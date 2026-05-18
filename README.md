# Beamax

[![CI](https://github.com/elma16/beamax/actions/workflows/run-tests.yml/badge.svg)](https://github.com/elma16/beamax/actions/workflows/run-tests.yml)

Implementation of the fast multiscale Gaussian wavepacket transform and multiscale Gaussian beam method of [Qian and Ying (2010)](https://doi.org/10.1137/100787313) in JAX, with tools for Fourier tilings, multiscale wave-packet transforms, and acoustic forward/reconstruction solvers.

> **Status:** v0.1.0 is the initial public release. The library follows semantic versioning, and the public API may evolve until v1.0; pin to a minor version if you need stability.

## What is included?

- JAX kernels for dyadic frequency tilings and a multiscale wave-packet transform (MSWPT).
- Gaussian beam solvers and utilities for hybrid MSGB / low-frequency solvers.
- Optional integrations (k-Wave, FNO) and plotting helpers (matplotlib/pyvista).
- MkDocs + mkdocstrings documentation powered by the docstrings in `src/beamax`.

## Installation

Python 3.11 or 3.12 is required.

1. Install JAX for your hardware (CPU/GPU/TPU) following the [official instructions](https://github.com/google/jax#installation).
2. Install `beamax` in one of the modes below.

Library usage (standard install):

```bash
# Minimal install
pip install .

# With plotting helpers
pip install .[viz]

# With optional solvers / operator-learning stacks
pip install .[kwave]       # k-wave-python
pip install .[fno]         # neuraloperator + pdequinox + torch

# Everything for development/docs/tests
pip install .[all]
```

Repository development (recommended when editing code/examples locally):

```bash
pip install -e .[dev]
```

If editable install fails because build dependencies cannot be fetched (offline/restricted network), use:

```bash
pip install -e .[dev] --no-build-isolation
```

Verify which package Python is loading:

```bash
python -c "import beamax; print(beamax.__file__)"
```

## Quickstart: 1D forward solve

This solves the 1D wave equation from initial pressure `p0` and records the wavefield on every grid point.

```python
import jax
import jax.numpy as jnp

from beamax import Domain, Sensor, DyadicDecomposition, MSWPT
from beamax.gb import gb_solvers
from beamax.solvers import MSGBSolver

jax.config.update("jax_enable_x64", True)

N = (128,)
domain = Domain(N=N, dx=(1.0 / N[0],), c=1.0, periodic=(True,))
ts = domain.generate_time_domain()
x = jnp.arange(N[0]) * domain.dx[0]
p0 = jnp.sin(2.0 * jnp.pi * x) + 0.5 * jnp.sin(4.0 * jnp.pi * x)

decomp = DyadicDecomposition(
    num_levels=2,
    N=domain.N,
    num_boxes_levels=(4, 8),
    box_aspect_ratio=(1,),
)
wpt = MSWPT(decomp, redundancy=2, windowing="rectangular")
sensors = Sensor(domain=domain, binary_mask=jnp.ones(domain.N))

solver = MSGBSolver(
    thr=int(wpt.total_coeffs),
    thr_strat="top_n",
    batch_size=256,
    input_type="spatial",
    ode_solver=gb_solvers.solve_ODE_base,
    sum_method="all_real",
)

sensor_data, params = solver.forward(p0, domain, sensors, ts, wpt)
print(sensor_data.shape)  # (Nt, 128)
```

For the complete version with non-zero initial velocity and a spectral reference check, run `examples/forward/1d_forward_solve.py`.

## Running examples

The supported public examples live under `examples/`. They are small,
self-contained, documented, paired with notebooks, linted, and smoke-tested in
CI. Research, profiling, comparison, and data-dependent scripts are preserved
under `examples/private/` as unsupported archived material.

```bash
python examples/forward/1d_forward_solve.py
```

If you are running headless (CI/remote shell), disable interactive plotting:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig python examples/forward/1d_forward_solve.py
```

### Google Colab

Each notebook in `examples/` carries an **Open in Colab** badge. The public gallery is indexed in [`examples/README.md`](examples/README.md) and is designed to run on a standard CPU Colab runtime.

Other optional environment variables:

- `BEAMAX_ROOT` — override the detected repository root (used by examples to locate `data/`, `plots/`).
- `BEAMAX_PROFILE=1` — enable timing/memory instrumentation in `beamax.utils.profiling`.

## Documentation

The API docs are generated from docstrings via MkDocs. After installing `.[dev]`, run:

```bash
mkdocs serve   # live preview at http://127.0.0.1:8000
```

The navigation in `mkdocs.yml` mirrors the modules in `src/beamax` (decomposition, transforms, geometry, plotter, utils, gb, solvers, …).

## References and related projects

Beamax's MSWPT/MSGB implementation follows:

- Jianliang Qian and Lexing Ying, ["Fast Multiscale Gaussian Wavepacket Transforms and Multiscale Gaussian Beams for the Wave Equation"](https://doi.org/10.1137/100787313), *Multiscale Modeling & Simulation*, 8(5), 1803-1837, 2010.

Related acoustic simulation projects:

- [k-Wave](http://www.k-wave.org/) — MATLAB/C++ toolbox for time-domain acoustic and ultrasound simulations.
- [k-Wave-python](https://github.com/waltsims/k-wave-python) / [docs](https://k-wave-python.readthedocs.io/) — Python wrapper and NumPy/CuPy implementation; Beamax uses this through the optional `[kwave]` extra.
- [j-Wave](https://github.com/ucl-bug/jwave) — differentiable acoustic simulations in JAX.

## Troubleshooting

### `ValueError: 0th dimension of all xs should be replicated`

This typically means Python is importing an older `beamax` build from `site-packages` instead of your local `src/beamax` checkout.

Fix:

```bash
pip uninstall -y beamax
pip install -e . --no-build-isolation
python -c "import beamax; print(beamax.__file__)"
```

The printed path should point into your repository (for example `.../python/beamax/src/beamax/__init__.py`).

### k-Wave on macOS

Beamax's `[kwave]` extra requires `k-wave-python>=0.6.2`, which includes the `binary_path` fix and the `v1.4.1` macOS OMP binary.

The macOS C++ binary is currently Apple Silicon (`arm64`) only. Intel Mac users should use `backend="python"` until upstream ships universal2 coverage.

Older Darwin OMP binaries, especially `v0.3.0rc3` and the bad `v1.4.0` release asset, can silently mishandle power-law absorption. `KWaveSolver` rejects those known-bad binaries when absorption is enabled. To test a custom binary, set `BEAMAX_KWAVE_BINARY_PATH` to either the `kspaceFirstOrder-OMP` file or the directory containing it.

Some older k-Wave binaries are linked against `libhdf5.310` while newer Homebrew installs provide `libhdf5.320`. `KWaveSolver` includes a runtime compatibility shim for this mismatch. If errors persist:

1. Ensure you are using this repo version (`python -c "import beamax; print(beamax.__file__)"`).
2. Reinstall your editable package in the active venv.
3. Confirm Homebrew HDF5 is installed (`ls /opt/homebrew/opt/hdf5/lib`).

### k-Wave time-reversal / adjoint uses the Python backend

The `KWaveSolver` uses the C++ binary for forward simulations (faster), but
automatically falls back to the pure-Python backend for time-reversal and
adjoint operations. This is due to missing source-term preprocessing in
k-wave-python's `CppSimulation` path for time-varying pressure sources.
This will be resolved in a future k-wave-python release.

### Matplotlib cache/font warnings

If you see cache permission warnings, set a writable config directory:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig python examples/forward/forward-2d.py
```

## Development

```bash
pip install .[dev]
pre-commit install
pytest
```

Useful commands:

- `ruff check src tests` for linting.
- `pytest --cov=src/beamax --cov-report=term-missing` for coverage.

## License

MIT; see `LICENSE`.

## Citation

If you use Beamax, please cite this repository. If you use the MSWPT/MSGB method, also cite Qian and Ying (2010). If you use results or implementation details from the forthcoming Beamax papers, please cite those papers once their citation details are available.
