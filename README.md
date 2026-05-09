# Beamax

[![CI](https://github.com/elma16/beamax/actions/workflows/run-tests.yml/badge.svg)](https://github.com/elma16/beamax/actions/workflows/run-tests.yml) [![codecov](https://codecov.io/gh/elma16/beamax/branch/main/graph/badge.svg)](https://codecov.io/gh/elma16/beamax)

Implementation of the fast multiscale Gaussian wavepacket transform and multiscale Gaussian beam method of [Qian and Ying (2010)](https://doi.org/10.1137/100787313) in JAX, with tools for Fourier tilings, multiscale wave-packet transforms, and acoustic forward/reconstruction solvers.

> **Status:** v0.1.0 is the initial public release. The library follows semantic versioning, and the public API may evolve until v1.0 — pin to a minor version if you need stability.

## What is included?

- JAX-first kernels for dyadic frequency tilings and a multiscale wave-packet transform (MSWPT).
- Gaussian beam solvers and utilities for hybrid MSGB / low-frequency methods.
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

## Quickstart

The conventional short alias is `bmx`:

```python
import jax
import jax.numpy as jnp
import beamax as bmx
from beamax import Domain, DyadicDecomposition, MSWPT

# Grid + frequency tiling
domain = Domain(N=(64, 64), dx=(1e-3, 1e-3), c=1500.0, periodic=(True, True))
dyadic = DyadicDecomposition(
    num_levels=2,
    N=domain.N,
    num_boxes_levels=(4, 8),
    box_aspect_ratio=(1, 1),
)
wpt = MSWPT(dyadic, redundancy=2, windowing="rectangular")

# Analyse + reconstruct a field
field = jax.random.normal(jax.random.PRNGKey(0), domain.N)
coeffs = wpt.forward(field, input_type="spatial")
recon = wpt.inverse(coeffs, output_type="spatial").real

# Plot helper (optional)
from beamax.plotter import PlotHelper
PlotHelper().plot_wavefield(recon, title="Reconstructed field")
```

## Running examples

Every example under `examples/` runs locally from a checkout:

```bash
python examples/forward/forward-2d.py
```

If you are running headless (CI/remote shell), disable interactive plotting:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig python examples/forward/forward-2d.py
```

### Google Colab

Each notebook in the `examples/` tree carries an **Open in Colab** badge so you can run it on a free GPU or TPU without any local setup — see the gallery in [`examples/README.md`](examples/README.md). After clicking the badge, switch the runtime (`Runtime → Change runtime type → GPU` or `TPU`) to take advantage of JAX's hardware acceleration.

Heavier benchmarks support a reduced default mode for faster validation. Opt into a full sweep with:

```bash
BEAMAX_FULL_BENCHMARKS=1 python examples/benchmarks/gb-vs-kw-runtime.py
```

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

### k-Wave fails on macOS with `libhdf5.310.dylib` not found

`k-wave-python` binaries may be linked against `libhdf5.310` while newer Homebrew installs provide `libhdf5.320`. `KWaveSolver` includes a runtime compatibility shim for this mismatch. If errors persist:

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

If you use this code, please cite Qian and Ying (2010) and this repository.
