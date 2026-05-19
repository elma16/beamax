# beamax

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

From a source checkout:

```bash
# Minimal install
pip install .

# With plotting helpers
pip install .[viz]

# With optional solver / operator-learning stacks
pip install .[kwave]       # k-wave-python integration
pip install .[fno]         # neuraloperator + pdequinox + torch

# Everything for development/docs/tests
pip install .[all]
```

Repository development (recommended when editing code/examples locally):

```bash
pip install -e ".[dev,kwave,viz]"
```

## Quickstart: 1D forward solve

This example solves the 1D acoustic wave equation on a periodic unit interval.
It starts from a simple initial pressure `p0`, records the wavefield at every
grid point, and prints the shape of the recorded data.

```python
import jax
import jax.numpy as jnp

from beamax import Domain, Sensor, DyadicDecomposition, MSWPT
from beamax.gb import gb_solvers
from beamax.solvers import MSGBSolver

# Use double precision for the small numerical example below.
jax.config.update("jax_enable_x64", True)

# Build a 1D periodic domain with 128 grid points on [0, 1).
# The sound speed is the constant c=1
N = (128,)
domain = Domain(N=N, dx=(1.0 / N[0],), c=1.0, periodic=(True,))

# Ask beamax for a stable time grid based on the domain spacing, wave speed,
# and CFL number.
ts = domain.generate_time_domain()

# Define the initial pressure on the grid: a sum of two sine waves.
x = jnp.arange(N[0]) * domain.dx[0]
p0 = jnp.sin(2.0 * jnp.pi * x) + 0.5 * jnp.sin(4.0 * jnp.pi * x)

# Describe how Fourier space is split into multiscale wave-packet boxes.
decomp = DyadicDecomposition(
    num_levels=2,
    N=domain.N,
    num_boxes_levels=(4, 8),
    box_aspect_ratio=(1,),
)

# Build the multiscale wave-packet transform used to convert p0 into beams.
wpt = MSWPT(decomp, redundancy=2, windowing="rectangular")

# Put a sensor at every grid point. The output will therefore be the full
# time history of the 1D wavefield.
sensors = Sensor(domain=domain, binary_mask=jnp.ones(domain.N))

# Configure the multiscale Gaussian beam solver. `top_n` with total_coeffs keeps
# all transform coefficients, which is the clearest setting for a first run.
solver = MSGBSolver(
    thr=int(wpt.total_coeffs),
    thr_strat="top_n",
    batch_size=256,
    input_type="spatial",
    ode_solver=gb_solvers.solve_ODE_base,
    sum_method="all_real",
)

# Run the forward solve. sensor_data has shape (time samples, sensors).
sensor_data, beam_params = solver.forward(p0, domain, sensors, ts, wpt)
print(sensor_data.shape)  # (len(ts), 128)
```

The printed shape is `(number_of_time_samples, number_of_sensors)`. Because the
sensor mask is all ones, this example records every grid point at every time.

For the complete version with non-zero initial velocity and a spectral reference check, run `examples/forward/1d_forward_solve.py`.

## Running examples

The supported public examples live under `examples/`. They are small,
self-contained, documented, paired with notebooks, linted, and smoke-tested in
CI. Install `.[viz-mpl]` before running examples that save matplotlib figures.
Examples marked optional in their module docstring require additional extras
such as `beamax[kwave,viz-mpl]` and are skipped by the default smoke suite.

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

- `BEAMAX_ROOT` — override the detected repository root (used by examples to locate `plots/`).
- `BEAMAX_PROFILE=1` — enable timing/memory instrumentation in `beamax.utils.profiling`.

## Documentation

The API docs are generated from docstrings via MkDocs. After installing `.[dev]`, run:

```bash
mkdocs serve   # live preview at http://127.0.0.1:8000
```

The navigation in `mkdocs.yml` mirrors the modules in `src/beamax` (decomposition, transforms, geometry, plotter, utils, gb, solvers, …).

## References and related projects

beamax's MSWPT/MSGB implementation follows:

- Jianliang Qian and Lexing Ying, ["Fast Multiscale Gaussian Wavepacket Transforms and Multiscale Gaussian Beams for the Wave Equation"](https://doi.org/10.1137/100787313), *Multiscale Modeling & Simulation*, 8(5), 1803-1837, 2010.

Related acoustic simulation projects:

- [k-Wave](http://www.k-wave.org/) — MATLAB/C++ toolbox for time-domain acoustic and ultrasound simulations.
- [k-Wave-python](https://github.com/waltsims/k-wave-python) — Python wrapper used by beamax through the optional `[kwave]` extra.
- [j-Wave](https://github.com/ucl-bug/jwave) — differentiable acoustic simulations in JAX.

## Development

```bash
pip install -e ".[dev,kwave,viz]"
tools/install-hooks.sh
pytest
```

Useful commands:

- `ruff check src tests` for linting.
- `pytest --cov=src/beamax --cov-report=term-missing` for coverage.

## License

MIT; see `LICENSE`.

## Citation

If you use beamax, please cite this repository. If you use the MSWPT/MSGB method, also cite Qian and Ying (2010).
