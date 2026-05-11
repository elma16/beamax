# Getting Started

## Prerequisites

- Python 3.11 or 3.12
- JAX installed for your hardware (CPU/GPU/TPU), following the [official JAX install guide](https://github.com/google/jax#installation)

## Install

From the repository root:

```bash
# Core package
pip install .

# Optional extras
pip install .[viz]      # plotting helpers
pip install .[kwave]    # k-wave-python integration
pip install .[fno]      # operator-learning stack
pip install .[all]      # everything (dev + optional extras)
```

If you are actively developing in this checkout, use editable mode:

```bash
pip install -e .[dev]
```

If build isolation cannot fetch dependencies (offline/restricted network):

```bash
pip install -e .[dev] --no-build-isolation
```

## Verify Your Import Path

```bash
python -c "import beamax; print(beamax.__file__)"
```

- Editable install should point to `.../src/beamax/__init__.py`.
- If it points to `.../site-packages/beamax/...`, your environment is using an installed wheel instead of your local source tree.

## Minimal 1D Forward Solve

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
print(sensor_data.shape)
```

## Run an Example

```bash
python examples/forward/forward-1d-v0.py
```

For headless environments:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig python examples/forward/forward-1d-v0.py
```
