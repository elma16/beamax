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
pip install -e ".[dev,kwave,viz]"
```

## Minimal 1D Forward Solve

This example solves the 1D acoustic wave equation on a periodic unit interval.
It starts from an initial pressure field, records the wavefield at every grid
point, and prints the shape of the recorded data.

```python
import jax
import jax.numpy as jnp

from beamax import Domain, Sensor, DyadicDecomposition, MSWPT
from beamax.gb import gb_solvers
from beamax.solvers import MSGBSolver

# Use double precision for this small numerical example.
jax.config.update("jax_enable_x64", True)

# Step 1: define the physical domain.
# N gives the number of grid points. dx gives the physical spacing. c=1.0
# makes the sound speed constant, and periodic=True wraps waves around [0, 1).
N = (128,)
domain = Domain(N=N, dx=(1.0 / N[0],), c=1.0, periodic=(True,))

# Step 2: generate a stable time grid from the domain spacing and wave speed.
ts = domain.generate_time_domain()

# Step 3: define the initial pressure on the grid.
# Here p0 is just a sum of two sine waves, so the signal is easy to recognise.
x = jnp.arange(N[0]) * domain.dx[0]
p0 = jnp.sin(2.0 * jnp.pi * x) + 0.5 * jnp.sin(4.0 * jnp.pi * x)

# Step 4: choose a multiscale Fourier tiling for the wave-packet transform.
decomp = DyadicDecomposition(
    num_levels=2,
    N=domain.N,
    num_boxes_levels=(4, 8),
    box_aspect_ratio=(1,),
)

# Step 5: build the wave-packet transform used to convert p0 into beam data.
wpt = MSWPT(decomp, redundancy=2, windowing="rectangular")

# Step 6: put a sensor at every grid point, so the output is the full wavefield
# sampled at each time in ts.
sensors = Sensor(domain=domain, binary_mask=jnp.ones(domain.N))

# Step 7: configure the multiscale Gaussian beam solver. The `top_n` strategy
# with `thr=wpt.total_coeffs` keeps every transform coefficient; this is a
# simple first-run setting before trying coefficient truncation.
solver = MSGBSolver(
    thr=int(wpt.total_coeffs),
    thr_strat="top_n",
    batch_size=256,
    input_type="spatial",
    ode_solver=gb_solvers.solve_ODE_base,
    sum_method="all_real",
)

# Step 8: run the forward solve.
# sensor_data has shape (number of time samples, number of sensors).
sensor_data, beam_params = solver.forward(p0, domain, sensors, ts, wpt)
print(sensor_data.shape)  # (len(ts), 128)
```

The printed shape is `(number_of_time_samples, number_of_sensors)`. Because the
sensor mask is all ones, this example records every grid point at every time.
