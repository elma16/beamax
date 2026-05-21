# Custom Low-Frequency Solvers

`HybridSolver` keeps MSGB as the high-frequency path and delegates the
low-frequency path to a `HybridBackend`. A backend is just one or more
callables with the stable signature:

```python
def operation(component_array, context):
    ...
    return component_domain_result
```

The backend does not need to subclass `Solver`. It must implement at least one
of `forward`, `time_reversal`, or `adjoint`; missing operations raise
`NotImplementedError` only when that operation is called.

## Data Flow

For each hybrid operation:

1. `HybridSolver` splits the input into LF and HF components.
2. If `downsample=True`, the LF component and sensor mask are moved to a
   component grid.
3. The HF component is solved by MSGB or another HF-compatible solver.
4. The LF component is passed to the backend with a `HybridContext`.
5. `HybridSolver` applies LF windowing/interpolation as needed.
6. HF and LF results are added on the target output shape.

The backend should return its native component-domain result. Do not upsample
inside the backend unless you also set `downsample=False` and deliberately own
the full-resolution LF behavior.

## Runnable Forward-Only Backend

This backend is intentionally small and has no dependencies beyond beamax's
normal JAX stack. It solves a homogeneous, periodic, 1D acoustic wave equation
with zero initial velocity:

```python
import jax
import jax.numpy as jnp

from beamax import Domain, DyadicDecomposition, MSWPT, Sensor
from beamax.gb import gb_solvers
from beamax.solvers import HybridBackend, HybridSolver, MSGBSolver


jax.config.update("jax_enable_x64", True)


def spectral_lf_forward(p0_lf, ctx):
    domain = ctx.component_domain
    if len(domain.N) != 1:
        raise ValueError("spectral_lf_forward is a 1D example backend.")

    n = domain.N[0]
    dx = domain.dx[0]
    c0 = float(jnp.max(domain.sound_speed_array))
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=dx)

    p0_hat = jnp.fft.fft(jnp.asarray(p0_lf))
    phase = jnp.cos(ctx.ts[:, None] * c0 * jnp.abs(k)[None, :])
    fields = jnp.fft.ifft(phase * p0_hat[None, :], axis=-1).real

    sensor_mask = jnp.asarray(ctx.component_sensor_mask).astype(bool)
    return fields[:, sensor_mask]
```

The important part is the signature: `spectral_lf_forward(p0_lf, ctx)`. The
adapter reads the component domain, time grid, and sensor mask from
`HybridContext`, then returns sensor data with time on axis 0.

Use it in a hybrid solve like this:

```python
n = 64
domain = Domain(N=(n,), dx=(1.0 / n,), c=1.0, periodic=(True,))
ts = jnp.linspace(0.0, 0.08, 5)

x = jnp.arange(n) * domain.dx[0]
p0 = jnp.exp(-200.0 * (x - 0.35) ** 2) * jnp.cos(18.0 * jnp.pi * x)

decomp = DyadicDecomposition(
    num_levels=2,
    N=domain.N,
    num_boxes_levels=(4, 8),
    box_aspect_ratio=(1,),
)
wpt = MSWPT(decomp, redundancy=2, windowing="rectangular")
sensors = Sensor(domain=domain, binary_mask=jnp.ones(domain.N))

msgb = MSGBSolver(
    thr=int(wpt.total_coeffs),
    thr_strat="top_n",
    batch_size=64,
    input_type="spatial",
    ode_solver=gb_solvers.solve_ODE_base,
    sum_method="all_real",
)

hybrid = HybridSolver(
    hf_solver=msgb,
    lf_backend=HybridBackend(
        forward=spectral_lf_forward,
        name="1D spectral LF example",
    ),
    box_corners=jnp.array([0, 1]),
    downsample=False,
    use_time_extension=False,
    dt_oversample=0,
)

sensor_data = hybrid.forward(p0, domain, sensors, ts, wpt)
print(sensor_data.shape)  # (5, 64)
```

The complete script is in `examples/forward/custom_lf_spectral_backend.py`.

## Minimal Adapter Shape

For a real backend, the same pattern usually reduces to:

```python
def lf_forward(component, ctx):
    return my_wave_solver.forward(
        component,
        domain=ctx.component_domain,
        sensors=ctx.component_sensor_mask,
        ts=ctx.ts,
    )


hybrid = HybridSolver(
    hf_solver=msgb,
    lf_backend=HybridBackend(forward=lf_forward, name="my LF solver"),
    cutoff_freq=0.35,
    downsample=False,
)
```

This backend can run `hybrid.forward(...)`. Calling
`hybrid.time_reversal(...)` or `hybrid.adjoint(...)` will fail clearly because
those LF operations were not provided.

## Wrapping Existing beamax-Style Solvers

Solvers that already use the beamax-style argument order can be wrapped with:

```python
from beamax.solvers import HybridBackend, HybridSolver, KWaveSolver, MSGBSolver

kwave = KWaveSolver(...)
msgb = MSGBSolver(...)

hybrid = HybridSolver(
    hf_solver=msgb,
    lf_backend=HybridBackend.from_beamax_solver(kwave),
    box_corners=...,
)
```

The helper maps `forward(component, ctx)` to
`solver.forward(component, ctx.component_domain, ctx.component_sensor_mask,
ctx.ts)` and does the analogous mapping for `time_reversal` and `adjoint`.
If your solver needs a different source layout, custom boundary weights, or
extra arguments, write an explicit adapter callable instead.

## Shape Expectations

`component_array` is the low-frequency component after splitting. With
`downsample=True`, it lives on `ctx.component_domain`; with `downsample=False`,
`ctx.component_domain` is the original full grid.

For `forward`, the backend usually returns sensor data with time on axis 0.
`ctx.ts` may be longer than `ctx.original_ts` because hybrid forward solves can
use time extension for LF/HF windowing. `HybridSolver` truncates back to the
original time grid after merging.

For `time_reversal` and `adjoint`, the backend usually returns an image on
`ctx.component_domain.N`. `HybridSolver` interpolates that image to
`ctx.target_shape` when downsampling is enabled.

Use `downsample=False` when the LF solver already owns its grid, uses off-grid
or sparse sensor objects, or cannot consume the interpolated component mask.
In that mode, the backend sees full-resolution fields and masks and hybrid
does not interpolate the LF result.

## Optional j-Wave Adapter Sketch

j-Wave is a good candidate for an LF adapter because it exposes JAX-based wave
simulation primitives and custom media. Keep it optional in your own
environment; it is not a beamax core dependency.

The current PyPI `jwave` package may pin older JAX/JaxDF versions than beamax
uses. Treat this as an environment-level integration rather than a copy-paste
first run until those dependency ranges are compatible with your beamax
environment.

```python
import jax.numpy as jnp

from beamax.solvers import HybridBackend, HybridSolver, MSGBSolver

try:
    from jwave import FourierSeries
    from jwave.acoustics import TimeWavePropagationSettings
    from jwave.acoustics import simulate_wave_propagation
    from jwave.geometry import Domain as JWaveDomain
    from jwave.geometry import Medium, Sensors, TimeAxis
except ImportError as exc:
    raise RuntimeError("Install j-Wave separately to run this adapter.") from exc


def mask_to_jwave_positions(mask):
    return tuple(jnp.asarray(axis_idx) for axis_idx in jnp.where(mask > 0))


def jwave_forward(p0_lf, ctx):
    jwave_domain = JWaveDomain(ctx.component_domain.N, ctx.component_domain.dx)
    medium = Medium(
        domain=jwave_domain,
        sound_speed=jnp.asarray(ctx.component_domain.sound_speed_array),
        density=1.0,
    )
    dt = float(ctx.ts[1] - ctx.ts[0])
    time_axis = TimeAxis(dt=dt, t_end=float(len(ctx.ts) * dt))
    sensors = Sensors(positions=mask_to_jwave_positions(ctx.component_sensor_mask))
    pressure = FourierSeries(jnp.asarray(p0_lf), jwave_domain)

    recorded = simulate_wave_propagation(
        medium,
        time_axis,
        p0=pressure,
        sensors=sensors,
        settings=TimeWavePropagationSettings(checkpoint=False),
    )
    recorded = jnp.asarray(recorded)[: len(ctx.ts)]
    if recorded.ndim == 3 and recorded.shape[-1] == 1:
        recorded = recorded[..., 0]
    return recorded


hybrid = HybridSolver(
    hf_solver=MSGBSolver(...),
    lf_backend=HybridBackend(forward=jwave_forward, name="j-Wave LF"),
    cutoff_freq=0.35,
    downsample=False,
)
```

The exact `Sensors` construction is application-specific. For dense on-grid
masks, convert active mask indices to a tuple of integer index arrays, as above.
For off-grid or sparse geometry, prefer `downsample=False` so the adapter can
preserve the geometry directly.

References:

- [j-Wave documentation](https://ucl-bug.github.io/jwave/index.html)
- [j-Wave PyPI package](https://pypi.org/project/jwave/)
