# Custom Low-Frequency Solvers

`HybridSolver` delegates low frequencies to a `HybridBackend` and uses MSGB
for high frequencies. A backend provides one or more callables with this
signature:

```python
def operation(component_array, context):
    ...
    return component_domain_result
```

No Beamax base class is required. Implement at least one of `forward`,
`time_reversal`, or `adjoint`; calling an omitted operation raises
`NotImplementedError`.

## Data flow

1. `HybridSolver` splits the input into LF and HF components.
2. With `downsample=True`, it moves the LF component and sensor mask to a
   component grid.
3. It solves the HF component and passes the LF component plus a
   `HybridContext` to the backend.
4. It applies LF windowing/interpolation and combines both results on the
   target shape.

Return a result on `ctx.component_domain`. `HybridSolver` handles interpolation
when `downsample=True`; otherwise the backend owns the full-resolution result.

## Minimal adapter

A typical adapter is:

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

This configuration supports only `hybrid.forward(...)`.

## Existing beamax-style solvers

Wrap solvers with beamax-style arguments using
`HybridBackend.from_beamax_solver`:

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

It maps the context to the corresponding solver arguments for `forward`,
`time_reversal`, and `adjoint`. Use an explicit adapter for custom source
layouts, boundary weights, or extra arguments.

## Shapes

`component_array` is the split LF component. With `downsample=True`, it lives
on `ctx.component_domain`; otherwise that domain is the original grid.

`forward` should return sensor data with time on axis 0. `ctx.ts` may include
time-extension samples; `HybridSolver` removes the extension after merging.

`time_reversal` and `adjoint` should return an image on
`ctx.component_domain.N`. With downsampling, `HybridSolver` interpolates it to
`ctx.target_shape`.

Use `downsample=False` when the LF solver owns its grid, uses off-grid or sparse
sensors, or cannot consume the interpolated mask. The backend then receives
full-resolution fields and masks, and its result is not interpolated.

## j-Wave

j-Wave is optional and not a beamax dependency. Its JAX and JaxDF constraints
may differ from beamax's, so check resolver compatibility before combining them.
Convert dense masks to j-Wave index arrays inside the adapter. Use
`downsample=False` for off-grid or sparse sensors.

References:

- [j-Wave documentation](https://ucl-bug.github.io/jwave/index.html)
- [j-Wave PyPI package](https://pypi.org/project/jwave/)
