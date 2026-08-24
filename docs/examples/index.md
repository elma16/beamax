# Examples

Run an example from the repository root:

```bash
python examples/<category>/<file>.py
```

Set `MPLBACKEND=Agg` when running headless.
Each example page lists any optional requirements and its Colab notebook.

## Gallery

### Forward propagation

- [MSGB vs k-Wave](forward.md#msgb-vs-k-wave) — compare 2D forward sensor data.

### Reconstruction

- [2D time reversal and adjoint](reconstruction.md#2d-time-reversal-and-adjoint) — optional k-Wave inverse comparison.

### Rays and autodiff

- [2D ray bending](rays.md#2d-ray-bending) — trace a fan of rays through a smooth speed field.
- [2D rays autodiff](rays.md#2d-rays-autodiff) — optimize a neural `c(x)` field with autodiff through the ray ODE.

### Single Gaussian beam diagnostics

- [Single Gaussian beam absorption](single-gaussian-beam.md#single-gaussian-beam-absorption) — absorbing-beam comparison.

### Diagnostics

- [Memory planning](diagnostics.md#memory-planning) — compare static MSGB memory estimates with a device budget.
