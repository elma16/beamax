# Examples

The repository ships a curated public example gallery organised by topic. Each page embeds the full runnable source; use the copy-button on the upper-right of each code block to paste it into your own session, or clone the repo and run the file directly:

```bash
python examples/<category>/<file>.py
```

Set `MPLBACKEND=Agg` when running headless (CI / remote shells).
Some public examples are marked optional in their module docstring; these
require `beamax[kwave,viz-mpl]` and are skipped by the default smoke suite.
CI installs those extras and runs the full public example suite with
`--include-optional`.

## Run on Google Colab

Notebooks that ship alongside the public scripts carry an **Open in Colab** badge. Click the badge on the category pages below (or on [examples/README.md](https://github.com/elma16/beamax/blob/main/examples/README.md)) to open the notebook. The public examples are small enough to run on a standard CPU Colab runtime.

## Gallery

### Forward propagation

- [Custom low-frequency backend](forward.md#custom-low-frequency-backend) — plug a pure-JAX LF backend into the hybrid solver.
- [2D photoacoustic forward](forward.md#2d-photoacoustic-forward) — optional PAT sensor-data comparison against k-Wave.

### Reconstruction

- [2D time reversal and adjoint](reconstruction.md#2d-time-reversal-and-adjoint) — optional k-Wave inverse comparison.

### Rays and autodiff

- [2D ray bending](rays.md#2d-ray-bending) — trace a fan of rays through a smooth speed field.
- [2D rays autodiff](rays.md#2d-rays-autodiff) — optimize a neural `c(x)` field with autodiff through the ray ODE.

### Single Gaussian Beam Diagnostics

- [Single Gaussian beam absorption](single-gaussian-beam.md#single-gaussian-beam-absorption) — absorbing-beam comparison.
