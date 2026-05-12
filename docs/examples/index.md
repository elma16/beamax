# Examples

The repository ships a curated public example gallery organised by topic. Each page embeds the full runnable source; use the copy-button on the upper-right of each code block to paste it into your own session, or clone the repo and run the file directly:

```bash
python examples/<category>/<file>.py
```

Set `MPLBACKEND=Agg` when running headless (CI / remote shells).

Research, profiling, comparison, and data-dependent scripts are preserved under
`examples/private/`, but they are not part of the public docs or CI smoke suite.
Some public examples are marked optional in their module docstring; these
require `beamax[kwave,viz-mpl]` and are skipped by the default CI smoke suite.

## Run on Google Colab

Notebooks that ship alongside the public scripts carry an **Open in Colab** badge. Click the badge on the category pages below (or on [examples/README.md](https://github.com/elma16/beamax/blob/main/examples/README.md)) to open the notebook. The public examples are small enough to run on a standard CPU Colab runtime.

## Gallery

### Frequency decomposition & wave-packet transforms

- [Wave packet frame atoms](decomposition.md#wave-packet-frame-atoms) — render MSWPT atoms.
- [Low and high pass filters](decomposition.md#low-and-high-pass-filters) — frequency separation via MSWPT.
- [Wave packet cutoff error](decomposition.md#wave-packet-cutoff-error) — error vs. box count / redundancy.

### Forward propagation

- [1D forward solve](forward.md#1d-forward-solve) — MSGB vs. spectral ground truth.
- [2D forward solve](forward.md#2d-forward-solve) — trigonometric initial data with error maps.
- [1D forward k-Wave reference](forward.md#1d-forward-k-wave-reference) — optional k-Wave reference comparison.
- [2D forward k-Wave reference](forward.md#2d-forward-k-wave-reference) — optional k-Wave boundary-sensor comparison.

### Time-reversal reconstruction

- [1D time reversal](time-reversal.md#1d-time-reversal) — one-sided MSGB time-reversal smoke test.
- [2D time reversal and adjoint](time-reversal.md#2d-time-reversal-and-adjoint) — optional k-Wave inverse comparison.

### Rays and autofocus

- [2D ray bending](rays.md#2d-ray-bending) — trace a fan of rays through a smooth speed field.
- [Neural sound speed autofocus](rays.md#neural-sound-speed-autofocus) — optimize a neural `c(x)` field with autodiff through the ray ODE.

### Single Gaussian beam diagnostics

- [Single Gaussian beam absorption](single-gaussian-beam.md#single-gaussian-beam-absorption) — optional absorbing-beam comparison.
