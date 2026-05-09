# Examples

The repository ships a curated example gallery organised by topic. Each page embeds the full runnable source; use the copy-button on the upper-right of each code block to paste it into your own session, or clone the repo and run the file directly:

```bash
python examples/<category>/<file>.py
```

Set `MPLBACKEND=Agg` when running headless (CI / remote shells). Examples that depend on k-Wave need `pip install 'beamax[kwave]'`.

## Run on Google Colab

Notebooks that ship alongside the scripts carry an **Open in Colab** badge. Click the badge on the category pages below (or on [examples/README.md](https://github.com/elma16/beamax/blob/main/examples/README.md)) to open the notebook on a free GPU or TPU runtime — no local setup required. Remember to switch to a GPU/TPU runtime in Colab (`Runtime → Change runtime type`) to see the hardware-acceleration benefits of JAX.

## Gallery

### Frequency decomposition & wave-packet transforms

- [Frame atoms figure](decomp.md#frames-figure) — render MSWPT atoms.
- [Frame atoms grid](decomp.md#frames-grid) — 3×3 grid of MSWPT atoms across scales.
- [Low/high-pass filter](decomp.md#lp-hp-filter) — frequency separation via MSWPT.
- [MSWPT reconstruction error](decomp.md#mswpt-error-plot) — error vs. box count / redundancy.

### Forward propagation

- [Forward 1D (Cauchy data)](forward.md#forward-1d-v0) — MSGB vs. spectral ground truth.
- [Forward 2D (periodic box)](forward.md#forward-2d-v0) — trigonometric initial data with error maps.
- [Forward 2D (MSGB vs. k-Wave vs. hybrid)](forward.md#forward-2d) — three-way comparison.

### Time-reversal reconstruction

- [TR 1D](time-reversal.md#tr-1d) — MSGB vs. k-Wave time-reversal in 1D.
- [TR 2D](time-reversal.md#tr-2d) — MSGB vs. k-Wave time-reversal in 2D.
- [TR 3D diagnostic](time-reversal.md#tr-3d) — per-stage diagnostics for 3D.

### Adjoint reconstruction

- [Adjoint 2D (v1)](adjoint.md#ad-2d) — MSGB adjoint reconstruction of `p0`.
- [Adjoint 2D (v2)](adjoint.md#ad-2d2) — linearised MSGB adjoint with autodiff.
- [Adjoint 2D (v3)](adjoint.md#ad-2d3) — full MSGB forward with autodiff.
- [Autodiff 1D](adjoint.md#autodiff-1d) — independent check via JAX autodiff.
- [Autodiff 2D](adjoint.md#autodiff-2d) — 2D autodiff cross-check.
- [Autodiff 3D](adjoint.md#autodiff-3d) — 3D autodiff cross-check.

### Inverse-operator comparisons

- [Adjoint vs. time-reversal on a box phantom](comparison.md#adj-vs-tr-boxtest) — contrast inverse operators.
- [Iterative adjoint + TR](comparison.md#iterative-adj-tr) — combined iterative scheme.
- [k-Wave TR vs. adjoint](comparison.md#kwave-tr-vs-adj) — reference sanity check.

### Benchmarks

- [k-Wave vs. MSGB runtime](benchmarks.md#gb-vs-kw-runtime) — grid-size sweep.
- [Memory footprint](benchmarks.md#memory-test) — peak RSS during MSGB forward solve.
- [Detailed memory + time](benchmarks.md#memory-test-detailed) — per-phase profiling including JAX async.
- [Strong & weak scaling](benchmarks.md#strong-weak-scaling) — multi-device beam parallelism.
- [MSWPT runtime](benchmarks.md#wpt-decomp-runtime) — forward/inverse vs. grid size and depth.
