# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- MSGB, k-Wave, and Hybrid accept a sampled `DetectorSurface` or a binary
  mask through `detectors=` for forward, time reversal, and adjoint solves.
  Forward pressure has one row per time sample and one column per detector.
  Pass those traces and the same detectors to either inverse.
- MSGB supports time reversal and adjoint on
  curved detector surfaces. On a complete, uniformly sampled inward-facing
  ring with a compatible exterior sound speed, `adjoint` selects the circular
  formula automatically. Surface adjoints use detector areas or arc lengths
  by default. [Detector geometry and adjoint weighting](docs/api/surfaces.md)
- MSGB evaluates forward pressure at physical receiver positions. The Python
  k-Wave backend can record off-grid traces by linear interpolation. Native
  k-Wave recording and all k-Wave inverse solves require grid-aligned
  detectors. `surface.to_mask(domain, snap=True)` explicitly maps receivers
  to their nearest grid points and reports the channel ordering.
- `KWaveSolver(backend="mac")` runs k-Wave on Apple silicon with macOS 15
  or newer. The first Metal solve downloads, verifies, and caches the
  NeuroHarmonics executable. Existing `device="metal"` calls also select it.
- The reconstruction examples show how to reuse a planar detector or a ring
  for forward and inverse solves. The 2D and 3D MSWPT coefficient plots
  accept `display_order="array"` to align their axes with NumPy array indices.

### Changed

- `KWaveSolver` uses the selected backend for forward simulation, time
  reversal, and the adjoint. The default `backend="cpp"` uses the native
  OpenMP CPU executable for all three. Select `backend="python"` explicitly
  to reproduce an inverse run that previously fell back to Python.
- `MSGBSolver.forward` propagates from time zero even when the first requested
  sample is later. The underlying Gaussian beam integrators take their
  initial state at the first supplied time.
- MSGB time reversal and adjoint now require a lossless domain. The k-Wave
  adjoint validates the lossless-medium assumption and rejects smoothed
  initial pressure, which would make the forward map nonlinear.
- Hybrid records both frequency bands at the same receivers in the same
  order. It keeps the low-frequency solve on the original grid when
  coarsening would move a receiver. Both inverse components use the
  detector geometry and sampling times of the supplied traces.

### Fixed

- Fourier cropping and padding preserve the DC bin when grid parity changes.
  MSWPT analysis and synthesis now agree on the frequency centres and
  widths of wave packets, including odd-width supports and rectangular grids.
- Gaussian beam propagation carries constant absorption through the analytic,
  numerical, and fused solvers and preserves phase through a focus. Batched
  Hamiltonian derivatives and gradients at zero beam amplitude are corrected.
- MSGB forward handles the initial pressure derivative's zero-frequency
  component on a periodic grid with constant sound speed. Planar MSGB
  inverses launch both frequency signs inward and count each conjugate
  packet pair once when selecting real data with `top_n`.
- Hybrid crops low-frequency fields along each spatial axis and passes
  Fourier coefficients to an MSGB component configured for Fourier input.
- The native C++ k-Wave backend preserves time and sensor order when both
  axes have the same length, and retains the channel axis for one sensor.
  k-Wave inverse calls validate the trace length. Simulations using legacy
  k-Wave options get independent scratch files by default.

## [0.3.0] — 2026-08-24

### Changed

- **Streamed top-n coefficient selection is now the default** for eligible
  real, windowed data with `thr_strat="top_n"`. At cutoff ties it retains the
  lower global flat index. Other configurations use the materialized fallback.
  Explicit `"streaming_top_n"` makes unsupported inputs an error.
- **Terminal-only XLA inverse evaluation is now the default** for time
  reversal and adjoint with real, single-device aggregation. This removes the
  all-time per-beam tensor. TR remains bitwise-identical. Under `all_*`,
  `time_reversal_with_params` and `adjoint_with_params` now return zero-padded
  `(num_batches, batch_size, ...)` parameters instead of a flat layout.
- **The zero-`dpdt` forward shortcut is always active**: `dpdt=None` uses
  $c_+ = \tfrac{1}{2}\operatorname{WPT}(p_0)$, skipping a zero-field transform
  and unused group-velocity geometry.
- Full-float32 `Precision.HIGHEST` is the portable Gaussian beam contraction
  contract; backend-default reduced precision is not the numerical reference.
- `MSGBExperimentalConfig` now defaults to `"auto"`. Pre-release
  `"materialized"` and `"legacy"` overrides were removed; those formulations
  remain automatic fallbacks only.
- Python 3.12 is now the minimum; Python 3.11 support was removed.

### Added

- Added experimental `pallas_real` MSGB aggregation, fusing real beam
  evaluation and reduction into detector tiles on GPU Triton and float32
  Mosaic TPU. It provides frozen `PallasConfig` tuning and CPU interpret-mode
  coverage. Native accelerator validation remains limited, so Pallas stays
  opt-in.
- Added experimental opt-in 3D pipeline stages: `trajectory_pallas` and
  `hom_diag_3d_pallas` forward kernels, plus a `terminal_pallas` inverse
  evaluator.
- Added memory planning under `beamax.utils`, with device inspection,
  allocation lower bounds, an explicit safety margin, and a budget verdict.
- Exposed the MSWPT half-frame mask and adjoint source/image weighting helpers.

### Removed

- Removed `beamax.solvers.Solver`, `TimedKWaveSolver`, and
  `MSWPT.convert_to_array`.
- Removed `PlotHelper`, animation and PyVista helpers, and the `viz` extra.
  `beamax.plotter` now contains Matplotlib styling and MSWPT coefficient plots.
- Moved thesis-only plotting, benchmarks, and diagnostics out of beamax, and
  removed the obsolete notebook converter.
- Removed gallery path helpers, custom Git-hook wrappers, redundant `solve_ivp`
  aliases, unused threshold policies, and duplicate `vmap_*` aggregators.
- Reduced `HybridContext` to the six values consumed by low-frequency
  backends: domains, component sensor mask, time grid, target shape, and
  optional sources.

### Fixed

- Time reversal now assembles results with the reconstruction domain's grid
  size, not the acquisition domain's.
- Adjoint parameter sharding now uses the time-reversal aggregation-policy
  gate instead of applying whenever a sharding strategy exists.
- Single-device Pallas aggregation and explicit experimental stages now reject
  solver sharding.
- k-Wave option objects can no longer be mixed with keyword options, and its
  stale-file workaround no longer monkeypatches k-Wave globally.
- Real MSGB aggregation methods now reject complex forward and inverse input;
  use the corresponding `*_complex` method instead.

## [0.2.2] — 2026-07-14

### Fixed

- Standardised standalone k-Wave sensor channels to NumPy C order, including planar 3D detector data used by TR and adjoint solvers.
- Made grid-valued medium fields usable by Gaussian beam ray solvers and passed evaluated absorption fields to k-Wave.
- Repaired data-dependent MSGB thresholding under JAX, dimensional Bao-energy indexing, top-n bounds, and threshold validation.
- Corrected rectangular MSWPT tilings and rejected decompositions with uncovered Fourier bins.
- Preserved heterogeneous medium parameters during Hybrid downsampling and fixed time windows for multidimensional detector arrays.
- Retained the validated complex Diffrax path while exposing its upstream support warning.
- Added strict interpolation, geometry, sensor, time-grid, PML, batching, and sharding validation.
- Fixed 3D complex wavefield and optional PyVista plotting paths.

### Changed

- Ambiguous square sensor data now requires an explicit `data_layout`.
- User-supplied k-Wave PML options are honoured; safe values are derived only when they are omitted.
- Numerical warnings are no longer globally suppressed.
- Release CI now checks the lockfile, builds and imports the wheel, and exercises a two-device sharding contract.

## [0.2.1] — 2026-07-12

### Fixed

- Folded detector sound speeds onto planar 3D detector grids when forming the adjoint source.

## [0.2.0] — 2026-07-12

### Changed

- Audited the principal-symbol adjoint, MSWPT analysis/synthesis, TR geometry, Hybrid splitting, and solver aggregation paths.
- Added public mapping and adjoint diagnostics with regression coverage.

### Removed

- Removed the optional FNO adapters and the `fno` extra.

## [0.1.0] — initial public release

### Added

- JAX-first kernels for dyadic frequency tilings (`beamax.decomposition`) and the multiscale wave-packet transform (`beamax.transforms.MSWPT`).
- Gaussian beam core (`beamax.gb`): field evaluators, ODE solvers for ray trajectories and amplitude evolution, and utility matrix operations on the Hamiltonian.
- High-level solver classes (`beamax.solvers`) sharing a common `forward` / `time_reversal` / `adjoint` interface:
  - `MSGBSolver` — multiscale Gaussian beams.
  - `KWaveSolver` — optional k-Wave reference backend behind the `kwave` extra.
  - `HybridSolver` — MSGB for high-frequency content combined with a low-frequency solver.
  - Optional FNO adapters (`FNONeuralOpsSolver`, `FNOpdequinoxSolver`) behind the `fno` extra.
- Plotting helpers (`beamax.plotter`) for wavefields, beam ellipses, and MSWPT coefficients.
- MkDocs + mkdocstrings documentation site generated from in-source docstrings.
- CI on Python 3.11 and 3.12 with ruff + pytest + example smoke tests.
- Example gallery covering forward simulation, time reversal, adjoint, ray tracing, single-beam diagnostics, and optional MSGB-vs-k-Wave comparisons.
