# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] — 2026-07-14

### Fixed

- Standardised standalone k-Wave sensor channels to NumPy C order, including planar 3D detector data used by TR and adjoint solvers.
- Made grid-valued medium fields usable by Gaussian-beam ray solvers and passed evaluated absorption fields to k-Wave.
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
