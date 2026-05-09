# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — initial public release

### Added

- JAX-first kernels for dyadic frequency tilings (`beamax.decomposition`) and the multiscale wave-packet transform (`beamax.transforms.MSWPT`).
- Gaussian-beam core (`beamax.gb`): field evaluators, ODE solvers for ray trajectories and amplitude evolution, and utility matrix operations on the Hamiltonian.
- High-level solver classes (`beamax.solvers`) sharing a common `forward` / `time_reversal` / `adjoint` interface:
  - `MSGBSolver` — multiscale Gaussian beams.
  - `KWaveSolver` — k-Wave pseudo-spectral reference, with a macOS HDF5 compatibility shim.
  - `HybridSolver` — MSGB for high-frequency content combined with a low-frequency solver.
  - Optional FNO adapters (`FNONeuralOpsSolver`, `FNOpdequinoxSolver`) behind the `fno` extra.
- Plotting helpers (`beamax.plotter`) for wavefields, beam ellipses, and MSWPT coefficients.
- MkDocs + mkdocstrings documentation site generated from in-source docstrings.
- CI on Python 3.11 and 3.12 with ruff + pytest + example smoke tests.
- Example gallery covering forward simulation, time reversal, adjoint, and MSGB-vs-k-Wave comparisons.
