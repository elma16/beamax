# beamax

**Multiscale Gaussian Beams in JAX** — dyadic Fourier tilings, multiscale wave-packet transforms, and acoustic forward/reconstruction solvers.

## What problem does this solve?

Simulating the acoustic wave equation on large, smoothly varying media is expensive with conventional grid-based solvers: runtime and memory scale poorly in 3D. beamax implements the fast multiscale Gaussian wavepacket transform and **Multiscale Gaussian Beams (MSGB)** method of [Qian and Ying (2010)](https://doi.org/10.1137/100787313), representing the wavefield as a superposition of many narrow, high-frequency beams that evolve independently along ray trajectories — which parallelises naturally and decouples accuracy from grid resolution at high frequencies.

`beamax` implements MSGB end-to-end in JAX, with companion tooling for the multiscale wave-packet transform (MSWPT) that bridges pixel-domain signals and beam-domain coefficients, and optional hybrid schemes that combine MSGB on the high-frequency content with a user-supplied low-frequency solver.

## Library at a glance

- **`beamax.decomposition`** — dyadic frequency tilings (the `DyadicDecomposition` data object).
- **`beamax.transforms`** — the multiscale wave-packet transform (`MSWPT.forward` / `MSWPT.inverse`), filter construction, frame analysis.
- **`beamax.geometry`** — `Domain` (grid, spacing, wave speed, periodicity) and `Sensor` containers.
- **`beamax.gb`** — low-level Gaussian beam kernels and ODE solvers for ray trajectories and amplitude evolution.
- **`beamax.solvers`** — high-level solver classes (`MSGBSolver`, `HybridSolver`, optional `KWaveSolver` and FNO adapters) sharing a common interface for forward, adjoint, and time-reversal operations.
- **`beamax.plotter`** — matplotlib / pyvista helpers for wavefields, beam trajectories, and MSWPT coefficients.
- **`beamax.utils`** — FFT/interpolation primitives, device placement, profiling, OA-Breast phantom loader.

## Start here

- **New users** — see [Getting Started](getting-started.md) for installation, a minimal 1D forward solve, and a tour of the solver hierarchy.
- **API reference** — use the navigation tree under **Basic API**; every module page is generated from the in-source docstrings via `mkdocstrings`.

## References and related projects

- Jianliang Qian and Lexing Ying, ["Fast Multiscale Gaussian Wavepacket Transforms and Multiscale Gaussian Beams for the Wave Equation"](https://doi.org/10.1137/100787313), *Multiscale Modeling & Simulation*, 8(5), 1803-1837, 2010.
- [j-Wave](https://github.com/ucl-bug/jwave) is a differentiable acoustic simulator in JAX.

## Build this site locally

```bash
pip install -e .[dev]
mkdocs serve  # http://127.0.0.1:8000
```

Docstring changes in `src/beamax` are picked up on save.
