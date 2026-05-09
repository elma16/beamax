# Examples

This directory holds the beamax example gallery. Almost every script has a
matching notebook with an **Open in Colab** badge so you can run it on a free
GPU or TPU without any local setup — click the badge, then pick a GPU or TPU
runtime in Colab (`Runtime → Change runtime type`).

Each notebook installs beamax from this repository in its first code cell:

```
%pip install --quiet "beamax[kwave] @ git+https://github.com/elma16/beamax.git"
```

When running locally from a checkout, that cell is a no-op (skip it, or leave
it — `pip install` will simply reinstall the current working copy).

A few examples need additional data — most notably the OA-Breast phantom from
the [Illinois OA-Breast database](https://anastasio.bioengineering.illinois.edu/downloadable-content/oa-breast-database/).
Download `Neg_07_Left.h5` from there and place it under
`<repo-root>/data/NumericalBreastPhantoms-selected/hdf5/`. Notebooks that
need this data carry a banner pointing here.

## Style

Examples that customise matplotlib import `use_beamax_style` from
`beamax.plotter` — the style file is bundled inside the installed package, so
it resolves identically in a checkout, an installed wheel, or on Colab.

## Gallery

### Frequency decomposition & MSWPT

- [`frames-figure.py`](examples/decomp/frames-figure.py) — Render the MSWPT frame atoms for a given dyadic decomposition, illustrating the multiscale tiling. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/decomp/frames-figure.ipynb)
- [`frames_grid.py`](examples/decomp/frames_grid.py) — Plot a 3×3 grid of representative atoms for: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/decomp/frames_grid.ipynb)
- [`lp_hp_filter.py`](examples/decomp/lp_hp_filter.py) — Low-pass / high-pass frequency separation built from MSWPT filters. Also part of the CI example smoke suite. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/decomp/lp_hp_filter.ipynb)
- [`mswpt_error_plot.py`](examples/decomp/mswpt_error_plot.py) — Reconstruction error of the MSWPT forward+inverse pipeline as a function of box count and redundancy. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/decomp/mswpt_error_plot.ipynb)

### Forward propagation

- [`forward-1d-v0.py`](examples/forward/forward-1d-v0.py) — Forward solve with non-zero initial velocity (v0) compared to a spectral ground truth. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/forward/forward-1d-v0.ipynb)
- [`forward-2d-v0.py`](examples/forward/forward-2d-v0.py) — 2D forward solve with non-zero initial velocity, compared to a spectral ground truth. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/forward/forward-2d-v0.ipynb)
- [`forward-2d.py`](examples/forward/forward-2d.py) — 2D forward solve comparing MSGB against k-Wave and the hybrid MSGB + low-frequency solver. Requires `[kwave]`. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/forward/forward-2d.ipynb)
- [`forward-3d.py`](examples/forward/forward-3d.py) — 3D forward solve on an OA-Breast phantom comparing MSGB, k-Wave, and the hybrid MSGB + low-frequency solver on a planar sensor array, with orthogonal MIP figures and relative L2 error reporting. Requires `[kwave]`. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/forward/forward-3d.ipynb)

### Time-reversal reconstruction

- [`TR-1d.py`](examples/reconstruction/time-reversal/TR-1d.py) — 1D time-reversal reconstruction with MSGB vs. k-Wave. Part of the CI example smoke suite. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/reconstruction/time-reversal/TR-1d.ipynb)
- [`TR-2d.py`](examples/reconstruction/time-reversal/TR-2d.py) — 2D time-reversal comparison between MSGB and k-Wave. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/reconstruction/time-reversal/TR-2d.ipynb)
- [`TR-3d2.py`](examples/reconstruction/time-reversal/TR-3d2.py) — 3D Time Reversal Diagnostic Script [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/reconstruction/time-reversal/TR-3d2.ipynb)
- [`fwd-tr-img.py`](examples/reconstruction/time-reversal/fwd-tr-img.py) — 2D k-Wave forward + time-reversal reconstruction on an OA-Breast phantom slice, plotting p0, the sensor measurement, and the time-reversal estimate side by side. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/reconstruction/time-reversal/fwd-tr-img.ipynb)

### Adjoint reconstruction

- [`AD-2d.py`](examples/reconstruction/adjoint/AD-2d.py) — Reconstruct p0 from k-Wave data using MSGB: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/reconstruction/adjoint/AD-2d.ipynb)
- [`AD-2d2.py`](examples/reconstruction/adjoint/AD-2d2.py) — 2D PAT inverse problem with a *linearised* MSGB forward map and JAX autodiff. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/reconstruction/adjoint/AD-2d2.ipynb)
- [`AD-2d3.py`](examples/reconstruction/adjoint/AD-2d3.py) — 2D PAT inverse problem with the full MSGB forward map and JAX autodiff. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/reconstruction/adjoint/AD-2d3.ipynb)
- [`autodiff-1d.py`](examples/reconstruction/adjoint/autodiff-1d.py) — 1D adjoint via JAX autodiff, used as an independent check on the analytic MSGB adjoint. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/reconstruction/adjoint/autodiff-1d.ipynb)
- [`autodiff-2d.py`](examples/reconstruction/adjoint/autodiff-2d.py) — 2D adjoint via JAX autodiff. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/reconstruction/adjoint/autodiff-2d.ipynb)
- [`autodiff-3d.py`](examples/reconstruction/adjoint/autodiff-3d.py) — 3D adjoint via JAX autodiff. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/reconstruction/adjoint/autodiff-3d.ipynb)

### Inverse-operator comparisons

- [`adj-vs-TR-boxtest.py`](examples/reconstruction/comparison/adj-vs-TR-boxtest.py) — Adjoint vs. time-reversal reconstruction on a synthetic box phantom, contrasting the two inverse operators. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/reconstruction/comparison/adj-vs-TR-boxtest.ipynb)
- [`breast_tr_adj_faces.py`](examples/reconstruction/comparison/breast_tr_adj_faces.py) — Create a 3-column figure comparing k-Wave time-reversal and adjoint [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/reconstruction/comparison/breast_tr_adj_faces.ipynb)
- [`iterative-adj-tr.py`](examples/reconstruction/comparison/iterative-adj-tr.py) — Iterative refinement combining MSGB adjoint and time-reversal updates. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/reconstruction/comparison/iterative-adj-tr.ipynb)
- [`kwave-TR-vs-Adj.py`](examples/reconstruction/comparison/kwave-TR-vs-Adj.py) — Reference comparison of k-Wave time-reversal against the k-Wave adjoint. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/reconstruction/comparison/kwave-TR-vs-Adj.ipynb)

### Benchmarks

- [`gb-vs-kw-runtime.py`](examples/benchmarks/gb-vs-kw-runtime.py) — Runtime comparison between the k-Wave forward solver and beamax's MSGB solver across a grid-size sweep. Requires `[kwave]` extra. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/benchmarks/gb-vs-kw-runtime.ipynb)
- [`memory_test.py`](examples/benchmarks/memory_test.py) — Measure peak memory usage of an MSGB forward solve, annotated for CPU and GPU runs. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/benchmarks/memory_test.ipynb)
- [`memory_test_detailed.py`](examples/benchmarks/memory_test_detailed.py) — Comprehensive profiling script that accounts for ALL time including JAX async ops. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/benchmarks/memory_test_detailed.ipynb)
- [`strong-weak-scaling.py`](examples/benchmarks/strong-weak-scaling.py) — Aim: Work out the scalability of the ODE solver across multiple devices. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/benchmarks/strong-weak-scaling.ipynb)
- [`wpt-decomp-runtime.py`](examples/benchmarks/wpt-decomp-runtime.py) — Runtime of the MSWPT forward and inverse transforms as a function of grid size and decomposition depth. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/benchmarks/wpt-decomp-runtime.ipynb)

### Single Gaussian beam diagnostics

- [`singleGBenergy.py`](examples/singleGB/singleGBenergy.py) — Track energy conservation along a single Gaussian beam trajectory. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/singleGB/singleGBenergy.ipynb)
- [`singleGBplot.py`](examples/singleGB/singleGBplot.py) — Visualise a single Gaussian beam's amplitude and ellipse over time. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/singleGB/singleGBplot.ipynb)
- [`singleGBpropagation.py`](examples/singleGB/singleGBpropagation.py) — Propagate a single Gaussian beam through a homogeneous medium. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/singleGB/singleGBpropagation.ipynb)
- [`singleGBrayleigh-error.py`](examples/singleGB/singleGBrayleigh-error.py) — Single Gaussian beam Rayleigh-range / focal-error diagnostic. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/singleGB/singleGBrayleigh-error.ipynb)
- [`singleGBwpt.py`](examples/singleGB/singleGBwpt.py) — Single Gaussian beam wave-packet transform diagnostic. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/singleGB/singleGBwpt.ipynb)

### Ray tracing & Hamiltonian

- [`_curves.py`](examples/rays/_curves.py) — Ray-tracing diagnostic for the Gaussian beam Hamiltonian. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/rays/_curves.ipynb)
- [`_velocitymaps.py`](examples/rays/_velocitymaps.py) — Ray-tracing diagnostic for the Gaussian beam Hamiltonian. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/rays/_velocitymaps.ipynb)
- [`col.py`](examples/rays/col.py) — Ray-tracing diagnostic for the Gaussian beam Hamiltonian. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/rays/col.ipynb)
- [`rays-stiffness.py`](examples/rays/rays-stiffness.py) — Investigate the stiffness of the Gaussian beam ray ODEs. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/rays/rays-stiffness.ipynb)
- [`rays2d-anim.py`](examples/rays/rays2d-anim.py) — Animation of ray trajectories propagating through a 2D/3D medium. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/rays/rays2d-anim.ipynb)
- [`rays2d-animTR.py`](examples/rays/rays2d-animTR.py) — Animation of ray trajectories propagating through a 2D/3D medium. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/rays/rays2d-animTR.ipynb)
- [`rays2d.py`](examples/rays/rays2d.py) — Ray-tracing diagnostic for the Gaussian beam Hamiltonian. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/rays/rays2d.ipynb)
- [`rays3d.py`](examples/rays/rays3d.py) — Ray-tracing diagnostic for the Gaussian beam Hamiltonian. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/rays/rays3d.ipynb)
- [`rays3d2.py`](examples/rays/rays3d2.py) — Ray-tracing diagnostic for the Gaussian beam Hamiltonian. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/rays/rays3d2.ipynb)

### Bowtie sensor configurations

- [`angle_mapping.py`](examples/bowtie/angle_mapping.py) — Angle-mapping experiment for planar line-sensor PAT data. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/bowtie/angle_mapping.ipynb)
- [`test2d.py`](examples/bowtie/test2d.py) — Planar line-sensor wave data: 2D FFT in (sensor coordinate, time) and support-energy analysis. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/bowtie/test2d.ipynb)
- [`test3d.py`](examples/bowtie/test3d.py) — Sweep CFL values for a 3D delta-source k-Wave run and measure the energy fractions of the planar-sensor (ky, kz, omega) spectrum that fall inside the continuum cone and curved Nyquist cap, diagnosing temporal aliasing as CFL crosses 1/sqrt(3). [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/bowtie/test3d.ipynb)
- [`test3d_aliasing.py`](examples/bowtie/test3d_aliasing.py) — Compare a coarse-CFL 3D k-Wave planar-sensor spectrum against an aliased fine-CFL reference, quantifying how much of the observed cone violation is explained purely by temporal aliasing. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/bowtie/test3d_aliasing.ipynb)

## Contributing a new example

1. Add the script under the appropriate `examples/<category>/` directory with
   a 1-2 sentence module docstring.
2. Run `python tools/finalize_examples.py` (or hand-edit a notebook) so a
   paired `.ipynb` exists with the Open-in-Colab badge + install cell pattern.
3. Add a bullet to the section above (or rerun the regeneration script).
4. If the example needs significant RAM or external data, mention it in the
   docstring so the auto-generated notebook can surface a warning.
