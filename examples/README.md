# Examples

Compare k-Wave, MSGB and Hybrid in the [forward notebook](forward/2d_forward_comparison.ipynb).
Run [time reversal and adjoint](reconstruction/2d_time_reversal_and_adjoint.ipynb)
on a single box-scanner acquisition, or run either operation separately.
Each notebook prints first-call and repeat runtimes.

## Gallery

### Wave packets

- [`2d_packet_selection.py`](wave-packets/2d_packet_selection.py) — Recover a 2D pressure field from its strongest wave-packet coefficients. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/v0.4.0/examples/wave-packets/2d_packet_selection.ipynb)

### Forward propagation

- [`2d_forward_comparison.py`](forward/2d_forward_comparison.py) — Compare k-Wave, MSGB and Hybrid records for the thesis mixed two-packet source. _(requires `beamax[kwave,viz-mpl]`; excluded from default smoke)_ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/v0.4.0/examples/forward/2d_forward_comparison.ipynb)

### Reconstruction

- [`2d_adjoint.py`](reconstruction/2d_adjoint.py) — Reproduce the adjoint panels for the first face of the thesis box scanner. _(requires `beamax[kwave,viz-mpl]`; excluded from default smoke)_ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/v0.4.0/examples/reconstruction/2d_adjoint.ipynb)
- [`2d_time_reversal.py`](reconstruction/2d_time_reversal.py) — Reproduce the time-reversal panels for the first face of the thesis box scanner. _(requires `beamax[kwave,viz-mpl]`; excluded from default smoke)_ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/v0.4.0/examples/reconstruction/2d_time_reversal.ipynb)
- [`2d_time_reversal_and_adjoint.py`](reconstruction/2d_time_reversal_and_adjoint.py) — Compare time reversal and adjoint on the first face of the thesis box scanner. _(requires `beamax[kwave,viz-mpl]`; excluded from default smoke)_ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/v0.4.0/examples/reconstruction/2d_time_reversal_and_adjoint.ipynb)

### Rays and autodiff

- [`2d_ray_bending.py`](rays/2d_ray_bending.py) — Trace 2D rays through a smooth field with $G(\mathbf{x},\mathbf{p})=c(\mathbf{x})|\mathbf{p}|$. _(requires `beamax[viz-mpl]`; excluded from default smoke)_ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/v0.4.0/examples/rays/2d_ray_bending.ipynb)
- [`2d_rays_autodiff.py`](rays/2d_rays_autodiff.py) — Optimise a neural sound-speed field through the Gaussian beam ray ODE. _(requires `beamax[viz-mpl,autodiff]`; excluded from default smoke)_ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/v0.4.0/examples/rays/2d_rays_autodiff.ipynb)

### Single Gaussian beam diagnostics

- [`single_gaussian_beam_absorption.py`](single-gaussian-beam/single_gaussian_beam_absorption.py) — Compare lossless and absorbing Gaussian beams with k-Wave. _(requires `beamax[kwave,viz-mpl]`; excluded from default smoke)_ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/v0.4.0/examples/single-gaussian-beam/single_gaussian_beam_absorption.ipynb)

### Diagnostics

- [`memory_planning.py`](diagnostics/memory_planning.py) — Compare static MSGB memory plans against a device budget.

## Smoke testing

Examples marked `Example smoke: false` are skipped by default. To include them:

```bash
python tools/run_examples.py --directory examples --include-optional --silent-figures
```
