# Examples

Start with [wave-packet selection](wave-packets/2d_packet_selection.py) using
the base installation. Then follow [forward propagation](forward/2d_forward_comparison.py)
and [reconstruction](reconstruction/2d_time_reversal_and_adjoint.py) for the
2D photoacoustic workflow. Run scripts from the repository root or open their
Colab notebooks below.

## Gallery

### Wave packets

- [`2d_packet_selection.py`](wave-packets/2d_packet_selection.py) — Recover a 2D pressure field from its strongest wave-packet coefficients. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/v0.4.0/examples/wave-packets/2d_packet_selection.ipynb)

### Forward propagation

- [`2d_forward_comparison.py`](forward/2d_forward_comparison.py) — Compare MSGB and k-Wave sensor data for a 2D wave packet. _(requires `beamax[kwave,viz-mpl]`; excluded from default smoke)_ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/v0.4.0/examples/forward/2d_forward_comparison.ipynb)

### Reconstruction

- [`2d_time_reversal_and_adjoint.py`](reconstruction/2d_time_reversal_and_adjoint.py) — Compare 2D MSGB and k-Wave time-reversal and adjoint reconstructions. _(requires `beamax[kwave,viz-mpl]`; excluded from default smoke)_ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/v0.4.0/examples/reconstruction/2d_time_reversal_and_adjoint.ipynb)

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
