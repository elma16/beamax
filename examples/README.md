# Examples

This directory holds the supported beamax example gallery. Base examples are
small, documented, paired with notebooks, linted, and smoke-tested in CI.
Every public script has a matching notebook with an **Open in Colab** badge.
The public examples are small enough to run on a standard CPU Colab runtime.

Each notebook installs beamax from this repository in its first code cell:

```
%pip install --quiet "beamax[viz-mpl] @ git+https://github.com/elma16/beamax.git"
```

When running locally from a checkout, that cell can be skipped.

Local script outputs are written under `plots/<category>/`, mirroring the
script's directory under `examples/`.

## Gallery

### Forward propagation

- [`custom_lf_spectral_backend.py`](forward/custom_lf_spectral_backend.py) — Custom low-frequency backend for HybridSolver. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/forward/custom_lf_spectral_backend.ipynb)
- [`2d_forward.py`](forward/2d_forward.py) — 2D photoacoustic forward comparison with MSGB, Hybrid, and k-Wave. _(optional; requires `beamax[kwave,viz-mpl]`; skipped by default smoke)_ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/forward/2d_forward.ipynb)

### Reconstruction

- [`2d_time_reversal_and_adjoint.py`](reconstruction/2d_time_reversal_and_adjoint.py) — 2D MSGB vs k-Wave reconstruction: time reversal + adjoint. _(optional; requires `beamax[kwave,viz-mpl]`; skipped by default smoke)_ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/reconstruction/2d_time_reversal_and_adjoint.ipynb)

### Rays and autodiff

- [`2d_ray_bending.py`](rays/2d_ray_bending.py) — Trace a small fan of 2D rays through a smooth speed field. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/rays/2d_ray_bending.ipynb)
- [`2d_rays_autodiff.py`](rays/2d_rays_autodiff.py) — Differentiate through 2D Gaussian beam rays. _(optional; requires `beamax[viz-mpl,autodiff]`; skipped by default smoke)_ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/rays/2d_rays_autodiff.ipynb)

### Single Gaussian beam diagnostics

- [`single_gaussian_beam_absorption.py`](single-gaussian-beam/single_gaussian_beam_absorption.py) — Single Gaussian beam with viscous absorption: MSGB vs k-Wave. _(optional; requires `beamax[kwave,viz-mpl]`; skipped by default smoke)_ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/single-gaussian-beam/single_gaussian_beam_absorption.ipynb)


## Smoke Testing

The local default smoke command runs the base examples and skips only examples
marked `Example smoke: false`. These are skipped because they require optional
runtime extras, not because they are unsupported.

CI installs the k-Wave and matplotlib extras and runs all public examples with:

```bash
python tools/run_examples.py --directory examples --include-optional --silent-figures
```

Optional examples skipped by default:

- [`forward/2d_forward.py`](forward/2d_forward.py) — requires `beamax[kwave,viz-mpl]`.
- [`rays/2d_rays_autodiff.py`](rays/2d_rays_autodiff.py) — requires `beamax[viz-mpl,autodiff]`.
- [`reconstruction/2d_time_reversal_and_adjoint.py`](reconstruction/2d_time_reversal_and_adjoint.py) — requires `beamax[kwave,viz-mpl]`.
- [`single-gaussian-beam/single_gaussian_beam_absorption.py`](single-gaussian-beam/single_gaussian_beam_absorption.py) — requires `beamax[kwave,viz-mpl]`.


## Contributing a new example

1. Add the script under the appropriate `examples/<category>/` directory with
   a 1-2 sentence module docstring.
   Use `Example extras: ...` and `Example smoke: false` for optional-runtime
   examples.
2. Run `python tools/finalize_examples.py` (or hand-edit a notebook) so a
   paired `.ipynb` exists with the Open-in-Colab badge + install cell pattern.
3. Add a bullet to the section above (or rerun the regeneration script).
4. Keep public examples self-contained and fast. Keep research/profiling/data-
   dependent material outside the tracked public gallery.
