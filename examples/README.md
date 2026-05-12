# Examples

This directory holds the supported beamax example gallery. Base examples are
small, documented, paired with notebooks, linted, and smoke-tested in CI.
Examples marked optional require extra dependencies and are skipped by default
smoke runs.

`private/` preserves research, profiling, comparison, and diagnostic scripts.
They may require extra data, optional solver backends, large memory, or local
hardware assumptions, and are not part of the public docs or CI smoke suite.

Every public script has a matching notebook with an **Open in Colab** badge.
The public examples are small enough to run on a standard CPU Colab runtime.

Each notebook installs beamax from this repository in its first code cell:

```
%pip install --quiet "beamax[viz-mpl] @ git+https://github.com/elma16/beamax.git"
```

When running locally from a checkout, that cell can be skipped.

## Style

Examples that customise matplotlib import `use_beamax_style` from
`beamax.plotter` — the style file is bundled inside the installed package, so
it resolves identically in a checkout, an installed wheel, or on Colab.

## Gallery

### Frequency decomposition & MSWPT

- [`low_high_pass_filters.py`](decomposition/low_high_pass_filters.py) — Visualise the low-pass/high-pass filter pairs used by the MSWPT. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/decomposition/low_high_pass_filters.ipynb)
- [`wave_packet_cutoff_error.py`](decomposition/wave_packet_cutoff_error.py) — Plot MSWPT frame cutoff error across dyadic scales. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/decomposition/wave_packet_cutoff_error.ipynb)
- [`wave_packet_frame_atoms.py`](decomposition/wave_packet_frame_atoms.py) — Render a small grid of MSWPT frame atoms in Fourier space. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/decomposition/wave_packet_frame_atoms.ipynb)

### Forward propagation

- [`1d_forward_solve.py`](forward/1d_forward_solve.py) — Forward solve with non-zero initial velocity (v0) compared to a spectral ground truth. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/forward/1d_forward_solve.ipynb)
- [`2d_forward_solve.py`](forward/2d_forward_solve.py) — 2D forward solve with non-zero initial velocity, compared to a spectral ground truth. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/forward/2d_forward_solve.ipynb)
- [`1d_forward_kwave_reference.py`](forward/1d_forward_kwave_reference.py) — Compare a compact 1D MSGB forward solve with a k-Wave strip reference. _(optional; requires `beamax[kwave,viz-mpl]`; skipped by default smoke)_ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/forward/1d_forward_kwave_reference.ipynb)
- [`2d_forward_kwave_reference.py`](forward/2d_forward_kwave_reference.py) — Compare a small 2D MSGB forward solve with a k-Wave boundary-sensor reference. _(optional; requires `beamax[kwave,viz-mpl]`; skipped by default smoke)_ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/forward/2d_forward_kwave_reference.ipynb)

### Time-reversal reconstruction

- [`1d_time_reversal.py`](reconstruction/time-reversal/1d_time_reversal.py) — Run a compact 1D MSGB time-reversal smoke test. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/reconstruction/time-reversal/1d_time_reversal.ipynb)
- [`2d_time_reversal_and_adjoint.py`](reconstruction/time-reversal/2d_time_reversal_and_adjoint.py) — Compare k-Wave time-reversal and adjoint reconstructions on a tiny 2D phantom. _(optional; requires `beamax[kwave,viz-mpl]`; skipped by default smoke)_ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/reconstruction/time-reversal/2d_time_reversal_and_adjoint.ipynb)

### Rays and autofocus

- [`2d_ray_bending.py`](rays/2d_ray_bending.py) — Trace a small fan of 2D rays through a smooth speed field. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/rays/2d_ray_bending.ipynb)
- [`neural_sound_speed_autofocus.py`](rays/neural_sound_speed_autofocus.py) — Optimize a neural sound-speed field so initially parallel rays focus. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/rays/neural_sound_speed_autofocus.ipynb)

### Single Gaussian beam diagnostics

- [`single_gaussian_beam_absorption.py`](single-gaussian-beam/single_gaussian_beam_absorption.py) — Compare lossless and absorbing single-Gaussian-beam propagation with k-Wave. _(optional; requires `beamax[kwave,viz-mpl]`; skipped by default smoke)_ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/single-gaussian-beam/single_gaussian_beam_absorption.ipynb)


## Contributing a new example

1. Add the script under the appropriate `examples/<category>/` directory with
   a 1-2 sentence module docstring.
   Use `Example extras: ...` and `Example smoke: false` for optional-runtime
   examples.
2. Run `python tools/finalize_examples.py` (or hand-edit a notebook) so a
   paired `.ipynb` exists with the Open-in-Colab badge + install cell pattern.
3. Add a bullet to the section above (or rerun the regeneration script).
4. Keep public examples self-contained and fast. Move research/profiling/data-
   dependent material to `examples/private`.
