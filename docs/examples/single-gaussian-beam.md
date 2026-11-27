# Single Gaussian beam diagnostics

## Single Gaussian beam absorption

Follow a 1D Gaussian beam in a constant-speed medium. Run the lossless case
first, then set the beam damping rate to `lam=5.0` and convert it to the
absorption coefficient used by k-Wave. Compare the space-time pressure plots
and the relative L2 differences from the same initial field.
Requires `beamax[kwave,viz-mpl]`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/v0.4.0/examples/single-gaussian-beam/single_gaussian_beam_absorption.ipynb)

```python
--8<-- "examples/single-gaussian-beam/single_gaussian_beam_absorption.py"
```
