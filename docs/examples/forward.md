# Forward propagation

MSGB forward solves and hybrid forward examples. k-Wave reference examples
require `beamax[kwave,viz-mpl]` and are skipped by the default smoke suite.

---

## Custom low-frequency backend

Run a 1D hybrid solve with MSGB for high frequencies and a tiny pure-JAX
spectral low-frequency backend.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/forward/custom_lf_spectral_backend.ipynb)

```python
--8<-- "examples/forward/custom_lf_spectral_backend.py"
```
