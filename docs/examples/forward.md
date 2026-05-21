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

---

## 2D photoacoustic forward

Optional: run the thesis two-packet homogeneous PAT setup on a small public
grid. The script saves setup panels for $p_0$ / MSWPT coefficients and a
sensor-data comparison between k-Wave, MSGB, and Hybrid.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/forward/2d_forward.ipynb)

```python
--8<-- "examples/forward/2d_forward.py"
```
