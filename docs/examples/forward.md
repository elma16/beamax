# Forward propagation

MSGB forward solves on 1D and 2D domains, benchmarked against spectral reference
solutions. Examples marked optional require `beamax[kwave,viz-mpl]` and are
skipped by the default smoke suite.

---

## 1D forward solve

1D forward solve with non-zero initial velocity (Cauchy data), compared against a spectral ground truth. Runs as a CI smoke test — relative L2 error against the spectral solution is reported.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/forward/1d_forward_solve.ipynb)

```python
--8<-- "examples/forward/1d_forward_solve.py"
```

---

## 2D forward solve

2D forward solve on a periodic box with constant sound speed and trigonometric initial data; plots snapshots and error maps against the spectral solution.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/forward/2d_forward_solve.ipynb)

```python
--8<-- "examples/forward/2d_forward_solve.py"
```

---

## 1D forward k-Wave reference

Optional: compare a compact 1D MSGB forward solve with a k-Wave strip reference.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/forward/1d_forward_kwave_reference.ipynb)

```python
--8<-- "examples/forward/1d_forward_kwave_reference.py"
```

---

## 2D forward k-Wave reference

Optional: compare a small 2D MSGB forward solve with a k-Wave boundary-sensor reference.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/forward/2d_forward_kwave_reference.ipynb)

```python
--8<-- "examples/forward/2d_forward_kwave_reference.py"
```
