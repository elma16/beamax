# Forward propagation

MSGB forward solves on 1D and 2D domains, benchmarked against spectral reference solutions and k-Wave.

---

## forward-1d-v0

1D forward solve with non-zero initial velocity (Cauchy data), compared against a spectral ground truth. Runs as a CI smoke test — relative L2 error against the spectral solution is reported.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/forward/forward-1d-v0.ipynb)

```python
--8<-- "examples/forward/forward-1d-v0.py"
```

---

## forward-2d-v0

2D forward solve on a periodic box with constant sound speed and trigonometric initial data; plots snapshots and error maps against the spectral solution.

```python
--8<-- "examples/forward/forward-2d-v0.py"
```

---

## forward-2d

2D forward solve comparing MSGB against k-Wave and the hybrid MSGB + low-frequency solver. Requires `[kwave]`.

```python
--8<-- "examples/forward/forward-2d.py"
```
