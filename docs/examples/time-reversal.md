# Time-reversal reconstruction

Time-reversal imaging with MSGB. Examples marked optional require
`beamax[kwave,viz-mpl]` and are skipped by the default smoke suite.

---

## 1D time reversal

1D one-sided boundary time-reversal smoke test. It records synthetic data on the right boundary, propagates that data backward, and reports compact diagnostics.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/reconstruction/time-reversal/1d_time_reversal.ipynb)

```python
--8<-- "examples/reconstruction/time-reversal/1d_time_reversal.py"
```

---

## 2D time reversal and adjoint

Optional: compare k-Wave time-reversal and adjoint reconstructions on a tiny 2D phantom.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/reconstruction/time-reversal/2d_time_reversal_and_adjoint.ipynb)

```python
--8<-- "examples/reconstruction/time-reversal/2d_time_reversal_and_adjoint.py"
```
