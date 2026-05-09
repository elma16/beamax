# Adjoint reconstruction

Initial-pressure reconstruction using the MSGB adjoint operator. Three variants of a 2D photoacoustic setup plus autodiff cross-checks in 1D / 2D / 3D.

---

## ad-2d

2D initial-pressure reconstruction from k-Wave sensor data using the MSGB adjoint.

```python
--8<-- "examples/reconstruction/adjoint/AD-2d.py"
```

---

## ad-2d2

Linearised 2D MSGB adjoint with autodiff (reorganised pipeline).

```python
--8<-- "examples/reconstruction/adjoint/AD-2d2.py"
```

---

## ad-2d3

Full 2D MSGB forward with autodiff — matches the `MSGBSolver` idiom used in `forward-2d` and supports a time-reversal warm start.

```python
--8<-- "examples/reconstruction/adjoint/AD-2d3.py"
```

---

## autodiff-1d

1D adjoint via JAX autodiff, as an independent check on the analytic MSGB adjoint.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/reconstruction/adjoint/autodiff-1d.ipynb)

```python
--8<-- "examples/reconstruction/adjoint/autodiff-1d.py"
```

---

## autodiff-2d

2D adjoint via JAX autodiff.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/reconstruction/adjoint/autodiff-2d.ipynb)

```python
--8<-- "examples/reconstruction/adjoint/autodiff-2d.py"
```

---

## autodiff-3d

3D adjoint via JAX autodiff. Memory-heavy — run on a Colab GPU/TPU runtime for best results.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/reconstruction/adjoint/autodiff-3d.ipynb)

```python
--8<-- "examples/reconstruction/adjoint/autodiff-3d.py"
```
