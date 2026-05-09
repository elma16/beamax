# Inverse-operator comparisons

Contrasting time-reversal and adjoint reconstructions side-by-side.

---

## adj-vs-tr-boxtest

Adjoint vs. time-reversal reconstruction on a synthetic box phantom, illustrating the characteristic differences between the two inverse operators.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/reconstruction/comparison/adj-vs-TR-boxtest.ipynb)

```python
--8<-- "examples/reconstruction/comparison/adj-vs-TR-boxtest.py"
```

---

## iterative-adj-tr

Iterative refinement combining MSGB adjoint and time-reversal updates.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/reconstruction/comparison/iterative-adj-tr.ipynb)

```python
--8<-- "examples/reconstruction/comparison/iterative-adj-tr.py"
```

---

## kwave-tr-vs-adj

Reference comparison of k-Wave time-reversal against the k-Wave adjoint. Useful as a sanity check independent of the MSGB implementation.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/reconstruction/comparison/kwave-TR-vs-adj.ipynb)

```python
--8<-- "examples/reconstruction/comparison/kwave-TR-vs-Adj.py"
```
