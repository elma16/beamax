# Benchmarks

Performance and scaling studies of beamax against k-Wave baselines.

---

## gb-vs-kw-runtime

Runtime comparison between the k-Wave forward solver and beamax's MSGB solver across a grid-size sweep. Requires `[kwave]` extra. Set `BEAMAX_FULL_BENCHMARKS=1` to expand the sweep.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/benchmarks/gb-vs-kw-runtime.ipynb)

```python
--8<-- "examples/benchmarks/gb-vs-kw-runtime.py"
```

---

## memory-test

Measure peak memory usage of an MSGB forward solve, annotated for CPU and GPU runs.

```python
--8<-- "examples/benchmarks/memory_test.py"
```

---

## memory-test-detailed

Detailed per-phase memory and time profiling of an MSGB forward solve, accounting for JAX asynchronous dispatch.

```python
--8<-- "examples/benchmarks/memory_test_detailed.py"
```

---

## strong-weak-scaling

Strong- and weak-scaling study of MSGB beam parallelism. Best run on a Colab TPU runtime to see the multi-device speedup.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/benchmarks/strong-weak-scaling.ipynb)

```python
--8<-- "examples/benchmarks/strong-weak-scaling.py"
```

---

## wpt-decomp-runtime

Runtime of the MSWPT forward and inverse transforms as a function of grid size and decomposition depth.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/benchmarks/wpt-decomp-runtime.ipynb)

```python
--8<-- "examples/benchmarks/wpt-decomp-runtime.py"
```
