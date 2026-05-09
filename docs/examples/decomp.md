# Frequency decomposition

Examples illustrating beamax's dyadic frequency tilings and the multiscale wave-packet transform (MSWPT). All of these depend only on `beamax` core + `[viz]` (`matplotlib`).

---

## frames-figure

Render the MSWPT frame atoms for a given dyadic decomposition, illustrating the multiscale tiling.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/decomp/frames-figure.ipynb)

```python
--8<-- "examples/decomp/frames-figure.py"
```

---

## frames-grid

3×3 grid of representative MSWPT atoms across coarse, mid, and fine scales.

```python
--8<-- "examples/decomp/frames_grid.py"
```

---

## lp-hp-filter

Low-pass / high-pass frequency separation built from MSWPT filters. Also part of the CI example smoke suite — serves as a fast end-to-end check that the transform chain is working.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/decomp/lp_hp_filter.ipynb)

```python
--8<-- "examples/decomp/lp_hp_filter.py"
```

---

## mswpt-error-plot

Reconstruction error of the MSWPT forward+inverse pipeline as a function of box count and redundancy.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/decomp/mswpt_error_plot.ipynb)

```python
--8<-- "examples/decomp/mswpt_error_plot.py"
```
