# Frequency decomposition

Examples illustrating beamax's dyadic frequency tilings and the multiscale wave-packet transform (MSWPT). All of these depend only on `beamax` core + `[viz]` (`matplotlib`).

---

## Wave packet frame atoms

Render the MSWPT frame atoms for a given dyadic decomposition, illustrating the multiscale tiling.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/decomposition/wave_packet_frame_atoms.ipynb)

```python
--8<-- "examples/decomposition/wave_packet_frame_atoms.py"
```

---

## Low and high pass filters

Low-pass / high-pass frequency separation built from MSWPT filters. Also part of the CI example smoke suite — serves as a fast end-to-end check that the transform chain is working.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/decomposition/low_high_pass_filters.ipynb)

```python
--8<-- "examples/decomposition/low_high_pass_filters.py"
```

---

## Wave packet cutoff error

Reconstruction error of the MSWPT forward+inverse pipeline as a function of box count and redundancy.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/decomposition/wave_packet_cutoff_error.ipynb)

```python
--8<-- "examples/decomposition/wave_packet_cutoff_error.py"
```
