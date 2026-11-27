# How many wave packets describe an initial pressure field?

Before propagating beams, MSGB represents the initial pressure field as a sum
of localized, oscillating packets. Try the multiscale wave-packet transform
(MSWPT) on one such field. This example runs on CPU with the base installation
of Beamax.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/v0.4.0/examples/wave-packets/2d_packet_selection.ipynb)

On a float32 CPU run the relative L2 error in the pressure field fell from
**0.41 with 16 coefficients** to **0.20 with 64**. All 4,096 coefficients
recovered the input to within `2.3e-7`. Rotate the carrier or narrow the
envelope and see what happens at the same coefficient budget.

```python
--8<-- "examples/wave-packets/2d_packet_selection.py"
```

To see how a packet approximation affects a measurement, continue to the
[forward simulation](forward.md#msgb-vs-k-wave).
