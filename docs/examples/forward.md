# Forward propagation

<a id="msgb-vs-k-wave"></a>

The Chapter 4 example records the mixed two-packet source at 128 receivers on a
periodic 128 × 128 grid, at 1,500 m/s with the absorbing layer disabled. One
script runs k-Wave, MSGB and Hybrid and prints the first and repeat runtime for
each. The full MSGB recording uses 1,849 coefficients for the 1% synthesis
target. Hybrid uses k-Wave on the coarse low-frequency grid and MSGB with
940 high-frequency coefficients. It tapers the extra 30 fine time samples before
Fourier interpolation, as in the thesis experiment.

The top row shows the k-Wave pressure and the two signed differences, each
computed as k-Wave minus the comparison solver. The differences share a colour
scale. The dashed line marks the interior receiver with the largest RMS
discrepancy across the two differences. All three traces appear below.
The thesis relative L2 differences over the full recording are 2.306% for
MSGB and 0.293% for Hybrid.

![Chapter 4 pressure record, both signed differences and the three traces at the marked receiver](../assets/examples/2d_forward_comparison.png)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/v0.4.0/examples/forward/2d_forward_comparison.ipynb)

```python
--8<-- "examples/forward/2d_forward_comparison.py"
```
