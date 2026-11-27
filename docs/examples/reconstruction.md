# Reconstruction

## 2D time reversal and adjoint

A detector line on one side of a 64 × 64 grid records the signal from two
Gaussian pressure sources of opposite sign. k-Wave generates the measurements.
Time reversal propagates the recordings backwards. The PAT adjoint maps
receiver data back to the image using the adjoint of the measurement model.
The example applies both operations with each solver. Requires
`beamax[kwave,viz-mpl]`.

![Initial pressure, single-sided detector data, four reconstructions and their centre-line profiles](../assets/examples/2d_time_reversal_and_adjoint.png)

Compare the source locations and the widths and signs of the peaks in the
line profiles. The script resamples the records in time for MSGB and fits one
scalar per reconstructed image to the known phantom before plotting and
reporting relative L2 error. Those errors measure agreement in image shape
after scaling. Testing an adjoint identity requires a separate dot-product
test with the chosen data and image inner products.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/v0.4.0/examples/reconstruction/2d_time_reversal_and_adjoint.ipynb)

```python
--8<-- "examples/reconstruction/2d_time_reversal_and_adjoint.py"
```
