# Forward propagation

## MSGB vs k-Wave

Launch an oscillating Gaussian pressure packet on a 48 × 48 grid with a
constant sound speed of 1,500 m/s. Both solvers record at the same 48
receivers. Each column of the resulting array is one receiver's pressure
trace over time. Requires `beamax[kwave,viz-mpl]`.

The plot overlays the traces at the receiver with the strongest k-Wave signal.
The printed relative L2 difference uses all 48 receivers and 227 time samples.
With 512 selected coefficients the difference was 1.14% on CPU.

![Two pressure traces at the receiver with the strongest signal](../assets/examples/2d_forward_comparison.png)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/v0.4.0/examples/forward/2d_forward_comparison.ipynb)

```python
--8<-- "examples/forward/2d_forward_comparison.py"
```
