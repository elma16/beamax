# Examples

Compare MSGB and k-Wave on a 48 × 48 grid in three standalone Colab notebooks.

| Example | What you will see | Notebook |
| --- | --- | --- |
| [Forward propagation](forward.md) | The pressure trace at one receiver and the relative difference across all 48 receivers. | [Open in Colab](https://colab.research.google.com/github/elma16/beamax/blob/v0.4.0/examples/forward/2d_forward_comparison.ipynb) |
| [Time reversal](reconstruction.md#time-reversal) | Two initial pressure sources reconstructed by propagating their recordings backwards. | [Open in Colab](https://colab.research.google.com/github/elma16/beamax/blob/v0.4.0/examples/reconstruction/2d_time_reversal.ipynb) |
| [Adjoint](reconstruction.md#adjoint) | The image each solver produces when its adjoint acts on the same detector recordings. | [Open in Colab](https://colab.research.google.com/github/elma16/beamax/blob/v0.4.0/examples/reconstruction/2d_adjoint.ipynb) |

Each notebook prints the elapsed time for the first solver call and an immediate
repeat. Times include any JAX compilation during the call and the transfer of the
result to the host. MSGB uses the selected JAX device with single-precision FFTs.
The Python k-Wave backend runs on CPU. Each example displays its plot in Colab and
saves a PNG under `plots/`.

To run the scripts locally, install the example dependencies from the checkout.

```sh
python -m pip install -e ".[kwave,viz-mpl]"
python examples/forward/2d_forward_comparison.py
python examples/reconstruction/2d_time_reversal.py
python examples/reconstruction/2d_adjoint.py
```

The [wave-packet tutorial](wave-packets.md) introduces the transform using the
base installation. On an Apple silicon Mac with `jax-mps` installed, use
`JAX_PLATFORMS=cpu` for the reconstruction examples. For a SciPy import error
on macOS 27, see the [local notebook setup](../guides/kwave-metal.md#scipy-import-error-on-an-apple-silicon-mac).

For a ring or an open detector arc, the [detector surface guide](../api/surfaces.md)
explains how to order your measured traces and reconstruct a region of
interest. To combine k-Wave at low frequencies with MSGB at high frequencies,
see the [hybrid backend guide](../guides/custom-low-frequency-solvers.md).
