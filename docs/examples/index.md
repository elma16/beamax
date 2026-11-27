# Examples

## From initial pressure to a reconstructed image

1. **Represent an image with wave packets.** Keep the 16 strongest
   coefficients of a 32 × 32 pressure field, then 64, then all of them.
   Measure how closely each selection recovers the image. This first exercise
   needs only the base installation. [Wave-packet selection](wave-packets.md)
2. **Predict what a detector records.** Propagate one oscillating packet in a
   homogeneous medium. Read the pressure at a line of 48 receivers and compare
   the MSGB and k-Wave records sample by sample. [Forward propagation](forward.md)
3. **Reconstruct from a measurement.** Start with two pressure sources and a
   single detector line. Use one k-Wave acquisition to compare four images
   from MSGB and k-Wave time reversal and adjoint. Look at the recovered
   shape and the effect of observing from one side. [Reconstruction](reconstruction.md)

Each page includes the source and a Colab notebook. To run the same sequence
locally, use Python 3.12 or later and run these commands from the checkout.
The two solver comparisons use the Python k-Wave backend on CPU.

```sh
python -m pip install -e .
python examples/wave-packets/2d_packet_selection.py
python -m pip install -e ".[kwave,viz-mpl]"
python examples/forward/2d_forward_comparison.py
python examples/reconstruction/2d_time_reversal_and_adjoint.py
```

The two comparisons save figures under `plots/`. In Colab, open the printed
PNG path in the Files pane. Set `MPLBACKEND=Agg` for headless runs. On an
Apple silicon Mac with `jax-mps` installed, select
`JAX_PLATFORMS=cpu` for the gallery. If k-Wave fails while importing SciPy
on macOS 27, see the [local notebook setup](../guides/kwave-metal.md#scipy-import-error-on-an-apple-silicon-mac).

## Explore a specific question

| Question | Example | What to inspect |
| --- | --- | --- |
| How does spatially varying sound speed bend a ray? | [Ray bending](rays.md#2d-ray-bending) | Follow a fan of rays through a smooth acoustic lens. |
| Can I differentiate through propagation? | [Ray focusing with autodiff](rays.md#2d-rays-autodiff) | Compare the ray endpoints before and after optimizing the sound-speed field. |
| How does damping affect a single beam? | [Gaussian beam absorption](single-gaussian-beam.md) | Compare lossless and absorbing wavefields with k-Wave from the same initial pressure. |
| Will a larger run fit in memory? | [Memory planning](diagnostics.md) | Estimate memory for a 3D grid before choosing the batch size. |

For a ring or an open detector arc, the [detector surface guide](../api/surfaces.md)
explains how to order your measured traces and reconstruct a region of
interest. To combine k-Wave at low frequencies with MSGB at high frequencies,
see the [hybrid backend guide](../guides/custom-low-frequency-solvers.md).
