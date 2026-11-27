# beamax

beamax provides multiscale Gaussian beam solvers and wave-packet transforms for photoacoustic tomography in JAX.

## Start here

Install Beamax with `python -m pip install beamax` (Python 3.12 or newer).
Use the [wave-packet tutorial](examples/wave-packets.md) to decompose and
recover an initial pressure field with the base installation. Follow the
[example gallery](examples/index.md) to simulate the pressure at a detector
array and reconstruct an image from the measurements.

On an Apple silicon Mac, the [Metal guide](guides/kwave-metal.md) covers
the native k-Wave solver and experimental JAX acceleration for MSGB.

## Packages

- `beamax.decomposition` and `beamax.transforms`: dyadic frequency tilings and MSWPT analysis/synthesis.
- `beamax.geometry`: computational domains and sensor geometries.
- `beamax.gb`: Gaussian beam kernels and trajectory solvers.
- `beamax.solvers`: MSGB, hybrid, and optional k-Wave solvers.
- `beamax.plotter`: optional styling and MSWPT coefficient visualisation.

The MSWPT/MSGB implementation follows Jianliang Qian and Lexing Ying, ["Fast Multiscale Gaussian Wavepacket Transforms and Multiscale Gaussian Beams for the Wave Equation"](https://doi.org/10.1137/100787313), *Multiscale Modeling & Simulation*, 8(5), 1803-1837, 2010. [Full text (PDF)](https://stanford.edu/~lexing/GBwave.pdf)

MSGB time reversal uses the boundary matching construction of Michele Berra, Maarten V. de Hoop, and José Luis Romero to launch Gaussian beams from measured pressure traces. [Published article (2022)](https://doi.org/10.1016/j.jde.2021.11.043) · [Open preprint](https://arxiv.org/abs/1705.00337)
