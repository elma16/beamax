# beamax

beamax provides multiscale Gaussian beam solvers and wave-packet transforms for photoacoustic tomography in JAX.

## Start here

- Install with `pip install beamax`.
- Browse the [example gallery](examples/index.md).
- Use the navigation under **Basic API** for generated API documentation.

## Packages

- `beamax.decomposition` and `beamax.transforms`: dyadic frequency tilings and MSWPT analysis/synthesis.
- `beamax.geometry`: computational domains and sensor geometries.
- `beamax.gb`: Gaussian beam kernels and trajectory solvers.
- `beamax.solvers`: MSGB, hybrid, and optional k-Wave solvers.
- `beamax.plotter`: optional styling and MSWPT coefficient visualisation.

The MSWPT/MSGB implementation follows Jianliang Qian and Lexing Ying, ["Fast Multiscale Gaussian Wavepacket Transforms and Multiscale Gaussian Beams for the Wave Equation"](https://doi.org/10.1137/100787313), *Multiscale Modeling & Simulation*, 8(5), 1803–1837, 2010.
