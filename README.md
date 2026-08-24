# beamax

[![CI](https://github.com/elma16/beamax/actions/workflows/run-tests.yml/badge.svg)](https://github.com/elma16/beamax/actions/workflows/run-tests.yml)
[![codecov](https://codecov.io/gh/elma16/beamax/branch/main/graph/badge.svg)](https://codecov.io/gh/elma16/beamax)

[Installation](#installation) | [Documentation](https://elma16.github.io/beamax/) | [Examples](https://elma16.github.io/beamax/examples/)

beamax is a [JAX](https://github.com/jax-ml/jax) library for photoacoustic tomography with multiscale Gaussian beams. It provides:

- dyadic Fourier decompositions and multiscale wave-packet transforms
- forward, adjoint, and time-reversal MSGB solvers
- hybrid low- and high-frequency composition and optional k-Wave integration
- CPU, GPU, and TPU execution through JAX.

## Installation

Python 3.12 or newer is required. Install JAX for your hardware using the [official instructions](https://docs.jax.dev/en/latest/installation.html), then install beamax:

```bash
pip install beamax
```

Optional extras provide plotting and k-Wave support:

```bash
pip install "beamax[viz-mpl]"
pip install "beamax[kwave]"
```

## Examples

See the [example gallery](https://elma16.github.io/beamax/examples/), including notebooks with Colab links. 

```bash
python examples/diagnostics/memory_planning.py
```

For development setup and tests, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Reference

- Jianliang Qian and Lexing Ying, ["Fast Multiscale Gaussian Wavepacket Transforms and Multiscale Gaussian Beams for the Wave Equation"](https://doi.org/10.1137/100787313), *Multiscale Modeling & Simulation*, 8(5), 1803–1837, 2010.

Related acoustic simulation projects include [k-Wave](https://www.k-wave.org/), [k-Wave-python](https://github.com/waltsims/k-wave-python), and [j-Wave](https://github.com/ucl-bug/jwave).

## Citation and license

If you use beamax, please cite this repository. For MSWPT/MSGB, also cite Qian and Ying (2010). beamax is released under the [MIT license](LICENSE).
