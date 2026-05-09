# Getting Started

## Prerequisites

- Python 3.11 or 3.12
- JAX installed for your hardware (CPU/GPU/TPU), following the [official JAX install guide](https://github.com/google/jax#installation)

## Install

From the repository root:

```bash
# Core package
pip install .

# Optional extras
pip install .[viz]      # plotting helpers
pip install .[kwave]    # k-wave-python integration
pip install .[fno]      # operator-learning stack
pip install .[all]      # everything (dev + optional extras)
```

If you are actively developing in this checkout, use editable mode:

```bash
pip install -e .[dev]
```

If build isolation cannot fetch dependencies (offline/restricted network):

```bash
pip install -e .[dev] --no-build-isolation
```

## Verify Your Import Path

```bash
python -c "import beamax; print(beamax.__file__)"
```

- Editable install should point to `.../src/beamax/__init__.py`.
- If it points to `.../site-packages/beamax/...`, your environment is using an installed wheel instead of your local source tree.

## Minimal Usage

```python
import jax
from beamax import Domain, DyadicDecomposition, MSWPT

domain = Domain(N=(64, 64), dx=(1e-3, 1e-3), c=1500.0, periodic=(True, True))
decomp = DyadicDecomposition(3, domain.N, (2, 4, 8), (1, 1))
wpt = MSWPT(decomp, redundancy=2, windowing="rectangular")

field = jax.random.normal(jax.random.PRNGKey(0), domain.N)
coeffs = wpt.forward(field, input_type="spatial")
recon = wpt.inverse(coeffs, output_type="spatial")
```

## Run an Example

```bash
python examples/forward/forward-2d.py
```

For headless environments:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig python examples/forward/forward-2d.py
```
