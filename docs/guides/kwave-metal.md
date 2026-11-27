# Run k-Wave on Apple Metal

On an Apple silicon Mac running macOS 15 or newer, Beamax can use the GPU to
propagate an initial pressure field on a 2D or 3D Cartesian grid. Install the
k-Wave extra in a native arm64 Python environment.

```sh
python -m pip install "beamax[kwave]"
```

From a Beamax source checkout, use `python -m pip install -e ".[kwave]"`.
Select `device="metal"` when creating the solver.

```python
from beamax.solvers import KWaveSolver

solver = KWaveSolver(backend="cpp", device="metal")
pressure = solver.forward(p0, domain, sensor_mask, ts)
```

On the first forward call, Beamax downloads the Metal solver, checks the
archive against the pinned SHA-256 digest, and caches the executable for
future runs. [NeuroHarmonics Metal release](https://github.com/NeuroHarmonics/k-wave-cpp-mac/releases/tag/v1.3.0-macos)

`pressure` has shape `(number of time samples, number of sensors)`. For a full
example that constructs the domain, initial pressure, and sensor mask, see
[MSGB vs k-Wave](../examples/forward.md#msgb-vs-k-wave).

To prepare for an offline run, download the same verified release ahead of time
with `python -m beamax.install_kwave_metal` while you have network access.
If you already have a copy of `kspaceFirstOrder-Metal`, pass its absolute path
as `binary_path` when constructing `KWaveSolver`, or set
`BEAMAX_KWAVE_BINARY_PATH` to that file for the run. Beamax checks
`binary_path` first, then the environment variable, then its user cache.

`KWaveSolver()` uses the CPU by default. With `device="metal"`, forward calls
use the Metal executable, while time reversal and adjoint calls use the Python
solver on the CPU. MSGB runs through JAX and uses JAX's device configuration.
