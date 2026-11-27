# Run k-Wave on Apple Metal

On an Apple silicon Mac running macOS 15 or newer, Beamax can use the GPU to
simulate pressure at sensors and reconstruct a pressure field from sensor
measurements on a 2D or 3D Cartesian grid. Install the k-Wave extra in a
native arm64 Python environment.

```sh
python -m pip install "beamax[kwave]"
```

From a Beamax source checkout, use `python -m pip install -e ".[kwave]"`.
Select `backend="mac"` when creating the solver. The existing spelling
`backend="cpp", device="metal"` selects the same executable.

```python
from beamax.solvers import KWaveSolver

solver = KWaveSolver(backend="mac")
pressure = solver.forward(p0, domain, sensor_mask, ts)
```

On the first solve that uses Metal, Beamax downloads the executable, checks the
archive against the pinned SHA-256 digest, and caches the executable for
future runs. [NeuroHarmonics Metal release](https://github.com/NeuroHarmonics/k-wave-cpp-mac/releases/tag/v1.3.0-macos)

`pressure` has shape `(number of time samples, number of sensors)`. For a full
example that constructs the domain, initial pressure, and sensor mask, see
[MSGB vs k-Wave](../examples/forward.md#msgb-vs-k-wave).

## Reconstruct from sensor data

`time_reversal` and `adjoint` use the same Metal executable as `forward`.
Both return an image with shape `domain.N` for their default `record="p_final"`.
Pass the acquisition mask as `sources` to inject each trace at its receiver
location. For traces during time reversal, use `record="p"` and pass the
desired recording mask as `sensors`.

```python
image = solver.time_reversal(
    pressure, domain, sensors=sensor_mask, sources=sensor_mask, ts=ts,
    data_layout="nt_ns",
)
```

`solver.adjoint` accepts the same arguments and requires a lossless medium,
equal grid spacing along each axis, and `smooth_p0=False` (the default).
Beamax prepares the time-varying sources in the units and order expected by the
native executable. The [solver API](../api/solvers/kwave_solver.md) lists
supported grids, settings, and output records.

To inspect the input and output HDF5 files, supply `data_path` when creating
the solver. Each Metal inverse call writes its own `beamax-inverse-*`
subdirectory under that path. Forward calls using a record other than `p`
write a `beamax-forward-*` subdirectory.

Comparisons with the Python CPU solver cover lossless 2D and 3D grids.
In a 2D adjoint dot-product test with data only at the last time sample,
enabling the perfectly matched layer (PML) increased the discrepancy on both
CPU and Metal. When an application requires a discrete transpose, check the
dot-product identity with its acquisition geometry, time window, and PML.

## Use an existing executable

To prepare for an offline run, download the same verified release ahead of time
with `python -m beamax.install_kwave_metal` while you have network access.
If you already have a copy of `kspaceFirstOrder-Metal`, pass its absolute path
as `binary_path` when constructing `KWaveSolver`, or set
`BEAMAX_KWAVE_BINARY_PATH` to that file for the run. Beamax checks
`binary_path` first, then the environment variable, then its user cache.

MSGB runs through JAX and uses JAX's device configuration.
