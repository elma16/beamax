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
pressure = solver.forward(p0, domain, detectors=sensor_mask, ts=ts)
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
    pressure, domain, detectors=sensor_mask, ts=ts,
)
```

For the Euclidean adjoint of the binary-mask recording, use `solver.adjoint(pressure, domain, detectors=sensor_mask, ts=ts, pairing="euclidean")`. It requires a lossless medium,
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

## SciPy import error on an Apple silicon Mac

If a local k-Wave notebook on macOS 27 fails to import `_spropack` with a
`__thread_bss` loader error, check which SciPy wheel is installed.
k-wave-python 0.6.2 pins SciPy 1.15.3. On an M4 with CPython 3.12, the
default `macosx_14_0_arm64` wheel failed to load, while the official
`macosx_12_0_arm64` wheel of the same version imported successfully.
[SciPy issue](https://github.com/scipy/scipy/issues/25635)

Run these commands in the Python 3.12 environment used by your notebook,
then restart its kernel.

```sh
python -m pip download --only-binary=:all: --no-deps --platform macosx_12_0_arm64 \
    --python-version 3.12 --implementation cp --abi cp312 scipy==1.15.3
python -m pip install --no-deps --force-reinstall ./scipy-1.15.3-cp312-cp312-macosx_12_0_arm64.whl
python -c "import scipy.sparse.linalg"
```

## Experimental JAX acceleration for MSGB

`KWaveSolver(backend="mac")` selects the native k-Wave executable. The
Gaussian beam calculations in `MSGBSolver` run through JAX. On Apple
silicon, the optional `jax-mps` plugin can run some of those calculations
on the GPU. Beamax's standard installation uses JAX on CPU on these Macs.

In a float32 test on one M4 Mac mini, a 128 × 128 homogeneous forward
simulation with 256 selected packets, 128 receivers and 256 time samples took
**202 ms on GPU and 575 ms on CPU**. These are medians of five calls after
compilation, each timed until its output was ready. The sensor records differed
by `7.4e-7` in relative L2 norm. The test used JAX, jaxlib and `jax-mps`
0.11.1. The plugin documents its compatible JAX versions and the
`JAX_PLATFORMS=mps` device setting. [jax-mps 0.11.1](https://pypi.org/project/jax-mps/0.11.1/)

Use the CPU backend (`JAX_PLATFORMS=cpu`) for the gallery and for float64
work. The tested plugin lacks float64 support. On the same M4, MSGB time
reversal and adjoint reached an unsupported complex triangular solve, and
forward propagation with spatially varying sound speed reached an unsupported
Python callback. The native k-Wave Metal backend described above supports
all three k-Wave operations. [Plugin source for precision handling](https://raw.githubusercontent.com/tillahoffmann/jax-mps/v0.11.1/src/pjrt_plugin/mlx_buffer.cc)
