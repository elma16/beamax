# `beamax.solvers.kwave_solver`

Two- and three-dimensional `k-wave-python` integration. Install with
`pip install "beamax[kwave]"`.

`KWaveSolver` uses the selected backend for `forward`, `time_reversal`, and
`adjoint`.

| Configuration | Solver |
| --- | --- |
| `KWaveSolver()` or `KWaveSolver(backend="cpp")` | Native OpenMP executable on the CPU |
| `KWaveSolver(backend="cpp", device="gpu")` | Native CUDA executable on an NVIDIA GPU |
| `KWaveSolver(backend="python")` | k-wave-python's Python solver on the CPU |
| `KWaveSolver(backend="python", device="gpu")` | k-wave-python's Python solver with CuPy |
| `KWaveSolver(backend="mac")` | Native Metal executable on an Apple silicon GPU |

`backend="cpp", device="metal"` also selects Metal. For installation and
reconstruction examples, see [Run k-Wave on Apple Metal](../../guides/kwave-metal.md).
For an inverse calculation that previously used the automatic Python fallback,
select `backend="python"`. Native solvers raise an error for unsupported
records or settings.

Native solves require 2D or 3D grids with at least two points per axis and
`use_kspace=True` and `use_sg=True`. In 2D the first axis must have at least
three points because k-wave-python interprets a two-row mask as coordinates.
Beamax's field-returning methods require `save_only=False`.

## Pressure recordings

For the native C++ and Mac backends, `record` selects the following results.
Here `Nt` is the number of time samples and `Ns` is the number of points in
the recording mask.

| Record | `forward` | `time_reversal` | `adjoint` |
| --- | --- | --- | --- |
| `p` | Sensor traces, `(Nt, Ns)` (default) | Sensor traces, `(Ns, Nt)` | — |
| `p_max`, `p_min`, `p_rms` | One value per sensor, `(Ns,)` | One value per sensor, `(Ns,)` | — |
| `p_final` | Final pressure on the grid | Final pressure on the grid (default) | Adjoint image on the grid (default) |
| `p_max_all`, `p_min_all` | One extremum per grid point | One extremum per grid point | — |

Full-grid results have shape `domain.N`. For velocity or intensity records,
select `backend="python"` and use the recording options supported by your
k-wave-python version. In 0.6.2, whole-grid pressure extrema
(`p_max_all`, `p_min_all`) require a native backend.

::: beamax.solvers.kwave_solver
