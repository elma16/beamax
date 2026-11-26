# Curved detector surfaces

`beamax.surfaces` reconstructs real pressure traces sampled on a regular detector chart in free space. Data axes are time followed by the chart axes. The first time row corresponds to zero, and detector coordinates follow `surface.positions()` in row-major chart order.

## Ring time reversal and a region of interest

This example uses 64 uniformly spaced elements on a 2.5 mm ring in water. Supply the measured pressure as an array of shape `(64, 64)`.

```python
from beamax import Domain, DyadicDecomposition, MSWPT
from beamax.surfaces import DetectorSurface, surface_time_reversal

domain = Domain((64, 64), (1e-4, 1e-4), (False, False), c=1500.0)
centre, radius = (3.2e-3, 3.2e-3), 2.5e-3
ring = DetectorSurface.circle(centre, radius, 64)
dd = Domain((64, 64), (5e-8, *ring.spacing), (False, True))
wpt = MSWPT(DyadicDecomposition(2, dd.N, (4, 8), (1, 1)), 2, "rectangular_mirror")
points = domain.grid[16:32, 28:44]
image, diagnostics = surface_time_reversal(
    pressure, dd, wpt, ring, domain, 2048, eval_points=points
)
```

The example returns a 16 × 16 image with the coordinates and order of `points`. Omit `eval_points` to render the full grid. A list of shape `(K, 2)` produces a vector of length K. Choosing a region reduces the number of beam evaluations while keeping packet selection and ray propagation identical. Diagnostics include each packet's hyperbolicity, incidence angle cosine and chart membership. Callable or gridded sound speeds select numerical propagation. A scalar speed in a lossless medium selects the homogeneous closed form.

## Detector mask order

A binary detector mask enumerates channels in C-order over the grid. To use those traces on a sparse ring, reorder the last axis with `pressure = mask_traces[..., ring.mask_to_chart(domain)]`. To inject chart traces into the same mask, use the inverse permutation `np.argsort(ring.mask_to_chart(domain))`.

A dense array can map multiple detector elements to the same grid point. In that case, the binary mask conversion raises an error.

## Open arcs and patches

For an open axis, taper the rim and pad the traces before the periodic MSWPT. Use `padded_data, padded_arc = arc.prepare_open_data(data, padding=(8,), taper=(3,))` to taper the outer three intervals and add eight zero samples at both ends. Build the data domain and wave-packet transform using `padded_data.shape` and `padded_arc.spacing`. Pass the padded data and chart together to `surface_time_reversal`. Packet centres in padding receive zero amplitude. On mixed charts such as a sphere band, specify zero widths on periodic axes.

## PAT adjoint on a detector surface

`surface_adjoint` maps a pressure residual on a detector chart to an initial-pressure image in a lossless medium (`lam=0`). For example, consider an ellipse with 64 elements spaced uniformly in angle. It uses the same time and angular sampling as the ring example above, so `dd` and `wpt` can be reused. Supply the residual measured on the ellipse in chart order.

```python
from beamax.surfaces import surface_adjoint

ellipse = DetectorSurface.ellipse(centre, a=2.5e-3, b=1.8e-3, n=64)
image, diagnostics = surface_adjoint(
    residual, dd, wpt, ellipse, domain, 2048,
    pairing="surface_area", eval_points=points,
)
```

Choose the pairing used to measure the residual in your objective function.

- `"surface_area"` weights each detector by the physical length or area it represents and each time sample by `dt`. On the ellipse, equal angle increments cover different arc lengths. On a 3D detector patch, the two chart increments and the local surface metric determine the area.
- `"element_sum"` weights every detector equally and each time sample by `dt`. Use this when the objective sums the squared residuals over detector channels, including when their spacing varies.

Both pairings weight the image inner product by pixel area in 2D or voxel volume in 3D. `surface_adjoint` differentiates the residual in time, weights each wave packet using the local sound speed and angle of incidence, and propagates it back to the image. Spectral differentiation treats the time record as periodic. If the residual has significant amplitude at either end, supply a real time taper of shape `(nt,)` through `window`. It multiplies all channels before differentiation.

The packet weight uses the leading high-frequency approximation to the boundary-source operator. The solver discards evanescent packets. The weight becomes singular at grazing incidence, so `guard=0.05` keeps packets whose absolute incidence cosine exceeds 0.05, measured from the detector normal. Inspect `diagnostics["boundary_multiplier"]` for each selected packet's complex weight. The result approximates the continuous adjoint. Selecting the strongest `top_n` packets makes that selection depend on the input, so a discrete forward/adjoint dot-product check can have a nonzero error.

Orient the chart towards the reconstruction region. The matching uses one local ray branch. On nonconvex detectors, rays can intersect the surface again, and identifying the relevant arrivals requires additional information about the acquisition.

For spatially varying sound speed, the beams use the numerical ODE solver. If it reaches its step limit, pass a smaller initial step in physical time through `solver_config=SolverConfig.from_precision(dt0=...)`, with `SolverConfig` from `beamax.gb.gb_solvers`.

## Circular PAT adjoint

`ring_adjoint(pressure, dd, wpt, domain, 2048, centre=centre, radius=radius, pairing="arc_length", eval_points=points)` returns the approximate continuous PAT adjoint and packet diagnostics. Use a lossless reconstruction domain with `lam=0`. The exact circular boundary symbol assumes homogeneous speed. For an interior with varying speed, the operator uses the sound speed shared by all boundary elements and propagates beams through the supplied interior. Time reversal and this thresholded adjoint approximation have different normalisations.

Choose `pairing="arc_length"` for the data inner product weighted by `dt * 2πR / ns`, or `"element_sum"` for the data inner product weighted by `dt` per element. Both use an area-weighted image inner product. The boundary multiplier is computed on the host, and beam propagation and rendering use JAX.
