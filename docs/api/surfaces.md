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
