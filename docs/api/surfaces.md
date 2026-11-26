# Detector surfaces

`DetectorSurface` defines a regular detector chart in row-major order. Built-in charts cover points, planes, circles, open arcs, ellipses, polar curves, sphere bands, cylinders and tori. Use `DetectorSurface.parametric` with a JAX-compatible mapping for a custom chart.

Chart geometry includes positions, a front-pointing unit normal, first and second derivatives, and the area element. Boundary data axes are time followed by the chart axes.

## Detector mask order

A binary detector mask enumerates channels in C-order over the grid. To use those traces on a sparse ring, reorder the last axis with `pressure = mask_traces[..., ring.mask_to_chart(domain)]`. To inject chart traces into the same mask, use the inverse permutation `np.argsort(ring.mask_to_chart(domain))`.

A dense array can map multiple detector elements to the same grid point. In that case, the binary mask conversion raises an error.

## Open arcs and patches

For an open axis, taper the rim and pad the traces before the periodic MSWPT. Use `padded_data, padded_arc = arc.prepare_open_data(data, padding=(8,), taper=(3,))` to taper the outer three intervals and add eight zero samples at both ends. Build the data domain and wave-packet transform using `padded_data.shape` and `padded_arc.spacing`. The padded chart marks the original detector region in `active_bounds`. On mixed charts such as a sphere band, specify zero widths on periodic axes.
