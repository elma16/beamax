"""Acquisition geometry shared by solver entry points."""

import jax.numpy as jnp
import numpy as np
from beamax.geometry import Sensor


def forward_positions(domain, detectors):
    from beamax.surfaces import DetectorSurface

    if isinstance(detectors, DetectorSurface):
        if detectors.ndim != domain.ndim:
            raise ValueError("Detector and propagation dimensions must match.")
        return detectors.positions()
    if isinstance(detectors, Sensor):
        return detectors.positions
    values = np.asarray(detectors)
    if values.shape == domain.N:
        return Sensor(domain, binary_mask=jnp.asarray(values)).positions
    if values.ndim != 2 or values.shape[1] != domain.ndim or not len(values):
        raise ValueError("Give a binary grid mask or an array of detector positions.")
    if not np.isrealobj(values) or not np.all(np.isfinite(values)):
        raise ValueError("Detector positions must be real and finite.")
    return jnp.asarray(values)


def oriented_surface(domain, detectors, front):
    from beamax.surfaces import DetectorSurface

    if isinstance(detectors, DetectorSurface):
        if front is not None or detectors.ndim != domain.ndim:
            raise ValueError(
                "Use a surface of the domain dimension with its own orientation."
            )
        return detectors
    if front not in (-1, 1):
        raise ValueError("A planar mask requires front=1 or front=-1.")
    positions = np.asarray(Sensor(domain, binary_mask=detectors).positions)
    axes = [np.unique(positions[:, i]) for i in range(domain.ndim)]
    fixed = [i for i, a in enumerate(axes) if len(a) == 1]
    if len(fixed) != 1 or np.prod([len(a) for a in axes]) != len(positions):
        raise ValueError("Give a complete planar mask or an oriented DetectorSurface.")
    normal = fixed[0]
    tangents = [a for i, a in enumerate(axes) if i != normal]
    if any(
        not np.allclose(
            np.diff(a),
            (a[-1] - a[0]) / (len(a) - 1),
            rtol=1e-5,
            atol=4 * np.finfo(a.dtype).eps * np.max(abs(a)),
        )
        for a in tangents
    ):
        raise ValueError("Planar detectors must be uniformly sampled.")
    return DetectorSurface.plane(
        domain.ndim,
        axis=normal,
        value=axes[normal][0],
        origin=tuple(a[0] for a in tangents),
        spacing=tuple((a[-1] - a[0]) / (len(a) - 1) for a in tangents),
        shape=tuple(len(a) for a in tangents),
        front=front,
    )


def chart_data(data, surface, ts, wpt):
    from beamax.geometry import Domain
    from beamax.solvers.msgb_solvers.msgb_solver import _validate_time_grid

    dt = _validate_time_grid(ts)
    assert dt is not None  # The inverse requires at least two samples.
    if ts[0] < 0:
        raise ValueError("Acquisition times must be nonnegative.")
    shape = (len(ts), *surface.shape)
    data = jnp.asarray(data)
    if data.shape not in (shape, (len(ts), surface.num_sensors)):
        raise ValueError("Expected time-first detector traces in chart order.")
    if wpt is None or tuple(wpt.dyadic_decomp.N) != shape:
        raise ValueError(
            "The data transform must match the time and detector sampling."
        )
    return data.reshape(shape), Domain(
        shape, (dt, *surface.spacing), (False, *surface.periodic)
    )


def detector_measure(detectors, pairing, element_measures, count):
    """Physical detector measure per channel in acquisition order."""
    from beamax.surfaces import DetectorSurface

    chart = isinstance(detectors, DetectorSurface)
    if pairing is None:
        if not chart:
            raise ValueError("Supply a pairing for a binary detector mask.")
        pairing = "surface_area"
    if pairing not in ("surface_area", "element_sum", "euclidean"):
        raise ValueError("Choose surface_area, element_sum or euclidean.")
    if pairing != "surface_area":
        if element_measures is not None:
            raise ValueError("Element measures apply only to the surface_area pairing.")
        return pairing, 1.0
    if chart:
        if element_measures is not None:
            raise ValueError("A chart supplies its physical element measures.")
        theta, _ = detectors.active_parameters(detectors.params_flat())
        measure = np.asarray(detectors.area_element(theta)) * np.prod(detectors.spacing)
    else:
        if element_measures is None:
            raise ValueError(
                "Supply element_measures for a mask with surface_area pairing."
            )
        measure = np.asarray(element_measures)
    if measure.shape not in ((), (count,)) or not np.all(
        np.isfinite(measure) & (measure > 0)
    ):
        raise ValueError(
            "Give positive finite element measures, scalar or one per receiver."
        )
    return pairing, measure
