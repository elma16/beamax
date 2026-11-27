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
