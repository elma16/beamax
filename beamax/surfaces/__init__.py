"""Detector geometry and curved MSGB boundary matching."""

from .matching import (
    curvature_block,
    decode_packets,
    launch_momentum,
    match_packets,
    reduced_hessian,
    solve_reduced_hessian,
    surface_time_reversal,
    surface_tr_parameters,
)
from .surface import DetectorSurface

__all__ = [
    "DetectorSurface",
    "curvature_block",
    "decode_packets",
    "launch_momentum",
    "match_packets",
    "reduced_hessian",
    "solve_reduced_hessian",
    "surface_time_reversal",
    "surface_tr_parameters",
]
