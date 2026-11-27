"""Detector geometry and curved MSGB boundary matching."""

from .adjoint import surface_adjoint
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
from .surface import DetectorMask, DetectorSurface
from .ring import ring_adjoint, ring_dirichlet_from_source

__all__ = [
    "DetectorMask",
    "DetectorSurface",
    "ring_adjoint",
    "ring_dirichlet_from_source",
    "curvature_block",
    "decode_packets",
    "launch_momentum",
    "match_packets",
    "reduced_hessian",
    "solve_reduced_hessian",
    "surface_adjoint",
    "surface_time_reversal",
    "surface_tr_parameters",
]
