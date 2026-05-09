"""
Utilities public API (explicit). Heavy things must be imported inside call-sites.
"""

# device/memory
from .device import (
    get_devices,
    memory_estimate,
    memory_str,
    array_str,
    detect_root,
)

# FFT helpers
from .fft import unitary_fft, unitary_ifft, convert_space

# Interpolation
from .interp import make_c_function_from_grid, Interpolator

# Misc helpers (careful: keep the surface tight)
from .misc import (
    batch_data,
    interpolate_nearest,
    pad_array,
    pad_edge,
    crop_centered,
    compute_coeff_shapes,
    find_tensor_and_multiindex,
    find_level,
    ellipsoid_superposition,
    interpolate_fourier,
    extract_centered_box,
    choose_K_by_tau,
    find_min_K_for_target_error,
    select_levelaware_topK_indices,
    reconstruct_from_selection,
    rel_l2,
)

# OA-breast I/O mapping (depends on h5py/scipy; users can import beamax.utils without calling these)
from .oabreast import load_oabreast_p0_c

from .profiling import profile_section, print_memory_summary, array_info

__all__ = [
    # device/mem
    "get_devices",
    "memory_estimate",
    "memory_str",
    "array_str",
    "detect_root",
    # fft
    "unitary_fft",
    "unitary_ifft",
    "convert_space",
    # interp
    "make_c_function_from_grid",
    "Interpolator",
    # misc
    "batch_data",
    "interpolate_nearest",
    "pad_array",
    "pad_edge",
    "crop_centered",
    "compute_coeff_shapes",
    "find_tensor_and_multiindex",
    "find_level",
    "ellipsoid_superposition",
    "interpolate_fourier",
    "extract_centered_box",
    "choose_K_by_tau",
    "find_min_K_for_target_error",
    "select_levelaware_topK_indices",
    "reconstruct_from_selection",
    # data I/O
    "load_oabreast_p0_c",
    "profile_section",
    "print_memory_summary",
    "array_info",
    "rel_l2",
]
