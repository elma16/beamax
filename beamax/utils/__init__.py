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
    example_plot_dir,
)

# FFT helpers
from .fft import unitary_fft, unitary_ifft, convert_space

# Interpolation
from .interp import make_c_function_from_grid, Interpolator

# Array shape and resampling helpers
from .arrays import (
    interpolate_nearest,
    pad_array,
    pad_zero,
    pad_edge,
    crop_centered,
    interpolate_fourier,
    extract_centered_box,
    rel_l2,
)

# Coefficient indexing helpers used by transforms and solvers
from .coeff_index import (
    batch_data,
    find_level,
    find_tensor_and_multiindex,
    compute_coeff_shapes,
)


__all__ = [
    # device/mem
    "get_devices",
    "memory_estimate",
    "memory_str",
    "array_str",
    "detect_root",
    "example_plot_dir",
    # fft
    "unitary_fft",
    "unitary_ifft",
    "convert_space",
    # interp
    "make_c_function_from_grid",
    "Interpolator",
    # arrays
    "interpolate_nearest",
    "pad_array",
    "pad_zero",
    "pad_edge",
    "crop_centered",
    "interpolate_fourier",
    "extract_centered_box",
    "rel_l2",
    # coeff index
    "batch_data",
    "find_level",
    "find_tensor_and_multiindex",
    "compute_coeff_shapes",
]
