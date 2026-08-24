"""Public utility API, loaded on first attribute access."""

from importlib import import_module
from typing import Any


__all__ = [
    "unitary_fft",
    "unitary_ifft",
    "convert_space",
    "make_c_function_from_grid",
    "Interpolator",
    "interpolate_nearest",
    "pad_array",
    "pad_zero",
    "pad_edge",
    "crop_centered",
    "interpolate_fourier",
    "extract_centered_box",
    "rel_l2",
    "batch_data",
    "find_level",
    "find_tensor_and_multiindex",
    "compute_coeff_shapes",
    "DeviceCapabilities",
    "MemoryEstimate",
    "device_capabilities",
    "estimate_msgb_memory",
]


_EXPORT_MODULES = {
    "unitary_fft": ".fft",
    "unitary_ifft": ".fft",
    "convert_space": ".fft",
    "make_c_function_from_grid": ".interp",
    "Interpolator": ".interp",
    "interpolate_nearest": ".arrays",
    "pad_array": ".arrays",
    "pad_zero": ".arrays",
    "pad_edge": ".arrays",
    "crop_centered": ".arrays",
    "interpolate_fourier": ".arrays",
    "extract_centered_box": ".arrays",
    "rel_l2": ".arrays",
    "batch_data": ".coeff_index",
    "find_level": ".coeff_index",
    "find_tensor_and_multiindex": ".coeff_index",
    "compute_coeff_shapes": ".coeff_index",
    "DeviceCapabilities": ".memory",
    "MemoryEstimate": ".memory",
    "device_capabilities": ".memory",
    "estimate_msgb_memory": ".memory",
}


def __getattr__(name: str) -> Any:
    """Load and cache a public utility on first access."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
