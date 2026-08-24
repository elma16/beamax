"""Minimal Beamax API with lazy optional imports."""

from importlib import import_module
from types import ModuleType
from typing import Any

__version__ = "0.3.0"

from .geometry import Domain, Sensor
from .decomposition import DyadicDecomposition
from .transforms import MSWPT

__all__ = [
    "__version__",
    "Domain",
    "Sensor",
    "DyadicDecomposition",
    "MSWPT",
    "coefficients",
    "gb",
    "utils",
    "plotter",
    "solvers",
]


def __getattr__(name: str) -> Any:
    """Load a public submodule on first access."""
    if name in {"coefficients", "gb", "utils", "plotter", "solvers"}:
        mod: ModuleType = import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
