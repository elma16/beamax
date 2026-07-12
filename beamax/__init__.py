"""
beamax — Multiscale Gaussian Beams in JAX. Minimal public API.
Avoids importing heavy/optional modules at package import time.
"""

from importlib import import_module
from types import ModuleType
from typing import Any

__version__ = "0.2.0"

# ---- Curated, always-light symbols (fast to import)
# geometry
from .geometry import Domain, Sensor

# decomposition / transforms
from .decomposition import DyadicDecomposition
from .transforms import MSWPT

__all__ = [
    "__version__",
    # core API
    "Domain",
    "Sensor",
    "DyadicDecomposition",
    "MSWPT",
    # names exposed lazily below
    "gb",
    "utils",
    "plotter",
    "solvers",
]


# ---- Lazy submodules (to keep import cost small)
# Access as: beamax.gb, beamax.utils, beamax.plotter, beamax.solvers
def __getattr__(name: str) -> Any:
    """
    Lazily import optional public submodules.

    Parameters
    ----------
    name : str
        Attribute requested from the package namespace.

    Returns
    -------
    Any
        Imported submodule, cached in ``globals()``.

    Raises
    ------
    AttributeError
        If ``name`` is not a lazily exposed submodule.
    """
    if name in {"gb", "utils", "plotter", "solvers"}:
        mod: ModuleType = import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
