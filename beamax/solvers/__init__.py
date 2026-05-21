"""
Solvers public API with lazy loading of optional stacks.
No imports of heavy third-party libs at module import time.
"""

from importlib import import_module
from typing import Any

# Always-light, internal base types
from .solverbase import Solver  # protocol/ABC only; safe

from .hybrid_solver import HybridSolver

# Always available solvers implemented within this package:
from .msgb_solvers.msgb_solver import MSGBSolver, ShardingStrategy

__all__ = [
    "Solver",
    "MSGBSolver",
    "ShardingStrategy",
    "HybridSolver",
    # optional solvers exposed lazily via __getattr__
    "KWaveSolver",
    "FNONeuralOpsSolver",
    "FNOpdequinoxSolver",
]

# Map attribute → (module path, symbol)
_LAZY = {
    "KWaveSolver": ("beamax.solvers.kwave_solver", "KWaveSolver"),
    "FNONeuralOpsSolver": ("beamax.solvers.fno_solver_neurops", "FNONeuralOpsSolver"),
    "FNOpdequinoxSolver": ("beamax.solvers.fno_solver_pdequinox", "FNOpdequinoxSolver"),
}


def __getattr__(name: str) -> Any:
    """
    Lazily import optional solver classes.

    Parameters
    ----------
    name : str
        Solver symbol requested from ``beamax.solvers``.

    Returns
    -------
    Any
        Imported solver class, cached in ``globals()``.

    Raises
    ------
    ImportError
        If the requested optional solver cannot be imported.
    AttributeError
        If ``name`` is not part of the lazy solver map.
    """
    if name in _LAZY:
        modpath, sym = _LAZY[name]
        try:
            mod = import_module(modpath)
        except Exception as e:
            # Defer the failure until first touch with a precise message.
            raise ImportError(
                f"{name} is optional. Install the relevant extra and its dependencies "
                f"to use it (failed to import {modpath!r}: {e})"
            ) from e
        obj = getattr(mod, sym)
        globals()[name] = obj
        return obj
    raise AttributeError(name)
