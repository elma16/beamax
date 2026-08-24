from importlib import import_module
from typing import Any

from .hybrid_solver import (
    HybridBackend,
    HybridContext,
    HybridSolver,
    HybridSolverConfig,
)

from .msgb_solvers.msgb_solver import (
    MSGBExperimentalConfig,
    MSGBSolver,
    ShardingStrategy,
)

__all__ = [
    "MSGBSolver",
    "MSGBExperimentalConfig",
    "ShardingStrategy",
    "HybridBackend",
    "HybridContext",
    "HybridSolver",
    "HybridSolverConfig",
    "KWaveSolver",
]


def __getattr__(name: str) -> Any:
    """Load the optional k-Wave solver on first access."""
    if name != "KWaveSolver":
        raise AttributeError(name)

    module_name = "beamax.solvers.kwave_solver"
    try:
        solver = getattr(import_module(module_name), name)
    except ModuleNotFoundError as error:
        missing = error.name or ""
        if missing != "kwave" and not missing.startswith("kwave."):
            raise
        raise ImportError(
            "KWaveSolver requires the optional k-Wave dependencies; "
            "install beamax[kwave]."
        ) from error
    globals()[name] = solver
    return solver
