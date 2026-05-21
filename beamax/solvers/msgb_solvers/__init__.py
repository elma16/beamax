"""
MSGB solver package.
"""

from .msgb_solver import MSGBSolver, ShardingStrategy

# forward utils
from .forward_solver_utils import (
    threshold_coefficients,
    compute_coefficients,
    compute_forward_parameters,
    compute_forward_result,
    compute_memory_requirements,
)

# TR utils
from .tr_solver_utils import (
    compute_TR_parameters,
    compute_TR_result,
    compute_mT_linear_system,
    mT_forward,
    mT_inverse,
)

__all__ = [
    "MSGBSolver",
    "ShardingStrategy",
    "threshold_coefficients",
    "compute_coefficients",
    "compute_forward_parameters",
    "compute_forward_result",
    "compute_memory_requirements",
    "compute_TR_parameters",
    "compute_TR_result",
    "compute_mT_linear_system",
    "mT_forward",
    "mT_inverse",
]
