from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, Union, Tuple

import jax.numpy as jnp
import numpy as np


Array = Union[jnp.ndarray, np.ndarray]


class SupportsArray(Protocol):
    """Minimal protocol to accept JAX/NumPy arrays without importing heavy deps."""

    shape: Tuple[int, ...]


class Solver(ABC):
    """
    Unified solver interface.

    All implementations MUST accept keyword-only arguments and MUST document
    the exact shapes they expect for each argument.

    Conventions
    -----------
    - `p0`: initial pressure in the image domain. Real-valued unless otherwise stated.
    - `dpdt`: initial time derivative (may be None; treat as zeros if not used).
    - `domain`: geometry/medium object with physical grid and c(x).
    - `sensors`: either a Sensor object or an array mask / positions consistent with the solver.
    - `ts`: time grid, shape (Nt,).
    - All methods return arrays with time leading: shape (Nt, *S) unless documented.

    Notes
    -----
    Implementations that do not support `time_reversal` or `adjoint` must raise
    `NotImplementedError` with a clear message at call time.
    """

    @abstractmethod
    def forward(
        self,
        p0: Array,
        domain: object,
        sensors: object,
        ts: Array,
        **solver_kwargs,
    ) -> Array:
        """
        Simulate forward propagation from initial conditions.

        Parameters
        ----------
        p0 : array
            Initial pressure field (image domain).
        domain : object
            Geometry/medium descriptor.
        sensors : object
            Sensor geometry or mask.
        ts : array, shape (Nt,)

        Returns
        -------
        array
            Sensor time series, shape `(Nt, *S)`.
        """

    @abstractmethod
    def time_reversal(
        self,
        data: Array,
        domain: object,
        sensors: object,
        sources: object,
        ts: Array,
        **solver_kwargs,
    ) -> Array:
        """
        Time-reversal reconstruction.

        Parameters
        ----------
        data : Array, shape (Nt, *S)
            Sensor measurements in time.
        domain : Domain
            Reconstruction domain (where to reconstruct on).
        sensors : Sensor
            Sensor geometry.
        sources : Sensor, optional
            Source positions for data injection. Pass ``None`` if the solver
            does not distinguish between sources and receivers.
        ts : Array, shape (Nt,)
            Time grid corresponding to ``data``.
        **solver_kwargs
            Solver-specific arguments (e.g. ``wpt``, ``domain_data`` for MSGB).

        Returns
        -------
        Array, shape (*N,)
            Reconstructed initial pressure on the image grid.
        """

    @abstractmethod
    def adjoint(
        self,
        data: Array,
        domain: object,
        sensors: object,
        sources: object,
        ts: Array,
        **solver_kwargs,
    ) -> Array:
        """
        Adjoint reconstruction.

        Parameters
        ----------
        data : Array, shape (Nt, *S)
            Sensor measurements in time.
        domain : Domain
            Reconstruction domain (where to reconstruct on).
        sensors : Sensor
            Sensor geometry.
        sources : Sensor, optional
            Source positions for data injection. Pass ``None`` if the solver
            does not distinguish between sources and receivers.
        ts : Array, shape (Nt,)
            Time grid corresponding to ``data``.
        **solver_kwargs
            Solver-specific arguments (e.g. ``wpt``, ``domain_data`` for MSGB).

        Returns
        -------
        Array, shape (*N,)
            Reconstructed initial pressure on the image grid. Unlike
            time-reversal, the adjoint is the *exact* transpose of the
            discretised forward map.
        """
