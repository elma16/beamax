from __future__ import annotations

from typing import Tuple, List, Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np


def _all_even(N: Tuple[int, ...]) -> bool:
    """
    Check whether all grid dimensions are even.

    Parameters
    ----------
    N : Tuple[int, ...]
        Grid shape.

    Returns
    -------
    bool
        ``True`` when every entry of ``N`` is even.
    """
    return all((int(n) % 2) == 0 for n in N)


def _validate_boxes_per_dim_levels(
    boxes_per_dim_levels: Tuple[Tuple[int, ...], ...],
    num_levels: int,
    ndim: int,
) -> None:
    """
    Validate explicit per-axis box counts for each dyadic level.

    Parameters
    ----------
    boxes_per_dim_levels : Tuple[Tuple[int, ...], ...]
        Per-level tuple of per-axis box counts.
    num_levels : int
        Expected number of levels.
    ndim : int
        Spatial dimensionality.

    Raises
    ------
    ValueError
        If level counts, dimensionality, positivity, parity, or monotonicity
        constraints are violated.
    """
    if len(boxes_per_dim_levels) != num_levels:
        raise ValueError(
            f"len(boxes_per_dim_levels)={len(boxes_per_dim_levels)} != num_levels={num_levels}."
        )
    for lvl, b in enumerate(boxes_per_dim_levels):
        if len(b) != ndim:
            raise ValueError(
                f"Level {lvl}: len(boxes_per_dim_levels[lvl])={len(b)} != ndim={ndim}."
            )
        if any(int(x) <= 0 for x in b):
            raise ValueError(f"Level {lvl}: boxes per axis must be positive; got {b}.")
        if any((int(x) % 2) != 0 for x in b):
            raise ValueError(
                f"Level {lvl}: boxes per axis must be even for symmetry; got {b}."
            )

    # nondecreasing per axis across levels
    for ax in range(ndim):
        seq = [int(boxes_per_dim_levels[lvl][ax]) for lvl in range(num_levels)]
        if any(seq[i] > seq[i + 1] for i in range(num_levels - 1)):
            raise ValueError(
                f"boxes_per_dim_levels must be nondecreasing per axis; axis {ax} has {seq}."
            )


def validate_params(
    num_levels: int,
    N: Tuple[int, ...],
    num_boxes_levels: Tuple[int, ...],
    box_aspect_ratio: Tuple[int, ...],
    boxes_per_dim_levels: Optional[Tuple[Tuple[int, ...], ...]] = None,
) -> None:
    """
    Validate parameters for :class:`DyadicDecomposition`.

    Parameters
    ----------
    num_levels : int
        Number of dyadic levels.
    N : Tuple[int, ...]
        Grid shape per spatial axis.
    num_boxes_levels : Tuple[int, ...]
        Number of boxes along the smallest axis at each level.
    box_aspect_ratio : Tuple[int, ...]
        Per-axis integer box aspect ratio.
    boxes_per_dim_levels : Tuple[Tuple[int, ...], ...], optional
        Explicit per-axis box counts per level.

    Raises
    ------
    ValueError
        If the parameters cannot define a valid dyadic tiling.
    """
    if not isinstance(num_levels, int) or num_levels < 1:
        raise ValueError(f"num_levels must be positive int; got {num_levels}.")

    N = tuple(int(x) for x in N)
    if any(n <= 0 for n in N):
        raise ValueError(f"N must be positive; got {N}.")
    if not _all_even(N):
        raise ValueError(f"N must be even along all axes; got {N}.")

    ndim = len(N)

    bar = tuple(int(x) for x in box_aspect_ratio)
    if len(bar) == 0:
        raise ValueError("box_aspect_ratio must be non-empty.")
    if len(bar) != ndim:
        raise ValueError(f"len(box_aspect_ratio)={len(bar)} != ndim={ndim}.")
    if not any(x == 1 for x in bar):
        raise ValueError("At least one aspect ratio component must equal 1.")
    if ndim == 1 and bar != (1,):
        raise ValueError("For 1D, box_aspect_ratio must be (1,).")

    num_boxes_levels = tuple(int(x) for x in num_boxes_levels)
    if len(num_boxes_levels) != num_levels:
        raise ValueError(
            f"len(num_boxes_levels)={len(num_boxes_levels)} != num_levels={num_levels}."
        )
    if any(x <= 0 for x in num_boxes_levels):
        raise ValueError(f"num_boxes_levels must be positive; got {num_boxes_levels}.")
    if any(
        num_boxes_levels[i] > num_boxes_levels[i + 1] for i in range(num_levels - 1)
    ):
        raise ValueError(
            f"num_boxes_levels must be non-decreasing with level; got {num_boxes_levels}."
        )

    if boxes_per_dim_levels is not None:
        _validate_boxes_per_dim_levels(
            boxes_per_dim_levels, num_levels=num_levels, ndim=ndim
        )

        # Enforce consistency with "boxes on the smallest axis" convention
        min_axis = int(np.argmin(np.asarray(N)))
        implied = tuple(
            int(boxes_per_dim_levels[lvl][min_axis]) for lvl in range(num_levels)
        )
        if implied != num_boxes_levels:
            raise ValueError(
                "boxes_per_dim_levels disagrees with num_boxes_levels on the smallest axis. "
                f"Expected smallest-axis sequence {num_boxes_levels}, got {implied}."
            )

    # Max levels given smallest side and coarsest tiling (by smallest-axis count).
    N_ref = min(N)
    base = num_boxes_levels[0]
    if base <= 0 or base > N_ref:
        raise ValueError(f"Invalid base boxes per side: {base} for N_ref={N_ref}.")
    max_levels = int(jnp.floor(jnp.log2(N_ref // base)) + 1)
    if not (1 <= num_levels <= max_levels):
        raise ValueError(
            f"num_levels must be in [1, {max_levels}] (N_ref={N_ref}, base={base})."
        )

    # Box length per level along the smallest axis; must divide N_ref.
    levels_desc = list(range(num_levels - 1, -1, -1))
    denom_last = num_boxes_levels[-1]
    box_lengths = [N_ref // (denom_last * (2**L)) for L in levels_desc]
    if any(N_ref % L != 0 for L in box_lengths):
        raise ValueError(
            f"Box lengths must divide N_ref={N_ref}; computed {box_lengths}."
        )


class DyadicDecomposition(eqx.Module):
    """
    Multi-level dyadic tiling of Fourier space on a rectangular grid.

    The decomposition partitions Fourier space into frequency boxes organised
    across ``num_levels`` scales. Each coarser level covers roughly half the
    resolution of the next, and the outer ring of boxes at each non-final
    level is retained while an inner cutout (covered by the finer level) is
    removed — this gives the "dyadic" structure.

    Parameters
    ----------
    num_levels : int
        Number of scale levels. Must satisfy ``1 <= num_levels <= floor(log2(N_ref / num_boxes_levels[0])) + 1``
        where ``N_ref = min(N)``.
    N : Tuple[int, ...]
        Grid shape of the underlying domain per spatial axis.
    num_boxes_levels : Tuple[int, ...]
        Boxes per side along the smallest spatial axis, one entry per level,
        from coarsest to finest. Typically a dyadic progression such as
        ``(4, 8)`` or ``(2, 4, 8)``.
    box_aspect_ratio : Tuple[int, ...]
        Integer aspect ratio of each box per axis, applied at every level.
    boxes_per_dim_levels : Tuple[Tuple[int, ...], ...], optional
        Explicit per-axis box count per level. If ``None`` (default), the
        per-axis count is derived from ``num_boxes_levels`` and the domain
        aspect ratio.

    Attributes
    ----------
    num_boxes_ndim : jnp.ndarray, shape (num_levels,)
        Number of boxes retained at each level (outer boxes minus inner cutout).
    centres_ndim : jnp.ndarray, shape (total_num_boxes, ndim)
        Integer Fourier-index centres of every box across all levels.
    total_num_boxes : int
        Sum of ``num_boxes_ndim``.

    Examples
    --------
    >>> from beamax import DyadicDecomposition
    >>> decomp = DyadicDecomposition(
    ...     num_levels=2,
    ...     N=(64, 64),
    ...     num_boxes_levels=(4, 8),
    ...     box_aspect_ratio=(1, 1),
    ... )
    >>> int(decomp.total_num_boxes)
    76

    Notes
    -----
    - Raises ``ValueError`` during construction if the parameter combination
      cannot tile the grid cleanly (e.g. box lengths do not divide the
      smallest side).
    - ``num_levels``, ``N``, ``num_boxes_levels``, ``box_aspect_ratio``,
      and ``boxes_per_dim_levels`` are marked static for JAX pytree flattening.
    """

    num_levels: int = eqx.field(static=True)
    N: Tuple[int, ...] = eqx.field(static=True)
    num_boxes_levels: Tuple[int, ...] = eqx.field(static=True)
    box_aspect_ratio: Tuple[int, ...] = eqx.field(static=True)
    boxes_per_dim_levels: Optional[Tuple[Tuple[int, ...], ...]] = eqx.field(static=True)

    num_boxes_ndim: jnp.ndarray
    centres_ndim: jnp.ndarray
    total_num_boxes: int = eqx.field(static=True)

    def __init__(
        self,
        num_levels: int,
        N: Tuple[int, ...],
        num_boxes_levels: Tuple[int, ...],
        box_aspect_ratio: Tuple[int, ...],
        boxes_per_dim_levels: Optional[Tuple[Tuple[int, ...], ...]] = None,
    ):
        """
        Construct a dyadic decomposition and precompute box metadata.

        Parameters
        ----------
        num_levels : int
            Number of dyadic levels.
        N : Tuple[int, ...]
            Grid shape per spatial axis.
        num_boxes_levels : Tuple[int, ...]
            Number of boxes along the smallest axis at each level.
        box_aspect_ratio : Tuple[int, ...]
            Per-axis integer box aspect ratio.
        boxes_per_dim_levels : Tuple[Tuple[int, ...], ...], optional
            Explicit per-axis box counts per level.

        Raises
        ------
        ValueError
            If the requested tiling is invalid.
        """
        validate_params(
            num_levels, N, num_boxes_levels, box_aspect_ratio, boxes_per_dim_levels
        )

        self.num_levels = int(num_levels)
        self.N = tuple(int(x) for x in N)
        self.num_boxes_levels = tuple(int(x) for x in num_boxes_levels)
        self.box_aspect_ratio = tuple(int(x) for x in box_aspect_ratio)
        self.boxes_per_dim_levels = boxes_per_dim_levels

        nb_per_level = self._compute_num_boxes_ndim_py()
        centres = self._compute_nd_centres_py(nb_per_level)

        self.num_boxes_ndim = jnp.asarray(nb_per_level, dtype=jnp.int32)
        self.centres_ndim = jnp.asarray(centres, dtype=jnp.int32)
        self.total_num_boxes = int(self.num_boxes_ndim.sum())

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions (``len(N)``)."""
        return len(self.N)

    @property
    def num_boxes_ndim_cumsum(self) -> jnp.ndarray:
        """Cumulative ``num_boxes_ndim``; useful for indexing per-level slices of ``centres_ndim``."""
        return jnp.cumsum(self.num_boxes_ndim)

    @property
    def fourier_meshgrid(self) -> jnp.ndarray:
        """
        Zero-centred Fourier-index meshgrid.

        Returns
        -------
        jnp.ndarray, shape (*N, ndim)
            Stacked meshgrid of axes ``arange(-Ni//2, Ni//2)``.
        """
        axes = [jnp.arange(-n // 2, n // 2, dtype=jnp.int32) for n in self.N]
        return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)

    @property
    def scaling(self) -> jnp.ndarray:
        """
        Per-axis scaling factor that maps the domain aspect ratio onto the box aspect ratio.

        Returns
        -------
        jnp.ndarray, shape (ndim,)
            ``(N // min(N)) / box_aspect_ratio``.
        """
        N_arr = jnp.asarray(self.N)
        domain_aspect = N_arr // N_arr.min()
        return domain_aspect / jnp.asarray(self.box_aspect_ratio)

    @property
    def box_lengths(self) -> jnp.ndarray:
        """
        Box side length (in Fourier indices, along the smallest axis) at each level.

        Returns
        -------
        jnp.ndarray, shape (num_levels,), int32
            Length at level ``L`` is ``min(N) // (num_boxes_levels[-1] * 2**(num_levels-1-L))``;
            halves moving coarse → fine.
        """
        N_ref = int(min(self.N))
        levels_desc = np.arange(self.num_levels - 1, -1, -1, dtype=np.int32)
        denom_last = self.num_boxes_levels[-1]
        bl = (N_ref // (denom_last * (2**levels_desc))).astype(np.int32)
        return jnp.asarray(bl)

    # ---------------------- Pure-Python builders ---------------------------

    def _outer_boxes_per_axis_py(self, lvl: int) -> np.ndarray:
        """
        Return per-axis outer box counts (NumPy int64).

        Parameters
        ----------
        lvl : int
            Dyadic level index.

        Returns
        -------
        np.ndarray, shape (ndim,), dtype=int64
            Number of boxes along each axis at the level.

        Notes
        -----
        If ``boxes_per_dim_levels`` is provided, it is used directly.
        Otherwise counts are derived from ``num_boxes_levels`` and the domain
        aspect ratio.
        """
        if self.boxes_per_dim_levels is not None:
            return np.asarray(self.boxes_per_dim_levels[lvl], dtype=np.int64)

        # Original behaviour:
        N_arr = np.asarray(self.N, dtype=np.int64)
        domain_aspect = N_arr // int(N_arr.min())
        bar = np.asarray(self.box_aspect_ratio, dtype=np.int64)
        scaling = (domain_aspect / bar).astype(np.int64)  # truncates like before

        nb_outer = int(self.num_boxes_levels[lvl])
        per_axis = nb_outer * scaling
        return per_axis.astype(np.int64)

    def _compute_num_boxes_ndim_py(self) -> np.ndarray:
        """
        Compute the number of retained boxes at each level.

        Returns
        -------
        np.ndarray, shape (num_levels,), dtype=int32
            Per-level box counts after removing inner cutouts covered by
            finer levels.
        """
        boxes_total: List[int] = []
        cutouts: List[int] = []

        for L in range(self.num_levels):
            per_axis = self._outer_boxes_per_axis_py(L)
            total = int(np.prod(per_axis))
            boxes_total.append(total)
            cutout = int(total // (2**self.ndim)) if L < self.num_levels - 1 else 0
            cutouts.append(cutout)

        result = []
        for i in range(self.num_levels):
            if i == 0:
                result.append(boxes_total[0])
            else:
                result.append(int(boxes_total[i] - cutouts[i - 1]))

        return np.asarray(result, dtype=np.int32)

    def _centres_1d(self, length: int, num_outer: int, scale: int) -> np.ndarray:
        """
        Construct one-dimensional box centres for an axis.

        Parameters
        ----------
        length : int
            Smallest-axis box length at the level.
        num_outer : int
            Number of outer boxes along the axis.
        scale : int
            Axis aspect multiplier.

        Returns
        -------
        np.ndarray, shape (num_outer,), dtype=int32
            Integer Fourier-index centres.
        """
        step = int(length * scale)
        half = num_outer // 2
        start = -half * step + step // 2
        stop = half * step
        return np.arange(start, stop, step, dtype=np.int32)

    def _compute_nd_centres_py(self, nb_per_level: np.ndarray) -> np.ndarray:
        """
        Compute all retained N-dimensional box centres.

        Parameters
        ----------
        nb_per_level : np.ndarray, shape (num_levels,)
            Expected per-level retained box counts.

        Returns
        -------
        np.ndarray, shape (total_num_boxes, ndim), dtype=int32
            Integer Fourier-index centres across all levels.

        Raises
        ------
        ValueError
            If generated centre counts disagree with ``nb_per_level``.
        """
        N_arr = np.asarray(self.N, dtype=np.int64)
        N_ref = int(N_arr.min())
        bar = np.asarray(self.box_aspect_ratio, dtype=np.int64)

        levels_desc = list(range(self.num_levels - 1, -1, -1))
        denom_last = self.num_boxes_levels[-1]
        box_lengths = [int(N_ref // (denom_last * (2**L))) for L in levels_desc]

        centres_all = []
        bounds_prev = np.zeros((self.ndim,), dtype=np.int32)

        for lvl in range(self.num_levels):
            step_len = box_lengths[lvl]
            per_axis_counts = self._outer_boxes_per_axis_py(lvl)

            per_axis = [
                self._centres_1d(
                    step_len,
                    int(per_axis_counts[ax]),
                    int(bar[ax]),
                )
                for ax in range(self.ndim)
            ]

            mesh = np.stack(np.meshgrid(*per_axis, indexing="ij"), axis=-1).reshape(
                -1, self.ndim
            )

            if mesh.size > 0 and np.any(bounds_prev > 0):
                keep = np.logical_not(np.all(np.abs(mesh) < bounds_prev, axis=1))
                mesh = mesh[keep]

            expected = int(nb_per_level[lvl])
            if mesh.shape[0] != expected:
                raise ValueError(
                    f"Level {lvl}: expected {expected} centres, got {mesh.shape[0]}. "
                    f"(per_axis_counts={per_axis_counts.tolist()})"
                )

            centres_all.append(mesh)

            if mesh.shape[0] > 0:
                bounds_prev = np.max(np.abs(mesh), axis=0) + (step_len // 2)

        return np.concatenate(centres_all, axis=0)
