from __future__ import annotations
from collections.abc import Sequence
import jax.numpy as jnp
import numpy as np
from scipy import ndimage
from typing import Optional, Tuple, List
from jax import hessian, grad


_BOUNDARY_ALIASES = {
    "clamp": "nearest",
    "nearest": "nearest",
    "edge": "nearest",
    "reflect": "reflect",
    "mirror": "mirror",
    "wrap": "wrap",
    "periodic": "wrap",
}


def _canonical_boundary(boundary: str) -> str:
    try:
        return _BOUNDARY_ALIASES[boundary]
    except KeyError as exc:
        valid = ", ".join(sorted(_BOUNDARY_ALIASES))
        raise ValueError(
            f"Unsupported boundary='{boundary}'. Expected one of: {valid}"
        ) from exc


def _prepare_grid(
    c_map: jnp.ndarray,
    *,
    method: str,
    boundary: str,
    smooth_sigma: Optional[float | Sequence[float]],
) -> jnp.ndarray:
    """Return grid values or spline coefficients used by the evaluator."""
    values = jnp.asarray(c_map)
    if values.ndim == 0:
        raise ValueError("c_map must have at least one dimension.")

    if smooth_sigma is None:
        prepared = values
    else:
        sigma = np.asarray(smooth_sigma, dtype=float)
        if sigma.ndim == 0:
            sigma = np.full(values.ndim, float(sigma))
        if sigma.shape != (values.ndim,):
            raise ValueError(
                f"smooth_sigma must be scalar or length {values.ndim}, got shape {sigma.shape}."
            )
        if np.any(sigma < 0):
            raise ValueError("smooth_sigma entries must be non-negative.")

        arr = np.asarray(values)
        prepared_np = ndimage.gaussian_filter(
            arr,
            sigma=tuple(float(s) for s in sigma),
            mode=boundary,
        )
        prepared = jnp.asarray(prepared_np, dtype=values.dtype)

    if method == "linear":
        return prepared

    if method == "bspline3":
        coeff_np = ndimage.spline_filter(
            np.asarray(prepared),
            order=3,
            mode=boundary,
        )
        return jnp.asarray(coeff_np, dtype=values.dtype)

    raise ValueError("method must be 'linear' or 'bspline3'.")


def _map_indices(idx: jnp.ndarray, n: int, boundary: str) -> jnp.ndarray:
    """Map integer sample indices according to the boundary extension."""
    if n <= 0:
        raise ValueError("Grid axes must have positive length.")
    if n == 1:
        return jnp.zeros_like(idx)

    if boundary == "nearest":
        return jnp.clip(idx, 0, n - 1)
    if boundary == "wrap":
        return jnp.mod(idx, n)
    if boundary == "reflect":
        period = 2 * n
        r = jnp.mod(idx, period)
        return jnp.where(r < n, r, period - 1 - r)
    if boundary == "mirror":
        period = 2 * n - 2
        r = jnp.mod(idx, period)
        return jnp.where(r < n, r, period - r)

    raise ValueError(f"Unsupported canonical boundary='{boundary}'.")


def _linear_weights(t: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack([1.0 - t, t], axis=-1)


def _cubic_bspline_weights(t: jnp.ndarray) -> jnp.ndarray:
    """Cubic cardinal B-spline weights for offsets -1, 0, 1, 2."""
    one_minus_t = 1.0 - t
    t2 = t * t
    t3 = t2 * t
    return jnp.stack(
        [
            one_minus_t**3 / 6.0,
            (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
            (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
            t3 / 6.0,
        ],
        axis=-1,
    )


def _tensor_product_eval(
    values: jnp.ndarray,
    x: jnp.ndarray,
    *,
    boundary: str,
    offsets: Tuple[int, ...],
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate a tensor-product local interpolant."""
    d = values.ndim
    shp = values.shape
    i0 = jnp.floor(x).astype(jnp.int32)

    acc = jnp.zeros(x.shape[:-1], dtype=values.dtype)
    n_offsets = len(offsets)
    n_corners = n_offsets**d

    def corner_acc(corner_num, running):
        rem = corner_num
        wt = jnp.ones(x.shape[:-1], dtype=values.dtype)
        gather_indices = []

        for ax in range(d):
            local = rem % n_offsets
            rem //= n_offsets
            sample_idx = i0[..., ax] + offsets[local]
            mapped_idx = _map_indices(sample_idx, shp[ax], boundary)
            gather_indices.append(mapped_idx)
            wt = wt * weights[..., ax, local]

        return running + wt * values[tuple(gather_indices)]

    for corner in range(n_corners):
        acc = corner_acc(corner, acc)
    return acc


def make_c_function_from_grid(
    c_map: jnp.ndarray,
    spacing: Optional[Tuple[float, ...]] = None,
    origin: Optional[Tuple[float, ...]] = None,
    *,
    method: str = "linear",
    boundary: str = "clamp",
    smooth_sigma: Optional[float | Sequence[float]] = None,
):
    """
    Build a JAX-differentiable interpolant over a rectilinear grid.

    Parameters
    ----------
    c_map : jnp.ndarray, shape (*N,)
        Grid values.
    spacing : Tuple[float, ...] | None
        Per-axis spacing. Defaults to 1.0.
    origin : Tuple[float, ...] | None
        Per-axis origin. Defaults to 0.0.
    method : {"linear", "bspline3"}, default "linear"
        Interpolation method. ``"linear"`` preserves the original n-linear
        behaviour. ``"bspline3"`` builds a tensor-product cubic B-spline
        interpolant whose value, gradient, and Hessian are continuous away from
        boundary handling.
    boundary : {"clamp", "nearest", "reflect", "mirror", "wrap", "periodic"}, default "clamp"
        Boundary extension used by the interpolant. ``"clamp"`` is an alias for
        nearest-edge extension; ``"periodic"`` is an alias for ``"wrap"``.
    smooth_sigma : float | sequence of float | None, default None
        Optional Gaussian pre-smoothing of the grid values, with standard
        deviation measured in grid cells. Smoothing is applied once when the
        interpolant is constructed.

    Returns
    -------
    Callable[[jnp.ndarray], jnp.ndarray]
        Function `c_fun(x)` with `x` shape `(..., d)` in physical units.

    Notes
    -----
    - The default ``method="linear"`` path is piecewise-linear and
      differentiable a.e.
    - The ``method="bspline3"`` path precomputes cubic spline coefficients
      using SciPy, then evaluates the spline using JAX operations so
      ``jax.grad`` and ``jax.hessian`` can differentiate with respect to query
      coordinates.
    """
    method = method.lower()
    boundary = _canonical_boundary(boundary)
    values = _prepare_grid(
        c_map,
        method=method,
        boundary=boundary,
        smooth_sigma=smooth_sigma,
    )

    d = values.ndim
    spacing_arr = jnp.array(spacing if spacing is not None else (1.0,) * d)
    origin_arr = jnp.array(origin if origin is not None else (0.0,) * d)
    if spacing_arr.shape != (d,):
        raise ValueError(f"spacing must be length {d}, got shape {spacing_arr.shape}.")
    if origin_arr.shape != (d,):
        raise ValueError(f"origin must be length {d}, got shape {origin_arr.shape}.")

    def c_fun(coords: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the interpolant at physical coordinates.

        Parameters
        ----------
        coords : jnp.ndarray, shape (..., d)
            Physical query coordinates.

        Returns
        -------
        jnp.ndarray, shape (...)
            Interpolated grid values.
        """
        coords = jnp.asarray(coords)
        x = (coords - origin_arr) / spacing_arr  # (..., d) in index space
        i0 = jnp.floor(x)
        t = x - i0

        if method == "linear":
            weights = _linear_weights(t)
            return _tensor_product_eval(
                values,
                x,
                boundary=boundary,
                offsets=(0, 1),
                weights=weights,
            )

        weights = _cubic_bspline_weights(t)
        return _tensor_product_eval(
            values,
            x,
            boundary=boundary,
            offsets=(-1, 0, 1, 2),
            weights=weights,
        )

    return c_fun


class Interpolator:
    """
    Thin wrapper around `make_c_function_from_grid` using axis vectors.

    Parameters
    ----------
    grid_points : List[jnp.ndarray]
        1D axis arrays (length d), each strictly increasing.
    values : jnp.ndarray
        Grid values shaped to match `grid_points`.

    Methods
    -------
    __call__(x)
        Evaluate interpolant at `x` (shape `(..., d)`).
    grad(x)
        Gradient `∇c(x)`.
    hessian(x)
        Hessian `∇²c(x)`.
    """

    def __init__(
        self,
        grid_points: List[jnp.ndarray],
        values: jnp.ndarray,
        *,
        method: str = "linear",
        boundary: str = "clamp",
        smooth_sigma: Optional[float | Sequence[float]] = None,
        **_,
    ):
        """
        Construct an interpolator from axis vectors and grid values.

        Parameters
        ----------
        grid_points : List[jnp.ndarray]
            One strictly increasing one-dimensional coordinate vector per
            axis.
        values : jnp.ndarray
            Grid values with dimensionality matching ``grid_points``.
        method, boundary, smooth_sigma
            Passed through to :func:`make_c_function_from_grid`.
        **_ : dict
            Ignored compatibility keyword arguments.

        Raises
        ------
        ValueError
            If axis count and value dimensionality disagree, or if any axis
            is not one-dimensional with at least two points.
        """
        if len(grid_points) != values.ndim:
            raise ValueError(
                f"grid_points dims ({len(grid_points)}) must match values.ndim ({values.ndim})"
            )
        self.grid_points = [jnp.asarray(g) for g in grid_points]
        self.values = jnp.asarray(values)
        # Infer uniform spacing + origin per axis
        spacings = []
        origins = []
        for g in self.grid_points:
            if g.ndim != 1 or g.size < 2:
                raise ValueError("Each grid axis must be 1D with >=2 points.")
            spacings.append(float(g[1] - g[0]))
            origins.append(float(g[0]))
        self._c = make_c_function_from_grid(
            self.values,
            spacing=tuple(spacings),
            origin=tuple(origins),
            method=method,
            boundary=boundary,
            smooth_sigma=smooth_sigma,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the interpolant.

        Parameters
        ----------
        x : jnp.ndarray, shape (..., d)
            Physical query coordinates.

        Returns
        -------
        jnp.ndarray, shape (...)
            Interpolated values.
        """
        return self._c(x)

    def grad(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the gradient of the interpolant.

        Parameters
        ----------
        x : jnp.ndarray, shape (d,)
            Physical query coordinate.

        Returns
        -------
        jnp.ndarray, shape (d,)
            Gradient at ``x``.
        """
        return grad(lambda z: self._c(z))(x)

    def hessian(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the Hessian of the interpolant.

        Parameters
        ----------
        x : jnp.ndarray, shape (d,)
            Physical query coordinate.

        Returns
        -------
        jnp.ndarray, shape (d, d)
            Hessian at ``x``.
        """
        return hessian(lambda z: self._c(z))(x)
