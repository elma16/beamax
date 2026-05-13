from __future__ import annotations
import jax.numpy as jnp
from typing import Optional, Tuple, List
from jax import hessian, grad


def make_c_function_from_grid(
    c_map: jnp.ndarray,
    spacing: Optional[Tuple[float, ...]] = None,
    origin: Optional[Tuple[float, ...]] = None,
):
    """
    Build a JAX-differentiable n-linear interpolant over a rectilinear grid.

    Parameters
    ----------
    c_map : jnp.ndarray, shape (*N,)
        Grid values.
    spacing : Tuple[float, ...] | None
        Per-axis spacing. Defaults to 1.0.
    origin : Tuple[float, ...] | None
        Per-axis origin. Defaults to 0.0.

    Returns
    -------
    Callable[[jnp.ndarray], jnp.ndarray]
        Function `c_fun(x)` with `x` shape `(..., d)` in physical units.

    Notes
    -----
    - Piecewise-linear; differentiable a.e.; gradients via `jax.grad`.
    - Index clamping at grid boundaries.
    """
    d = c_map.ndim
    shp = jnp.array(c_map.shape)

    spacing = jnp.array(spacing if spacing is not None else (1.0,) * d)
    origin = jnp.array(origin if origin is not None else (0.0,) * d)

    def c_fun(coords: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the n-linear interpolant at physical coordinates.

        Parameters
        ----------
        coords : jnp.ndarray, shape (..., d)
            Physical query coordinates.

        Returns
        -------
        jnp.ndarray, shape (...)
            Interpolated grid values.
        """
        # coords: (..., d) in physical units
        x = (coords - origin) / spacing  # (..., d) in index space
        i0 = jnp.floor(x).astype(jnp.int32)
        t = jnp.clip(x - i0, 0.0, 1.0)  # (..., d)
        i0 = jnp.clip(i0, 0, shp - 1)
        i1 = jnp.clip(i0 + 1, 0, shp - 1)

        # Iterate over 2^d corners via integer bit masks
        def corner_acc(mask, acc):
            """
            Add one hypercube corner contribution to the interpolation sum.

            Parameters
            ----------
            mask : int
                Corner bit mask in ``[0, 2**d)``.
            acc : jnp.ndarray, shape (...)
                Running interpolation sum.

            Returns
            -------
            jnp.ndarray, shape (...)
                Updated interpolation sum.
            """
            # mask in [0, 2^d)
            bits = jnp.array(
                [(mask >> k) & 1 for k in range(d)], dtype=jnp.int32
            )  # (d,)
            idx = jnp.where(bits == 0, i0, i1)  # (..., d)
            wt = jnp.prod(jnp.where(bits == 0, 1.0 - t, t), axis=-1)  # (...,)

            # Gather corner values
            corner_val = c_map[tuple([idx[..., k] for k in range(d)])]  # (...)
            return acc + wt * corner_val

        acc = jnp.zeros(x.shape[:-1], dtype=c_map.dtype)
        for m in range(1 << d):
            acc = corner_acc(m, acc)
        return acc

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

    def __init__(self, grid_points: List[jnp.ndarray], values: jnp.ndarray, **_):
        """
        Construct an interpolator from axis vectors and grid values.

        Parameters
        ----------
        grid_points : List[jnp.ndarray]
            One strictly increasing one-dimensional coordinate vector per
            axis.
        values : jnp.ndarray
            Grid values with dimensionality matching ``grid_points``.
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
            self.values, spacing=tuple(spacings), origin=tuple(origins)
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
