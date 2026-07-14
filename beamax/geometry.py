from typing import Callable, Optional, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial
from jaxtyping import Array, Float, Int, Num


ScalarLike = Union[int, float, Float[Array, ""]]
FieldFn = Callable[[Float[Array, "... d"]], Float[Array, "..."]]
Param = Optional[Union[FieldFn, ScalarLike, Num[Array, "..."]]]


class Domain(eqx.Module):
    """
    Axis-aligned rectangular domain with physical spacing and medium fields.

    Attributes
    ----------
    N : Tuple[int, ...]
        Grid shape per axis.
    dx : Tuple[float, ...]
        Physical spacing per axis (same length as ``N``).
    periodic : Tuple[bool, ...]
        Per-axis periodicity flags.
    cfl : float
        CFL number used to pick ``dt`` in ``generate_time_domain`` (default 0.3).
    c : Callable | float
        Speed of sound ``c(x)`` or constant (default 1500.0).
    density : Callable | float | None
        Density ``rho(x)`` or constant.
    alpha_coeff : Callable | float | None
        Absorption prefactor ``alpha0(x)``.
    lam : float
        Absorption coefficient for GB ODEs (default 0.0).
    alpha_power : Callable | float | None
        Absorption exponent ``y(x)`` in ``alpha = alpha0 * f**y``.

    Notes
    -----
    - All callable fields are evaluated lazily on accessors (pure, JAX-friendly).
    - ``N``, ``periodic`` are static; others can be traced.
    - Derived arrays (``grid``, ``sound_speed_array``, ``density_array``,
      ``alpha_coeff_array``, ``alpha_power_array``) and ``ndim`` are available
      as properties.
    """

    # geometry
    N: Tuple[int, ...] = eqx.field(static=True)
    dx: Tuple[float, ...] = eqx.field()
    periodic: Tuple[bool, ...] = eqx.field(static=True)
    cfl: float = eqx.field(default=0.3)

    # material parameters
    c: Param = eqx.field(default=1500.0)  # speed of sound (m s⁻¹)
    density: Param = eqx.field(default=1.0)  # ρ
    alpha_coeff: Param = eqx.field(default=None)  # α₀
    lam: float = eqx.field(default=0.0)  # absorption coefficient
    alpha_power: Param = eqx.field(default=None)  # y in α=α₀ fʸ

    def __init__(
        self,
        N: Tuple[int, ...],
        dx: Tuple[float, ...],
        periodic: Tuple[bool, ...],
        cfl: float = 0.3,
        c: Param = 1500.0,
        density: Param = 1.0,
        alpha_coeff: Param = None,
        lam: float = 0.0,
        alpha_power: Param = None,
    ) -> None:
        """Construct and validate a computational domain."""
        N_values = np.asarray(N)
        if N_values.ndim != 1 or N_values.size == 0:
            raise ValueError("N must contain at least one spatial dimension.")
        if any(
            isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
            for n in N_values
        ):
            raise ValueError(f"N entries must be integers; got {N}.")
        N = tuple(int(n) for n in N_values)
        if any(n <= 0 for n in N):
            raise ValueError(f"N entries must be positive; got {N}.")

        if len(dx) != len(N):
            raise ValueError(f"dx must have length {len(N)}, got {len(dx)}.")
        dx_arr = np.asarray(dx, dtype=float)
        if not np.all(np.isfinite(dx_arr)) or np.any(dx_arr <= 0):
            raise ValueError(f"dx entries must be finite and positive; got {dx}.")

        if len(periodic) != len(N):
            raise ValueError(
                f"periodic must have length {len(N)}, got {len(periodic)}."
            )
        if any(not isinstance(p, (bool, np.bool_)) for p in periodic):
            raise ValueError(f"periodic entries must be boolean; got {periodic}.")

        if not np.isfinite(cfl) or cfl <= 0:
            raise ValueError(f"cfl must be finite and positive; got {cfl}.")
        if not np.isfinite(lam) or lam < 0:
            raise ValueError(f"lam must be finite and non-negative; got {lam}.")

        def _validate_param(
            name: str,
            value: Param,
            *,
            allow_none: bool,
            strictly_positive: bool,
        ) -> Param:
            if value is None:
                if allow_none:
                    return None
                raise ValueError(f"{name} cannot be None.")
            if callable(value):
                return value if isinstance(value, Partial) else Partial(value)
            arr = np.asarray(value)
            if arr.ndim != 0 and tuple(arr.shape) != N:
                raise ValueError(
                    f"{name} must be scalar, callable, or have shape {N}; "
                    f"got {arr.shape}."
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must contain only finite values.")
            if strictly_positive and np.any(arr <= 0):
                raise ValueError(f"{name} must be strictly positive.")
            if not strictly_positive and np.any(arr < 0):
                raise ValueError(f"{name} must be non-negative.")
            return value if arr.ndim == 0 else jnp.asarray(value)

        self.N = N
        self.dx = tuple(float(x) for x in dx_arr)
        self.periodic = tuple(bool(p) for p in periodic)
        self.cfl = float(cfl)
        self.c = _validate_param("c", c, allow_none=False, strictly_positive=True)
        self.density = _validate_param(
            "density", density, allow_none=True, strictly_positive=True
        )
        self.alpha_coeff = _validate_param(
            "alpha_coeff", alpha_coeff, allow_none=True, strictly_positive=False
        )
        self.lam = float(lam)
        self.alpha_power = _validate_param(
            "alpha_power", alpha_power, allow_none=True, strictly_positive=False
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _eval(self, p: Param) -> Optional[Num[Array, "*N"]]:
        """
        Evaluate a parameter on the spatial grid.

        Parameters
        ----------
        p : Callable[[jnp.ndarray], jnp.ndarray] | scalar | None
            If callable, receives `self.grid` with shape `(*N, d)`.

        Returns
        -------
        jnp.ndarray | None
            Array broadcast to `(*N,)` if scalar; otherwise `None` if `p` is None.
        """
        if p is None:
            return None
        arr = jnp.asarray(p(self.grid) if callable(p) else p)
        if arr.ndim == 0:  # broadcast scalar
            arr = jnp.broadcast_to(arr, self.grid.shape[:-1])
        elif tuple(arr.shape) != self.N:
            raise ValueError(
                f"Medium field evaluated to shape {arr.shape}; expected {self.N}."
            )
        return arr

    @property
    def c_fn(self) -> FieldFn:
        """
        Callable sound speed, wrapping scalars into a JAX-friendly function.

        Returns
        -------
        Callable[[jnp.ndarray], jnp.ndarray]
            If `c` is already callable, returns it. Otherwise returns a function
            that broadcasts the constant `c` over the leading shape of `x`.
        """
        if callable(self.c):
            return self.c
        val = jnp.asarray(self.c)

        if val.ndim > 0:
            if tuple(val.shape) != self.N:
                raise ValueError(
                    f"Grid-valued c must have shape {self.N}; got {val.shape}."
                )
            from beamax.utils.interp import make_c_function_from_grid

            return make_c_function_from_grid(
                val,
                spacing=self.dx,
                origin=(0.0,) * self.ndim,
                boundary="wrap" if all(self.periodic) else "clamp",
            )

        def _const_c(x: Float[Array, "... d"]) -> Float[Array, "..."]:
            """
            Broadcast a constant sound speed over query coordinates.

            Parameters
            ----------
            x : jnp.ndarray, shape (..., d)
                Query coordinates.

            Returns
            -------
            jnp.ndarray, shape (...)
                Constant sound speed with the leading shape of ``x``.
            """
            return val + jnp.zeros(x.shape[:-1], dtype=val.dtype)

        return _const_c

    # ------------------------------------------------------------------
    # public accessors
    # ------------------------------------------------------------------
    @property
    def grid_size(self) -> Float[Array, " d"]:
        """
        Physical size of the domain per axis.

        Returns
        -------
        jnp.ndarray, shape (d,)
            `N * dx` per axis.
        """
        return jnp.array(self.N) * jnp.array(self.dx)

    @property
    def xmax(self) -> Float[Array, ""]:
        """
        Max extent (Euclidean norm of `grid_size`).

        Returns
        -------
        float
        """
        return jnp.linalg.norm(self.grid_size)

    @property
    def k_max(self) -> Float[Array, ""]:
        """
        Max wavenumber given sampling.

        Returns
        -------
        float
            π * min(1/dx).
        """
        return jnp.pi * jnp.min(1 / jnp.array(self.dx))

    @property
    def ndim(self) -> int:
        """
        Number of spatial dimensions.

        Returns
        -------
        int
        """
        return len(self.N)

    @property
    def grid(self) -> Float[Array, "*N d"]:
        """
        Stacked spatial coordinates.

        Returns
        -------
        jnp.ndarray, shape (*N, d)
            Meshgrid stacked along last axis.
        """
        return jnp.stack(self.generate_meshgrid()[0], axis=-1)

    @property
    def sound_speed_array(self) -> Float[Array, "*N"]:
        """
        Speed of sound evaluated on grid.

        Returns
        -------
        jnp.ndarray, shape (*N,)
        """
        out = self._eval(self.c)
        assert out is not None  # ``self.c`` defaults to 1500.0; never None.
        return out

    @property
    def density_array(self) -> Optional[Float[Array, "*N"]]:
        """
        Density evaluated on grid.

        Returns
        -------
        jnp.ndarray | None
        """
        return self._eval(self.density)

    @property
    def alpha_coeff_array(self) -> Optional[Float[Array, "*N"]]:
        """
        Absorption prefactor evaluated on grid.

        Returns
        -------
        jnp.ndarray | None
        """
        return self._eval(self.alpha_coeff)

    @property
    def alpha_power_array(self) -> Optional[Float[Array, "*N"]]:
        """
        Absorption exponent evaluated on grid.

        Returns
        -------
        jnp.ndarray | None
        """
        return self._eval(self.alpha_power)

    def compute_max_speed(self) -> Float[Array, ""]:
        """
        Maximum sound speed on grid.

        Returns
        -------
        jnp.ndarray
            Scalar (0-D array) with ``max(c)``.
        """
        return jnp.max(self.sound_speed_array)

    def compute_min_speed(self) -> Float[Array, ""]:
        """
        Minimum sound speed on grid.

        Returns
        -------
        jnp.ndarray
            Scalar (0-D array) with ``min(c)``.
        """
        return jnp.min(self.sound_speed_array)

    def generate_meshgrid(
        self,
    ) -> Tuple[list[Float[Array, "*N"]], list[Int[Array, "*N"]]]:
        """
        Spatial and Fourier meshgrids.

        Returns
        -------
        (spatial_meshgrid, fourier_meshgrid)
            Each a tuple of length `d` with arrays shaped `N[i]` per axis.
            - Spatial coordinates: 0..(N[i]-1) * dx[i].
            - Fourier indices:     -N[i]//2 .. N[i]//2 - 1.
        """
        spatial_coords = [
            jnp.arange(0, self.N[idx], 1) * self.dx[idx] for idx in range(self.ndim)
        ]
        fourier_coords = [
            jnp.arange(-self.N[idx] // 2, self.N[idx] // 2, 1)
            for idx in range(self.ndim)
        ]
        spatial_meshgrid = jnp.meshgrid(*spatial_coords, indexing="ij")
        fourier_meshgrid = jnp.meshgrid(*fourier_coords, indexing="ij")
        return spatial_meshgrid, fourier_meshgrid

    def compute_max_freq(self) -> Float[Array, ""]:
        """
        Max frequency allowed by grid / CFL proxy.

        Returns
        -------
        jnp.ndarray
            Scalar ``≈ max(c) / (2 * min(dx))``.
        """
        return self.compute_max_speed() / (2 * min(self.dx))

    def generate_time_domain(self) -> Float[Array, " Nt"]:
        """
        CFL-based uniform time grid covering one diameter-crossing.

        Returns
        -------
        jnp.ndarray, shape (Nt,)
            With ``dt = cfl * min(dx) / max(c)`` and
            ``tmax = ||grid_size|| / min(c)``, this returns ``arange(0, tmax, dt)``.
        """
        dt = self.cfl * min(self.dx) / self.compute_max_speed()
        tmax = self.xmax / self.compute_min_speed()
        time_domain = jnp.arange(0, tmax, dt)
        return time_domain


class Sensor(eqx.Module):
    """
    Sampling geometry for receivers or sources.

    Construct with exactly one of ``positions`` or ``binary_mask``. The other
    representation is derived deterministically from whatever you provided and
    is available via the :attr:`positions` / :attr:`binary_mask` properties.

    Parameters
    ----------
    domain : Domain
        The spatial domain the sensor lives on.
    positions : jnp.ndarray, shape (Ns, d), optional
        Cartesian positions in physical units.
    binary_mask : jnp.ndarray, shape (*domain.N,), optional
        Mask with 1 at sensor voxels, 0 elsewhere.

    Raises
    ------
    ValueError
        If neither or both of ``positions`` and ``binary_mask`` are provided.

    Notes
    -----
    ``positions`` are converted to the nearest integer grid index when the mask
    is derived; sub-pixel positions are quantised to grid voxels.
    """

    domain: Domain
    _positions: Optional[Float[Array, "Ns d"]]
    _binary_mask: Optional[Num[Array, "*N"]]

    def __init__(
        self,
        domain: Domain,
        positions: Optional[Float[Array, "Ns d"]] = None,
        binary_mask: Optional[Num[Array, "*N"]] = None,
    ) -> None:
        """
        Construct from positions or mask.

        Parameters
        ----------
        domain : Domain
            Spatial domain the sensor belongs to.
        positions : jnp.ndarray, shape (Ns, d), optional
            Physical sensor positions.
        binary_mask : jnp.ndarray, shape (*domain.N,), optional
            Binary sensor mask.

        Raises
        ------
        ValueError
            If neither or both of `positions` and `binary_mask` are provided.
        """
        if positions is None and binary_mask is None:
            raise ValueError("Either positions or binary_mask must be provided")
        if positions is not None and binary_mask is not None:
            raise ValueError("Cannot provide both positions and binary_mask")

        self.domain = domain

        if positions is not None:
            positions = self._normalize_positions(positions)
            self._positions = positions
            self._binary_mask = self._positions_to_mask(positions)
        else:
            assert binary_mask is not None  # exclusive-or guard above
            binary_mask = self._normalize_mask(binary_mask)
            self._binary_mask = binary_mask
            self._positions = self._mask_to_positions(binary_mask)

    def _normalize_positions(
        self, positions: Float[Array, "..."]
    ) -> Float[Array, "Ns d"]:
        """
        Validate positions and coerce a single point to shape ``(1, d)``.

        Parameters
        ----------
        positions : array-like
            Physical sensor positions.

        Returns
        -------
        jnp.ndarray, shape (Ns, d)

        Raises
        ------
        ValueError
            If the shape is invalid or any point falls outside the domain.
        """
        positions = jnp.asarray(positions)
        if positions.ndim == 1:
            positions = positions[None, :]
        if positions.ndim != 2 or positions.shape[1] != self.domain.ndim:
            raise ValueError(
                "positions must have shape (Ns, ndim) or (ndim,), "
                f"got {positions.shape} for ndim={self.domain.ndim}."
            )

        pos_np = np.asarray(positions)
        if not np.all(np.isfinite(pos_np)):
            raise ValueError("positions must contain only finite values.")
        lower = np.zeros(self.domain.ndim)
        upper = np.asarray(self.domain.grid_size)
        if np.any(pos_np < lower) or np.any(pos_np >= upper):
            raise ValueError(
                "positions must lie inside the half-open domain [0, N*dx)."
            )
        indices = np.rint(pos_np / np.asarray(self.domain.dx)).astype(int)
        indices = np.clip(indices, 0, np.asarray(self.domain.N) - 1)
        if np.unique(indices, axis=0).shape[0] != indices.shape[0]:
            raise ValueError(
                "positions must map to distinct grid points after quantisation."
            )
        return positions

    def _normalize_mask(self, mask: Num[Array, "..."]) -> Num[Array, "*N"]:
        """
        Validate and coerce a binary mask.

        Parameters
        ----------
        mask : array-like, shape (*domain.N,)

        Returns
        -------
        jnp.ndarray

        Raises
        ------
        ValueError
            If the mask shape does not match ``domain.N`` or contains no
            positive entries.
        """
        mask = jnp.asarray(mask)
        if tuple(mask.shape) != tuple(self.domain.N):
            raise ValueError(
                f"binary_mask must have shape {self.domain.N}, got {mask.shape}."
            )
        mask_np = np.asarray(mask)
        if not np.all(np.isfinite(mask_np)):
            raise ValueError("binary_mask must contain only finite values.")
        if not np.all((mask_np == 0) | (mask_np == 1)):
            raise ValueError("binary_mask must contain only binary values 0 and 1.")
        if not bool(np.any(mask_np == 1)):
            raise ValueError(
                "binary_mask must contain at least one positive active entry."
            )
        return mask

    def _positions_to_mask(self, positions: Float[Array, "Ns d"]) -> Num[Array, "*N"]:
        """
        Convert physical positions to a binary mask.

        Parameters
        ----------
        positions : jnp.ndarray, shape (Ns, d)
            Physical coordinates.

        Returns
        -------
        jnp.ndarray, shape (*N,), dtype=int
            1 at nearest grid indices (rounded), else 0.
        """
        sensor_indices = jnp.round(positions / jnp.array(self.domain.dx)).astype(int)
        upper_indices = jnp.array(self.domain.N) - 1
        sensor_indices = jnp.clip(sensor_indices, 0, upper_indices)
        mask = jnp.zeros(self.domain.N)
        return mask.at[tuple(sensor_indices.T)].set(1)

    def _mask_to_positions(self, mask: Num[Array, "*N"]) -> Float[Array, "Ns d"]:
        """
        Convert binary mask to physical positions.

        Parameters
        ----------
        mask : jnp.ndarray, shape (*N,)

        Returns
        -------
        jnp.ndarray, shape (Ns, d)
            Positions in physical units: `indices * dx`.
        """
        indices = jnp.where(mask > 0)
        positions = jnp.stack(indices, axis=1) * jnp.array(self.domain.dx)
        return positions

    @property
    def positions(self) -> Float[Array, "Ns d"]:
        """
        Sensor positions (physical units).

        Returns
        -------
        jnp.ndarray, shape (Ns, d)
        """
        out = self._positions
        assert out is not None  # always derived by __init__
        return out

    @property
    def binary_mask(self) -> Num[Array, "*N"]:
        """
        Sensor mask aligned to `domain.N`.

        Returns
        -------
        jnp.ndarray, shape (*N,), dtype=int
        """
        out = self._binary_mask
        assert out is not None  # always derived by __init__
        return out
