from typing import Callable, Tuple, Optional, Union
import jax.numpy as jnp
import equinox as eqx

ArrayLike = Union[int, float, jnp.ndarray]
Param = Optional[Union[Callable[[jnp.ndarray], jnp.ndarray], ArrayLike]]


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
    periodic: Tuple[bool, ...] = eqx.field()
    cfl: float = eqx.field(default=0.3)

    # material parameters
    c: Param = eqx.field(default=1500.0, static=True)  # speed of sound (m s⁻¹)
    density: Param = eqx.field(default=1.0)  # ρ
    alpha_coeff: Param = eqx.field(default=None)  # α₀
    lam: float = eqx.field(default=0.0)  # absorption coefficient
    alpha_power: Param = eqx.field(default=None)  # y in α=α₀ fʸ

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _eval(self, p: Param) -> Optional[jnp.ndarray]:
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
        arr = p(self.grid) if callable(p) else jnp.asarray(p)
        if arr.ndim == 0:  # broadcast scalar
            arr = jnp.broadcast_to(arr, self.grid.shape[:-1])
        return arr

    @property
    def c_fn(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
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

        def _const_c(x):
            return val + jnp.zeros(x.shape[:-1], dtype=val.dtype)

        return _const_c

    # ------------------------------------------------------------------
    # public accessors
    # ------------------------------------------------------------------
    @property
    def grid_size(self) -> jnp.ndarray:
        """
        Physical size of the domain per axis.

        Returns
        -------
        jnp.ndarray, shape (d,)
            `N * dx` per axis.
        """
        return jnp.array(self.N) * jnp.array(self.dx)

    @property
    def xmax(self) -> float:
        """
        Max extent (Euclidean norm of `grid_size`).

        Returns
        -------
        float
        """
        return jnp.linalg.norm(self.grid_size)

    @property
    def k_max(self) -> float:
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
    def grid(self) -> jnp.ndarray:
        """
        Stacked spatial coordinates.

        Returns
        -------
        jnp.ndarray, shape (*N, d)
            Meshgrid stacked along last axis.
        """
        return jnp.stack(self.generate_meshgrid()[0], axis=-1)

    @property
    def sound_speed_array(self) -> jnp.ndarray:
        """
        Speed of sound evaluated on grid.

        Returns
        -------
        jnp.ndarray, shape (*N,)
        """
        return self._eval(self.c)

    @property
    def density_array(self) -> Optional[jnp.ndarray]:
        """
        Density evaluated on grid.

        Returns
        -------
        jnp.ndarray | None
        """
        return self._eval(self.density)

    def compute_max_speed(self) -> jnp.ndarray:
        """
        Maximum sound speed on grid.

        Returns
        -------
        jnp.ndarray
            Scalar (0-D array) with ``max(c)``.
        """
        return jnp.max(self.sound_speed_array)

    def compute_min_speed(self) -> jnp.ndarray:
        """
        Minimum sound speed on grid.

        Returns
        -------
        jnp.ndarray
            Scalar (0-D array) with ``min(c)``.
        """
        return jnp.min(self.sound_speed_array)

    def generate_meshgrid(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
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

    def compute_max_freq(self) -> jnp.ndarray:
        """
        Max frequency allowed by grid / CFL proxy.

        Returns
        -------
        jnp.ndarray
            Scalar ``≈ max(c) / (2 * min(dx))``.
        """
        return self.compute_max_speed() / (2 * min(self.dx))

    def generate_time_domain(self) -> jnp.ndarray:
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
    _positions: Optional[jnp.ndarray]
    _binary_mask: Optional[jnp.ndarray]

    def __init__(
        self,
        domain: Domain,
        positions: Optional[jnp.ndarray] = None,
        binary_mask: Optional[jnp.ndarray] = None,
    ):
        """
        Construct from positions or mask.

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
            self._positions = positions
            self._binary_mask = self._positions_to_mask(positions)
        else:
            self._binary_mask = binary_mask
            self._positions = self._mask_to_positions(binary_mask)

    def _positions_to_mask(self, positions: jnp.ndarray) -> jnp.ndarray:
        """
        Convert positions (physical) → binary mask.

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
        mask = jnp.zeros(self.domain.N)
        return mask.at[tuple(sensor_indices.T)].set(1)

    def _mask_to_positions(self, mask: jnp.ndarray) -> jnp.ndarray:
        """
        Convert binary mask → positions (physical).

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
    def positions(self) -> jnp.ndarray:
        """
        Sensor positions (physical units).

        Returns
        -------
        jnp.ndarray, shape (Ns, d)
        """
        return self._positions

    @property
    def binary_mask(self) -> jnp.ndarray:
        """
        Sensor mask aligned to `domain.N`.

        Returns
        -------
        jnp.ndarray, shape (*N,), dtype=int
        """
        return self._binary_mask
