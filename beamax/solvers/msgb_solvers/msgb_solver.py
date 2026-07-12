import math
from dataclasses import dataclass
from typing import Union, Optional, Tuple
import jax
import jax.numpy as jnp
import equinox as eqx
from jax.sharding import NamedSharding, PartitionSpec, Mesh

from beamax.solvers.msgb_solvers.forward_solver_utils import (
    compute_coefficients,
    threshold_coefficients,
    compute_forward_parameters,
    compute_forward_result,
)
from beamax.solvers.msgb_solvers.tr_solver_utils import (
    compute_TR_result,
    compute_TR_parameters,
)
from beamax.geometry import Domain, Sensor
from beamax.transforms import MSWPT
from beamax import utils
from beamax.gb.gb_solvers import SolverFn, SolverConfig
from beamax.solvers.msgb_solvers.adjoint_solver_utils import compute_adj_parameters


__all__ = ["MSGBSolver", "ShardingStrategy"]

complex_dtypes = (jnp.complex64, jnp.complex128)


def _form_adjoint_source(
    data: jnp.ndarray,
    dt: float,
    c_at_sources: jnp.ndarray,
    window: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Form the acquisition-time source for the unweighted wave equation.

    For detector residual ``r(s, x_s)`` this returns

    ``-c(x_s)**2 * d_s(window(s, x_s) * r(s, x_s))``.

    It is the acquisition-time representation of the time-reversed source in
    the continuous PAT adjoint.  A one-dimensional window is interpreted as a
    time window and broadcast over detector axes.
    """
    if window is None:
        windowed_data = data
    else:
        if window.ndim == 1 and data.ndim > 1:
            window = window.reshape((window.shape[0],) + (1,) * (data.ndim - 1))
        windowed_data = window * data

    # The adjoint construction is microlocal/Fourier based.  A centred
    # two-point difference has multiplier i*sin(Omega*dt)/dt and therefore
    # suppresses precisely the high temporal frequencies on which the
    # principal-symbol approximation operates.  Differentiate on the sampled
    # Fourier grid instead.  The documented endpoint/taper condition makes the
    # periodic extension appropriate; for real data the Nyquist contribution
    # is projected to the real derivative (hence zero, as required).
    frequencies = jnp.fft.fftfreq(windowed_data.shape[0], d=dt).astype(
        windowed_data.real.dtype
    )
    multiplier_shape = (frequencies.shape[0],) + (1,) * (windowed_data.ndim - 1)
    multiplier = (2j * jnp.pi * frequencies).reshape(multiplier_shape)
    derivative = jnp.fft.ifft(multiplier * jnp.fft.fft(windowed_data, axis=0), axis=0)
    if not jnp.issubdtype(windowed_data.dtype, jnp.complexfloating):
        derivative = derivative.real

    # ``c_at_sources`` is flat over detectors, ``(Ns,)``.  Data may carry the
    # detector grid unflattened -- a planar 3D array is ``(Nt, Ny, Nz)`` -- so
    # fold the speeds back onto those trailing axes before multiplying.  In 2D
    # (``(Nt, Ns)``) the flat vector already broadcasts and is left untouched.
    c_at_sources = jnp.asarray(c_at_sources)
    detector_shape = derivative.shape[1:]
    if c_at_sources.ndim == 1 and c_at_sources.shape != detector_shape:
        if c_at_sources.size != math.prod(detector_shape):
            raise ValueError(
                f"c_at_sources has {c_at_sources.size} detector samples, which "
                f"does not match the data detector grid {detector_shape}."
            )
        c_at_sources = c_at_sources.reshape(detector_shape)
    return -(c_at_sources**2) * derivative


def _apply_adjoint_image_weight(
    terminal_field: jnp.ndarray, c_at_image: jnp.ndarray
) -> jnp.ndarray:
    """Apply the ``c^{-2}`` weight for an unweighted image-space pairing."""
    return terminal_field / (c_at_image**2)


@dataclass(frozen=True)
class ShardingStrategy:
    """
    Strategy for sharding beam parameters across devices.

    Attributes
    ----------
    mesh : Mesh
        JAX device mesh for multi-device parallelization.
    beam_axis : str
        Mesh axis used to shard beams.
    """

    mesh: Mesh
    beam_axis: str = "x"

    def _beam_sharding_spec(self, ndim: int, *, is_batched: bool) -> PartitionSpec:
        """
        Build a partition spec for beam parameters.

        Parameters
        ----------
        ndim : int
            Number of dimensions of the parameter array.
        is_batched : bool
            Whether the parameter array has a leading batch axis.

        Returns
        -------
        PartitionSpec
            Sharding specification for the parameter array.

        Raises
        ------
        ValueError
            If a scalar or malformed batched array is requested.

        Notes
        -----
        For batched tensors `(num_batches, batch_size, ...)` used by scan/vmap
        aggregation, we keep all axes replicated. This avoids unsupported
        sharding interactions inside nested `scan`/`diffrax` transforms.
        """
        if ndim < 1:
            raise ValueError("Cannot shard scalar tensors.")

        if is_batched:
            if ndim < 2:
                raise ValueError(
                    "Batched beam parameters must have at least two dimensions."
                )
            return PartitionSpec(*([None] * ndim))

        return PartitionSpec(self.beam_axis, *([None] * (ndim - 1)))

    def shard_beam_params(
        self,
        p0: jnp.ndarray,
        M0: jnp.ndarray,
        x0: jnp.ndarray,
        omega: jnp.ndarray,
        a0: jnp.ndarray,
        modes: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, ...]:
        """
        Shard forward beam parameters along the beam dimension.

        Parameters
        ----------
        p0 : jnp.ndarray
            Beam momenta.
        M0 : jnp.ndarray
            Beam Hessian matrices.
        x0 : jnp.ndarray
            Beam positions.
        omega : jnp.ndarray
            Beam frequencies.
        a0 : jnp.ndarray
            Beam amplitudes.
        modes : jnp.ndarray
            Beam branch signs.

        Returns
        -------
        Tuple[jnp.ndarray, ...]
            Device-placed arrays with sharding specifications applied.
        """
        is_batched = p0.ndim == 3

        return (
            jax.device_put(
                p0,
                NamedSharding(
                    self.mesh, self._beam_sharding_spec(p0.ndim, is_batched=is_batched)
                ),
            ),
            jax.device_put(
                M0,
                NamedSharding(
                    self.mesh, self._beam_sharding_spec(M0.ndim, is_batched=is_batched)
                ),
            ),
            jax.device_put(
                x0,
                NamedSharding(
                    self.mesh, self._beam_sharding_spec(x0.ndim, is_batched=is_batched)
                ),
            ),
            jax.device_put(
                omega,
                NamedSharding(
                    self.mesh,
                    self._beam_sharding_spec(omega.ndim, is_batched=is_batched),
                ),
            ),
            jax.device_put(
                a0,
                NamedSharding(
                    self.mesh, self._beam_sharding_spec(a0.ndim, is_batched=is_batched)
                ),
            ),
            jax.device_put(
                modes,
                NamedSharding(
                    self.mesh,
                    self._beam_sharding_spec(modes.ndim, is_batched=is_batched),
                ),
            ),
        )

    def shard_tr_params(
        self,
        pts: jnp.ndarray,
        Mts: jnp.ndarray,
        xts: jnp.ndarray,
        omega_ts: jnp.ndarray,
        ats: jnp.ndarray,
        signum: jnp.ndarray,
        ts: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, ...]:
        """
        Shard time-reversal beam parameters along the beam dimension.

        Parameters
        ----------
        pts : jnp.ndarray
            Beam momenta at the boundary.
        Mts : jnp.ndarray
            Beam Hessians at the boundary.
        xts : jnp.ndarray
            Boundary positions.
        omega_ts : jnp.ndarray
            Beam frequencies.
        ats : jnp.ndarray
            Beam amplitudes.
        signum : jnp.ndarray
            Beam branch signs.
        ts : jnp.ndarray
            Per-beam time intervals.

        Returns
        -------
        Tuple[jnp.ndarray, ...]
            Device-placed arrays with sharding specifications applied.
        """
        is_batched = pts.ndim == 3

        return (
            jax.device_put(
                pts,
                NamedSharding(
                    self.mesh, self._beam_sharding_spec(pts.ndim, is_batched=is_batched)
                ),
            ),
            jax.device_put(
                Mts,
                NamedSharding(
                    self.mesh, self._beam_sharding_spec(Mts.ndim, is_batched=is_batched)
                ),
            ),
            jax.device_put(
                xts,
                NamedSharding(
                    self.mesh, self._beam_sharding_spec(xts.ndim, is_batched=is_batched)
                ),
            ),
            jax.device_put(
                omega_ts,
                NamedSharding(
                    self.mesh,
                    self._beam_sharding_spec(omega_ts.ndim, is_batched=is_batched),
                ),
            ),
            jax.device_put(
                ats,
                NamedSharding(
                    self.mesh, self._beam_sharding_spec(ats.ndim, is_batched=is_batched)
                ),
            ),
            jax.device_put(
                signum,
                NamedSharding(
                    self.mesh,
                    self._beam_sharding_spec(signum.ndim, is_batched=is_batched),
                ),
            ),
            jax.device_put(
                ts,
                NamedSharding(
                    self.mesh, self._beam_sharding_spec(ts.ndim, is_batched=is_batched)
                ),
            ),
        )


class MSGBSolver(eqx.Module):
    """
    Multiscale Gaussian Beam solver for the linear wave equation.

    Implements forward, time-reversal, and adjoint operators by:

    1. Decomposing the initial pressure into wave-packet coefficients via
       :class:`beamax.transforms.MSWPT`.
    2. Thresholding to retain only significant coefficients.
    3. Integrating a small Hamiltonian ODE per retained beam.
    4. Summing (or scanning) the beam contributions at the sensor positions.

    Parameters
    ----------
    thr : int or float
        Threshold value for coefficient selection. Semantics depend on
        ``thr_strat`` (e.g. absolute magnitude, percentile, top-k count).
    thr_strat : str
        Thresholding strategy; one of ``"hard"``, ``"top_n"``,
        ``"percentile"``, ``"hard_reassign"``, ``"bao_energy"``, or
        ``"perc_max_abs"``.
    batch_size : int
        Batch size along the beam axis for ODE integration. Tune to fit
        device memory; larger values amortise kernel launches.
    input_type : {"spatial", "fourier"}
        Domain the caller provides ``p0`` in.
    ode_solver : SolverFn
        Forward-time ODE integrator for beam dynamics (typically one of
        :mod:`beamax.gb.gb_solvers`).
    sum_method : str
        Method for summing beam contributions. One of ``"all_real"``,
        ``"scan_real"``, ``"vmap_real"``, ``"all_complex"``,
        ``"scan_complex"``, or ``"vmap_complex"``.
    tr_ode_solver : SolverFn, optional
        ODE integrator for the time-reversal dynamics. Falls back to
        ``ode_solver`` when ``None``.
    sharding : ShardingStrategy, optional
        Multi-device sharding strategy. ``None`` runs on a single device.
    ode_config : SolverConfig, optional
        Numerical configuration passed through to the ODE integrator. Falls
        back to ``SolverConfig.from_precision()``.
    adjoint_relative_guard : float, default=5e-2
        Dimensionless near-grazing exclusion ``Gamma / abs(tau)`` used by the
        principal-symbol adjoint. Tune together with the ODE configuration.
    """

    thr: Union[int, float] = eqx.field()
    thr_strat: str = eqx.field()
    batch_size: int = eqx.field(static=True)
    input_type: str = eqx.field(static=True)
    ode_solver: SolverFn = eqx.field()
    tr_ode_solver: SolverFn = eqx.field()
    use_real: bool = eqx.field(static=True)
    aggregate_method: str = eqx.field(static=True)
    sharding: Optional[ShardingStrategy] = eqx.field(default=None, static=True)
    ode_config: Optional[SolverConfig] = eqx.field(default=None, static=True)
    adjoint_relative_guard: float = eqx.field(default=5e-2, static=True)

    def __init__(
        self,
        thr: Union[int, float],
        thr_strat: str,
        batch_size: int,
        input_type: str,
        ode_solver: SolverFn,
        sum_method: str,
        tr_ode_solver: Optional[SolverFn] = None,
        sharding: Optional[ShardingStrategy] = None,
        ode_config: Optional[SolverConfig] = None,
        adjoint_relative_guard: float = 5e-2,
    ):
        """
        Initialize the MSGB solver.

        Parameters
        ----------
        thr : int or float
            Threshold value for coefficient selection.
        thr_strat : str
            Thresholding strategy name.
        batch_size : int
            Number of beams per batch for scan/vmap aggregation.
        input_type : {"spatial", "fourier"}
            Domain of inputs supplied to the solver.
        ode_solver : SolverFn
            Forward ODE solver.
        sum_method : str
            Aggregation mode string. Must be one of the values listed in the
            class-level parameter documentation.
        tr_ode_solver : SolverFn, optional
            ODE solver for time reversal. Defaults to ``ode_solver``.
        sharding : ShardingStrategy, optional
            Multi-device sharding strategy.
        ode_config : SolverConfig, optional
            Numerical ODE solver configuration. Defaults to
            ``SolverConfig.from_precision()``.
        adjoint_relative_guard : float, default=5e-2
            Dimensionless near-grazing exclusion for the adjoint. Must lie in
            ``[0, 1)``.
        """
        valid_thresholds = {
            "hard",
            "top_n",
            "percentile",
            "hard_reassign",
            "bao_energy",
            "perc_max_abs",
        }
        if thr_strat not in valid_thresholds:
            allowed = ", ".join(sorted(valid_thresholds))
            raise ValueError(f"thr_strat must be one of {allowed}; got {thr_strat!r}.")
        if input_type not in {"spatial", "fourier"}:
            raise ValueError(
                f"input_type must be 'spatial' or 'fourier'; got {input_type!r}."
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {batch_size}.")
        if not 0.0 <= adjoint_relative_guard < 1.0:
            raise ValueError(
                "adjoint_relative_guard must lie in [0, 1); got "
                f"{adjoint_relative_guard}."
            )

        valid_sum_methods = {
            "all_real",
            "scan_real",
            "vmap_real",
            "all_complex",
            "scan_complex",
            "vmap_complex",
        }
        if sum_method not in valid_sum_methods:
            allowed = ", ".join(sorted(valid_sum_methods))
            raise ValueError(
                f"sum_method must be one of {allowed}; got {sum_method!r}."
            )

        self.thr = thr
        self.thr_strat = thr_strat
        self.batch_size = batch_size
        self.input_type = input_type
        self.ode_solver = ode_solver
        self.tr_ode_solver = ode_solver if tr_ode_solver is None else tr_ode_solver
        self.sharding = sharding
        self.ode_config = (
            ode_config if ode_config is not None else SolverConfig.from_precision()
        )
        self.adjoint_relative_guard = float(adjoint_relative_guard)

        # Parse sum_method
        self.use_real = "real" in sum_method
        if "scan" in sum_method:
            self.aggregate_method = "scan"
        elif "vmap" in sum_method:
            self.aggregate_method = "vmap"
        else:
            self.aggregate_method = "all"

    def _replicate_array(self, arr: jnp.ndarray) -> jnp.ndarray:
        """
        Materialize an array with fully replicated sharding on the active mesh.

        Parameters
        ----------
        arr : jnp.ndarray
            Array to replicate.

        Returns
        -------
        jnp.ndarray
            Replicated array, or ``arr`` unchanged if no sharding is active.
        """
        if self.sharding is None:
            return arr
        replicated = PartitionSpec(*([None] * arr.ndim))
        return jax.device_put(arr, NamedSharding(self.sharding.mesh, replicated))

    def _prepare_forward_params_real(
        self, p0: jnp.ndarray, dpdt: jnp.ndarray, domain: Domain, wpt: MSWPT
    ) -> Tuple[jnp.ndarray, ...]:
        """
        Prepare beam parameters for a real-valued forward solve.

        Parameters
        ----------
        p0 : jnp.ndarray
            Initial pressure field.
        dpdt : jnp.ndarray
            Initial pressure time derivative.
        domain : Domain
            Physical domain.
        wpt : MSWPT
            Wave-packet transform.

        Returns
        -------
        Tuple[jnp.ndarray, ...]
            Beam parameters ``(p0s, M0s, x0s, omegas, a0s, modes)``.
        """
        c_pos = compute_coefficients(
            p0, dpdt, self.input_type, domain, wpt, mode="pos_only"
        )

        threshold = self.thr
        if self.thr_strat == "top_n":
            # ``pos_only`` has already zeroed one conjugate-frequency half of
            # every level.  Requesting more rows than the retained half-frame
            # used to select zero coefficients and propagate zero-amplitude
            # beams.  Clamp to the exact static capacity of ``_half_mask``;
            # this removes wasted trajectories without changing the field.
            frequency_half_capacity = sum(
                (end - start) // 2
                for start, end in zip(wpt.coeffs_cumsum[:-1], wpt.coeffs_cumsum[1:])
            )
            threshold = min(self.thr, frequency_half_capacity)

        coeff_pos_idx, max_pos_coeffs = threshold_coefficients(
            c_pos, threshold, self.thr_strat, wpt
        )

        p0s, M0s, x0s, ωs, a0s, modes = compute_forward_parameters(
            coeff_pos_idx, wpt, domain
        )
        a0s = a0s * max_pos_coeffs

        # Mirror to negative frequencies for real-valued field
        params_to_concat = (p0s, M0s, x0s, ωs, a0s)
        p0s, M0s, x0s, ωs, a0s = tuple(
            jnp.concatenate([p, p]) for p in params_to_concat
        )
        modes = jnp.concatenate([modes, -modes])

        # Batch if using scan or vmap
        if self.aggregate_method in ["scan", "vmap"]:
            p0s, M0s, x0s, ωs, a0s, modes = utils.batch_data(
                p0s,
                M0s,
                x0s,
                ωs,
                a0s,
                modes,
                batch_size=self.batch_size,
                zero_padded_args=(4,),
            )
        return (p0s, M0s, x0s, ωs, a0s, modes)

    def _prepare_forward_params_complex(
        self, p0: jnp.ndarray, dpdt: jnp.ndarray, domain: Domain, wpt: MSWPT
    ) -> Tuple[jnp.ndarray, ...]:
        """
        Prepare beam parameters for a complex-valued forward solve.

        Parameters
        ----------
        p0 : jnp.ndarray
            Initial pressure field.
        dpdt : jnp.ndarray
            Initial pressure time derivative.
        domain : Domain
            Physical domain.
        wpt : MSWPT
            Wave-packet transform.

        Returns
        -------
        Tuple[jnp.ndarray, ...]
            Beam parameters ``(p0s, M0s, x0s, omegas, a0s, modes)``.
        """
        c_pos, c_neg = compute_coefficients(
            p0, dpdt, self.input_type, domain, wpt, mode="both"
        )

        (coeff_pos_idx, max_pos_coeffs), (coeff_neg_idx, max_neg_coeffs) = (
            threshold_coefficients(c_pos, self.thr, self.thr_strat, wpt),
            threshold_coefficients(c_neg, self.thr, self.thr_strat, wpt),
        )

        p0s, M0s, x0s, ωs, a0s, modes = compute_forward_parameters(
            (coeff_pos_idx, coeff_neg_idx), wpt, domain
        )
        max_coeffs = jnp.concatenate([max_pos_coeffs, max_neg_coeffs])
        a0s = a0s * max_coeffs

        # Batch if using scan or vmap
        if self.aggregate_method in ["scan", "vmap"]:
            p0s, M0s, x0s, ωs, a0s, modes = utils.batch_data(
                p0s,
                M0s,
                x0s,
                ωs,
                a0s,
                modes,
                batch_size=self.batch_size,
                zero_padded_args=(4,),
            )

        return (p0s, M0s, x0s, ωs, a0s, modes)

    def _prepare_tr_params(
        self, data: jnp.ndarray, data_domain: Domain, data_wpt: MSWPT, sources
    ) -> Tuple[jnp.ndarray, ...]:
        """
        Prepare beam parameters for a time-reversal solve.

        Parameters
        ----------
        data : jnp.ndarray
            Sensor time-series data.
        data_domain : Domain
            Domain describing ``data``.
        data_wpt : MSWPT
            Wave-packet transform for ``data``.
        sources : Sensor
            Boundary source geometry.

        Returns
        -------
        Tuple[jnp.ndarray, ...]
            Time-reversal beam parameters
            ``(pts, Mts, xts, omegas, ats, signum, ts)``.
        """
        dpdt = jnp.zeros_like(data)

        c_pos, _ = compute_coefficients(
            data, dpdt, self.input_type, data_domain, data_wpt, mode="both"
        )

        coeff_idx, max_coeffs = threshold_coefficients(
            c_pos, self.thr, self.thr_strat, data_wpt
        )

        pts, Mts, xts, ωts, ats, signum, ts = compute_TR_parameters(
            coeff_idx, data_domain, data_wpt, sources
        )

        ats = ats * max_coeffs[:, None]

        # Only reshape into (num_batches, batch_size, ...) when the downstream
        # aggregator expects that layout. The "all" aggregator passes params
        # straight to `solve_ODE_batch_t` whose internal vmap strips one batch
        # axis; if we pre-batch here, that axis is mis-stripped and shapes
        # collide inside `coupled_rhs` for d > 1.
        if self.aggregate_method in ["scan", "vmap"]:
            pts, Mts, xts, ωts, ats, signum, ts = utils.batch_data(
                pts,
                Mts,
                xts,
                ωts,
                ats,
                signum,
                ts,
                batch_size=self.batch_size,
                zero_padded_args=(4,),
            )
        return pts, Mts, xts, ωts, ats, signum, ts

    def _infer_planar_surface(self, sensor_positions: jnp.ndarray, eps: float = 1e-9):
        """
        Infer a planar detector surface x_axis = const from sensor positions.

        Parameters
        ----------
        sensor_positions : jnp.ndarray, shape (Ns, d)
            Sensor coordinates.
        eps : float, default=1e-9
            Maximum standard deviation allowed on the inferred normal axis.

        Returns
        -------
        surface : Callable[[jnp.ndarray], jnp.ndarray]
            Implicit surface function.
        axis : int
            Inferred normal axis.
        coord : float
            Constant coordinate value on that axis.

        Raises
        ------
        ValueError
            If no nearly constant sensor-position axis is found.
        """
        stds = jnp.std(sensor_positions, axis=0)
        axis = int(jnp.argmin(stds))
        if stds[axis] > eps:
            raise ValueError(
                "Cannot infer planar surface from sensor positions; please provide `surface`."
            )
        coord = float(sensor_positions[0, axis])

        def surface(x):
            """
            Evaluate the inferred planar surface function.

            Parameters
            ----------
            x : jnp.ndarray, shape (d,)
                Query coordinate.

            Returns
            -------
            jnp.ndarray
                Signed distance-like residual ``x[axis] - coord``.
            """
            return x[axis] - coord

        return surface, axis, coord

    def forward(
        self,
        p0: jnp.ndarray,
        domain: Domain,
        sensors: Union[Sensor, jnp.ndarray],
        ts: jnp.ndarray,
        wpt: MSWPT,
        *,
        dpdt: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Solve the forward wave equation ``u_tt - c²∇²u = 0`` with MSGB.

        Initial conditions are ``u(0, x) = p0`` and ``u_t(0, x) = dpdt`` (zero
        for standard photoacoustic tomography).

        Parameters
        ----------
        p0 : jnp.ndarray, shape (*N,)
            Initial pressure field. Real or complex; dtype selects the
            underlying beam formulation.
        domain : Domain
            Computational domain and medium.
        sensors : Sensor or jnp.ndarray
            Sensor geometry. Either a :class:`Sensor` or an array of
            positions in physical units, shape ``(Ns, ndim)``.
        ts : jnp.ndarray, shape (Nt,)
            Time grid.
        wpt : MSWPT
            Wave-packet transform used to build the beam decomposition.
        dpdt : jnp.ndarray, optional
            Initial time derivative. Defaults to zeros (standard PAT).

        Returns
        -------
        jnp.ndarray, shape (Nt, Ns)
            Pressure at each sensor over time.
        """
        sensor_data, _ = self.forward_with_params(
            p0, domain, sensors, ts, wpt, dpdt=dpdt
        )
        return sensor_data

    @eqx.filter_jit
    def forward_with_params(
        self,
        p0: jnp.ndarray,
        domain: Domain,
        sensors: Union[Sensor, jnp.ndarray],
        ts: jnp.ndarray,
        wpt: MSWPT,
        *,
        dpdt: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
        """
        Forward MSGB solve plus diagnostic beam parameters.

        This is the explicit diagnostic variant of :meth:`forward`. Most users
        should call :meth:`forward`, which returns only sensor data.

        Returns
        -------
        sensor_data : jnp.ndarray, shape (Nt, Ns)
            Pressure at each sensor over time.
        params : tuple of jnp.ndarray
            Beam parameters used in the solve:
            ``(p0s, M0s, x0s, omegas, a0s, modes)``.
        """
        if dpdt is None:
            dpdt = jnp.zeros_like(p0)

        use_sharding = self.sharding is not None and self.aggregate_method == "all"

        sensor_positions = (
            sensors.positions
            if isinstance(sensors, Sensor)
            else sensors
            if isinstance(sensors, jnp.ndarray)
            else None
        )
        if sensor_positions is None:
            raise ValueError("Unsupported sensor type")

        # Prepare beam parameters
        if (p0.dtype in complex_dtypes) or (dpdt.dtype in complex_dtypes):
            params = self._prepare_forward_params_complex(p0, dpdt, domain, wpt)
        else:
            params = self._prepare_forward_params_real(p0, dpdt, domain, wpt)

        # Shard across devices if sharding strategy provided
        if use_sharding:
            assert self.sharding is not None  # implied by `use_sharding`
            params = self.sharding.shard_beam_params(*params)

        # Compute forward solution
        sensor_data = compute_forward_result(
            params=params,
            c=domain.c_fn,
            lam=domain.lam,
            ts=ts,
            ode_solver=self.ode_solver,
            sensors=sensor_positions,
            domain_size=domain.grid_size,
            periodic=jnp.array(domain.periodic),
            use_real=self.use_real,
            aggregate_method=self.aggregate_method,
            solver_config=self.ode_config,
        )

        if len(p0.shape) == 3:
            Nt = len(ts)
            sensor_data = sensor_data.reshape(
                Nt, p0.shape[0], p0.shape[1], order="F"
            ).reshape(Nt, p0.shape[0] * p0.shape[1])

        if use_sharding:
            sensor_data = self._replicate_array(sensor_data)

        return sensor_data, params

    def time_reversal(
        self,
        data: jnp.ndarray,
        domain: Domain,
        sensors: Sensor,
        sources: Sensor,
        ts,
        data_domain: Domain,
        data_wpt: MSWPT,
    ) -> jnp.ndarray:
        """
        MSGB time-reversal reconstruction.

        Parameters
        ----------
        data : jnp.ndarray, shape (Nt, Ns)
            Sensor time series to time-reverse.
        domain : Domain
            Reconstruction domain. Must have ``periodic`` all False —
            time-reversal here assumes free-space boundaries.
        sensors : Sensor
            Sensor geometry corresponding to ``data``.
        sources : Sensor
            Source positions used to seed the TR beams (often the same
            boundary as ``sensors``).
        ts : jnp.ndarray, shape (Nt,)
            Time grid corresponding to ``data``.
        data_domain : Domain
            Domain on which ``data`` was acquired (may differ from ``domain``
            under downsampling).
        data_wpt : MSWPT
            Wave-packet transform on ``data_domain`` used to analyse ``data``.

        Returns
        -------
        jnp.ndarray, shape (*N,)
            Reconstructed initial pressure. Scaled by 2 to match the
            standard full-field time-reversal convention.

        Raises
        ------
        ValueError
            If any axis of ``domain`` is periodic.

        Notes
        -----
        Time reversal here uses per-beam time intervals (via
        :func:`beamax.gb.solve_ODE_batch_t`) regardless of the forward
        integrator, to accommodate the variable emission time of each beam.
        """
        p0_recon, _ = self.time_reversal_with_params(
            data, domain, sensors, sources, ts, data_domain, data_wpt
        )
        return p0_recon

    @eqx.filter_jit
    def time_reversal_with_params(
        self,
        data: jnp.ndarray,
        domain: Domain,
        sensors: Sensor,
        sources: Sensor,
        ts,
        data_domain: Domain,
        data_wpt: MSWPT,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
        """
        Time-reversal MSGB solve plus diagnostic beam parameters.

        This is the explicit diagnostic variant of :meth:`time_reversal`. Most
        users should call :meth:`time_reversal`, which returns only the
        reconstructed field.

        Returns
        -------
        p0_recon : jnp.ndarray, shape (*N,)
            Reconstructed initial pressure.
        params : tuple of jnp.ndarray
            Beam parameters used in the solve.
        """
        if any(domain.periodic):
            raise ValueError(
                "The MSGB time reversal solver only supports free space boundary conditions."
            )
        use_sharding = self.sharding is not None and self.aggregate_method == "all"

        if use_sharding:
            data = self._replicate_array(data)

        sensor_positions = (
            sensors.positions
            if isinstance(sensors, Sensor)
            else sensors
            if isinstance(sensors, jnp.ndarray)
            else None
        )
        if sensor_positions is None:
            raise ValueError("Unsupported sensor type")

        # Prepare TR parameters on host
        params = self._prepare_tr_params(data, data_domain, data_wpt, sources)

        if use_sharding:
            assert self.sharding is not None  # implied by `use_sharding`
            params = self.sharding.shard_tr_params(*params)

        p0_recon = compute_TR_result(
            params=params,
            c=domain.c_fn,
            lam=domain.lam,
            sensors=sensor_positions,
            domain_size=data_domain.grid_size,
            periodic=jnp.array(domain.periodic),
            ode_solver=self.tr_ode_solver,
            aggregate_method=self.aggregate_method,
            solver_config=self.ode_config,
        )

        p0_recon = p0_recon * 2

        if use_sharding:
            p0_recon = self._replicate_array(p0_recon)

        return p0_recon, params

    def solve_ivp(
        self,
        p0: jnp.ndarray,
        dpdt: jnp.ndarray,
        domain: Domain,
        wpt: MSWPT,
        sensors: Union[Sensor, jnp.ndarray],
        ts: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Solve the wave-equation IVP with non-zero initial velocity.

        Equivalent to :meth:`forward` but requires an explicit ``dpdt``.
        Use this when ``u_t(0, x) ≠ 0`` (Cauchy data); use :meth:`forward`
        for standard photoacoustic settings where ``dpdt = 0``.

        Parameters
        ----------
        p0 : jnp.ndarray, shape (*N,)
            Initial pressure field.
        dpdt : jnp.ndarray, shape (*N,)
            Initial time derivative of the pressure.
        domain : Domain
            Computational domain.
        wpt : MSWPT
            Wave-packet transform for the beam decomposition.
        sensors : Sensor or jnp.ndarray
            Sensor geometry.
        ts : jnp.ndarray, shape (Nt,)
            Time grid.

        Returns
        -------
        jnp.ndarray
            Sensor time series. Equivalent to :meth:`forward` with explicit
            ``dpdt``.
        """
        return self.forward(p0, domain, sensors, ts, wpt, dpdt=dpdt)

    def solve_ivp_with_params(
        self,
        p0: jnp.ndarray,
        dpdt: jnp.ndarray,
        domain: Domain,
        wpt: MSWPT,
        sensors: Union[Sensor, jnp.ndarray],
        ts: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
        """
        IVP solve plus diagnostic beam parameters.

        This is the explicit diagnostic variant of :meth:`solve_ivp`.
        """
        return self.forward_with_params(p0, domain, sensors, ts, wpt, dpdt=dpdt)

    def _prepare_adj_params(
        self,
        source: jnp.ndarray,
        data_domain: Domain,
        data_wpt: MSWPT,
        sources: Sensor,
    ) -> Tuple[jnp.ndarray, ...]:
        """
        Prepare beam parameters for the principal-symbol adjoint backprojection.

        Parameters
        ----------
        source : jnp.ndarray
            Boundary source density appearing on the RHS of the unweighted
            second-order wave equation. It is already windowed, differentiated,
            and expressed on the time grid used by the backpropagator.
        data_domain : Domain
            Domain describing the (t, x_s) grid of `source`.
        data_wpt : MSWPT
            Transform used to analyse `source` in the MSWPT frame.
        sources : Sensor
            Geometry of the injection boundary (usually the same as `sensors`).

        Returns
        -------
        Tuple of beam parameters suitable for `compute_TR_result`.
        """
        # Analyse the spacetime source directly.  This is not an initial-value
        # half-wave split: applying ``compute_coefficients(source, 0, ...)``
        # would insert an erroneous factor 1/2.  The raw MSWPT coefficients
        # still cover signed temporal Fourier boxes, which are required by the
        # odd B^{-1} branch factor.
        source_coeffs = data_wpt.forward(source, self.input_type)

        coeff_idx, max_coeffs = threshold_coefficients(
            source_coeffs, self.thr, self.thr_strat, data_wpt
        )

        pts, Mts, xts, omegas, ats, signum, ts = compute_adj_parameters(
            coeff_idx,
            data_domain,
            data_wpt,
            sources,
            relative_guard=self.adjoint_relative_guard,
        )

        # Attach the (preconditioned) MSWPT coefficients to the beam amplitudes
        ats = ats * max_coeffs[:, None]

        if self.aggregate_method in ["scan", "vmap"]:
            pts, Mts, xts, omegas, ats, signum, ts = utils.batch_data(
                pts,
                Mts,
                xts,
                omegas,
                ats,
                signum,
                ts,
                batch_size=self.batch_size,
                zero_padded_args=(4,),  # ts is the zero-padded arg
            )

        return pts, Mts, xts, omegas, ats, signum, ts

    def adjoint(
        self,
        data: jnp.ndarray,
        domain: Domain,
        sensors: Union[Sensor, jnp.ndarray],
        sources: Sensor,
        ts: jnp.ndarray,
        data_domain: Domain,
        data_wpt: MSWPT,
        *,
        window: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Principal-symbol MSGB approximation of the continuous PAT adjoint.

        Parameters
        ----------
        data : jnp.ndarray
            Boundary residual r(t, x_s) on Gamma. Shape (Nt, Ns) or (Nt,),
            with time along axis 0.
        domain : Domain
            Reconstruction (image) domain where we want q(T, x).
        sensors : Sensor or jnp.ndarray
            Locations at which to evaluate the adjoint field. For image
            reconstruction this is typically `domain.grid` (so we get
            q_T on the full grid).
        sources : Sensor
            Source geometry on Gamma, used to construct the boundary
            beam parameters (same role as in time reversal).
        ts : jnp.ndarray
            Time grid, shape (Nt,). Currently not used directly, but kept
            for interface symmetry and possible future extensions.
        data_domain : Domain
            Domain describing the (t, x_s) grid of the boundary data.
            Its `dx[0]` is used as the time step dt.
        data_wpt : MSWPT
            MSWPT instance for analysing the boundary data / source.

        window : jnp.ndarray, optional
            Sampled acquisition window. It may have the same shape as ``data``
            or shape (Nt,), in which case it is broadcast over sensors. The
            default is one on the sampled acquisition array and assumes the
            residual is negligible at the temporal endpoints; otherwise pass a
            taper that vanishes there.

        Returns
        -------
        jnp.ndarray
            Principal-symbol approximation to P*data for unweighted image and
            data L2 pairings. This is not the exact transpose of the
            thresholded discrete MSGB forward solver.
        """
        q_T, _ = self.adjoint_with_params(
            data,
            domain,
            sensors,
            sources,
            ts,
            data_domain,
            data_wpt,
            window=window,
        )
        return q_T

    @eqx.filter_jit
    def adjoint_with_params(
        self,
        data: jnp.ndarray,
        domain: Domain,
        sensors: Union[Sensor, jnp.ndarray],
        sources: Sensor,
        ts: jnp.ndarray,
        data_domain: Domain,
        data_wpt: MSWPT,
        *,
        window: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
        """
        Adjoint MSGB solve plus diagnostic beam parameters.

        This is the explicit diagnostic variant of :meth:`adjoint`. Most users
        should call :meth:`adjoint`, which returns only the adjoint field.

        Returns
        -------
        q_T : jnp.ndarray
            Principal-symbol approximation to P*data on the reconstruction
            domain under unweighted L2 pairings.
        params : tuple of jnp.ndarray
            Beam parameters used internally.
        """
        if any(domain.periodic):
            raise ValueError(
                "MSGBSolver.adjoint currently assumes non-periodic spatial "
                "boundaries in the reconstruction domain."
            )

        # Where do we want to evaluate q(T, ·)?
        sensor_positions = (
            sensors.positions
            if isinstance(sensors, Sensor)
            else sensors
            if isinstance(sensors, jnp.ndarray)
            else None
        )
        if sensor_positions is None:
            raise ValueError("Unsupported sensor type for `sensors` in adjoint().")

        # ------------------------------------------------------------------
        # 1. Build the unweighted-equation source in acquisition time:
        #    F_acq = -c_Gamma^2 d_s(window * r).
        # ------------------------------------------------------------------

        dt = float(data_domain.dx[0])
        # The acquisition geometry owns the boundary medium. Using the image
        # domain here could make the c_Gamma^2 source inconsistent with the TR
        # geometry and B^{-1} multiplier when the two Domain objects differ.
        c_at_sources = sources.domain.c_fn(sources.positions)
        source = _form_adjoint_source(data, dt, c_at_sources, window)

        # ------------------------------------------------------------------
        # 2. Prepare adjoint (B^{-1}F) beams using the MSWPT + symbol logic.
        # ------------------------------------------------------------------
        params = self._prepare_adj_params(source, data_domain, data_wpt, sources)
        if self.sharding is not None:
            params = self.sharding.shard_tr_params(*params)

        # ------------------------------------------------------------------
        # 3. Propagate beams with the TR machinery, evaluate in the image
        #    domain, and apply the c^{-2} image weight required by the
        #    unweighted L2 image pairing.
        # ------------------------------------------------------------------
        q_T = compute_TR_result(
            params=params,
            c=domain.c_fn,
            lam=domain.lam,
            sensors=sensor_positions,
            domain_size=domain.grid_size,
            periodic=jnp.array(domain.periodic),
            ode_solver=self.tr_ode_solver,
            aggregate_method=self.aggregate_method,
            solver_config=self.ode_config,
        )

        q_T = _apply_adjoint_image_weight(q_T, domain.c_fn(sensor_positions))

        return q_T, params
