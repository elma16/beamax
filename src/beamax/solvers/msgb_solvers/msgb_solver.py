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
    compute_TR_result,  # New unified function
    compute_TR_parameters,
)
from beamax.geometry import Domain, Sensor
from beamax.transforms import MSWPT
from beamax import utils
from beamax.gb.gb_solvers import SolverFn, SolverConfig
from beamax.solvers.msgb_solvers.adjoint_solver_utils import compute_adj_parameters


__all__ = ["MSGBSolver", "ShardingStrategy"]

complex_dtypes = (jnp.complex64, jnp.complex128)


@dataclass(frozen=True)
class ShardingStrategy:
    """
    Strategy for sharding beam parameters across devices.

    Attributes:
        mesh: JAX device mesh for multi-device parallelization
        beam_axis: Which mesh axis to shard beams along (default: "x")
    """

    mesh: Mesh
    beam_axis: str = "x"

    def _beam_sharding_spec(self, ndim: int, *, is_batched: bool) -> PartitionSpec:
        """
        Build a partition spec for beam parameters.

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
        """Shard beam parameters along the beam dimension."""
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
        """Shard time-reversal parameters along the beam dimension."""
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
        Thresholding strategy; one of ``"top_k"``, ``"percentile"``,
        ``"magnitude"``, or another value understood by
        :func:`threshold_coefficients`.
    batch_size : int
        Batch size along the beam axis for ODE integration. Tune to fit
        device memory; larger values amortise kernel launches.
    input_type : {"spatial", "fourier"}
        Domain the caller provides ``p0`` in.
    ode_solver : SolverFn
        Forward-time ODE integrator for beam dynamics (typically one of
        :mod:`beamax.gb.gb_solvers`).
    sum_method : str
        Method for summing beam contributions. Recognised substrings include
        ``"real"`` (use real-valued beam paths), ``"scan"`` (use
        :func:`jax.lax.scan` over the beam dimension) and ``"vmap"``
        (use :func:`jax.vmap`).
    tr_ode_solver : SolverFn, optional
        ODE integrator for the time-reversal dynamics. Falls back to
        ``ode_solver`` when ``None``.
    sharding : ShardingStrategy, optional
        Multi-device sharding strategy. ``None`` runs on a single device.
    ode_config : SolverConfig, optional
        Numerical configuration passed through to the ODE integrator. Falls
        back to ``SolverConfig.from_precision()``.
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
    ):
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

        # Parse sum_method
        self.use_real = "real" in sum_method
        if "scan" in sum_method:
            self.aggregate_method = "scan"
        elif "vmap" in sum_method:
            self.aggregate_method = "vmap"
        else:
            self.aggregate_method = "all"

    def _replicate_array(self, arr: jnp.ndarray) -> jnp.ndarray:
        """Materialize an array with fully replicated sharding on the active mesh."""
        if self.sharding is None:
            return arr
        replicated = PartitionSpec(*([None] * arr.ndim))
        return jax.device_put(arr, NamedSharding(self.sharding.mesh, replicated))

    def _prepare_forward_params_real(
        self, p0: jnp.ndarray, dpdt: jnp.ndarray, domain: Domain, wpt: MSWPT
    ) -> Tuple[jnp.ndarray, ...]:
        """Prepare beam parameters for real-valued forward solve."""
        c_pos = compute_coefficients(
            p0, dpdt, self.input_type, domain, wpt, mode="pos_only"
        )

        coeff_pos_idx, max_pos_coeffs = threshold_coefficients(
            c_pos, self.thr, self.thr_strat, wpt
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
        """Prepare beam parameters for complex-valued forward solve."""
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
        """Prepare beam parameters for time-reversal solve."""
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
        """
        stds = jnp.std(sensor_positions, axis=0)
        axis = int(jnp.argmin(stds))
        if stds[axis] > eps:
            raise ValueError(
                "Cannot infer planar surface from sensor positions; please provide `surface`."
            )
        coord = float(sensor_positions[0, axis])

        def surface(x):
            return x[axis] - coord

        return surface, axis, coord

    @eqx.filter_jit
    def forward(
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
        sensor_data : jnp.ndarray, shape (Nt, Ns)
            Pressure at each sensor over time.
        params : tuple of jnp.ndarray
            Beam parameters used in the solve, exposed for diagnostics:
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

    @eqx.filter_jit
    def time_reversal(
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
        p0_recon : jnp.ndarray, shape (*N,)
            Reconstructed initial pressure. Scaled by 2 to match the k-Wave
            time-reversal convention.
        params : tuple of jnp.ndarray
            Beam parameters used in the solve, exposed for diagnostics.

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

    @eqx.filter_jit
    def solve_ivp(
        self,
        p0: jnp.ndarray,
        dpdt: jnp.ndarray,
        domain: Domain,
        wpt: MSWPT,
        sensors: Union[Sensor, jnp.ndarray],
        ts: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
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
        Same as :meth:`forward`.
        """
        return self.forward(p0, domain, sensors, ts, wpt, dpdt=dpdt)

    def _prepare_adj_params(
        self,
        source: jnp.ndarray,
        data_domain: Domain,
        data_wpt: MSWPT,
        sources: Sensor,
    ) -> Tuple[jnp.ndarray, ...]:
        """
        Prepare beam parameters for the Arridge-style adjoint solve.

        Parameters
        ----------
        source : jnp.ndarray
            Boundary *mass source* F(t, x_s) appearing on the RHS of the
            second-order adjoint wave equation (already windowed and
            time-reversed / differentiated as needed).
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
        # Analyse the source in the MSWPT frame; we only need the positive
        # "half" since the underlying beam dynamics are real.
        dpdt = jnp.zeros_like(source)
        c_pos, _ = compute_coefficients(
            source, dpdt, self.input_type, data_domain, data_wpt, mode="both"
        )

        coeff_idx, max_coeffs = threshold_coefficients(
            c_pos, self.thr, self.thr_strat, data_wpt
        )

        pts, Mts, xts, omegas, ats, signum, ts = compute_adj_parameters(
            coeff_idx, data_domain, data_wpt, sources
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

    @eqx.filter_jit
    def adjoint(
        self,
        data: jnp.ndarray,
        domain: Domain,
        sensors: Union[Sensor, jnp.ndarray],
        sources: Sensor,
        ts: jnp.ndarray,
        data_domain: Domain,
        data_wpt: MSWPT,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
        """
        MSGB adjoint solve (Arridge-style): F = w ∂_t r(T - t), then B^{-1}F + TR.

        Parameters
        ----------
        data : jnp.ndarray
            Boundary measurement r(t, x_s) on Gamma, or (if use_raw_source=True)
            an already-formed adjoint source F(t, x_s). Shape (Nt, Ns) or (Nt,)
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

        Keyword Parameters
        ------------------
        use_raw_source : bool, default False
            If False (default), `data` is interpreted as a boundary
            measurement r(t, x_s) and we internally form the adjoint
            source

                F(t, x_s) = w(x_s) ∂_t r(T - t, x_s),

            using a simple finite-difference in time and unit weights w≡1.
            If True, `data` is assumed to already be F(t, x_s) and is
            passed to `_prepare_adj_params` unchanged.

        Returns
        -------
        q_T : jnp.ndarray
            Adjoint field q(T, x) on the reconstruction domain (same
            shape as a forward initial condition).
        params : Tuple
            Beam parameters used internally (for inspection / debugging).
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
        # 1. Build F(t, x_s) from measured data r(t, x_s), unless user has
        #    explicitly asked to treat `data` as a raw source.
        # ------------------------------------------------------------------

        # data is r(t, x_s) on [0, T]; we construct F(t, x_s) ≈ ∂_t r(T - t, x_s)
        # using a simple finite-difference in time.
        dt = float(data_domain.dx[0])
        r = data * domain.c_fn(sensor_positions)[0, :]
        source = jnp.gradient(r, dt, axis=0)

        # ------------------------------------------------------------------
        # 2. Prepare adjoint (B^{-1}F) beams using the MSWPT + symbol logic.
        # ------------------------------------------------------------------
        params = self._prepare_adj_params(source, data_domain, data_wpt, sources)
        if self.sharding is not None:
            params = self.sharding.shard_tr_params(*params)

        # ------------------------------------------------------------------
        # 3. Propagate beams with the TR machinery, but evaluate in the
        #    reconstruction domain (domain.grid_size).
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

        # should maybe rescale by 2/3?
        q_T = -q_T

        return q_T, params
