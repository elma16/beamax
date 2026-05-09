from functools import partial

from jax import lax, vmap
import jax.numpy as jnp
from einops import rearrange
from typing import Tuple, Union, Callable, Optional


from beamax import utils
from beamax.gb import core, gb_utils, gb_solvers
from beamax.geometry import Domain, Sensor
from beamax.transforms import MSWPT
from beamax.gb.gb_solvers import SolverFn, SolverConfig


# ============================================================================
# TR Parameter Computation (unchanged)
# ============================================================================


def compute_mT_linear_system(
    xT: jnp.ndarray,
    pT: jnp.ndarray,
    mT_spc: Union[None, jnp.ndarray],
    mT_spc_time: Union[None, jnp.ndarray],
    mode: jnp.ndarray,
    c: Callable,
) -> jnp.ndarray:
    """Compute the linear system involving mT and m_init at the final time."""
    if mT_spc is None and mT_spc_time is None:
        raise ValueError("Either m_init or mT must be provided.")
    elif mT_spc_time is None:
        return mT_forward(xT, pT, mT_spc, mode, c)
    elif mT_spc is None:
        return mT_inverse(xT, pT, mT_spc_time, mode, c)


def mT_forward(
    xT: jnp.ndarray,
    pT: jnp.ndarray,
    mT_spc: jnp.ndarray,
    mode: jnp.ndarray,
    c: Callable,
) -> jnp.ndarray:
    """Given the matrix mT in space, compute the matrix mT in space-time."""
    b, d = xT.shape
    xdot = gb_utils.vmap_gp(xT, pT, mode, c)
    pdot = -gb_utils.vmap_gx(xT, pT, mode, c)

    M_lowerright = mT_spc[:, 1:, 1:]
    M_upperleft = jnp.einsum("bi, bij, bj -> b", xdot, mT_spc, xdot) - jnp.einsum(
        "bi, bi -> b", pdot, xdot
    )
    M_upperleft = jnp.reshape(M_upperleft, (b, 1, 1))
    M_upperright = (pdot - jnp.einsum("bij, bj -> bi", mT_spc, xdot))[:, 1:]
    M_upperright = jnp.reshape(M_upperright, (b, 1, d - 1))

    M = jnp.block(
        [
            [M_upperleft, M_upperright],
            [jnp.transpose(M_upperright, (0, 2, 1)), M_lowerright],
        ]
    )
    return M


def mT_inverse(xT, pT, mT_spc_time, mode, c):
    """Given the matrix mT in space-time, compute the matrix mT in space."""
    xdot = gb_utils.vmap_gp(xT, pT, mode, c)
    pdot = -gb_utils.vmap_gx(xT, pT, mode, c)

    b, d = xT.shape

    xdot_1 = xdot[:, 0:1]
    pdot_1 = pdot[:, 0:1]
    xdot_star = xdot[:, 1:]
    pdot_star = pdot[:, 1:]
    A = mT_spc_time[:, 0, 0]
    B = mT_spc_time[:, 0, 1:]
    C = mT_spc_time[:, 1:, 1:]

    M_lower_right = C
    M_star_star_xdot_star = jnp.einsum("bij,bj->bi", C, xdot_star)
    M_upper_right = ((pdot_star - M_star_star_xdot_star - B) / xdot_1).reshape(
        b, 1, d - 1
    )

    pdotxdot = jnp.einsum("bi,bi->b", pdot_1, xdot_1)
    pdot_star_dot_xdot_star = jnp.einsum("bi,bi->b", pdot_star, xdot_star)
    xdot_star_M_xdot_star = jnp.einsum("bi,bij,bj->b", xdot_star, C, xdot_star)
    Bx = jnp.einsum("bi,bi->b", B, xdot_star)

    numerator = (
        A + pdotxdot - pdot_star_dot_xdot_star + xdot_star_M_xdot_star + 2 * Bx
    ).reshape(b, 1, 1)

    M_upper_left = numerator / (xdot_1**2).reshape(b, 1, 1)
    M_lower_left = jnp.transpose(M_upper_right, (0, 2, 1))

    M = jnp.block([[M_upper_left, M_upper_right], [M_lower_left, M_lower_right]])
    return M


def find_constant_columns(arr, max_constant_axes=None):
    """
    Find columns where all values are the same (e.g., the boundary normal axis).

    Args:
        arr: Array to check, shape (N, d)
        max_constant_axes: expected #constant axes; for TR typically 1 (plane/line)

    Returns:
        constant_mask: boolean mask of constant columns
        constant_values: values of those columns (padded with 0)
        constant_axes: indices of constant columns (padded with -1)
    """
    d = arr.shape[1]
    if max_constant_axes is None:
        max_constant_axes = d

    constant_mask = jnp.all(arr == arr[0], axis=0)
    constant_axes = jnp.where(constant_mask, size=max_constant_axes, fill_value=-1)[0]
    constant_values = jnp.take(arr[0], jnp.maximum(constant_axes, 0))
    valid_mask = constant_axes >= 0
    constant_values = jnp.where(valid_mask, constant_values, 0.0)
    return constant_mask, constant_values, constant_axes


def compute_TR_parameters(
    significant_coeffs: jnp.ndarray,
    domain_data: Domain,
    wpt_data: MSWPT,
    sources: Sensor,
) -> Tuple[jnp.ndarray, ...]:
    """
    Compute the components for the time reversal problem.

    Returns:
        pts: The momentum at the final time (spatial, shape (B, d_spatial))
        Mts: The matrix M at the final time (B, d_spatial, d_spatial)
        xts: The position at the final time (B, d_spatial)
        ωs: The scale at the final time (B,)
        ats: The scale at the initial time (B, 1)
        signum: The mode of the Gaussian beam (B, 1)
        ts: The time interval per beam (B, 2)
    """
    # -------------------------------------------------------------------------
    # 0. Boundary geometry: which spatial axis is normal to the detector?
    # -------------------------------------------------------------------------
    _, const_vals, const_axes = find_constant_columns(
        sources.positions, max_constant_axes=1
    )
    valid = const_axes >= 0
    normal_axis = jnp.where(valid, const_axes, 0)[0]  # first valid or 0
    normal_value = jnp.where(valid, const_vals, 0.0)[0]

    # This is safe with jitted code: the f-string is fully resolved on host.
    # debug.print(f"TR: normal axis = {normal_axis}, value = {normal_value}")

    # Spatial dimension comes from the physical domain of the sensors
    d_spatial = sources.domain.ndim  # 1D/2D/3D spatial domain

    # -------------------------------------------------------------------------
    # 1. Dyadic decomposition / packet indices in (t, x_*)-space
    # -------------------------------------------------------------------------
    decomp = wpt_data.dyadic_decomp
    red = wpt_data.redundancy

    box_lengths = jnp.array(decomp.box_lengths)  # per level
    box_aspect = jnp.array(decomp.box_aspect_ratio)  # per axis (t, x_*)
    N_data = jnp.array(domain_data.N)  # (d_data,)
    L_phys = jnp.array(domain_data.grid_size)  # (d_data,)

    shapes = utils.compute_coeff_shapes(decomp, red, jnp.arange(decomp.num_levels))
    cumsum_boxes = jnp.r_[0, jnp.cumsum(decomp.num_boxes_ndim)]
    nn_level, nn_idx = utils.find_tensor_and_multiindex(significant_coeffs, shapes)
    box_idx = nn_idx[0, :] + cumsum_boxes[nn_level]

    # centres_hat: physical Fourier center in (t, x_*) coords
    # First axis = time frequency τ; others = tangential spatial frequencies k_tan
    centres_hat = decomp.centres_ndim[box_idx, :] / L_phys  # (B, d_data)
    norm_xi = jnp.linalg.norm(centres_hat, axis=-1, keepdims=True)  # |(τ, k_tan)|
    centres_normed = centres_hat / norm_xi

    # Split into time and tangential components in the *data* coordinates
    xi_tau = centres_normed[:, :1]  # normalized time frequency direction
    xi_tan_hat = centres_normed[:, 1:]  # normalized tangential directions

    # Physical (un-normalized) τ and k_tan — needed for correct dispersion
    tau = centres_hat[:, 0:1]  # (B, 1)
    k_tan = centres_hat[:, 1:]  # (B, d_data-1)

    # Physical box sizes and Gaussian widths in the data domain
    bl = (
        rearrange(box_lengths[nn_level], "j -> j 1") / L_phys * box_aspect
    )  # (B, d_data)
    Lls = bl * 2.0
    sigmas = bl / 2.0

    # Mode sign from sign(τ): signum = -sign τ selects outgoing vs incoming branch
    sign_tau = jnp.sign(centres_hat[:, 0])
    signum = rearrange(-sign_tau, "b -> b 1")

    ωs = rearrange(norm_xi, "b 1 -> b")  # scalar frequency parameter per beam

    # Time coordinate for each beam: where in the time grid it lives (start)
    ts = jnp.zeros((nn_idx.shape[1], 2))
    ts = ts.at[:, 0].set(nn_idx[1, :] / Lls[:, 0])  # consistent with MSWPT indexing

    # Indices along tangential axes in the data domain → physical boundary coords
    if d_spatial == 1:
        xstar_idx = nn_idx[1:, :]  # (1, B)
    else:
        xstar_idx = nn_idx[2:, :]  # (d_data-1, B)
    xstar = jnp.stack(xstar_idx, axis=-1) / Lls[:, 1:]  # (B, d_data-1)

    # -------------------------------------------------------------------------
    # 2. Map data-domain tangential coords to spatial boundary coords x_T
    # -------------------------------------------------------------------------
    B_ = xstar.shape[0]
    xts = jnp.zeros((B_, d_spatial))
    p_unit_spatial = jnp.zeros((B_, d_spatial))

    # Build tangential/normal masks with integer gathers (JIT-friendly)
    axis_ids = jnp.arange(d_spatial)
    mask_tan = axis_ids != normal_axis  # (d_spatial,)
    tan_index = jnp.where(
        mask_tan, jnp.cumsum(mask_tan.astype(jnp.int32)) - 1, 0
    ).astype(jnp.int32)

    if d_spatial == 1:
        xts = xts.at[:, 0].set(normal_value)
    else:
        xts = jnp.where(mask_tan, xstar[:, tan_index], normal_value)

    # Local wave speed at x_T
    cxts = domain_data.c(xts).reshape(-1, 1)  # (B_, 1)

    # -------------------------------------------------------------------------
    # 3. Spatial momentum direction p_unit(x_T) from (τ, k_tan)
    #
    # For 1D spatial: keep existing behaviour (which you've verified works)
    # For d_spatial > 1: use dispersion relation
    #
    #   ω = |τ|, k = (k_n, k_tan),  ω^2 = c^2 |k|^2
    #   => |k| = |ω| / c, p̂ = k / |k|
    #   => p̂_tan = c k_tan / |ω|,  p̂_n = ±sqrt(1 - |p̂_tan|^2)
    # -------------------------------------------------------------------------
    # Default: 1D spatial case – preserve your current behaviour
    if d_spatial == 1:
        p_tan_unit = xi_tan_hat  # empty for pure 1D boundary, harmless
        radicand = jnp.maximum(
            (xi_tau / cxts) ** 2 - jnp.sum(xi_tan_hat**2, axis=1, keepdims=True),
            0.0,
        )
    else:
        # Multi-D spatial case: use ω–k dispersion for the tangential angles
        tau_abs = jnp.maximum(jnp.abs(tau), 1e-6)  # avoid divide-by-zero
        # Tangential components of unit spatial momentum
        p_tan_unit = cxts * k_tan / tau_abs  # (B_, d_data-1)
        tan_norm_sq = jnp.sum(p_tan_unit**2, axis=1, keepdims=True)
        radicand = jnp.maximum(1.0 - tan_norm_sq, 0.0)

    p_n_unit = jnp.sqrt(radicand)  # (B_, 1)

    # Orient the normal component so beams point *into* the domain:
    dx_arr = jnp.array(domain_data.dx)
    grid_max = domain_data.grid_size[normal_axis] - dx_arr[normal_axis]
    inward_sign = jnp.where(
        jnp.isclose(normal_value, 0.0),
        1.0,
        jnp.where(
            jnp.isclose(normal_value, grid_max),
            -1.0,
            jnp.sign(grid_max / 2.0 - normal_value),
        ),
    )
    inward_sign = jnp.where(inward_sign == 0.0, 1.0, inward_sign)
    p_n_unit = inward_sign * p_n_unit  # (B_, 1)

    if d_spatial == 1:
        p_unit_spatial = p_unit_spatial.at[:, 0].set(p_n_unit[:, 0])
    else:
        p_unit_spatial = jnp.where(mask_tan, p_tan_unit[:, tan_index], p_n_unit)

    # Scale by 2π to get actual spatial momentum at the boundary
    pts = (2.0 * jnp.pi) * p_unit_spatial  # (B_, d_spatial)

    if d_spatial != 1:
        pts = pts / cxts  # scale by local c to get correct momentum

    # -------------------------------------------------------------------------
    # 4. Initial complex curvature and geometric amplitude
    # -------------------------------------------------------------------------
    alpha = 2j * (jnp.pi * sigmas) ** 2 / norm_xi  # (B_, d_data)

    M_init = jnp.einsum("bi,ij->bij", alpha, jnp.eye(d_spatial))

    Mts = compute_mT_linear_system(xts, pts, None, M_init, signum, domain_data.c)

    ats = jnp.prod(
        jnp.sqrt(
            (jnp.pi * rearrange(L_phys, "d -> 1 d"))
            / (Lls * rearrange(N_data, "d -> 1 d"))
        )
        * sigmas,
        axis=1,
        keepdims=True,
    )

    # -------------------------------------------------------------------------
    # 5. Grazing handling: kill contributions, keep dummy placeholders
    # -------------------------------------------------------------------------
    is_grazing_final = jnp.abs(jnp.take(pts, normal_axis, axis=1)) == 0.0
    is_grazing_final = rearrange(is_grazing_final, "b -> b 1")

    pts = jnp.where(is_grazing_final, jnp.ones_like(pts), pts)
    ats = jnp.where(is_grazing_final, jnp.zeros_like(ats), ats)

    identity_mats = 1j * jnp.eye(d_spatial)[None, :, :].repeat(B_, axis=0)
    Mts = jnp.where(is_grazing_final[:, :, None], identity_mats, Mts)

    return pts, Mts, xts, ωs, ats, signum, ts


# ============================================================================
# TR Beam Computation (refactored to match forward solver pattern)
# ============================================================================


def _compute_tr_beams(
    x0: jnp.ndarray,
    p0: jnp.ndarray,
    M0: jnp.ndarray,
    a0: jnp.ndarray,
    ω: jnp.ndarray,
    mode: jnp.ndarray,
    c: Callable,
    lam: float,
    ts: jnp.ndarray,
    sensors: jnp.ndarray,
    domain_size: jnp.ndarray,
    periodic: jnp.ndarray,
    ode_solver: SolverFn,
    sum_beams: bool = False,
    solver_config: Optional[SolverConfig] = None,
):
    """
    Unified TR beam computation function.

    Similar to _compute_beams in forward_solver_utils.py but for time reversal.
    TR always uses real-valued computation.

    Args:
        x0: Initial positions
        p0: Initial momentum vectors
        M0: Initial M matrices
        a0: Initial amplitudes
        ω: Angular frequencies
        mode: Beam modes
        c: Speed of sound function
        lam: Lambda parameter
        ts: Time points (can be per-beam)
        sensors: Sensor positions
        domain_size: Domain size
        periodic: Boundary conditions
        ode_solver: ODE solver function
        sum_beams: Whether to sum over beams

    Returns:
        Computed TR beams (summed if sum_beams=True)
    """
    beams = core.compute_gaussian_beam_real_TR(
        x0=x0,
        p0=p0,
        M0=M0,
        a0=a0,
        omega0=ω,
        mode=mode,
        c=c,
        lam=lam,
        ts=ts,
        sensors=sensors,
        domain_size=domain_size,
        periodic=periodic,
        ode_solver=ode_solver,
        solver_config=solver_config,
    )

    return jnp.sum(beams, axis=-1) if sum_beams else beams


def _aggregate_tr_beams(
    params: Tuple[jnp.ndarray, ...],
    aggregate_method: str,
    init_shape: Tuple,
    c: Callable,
    lam: float,
    sensors: jnp.ndarray,
    domain_size: jnp.ndarray,
    periodic: jnp.ndarray,
    ode_solver: SolverFn,
    solver_config: Optional[SolverConfig] = None,
):
    """
    Generic TR beam aggregation supporting scan, vmap, or direct computation.

    Similar structure to _aggregate_beams but for time reversal.
    Takes the final time point (t=0 for TR) from each batch.

    Args:
        params: Tuple of beam parameters (p0, M0, x0, ω, a0, mode, ts)
        aggregate_method: 'scan', 'vmap', or 'all'
        init_shape: Shape of result array
        c: Speed of sound function
        lam: Lambda parameter
        sensors: Sensor positions
        domain_size: Domain size
        periodic: Boundary conditions
        ode_solver: ODE solver function

    Returns:
        Aggregated TR result at final time point
    """
    (
        p0_batches,
        M0_batches,
        x0_batches,
        ω_batches,
        a0_batches,
        mode_batches,
        ts_batches,
    ) = params

    if aggregate_method == "scan":
        init = jnp.zeros(init_shape)

        def scan_fn(carry, inp):
            p0, M0, x0, ω, a0, mode, ts_batch = inp
            batch_result = _compute_tr_beams(
                x0,
                p0,
                M0,
                a0,
                ω,
                mode,
                c,
                lam,
                ts_batch,
                sensors,
                domain_size,
                periodic,
                ode_solver,
                sum_beams=True,
                solver_config=solver_config,
            )
            # Take the final time point (t=0 for TR)
            return carry + batch_result[-1, ...], None

        result, _ = lax.scan(scan_fn, init, params)
        return result

    elif aggregate_method == "vmap":
        beam_sums = vmap(
            lambda p0, M0, x0, ω, a0, mode, ts_batch: _compute_tr_beams(
                x0,
                p0,
                M0,
                a0,
                ω,
                mode,
                c,
                lam,
                ts_batch,
                sensors,
                domain_size,
                periodic,
                ode_solver,
                sum_beams=True,
                solver_config=solver_config,
            )
        )(
            p0_batches,
            M0_batches,
            x0_batches,
            ω_batches,
            a0_batches,
            mode_batches,
            ts_batches,
        )
        # Take final time point and sum over batches
        return jnp.sum(beam_sums[:, -1, ...], axis=0)

    else:  # "all"
        beams = _compute_tr_beams(
            x0_batches,
            p0_batches,
            M0_batches,
            a0_batches,
            ω_batches,
            mode_batches,
            c,
            lam,
            ts_batches,
            sensors,
            domain_size,
            periodic,
            ode_solver,
            sum_beams=True,
            solver_config=solver_config,
        )
        return beams[-1, ...]  # Final time point


def compute_TR_result(
    params: Tuple[jnp.ndarray, ...],
    c: Callable,
    lam: float,
    sensors: jnp.ndarray,
    domain_size: jnp.ndarray,
    periodic: jnp.ndarray,
    ode_solver: SolverFn = None,
    aggregate_method: str = "scan",
    solver_config: Optional[SolverConfig] = None,
    dt0: float | None = None,
) -> jnp.ndarray:
    """
    Compute time-reversal solution using Gaussian beams.

    Unified interface similar to compute_forward_result.

    Args:
        params: Tuple of beam parameters (p0, M0, x0, ω, a0, mode, ts)
        c: Speed of sound function
        lam: Lambda parameter (absorption)
        sensors: Sensor positions
        domain_size: Domain size
        periodic: Boundary conditions
        ode_solver: ODE solver function (if None, uses solve_ODE_batch_t)
        aggregate_method: 'scan', 'vmap', or 'all'
        dt0: Optional initial time step passed to solve_ODE_batch_t

    Returns:
        Time-reversed field at sensor locations (final time point)

    Notes:
        TR requires per-beam time intervals (shape (b, 2)), so the ODE solver
        must support this. Default is solve_ODE_batch_t. If passing a custom
        solver, ensure it handles per-beam time arrays correctly.
    """
    # TR requires solve_ODE_batch_t (or compatible) due to per-beam time intervals
    if ode_solver is None:
        print("Using default ODE solver: solve_ODE_batch_t for TR.")
        ode_solver = gb_solvers.solve_ODE_batch_t
    if dt0 is None and solver_config is not None and hasattr(solver_config, "dt0"):
        dt0 = getattr(solver_config, "dt0")
    if dt0 is not None:
        ode_solver = partial(ode_solver, dt0=dt0)

    init_shape = sensors.shape[:-1]

    return _aggregate_tr_beams(
        params=params,
        aggregate_method=aggregate_method,
        init_shape=init_shape,
        c=c,
        lam=lam,
        sensors=sensors,
        domain_size=domain_size,
        periodic=periodic,
        ode_solver=ode_solver,
        solver_config=solver_config,
    )
