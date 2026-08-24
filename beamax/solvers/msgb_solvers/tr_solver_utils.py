import math

import jax
from jax import lax
import jax.numpy as jnp
from einops import rearrange
from typing import Tuple, Union, Callable, Optional


from beamax import utils
from beamax.gb import core, gb_utils, gb_solvers
from beamax.geometry import Domain, Sensor
from beamax.transforms import MSWPT, compute_frame_phase
from beamax.gb.gb_solvers import SolverFn, SolverConfig
from beamax.gb.pallas_config import PallasConfig


def compute_mT_linear_system(
    xT: jnp.ndarray,
    pT: jnp.ndarray,
    mT_spc: Union[None, jnp.ndarray],
    mT_spc_time: Union[None, jnp.ndarray],
    mode: jnp.ndarray,
    c: Callable,
) -> jnp.ndarray:
    """
    Compute the linear system relating spatial and spacetime Hessians.

    Parameters
    ----------
    xT : jnp.ndarray, shape (b, d)
        Beam positions at the final time.
    pT : jnp.ndarray, shape (b, d)
        Beam momenta at the final time.
    mT_spc : jnp.ndarray or None
        Spatial Hessian representation.
    mT_spc_time : jnp.ndarray or None
        Spacetime Hessian representation.
    mode : jnp.ndarray
        Beam branch signs.
    c : Callable
        Sound-speed function.

    Returns
    -------
    jnp.ndarray
        Converted Hessian representation.

    Raises
    ------
    ValueError
        If both Hessian representations are ``None``.
    """
    if mT_spc is None and mT_spc_time is None:
        raise ValueError("Either m_init or mT must be provided.")
    elif mT_spc_time is None:
        assert mT_spc is not None
        return mT_forward(xT, pT, mT_spc, mode, c)
    elif mT_spc is None:
        return mT_inverse(xT, pT, mT_spc_time, mode, c)
    raise ValueError("Provide exactly one of mT_spc and mT_spc_time.")


def mT_forward(
    xT: jnp.ndarray,
    pT: jnp.ndarray,
    mT_spc: jnp.ndarray,
    mode: jnp.ndarray,
    c: Callable,
) -> jnp.ndarray:
    """
    Convert final-time spatial Hessians to spacetime Hessians.

    Parameters
    ----------
    xT : jnp.ndarray, shape (b, d)
        Beam positions at the final time.
    pT : jnp.ndarray, shape (b, d)
        Beam momenta at the final time.
    mT_spc : jnp.ndarray, shape (b, d, d)
        Spatial Hessian matrices.
    mode : jnp.ndarray
        Beam branch signs.
    c : Callable
        Sound-speed function.

    Returns
    -------
    jnp.ndarray, shape (b, d, d)
        Spacetime Hessian matrices.
    """
    b, d = xT.shape
    xdot = gb_utils.vmap_gp(xT, pT, mode, c)
    pdot = -gb_utils.vmap_gx(xT, pT, mode, c)

    M_lowerright = mT_spc[:, 1:, 1:]
    M_upperleft = jnp.einsum(
        "bi, bij, bj -> b",
        xdot,
        mT_spc,
        xdot,
        precision=jax.lax.Precision.HIGHEST,
    ) - jnp.einsum(
        "bi, bi -> b",
        pdot,
        xdot,
        precision=jax.lax.Precision.HIGHEST,
    )
    M_upperleft = jnp.reshape(M_upperleft, (b, 1, 1))
    M_upperright = (
        pdot
        - jnp.einsum(
            "bij, bj -> bi",
            mT_spc,
            xdot,
            precision=jax.lax.Precision.HIGHEST,
        )
    )[:, 1:]
    M_upperright = jnp.reshape(M_upperright, (b, 1, d - 1))

    M = jnp.block(
        [
            [M_upperleft, M_upperright],
            [jnp.transpose(M_upperright, (0, 2, 1)), M_lowerright],
        ]
    )
    return M


def mT_inverse(xT, pT, mT_spc_time, mode, c):
    """
    Convert final-time spacetime Hessians to spatial Hessians.

    Parameters
    ----------
    xT : jnp.ndarray, shape (b, d)
        Beam positions at the final time.
    pT : jnp.ndarray, shape (b, d)
        Beam momenta at the final time.
    mT_spc_time : jnp.ndarray, shape (b, d, d)
        Spacetime Hessian matrices.
    mode : jnp.ndarray
        Beam branch signs.
    c : Callable
        Sound-speed function.

    Returns
    -------
    jnp.ndarray, shape (b, d, d)
        Spatial Hessian matrices.
    """
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
    M_star_star_xdot_star = jnp.einsum(
        "bij,bj->bi",
        C,
        xdot_star,
        precision=jax.lax.Precision.HIGHEST,
    )
    M_upper_right = ((pdot_star - M_star_star_xdot_star - B) / xdot_1).reshape(
        b, 1, d - 1
    )

    pdotxdot = jnp.einsum(
        "bi,bi->b",
        pdot_1,
        xdot_1,
        precision=jax.lax.Precision.HIGHEST,
    )
    pdot_star_dot_xdot_star = jnp.einsum(
        "bi,bi->b",
        pdot_star,
        xdot_star,
        precision=jax.lax.Precision.HIGHEST,
    )
    xdot_star_M_xdot_star = jnp.einsum(
        "bi,bij,bj->b",
        xdot_star,
        C,
        xdot_star,
        precision=jax.lax.Precision.HIGHEST,
    )
    Bx = jnp.einsum(
        "bi,bi->b",
        B,
        xdot_star,
        precision=jax.lax.Precision.HIGHEST,
    )

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

    Parameters
    ----------
    arr : jnp.ndarray, shape (N, d)
        Array to check.
    max_constant_axes : int, optional
        Maximum number of constant axes to return. For TR this is typically 1.

    Returns
    -------
    constant_mask : jnp.ndarray, shape (d,)
        Boolean mask of constant columns.
    constant_values : jnp.ndarray
        Values of constant columns, padded with zeros.
    constant_axes : jnp.ndarray
        Indices of constant columns, padded with ``-1``.
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

    Parameters
    ----------
    significant_coeffs : jnp.ndarray
        Flattened significant coefficient indices in the boundary-data WPT.
    domain_data : Domain
        Domain describing the boundary data coordinates.
    wpt_data : MSWPT
        Boundary-data wave-packet transform.
    sources : Sensor
        Source/sensor geometry on the acquisition boundary.

    Returns
    -------
    pts : jnp.ndarray, shape (B, d_spatial)
        Momentum at the final time.
    Mts : jnp.ndarray, shape (B, d_spatial, d_spatial)
        Hessian matrix at the final time.
    xts : jnp.ndarray, shape (B, d_spatial)
        Position at the final time.
    ωs : jnp.ndarray, shape (B,)
        Beam frequency scale at the final time.
    ats : jnp.ndarray, shape (B, 1)
        Beam amplitude scale at the final time.
    signum : jnp.ndarray, shape (B, 1)
        Gaussian beam mode sign.
    ts : jnp.ndarray, shape (B, 2)
        Per-beam time interval.
    """
    _, const_vals, const_axes = find_constant_columns(
        sources.positions, max_constant_axes=1
    )
    valid = const_axes >= 0
    normal_axis = jnp.where(valid, const_axes, 0)[0]
    normal_value = jnp.where(valid, const_vals, 0.0)[0]

    d_spatial = sources.domain.ndim

    decomp = wpt_data.dyadic_decomp
    red = wpt_data.redundancy

    box_lengths = jnp.array(decomp.box_lengths)
    box_aspect = jnp.array(decomp.box_aspect_ratio)
    N_data = jnp.array(domain_data.N)
    L_phys = jnp.array(domain_data.grid_size)

    shapes = utils.compute_coeff_shapes(decomp, red, jnp.arange(decomp.num_levels))
    cumsum_boxes = jnp.r_[0, jnp.cumsum(decomp.num_boxes_ndim)]
    nn_level, nn_idx = utils.find_tensor_and_multiindex(significant_coeffs, shapes)
    box_idx = nn_idx[0, :] + cumsum_boxes[nn_level]

    # Physical frequencies are ordered as (tau, tangential wave numbers).
    centres_hat = decomp.centres_ndim[box_idx, :] / L_phys
    # Normalize direction only; the beam frequency remains |tau|.
    norm_xi = jnp.linalg.norm(centres_hat, axis=-1, keepdims=True)
    centres_normed = centres_hat / norm_xi

    xi_tau = centres_normed[:, :1]
    xi_tan_hat = centres_normed[:, 1:]

    tau = centres_hat[:, 0:1]
    k_tan = centres_hat[:, 1:]

    bl = rearrange(box_lengths[nn_level], "j -> j 1") / L_phys * box_aspect
    # Match the support convention in compute_frames and compute_coeff_shapes.
    Lls = bl * red
    sigmas = bl / 2.0

    sign_tau = jnp.sign(centres_hat[:, 0])
    signum = rearrange(-sign_tau, "b -> b 1")

    omega_cyc = jnp.maximum(jnp.abs(tau), 1e-6)
    ωs = rearrange(omega_cyc, "b 1 -> b")

    ts = jnp.zeros((nn_idx.shape[1], 2))
    ts = ts.at[:, 0].set(nn_idx[1, :] / Lls[:, 0])

    if d_spatial == 1:
        xstar_idx = nn_idx[1:, :]
    else:
        xstar_idx = nn_idx[2:, :]
    xstar = jnp.stack(xstar_idx, axis=-1) / Lls[:, 1:]

    B_ = xstar.shape[0]
    xts = jnp.zeros((B_, d_spatial))
    p_unit_spatial = jnp.zeros((B_, d_spatial))

    axis_ids = jnp.arange(d_spatial)
    mask_tan = axis_ids != normal_axis
    tan_index = jnp.where(
        mask_tan, jnp.cumsum(mask_tan.astype(jnp.int32)) - 1, 0
    ).astype(jnp.int32)

    if d_spatial == 1:
        xts = xts.at[:, 0].set(normal_value)
    else:
        xts = jnp.where(mask_tan, xstar[:, tan_index], normal_value)

    # Source geometry, not boundary-data coordinates, owns the sound speed.
    physical_c = sources.domain.c_fn
    cxts = physical_c(xts).reshape(-1, 1)

    # For d > 1, dispersion gives p̂_tan = c k_tan / |tau| and
    # p̂_n = ±sqrt(1 - |p̂_tan|^2).
    if d_spatial == 1:
        p_tan_unit = xi_tan_hat
        radicand = jnp.maximum(
            (xi_tau / cxts) ** 2 - jnp.sum(xi_tan_hat**2, axis=1, keepdims=True),
            0.0,
        )
    else:
        tau_abs = jnp.maximum(jnp.abs(tau), 1e-6)
        p_tan_unit = cxts * k_tan / tau_abs
        tan_norm_sq = jnp.sum(p_tan_unit**2, axis=1, keepdims=True)
        radicand = jnp.maximum(1.0 - tan_norm_sq, 0.0)

    p_n_unit = jnp.sqrt(radicand)

    # Orient the normal component inward using the spatial sensor domain.
    spatial_dx = jnp.array(sources.domain.dx)
    spatial_size = jnp.array(sources.domain.grid_size)
    grid_max = spatial_size[normal_axis] - spatial_dx[normal_axis]
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
    p_n_unit = inward_sign * p_n_unit

    if d_spatial == 1:
        p_unit_spatial = p_unit_spatial.at[:, 0].set(p_n_unit[:, 0])
    else:
        p_unit_spatial = jnp.where(mask_tan, p_tan_unit[:, tan_index], p_n_unit)

    pts = (2.0 * jnp.pi) * p_unit_spatial

    if d_spatial != 1:
        pts = pts / cxts

    alpha = 2j * (jnp.pi * sigmas) ** 2 / omega_cyc

    M_init = jnp.einsum(
        "bi,ij->bij",
        alpha,
        jnp.eye(d_spatial),
        precision=jax.lax.Precision.HIGHEST,
    )

    Mts = compute_mT_linear_system(xts, pts, None, M_init, signum, physical_c)

    ats = jnp.prod(
        jnp.sqrt(
            (jnp.pi * rearrange(L_phys, "d -> 1 d"))
            / (Lls * rearrange(N_data, "d -> 1 d"))
        )
        * sigmas,
        axis=1,
        keepdims=True,
    )
    local_k = jnp.stack(nn_idx[1:, :], axis=-1)
    ats = ats * compute_frame_phase(decomp, box_idx, local_k, red)[:, None]

    # Replace zero-contribution grazing geometry with safe ODE inputs.
    is_grazing_final = jnp.abs(jnp.take(pts, normal_axis, axis=1)) == 0.0
    is_grazing_final = rearrange(is_grazing_final, "b -> b 1")

    pts = jnp.where(is_grazing_final, jnp.ones_like(pts), pts)
    ats = jnp.where(is_grazing_final, jnp.zeros_like(ats), ats)

    identity_mats = 1j * jnp.eye(d_spatial)[None, :, :].repeat(B_, axis=0)
    Mts = jnp.where(is_grazing_final[:, :, None], identity_mats, Mts)

    return pts, Mts, xts, ωs, ats, signum, ts


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

    Parameters
    ----------
    x0 : jnp.ndarray
        Initial positions.
    p0 : jnp.ndarray
        Initial momentum vectors.
    M0 : jnp.ndarray
        Initial Hessian matrices.
    a0 : jnp.ndarray
        Initial amplitudes.
    ω : jnp.ndarray
        Angular frequencies.
    mode : jnp.ndarray
        Beam modes.
    c : Callable
        Sound-speed function.
    lam : float
        Absorption parameter.
    ts : jnp.ndarray
        Time points, possibly per beam.
    sensors : jnp.ndarray
        Sensor positions.
    domain_size : jnp.ndarray
        Domain size.
    periodic : jnp.ndarray
        Boundary periodicity flags.
    ode_solver : SolverFn
        ODE solver.
    sum_beams : bool, default=False
        Whether to sum over beams.
    solver_config : SolverConfig, optional
        Numerical ODE configuration.

    Returns
    -------
    jnp.ndarray
        Computed TR beams, summed if ``sum_beams=True``.
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
    pallas_config: PallasConfig | None = None,
):
    """
    Generic TR beam aggregation supporting scan, vmap, Pallas, or direct computation.

    Similar structure to _aggregate_beams but for time reversal.
    Takes the final time point ($t=0$ for TR) from each batch.

    Parameters
    ----------
    params : Tuple[jnp.ndarray, ...]
        Beam parameters ``(p0, M0, x0, omega, a0, mode, ts)``.
    aggregate_method : {"scan", "pallas", "terminal_xla", "all"}
        Aggregation strategy.
    init_shape : Tuple[int, ...]
        Shape of the output field.
    c : Callable
        Sound-speed function.
    lam : float
        Absorption parameter.
    sensors : jnp.ndarray
        Sensor positions.
    domain_size : jnp.ndarray
        Domain size.
    periodic : jnp.ndarray
        Boundary periodicity flags.
    ode_solver : SolverFn
        ODE solver.
    solver_config : SolverConfig, optional
        Numerical ODE configuration.

    Returns
    -------
    jnp.ndarray
        Aggregated TR result at the final time point.
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

    if aggregate_method in {"scan", "pallas", "terminal_xla"}:
        effective_pallas_config = pallas_config or PallasConfig()
        fused_pallas_carry = (
            aggregate_method == "pallas" and jax.default_backend() != "tpu"
        )
        num_flat_sensors = math.prod(sensors.shape[:-1])
        padded_sensors = (
            math.ceil(num_flat_sensors / effective_pallas_config.sensor_block_size)
            * effective_pallas_config.sensor_block_size
        )
        init = jnp.zeros(
            (padded_sensors,) if fused_pallas_carry else init_shape,
            dtype=x0_batches.dtype,
        )

        def scan_fn(carry, inp):
            """
            Accumulate one time-reversal beam batch.

            Parameters
            ----------
            carry : jnp.ndarray
                Running reconstructed field.
            inp : Tuple[jnp.ndarray, ...]
                One batch of TR beam parameters.

            Returns
            -------
            carry : jnp.ndarray
                Updated reconstructed field.
            aux : None
                Empty scan output.
            """
            p0, M0, x0, ω, a0, mode, ts_batch = inp
            if aggregate_method == "pallas":
                terminal_result = core.compute_gaussian_beam_real_TR_pallas_terminal(
                    x0=x0,
                    p0=p0,
                    M0=M0,
                    a0=a0,
                    omega0=ω,
                    mode=mode,
                    c=c,
                    lam=lam,
                    ts=ts_batch,
                    sensors=sensors,
                    domain_size=domain_size,
                    periodic=periodic,
                    ode_solver=ode_solver,
                    solver_config=solver_config,
                    pallas_config=effective_pallas_config,
                    initial_field=carry if fused_pallas_carry else None,
                    return_padded=fused_pallas_carry,
                )
            elif aggregate_method == "terminal_xla":
                terminal_result = core.compute_gaussian_beam_real_TR_xla_terminal(
                    x0=x0,
                    p0=p0,
                    M0=M0,
                    a0=a0,
                    omega0=ω,
                    mode=mode,
                    c=c,
                    lam=lam,
                    ts=ts_batch,
                    sensors=sensors,
                    domain_size=domain_size,
                    periodic=periodic,
                    ode_solver=ode_solver,
                    solver_config=solver_config,
                )
            else:
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
                terminal_result = batch_result[-1, ...]
            if fused_pallas_carry:
                return terminal_result, None
            return carry + terminal_result, None

        result, _ = lax.scan(scan_fn, init, params)
        if fused_pallas_carry:
            return result[:num_flat_sensors].reshape(init_shape)
        return result

    else:
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
        return beams[-1, ...]


def compute_TR_result(
    params: Tuple[jnp.ndarray, ...],
    c: Callable,
    lam: float,
    sensors: jnp.ndarray,
    domain_size: jnp.ndarray,
    periodic: jnp.ndarray,
    ode_solver: Optional[SolverFn] = None,
    aggregate_method: str = "scan",
    solver_config: Optional[SolverConfig] = None,
    pallas_config: PallasConfig | None = None,
) -> jnp.ndarray:
    """
    Compute time-reversal solution using Gaussian beams.

    Unified interface similar to compute_forward_result.

    Parameters
    ----------
    params : Tuple[jnp.ndarray, ...]
        Beam parameters ``(p0, M0, x0, omega, a0, mode, ts)``.
    c : Callable
        Sound-speed function.
    lam : float
        Absorption parameter.
    sensors : jnp.ndarray
        Sensor positions.
    domain_size : jnp.ndarray
        Domain size.
    periodic : jnp.ndarray
        Boundary periodicity flags.
    ode_solver : SolverFn, optional
        ODE solver. If ``None``, uses :func:`solve_ODE_batch_t`.
    aggregate_method : {"scan", "pallas", "terminal_xla", "all"}, default="scan"
        Beam aggregation method.
    solver_config : SolverConfig, optional
        Numerical ODE configuration.

    Returns
    -------
    jnp.ndarray
        Time-reversed field at sensor locations at the final time point.

    Notes
    -----
        TR requires per-beam time intervals (shape (b, 2)), so the ODE solver
        must support this. Default is solve_ODE_batch_t. If passing a custom
        solver, ensure it handles per-beam time arrays correctly.
    """
    if ode_solver is None:
        ode_solver = gb_solvers.solve_ODE_batch_t

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
        pallas_config=pallas_config,
    )
