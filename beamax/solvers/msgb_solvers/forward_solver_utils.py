import math

import jax
from jax import lax
import jax.numpy as jnp
from einops import rearrange
from typing import Callable, Literal, Optional, Tuple, Union, overload

from beamax import utils
from beamax.gb import core, gb_utils
from beamax.gb.gb_solvers import SolverFn, SolverConfig, solve_hom_diag
from beamax.gb.pallas_config import PallasConfig
from beamax.transforms import MSWPT, compute_frame_phase
from beamax.geometry import Domain


def _threshold_hard(coeff, val):
    r"""
    Select coefficients $c_j$ satisfying $|c_j|>\lambda$.

    Parameters
    ----------
    coeff : jnp.ndarray
        Coefficient vector.
    val : float
        Absolute magnitude threshold $\lambda$.

    Returns
    -------
    idx : jnp.ndarray
        Selected coefficient indices.
    values : jnp.ndarray
        Selected coefficient values.
    """
    idx = jnp.where(jnp.abs(coeff) > val)[0]
    return idx, coeff[idx]


def _threshold_top_n(coeff, val):
    """
    Select the largest ``val`` coefficients by magnitude.

    Parameters
    ----------
    coeff : jnp.ndarray
        Coefficient vector.
    val : int
        Number of coefficients to select.

    Returns
    -------
    idx : jnp.ndarray
        Selected indices.
    values : jnp.ndarray
        Selected coefficient values.
    """
    N = min(int(val), int(coeff.shape[0]))
    if N <= 0:
        raise ValueError("top_n threshold must be a positive integer.")
    abs_c = jnp.abs(coeff)
    idx_unsorted = jnp.argpartition(abs_c, abs_c.size - N)[-N:]
    idx = idx_unsorted[jnp.argsort(abs_c[idx_unsorted])]
    return idx, coeff[idx]


def threshold_coefficients(
    coeffs: jnp.ndarray,
    val: float,
    strategy: str = "hard",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply thresholding to wavelet coefficients.

    Parameters
    ----------
    coeffs : jnp.ndarray
        Coefficients to threshold.
    val : float
        Threshold value.
    strategy : str, default="hard"
        Thresholding strategy.
    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        Selected indices and values.

    Raises
    ------
    ValueError
        If ``strategy`` is unknown.
    """

    funcs = {
        "hard": lambda c: _threshold_hard(c, val),
        "top_n": lambda c: _threshold_top_n(c, val),
    }
    if strategy not in funcs:
        raise ValueError(f"Invalid thresholding strategy: {strategy}")
    return funcs[strategy](coeffs)


def _coefficient_positions(
    nn_level: jnp.ndarray,
    nn_idx: jnp.ndarray,
    wpt: MSWPT,
    domain: Domain,
) -> jnp.ndarray:
    r"""Map local MSWPT coefficient indices to physical packet centres.

    Writing $r$ for the redundancy, $b_\ell$ for the level's box length,
    and $a_s$ for its aspect ratio, the support length on axis $s$ is
    $S_{\ell,s}=r b_\ell a_s$. Index $k_s$ maps to
    $x_s=k_sL_s/S_{\ell,s}$.
    """
    local_indices = jnp.stack(nn_idx[1:, :], axis=-1)
    box_lengths = jnp.asarray(wpt.dyadic_decomp.box_lengths)
    aspect = jnp.asarray(wpt.dyadic_decomp.box_aspect_ratio)
    support_lengths = (
        rearrange(box_lengths[nn_level], "b -> b 1") * aspect * wpt.redundancy
    )
    return local_indices * jnp.asarray(domain.grid_size) / support_lengths


def compute_forward_parameters(
    significant_coeffs: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
    wpt: MSWPT,
    domain: Domain,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """
    Compute Gaussian beam parameters from wavelet coefficients.

    Parameters
    ----------
    significant_coeffs : jnp.ndarray or Tuple[jnp.ndarray, jnp.ndarray]
        Significant coefficient indices for positive, or positive/negative,
        modes.
    wpt : MSWPT
        Wave-packet transform.
    domain : Domain
        Physical domain.

    Returns
    -------
    p0s : jnp.ndarray
        Initial beam momenta.
    M0s : jnp.ndarray
        Initial beam Hessians.
    x0s : jnp.ndarray
        Initial beam positions.
    ωs : jnp.ndarray
        Beam frequencies.
    a0s : jnp.ndarray
        Initial beam amplitudes.
    modes : jnp.ndarray
        Beam branch signs.
    """

    def compute_params(coeffs: jnp.ndarray, sign: int):
        """
        Compute beam parameters for one sign branch.

        Parameters
        ----------
        coeffs : jnp.ndarray
            Significant coefficient indices for one sign branch.
        sign : int
            Mode sign multiplier.

        Returns
        -------
        p0s : jnp.ndarray
            Initial beam momenta.
        M0s : jnp.ndarray
            Initial beam Hessians.
        x0s : jnp.ndarray
            Initial beam positions.
        ωs : jnp.ndarray
            Beam frequencies.
        a0s : jnp.ndarray
            Initial beam amplitudes.
        modes : jnp.ndarray
            Beam branch signs.
        """
        grid_size = domain.grid_size
        box_lengths = jnp.array(wpt.dyadic_decomp.box_lengths)
        box_aspect_ratio = jnp.array(wpt.dyadic_decomp.box_aspect_ratio)
        N = jnp.array(domain.N)

        shapes = utils.compute_coeff_shapes(
            wpt.dyadic_decomp, wpt.redundancy, jnp.arange(wpt.dyadic_decomp.num_levels)
        )
        cumsum = jnp.r_[0, jnp.cumsum(wpt.dyadic_decomp.num_boxes_ndim)]
        nn_level, nn_idx = utils.find_tensor_and_multiindex(coeffs, shapes)
        box_idx = nn_idx[0, :] + cumsum[nn_level]

        centres = wpt.dyadic_decomp.centres_ndim[box_idx, :] / grid_size
        norm = jnp.linalg.norm(centres, axis=-1, keepdims=True)
        p0s = 2 * jnp.pi * centres / norm

        bl = rearrange(box_lengths[nn_level], "j -> j 1") / grid_size * box_aspect_ratio
        Lls = bl * wpt.redundancy
        sigmas = bl / 2

        αs = 2j * (jnp.pi * sigmas) ** 2 / norm
        M0s = gb_utils.prepare_M0(αs, None)
        a0s = jnp.prod(
            jnp.sqrt(
                (jnp.pi * rearrange(grid_size, "d -> 1 d"))
                / (Lls * rearrange(N, "d -> 1 d"))
            )
            * sigmas,
            axis=1,
            keepdims=True,
        )
        a0s = rearrange(a0s, "b 1 -> b")
        local_k = jnp.stack(nn_idx[1:, :], axis=-1)
        a0s = a0s * compute_frame_phase(
            wpt.dyadic_decomp, box_idx, local_k, wpt.redundancy
        )
        x0s = _coefficient_positions(nn_level, nn_idx, wpt, domain)
        ωs = rearrange(norm, "b 1 -> b")
        modes = sign * jnp.ones((p0s.shape[0],))

        return p0s, M0s, x0s, ωs, a0s, modes

    if isinstance(significant_coeffs, tuple):
        pos = compute_params(significant_coeffs[0], 1)
        neg = compute_params(significant_coeffs[1], -1)
        return (
            jnp.concatenate((pos[0], neg[0]), axis=0),
            jnp.concatenate((pos[1], neg[1]), axis=0),
            jnp.concatenate((pos[2], neg[2]), axis=0),
            jnp.concatenate((pos[3], neg[3]), axis=0),
            jnp.concatenate((pos[4], neg[4]), axis=0),
            jnp.concatenate((pos[5], neg[5]), axis=0),
        )
    return compute_params(significant_coeffs, 1)


def _compute_beams(
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
    use_real: bool = True,
    sum_beams: bool = False,
    solver_config: Optional[SolverConfig] = None,
):
    """
    Unified beam computation function.

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
        Time points.
    sensors : jnp.ndarray
        Sensor positions.
    domain_size : jnp.ndarray
        Domain size.
    periodic : jnp.ndarray
        Boundary periodicity flags.
    ode_solver : SolverFn
        ODE solver.
    use_real : bool, default=True
        Whether to use real-valued beam computation.
    sum_beams : bool, default=False
        Whether to sum over the beam axis.
    solver_config : SolverConfig, optional
        Numerical ODE configuration.

    Returns
    -------
    jnp.ndarray
        Computed beams, summed if ``sum_beams=True``.
    """
    compute_fn = (
        core.compute_gaussian_beam_real if use_real else core.compute_gaussian_beam
    )

    beams = compute_fn(
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


def _aggregate_beams(
    params: Tuple[jnp.ndarray, ...],
    aggregate_method: str,
    init_shape: Tuple,
    use_real: bool,
    c: Callable,
    lam: float,
    ts: jnp.ndarray,
    sensors: jnp.ndarray,
    domain_size: jnp.ndarray,
    periodic: jnp.ndarray,
    ode_solver: SolverFn,
    solver_config: Optional[SolverConfig] = None,
    pallas_config: PallasConfig | None = None,
):
    """
    Generic beam aggregation supporting scan, vmap, Pallas, or direct computation.

    Parameters
    ----------
    params : Tuple[jnp.ndarray, ...]
        Beam parameter tuple ``(p0, M0, x0, omega, a0, mode)``.
    aggregate_method : {"scan", "pallas", "pallas_fused_hom_diag_3d", "all"}
        Aggregation strategy.
    init_shape : Tuple[int, ...]
        Shape of the running accumulated field.
    use_real : bool
        Whether beam computation is real-valued.
    c : Callable
        Sound-speed function.
    lam : float
        Absorption parameter.
    ts : jnp.ndarray
        Time grid.
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
        Aggregated field.

    Notes
    -----
    ``compute_gaussian_beam_real`` already sums over beams internally, while
    ``compute_gaussian_beam`` keeps the beam axis.
    """
    p0_batches, M0_batches, x0_batches, ω_batches, a0_batches, mode_batches = params

    pallas_methods = {"pallas", "pallas_fused_hom_diag_3d"}
    if aggregate_method in pallas_methods and not use_real:
        raise ValueError("Pallas aggregation is only available for real-valued fields.")

    if aggregate_method in {"scan", *pallas_methods}:
        effective_pallas_config = pallas_config or PallasConfig()
        fused_pallas_carry = (
            aggregate_method in pallas_methods and jax.default_backend() != "tpu"
        )
        num_flat_sensors = math.prod(sensors.shape[:-1])
        padded_sensors = (
            math.ceil(num_flat_sensors / effective_pallas_config.sensor_block_size)
            * effective_pallas_config.sensor_block_size
        )
        padded_times = (
            math.ceil(ts.shape[0] / effective_pallas_config.gpu_time_block_size)
            * effective_pallas_config.gpu_time_block_size
            if aggregate_method == "pallas_fused_hom_diag_3d"
            else ts.shape[0]
        )
        if use_real:
            if fused_pallas_carry:
                init = jnp.zeros((padded_times, padded_sensors), dtype=x0_batches.dtype)
            else:
                init = jnp.zeros(init_shape, dtype=x0_batches.dtype)
        else:
            x64_enabled = bool(getattr(jax.config, "x64_enabled", False))
            complex_dtype = jnp.complex128 if x64_enabled else jnp.complex64
            init = jnp.zeros(init_shape, dtype=complex_dtype)

        def scan_fn(carry, inp):
            """Accumulate one beam batch."""
            p0, M0, x0, ω, a0, mode = inp
            if aggregate_method == "pallas":
                batch_result = core.compute_gaussian_beam_real_pallas(
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
                    pallas_config=effective_pallas_config,
                    initial_field=carry if fused_pallas_carry else None,
                    return_padded=fused_pallas_carry,
                )
            elif aggregate_method == "pallas_fused_hom_diag_3d":
                from beamax.gb.pallas_kernels import (
                    sum_gaussian_beam_real_hom_diag_3d_pallas,
                )

                c0 = jnp.asarray(c(jnp.zeros((3,), dtype=x0.dtype)), dtype=x0.dtype)
                batch_result = sum_gaussian_beam_real_hom_diag_3d_pallas(
                    x0=x0,
                    p0=p0,
                    M0=M0,
                    a0=a0,
                    omega0=ω,
                    mode=mode,
                    c0=c0,
                    ts=ts,
                    sensors=sensors,
                    domain_size=domain_size,
                    periodic=periodic,
                    config=effective_pallas_config,
                    initial_field=carry if fused_pallas_carry else None,
                    return_padded=fused_pallas_carry,
                )
            else:
                batch_result = _compute_beams(
                    x0,
                    p0,
                    M0,
                    a0,
                    ω,
                    mode,
                    c,
                    lam,
                    ts,
                    sensors,
                    domain_size,
                    periodic,
                    ode_solver,
                    use_real,
                    sum_beams=False,
                    solver_config=solver_config,
                )
            if use_real:
                if fused_pallas_carry:
                    return batch_result, None
                return carry + batch_result, None
            else:
                return carry + jnp.sum(batch_result, axis=-1), None

        result, _ = lax.scan(scan_fn, init, params)
        if fused_pallas_carry:
            return result[: ts.shape[0], :num_flat_sensors].reshape(init_shape)
        return result

    else:
        beams = _compute_beams(
            x0_batches,
            p0_batches,
            M0_batches,
            a0_batches,
            ω_batches,
            mode_batches,
            c,
            lam,
            ts,
            sensors,
            domain_size,
            periodic,
            ode_solver,
            use_real,
            sum_beams=False,
            solver_config=solver_config,
        )
        return beams if use_real else jnp.sum(beams, axis=-1)


def compute_forward_result(
    params: Tuple[jnp.ndarray, ...],
    c: Callable,
    lam: float,
    ts: jnp.ndarray,
    ode_solver: SolverFn,
    sensors: jnp.ndarray,
    domain_size: jnp.ndarray,
    periodic: jnp.ndarray,
    use_real: bool = True,
    aggregate_method: str = "scan",
    solver_config: Optional[SolverConfig] = None,
    pallas_config: PallasConfig | None = None,
) -> jnp.ndarray:
    """
    Compute forward solution to the wave equation using Gaussian beams.

    Parameters
    ----------
    params : Tuple[jnp.ndarray, ...]
        Beam parameters ``(p0, M0, x0, omega, a0, mode)``.
    c : Callable
        Sound-speed function.
    lam : float
        Absorption parameter.
    ts : jnp.ndarray
        Time points.
    ode_solver : SolverFn
        ODE solver.
    sensors : jnp.ndarray
        Sensor positions.
    domain_size : jnp.ndarray
        Domain size.
    periodic : jnp.ndarray
        Boundary periodicity flags.
    use_real : bool, default=True
        Whether to use real-valued beam computation.
    aggregate_method : {"scan", "pallas", "pallas_fused_hom_diag_3d", "all"}, default="scan"
        Beam aggregation method.
    solver_config : SolverConfig, optional
        Numerical ODE configuration.

    Returns
    -------
    jnp.ndarray
        Forward solution at sensor locations.
    """
    if (
        aggregate_method == "pallas_fused_hom_diag_3d"
        and ode_solver is not solve_hom_diag
    ):
        raise ValueError("pallas_fused_hom_diag_3d requires ode_solver=solve_hom_diag.")
    init_shape = ts.shape + sensors.shape[:-1]

    return _aggregate_beams(
        params=params,
        aggregate_method=aggregate_method,
        init_shape=init_shape,
        use_real=use_real,
        c=c,
        lam=lam,
        ts=ts,
        sensors=sensors,
        domain_size=domain_size,
        periodic=periodic,
        ode_solver=ode_solver,
        solver_config=solver_config,
        pallas_config=pallas_config,
    )


@overload
def compute_coefficients(
    p0: jnp.ndarray,
    dpdt: jnp.ndarray,
    input_type: str,
    domain: Domain,
    wpt: MSWPT,
    mode: Literal["pos_only"],
) -> jnp.ndarray: ...


@overload
def compute_coefficients(
    p0: jnp.ndarray,
    dpdt: jnp.ndarray,
    input_type: str,
    domain: Domain,
    wpt: MSWPT,
    mode: Literal["both"] = "both",
) -> Tuple[jnp.ndarray, jnp.ndarray]: ...


def compute_coefficients(
    p0: jnp.ndarray,
    dpdt: jnp.ndarray,
    input_type: str,
    domain: Domain,
    wpt: MSWPT,
    mode: str = "both",
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Compute wavelet packet transform coefficients.

    Parameters
    ----------
    p0 : jnp.ndarray
        Initial pressure field.
    dpdt : jnp.ndarray
        Initial pressure time derivative.
    input_type : {"spatial", "fourier"}
        Domain of ``p0`` and ``dpdt``.
    domain : Domain
        Physical domain.
    wpt : MSWPT
        Wave-packet transform.
    mode : {"both", "pos_only"}, default="both"
        ``"both"`` returns positive and negative frequency coefficients.
        ``"pos_only"`` returns masked positive coefficients.

    Returns
    -------
    jnp.ndarray or Tuple[jnp.ndarray, jnp.ndarray]
        Coefficients ``(cpos, cneg)`` if ``mode="both"``, otherwise masked
        ``cpos``.

    Raises
    ------
    ValueError
        If ``mode`` is invalid.
    """
    a_coeff = wpt.forward(p0, input_type)
    b_coeff = wpt.forward(dpdt, input_type)

    shapes = utils.compute_coeff_shapes(
        wpt.dyadic_decomp, wpt.redundancy, jnp.arange(wpt.dyadic_decomp.num_levels)
    )
    cumsum = jnp.r_[0, jnp.cumsum(wpt.dyadic_decomp.num_boxes_ndim)]
    nn_level, nn_idx = utils.find_tensor_and_multiindex(
        jnp.arange(wpt.total_coeffs), shapes
    )
    b = nn_idx.shape[1]
    x_b = _coefficient_positions(nn_level, nn_idx, wpt, domain)
    box_idx = nn_idx[0, :] + cumsum[nn_level]
    centres = wpt.dyadic_decomp.centres_ndim[box_idx] / domain.grid_size
    p_b = 2 * jnp.pi * centres
    mode_b = jnp.ones(b)

    vg = gb_utils.vmap_g(x_b, p_b, mode_b, domain.c_fn)

    cpos = 0.5 * (a_coeff + 1j * b_coeff / vg)

    if mode == "pos_only":
        return cpos * wpt.half_mask
    elif mode == "both":
        cneg = 0.5 * (a_coeff - 1j * b_coeff / vg)
        return cpos, cneg
    else:
        raise ValueError(f"Invalid mode: {mode}")
