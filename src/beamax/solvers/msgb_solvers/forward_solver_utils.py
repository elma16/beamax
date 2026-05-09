import jax
from jax import vmap, lax
import jax.numpy as jnp
from einops import rearrange
from typing import Callable, Union, Tuple, Optional

from beamax import utils
from beamax.gb import core, gb_utils
from beamax.gb.gb_solvers import SolverFn, SolverConfig
from beamax.transforms import MSWPT
from beamax.geometry import Domain


def _threshold_hard(coeff, val):
    idx = jnp.where(jnp.abs(coeff) > val)[0]
    return idx, coeff[idx]


def _threshold_percentile(coeff, val, max_size=None):
    if max_size is None:
        max_size = coeff.shape[0]
    thresh = jnp.percentile(jnp.abs(coeff), val)
    mask = jnp.abs(coeff) > thresh
    idx = jnp.nonzero(mask, size=max_size, fill_value=-1)[0]
    values = jnp.where(idx >= 0, coeff[idx], 0.0)
    return idx, values


def _threshold_top_n(coeff, val):
    N = int(val)
    abs_c = jnp.abs(coeff)
    idx_unsorted = jnp.argpartition(abs_c, abs_c.size - N)[-N:]
    idx = idx_unsorted[jnp.argsort(abs_c[idx_unsorted])]
    return idx, coeff[idx]


def _threshold_hard_reassign(coeff, val):
    thr = jnp.where(jnp.abs(coeff) / jnp.max(jnp.abs(coeff)) >= val, coeff, 0)
    ratio = jnp.sqrt(jnp.sum(jnp.abs(coeff) ** 2) / jnp.sum(jnp.abs(thr) ** 2))
    reassigned = thr * ratio
    idx = jnp.where(jnp.abs(reassigned) > 0)[0]
    return idx, reassigned[idx]


def _threshold_bao_energy(coeff, val, decomp, red):
    shapes = utils.compute_coeff_shapes(decomp, red, jnp.arange(decomp.num_levels))
    cumsum = jnp.r_[0, jnp.cumsum(decomp.num_boxes_ndim)]
    nn_level, nn_idx = utils.find_tensor_and_multiindex(
        jnp.arange(4 * jnp.prod(jnp.array(decomp.N))), shapes
    )
    box_idx = nn_idx[0, :] + cumsum[nn_level]
    normxi = jnp.linalg.norm(decomp.centres_ndim[box_idx], axis=1) ** decomp.ndim
    energy = coeff * normxi
    idx = jnp.where(jnp.abs(energy) > val)[0]
    return idx, energy[idx]


def _threshold_perc_max_abs(coeff, val):
    thresh = jnp.max(jnp.abs(coeff)) * val
    mask = jnp.abs(coeff) > thresh
    size = int(coeff.shape[0])
    idx = jnp.nonzero(mask, size=size, fill_value=-1)[0]
    values = jnp.where(idx >= 0, coeff[idx], 0.0)
    return idx, values


def threshold_coefficients(
    coeffs: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
    val: float,
    strategy: str = "hard",
    wpt: MSWPT = None,
):
    """
    Apply thresholding to wavelet coefficients.

    Args:
        coeffs: Coefficients to threshold (single array or tuple of arrays)
        val: Threshold value
        strategy: Thresholding strategy
        wpt: Wavelet packet transform (required for some strategies)

    Returns:
        Thresholded coefficient indices and values
    """
    funcs = {
        "hard": lambda c: _threshold_hard(c, val),
        "top_n": lambda c: _threshold_top_n(c, val),
        "percentile": lambda c: _threshold_percentile(c, val),
        "hard_reassign": lambda c: _threshold_hard_reassign(c, val),
        "bao_energy": lambda c: _threshold_bao_energy(
            c, val, wpt.dyadic_decomp, wpt.redundancy
        ),
        "perc_max_abs": lambda c: _threshold_perc_max_abs(c, val),
    }
    if strategy not in funcs:
        raise ValueError(f"Invalid thresholding strategy: {strategy}")
    f = funcs[strategy]
    return (f(coeffs[0]), f(coeffs[1])) if isinstance(coeffs, tuple) else f(coeffs)


def compute_forward_parameters(
    significant_coeffs: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
    wpt: MSWPT,
    domain: Domain,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """
    Compute Gaussian beam parameters from wavelet coefficients.

    Args:
        significant_coeffs: Significant coefficient indices (single array or tuple)
        wpt: Wavelet packet transform
        domain: Domain object

    Returns:
        Tuple of (p0s, M0s, x0s, ωs, a0s, modes)
    """

    def compute_params(coeffs: jnp.ndarray, sign: int):
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

        # Compute normalized centres and momenta
        centres = wpt.dyadic_decomp.centres_ndim[box_idx, :] / grid_size
        norm = jnp.linalg.norm(centres, axis=-1, keepdims=True)
        p0s = 2 * jnp.pi * centres / norm

        # Compute box parameters
        bl = rearrange(box_lengths[nn_level], "j -> j 1") / grid_size * box_aspect_ratio
        Lls = bl * 2
        sigmas = bl / 2

        # Compute beam parameters
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
        x0s = jnp.stack(nn_idx[1:, :], axis=-1) / Lls
        ωs = rearrange(norm, "b 1 -> b")
        modes = sign * jnp.ones((p0s.shape[0],))

        return p0s, M0s, x0s, ωs, a0s, modes

    if isinstance(significant_coeffs, tuple):
        pos = compute_params(significant_coeffs[0], 1)
        neg = compute_params(significant_coeffs[1], -1)
        return tuple(jnp.concatenate((p, n), axis=0) for p, n in zip(pos, neg))
    return compute_params(significant_coeffs, 1)


def compute_memory_requirements(b: int, N: Tuple, Nt: int) -> str:
    """
    Estimate memory requirements for Gaussian beam computation.

    Args:
        b: Number of beams
        N: Grid dimensions
        Nt: Number of time points

    Returns:
        Memory estimate string
    """
    dims = (Nt,) + N + (b,)
    dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    return utils.memory_estimate(jnp.array(dims), dtype)


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

    Args:
        x0: Initial positions
        p0: Initial momentum vectors
        M0: Initial M matrices
        a0: Initial amplitudes
        ω: Angular frequencies
        mode: Beam modes
        c: Speed of sound function
        lam: Lambda parameter
        ts: Time points
        sensors: Sensor positions
        domain_size: Domain size
        periodic: Boundary conditions
        ode_solver: ODE solver function
        use_real: Whether to use real-valued computation
        sum_beams: Whether to sum over beams

    Returns:
        Computed beams (summed if sum_beams=True)
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
):
    """
    Generic beam aggregation supporting scan, vmap, or direct computation.

    Note: compute_gaussian_beam_real already sums over beams internally,
    while compute_gaussian_beam keeps the beam axis.
    """
    p0_batches, M0_batches, x0_batches, ω_batches, a0_batches, mode_batches = params

    if aggregate_method == "scan":
        # Initialize with correct dtype based on use_real flag
        if use_real:
            init = jnp.zeros(init_shape)
        else:
            # Complex computation - respect JAX precision setting
            complex_dtype = jnp.complex128 if jax.config.x64_enabled else jnp.complex64
            init = jnp.zeros(init_shape, dtype=complex_dtype)

        def scan_fn(carry, inp):
            p0, M0, x0, ω, a0, mode = inp
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
            # Real version already summed over beams, complex version has beam axis
            if use_real:
                # batch_result shape: (Nt, *S) - no beam axis
                return carry + batch_result, None
            else:
                # batch_result shape: (Nt, *S, b) - has beam axis
                return carry + jnp.sum(batch_result, axis=-1), None

        result, _ = lax.scan(scan_fn, init, params)
        return result

    elif aggregate_method == "vmap":
        beam_sums = vmap(
            lambda p0, M0, x0, ω, a0, mode: _compute_beams(
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
        )(p0_batches, M0_batches, x0_batches, ω_batches, a0_batches, mode_batches)
        # Real version: beam_sums shape (num_batches, Nt, *S)
        # Complex version: beam_sums shape (num_batches, Nt, *S, batch_size)
        if use_real:
            return jnp.sum(beam_sums, axis=0)  # sum over batches
        else:
            return jnp.sum(
                jnp.sum(beam_sums, axis=-1), axis=0
            )  # sum over beams, then batches

    else:  # "all"
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
        # Real version already summed, complex version has beam axis
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
) -> jnp.ndarray:
    """
    Compute forward solution to the wave equation using Gaussian beams.

    Args:
        params: Tuple of beam parameters (p0, M0, x0, ω, a0, mode)
        c: Speed of sound function
        lam: Lambda parameter
        ts: Time points
        ode_solver: ODE solver function
        sensors: Sensor positions
        domain_size: Domain size
        periodic: Boundary conditions
        use_real: Whether to use real-valued computation
        aggregate_method: 'scan', 'vmap', or 'all'

    Returns:
        Forward solution at sensor locations
    """
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
    )


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

    Args:
        p0: Initial pressure field
        dpdt: Time derivative of pressure field
        input_type: Type of input data
        domain: Domain object
        wpt: Wavelet packet transform object
        mode: 'both' returns (cpos, cneg), 'pos_only' returns cpos masked

    Returns:
        Coefficients (cpos, cneg) if mode='both', or masked cpos if mode='pos_only'
    """
    # Compute wavelet transforms
    a_coeff = wpt.forward(p0, input_type)
    b_coeff = wpt.forward(dpdt, input_type)

    # Compute geometric information
    shapes = utils.compute_coeff_shapes(
        wpt.dyadic_decomp, wpt.redundancy, jnp.arange(wpt.dyadic_decomp.num_levels)
    )
    cumsum = jnp.r_[0, jnp.cumsum(wpt.dyadic_decomp.num_boxes_ndim)]
    nn_level, nn_idx = utils.find_tensor_and_multiindex(
        jnp.arange(wpt.total_coeffs), shapes
    )
    k = nn_idx[1:, :]
    b = k.shape[1]
    x_b = (k / wpt.dyadic_decomp.box_lengths[nn_level]).T
    box_idx = nn_idx[0, :] + cumsum[nn_level]
    centres = wpt.dyadic_decomp.centres_ndim[box_idx] / domain.grid_size
    p_b = 2 * jnp.pi * centres
    mode_b = jnp.ones(b)

    # Compute group velocity
    vg = gb_utils.vmap_g(x_b, p_b, mode_b, domain.c_fn)

    # Compute positive and negative frequency coefficients
    cpos = 0.5 * (a_coeff + 1j * b_coeff / vg)

    if mode == "pos_only":
        # Return masked positive coefficients only
        return cpos * wpt._half_mask
    elif mode == "both":
        cneg = 0.5 * (a_coeff - 1j * b_coeff / vg)
        return cpos, cneg
    else:
        raise ValueError(f"Invalid mode: {mode}")
