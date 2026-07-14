import jax
from jax import vmap, lax
import jax.numpy as jnp
from einops import rearrange
from typing import Callable, Union, Tuple, Optional

from beamax import utils
from beamax.gb import core, gb_utils
from beamax.gb.gb_solvers import SolverFn, SolverConfig
from beamax.transforms import MSWPT, compute_frame_phase
from beamax.geometry import Domain


def _threshold_hard(coeff, val):
    """
    Select coefficients whose magnitude is strictly greater than a threshold.

    Parameters
    ----------
    coeff : jnp.ndarray
        Coefficient vector.
    val : float
        Absolute magnitude threshold.

    Returns
    -------
    idx : jnp.ndarray
        Selected coefficient indices.
    values : jnp.ndarray
        Selected coefficient values.
    """
    idx = jnp.where(jnp.abs(coeff) > val)[0]
    return idx, coeff[idx]


def _threshold_percentile(coeff, val, max_size=None):
    """
    Select coefficients above a magnitude percentile.

    Parameters
    ----------
    coeff : jnp.ndarray
        Coefficient vector.
    val : float
        Percentile threshold in ``[0, 100]``.
    max_size : int, optional
        Fixed output size for ``jnp.nonzero``. Defaults to ``len(coeff)``.

    Returns
    -------
    idx : jnp.ndarray
        Selected indices, padded with ``-1`` if needed.
    values : jnp.ndarray
        Selected values, padded with zeros if needed.
    """
    if max_size is None:
        max_size = coeff.shape[0]
    thresh = jnp.percentile(jnp.abs(coeff), val)
    mask = jnp.abs(coeff) > thresh
    idx = jnp.nonzero(mask, size=max_size, fill_value=-1)[0]
    values = jnp.where(idx >= 0, coeff[idx], 0.0)
    return idx, values


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


def _threshold_hard_reassign(coeff, val):
    """
    Hard-threshold coefficients and rescale retained energy.

    Parameters
    ----------
    coeff : jnp.ndarray
        Coefficient vector.
    val : float
        Relative threshold as a fraction of the maximum magnitude.

    Returns
    -------
    idx : jnp.ndarray
        Selected indices.
    values : jnp.ndarray
        Reassigned coefficient values.
    """
    max_abs = jnp.max(jnp.abs(coeff))
    thr = jnp.where((max_abs > 0) & (jnp.abs(coeff) >= val * max_abs), coeff, 0)
    retained_energy = jnp.sum(jnp.abs(thr) ** 2)
    total_energy = jnp.sum(jnp.abs(coeff) ** 2)
    ratio = jnp.sqrt(total_energy / jnp.where(retained_energy > 0, retained_energy, 1))
    ratio = jnp.where(retained_energy > 0, ratio, 0)
    reassigned = thr * ratio
    idx = jnp.where(jnp.abs(reassigned) > 0)[0]
    return idx, reassigned[idx]


def _threshold_bao_energy(coeff, val, decomp, red):
    """
    Threshold coefficients by Bao-style frequency-weighted energy.

    Parameters
    ----------
    coeff : jnp.ndarray
        Coefficient vector.
    val : float
        Weighted-energy threshold.
    decomp : DyadicDecomposition
        Dyadic decomposition used to map coefficients to boxes.
    red : int
        Transform redundancy.

    Returns
    -------
    idx : jnp.ndarray
        Selected indices.
    values : jnp.ndarray
        Weighted coefficient values.
    """
    shapes = utils.compute_coeff_shapes(decomp, red, jnp.arange(decomp.num_levels))
    cumsum = jnp.r_[0, jnp.cumsum(decomp.num_boxes_ndim)]
    nn_level, nn_idx = utils.find_tensor_and_multiindex(
        jnp.arange(coeff.shape[0]), shapes
    )
    box_idx = nn_idx[0, :] + cumsum[nn_level]
    normxi = jnp.linalg.norm(decomp.centres_ndim[box_idx], axis=1) ** decomp.ndim
    weighted_coeff = coeff * normxi
    idx = jnp.where(jnp.abs(weighted_coeff) > val)[0]
    return idx, coeff[idx]


def _threshold_perc_max_abs(coeff, val):
    """
    Select coefficients above a fraction of the maximum magnitude.

    Parameters
    ----------
    coeff : jnp.ndarray
        Coefficient vector.
    val : float
        Fraction of ``max(abs(coeff))`` used as the threshold.

    Returns
    -------
    idx : jnp.ndarray
        Selected indices, padded with ``-1`` if needed.
    values : jnp.ndarray
        Selected values, padded with zeros if needed.
    """
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
    wpt: Optional[MSWPT] = None,
):
    """
    Apply thresholding to wavelet coefficients.

    Parameters
    ----------
    coeffs : jnp.ndarray or Tuple[jnp.ndarray, jnp.ndarray]
        Coefficients to threshold.
    val : float
        Threshold value.
    strategy : str, default="hard"
        Thresholding strategy.
    wpt : MSWPT, optional
        Wave-packet transform required by strategies that depend on the
        dyadic layout.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray] or Tuple[Tuple[jnp.ndarray, jnp.ndarray], ...]
        Selected indices and values. If ``coeffs`` is a tuple, returns one
        ``(idx, values)`` pair for each coefficient vector.

    Raises
    ------
    ValueError
        If ``strategy`` is unknown.
    """

    def threshold_bao(c):
        if wpt is None:
            raise ValueError("bao_energy thresholding requires wpt.")
        return _threshold_bao_energy(c, val, wpt.dyadic_decomp, wpt.redundancy)

    funcs = {
        "hard": lambda c: _threshold_hard(c, val),
        "top_n": lambda c: _threshold_top_n(c, val),
        "percentile": lambda c: _threshold_percentile(c, val),
        "hard_reassign": lambda c: _threshold_hard_reassign(c, val),
        "bao_energy": threshold_bao,
        "perc_max_abs": lambda c: _threshold_perc_max_abs(c, val),
    }
    if strategy not in funcs:
        raise ValueError(f"Invalid thresholding strategy: {strategy}")
    f = funcs[strategy]
    return (f(coeffs[0]), f(coeffs[1])) if isinstance(coeffs, tuple) else f(coeffs)


def _coefficient_positions(
    nn_level: jnp.ndarray,
    nn_idx: jnp.ndarray,
    wpt: MSWPT,
    domain: Domain,
) -> jnp.ndarray:
    """Map local MSWPT coefficient indices to physical packet centres.

    A level's coefficient block is the inverse FFT of a support whose length
    on axis ``s`` is ``redundancy * box_length * box_aspect_ratio[s]``.
    Consequently index ``k_s`` represents the physical position
    ``k_s * domain_size_s / support_length_s``.
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

        # Compute normalized centres and momenta
        centres = wpt.dyadic_decomp.centres_ndim[box_idx, :] / grid_size
        norm = jnp.linalg.norm(centres, axis=-1, keepdims=True)
        p0s = 2 * jnp.pi * centres / norm

        # Compute box parameters
        bl = rearrange(box_lengths[nn_level], "j -> j 1") / grid_size * box_aspect_ratio
        Lls = bl * wpt.redundancy
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


def compute_memory_requirements(b: int, N: Tuple, Nt: int) -> str:
    """
    Estimate memory requirements for Gaussian beam computation.

    Parameters
    ----------
    b : int
        Number of beams.
    N : Tuple[int, ...]
        Grid dimensions.
    Nt : int
        Number of time points.

    Returns
    -------
    str
        Human-readable memory estimate.
    """
    dims = (Nt,) + N + (b,)
    x64_enabled = bool(getattr(jax.config, "x64_enabled", False))
    dtype = jnp.float64 if x64_enabled else jnp.float32
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
):
    """
    Generic beam aggregation supporting scan, vmap, or direct computation.

    Parameters
    ----------
    params : Tuple[jnp.ndarray, ...]
        Beam parameter tuple ``(p0, M0, x0, omega, a0, mode)``.
    aggregate_method : {"scan", "vmap", "all"}
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

    if aggregate_method == "scan":
        # Initialize with correct dtype based on use_real flag
        if use_real:
            init = jnp.zeros(init_shape)
        else:
            # Complex computation - respect JAX precision setting
            x64_enabled = bool(getattr(jax.config, "x64_enabled", False))
            complex_dtype = jnp.complex128 if x64_enabled else jnp.complex64
            init = jnp.zeros(init_shape, dtype=complex_dtype)

        def scan_fn(carry, inp):
            """
            Accumulate one beam batch into the scanned field.

            Parameters
            ----------
            carry : jnp.ndarray
                Running accumulated field.
            inp : Tuple[jnp.ndarray, ...]
                One batch of beam parameters.

            Returns
            -------
            carry : jnp.ndarray
                Updated accumulated field.
            aux : None
                Empty scan output.
            """
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
    aggregate_method : {"scan", "vmap", "all"}, default="scan"
        Beam aggregation method.
    solver_config : SolverConfig, optional
        Numerical ODE configuration.

    Returns
    -------
    jnp.ndarray
        Forward solution at sensor locations.
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
    b = nn_idx.shape[1]
    x_b = _coefficient_positions(nn_level, nn_idx, wpt, domain)
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
