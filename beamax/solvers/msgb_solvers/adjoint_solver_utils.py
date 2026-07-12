from typing import Tuple

import jax.numpy as jnp

from beamax import utils
from beamax.geometry import Domain, Sensor
from beamax.transforms import MSWPT

# Reuse the *tested* TR parameter builder
from .tr_solver_utils import compute_TR_parameters

Array = jnp.ndarray

__all__ = ["compute_adj_parameters", "principal_b_inverse"]


def principal_b_inverse(
    tau: Array,
    k_tan: Array,
    c: Array,
    relative_guard: float = 1e-6,
) -> Array:
    """Evaluate the retarded/outgoing half-space principal symbol ``B^{-1}``.

    ``tau`` and ``k_tan`` are cyclic frequencies.  They are converted to
    angular variables before evaluating

    ``-i sign(Omega) / (2 c sqrt(Omega**2 - c**2 |Q_tan|**2))``.

    The temporal sign is required for conjugate symmetry, hence for a real
    boundary operator. Evanescent and near-grazing entries are set to zero.
    A terminal-value (advanced) backpropagator has the negative of this
    multiplier; :func:`compute_adj_parameters` applies that sign.

    Parameters
    ----------
    tau : array, shape (B,)
        Cyclic temporal frequencies.
    k_tan : array, shape (B, d_tan)
        Cyclic tangential spatial frequencies.
    c : array, shape (B,)
        Sound speed at each boundary packet centre.
    relative_guard : float, default=1e-6
        Minimum ratio ``gamma_ang / abs(omega_ang)``. This is a local,
        dimensionless near-grazing guard.

    Returns
    -------
    array, shape (B,)
        Complex principal multiplier at each packet centre.
    """
    omega_ang = 2.0 * jnp.pi * tau
    q_tan_ang = 2.0 * jnp.pi * k_tan

    if k_tan.shape[1] > 0:
        q_tan_sq = jnp.sum(q_tan_ang**2, axis=1)
    else:
        q_tan_sq = jnp.zeros_like(omega_ang)

    rad = omega_ang**2 - (c**2) * q_tan_sq
    gamma_ang = jnp.sqrt(jnp.maximum(rad, 0.0))
    valid = (
        (rad > 0.0)
        & (jnp.abs(omega_ang) > 0.0)
        & (gamma_ang > relative_guard * jnp.abs(omega_ang))
    )
    safe_gamma_ang = jnp.where(valid, gamma_ang, 1.0)
    return jnp.where(
        valid,
        -1j * jnp.sign(omega_ang) / (2.0 * c * safe_gamma_ang),
        0.0,
    )


def compute_adj_parameters(
    coeff_indices: Array,
    domain_data: Domain,
    wpt_data: MSWPT,
    sources: Sensor,
    relative_guard: float = 5e-2,
) -> Tuple[Array, ...]:
    """
    Compute Gaussian beam parameters for the adjoint solve.

    Design:
        1. Reuse the time-reversal packet→beam mapping in
           `compute_TR_parameters`. This gives us beam geometry
           (x_T, p_T, M_T, ω, sign, time-intervals) that is already
           tested in the TR pipeline.

        2. On top of that, apply the approximate advanced B^{-1} symbol in
           (ω, k_tan) to the beam amplitudes only. This implements the
           microlocal equivalence

               L q = F(x^*, t) δ(x_1)  ↔  L q = 0,  q|_{Γ} = h,
               with h ≈ B^{-1} F,

           where B has outgoing principal symbol

               b_0(x^*; ω, ξ^*)
                 = 2 i c(0, x^*) sign(ω)
                   sqrt(ω^2 - c(0, x^*)^2 |ξ^*|^2).

           We keep the TR geometry and curvature from `compute_TR_parameters`.
           Because that routine evolves a terminal-value problem from
           acquisition time back to zero, time reversal gives

               a_γ  ←  -a_γ · b_0^{-1}(x^*_γ; ω_γ, ξ^*_γ).

    Parameters
    ----------
    coeff_indices : Array, shape (K,)
        Flattened indices of *significant* MSWPT coefficients of the
        boundary source F(t, x_s) (already windowed / time-reversed /
        differentiated as required by your adjoint formula).
    domain_data : Domain
        Domain describing the (t, x_*) grid of the boundary data.
        Its `grid_size` is used to scale discrete frequencies to
        physical ones. The physical sound speed is taken from
        ``sources.domain.c_fn``.
    wpt_data : MSWPT
        MSWPT transform instance for the boundary data.
    sources : Sensor
        Sensor geometry; used both for TR parameter construction and
        (implicitly) to locate the acquisition surface Γ.
    relative_guard : float, default=5e-2
        Exclude packets with ``Gamma / abs(tau) <= relative_guard``. The
        default removes the final approximately 2.9 degrees next to grazing;
        it is an operational safeguard for the default adaptive ODE solver,
        not a universal theorem-derived constant.

    Returns
    -------
    pts : (B, d)
        Initial momenta at the boundary for each beam (same as TR).
    Mts : (B, d, d)
        Complex curvature matrices m_T for each beam (same as TR).
    xts : (B, d)
        Boundary points x_T (intersection of rays with acquisition
        surface Γ).
    omegas : (B,)
        Cyclic temporal carrier |tau| for each packet (same as TR).
    ats : (B, 1)
        Geometric amplitudes including the B^{-1} prefactor.
    signum : (B, 1)
        Mode sign (+/-1) for TR propagation (same as TR).
    ts : (B, 2)
        Time intervals [t_start, t_end] for beam ODE integration.
    """
    # -------------------------------------------------------------------------
    # 1. Reuse the TR packet → beam mapping
    # -------------------------------------------------------------------------
    pts, Mts, xts, omegas, ats_geom, signum, ts = compute_TR_parameters(
        coeff_indices, domain_data, wpt_data, sources
    )

    # -------------------------------------------------------------------------
    # 2. Recover packet centres (ω, k_tan) to build B^{-1} symbol.
    #
    #    centres_ndim[k] gives the discrete frequency centre for box k in
    #    (τ, k_tan_1, ..., k_tan_{d_data-1}) coordinates. We rescale by
    #    L_phys to get physical frequencies.
    # -------------------------------------------------------------------------
    decomp = wpt_data.dyadic_decomp
    red = wpt_data.redundancy  # not used explicitly, but part of shapes logic

    L_phys = jnp.array(domain_data.grid_size)  # (d_data,)

    # Map flattened indices → (level, local multi-index)
    shapes = utils.compute_coeff_shapes(decomp, red, jnp.arange(decomp.num_levels))
    cumsum_boxes = jnp.r_[0, jnp.cumsum(decomp.num_boxes_ndim)]
    nn_level, nn_idx = utils.find_tensor_and_multiindex(coeff_indices, shapes)
    box_idx = nn_idx[0, :] + cumsum_boxes[nn_level]  # (B,)

    # Physical Fourier centres in (τ, k_tan) coords
    # centres_hat[b, 0] = τ_b, centres_hat[b, j>0] = (k_tan)_b,j
    centres_hat = decomp.centres_ndim[box_idx, :] / L_phys  # (B, d_data)

    tau = centres_hat[:, 0]  # signed cyclic temporal frequency, (B,)
    k_tan = centres_hat[:, 1:]  # (B, d_data-1) or (B, 0) in 1D

    # Local wave speed on Γ at each packet's spatial centre x_T
    # This is c(0, x^*) in the half-space picture; xts already lives
    # on the boundary hyperplane in physical coordinates.
    c_flat = sources.domain.c_fn(xts).reshape(-1)  # (B,)

    # -------------------------------------------------------------------------
    # 3. Principal symbol of B^{-1} in the hyperbolic region
    #
    #    The packet centres are cyclic frequencies.  `principal_b_inverse`
    #    converts them to angular variables and includes the temporal branch
    #    sign required by the outgoing half-space solution.
    # -------------------------------------------------------------------------
    # `compute_TR_parameters` evolves from acquisition time back to zero, so it
    # is an advanced/terminal-value propagator.  Time reversal changes the
    # retarded outgoing symbol b^{-1}(tau) to b^{-1}(-tau)=-b^{-1}(tau).
    Binv = -principal_b_inverse(tau, k_tan, c_flat, relative_guard=relative_guard)
    # print("Min/Max |B^{-1}| =", jnp.min(jnp.abs(Binv)), jnp.max(jnp.abs(Binv)))
    # A zero multiplier is also a structural exclusion. Merely zeroing the
    # amplitude leaves the near-grazing Hessian in the ODE batch, where its
    # 1/Gamma factors can exhaust the adaptive solver even though the beam can
    # contribute nothing. Replace excluded geometry with the same benign
    # placeholders used for exactly grazing TR packets before propagation.
    excluded = Binv == 0.0
    pts = jnp.where(excluded[:, None], jnp.ones_like(pts), pts)
    identity_mats = 1j * jnp.eye(Mts.shape[-1], dtype=Mts.dtype)[None, :, :]
    Mts = jnp.where(excluded[:, None, None], identity_mats, Mts)
    Binv = Binv.reshape(-1, 1)  # (B, 1)

    # -------------------------------------------------------------------------
    # 4. Fold B^{-1} into the geometric amplitudes
    #
    #    ats_geom already includes the MSWPT → beam Jacobian and TR
    #    normalisation; we only add the B^{-1} microlocal factor here.
    #    Grazing packets have either Γ≈0 (⇒ Binv = 0) and/or were already
    #    damped by TR parameter construction; both mechanisms are consistent.
    # -------------------------------------------------------------------------
    ats = ats_geom * Binv  # (B, 1)

    return pts, Mts, xts, omegas, ats, signum, ts
