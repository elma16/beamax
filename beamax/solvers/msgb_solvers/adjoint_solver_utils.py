from typing import Tuple

import jax.numpy as jnp

from beamax import utils
from beamax.geometry import Domain, Sensor
from beamax.transforms import MSWPT

# Reuse the *tested* TR parameter builder
from .tr_solver_utils import compute_TR_parameters

Array = jnp.ndarray

__all__ = ["compute_adj_parameters"]


def compute_adj_parameters(
    coeff_indices: Array,
    domain_data: Domain,
    wpt_data: MSWPT,
    sources: Sensor,
) -> Tuple[Array, ...]:
    """
    Compute Gaussian beam parameters for the adjoint solve.

    Design:
        1. Reuse the time-reversal packet→beam mapping in
           `compute_TR_parameters`. This gives us beam geometry
           (x_T, p_T, M_T, ω, sign, time-intervals) that is already
           tested in the TR pipeline.

        2. On top of that, apply the approximate B^{-1} symbol in
           (ω, k_tan) to the beam amplitudes only. This implements the
           microlocal equivalence

               L q = F(x^*, t) δ(x_1)  ↔  L q = 0,  q|_{Γ} = h,
               with h ≈ B^{-1} F,

           where B has principal symbol

               b_0(x^*; ω, ξ^*)
                 = -2 i c(0, x^*) sqrt(ω^2 - c(0, x^*)^2 |ξ^*|^2).

           We keep the TR geometry and curvature from `compute_TR_parameters`
           and only modify amplitudes via

               a_γ  ←  a_γ · b_0^{-1}(x^*_γ; ω_γ, ξ^*_γ).

    Parameters
    ----------
    coeff_indices : Array, shape (K,)
        Flattened indices of *significant* MSWPT coefficients of the
        boundary source F(t, x_s) (already windowed / time-reversed /
        differentiated as required by your adjoint formula).
    domain_data : Domain
        Domain describing the (t, x_*) grid of the boundary data.
        Its `grid_size` is used to scale discrete frequencies to
        physical ones; its `c(x)` is the physical sound speed.
    wpt_data : MSWPT
        MSWPT transform instance for the boundary data.
    sources : Sensor
        Sensor geometry; used both for TR parameter construction and
        (implicitly) to locate the acquisition surface Γ.

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

    tau = centres_hat[:, 0]  # (B,)
    k_tan = centres_hat[:, 1:]  # (B, d_data-1) or (B, 0) in 1D
    omega = jnp.abs(tau)  # |τ| ≡ ω  (B,)

    if k_tan.shape[1] > 0:
        k_tan_sq = jnp.sum(k_tan**2, axis=1)  # |k_tan|^2
    else:
        k_tan_sq = jnp.zeros_like(omega)

    # Local wave speed on Γ at each packet's spatial centre x_T
    # This is c(0, x^*) in the half-space picture; xts already lives
    # on the boundary hyperplane in physical coordinates.
    c_flat = domain_data.c(xts).reshape(-1)  # (B,)

    # -------------------------------------------------------------------------
    # 3. Principal symbol of B^{-1} in the hyperbolic region
    #
    #    Γ(x^*; τ, ξ^*) = sqrt(τ^2 - c(x^*)^2 |ξ^*|^2),
    #    b_0(x^*; ω, ξ^*) = -2 i c(x^*) Γ,
    #    => |b_0| ≈ 2 c(x^*) Γ for the rescaled cyclic multiplier used here.
    # -------------------------------------------------------------------------
    rad = omega**2 - (c_flat**2) * k_tan_sq
    rad = jnp.maximum(rad, 0.0)
    Gamma = jnp.sqrt(rad)  # (B,)

    # Avoid blow-up at grazing / evanescent modes: set B^{-1}=0 there.
    # eps is scaled to the max Γ to be robust across experiments.
    eps = 1e-6 * (1.0 + jnp.max(Gamma))
    Binv = jnp.where(Gamma > eps, 1j / (2.0 * c_flat * Gamma), 0.0)
    # print("Min/Max |B^{-1}| =", jnp.min(jnp.abs(Binv)), jnp.max(jnp.abs(Binv)))
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
