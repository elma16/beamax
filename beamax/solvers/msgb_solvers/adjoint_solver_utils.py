from typing import Tuple

import jax.numpy as jnp

from beamax import utils
from beamax.geometry import Domain, Sensor
from beamax.transforms import MSWPT

from .tr_solver_utils import compute_TR_parameters

Array = jnp.ndarray

__all__ = ["compute_adj_parameters", "principal_b_inverse"]


def principal_b_inverse(
    tau: Array,
    k_tan: Array,
    c: Array,
    relative_guard: float = 1e-6,
) -> Array:
    r"""Evaluate the retarded/outgoing half-space principal symbol $B^{-1}$.

    ``tau`` and ``k_tan`` are cyclic frequencies.  They are converted to
    angular variables before evaluating

    $$
    -\frac{i\,\operatorname{sgn}(\Omega)}
    {2c\sqrt{\Omega^2-c^2\lVert Q_{\mathrm{tan}}\rVert^2}}.
    $$

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
        Minimum ratio $\gamma_{\mathrm{ang}}/|\omega_{\mathrm{ang}}|$.
        This is a local, dimensionless near-grazing guard.

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
    r"""Apply the advanced $B^{-1}$ symbol to TR beam parameters.

    Geometry and curvature come from :func:`compute_TR_parameters`; only the
    amplitudes receive the microlocal multiplier.

    Parameters
    ----------
    coeff_indices : Array, shape (K,)
        Significant MSWPT coefficient indices of the prepared boundary source.
    domain_data : Domain
        Boundary-data domain used to scale discrete frequencies.
    wpt_data : MSWPT
        MSWPT transform instance for the boundary data.
    sources : Sensor
        Acquisition geometry and physical sound-speed domain.
    relative_guard : float, default=5e-2
        Exclude packets with
        $\Gamma/|\tau|\le\mathtt{relative\_guard}$.

    Returns
    -------
    pts : (B, d)
        Initial boundary momenta.
    Mts : (B, d, d)
        Complex boundary curvature matrices.
    xts : (B, d)
        Boundary points $x_T$ (intersection of rays with acquisition
        surface $\Gamma$).
    omegas : (B,)
        Cyclic temporal carrier $|\tau|$.
    ats : (B, 1)
        Geometric amplitudes including the $B^{-1}$ prefactor.
    signum : (B, 1)
        TR propagation mode sign.
    ts : (B, 2)
        Time intervals $[t_{\mathrm{start}},t_{\mathrm{end}}]$ for beam ODE
        integration.
    """
    pts, Mts, xts, omegas, ats_geom, signum, ts = compute_TR_parameters(
        coeff_indices, domain_data, wpt_data, sources
    )

    decomp = wpt_data.dyadic_decomp
    red = wpt_data.redundancy

    L_phys = jnp.array(domain_data.grid_size)

    shapes = utils.compute_coeff_shapes(decomp, red, jnp.arange(decomp.num_levels))
    cumsum_boxes = jnp.r_[0, jnp.cumsum(decomp.num_boxes_ndim)]
    nn_level, nn_idx = utils.find_tensor_and_multiindex(coeff_indices, shapes)
    box_idx = nn_idx[0, :] + cumsum_boxes[nn_level]

    # Physical Fourier centres are ordered as (tau, tangential wave numbers).
    centres_hat = decomp.centres_ndim[box_idx, :] / L_phys

    tau = centres_hat[:, 0]
    k_tan = centres_hat[:, 1:]

    c_flat = sources.domain.c_fn(xts).reshape(-1)

    # Terminal propagation changes retarded b^{-1}(tau) to -b^{-1}(tau).
    Binv = -principal_b_inverse(tau, k_tan, c_flat, relative_guard=relative_guard)
    # Excluded beams need benign geometry as well as zero amplitude; their
    # near-grazing Hessians can otherwise exhaust the adaptive solver.
    excluded = Binv == 0.0
    pts = jnp.where(excluded[:, None], jnp.ones_like(pts), pts)
    identity_mats = 1j * jnp.eye(Mts.shape[-1], dtype=Mts.dtype)[None, :, :]
    Mts = jnp.where(excluded[:, None, None], identity_mats, Mts)
    Binv = Binv.reshape(-1, 1)

    ats = ats_geom * Binv

    return pts, Mts, xts, omegas, ats, signum, ts
