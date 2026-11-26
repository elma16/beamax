"""Principal-symbol PAT adjoint on a sampled detector surface."""

from functools import partial
from math import prod

import jax
import jax.numpy as jnp
import numpy as np

from beamax.solvers.msgb_solvers import apply_adjoint_image_weight, form_adjoint_source
from beamax.solvers.msgb_solvers.adjoint_solver_utils import _advanced_b_inverse

from .matching import _backpropagate_surface, surface_tr_parameters
from .surface import DetectorSurface


def _surface_adjoint_parameters(
    coeff_idx, data_domain, data_wpt, surface, c_fn, include_Q, *, guard
):
    params, diag = surface_tr_parameters(
        coeff_idx, data_domain, data_wpt, surface, c_fn, include_Q
    )
    p, M, x, omega, amp, mode, ts = params
    omega_ang = 2 * jnp.pi * diag["tau"]
    # Launch momentum already accounts for the inverse metric of the chart.
    # Its normal incidence cosine is gamma / |angular temporal frequency|.
    gamma = jnp.where(
        diag["hyperbolic"] & diag["inside_chart"],
        jnp.abs(omega_ang) * diag["cos_inc"],
        0.0,
    )
    multiplier = _advanced_b_inverse(omega_ang, gamma, diag["c"], guard)
    diag["boundary_multiplier"] = multiplier
    # As in the planar adjoint, keep rejected grazing Hessians out of the ODE.
    active = multiplier != 0
    p = jnp.where(active[:, None], p, jnp.ones_like(p))
    M = jnp.where(active[:, None, None], M, 1j * jnp.eye(surface.ndim, dtype=M.dtype))
    return (p, M, x, omega, amp * multiplier[:, None], mode, ts), diag


def surface_adjoint(
    data,
    data_domain,
    data_wpt,
    surface: DetectorSurface,
    domain,
    top_n,
    *,
    pairing="surface_area",
    guard=0.05,
    window=None,
    eval_points=None,
    **tr_options,
):
    """Approximate the continuous lossless PAT adjoint on an oriented regular chart.

    The default data pairing uses dt times the physical surface measure.
    `element_sum` uses dt per detector. Both use the volume measure in the image.
    A real time window or a window broadcastable over the traces is applied before
    spectral differentiation. Taper the residual at the ends of the time record.

    The advanced principal symbol weights each retained boundary packet at its
    centre. Packets with incidence cosine at or below `guard` are excluded.
    Rendering options and the returned (image, diagnostics) follow
    `surface_time_reversal`. Diagnostics also contain the complex boundary multiplier.
    """
    if domain.lam != 0.0:
        raise NotImplementedError(
            "The surface adjoint requires a lossless domain (lam=0)."
        )
    if domain.ndim != surface.ndim or any(domain.periodic):
        raise ValueError(
            "The surface adjoint requires a free-space domain of the surface dimension."
        )
    if pairing not in ("surface_area", "element_sum"):
        raise ValueError("pairing must be 'surface_area' or 'element_sum'.")
    if not np.isfinite(guard) or not 0 <= guard < 1:
        raise ValueError("guard must be in [0, 1).")
    surface.check_data_domain(data_domain)
    data = jnp.asarray(data)
    if data.shape != tuple(data_domain.N) or data.shape[0] < 2:
        raise ValueError(
            "The residual must match the acquisition grid and have at least two time samples."
        )
    if jnp.iscomplexobj(data) or not jnp.all(jnp.isfinite(data)):
        raise ValueError("The surface adjoint requires real, finite residuals.")
    data = data.astype(jnp.result_type(data, 1.0))
    if window is not None:
        window = jnp.asarray(window)
        if jnp.iscomplexobj(window) or not jnp.all(jnp.isfinite(window)):
            raise ValueError("The adjoint window must be real and finite.")
        if window.ndim == 1:
            window = window.reshape((-1,) + (1,) * surface.m)
        window = jnp.broadcast_to(window, data.shape)

    # Padded open charts can be singular beyond their physical aperture.
    theta, inside = surface.active_parameters(surface.params_flat())
    cell_measure = surface.area_element(theta).reshape(surface.shape) * prod(
        surface.spacing
    )
    if not jnp.all(jnp.isfinite(cell_measure) & (cell_measure > 0)):
        raise ValueError(
            "The detector chart must have a positive finite surface measure."
        )
    c_sensor = jax.vmap(domain.c_fn)(surface.points(theta)).reshape(surface.shape)
    if not jnp.all(jnp.isfinite(c_sensor) & (c_sensor > 0)):
        raise ValueError(
            "The sound speed at detector samples must be positive and finite."
        )
    source = form_adjoint_source(data, data_domain.dx[0], c_sensor, window)
    if pairing == "element_sum":
        source = source / cell_measure
    source = jnp.where(inside.reshape(surface.shape), source, 0.0)

    points = domain.grid if eval_points is None else jnp.asarray(eval_points)
    image, diag = _backpropagate_surface(
        source,
        data_domain,
        data_wpt,
        surface,
        domain,
        top_n,
        partial(_surface_adjoint_parameters, guard=guard),
        eval_points=points,
        **tr_options,
    )
    c_image = jax.vmap(domain.c_fn)(points.reshape(-1, domain.ndim)).reshape(
        points.shape[:-1]
    )
    return apply_adjoint_image_weight(image, c_image), diag
