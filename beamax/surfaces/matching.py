"""Curved packet-to-beam matching on a ``DetectorSurface`` in 1D, 2D and 3D.

Implements Steps 3--4 of ``extensions/curved_sensors.tex`` for any dimension, in beamax's
time-reversal conventions: raw acquisition time, forward-time beams leaving the
reconstruction side through the sensor at their packet time, ``mode = -sign(tau)``, and
normal momentum ``sign(tau)`` times the front normal.  For an axis-aligned plane it
reproduces beamax's ``compute_TR_parameters`` exactly.

    Step 3   J^T (omega p) = 2 pi eta,  |p| = 2 pi / c,  p = p_tan + sign(tau) p_n n
    Step 4   M = E^{-T} N E^{-1},  E = [xdot | J],
             N = [[Mt_tt + pdot.xdot, (J^T pdot - Mt_t*)^T], [J^T pdot - Mt_t*, Mt_** - Q]],
             Q = p . D^2 X
"""

from __future__ import annotations

from numbers import Integral

import jax
import jax.numpy as jnp

from beamax import utils
from beamax.gb import gb_utils, gb_solvers
from beamax.solvers.msgb_solvers import tr_solver_utils
from beamax.solvers.msgb_solvers.forward_solver_utils import threshold_coefficients
from beamax.transforms import compute_frame_phase

from .surface import DetectorSurface


def decode_packets(coeff_idx, data_domain, data_wpt):
    """Phase-space centres, widths and synthesis amplitudes of MSWPT packets on a ``(t, vth)`` grid."""
    decomp = data_wpt.dyadic_decomp
    red = data_wpt.redundancy
    d = data_domain.ndim
    box_lengths = jnp.array(decomp.box_lengths)
    box_aspect = jnp.array(decomp.box_aspect_ratio)
    N_data = jnp.array(data_domain.N)
    L_phys = jnp.array(data_domain.grid_size)

    shapes = utils.compute_coeff_shapes(decomp, red, jnp.arange(decomp.num_levels))
    cumsum_boxes = jnp.r_[0, jnp.cumsum(decomp.num_boxes_ndim)]
    nn_level, nn_idx = utils.find_tensor_and_multiindex(coeff_idx, shapes)
    box_idx = nn_idx[0, :] + cumsum_boxes[nn_level]
    B = box_idx.shape[0]

    centres = (
        decomp.centres_ndim[box_idx, :] / L_phys
    )  # (tau, eta) in cycles per data unit
    bl = box_lengths[nn_level][:, None] / L_phys * box_aspect
    Lls = bl * red
    sigmas = bl / 2.0
    t_gamma = nn_idx[1, :] / Lls[:, 0]
    if d > 1:
        vth_rel = jnp.stack(nn_idx[2:, :], axis=-1) / Lls[:, 1:]
    else:
        vth_rel = jnp.zeros((B, 0))
    local_k = jnp.stack(nn_idx[1:, :], axis=-1)
    amp = jnp.prod(
        jnp.sqrt(jnp.pi * L_phys[None, :] / (Lls * N_data[None, :])) * sigmas, axis=1
    )
    amp = amp * compute_frame_phase(decomp, box_idx, local_k, red)
    tau = centres[:, 0]
    return dict(
        tau=tau,
        eta=centres[:, 1:],
        t_gamma=t_gamma,
        vth_rel=vth_rel,
        sigmas=sigmas,
        amp=amp,
        omega=jnp.where(tau != 0, jnp.abs(tau), 1e-6),
        level=nn_level,
    )


def launch_momentum(surface: DetectorSurface, vth, tau, eta, c_fn):
    """Step 3.  Returns ``(p, hyperbolic, X, J, n, c)``; non-hyperbolic packets get a finite placeholder."""
    X, J, n = surface.points(vth), surface.jacobian(vth), surface.normal(vth)
    c0 = jax.vmap(c_fn)(X)
    omega = jnp.where(
        tau != 0, jnp.abs(tau), 1.0
    )  # exact for every tau != 0; tau = 0 only feeds the placeholder
    if surface.m:
        I1 = jnp.einsum("bki,bkj->bij", J, J)
        Iinv_eta = jnp.linalg.solve(I1, eta[..., None])[..., 0]
        tangent = jnp.einsum("bki,bi->bk", J, (c0 / omega)[:, None] * Iinv_eta)
    else:
        tangent = jnp.zeros_like(X)
    # Use the same dimensionless discriminant to select and construct the
    # normal component. A rounded-zero normal makes the Hessian solve singular.
    disc = 1.0 - jnp.sum(tangent**2, axis=-1)
    hyperbolic = (tau != 0) & (disc > 8 * jnp.finfo(disc.dtype).eps)
    tangent = jnp.where(hyperbolic[:, None], tangent, 0)
    normal = jnp.sqrt(jnp.where(hyperbolic, disc, 1.0))
    signed_normal = jnp.where(tau < 0, -normal, normal)
    p = (2 * jnp.pi / c0)[:, None] * (tangent + signed_normal[:, None] * n)
    return p, hyperbolic, X, J, n, c0


def curvature_block(surface: DetectorSurface, vth, p):
    """``Q = p . D^2 X``, shape ``(B, m, m)``."""
    return jnp.einsum("bk,bkij->bij", p, surface.hessian(vth))


def reduced_hessian(surface: DetectorSurface, vth, p, M, xdot, pdot, include_Q=True):
    """Curved trace lemma: Hessian of the beam phase restricted to the surface, in ``(t, vth)``."""
    J = surface.jacobian(vth)
    B, d, m = J.shape
    out = jnp.zeros((B, d, d), dtype=jnp.result_type(M, 1j))
    Mxd = jnp.einsum("bij,bj->bi", M, xdot)
    out = out.at[:, 0, 0].set(
        jnp.einsum("bi,bi->b", xdot, Mxd) - jnp.einsum("bi,bi->b", pdot, xdot)
    )
    if m:
        cross = jnp.einsum("bki,bk->bi", J, pdot - Mxd)
        out = out.at[:, 0, 1:].set(cross).at[:, 1:, 0].set(cross)
        tang = jnp.einsum("bki,bkl,blj->bij", J, M, J)
        if include_Q:
            tang = tang + curvature_block(surface, vth, p)
        out = out.at[:, 1:, 1:].set(tang)
    return out


def solve_reduced_hessian(
    surface: DetectorSurface, vth, p, xdot, pdot, Mt, include_Q=True
):
    """Step 4: the unique symmetric ``M`` with ``reduced_hessian(M) = Mt``.  Returns ``(M, E^{-1}, Q)``."""
    J = surface.jacobian(vth)
    B, d, m = J.shape
    E = jnp.concatenate([xdot[:, :, None], J], axis=2)
    Mt = jnp.asarray(Mt, dtype=jnp.result_type(Mt, 1j))
    N = jnp.zeros((B, d, d), dtype=Mt.dtype)
    N = N.at[:, 0, 0].set(Mt[:, 0, 0] + jnp.einsum("bi,bi->b", pdot, xdot))
    Q = jnp.zeros((B, m, m))
    if m:
        cross = jnp.einsum("bki,bk->bi", J, pdot) - Mt[:, 0, 1:]
        N = N.at[:, 0, 1:].set(cross).at[:, 1:, 0].set(cross)
        Q = curvature_block(surface, vth, p) if include_Q else Q
        N = N.at[:, 1:, 1:].set(Mt[:, 1:, 1:] - Q)
    Einv = jnp.linalg.inv(E)
    M = jnp.einsum("bji,bjk,bkl->bil", Einv, N, Einv)
    return 0.5 * (M + jnp.swapaxes(M, 1, 2)), Einv, Q


def match_packets(surface: DetectorSurface, vth, tau, eta, Mt, c_fn, include_Q=True):
    """Steps 3--4 for packets at ``vth`` with frequency ``(tau, eta)`` and target ``Mt`` in ``(t, vth)``.

    Returns a dict with the beam launch state ``p, M, X``, ``mode = -sign(tau)``, ``hyperbolic`` and the
    intermediate ``xdot, pdot, normal, c, Q, Einv``.  ``M`` is not masked for non-hyperbolic packets.
    """
    vth = surface._batch(vth)
    p, hyperbolic, X, J, n, c0 = launch_momentum(surface, vth, tau, eta, c_fn)
    mode = jnp.where(
        tau < 0, 1.0, -1.0
    )  # -sign(tau); tau = 0 (non-hyperbolic) gets a finite placeholder
    xdot = gb_utils.vmap_gp(X, p, mode, c_fn)
    pdot = -gb_utils.vmap_gx(X, p, mode, c_fn)
    M, Einv, Q = solve_reduced_hessian(surface, vth, p, xdot, pdot, Mt, include_Q)
    return dict(
        p=p,
        M=M,
        X=X,
        mode=mode,
        hyperbolic=hyperbolic,
        xdot=xdot,
        pdot=pdot,
        normal=n,
        c=c0,
        Q=Q,
        Einv=Einv,
    )


def surface_tr_parameters(
    coeff_idx, data_domain, data_wpt, surface: DetectorSurface, c_fn, include_Q=True
):
    """Curved analogue of beamax ``compute_TR_parameters``.

    Returns ``(pts, Mts, xts, omegas, ats, signum, ts)`` in beamax's layout and a diagnostics dict.
    """
    surface.check_data_domain(data_domain)
    P = decode_packets(coeff_idx, data_domain, data_wpt)
    vth = jnp.asarray(surface.origin, dtype=float)[None, :] + P["vth_rel"]
    tau, omega = P["tau"], P["omega"]
    alpha = 2j * (jnp.pi * P["sigmas"]) ** 2 / omega[:, None]
    Mt = jax.vmap(jnp.diag)(alpha)
    launch_vth, inside_chart = surface.active_parameters(vth)
    B = match_packets(surface, launch_vth, tau, P["eta"], Mt, c_fn, include_Q)
    hyperbolic, n, c0, xdot = B["hyperbolic"], B["normal"], B["c"], B["xdot"]
    M = jnp.where(hyperbolic[:, None, None], B["M"], 1j * jnp.eye(surface.ndim)[None])
    ats = jnp.where(hyperbolic & inside_chart, P["amp"], 0.0)[:, None]
    ts = jnp.stack([P["t_gamma"], jnp.zeros_like(P["t_gamma"])], axis=-1)
    diag = dict(
        vth=vth,
        inside_chart=inside_chart,
        tau=tau,
        eta=P["eta"],
        t_gamma=P["t_gamma"],
        hyperbolic=hyperbolic,
        Mt=Mt,
        Q=B["Q"],
        Einv=B["Einv"],
        xdot=xdot,
        pdot=B["pdot"],
        normal=n,
        c=c0,
        level=P["level"],
        cos_inc=jnp.abs(jnp.einsum("bi,bi->b", xdot, n)) / c0,
    )
    return (B["p"], M, B["X"], omega, ats, B["mode"][:, None], ts), diag


def boundary_coefficients(data, data_wpt, top_n):
    """Select pressure coefficients with the weights for real beam rendering."""
    if isinstance(top_n, bool) or not isinstance(top_n, Integral) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")
    coeffs = 0.5 * data_wpt.forward(jnp.copy(jnp.asarray(data)), "spatial")
    if data_wpt.has_conjugate_frame:
        coeffs = coeffs * data_wpt.half_mask
        top_n = min(top_n, data_wpt.total_coeffs // 2)
    idx, values = threshold_coefficients(coeffs, int(top_n), "top_n")
    # The real renderer doubles every beam. An unpaired frame uses both halves.
    return idx, values if data_wpt.has_conjugate_frame else 0.5 * values


def surface_time_reversal(
    data,
    data_domain,
    data_wpt,
    surface: DetectorSurface,
    domain,
    top_n,
    include_Q=True,
    batch_size=256,
    ode_solver=None,
    solver_config=None,
    eval_points=None,
    keep_fn=None,
    aggregate_method="scan",
    pallas_config=None,
):
    """MSGB time reversal from boundary data on ``surface`` (beamax pipeline with curved matching).

    Scalar sound speed and zero absorption use the homogeneous closed form.
    Gridded or callable speeds and absorbing media use the ODE solver.
    ``eval_points`` has shape ``(..., ndim)``. Its leading dimensions and
    point order are preserved in the reconstruction.
    """
    if aggregate_method not in ("scan", "terminal_xla", "pallas", "all"):
        raise ValueError("aggregate_method must be scan, terminal_xla, pallas or all.")
    if domain.ndim != surface.ndim or any(domain.periodic):
        raise ValueError(
            "Curved time reversal requires a free-space domain of the surface dimension."
        )
    surface.check_data_domain(data_domain)
    data = jnp.asarray(data)
    if data.shape != tuple(data_domain.N) or data.shape != tuple(
        data_wpt.dyadic_decomp.N
    ):
        raise ValueError(
            "Data, acquisition domain and transform must have the same shape."
        )
    if jnp.iscomplexobj(data):
        raise ValueError("Curved time reversal requires real-valued pressure traces.")
    points = domain.grid if eval_points is None else jnp.asarray(eval_points)
    if points.ndim < 1 or points.shape[-1] != domain.ndim:
        raise ValueError(
            "eval_points must have a final coordinate axis of length domain.ndim."
        )
    if ode_solver is None:
        homogeneous = (
            not callable(domain.c) and jnp.ndim(domain.c) == 0 and domain.lam == 0
        )
        ode_solver = (
            gb_solvers.solve_hom_TR if homogeneous else gb_solvers.solve_ODE_batch_t
        )
    coeff_idx, values = boundary_coefficients(data, data_wpt, top_n)
    params, diag = surface_tr_parameters(
        coeff_idx, data_domain, data_wpt, surface, domain.c_fn, include_Q
    )
    pts, Mts, xts, oms, ats, signum, ts = params
    ats = ats * values[:, None]
    if keep_fn is not None:
        ats = ats * jnp.asarray(keep_fn(params, diag))[:, None]
    render_params = (pts, Mts, xts, oms, ats, signum, ts)
    if aggregate_method != "all":
        render_params = utils.batch_data(
            *render_params, batch_size=batch_size, zero_padded_args=(4,)
        )
    rec = tr_solver_utils.compute_TR_result(
        render_params,
        c=domain.c_fn,
        lam=domain.lam,
        sensors=points,
        domain_size=domain.grid_size,
        periodic=jnp.array(domain.periodic),
        ode_solver=ode_solver,
        aggregate_method=aggregate_method,
        pallas_config=pallas_config,
        solver_config=solver_config,
    )
    diag["coefficients"] = values
    return 2.0 * rec, diag
