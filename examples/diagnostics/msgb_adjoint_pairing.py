#!/usr/bin/env python3
"""Controlled pairing diagnostic for the principal-symbol MSGB PAT adjoint.

This is deliberately a diagnostic, not a unit test and not thesis evidence.
``MSGBSolver.adjoint`` is a leading-order continuous-adjoint approximation; it
is not the algebraic transpose of the thresholded discrete MSGB forward map.
The script therefore reports two distinct comparisons:

1. JAX's VJP of the *implemented, windowed* forward map.  This is the local
   exact discrete transpose (the top-N packet set is frozen by the VJP), and
   its dot-product residual should be at roundoff.
2. The principal-symbol MSGB backprojection under the continuous rectangle-rule
   pairing ``dt * sum(data)`` versus ``dx * sum(image)``.  Its residual is an
   approximation diagnostic, not an exact-transpose acceptance criterion.

The case is homogeneous and one dimensional, so there is no tangential or
grazing complication and the half-space principal symbol is exact at normal
incidence.  A smooth high-frequency packet and a tapered acquisition window
keep the experiment inside the intended microlocal regime.

Run from the public package root::

    .venv/bin/python examples/diagnostics/msgb_adjoint_pairing.py
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from beamax import geometry
from beamax.decomposition import DyadicDecomposition
from beamax.gb import gb_solvers
from beamax.solvers import MSGBSolver
from beamax.transforms import MSWPT


jax.config.update("jax_enable_x64", True)


def _constant_speed(speed: float):
    def c(x: jnp.ndarray) -> jnp.ndarray:
        return speed + jnp.zeros(x.shape[:-1], dtype=x.dtype)

    return c


def _packet(
    x: jnp.ndarray, *, centre: float, width: float, cycles: float
) -> jnp.ndarray:
    envelope = jnp.exp(-0.5 * ((x - centre) / width) ** 2)
    return envelope * jnp.cos(2.0 * jnp.pi * cycles * (x - centre))


def _relative_pairing_error(lhs: float, rhs: float) -> float:
    return abs(lhs - rhs) / max(abs(lhs), abs(rhs), np.finfo(float).eps)


def _fit(reference: jnp.ndarray, candidate: jnp.ndarray) -> tuple[float, float, float]:
    """Return candidate scale, best-scaled error, and cosine against reference."""
    ref_norm = float(jnp.linalg.norm(reference))
    cand_norm = float(jnp.linalg.norm(candidate))
    dot = float(jnp.vdot(candidate, reference).real)
    scale = dot / max(cand_norm**2, np.finfo(float).eps)
    error = float(
        jnp.linalg.norm(scale * candidate - reference)
        / max(ref_norm, np.finfo(float).eps)
    )
    cosine = dot / max(cand_norm * ref_norm, np.finfo(float).eps)
    return scale, error, cosine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=128, help="image/data grid size")
    parser.add_argument(
        "--cycles", type=float, default=24.0, help="packet carrier cycles per unit"
    )
    parser.add_argument(
        "--top-n", type=int, default=32, help="retained coefficients per solve"
    )
    parser.add_argument("--speed", type=float, default=1.0, help="constant wave speed")
    args = parser.parse_args()

    n = args.n
    cycles = args.cycles
    top_n = args.top_n
    speed = args.speed
    dx = 1.0 / n
    nt = n
    dt = 1.0 / (speed * nt)
    c_fn = _constant_speed(speed)

    domain = geometry.Domain(
        N=(n,),
        dx=(dx,),
        c=c_fn,
        cfl=0.3,
        periodic=(False,),
    )
    image_decomp = DyadicDecomposition(
        num_levels=2,
        N=(n,),
        num_boxes_levels=(4, 8),
        box_aspect_ratio=(1,),
    )
    image_wpt = MSWPT(image_decomp, redundancy=2, windowing="rectangular")

    ts = jnp.arange(nt, dtype=jnp.float64) * dt
    data_domain = geometry.Domain(
        N=(nt,),
        dx=(dt,),
        c=c_fn,
        cfl=0.3,
        periodic=(False,),
    )
    data_decomp = DyadicDecomposition(
        num_levels=2,
        N=(nt,),
        num_boxes_levels=(4, 8),
        box_aspect_ratio=(1,),
    )
    data_wpt = MSWPT(
        data_decomp,
        redundancy=2,
        windowing="rectangular_mirror",
    )

    boundary_mask = jnp.zeros((n,), dtype=jnp.float64).at[0].set(1.0)
    boundary = geometry.Sensor(domain=domain, binary_mask=boundary_mask)
    image_grid = geometry.Sensor(
        domain=domain,
        binary_mask=jnp.ones((n,), dtype=jnp.float64),
    )

    solver = MSGBSolver(
        thr=top_n,
        thr_strat="top_n",
        batch_size=32,
        input_type="spatial",
        ode_solver=gb_solvers.solve_ODE_base,
        tr_ode_solver=gb_solvers.solve_ODE_batch_t,
        sum_method="all_real",
    )

    x = domain.grid[..., 0]
    f = _packet(x, centre=0.58, width=0.075, cycles=cycles)

    # The taper is exactly zero at both endpoints.  The implemented forward
    # being paired is Bf = window * Pf, matching the window used by adjoint().
    window = jnp.sin(jnp.pi * jnp.arange(nt) / (nt - 1)) ** 2

    def windowed_forward(image: jnp.ndarray) -> jnp.ndarray:
        data = solver.forward(image, domain, boundary, ts, image_wpt)
        return window * jnp.ravel(data)

    bf, pullback = jax.vjp(windowed_forward, f)
    # y := Pf makes the nonzero pairing numerically stable.  Since
    # Bf = window * Pf, B^T y contains one (and only one) taper factor.
    pf = jnp.ravel(solver.forward(f, domain, boundary, ts, image_wpt))
    y = pf
    exact_transpose = pullback(y)[0]

    principal_adjoint = jnp.ravel(
        solver.adjoint(
            data=y,
            domain=domain,
            sensors=image_grid,
            sources=boundary,
            ts=ts,
            data_domain=data_domain,
            data_wpt=data_wpt,
            window=window,
        )
    )

    lhs = float(dt * jnp.vdot(bf, y).real)
    rhs_exact = float(dt * jnp.vdot(f, exact_transpose).real)
    rhs_principal = float(dx * jnp.vdot(f, principal_adjoint).real)

    # The VJP is an unweighted-sum transpose.  dt/dx converts it to samples of
    # the continuous-adjoint field under rectangle-rule quadrature.
    exact_continuous_adjoint = (dt / dx) * exact_transpose
    fit, fitted_error, cosine = _fit(exact_continuous_adjoint, principal_adjoint)

    # D'Alembert reference for a full-line wave with f supported in x > 0:
    # (P f)(0,t) = (f(c*t) + f(-c*t))/2 ~= f(c*t)/2, and
    # (P* h)(x) = h(x/c)/(2c).  Here dt=dx/c, so the image and data arrays
    # align sample by sample.  The negative-x tail of this packet is below
    # 1e-13, so the one-sided formula is adequate here.
    analytic_forward = 0.5 * _packet(
        speed * ts, centre=0.58, width=0.075, cycles=cycles
    )
    analytic_adjoint = 0.5 * window * y / speed
    fwd_fit, fwd_error, fwd_cosine = _fit(analytic_forward, pf)
    adj_fit, adj_error, adj_cosine = _fit(analytic_adjoint, principal_adjoint)

    print("Controlled 1D homogeneous MSGB adjoint pairing diagnostic")
    print(f"grid: image={n}, data={nt}, dx={dx:.8f}, dt={dt:.8f}, c={speed:g}")
    print(f"carrier: {cycles:g} cycles/unit; top-N packets: {top_n}; y = P f")
    print()
    print(f"lhs = dt <window * Pf, y>            {lhs:+.12e}")
    print(f"rhs (JAX local discrete transpose)   {rhs_exact:+.12e}")
    print(
        "exact-transpose relative residual         "
        f"{_relative_pairing_error(lhs, rhs_exact):.6e}"
    )
    print(f"rhs (principal-symbol MSGB adjoint)  {rhs_principal:+.12e}")
    print(
        "principal-symbol relative pairing residual "
        f"{_relative_pairing_error(lhs, rhs_principal):.6e}"
    )
    print(f"principal/exact pairing ratio         {rhs_principal / lhs:+.6e}")
    print(f"image cosine against scaled VJP       {cosine:+.6e}")
    print(f"best scalar applied to principal image {fit:+.6e}")
    print(f"best-scaled image relative L2 error   {fitted_error:.6e}")
    print()
    print("D'Alembert amplitude/shape references")
    print(
        f"forward:  scale={fwd_fit:+.6e}, rel-L2={fwd_error:.6e}, "
        f"cosine={fwd_cosine:+.6e}"
    )
    print(
        f"adjoint:  scale={adj_fit:+.6e}, rel-L2={adj_error:.6e}, "
        f"cosine={adj_cosine:+.6e}"
    )


if __name__ == "__main__":
    main()
