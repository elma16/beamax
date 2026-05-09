#!/usr/bin/env python
"""
Forward solve with non-zero initial velocity (v0) compared to a spectral ground truth.

This mirrors the Cauchy data setup in Qian (2010): u(0,x)=p0, ut(0,x)=v0.
We use the MSGB forward solver with dpdt ≠ 0 and compare against the exact
solution of the constant-coefficient wave equation on a periodic domain.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from beamax.decomposition import DyadicDecomposition
from beamax.transforms import MSWPT
from beamax.geometry import Domain, Sensor
from beamax.solvers import MSGBSolver
from beamax.gb import gb_solvers


jax.config.update("jax_enable_x64", True)


def spectral_solution(
    p0: jnp.ndarray, v0: jnp.ndarray, c: float, ts: np.ndarray, dx: float
) -> np.ndarray:
    """
    Exact solution of utt - c^2 u_xx = 0 with periodic BCs via Fourier series.
    """
    p0_np = np.asarray(p0)
    v0_np = np.asarray(v0)
    t = np.asarray(ts)[:, None]

    k = 2 * np.pi * np.fft.fftfreq(p0_np.shape[0], d=float(dx))
    omega = c * np.abs(k)[None, :]

    p_hat = np.fft.fft(p0_np)
    v_hat = np.fft.fft(v0_np)

    sin_term = np.sin(omega * t)
    omega_safe = np.where(omega == 0.0, 1.0, omega)
    sin_over_omega = sin_term / omega_safe
    sin_over_omega = np.where(omega == 0.0, t, sin_over_omega)  # limit sin(0)/0 -> t

    cos_term = np.cos(omega * t)
    u_hat = p_hat[None, :] * cos_term + v_hat[None, :] * sin_over_omega
    return np.fft.ifft(u_hat, axis=1).real


def main():
    # Problem setup (1D, periodic)
    N = (128,)
    domain = Domain(N=N, dx=(1.0 / N[0],), c=1.0, periodic=(True,))
    ts = domain.generate_time_domain()
    x = jnp.arange(N[0]) * domain.dx[0]

    # Initial displacement and velocity
    p0 = jnp.sin(2 * jnp.pi * x) + 0.5 * jnp.sin(4 * jnp.pi * x)
    v0 = 0.3 * jnp.cos(2 * jnp.pi * x) - 0.1 * jnp.cos(6 * jnp.pi * x)

    # MSWPT + solver configuration
    dyadic = DyadicDecomposition(
        num_levels=2,
        N=domain.N,
        num_boxes_levels=(4, 8),
        box_aspect_ratio=(1,),
    )
    wpt = MSWPT(dyadic, redundancy=2, windowing="rectangular")
    sensors = Sensor(domain, binary_mask=jnp.ones(domain.N))

    # Use a fixed-size selection (`top_n`) to keep JIT shapes static.
    # Here we keep all coefficients to mimic the τ=0, no-threshold case.
    num_beams = int(wpt.total_coeffs)
    msgb_solver = MSGBSolver(
        thr=num_beams,
        thr_strat="top_n",
        batch_size=256,
        input_type="spatial",
        ode_solver=gb_solvers.solve_ODE_base,
        sum_method="all_real",
    )

    sensor_data, _ = msgb_solver.forward(p0, domain, sensors, ts, wpt, dpdt=v0)
    sensor_np = np.asarray(sensor_data.block_until_ready())

    # Spectral ground truth (periodic analytic solution)
    analytic = spectral_solution(p0, v0, c=1.0, ts=np.asarray(ts), dx=domain.dx[0])

    diff = analytic - sensor_np
    rel_l2 = np.linalg.norm(diff) / np.linalg.norm(analytic)
    rel_linf = np.max(np.abs(diff)) / np.max(np.abs(analytic))

    print(f"Grid N={domain.N}, Nt={ts.shape[0]}, dt={float(ts[1] - ts[0]):.3e}")
    print(f"Non-zero initial velocity: |v0|_inf={float(jnp.max(jnp.abs(v0))):.3f}")
    print(f"Relative L2 error vs spectral solution:  {rel_l2:.2e}")
    print(f"Relative Linf error vs spectral solution: {rel_linf:.2e}")

    # Basic plots: heatmaps + a time trace at x=0.4
    extent = [0.0, domain.grid_size[0], ts[0], ts[-1]]
    trace_x = 0.4 * domain.grid_size[0]
    ix = int(trace_x / domain.dx[0]) % domain.N[0]

    fig, axes = plt.subplots(
        2, 2, figsize=(10, 6), gridspec_kw={"height_ratios": [1.4, 1.0]}
    )
    ax_msgb, ax_true, ax_diff = axes[0, 0], axes[0, 1], axes[1, 0]
    ax_trace = axes[1, 1]

    im0 = ax_msgb.imshow(sensor_np, extent=extent, origin="lower", aspect="auto")
    ax_msgb.set_title("MSGB $u(t,x)$")
    ax_msgb.set_xlabel("x")
    ax_msgb.set_ylabel("t")
    fig.colorbar(im0, ax=ax_msgb, fraction=0.046, pad=0.04)

    im1 = ax_true.imshow(analytic, extent=extent, origin="lower", aspect="auto")
    ax_true.set_title("Spectral $u(t,x)$")
    ax_true.set_xlabel("x")
    ax_true.set_ylabel("t")
    fig.colorbar(im1, ax=ax_true, fraction=0.046, pad=0.04)

    im2 = ax_diff.imshow(
        diff, extent=extent, origin="lower", aspect="auto", cmap="RdBu_r"
    )
    ax_diff.set_title("Difference (spectral - MSGB)")
    ax_diff.set_xlabel("x")
    ax_diff.set_ylabel("t")
    fig.colorbar(im2, ax=ax_diff, fraction=0.046, pad=0.04)

    ax_trace.plot(ts, analytic[:, ix], label="Spectral")
    ax_trace.plot(ts, sensor_np[:, ix], "--", label="MSGB")
    ax_trace.set_title(f"Trace at x={trace_x:.3f}")
    ax_trace.set_xlabel("t")
    ax_trace.set_ylabel("u(t,x)")
    ax_trace.legend()
    ax_trace.grid(True, alpha=0.3)

    fig.tight_layout()
    outdir = Path(__file__).resolve().parents[2] / "plots"
    outdir.mkdir(exist_ok=True)
    outpath = outdir / "forward-1d-v0.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {outpath}")
    plt.close(fig)


if __name__ == "__main__":
    main()
