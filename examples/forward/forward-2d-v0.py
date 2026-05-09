#!/usr/bin/env python
"""
2D forward solve with non-zero initial velocity, compared to a spectral ground truth.

Periodic box, constant sound speed. Initial data: sum of sines/cosines in x and y.
We keep a capped number of beams (top_n) for stability and plot snapshots + error maps.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from beamax.decomposition import DyadicDecomposition
from beamax.transforms import MSWPT
from beamax.geometry import Domain, Sensor
from beamax.solvers import MSGBSolver
from beamax.gb import gb_solvers


jax.config.update("jax_enable_x64", True)


def spectral_solution_2d(
    p0: jnp.ndarray, v0: jnp.ndarray, c: float, ts: np.ndarray, dx: tuple[float, float]
) -> np.ndarray:
    """
    Exact solution of utt - c^2 Δu = 0 on a 2D periodic box via Fourier series.
    """
    p0_np = np.asarray(p0)
    v0_np = np.asarray(v0)
    t = np.asarray(ts)[:, None, None]

    kx = 2 * np.pi * np.fft.fftfreq(p0_np.shape[0], d=float(dx[0]))
    ky = 2 * np.pi * np.fft.fftfreq(p0_np.shape[1], d=float(dx[1]))
    kxg, kyg = np.meshgrid(kx, ky, indexing="ij")
    omega = c * np.sqrt(kxg**2 + kyg**2)[None, ...]

    p_hat = np.fft.fftn(p0_np)
    v_hat = np.fft.fftn(v0_np)

    sin_term = np.sin(omega * t)
    omega_safe = np.where(omega == 0.0, 1.0, omega)
    sin_over_omega = sin_term / omega_safe
    sin_over_omega = np.where(omega == 0.0, t, sin_over_omega)

    cos_term = np.cos(omega * t)
    u_hat = p_hat[None, ...] * cos_term + v_hat[None, ...] * sin_over_omega
    return np.fft.ifftn(u_hat, axes=(1, 2)).real


def main():
    full_run = os.environ.get("BEAMAX_FULL_EXAMPLES", "0") == "1"

    # Problem setup (2D, periodic)
    N = (128, 128) if full_run else (64, 64)
    dx = (1.0 / N[0], 1.0 / N[1])
    domain = Domain(N=N, dx=dx, c=1.0, periodic=(True, True))
    ts = domain.generate_time_domain()
    X = jnp.meshgrid(
        jnp.arange(N[0]) * dx[0],
        jnp.arange(N[1]) * dx[1],
        indexing="ij",
    )
    x, y = X

    # Initial displacement and velocity
    p0 = jnp.sin(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y) + 0.3 * jnp.sin(
        4 * jnp.pi * x + 1.1
    )
    v0 = 0.25 * jnp.cos(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y) - 0.15 * jnp.cos(
        3 * jnp.pi * x - 0.7
    )

    # MSWPT + solver configuration
    dyadic = DyadicDecomposition(
        num_levels=2,
        N=domain.N,
        num_boxes_levels=(4, 8),
        box_aspect_ratio=(1, 1),
    )
    wpt = MSWPT(dyadic, redundancy=2, windowing="rectangular")
    sensors = Sensor(domain, binary_mask=jnp.ones(domain.N))

    total_coeffs = int(wpt.total_coeffs)
    thr_cap = 6000 if full_run else 2000
    thr = min(total_coeffs, thr_cap)  # cap to keep runtime/memory reasonable
    msgb_solver = MSGBSolver(
        thr=thr,
        thr_strat="top_n",
        batch_size=256,
        input_type="spatial",
        ode_solver=gb_solvers.solve_ODE_base,
        sum_method="scan_real",
    )

    sensor_data, _ = msgb_solver.forward(p0, domain, sensors, ts, wpt, dpdt=v0)
    sensor_np = np.asarray(sensor_data.block_until_ready())
    if sensor_np.ndim == 2 and sensor_np.shape[1] == N[0] * N[1]:
        sensor_np = sensor_np.reshape(sensor_np.shape[0], N[0], N[1])

    # Spectral ground truth
    analytic = spectral_solution_2d(p0, v0, c=1.0, ts=np.asarray(ts), dx=dx)

    diff = analytic - sensor_np
    rel_l2 = np.linalg.norm(diff) / np.linalg.norm(analytic)
    rel_linf = np.max(np.abs(diff)) / np.max(np.abs(analytic))

    print(f"Grid N={domain.N}, Nt={ts.shape[0]}, dt={float(ts[1] - ts[0]):.3e}")
    print(f"Non-zero initial velocity: |v0|_inf={float(jnp.max(jnp.abs(v0))):.3f}")
    print(f"Relative L2 error vs spectral solution:  {rel_l2:.2e}")
    print(f"Relative Linf error vs spectral solution: {rel_linf:.2e}")

    # Plots: snapshots and a time trace at (x,y) = (0.4 Lx, 0.3 Ly)
    outdir = Path(__file__).resolve().parents[2] / "plots"
    outdir.mkdir(exist_ok=True)
    outpath = outdir / "forward-2d-v0.png"

    snap_indices = [
        0,
        ts.shape[0] // 3,
        2 * ts.shape[0] // 3,
        ts.shape[0] - 1,
    ]
    trace_x = 0.4 * domain.grid_size[0]
    trace_y = 0.3 * domain.grid_size[1]
    ix = int(trace_x / dx[0]) % N[0]
    iy = int(trace_y / dx[1]) % N[1]

    fig, axes = plt.subplots(3, len(snap_indices), figsize=(12, 8))
    for col, ti in enumerate(snap_indices):
        im0 = axes[0, col].imshow(
            sensor_np[ti],
            origin="lower",
            extent=[0, domain.grid_size[1], 0, domain.grid_size[0]],
            aspect="auto",
        )
        axes[0, col].set_title(f"MSGB t={ts[ti]:.3f}")
        fig.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)

        im1 = axes[1, col].imshow(
            analytic[ti],
            origin="lower",
            extent=[0, domain.grid_size[1], 0, domain.grid_size[0]],
            aspect="auto",
        )
        axes[1, col].set_title(f"Spectral t={ts[ti]:.3f}")
        fig.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)

        im2 = axes[2, col].imshow(
            diff[ti],
            origin="lower",
            extent=[0, domain.grid_size[1], 0, domain.grid_size[0]],
            aspect="auto",
            cmap="RdBu_r",
        )
        axes[2, col].set_title("Diff")
        fig.colorbar(im2, ax=axes[2, col], fraction=0.046, pad=0.04)

        for row in range(3):
            axes[row, col].set_xlabel("y")
            axes[row, col].set_ylabel("x")

    fig.suptitle("MSGB vs Spectral (2D, v0 ≠ 0)", y=0.995)
    fig.tight_layout()

    # Time trace
    fig2, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(ts, analytic[:, ix, iy], label="Spectral")
    ax.plot(ts, sensor_np[:, ix, iy], "--", label="MSGB")
    ax.set_title(f"Trace at x={trace_x:.3f}, y={trace_y:.3f}")
    ax.set_xlabel("t")
    ax.set_ylabel("u(t,x,y)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    fig2.savefig(outdir / "forward-2d-v0-trace.png", dpi=200, bbox_inches="tight")
    print(f"Saved plots to {outpath} and {outdir / 'forward-2d-v0-trace.png'}")
    plt.close(fig)
    plt.close(fig2)


if __name__ == "__main__":
    main()
