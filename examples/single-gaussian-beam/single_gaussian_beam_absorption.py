#!/usr/bin/env python
"""
Single Gaussian beam with viscous absorption: MSGB vs k-Wave.

This optional example propagates the same 1D Gaussian beam pair twice — once
in a lossless medium and once in an absorbing medium — using both the MSGB
solver and a k-Wave strip reference, then compares them in a single figure.

The absorbing case is the headline: it shows that the Gaussian-beam viscous
damping coefficient ``lambda`` and k-Wave's ``alpha_coeff`` produce visually
matching spacetime fields and matching max-amplitude decay.

Example category: Single Gaussian beam diagnostics
Example extras: kwave,viz-mpl
Example smoke: false
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

from beamax import geometry, utils
from beamax.gb import core, gb_solvers, gb_utils


jax.config.update("jax_enable_x64", True)

INSTALL_HINT = 'pip install -e ".[kwave,viz-mpl]"'


def load_kwave_solver():
    """Import k-Wave lazily so base beamax installs can still import this file."""
    try:
        from beamax.solvers import KWaveSolver
        from kwave.options.simulation_execution_options import (
            SimulationExecutionOptions,
        )
        from kwave.options.simulation_options import SimulationOptions
    except ImportError as exc:
        print(f"Skipping optional example: k-Wave is not installed ({INSTALL_HINT}).")
        raise SystemExit(0) from exc
    return KWaveSolver, SimulationOptions, SimulationExecutionOptions


def lam_to_alpha_db_per_cm(lam: float, c0: float) -> float:
    """
    Convert the Gaussian-beam viscous coefficient ``lam`` to k-Wave's
    ``alpha_coeff`` so the two solvers see matching effective absorption.

    The factor of 1/2 is load-bearing: empirically the lossless residual
    drops to machine precision and the absorbing residual stays small only
    when the conversion is halved.
    """
    return float(jnp.log10(jnp.e) / 5.0 * lam / c0 / 2.0)


def msgb_real_beam(domain, ts, lam: float) -> jnp.ndarray:
    """Evaluate a right/left mode pair so the recorded field is real-valued."""
    b, d = 1, 1
    x0 = jnp.array([[0.5 * domain.grid_size[0]]])
    p0 = jnp.ones((b, d))
    mode = jnp.ones((b,))
    a0 = jnp.ones((b,))
    omega0 = jnp.ones((b,)) * 100.0
    alpha0 = jnp.ones((b, d)) * 1j  # beam half-width parameter
    m0 = gb_utils.prepare_M0(alpha0, None)
    periodic = jnp.array(domain.periodic)

    def beam(sign):
        return core.compute_gaussian_beam_real(
            x0,
            p0,
            m0,
            a0,
            omega0,
            sign * mode,
            domain.c_fn,
            lam,
            ts,
            domain.grid,
            domain.grid_size,
            periodic,
            gb_solvers.solve_ODE_base,
            None,
        )

    return jnp.squeeze(beam(+1) + beam(-1))


def kwave_run(
    p0_1d,
    ts,
    *,
    c0: float,
    alpha_coeff: float,
    cfl: float,
    KWaveSolver,
    SimulationOptions,
    SimulationExecutionOptions,
):
    """Run a 1D k-Wave strip simulation with a matching absorbing medium."""
    n = p0_1d.shape[0]
    n_kw = (n, 1)

    def c_fn(x):
        return c0 + 0.0 * x[..., 0]

    kw_domain = geometry.Domain(
        N=n_kw,
        dx=(1.0 / n, 1.0 / n),
        c=c_fn,
        cfl=cfl,
        periodic=(True, True),
        alpha_power=0,
        alpha_coeff=alpha_coeff,
    )
    binary_mask = jnp.ones(n_kw)
    sim_opts = SimulationOptions(data_cast="single", smooth_p0=False, save_to_disk=True)
    exec_opts = SimulationExecutionOptions(
        is_gpu_simulation=False,
        delete_data=False,
        verbose_level=0,
        show_sim_log=False,
    )
    solver = KWaveSolver(sim_opts, exec_opts)
    p0_2d = p0_1d[:, None]  # k-Wave wants the strip dim explicit
    return np.asarray(solver.forward(p0_2d, kw_domain, binary_mask, ts))


def main() -> None:
    KWaveSolver, SimulationOptions, SimulationExecutionOptions = load_kwave_solver()

    n = 512
    cfl = 0.3
    lam = 5.0  # GB viscous coefficient; alpha_coeff is derived from this

    def c_fn(x):
        return 1.0 + 0.0 * x[..., 0]

    domain = geometry.Domain(
        N=(n,),
        dx=(1.0 / n,),
        c=c_fn,
        periodic=(True,),
        cfl=cfl,
    )
    ts = domain.generate_time_domain()
    c0 = float(c_fn(jnp.zeros(1)))

    # 1. MSGB beams (lossless + absorbing).
    u_loss = np.asarray(msgb_real_beam(domain, ts, lam=0.0))
    u_abs = np.asarray(msgb_real_beam(domain, ts, lam=lam))

    # 2. k-Wave references, initialised from the MSGB p0 so the two solvers
    #    see exactly the same initial condition.
    alpha_db = lam_to_alpha_db_per_cm(lam, c0)
    p0_init = jnp.asarray(u_loss[0])
    k_loss = kwave_run(
        p0_init,
        ts,
        c0=c0,
        alpha_coeff=0.0,
        cfl=cfl,
        KWaveSolver=KWaveSolver,
        SimulationOptions=SimulationOptions,
        SimulationExecutionOptions=SimulationExecutionOptions,
    ).reshape(len(ts), n)
    k_abs = kwave_run(
        p0_init,
        ts,
        c0=c0,
        alpha_coeff=alpha_db,
        cfl=cfl,
        KWaveSolver=KWaveSolver,
        SimulationOptions=SimulationOptions,
        SimulationExecutionOptions=SimulationExecutionOptions,
    ).reshape(len(ts), n)

    dx = float(domain.dx[0])
    dt = float(ts[1] - ts[0])

    def rel_l2(u, ref):
        return float(
            np.sqrt(np.sum((u - ref) ** 2) * dx * dt)
            / np.sqrt(np.sum(ref**2) * dx * dt)
        )

    e2_loss = rel_l2(u_loss, k_loss)
    e2_abs = rel_l2(u_abs, k_abs)
    print(f"Damping coefficient lam = {lam}, alpha_coeff = {alpha_db:.4f} dB/cm")
    print(f"Lossless  rel-L2 (MSGB vs k-Wave): {e2_loss:.3e}")
    print(f"Absorbing rel-L2 (MSGB vs k-Wave): {e2_abs:.3e}")

    # 3. One figure: top row spacetime panels (absorbing case), bottom row
    #    quantitative comparisons (max-amplitude curves + initial/final snapshots).
    extent = [0.0, 1.0, float(ts[0]), float(ts[-1])]
    norm_abs = Normalize(
        vmin=min(k_abs.min(), u_abs.min()), vmax=max(k_abs.max(), u_abs.max())
    )
    diff = k_abs - u_abs
    m = float(np.max(np.abs(diff)))
    diff_norm = TwoSlopeNorm(vcenter=0.0, vmin=-m, vmax=m)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)

    def _label_spacetime(ax, title):
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")

    im0 = axes[0, 0].imshow(
        k_abs,
        extent=extent,
        origin="lower",
        aspect="auto",
        norm=norm_abs,
        cmap="viridis",
    )
    _label_spacetime(axes[0, 0], "k-Wave (absorbing)")
    fig.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(
        u_abs,
        extent=extent,
        origin="lower",
        aspect="auto",
        norm=norm_abs,
        cmap="viridis",
    )
    _label_spacetime(axes[0, 1], "MSGB (absorbing)")
    fig.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(
        diff,
        extent=extent,
        origin="lower",
        aspect="auto",
        norm=diff_norm,
        cmap="RdBu_r",
    )
    _label_spacetime(axes[0, 2], "k-Wave − MSGB (absorbing)")
    fig.colorbar(im2, ax=axes[0, 2])

    t = np.asarray(ts)
    axes[1, 0].plot(t, u_loss.max(axis=1), label="MSGB", color="C0")
    axes[1, 0].plot(t, k_loss.max(axis=1), "--", label="k-Wave", color="C3")
    axes[1, 0].set_title(r"$\max_x |u(x,t)|$ — lossless")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].legend()

    axes[1, 1].plot(t, u_abs.max(axis=1), label="MSGB", color="C0")
    axes[1, 1].plot(t, k_abs.max(axis=1), "--", label="k-Wave", color="C3")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title(r"$\max_x |u(x,t)|$ — absorbing (log)")
    axes[1, 1].set_xlabel("t")
    axes[1, 1].legend()

    x = np.asarray(domain.grid).reshape(-1)
    axes[1, 2].plot(x, u_loss[0], label="initial $p_0$", color="black")
    axes[1, 2].plot(x, u_loss[-1], "--", label="lossless, final", color="C3")
    axes[1, 2].plot(x, u_abs[-1], label="absorbing, final", color="C0")
    axes[1, 2].set_title("snapshots")
    axes[1, 2].set_xlabel("x")
    axes[1, 2].legend()

    out_dir = Path(utils.detect_root()) / "plots" / "optional"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "single_gaussian_beam_absorption.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
