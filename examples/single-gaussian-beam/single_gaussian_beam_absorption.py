"""Compare lossless and absorbing Gaussian beams with k-Wave.

Example extras: kwave,viz-mpl
Example smoke: false
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from pathlib import Path

from beamax import geometry, utils
from beamax.gb import core, gb_solvers, gb_utils
from beamax.solvers import KWaveSolver

jax.config.update("jax_enable_x64", True)


def lam_to_alpha_db_per_cm(lam: float, c0: float) -> float:
    r"""Return $\alpha_{\mathrm{dB/cm}}=(\log_{10}e)\lambda/(10c_0)$."""
    return float(jnp.log10(jnp.e) / 5.0 * lam / c0 / 2.0)


def msgb_real_beam(domain, ts, lam: float) -> jnp.ndarray:
    """Evaluate a right/left mode pair so the recorded field is real-valued."""
    b, d = 1, 1
    x0 = jnp.array([[0.5 * domain.grid_size[0]]])
    p0 = jnp.ones((b, d))
    mode = jnp.ones((b,))
    a0 = jnp.ones((b,))
    omega0 = jnp.ones((b,)) * 100.0
    alpha0 = jnp.ones((b, d)) * 1j
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
    solver = KWaveSolver(
        backend="python",
        device="cpu",
        smooth_p0=False,
        debug=False,
        quiet=True,
    )
    p0_2d = p0_1d[:, None]
    return np.asarray(solver.forward(p0_2d, kw_domain, binary_mask, ts))


def main() -> None:
    n = 512
    cfl = 0.3
    lam = 5.0

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

    u_loss = np.asarray(msgb_real_beam(domain, ts, lam=0.0))
    u_abs = np.asarray(msgb_real_beam(domain, ts, lam=lam))

    # Use the same initial field in both solvers.
    alpha_db = lam_to_alpha_db_per_cm(lam, c0)
    p0_init = jnp.asarray(u_loss[0])
    k_loss = kwave_run(
        p0_init,
        ts,
        c0=c0,
        alpha_coeff=0.0,
        cfl=cfl,
    ).reshape(len(ts), n)
    k_abs = kwave_run(
        p0_init,
        ts,
        c0=c0,
        alpha_coeff=alpha_db,
        cfl=cfl,
    ).reshape(len(ts), n)

    e2_loss = utils.rel_l2(k_loss, u_loss)
    e2_abs = utils.rel_l2(k_abs, u_abs)
    print(f"Damping coefficient lam = {lam}, alpha_coeff = {alpha_db:.4f} dB/cm")
    print(f"Lossless  rel-L2 (MSGB vs k-Wave): {e2_loss:.3e}")
    print(f"Absorbing rel-L2 (MSGB vs k-Wave): {e2_abs:.3e}")

    extent = [0.0, 1.0, float(ts[0]), float(ts[-1])]
    norm_abs = Normalize(
        vmin=min(k_abs.min(), u_abs.min()), vmax=max(k_abs.max(), u_abs.max())
    )
    diff = k_abs - u_abs
    m = float(np.max(np.abs(diff)))
    diff_norm = TwoSlopeNorm(vcenter=0.0, vmin=-m, vmax=m)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)

    spacetime_panels = (
        (k_abs, norm_abs, "viridis", "k-Wave (absorbing)"),
        (u_abs, norm_abs, "viridis", "MSGB (absorbing)"),
        (diff, diff_norm, "RdBu_r", "k-Wave − MSGB (absorbing)"),
    )
    for ax, (image, norm, cmap, title) in zip(axes[0], spacetime_panels):
        im = ax.imshow(
            image,
            extent=extent,
            origin="lower",
            aspect="auto",
            norm=norm,
            cmap=cmap,
        )
        ax.set(title=title, xlabel="x", ylabel="t")
        fig.colorbar(im, ax=ax)

    t = np.asarray(ts)
    axes[1, 0].plot(t, u_loss.max(axis=1), label="MSGB", color="C0")
    axes[1, 0].plot(t, k_loss.max(axis=1), "--", label="k-Wave", color="C3")
    axes[1, 0].set(title=r"$\max_x |u(x,t)|$ — lossless", xlabel="t")
    axes[1, 0].legend()

    axes[1, 1].plot(t, u_abs.max(axis=1), label="MSGB", color="C0")
    axes[1, 1].plot(t, k_abs.max(axis=1), "--", label="k-Wave", color="C3")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(title=r"$\max_x |u(x,t)|$ — absorbing (log)", xlabel="t")
    axes[1, 1].legend()

    x = np.asarray(domain.grid).reshape(-1)
    axes[1, 2].plot(x, u_loss[0], label="initial $p_0$", color="black")
    axes[1, 2].plot(x, u_loss[-1], "--", label="lossless, final", color="C3")
    axes[1, 2].plot(x, u_abs[-1], label="absorbing, final", color="C0")
    axes[1, 2].set(title="snapshots", xlabel="x")
    axes[1, 2].legend()

    out_dir = Path("plots/single-gaussian-beam")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "single_gaussian_beam_absorption.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
