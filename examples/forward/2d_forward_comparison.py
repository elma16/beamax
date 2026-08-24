r"""Compare MSGB and k-Wave sensor data for a 2D wave packet.

Example extras: kwave,viz-mpl
Example smoke: false
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from beamax import Domain, DyadicDecomposition, MSWPT, Sensor, utils
from beamax.gb import gb_solvers
from beamax.plotter import use_beamax_style
from beamax.solvers import KWaveSolver, MSGBSolver

jax.config.update("jax_enable_x64", True)


def wave_packet(domain: Domain) -> jnp.ndarray:
    r"""Return a Gaussian wave packet with phase $18\pi(x/L_x+y/L_y)$."""
    x, y = domain.generate_meshgrid()[0]
    lx, ly = domain.grid_size
    xi = x / lx - 0.42
    eta = y / ly - 0.42
    envelope = jnp.exp(-(xi**2 + eta**2) / (2.0 * 0.12**2))
    p0 = envelope * jnp.cos(18.0 * jnp.pi * (xi + eta))
    return p0 / jnp.max(jnp.abs(p0))


def plot_comparison(
    p0, kwave_data, msgb_data, domain, ts, sensor_index, out_path
) -> None:
    residual = msgb_data - kwave_data
    record_limit = float(max(np.max(np.abs(kwave_data)), np.max(np.abs(msgb_data))))
    residual_limit = float(np.max(np.abs(residual)))
    lx, ly = np.asarray(domain.grid_size)
    time_extent = [0.0, ly * 1e3, float(ts[0]) * 1e6, float(ts[-1]) * 1e6]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), constrained_layout=True)
    panels = (
        (
            axes[0],
            np.asarray(p0).T,
            [0.0, lx * 1e3, 0.0, ly * 1e3],
            1.0,
            r"initial $p_0$",
        ),
        (axes[1], kwave_data, time_extent, record_limit, "k-Wave"),
        (axes[2], msgb_data, time_extent, record_limit, "MSGB"),
        (axes[3], residual, time_extent, residual_limit, "MSGB − k-Wave"),
    )
    for ax, image, extent, limit, title in panels:
        im = ax.imshow(
            image,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)

    axes[0].axvline(sensor_index * domain.dx[0] * 1e3, color="black", ls="--")
    axes[0].set(xlabel="$x$ [mm]", ylabel="$y$ [mm]")
    for ax in axes[1:]:
        ax.set(xlabel="sensor position [mm]", ylabel=r"$t$ [$\mu$s]")

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    use_beamax_style()

    shape = (48, 48)
    domain = Domain(
        N=shape,
        dx=(1.0e-4, 1.0e-4),
        c=1500.0,
        cfl=0.3,
        periodic=(False, False),
    )
    ts = domain.generate_time_domain()
    p0 = wave_packet(domain)

    sensor_index = 5
    sensor_mask = jnp.zeros(shape).at[sensor_index, :].set(1.0)
    sensors = Sensor(domain=domain, binary_mask=sensor_mask)

    decomposition = DyadicDecomposition(
        num_levels=2,
        N=shape,
        num_boxes_levels=(4, 8),
        box_aspect_ratio=(1, 1),
    )
    wpt = MSWPT(decomposition, redundancy=2, windowing="rectangular_mirror")
    msgb = MSGBSolver(
        thr=512,
        thr_strat="top_n",
        batch_size=64,
        input_type="spatial",
        ode_solver=gb_solvers.solve_hom_diag,
        sum_method="scan_real",
    )
    kwave = KWaveSolver(
        backend="python",
        device="cpu",
        pml_inside=False,
        pml_size=6,
        smooth_p0=False,
        quiet=True,
    )

    msgb_data = np.asarray(msgb.forward(p0, domain, sensors, ts, wpt))
    kwave_data = np.asarray(kwave.forward(p0, domain, sensor_mask, ts))
    error = utils.rel_l2(kwave_data, msgb_data)

    plot_dir = Path("plots/forward")
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / "2d_forward_comparison.png"
    plot_comparison(
        p0,
        kwave_data,
        msgb_data,
        domain,
        ts,
        sensor_index,
        out_path,
    )
    print(f"Sensor data shape: {msgb_data.shape}")
    print(f"Relative L2 error (MSGB vs k-Wave): {error:.3e}")
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
