#!/usr/bin/env python
"""
2D photoacoustic forward comparison with MSGB, Hybrid, and k-Wave.

This public example ports the thesis two-packet homogeneous forward case to a
128 x 128 grid. It builds a high-frequency $p_0$, records sensor data on a
one-sided detector line, and saves thesis-style setup and sensor panels.

Example category: Forward propagation
Example extras: kwave,viz-mpl
Example smoke: false
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from beamax import plotter, transforms, utils
from beamax.decomposition import DyadicDecomposition
from beamax.gb import gb_solvers
from beamax.geometry import Domain, Sensor
from beamax.plotter import use_beamax_style
from beamax.solvers import HybridBackend, HybridSolver, MSGBSolver
from beamax.solvers.hybrid_solver import HybridSolverConfig
from beamax.transforms import MSWPT


INSTALL_HINT = 'pip install -e ".[kwave,viz-mpl]"'
N = (128, 128)
DX = (1.0e-4, 1.0e-4)
BOUNDS_FOR_LF_SOLVER = jnp.array([16, 75])
NUM_BEAMS = 4096

jax.config.update("jax_enable_x64", True)


def load_kwave_solver():
    """Import k-Wave lazily so base beamax installs can still import this file."""
    try:
        from beamax.solvers import KWaveSolver
    except ImportError as exc:
        print(f"Skipping optional example: k-Wave is not installed ({INSTALL_HINT}).")
        raise SystemExit(0) from exc
    return KWaveSolver


def c_homogeneous(x: jnp.ndarray) -> jnp.ndarray:
    """Homogeneous sound speed used by the thesis two-packet case."""
    return 1500.0 + 0.0 * x[..., 0]


def time_first(data: np.ndarray, nt: int) -> np.ndarray:
    """Return k-Wave sensor data as ``(Nt, Ns)``."""
    arr = np.asarray(data)
    if arr.ndim == 1:
        return arr[:, None] if arr.shape[0] == nt else arr[None, :]
    if arr.shape[0] == nt:
        return arr
    if arr.shape[-1] == nt:
        return np.moveaxis(arr, -1, 0).reshape(nt, -1)
    return arr.reshape(nt, -1)


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def relative_linf(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def make_wpts() -> tuple[DyadicDecomposition, MSWPT, MSWPT]:
    """Build the MSWPT pair used by the two-packet thesis example."""
    decomp = DyadicDecomposition(
        num_levels=3,
        N=N,
        num_boxes_levels=(4, 8, 16),
        box_aspect_ratio=(1, 1),
    )
    return (
        decomp,
        MSWPT(decomp, redundancy=2, windowing="rectangular_mirror"),
        MSWPT(decomp, redundancy=2, windowing="none"),
    )


def make_two_packet_p0(decomp: DyadicDecomposition) -> jnp.ndarray:
    """Build the thesis two-wave-packet $p_0$ from two MSWPT frame atoms."""
    grid = decomp.fourier_meshgrid
    high = transforms.compute_frames(
        decomp,
        125,
        jnp.array([11, 6]),
        grid,
        redundancy=2,
        windowing="none",
    )
    low = transforms.compute_frames(
        decomp,
        44,
        jnp.array([11, 3]),
        grid,
        redundancy=2,
        windowing="none",
    )

    p0 = utils.unitary_ifft(high) + utils.unitary_ifft(low)
    p0 = p0 / jnp.max(jnp.abs(p0))
    return p0.T.real


def top_k_reconstruction(
    coeffs: jnp.ndarray,
    inverse_wpt: MSWPT,
    k: int,
) -> jnp.ndarray:
    """Reconstruct from the top-``k`` coefficients for the setup residual panel."""
    k = min(k, int(coeffs.size))
    indices = jnp.argsort(jnp.abs(coeffs))[::-1][:k]
    selected = jnp.zeros_like(coeffs).at[indices].set(coeffs[indices])
    return inverse_wpt.inverse(selected, output_type="spatial").real


def crop_to_common_time_sensor(
    ts: jnp.ndarray,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Crop solver outputs to a common ``(Nt, Ns)`` shape."""
    nt = min(arr.shape[0] for arr in arrays)
    ns = min(arr.shape[1] for arr in arrays)
    return np.asarray(ts[:nt]), [np.asarray(arr[:nt, :ns]) for arr in arrays]


def colorbar(fig: plt.Figure, ax: plt.Axes, im, side: str = "right", size: str = "5%"):
    cax = make_axes_locatable(ax).append_axes(side, size=size, pad=0.1)
    cb = fig.colorbar(im, cax=cax)
    if side == "left":
        cax.yaxis.set_ticks_position("left")
        cax.yaxis.set_label_position("left")
    return cb


def plot_setup_panels(
    out_path: Path,
    c_grid: np.ndarray,
    p0: np.ndarray,
    coeffs_array: jnp.ndarray,
    p0_recon: np.ndarray,
    sensor_mask: np.ndarray,
    decomp: DyadicDecomposition,
) -> None:
    """Save the thesis-style setup panels for the forward experiment."""
    recon_diff = p0_recon - p0
    rd_max = float(np.max(np.abs(recon_diff)))
    rd_norm = mcolors.Normalize(vmin=-rd_max, vmax=rd_max) if rd_max > 0 else None

    fig = plt.figure(figsize=(13, 4))
    gs = fig.add_gridspec(nrows=1, ncols=4, wspace=0.5)

    ax_c = fig.add_subplot(gs[0, 0])
    im_c = ax_c.imshow(c_grid, origin="lower")
    ax_c.set_title(r"$c(\mathbf{x})$")
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    colorbar(fig, ax_c, im_c)

    ax_p0 = fig.add_subplot(gs[0, 1])
    im_p0 = ax_p0.imshow(p0, origin="lower")
    sensor_rows, sensor_cols = np.nonzero(sensor_mask)
    ax_p0.scatter(sensor_cols, sensor_rows, marker="^", color="r")
    ax_p0.set_title(r"$p_0$")
    ax_p0.set_xticks([])
    ax_p0.set_yticks([])
    colorbar(fig, ax_p0, im_p0)

    ax_coeff = fig.add_subplot(gs[0, 2])
    im_coeff = plotter.plot_mswpt_coeffs(
        ax_coeff,
        coeffs_array,
        decomp,
        cutoff_freq=None,
        box_corners=BOUNDS_FOR_LF_SOLVER,
        asymptote=False,
        log_scale=True,
    )
    ax_coeff.set_aspect("equal")
    ax_coeff.set_xticks([])
    ax_coeff.set_yticks([])
    colorbar(fig, ax_coeff, im_coeff)

    ax_diff = fig.add_subplot(gs[0, 3])
    im_diff = ax_diff.imshow(recon_diff, origin="lower", norm=rd_norm, cmap="RdBu_r")
    ax_diff.set_title(r"$p_0^{\mathrm{GB}} - p_0$")
    ax_diff.set_xticks([])
    ax_diff.set_yticks([])
    colorbar(fig, ax_diff, im_diff)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sensor_panels(
    out_path: Path,
    kwave: np.ndarray,
    msgb: np.ndarray,
    hybrid: np.ndarray,
    ts: np.ndarray,
    domain: Domain,
) -> None:
    """Save the thesis-style sensor-data comparison panels."""
    sensor_vals = [kwave, msgb, hybrid]
    sensor_norm = mcolors.Normalize(
        vmin=float(min(arr.min() for arr in sensor_vals)),
        vmax=float(max(arr.max() for arr in sensor_vals)),
    )

    diff_msgb = kwave - msgb
    diff_hybrid = kwave - hybrid
    diff_absmax = float(max(np.max(np.abs(diff_msgb)), np.max(np.abs(diff_hybrid))))
    diff_norm = mcolors.Normalize(vmin=-diff_absmax, vmax=diff_absmax)

    extent_sensor = [
        0.0,
        float(domain.N[1] * domain.dx[1]),
        float(np.max(ts)),
        float(np.min(ts)),
    ]

    fig = plt.figure(figsize=(9, 7))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=[1.0, 0.75],
        hspace=0.2,
        wspace=0.15,
    )

    ax_kw = fig.add_subplot(gs[0, 0])
    ax_msgb = fig.add_subplot(gs[0, 1])
    ax_hybrid = fig.add_subplot(gs[0, 2])

    im_kw = ax_kw.imshow(kwave, extent=extent_sensor, aspect="auto", norm=sensor_norm)
    ax_msgb.imshow(
        diff_msgb,
        extent=extent_sensor,
        aspect="auto",
        norm=diff_norm,
        cmap="RdBu_r",
    )
    im_hybrid = ax_hybrid.imshow(
        diff_hybrid,
        extent=extent_sensor,
        aspect="auto",
        norm=diff_norm,
        cmap="RdBu_r",
    )

    ax_kw.set_title(r"$g^{\mathrm{k\!-\!Wave}}$")
    ax_kw.set_xlabel(r"$x_s$")
    ax_kw.set_ylabel(r"$t$")
    ax_msgb.set_title(r"$g^{\mathrm{k\!-\!Wave}} - g^{\mathrm{MSGB}}$")
    ax_hybrid.set_title(r"$g^{\mathrm{k\!-\!Wave}} - g^{\mathrm{Hybrid}}$")
    for ax in (ax_kw, ax_msgb, ax_hybrid):
        ax.set_xticks([])
        ax.set_yticks([])

    colorbar(fig, ax_kw, im_kw, side="left", size="7%")
    colorbar(fig, ax_hybrid, im_hybrid, size="7%")

    _, nx = kwave.shape
    xs = np.linspace(extent_sensor[0], extent_sensor[1], nx)
    max_diff_per_sensor = np.max(np.abs(diff_msgb), axis=0)
    if nx > 2:
        sensor_idx = int(np.argmax(max_diff_per_sensor[1:-1]) + 1)
    else:
        sensor_idx = int(np.argmax(max_diff_per_sensor))
    sensor_idx = int(np.clip(sensor_idx, 0, nx - 1))
    profile_x = xs[sensor_idx] if nx > 0 else 0.0

    for ax in (ax_kw, ax_msgb, ax_hybrid):
        line = ax.axvline(profile_x, ls="--", lw=0.9, color="k", zorder=5)
        line.set_path_effects([pe.Stroke(linewidth=1.6, foreground="k"), pe.Normal()])

    ax_profile = fig.add_subplot(gs[1, :])
    ax_profile.plot(ts, kwave[:, sensor_idx], label="k-Wave", lw=1.5)
    ax_profile.plot(ts, msgb[:, sensor_idx], "--", label="MSGB", lw=1.5)
    ax_profile.plot(ts, hybrid[:, sensor_idx], "--", label="Hybrid", lw=1.5)
    ax_profile.set_xlabel(r"$t$")
    ax_profile.set_ylabel(r"$g(x_s,t)$")
    ax_profile.legend(frameon=False)
    ax_profile.grid(True, alpha=0.4)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_msgb_solver(num_beams: int) -> MSGBSolver:
    """Configure the homogeneous MSGB forward solver."""
    return MSGBSolver(
        thr=num_beams,
        thr_strat="top_n",
        batch_size=128,
        input_type="spatial",
        ode_solver=gb_solvers.solve_hom_diag,
        tr_ode_solver=gb_solvers.solve_hom_diag,
        sum_method="scan_real",
    )


def main() -> None:
    KWaveSolver = load_kwave_solver()
    use_beamax_style()

    cfl = float((jnp.sqrt(2.0) / 4.0).round(3))
    domain = Domain(
        N=N,
        dx=DX,
        c=c_homogeneous,
        cfl=cfl,
        periodic=(True, True),
    )
    ts = domain.generate_time_domain()

    decomp, wpt, wpt_none = make_wpts()
    p0 = make_two_packet_p0(decomp)
    coeffs = wpt.forward(p0, input_type="spatial")
    coeffs_array = wpt.convert_to_array(coeffs)
    p0_recon = top_k_reconstruction(coeffs, wpt_none, NUM_BEAMS)

    sensor_mask = jnp.zeros(N).at[0, :].set(1.0)
    sensors = Sensor(domain=domain, binary_mask=sensor_mask)

    kwave = KWaveSolver(
        backend="python",
        device="cpu",
        pml_size=8,
        smooth_p0=False,
        debug=False,
    )

    msgb_data = make_msgb_solver(NUM_BEAMS).forward(p0, domain, sensors, ts, wpt)
    msgb_data = np.asarray(msgb_data.block_until_ready()).real

    kwave_data = time_first(
        kwave.forward(p0, domain, sensor_mask, ts),
        nt=len(ts),
    )

    hybrid = HybridSolver(
        hf_solver=make_msgb_solver(NUM_BEAMS),
        lf_backend=HybridBackend.from_beamax_solver(kwave, name="k-Wave LF"),
        config=HybridSolverConfig(
            box_corners=BOUNDS_FOR_LF_SOLVER,
            downsample=False,
            use_time_extension=False,
            dt_oversample=0,
        ),
    )
    hybrid_data = np.asarray(hybrid.forward(p0, domain, sensors, ts, wpt)).real

    ts_np, (kwave_data, msgb_data, hybrid_data) = crop_to_common_time_sensor(
        ts,
        kwave_data,
        msgb_data,
        hybrid_data,
    )

    print(f"p0 shape: {p0.shape}")
    print(f"Sensor data shape: {msgb_data.shape}")
    print(f"MSGB relative L2 vs k-Wave:   {relative_l2(msgb_data, kwave_data):.3e}")
    print(f"MSGB relative Linf vs k-Wave: {relative_linf(msgb_data, kwave_data):.3e}")
    print(f"Hybrid relative L2 vs k-Wave:   {relative_l2(hybrid_data, kwave_data):.3e}")
    print(
        f"Hybrid relative Linf vs k-Wave: {relative_linf(hybrid_data, kwave_data):.3e}"
    )

    out_dir = utils.example_plot_dir(__file__)
    setup_path = out_dir / "2d_forward_setup_panels.png"
    sensor_path = out_dir / "2d_forward.png"

    plot_setup_panels(
        setup_path,
        np.asarray(domain.c_fn(domain.grid)),
        np.asarray(p0),
        coeffs_array,
        np.asarray(p0_recon),
        np.asarray(sensor_mask),
        decomp,
    )
    plot_sensor_panels(sensor_path, kwave_data, msgb_data, hybrid_data, ts_np, domain)

    print(f"Saved setup panels to {setup_path}")
    print(f"Saved sensor comparison to {sensor_path}")


if __name__ == "__main__":
    main()
