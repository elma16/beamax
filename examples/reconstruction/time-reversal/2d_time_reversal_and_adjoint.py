#!/usr/bin/env python
"""
2D MSGB vs k-Wave reconstruction: time reversal + adjoint.

Distills the thesis ``inverse_2d_full`` figure to one small 2D problem. Steps:

  1. Build a smooth two-Gaussian phantom and a one-sided boundary sensor line.
  2. Forward-simulate with k-Wave to get the sensor record.
  3. Reconstruct with both k-Wave and MSGB via time reversal AND adjoint
     back-propagation (four reconstructions in total).
  4. Plot the thesis-style 3-row figure: phantom + 2 TR images on top,
     blank + 2 adjoint images in the middle, 1D profile through them on the
     bottom. Print relative-L2 metrics against the truth.

MSGB time-reversal in 2D needs a frequency-cropped data domain and a paired
data-WPT — the helper ``prepare_data_domain_for_msgb`` below mirrors what the
thesis script does in ~30 lines.

Example category: Time-reversal reconstruction
Example extras: kwave,viz-mpl
Example smoke: false
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from beamax import utils
from beamax.decomposition import DyadicDecomposition
from beamax.geometry import Domain, Sensor
from beamax.gb import gb_solvers
from beamax.solvers import MSGBSolver
from beamax.transforms import MSWPT


jax.config.update("jax_enable_x64", True)

INSTALL_HINT = 'pip install -e ".[kwave,viz-mpl]"'


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def load_kwave_solver():
    """Import k-Wave lazily so base beamax installs can still import this file."""
    try:
        from beamax.solvers import KWaveSolver
    except ImportError as exc:
        print(f"Skipping optional example: k-Wave is not installed ({INSTALL_HINT}).")
        raise SystemExit(0) from exc
    return KWaveSolver


def c_homogeneous(x: jnp.ndarray) -> jnp.ndarray:
    return 1500.0 + 0.0 * x[..., 0]


def make_two_gaussian_phantom(domain: Domain) -> jnp.ndarray:
    """Two smooth Gaussian inclusions with zero mean, normalised to peak |p| = 1."""
    lx, ly = domain.grid_size
    x, y = jnp.meshgrid(
        jnp.arange(domain.N[0]) * domain.dx[0],
        jnp.arange(domain.N[1]) * domain.dx[1],
        indexing="ij",
    )
    p0 = jnp.exp(
        -((x - 0.38 * lx) ** 2 + (y - 0.45 * ly) ** 2) / (2.0 * (0.08 * lx) ** 2)
    )
    p0 -= 0.7 * jnp.exp(
        -((x - 0.62 * lx) ** 2 + (y - 0.58 * ly) ** 2) / (2.0 * (0.09 * lx) ** 2)
    )
    p0 = p0 - jnp.mean(p0)
    return p0 / jnp.max(jnp.abs(p0))


def coerce_image(arr: jnp.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Coerce k-Wave image output to ``shape``; handle the transposed-output case."""
    image = np.asarray(arr)
    if image.shape == shape:
        return image
    if image.T.shape == shape:
        return image.T
    return image.reshape(shape)


def scaled(recon: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, float]:
    """Best L2 scale of recon onto truth; returns (scaled, rel_l2)."""
    r = np.asarray(recon).real
    t = np.asarray(truth).real
    s = float(np.vdot(r, t) / (np.vdot(r, r) + 1e-30))
    out = s * r
    rel_l2 = float(np.linalg.norm(out - t) / (np.linalg.norm(t) + 1e-30))
    return out, rel_l2


# ---------------------------------------------------------------------------
# MSGB data-domain construction (thesis: prepare_sensor_data_for_tr)
# ---------------------------------------------------------------------------


def _cut_out_middle(arr: jnp.ndarray, size: int) -> jnp.ndarray:
    """Keep the middle ``size`` samples along axis 0 (after fftshift)."""
    mid = arr.shape[0] // 2
    return arr[mid - size // 2 : mid + size // 2]


def prepare_data_domain_for_msgb(
    sensor_data_kw: jnp.ndarray,
    domain: Domain,
    ts: jnp.ndarray,
    *,
    over_resolve: int = 2,
):
    """
    Build the (Nt', Ns) data domain and its paired MSWPT that MSGB needs for
    TR/adjoint, by Fourier-cropping the k-Wave sensor record in time.

    Returns
    -------
    sensor_data_cropped : (Nt', Ns)
    domain_data : Domain on (Nt', Ns) with dx = (dt', dx_y)
    wpt_data : MSWPT on the data domain
    ts_data : (Nt',) new time grid
    """
    sensor_arr = jnp.asarray(sensor_data_kw)
    if sensor_arr.ndim != 2:
        raise ValueError(f"Expected (Nt, Ns) sensor data; got {sensor_arr.shape}")

    nt_cropped = over_resolve * domain.N[0]
    if sensor_arr.shape[0] < nt_cropped:
        raise ValueError(
            f"Need >= {nt_cropped} time samples; got {sensor_arr.shape[0]}."
        )
    fft = utils.unitary_fft(sensor_arr)
    cropped_fft = _cut_out_middle(fft, nt_cropped)
    sensor_data_cropped = utils.unitary_ifft(cropped_fft).real

    nt_data, ns = sensor_data_cropped.shape
    ts_data = jnp.linspace(float(ts[0]), float(ts[-1]), nt_data)
    dt_data = float(ts_data[1] - ts_data[0])
    dx_y = float(domain.dx[1])
    domain_data = Domain(
        N=(nt_data, ns),
        dx=(dt_data, dx_y),
        c=domain.c,
        periodic=domain.periodic,
        cfl=domain.cfl,
    )

    # Aspect ratio set so the dyadic decomposition matches the rectangular data
    # shape (nt_data is typically over_resolve * ns for over_resolve == 2).
    n_min = min(nt_data, ns)
    box_aspect = (nt_data // n_min, ns // n_min)
    dyadic_data = DyadicDecomposition(
        num_levels=2,
        N=(nt_data, ns),
        num_boxes_levels=(4, 8),
        box_aspect_ratio=box_aspect,
    )
    wpt_data = MSWPT(dyadic_data, redundancy=2, windowing="rectangular_mirror")
    return sensor_data_cropped, domain_data, wpt_data, ts_data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_comparison(
    p0,
    tr_kw,
    tr_msgb,
    adj_kw,
    adj_msgb,
    domain,
    sensors,
    *,
    out_path,
):
    """Thesis-style 3-row layout: 2 imshow rows + 1 profile row."""
    arrays = [np.asarray(a).real for a in (p0, tr_kw, tr_msgb, adj_kw, adj_msgb)]
    vmax = max(float(np.max(np.abs(a))) for a in arrays)
    extent = (0.0, float(domain.grid_size[1]), 0.0, float(domain.grid_size[0]))

    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(
        3,
        3,
        height_ratios=[1.0, 1.0, 0.85],
        hspace=0.25,
        wspace=0.08,
        figure=fig,
    )

    top_titles = [
        r"$p_0$",
        r"$p_{\rm TR}^{\rm k\text{-}Wave}$",
        r"$p_{\rm TR}^{\rm MSGB}$",
    ]
    mid_titles = [
        None,
        r"$p_{\rm Adj}^{\rm k\text{-}Wave}$",
        r"$p_{\rm Adj}^{\rm MSGB}$",
    ]
    top_arrays = [arrays[0], arrays[1], arrays[2]]
    mid_arrays = [None, arrays[3], arrays[4]]
    image_axes = []

    for j, (title, arr) in enumerate(zip(top_titles, top_arrays)):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(
            arr,
            origin="lower",
            extent=extent,
            vmin=-vmax,
            vmax=vmax,
            cmap="RdBu_r",
            aspect="equal",
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        image_axes.append(ax)

    for j, (title, arr) in enumerate(zip(mid_titles, mid_arrays)):
        ax = fig.add_subplot(gs[1, j])
        if arr is None:
            ax.axis("off")
        else:
            ax.imshow(
                arr,
                origin="lower",
                extent=extent,
                vmin=-vmax,
                vmax=vmax,
                cmap="RdBu_r",
                aspect="equal",
            )
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            image_axes.append(ax)

    # Overlay sensor positions on every image panel.
    rr, cc = jnp.where(sensors.binary_mask)
    xs = (np.asarray(cc) + 0.5) * float(domain.dx[1])
    ys = (np.asarray(rr) + 0.5) * float(domain.dx[0])
    for ax in image_axes:
        ax.scatter(xs, ys, s=10, c="black", marker="^", alpha=0.85, zorder=10)

    # 1D profile down the middle column of the image.
    ax_prof = fig.add_subplot(gs[2, :])
    idx = arrays[0].shape[1] // 2
    y_axis = np.arange(arrays[0].shape[0]) * float(domain.dx[0])
    ax_prof.plot(y_axis, arrays[0][:, idx], color="black", lw=2.0, label=r"$p_0$")
    ax_prof.plot(y_axis, arrays[1][:, idx], color="C0", lw=1.5, label="TR k-Wave")
    ax_prof.plot(
        y_axis, arrays[2][:, idx], color="C0", lw=1.5, ls="--", label="TR MSGB"
    )
    ax_prof.plot(y_axis, arrays[3][:, idx], color="C3", lw=1.5, label="Adj k-Wave")
    ax_prof.plot(
        y_axis, arrays[4][:, idx], color="C3", lw=1.5, ls="--", label="Adj MSGB"
    )
    ax_prof.set_xlabel("y [m]")
    ax_prof.set_ylabel("pressure")
    ax_prof.set_title(f"profile at x = {idx * float(domain.dx[1]):.1e} m")
    ax_prof.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=5)
    ax_prof.axvline(idx * float(domain.dx[0]), color="grey", ls=":", lw=0.8)
    for ax in image_axes:
        ax.axvline(idx * float(domain.dx[1]), color="grey", ls=":", lw=0.8)

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    KWaveSolver = load_kwave_solver()

    n = (64, 64)
    dx = (1.0e-4, 1.0e-4)
    domain = Domain(
        N=n,
        dx=dx,
        c=c_homogeneous,
        cfl=0.3,
        periodic=(False, False),
    )
    ts = domain.generate_time_domain()
    p0 = make_two_gaussian_phantom(domain)

    # Boundary sensors on the x = 0 row.
    sensor_mask = jnp.zeros(n).at[0, :].set(1.0)
    sensors = Sensor(domain=domain, binary_mask=sensor_mask)
    image_mask = jnp.ones(n)

    # --- k-Wave forward to generate sensor data ---
    kwave = KWaveSolver(
        backend="python",
        device="cpu",
        pml_size=8,
        smooth_p0=False,
        debug=False,
    )
    data = kwave.forward(p0, domain, sensor_mask, ts)

    # --- k-Wave TR and Adjoint ---
    tr_kw = -coerce_image(
        kwave.time_reversal(
            data=data,
            domain=domain,
            sensors=image_mask,
            sources=sensor_mask,
            ts=ts,
            data_layout="nt_ns",
        ),
        n,
    )
    adj_kw = -coerce_image(
        kwave.adjoint(
            data=data,
            domain=domain,
            sensors=image_mask,
            sources=sensor_mask,
            ts=ts,
            data_layout="nt_ns",
        ),
        n,
    )

    # --- MSGB TR and Adjoint (with frequency-cropped data domain) ---
    sensor_cropped, domain_data, wpt_data, _ts_data = prepare_data_domain_for_msgb(
        data,
        domain,
        ts,
    )
    img_dyadic = DyadicDecomposition(
        num_levels=2,
        N=n,
        num_boxes_levels=(4, 8),
        box_aspect_ratio=(1, 1),
    )
    img_wpt = MSWPT(img_dyadic, redundancy=2, windowing="rectangular_mirror")
    msgb = MSGBSolver(
        thr=int(img_wpt.total_coeffs),
        thr_strat="top_n",
        batch_size=64,
        input_type="spatial",
        ode_solver=gb_solvers.solve_ODE_base,
        tr_ode_solver=gb_solvers.solve_ODE_batch_t,
        sum_method="scan_real",
    )
    sensors_eval = Sensor(domain=domain, binary_mask=image_mask)
    tr_msgb_raw, _ = msgb.time_reversal(
        data=sensor_cropped,
        domain=domain,
        sensors=sensors_eval,
        sources=sensors,
        ts=ts,
        data_domain=domain_data,
        data_wpt=wpt_data,
    )
    adj_msgb_raw, _ = msgb.adjoint(
        data=sensor_cropped,
        domain=domain,
        sensors=sensors_eval,
        sources=sensors,
        ts=ts,
        data_domain=domain_data,
        data_wpt=wpt_data,
    )
    tr_msgb = np.asarray(tr_msgb_raw).real.reshape(n)
    adj_msgb = np.asarray(adj_msgb_raw).real.reshape(n)

    # --- Best-L2 scale each reconstruction and print metrics ---
    truth = np.asarray(p0)
    tr_kw_s, tr_kw_l2 = scaled(tr_kw, truth)
    adj_kw_s, adj_kw_l2 = scaled(adj_kw, truth)
    tr_msgb_s, tr_msgb_l2 = scaled(tr_msgb, truth)
    adj_msgb_s, adj_msgb_l2 = scaled(adj_msgb, truth)

    print(f"TR k-Wave   rel L2 = {tr_kw_l2:.3f}")
    print(f"TR MSGB     rel L2 = {tr_msgb_l2:.3f}")
    print(f"Adj k-Wave  rel L2 = {adj_kw_l2:.3f}")
    print(f"Adj MSGB    rel L2 = {adj_msgb_l2:.3f}")

    out_dir = Path(utils.detect_root()) / "plots" / "optional"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "2d_time_reversal_and_adjoint.png"
    plot_comparison(
        truth,
        tr_kw_s,
        tr_msgb_s,
        adj_kw_s,
        adj_msgb_s,
        domain=domain,
        sensors=sensors,
        out_path=out_path,
    )
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
