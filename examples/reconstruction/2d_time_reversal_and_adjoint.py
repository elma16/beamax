"""Compare 2D MSGB and k-Wave time-reversal and adjoint reconstructions.

Example extras: kwave,viz-mpl
Example smoke: false
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from beamax import utils
from beamax.decomposition import DyadicDecomposition
from beamax.geometry import Domain, Sensor
from beamax.gb import gb_solvers
from beamax.solvers import KWaveSolver, MSGBSolver
from beamax.transforms import MSWPT

jax.config.update("jax_enable_x64", True)


def c_homogeneous(x: jnp.ndarray) -> jnp.ndarray:
    return 1500.0 + 0.0 * x[..., 0]


def make_two_gaussian_phantom(domain: Domain) -> jnp.ndarray:
    r"""Return a zero-mean phantom normalised to $\max|p_0|=1$."""
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
    """Match k-Wave image orientation to ``shape``."""
    image = np.asarray(arr)
    if image.shape == shape:
        return image
    if image.T.shape == shape:
        return image.T
    return image.reshape(shape)


def scaled(recon: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, float]:
    r"""Fit the $\ell^2$ scale and return the scaled image and relative error."""
    r = np.asarray(recon).real
    t = np.asarray(truth).real
    s = float(np.vdot(r, t) / (np.vdot(r, r) + 1e-30))
    out = s * r
    return out, utils.rel_l2(t, out)


def prepare_data_domain_for_msgb(
    sensor_data_kw: jnp.ndarray,
    domain: Domain,
    ts: jnp.ndarray,
    *,
    over_resolve: int = 2,
):
    r"""Fourier-crop the record and build its $(N_t',N_s)$ MSGB data domain."""
    sensor_arr = jnp.asarray(sensor_data_kw)
    if sensor_arr.ndim != 2:
        raise ValueError(f"Expected (Nt, Ns) sensor data; got {sensor_arr.shape}")

    nt_cropped = over_resolve * domain.N[0]
    if sensor_arr.shape[0] < nt_cropped:
        raise ValueError(
            f"Need >= {nt_cropped} time samples; got {sensor_arr.shape[0]}."
        )
    fft = utils.unitary_fft(sensor_arr)
    mid = fft.shape[0] // 2
    cropped_fft = fft[mid - nt_cropped // 2 : mid + nt_cropped // 2]
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

    # Match the dyadic aspect ratio to the rectangular data grid.
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


def plot_comparison(
    p0,
    sensor_data,
    tr_kw,
    tr_msgb,
    adj_kw,
    adj_msgb,
    domain,
    sensors,
    *,
    out_path,
):
    arrays = [np.asarray(a).real for a in (p0, tr_kw, tr_msgb, adj_kw, adj_msgb)]
    sensor_arr = np.asarray(sensor_data).real
    if sensor_arr.ndim != 2:
        sensor_arr = sensor_arr.reshape(sensor_arr.shape[0], -1)
    vmax = max(float(np.max(np.abs(a))) for a in arrays)
    sensor_vmax = float(np.percentile(np.abs(sensor_arr), 99.5))
    if sensor_vmax == 0.0:
        sensor_vmax = 1.0
    extent = (0.0, float(domain.grid_size[1]), 0.0, float(domain.grid_size[0]))

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.0, 1.0, 0.85],
        hspace=0.25,
        wspace=0.08,
    )

    panels = (
        (0, 0, r"$p_0$", arrays[0]),
        (0, 1, r"$p_{\mathrm{TR}}^{\mathrm{k\!-\!Wave}}$", arrays[1]),
        (0, 2, r"$p_{\mathrm{TR}}^{\mathrm{MSGB}}$", arrays[2]),
        (1, 1, r"$p_{\mathrm{Adj}}^{\mathrm{k\!-\!Wave}}$", arrays[3]),
        (1, 2, r"$p_{\mathrm{Adj}}^{\mathrm{MSGB}}$", arrays[4]),
    )
    image_axes = []
    for row, column, title, arr in panels:
        ax = fig.add_subplot(gs[row, column])
        ax.imshow(
            arr,
            origin="lower",
            extent=extent,
            vmin=-vmax,
            vmax=vmax,
            cmap="RdBu_r",
            aspect="equal",
        )
        ax.set(title=title, xticks=[], yticks=[])
        image_axes.append(ax)

    ax_data = fig.add_subplot(gs[1, 0])
    ax_data.imshow(
        sensor_arr,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=-sensor_vmax,
        vmax=sensor_vmax,
    )
    ax_data.set(title="sensor data", xlabel=r"$x_s$", ylabel=r"$t$")
    ax_data.set_box_aspect(1)
    ax_data.set(xticks=[], yticks=[])

    rr, cc = jnp.where(sensors.binary_mask)
    xs = (np.asarray(cc) + 0.5) * float(domain.dx[1])
    ys = (np.asarray(rr) + 0.5) * float(domain.dx[0])
    for ax in image_axes:
        ax.scatter(
            xs,
            ys,
            s=16,
            c="red",
            marker="^",
            alpha=0.9,
            edgecolors="white",
            linewidths=0.3,
            zorder=10,
        )

    ax_prof = fig.add_subplot(gs[2, :])
    idx = arrays[0].shape[1] // 2
    y_axis = np.arange(arrays[0].shape[0]) * float(domain.dx[0])
    profiles = (
        (arrays[0], "black", None, 2.0, r"$p_0$"),
        (arrays[1], "C0", None, 1.5, "TR k-Wave"),
        (arrays[2], "C0", "--", 1.5, "TR MSGB"),
        (arrays[3], "C3", None, 1.5, "Adj k-Wave"),
        (arrays[4], "C3", "--", 1.5, "Adj MSGB"),
    )
    for image, color, linestyle, width, label in profiles:
        ax_prof.plot(
            y_axis,
            image[:, idx],
            color=color,
            lw=width,
            label=label,
            **({"ls": linestyle} if linestyle else {}),
        )
    ax_prof.set(
        xlabel="y [m]",
        ylabel="pressure",
        title=f"profile at x = {idx * float(domain.dx[1]):.1e} m",
    )
    ax_prof.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=5)
    ax_prof.axvline(idx * float(domain.dx[0]), color="grey", ls=":", lw=0.8)
    for ax in image_axes:
        ax.axvline(idx * float(domain.dx[1]), color="grey", ls=":", lw=0.8)

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
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

    sensor_mask = jnp.zeros(n).at[0, :].set(1.0)
    sensors = Sensor(domain=domain, binary_mask=sensor_mask)
    image_mask = jnp.ones(n)

    kwave = KWaveSolver(
        backend="python",
        device="cpu",
        pml_size=8,
        smooth_p0=False,
        debug=False,
    )
    data = kwave.forward(p0, domain, sensor_mask, ts)

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
    tr_msgb_raw = msgb.time_reversal(
        data=sensor_cropped,
        domain=domain,
        sensors=sensors_eval,
        sources=sensors,
        ts=ts,
        data_domain=domain_data,
        data_wpt=wpt_data,
    )
    adj_msgb_raw = msgb.adjoint(
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

    truth = np.asarray(p0)
    tr_kw_s, tr_kw_l2 = scaled(tr_kw, truth)
    adj_kw_s, adj_kw_l2 = scaled(adj_kw, truth)
    tr_msgb_s, tr_msgb_l2 = scaled(tr_msgb, truth)
    adj_msgb_s, adj_msgb_l2 = scaled(adj_msgb, truth)

    print(f"TR k-Wave   rel L2 = {tr_kw_l2:.3f}")
    print(f"TR MSGB     rel L2 = {tr_msgb_l2:.3f}")
    print(f"Adj k-Wave  rel L2 = {adj_kw_l2:.3f}")
    print(f"Adj MSGB    rel L2 = {adj_msgb_l2:.3f}")

    out_dir = Path("plots/reconstruction")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "2d_time_reversal_and_adjoint.png"
    plot_comparison(
        truth,
        data,
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
