"""Matplotlib panels adapted from thesis_experiments.figures and the Chapter 4/5 replotters."""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np


# Match the final-width typography of thesis_experiments.figures.apply_dense_thesis_style.
_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
    "image.cmap": "viridis",
    "image.origin": "upper",
    "mathtext.fontset": "dejavusans",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.formatter.use_mathtext": True,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.05,
    "figure.constrained_layout.use": False,
    "axes.grid": False,
}


def _profile_column(errors):
    """Largest linewise RMS discrepancy; break ties by peak error as in the thesis."""
    stacked = np.abs(np.stack(errors))
    rms = np.sqrt(np.mean(stacked**2, axis=1)).max(axis=0)
    peak = stacked.max(axis=(0, 1))
    columns = np.arange(rms.size)
    margin = max(1, int(np.ceil(0.05 * rms.size)))
    if 2 * margin < rms.size:
        columns = columns[margin:-margin]
    return int(columns[np.lexsort((peak[columns], rms[columns]))[-1]])


_WIDTH = 140 / 25.4  # Thesis text width in inches.


def _compact_difference_colorbar(bar):
    ticks = bar.get_ticks()
    tolerance = (
        np.finfo(float).eps * max(np.max(np.abs(ticks)), np.finfo(float).tiny) * 64
    )
    if np.any(np.abs(ticks - ticks.round(2)) > tolerance):
        formatter = ScalarFormatter(useMathText=True, useOffset=False)
        formatter.set_powerlimits((0, 0))
        bar.formatter = formatter
        bar.update_ticks()


@plt.rc_context(_STYLE)
def plot_forward_records(kwave, msgb, hybrid, ts) -> Figure:
    """Chapter 4 sensor plate, with time-first records and signed residuals."""
    kw, msgb, hyb, ts = map(np.asarray, (kwave, msgb, hybrid, ts))
    if (
        kw.ndim != 2
        or kw.shape != msgb.shape
        or kw.shape != hyb.shape
        or ts.shape != (kw.shape[0],)
    ):
        raise ValueError(
            "Expected three records of shape (time, receiver) and a matching time vector."
        )
    d_kw_msgb, d_kw_hyb = kw - msgb, kw - hyb
    s_min = float(min(arr.min() for arr in (kw, msgb, hyb)))
    s_max = float(max(arr.max() for arr in (kw, msgb, hyb)))
    s_norm = mcolors.Normalize(vmin=s_min, vmax=s_max)
    d_absmax = float(max(np.max(np.abs(d_kw_msgb)), np.max(np.abs(d_kw_hyb))))
    d_norm = mcolors.Normalize(vmin=-d_absmax, vmax=d_absmax)

    _, nx = kw.shape
    extent_sensor = [0, 1.0, float(ts.max()), float(ts.min())]
    xs = np.linspace(0, 1.0, nx)
    idx_profile = _profile_column((d_kw_msgb, d_kw_hyb))
    profile_pos = xs[idx_profile]
    y_kw = kw[:, idx_profile]
    y_msgb = msgb[:, idx_profile]
    y_hyb = hyb[:, idx_profile]

    # Paper-space anchors from the 300 dpi PNG included in Chapter 4. Freezing
    # its final crop keeps tick extents from resizing the plate in Colab.
    width, height = 1560, 1222
    fig = plt.figure(figsize=(width / 300, height / 300))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=6,
        # The narrow blank column separates the left colourbar from the
        # first panel's vertical time label without changing image widths.
        width_ratios=[0.07, 0.02, 1.0, 1.0, 1.0, 0.07],
        height_ratios=[1.0, 0.78],
        left=115 / width,
        right=1446 / width,
        bottom=1 - 1025 / height,
        top=1 - 73 / height,
        hspace=0.30,
        wspace=0.17,
    )

    cax_l = fig.add_subplot(gs[0, 0])
    ax_kw = fig.add_subplot(gs[0, 2])
    ax_dmsgb = fig.add_subplot(gs[0, 3])
    ax_dhyb = fig.add_subplot(gs[0, 4])
    cax_r = fig.add_subplot(gs[0, 5])

    im_kw = ax_kw.imshow(kw, extent=extent_sensor, aspect="auto", norm=s_norm)
    ax_dmsgb.imshow(
        d_kw_msgb, extent=extent_sensor, aspect="auto", norm=d_norm, cmap="RdBu_r"
    )
    im_dhyb = ax_dhyb.imshow(
        d_kw_hyb, extent=extent_sensor, aspect="auto", norm=d_norm, cmap="RdBu_r"
    )

    ax_kw.set_title(r"$g^{\mathrm{k\text{-}Wave}}$")
    ax_kw.set_xlabel("$x_{s}$")
    ax_kw.set_ylabel("t")
    ax_dmsgb.set_title(r"$g^{\mathrm{k\text{-}Wave}}-g^{\mathrm{MSGB}}$")
    ax_dhyb.set_title(r"$g^{\mathrm{k\text{-}Wave}}-g^{\mathrm{Hybrid}}$")
    for ax in (ax_kw, ax_dmsgb, ax_dhyb):
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im_kw, cax=cax_l)
    cax_l.yaxis.set_ticks_position("left")
    cax_l.yaxis.set_label_position("left")
    difference_colorbar = fig.colorbar(im_dhyb, cax=cax_r)
    _compact_difference_colorbar(difference_colorbar)

    for ax in (ax_kw, ax_dmsgb, ax_dhyb):
        ax.axvline(profile_pos, ls="--", lw=1.0, color="k", zorder=5)

    ax_prof = fig.add_subplot(gs[1, 2:5])
    lines = [
        ax_prof.plot(ts, y_kw, label="k-Wave", lw=1.3)[0],
        ax_prof.plot(ts, y_msgb, "--", label="MSGB", lw=1.3)[0],
        ax_prof.plot(ts, y_hyb, ":", label="Hybrid", lw=1.3)[0],
    ]
    ax_prof.set_xlabel("t")
    ax_prof.set_ylabel(r"$g(x_{s},t)$")
    ax_prof.grid(True, alpha=0.4)
    fig.legend(
        lines,
        [line.get_label() for line in lines],
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(768.5 / width, 0.7 / height),
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.5,
    )

    return fig


@plt.rc_context(_STYLE)
def plot_packet_box_scanner(
    p0, mask, grid_size, *, time_reversal=None, adjoint=None
) -> Figure:
    """Draw the one-face excerpt of the Chapter 5 homogeneous two-packet plate.

    Supply (k-Wave, MSGB) pairs of raw fields. The original four-face figure
    determines the marked slice, colour limits, profile limits and adjoint
    display gain. Keeping that calibration lets a one-face run reproduce its row.
    """
    p0 = np.asarray(p0)
    if p0.shape != (128, 128) or np.shape(mask) != p0.shape:
        raise ValueError(
            "The thesis two-packet plate uses a 128 × 128 image and detector mask."
        )
    # inverse_2d_box.py, 2d_1_two_packets_hom_tau1pct: fit against the final
    # four-face k-Wave adjoint once, before selecting the one-face excerpt.
    adjoint_scale = 2296.5277637882564
    norm = mcolors.Normalize(-1.0806082602212956, 1.1509535463142675)
    column = 87
    xs = np.linspace(0, float(grid_size[0]), p0.shape[1])
    ys = np.linspace(0, float(grid_size[1]), p0.shape[0])
    extent = (0, xs[-1], 0, ys[-1])
    groups = []
    if time_reversal is not None:
        groups.append(
            ("TR", r"$A^{\mathrm{TR}}$", tuple(map(np.asarray, time_reversal)))
        )
    if adjoint is not None:
        groups.append(
            (
                "Adj",
                r"$A^{\mathrm{adj}}$",
                tuple(adjoint_scale * np.asarray(a) for a in adjoint),
            )
        )
    if not groups:
        raise ValueError("Supply time_reversal, adjoint, or both pairs.")
    if any(
        len(pair) != 2 or any(field.shape != p0.shape for field in pair)
        for _, _, pair in groups
    ):
        raise ValueError(
            "Each operation requires two fields with the same shape as the source."
        )

    # Keep the thesis panel dimensions when removing the three other detector
    # rows. Use the original layout as an inch ruler for both images and profiles.
    height = _WIDTH / 0.75
    fig = plt.figure(figsize=(_WIDTH, height))
    grid = fig.add_gridspec(
        6,
        6,
        height_ratios=[1, 1, 1, 1, 0.72, 0.72],
        width_ratios=[1, 1, 1, 1, 1, 0.04],
        left=0.08,
        right=0.93,
        bottom=0.09,
        top=0.95,
        hspace=0.42,
        wspace=0.08,
    )
    removed = grid[0, 0].get_position(fig).y0 - grid[3, 0].get_position(fig).y0
    if len(groups) == 1:
        removed += grid[4, 0].get_position(fig).y0 - grid[5, 0].get_position(fig).y0
    fig.set_size_inches(_WIDTH, height * (1 - removed))

    def panel(row, cols, **kwargs):
        pos = grid[row, cols].get_position(fig)
        bottom = pos.y0 - removed if row == 0 else pos.y0
        return fig.add_axes(
            (pos.x0, bottom / (1 - removed), pos.width, pos.height / (1 - removed)),
            **kwargs,
        )

    names = (r"\mathrm{k\text{-}Wave}", r"\mathrm{MSGB}")
    images = [(r"$p_0$", p0)]
    for op, _, pair in groups:
        images.extend(
            (rf"$p_{{\mathrm{{{op}}}}}^{{{name}}}$", field)
            for name, field in zip(names, pair, strict=True)
        )
    rr, cc = np.nonzero(mask)
    # Centre the three image panels in the single-operation excerpts.
    first = (5 - len(images)) // 2
    for i, (title, field) in enumerate(images):
        ax = panel(0, i + first)
        im = ax.imshow(field, origin="lower", extent=extent, norm=norm, aspect="equal")
        ax.set(title=title, xticks=[], yticks=[])
        ax.axvline(xs[column], color="k", lw=1.0, ls="--")
        if i:
            ax.scatter(xs[cc], ys[rr], s=5, c="red", marker="^", alpha=0.9, zorder=10)
    fig.colorbar(im, cax=panel(0, 5))

    profile_axes = []
    for i, (op, ylabel, fields) in enumerate(groups):
        ax = panel(6 - len(groups) + i, slice(0, 5))
        ax.plot(ys, p0[:, column], "k-", lw=1.6)
        handles = [Line2D([], [], color="black", lw=1.6, label=r"$p_0$")]
        for color, name, field in zip(("C0", "C1"), names, fields, strict=True):
            ax.plot(ys, field[:, column], color=color, lw=1.0, alpha=0.55)
            handles.append(
                Line2D(
                    [],
                    [],
                    color=color,
                    lw=1.3,
                    label=rf"$p_{{\mathrm{{{op}}}}}^{{{name}}}$",
                )
            )
        ax.set(ylabel=ylabel, ylim=(-1.1828816429136018, 1.0671327763171332))
        ax.grid(True, alpha=0.35)
        ax.legend(
            handles=handles,
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            frameon=False,
            columnspacing=1.0,
            handletextpad=0.45,
        )
        profile_axes.append(ax)
    if len(groups) == 2:
        profile_axes[0].tick_params(labelbottom=False)
    profile_axes[-1].set_xlabel("y [m]")
    fig.legend(
        handles=[Line2D([], [], color="gray", lw=1.0, alpha=0.55, label="1 face")],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012 / (1 - removed)),
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.45,
    )
    return fig


def save_comparison_figure(
    fig: Figure, path: str | Path, *, fixed_canvas=False, show=False
):
    """Export a thesis comparison at 300 dpi and optionally display the saved PNG.

    Use fixed_canvas=True for the forward plate to preserve its published margins.
    The one-face inverse plate uses a tight crop. With show=True the figure is
    displayed in a notebook or interactive backend and closed after display.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Tick locators also consult rcParams at save time. Display the exported
    # PNG in Colab so its inline formatter cannot change the ticks or crop.
    with plt.rc_context(_STYLE):
        bounds = fig.bbox_inches if fixed_canvas else "tight"
        fig.savefig(path, dpi=300, bbox_inches=bounds, pad_inches=0.05)
        backend = plt.get_backend().lower()
        if show and "inline" in backend:
            from IPython.display import Image, display

            display(Image(filename=str(path), retina=True))
        elif show and backend != "agg":
            plt.show()
    if show:
        plt.close(fig)
