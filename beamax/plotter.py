"""Optional Matplotlib helpers for Beamax examples."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from beamax.decomposition import DyadicDecomposition

__all__ = [
    "plot_mswpt_coeffs",
    "plot_mswpt_coeffs_3d",
    "use_beamax_style",
]


def use_beamax_style() -> None:
    """Apply the Matplotlib style used by Beamax examples."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": "medium",
            "axes.labelsize": "medium",
            "legend.frameon": False,
            "image.cmap": "viridis",
        }
    )


def _indices_with_norm_at_most(centres: np.ndarray, norm: float) -> np.ndarray:
    norms = np.linalg.norm(centres, axis=1, ord=np.inf)
    return np.flatnonzero(norms <= norm)


def _indices_between_corners(
    centres: np.ndarray,
    corner1_idx: int,
    corner2_idx: int,
) -> np.ndarray:
    corner1 = centres[corner1_idx]
    corner2 = centres[corner2_idx]
    if np.array_equal(corner1, corner2):
        raise ValueError("corner1 and corner2 must differ to define a box.")

    lower = np.minimum(corner1, corner2)
    upper = np.maximum(corner1, corner2)
    return np.flatnonzero(np.all((centres >= lower) & (centres <= upper), axis=1))


def _mswpt_highlight_indices(
    dyadic_decomp: DyadicDecomposition,
    cutoff_freq: float | None,
    box_corners: np.ndarray | tuple[int, int] | None,
) -> np.ndarray | None:
    centres = np.asarray(dyadic_decomp.centres_ndim)
    if cutoff_freq is not None:
        return _indices_with_norm_at_most(centres, cutoff_freq)
    if box_corners is None:
        return None

    corners = np.asarray(box_corners, dtype=int)
    if corners.shape != (2,):
        raise ValueError("box_corners must contain exactly two box indices.")
    return _indices_between_corners(centres, int(corners[0]), int(corners[1]))


def _find_level(dyadic_decomp: DyadicDecomposition, box_index: int) -> int:
    cumulative_counts = np.asarray(dyadic_decomp.num_boxes_ndim_cumsum)
    return int(np.searchsorted(cumulative_counts, box_index, side="right"))


def plot_mswpt_coeffs(
    ax: Axes,
    coeffs_array,
    dyadic_decomp: DyadicDecomposition,
    cutoff_freq: float | None = None,
    box_corners: np.ndarray | tuple[int, int] | None = None,
    asymptote: bool = False,
    asymptote_slope: float | None = None,
    normalized_frequency_axes: bool = False,
    log_scale: bool = False,
):
    r"""Plot 2D MSWPT coefficient magnitudes and their dyadic boxes.

    ``cutoff_freq`` or ``box_corners`` highlights the coefficient region used
    by a low-frequency solve. Frequency axes can be Fourier-bin coordinates or
    cycles per sample in $[-0.5, 0.5]$.

    Returns
    -------
    matplotlib.image.AxesImage
        The image returned by :meth:`matplotlib.axes.Axes.imshow`.
    """
    # Decomposition axis 0 is horizontal and axis 1 is vertical.
    coeffs_mag = np.asarray(np.abs(coeffs_array)).T

    N0, N1 = dyadic_decomp.N
    x_scale = 1.0 / N0 if normalized_frequency_axes else 1.0
    y_scale = 1.0 / N1 if normalized_frequency_axes else 1.0
    extent_coeffs = (
        -N0 / 2 * x_scale,
        N0 / 2 * x_scale,
        -N1 / 2 * y_scale,
        N1 / 2 * y_scale,
    )

    if log_scale:
        c_max = float(coeffs_mag.max()) or 1.0
        c_min = c_max * 1e-2
        coeffs_mag_plot = np.maximum(coeffs_mag, c_min)
        norm = mcolors.LogNorm(vmin=c_min, vmax=c_max)
        title = r"$\log |c_{\ell,j,k}|$"
    else:
        coeffs_mag_plot = coeffs_mag
        norm = None
        title = r"$|c_{\ell,j,k}|$"

    image = ax.imshow(
        coeffs_mag_plot,
        origin="lower",
        extent=extent_coeffs,
        aspect="auto",
        norm=norm,
    )
    ax.set_title(title)

    cumulative_boxes = np.r_[0, np.cumsum(dyadic_decomp.num_boxes_ndim)]
    box_lengths = dyadic_decomp.box_lengths
    box_aspect = dyadic_decomp.box_aspect_ratio
    colors = ["gray", "darkgray", "silver", "lightgray"]

    for level in range(dyadic_decomp.num_levels):
        start_idx = int(cumulative_boxes[level])
        end_idx = int(cumulative_boxes[level + 1])
        centres = dyadic_decomp.centres_ndim[start_idx:end_idx]
        box_length = float(box_lengths[level])
        box_width_x = float(box_length * box_aspect[0] * x_scale)
        box_width_y = float(box_length * box_aspect[1] * y_scale)

        for centre in centres:
            cx = float(centre[0]) * x_scale
            cy = float(centre[1]) * y_scale
            ax.add_patch(
                patches.Rectangle(
                    (cx - box_width_x / 2, cy - box_width_y / 2),
                    box_width_x,
                    box_width_y,
                    linewidth=1.0,
                    edgecolor=colors[level % len(colors)],
                    facecolor="none",
                    linestyle=":",
                    alpha=0.6,
                )
            )

    selected = _mswpt_highlight_indices(dyadic_decomp, cutoff_freq, box_corners)
    if selected is not None and selected.size:
        bounds_x: list[float] = []
        bounds_y: list[float] = []
        for global_idx in selected:
            level = _find_level(dyadic_decomp, int(global_idx))
            centre = dyadic_decomp.centres_ndim[int(global_idx)]
            box_length = float(box_lengths[level])
            box_width_x = float(box_length * box_aspect[0]) * x_scale
            box_width_y = float(box_length * box_aspect[1]) * y_scale
            cx = float(centre[0]) * x_scale
            cy = float(centre[1]) * y_scale
            bounds_x.extend([cx - box_width_x / 2, cx + box_width_x / 2])
            bounds_y.extend([cy - box_width_y / 2, cy + box_width_y / 2])

        ax.add_patch(
            patches.Rectangle(
                (min(bounds_x), min(bounds_y)),
                max(bounds_x) - min(bounds_x),
                max(bounds_y) - min(bounds_y),
                linewidth=2.0,
                edgecolor="red",
                facecolor="none",
                linestyle="--",
                zorder=5,
            )
        )

    if asymptote:
        slope = 4.0 if asymptote_slope is None else float(asymptote_slope)
        if not np.isfinite(slope) or slope <= 0.0:
            raise ValueError("asymptote_slope must be positive and finite.")
        x_limit = min(abs(extent_coeffs[1]), abs(extent_coeffs[3]) / slope)
        y_limit = slope * x_limit
        ax.plot(
            [-x_limit, x_limit],
            [-y_limit, y_limit],
            color="#ffa500",
            linestyle="-.",
            linewidth=2.5,
            zorder=6,
        )
        ax.plot(
            [-x_limit, x_limit],
            [y_limit, -y_limit],
            color="#ffa500",
            linestyle="-.",
            linewidth=2.5,
            zorder=6,
        )

    return image


def plot_mswpt_coeffs_3d(
    ax: Axes,
    coeffs_array,
    dyadic_decomp: DyadicDecomposition,
    cutoff_freq: float | None = None,
    box_corners: np.ndarray | tuple[int, int] | None = None,
    asymptote: bool = False,
):
    """Plot a 2D projection of 3D MSWPT coefficients and dyadic boxes."""
    coeffs_mag = np.asarray(np.abs(coeffs_array)).T
    if len(dyadic_decomp.N) < 2:
        raise ValueError(
            "dyadic_decomp.N must have at least two dimensions for 3D plots."
        )

    N0, N1 = dyadic_decomp.N[:2]
    extent_coeffs = (-N1 // 2, N1 // 2, -N0 // 2, N0 // 2)
    c_max = float(coeffs_mag.max()) or 1.0
    c_min = c_max * 1e-2
    image = ax.imshow(
        np.maximum(coeffs_mag, c_min),
        origin="lower",
        extent=extent_coeffs,
        aspect="auto",
        norm=mcolors.LogNorm(vmin=c_min, vmax=c_max),
    )

    cumulative_boxes = np.r_[0, np.cumsum(dyadic_decomp.num_boxes_ndim)]
    box_lengths = dyadic_decomp.box_lengths
    box_aspect = dyadic_decomp.box_aspect_ratio
    colors = ["gray", "darkgray", "silver", "lightgray"]

    for level in range(dyadic_decomp.num_levels):
        start_idx = int(cumulative_boxes[level])
        end_idx = int(cumulative_boxes[level + 1])
        centres = dyadic_decomp.centres_ndim[start_idx:end_idx]
        box_length = float(box_lengths[level])
        box_width_x = float(box_length * box_aspect[1])
        box_width_y = float(box_length * box_aspect[0])

        seen_projected: set[tuple[float, float]] = set()
        for centre in centres:
            cx, cy = float(centre[1]), float(centre[0])
            key = (round(cx, 9), round(cy, 9))
            if key in seen_projected:
                continue
            seen_projected.add(key)
            ax.add_patch(
                patches.Rectangle(
                    (cx - box_width_x / 2, cy - box_width_y / 2),
                    box_width_x,
                    box_width_y,
                    linewidth=1.0,
                    edgecolor=colors[level % len(colors)],
                    facecolor="none",
                    linestyle=":",
                    alpha=0.6,
                )
            )

    selected = _mswpt_highlight_indices(dyadic_decomp, cutoff_freq, box_corners)
    if selected is not None and selected.size:
        bounds_x: list[float] = []
        bounds_y: list[float] = []
        for global_idx in selected:
            level = _find_level(dyadic_decomp, int(global_idx))
            centre = dyadic_decomp.centres_ndim[int(global_idx)]
            box_length = float(box_lengths[level])
            box_width_x = float(box_length * box_aspect[1])
            box_width_y = float(box_length * box_aspect[0])
            cx, cy = float(centre[1]), float(centre[0])
            bounds_x.extend([cx - box_width_x / 2, cx + box_width_x / 2])
            bounds_y.extend([cy - box_width_y / 2, cy + box_width_y / 2])

        ax.add_patch(
            patches.Rectangle(
                (min(bounds_x), min(bounds_y)),
                max(bounds_x) - min(bounds_x),
                max(bounds_y) - min(bounds_y),
                linewidth=2.0,
                edgecolor="red",
                facecolor="none",
                linestyle="--",
                zorder=5,
            )
        )

    if asymptote:
        limit = abs(extent_coeffs[1])
        ax.plot(
            [-limit / 2, limit / 2],
            [-2 * limit, 2 * limit],
            color="orange",
            linestyle="-.",
            linewidth=1.5,
            zorder=6,
        )
        ax.plot(
            [-limit / 2, limit / 2],
            [2 * limit, -2 * limit],
            color="orange",
            linestyle="-.",
            linewidth=1.5,
            zorder=6,
        )

    return image
