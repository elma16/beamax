#!/usr/bin/env python
"""
Visualise the low-pass/high-pass filter pairs used by the MSWPT.

The transform builds analysis filters `g` and synthesis filters `h` whose
pointwise products form a partition of unity. This example checks that
partition numerically in 1D and 2D, then saves the corresponding diagnostic
plots.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from beamax import utils
from beamax.decomposition import DyadicDecomposition
from beamax.plotter import use_beamax_style
from beamax.transforms import compute_gh_filters


jax.config.update("jax_enable_x64", True)


def _filter_setup(n: tuple[int, ...]) -> tuple[DyadicDecomposition, int, str]:
    num_levels = 2
    num_boxes_levels = tuple(2 ** (level + 2) for level in range(num_levels))
    decomp = DyadicDecomposition(
        num_levels=num_levels,
        N=n,
        num_boxes_levels=num_boxes_levels,
        box_aspect_ratio=(1,) * len(n),
    )
    return decomp, 2, "rectangular_mirror"


def main() -> None:
    root_dir = utils.detect_root()
    plot_dir = Path(root_dir) / "plots"
    plot_dir.mkdir(exist_ok=True)
    use_beamax_style()

    decomp_1d, redundancy, windowing = _filter_setup((256,))
    gs_1d, hs_1d = compute_gh_filters(decomp_1d, redundancy, windowing)
    partition_1d = jnp.sum(gs_1d * hs_1d, axis=0)
    err_1d = float(jnp.max(jnp.abs(partition_1d - 1.0)))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gs_1d[0], label="analysis g")
    ax.plot(hs_1d[0], label="synthesis h")
    ax.plot(partition_1d, label="sum g*h")
    ax.set_title("1D MSWPT filters")
    ax.set_xlabel("frequency index")
    ax.legend()
    fig.tight_layout()
    out_1d = plot_dir / "low_high_pass_filters_1d.png"
    fig.savefig(out_1d, dpi=180, bbox_inches="tight")
    plt.close(fig)

    decomp_2d, redundancy, windowing = _filter_setup((64, 64))
    gs_2d, hs_2d = compute_gh_filters(decomp_2d, redundancy, windowing)
    partition_2d = jnp.sum(gs_2d * hs_2d, axis=0)
    err_2d = float(jnp.max(jnp.abs(partition_2d - 1.0)))

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    for ax, arr, title in [
        (axes[0], gs_2d[0], "analysis g"),
        (axes[1], hs_2d[0], "synthesis h"),
        (axes[2], partition_2d, "sum g*h"),
    ]:
        im = ax.imshow(arr, origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_2d = plot_dir / "low_high_pass_filters_2d.png"
    fig.savefig(out_2d, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"1D partition max error: {err_1d:.2e}; saved {out_1d}")
    print(f"2D partition max error: {err_2d:.2e}; saved {out_2d}")


if __name__ == "__main__":
    main()
