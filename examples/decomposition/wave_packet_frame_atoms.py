#!/usr/bin/env python
"""
Render a small grid of MSWPT frame atoms in Fourier space.

The multiscale wave-packet transform partitions Fourier space into boxes at
several scales. This example picks representative boxes and intra-box shifts,
evaluates their frame functions, and saves a compact figure that shows how the
tiling changes from coarse to fine scales.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from beamax import utils
from beamax.decomposition import DyadicDecomposition
from beamax.plotter import use_beamax_style
from beamax.transforms import compute_frames


jax.config.update("jax_enable_x64", True)


def main() -> None:
    root_dir = utils.detect_root()
    plot_dir = Path(root_dir) / "plots"
    plot_dir.mkdir(exist_ok=True)
    use_beamax_style()

    n = (64, 64)
    num_levels = 2
    num_boxes_levels = (4, 8)
    redundancy = 2
    windowing = "rectangular"

    decomp = DyadicDecomposition(
        num_levels=num_levels,
        N=n,
        num_boxes_levels=num_boxes_levels,
        box_aspect_ratio=(1, 1),
    )
    kxy = decomp.fourier_meshgrid
    total_boxes = int(jnp.sum(decomp.num_boxes_ndim))

    box_indices = jnp.linspace(0, total_boxes - 1, 6).round().astype(int)
    shifts = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1], [1, 2]])

    fig, axes = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
    for ax, box_idx, shift in zip(axes.ravel(), box_indices, shifts):
        frame = compute_frames(
            decomp,
            jnp.array([int(box_idx)]),
            shift,
            kxy,
            redundancy,
            windowing,
        )
        level = int(np.asarray(utils.find_level(decomp, jnp.array([int(box_idx)]))).ravel()[0])
        ax.imshow(np.asarray(jnp.real(frame)), origin="lower", cmap="viridis")
        ax.set_title(f"level {level}, box {int(box_idx)}, k={tuple(map(int, shift))}")
        ax.set_xticks([])
        ax.set_yticks([])

    out_path = plot_dir / "wave_packet_frame_atoms.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved representative MSWPT frame atoms to {out_path}")


if __name__ == "__main__":
    main()
