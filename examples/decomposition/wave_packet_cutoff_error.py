#!/usr/bin/env python
"""
Plot MSWPT frame cutoff error across dyadic scales.

For each Fourier box, the example compares rectangular and mirrored-rectangular
frame functions against the unwindowed frame. The result illustrates that the
cutoff error is scale dependent but effectively constant within a scale.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.special import erf

from beamax import transforms, utils
from beamax.decomposition import DyadicDecomposition
from beamax.plotter import use_beamax_style


jax.config.update("jax_enable_x64", True)


def main() -> None:
    root_dir = utils.detect_root()
    plot_dir = Path(root_dir) / "plots"
    plot_dir.mkdir(exist_ok=True)
    use_beamax_style()

    d = 2
    n = (128,) * d
    num_levels = 3
    num_boxes_levels = tuple(2 ** (level + 2) for level in range(num_levels))
    redundancy = 2
    decomp = DyadicDecomposition(
        num_levels=num_levels,
        N=n,
        num_boxes_levels=num_boxes_levels,
        box_aspect_ratio=(1,) * d,
    )
    kxy = decomp.fourier_meshgrid
    shift = jnp.ones((d,))

    def compute_error(box_idx: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        omega = jnp.linalg.norm(decomp.centres_ndim[box_idx])
        idx = jnp.array([box_idx])
        frame_rect = transforms.compute_frames(
            decomp, idx, shift, kxy, redundancy, "rectangular"
        )
        frame_mirror = transforms.compute_frames(
            decomp, idx, shift, kxy, redundancy, "rectangular_mirror"
        )
        frame_none = transforms.compute_frames(
            decomp, idx, shift, kxy, redundancy, "none"
        )
        return (
            omega,
            jnp.linalg.norm(frame_rect - frame_none) ** 2,
            jnp.linalg.norm(frame_mirror - frame_none) ** 2,
        )

    indices = jnp.arange(jnp.sum(decomp.num_boxes_ndim))
    omegas, errors_rect, errors_mirror = jax.vmap(compute_error)(indices)
    cumsum = jnp.concatenate([jnp.array([0]), decomp.num_boxes_ndim_cumsum])

    for level in range(num_levels):
        level_errors = errors_rect[cumsum[level] : cumsum[level + 1]]
        if not bool(jnp.allclose(level_errors, level_errors[0], atol=1e-12)):
            raise RuntimeError(f"Frame error is not constant within level {level}")

    bound = (jnp.sqrt(2 * jnp.pi) / (8 * redundancy)) ** d * (
        1 - erf(jnp.sqrt(2)) ** d
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hlines(
        bound,
        xmin=float(jnp.min(omegas)),
        xmax=float(jnp.max(omegas)),
        color="red",
        linestyle="--",
        label="rectangular bound",
    )
    ax.loglog(omegas, errors_rect, ".", label="rectangular")
    ax.loglog(omegas, errors_mirror, ".", label="rectangular mirror")
    ax.set_xlabel("box center frequency norm")
    ax.set_ylabel("squared L2 frame error")
    ax.set_title("MSWPT frame cutoff error")
    ax.legend()
    fig.tight_layout()

    out_path = plot_dir / "wave_packet_cutoff_error.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(
        "Frame error range: "
        f"rect=[{float(errors_rect.min()):.2e}, {float(errors_rect.max()):.2e}], "
        f"mirror=[{float(errors_mirror.min()):.2e}, {float(errors_mirror.max()):.2e}]"
    )
    print(f"Saved MSWPT error plot to {out_path}")


if __name__ == "__main__":
    main()
