#!/usr/bin/env python
"""
Plot a 3×3 grid of representative atoms for:
  - MSWPT (top row),
  - Curvelets (middle row),
  - Wavelets (bottom row).

Columns move from coarse → mid → fine scales.

Dependencies:
  pip install curvelops PyWavelets matplotlib
"""

from pathlib import Path
from typing import List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from beamax import utils
from beamax.decomposition import DyadicDecomposition
from beamax.transforms import MSWPT
from beamax.plotter import use_beamax_style

try:
    import curvelops as cl
except ModuleNotFoundError as exc:
    print("Skipping example: curvelops is not installed (`pip install curvelops`).")
    raise SystemExit(0) from exc

try:
    import pywt
except ModuleNotFoundError as exc:
    print("Skipping example: PyWavelets is not installed (`pip install PyWavelets`).")
    raise SystemExit(0) from exc

# ----------------------------------------------------------------------------- #
# Config
# ----------------------------------------------------------------------------- #

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

ROOT_DIR = utils.detect_root()
PLOT_DIR = Path(ROOT_DIR / "plots")
DATA_DIR = Path(ROOT_DIR / "data")
PLOT_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)

use_beamax_style()

N = (128, 128)
num_levels = 3  # coarse, mid, fine
box_aspect_ratio = (1, 1)
num_boxes_levels = tuple(2 ** (i + 2) for i in range(num_levels))
redundancy = 2
windowing = "rectangular_mirror"

wavelet = "db4"
wavelet_mode = "periodization"

# ----------------------------------------------------------------------------- #
# MSWPT atoms
# ----------------------------------------------------------------------------- #


def mswpt_atom(wpt: MSWPT, level: int) -> np.ndarray:
    """Single MSWPT coefficient at `level` reconstructed to spatial domain."""
    coeffs = jnp.zeros((wpt.total_coeffs,), dtype=wpt.complex_dtype)
    shape = wpt.coeff_shapes[level]
    center = (0,) + tuple(int(s // 2) for s in shape[1:])
    flat = int(np.ravel_multi_index(center, shape))
    idx = int(wpt.coeffs_cumsum[level] + flat)
    coeffs = coeffs.at[idx].set(1.0)
    atom = wpt.inverse(coeffs, output_type="spatial").real
    return np.asarray(atom)


# ----------------------------------------------------------------------------- #
# Curvelet atoms
# ----------------------------------------------------------------------------- #


def _curvelet_partition(F, coeff_len: int) -> List[int]:
    """
    Return coefficient counts per scale if available; otherwise split into thirds.
    """
    part = getattr(F, "partition", None)
    if part is not None:
        return [int(x) for x in part]
    # fallback: even split
    base = coeff_len // 3
    return [base, base, coeff_len - 2 * base]


def curvelet_atom(F, scale_idx: int, img_shape) -> np.ndarray:
    """Reconstruct a curvelet atom near the spatial center for a given scale."""
    # Determine coefficient length
    coeff_probe = F @ np.zeros(img_shape)
    coeff_len = int(np.asarray(coeff_probe).size)
    partition = _curvelet_partition(F, coeff_len)
    scale_idx = int(np.clip(scale_idx, 0, len(partition) - 1))
    start = int(sum(partition[:scale_idx]))
    length = int(partition[scale_idx])

    # Sample a handful of coefficients in this scale and pick the one whose atom
    # has energy centroid closest to the image center. For the coarsest scale,
    # bias candidates away from extreme corners by skipping the first few indices.
    num_candidates = min(32, max(1, length))
    if scale_idx == 0:
        candidate_offsets = np.linspace(
            max(2, length // 8),
            max(2, length - 1),
            num=num_candidates,
            dtype=int,
        )
    else:
        candidate_offsets = np.linspace(
            0, max(0, length - 1), num=num_candidates, dtype=int
        )
    candidate_indices = start + candidate_offsets

    center = np.array([(img_shape[0] - 1) / 2.0, (img_shape[1] - 1) / 2.0])
    best_atom = None
    best_dist = np.inf

    for idx in candidate_indices:
        coeffs = np.zeros(coeff_len, dtype=complex)
        coeffs[int(idx)] = 1.0
        atom = np.asarray((F.H @ coeffs)).real

        w = np.abs(atom)
        total = w.sum()
        if total <= 0:
            continue
        grid = np.indices(atom.shape)
        centroid = np.array([(w * grid[0]).sum() / total, (w * grid[1]).sum() / total])
        dist = np.linalg.norm(centroid - center)
        if dist < best_dist:
            best_dist = dist
            best_atom = atom

    if best_atom is None:
        # fallback to mid coefficient if all candidates were zero
        mid = start + max(0, length // 2)
        coeffs = np.zeros(coeff_len, dtype=complex)
        coeffs[mid] = 1.0
        best_atom = np.asarray((F.H @ coeffs)).real

    return best_atom


# ----------------------------------------------------------------------------- #
# Wavelet atoms
# ----------------------------------------------------------------------------- #


def _empty_wavelet_coeffs(shape, level):
    coeffs = pywt.wavedec2(
        np.zeros(shape, dtype=float),
        wavelet=wavelet,
        mode=wavelet_mode,
        level=level,
    )
    return coeffs


def wavelet_atom(shape, which: str, level: int) -> np.ndarray:
    """
    Reconstruct a single wavelet coefficient.

    which: "approx", "detail_coarse", "detail_fine"
    """
    coeffs = _empty_wavelet_coeffs(shape, level=level)
    # zero everything
    coeffs = list(coeffs)
    for i in range(1, len(coeffs)):
        cH, cV, cD = coeffs[i]
        coeffs[i] = (np.zeros_like(cH), np.zeros_like(cV), np.zeros_like(cD))

    if which == "approx":
        cA = np.zeros_like(coeffs[0])
        center = tuple(int(s // 2) for s in cA.shape)
        cA[center] = 1.0
        coeffs[0] = cA
    elif which == "detail_coarse":
        # highest-level details are at index 1
        cH, cV, cD = coeffs[1]
        center = tuple(int(s // 2) for s in cH.shape)
        cH[center] = 1.0
        coeffs[1] = (cH, cV, cD)
    elif which == "detail_fine":
        # finest details are at the end
        cH, cV, cD = coeffs[-1]
        center = tuple(int(s // 2) for s in cH.shape)
        cH[center] = 1.0
        coeffs[-1] = (cH, cV, cD)
    else:
        raise ValueError(f"Unknown wavelet atom type: {which}")

    atom = pywt.waverec2(coeffs, wavelet=wavelet, mode=wavelet_mode)
    return np.asarray(atom)


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #


def main():
    # MSWPT setup
    dyadic = DyadicDecomposition(num_levels, N, num_boxes_levels, box_aspect_ratio)
    wpt = MSWPT(dyadic, redundancy, windowing)

    # Curvelet setup
    F = cl.FDCT2D(dims=N)

    # Wavelet setup
    wavelet_levels = 3  # aligns with num_levels for comparison

    mswpt_atoms = [
        mswpt_atom(wpt, level=0),
        mswpt_atom(wpt, level=1),
        mswpt_atom(wpt, level=2),
    ]
    curvelet_atoms = [
        curvelet_atom(F, scale_idx=0, img_shape=N),
        curvelet_atom(F, scale_idx=1, img_shape=N),
        curvelet_atom(F, scale_idx=2, img_shape=N),
    ]
    wavelet_atoms = [
        wavelet_atom(N, "approx", level=wavelet_levels),
        wavelet_atom(N, "detail_coarse", level=wavelet_levels),
        wavelet_atom(N, "detail_fine", level=wavelet_levels),
    ]

    atoms = [mswpt_atoms, curvelet_atoms, wavelet_atoms]
    row_titles = ["MSWPT", "Curvelets", f"Wavelets({wavelet})"]
    col_titles = ["Coarse", "Mid", "Fine"]

    fig, axes = plt.subplots(
        nrows=3, ncols=3, figsize=(10, 10), constrained_layout=True
    )

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            atom = atoms[i][j]
            vmax = np.max(np.abs(atom)) + 1e-12
            ax.imshow(atom, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(col_titles[j])
            if j == 0:
                ax.set_ylabel(row_titles[i], rotation=90, fontsize=10, weight="bold")

    outpath = PLOT_DIR / "frames_grid.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved grid to {outpath}")


if __name__ == "__main__":
    main()
