#!/usr/bin/env python
# coding: utf-8



"""
Render the MSWPT frame atoms for a given dyadic decomposition, illustrating the multiscale tiling.
"""
import jax.numpy as jnp
import jax as jax
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.widgets import Slider
import numpy as np
from time import time

from beamax import plotter, utils
from beamax.decomposition import DyadicDecomposition
from beamax.transforms import MSWPT

ROOT_DIR = utils.detect_root()
PLOT_DIR = Path(ROOT_DIR / "plots")
DATA_DIR = Path(ROOT_DIR / "data")
PROF_DIR = Path(ROOT_DIR / "profiler")
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
PROF_DIR.mkdir(exist_ok=True)

from beamax.plotter import use_beamax_style
use_beamax_style()

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

d = 2
N = (64,) * d
box_aspect_ratio = (1,) * d
num_levels = 2
num_boxes_level = (4, 8)

windowing = "rectangular"
input_type = "spatial"
output_type = "spatial"
redundancy = 2

dyadic_decomp = DyadicDecomposition(num_levels, N, num_boxes_level, box_aspect_ratio)
wpt_img = MSWPT(dyadic_decomp, redundancy, windowing)

total_coeffs = jnp.prod(redundancy * jnp.array(N))
coeffs = jnp.zeros((total_coeffs,))


def _plot_quadrants(ax, array, fixed_coord, cmap, rstride=1, cstride=1):
    nx, ny, nz = array.shape
    idx = {
        "x": (nx // 2, slice(None), slice(None)),
        "y": (slice(None), ny // 2, slice(None)),
        "z": (slice(None), slice(None), nz // 2),
    }[fixed_coord]
    plane = array[idx]
    n0, n1 = plane.shape
    q = [
        plane[: n0 // 2, : n1 // 2],
        plane[: n0 // 2, n1 // 2 :],
        plane[n0 // 2 :, : n1 // 2],
        plane[n0 // 2 :, n1 // 2 :],
    ]
    vmin, vmax = array.min(), array.max()
    norm = (q[0] - vmin) / (vmax - vmin + 1e-12)  # noqa
    cmap = plt.get_cmap(cmap)

    for i, quadrant in enumerate(q):
        fc = cmap((quadrant - vmin) / (vmax - vmin + 1e-12))
        if fixed_coord == "x":
            Y, Z = np.mgrid[0 : ny // 2, 0 : nz // 2]
            X = (nx // 2) * np.ones_like(Y)
            Y += (i // 2) * (ny // 2)
            Z += (i % 2) * (nz // 2)
            ax.plot_surface(
                X, Y, Z, rstride=rstride, cstride=cstride, facecolors=fc, shade=False
            )
        elif fixed_coord == "y":
            X, Z = np.mgrid[0 : nx // 2, 0 : nz // 2]
            Y = (ny // 2) * np.ones_like(X)
            X += (i // 2) * (nx // 2)
            Z += (i % 2) * (nz // 2)
            ax.plot_surface(
                X, Y, Z, rstride=rstride, cstride=cstride, facecolors=fc, shade=False
            )
        else:  # 'z'
            X, Y = np.mgrid[0 : nx // 2, 0 : ny // 2]
            Z = (nz // 2) * np.ones_like(X)
            X += (i // 2) * (nx // 2)
            Y += (i % 2) * (ny // 2)
            ax.plot_surface(
                X, Y, Z, rstride=rstride, cstride=cstride, facecolors=fc, shade=False
            )


def _render_3d_slices(ax, arr3d, cmap="viridis", stride=2):
    ax.cla()
    nx, ny, nz = arr3d.shape
    # Try to set aspect for Matplotlib versions that support 3‑tuple; otherwise skip
    try:
        ax.set_box_aspect((nx, ny, nz))  # works on newer mplot3d
    except Exception:
        pass
    _plot_quadrants(ax, arr3d, "x", cmap, rstride=stride, cstride=stride)
    _plot_quadrants(ax, arr3d, "y", cmap, rstride=stride, cstride=stride)
    _plot_quadrants(ax, arr3d, "z", cmap, rstride=stride, cstride=stride)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)


if d == 1:

    def plotter(val, coeff_idx):
        ax.clear()
        ax.plot(val.real)
        ax.set_title(f"Coefficient Index: {coeff_idx}")
elif d == 2:

    def plotter(val, coeff_idx):
        ax.clear()
        ax.imshow(val.real)
        ax.set_title(f"Coefficient Index: {coeff_idx}")
elif d == 3:

    def plotter(val, coeff_idx, stride=2, cmap="viridis"):
        arr = np.asarray(val.real)  # host transfer from JAX
        _render_3d_slices(ax, arr, cmap=cmap, stride=stride)
        ax.set_title(f"Coefficient Index: {coeff_idx}")


if d < 3:
    fig, ax = plt.subplots()
else:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

plt.subplots_adjust(bottom=0.25)
plt.title("Wavelet Coefficient Visualization")


# @jax.jit
def inv(coeffs):
    return wpt_img.inverse(coeffs, output_type)


def update(val):
    coeff_idx = int(slider.val)
    coeffs = jnp.zeros((total_coeffs,))
    coeffs = coeffs.at[coeff_idx].set(1.0)

    t1 = time()
    f_rect = inv(coeffs)
    t2 = time()
    print(f"Inverse transform took {t2 - t1:.4f} seconds")

    shapes = utils.compute_coeff_shapes(
        wpt_img.dyadic_decomp,
        wpt_img.redundancy,
        jnp.arange(wpt_img.dyadic_decomp.num_levels),
    )
    nn_level, nn_indices = utils.find_tensor_and_multiindex(
        jnp.array([coeff_idx]), shapes
    )
    cumsum_boxes = jnp.r_[0, jnp.cumsum(dyadic_decomp.num_boxes_ndim)]
    idx = nn_indices[0, :] + cumsum_boxes[nn_level]
    level = utils.find_level(wpt_img.dyadic_decomp, idx)
    k = nn_indices[1:, :]

    # level = utils.find_level(dyadic_decomp, idx)
    # k_guess = rearrange(k_guess, "d 1 -> d")
    # cumsum_boxes = jnp.r_[0, jnp.cumsum(dyadic_decomp.num_boxes_ndim)]
    # idx_guess = idx[0] + cumsum_boxes[level]
    # k_guess = idx[1:]

    print(
        f"f_rect.shape = {f_rect.shape}, coeff_idx = {coeff_idx}, level = {level}, idx = {idx}, k = {k}"
    )
    plotter(f_rect, coeff_idx)
    plt.draw()


slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(
    slider_ax, "Coefficient Index", 0, total_coeffs - 1, valinit=0, valstep=1
)
slider.on_changed(update)
update(0)
plt.show()

from beamax.transforms import compute_frames
from beamax import utils

"""
make a four by four grid of subplots for different frame elements and their corresponding level, idx, k
"""
KXY = dyadic_decomp.fourier_meshgrid

num_boxes_total = int(jnp.sum(dyadic_decomp.num_boxes_ndim))
idxs = jnp.linspace(0, num_boxes_total - 1, 16).round().astype(int)
ks = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]])
ks = jnp.tile(ks, (4, 1))

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for n, (idx_i, k_i) in enumerate(zip(idxs, ks)):
    idx = jnp.array([int(idx_i)])
    k = jnp.array(k_i)
    phi_kx = compute_frames(dyadic_decomp, idx, k, KXY, redundancy, windowing)

    level = utils.find_level(dyadic_decomp, idx)
    # print(f"level {level}")
    ax = axes[n // 4, n % 4]
    ax.imshow(np.asarray(phi_kx.real))
    ax.set_title(
        f"L{level}  idx={int(idx_i)}  k={tuple(int(x) for x in k_i)}", fontsize=8
    )
    ax.axis("off")
plt.tight_layout()
out_path = PLOT_DIR / "frames_4x4.png"
plt.savefig(out_path, dpi=200)
print(f"Saved {out_path}")
plt.show()
