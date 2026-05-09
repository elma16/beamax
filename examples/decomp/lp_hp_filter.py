#!/usr/bin/env python
# coding: utf-8



"""
Low-pass / high-pass frequency separation built from MSWPT filters. Also part of the CI example smoke suite.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

from beamax.decomposition import DyadicDecomposition
from beamax import plotter, utils
from beamax.transforms import compute_frames, compute_gh_filters
from beamax.plotter import use_beamax_style

jax.config.update("jax_enable_x64", True)

pltgb = plotter.PlotHelper()

ROOT_DIR = utils.detect_root()
PLOT_DIR = Path(ROOT_DIR / "plots")
DATA_DIR = Path(ROOT_DIR / "data")
PROF_DIR = Path(ROOT_DIR / "profiler")
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
PROF_DIR.mkdir(exist_ok=True)

use_beamax_style()

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

############################
###### 1D Example ##########
############################

N = (512,)
d = len(N)
dx = (1e-4,) * d
box_aspect_ratio = (1,) * d
num_levels = 2
num_boxes_levels = tuple([2 ** (level + 2) for level in range(num_levels)])
redundancy = 2
windowing = "rectangular_mirror"

decomp = DyadicDecomposition(num_levels, N, num_boxes_levels, box_aspect_ratio)

gs, hs = compute_gh_filters(decomp, redundancy, windowing)

# plot single filter
plt.plot(gs[0], label=r"$g_{\ell,j}(\xi)$")
plt.plot(hs[0], label=r"$h_{\ell,j}(\xi)$")
plt.legend()
plt.show()

gs_sum = jnp.sum(gs, axis=0)
hs_sum = jnp.sum(hs, axis=0)
gh_sum = jnp.sum(gs * hs, axis=0)

# plot sum of filters
plt.plot(gs_sum, label=r"$\sum_{(\ell,j,k)}g_{\ell,j}(\xi)$")
plt.plot(hs_sum, label=r"$\sum_{(\ell,j,k)}h_{\ell,j}(\xi)$")
plt.plot(gh_sum, label=r"$\sum_{(\ell,j,k)}g_{\ell,j}(\xi)h_{\ell,j}(\xi) = 1$")
plt.legend()
plt.savefig(PLOT_DIR / "gh_sum_plot.png", dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()

# ############################
# ###### 2D Example ##########
# ############################

N = (128, 128)
d = len(N)
dx = (1e-4,) * d
box_aspect_ratio = (1,) * d
num_levels = 2
num_boxes_levels = tuple([2 ** (level + 2) for level in range(num_levels)])
redundancy = 2
windowing = "rectangular"

decomp = DyadicDecomposition(num_levels, N, num_boxes_levels, box_aspect_ratio)

gs, hs = compute_gh_filters(decomp, redundancy, windowing)

# Plot the filters
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(gs[0], cmap="viridis")
axs[0].set_title(r"$g_{\ell,j}(\xi)$")
axs[1].imshow(hs[0], cmap="viridis")
axs[1].set_title(r"$h_{\ell,j}(\xi)$")
plt.show()

gs_sum = jnp.sum(gs, axis=0)
hs_sum = jnp.sum(hs, axis=0)

# Plot the sum of the filters
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
im0 = axs[0].imshow(gs_sum, cmap="viridis")
axs[0].set_title(r"$\sum_{(\ell,j,k)}g_{\ell,j}(\xi)$")
cbar0 = fig.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(hs_sum, cmap="viridis")
axs[1].set_title(r"$\sum_{(\ell,j,k)}h_{\ell,j}(\xi)$")
cbar1 = fig.colorbar(im1, ax=axs[1])

plt.tight_layout()
plt.savefig(PLOT_DIR / "gh_sum_plot_2d.png", dpi=300, bbox_inches="tight")
plt.show()

####################################
###### Filters vs GB ###############
####################################

KXY = decomp.fourier_meshgrid

pltgb.plot_centers(decomp.centres_ndim)

phi = 0

ls = [(34, jnp.array([20, 10])), (6, jnp.array([2, 5]))]

# ls = [(34, jnp.array([20, 10]))]
# ls = [(6, jnp.array([8, 3]))]

for idx_k in ls:
    idx = idx_k[0]
    k = idx_k[1]
    frame = compute_frames(decomp, idx, k, KXY, redundancy, windowing)
    phi += frame / jnp.max(frame)

plt.imshow(
    jnp.real(phi),
    origin="lower",
    extent=(-0.5 * N[0], 0.5 * N[0] - 1, -0.5 * N[1], 0.5 * N[1] - 1),
    cmap="viridis",
)
plt.colorbar()
plt.savefig(PLOT_DIR / "frame_plot.png", dpi=300, bbox_inches="tight")
plt.show()
