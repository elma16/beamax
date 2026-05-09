#!/usr/bin/env python
# coding: utf-8



"""
Reconstruction error of the MSWPT forward+inverse pipeline as a function of box count and redundancy.
"""
import jax
import jax.numpy as jnp
from beamax.decomposition import DyadicDecomposition
from beamax import transforms, plotter, utils
from beamax.plotter import use_beamax_style
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import erf

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

ROOT_DIR = utils.detect_root()
PLOT_DIR = Path(ROOT_DIR / "plots")
DATA_DIR = Path(ROOT_DIR / "data")
PROF_DIR = Path(ROOT_DIR / "profiler")
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
PROF_DIR.mkdir(exist_ok=True)

use_beamax_style()

"""
1. The errors of the frame is scale dependent.
2. The errors of the frame is not frequency dependent.
3. Assuming the windowing is a rectangular window, the error is bounded by
    (sqrt(2 * pi)/8)^d * (1 - erf(sqrt(2)))^d

4. Observation. The error seems to be dependent on the length of the box at each scale.

N.B: This error bound does not apply for the rectangular_mirror window.
"""

pltgb = plotter.PlotHelper()

d = 2
N = (256,) * d
box_aspect_ratio = (1,) * d
num_levels = 3
num_boxes_outer_level = tuple([2 ** (i + 2) for i in range(num_levels)])
redundancy = 2

dyadic_decomp = DyadicDecomposition(
    num_levels, N, num_boxes_outer_level, box_aspect_ratio
)

box_lengths = dyadic_decomp.box_lengths

# pltgb.plot_centers(dyadic_decomp.centres_ndim)

KXY = dyadic_decomp.fourier_meshgrid
k = jnp.ones((d,))


def compute_error(idx):
    omega = jnp.linalg.norm(dyadic_decomp.centres_ndim[idx])

    idx = jnp.array([idx])

    phi_r = transforms.compute_frames(
        dyadic_decomp, idx, k, KXY, redundancy, "rectangular"
    )
    phi_rm = transforms.compute_frames(
        dyadic_decomp, idx, k, KXY, redundancy, "rectangular_mirror"
    )
    phi_n = transforms.compute_frames(dyadic_decomp, idx, k, KXY, redundancy, "none")

    error_rn = jnp.linalg.norm(phi_r - phi_n) ** 2
    error_rm = jnp.linalg.norm(phi_rm - phi_n) ** 2
    return omega, error_rn, error_rm


indices = jnp.arange(jnp.sum(dyadic_decomp.num_boxes_ndim))

omegas, errors_rn, errors_rm = jax.vmap(compute_error)(indices)

boxes_cumsum = jnp.concatenate([jnp.array([0]), dyadic_decomp.num_boxes_ndim_cumsum])

for scale in range(num_levels):
    errors_scale_rn = errors_rn[boxes_cumsum[scale] : boxes_cumsum[scale + 1]]
    errors_scale_rm = errors_rm[boxes_cumsum[scale] : boxes_cumsum[scale + 1]]
    print(
        f"Scale {scale} errors rect: {errors_scale_rn[0]}, errors rect mirror {errors_scale_rm[0]}, box lengths: {box_lengths[scale]}"
    )
    assert jnp.allclose(errors_scale_rn, errors_scale_rn[0], atol=1e-16)

errors_allscales = errors_rn[boxes_cumsum[:-1]]

# upper bound for the error for a rectangular window
error_bound = (jnp.sqrt(2 * jnp.pi) / (8 * redundancy)) ** d * (
    1 - erf(jnp.sqrt(2)) ** d
)

plt.hlines(
    y=error_bound,
    xmin=jnp.min(omegas),
    xmax=jnp.max(omegas),
    color="red",
    linestyle="--",
    label="Error bound",
)
plt.loglog(omegas, errors_rn, ".", label="Rectangular window")
plt.loglog(omegas, errors_rm, ".", label="Rectangular mirror")
plt.xlabel(r"$\omega$")
plt.ylabel("$L^{2}_2$ Error")
plt.title(f"Frame element cutoff error in {d}D for {num_levels} levels")
plt.legend()
plt.savefig(PLOT_DIR / f"frame_error_{d}D.png", dpi=300, bbox_inches="tight")
plt.show()

for d in range(1, 4):
    error_bound = (jnp.sqrt(2 * jnp.pi) / (8 * redundancy)) ** d * (
        1 - erf(jnp.sqrt(2)) ** d
    )
    print(f"d: {d}, error bound: {error_bound}")
