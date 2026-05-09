#!/usr/bin/env python
# coding: utf-8



"""
Single Gaussian beam wave-packet transform diagnostic.
"""
import jax.numpy as jnp
import jax
from jax import vmap
import numpy as np

from beamax import geometry, plotter, utils
from beamax.decomposition import DyadicDecomposition
from beamax.transforms import MSWPT
from beamax.gb import core, gb_utils, gb_solvers
from pathlib import Path

from time import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

"""
Example script to show how the GB changes over time.

Specifically, we can show how keeping the M0 symmetric and the imaginary part positive definite gives some funky results.

1. shows the gb at snapshots in time
2. real and imag parts of GB in spatial and fourier domain
3. eigenvalues of the hessian matrix
4. wavefront set

"""

ROOT_DIR = utils.detect_root()
PLOT_DIR = Path(ROOT_DIR / "plots")
PLOT_DIR.mkdir(exist_ok=True)

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)
jax.config.update("jax_enable_x64", True)
pltgb = plotter.PlotHelper()

b = 2
d = 2
N = (256,) * d
dx = (10 / N[0],) * d
periodic = (True,) * d
box_aspect_ratio = (1,) * d
num_levels = 2


def c(x):
    return 1 + 0 * x[..., 0]


start = tuple([2 ** (level + 2) for level in range(num_levels)])
windowing = "rectangular"
input_type = "spatial"
output_type = "spatial"
redundancy = 2

cfl = 0.3
domain = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)
space, fourier = domain.generate_meshgrid()
XY = domain.grid

# ts = domain.generate_time_domain()
ts = jnp.linspace(0, 10, 100)
dyadic_decomp = DyadicDecomposition(num_levels, N, start, box_aspect_ratio)
wpt = MSWPT(dyadic_decomp, redundancy, windowing)

domain_size = domain.grid_size

mode = jnp.array([1, -1])
x0 = jnp.zeros((b, d)) + 0.5 * domain_size
p0 = jnp.ones((b, d))
p0 = p0.at[:, 1].set(0)
p0 = p0 / jnp.linalg.norm(p0, axis=-1, keepdims=True)
p0 = p0
a0 = jnp.ones((b,))
alpha0 = jnp.ones((b, d)) * 1j
M0 = None
ω0 = jnp.ones((b,)) * 50
lam = 0

M0 = gb_utils.prepare_M0(alpha0, M0)
is_M0_diagonal = gb_utils.is_diagonal(M0)

solver = gb_solvers.solve_ODE_base
# solver = gb_solvers.solve_hom_diag
solver_config = None

print("Is M0 diagonal?", is_M0_diagonal)

t1 = time()
(xt, pt, mt, at) = solver(x0, p0, M0, a0, mode, ts, c, lam, solver_config)

u0 = core.compute_gaussian_beam(
    x0,
    p0,
    M0,
    a0,
    ω0,
    mode,
    c,
    lam,
    ts,
    XY,
    domain_size,
    jnp.array(periodic),
    solver,
    solver_config,
)
u0 = jnp.sum(u0, axis=-1)


def apply_mswpt_batch(mswpt, data, input_type="spatial"):
    """Apply the forward MSWPT transform to a batch of data along the first dimension."""

    # Create a vmapped version of the forward method
    forward_vmapped = vmap(
        lambda slice_data: mswpt.convert_to_array(mswpt.forward(slice_data, input_type))
    )

    # Apply to the batch
    return forward_vmapped(data)


coeffs = apply_mswpt_batch(wpt, u0, input_type)

print("u0 shape:", u0.shape)
print("coeffs shape:", coeffs.shape)


def plot_gb_and_coeffs(u0, coeffs, space, ts, dyadic_decomp):
    """
    Create an interactive plot showing the real part of the Gaussian Beam at
    snapshots in time alongside the relevant slice of coefficients.

    Args:
        u0: Gaussian beam data (time, x, y)
        coeffs: MSWPT coefficients (time, 2*N[0], 2*N[1])
        space: Spatial domain coordinates
        ts: Time points
        dyadic_decomp: Dyadic decomposition for coefficient plotting
    """
    # Convert JAX arrays to NumPy for matplotlib compatibility
    u0_np = np.array(u0)
    coeffs_np = np.array(coeffs)
    ts_np = np.array(ts)
    u0_real = np.real(u0_np)

    # Stable scaling for the GB plot (symmetric about 0)
    gb_max = float(np.max(np.abs(u0_real)))
    if gb_max == 0:
        gb_max = 1.0

    # Stable log scaling for coefficient plots
    coeffs_mag = np.abs(coeffs_np)
    coeffs_max = float(np.max(coeffs_mag))
    if coeffs_max == 0:
        coeffs_max = 1.0
    coeffs_min = coeffs_max * 1e-2
    coeffs_norm = mcolors.LogNorm(vmin=coeffs_min, vmax=coeffs_max)
    coeffs_mag_plot = np.maximum(coeffs_mag, coeffs_min)

    # Create figure with GridSpec
    fig = plt.figure(figsize=(17, 8))
    gs = gridspec.GridSpec(
        3,
        4,  # 3 rows, 4 columns (add coeff colorbar)
        height_ratios=[1, 1, 0.1],
        width_ratios=[1, 0.05, 1, 0.05],
        hspace=0.3,
        wspace=0.4,
    )

    # Create axes
    ax_gb = plt.subplot(gs[0:2, 0])  # Gaussian Beam plot
    cax_gb = plt.subplot(gs[0:2, 1])  # Colorbar for GB
    ax_coeffs = plt.subplot(gs[0:2, 2])  # Coefficients plot
    cax_coeffs = plt.subplot(gs[0:2, 3])  # Colorbar for coeffs
    ax_slider = plt.subplot(gs[2, :])  # Slider takes full width

    # Calculate extents for spatial domain
    spatial_extent = [
        float(np.min(space[0])),
        float(np.max(space[0])),
        float(np.min(space[1])),
        float(np.max(space[1])),
    ]

    # Initial plots
    # Gaussian Beam plot (real part)
    im_gb = ax_gb.imshow(
        u0_real[0, ...].T,
        extent=spatial_extent,
        cmap="RdBu_r",
        origin="lower",
        vmin=-gb_max,
        vmax=gb_max,
    )
    ax_gb.set_title(f"Real Part of GB (t = {ts_np[0]:.2f})")
    ax_gb.set_xlabel("x")
    ax_gb.set_ylabel("y")

    # Add colorbar
    plt.colorbar(im_gb, cax=cax_gb, label="Amplitude")

    # Coefficients plot (log magnitude with dyadic boxes)
    im_coeffs = plotter.plot_mswpt_coeffs(
        ax_coeffs,
        coeffs_np[0, ...],
        dyadic_decomp,
        cutoff_freq=None,
        box_corners=None,
        asymptote=False,
        log_scale=True,
    )
    im_coeffs.set_norm(coeffs_norm)
    im_coeffs.set_data(coeffs_mag_plot[0, ...].T)
    ax_coeffs.set_aspect("equal")
    ax_coeffs.set_xticks([])
    ax_coeffs.set_yticks([])
    ax_coeffs.set_title(f"MSWPT Coefficients (t = {ts_np[0]:.2f})")
    plt.colorbar(im_coeffs, cax=cax_coeffs, label="|coeff|")

    # Create time slider
    slider = Slider(
        ax=ax_slider,
        label="Time",
        valmin=float(ts_np[0]),
        valmax=float(ts_np[-1]),
        valinit=float(ts_np[0]),
        valstep=float(ts_np[1] - ts_np[0]),
    )

    # Update function for slider
    def update(val):
        t_idx = int(np.argmin(np.abs(ts_np - val)))

        # Update Gaussian Beam plot
        im_gb.set_array(u0_real[t_idx, ...].T)

        # Update Coefficients plot
        im_coeffs.set_data(coeffs_mag_plot[t_idx, ...].T)

        # Update title with current time
        t_label = ts_np[t_idx]
        ax_gb.set_title(f"Real Part of GB (t = {t_label:.2f})")
        ax_coeffs.set_title(f"MSWPT Coefficients (t = {t_label:.2f})")

        fig.canvas.draw_idle()

    # Register the update function with the slider
    slider.on_changed(update)
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "gb_and_coeffs_interactive.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


# To use this function, add the following after the coefficients are computed:
plot_gb_and_coeffs(u0, coeffs, space, ts, dyadic_decomp)
