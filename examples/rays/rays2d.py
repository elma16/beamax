#!/usr/bin/env python
# coding: utf-8



"""
Ray-tracing diagnostic for the Gaussian beam Hamiltonian.
"""
import jax.numpy as jnp
import jax
import diffrax
from pathlib import Path
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from _velocitymaps import peaks_function_2d, gaussian_bump_2d, vertical_gradient_2d
from _curves import circle_curve

from beamax import plotter, utils
from beamax.gb import gb_utils, gb_solvers
from beamax.plotter import use_beamax_style

jax.config.update("jax_enable_x64", True)

ROOT_DIR = utils.detect_root()
PLOT_DIR = Path(ROOT_DIR / "plots")
DATA_DIR = Path(ROOT_DIR / "data")
PROF_DIR = Path(ROOT_DIR / "profiler")
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
PROF_DIR.mkdir(exist_ok=True)

import matplotlib.pyplot as plt

use_beamax_style()

pltgb = plotter.PlotHelper()

b = 90
d = 2
# N = jnp.array([128] * d)
N = jnp.array([128, 256])
xmax = jnp.array([1, 2])
# xmax = jnp.array([1] * d)
dx = xmax / N
domain_size = xmax
x_linspace = [jnp.linspace(0, xmax[i], N[i]) for i in range(d)]

XY = jnp.stack(
    jnp.meshgrid(*x_linspace, indexing="ij"),
    axis=-1,
)

key = jax.random.PRNGKey(0)
c_vals = jax.random.uniform(key, shape=N) * 2
sigma = 1.0
c_vals_smooth = gaussian_filter(c_vals, sigma)

# c_vel = gb_utils.Interpolator(
#     x_linspace, c_vals_smooth, interpolator_class=Interpolator2D, method="cubic2"
# )

pth = Path(DATA_DIR / "Logo_KIT.png")
ucl_vals = jnp.rot90(jnp.mean(plt.imread(pth), axis=-1), -1)
ucl_vals = ucl_vals[20:-20, 60:-60]
ucl_vals = (ucl_vals - ucl_vals.min()) / (ucl_vals.max() - ucl_vals.min())
ucl_vals = 0.4 + ucl_vals / ucl_vals.max()

sigma = 0.0
ucl_vals_smooth = gaussian_filter(ucl_vals, sigma)

ucl_linspace = [jnp.linspace(0, xmax[i], ucl_vals.shape[i]) for i in range(d)]

c_vel = utils.make_c_function_from_grid(c_vals, spacing=dx, origin=(0.0, 0.0))

ts = jnp.linspace(0, 1, 100)
periodic = (False,) * d
mode = jnp.ones((b, 1))
curve_points = jnp.linspace(0, 1, b)

x0 = circle_curve(curve_points, radius=0.01, center=(0.5, 1.5))

if b == 1:
    tangents = jnp.array([[1.0, 0.0]])
else:
    tangents = jnp.gradient(x0, axis=0)

p0 = jnp.stack([-tangents[:, 1], tangents[:, 0]], axis=-1)
p0 = p0 / jnp.linalg.norm(p0, axis=1, keepdims=True)

a0 = jnp.ones((b, 1)) * 0.1
alpha0 = jnp.ones((b, d)) * 1j
ω0 = jnp.ones((b,)) * 100
lam = 0


def generate_complex_positive_definite_matrix(b, d):
    key = jax.random.PRNGKey(0)

    A = jax.random.uniform(key, shape=(b, d, d)) * 5
    real_part = jnp.einsum("bij,bkj->bik", A, A)

    key, _ = jax.random.split(key)
    B = jax.random.normal(key, shape=(b, d, d)) * 0.5
    imag_part = jnp.einsum("bij,bkj->bik", B, B)
    M0 = real_part + 1j * imag_part
    return M0


# alpha0 = None
M0 = None
# M0 = generate_complex_positive_definite_matrix(b, d)

M0 = gb_utils.prepare_M0(alpha0, M0)
is_M0_diagonal = gb_utils.is_diagonal(M0)

solver = gb_solvers.solve_ODE_base

solver_config = gb_solvers.SolverConfig(
    solver=diffrax.Tsit5(),
    max_steps=4096,  # Increase max steps
    rtol=1e-5,  # Tighter tolerance
    pcoeff=0.1,  # Modified PID coefficients
    icoeff=0.3,
    dcoeff=0.0,
)

print("Is M0 diagonal?", is_M0_diagonal)

map_funcs = [peaks_function_2d, gaussian_bump_2d, vertical_gradient_2d]
map_names = ["Peaks\nFunction", "Gaussian\nBump", "Vertical\nGradient"]


# Function to compute gradient and Hessian norms
def compute_gradient_and_hessian(c_func, XY):
    grad_c = jax.vmap(jax.vmap(jax.grad(c_func)))
    hessian_c = jax.vmap(jax.vmap(jax.hessian(c_func)))

    c_vals = jax.vmap(jax.vmap(c_func))(XY)
    grad_vals = grad_c(XY)
    grad_norms = jnp.linalg.norm(grad_vals, axis=-1)

    hess_vals = hessian_c(XY)
    hess_norms = jnp.sqrt(jnp.sum(hess_vals**2, axis=(-2, -1)))

    return c_vals, grad_norms, hess_norms


# Plot velocity map, gradient norms, and Hessian norms
fig = plt.figure(figsize=(15, 15))
gs = GridSpec(len(map_funcs), 3, wspace=0.3, hspace=0.3)

for i, (c_func, map_name) in enumerate(zip(map_funcs, map_names)):
    c_vals, grad_norms, hess_norms = compute_gradient_and_hessian(c_func, XY)

    ax1 = fig.add_subplot(gs[i, 0])
    im1 = ax1.imshow(
        c_vals.T, extent=[0, xmax[0], 0, xmax[1]], origin="lower", cmap="viridis"
    )
    ax1.set_title(f"{map_name} - Velocity Map")
    plt.colorbar(im1, ax=ax1)

    ax2 = fig.add_subplot(gs[i, 1])
    im2 = ax2.imshow(
        grad_norms.T, extent=[0, xmax[0], 0, xmax[1]], origin="lower", cmap="plasma"
    )
    ax2.set_title(f"{map_name} - Gradient Norm")
    plt.colorbar(im2, ax=ax2)

    ax3 = fig.add_subplot(gs[i, 2])
    im3 = ax3.imshow(
        hess_norms.T, extent=[0, xmax[0], 0, xmax[1]], origin="lower", cmap="inferno"
    )
    ax3.set_title(f"{map_name} - Hessian Norm")
    plt.colorbar(im3, ax=ax3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "velocity_map_grad_hess_norms.png", dpi=300, bbox_inches="tight")
plt.show()


def plot_rays_and_map(ax, XY, c_func, xt, t_idx, add_colorbar=False):
    # Plot speed of sound map
    c_vals = jax.vmap(jax.vmap(c_func))(XY)
    im = ax.imshow(
        c_vals.T,
        extent=[0, xmax[0], 0, xmax[1]],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    if add_colorbar:
        plt.colorbar(im, ax=ax)

    # Plot rays up to time t_idx
    for i in range(xt.shape[0]):
        ax.plot(xt[i, :t_idx, 0], xt[i, :t_idx, 1], "r-", linewidth=1, alpha=0.5)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Set axis limits to match domain
    ax.set_xlim(0, xmax[0])
    ax.set_ylim(0, xmax[1])


# Create figure with a wider grid to accommodate labels
fig = plt.figure(figsize=(15, 15))
gs = GridSpec(3, 4, width_ratios=[0.1, 1, 1, 1])

# Time indices for visualization
t_indices = [0, 50, 99]
t_labels = [f"t = {ts[t_idx]:.2f}" for t_idx in t_indices]

# Generate ray traces for each map
rays_data = []
for c_func in map_funcs:
    xt, pt, mt, at = solver(x0, p0, M0, a0, mode, ts, c_func, lam, None)
    rays_data.append(xt)

# Create all plots
for i, (c_func, map_name, xt) in enumerate(zip(map_funcs, map_names, rays_data)):
    # Add map name in the leftmost column
    ax_label = fig.add_subplot(gs[i, 0])
    ax_label.text(0.5, 0.5, map_name, rotation=0, ha="center", va="center")
    ax_label.axis("off")

    for j, t_idx in enumerate(t_indices):
        ax = fig.add_subplot(gs[i, j + 1])

        # Only add time labels to top row
        if i == 0:
            ax.set_title(t_labels[j])

        # Only add colorbar to rightmost column
        im = plot_rays_and_map(ax, XY, c_func, xt, t_idx + 1, add_colorbar=(j == 2))

plt.tight_layout()
plt.savefig(PLOT_DIR / "rays_and_velocity_maps.png", dpi=300, bbox_inches="tight")
plt.show()
