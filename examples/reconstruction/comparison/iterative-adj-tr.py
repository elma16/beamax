#!/usr/bin/env python
# coding: utf-8



"""
Iterative refinement combining MSGB adjoint and time-reversal updates.
"""
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from beamax import geometry, utils
from beamax.solvers.kwave_solver import KWaveSolver
from beamax.plotter import use_beamax_style
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions

try:
    import tqdm.auto as tqdm
except ModuleNotFoundError:

    class _ProgressFallback:
        def __init__(self, iterable):
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix(self, *args, **kwargs):
            return None

        def set_description(self, *args, **kwargs):
            return None

        def update(self, *args, **kwargs):
            return None

        def close(self):
            return None

    class _TqdmFallback:
        @staticmethod
        def tqdm(iterable, *args, **kwargs):
            return _ProgressFallback(iterable)

        @staticmethod
        def trange(*args, **kwargs):
            return _ProgressFallback(range(*args))

    tqdm = _TqdmFallback()

ROOT_DIR = utils.detect_root()
PLOT_DIR = Path(ROOT_DIR / "plots")
DATA_DIR = Path(ROOT_DIR / "data")
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

use_beamax_style()

"""
Improved iterative time-reversal reconstruction with stabilization techniques.
"""

N = (64, 64)
d = len(N)
dx = (1e-4,) * d
cfl = 0.3
periodic = (False,) * d
p0 = jnp.zeros(N).at[10:30, 30:50].set(1)


def c(x):
    return 1500 + 0 * x[..., 0]


domain = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)
ts = domain.generate_time_domain()

sim_opts = SimulationOptions(data_cast="double", smooth_p0=False, save_to_disk=True)
exec_opts = SimulationExecutionOptions(
    is_gpu_simulation=False, delete_data=False, verbose_level=0, show_sim_log=False
)
solver = KWaveSolver(sim_opts, exec_opts)

# Define sensor mask (left boundary)
mask = jnp.zeros(N)
mask = mask.at[:, 0].set(1)
all_sensors = jnp.ones(N)

print("Generating initial measurement data...")
meas = solver.forward(p0, domain, mask, ts)
print("Measurement data generated.")


def compute_step_size(operator_type="adjoint", power_iter=5):
    """
    Estimate optimal step size using power iteration to estimate operator norm.
    """
    # Initialize with random vector
    x = jnp.ones(N)
    x = x / jnp.linalg.norm(x)

    for _ in range(power_iter):
        # Apply forward then adjoint/time-reversal (A^T A)
        y = solver.forward(x, domain, mask, ts)
        if operator_type == "adjoint":
            x_new = solver.adjoint(y.T, domain, all_sensors, mask, ts).T
        else:
            x_new = solver.time_reversal(y.T, domain, all_sensors, mask, ts).T

        # Normalize
        norm = jnp.linalg.norm(x_new)
        x = x_new / norm

    # The spectral radius approximation
    spectral_radius = norm
    # Conservative step size (< 2/spectral_radius for convergence)
    step_size = 1.0 / spectral_radius
    return step_size


def apply_non_negativity(x):
    """Apply non-negativity constraint."""
    return jnp.maximum(x, 0)


def apply_smoothing(x, sigma=0.5):
    """Apply mild Gaussian smoothing for regularization."""
    from scipy.ndimage import gaussian_filter

    return jnp.array(gaussian_filter(x, sigma=sigma))


def compute_error(estimate, ground_truth):
    """Compute relative error."""
    return jnp.linalg.norm(estimate - ground_truth) / jnp.linalg.norm(ground_truth)


# --- 3. Compute Optimal Step Sizes ---
print("Computing optimal step sizes...")
alpha_tr = compute_step_size("time_reversal")
alpha_adj = compute_step_size("adjoint")

# Make adjoint step size more conservative if needed
alpha_adj *= 0.5  # Additional safety factor for adjoint

print(f"Time-reversal step size: {alpha_tr:.4f}")
print(f"Adjoint step size: {alpha_adj:.4f}")

# --- 4. Iterative Reconstruction with Improvements ---

n_iterations = 50  # Increased for better convergence
use_constraints = True  # Enable non-negativity constraint
use_regularization = True  # Enable smoothing regularization
reg_interval = 5  # Apply regularization every N iterations

# Initialize estimates
p_tr_est = jnp.zeros_like(p0)
p_adj_est = jnp.zeros_like(p0)

# Storage for iterations and metrics
tr_iterations = []
adj_iterations = []
tr_errors = []
adj_errors = []
tr_residuals = []
adj_residuals = []

print(f"Starting {n_iterations} iterations of improved reconstruction...")

for i in tqdm.trange(n_iterations, desc="Reconstruction Iterations"):
    # --- Time-Reversal Iteration ---
    # Forward model
    est_meas_tr = solver.forward(p_tr_est, domain, mask, ts)
    residual_tr = meas - est_meas_tr

    # Compute update with step size
    update_tr = solver.time_reversal(residual_tr.T, domain, all_sensors, mask, ts).T
    p_tr_est = p_tr_est + alpha_tr * update_tr

    # Apply constraints
    if use_constraints:
        p_tr_est = apply_non_negativity(p_tr_est)

    # Apply regularization periodically
    if use_regularization and (i + 1) % reg_interval == 0:
        p_tr_est = apply_smoothing(p_tr_est, sigma=0.3)

    # Store iteration and metrics
    tr_iterations.append(p_tr_est.copy())
    tr_errors.append(compute_error(p_tr_est, p0))
    tr_residuals.append(jnp.linalg.norm(residual_tr))

    # --- Adjoint Iteration ---
    # Forward model
    est_meas_adj = solver.forward(p_adj_est, domain, mask, ts)
    residual_adj = meas - est_meas_adj

    # Compute update with step size
    update_adj = solver.adjoint(residual_adj.T, domain, all_sensors, mask, ts).T
    p_adj_est = p_adj_est + alpha_adj * update_adj

    # Apply constraints
    if use_constraints:
        p_adj_est = apply_non_negativity(p_adj_est)

    # Apply regularization periodically
    if use_regularization and (i + 1) % reg_interval == 0:
        p_adj_est = apply_smoothing(p_adj_est, sigma=0.3)

    # Store iteration and metrics
    adj_iterations.append(p_adj_est.copy())
    adj_errors.append(compute_error(p_adj_est, p0))
    adj_residuals.append(jnp.linalg.norm(residual_adj))

print("Iterative reconstruction complete.")

# --- 5. Create Enhanced Visualization ---

# Create figure with animation and convergence plots
fig = plt.figure(figsize=(20, 10))

# Top row: Reconstructions
ax1 = plt.subplot(2, 3, 1)
ax2 = plt.subplot(2, 3, 2)
ax3 = plt.subplot(2, 3, 3)

# Bottom row: Convergence metrics
ax4 = plt.subplot(2, 2, 3)
ax5 = plt.subplot(2, 2, 4)

# Shared normalization from ground truth
vmin, vmax = float(jnp.min(p0)), float(jnp.max(p0))

# Ground truth
ax1.set_title("Ground Truth")
im1 = ax1.imshow(p0, cmap="viridis", vmin=vmin, vmax=vmax)
ax1.set_xticks([])
ax1.set_yticks([])

# Overlay sensors as red triangles (mask is a JAX array; convert for numpy)
ys, xs = np.where(np.asarray(mask) > 0)
ax1.plot(
    xs,
    ys,
    "^",
    linestyle="None",
    markerfacecolor="none",
    markeredgecolor="red",
    markersize=6,
    alpha=0.95,
    zorder=5,
)

# Reconstructions (use same vmin/vmax)
ax2.set_title("Time-Reversal (Iteration 0)")
im2 = ax2.imshow(tr_iterations[0], cmap="viridis", vmin=vmin, vmax=vmax)
ax2.set_xticks([])
ax2.set_yticks([])

ax3.set_title("Adjoint Method (Iteration 0)")
im3 = ax3.imshow(adj_iterations[0], cmap="viridis", vmin=vmin, vmax=vmax)
ax3.set_xticks([])
ax3.set_yticks([])

divider = make_axes_locatable(ax1)
cax = divider.append_axes("left", size="4%", pad=0.15)  # width, gap
cbar = fig.colorbar(im1, cax=cax)
cbar.set_label("Amplitude (shared scale from ground truth)")
cax.yaxis.set_ticks_position("left")
cax.yaxis.set_label_position("left")

# --- Convergence plots ---
ax4.set_title("Relative Error")
(line1,) = ax4.plot([], [], "b-", label="Time-Reversal")
(line2,) = ax4.plot([], [], "r-", label="Adjoint")
ax4.set_xlabel("Iteration")
ax4.set_ylabel("Relative Error")
ax4.set_xlim(0, n_iterations)
ax4.set_ylim(0, max(max(tr_errors), max(adj_errors)) * 1.1)
ax4.legend()
ax4.grid(True, alpha=0.3)

ax5.set_title("Residual Norm")
(line3,) = ax5.plot([], [], "b-", label="Time-Reversal")
(line4,) = ax5.plot([], [], "r-", label="Adjoint")
ax5.set_xlabel("Iteration")
ax5.set_ylabel("||meas - meas_est||")
ax5.set_xlim(0, n_iterations)
ax5.set_ylim(0, max(max(tr_residuals), max(adj_residuals)) * 1.1)
ax5.legend()
ax5.grid(True, alpha=0.3)

fig.tight_layout()


def update(frame):
    """Updates plots for each frame."""
    # Update reconstructions
    im2.set_data(tr_iterations[frame])
    ax2.set_title(f"Time-Reversal (Iteration {frame + 1})")

    im3.set_data(adj_iterations[frame])
    ax3.set_title(f"Adjoint Method (Iteration {frame + 1})")

    # Update convergence plots
    x_data = range(frame + 1)
    line1.set_data(x_data, tr_errors[: frame + 1])
    line2.set_data(x_data, adj_errors[: frame + 1])
    line3.set_data(x_data, tr_residuals[: frame + 1])
    line4.set_data(x_data, adj_residuals[: frame + 1])

    return [im2, im3, line1, line2, line3, line4]


# Create animation
ani = animation.FuncAnimation(
    fig, update, frames=n_iterations, interval=500, blit=False
)

# Save animation
ani.save(
    PLOT_DIR / "improved_reconstruction_animation.gif", writer="imagemagick", fps=2
)

# --- 6. Print Final Statistics ---
print("\n" + "=" * 50)
print("FINAL RECONSTRUCTION STATISTICS")
print("=" * 50)
print("Time-Reversal:")
print(f"  Final relative error: {tr_errors[-1]:.4f}")
print(f"  Final residual norm: {tr_residuals[-1]:.4f}")
print(
    f"  Min/Max values: {jnp.min(tr_iterations[-1]):.4f} / {jnp.max(tr_iterations[-1]):.4f}"
)

print("\nAdjoint Method:")
print(f"  Final relative error: {adj_errors[-1]:.4f}")
print(f"  Final residual norm: {adj_residuals[-1]:.4f}")
print(
    f"  Min/Max values: {jnp.min(adj_iterations[-1]):.4f} / {jnp.max(adj_iterations[-1]):.4f}"
)

plt.show()

# --- 7. Optional: Try Conjugate Gradient Method ---
print("\n" + "=" * 50)
print("BONUS: Conjugate Gradient Reconstruction")
print("=" * 50)


def conjugate_gradient_reconstruction(n_iter=10):
    """Implement CG for normal equations: A^T A x = A^T b"""
    p_cg = jnp.zeros_like(p0)

    # Initial residual: r = A^T b - A^T A x
    Ax = solver.forward(p_cg, domain, mask, ts)
    ATAx = solver.adjoint(Ax.T, domain, all_sensors, mask, ts).T
    ATb = solver.adjoint(meas.T, domain, all_sensors, mask, ts).T
    r = ATb - ATAx
    p = r.copy()
    rsold = jnp.sum(r * r)

    cg_iterations = [p_cg.copy()]

    for i in range(n_iter):
        # Compute A^T A p
        Ap = solver.forward(p, domain, mask, ts)
        ATAp = solver.adjoint(Ap.T, domain, all_sensors, mask, ts).T

        # Step size
        alpha = rsold / jnp.sum(p * ATAp)

        # Update estimate
        p_cg = p_cg + alpha * p

        # Apply constraints
        if use_constraints:
            p_cg = apply_non_negativity(p_cg)

        # Update residual
        r = r - alpha * ATAp
        rsnew = jnp.sum(r * r)

        # Check convergence
        if jnp.sqrt(rsnew) < 1e-10:
            break

        # Update search direction
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew

        cg_iterations.append(p_cg.copy())
        print(f"  CG Iteration {i + 1}: residual = {jnp.sqrt(rsnew):.6f}")

    return p_cg, cg_iterations


# Run CG reconstruction
p_cg_final, cg_iters = conjugate_gradient_reconstruction(n_iter=15)
cg_error = compute_error(p_cg_final, p0)
print(f"\nConjugate Gradient final error: {cg_error:.4f}")

# Plot final comparison with ONE shared colorbar and GT-based scaling
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
titles = ["Ground Truth", "Time-Reversal", "Adjoint", "Conjugate Gradient"]
images = [p0, tr_iterations[-1], adj_iterations[-1], p_cg_final]
errors = [0, tr_errors[-1], adj_errors[-1], cg_error]

vmin, vmax = float(jnp.min(p0)), float(jnp.max(p0))
ims = []
for ax, title, img, err in zip(axes, titles, images, errors):
    im = ax.imshow(img, cmap="viridis", vmin=vmin, vmax=vmax)
    ims.append(im)
    ax.set_title(f"{title}\nError: {err:.4f}")
    ax.set_xticks([])
    ax.set_yticks([])
    if title == "Ground Truth":
        ax.plot(
            xs,
            ys,
            "^",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="red",
            markersize=6,
            alpha=0.95,
            zorder=5,
        )

ax_gt = axes[0]
divider = make_axes_locatable(ax_gt)
cax = divider.append_axes("left", size="4%", pad=0.15)
cbar = fig.colorbar(axes[0].images[0], cax=cax)  # tie to GT's image; scale is shared
cbar.set_label("Amplitude (shared scale from ground truth)")
cax.yaxis.set_ticks_position("left")
cax.yaxis.set_label_position("left")

plt.tight_layout()
plt.savefig(PLOT_DIR / "final_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
