#!/usr/bin/env python
# coding: utf-8



"""
1D adjoint via JAX autodiff, used as an independent check on the analytic MSGB adjoint.
"""
import jax.numpy as jnp
import jax
from time import time
import matplotlib.pyplot as plt
from pathlib import Path
import equinox as eqx
from beamax import geometry, utils
from beamax.decomposition import DyadicDecomposition
from beamax.transforms import MSWPT
from beamax.gb import gb_solvers
from beamax.solvers import MSGBSolver, ShardingStrategy
from beamax.plotter import use_beamax_style
import numpy as np
from matplotlib import animation
from matplotlib.gridspec import GridSpec

try:
    import optax
except ModuleNotFoundError:
    print("Skipping example: optax is not installed (`pip install optax`).")
    raise SystemExit(0)

try:
    from tqdm import tqdm
except ModuleNotFoundError:

    def tqdm(iterable, *args, **kwargs):
        return iterable


# Configure JAX for double precision
jax.config.update("jax_enable_x64", True)

# Setup directories
ROOT_DIR = utils.detect_root()
PLOT_DIR = Path(ROOT_DIR / "plots")
DATA_DIR = Path(ROOT_DIR / "data")
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Try to use thesis style if available
use_beamax_style()

# ============================================
# PART 1: DOMAIN AND SOLVER SETUP
# ============================================
# Domain parameters
d = 1
N = (256,) * d
dx = (1e-3,) * d
periodic = (False,) * d
box_aspect_ratio = (1,) * d
num_levels = 3
num_boxes_level = (4,) * num_levels


def c(x):
    return 1 + 0 * x[..., 0]


c0 = c(jnp.zeros(N))

# MSGB parameters
windowing = "rectangular_mirror"
input_type = "spatial"
output_type = "spatial"
redundancy = 2
cfl = 0.5

# Create domain
domain = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)
XY, KXY = domain.generate_meshgrid()
KXY = jnp.stack(KXY, axis=-1)

# Time domain
ts = domain.generate_time_domain()
Nt = len(ts)
dt = ts[1] - ts[0]
print(f"Spatial grid: {N[0]} points, dx = {dx[0]:.3e}")
print(f"Time grid: {Nt} points, dt = {dt:.3e}, T_max = {ts[-1]:.3f}")

# Create decomposition and transform
t1 = time()
dyadic_decomp = DyadicDecomposition(num_levels, N, num_boxes_level, box_aspect_ratio)
wpt = MSWPT(dyadic_decomp, redundancy, windowing)
t2 = time()
print(f"MSGB setup time: {t2 - t1:.3f}s")

# Setup sensors
binary_mask = jnp.zeros(N)
# sensor_positions = [30, 50, 100, 150, 200, 220]  # More sensors for better reconstruction
sensor_positions = [0]
for pos in sensor_positions:
    binary_mask = binary_mask.at[pos, ...].set(1)

sensors = geometry.Sensor(domain, binary_mask=binary_mask)
sensor_idx = jnp.where(binary_mask == 1)[0]
x_grid = jnp.arange(N[0]) * dx[0]
sensor_x = x_grid[sensor_idx]
print(f"Number of sensors: {len(sensor_idx)}")
print(f"Sensor locations (m): {sensor_x}")

# Create MSGB solver
threshold = 2 * N[0]
strategy = "top_n"
batch_size = 100
method = "scan_real"
solver = gb_solvers.solve_hom_diag

num_devices = jax.device_count()

mesh = jax.make_mesh((num_devices,), ("x",))

# Create sharding strategy
sharding_strategy = ShardingStrategy(mesh, beam_axis="x")

msgb_solver = MSGBSolver(
    thr=threshold,
    thr_strat=strategy,
    batch_size=batch_size,
    input_type="spatial",
    ode_solver=solver,
    sum_method=method,
    sharding=sharding_strategy,
)

# ============================================
# PART 2: FORWARD SOLVE
# ============================================

# Create initial pressure distribution
p0_true = jnp.zeros(N)
# Add a Gaussian-like source
center = N[0] // 2
width = 10
x_indices = jnp.arange(N[0])
p0_true = jnp.exp(-((x_indices - center) ** 2) / (2 * width**2))
p0_true = p0_true / jnp.max(jnp.abs(p0_true))  # Normalize

# Alternative: Box function
# p0_true = p0_true.at[N[0] // 2 - 10 : N[0] // 2 + 10].set(1)
# p0_true = p0_true / jnp.max(jnp.abs(p0_true))

dpdt_true = jnp.zeros_like(p0_true)

# Make sure they're real
p0_true = p0_true.real
dpdt_true = dpdt_true.real

print(f"Initial pressure norm: {jnp.linalg.norm(p0_true):.3e}")

# Forward solve (with JIT warm-up)
print("Running forward solve...")
for i in range(2):
    t1 = time()
    sensor_data, full_field = msgb_solver.forward(p0_true, domain, sensors, ts, wpt)
    t2 = time()
    if i == 1:  # Report second run (after JIT)
        print(f"Forward solve time: {t2 - t1:.3f}s")

sensor_data = sensor_data.real
measurements = sensor_data

print(f"Sensor data shape: {sensor_data.shape} (time_steps, n_sensors)")
print(f"Sensor data range: [{jnp.min(sensor_data):.3e}, {jnp.max(sensor_data):.3e}]")


# Define forward function for optimization
def forward_sensor_data(p0, dpdt):
    sd, _ = msgb_solver.forward(p0, domain, sensors, ts, wpt)
    return sd.real


# ============================================
# PART 3: VISUALIZE FORWARD SOLVE
# ============================================
# Create comprehensive forward solve visualization
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# 1. Initial condition
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(x_grid, p0_true, "b-", linewidth=2)
ax1.set_title("Initial Pressure p0", fontweight="bold")
ax1.set_xlabel("Position (m)")
ax1.set_ylabel("Pressure")
ax1.grid(True, alpha=0.3)
# Mark sensors
for idx in sensor_idx:
    ax1.axvline(x=x_grid[idx], color="r", linestyle="--", alpha=0.5, linewidth=1)
ax1.legend(["p0", "Sensors"], loc="upper right")

# 2. Sensor signals (waterfall plot)
ax2 = fig.add_subplot(gs[0, 1:])
time_mesh, sensor_mesh = jnp.meshgrid(ts, jnp.arange(len(sensor_idx)))
c = ax2.pcolormesh(time_mesh, sensor_mesh, sensor_data.T, shading="auto", cmap="RdBu_r")
ax2.set_title("Sensor Measurements (Waterfall)", fontweight="bold")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Sensor Index")
plt.colorbar(c, ax=ax2, label="Amplitude")

# 3. Individual sensor traces
ax3 = fig.add_subplot(gs[1, :])
colors = plt.cm.viridis(np.linspace(0, 1, len(sensor_idx)))
for i, (idx, color) in enumerate(zip(sensor_idx, colors)):
    ax3.plot(
        ts,
        sensor_data[:, i],
        color=color,
        label=f"Sensor @ x={x_grid[idx]:.3f}m",
        linewidth=1.5,
        alpha=0.8,
    )
ax3.set_title("Individual Sensor Traces", fontweight="bold")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Amplitude")
ax3.grid(True, alpha=0.3)
ax3.legend(loc="upper right", ncol=2, fontsize=8)

# # 4. Wave propagation snapshots
# snapshot_times = [0, Nt//4, Nt//2, 3*Nt//4]
# for i, t_idx in enumerate(snapshot_times):
#     ax = fig.add_subplot(gs[2, i]) if i < 3 else None
#     if ax and full_field is not None and len(full_field.shape) > 1:
#         try:
#             field_snapshot = full_field[t_idx] if full_field.shape[0] > t_idx else jnp.zeros(N)
#             ax.plot(x_grid, field_snapshot.real, 'b-', linewidth=1.5)
#             ax.set_title(f't = {ts[t_idx]:.3f}s', fontsize=10)
#             ax.set_xlabel('Position (m)')
#             ax.set_ylabel('Pressure')
#             ax.grid(True, alpha=0.3)
#             # Mark sensors
#             for idx in sensor_idx:
#                 ax.axvline(x=x_grid[idx], color='r', linestyle='--', alpha=0.3, linewidth=0.5)
#         except:
#             pass

plt.suptitle("FORWARD WAVE PROPAGATION ANALYSIS", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOT_DIR / "forward_solve_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================
# PART 4: GRADIENT-BASED RECONSTRUCTION
# ============================================
# Optimization parameters
iters = 150
lr = 2e-2
regularization = 1e-6

# Initialize reconstruction variables (poor initial guess)
recon_p0 = jnp.ones_like(p0_true) * 0.1  # Flat initial guess
recon_dpdt = jnp.zeros_like(dpdt_true)


# Define loss function
@eqx.filter_jit
def loss_fn(p0_var):
    pred = forward_sensor_data(p0_var, recon_dpdt)
    data_fidelity = 0.5 * jnp.mean((pred - measurements) ** 2)
    regularization_term = regularization * 0.5 * jnp.mean(p0_var**2)
    return data_fidelity + regularization_term


# Setup optimizer
optimizer = optax.adam(lr)
opt_state = optimizer.init(recon_p0)


@eqx.filter_jit
def optimization_step(var, opt_state):
    loss, grad = jax.value_and_grad(loss_fn)(var)
    updates, opt_state = optimizer.update(grad, opt_state, params=var)
    var = optax.apply_updates(var, updates)
    return var, opt_state, loss, grad


# Run optimization
print(f"Running {iters} iterations of Adam optimization...")
print(f"Learning rate: {lr}, Regularization: {regularization}")

hist_p0 = [recon_p0]
hist_loss = []
hist_grad_norm = []

for it in tqdm(range(iters)):
    recon_p0, opt_state, loss, grad = optimization_step(recon_p0, opt_state)
    hist_p0.append(recon_p0)
    hist_loss.append(loss)
    hist_grad_norm.append(jnp.linalg.norm(grad))

# Convert to numpy for plotting
hist_p0 = np.stack(hist_p0, axis=0)

# Calculate reconstruction error
final_error = jnp.linalg.norm(recon_p0 - p0_true) / jnp.linalg.norm(p0_true)
print(f"Final reconstruction error: {final_error:.3e}")
print(f"Final loss: {hist_loss[-1]:.3e}")


# ============================================
# PART 5: ADJOINT SOLVE AND TEST
# ============================================
@eqx.filter_jit
def adjoint_apply(p0, dpdt, residual):
    """
    Apply adjoint operator: J^T · residual
    Returns gradients with respect to p0 and dpdt
    """
    _, pullback = jax.vjp(forward_sensor_data, p0, dpdt)
    g_p0, g_dpdt = pullback(residual)
    return g_p0, g_dpdt


# Compute adjoint field for current reconstruction
pred = forward_sensor_data(recon_p0, recon_dpdt)
residual = pred - measurements
adjoint_p0, adjoint_dpdt = adjoint_apply(recon_p0, recon_dpdt, residual)

print(f"Adjoint field norm (p0): {jnp.linalg.norm(adjoint_p0):.3e}")
print(f"Adjoint field norm (dpdt): {jnp.linalg.norm(adjoint_dpdt):.3e}")


# Adjoint test: verify <Jv, w> = <v, J^T w>
def adjoint_test(base_p0, base_dpdt, trials=5, seed=42):
    """
    Adjoint test to verify correctness of adjoint implementation.
    Tests if <J·v, w> = <v, J^T·w> for random v and w.
    """
    print("\nRunning adjoint test...")
    print("-" * 40)

    key = jax.random.PRNGKey(seed)
    errors = []

    for i in range(trials):
        # Generate random test vectors
        key, k1, k2, k3 = jax.random.split(key, 4)
        v_p0 = jax.random.normal(k1, base_p0.shape)
        v_dpdt = jax.random.normal(k2, base_dpdt.shape)
        w = jax.random.normal(k3, measurements.shape)

        # Compute J·v (forward)
        _, Jv = jax.jvp(forward_sensor_data, (base_p0, base_dpdt), (v_p0, v_dpdt))

        # Compute J^T·w (adjoint)
        JTw_p0, JTw_dpdt = adjoint_apply(base_p0, base_dpdt, w)

        # Inner products
        lhs = jnp.vdot(Jv, w).real  # <J·v, w>
        rhs = (jnp.vdot(v_p0, JTw_p0) + jnp.vdot(v_dpdt, JTw_dpdt)).real  # <v, J^T·w>

        # Error
        abs_err = jnp.abs(lhs - rhs)
        rel_err = abs_err / jnp.maximum(jnp.maximum(jnp.abs(lhs), jnp.abs(rhs)), 1e-12)
        errors.append(rel_err)

        print(f"Trial {i + 1}: <Jv,w> = {lhs:+.6e}, <v,J^Tw> = {rhs:+.6e}")
        print(f"         Absolute error: {abs_err:.3e}, Relative error: {rel_err:.3e}")

    mean_error = np.mean(errors)
    max_error = np.max(errors)
    print("-" * 40)
    print(f"Mean relative error: {mean_error:.3e}")
    print(f"Max relative error: {max_error:.3e}")
    print(f"Adjoint test {'PASSED' if max_error < 1e-10 else 'FAILED'}")

    return errors


adjoint_errors = adjoint_test(recon_p0, recon_dpdt, trials=5)

# ============================================
# PART 6: COMPREHENSIVE VISUALIZATION
# ============================================
# Create main results figure
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

# 1. True vs Reconstructed
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(x_grid, p0_true, "k-", linewidth=2, label="True p0")
ax1.plot(
    x_grid,
    recon_p0,
    "r--",
    linewidth=2,
    label=f"Reconstructed (error={final_error:.2e})",
)
ax1.fill_between(x_grid, p0_true, recon_p0, alpha=0.3, color="gray")
ax1.set_title("Reconstruction Result", fontweight="bold", fontsize=12)
ax1.set_xlabel("Position (m)")
ax1.set_ylabel("Pressure")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)
# Mark sensors
for idx in sensor_idx:
    ax1.axvline(x=x_grid[idx], color="g", linestyle=":", alpha=0.5, linewidth=1)

# 2. Reconstruction error
ax2 = fig.add_subplot(gs[1, 0])
error_spatial = np.abs(recon_p0 - p0_true)
ax2.plot(x_grid, error_spatial, "b-", linewidth=2)
ax2.fill_between(x_grid, 0, error_spatial, alpha=0.3, color="blue")
ax2.set_title("Spatial Error Distribution", fontweight="bold")
ax2.set_xlabel("Position (m)")
ax2.set_ylabel("|p_recon - p_true|")
ax2.grid(True, alpha=0.3)

# 3. Loss convergence
ax3 = fig.add_subplot(gs[1, 1])
ax3.semilogy(hist_loss, "r-", linewidth=2)
ax3.set_title("Loss Convergence", fontweight="bold")
ax3.set_xlabel("Iteration")
ax3.set_ylabel("Loss (log scale)")
ax3.grid(True, alpha=0.3, which="both")

# 4. Gradient norm evolution
ax4 = fig.add_subplot(gs[1, 2])
ax4.semilogy(hist_grad_norm, "g-", linewidth=2)
ax4.set_title("Gradient Norm Evolution", fontweight="bold")
ax4.set_xlabel("Iteration")
ax4.set_ylabel("||∇L|| (log scale)")
ax4.grid(True, alpha=0.3, which="both")

# 5. Adjoint field visualization
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(x_grid, adjoint_p0, "purple", linewidth=2)
ax5.set_title("Adjoint Field (∂L/∂p0)", fontweight="bold")
ax5.set_xlabel("Position (m)")
ax5.set_ylabel("Adjoint amplitude")
ax5.grid(True, alpha=0.3)

# 6. Residual in sensor space
ax6 = fig.add_subplot(gs[2, 1:])
residual_plot = ax6.pcolormesh(
    ts, jnp.arange(len(sensor_idx)), residual.T, shading="auto", cmap="RdBu_r"
)
ax6.set_title("Sensor Space Residual (pred - meas)", fontweight="bold")
ax6.set_xlabel("Time (s)")
ax6.set_ylabel("Sensor Index")
plt.colorbar(residual_plot, ax=ax6, label="Residual")

# 7. Reconstruction evolution (selected iterations)
ax7 = fig.add_subplot(gs[3, :])
iterations_to_plot = [0, iters // 4, iters // 2, 3 * iters // 4, iters]
colors = plt.cm.viridis(np.linspace(0, 1, len(iterations_to_plot)))
for i, (it, color) in enumerate(zip(iterations_to_plot, colors)):
    it = min(it, len(hist_p0) - 1)
    ax7.plot(
        x_grid, hist_p0[it], color=color, label=f"Iter {it}", linewidth=1.5, alpha=0.8
    )
ax7.plot(x_grid, p0_true, "k--", linewidth=2, label="True", alpha=0.7)
ax7.set_title("Reconstruction Evolution", fontweight="bold")
ax7.set_xlabel("Position (m)")
ax7.set_ylabel("Pressure")
ax7.legend(loc="upper right", ncol=3)
ax7.grid(True, alpha=0.3)

plt.suptitle(
    "PAT RECONSTRUCTION: GRADIENT-BASED METHOD WITH ADJOINT ANALYSIS",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(PLOT_DIR / "pat_complete_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================
# PART 7: ANIMATION OF RECONSTRUCTION
# ============================================
fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}
)

# Top: reconstruction progress
(line_true,) = ax_top.plot(x_grid, p0_true, "k-", linewidth=2, label="True p0")
(line_recon,) = ax_top.plot(
    x_grid, hist_p0[0], "r-", linewidth=2, label="Reconstruction"
)
(line_error,) = ax_top.plot(
    x_grid, np.abs(hist_p0[0] - p0_true), "b--", linewidth=1.5, alpha=0.7, label="Error"
)

# Mark sensors
for idx in sensor_idx:
    ax_top.axvline(x=x_grid[idx], color="g", linestyle=":", alpha=0.3)

ax_top.set_ylim(-0.1, 1.2)
ax_top.set_xlabel("Position (m)")
ax_top.set_ylabel("Amplitude")
ax_top.legend(loc="upper right")
ax_top.grid(True, alpha=0.3)
ax_top.set_title("Iteration 0")

# Bottom: loss evolution
(line_loss,) = ax_bottom.semilogy(
    [0], [hist_loss[0] if hist_loss else 1], "r-", linewidth=2
)
ax_bottom.set_xlabel("Iteration")
ax_bottom.set_ylabel("Loss (log scale)")
ax_bottom.set_xlim(0, iters)
ax_bottom.set_ylim(
    min(hist_loss) * 0.5 if hist_loss else 1e-6, max(hist_loss) * 2 if hist_loss else 1
)
ax_bottom.grid(True, alpha=0.3, which="both")


def animate(frame):
    if frame < len(hist_p0):
        # Update reconstruction
        line_recon.set_ydata(hist_p0[frame])
        line_error.set_ydata(np.abs(hist_p0[frame] - p0_true))

        # Update title with current stats
        current_error = np.linalg.norm(hist_p0[frame] - p0_true) / np.linalg.norm(
            p0_true
        )
        current_loss = (
            hist_loss[frame - 1] if frame > 0 and frame - 1 < len(hist_loss) else 0
        )
        ax_top.set_title(
            f"Iteration {frame} | Error: {current_error:.3e} | Loss: {current_loss:.3e}"
        )

        # Update loss plot
        if frame > 0 and frame <= len(hist_loss):
            line_loss.set_data(range(frame), hist_loss[:frame])

    return line_recon, line_error, line_loss


# Create and save animation
ani = animation.FuncAnimation(
    fig, animate, frames=len(hist_p0), interval=50, blit=True, repeat=True
)
ani.save(PLOT_DIR / "reconstruction_animation.mp4", fps=10, dpi=100)
print(f"Animation saved to {PLOT_DIR / 'reconstruction_animation.mp4'}")
plt.close(fig)

# ============================================
# PART 8: SUMMARY REPORT
# ============================================
print("\n" + "=" * 70)
print(" " * 25 + "SUMMARY REPORT")
print("=" * 70)

print("\n1. FORWARD SOLVE:")
print(f"   - Grid points: {N[0]}")
print(f"   - Time steps: {Nt}")
print(f"   - Number of sensors: {len(sensor_idx)}")
print(f"   - Forward solve time: {t2 - t1:.3f}s")

print("\n2. GRADIENT-BASED RECONSTRUCTION:")
print("   - Optimization method: Adam")
print(f"   - Learning rate: {lr}")
print(f"   - Regularization: {regularization}")
print(f"   - Iterations: {iters}")
print(f"   - Final loss: {hist_loss[-1]:.3e}")
print(f"   - Final error: {final_error:.3e}")
print(f"   - Final gradient norm: {hist_grad_norm[-1]:.3e}")

print("\n3. ADJOINT VERIFICATION:")
print(f"   - Adjoint field norm: {jnp.linalg.norm(adjoint_p0):.3e}")
print(f"   - Mean adjoint test error: {np.mean(adjoint_errors):.3e}")
print(f"   - Max adjoint test error: {np.max(adjoint_errors):.3e}")
print(
    f"   - Adjoint test: {'✓ PASSED' if np.max(adjoint_errors) < 1e-10 else '✗ FAILED'}"
)

print("\n4. OUTPUT FILES:")
print(f"   - Forward analysis: {PLOT_DIR / 'forward_solve_analysis.png'}")
print(f"   - Complete analysis: {PLOT_DIR / 'pat_complete_analysis.png'}")
print(f"   - Animation: {PLOT_DIR / 'reconstruction_animation.mp4'}")
