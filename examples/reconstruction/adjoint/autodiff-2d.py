#!/usr/bin/env python
# coding: utf-8



"""
2D adjoint via JAX autodiff.
"""
import jax.numpy as jnp
import jax
from time import time
import matplotlib.pyplot as plt
from pathlib import Path
import equinox as eqx
from beamax import geometry, utils, transforms
from beamax.decomposition import DyadicDecomposition
from beamax.transforms import MSWPT
from beamax.gb import gb_solvers
from beamax.solvers import MSGBSolver, KWaveSolver, ShardingStrategy
from beamax.plotter import use_beamax_style
import numpy as np
from matplotlib.gridspec import GridSpec
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions

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


jax.config.update("jax_enable_x64", True)

ROOT_DIR = utils.detect_root()
PLOT_DIR = Path(ROOT_DIR / "plots")
DATA_DIR = Path(ROOT_DIR / "data")
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

use_beamax_style()

# ============================================
# PART 1: DOMAIN AND SOLVER SETUP
# ============================================
# Domain parameters
d = 2
N = (128,) * d
dx = (1e-4,) * d
extent = tuple([dx[i] * N[i] for i in range(d)])
periodic = (False,) * d
box_aspect_ratio = (1,) * d
num_levels = 1
num_boxes_level = (4,) * num_levels


def c(x):
    return 1 + 0 * x[..., 0]


c0 = c(jnp.zeros(N))

# MSGB parameters
windowing = "rectangular_mirror"
input_type = "spatial"
output_type = "spatial"
redundancy = 2
cfl = jnp.sqrt(d) / 4

# Create domain
domain = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)
XY, KXY = domain.generate_meshgrid()
X, Y = XY
KXY = jnp.stack(KXY, axis=-1)

# Time domain
ts = domain.generate_time_domain()
Nt = len(ts)
dt = ts[1] - ts[0]

print(f"Spatial grid: {N[0]}x{N[1]} points")
print(f"Physical extent: {extent[0]:.3e} x {extent[1]:.3e} m")
print(f"Grid spacing: dx = {dx[0]:.3e} m, dy = {dx[1]:.3e} m")
print(f"Time grid: {Nt} points, dt = {dt:.3e}s, T_max = {ts[-1]:.3f}s")

# Create decomposition and transform
t1 = time()
dyadic_decomp = DyadicDecomposition(num_levels, N, num_boxes_level, box_aspect_ratio)
wpt = MSWPT(dyadic_decomp, redundancy, windowing)
t2 = time()
print(f"MSGB setup time: {t2 - t1:.3f}s")

# Setup sensors - circular arc or line array
print("\nSensor Configuration:")
sensor_config = "line"
binary_mask = jnp.zeros(N)
if sensor_config == "circle":
    # Circular sensor array
    R = extent[0] / 2 - dx[0]  # radius of the circle
    dx2 = dx[0] / 2  # half the grid spacing
    r = jnp.sqrt((X - extent[0] / 2) ** 2 + (Y - extent[1] / 2) ** 2)
    binary_mask = jnp.logical_and(r >= (R - dx2), r <= (R + dx2)).astype(jnp.int32)
elif sensor_config == "line":
    binary_mask = binary_mask.at[0, ...].set(1)

sensors = geometry.Sensor(domain, binary_mask=binary_mask)
sensor_idx = np.argwhere(np.array(binary_mask) == 1)  # shape (Ns, 2) with [iy, ix]
sensor_y = sensor_idx[:, 0]
sensor_x = sensor_idx[:, 1]
print(f"Number of sensors: {len(sensor_idx)}")
print(f"Sensor configuration: {sensor_config}")

# Create MSGB solver
threshold = 1000
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
# PART 2: CREATE INITIAL PRESSURE DISTRIBUTION
# ============================================

p0_true = jnp.zeros(N)
p0_true = p0_true.at[N[0] // 2 - 3 : N[0] // 2 + 3, N[1] // 2 - 3 : N[1] // 2 + 3].set(
    1.0
)

KXY = dyadic_decomp.fourier_meshgrid

# pltgb.plot_centers(dyadic_decomp.centres_ndim)

boxhf = 44
boxlf = 10
# probably need to multiply by the ratio between (64,64) and the desired res.
khf = jnp.array([30, 12])
klf = jnp.array([30, 3])
kerft_hf = transforms.compute_frames(dyadic_decomp, boxhf, khf, KXY, redundancy, "none")
kerft_lf = transforms.compute_frames(dyadic_decomp, boxlf, klf, KXY, redundancy, "none")
p0 = utils.unitary_ifft(kerft_hf) + utils.unitary_ifft(kerft_lf)
p0 = p0 / jnp.max(jnp.abs(p0))
p0_true = p0.T.real

# Initial velocity (zero)
dpdt_true = jnp.zeros_like(p0_true).real

print(f"Initial pressure norm: {jnp.linalg.norm(p0_true):.3e}")
print(f"Initial pressure range: [{jnp.min(p0_true):.3e}, {jnp.max(p0_true):.3e}]")

# ============================================
# PART 3: FORWARD SOLVE WITH K-WAVE (TRUE) AND MSGB (SIMULATED)
# ============================================

# K-Wave forward solve for TRUE measurements
print("\nComputing k-Wave forward solve (ground truth)...")
simulation_options = SimulationOptions(
    data_cast="single",
    smooth_p0=False,
    save_to_disk=True,
)
execution_options = SimulationExecutionOptions(
    is_gpu_simulation=False, delete_data=False, verbose_level=0, show_sim_log=False
)
kwave_solver = KWaveSolver(simulation_options, execution_options)

t1 = time()
measurements = kwave_solver.forward(
    p0_true, domain, binary_mask, ts
)  # TRUE measurements from k-Wave
t2 = time()
print(f"k-Wave forward solve time: {t2 - t1:.3f}s")

# MSGB forward solve (for comparison and to setup the adjoint machinery)
print("\nComputing MSGB forward solve (for comparison)...")
# Warm-up JIT compilation
for i in range(2):
    t1 = time()
    msgb_sensor_data, msgb_full_field = msgb_solver.forward(
        p0_true,
        domain,
        sensors,
        ts,
        wpt,
    )
    t2 = time()
    if i == 1:  # Report second run (after JIT)
        print(f"MSGB forward solve time: {t2 - t1:.3f}s")

msgb_measurements = msgb_sensor_data.real

print("\nSensor data comparison:")
print(f"k-Wave measurements shape: {measurements.shape}")
print(
    f"k-Wave measurements range: [{jnp.min(measurements):.3e}, {jnp.max(measurements):.3e}]"
)
print(f"MSGB measurements shape: {msgb_measurements.shape}")
print(
    f"MSGB measurements range: [{jnp.min(msgb_measurements):.3e}, {jnp.max(msgb_measurements):.3e}]"
)
print(
    f"Difference (RMS): {jnp.sqrt(jnp.mean((measurements - msgb_measurements) ** 2)):.3e}"
)


# Define forward function for optimization (use MSGB for inverse)
def forward_sensor_data(p0, dpdt):
    sd, _ = msgb_solver.forward(p0, domain, wpt, sensors, ts)
    return sd.real


# ============================================
# PART 4: VISUALIZE FORWARD SOLVE COMPARISON
# ============================================
# Create comprehensive forward solve visualization comparing k-Wave and MSGB
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

# 1. Initial pressure distribution
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(
    p0_true,
    origin="lower",
    cmap="RdBu_r",
    extent=[0, extent[0] * 1e3, 0, extent[1] * 1e3],
)
ax1.set_title("Initial Pressure p0", fontweight="bold")
ax1.set_xlabel("x (mm)")
ax1.set_ylabel("y (mm)")
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
# Overlay sensors
ax1.scatter(
    sensor_x * dx[0] * 1e3,
    sensor_y * dx[1] * 1e3,
    c="green",
    s=20,
    alpha=0.7,
    marker="^",
    edgecolors="k",
    linewidth=0.5,
)

# 2. Sensor mask
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(
    binary_mask,
    origin="lower",
    cmap="binary",
    extent=[0, extent[0] * 1e3, 0, extent[1] * 1e3],
)
ax2.set_title("Sensor Locations", fontweight="bold")
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("y (mm)")

# 3. k-Wave sensor signals (waterfall)
ax3 = fig.add_subplot(gs[0, 2:])
time_mesh, sensor_mesh = jnp.meshgrid(ts * 1e3, jnp.arange(len(sensor_idx)))
c = ax3.pcolormesh(
    time_mesh, sensor_mesh, measurements.T, shading="auto", cmap="RdBu_r"
)
ax3.set_title("k-Wave Measurements (Ground Truth)", fontweight="bold")
ax3.set_xlabel("Time (ms)")
ax3.set_ylabel("Sensor Index")
plt.colorbar(c, ax=ax3, label="Amplitude")

# 4. MSGB sensor signals (waterfall)
ax4 = fig.add_subplot(gs[1, 2:])
c = ax4.pcolormesh(
    time_mesh, sensor_mesh, msgb_measurements.T, shading="auto", cmap="RdBu_r"
)
ax4.set_title("MSGB Measurements (Simulated)", fontweight="bold")
ax4.set_xlabel("Time (ms)")
ax4.set_ylabel("Sensor Index")
plt.colorbar(c, ax=ax4, label="Amplitude")

# 5. Difference between k-Wave and MSGB
ax5 = fig.add_subplot(gs[2, 2:])
diff = measurements - msgb_measurements
max_diff = jnp.max(jnp.abs(diff))
c = ax5.pcolormesh(
    time_mesh,
    sensor_mesh,
    diff.T,
    shading="auto",
    cmap="seismic",
    vmin=-max_diff,
    vmax=max_diff,
)
ax5.set_title("Difference (k-Wave - MSGB)", fontweight="bold")
ax5.set_xlabel("Time (ms)")
ax5.set_ylabel("Sensor Index")
plt.colorbar(c, ax=ax5, label="Difference")

# 6. Selected sensor traces comparison
ax6 = fig.add_subplot(gs[1:2, :2])
n_traces = min(4, len(sensor_idx))
trace_indices = np.linspace(0, len(sensor_idx) - 1, n_traces, dtype=int)
colors = plt.cm.viridis(np.linspace(0, 1, n_traces))
for i, idx in enumerate(trace_indices):
    ax6.plot(
        ts * 1e3,
        measurements[:, idx],
        color=colors[i],
        label=f"k-Wave S{idx}",
        linewidth=1.5,
        alpha=0.8,
    )
    ax6.plot(
        ts * 1e3,
        msgb_measurements[:, idx],
        "--",
        color=colors[i],
        label=f"MSGB S{idx}",
        linewidth=1.5,
        alpha=0.8,
    )
ax6.set_title("Selected Sensor Trace Comparison", fontweight="bold")
ax6.set_xlabel("Time (ms)")
ax6.set_ylabel("Amplitude")
ax6.grid(True, alpha=0.3)
ax6.legend(loc="upper right", ncol=2, fontsize=8)

# 7. Statistics
ax7 = fig.add_subplot(gs[2:3, :2])
ax7.axis("off")
stats_text = f"""Forward Solve Statistics:
━━━━━━━━━━━━━━━━━━━━━
Grid: {N[0]}×{N[1]} points
Physical size: {extent[0] * 1e3:.1f}×{extent[1] * 1e3:.1f} mm
Time steps: {Nt}
dt: {dt * 1e6:.2f} μs
Sensors: {len(sensor_idx)}
Max |p0|: {jnp.max(jnp.abs(p0_true)):.3f}
Max |k-Wave|: {jnp.max(jnp.abs(measurements)):.3f}
Max |MSGB|: {jnp.max(jnp.abs(msgb_measurements)):.3f}
RMS diff: {jnp.sqrt(jnp.mean(diff**2)):.3e}
"""
ax7.text(
    0.1, 0.5, stats_text, fontsize=10, family="monospace", verticalalignment="center"
)

plt.suptitle(
    "FORWARD WAVE PROPAGATION: k-WAVE vs MSGB COMPARISON",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(
    PLOT_DIR / "forward_kwave_msgb_comparison.png", dpi=150, bbox_inches="tight"
)
plt.show()

# ============================================
# PART 5: GRADIENT-BASED RECONSTRUCTION
# ============================================
iters = 150
lr = 2e-2
regularization = 1e-6

# Initialize reconstruction variables (poor initial guess)
recon_p0 = jnp.ones_like(p0_true)
recon_dpdt = jnp.zeros_like(dpdt_true)


# Define loss function (using k-Wave measurements as ground truth)
def loss_fn(p0_var):
    pred = forward_sensor_data(p0_var, recon_dpdt)  # MSGB forward
    data_fidelity = 0.5 * jnp.mean((pred - measurements) ** 2)  # Compare to k-Wave
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
print(f"\nRunning {iters} iterations of Adam optimization...")
print(f"Learning rate: {lr}, Regularization: {regularization}")
print("Using k-Wave measurements as ground truth, MSGB for reconstruction")

hist_p0 = [recon_p0]
hist_loss = []
hist_grad_norm = []
hist_error = []

for it in tqdm(range(iters)):
    recon_p0, opt_state, loss, grad = optimization_step(recon_p0, opt_state)
    hist_p0.append(recon_p0)
    hist_loss.append(loss)
    hist_grad_norm.append(jnp.linalg.norm(grad))

    # Calculate reconstruction error
    rel_error = jnp.linalg.norm(recon_p0 - p0_true) / jnp.linalg.norm(p0_true)
    hist_error.append(rel_error)

    if it % 10 == 0:
        print(f"  Iter {it}: loss = {loss:.3e}, rel_error = {rel_error:.3e}")

# Convert to numpy for plotting
hist_p0 = np.stack(hist_p0, axis=0)

# Final statistics
final_error = hist_error[-1]
print(f"\nFinal reconstruction error: {final_error:.3e}")
print(f"Final loss: {hist_loss[-1]:.3e}")
print(f"Final gradient norm: {hist_grad_norm[-1]:.3e}")


# ============================================
# PART 6: COMPUTE AND VISUALIZE ADJOINT SOLUTION
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
residual = pred - measurements  # Residual at sensors
adjoint_p0, adjoint_dpdt = adjoint_apply(recon_p0, recon_dpdt, residual)

print(f"Adjoint field norm (p0): {jnp.linalg.norm(adjoint_p0):.3e}")
print(f"Adjoint field norm (dpdt): {jnp.linalg.norm(adjoint_dpdt):.3e}")

# ============================================
# COMPUTE TIME-REVERSED ADJOINT PROPAGATION
# ============================================
print("\nComputing time-reversed adjoint propagation...")

# Create time-reversed residual signal at sensors
residual_time_reversed = jnp.flip(residual, axis=0)

# Create initial condition for adjoint propagation
# The adjoint initial condition is zero pressure, but with sensor residuals as sources
adjoint_p0_init = jnp.zeros_like(p0_true)
adjoint_dpdt_init = jnp.zeros_like(dpdt_true)

# We need to inject the time-reversed residual at sensor locations
# This is a simplified approach - in practice you'd solve the adjoint wave equation
# For visualization, we'll show the adjoint gradient field at different time points

# Compute adjoint solution at multiple time points
print("Computing adjoint snapshots...")
n_snapshots = 5
snapshot_times = np.linspace(0, len(ts) - 1, n_snapshots, dtype=int)
adjoint_snapshots = []

for t_idx in snapshot_times:
    # Create partial residual up to time t_idx
    partial_residual = jnp.zeros_like(residual)
    partial_residual = partial_residual.at[:t_idx].set(residual[:t_idx])

    # Compute adjoint field for this partial residual
    adj_p0_t, adj_dpdt_t = adjoint_apply(recon_p0, recon_dpdt, partial_residual)
    adjoint_snapshots.append(adj_p0_t)

# ============================================
# PART 7: ADJOINT VISUALIZATION
# ============================================
# Create comprehensive adjoint visualization
fig = plt.figure(figsize=(18, 14))
gs = GridSpec(4, 5, figure=fig, hspace=0.3, wspace=0.3)

# Row 1: Adjoint snapshots over time
for i, (t_idx, adj_snapshot) in enumerate(
    zip(snapshot_times[:5], adjoint_snapshots[:5])
):
    ax = fig.add_subplot(gs[0, i])
    max_val = np.max(np.abs(adj_snapshot))
    im = ax.imshow(
        adj_snapshot,
        origin="lower",
        cmap="PuOr",
        vmin=-max_val,
        vmax=max_val,
        extent=[0, extent[0] * 1e3, 0, extent[1] * 1e3],
    )
    ax.set_title(f"Adjoint t={ts[t_idx] * 1e3:.2f}ms", fontsize=10, fontweight="bold")
    ax.set_xlabel("x (mm)", fontsize=9)
    ax.set_ylabel("y (mm)", fontsize=9)
    if i == len(snapshot_times) - 1:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Row 2: Reconstruction results
# True p0
ax1 = fig.add_subplot(gs[1, 0])
im1 = ax1.imshow(
    p0_true,
    origin="lower",
    cmap="RdBu_r",
    extent=[0, extent[0] * 1e3, 0, extent[1] * 1e3],
)
ax1.set_title("True p0", fontweight="bold")
ax1.set_xlabel("x (mm)")
ax1.set_ylabel("y (mm)")
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# Reconstructed p0
ax2 = fig.add_subplot(gs[1, 1])
im2 = ax2.imshow(
    recon_p0,
    origin="lower",
    cmap="RdBu_r",
    extent=[0, extent[0] * 1e3, 0, extent[1] * 1e3],
    vmin=p0_true.min(),
    vmax=p0_true.max(),
)
ax2.set_title(f"Reconstructed (error={final_error:.2e})", fontweight="bold")
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("y (mm)")
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

# Final adjoint field (full)
ax3 = fig.add_subplot(gs[1, 2])
max_adj = np.max(np.abs(adjoint_p0))
im3 = ax3.imshow(
    adjoint_p0,
    origin="lower",
    cmap="PuOr",
    vmin=-max_adj,
    vmax=max_adj,
    extent=[0, extent[0] * 1e3, 0, extent[1] * 1e3],
)
ax3.set_title("Final Adjoint Field (∂L/∂p0)", fontweight="bold")
ax3.set_xlabel("x (mm)")
ax3.set_ylabel("y (mm)")
plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

# Reconstruction error
ax4 = fig.add_subplot(gs[1, 3])
error_spatial = np.abs(recon_p0 - p0_true)
im4 = ax4.imshow(
    error_spatial,
    origin="lower",
    cmap="hot",
    extent=[0, extent[0] * 1e3, 0, extent[1] * 1e3],
)
ax4.set_title("Absolute Error", fontweight="bold")
ax4.set_xlabel("x (mm)")
ax4.set_ylabel("y (mm)")
plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

# Gradient direction
ax5 = fig.add_subplot(gs[1, 4])
gradient = jax.grad(loss_fn)(recon_p0)
max_grad = np.max(np.abs(gradient))
im5 = ax5.imshow(
    gradient,
    origin="lower",
    cmap="seismic",
    vmin=-max_grad,
    vmax=max_grad,
    extent=[0, extent[0] * 1e3, 0, extent[1] * 1e3],
)
ax5.set_title("Gradient Direction", fontweight="bold")
ax5.set_xlabel("x (mm)")
ax5.set_ylabel("y (mm)")
plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

# Row 3: Cross-sections through adjoint field
ax6 = fig.add_subplot(gs[2, :3])
mid_y = N[0] // 2
x_coord = np.arange(N[1]) * dx[1] * 1e3

# Plot cross-sections of adjoint snapshots
for i, (t_idx, adj_snapshot) in enumerate(zip(snapshot_times, adjoint_snapshots)):
    alpha = 0.3 + 0.7 * (i / (len(snapshot_times) - 1))
    ax6.plot(
        x_coord,
        adj_snapshot[mid_y, :],
        label=f"t={ts[t_idx] * 1e3:.2f}ms",
        alpha=alpha,
        linewidth=2,
    )

ax6.set_title(f"Adjoint Field Evolution (y={N[0] // 2})", fontweight="bold")
ax6.set_xlabel("x (mm)")
ax6.set_ylabel("Adjoint Field Amplitude")
ax6.legend(loc="upper right")
ax6.grid(True, alpha=0.3)

# Sensor residual waterfall
ax7 = fig.add_subplot(gs[2, 3:])
time_mesh, sensor_mesh = jnp.meshgrid(ts * 1e3, jnp.arange(len(sensor_idx)))
sensor_residual = residual.T
max_res = jnp.max(jnp.abs(sensor_residual))
c = ax7.pcolormesh(
    time_mesh,
    sensor_mesh,
    sensor_residual,
    shading="auto",
    cmap="RdBu_r",
    vmin=-max_res,
    vmax=max_res,
)
ax7.set_title("Sensor Space Residual (MSGB - k-Wave)", fontweight="bold")
ax7.set_xlabel("Time (ms)")
ax7.set_ylabel("Sensor Index")
plt.colorbar(c, ax=ax7, label="Residual")

# Row 4: Convergence plots
# Loss convergence
ax8 = fig.add_subplot(gs[3, 0])
ax8.semilogy(hist_loss, "r-", linewidth=2)
ax8.set_title("Loss Convergence", fontweight="bold")
ax8.set_xlabel("Iteration")
ax8.set_ylabel("Loss (log scale)")
ax8.grid(True, alpha=0.3, which="both")

# Error evolution
ax9 = fig.add_subplot(gs[3, 1])
ax9.semilogy(hist_error, "b-", linewidth=2)
ax9.set_title("Reconstruction Error", fontweight="bold")
ax9.set_xlabel("Iteration")
ax9.set_ylabel("Relative Error (log scale)")
ax9.grid(True, alpha=0.3, which="both")

# Gradient norm
ax10 = fig.add_subplot(gs[3, 2])
ax10.semilogy(hist_grad_norm, "g-", linewidth=2)
ax10.set_title("Gradient Norm", fontweight="bold")
ax10.set_xlabel("Iteration")
ax10.set_ylabel("||∇L|| (log scale)")
ax10.grid(True, alpha=0.3, which="both")

# Adjoint field statistics
ax11 = fig.add_subplot(gs[3, 3:])
ax11.axis("off")
adj_stats_text = f"""Adjoint Solution Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━
Final adjoint norm (p0): {jnp.linalg.norm(adjoint_p0):.3e}
Final adjoint norm (dpdt): {jnp.linalg.norm(adjoint_dpdt):.3e}
Max |adjoint p0|: {jnp.max(jnp.abs(adjoint_p0)):.3e}
Min |adjoint p0|: {jnp.min(jnp.abs(adjoint_p0)):.3e}
Residual RMS: {jnp.sqrt(jnp.mean(residual**2)):.3e}
Number of snapshots: {n_snapshots}
"""
ax11.text(
    0.1,
    0.5,
    adj_stats_text,
    fontsize=10,
    family="monospace",
    verticalalignment="center",
)

plt.suptitle(
    "ADJOINT SOLUTION VISUALIZATION: k-WAVE TRUTH vs MSGB RECONSTRUCTION",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(
    PLOT_DIR / "adjoint_solution_visualization.png", dpi=150, bbox_inches="tight"
)
plt.show()


# ============================================
# PART 8: ADJOINT TEST WITH k-WAVE MEASUREMENTS
# ============================================
def adjoint_test(base_p0, base_dpdt, trials=5, seed=42):
    """
    Adjoint test to verify correctness of adjoint implementation.
    Tests if <J·v, w> = <v, J^T·w> for random v and w.
    """
    print("\nRunning adjoint test with k-Wave measurements...")
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
    print(f"Adjoint test {'PASSED ✓' if max_error < 1e-10 else 'FAILED ✗'}")

    return errors


# Run adjoint test
adjoint_errors = adjoint_test(recon_p0, recon_dpdt, trials=5)

# ============================================
# PART 9: SUMMARY REPORT
# ============================================
print("\n" + "=" * 70)
print("FINAL SUMMARY REPORT")
print("=" * 70)

print("\n1. FORWARD SOLVE:")
print(f"   - Grid: {N[0]}×{N[1]} points")
print(f"   - Physical size: {extent[0] * 1e3:.1f}×{extent[1] * 1e3:.1f} mm")
print(f"   - Grid spacing: {dx[0] * 1e6:.1f}×{dx[1] * 1e6:.1f} μm")
print(f"   - Time steps: {Nt}")
print(f"   - Time step: {dt * 1e6:.2f} μs")
print(f"   - Number of sensors: {len(sensor_idx)}")
print(f"   - Sensor configuration: {sensor_config}")
print("   - Ground truth: k-Wave")
print("   - Reconstruction solver: MSGB")

print("\n2. MEASUREMENT COMPARISON:")
print(f"   - Max |k-Wave measurements|: {jnp.max(jnp.abs(measurements)):.3e}")
print(f"   - Max |MSGB measurements|: {jnp.max(jnp.abs(msgb_measurements)):.3e}")
print(
    f"   - RMS difference: {jnp.sqrt(jnp.mean((measurements - msgb_measurements) ** 2)):.3e}"
)

print("\n3. GRADIENT-BASED RECONSTRUCTION:")
print("   - Optimization method: Adam")
print(f"   - Learning rate: {lr}")
print(f"   - Regularization: {regularization}")
print(f"   - Iterations: {iters}")
print(f"   - Final loss: {hist_loss[-1]:.3e}")
print(f"   - Final relative error: {final_error:.3e}")
print(f"   - Final gradient norm: {hist_grad_norm[-1]:.3e}")

print("\n4. ADJOINT SOLUTION:")
print(f"   - Adjoint field norm (p0): {jnp.linalg.norm(adjoint_p0):.3e}")
print(f"   - Adjoint field norm (dpdt): {jnp.linalg.norm(adjoint_dpdt):.3e}")
print(f"   - Number of snapshots computed: {n_snapshots}")
print(f"   - Mean adjoint test error: {np.mean(adjoint_errors):.3e}")
print(f"   - Max adjoint test error: {np.max(adjoint_errors):.3e}")
print(
    f"   - Adjoint test: {'✓ PASSED' if np.max(adjoint_errors) < 1e-10 else '✗ FAILED'}"
)

print("\n5. OUTPUT FILES:")
print(f"   - Forward comparison: {PLOT_DIR / 'forward_kwave_msgb_comparison.png'}")
print(f"   - Adjoint visualization: {PLOT_DIR / 'adjoint_solution_visualization.png'}")
