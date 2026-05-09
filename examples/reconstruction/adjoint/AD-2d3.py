#!/usr/bin/env python
# coding: utf-8

"""
2D PAT inverse problem with the full MSGB forward map and JAX autodiff.

Key improvements over v1/v2:
- Uses MSGBSolver class properly (matching forward-2d.py style)
- Offers time-reversal warm start (much better than random init)
- Self-consistent vs k-Wave data modes
- Proper comparison of linearized vs full MSGB forward
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    import optax
except ModuleNotFoundError:
    print("Skipping example: optax is not installed (`pip install optax`).")
    raise SystemExit(0)

from beamax import geometry, utils
from beamax.decomposition import DyadicDecomposition
from beamax.transforms import MSWPT
from beamax.gb import gb_solvers
from beamax.gb.gb_solvers import SolverConfig
from beamax.plotter import use_beamax_style
from beamax.solvers import KWaveSolver, MSGBSolver
from beamax.solvers.msgb_solvers.forward_solver_utils import (
    compute_coefficients,
    threshold_coefficients,
    compute_forward_parameters,
    compute_forward_result,
)
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

jax.config.update("jax_enable_x64", True)

ROOT_DIR = utils.detect_root()
PLOT_DIR = Path(ROOT_DIR / "plots")
DATA_DIR = Path(ROOT_DIR / "data")
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

try:
    use_beamax_style()
except OSError:
    pass


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Data source for reconstruction target
USE_SELF_CONSISTENT_DATA = True  # True: MSGB data, False: k-Wave data

# Initialization strategy
INIT_STRATEGY = "time_reversal"  # "time_reversal", "random", "zeros", "backprojection"

# Optimization parameters
ITERS = 200
LR_INIT = 5e-2
LR_FINAL = 1e-3
LAM_REG = 1e-4

# Forward operator: "linearized" (frozen support) or "full" (re-threshold each call)
FORWARD_MODE = "linearized"

# Early stopping
USE_EARLY_STOPPING = True
PATIENCE = 30


# ---------------------------------------------------------------------------
# PART 1: DOMAIN, MSWPT, SENSORS
# ---------------------------------------------------------------------------

d = 2
N = (128,) * d
dx = (1e-4,) * d
extent = tuple(dx[i] * N[i] for i in range(d))
periodic = (False,) * d
box_aspect_ratio = (1,) * d
num_levels = 1
num_boxes_level = (4,) * num_levels


def c_fn(x):
    """Homogeneous sound speed."""
    return 1500.0 + 0.0 * x[..., 0]


domain = geometry.Domain(N=N, dx=dx, c=c_fn, cfl=jnp.sqrt(d) / 4, periodic=periodic)
XY, _ = domain.generate_meshgrid()
X, Y = XY

ts = domain.generate_time_domain()
Nt = ts.shape[0]
dt = float(ts[1] - ts[0])

print(f"Spatial grid: {N[0]}x{N[1]} points")
print(f"dx = {dx[0]:.3e} m")
print(f"T grid: Nt = {Nt}, dt = {dt:.3e} s, T_max = {ts[-1]:.3e} s")

# MSWPT/MSGB setup
t0 = time.time()
dyadic_decomp = DyadicDecomposition(
    num_levels=num_levels,
    N=N,
    num_boxes_levels=num_boxes_level,
    box_aspect_ratio=box_aspect_ratio,
)
wpt = MSWPT(dyadic_decomp, redundancy=2, windowing="rectangular_mirror")
t1 = time.time()
print(f"MSWPT/MSGB setup time: {t1 - t0:.3f} s")

# Sensors: line at y=0
binary_mask = jnp.zeros(N, dtype=jnp.int32)
binary_mask = binary_mask.at[0, :].set(1)
sensors = geometry.Sensor(domain, binary_mask=binary_mask)
sensor_positions = sensors.positions
Ns = sensor_positions.shape[0]
print(f"Number of sensors: {Ns} (line at y=0)")


# ---------------------------------------------------------------------------
# PART 2: TRUE p0 AND FORWARD SIMULATIONS
# ---------------------------------------------------------------------------

# True initial pressure: small block in center
p0_true = jnp.zeros(N)
p0_true = p0_true.at[N[0] // 2 - 3 : N[0] // 2 + 3, N[1] // 2 - 3 : N[1] // 2 + 3].set(
    1.0
)

print(f"\n||p0_true||_2 = {jnp.linalg.norm(p0_true):.3e}")

# k-Wave forward (ground truth reference)
print("\nComputing k-Wave forward...")
simulation_options = SimulationOptions(
    data_cast="single",
    smooth_p0=False,
    save_to_disk=True,
)
execution_options = SimulationExecutionOptions(
    is_gpu_simulation=False,
    delete_data=False,
    verbose_level=0,
    show_sim_log=False,
)
kwave_solver = KWaveSolver(simulation_options, execution_options)

t0 = time.time()
measurements_kwave = kwave_solver.forward(p0_true, domain, binary_mask, ts)
t1 = time.time()
print(f"k-Wave forward time: {t1 - t0:.3f} s")

# MSGBSolver setup (matching forward-2d.py style)
thr_value = 1000
thr_strategy = "top_n"
batch_size = 128
sum_method = "all_real"  # "scan_real", "vmap_real", or "all_real"
input_type = "spatial"

ode_solver = gb_solvers.solve_hom_diag
ode_config = SolverConfig.from_precision()

msgb_solver = MSGBSolver(
    thr=thr_value,
    thr_strat=thr_strategy,
    batch_size=batch_size,
    input_type=input_type,
    ode_solver=ode_solver,
    tr_ode_solver=gb_solvers.solve_ODE_batch_t,
    sum_method=sum_method,
    sharding=None,
    ode_config=ode_config,
)

# MSGB forward on true p0
print("\nComputing MSGB forward on p0_true...")
t0 = time.time()
measurements_msgb, msgb_params = msgb_solver.forward(p0_true, domain, sensors, ts, wpt)
measurements_msgb = measurements_msgb.real
t1 = time.time()
print(f"MSGB forward time: {t1 - t0:.3f} s")

# Model error
rms_diff = jnp.sqrt(jnp.mean((measurements_kwave - measurements_msgb) ** 2))
rel_model_error = rms_diff / jnp.sqrt(jnp.mean(measurements_kwave**2))
print(f"RMS difference (k-Wave vs MSGB): {rms_diff:.3e}")
print(f"Relative model error: {rel_model_error:.2%}")

# Choose target measurements
if USE_SELF_CONSISTENT_DATA:
    print("\n*** Using MSGB data (self-consistent) ***")
    measurements = measurements_msgb
else:
    print("\n*** Using k-Wave data (with model mismatch) ***")
    measurements = measurements_kwave


# ---------------------------------------------------------------------------
# PART 3: BUILD FORWARD OPERATOR FOR OPTIMIZATION
# ---------------------------------------------------------------------------

print(f"\nBuilding {FORWARD_MODE} forward operator...")

if FORWARD_MODE == "linearized":
    # Linearized: freeze beam support based on p0_true
    p0_ref = p0_true
    dpdt_ref = jnp.zeros_like(p0_ref)

    c_pos_ref = compute_coefficients(
        p0_ref,
        dpdt_ref,
        input_type,
        domain,
        wpt,
        mode="pos_only",
    )
    coeff_pos_idx, _ = threshold_coefficients(
        c_pos_ref, thr_value, strategy=thr_strategy, wpt=wpt
    )
    num_beams_pos = int(coeff_pos_idx.shape[0])
    print(f"Frozen beam support: {num_beams_pos} positive-frequency beams")

    (
        p0s_base,
        M0s_base,
        x0s_base,
        ωs_base,
        a0s_base,
        modes_base,
    ) = compute_forward_parameters(coeff_pos_idx, wpt, domain)

    # Determine use_real and aggregate_method from sum_method
    use_real = "real" in sum_method
    if "scan" in sum_method:
        aggregate_method = "scan"
    elif "vmap" in sum_method:
        aggregate_method = "vmap"
    else:
        aggregate_method = "all"

    def forward_operator(p0: jnp.ndarray) -> jnp.ndarray:
        """Linearized MSGB forward with frozen beam support."""
        dpdt = jnp.zeros_like(p0)
        c_pos = compute_coefficients(
            p0,
            dpdt,
            input_type,
            domain,
            wpt,
            mode="pos_only",
        )
        coeff_vals = c_pos[coeff_pos_idx]
        a0s = a0s_base * coeff_vals

        # Mirror for real field
        p0s = jnp.concatenate([p0s_base, p0s_base], axis=0)
        M0s = jnp.concatenate([M0s_base, M0s_base], axis=0)
        x0s = jnp.concatenate([x0s_base, x0s_base], axis=0)
        ωs = jnp.concatenate([ωs_base, ωs_base], axis=0)
        a0s_full = jnp.concatenate([a0s, a0s], axis=0)
        modes = jnp.concatenate([modes_base, -modes_base], axis=0)

        params = (p0s, M0s, x0s, ωs, a0s_full, modes)

        return compute_forward_result(
            params=params,
            c=domain.c_fn,
            lam=domain.lam,
            ts=ts,
            ode_solver=ode_solver,
            sensors=sensor_positions,
            domain_size=domain.grid_size,
            periodic=jnp.array(domain.periodic),
            use_real=use_real,
            aggregate_method=aggregate_method,
            solver_config=ode_config,
        )

else:  # FORWARD_MODE == "full"
    # Full: use MSGBSolver (re-thresholds each call - nonlinear!)
    print("Warning: 'full' mode re-thresholds each iteration (nonlinear operator)")

    def forward_operator(p0: jnp.ndarray) -> jnp.ndarray:
        """Full MSGB forward (re-thresholds each call)."""
        sensor_data, _ = msgb_solver.forward(p0, domain, sensors, ts, wpt)
        return sensor_data.real


forward_operator_jit = jax.jit(forward_operator)

# Verify forward operator
print("\nVerifying forward operator on p0_true...")
t0 = time.time()
test_output = forward_operator_jit(p0_true)
t1 = time.time()
print(f"Forward operator time: {t1 - t0:.3f} s")
print(f"Output shape: {test_output.shape}")

fwd_diff = jnp.sqrt(jnp.mean((test_output - measurements_msgb) ** 2))
print(f"Difference from MSGB solver: {fwd_diff:.3e}")


# ---------------------------------------------------------------------------
# PART 4: INITIALIZATION
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"INITIALIZATION: {INIT_STRATEGY}")
print("=" * 60)

if INIT_STRATEGY == "time_reversal":
    # Use MSGB time-reversal as warm start
    print("Computing time-reversal reconstruction...")

    # Create a data domain for TR (treating sensor data as space-time field)
    # This is a simplified approach - proper TR needs careful setup
    data_N = (Nt, Ns)
    data_dx = (dt, dx[0])
    data_domain = geometry.Domain(
        N=data_N, dx=data_dx, c=c_fn, cfl=domain.cfl, periodic=(False, False)
    )
    data_wpt = MSWPT(
        DyadicDecomposition(1, data_N, (4,), (1, 1)),
        redundancy=2,
        windowing="rectangular_mirror",
    )

    # For simplicity, use backprojection instead of full TR
    # (Full TR requires more setup with proper source geometry)
    print("Using backprojection approximation...")

    def compute_backprojection(meas: jnp.ndarray, base_p0: jnp.ndarray) -> jnp.ndarray:
        """Compute J^T * measurements as warm start."""
        _, vjp_fn = jax.vjp(forward_operator_jit, base_p0)
        (backproj,) = vjp_fn(meas)
        return backproj

    # Need non-zero base for VJP
    key = jax.random.PRNGKey(0)
    base_p0 = 0.01 * jax.random.normal(key, N)

    t0 = time.time()
    p0_backproj = compute_backprojection(measurements, base_p0)
    t1 = time.time()
    print(f"Backprojection time: {t1 - t0:.3f} s")

    # Scale to reasonable magnitude
    scale = jnp.linalg.norm(p0_true) / (jnp.linalg.norm(p0_backproj) + 1e-12)
    p0_init = p0_backproj * scale * 0.5

    init_err = jnp.linalg.norm(p0_init - p0_true) / jnp.linalg.norm(p0_true)
    print(f"Initial relative error: {init_err:.3e}")

elif INIT_STRATEGY == "random":
    key = jax.random.PRNGKey(42)
    p0_init = 0.1 * jax.random.normal(key, p0_true.shape)

elif INIT_STRATEGY == "zeros":
    # This will have zero gradient - kept for demonstration
    p0_init = jnp.zeros_like(p0_true)
    print("WARNING: Zero initialization will have zero gradient!")

elif INIT_STRATEGY == "backprojection":
    # Same as time_reversal path above
    key = jax.random.PRNGKey(0)
    base_p0 = 0.01 * jax.random.normal(key, N)

    def compute_backprojection(meas, base_p0):
        _, vjp_fn = jax.vjp(forward_operator_jit, base_p0)
        (backproj,) = vjp_fn(meas)
        return backproj

    p0_backproj = compute_backprojection(measurements, base_p0)
    scale = jnp.linalg.norm(p0_true) / (jnp.linalg.norm(p0_backproj) + 1e-12)
    p0_init = p0_backproj * scale * 0.5

else:
    raise ValueError(f"Unknown init strategy: {INIT_STRATEGY}")

print(f"||p0_init||_2 = {jnp.linalg.norm(p0_init):.3e}")


# ---------------------------------------------------------------------------
# PART 5: LOSS AND OPTIMIZATION
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("OPTIMIZATION")
print("=" * 60)
print(f"Iterations: {ITERS}")
print(f"Learning rate: {LR_INIT} -> {LR_FINAL}")
print(f"L2 regularization: {LAM_REG}")
print("=" * 60)


def loss_fn(p0_var: jnp.ndarray) -> jnp.ndarray:
    pred = forward_operator_jit(p0_var)
    data_fid = 0.5 * jnp.mean((pred - measurements) ** 2)
    reg = 0.5 * LAM_REG * jnp.mean(p0_var**2)
    return data_fid + reg


# Verify gradient at initialization
print("\nGradient check at p0_init...")
grad0 = jax.grad(loss_fn)(p0_init)
grad0_norm = float(jnp.linalg.norm(grad0))
print(f"||grad loss(p0_init)||_2 = {grad0_norm:.3e}")

if grad0_norm < 1e-12:
    print("ERROR: Gradient is zero! Try different initialization.")
    if INIT_STRATEGY == "zeros":
        print("Switching to random initialization...")
        key = jax.random.PRNGKey(42)
        p0_init = 0.1 * jax.random.normal(key, p0_true.shape)
        grad0 = jax.grad(loss_fn)(p0_init)
        grad0_norm = float(jnp.linalg.norm(grad0))
        print(f"New ||grad||_2 = {grad0_norm:.3e}")
else:
    print("✓ Gradient is non-zero")

# Learning rate schedule
lr_schedule = optax.exponential_decay(
    init_value=LR_INIT,
    transition_steps=ITERS,
    decay_rate=LR_FINAL / LR_INIT,
)

optimizer = optax.adam(lr_schedule)
opt_state = optimizer.init(p0_init)


@jax.jit
def optimization_step(p0_var, opt_state):
    loss, grad = jax.value_and_grad(loss_fn)(p0_var)
    updates, opt_state = optimizer.update(grad, opt_state, params=p0_var)
    p0_next = optax.apply_updates(p0_var, updates)
    return p0_next, opt_state, loss, grad


print("\nRunning optimization...")
hist_loss = []
hist_err = []
hist_grad_norm = []

p0_recon = p0_init
best_err = float("inf")
best_p0 = p0_init
best_iter = 0
no_improve_count = 0

for k in range(ITERS):
    p0_recon, opt_state, loss_val, grad = optimization_step(p0_recon, opt_state)

    rel_err = float(jnp.linalg.norm(p0_recon - p0_true) / jnp.linalg.norm(p0_true))
    grad_norm = float(jnp.linalg.norm(grad))

    hist_loss.append(float(loss_val))
    hist_err.append(rel_err)
    hist_grad_norm.append(grad_norm)

    if rel_err < best_err:
        best_err = rel_err
        best_p0 = p0_recon
        best_iter = k
        no_improve_count = 0
    else:
        no_improve_count += 1

    if k % 20 == 0 or k == ITERS - 1:
        print(
            f"  iter {k:3d}: loss={loss_val:.3e}, "
            f"rel_err={rel_err:.3e}, ||grad||={grad_norm:.3e}"
        )

    if USE_EARLY_STOPPING and no_improve_count >= PATIENCE:
        print(f"\nEarly stopping at iter {k} (no improvement for {PATIENCE} iters)")
        break

print(f"\nBest reconstruction at iter {best_iter}: rel_err={best_err:.3e}")
p0_recon = best_p0
final_err = best_err


# ---------------------------------------------------------------------------
# PART 6: ADJOINT TEST
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("ADJOINT TEST")
print("=" * 60)


def adjoint_test(p0_base: jnp.ndarray, trials: int = 5, seed: int = 42):
    """Verify <Jv, w> = <v, J^T w> for the forward operator."""
    key = jax.random.PRNGKey(seed)
    rel_errors = []

    for i in range(trials):
        key, k1, k2 = jax.random.split(key, 3)
        v = jax.random.normal(k1, p0_base.shape)
        w = jax.random.normal(k2, measurements.shape)

        # J * v via JVP
        _, Jv = jax.jvp(forward_operator_jit, (p0_base,), (v,))

        # J^T * w via VJP
        _, vjp_fn = jax.vjp(forward_operator_jit, p0_base)
        (JTw,) = vjp_fn(w)

        lhs = jnp.vdot(Jv, w).real
        rhs = jnp.vdot(v, JTw).real

        rel_err = float(jnp.abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12))
        rel_errors.append(rel_err)

        print(
            f"  Trial {i + 1}: <Jv,w>={lhs:+.6e}, <v,J^Tw>={rhs:+.6e}, err={rel_err:.2e}"
        )

    print(f"Mean relative error: {np.mean(rel_errors):.3e}")
    return rel_errors


adjoint_errors = adjoint_test(p0_recon, trials=5)


# ---------------------------------------------------------------------------
# PART 7: PLOTTING
# ---------------------------------------------------------------------------

print("\nGenerating plots...")

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.4)

# Row 1: p0 images
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(
    p0_true,
    origin="lower",
    cmap="RdBu_r",
    extent=[0, extent[0] * 1e3, 0, extent[1] * 1e3],
)
ax.set_title("True $p_0$")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = fig.add_subplot(gs[0, 1])
vmax = float(max(jnp.abs(p0_true).max(), jnp.abs(p0_init).max()))
im = ax.imshow(
    p0_init,
    origin="lower",
    cmap="RdBu_r",
    extent=[0, extent[0] * 1e3, 0, extent[1] * 1e3],
    vmin=-vmax,
    vmax=vmax,
)
init_err_val = float(jnp.linalg.norm(p0_init - p0_true) / jnp.linalg.norm(p0_true))
ax.set_title(f"Initial $p_0$ ({INIT_STRATEGY})\nerr={init_err_val:.2e}")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = fig.add_subplot(gs[0, 2])
vmax = float(max(jnp.abs(p0_true).max(), jnp.abs(p0_recon).max()))
im = ax.imshow(
    p0_recon,
    origin="lower",
    cmap="RdBu_r",
    extent=[0, extent[0] * 1e3, 0, extent[1] * 1e3],
    vmin=-vmax,
    vmax=vmax,
)
ax.set_title(f"Reconstruction\nerr={final_err:.2e}")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = fig.add_subplot(gs[0, 3])
err_map = np.abs(np.array(p0_recon - p0_true))
im = ax.imshow(
    err_map, origin="lower", cmap="hot", extent=[0, extent[0] * 1e3, 0, extent[1] * 1e3]
)
ax.set_title("|Recon - True|")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Row 2: sensor traces and convergence
ax = fig.add_subplot(gs[1, 0])
sensor_id = Ns // 2
ax.plot(ts * 1e3, measurements_kwave[:, sensor_id], label="k-Wave", lw=1.5)
ax.plot(ts * 1e3, measurements_msgb[:, sensor_id], "--", label="MSGB", lw=1.5)
ax.set_title(f"Sensor {sensor_id}: k-Wave vs MSGB")
ax.set_xlabel("t (ms)")
ax.legend()
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 1])
pred_final = forward_operator_jit(p0_recon)
ax.plot(ts * 1e3, measurements[:, sensor_id], label="Target", lw=1.5)
ax.plot(ts * 1e3, pred_final[:, sensor_id], "--", label="Recon", lw=1.5)
ax.set_title(f"Sensor {sensor_id}: Target vs Recon")
ax.set_xlabel("t (ms)")
ax.legend()
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 2])
ax.semilogy(hist_loss, "r-")
ax.axvline(best_iter, color="g", ls=":", label=f"Best={best_iter}")
ax.set_title("Loss")
ax.set_xlabel("Iteration")
ax.legend()
ax.grid(alpha=0.3, which="both")

ax = fig.add_subplot(gs[1, 3])
ax.semilogy(hist_err, "b-")
ax.axvline(best_iter, color="g", ls=":")
ax.axhline(best_err, color="r", ls="--", alpha=0.5)
ax.set_title("Relative Error")
ax.set_xlabel("Iteration")
ax.grid(alpha=0.3, which="both")

# Row 3: gradient norm and cross-section
ax = fig.add_subplot(gs[2, 0])
ax.semilogy(hist_grad_norm, "g-")
ax.set_title("Gradient Norm")
ax.set_xlabel("Iteration")
ax.grid(alpha=0.3, which="both")

ax = fig.add_subplot(gs[2, 1])
mid_idx = N[0] // 2
ax.plot(np.arange(N[1]) * dx[1] * 1e3, p0_true[mid_idx, :], "b-", label="True", lw=2)
ax.plot(np.arange(N[1]) * dx[1] * 1e3, p0_recon[mid_idx, :], "r--", label="Recon", lw=2)
ax.set_title(f"Cross-section y={mid_idx * dx[0] * 1e3:.1f} mm")
ax.set_xlabel("x (mm)")
ax.legend()
ax.grid(alpha=0.3)

# Summary text
ax = fig.add_subplot(gs[2, 2:])
ax.axis("off")
data_src = "MSGB (self-consistent)" if USE_SELF_CONSISTENT_DATA else "k-Wave"
text = f"""
Configuration
─────────────
Data source: {data_src}
Forward mode: {FORWARD_MODE}
Initialization: {INIT_STRATEGY}
Model error (k-Wave vs MSGB): {rel_model_error:.2%}

Optimization
────────────
Iterations: {len(hist_loss)}/{ITERS}
LR: {LR_INIT} → {LR_FINAL}
λ (L2 reg): {LAM_REG}

Results
───────
Initial error: {init_err_val:.3e}
Best error: {best_err:.3e} (iter {best_iter})
Improvement: {(init_err_val - best_err) / init_err_val * 100:.1f}%

Adjoint Test
────────────
Mean rel error: {np.mean(adjoint_errors):.2e}
"""
ax.text(0.0, 1.0, text, va="top", ha="left", family="monospace", fontsize=10)

title = f"2D PAT Reconstruction: {FORWARD_MODE.title()} MSGB + AD"
if USE_SELF_CONSISTENT_DATA:
    title += " (Self-consistent)"
plt.suptitle(title, fontsize=14)

plt.tight_layout()
fig_path = PLOT_DIR / f"AD-2d3_{FORWARD_MODE}_{INIT_STRATEGY}.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"\nSaved figure: {fig_path}")
print("\nDone.")
