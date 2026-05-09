#!/usr/bin/env python
# coding: utf-8

"""
2D PAT inverse problem with a *linearised* MSGB forward map and JAX autodiff.

- Forward "truth": k-Wave
- Reconstruction model: MSGB with frozen beam support (linear operator)
- Optimization variable: p0 (initial pressure)
- Gradient: reverse-mode AD through the linearised MSGB forward map
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
from beamax.plotter import use_beamax_style
from beamax.solvers import KWaveSolver
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
    # homogeneous c for now
    return 1.0 + 0.0 * x[..., 0]


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
sensor_positions = sensors.positions  # (Ns, 2)
Ns = sensor_positions.shape[0]
print(f"Number of sensors: {Ns} (line at y=0)")


# ---------------------------------------------------------------------------
# PART 2: TRUE p0 AND k-WAVE FORWARD
# ---------------------------------------------------------------------------

p0_true = jnp.zeros(N)
p0_true = p0_true.at[N[0] // 2 - 3 : N[0] // 2 + 3, N[1] // 2 - 3 : N[1] // 2 + 3].set(
    1.0
)
dpdt_true = jnp.zeros_like(p0_true)

print(f"\n||p0_true||_2 = {jnp.linalg.norm(p0_true):.3e}")

print("\nComputing k-Wave forward (ground truth)...")
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
measurements = kwave_solver.forward(p0_true, domain, binary_mask, ts)  # (Nt, Ns)
t1 = time.time()
print(f"k-Wave forward time: {t1 - t0:.3f} s")
print(f"measurements shape: {measurements.shape}")


# ---------------------------------------------------------------------------
# PART 3: BUILD A LINEARISED MSGB FORWARD OPERATOR
# ---------------------------------------------------------------------------

"""
We construct a *linear* operator P_lin : p0 -> data by:

1. Choose a reference p0_ref and compute positive-frequency MSWPT coeffs c_pos_ref.
2. Threshold once to pick coefficient indices (support) idx_pos.
3. From idx_pos, build geometric GB parameters (p0s_base, M0s_base, x0s_base, ωs_base, a0s_base, modes_base).
4. For any p0, compute c_pos(p0), take coeff_vals = c_pos[idx_pos], and use these as amplitudes.

For real fields we mirror beams to negative frequencies.
This mapping is linear in p0, so JAX reverse-mode AD gives an exact discrete adjoint.
"""

input_type = "spatial"
thr_value = 1000
thr_strategy = "top_n"

ode_solver = gb_solvers.solve_hom_diag
ode_config = gb_solvers.SolverConfig.from_precision()
aggregate_method = "all"
use_real = True

print("\nPrecomputing linearised MSGB support...")

p0_ref = p0_true
dpdt_ref = jnp.zeros_like(p0_ref)

# 1) MSWPT positive-frequency coefficients (reference)
c_pos_ref = compute_coefficients(
    p0_ref,
    dpdt_ref,
    input_type,
    domain,
    wpt,
    mode="pos_only",
)

# 2) Threshold ONCE – outside grad graph
coeff_pos_idx, _ = threshold_coefficients(
    c_pos_ref, thr_value, strategy=thr_strategy, wpt=wpt
)
num_beams_pos = int(coeff_pos_idx.shape[0])
print(f"Number of positive-frequency beams (frozen): {num_beams_pos}")

# 3) Beam parameters from indices
(
    p0s_base,
    M0s_base,
    x0s_base,
    ωs_base,
    a0s_base,
    modes_base,
) = compute_forward_parameters(coeff_pos_idx, wpt, domain)

print(
    f"Base beam arrays: p0s={p0s_base.shape}, "
    f"M0s={M0s_base.shape}, x0s={x0s_base.shape}, "
    f"ωs={ωs_base.shape}, a0s={a0s_base.shape}"
)


def msgb_linear_forward(p0: jnp.ndarray) -> jnp.ndarray:
    """
    Linearised MSGB forward operator P_lin : p0 -> sensor data (Nt, Ns).

    - Compute positive-frequency MSWPT coefficients c_pos(p0).
    - Take coeff_vals = c_pos[idx_pos] as amplitudes (up to a0s_base factor).
    - Mirror beams for real field and propagate.
    """
    dpdt = jnp.zeros_like(p0)

    c_pos = compute_coefficients(
        p0,
        dpdt,
        input_type,
        domain,
        wpt,
        mode="pos_only",
    )  # (total_coeffs,)

    coeff_vals = c_pos[coeff_pos_idx]  # (num_beams_pos,)

    # attach to base amplitudes
    a0s = a0s_base * coeff_vals  # complex amplitudes

    # mirror to negative frequencies
    p0s = jnp.concatenate([p0s_base, p0s_base], axis=0)
    M0s = jnp.concatenate([M0s_base, M0s_base], axis=0)
    x0s = jnp.concatenate([x0s_base, x0s_base], axis=0)
    ωs = jnp.concatenate([ωs_base, ωs_base], axis=0)
    a0s_full = jnp.concatenate([a0s, a0s], axis=0)
    modes = jnp.concatenate([modes_base, -modes_base], axis=0)

    params = (p0s, M0s, x0s, ωs, a0s_full, modes)

    sensor_data = compute_forward_result(
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
    # (Nt, Ns)
    return sensor_data


# JIT the forward
msgb_linear_forward_jit = jax.jit(msgb_linear_forward)

# Quick forward check
print("\nMSGB linear forward on p0_true...")
t0 = time.time()
msgb_meas_true = msgb_linear_forward_jit(p0_true)
t1 = time.time()
print(f"MSGB linear forward time: {t1 - t0:.3f} s")
print(f"MSGB measurements shape: {msgb_meas_true.shape}")

rms_diff = jnp.sqrt(jnp.mean((measurements - msgb_meas_true) ** 2))
print(f"RMS difference (k-Wave vs MSGB-linear): {rms_diff:.3e}")


# ---------------------------------------------------------------------------
# PART 3.5: GRADIENT FLOW DIAGNOSTICS
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("GRADIENT FLOW DIAGNOSTICS")
print("=" * 60)

# Test at p0_true (should work)
print("\nTesting gradient flow at p0_true:")


def test_full_forward(p0):
    return jnp.sum(msgb_linear_forward(p0) ** 2)


try:
    grad_at_true = jax.grad(test_full_forward)(p0_true)
    print(f"  ||grad(p0_true)||_2 = {jnp.linalg.norm(grad_at_true):.3e} ✓")
except Exception as e:
    print(f"  FAILED: {e}")

# Test at p0=0 (likely zero gradient - degenerate case)
print("\nTesting gradient flow at p0=0 (degenerate case):")
p0_zero = jnp.zeros(N)
try:
    grad_at_zero = jax.grad(test_full_forward)(p0_zero)
    grad_zero_norm = float(jnp.linalg.norm(grad_at_zero))
    print(f"  ||grad(p0=0)||_2 = {grad_zero_norm:.3e}")
    if grad_zero_norm < 1e-12:
        print("  ⚠ WARNING: Gradient is zero at p0=0 (degenerate initialization)")
        print("  → This is expected due to multiplicative beam dynamics")
        print("  → Solution: Use non-zero initialization")
except Exception as e:
    print(f"  FAILED: {e}")

print("=" * 60)


# ---------------------------------------------------------------------------
# PART 4: LOSS + AUTODIFF SETUP
# ---------------------------------------------------------------------------

iters = 150
lr = 2e-2
lam_reg = 1e-6

# ---------------------------------------------------------------------------
# IMPORTANT: Use non-zero initialization to avoid degenerate gradient
# ---------------------------------------------------------------------------
# Option 1: Small random initialization (recommended)
key = jax.random.PRNGKey(42)
p0_init = 0.01 * jax.random.normal(key, p0_true.shape)

# Option 2: Small constant initialization
# p0_init = jnp.full_like(p0_true, 0.01)

# Option 3: Backprojection warm start (often converges faster)
# def compute_backprojection(meas):
#     """Compute J^T * measurements as a warm start."""
#     def forward_wrapper(p0):
#         return msgb_linear_forward_jit(p0)
#     _, vjp_fn = jax.vjp(forward_wrapper, jnp.ones(N) * 0.01)  # Need non-zero base
#     (backproj,) = vjp_fn(meas)
#     scale = jnp.linalg.norm(p0_true) / (jnp.linalg.norm(backproj) + 1e-12)
#     return backproj * scale * 0.1
# p0_init = compute_backprojection(measurements)

print(f"\nInitialization: ||p0_init||_2 = {jnp.linalg.norm(p0_init):.3e}")


def forward_sensor_data(p0_var: jnp.ndarray) -> jnp.ndarray:
    """Wrapper used for JVP/VJP and loss."""
    return msgb_linear_forward_jit(p0_var)


def loss_fn(p0_var: jnp.ndarray) -> jnp.ndarray:
    pred = forward_sensor_data(p0_var)
    data_fid = 0.5 * jnp.mean((pred - measurements) ** 2)
    reg = 0.5 * lam_reg * jnp.mean(p0_var**2)
    return data_fid + reg


# ---------------------------------------------------------------------------
# CRITICAL: Verify gradient is non-zero at initial point
# ---------------------------------------------------------------------------
print("\nSanity check: gradient at p0_init...")
grad0 = jax.grad(loss_fn)(p0_init)
grad0_norm = float(jnp.linalg.norm(grad0))
print(f"  ||grad loss(p0_init)||_2 = {grad0_norm:.3e}")

if grad0_norm < 1e-12:
    raise ValueError(
        "Gradient at initial guess is ~0. "
        "Try a different initialization (non-zero p0_init). "
        "This happens because the beam amplitude dynamics are multiplicative, "
        "so p0=0 → a0=0 → output=0 with zero Jacobian."
    )
else:
    print("  ✓ Gradient is non-zero, optimization should proceed normally")

# ---------------------------------------------------------------------------
# Setup optimizer
# ---------------------------------------------------------------------------
optimizer = optax.adam(lr)
opt_state = optimizer.init(p0_init)


@jax.jit
def optimisation_step(p0_var, opt_state):
    loss, grad = jax.value_and_grad(loss_fn)(p0_var)
    updates, opt_state = optimizer.update(grad, opt_state, params=p0_var)
    p0_next = optax.apply_updates(p0_var, updates)
    return p0_next, opt_state, loss, grad


print(f"\nRunning {iters} Adam iterations on p0...")
print(f"lr = {lr}, λ = {lam_reg}")
hist_loss = []
hist_err = []
hist_grad_norm = []

p0_recon = p0_init

for k in range(iters):
    p0_recon, opt_state, loss_val, grad = optimisation_step(p0_recon, opt_state)
    rel_err = jnp.linalg.norm(p0_recon - p0_true) / jnp.linalg.norm(p0_true)
    grad_norm = jnp.linalg.norm(grad)

    hist_loss.append(float(loss_val))
    hist_err.append(float(rel_err))
    hist_grad_norm.append(float(grad_norm))

    if k % 10 == 0 or k == iters - 1:
        print(
            f"  iter {k:3d}: loss={loss_val:.3e}, "
            f"rel_err={rel_err:.3e}, ||grad||={grad_norm:.3e}"
        )

final_err = hist_err[-1]
print(f"\nFinal rel reconstruction error: {final_err:.3e}")
print(f"Final loss: {hist_loss[-1]:.3e}")


# ---------------------------------------------------------------------------
# PART 5: ADJOINT TEST FOR THE LINEARISED FORWARD
# ---------------------------------------------------------------------------


def adjoint_apply(p0_base: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Compute J^T w at p0_base via VJP."""
    _, vjp_fn = jax.vjp(forward_sensor_data, p0_base)
    (g_p0,) = vjp_fn(w)
    return g_p0


def adjoint_test(p0_base: jnp.ndarray, trials: int = 5, seed: int = 42):
    print("\nAdjoint test: <Jv,w> vs <v,J^T w>...")
    key = jax.random.PRNGKey(seed)
    rel_errors = []

    for i in range(trials):
        key, k1, k2 = jax.random.split(key, 3)
        v = jax.random.normal(k1, p0_base.shape)
        w = jax.random.normal(k2, measurements.shape)

        # J v
        _, Jv = jax.jvp(forward_sensor_data, (p0_base,), (v,))
        # J^T w
        JTw = adjoint_apply(p0_base, w)

        lhs = jnp.vdot(Jv, w).real
        rhs = jnp.vdot(v, JTw).real

        abs_err = jnp.abs(lhs - rhs)
        denom = jnp.maximum(jnp.maximum(jnp.abs(lhs), jnp.abs(rhs)), 1e-12)
        rel_err = abs_err / denom
        rel_errors.append(float(rel_err))

        print(
            f"  Trial {i + 1}: <Jv,w>={lhs:+.6e}, <v,J^T w>={rhs:+.6e}, "
            f"rel_err={rel_err:.3e}"
        )

    print("-" * 40)
    print(f"Mean relative error: {np.mean(rel_errors):.3e}")
    print(f"Max  relative error: {np.max(rel_errors):.3e}")
    return rel_errors


adjoint_errors = adjoint_test(p0_recon, trials=5)


# ---------------------------------------------------------------------------
# PART 6: PLOTTING
# ---------------------------------------------------------------------------

print("\nGenerating plots...")

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.4)

# True p0
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

# Reconstructed p0
ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(
    p0_recon,
    origin="lower",
    cmap="RdBu_r",
    extent=[0, extent[0] * 1e3, 0, extent[1] * 1e3],
    vmin=float(p0_true.min()),
    vmax=float(p0_true.max()),
)
ax.set_title(f"Reconstruction\nrel_err={final_err:.2e}")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Error map
ax = fig.add_subplot(gs[0, 2])
err_map = np.abs(np.array(p0_recon - p0_true))
im = ax.imshow(
    err_map,
    origin="lower",
    cmap="hot",
    extent=[0, extent[0] * 1e3, 0, extent[1] * 1e3],
)
ax.set_title("|p0_recon - p0_true|")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# One sensor trace (k-Wave vs MSGB lin at true p0)
ax = fig.add_subplot(gs[0, 3])
sensor_id = Ns // 2
ax.plot(ts * 1e3, measurements[:, sensor_id], label="k-Wave", lw=1.5)
ax.plot(ts * 1e3, msgb_meas_true[:, sensor_id], "--", label="MSGB lin", lw=1.5)
ax.set_title(f"Sensor {sensor_id} trace")
ax.set_xlabel("t (ms)")
ax.set_ylabel("Amplitude")
ax.legend()
ax.grid(alpha=0.3)

# Loss
ax = fig.add_subplot(gs[1, 0])
ax.semilogy(hist_loss, "r-")
ax.set_title("Loss")
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.grid(alpha=0.3, which="both")

# Relative error
ax = fig.add_subplot(gs[1, 1])
ax.semilogy(hist_err, "b-")
ax.set_title("Relative reconstruction error")
ax.set_xlabel("Iteration")
ax.set_ylabel("Rel. error")
ax.grid(alpha=0.3, which="both")

# Gradient norm
ax = fig.add_subplot(gs[1, 2])
ax.semilogy(hist_grad_norm, "g-")
ax.set_title("Gradient norm")
ax.set_xlabel("Iteration")
ax.set_ylabel("||∇L||")
ax.grid(alpha=0.3, which="both")

# Residual waterfall for true p0
ax = fig.add_subplot(gs[1, 3])
time_mesh, sensor_mesh = jnp.meshgrid(ts * 1e3, jnp.arange(Ns))
residual_true = (measurements - msgb_meas_true).T
max_res = float(jnp.max(jnp.abs(residual_true)))
pc = ax.pcolormesh(
    time_mesh,
    sensor_mesh,
    residual_true,
    shading="auto",
    cmap="RdBu_r",
    vmin=-max_res,
    vmax=max_res,
)
ax.set_title("Residual (k-Wave - MSGB lin)")
ax.set_xlabel("t (ms)")
ax.set_ylabel("Sensor index")
plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)

# Adjoint stats
ax = fig.add_subplot(gs[2, :])
ax.axis("off")
text = f"""
Adjoint check (linearised MSGB forward)
---------------------------------------
Trials: {len(adjoint_errors)}
Mean rel error: {np.mean(adjoint_errors):.3e}
Max  rel error: {np.max(adjoint_errors):.3e}

Forward operator:
  - Linearised MSGB with frozen positive-frequency support
  - Pos beams: {num_beams_pos} (2× for real field)
  - aggregate_method = '{aggregate_method}', use_real = {use_real}

Optimisation:
  - Adam, lr={lr}, λ={lam_reg}, iters={iters}
  - Final rel error={final_err:.3e}
  - Final loss={hist_loss[-1]:.3e}

Initialization:
  - Used non-zero random initialization to avoid degenerate gradient at p0=0
  - ||p0_init||={jnp.linalg.norm(p0_init):.3e}
  - ||∇L(p0_init)||={grad0_norm:.3e}
"""
ax.text(0.01, 0.95, text, va="top", ha="left", family="monospace")

plt.suptitle("2D PAT: k-Wave vs Linearised MSGB + AD Inversion", fontsize=14)
plt.tight_layout()
fig_path = PLOT_DIR / "AD-2d3_linear_msgb.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"\nSaved figure: {fig_path}")
print("\nDone.")
