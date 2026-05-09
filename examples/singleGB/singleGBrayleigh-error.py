#!/usr/bin/env python
# coding: utf-8



"""
Single Gaussian beam Rayleigh-range / focal-error diagnostic.
"""
import jax
import jax.numpy as jnp

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from time import time

from beamax import geometry, plotter, utils
from beamax.gb import core, gb_utils, gb_solvers
from beamax.solvers.kwave_solver import TimedKWaveSolver
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions

ROOT_DIR = utils.detect_root()
CACHE_DIR = Path(ROOT_DIR / "cache")
PLOT_DIR = Path(ROOT_DIR / "plots")
CACHE_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)
pltgb = plotter.PlotHelper()

# -------------------- domain --------------------
b = 1
d = 2
N = (256,) * d
dx = (3.0 / N[0],) * d  # physical size ≈ (3,3)
periodic = (False,) * d


def c(x):
    return 1.0 + 0.0 * x[..., 0]


c0 = 1.0
cfl = 0.3
lam = 0.0
domain = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)
XY = domain.grid
domain_size = domain.grid_size

# Time grid: must cover r_max * zR. With alpha=1 -> zR=0.5; r_max=3 -> need t_end >= 1.5
ts = jnp.linspace(0.0, 3.0, 200)

# -------------------- GB initialization --------------------
mode = jnp.ones((b,))
# start near left, propagate +x
x0 = jnp.array([[0.2 * domain_size[0], 0.5 * domain_size[1]]])  # (b,d)
p0 = jnp.array([[1.0, 0.0]])
p0 = p0 / jnp.linalg.norm(p0, axis=-1, keepdims=True)
a0 = jnp.ones((b,))

# Isotropic curvature: M0 = i*alpha*I (alpha>0) and carrier omega
alpha_scalar = 1.0
alpha0 = 1j * jnp.ones((b, d)) * alpha_scalar
ω0 = jnp.ones((b,)) * 50.0

M0 = gb_utils.prepare_M0(alpha0, None)
print("Is M0 diagonal?", bool(gb_utils.is_diagonal(M0)))

# Ray/GB solver
solver = gb_solvers.solve_ODE_base
solver_config = None

# -------------------- Ray + GB field --------------------
t1 = time()
(xt, pt, mt, at) = solver(x0, p0, M0, a0, mode, ts, c, lam, solver_config)
# IMPORTANT: xt has shape (b, T, d), not (T, b, d)
xt0 = xt[0, :, :]  # (T, d) centerline for beam 0

u_all = core.compute_gaussian_beam(
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
)  # (T, *N, b)
t2 = time()
print(f"Time to compute GB (sum of {b} beams): {t2 - t1:.3f}s")

# Sum beams; compare real parts to k-Wave
u_gb = jnp.sum(u_all, axis=-1)  # (T, *N)
u_gb_real = jnp.real(u_gb)

# -------------------- Arclength s(t) --------------------
# Robust arclength using the actual ray path; handles hetero c, curvature
dx_steps = jnp.linalg.norm(xt0[1:] - xt0[:-1], axis=-1)  # (T-1,)
s_of_t = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dx_steps)])  # (T,)

# -------------------- Rayleigh range zR --------------------
# For M0 = i*alpha*I and constant c: zR = 1/(2 c alpha)
zR = 1.0 / (2.0 * c0 * alpha_scalar)  # -> 0.5 with alpha=1, c=1
print(f"Rayleigh range zR = {zR:.6g}")

# Normalized distance samples r in [0, 3] -> s_targets = r*zR
r_values = jnp.linspace(0.0, 3.0, 60)
s_targets = r_values * zR

# Clip if time window too short
if float(s_targets[-1]) > float(s_of_t[-1]):
    max_r = float(s_of_t[-1] / zR)
    print(f"WARNING: time window ends at r ≈ {max_r:.2f}; clipping r grid.")
    mask = s_targets <= s_of_t[-1]
    r_values = r_values[mask]
    s_targets = s_targets[mask]


# Map desired distances to nearest time indices
def nearest_time_indices(s_targets, s_of_t):
    idxs = []
    for s in list(s_targets):
        i = int(jnp.clip(jnp.searchsorted(s_of_t, s), 1, len(s_of_t) - 1))
        i = i if abs(float(s_of_t[i] - s)) < abs(float(s_of_t[i - 1] - s)) else (i - 1)
        idxs.append(i)
    return jnp.array(idxs, dtype=int)


t_idx_targets = nearest_time_indices(s_targets, s_of_t)  # (R,)

# The Rayleigh plane location (point on centerline at s = zR)
t_idx_R = int(jnp.searchsorted(s_of_t, zR))
t_idx_R = int(jnp.clip(t_idx_R, 0, len(s_of_t) - 1))
x_R = xt0[t_idx_R, 0]  # x-coordinate of centerline at zR

# -------------------- k-Wave reference --------------------
simulation_options = SimulationOptions(
    data_cast="double",
    smooth_p0=False,
    save_to_disk=True,
)
execution_options = SimulationExecutionOptions(
    is_gpu_simulation=False,
    delete_data=False,
    verbose_level=0,
)
kwave_solver = TimedKWaveSolver(simulation_options, execution_options)

sensors_all = jnp.ones(N)
u0_init_real = u_gb_real[0, ...]  # physical initial pressure

pt_kwave_all, runtime = kwave_solver.forward(u0_init_real, domain, sensors_all, ts)
pt_kwave_all = jnp.transpose(
    pt_kwave_all.reshape(-1, *N), (0, len(N), *(range(1, len(N))))
)

assert jnp.allclose(pt_kwave_all[0, ...], u0_init_real), "k-Wave t=0 mismatch."

# -------------------- Errors at those distances --------------------
axis = tuple(range(1, 1 + len(N)))
diff_at_idx = pt_kwave_all[t_idx_targets, ...] - u_gb_real[t_idx_targets, ...]
true_at_idx = pt_kwave_all[t_idx_targets, ...]
l2_err = jnp.sqrt(jnp.sum(diff_at_idx**2, axis=axis))  # (R,)
l2_true = jnp.sqrt(jnp.sum(true_at_idx**2, axis=axis))  # (R,)
rel_l2 = l2_err / jnp.maximum(l2_true, 1e-16)
linf_err = jnp.max(jnp.abs(diff_at_idx), axis=axis)

# -------------------- Plots vs r = z/zR --------------------
plt.figure(figsize=(7, 5))
plt.plot(np.array(r_values), np.array(l2_err), marker="o")
plt.xlabel(r"normalized distance $r = z/z_R$")
plt.ylabel(r"$\|u - u_{\rm GB}\|_2$")
plt.title("Absolute L2 error vs Rayleigh multiples")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "error_vs_rayleigh_abs.png", dpi=160)

plt.figure(figsize=(7, 5))
plt.plot(np.array(r_values), np.array(rel_l2), marker="o")
plt.xlabel(r"normalized distance $r = z/z_R$")
plt.ylabel(r"$\|u - u_{\rm GB}\|_2 / \|u\|_2$")
plt.title("Relative L2 error vs Rayleigh multiples")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "error_vs_rayleigh_rel.png", dpi=160)

plt.figure(figsize=(7, 5))
plt.plot(np.array(r_values), np.array(linf_err), marker="o")
plt.xlabel(r"normalized distance $r = z/z_R$")
plt.ylabel(r"$\|u - u_{\rm GB}\|_\infty$")
plt.title("L∞ error vs Rayleigh multiples")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "error_vs_rayleigh_linf.png", dpi=160)
plt.show()

# -------------------- GB snapshots with Rayleigh plane overlaid --------------------
# pick snapshot distances (in Rayleigh units), clip to available range
r_snaps = jnp.array([0.0, 0.5, 1.0, 2.0, 3.0])
s_snaps = r_snaps * zR
mask_snaps = s_snaps <= s_of_t[-1]
r_snaps = r_snaps[mask_snaps]
t_idx_snaps = nearest_time_indices(s_snaps[mask_snaps], s_of_t)

# spatial extent for imshow
X0, X1 = float(jnp.min(XY[..., 0])), float(jnp.max(XY[..., 0]))
Y0, Y1 = float(jnp.min(XY[..., 1])), float(jnp.max(XY[..., 1]))

n = int(len(r_snaps))
cols = 3
rows = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.8 * rows), squeeze=False)
axes = axes.ravel()

for k, (ax, tidx, r) in enumerate(
    zip(axes, list(map(int, t_idx_snaps)), list(map(float, r_snaps)))
):
    if k >= n:
        ax.axis("off")
        continue
    img = ax.imshow(
        np.array(u_gb_real[tidx, ...]).T,
        extent=[X0, X1, Y0, Y1],
        origin="lower",
        cmap="RdBu_r",
    )
    ax.set_title(f"GB real part @ r={r:.2f} (t≈{float(ts[tidx]):.2f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # Rayleigh plane x = x_R
    ax.axvline(float(x_R), linestyle="--")
    fig.colorbar(img, ax=ax, shrink=0.8)

# turn off unused panes
for ax in axes[n:]:
    ax.axis("off")

plt.tight_layout()
plt.savefig(PLOT_DIR / "gb_snapshots_with_rayleigh.png", dpi=180)
plt.show()

# -------------------- crude slope (mid-range) --------------------
r_np = np.array(r_values)
rel_np = np.array(rel_l2)
mask = (r_np >= 0.5) & (r_np <= 2.5) & np.isfinite(rel_np) & (rel_np > 0)
if mask.sum() >= 3:
    p = np.polyfit(np.log(r_np[mask]), np.log(rel_np[mask]), 1)[0]
    print(f"Empirical log-log slope of relative error vs r on [0.5, 2.5]: {p:.3f}")
else:
    print("Not enough valid points in [0.5, 2.5] to estimate a slope.")
