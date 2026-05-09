#!/usr/bin/env python
# coding: utf-8
"""
Reconstruct p0 from k-Wave data using MSGB:
- k-Wave generates measured data d_kw
- Add AWGN at target SNR and whiten residuals
- MSGB provides A(p0); adjoint via dynamic VJP (re-linearized each iter)
- CGLS solves min 0.5 || W (A(p0) - d_noisy) ||_2^2
- Plots with sensors overlaid as red circles on the true p0 image
"""

import numpy as np
from pathlib import Path
from time import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from beamax import geometry, utils
from beamax.decomposition import DyadicDecomposition
from beamax.plotter import use_beamax_style
from beamax.transforms import MSWPT
from beamax.gb import gb_solvers
from beamax.solvers import MSGBSolver, ShardingStrategy, KWaveSolver
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
jax.config.update("jax_enable_x64", True)

ROOT_DIR = utils.detect_root()
PLOT_DIR = Path(ROOT_DIR / "plots")
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR = Path(ROOT_DIR / "data")
DATA_DIR.mkdir(exist_ok=True)
try:
    use_beamax_style()
except Exception:
    pass

# ---------------------------------------------------------------------
# Domain & sensors
# ---------------------------------------------------------------------
d = 2
N = (64,) * d  # (Ny, Nx) small for quick runs
dx = (1.0e-4,) * d  # (dy, dx) in meters
extent = (N[0] * dx[0], N[1] * dx[1])  # (Ly, Lx) in meters
periodic = (False,) * d
cfl = jnp.sqrt(d) / 4


def c_fn(x):
    return 1.0 + 0.0 * x[..., 0]  # constant c for clean comparisons


domain = geometry.Domain(N=N, dx=dx, c=c_fn, cfl=cfl, periodic=periodic)
ts = domain.generate_time_domain()
Nt = len(ts)
dt = float(ts[1] - ts[0])

print(f"Grid: {N}, dx={dx}, extent={extent}")
print(f"Time: Nt={Nt}, dt={dt:.3e}, T={float(ts[-1]):.3e}")

# Sensors: a horizontal line at y=0

radius = N[0] // 2 - 3
tol = 0.5
idx = jnp.indices(N)
c = jnp.array(N) // 2
d = jnp.sqrt((idx[0] - c[0]) ** 2 + (idx[1] - c[1]) ** 2)
binary_mask = (jnp.abs(d - radius) <= tol).astype(jnp.int32)
sensors = geometry.Sensor(binary_mask=binary_mask, domain=domain)

# binary_mask = jnp.zeros(N)
# binary_mask = binary_mask.at[0, :].set(1)  # top row active
# sensors = geometry.Sensor(domain, binary_mask=binary_mask)
# num_sensors = int(binary_mask.sum())
# print(f"Num sensors: {num_sensors}")

# For overlay in mm
sensor_idx = np.argwhere(np.array(binary_mask) == 1)  # (Ns, 2) of (iy, ix)
sensor_x_mm = sensor_idx[:, 1] * dx[1] * 1e3
sensor_y_mm = sensor_idx[:, 0] * dx[0] * 1e3
extent_mm = [0, extent[1] * 1e3, 0, extent[0] * 1e3]  # [x_min, x_max, y_min, y_max]

# ---------------------------------------------------------------------
# MSWPT & MSGB solver setup
# ---------------------------------------------------------------------
num_levels = 1
num_boxes_level = (4,)
redundancy = 2
windowing = "rectangular_mirror"
box_aspect_ratio = (1.0, 1.0)

dyadic_decomp = DyadicDecomposition(num_levels, N, num_boxes_level, box_aspect_ratio)
wpt = MSWPT(dyadic_decomp, redundancy, windowing)

threshold = 1000
strategy = "top_n"  # hard selection → piecewise linear
batch_size = 100
sum_method = "scan_real"
ode_solver = gb_solvers.solve_hom_diag

num_devices = jax.device_count()
mesh = jax.make_mesh((num_devices,), ("x",))
sharding = ShardingStrategy(mesh, beam_axis="x")

msgb_solver = MSGBSolver(
    thr=threshold,
    thr_strat=strategy,
    batch_size=batch_size,
    input_type="spatial",
    ode_solver=ode_solver,
    sum_method=sum_method,
    sharding=sharding,
)

# ---------------------------------------------------------------------
# Synthetic truth p0 (Gaussian blob)
# ---------------------------------------------------------------------

from beamax import transforms

KXY = dyadic_decomp.fourier_meshgrid

# pltgb.plot_centers(dyadic_decomp.centres_ndim)

boxhf = 44
boxlf = 10
# probably need to multiply by the ratio between (64,64) and the desired res.
khf = jnp.array([12, 14])
klf = jnp.array([12, 6])
kerft_hf = transforms.compute_frames(dyadic_decomp, boxhf, khf, KXY, redundancy, "none")
kerft_lf = transforms.compute_frames(dyadic_decomp, boxlf, klf, KXY, redundancy, "none")
p0 = utils.unitary_ifft(kerft_hf) + utils.unitary_ifft(kerft_lf)
p0 = p0.T.real
p0_true = p0 / jnp.max(p0)

# ---------------------------------------------------------------------
# k-Wave forward to create measured data (ground truth)
# ---------------------------------------------------------------------
simulation_options = SimulationOptions(
    data_cast="single", smooth_p0=False, save_to_disk=True
)
execution_options = SimulationExecutionOptions(
    is_gpu_simulation=False, delete_data=False, verbose_level=0, show_sim_log=False
)
kwave_solver = KWaveSolver(simulation_options, execution_options)

print("\nRunning k-Wave forward for measured data...")
t1 = time()
d_kw = kwave_solver.forward(p0_true.T, domain, binary_mask, ts)  # shape (Nt, Ns)
t2 = time()
print(f"k-Wave time: {t2 - t1:.2f}s; d_kw shape={tuple(d_kw.shape)}")


# ---------------------------------------------------------------------
# Add AWGN & build whitening
# ---------------------------------------------------------------------
def add_awgn(y, *, snr_db=None, sigma=None, key=jax.random.PRNGKey(0)):
    """Add white Gaussian noise by SNR (dB) or std sigma; returns (y_noisy, sigma)."""
    y = jnp.asarray(y)
    if sigma is None:
        if snr_db is None:
            return y, None
        Ntot = y.size
        y_norm = jnp.linalg.norm(y)
        sigma = (y_norm / (10 ** (snr_db / 20.0))) / jnp.sqrt(Ntot)
    n = jax.random.normal(key, y.shape) * sigma
    return y + n, sigma


SNR_DB = 0.0  # e.g., 20–40 dB typical
d_noisy, sigma = add_awgn(d_kw, snr_db=SNR_DB, key=jax.random.PRNGKey(123))
print(f"Noise sigma (implied): {float(sigma):.3e}")


def apply_W(y):  # Σ^{-1/2}
    return y / sigma


def apply_WT(y):  # = W for iid noise
    return y / sigma


# ---------------------------------------------------------------------
# MSGB forward A(p0) — keyword args only
# ---------------------------------------------------------------------
def A_fn(p0):
    sd, _ = msgb_solver.forward(
        p0=p0,
        domain=domain,
        sensors=sensors,
        ts=ts,
        wpt=wpt,
    )
    return sd.real  # (Nt, Ns)


# Weighted forward for solver
def A_w(p0):
    return apply_W(A_fn(p0))


# ---------------------------------------------------------------------
# Dynamic VJP adjoint (re-linearize at current iterate) with whitening
# ---------------------------------------------------------------------
def make_ATw_at(x):
    """Build weighted adjoint at x: ATw(r) = A^T (W^T r)."""

    def A_single(p):
        return A_fn(p)

    _, pullback = jax.vjp(A_single, x)

    def ATw(r):
        (g_p0,) = pullback(apply_WT(r))
        return g_p0

    return ATw


# Quick derivative sanity (optional)
key = jax.random.PRNGKey(0)
v = jax.random.normal(key, p0_true.shape)
Av, Jv = jax.jvp(lambda p: A_fn(p), (p0_true,), (v,))
jvp_norm = float(jnp.linalg.norm(Jv))
print(f"\nSanity: ||A'(p0_true)[v]||_2 = {jvp_norm:.3e}")


# ---------------------------------------------------------------------
# CGLS (dynamic VJP) to solve min 0.5 || W(A(p0)-d_noisy) ||^2
# ---------------------------------------------------------------------
def cgls_dynamic(Aw, make_ATw, b_w, x0, iters=30, verbose=True):
    """
    Re-linearizes at current x each iteration:
        r_w = b_w - A_w(x)        (weighted residual)
        s   = A^T( W^T r_w )      (unweighted normal residual)
    """
    x = x0
    r_w = b_w - Aw(x)
    ATw = make_ATw(x)
    s = ATw(r_w)
    p = s
    gamma = jnp.vdot(s, s).real

    loss_hist, rel_hist = [], []
    for k in range(1, iters + 1):
        Ap_w = Aw(p)
        denom = jnp.vdot(Ap_w, Ap_w).real + 1e-30
        alpha = gamma / denom

        x = x + alpha * p
        r_w = r_w - alpha * Ap_w

        ATw = make_ATw(x)  # re-linearize
        s_new = ATw(r_w)
        gamma_new = jnp.vdot(s_new, s_new).real
        beta = gamma_new / (gamma + 1e-30)

        p = s_new + beta * p
        s = s_new
        gamma = gamma_new

        loss = 0.5 * float(jnp.linalg.norm(r_w) ** 2)
        rel = float(jnp.linalg.norm(x - p0_true) / (jnp.linalg.norm(p0_true) + 1e-30))
        loss_hist.append(loss)
        rel_hist.append(rel)

        if verbose and (k % max(1, iters // 10) == 0):
            print(f"  iter {k:3d}: loss={loss:.3e}, relerr={rel:.3e}")

    return x, {"loss": loss_hist, "relerr": rel_hist}


print("\nRunning CGLS (weighted, dynamic VJP)…")
x0 = jnp.ones_like(p0_true) * 0.05  # avoid empty active set at zero
bw = apply_W(d_noisy)  # whitened data
t1 = time()
p0_rec, hist = cgls_dynamic(A_w, make_ATw_at, bw, x0, iters=10, verbose=True)
t2 = time()
print(f"CGLS time: {t2 - t1:.2f}s; final relerr={hist['relerr'][-1]:.3e}")

# ---------------------------------------------------------------------
# Plots with sensor overlay
# ---------------------------------------------------------------------
vmin = float(jnp.min(p0_true))
vmax = float(jnp.max(p0_true))

fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# True p0 with sensors
im0 = axs[0].imshow(
    np.asarray(p0_true),
    origin="lower",
    cmap="RdBu_r",
    vmin=vmin,
    vmax=vmax,
    extent=extent_mm,
    interpolation="nearest",
)
axs[0].scatter(
    sensor_x_mm,
    sensor_y_mm,
    s=36,
    facecolors="none",
    edgecolors="red",
    linewidths=1.5,
    marker="o",
    label="Sensors",
)
axs[0].set_title("True $p_0$ (sensors in red)")
axs[0].set_xlabel("x (mm)")
axs[0].set_ylabel("y (mm)")
axs[0].legend(loc="upper right", frameon=False)
plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

# Reconstruction
im1 = axs[1].imshow(
    np.asarray(p0_rec),
    origin="lower",
    cmap="RdBu_r",
    vmin=vmin,
    vmax=vmax,
    extent=extent_mm,
    interpolation="nearest",
)
axs[1].set_title(f"Reconstruction (rel={hist['relerr'][-1]:.2e})")
axs[1].set_xlabel("x (mm)")
axs[1].set_ylabel("y (mm)")
plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

# Absolute error
im2 = axs[2].imshow(
    np.abs(np.asarray(p0_rec - p0_true)),
    origin="lower",
    cmap="hot",
    extent=extent_mm,
    interpolation="nearest",
)
axs[2].set_title("|Error|")
axs[2].set_xlabel("x (mm)")
axs[2].set_ylabel("y (mm)")
plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(PLOT_DIR / "recon_kwave_msgb_whitened.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

# Curves
plt.figure(figsize=(5, 3))
plt.semilogy(hist["loss"], "r-")
plt.grid(True, which="both", alpha=0.3)
plt.title("Weighted data loss")
plt.xlabel("iteration")
plt.ylabel("0.5 ||W(Ax-d)||^2")
plt.tight_layout()
plt.savefig(PLOT_DIR / "recon_kwave_msgb_loss.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(5, 3))
plt.semilogy(hist["relerr"], "b-")
plt.grid(True, which="both", alpha=0.3)
plt.title("Reconstruction relative error")
plt.xlabel("iteration")
plt.ylabel("||x-p*||/||p*||")
plt.tight_layout()
plt.savefig(PLOT_DIR / "recon_kwave_msgb_relerr.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

print("\nSaved:")
print(f"  {PLOT_DIR / 'recon_kwave_msgb_whitened.png'}")
print(f"  {PLOT_DIR / 'recon_kwave_msgb_loss.png'}")
print(f"  {PLOT_DIR / 'recon_kwave_msgb_relerr.png'}")
