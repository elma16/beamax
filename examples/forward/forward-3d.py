#!/usr/bin/env python
# coding: utf-8

"""
3D forward solve on an OA-Breast phantom comparing MSGB, k-Wave, and the hybrid MSGB + low-frequency solver on a planar sensor array. Reports relative L2 sensor errors and saves orthogonal MIP figures of the initial pressure, MSWPT coefficients, and reconstruction error. Requires `[kwave]`.
"""
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1 import make_axes_locatable

from beamax import geometry, utils, plotter
from beamax.decomposition import DyadicDecomposition
from beamax.transforms import MSWPT
from beamax.gb import gb_solvers
from beamax.solvers import KWaveSolver, MSGBSolver, HybridSolver, ShardingStrategy

from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions

# ============================================================================
# CONFIGURATION
# ============================================================================
jax.config.update("jax_enable_x64", False)

ROOT_DIR = utils.detect_root()
PLOT_DIR = Path(ROOT_DIR / "plots")
DATA_DIR = Path(ROOT_DIR / "data")
PLOT_DIR.mkdir(exist_ok=True)
pltgb = plotter.PlotHelper()


# ============================================================================
# DOMAIN SETUP
# ============================================================================
d = 3
N = (64,) * d
dx = (1e-4,) * d
periodic = (True,) * d
box_aspect_ratio = (1,) * d
num_levels = 2
num_boxes_levels = tuple([2 ** (i + 2) for i in range(num_levels)])


def c_fn(x):
    return 1500 + 0 * x[..., 0]


cfl = (jnp.sqrt(3) / 4).round(3)
domain = geometry.Domain(N=N, dx=dx, c=c_fn, cfl=cfl, periodic=periodic)
ts = domain.generate_time_domain()

print(f"Grid: {N}, dx={dx}")
print(f"Time steps: {len(ts)}, dt={float(ts[1] - ts[0]):.3e}s")

# ============================================================================
# LOAD DATA (OA-BREAST)
# ============================================================================
phantom_path = DATA_DIR / "NumericalBreastPhantoms-selected/hdf5/Neg_07_Left.h5"

p0, c_3d, meta3d = utils.load_oabreast_p0_c(
    phantom_path,
    dim="3d",
    axis_order="ZYX",
    source_spacing_mm=dx,
    target_shape=N,
    normalize_p0=False,
)

# Update c function
# c_fn = utils.make_c_function_from_grid(c_3d, dx, (0, 0, 0))
domain = geometry.Domain(N=N, dx=dx, c=c_fn, cfl=cfl, periodic=periodic)

p0 = p0.real
print(f"p0 range: [{p0.min():.2e}, {p0.max():.2e}]")

# ============================================================================
# SENSORS (planar array at z=0)
# ============================================================================
binary_mask = jnp.zeros(N).at[..., 0].set(1)
sensors = geometry.Sensor(binary_mask=binary_mask, domain=domain)
num_sensors = int(binary_mask.sum())
print(f"Sensors: {num_sensors} points")

# ============================================================================
# MSWPT & COEFFICIENT THRESHOLDING
# ============================================================================
windowing = "rectangular_mirror"
redundancy = 2

dyadic_decomp = DyadicDecomposition(num_levels, N, num_boxes_levels, box_aspect_ratio)
wpt = MSWPT(dyadic_decomp, redundancy, windowing)
wpt_none = MSWPT(dyadic_decomp, redundancy, "none")

coeffs = wpt.forward(p0, input_type="spatial")
coeffs_array = wpt.convert_to_array(coeffs)

# Adaptive thresholding
tau = 0.01
K = utils.choose_K_by_tau(
    coeffs, p0, wpt_none, dyadic_decomp, wpt, tau=tau, Kmin=512, Kmax=None, num_steps=10
)
print(f"Chosen K={K} for τ={tau:.1%}")

indices, coeff_vals = utils.select_levelaware_topK_indices(
    coeffs, dyadic_decomp, wpt, K
)
# threshold = int(indices.size)
threshold = 100

# Reconstruct from thresholded coefficients
thresholded_coeffs = jnp.zeros_like(coeffs).at[indices].set(coeff_vals)
data_recon = wpt_none.inverse(thresholded_coeffs, output_type="spatial")

# ============================================================================
# SOLVERS
# ============================================================================
batch_size = 100
sum_method = "scan_real"
strategy = "top_n"

# MSGB
num_devices = jax.device_count()
mesh = jax.make_mesh((num_devices,), ("x",))
sharding_strategy = ShardingStrategy(mesh, beam_axis="x")

msgb_solver = MSGBSolver(
    thr=threshold,
    thr_strat=strategy,
    batch_size=batch_size,
    input_type="spatial",
    ode_solver=gb_solvers.solve_hom_diag,
    sum_method=sum_method,
    sharding=sharding_strategy,
)

# k-Wave
simulation_options = SimulationOptions(
    data_cast="double",
    smooth_p0=False,
    save_to_disk=True,
)
execution_options = SimulationExecutionOptions(
    is_gpu_simulation=False, delete_data=False, verbose_level=0, show_sim_log=False
)
kwave_solver = KWaveSolver(simulation_options, execution_options)

# Hybrid
cutoff_freq = 16
hybrid_solver = HybridSolver(
    lf_solver=kwave_solver,
    hf_solver=msgb_solver,
    downsample=False,
    cutoff_freq=cutoff_freq,
    input_type="spatial",
    interp_method="fourier",
    dt_oversample=0,
    beta=12.0,
)

# ============================================================================
# RUN FORWARD SIMULATIONS
# ============================================================================
print("\n" + "=" * 60)
print("RUNNING FORWARD SIMULATIONS")
print("=" * 60)

# MSGB
print("\nMSGB...")
t1 = time()
gb_data = msgb_solver.forward(p0, domain, sensors, ts, wpt)[0].block_until_ready()
gb_data = gb_data.reshape(len(ts), N[0], N[1]).transpose(0, 2, 1)
t_msgb = time() - t1
print(f"  Time: {t_msgb:.2f}s")

# k-Wave
print("\nk-Wave...")
t1 = time()
kw_data = kwave_solver.forward(p0, domain, sensors.binary_mask, ts)
kw_data = kw_data.reshape(len(ts), N[0], N[1]).transpose(0, 2, 1)
t_kwave = time() - t1
print(f"  Time: {t_kwave:.2f}s")

# Hybrid
print("\nHybrid...")
t1 = time()
hyb_data = hybrid_solver.forward(p0, domain, sensors, ts, wpt)
hyb_data = hyb_data.reshape(len(ts), N[0], N[1]).transpose(0, 2, 1)
t_hybrid = time() - t1
print(f"  Time: {t_hybrid:.2f}s")

# Error metrics
print("\n" + "=" * 60)
print("ERROR METRICS")
print("=" * 60)
err_msgb = jnp.linalg.norm(gb_data - kw_data) / jnp.linalg.norm(kw_data)
err_hyb = jnp.linalg.norm(hyb_data - kw_data) / jnp.linalg.norm(kw_data)
print(f"MSGB  rel L2 error: {100 * err_msgb:.2f}%")
print(f"Hybrid rel L2 error: {100 * err_hyb:.2f}%")


# ============================================================================
# PLOTTING UTILITIES
# ============================================================================
def mip(arr, axis):
    """Maximum intensity projection."""
    return np.max(arr, axis=axis)


def shared_norm(arr_list):
    """Shared normalization across arrays."""
    vmin = min(float(np.min(a)) for a in arr_list)
    vmax = max(float(np.max(a)) for a in arr_list)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def symmetric_norm(arr_list):
    """Symmetric normalization for difference plots."""
    m = max(float(np.max(np.abs(a))) for a in arr_list)
    return mcolors.Normalize(vmin=-m, vmax=+m)


def add_colorbar(fig, ax, im, where="right", size="5%", pad=0.08):
    """Add colorbar to axis."""
    div = make_axes_locatable(ax)
    cax = div.append_axes(where, size=size, pad=pad)
    fig.colorbar(im, cax=cax)
    return cax


# ============================================================================
# FIGURE 1: INITIAL CONDITIONS
# ============================================================================
print("\nGenerating Figure 1: Initial Conditions...")

# Prepare data
_p0 = np.asarray(p0)
_c3d = np.asarray(c_3d)
_coeffs_mag = np.asarray(np.abs(coeffs_array))
_recon_err = np.asarray(np.abs(data_recon.real - p0))

# MIP projections
p0_mip = mip(_p0, axis=2)
c_mip = mip(_c3d, axis=2)
coeff_mip = mip(_coeffs_mag, axis=2)
err_mip = mip(_recon_err, axis=2)

# Create figure
fig1 = plt.figure(figsize=(14, 3.5))
gs1 = fig1.add_gridspec(nrows=1, ncols=4, wspace=0.35)

# Panel 1: c(x)
ax1 = fig1.add_subplot(gs1[0, 0])
im1 = ax1.imshow(c_mip, origin="lower", cmap="viridis")
ax1.set_title(r"$\max_z\, c(\mathbf{x})$")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_xticks([])
ax1.set_yticks([])
add_colorbar(fig1, ax1, im1)

# Panel 2: p0
ax2 = fig1.add_subplot(gs1[0, 1])
im2 = ax2.imshow(p0_mip, origin="lower", cmap="hot")
ax2.set_title(r"$\max_z\, p_0(\mathbf{x})$")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_xticks([])
ax2.set_yticks([])
add_colorbar(fig1, ax2, im2)

# Panel 3: Coefficients with dyadic boxes
ax3 = fig1.add_subplot(gs1[0, 2])
extent_k = [-N[0] // 2, N[0] // 2, -N[1] // 2, N[1] // 2]
pltgb.plot_coeffs_with_boxes(
    coeff_mip,
    dyadic_decomp,
    ax=ax3,
    plane="xy",
    extent=extent_k,
    title=r"$\max_{k_z} |c_{\ell,j,k}|$",
    show_colorbar=True,
)
ax3.set_xlabel(r"$k_x$")
ax3.set_ylabel(r"$k_y$")
ax3.set_xticks([])
ax3.set_yticks([])

# Panel 4: Reconstruction error
ax4 = fig1.add_subplot(gs1[0, 3])
im4 = ax4.imshow(err_mip, origin="lower", cmap="hot")
ax4.set_title(r"$\max_z |p_0^{\mathrm{recon}} - p_0|$")
ax4.set_xlabel("x")
ax4.set_ylabel("y")
ax4.set_xticks([])
ax4.set_yticks([])
add_colorbar(fig1, ax4, im4)

fig1.tight_layout()
out1 = PLOT_DIR / "thesis_3d_fig1_initial.png"
fig1.savefig(out1, dpi=300, bbox_inches="tight")
print(f"  Saved: {out1}")
plt.close(fig1)

# ============================================================================
# FIGURE 2: SENSOR DATA COMPARISON
# ============================================================================
print("\nGenerating Figure 2: Sensor Data...")

# Prepare data (MIP over z-axis of sensor plane)
_kw = mip(np.asarray(kw_data), axis=2)  # (Nt, Ny)
_msgb = mip(np.asarray(gb_data), axis=2)
_hyb = mip(np.asarray(hyb_data), axis=2)
_ts = np.asarray(ts)

# Differences
_d_msgb = _kw - _msgb
_d_hyb = _kw - _hyb

# Normalizations
s_norm = shared_norm([_kw, _msgb, _hyb])
d_norm = symmetric_norm([_d_msgb, _d_hyb])

# Extent for imshow
Ny = _kw.shape[1]
extent = [0, N[1] * dx[1], float(_ts.max()), float(_ts.min())]

# Profile position
y_idx = Ny // 2
y_pos = (N[1] * dx[1]) * (y_idx / Ny)

# Create figure
fig2 = plt.figure(figsize=(10, 7))
gs2 = fig2.add_gridspec(
    nrows=2, ncols=3, height_ratios=[1.0, 0.7], hspace=0.35, wspace=0.12
)

# Top row: Image panels
ax_kw = fig2.add_subplot(gs2[0, 0])
ax_dmsgb = fig2.add_subplot(gs2[0, 1])
ax_dhyb = fig2.add_subplot(gs2[0, 2])

im_kw = ax_kw.imshow(_kw, extent=extent, aspect="auto", norm=s_norm, cmap="seismic")
im_dmsgb = ax_dmsgb.imshow(
    _d_msgb, extent=extent, aspect="auto", norm=d_norm, cmap="RdBu_r"
)
im_dhyb = ax_dhyb.imshow(
    _d_hyb, extent=extent, aspect="auto", norm=d_norm, cmap="RdBu_r"
)

ax_kw.set_title(r"$\max_z\, p_t^{\mathrm{k\text{-}Wave}}$", fontsize=11)
ax_dmsgb.set_title(
    r"$\max_z\, (p_t^{\mathrm{k\text{-}Wave}} - p_t^{\mathrm{MSGB}})$", fontsize=11
)
ax_dhyb.set_title(
    r"$\max_z\, (p_t^{\mathrm{k\text{-}Wave}} - p_t^{\mathrm{Hybrid}})$", fontsize=11
)

for ax in (ax_kw, ax_dmsgb, ax_dhyb):
    ax.set_xlabel(r"$y_s$ (m)", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
ax_kw.set_ylabel("t (s)", fontsize=10)

# Colorbars
divL = make_axes_locatable(ax_kw)
caxL = divL.append_axes("left", size="5%", pad=0.15)
fig2.colorbar(im_kw, cax=caxL)
caxL.yaxis.set_ticks_position("left")
caxL.yaxis.set_label_position("left")

divR = make_axes_locatable(ax_dhyb)
caxR = divR.append_axes("right", size="5%", pad=0.15)
fig2.colorbar(im_dhyb, cax=caxR)

# Vertical profile line
for ax in (ax_kw, ax_dmsgb, ax_dhyb):
    ln = ax.axvline(y_pos, ls="--", lw=1.5, color="k", zorder=5)
    ln.set_path_effects([pe.Stroke(linewidth=3, foreground="white"), pe.Normal()])

# Bottom row: Time profile
ax_prof = fig2.add_subplot(gs2[1, :])
y_kw = _kw[:, y_idx]
y_msgb = _msgb[:, y_idx]
y_hyb = _hyb[:, y_idx]

ax_prof.plot(_ts, y_kw, label="k-Wave", lw=2, alpha=0.9)
ax_prof.plot(_ts, y_msgb, "--", label="MSGB", lw=2, alpha=0.9)
ax_prof.plot(_ts, y_hyb, ":", label="Hybrid", lw=2.5, alpha=0.9)
ax_prof.set_xlabel("Time (s)", fontsize=11)
ax_prof.set_ylabel(r"$\max_z\, p_t(y_s)$", fontsize=11)
ax_prof.set_title(f"Temporal profile at $y_s$ = {y_pos:.2e} m", fontsize=11)
ax_prof.legend(frameon=True, fancybox=True, shadow=True, loc="best")
ax_prof.grid(True, alpha=0.3, ls=":")

fig2.tight_layout()
out2 = PLOT_DIR / "thesis_3d_fig2_sensor.png"
fig2.savefig(out2, dpi=300, bbox_inches="tight")
print(f"  Saved: {out2}")
plt.close(fig2)

# ============================================================================
# FIGURE 3: ORTHOGONAL MIP VIEWS
# ============================================================================
print("\nGenerating Figure 3: Orthogonal MIP Views...")


def orth_mips(vol):
    """Return XY, YZ, ZX MIP projections."""
    xy = np.max(vol, axis=2).T  # (Ny, Nx)
    yz = np.max(vol, axis=0).T  # (Nz, Ny)
    zx = np.max(vol, axis=1).T  # (Nz, Nx)
    return xy, yz, zx


# Prepare data
p0_xy, p0_yz, p0_zx = orth_mips(_p0)
err_xy, err_yz, err_zx = orth_mips(_recon_err)

fig3 = plt.figure(figsize=(14, 6))
gs3 = fig3.add_gridspec(nrows=2, ncols=3, hspace=0.3, wspace=0.3)

# Row 1: p0 projections
ax_xy = fig3.add_subplot(gs3[0, 0])
im_xy = ax_xy.imshow(p0_xy, origin="lower", cmap="hot")
ax_xy.set_title(r"$p_0$ (XY)", fontsize=12)
ax_xy.set_xlabel("X")
ax_xy.set_ylabel("Y")
ax_xy.set_xticks([])
ax_xy.set_yticks([])
add_colorbar(fig3, ax_xy, im_xy, size="4%", pad=0.05)

ax_yz = fig3.add_subplot(gs3[0, 1])
im_yz = ax_yz.imshow(p0_yz, origin="lower", cmap="hot")
ax_yz.set_title(r"$p_0$ (YZ)", fontsize=12)
ax_yz.set_xlabel("Y")
ax_yz.set_ylabel("Z")
ax_yz.set_xticks([])
ax_yz.set_yticks([])
add_colorbar(fig3, ax_yz, im_yz, size="4%", pad=0.05)

ax_zx = fig3.add_subplot(gs3[0, 2])
im_zx = ax_zx.imshow(p0_zx, origin="lower", cmap="hot")
ax_zx.set_title(r"$p_0$ (ZX)", fontsize=12)
ax_zx.set_xlabel("X")
ax_zx.set_ylabel("Z")
ax_zx.set_xticks([])
ax_zx.set_yticks([])
add_colorbar(fig3, ax_zx, im_zx, size="4%", pad=0.05)

# Row 2: Error projections
ax_exy = fig3.add_subplot(gs3[1, 0])
im_exy = ax_exy.imshow(err_xy, origin="lower", cmap="hot")
ax_exy.set_title(r"$|p_0^{\mathrm{recon}} - p_0|$ (XY)", fontsize=12)
ax_exy.set_xlabel("X")
ax_exy.set_ylabel("Y")
ax_exy.set_xticks([])
ax_exy.set_yticks([])
add_colorbar(fig3, ax_exy, im_exy, size="4%", pad=0.05)

ax_eyz = fig3.add_subplot(gs3[1, 1])
im_eyz = ax_eyz.imshow(err_yz, origin="lower", cmap="hot")
ax_eyz.set_title(r"$|p_0^{\mathrm{recon}} - p_0|$ (YZ)", fontsize=12)
ax_eyz.set_xlabel("Y")
ax_eyz.set_ylabel("Z")
ax_eyz.set_xticks([])
ax_eyz.set_yticks([])
add_colorbar(fig3, ax_eyz, im_eyz, size="4%", pad=0.05)

ax_ezx = fig3.add_subplot(gs3[1, 2])
im_ezx = ax_ezx.imshow(err_zx, origin="lower", cmap="hot")
ax_ezx.set_title(r"$|p_0^{\mathrm{recon}} - p_0|$ (ZX)", fontsize=12)
ax_ezx.set_xlabel("X")
ax_ezx.set_ylabel("Z")
ax_ezx.set_xticks([])
ax_ezx.set_yticks([])
add_colorbar(fig3, ax_ezx, im_ezx, size="4%", pad=0.05)

fig3.tight_layout()
out3 = PLOT_DIR / "thesis_3d_fig3_orthogonal.png"
fig3.savefig(out3, dpi=300, bbox_inches="tight")
print(f"  Saved: {out3}")
plt.close(fig3)

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
print(f"All figures saved to: {PLOT_DIR}")
