#!/usr/bin/env python
# coding: utf-8


"""
2D forward solve comparing MSGB against k-Wave and the hybrid MSGB + low-frequency solver. Requires `[kwave]`.
"""
import jax as jax
import jax.numpy as jnp
import numpy as np

from time import time
from pathlib import Path

from beamax import geometry, plotter, utils
from beamax.decomposition import DyadicDecomposition
from beamax.transforms import MSWPT
from beamax.gb import gb_solvers
from beamax.solvers import KWaveSolver, MSGBSolver, HybridSolver, ShardingStrategy
from beamax.solvers.hybrid_solver_utils import get_indices_with_norm_less_than
from beamax.plotter import use_beamax_style

from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions

ROOT_DIR = utils.detect_root()
PLOT_DIR = Path(ROOT_DIR / "plots")
DATA_DIR = Path(ROOT_DIR / "data")
PROF_DIR = Path(ROOT_DIR / "profiler")
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
PROF_DIR.mkdir(exist_ok=True)

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import matplotlib.patches as patches

use_beamax_style()

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

num_devices = len(jax.devices())
is_cpu, is_gpu, is_tpu = utils.get_devices()

pltgb = plotter.PlotHelper()

# # Setup

d = 2
N = (128, 64)
dx = (1e-4,) * d
periodic = (True,) * d
box_aspect_ratio = (1,) * d
num_levels = 2
num_boxes_levels = tuple([2 ** (i + 2) for i in range(num_levels)])


def c(x):
    return 1500 + 0 * x[..., 0]


windowing = "rectangular_mirror"
none_windowing = "none"
input_type = "spatial"
output_type = "spatial"
redundancy = 2

cfl = (jnp.sqrt(2) / 4).round(3)
domain = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)

ts = domain.generate_time_domain()

dyadic_decomp = DyadicDecomposition(num_levels, N, num_boxes_levels, box_aspect_ratio)

# def c(x):
#     """
#     Defines the speed of sound with one fast and one slow Gaussian blob.
#     """
#     extent = tuple([N[i] * dx[i] for i in range(d)])
#     # 1. Define a constant background speed
#     background_speed = 1500.0

#     # 2. Define the FAST blob (positive perturbation)
#     center_fast = (0.4 * extent[0], 0.4 * extent[1])  # Moved closer to center
#     width_fast = 0.05 * extent[0]  # Width parameter (sigma)
#     amplitude_fast = 50.0  # Speed increases by 50 m/s at the center

#     # Fixed Gaussian formula with proper width^2
#     fast_blob = jnp.exp(
#         -((x[..., 0] - center_fast[0]) ** 2 + (x[..., 1] - center_fast[1]) ** 2)
#         / (2 * width_fast**2)
#     )

#     # 3. Define the SLOW blob (negative perturbation)
#     center_slow = (0.6 * extent[0], 0.6 * extent[1])  # Moved closer to center
#     width_slow = 0.08 * extent[0]  # Made slightly wider
#     amplitude_slow = 50.0  # Speed decreases by 50 m/s at the center

#     # Fixed Gaussian formula with proper width^2
#     slow_blob = jnp.exp(
#         -((x[..., 0] - center_slow[0]) ** 2 + (x[..., 1] - center_slow[1]) ** 2)
#         / (2 * width_slow**2)
#     )

#     # 4. Combine them: background + fast blob - slow blob
#     return (
#         background_speed + (amplitude_fast * fast_blob) - (amplitude_slow * slow_blob)
#     )

t1 = time()
dyadic_decomp = DyadicDecomposition(num_levels, N, num_boxes_levels, box_aspect_ratio)
wpt = MSWPT(dyadic_decomp, redundancy, windowing)
wptNone = MSWPT(dyadic_decomp, redundancy, none_windowing)
t2 = time()
print("Time to create params", t2 - t1)

#################
### planar ######
#################

binary_mask = jnp.zeros(N)
binary_mask = binary_mask.at[0, ...].set(1)
sensors = geometry.Sensor(binary_mask=binary_mask, domain=domain)

#################
### circ ########
#################

# # circ
# radius = 30
# tol = 0.5
# idx = jnp.indices(N); c = jnp.array(N)//2
# d = jnp.sqrt((idx[0]-c[0])**2 + (idx[1]-c[1])**2)
# binary_mask = (jnp.abs(d-radius) <= tol).astype(jnp.int32)
# sensors_circ = geometry.Sensor(binary_mask=binary_mask, domain=domain)

# #######################################
# ### wave packets ######################
# #######################################
# from beamax import transforms

# KXY = dyadic_decomp.fourier_meshgrid

# # pltgb.plot_centers(dyadic_decomp.centres_ndim)

# boxhf = 44
# boxlf = 10
# # probably need to multiply by the ratio between (64,64) and the desired res.
# khf = jnp.array([10, 12])
# klf = jnp.array([10, 3])
# kerft_hf = transforms.compute_frames(dyadic_decomp, boxhf, khf, KXY, redundancy, "none")
# kerft_lf = transforms.compute_frames(dyadic_decomp, boxlf, klf, KXY, redundancy, "none")
# p0 = utils.unitary_ifft(kerft_hf) + utils.unitary_ifft(kerft_lf)
# p0 = p0 / jnp.max(jnp.abs(p0))
# p0 = p0.T
# exp = 1

#######################################
### POINT SOURCE ######################
#######################################

p0 = jnp.zeros(N)
p0 = p0.at[N[0] // 4, N[1] // 2].set(1)
exp = 2

#######################################
##  PALM VESSELS ######################
#######################################

# from scipy import io as sio

# # data = sio.loadmat(DATA_DIR / "vessels_BB.mat")["p0"]
# # data = np.load(DATA_DIR / "chinese.npy")
# data = sio.loadmat(DATA_DIR / "palm5.mat")["p0"]
# data = data[: N[0], : N[1]]
# p0 = data / jnp.max(jnp.abs(data))
# exp = 3

######################################
### CIRCLES PHANTOM ##################
######################################
# from kwave.utils.mapgen import make_disc
# from kwave.data import Vector

# # # create initial pressure distribution using makeDisc
# N_vec = Vector(N)  # [grid points]
# disc_magnitude = 1  # [Pa]
# disc_pos = Vector([30, 60])  # [grid points]
# disc_radius = 5  # [grid points]
# disc_2 = disc_magnitude * make_disc(N_vec, disc_pos, disc_radius)

# disc_pos = Vector([80, 50])  # [grid points]
# disc_radius = 20  # [grid points]
# disc_magnitude = 2  # [Pa]
# disc_1 = disc_magnitude * make_disc(N_vec, disc_pos, disc_radius)

# p0 = disc_1 + disc_2
# exp = 4

########################################
###  OA DATASET ########################
########################################

# phantom_path = Path(DATA_DIR / "NumericalBreastPhantoms-selected/hdf5/Neg_07_Left.h5")

# p0, c_2d, meta2d = utils.load_oabreast_p0_c(
#     phantom_path,
#     dim="2d",
#     axis_order="XYZ",
#     source_spacing_mm=(dx[0], dx[1], dx[0]),
#     slice_axis=0,
#     slice_policy="middle",  # or "max_variance"
#     target_shape=N,
#     normalize_p0=True,
#     label_to_sos={4: 1550.0},
# )
# exp = 5

# c = utils.make_c_function_from_grid(c_2d, spacing=dx, origin=(0.0, 0.0))

# XY, _ = domain.generate_meshgrid()
# XY = jnp.stack(XY, axis=-1)
# cval = c(XY)

# plt.imshow(cval, extent=[0, N[0] * dx[0], 0, N[1] * dx[1]], origin="lower")
# plt.show()

# domain = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)

########################################
###  SETUP           ###################
########################################

# p0 = p0 / jnp.max(jnp.abs(p0))
p0 = p0.real
coeffs = wpt.forward(p0, input_type="spatial")

# forward coefficients
coeffs = wpt.forward(p0, input_type="spatial")  # (total_coeffs,)
coeffs_array = wpt.convert_to_array(coeffs)


print(f"Total coefficients: {len(coeffs)}")

# Find minimum K
tau = 0.01
K, indices, coeff_vals = utils.find_min_K_for_target_error(
    coeffs=coeffs, p0=p0, inv_wpt=wptNone, tau=tau, Kmin=512, Kmax=None, verbose=True
)

print(f"\n[result] K={K} coefficients needed for {tau * 100:.1f}% error")
print(f"Selected coefficients: {len(indices)}")

# (C) Reconstruct
data_recon = utils.reconstruct_from_selection(
    coeffs, indices, coeff_vals, wptNone, output_type="spatial"
)
threshold = int(indices.size)

coeffs_array = wpt.convert_to_array(coeffs)

# total_coeffs = jnp.prod(redundancy * jnp.array(N))
strategy = "top_n"

# threshold = int(total_coeffs * 0.1)
# indices, coeff_vals = forward_solver_utils.threshold_coefficients(
#     coeffs, threshold, strategy
# )
# print("Number of coefficients:", len(indices))

thresholded_coeffs = jnp.zeros_like(coeffs)
thresholded_coeffs = thresholded_coeffs.at[indices].set(coeff_vals)

data_recon = wptNone.inverse(thresholded_coeffs, output_type="spatial")
##########################################
#  COEFF RECONSTRUCTION WITH BOX OVERLAY #
##########################################
# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ax1, ax2, ax3 = axes

im1 = ax1.imshow(p0, origin="lower")
ax1.scatter(jnp.where(sensors.binary_mask)[1], jnp.where(sensors.binary_mask)[0], c="r")
ax1.axis("off")
ax1.set_title("Initial Pressure")

# Add colorbar for first subplot
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im1, cax=cax1)

# --- Second subplot: MSWPT coefficients with box overlay ---
pltgb.plot_coeffs_with_boxes(
    coeffs_array,
    dyadic_decomp,
    ax=ax2,
    plane="xy",
    extent=[-N[0] // 2, N[0] // 2, -N[1] // 2, N[1] // 2],
    title="MSWPT Coefficients with Dyadic Boxes",
    show_colorbar=True,
)

# --- Third subplot: Reconstructed - Original ---
im3 = ax3.imshow(data_recon.real - p0, origin="lower")
ax3.axis("off")
ax3.set_title("Reconstructed Data - Original Data")

# Add colorbar for third subplot
divider3 = make_axes_locatable(ax3)
cax3 = divider3.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im3, cax=cax3)

plt.tight_layout()
plt.savefig(PLOT_DIR / f"2d_coeffs_with_boxes_{exp}.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# # Forward Solve

batch_size = 128
sum_method = "scan_real"
solver = gb_solvers.solve_ODE_base

num_devices = jax.device_count()
print(f"\nNumber of devices available: {num_devices}")

mesh = jax.make_mesh((num_devices,), ("x",))

# Create sharding strategy
sharding_strategy = ShardingStrategy(mesh, beam_axis="x")

# Create solver with sharding
msgb_solver = MSGBSolver(
    thr=threshold,
    thr_strat=strategy,
    batch_size=batch_size,
    input_type=input_type,
    ode_solver=solver,
    tr_ode_solver=gb_solvers.solve_hom_TR,
    sum_method=sum_method,
    sharding=sharding_strategy,  # ← Only change!
)

print(f"\nRunning on {num_devices} devices...")
t1 = time()
gb_multi = msgb_solver.forward(p0, domain, sensors, ts, wpt)[0].block_until_ready()
t2 = time()
print(f"Multi-device: {t2 - t1:.3f}s")

simulation_options = SimulationOptions(
    data_cast="double",
    smooth_p0=False,
    save_to_disk=True,
)

execution_options = SimulationExecutionOptions(
    is_gpu_simulation=is_gpu, delete_data=False, verbose_level=0, show_sim_log=False
)

kwave_solver = KWaveSolver(simulation_options, execution_options)

box_corners = jnp.array([0, 15])
# box_corners = None
cutoff_freq = 15
cutoff_freq = None

if cutoff_freq is not None:
    box_idx = get_indices_with_norm_less_than(dyadic_decomp.centres_ndim, cutoff_freq)
    plt.plot(
        dyadic_decomp.centres_ndim[:, 0],
        dyadic_decomp.centres_ndim[:, 1],
        "o",
        color="red",
    )
    plt.plot(
        dyadic_decomp.centres_ndim[box_idx, 0],
        dyadic_decomp.centres_ndim[box_idx, 1],
        "o",
        color="green",
    )
    plt.title("Box indices")
    plt.savefig(PLOT_DIR / "box_indices.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

hybrid_solver = HybridSolver(
    lf_solver=kwave_solver,
    hf_solver=msgb_solver,
    downsample=False,
    box_corners=box_corners,
    cutoff_freq=cutoff_freq,
    input_type="spatial",
    interp_method="fourier",
    dt_oversample=0,
    beta=12.0,
)

gb_init = gb_multi.real

t1 = time()
sensor_data_kw = kwave_solver.forward(p0, domain, sensors.binary_mask, ts)
t2 = time()
print(f"k-Wave runtime: {t2 - t1:.3f}s")

t1 = time()
hybrid_data = hybrid_solver.forward(p0, domain, sensors, ts, wpt)
t2 = time()
print(f"Hybrid runtime: {t2 - t1:.3f}s")

extent = [0, N[1] * dx[1], jnp.max(ts), jnp.min(ts)]

# Keep your error prints as-is below
print(f"Exp: {exp}, periodic: {periodic}")
print("Max |hybrid - k-Wave| = ", jnp.max(jnp.abs(hybrid_data - sensor_data_kw)))
print("Max |MSGB - k-Wave| = ", jnp.max(jnp.abs(gb_init - sensor_data_kw)))
print("L2 error hybrid vs k-Wave = ", jnp.linalg.norm(hybrid_data - sensor_data_kw))
print("L2 error MSGB vs k-Wave = ", jnp.linalg.norm(gb_init - sensor_data_kw))
print(
    "Relative L2 error hybrid vs k-Wave = ",
    jnp.linalg.norm(hybrid_data - sensor_data_kw)
    / jnp.linalg.norm(sensor_data_kw)
    * 100,
    "%",
)
print(
    "Relative L2 error MSGB vs k-Wave = ",
    jnp.linalg.norm(gb_init - sensor_data_kw) / jnp.linalg.norm(sensor_data_kw) * 100,
    "%",
)

# # Figures

_kw = np.asarray(sensor_data_kw)
_msgb = np.asarray(gb_init)
_hyb = np.asarray(hybrid_data)
_ts = np.asarray(ts)

_p0 = np.asarray(p0.real)
_coeffs_mag = np.asarray(np.abs(np.asarray(coeffs_array)))
_recon_diff = np.asarray((data_recon.real - p0))

_s_min = float(min(_kw.min(), _msgb.min(), _hyb.min()))
_s_max = float(max(_kw.max(), _msgb.max(), _hyb.max()))
_s_norm = mcolors.Normalize(vmin=_s_min, vmax=_s_max)

# --- symmetric norm for top-right (recon diff) ---
_rd_max = float(np.max(np.abs(_recon_diff)))
_rd_norm = mcolors.Normalize(vmin=-_rd_max, vmax=_rd_max)

# --- middle-row diffs and symmetric norm shared by both diff panels ---
_d_kw_msgb = _kw - _msgb
_d_kw_hyb = _kw - _hyb
_d_absmax = float(max(np.max(np.abs(_d_kw_msgb)), np.max(np.abs(_d_kw_hyb))))
_d_norm = mcolors.Normalize(vmin=-_d_absmax, vmax=_d_absmax)

_extent_coeffs = [-N[0] // 2, N[0] // 2, -N[1] // 2, N[1] // 2]

# --- FIGURE 1: Initial Conditions, Coefficients, and Reconstruction ---
_profile_pos = (N[1] * dx[1]) * 100 / N[0]

fig1 = plt.figure(figsize=(13, 4))
gs1 = fig1.add_gridspec(nrows=1, ncols=4, wspace=0.5)

# Panel 4: Speed of Sound c(x)
ax_cmap = fig1.add_subplot(gs1[0, 0])
im_c = ax_cmap.imshow(c(domain.grid), origin="lower")
ax_cmap.set_title("$c(x)$")
ax_cmap.set_xticks([])
ax_cmap.set_yticks([])
_div4 = make_axes_locatable(ax_cmap)
_cax4 = _div4.append_axes("right", size="5%", pad=0.1)
fig1.colorbar(im_c, cax=_cax4)

# Panel 2: Initial Pressure p0
ax_p0 = fig1.add_subplot(gs1[0, 1])
im_p0 = ax_p0.imshow(_p0, origin="lower")
ax_p0.scatter(
    jnp.where(sensors.binary_mask)[1],
    jnp.where(sensors.binary_mask)[0],
    marker="^",
    color="r",
)
ax_p0.set_title("$p_0$")
ax_p0.set_xticks([])
ax_p0.set_yticks([])
_div1 = make_axes_locatable(ax_p0)
_cax1 = _div1.append_axes("right", size="5%", pad=0.1)
fig1.colorbar(im_p0, cax=_cax1)

# Panel 3: MSWPT Coefficients
ax_coeff = fig1.add_subplot(gs1[0, 2])
im_coeff = ax_coeff.imshow(_coeffs_mag, origin="lower", extent=_extent_coeffs)

# ADD BOX OVERLAY TO COEFFICIENTS PLOT
# First, add the dyadic decomposition boxes
cumsum_boxes = jnp.r_[0, jnp.cumsum(dyadic_decomp.num_boxes_ndim)]
box_lengths = dyadic_decomp.box_lengths
colors = ["gray", "darkgray", "silver", "lightgray"]

for level in range(dyadic_decomp.num_levels):
    start_idx = cumsum_boxes[level]
    end_idx = cumsum_boxes[level + 1]
    centers = dyadic_decomp.centres_ndim[start_idx:end_idx]

    box_length = box_lengths[level]
    box_width_x = box_length * dyadic_decomp.box_aspect_ratio[0]
    box_width_y = box_length * dyadic_decomp.box_aspect_ratio[1]

    color = colors[level % len(colors)]

    for center in centers:
        x_min = center[0] - box_width_x // 2
        x_max = center[0] + box_width_x // 2
        y_min = center[1] - box_width_y // 2
        y_max = center[1] + box_width_y // 2

        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=1.0,
            edgecolor=color,
            facecolor="none",
            linestyle=":",
            alpha=0.6,
        )
        ax_coeff.add_patch(rect)

# ADD RED DASHED CUTOFF BOUNDARY (if specified)
if box_corners is not None:
    # Calculate exact bounds for boxes in the cutoff range
    bounds_x = []
    bounds_y = []

    for idx in range(box_corners[0], box_corners[1] + 1):
        # Find which level this box belongs to
        level = 0
        for lv in range(dyadic_decomp.num_levels):
            if idx < cumsum_boxes[lv + 1]:
                level = lv
                break

        center = dyadic_decomp.centres_ndim[idx]
        box_length = box_lengths[level]
        box_width_x = box_length * dyadic_decomp.box_aspect_ratio[0]
        box_width_y = box_length * dyadic_decomp.box_aspect_ratio[1]

        bounds_x.extend([center[0] - box_width_x // 2, center[0] + box_width_x // 2])
        bounds_y.extend([center[1] - box_width_y // 2, center[1] + box_width_y // 2])

    # Draw red dashed rectangle around the entire cutoff region
    x_min, x_max = min(bounds_x), max(bounds_x)
    y_min, y_max = min(bounds_y), max(bounds_y)

    cutoff_rect = patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=2.5,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
        zorder=100,  # Make sure it's on top
    )
    ax_coeff.add_patch(cutoff_rect)

ax_coeff.set_title(r"$|c_{\ell,j,k}|$")
ax_coeff.set_xticks([])
ax_coeff.set_yticks([])
_div2 = make_axes_locatable(ax_coeff)
_cax2 = _div2.append_axes("right", size="5%", pad=0.1)
fig1.colorbar(im_coeff, cax=_cax2)

# Panel 4: Reconstruction Difference
ax_rdif = fig1.add_subplot(gs1[0, 3])
im_rdif = ax_rdif.imshow(_recon_diff, origin="lower", norm=_rd_norm, cmap="RdBu_r")
ax_rdif.set_title("" + r"$p_0^{\mathrm{recon}} - p_0$")
ax_rdif.set_xticks([])
ax_rdif.set_yticks([])
_div3 = make_axes_locatable(ax_rdif)
_cax3 = _div3.append_axes("right", size="5%", pad=0.1)
fig1.colorbar(im_rdif, cax=_cax3)

# --- Finalize and Save Figure 1 ---
fig1.tight_layout()
out1_png = PLOT_DIR / f"report_fig1_initial_{exp}.png"
fig1.savefig(out1_png, dpi=300, bbox_inches="tight")
plt.close(fig1)

# --- FIGURE 2: Sensor Data and Time-Domain Profile ---
fig2 = plt.figure(figsize=(9, 7))
gs2 = fig2.add_gridspec(
    nrows=2, ncols=3, height_ratios=[1.0, 0.75], hspace=0.4, wspace=0.15
)

# Top Row: Sensor Data Panels
ax_kw = fig2.add_subplot(gs2[0, 0])
ax_dmsgb = fig2.add_subplot(gs2[0, 1])
ax_dhyb = fig2.add_subplot(gs2[0, 2])

im_kw = ax_kw.imshow(_kw, extent=extent, aspect="auto", norm=_s_norm)
im_dmsgb = ax_dmsgb.imshow(
    _d_kw_msgb, extent=extent, aspect="auto", norm=_d_norm, cmap="RdBu_r"
)
im_dhyb = ax_dhyb.imshow(
    _d_kw_hyb, extent=extent, aspect="auto", norm=_d_norm, cmap="RdBu_r"
)

ax_kw.set_title("$p_t^{\\text{k-Wave}}$")
ax_kw.set_xlabel("$x_{s}$")
ax_kw.set_ylabel("t")
ax_kw.set_xticks([])
ax_kw.set_yticks([])

ax_dmsgb.set_title("$p_t^{\\text{k-Wave}} - p_t^{\\text{MSGB}}$")
ax_dhyb.set_title("$p_t^{\\text{k-Wave}} - p_t^{\\text{Hybrid}}$")
for ax in (ax_dmsgb, ax_dhyb):
    ax.set_xticks([])
    ax.set_yticks([])

# Colorbars for sensor data
_divL = make_axes_locatable(ax_kw)
_caxL = _divL.append_axes("left", size="7%", pad=0.22)
fig2.colorbar(im_kw, cax=_caxL)
_caxL.yaxis.set_ticks_position("left")
_caxL.yaxis.set_label_position("left")

_divR = make_axes_locatable(ax_dhyb)
_caxR = _divR.append_axes("right", size="7%", pad=0.2)
fig2.colorbar(im_dhyb, cax=_caxR)

# Overlay the profile line
Ny, Nx = _kw.shape
_xs = np.linspace(extent[0], extent[1], Nx)
_idx = int(np.clip(np.rint((_profile_pos - _xs[0]) / (_xs[1] - _xs[0])), 0, Nx - 1))
for ax in (ax_kw, ax_dmsgb, ax_dhyb):
    ln = ax.axvline(_profile_pos, ls="--", lw=1.2, color="k", zorder=5)
    ln.set_path_effects([pe.Stroke(linewidth=2.6, foreground="k"), pe.Normal()])

# Bottom Row: Profile Plot
ax_prof = fig2.add_subplot(gs2[1, :])
y_kw, y_msgb, y_hyb = _kw[:, _idx], _msgb[:, _idx], _hyb[:, _idx]
ax_prof.plot(_ts, y_kw, label="k-Wave", lw=2)
ax_prof.plot(_ts, y_msgb, "--", label="MSGB", lw=2)
ax_prof.plot(_ts, y_hyb, "--", label="Hybrid", lw=2)
ax_prof.set_xlabel("t")
ax_prof.set_ylabel("$p_t(x_s)$")
ax_prof.set_title(f"Profile at x = {_profile_pos:.4g} m")
ax_prof.legend(frameon=False)
ax_prof.grid(True, alpha=0.4)

# --- Finalize and Save Figure 2 ---
out2_png = PLOT_DIR / f"report_fig2_sensor_{exp}.png"
fig2.savefig(out2_png, dpi=300, bbox_inches="tight")
plt.close(fig2)
