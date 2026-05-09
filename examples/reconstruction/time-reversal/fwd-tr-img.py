#!/usr/bin/env python
# coding: utf-8



"""
2D k-Wave forward simulation followed by time-reversal reconstruction on an OA-Breast phantom slice with a planar line-sensor array. Plots the initial pressure p0, the sensor measurement, and the time-reversal estimate p0_TR side by side.
"""
import jax.numpy as jnp
import jax as jax
import matplotlib.pyplot as plt
from pathlib import Path
from beamax.solvers import KWaveSolver
import numpy as np

from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions

from beamax import geometry, utils

ROOT_DIR = utils.detect_root()
DATA_DIR = Path(ROOT_DIR / "data")
PLOT_DIR = Path(ROOT_DIR / "plots")
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
jax.config.update("jax_enable_x64", True)

d = 2
N = (1024,) * d
dx = (1e-4,) * d
periodic = (False,) * d
box_aspect_ratio = (1,) * d
num_levels = 2
num_boxes_level = (4, 8)
c0 = 1500.0
cfl = (jnp.sqrt(2) / 4).round(3)
domain = geometry.Domain(N=N, dx=dx, c=c0, cfl=cfl, periodic=periodic)

ts = domain.generate_time_domain()

sensor_mask = jnp.zeros(N)
sensors_all = jnp.ones(N)
sensor_mask = sensor_mask.at[..., 0].set(1)
# sensor_mask = sensor_mask.at[-1, ...].set(1)
# sensor_mask = sensor_mask.at[..., -1].set(1)
# sensor_mask = sensor_mask.at[0, ...].set(1)

simulation_options = SimulationOptions(
    data_cast="double",
    smooth_p0=False,
    save_to_disk=True,
)

execution_options = SimulationExecutionOptions(
    is_gpu_simulation=False, delete_data=False, verbose_level=0, show_sim_log=False
)

kwave_solver = KWaveSolver(simulation_options, execution_options)


phantom_path = Path(DATA_DIR / "NumericalBreastPhantoms-selected/hdf5/Neg_07_Left.h5")

p0, c_2d, meta = utils.load_oabreast_p0_c(
    phantom_path,
    dim="2d",
    axis_order="XYZ",
    slice_axis=1,  # MIP along Z
    vessels_mip_2d=True,
    c_exclude_vessels=True,
    c_fill_strategy="background",  # simplest: set vessel pixels to background speed
    background_speed=1500.0,
    target_shape=N,
)

# data = plt.imread(DATA_DIR / "athletics2.png")

# p0 = jnp.mean(data, axis=2)
# logo_mask = p0 < 0.8
# binary_logo = logo_mask.astype(bool)

# p0 = gaussian_filter(binary_logo, sigma=0.1)

# zoom_factor_height = 266 / smoothed_logo.shape[0]  # 512 / 133
# zoom_factor_width = 440 / smoothed_logo.shape[1]   # 512 / 378

# # The 'zoom' function can take a tuple of factors, one for each dimension
# p0 = zoom(smoothed_logo, (zoom_factor_height, zoom_factor_width))

# plt.imshow(smoothed_logo)

p0 = jnp.rot90(p0, k=1)

measurement = kwave_solver.forward(p0, domain, sensor_mask, ts)
p0_estimate_tr = kwave_solver.time_reversal(
    measurement.T, domain, jnp.ones(N), sensor_mask, ts
).T


SQUARE_DIM = N[0]  # Height and width for square images (p0, p0_estimate_tr)
MEASUREMENT_PLOTTED_HEIGHT = SQUARE_DIM
MEASUREMENT_PLOTTED_WIDTH_ORIG = 4 * SQUARE_DIM  # Original data is 1:4 (height:width)
MEASUREMENT_DESIRED_PLOTTED_WIDTH = (
    0.5 * MEASUREMENT_PLOTTED_WIDTH_ORIG
)  # Desired plot width is 2 * SQUARE_DIM

fig = plt.figure(figsize=(20, 8), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 2, 1])

# 2. Create the subplots using GridSpec
ax1 = fig.add_subplot(gs[0, 0])  # Leftmost (Initial Pressure)
ax2 = fig.add_subplot(gs[0, 1])  # Middle (Measurement)
ax3 = fig.add_subplot(gs[0, 2])  # Rightmost (Time Reversal)

# 3. Find global min/max for shared colorbar
all_data = np.concatenate(
    [p0.flatten(), measurement.flatten(), p0_estimate_tr.flatten()]
)
vmin = all_data.min()
vmax = all_data.max()

# Plot 1: Initial Pressure Distribution (Square)
im1 = ax1.imshow(p0, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
ax1.set_box_aspect(1)  # Force axes box to be square (height/width = 1)
x_coords = np.where(sensor_mask)[1]
y_coords = np.where(sensor_mask)[0]
ax1.scatter(
    x_coords,
    y_coords,
    color="red",
    marker="^",
    label="Sensor",
    s=50,
    linewidth=0.5,
    edgecolors="r",
)
ax1.set_title("Initial Pressure: $p_0$")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
# ax1.legend()

# Plot 2: Forward Simulation Result (Rectangular, scaled)
im2 = ax2.imshow(measurement.T, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
# Calculate the aspect ratio for the axes box: desired_height / desired_width
ax2.set_box_aspect(MEASUREMENT_PLOTTED_HEIGHT / MEASUREMENT_DESIRED_PLOTTED_WIDTH)
ax2.set_title("Measurement Data: $g(x_s,t)$")
ax2.set_xlabel("t")
ax2.set_ylabel("x")

# Plot 3: Time Reversal Result (Square)
im3 = ax3.imshow(p0_estimate_tr, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
ax3.set_box_aspect(1)  # Force axes box to be square (height/width = 1)
ax3.set_title("Time Reversal: $p_0^{TR}$")
ax3.set_xlabel("x")
ax3.set_ylabel("y")

# 4. Remove numbers/ticks from axes for all subplots
for ax in [ax1, ax2, ax3]:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

fig.canvas.draw()
pos3 = ax3.get_position()
cbar_width = 0.015  # Width of colorbar
cbar_pad = 0.02  # Padding between ax3 and colorbar
cbar_ax = fig.add_axes([pos3.x1 + cbar_pad, pos3.y0, cbar_width, pos3.height])
cbar = fig.colorbar(im1, cax=cbar_ax)
# cbar_ax.set_ylabel('Value', rotation=270, labelpad=15)
fig.savefig(PLOT_DIR / "london-skyline.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
