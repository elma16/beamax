#!/usr/bin/env python

"""
Create a 3-column figure comparing k-Wave time-reversal and adjoint
reconstructions for increasing boundary sensor coverage (1–4 faces).

Layout (9 images total):
  - Column 0 (spans all rows): initial pressure p0.
  - Column 1: time reversal for 1, 2, 3, and 4 faces of sensors.
  - Column 2: adjoint for the same sensor configurations.

Sensor positions are shown as red triangles on every panel.
"""

from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions

from beamax import geometry, utils
from beamax.solvers import KWaveSolver


jax.config.update("jax_enable_x64", True)

ROOT_DIR = utils.detect_root()
DATA_DIR = Path(ROOT_DIR / "data")
PLOT_DIR = Path(ROOT_DIR / "plots")
PLOT_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------#
# Domain and phantom
# -----------------------------------------------------------------------------#

N = (128, 128)
dx = (1e-4,) * 2
periodic = (False, False)
cfl = float((jnp.sqrt(2) / 4).round(3))

phantom_path = DATA_DIR / "NumericalBreastPhantoms-selected/hdf5/Neg_07_Left.h5"
p0, c_map, meta = utils.load_oabreast_p0_c(
    phantom_path,
    dim="2d",
    axis_order="XYZ",
    slice_axis=1,  # MIP along Z
    vessels_mip_2d=True,
    c_exclude_vessels=True,
    c_fill_strategy="background",
    background_speed=1500.0,
    target_shape=N,
)

# Rotate to match the orientation used elsewhere.
p0 = jnp.rot90(p0, k=2)
c_map = jnp.rot90(c_map, k=2)

domain = geometry.Domain(N=N, dx=dx, c=c_map, cfl=cfl, periodic=periodic)
ts = domain.generate_time_domain()

sim_opts = SimulationOptions(data_cast="double", smooth_p0=False, save_to_disk=True)
exec_opts = SimulationExecutionOptions(
    is_gpu_simulation=False, delete_data=False, verbose_level=0, show_sim_log=False
)
solver = KWaveSolver(sim_opts, exec_opts)

# -----------------------------------------------------------------------------#
# Sensor configurations (faces = domain edges)
# -----------------------------------------------------------------------------#


def make_boundary_mask(shape, faces):
    """Build a sensor mask covering the specified faces."""
    h, w = shape
    mask = jnp.zeros(shape)
    if "top" in faces:
        mask = mask.at[0, :].set(1)
    if "bottom" in faces:
        mask = mask.at[h - 1, :].set(1)
    if "left" in faces:
        mask = mask.at[:, 0].set(1)
    if "right" in faces:
        mask = mask.at[:, w - 1].set(1)
    return mask


sensor_configs = [
    ("1 face (top)", ("top",)),
    ("2 faces (top+right)", ("top", "right")),
    ("3 faces (top+right+bottom)", ("top", "right", "bottom")),
    ("4 faces (all)", ("top", "right", "bottom", "left")),
]

all_sensors = jnp.ones(N)

# -----------------------------------------------------------------------------#
# Forward and inverse solves
# -----------------------------------------------------------------------------#

recon_data = []
for label, faces in sensor_configs:
    mask = make_boundary_mask(N, faces)
    measurement = solver.forward(p0, domain, mask, ts)
    tr = solver.time_reversal(measurement.T, domain, all_sensors, mask, ts).T
    adj = solver.adjoint(measurement.T, domain, all_sensors, mask, ts).T
    recon_data.append({"label": label, "mask": mask, "tr": tr, "adj": adj})

# -----------------------------------------------------------------------------#
# Plotting
# -----------------------------------------------------------------------------#

# Collect values for shared color scaling.
all_arrays = [p0] + [d["tr"] for d in recon_data] + [d["adj"] for d in recon_data]
vmin = float(np.min([np.min(np.asarray(a)) for a in all_arrays]))
vmax = float(np.max([np.max(np.asarray(a)) for a in all_arrays]))

fig = plt.figure(figsize=(12, 14), constrained_layout=True)
gs = fig.add_gridspec(4, 3, width_ratios=[1.05, 1, 1])

axes_tr = []
axes_adj = []

# Initial pressure axis spans all rows in column 0.
ax_p0 = fig.add_subplot(gs[:, 0])
im_p0 = ax_p0.imshow(p0, cmap="magma", vmin=vmin, vmax=vmax, origin="upper")
mask_all = recon_data[-1]["mask"]
y_all, x_all = np.where(np.asarray(mask_all))
ax_p0.scatter(x_all, y_all, marker="^", c="red", s=12, linewidths=0.4)
ax_p0.set_title(r"Initial pressure $p_0$")
ax_p0.set_xticks([])
ax_p0.set_yticks([])

# Rows for sensor configurations.
for row_idx, (label, faces) in enumerate(sensor_configs):
    data = recon_data[row_idx]

    ax_tr = fig.add_subplot(gs[row_idx, 1])
    im_tr = ax_tr.imshow(data["tr"], cmap="magma", vmin=vmin, vmax=vmax, origin="upper")
    y, x = np.where(np.asarray(data["mask"]))
    ax_tr.scatter(x, y, marker="^", c="red", s=12, linewidths=0.4)
    ax_tr.set_title(f"TR — {label}")
    ax_tr.set_xticks([])
    ax_tr.set_yticks([])
    axes_tr.append(im_tr)

    ax_adj = fig.add_subplot(gs[row_idx, 2])
    im_adj = ax_adj.imshow(
        data["adj"], cmap="magma", vmin=vmin, vmax=vmax, origin="upper"
    )
    ax_adj.scatter(x, y, marker="^", c="red", s=12, linewidths=0.4)
    ax_adj.set_title(f"Adjoint — {label}")
    ax_adj.set_xticks([])
    ax_adj.set_yticks([])
    axes_adj.append(im_adj)

    # Row label on the left edge of the TR column.
    ax_tr.text(
        -0.08,
        0.5,
        label,
        ha="right",
        va="center",
        rotation=90,
        transform=ax_tr.transAxes,
        fontsize=9,
    )

# Shared colorbar across all panels.
cbar = fig.colorbar(
    im_p0,
    ax=[ax_p0] + [im.axes for im in axes_tr + axes_adj],
    fraction=0.03,
    pad=0.02,
)
cbar.set_label("Amplitude")

output_path = PLOT_DIR / "breast_tr_adj_faces.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)
