#!/usr/bin/env python
# coding: utf-8



"""
Ray-tracing diagnostic for the Gaussian beam Hamiltonian.
"""
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from pathlib import Path
from beamax import utils
from beamax.plotter import use_beamax_style

try:
    from skimage import measure
except ModuleNotFoundError:
    print(
        "Skipping example: scikit-image is not installed (`pip install scikit-image`)."
    )
    raise SystemExit(0)

ROOT_DIR = utils.detect_root()
PLOT_DIR = Path(ROOT_DIR / "plots")
DATA_DIR = Path(ROOT_DIR / "data")
PROF_DIR = Path(ROOT_DIR / "profiler")
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
PROF_DIR.mkdir(exist_ok=True)

use_beamax_style()


# =================================================================
# Placeholder functions for your custom modules to make the script runnable.
# In your actual use, you would keep your original imports and code here.
# =================================================================
def sphere_points(b, radius):
    """Generates mock points on a sphere."""
    phi = np.linspace(0, np.pi, int(np.sqrt(b)))
    theta = np.linspace(0, 2 * np.pi, int(np.sqrt(b)))
    phi, theta = np.meshgrid(phi, theta)
    phi, theta = phi.flatten(), theta.flatten()

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return jnp.array(np.vstack([x, y, z]).T) + 0.5


def velocity_field_3d(XYZ):
    """Generates a mock 3D velocity field."""
    X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]
    # A sample field with interesting isosurfaces
    val = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) + np.sin(2 * np.pi * Z)
    return val


# Dummy solver for demonstration
def solve_ODE_base(x0, p0, M0, a0, mode, ts, velocity, _):
    """Generates mock ray trajectories."""
    num_timesteps = len(ts)
    b, d = x0.shape
    xt = np.zeros((b, num_timesteps, d))
    for i in range(b):
        # Create a simple helical trajectory for each ray
        t = ts.reshape(-1, 1)
        xt[i, :, 0] = x0[i, 0] + 0.2 * np.sin(5 * np.pi * t).flatten()
        xt[i, :, 1] = x0[i, 1] + 0.2 * np.cos(5 * np.pi * t).flatten()
        xt[i, :, 2] = x0[i, 2] + (t.flatten() - 0.5) * 0.5
    return jnp.array(xt), None, None, None


# =================================================================
# Your original data generation code (with placeholders above)
# =================================================================
d = 3
b = 81  # Adjusted to be a perfect square for the mock sphere_points
N = (32,) * d  # Using a slightly smaller grid for faster isosurface calculation
dx = (1 / N[0],) * d
xmax = jnp.array(N) * jnp.array(dx)

x_linspace = [jnp.linspace(0, xmax[i], N[i]) for i in range(d)]
x, y, z = np.meshgrid(*x_linspace, indexing="ij")
XYZ = jnp.stack([x, y, z], axis=-1)

x0 = sphere_points(b, radius=0.2)
p0 = jnp.zeros_like(x0)  # Dummy value

a0 = jnp.ones((b, 1)) * 0.1
mode = jnp.ones((b, 1))
ts = jnp.linspace(0, 1, 100)

xt, _, _, _ = solve_ODE_base(x0, p0, None, a0, mode, ts, velocity_field_3d, None)
beam_points = np.array(xt)  # Shape: (b, num_timesteps, 3)

# Calculate velocity field for contours
velocity_data = velocity_field_3d(XYZ)

# =================================================================
# Matplotlib Visualization
# =================================================================
# --- 1. Set up the Figure and 3D Axes ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(bottom=0.2)  # Make room for widgets

# --- 2. Plot the Velocity Field Isosurface ---
# Use marching_cubes to find the surface for a given isovalue
iso_value = 0.5
try:
    verts, faces, _, _ = measure.marching_cubes(velocity_data, iso_value, spacing=dx)
    # The 'verts' are in grid coordinates, scale them to the physical domain
    verts = verts * np.array(dx)
    velocity_surface = ax.plot_trisurf(
        verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap="viridis", lw=0.2, alpha=0.3
    )
    velocity_surface.set_visible(True)  # Initially visible
except ValueError:
    print(
        f"No isosurface found at value {iso_value}. The velocity field might be flat."
    )
    velocity_surface = None

# --- 3. Plot the Bounding Box Outline ---
outline_pts = np.array(
    [
        [0, 0, 0],
        [xmax[0], 0, 0],
        [xmax[0], xmax[1], 0],
        [0, xmax[1], 0],
        [0, 0, 0],
        [0, 0, xmax[2]],
        [xmax[0], 0, xmax[2]],
        [xmax[0], xmax[1], xmax[2]],
        [0, xmax[1], xmax[2]],
        [0, 0, xmax[2]],
        [xmax[0], 0, xmax[2]],
        [xmax[0], 0, 0],
        [xmax[0], xmax[1], 0],
        [xmax[0], xmax[1], xmax[2]],
        [0, xmax[1], xmax[2]],
        [0, xmax[1], 0],
    ]
)
ax.plot(outline_pts[:, 0], outline_pts[:, 1], outline_pts[:, 2], color="white", lw=1.5)

# --- 4. Plot Initial Ray Trajectories ---
# Plot each ray as a line object and store them in a list
lines = []
for i in range(b):
    # Start with a single point
    (line,) = ax.plot(
        beam_points[i, 0:1, 0],
        beam_points[i, 0:1, 1],
        beam_points[i, 0:1, 2],
        "r-",
        lw=2,
    )
    lines.append(line)

# --- 5. Scene Configuration ---
ax.set_facecolor("grey")
ax.set_xlim(0, xmax[0])
ax.set_ylim(0, xmax[1])
ax.set_zlim(0, xmax[2])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Ray Trajectories with Matplotlib")
ax.grid(False)

# --- 6. Create Interactive Widgets and Update Functions ---

# Slider
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow")
time_slider = Slider(
    ax=ax_slider, label="Time", valmin=0, valmax=len(ts) - 1, valinit=0, valstep=1
)


def update(val):
    """Updates the plot when the slider is moved."""
    time_idx = int(time_slider.val)
    for i, line in enumerate(lines):
        # Update the data for each line up to the current time index
        points_to_plot = beam_points[i, : time_idx + 1]
        line.set_data(points_to_plot[:, 0], points_to_plot[:, 1])
        line.set_3d_properties(points_to_plot[:, 2])
    fig.canvas.draw_idle()


time_slider.on_changed(update)

# Checkbox
ax_check = plt.axes([0.05, 0.8, 0.15, 0.15], facecolor="lightgoldenrodyellow")
check = CheckButtons(ax=ax_check, labels=["Velocity"], actives=[True])


def toggle_visibility(label):
    """Toggles the visibility of the velocity field."""
    if velocity_surface:
        is_visible = velocity_surface.get_visible()
        velocity_surface.set_visible(not is_visible)
        fig.canvas.draw_idle()


check.on_clicked(toggle_visibility)

plt.savefig(PLOT_DIR / "3d_ray_trajectories.png", dpi=300, bbox_inches="tight")
# --- 7. Show the Plot ---
plt.show()
