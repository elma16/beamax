#!/usr/bin/env python
# coding: utf-8



"""
Animation of ray trajectories propagating through a 2D/3D medium.
"""
import jax.numpy as jnp
import jax
from functools import partial
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

from _velocitymaps import gaussian_dips_2d
from _curves import circle_curve

import numpy as np

try:
    import pyvista as pv
except ModuleNotFoundError:
    print("Skipping example: pyvista is not installed (`pip install pyvista`).")
    raise SystemExit(0)

from beamax import plotter, utils
from beamax.geometry import Domain
from beamax.gb import core, gb_utils, gb_solvers
from beamax.plotter import use_beamax_style

ROOT_DIR = utils.detect_root()
PLOT_DIR = Path(ROOT_DIR / "plots")
DATA_DIR = Path(ROOT_DIR / "data")
PROF_DIR = Path(ROOT_DIR / "profiler")
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
PROF_DIR.mkdir(exist_ok=True)

use_beamax_style()

jax.config.update("jax_enable_x64", False)
pltgb = plotter.PlotHelper()

b = 120
d = 2
N = (128,) * d
dx = (1 / N[0],) * d
periodic = (False,) * d
velocity = partial(gaussian_dips_2d)

cfl = 0.3
lam = 0
domain = Domain(N=N, dx=dx, c=velocity, cfl=cfl, periodic=periodic)

xmax = jnp.array(dx) * jnp.array(N)
XY = domain.grid
domain_size = domain.grid_size

ts = jnp.linspace(0, 0.4, 40)
mode = jnp.ones((b, 1))
curve_points = jnp.linspace(0, 1, b)
x0 = circle_curve(curve_points, radius=0.001, center=(0.5, 0.5))

if b == 1:
    tangents = jnp.array([[1.0, 0.0]])
else:
    tangents = jnp.gradient(x0, axis=0)

p0 = jnp.stack([-tangents[:, 1], tangents[:, 0]], axis=-1)
p0 = p0 / jnp.linalg.norm(p0, axis=1, keepdims=True)

a0 = jnp.ones((b, 1)) * 0.1
alpha0 = jnp.ones((b, d)) * 1j
ω0 = jnp.ones((b,)) * 100

M0 = gb_utils.prepare_M0(alpha0, None)
is_M0_diagonal = gb_utils.is_diagonal(M0)
solver = gb_solvers.solve_ODE_base

print("Is M0 diagonal?", is_M0_diagonal)

################################

# Setup speed of sound map
c_field = velocity(XY)
wavelength = 2 * jnp.pi * c_field / ω0[0]
lam = 0

grad_c = jax.grad(lambda x: velocity(x.reshape(-1, 2)).sum())(
    XY.reshape(-1, 2)
).reshape(XY.shape)
grad_c_magnitude = jnp.sqrt(jnp.sum(grad_c**2, axis=-1))

# Compute solutions
(xt, pt, mt, at) = solver(x0, p0, M0, a0, mode, ts, velocity, lam, None)
phase = core.compute_phase(xt, pt, mt, XY, domain_size, jnp.array(periodic))
u0 = core.compute_gaussian_beam(
    x0,
    p0,
    M0,
    a0,
    ω0,
    mode,
    velocity,
    lam,
    ts,
    XY,
    domain_size,
    jnp.array(periodic),
    solver,
    None,
)

phase = jnp.sum(phase, axis=0)
u0 = jnp.sum(u0, axis=-1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
c_map = velocity(XY)
im1 = ax1.imshow(
    c_map,
    extent=[0, N[0] * dx[0], 0, N[1] * dx[1]],
    origin="lower",
    cmap="RdBu",
    alpha=0.6,
)
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax1, label="Speed of sound")

line_collections = [None] * b
for i in range(b):
    line_collections[i] = LineCollection([], color="lightgreen")
    ax1.add_collection(line_collections[i])
ax1.set_title("Ray Paths")
ax1.set_xlim(0, xmax[0])
ax1.set_ylim(0, xmax[1])

im2 = ax2.imshow(
    jnp.zeros_like(u0[0].real),
    extent=[0, N[0] * dx[0], 0, N[1] * dx[1]],
    origin="lower",
    cmap="RdBu",
)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im2, cax=cax2, label="Real Part of Wavefield")
ax2.set_title("Real Part of Wavefield")


def init():
    for i in range(b):
        line_collections[i].set_segments([])
    im2.set_array(jnp.zeros_like(u0[0].real))
    return line_collections + [im2]


def update_frame(frame):
    for i in range(b):
        pts = np.array([xt[i, :frame, 0], xt[i, :frame, 1]]).T.reshape(-1, 1, 2)
        if frame > 1:
            segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
            colors = np.array(at[i, : frame - 1].flatten())
            line_collections[i].set_segments(segments)
            line_collections[i].set_array(np.abs(colors))
            line_collections[i].set_clim(np.min(np.abs(at)), np.max(np.abs(at)))
    im2.set_array(jnp.real(u0[frame]).T)
    ax1.set_title(f"Ray Paths (t = {ts[frame]:.2f})")
    ax2.set_title(f"Wavefield (t = {ts[frame]:.2f})")
    return line_collections + [im2]


init()
# if you want to save the plots!
# for i in range(len(ts)):
#     update_frame(i)
#     plt.savefig(f"frame_{i:03d}.png")

ani = FuncAnimation(
    fig, update_frame, frames=len(ts), init_func=init, blit=False, interval=50
)

ani.save(PLOT_DIR / "rays2d_animation.mp4", fps=10, dpi=150)
plt.close(fig)


def create_3d_ray_visualization():
    plotter = pv.Plotter(window_size=[1024, 768])

    xt_np = np.array(xt)
    ts_np = np.array(ts)

    plotter.add_axes(xlabel="X", ylabel="Y", zlabel="Time", line_width=2)
    plotter.show_grid()

    # Time scale factor to make the time dimension more visible
    time_scale = 1.0  # Adjust this if rays are too flat or too tall

    for i in range(b):
        # Create line points with time as z-coordinate
        points = np.column_stack(
            (
                xt_np[i, :, 0],  # x coordinate
                xt_np[i, :, 1],  # y coordinate
                ts_np * time_scale,  # time (z coordinate) with scaling
            )
        )

        # Verify points are valid
        if np.isnan(points).any():
            print(f"Warning: NaN values in ray {i}")
            continue

        # Create polyline
        line = pv.PolyData()
        line.points = points

        # Create line cells
        cells = np.full((len(points) - 1, 3), 2, dtype=np.int64)
        cells[:, 1] = np.arange(0, len(points) - 1, dtype=np.int64)
        cells[:, 2] = np.arange(1, len(points), dtype=np.int64)
        line.lines = cells

        # Add line to plotter with fixed color for better visibility
        color = [0, 0.8, 0.8]  # Cyan color
        plotter.add_mesh(line, color=color, line_width=5, render_lines_as_tubes=True)

    # Add a text label
    plotter.add_text("Ray Paths in 3D (X, Y, Time)", font_size=24)

    # Set initial view angle to see the 3D structure clearly
    plotter.view_isometric()
    plotter.set_background("white")

    plotter.show()


create_3d_ray_visualization()
