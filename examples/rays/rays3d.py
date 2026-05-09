#!/usr/bin/env python
# coding: utf-8



"""
Ray-tracing diagnostic for the Gaussian beam Hamiltonian.
"""
import jax.numpy as jnp
import numpy as np
from functools import partial
from pathlib import Path
from _curves import sphere_points
from _velocitymaps import velocity_field_3d

try:
    import pyvista as pv
except ModuleNotFoundError:
    print("Skipping example: pyvista is not installed (`pip install pyvista`).")
    raise SystemExit(0)

from beamax import utils
from beamax.gb import gb_utils, gb_solvers
from beamax.plotter import use_beamax_style

ROOT_DIR = utils.detect_root()
PLOT_DIR = Path(ROOT_DIR / "plots")
DATA_DIR = Path(ROOT_DIR / "data")
PROF_DIR = Path(ROOT_DIR / "profiler")
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
PROF_DIR.mkdir(exist_ok=True)

import matplotlib.pyplot as plt

use_beamax_style()

d = 3
b = 90
N = (64,) * d
dx = (1 / N[0],) * d
xmax = jnp.array(N) * jnp.array(dx)

x_linspace = [jnp.linspace(0, xmax[i], N[i]) for i in range(d)]
x, y, z = np.meshgrid(*x_linspace, indexing="ij")
XYZ = jnp.stack([x, y, z], axis=-1)

# x0 = jnp.zeros((b, d)) + jnp.array([0.5, 0.5, 0.05])
x0 = sphere_points(b, radius=0.3)
tangents = jnp.gradient(x0, axis=0)
p0 = jnp.stack([-tangents[:, 1], tangents[:, 0], tangents[:, 2]], axis=-1)
p0 = -1 * p0 / jnp.linalg.norm(p0, axis=1, keepdims=True)

# Initial conditions for other parameters
a0 = jnp.ones((b, 1)) * 0.1
mode = jnp.ones((b, 1))
ts = jnp.linspace(0, 1, 100)
alpha0 = jnp.ones((b, d)) * 1j
lam = 0

# Solve for ray trajectories
solver = gb_solvers.solve_ODE_base
M0 = None
M0 = gb_utils.prepare_M0(alpha0, M0)
is_M0_diagonal = gb_utils.is_diagonal(M0)
velocity = partial(velocity_field_3d)

xt, pt, mt, at = solver(x0, p0, M0, a0, mode, ts, velocity, lam, None)

# Create PyVista grid
grid = pv.StructuredGrid(x, y, z)
grid.point_data["c"] = velocity(XYZ).flatten()

# Convert ray trajectories to numpy
beam_points = np.array(xt)


def create_mesh_actor():
    pl = pv.Plotter()
    pl.add_mesh(grid.outline(), color="white")

    # Add velocity field contours
    contours = grid.contour(scalars="c", isosurfaces=5)
    pl.add_mesh(contours, opacity=0.3, cmap="viridis", name="velocity")

    return pl


def update_scene(time_idx):
    pl.clear_actors()
    pl.add_mesh(grid.outline(), color="white")

    # Add velocity field contours
    contours = grid.contour(scalars="c", isosurfaces=5)
    pl.add_mesh(contours, opacity=0.3, cmap="viridis", name="velocity")

    # Add ray trajectories
    for i in range(b):
        points = beam_points[i, : time_idx + 1]
        if len(points) > 1:
            beam = pv.Line(points[0], points[-1], resolution=len(points) - 1)
            pl.add_mesh(beam, color="red", line_width=2)


# Create plotter and add widgets
pl = create_mesh_actor()

# Add time slider
pl.add_slider_widget(
    callback=lambda value: update_scene(int(value * (len(ts) - 1))),
    rng=[0, 1],
    value=0,
    title="Time",
    pointa=(0.4, 0.9),
    pointb=(0.9, 0.9),
)


# Add checkbox to toggle velocity field visibility
def toggle_velocity(flag):
    pl.add_mesh(
        grid.contour(scalars="c", isosurfaces=5),
        opacity=0.3 if flag else 0.0,
        cmap="viridis",
        name="velocity",
    )


pl.add_checkbox_button_widget(
    toggle_velocity,
    value=True,
    position=(10, 10),
    size=30,
    border_size=1,
    color_on="white",
    color_off="grey",
)

# Set camera and display options
pl.set_background("grey")
pl.add_axes()
pl.show_grid()

# Show the visualization
pl.show()
