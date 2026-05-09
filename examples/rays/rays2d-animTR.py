#!/usr/bin/env python
# coding: utf-8

"""
Animation of ray trajectories propagating through a 2D/3D medium.
"""
# # Rays Time Reversal



from beamax.gb import core, gb_solvers
from beamax import geometry, utils
from beamax.plotter import use_beamax_style
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

jax.config.update("jax_enable_x64", True)

ROOT_DIR = utils.detect_root()
PLOT_DIR = Path(ROOT_DIR / "plots")
DATA_DIR = Path(ROOT_DIR / "data")
PROF_DIR = Path(ROOT_DIR / "profiler")
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
PROF_DIR.mkdir(exist_ok=True)

import matplotlib.pyplot as plt
import matplotlib.animation as animation

use_beamax_style()


def c(x):
    return 1 + 0.5 * x[..., 0] + 0.5 * x[..., 1]


b = 1
Nt = 100
d = 2
ts = jnp.linspace(0, 1, Nt)

N = (256,) * d
dx = (1 / N[0],) * d
periodic = (False,) * d
cfl = 0.3
lam = 0

domain = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)

XY = domain.grid
domain_size = domain.grid_size

# Set random seed
key = jax.random.PRNGKey(0)

# Generate random initial conditions
x0 = jax.random.uniform(key, (b, d))
p0 = jax.random.uniform(key, (b, d))
a0 = jax.random.uniform(key, (b,))
alpha0 = jax.random.uniform(key, (b, d))
mode = jnp.ones((b,))
ω0 = jnp.ones((b,))
lam = 0
solver_configs = None

m0 = 1j * jnp.einsum("bd,dj->bdj", alpha0, jnp.eye(d))

# Solve the forward ODE
solver = gb_solvers.solve_ODE_base
xt, pt, mt, at = solver(x0, p0, m0, a0, mode, ts, c, lam, solver_configs)

u0 = core.compute_gaussian_beam(
    x0,
    p0,
    m0,
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
    solver_configs,
)

# Extract final state
xT = xt[:, -1, :]
pT = pt[:, -1, :]
mT = mt[:, -1, ...]
aT = at[:, -1, :]

# Reverse time array for backward propagation
ts_inv = ts[::-1]

# Backward propagation from final state
xt_inv, pt_inv, mt_inv, at_inv = solver(
    xT, pT, mT, aT, mode, ts_inv, c, lam, solver_configs
)

# Convert to numpy for matplotlib
xt_np = np.array(xt[0])
xt_inv_np = np.array(xt_inv[0])

# Create animation
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(np.min(xt_np[:, 0]) - 0.05, np.max(xt_np[:, 0]) + 0.05)
ax.set_ylim(np.min(xt_np[:, 1]) - 0.05, np.max(xt_np[:, 1]) + 0.05)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Gaussian Beam Trajectory Animation")

# Initialize forward trajectory line (blue dashed)
(forward_line,) = ax.plot([], [], "b--", lw=2, label="Forward Trajectory")
(forward_point,) = ax.plot([], [], "bo", ms=6)

# Initialize backward trajectory line (red dashed)
(backward_line,) = ax.plot([], [], "r--", lw=2, label="Time-Reversed Trajectory")
(backward_point,) = ax.plot([], [], "ro", ms=6)

# Add legend
ax.legend()

# Text for indicating current phase
phase_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)


# Animation initialization function
def init():
    forward_line.set_data([], [])
    forward_point.set_data([], [])
    backward_line.set_data([], [])
    backward_point.set_data([], [])
    phase_text.set_text("")
    return forward_line, forward_point, backward_line, backward_point, phase_text


# Animation update function
def update(frame):
    if frame < Nt:
        # Forward trajectory phase
        x_data = xt_np[: frame + 1, 0]
        y_data = xt_np[: frame + 1, 1]

        forward_line.set_data(x_data, y_data)
        forward_point.set_data([x_data[-1]], [y_data[-1]])
        phase_text.set_text("Forward Propagation")

        # Hide backward trajectory
        backward_line.set_data([], [])
        backward_point.set_data([], [])
    else:
        # Keep full forward trajectory
        forward_line.set_data(xt_np[:, 0], xt_np[:, 1])
        forward_point.set_data([xt_np[-1, 0]], [xt_np[-1, 1]])

        # Backward trajectory phase
        backward_frame = frame - Nt
        x_data = xt_inv_np[: backward_frame + 1, 0]
        y_data = xt_inv_np[: backward_frame + 1, 1]

        backward_line.set_data(x_data, y_data)
        backward_point.set_data([x_data[-1]], [y_data[-1]])
        phase_text.set_text("Time-Reversed Propagation")

    return forward_line, forward_point, backward_line, backward_point, phase_text


# Create animation
anim = animation.FuncAnimation(
    fig, update, frames=2 * Nt, init_func=init, interval=50, blit=True
)
anim.save(PLOT_DIR / "gb_trajectory_animation.mp4", fps=20, dpi=150)
# plt.show(fig)
