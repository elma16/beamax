#!/usr/bin/env python
# coding: utf-8



"""
Reference comparison of k-Wave time-reversal against the k-Wave adjoint.
"""
import jax.numpy as jnp
from matplotlib import colors
import numpy as np
from time import time
from pathlib import Path
from beamax import geometry, utils
from beamax.solvers.kwave_solver import KWaveSolver
from beamax.plotter import use_beamax_style
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT_DIR = utils.detect_root()
PLOT_DIR = Path(ROOT_DIR / "plots")
DATA_DIR = Path(ROOT_DIR / "data")
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

import matplotlib.pyplot as plt
import matplotlib.animation as animation

use_beamax_style()

"""
Compare the forward simulation, time reversal reconstruction, and adjoint reconstruction in k-Wave with animations.
"""

N = (64, 64)
d = len(N)
extent = 5e-2
dx = (extent / N[0],) * d
cfl = 0.3
periodic = (False,) * d
coords = jnp.linspace(-extent / 2, extent / 2, N[0])  # physical coord vector
Y, X = jnp.meshgrid(coords, coords, indexing="ij")


def c(x):
    return 1500 + 0 * x[..., 0]


domain = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)
ts = domain.generate_time_domain()
Nt = len(ts)

#########################
### square ##############
#########################

if d == 2:
    p0 = (
        jnp.zeros(N)
        .at[N[0] // 2 - 10 : N[0] // 2 + 10, N[1] // 2 - 10 : N[1] // 2 + 10]
        .set(1.0)
    )
elif d == 3:
    p0 = (
        jnp.zeros(N)
        .at[
            N[0] // 2 - 10 : N[0] // 2 + 10,
            N[1] // 3 - 10 : N[1] // 3 + 10,
            N[2] // 2 - 10 : N[2] // 2 + 10,
        ]
        .set(1.0)
    )

#########################
## circles and bars #####
#########################

# N = (256, 256)  # pixels (y, x)
# d = len(N)
# extent = 5e-2  # physical width [m] (assume square)
# dx = (extent / N[0],) * 2  # isotropic spacing (dy, dx)
# cfl = 0.3  # CFL condition
# periodic = (False,) * d
# coords = jnp.linspace(-extent / 2, extent / 2, N[0])  # physical coord vector
# Y, X = jnp.meshgrid(coords, coords, indexing="ij")

# p0 = jnp.zeros(N)

# circles = [(-0.015, 0.019, 0.004), (-0.015, 0.008, 0.0035), (-0.015, -0.012, 0.007)]
# for xc, yc, r in circles:
#     p0 = p0.at[((X - xc) ** 2 + (Y - yc) ** 2) < r**2].set(1.0)

# bar_x0, bar_w, gap, n_bars = 0.01, 0.0015, 0.0025, 6
# for k in range(n_bars):
#     x0 = bar_x0 + k * gap
#     p0 = p0.at[(X > x0) & (X < x0 + bar_w) & (Y > 0.005) & (Y < 0.023)].set(1.0)

# bar_y0, bar_h, gap, n_bars = -0.005, 0.0015, 0.0025, 6
# for k in range(n_bars):
#     y0 = bar_y0 - k * gap
#     p0 = p0.at[(Y < y0) & (Y > y0 - bar_h) & (X > 0.006) & (X < 0.023)].set(1.0)

sim_opts = SimulationOptions(data_cast="double", smooth_p0=False, save_to_disk=True)
exec_opts = SimulationExecutionOptions(
    is_gpu_simulation=False, delete_data=False, verbose_level=0, show_sim_log=False
)
solver = KWaveSolver(sim_opts, exec_opts)

sensor_mask = jnp.zeros(N)
# sensor_mask = sensor_mask.at[:, 0].set(1)
sensor_mask = sensor_mask.at[:, -1].set(1)
sensor_mask = sensor_mask.at[0, :].set(1)
sensor_ones = jnp.ones(N)

# build an inscribed circular sensor mask
# R = extent / 2
# dx2 = dx[0] / 2  # half the grid spacing
# r = jnp.sqrt(X**2 + Y**2)
# sensor_mask = jnp.logical_and(
#     r >= (R - dx2),
#     r <= (R + dx2)
# ).astype(jnp.int32)

t1 = time()
measurement = solver.forward(p0, domain, sensor_mask, ts, record="p")
t2 = time()
print("Time to forward solve", t2 - t1)

N_reverse = tuple(reversed(N))

measurement_anim = solver.forward(p0, domain, sensor_ones, ts, record="p").reshape(
    Nt, *N_reverse
)

t1 = time()
adj_recon = solver.adjoint(
    measurement.T, domain, sensor_ones, sensor_mask, ts, record="p"
).reshape(Nt, *N_reverse)
t2 = time()
print("Time to adjoint solve", t2 - t1)

t1 = time()
tr_recon = solver.time_reversal(
    measurement.T, domain, sensor_ones, sensor_mask, ts, record="p"
).reshape(Nt, *N_reverse)
t2 = time()
print("Time to time reversal solve", t2 - t1)


def create_wave_animation(
    data, title="Wave Propagation", figsize=(10, 8), interval=50, save_path=None
):
    """
    Create an animation from 3D wave data.

    Parameters:
    -----------
    data : array-like, shape (Nt, Ny, Nx)
        The wave data to animate
    title : str
        Title for the animation
    figsize : tuple
        Figure size (width, height)
    interval : int
        Delay between frames in milliseconds
    save_path : str or None
        If provided, save animation to this path

    Returns:
    --------
    fig, anim : matplotlib figure and animation objects
    """

    # Convert JAX array to numpy if needed
    if hasattr(data, "device"):  # JAX array
        data = np.array(data)

    Nt, Ny, Nx = data.shape

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate vmin/vmax for consistent color scaling
    vmin = np.min(data)
    vmax = np.max(data)

    # Create initial plot
    im = ax.imshow(
        data[0],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        extent=[0, Nx, 0, Ny],
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pressure")

    # Set labels and title
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{title} - Frame 0/{Nt - 1}")

    # Animation function
    def animate(frame):
        im.set_array(data[frame])
        ax.set_title(f"{title} - Frame {frame}/{Nt - 1}")
        return [im]

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=Nt, interval=interval, blit=True, repeat=True
    )

    # Save if path provided
    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer="pillow", fps=20)
        print("Animation saved!")

    return fig, anim


def create_side_by_side_animation(
    data1,
    data2,
    sensor_mask=None,  # << optional (Ny×Nx) boolean array
    titles=("Forward", "Time Reversal"),
    figsize=(15, 6),
    interval=50,
    save_path=None,
):
    # ---- convert JAX → NumPy if needed
    if hasattr(data1, "device"):
        data1 = np.asarray(data1)
    if hasattr(data2, "device"):
        data2 = np.asarray(data2)
    if sensor_mask is not None and hasattr(sensor_mask, "device"):
        sensor_mask = np.asarray(sensor_mask)

    Nt, Ny, Nx = data1.shape

    # ---- figure and axes ------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    vmin = min(data1.min(), data2.min())
    vmax = max(data1.max(), data2.max())

    im1 = ax1.imshow(data1[0], cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
    im2 = ax2.imshow(data2[0], cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")

    # ---- single colour‑bar, locked to rightmost edge --------------------
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = fig.colorbar(im2, cax=cax)  # ONE bar for both panels
    cbar.set_label("Pressure")

    # cbar = fig.colorbar(im1, ax=[ax1, ax2],
    #                     location="right", shrink=0.8, pad=0.03)
    # cbar.set_label("Pressure")

    # ---- overlay sensor markers ----------------------------------------
    if sensor_mask is not None:
        rows, cols = np.where(sensor_mask)
        ax1.plot(cols, rows, "^", ms=4, mfc="r", mec="k", linestyle="none")
        ax2.plot(cols, rows, "^", ms=4, mfc="r", mec="k", linestyle="none")

    # ---- axis cosmetics -------------------------------------------------
    for ax, t in zip((ax1, ax2), titles):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{t} - Frame 0/{Nt - 1}")
        ax.set_xlim(-0.5, Nx - 0.5)
        ax.set_ylim(Ny - 0.5, -0.5)  # keep (0,0) at top‑left

    plt.tight_layout()

    # ---- animation callback --------------------------------------------
    def animate(k):
        im1.set_array(data1[k])
        im2.set_array(data2[k])
        ax1.set_title(f"{titles[0]} - Frame {k}/{Nt - 1}")
        ax2.set_title(f"{titles[1]} - Frame {k}/{Nt - 1}")
        return im1, im2

    anim = animation.FuncAnimation(
        fig, animate, frames=Nt, interval=interval, blit=True, repeat=True
    )

    if save_path:
        print(f"Saving animation to {save_path} ...")
        anim.save(save_path, writer="pillow", fps=20)
        print("done.")

    return fig, anim


def compute_projections(vol4d, mode="max"):
    # vol4d: (Nt, Z, Y, X)
    if mode == "max":
        px = jnp.max(vol4d, axis=1)  # MIP over Z → (Nt,Y,X)
        py = jnp.max(vol4d, axis=2)  # over Y → (Nt,Z,X)
        pz = jnp.max(vol4d, axis=3)  # over X → (Nt,Z,Y)
    elif mode == "mean":
        px = jnp.mean(vol4d, axis=1)
        py = jnp.mean(vol4d, axis=2)
        pz = jnp.mean(vol4d, axis=3)
    elif mode == "central":
        zc = vol4d.shape[1] // 2
        yc = vol4d.shape[2] // 2
        xc = vol4d.shape[3] // 2
        px = vol4d[:, zc]  # central Z slice → (Nt,Y,X)
        py = vol4d[:, :, yc, :]  # (Nt,Z,X)
        pz = vol4d[:, :, :, xc]  # (Nt,Z,Y)
    else:
        raise ValueError(mode)
    return px, py, pz  # (Nt,Y,X),(Nt,Z,X),(Nt,Z,Y)


def prepare_projection_stack(vol4d, mode="max"):
    px, py, pz = compute_projections(vol4d, mode)
    # reorder all to (Nt, H, W) with consistent orientation
    # For visualization you may want Y increasing upward; adjust origin later.
    return [px, py, pz]


def animate_projection_grid(data_list, titles, interval=40, save_path=None):
    # data_list: list of arrays each (Nt,H,W), all same Nt
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    arrs = [np.asarray(a) for a in data_list]
    Nt = arrs[0].shape[0]
    vmin = min(a.min() for a in arrs)
    vmax = max(a.max() for a in arrs)
    fig, axes = plt.subplots(1, len(arrs), figsize=(5 * len(arrs), 5))
    if len(arrs) == 1:
        axes = [axes]
    ims = []
    for ax, a, t in zip(axes, arrs, titles):
        im = ax.imshow(a[0], cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(f"{t} 0/{Nt - 1}")
        ax.set_xticks([])
        ax.set_yticks([])
        ims.append(im)
    fig.tight_layout()

    def update(k):
        for im, a, ax, t in zip(ims, arrs, axes, titles):
            im.set_array(a[k])
            ax.set_title(f"{t} {k}/{Nt - 1}")
        return ims

    anim = animation.FuncAnimation(fig, update, frames=Nt, interval=interval, blit=True)
    if save_path:
        anim.save(save_path, writer="pillow", fps=int(1000 / interval))
    return fig, anim


def animate_projection_comparison(
    volA,
    volB,
    labels=("Adjoint", "Time Reversal"),
    mode="max",
    decimate=1,
    interval=40,
    save_path=None,
    cmap="RdBu_r",
):
    A = volA[::decimate]
    B = volB[::decimate]
    Aproj = prepare_projection_stack(A, mode)  # list len 3
    Bproj = prepare_projection_stack(B, mode)
    Nt = Aproj[0].shape[0]

    # Collect all data for global min/max
    all_proj = Aproj + Bproj
    vmin = min(p.min() for p in all_proj)
    vmax = max(p.max() for p in all_proj)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    names = ["MIP_Z→XY", "MIP_Y→ZX", "MIP_X→ZY"]

    fig, axes = plt.subplots(3, 2, figsize=(10, 12), constrained_layout=False)
    ims = []

    # Create images
    for r in range(3):
        for c, (proj, label) in enumerate(zip((Aproj[r], Bproj[r]), labels)):
            ax = axes[r, c]
            im = ax.imshow(proj[0], cmap=cmap, norm=norm, origin="lower")
            ax.set_title(f"{names[r]} {label} 0/{Nt - 1}")
            ax.set_xticks([])
            ax.set_yticks([])
            ims.append(im)

    # Single colorbar to right of the whole grid
    # Use the rightmost axes (bottom-right for positioning) to append a new axis
    right_ax = axes[0, -1]
    divider = make_axes_locatable(right_ax)
    cax = divider.append_axes("right", size="3%", pad=0.15)
    # Because append attaches only to that axes, expand its vertical span:
    # easiest: remove current cax and create new one spanning figure via fig.add_axes
    # (simpler method: just use fig.colorbar with ax=axes.ravel().)
    # We'll do the latter for brevity:

    # Remove the small appended cax; instead build a unified colorbar outside the grids.
    cax.remove()
    fig.subplots_adjust(right=0.88, wspace=0.08, hspace=0.25)
    cbar_ax = fig.add_axes([0.90, 0.10, 0.02, 0.80])  # [left,bottom,width,height]
    cbar = fig.colorbar(ims[0], cax=cbar_ax)
    cbar.set_label("Pressure", rotation=90)

    def update(k):
        for idx, (r, c) in enumerate(((r, c) for r in range(3) for c in range(2))):
            proj = (Aproj if c == 0 else Bproj)[r]
            ims[idx].set_array(proj[k])
            axes[r, c].set_title(f"{names[r]} {labels[c]} {k}/{Nt - 1}")
        return ims

    anim = animation.FuncAnimation(fig, update, frames=Nt, interval=interval, blit=True)

    if save_path:
        anim.save(save_path, writer="pillow", fps=int(1000 / interval))

    return fig, anim


if d == 2:
    create_wave_animation(
        measurement_anim[::10],
        title="Forward Simulation",
        figsize=(10, 5),
        interval=50,
        save_path=PLOT_DIR / "forward_simulation.gif",
    )

    # 3. Side-by-side comparison animation
    print("Creating comparison animation...")
    fig3, anim3 = create_side_by_side_animation(
        adj_recon[::10],
        tr_recon[::10],
        sensor_mask=sensor_mask.T,
        titles=["Adjoint", "Time Reversal"],
        interval=1,
        save_path=PLOT_DIR / "comparison_animation.gif",
    )
elif d == 3:
    # # Single reconstruction, 3 orthogonal MIPs
    # fig_mip_adj, anim_mip_adj = animate_projection_grid(
    #     prepare_projection_stack(adj_recon, mode='max'),
    #     ['Adjoint MIP_Z→XY','Adjoint MIP_Y→ZX','Adjoint MIP_X→ZY'],
    #     interval=30,
    #     save_path=PLOT_DIR/'adjoint_mips.gif'
    # )

    # Comparison grid between adjoint and time reversal (3 projections × 2 methods)
    fig_cmp, anim_cmp = animate_projection_comparison(
        adj_recon,
        tr_recon,
        labels=("Adjoint", "TimeReversal"),
        mode="max",
        decimate=2,
        interval=40,
        save_path=PLOT_DIR / "adjoint_vs_tr_mips.gif",
    )
