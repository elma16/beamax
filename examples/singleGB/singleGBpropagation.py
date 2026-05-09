#!/usr/bin/env python
# coding: utf-8

"""
Propagate a single Gaussian beam through a homogeneous medium.
"""
# # Example script to show how the GB changes over time.
#
# Specifically, we can show how keeping the M0 symmetric and the imaginary part positive definite gives some funky results.
#
# 1. shows the gb at snapshots in time
# 2. real and imag parts of GB in spatial and fourier domain
# 3. eigenvalues of the hessian matrix



import jax.numpy as jnp
import jax
from jax import vmap, jit
import numpy as np

from beamax import geometry, plotter, utils
from beamax.decomposition import DyadicDecomposition
from beamax.transforms import MSWPT
from beamax.gb import core, gb_utils, gb_solvers
from pathlib import Path

from time import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
from matplotlib.colors import LogNorm, Normalize

from matplotlib.patches import Ellipse

ROOT_DIR = utils.detect_root()
CACHE_DIR = Path(ROOT_DIR / "cache")
PLOT_DIR = Path(ROOT_DIR / "plots")
CACHE_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

pltgb = plotter.PlotHelper()

# ## domain setup

b = 1
d = 2
N = (512,) * d
dx = (10 / N[0],) * d
box_aspect_ratio = (1,) * d
periodic = (False,) * d
num_levels = 2


def c(x):
    return 1 + 0 * x[..., 0]


start = tuple([2 ** (level + 2) for level in range(num_levels)])
windowing = "rectangular"
input_type = "spatial"
output_type = "spatial"
redundancy = 2

cfl = 0.3
lam = 0
domain = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)
space, fourier = domain.generate_meshgrid()
XY = domain.grid
domain_size = domain.grid_size

ts = jnp.linspace(0, 10, 100)
dyadic_decomp = DyadicDecomposition(num_levels, N, start, box_aspect_ratio)
wpt = MSWPT(dyadic_decomp, redundancy, windowing)

mode = jnp.ones((b,))
if d == 1:
    x0 = jnp.array([0.2 * domain_size[0]])

elif d == 2:
    x0 = jnp.array([[0.1 * domain_size[0], 0.5 * domain_size[1]]])
elif d == 3:
    x0 = jnp.array([[0.2 * domain_size[0], 0.5 * domain_size[1], 0.2 * domain_size[2]]])

x0 = x0.reshape((b, d))
p0 = jnp.ones((b, d))
p0 = p0.at[:, 1].set(0)
p0 = p0 / jnp.linalg.norm(p0, axis=-1, keepdims=True)
a0 = jnp.ones((b,))
alpha0 = jnp.ones((b, d)) * 1j
M0 = None
ω0 = jnp.ones((b,)) * 50

# def generate_complex_positive_definite_matrix(b, d):
#     key = jax.random.PRNGKey(0)

#     A = jax.random.uniform(key, shape=(b, d, d)) * 5
#     real_part = jnp.einsum("bij,bkj->bik", A, A)

#     key, _ = jax.random.split(key)
#     B = jax.random.normal(key, shape=(b, d, d)) * 0.5
#     imag_part = jnp.einsum("bij,bkj->bik", B, B)
#     M0 = real_part + 1j * imag_part
#     return M0
# a1 = 1
# a2 = 1
# a3 = 0
# M0 = jnp.array([[[a1 + 1j, a3], [a3, a2 + 1j]]])

# M0 = generate_complex_positive_definite_matrix(b, d)

M0 = gb_utils.prepare_M0(alpha0, M0)
is_M0_diagonal = gb_utils.is_diagonal(M0)

solver = gb_solvers.solve_ODE_base
# solver = gb_solvers.solve_hom_diag
solver_config = None

print("Is M0 diagonal?", is_M0_diagonal)

t1 = time()
(xt, pt, mt, at) = solver(x0, p0, M0, a0, mode, ts, c, lam, solver_config)

u0 = core.compute_gaussian_beam(
    x0,
    p0,
    M0,
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
    solver_config,
)

phase = core.compute_phase(xt, pt, mt, XY, domain_size, jnp.array(periodic))  # * ω0[0]

print("u0 shape", u0.shape)
t2 = time()
print("Time taken to compute GB", t2 - t1)

t1 = time()
u0 = jnp.sum(u0, axis=-1)
t2 = time()
print("Time taken to compute sum", t2 - t1)


def fft_spatial(slice):
    return utils.unitary_fft(slice)


fft_result = vmap(fft_spatial, in_axes=0)(u0)

# plotter.animate_wavefield_2d(jnp.abs(fft_result), ts)

val = jnp.real(jnp.sum(u0[::20, ...], axis=0))
if d == 1:
    pltgb.plot_wavefield(
        val,
        X=space[0],
        # filename="initial_gb.png",
        title="Snapshots of a Gaussian Beam",
    )
elif d == 2:
    pltgb.plot_wavefield(
        val,
        X=space[0],
        Y=space[1],
        # filename="initial_gb.png",
        title="Snapshots of a Gaussian Beam",
        plot_type="pcolor",
    )

ts = np.array(ts)

from mpl_toolkits.axes_grid1 import make_axes_locatable

# Example array
# val = ...

h, w = val.T.shape
stretch = 1.5  # horizontal stretch factor
fig, ax = plt.subplots(figsize=(stretch * w / h * 5, 5))

# Plot the image
im = ax.imshow(val.T, aspect="equal", origin="lower")
ax.axis("off")

# Create a colorbar that matches the image height
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=8)

plt.show()

# ## Interactive plot (best viewed in script)


def plot_gb_1d(u0, space, dx, ts):
    """
    Plot 1D Gaussian beam evolution in spatial and fourier domains
    """
    # Create figure with GridSpec
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(
        3,
        2,  # 3 rows (2 for plots, 1 for slider), 2 columns
        height_ratios=[1, 1, 0.1],
        hspace=0.3,
        wspace=0.3,
    )

    # Create axes
    ax_spatial = plt.subplot(gs[0, :])  # spatial domain takes full width
    ax_fourier = plt.subplot(gs[1, :])  # fourier domain takes full width
    ax_slider = plt.subplot(gs[2, :])  # Slider takes full width

    # Initial plots
    # spatial domain
    (line_real,) = ax_spatial.plot(
        space[0], jnp.real(u0[0, ...]), "b-", label="Real Part"
    )
    (line_imag,) = ax_spatial.plot(
        space[0], jnp.imag(u0[0, ...]), "r--", label="Imaginary Part"
    )
    ax_spatial.set_title("spatial Domain GB")
    ax_spatial.set_xlabel("x")
    ax_spatial.set_ylabel("Amplitude")
    ax_spatial.legend()
    ax_spatial.grid(True)

    # fourier domain
    gb_fft = utils.unitary_fft(u0[0, ...])
    freqs = jnp.fft.fftfreq(len(space[0]), dx[0]) * 2 * jnp.pi
    (line_amp,) = ax_fourier.semilogy(freqs, jnp.abs(gb_fft), "g-", label="Amplitude")
    ax_phase = ax_fourier.twinx()  # Create second y-axis for phase
    (line_phase,) = ax_phase.plot(
        freqs, jnp.angle(gb_fft), "r.", label="Phase", markersize=1
    )

    ax_fourier.set_title("fourier Domain GB")
    ax_fourier.set_xlabel(r"$k$")
    ax_fourier.set_ylabel("Amplitude (log scale)")
    ax_phase.set_ylabel("Phase")

    # Set phase limits
    ax_phase.set_ylim(-np.pi, np.pi)
    ax_phase.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax_phase.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    # Add legends for both y-axes
    lines1, labels1 = ax_fourier.get_legend_handles_labels()
    lines2, labels2 = ax_phase.get_legend_handles_labels()
    ax_phase.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Create time slider
    ts_np = np.array(ts)
    slider = Slider(
        ax=ax_slider,
        label="Time",
        valmin=float(ts_np[0]),
        valmax=float(ts_np[-1]),
        valinit=float(ts_np[0]),
        valstep=float(ts_np[1] - ts_np[0]),
    )

    # Update function
    def update(val):
        t_idx = int(np.argmin(np.abs(ts_np - val)))

        # Update spatial domain
        line_real.set_ydata(jnp.real(u0[t_idx, ...]))
        line_imag.set_ydata(jnp.imag(u0[t_idx, ...]))

        # Update fourier domain
        gb_fft = utils.unitary_fft(u0[t_idx, ...])
        line_amp.set_ydata(jnp.abs(gb_fft))
        line_phase.set_ydata(jnp.angle(gb_fft))

        # Update limits
        ax_spatial.relim()
        ax_spatial.autoscale()
        ax_fourier.relim()
        ax_fourier.autoscale()

        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "gaussian_beam_1d.png", dpi=300, bbox_inches="tight")
    plt.show()


t_idx = 50
gb_fft = utils.unitary_fft(u0[t_idx, ...])
amp = jnp.abs(gb_fft)
ang = jnp.angle(gb_fft)

# Assuming you have spatial grid parameters - adjust these to match your simulation
# For example, if your spatial domain goes from -L/2 to L/2 with N points:
# N = amp.shape[0]  # assuming square grid
# L = 10.0  # adjust this to your actual domain size
# dx = L / N
# k_max = np.pi / dx  # Nyquist frequency
N_arr = jnp.array(N)
dx_arr = jnp.array(dx)
# k_coords = np.fft.fftfreq(N_arr, dx_arr) * 2 * np.pi  # frequency coordinates
# k_extent = [k_coords.min(), k_coords.max(), k_coords.min(), k_coords.max()]

plt.figure(figsize=(12, 5))

# Amplitude plot
plt.subplot(1, 2, 1)
im1 = plt.imshow(amp, origin="lower", cmap="viridis")
# extent=k_extent
plt.title(f"$|\\mathcal{{F}}[u_{{GB}}]|$ at $t = {ts[t_idx]:.2f}$")
plt.xlabel("$k_x$")
plt.ylabel("$k_y$")
plt.colorbar(im1, label="|FFT|")

# Phase plot
plt.subplot(1, 2, 2)
# cmap hsv
im2 = plt.imshow(ang, origin="lower", cmap="viridis", vmin=-np.pi, vmax=np.pi)
# extent = k_extent
plt.title(f"$\\arg[\\mathcal{{F}}[u_{{GB}}]]$ at $t = {ts[t_idx]:.2f}$")
plt.xlabel("$k_x$")
plt.ylabel("$k_y$")
cbar2 = plt.colorbar(im2, label="Phase (radians)")
cbar2.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
cbar2.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

plt.tight_layout()
plt.savefig(PLOT_DIR / "fourier_transform_gb.png", dpi=300, bbox_inches="tight")
plt.show()


def plot_gb_2d(u0, space, dx, ts):
    """
    Plot 2D Gaussian beam evolution in spatial and fourier domains

    Args:
        u0: Gaussian beam data (time, x, y)
        space: spatial domain coordinates
        fourier: fourier domain coordinates
        dx: Grid spacing
        ts: Time points
    """
    # Create figure with GridSpec for better control of layout
    fig = plt.figure(figsize=(14, 13))

    # Create GridSpec with extra space for colorbars
    gs = gridspec.GridSpec(
        3,
        4,  # Changed to 4 columns to accommodate separate colorbars
        height_ratios=[1, 1, 0.1],
        width_ratios=[1, 0.05, 1, 0.05],  # Adjusted for separate colorbars
        hspace=0.3,
        wspace=0.4,
    )

    # Create axes for plots and colorbars
    ax1 = plt.subplot(gs[0, 0])
    cax1 = plt.subplot(gs[0, 1])  # Colorbar for first plot
    ax2 = plt.subplot(gs[0, 2])
    cax2 = plt.subplot(gs[0, 3])  # Colorbar for second plot
    ax3 = plt.subplot(gs[1, 0])
    cax3 = plt.subplot(gs[1, 1])  # Colorbar for third plot
    ax4 = plt.subplot(gs[1, 2])
    cax4 = plt.subplot(gs[1, 3])  # Colorbar for fourth plot

    # Create slider axis
    ax_slider = plt.subplot(gs[2, :])

    # Calculate extents
    spatial_extent = [
        float(jnp.min(space[0])),
        float(jnp.max(space[0])),
        float(jnp.min(space[1])),
        float(jnp.max(space[1])),
    ]
    fourier_extent = [
        -1 / (2 * dx[0]),
        1 / (2 * dx[0]),
        -1 / (2 * dx[0]),
        1 / (2 * dx[0]),
    ]

    # Initial plots with modified colormaps and normalization
    im1 = ax1.imshow(
        jnp.real(u0[0, ...]).T,
        extent=spatial_extent,
        cmap="RdBu_r",  # Better for showing positive/negative values
        origin="lower",
    )
    ax1.set_title("Real Part")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    im2 = ax2.imshow(
        jnp.imag(u0[0, ...]).T, extent=spatial_extent, cmap="RdBu_r", origin="lower"
    )
    ax2.set_title("Imaginary Part")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    # FFT plots with separate amplitude and phase
    gb_fft = utils.unitary_fft(u0[0, ...])
    gb_amplitude = jnp.abs(gb_fft)
    gb_phase = jnp.angle(gb_fft)  # Returns values in [-π, π]

    im3 = ax3.imshow(
        gb_amplitude.T,
        extent=fourier_extent,
        norm=LogNorm(),  # Use log scale for amplitude
        cmap="viridis",
        origin="lower",
    )
    ax3.set_title("Amplitude (log scale)")
    ax3.set_xlabel(r"$k_x$")
    ax3.set_ylabel(r"$k_y$")

    im4 = ax4.imshow(
        gb_phase.T,
        extent=fourier_extent,
        cmap="twilight",  # Cyclic colormap for phase
        vmin=-np.pi,
        vmax=np.pi,
        origin="lower",
    )
    ax4.set_title("Phase")
    ax4.set_xlabel(r"$k_x$")
    ax4.set_ylabel(r"$k_y$")

    # Add colorbars with appropriate labels
    plt.colorbar(im1, cax=cax1, label="Real Value")
    plt.colorbar(im2, cax=cax2, label="Imaginary Value")
    plt.colorbar(im3, cax=cax3, label="Amplitude")
    phase_cbar = plt.colorbar(im4, cax=cax4, label="Phase")
    phase_cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    phase_cbar.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    # Add row titles
    fig.text(0.08, 0.75, "spatial Domain GB", rotation=90, va="center", fontsize=12)
    fig.text(0.08, 0.25, "fourier Domain GB", rotation=90, va="center", fontsize=12)

    # Update function for slider
    def update(val):
        t_idx = int(np.argmin(np.abs(ts - val)))

        # Update spatial domain plots
        im1.set_array(jnp.real(u0[t_idx, ...]).T)
        im2.set_array(jnp.imag(u0[t_idx, ...]).T)

        # Update fourier domain plots
        gb_fft = utils.unitary_fft(u0[t_idx, ...])
        gb_amplitude = jnp.abs(gb_fft)
        gb_phase = jnp.angle(gb_fft)

        im3.set_array(gb_amplitude.T)
        im4.set_array(gb_phase.T)

        # Only autoscale the spatial domain plots and amplitude
        im1.autoscale()
        im2.autoscale()
        im3.autoscale()
        # Phase plot keeps fixed scale

        fig.canvas.draw_idle()

    # Create time slider
    ts_np = np.array(ts)
    slider = Slider(
        ax=ax_slider,
        label="Time",
        valmin=float(ts_np[0]),
        valmax=float(ts_np[-1]),
        valinit=float(ts_np[0]),
        valstep=float(ts_np[1] - ts_np[0]),
    )

    # Register the update function with the slider
    slider.on_changed(update)

    # Adjust spacing for row titles
    plt.subplots_adjust(left=0.15)
    plt.savefig(PLOT_DIR / "gaussian_beam_2d.png", dpi=300, bbox_inches="tight")
    plt.show()


if d == 1:
    plot_gb_1d(u0, space, dx, ts)
else:
    plot_gb_2d(u0, space, dx, ts)

# ## Behaviour of eigenvalues


@jit
def compute_eigenvalues(array: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the eigenvalues of a batch of matrices.

    :arg array: The array of matrices. (b, d, d)
    :returns: The eigenvalues of the matrices. (b, d)
    """

    def eig_slice(slice):
        return jnp.linalg.eigvals(slice)

    return vmap(eig_slice)(array)


mi = compute_eigenvalues(jnp.imag(mt)).real
mr = compute_eigenvalues(jnp.real(mt)).real

print("Imaginary eigenvalues", mi.shape)
for i in range(d):
    plt.plot(ts, mi[0, ..., i], label=f"Imaginary Eigenvalue {i}")
    plt.plot(ts, mr[0, ..., i], label=f"Real Eigenvalue {i}")
plt.legend()
plt.xlabel("t")
plt.ylabel("Eigenvalue")
plt.title("Eigenvalues of the Hessian Matrix")
plt.savefig(PLOT_DIR / "eigenvalues.png", dpi=300, bbox_inches="tight")
plt.show()

t_idx = jnp.argmax(mr[0, ..., i])
tR = ts[t_idx]
zR2 = x0 + p0 * tR
zR = xt[0, t_idx, ...]

print("Rayleigh range", zR)
print("Time at rayleigh range", ts[t_idx])

guoy_phase = jnp.arctan(xt / zR)

plt.figure(figsize=(10, 6))
plt.plot(ts, guoy_phase[0, :, 0], label="Component 1")
plt.plot(ts, guoy_phase[0, :, 1], label="Component 2")
plt.xlabel("Time")
plt.ylabel("Gouy Phase (π/2 units)")
plt.yticks([0, np.pi / 4, np.pi / 2], ["0", "π/4", "π/2"])
plt.legend()
plt.title("Gouy Phase Evolution")
plt.savefig(PLOT_DIR / "guoy_phase.png", dpi=300, bbox_inches="tight")
plt.show()

ep = 1 / jnp.e**2
semi_axes = jnp.sqrt(2 * jnp.log(1 / ep) / (ω0[:, None, None] * mi))
# ep = 1e-1
# semi_axes = jnp.sqrt(2 * jnp.log(ep) / (ω0 * mi))
prin_curv = c(jnp.zeros((b, d))) * mr
radius_curv = 1 / prin_curv

for i in range(d):
    plt.plot(ts, semi_axes[0, ..., i], label=f"Semi Axis {i}")
plt.xlabel("t")
plt.ylabel("Semi Axis")
plt.legend()
plt.title("Semi Axes of Gaussian Beam")
plt.savefig(PLOT_DIR / "semi_axes.png", dpi=300, bbox_inches="tight")
plt.show()

for i in range(d):
    plt.plot(ts, prin_curv[0, ..., i], label=f"Principal Curvature {i}")
plt.xlabel("t")
plt.ylabel("Principal Curvature")
plt.legend()
plt.title("Principal Curvature")
plt.savefig(PLOT_DIR / "principal_curvature.png", dpi=300, bbox_inches="tight")
plt.show()

for i in range(d):
    plt.plot(ts, radius_curv[0, ..., i], label=f"Radius of Curvature {i}")
plt.xlabel("t")
plt.ylabel("Radius of Curvature")
plt.legend()
plt.title("Radius of Curvature")
plt.savefig(PLOT_DIR / "radius_of_curvature.png", dpi=300, bbox_inches="tight")
plt.show()

# the rate of change of the principal curvatures is largest at the rayleigh range
prin_curve_roc = jnp.gradient(prin_curv, ts, axis=1)

for i in range(d):
    plt.plot(ts, prin_curve_roc[0, ..., i], label=f"Principal Curvature ROC {i}")
plt.xlabel("t")
plt.ylabel("Rate of Change")
plt.legend()
plt.title("Rate of Change of Principal Curvature")
plt.savefig(PLOT_DIR / "principal_curvature_roc.png", dpi=300, bbox_inches="tight")
plt.show()

# ## Intensity Plots and Rayleigh range

intensity = jnp.abs(u0) ** 2
print("Intensity shape", intensity.shape)

intensity_max = jnp.expand_dims(
    jnp.max(intensity, axis=tuple(range(1, d + 1))), axis=tuple(range(1, d + 1))
)
wz = intensity_max / intensity < jnp.exp(2)

pltgb.plot_wavefield(
    jnp.sum(wz[::5, ...], axis=0),
    X=space[0],
    Y=space[1],
    # filename="initial_gb.png",
    title="Snapshots of a Gaussian Beam",
)
print("WZ shape", wz.shape)

FWHM = intensity_max / 2 < intensity
wz_alt = FWHM / jnp.sqrt(2 * jnp.log(2))


def _width_1d(u, x):
    I = jnp.abs(u) ** 2  # noqa
    s = I.sum()
    I = I / s  # noqa
    x0 = (I * x).sum()
    return 2 * jnp.sqrt(((I * ((x - x0) ** 2)).sum()))


def _width_2d(u, X, Y):
    I = jnp.abs(u) ** 2  # noqa
    s = I.sum()
    I = I / s  # noqa
    x0 = (I * X).sum()
    y0 = (I * Y).sum()
    r2 = (X - x0) ** 2 + (Y - y0) ** 2
    return 2 * jnp.sqrt((I * r2).sum()), x0, y0


if d == 1:
    X = space[0]
    w = jax.vmap(_width_1d, in_axes=(0, None))(u0, X)
    i0 = int(jnp.argmin(w))
    w0 = float(w[i0])
    iR = int(jnp.argmin(jnp.abs(w - w0 * jnp.sqrt(2.0))))
    plt.figure(figsize=(8, 3))
    plt.plot(ts, w)
    plt.axvline(float(ts[i0]), ls="--", label="waist t0")
    plt.axvline(float(ts[iR]), ls="--", label="Rayleigh tR")
    plt.xlabel("t")
    plt.ylabel("w")
    plt.legend()
    plt.savefig(PLOT_DIR / "beam_width_1d.png", dpi=300, bbox_inches="tight")
    plt.show()
    I = jnp.abs(u0) ** 2  # noqa
    extent = [float(ts[0]), float(ts[-1]), float(X.min()), float(X.max())]
    plt.figure(figsize=(8, 3))
    plt.imshow(I.T, extent=extent, origin="lower", aspect="auto")
    plt.axvline(float(ts[i0]), c="w", ls="--")
    plt.axvline(float(ts[iR]), c="w", ls="--")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("On-axis intensity")
    plt.savefig(PLOT_DIR / "spacetime_intensity_1d.png", dpi=300, bbox_inches="tight")
    plt.show()
else:
    x = jnp.linspace(0, domain_size[0], N[0])
    y = jnp.linspace(0, domain_size[1], N[1])
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    out = jax.vmap(_width_2d, in_axes=(0, None, None))(u0, X, Y)
    w, outx, outy = out[0], out[1], out[2]
    i0 = int(jnp.argmin(w))
    w0 = float(w[i0])
    iR = int(jnp.argmin(jnp.abs(w - w0 * jnp.sqrt(2.0))))

    # Improved Rayleigh-range visualization
    w_np = np.asarray(w)
    ts_np = np.asarray(ts)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(ts_np, w_np, color="C0", lw=2, label="Beam width $w(t)$")
    ax.axhline(w0, color="0.4", ls="--", lw=1.2, label="Waist $w_0$")
    ax.axhline(w0 * np.sqrt(2.0), color="0.6", ls=":", lw=1.4, label=r"$w_0 \sqrt{2}$")
    ax.axvline(float(ts[i0]), color="C1", ls="--", lw=1.4, label="Waist time")
    ax.axvline(float(ts[iR]), color="C2", ls="--", lw=1.6, label="Rayleigh time")
    ax.fill_between(
        ts_np,
        w_np,
        w0,
        where=ts_np <= ts_np[iR],
        color="C0",
        alpha=0.15,
        label="Within Rayleigh range",
    )
    ax.set_xlabel("t")
    ax.set_ylabel("Beam width")
    ax.set_title("Gaussian beam width and Rayleigh range")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(alpha=0.3, ls="--")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "rayleigh_range_profile.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 3))
    plt.plot(ts, w)
    # plt.axvline(float(ts[i0]), ls="--", label="waist t0")
    plt.axvline(float(ts[iR]), ls="--", label="Rayleigh tR")
    plt.xlabel("t")
    plt.ylabel("w")
    plt.legend()
    plt.savefig(PLOT_DIR / "beam_width_2d.png", dpi=300, bbox_inches="tight")
    plt.show()

    def _frame(idx, name):
        I = jnp.abs(u0[idx]) ** 2  # noqa
        extent = [float(X.min()), float(X.max()), float(Y.min()), float(Y.max())]
        plt.figure(figsize=(4.8, 4))
        plt.imshow(I.T, extent=extent, origin="lower", cmap="inferno")
        cx, cy = float(outx[idx]), float(outy[idx])
        rr = float(w[idx])
        circ = plt.Circle((cx, cy), rr, fill=False, linewidth=1.8, ec="w")
        plt.gca().add_patch(circ)
        plt.title(f"{name} at t={float(ts[idx]):.3g}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(
            PLOT_DIR / f"intensity_{name.lower()}.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    _frame(i0, "Waist")
    _frame(iR, "Rayleigh")


def _moments2(u, X, Y):
    I = jnp.abs(u) ** 2  # noqa
    S = I.sum()
    I = I / (S + 1e-16)  # noqa
    x0 = (I * X).sum()
    y0 = (I * Y).sum()
    dx = X - x0
    dy = Y - y0
    Cxx = (I * dx * dx).sum()
    Cyy = (I * dy * dy).sum()
    Cxy = (I * dx * dy).sum()
    return x0, y0, jnp.array([[Cxx, Cxy], [Cxy, Cyy]])


def _pick_idxs(ts, k, extra=()):
    base = jnp.linspace(0, len(ts) - 1, k).round().astype(int).tolist()
    return sorted(set([int(i) for i in base] + [int(e) for e in extra]))


def plot_montage_with_envelope(
    u0, space, ts, snapshots=6, alpha_max=0.9, gamma=2.0, cmap="RdBu_r", save=True
):
    assert u0.ndim == 3, "expect (T, Nx, Ny)"
    X, Y = space
    x0, y0, C = jax.vmap(_moments2, in_axes=(0, None, None))(u0, X, Y)
    # width along x2 (vertical) using second moment: w = 2*sqrt(Var_y)
    var_y = C[:, 1, 1]
    w = 2 * jnp.sqrt(jnp.maximum(var_y, 0))
    w_half = 0.5 * w  # half-width radius for 1/e^2
    # choose snapshots incl. waist and near Rayleigh
    i0 = int(jnp.argmin(w))
    iR = int(jnp.argmin(jnp.abs(w - w[i0] * jnp.sqrt(2.0))))
    idxs = _pick_idxs(ts, snapshots, extra=(i0, iR))

    x0_np = np.asarray(x0)
    y0_np = np.asarray(y0)
    w_half_np = np.asarray(w_half)
    C_np = np.asarray(C)

    order = np.argsort(x0_np)
    x0_ord = x0_np[order]
    y0_ord = y0_np[order]
    w_ord = w_half_np[order]

    x_min_dom = float(np.min(np.asarray(X)))
    x_max_dom = float(np.max(np.asarray(X)))
    y_min_dom = float(np.min(np.asarray(Y)))
    y_max_dom = float(np.max(np.asarray(Y)))
    extent = [x_min_dom, x_max_dom, y_min_dom, y_max_dom]

    idxs_np = np.array(idxs, dtype=int)
    u0_sel = np.asarray(u0[idxs_np])
    u0_sel_real = np.real(u0_sel)
    I_sel = np.abs(u0_sel) ** 2

    vmax = float(np.percentile(np.abs(u0_sel_real), 99))
    if vmax == 0:
        vmax = 1.0
    I_ref = float(np.percentile(I_sel, 99))
    if I_ref == 0:
        I_ref = 1.0

    norm = Normalize(vmin=-vmax, vmax=vmax)

    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    ax.set_facecolor("0.98")

    ax.fill_between(
        x0_ord,
        y0_ord - w_ord,
        y0_ord + w_ord,
        color="C0",
        alpha=0.18,
        label=r"$y_c \pm w_{1/e^2}(t)$",
        zorder=3,
    )
    ax.plot(x0_ord, y0_ord + w_ord, color="C0", lw=2.1, zorder=4)
    ax.plot(x0_ord, y0_ord - w_ord, color="C0", lw=2.1, zorder=4)
    ax.plot(x0_ord, y0_ord, color="0.2", ls=":", lw=1.2, label="Beam center", zorder=5)

    for j, idx in enumerate(idxs_np):
        U = u0_sel_real[j]
        intensity = I_sel[j]
        A = np.clip((intensity / I_ref) ** gamma, 0, 1) * alpha_max
        ax.imshow(
            U.T,
            extent=extent,
            origin="lower",
            cmap=cmap,
            norm=norm,
            alpha=A.T,
            interpolation="bilinear",
            zorder=1,
        )

        evals, evecs = np.linalg.eigh(C_np[idx])
        radii = 2 * np.sqrt(np.maximum(evals, 0))
        angle = float(np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0])))
        ec = "0.25"
        lw = 1.2
        if idx == i0:
            ec = "C2"
            lw = 1.8
        elif idx == iR:
            ec = "C1"
            lw = 1.8
        ell = Ellipse(
            (float(x0_np[idx]), float(y0_np[idx])),
            width=2.0 * float(radii[1]),
            height=2.0 * float(radii[0]),
            angle=angle,
            ec=ec,
            ls="--",
            lw=lw,
            fill=False,
            alpha=0.9,
            zorder=6,
        )
        ax.add_patch(ell)

    ax.scatter(
        x0_np[i0],
        y0_np[i0],
        s=36,
        marker="o",
        color="C2",
        edgecolor="white",
        linewidth=0.6,
        label="Waist",
        zorder=7,
    )
    ax.scatter(
        x0_np[iR],
        y0_np[iR],
        s=36,
        marker="D",
        color="C1",
        edgecolor="white",
        linewidth=0.6,
        label="Rayleigh",
        zorder=7,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label=r"$\Re(u)$")

    pad_frac = 0.05
    x_min = float(np.min(x0_np[idxs_np]) - pad_frac * (x_max_dom - x_min_dom))
    x_max = float(np.max(x0_np[idxs_np]) + pad_frac * (x_max_dom - x_min_dom))
    y_min = float(
        np.min((y0_np - w_half_np)[idxs_np]) - pad_frac * (y_max_dom - y_min_dom)
    )
    y_max = float(
        np.max((y0_np + w_half_np)[idxs_np]) + pad_frac * (y_max_dom - y_min_dom)
    )
    ax.set_xlim(max(x_min, x_min_dom), min(x_max, x_max_dom))
    ax.set_ylim(max(y_min, y_min_dom), min(y_max, y_max_dom))

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(r"Gaussian beam snapshots with $1/e^2$ envelope")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    ax.tick_params(direction="out", length=3, width=0.8, colors="0.2")
    plt.tight_layout()

    fp = PLOT_DIR / "gb_montage_envelope.png"
    if save:
        plt.savefig(fp, dpi=300, bbox_inches="tight")
        print(f"Saved {fp}")

    plt.show()


plot_montage_with_envelope(
    u0, space, ts, snapshots=6, alpha_max=0.9, gamma=2.0, save=True
)

u0_fft = utils.unitary_fft(u0)
t_idx_plot = u0_fft.shape[0] // 2
plt.figure(figsize=(8, 4))
plt.imshow(
    jnp.log(jnp.abs(u0_fft[t_idx_plot, ...]) + 1e-16).T,
    origin="lower",
    cmap="inferno",
)
plt.title(f"Fourier Transform of the Gaussian Beam at t={ts[t_idx_plot]:.2f}")
plt.xlabel("k_x")
plt.ylabel("k_y")
plt.colorbar(label="Log Amplitude")
plt.savefig(PLOT_DIR / "fourier_transform_initial_gb.png", dpi=300, bbox_inches="tight")
plt.show()
# utils.unitary_fft(u0[0, ...])
