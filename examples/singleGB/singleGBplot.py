#!/usr/bin/env python
# coding: utf-8



"""
Visualise a single Gaussian beam's amplitude and ellipse over time.
"""
import jax.numpy as jnp
import jax

from beamax import geometry, plotter, utils
from beamax.gb import core, gb_utils, gb_solvers
from pathlib import Path
import numpy as np

try:
    import pyvista as pv
except ImportError:
    pv = None

import matplotlib.pyplot as plt
import jax.profiler

ROOT_DIR = utils.detect_root()
CACHE_DIR = Path(ROOT_DIR / "cache")
PLOT_DIR = Path(ROOT_DIR / "plots")
PROF_DIR = Path(ROOT_DIR / "profiler")
CACHE_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)
PROF_DIR.mkdir(exist_ok=True)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

pltgb = plotter.PlotHelper()

b = 1
d = 1
N = (512,) * d
dx = (1 / N[0],) * d


def c(x):
    return 1 + 0 * x[..., 0]


periodic = (False,) * d
cfl = 0.3
lam = 0
domain = geometry.Domain(N, dx, c, periodic, cfl)
XY = domain.grid
ts = jnp.linspace(0, 0.1, 10)
domain_size = domain.grid_size

mode = jnp.ones((b,))
x0 = jnp.array([[0.5 * domain_size[0]]])
p0 = jnp.ones((b, d))
p0 = p0.at[:, 1].set(0)
p0 = p0 / jnp.linalg.norm(p0, axis=-1, keepdims=True)
a0 = jnp.ones((b,))

# related to the width
size = 2
alpha0 = jnp.ones((b, d)) * 1j * size
M0 = None

num_osc = 2
ω0 = jnp.ones((b,)) * num_osc**2 * jnp.pi**2 * size / 2


def generate_complex_positive_definite_matrix(b, d):
    key = jax.random.PRNGKey(0)

    A = jax.random.uniform(key, shape=(b, d, d)) * 5
    real_part = jnp.einsum("bij,bkj->bik", A, A)

    key, _ = jax.random.split(key)
    B = jax.random.normal(key, shape=(b, d, d)) * 0.5
    imag_part = jnp.einsum("bij,bkj->bik", B, B)
    M0 = 0.3 * real_part + 1j * imag_part
    return M0


if d == 1:
    M0 = gb_utils.prepare_M0(alpha0, M0)

if d == 2:
    beta1 = 0.25
    beta2 = 2.0
    phi_deg = 45
    phi = jnp.deg2rad(phi_deg)
    kappa = 0

    # rotation matrix
    cs, s = jnp.cos(phi), jnp.sin(phi)
    R = jnp.array([[cs, -s], [s, cs]], dtype=jnp.float64)

    # imaginary part (waist matrix)
    B = R @ jnp.diag(jnp.array([beta1, beta2])) @ R.T  # shape (2,2)

    eigenvals, eigenvecs = jnp.linalg.eigh(B)  # Remove batch dimension
    # print(f"Eigenvalues at {phi_deg}°: {eigenvals}")
    # print(f"Eigenvectors at {phi_deg}°: {eigenvecs}")
    # print(f"Expected: [{beta1}, {beta2}]")

    # assert False

    # real part (phase curvature)
    A = kappa * (R @ jnp.diag(jnp.array([1.0, -1.0])) @ R.T)

    # batch it and form M0
    A = A[jnp.newaxis, ...]  # shape (1,2,2)
    B = B[jnp.newaxis, ...]  # shape (1,2,2)
    M0 = A + 1j * B  # non-diagonal, symmetric

    # integrate with your existing setup
    alpha0 = None
    # M0 = gb_utils.prepare_M0(alpha0, M0)
    M0 = generate_complex_positive_definite_matrix(b, d)
    print("M0", M0)
    print("Is M0 diagonal?", gb_utils.is_diagonal(M0))

# solver = gb_solvers.solve_ODE_base
solver = gb_solvers.solve_hom_diag
solver_config = None

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

M0 = None
alpha0 = jnp.ones((b, d)) * 1j * size / 2  # Reuse the same alpha0
M0 = gb_utils.prepare_M0(alpha0, M0)
u1 = core.compute_gaussian_beam(
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

if d == 1:
    u = u0[0, :, 0]
    mag = np.abs(np.array(u))  # Magnitude at t=ts[0]
    x = np.array(XY[:, 0])

    # Define the new threshold for 1/e amplitude
    thr_1_e = 1 / np.e  # Using np.e for numpy array operations

    # find left/right crossings for each threshold
    def crossings(m, t):
        # Ensure magnitude is normalized or peak is known if not 1
        # Assuming peak magnitude is close to 1 as a0 is 1 and it's t=0
        peak_mag = np.max(m)
        idx = np.where(m < (peak_mag * t))[0]  # Compare with fraction of peak

        # Handle cases where threshold is not crossed or crossed multiple times
        # This simple logic assumes a single peak at center and symmetric decay
        left_candidates = idx[idx < N[0] // 2]
        right_candidates = idx[idx > N[0] // 2]

        if not left_candidates.size or not right_candidates.size:
            # Fallback or error if crossings not found (e.g., beam too wide/narrow for domain/threshold)
            print(f"Warning: Could not find crossings for threshold {t}")
            # Return NaNs or domain edges as a fallback to avoid crashing
            return x[0], x[-1]

        left = x[left_candidates[-1]]
        right = x[right_candidates[0]]
        return left, right

    x_1e_l, x_1e_r = crossings(mag, thr_1_e)

    a_val = float(np.array(alpha0.imag)[0])
    n_val = int(num_osc)

    plt.figure(figsize=(10, 6))
    plt.plot(x, u.real, label=r"$\Re(u_{\rm GB})$")
    plt.plot(x, mag, "--", label="Envelope")

    # vertical 1/e lines
    plt.axvline(x_1e_l, linestyle=":", color="green", label="1/e lines")
    plt.axvline(x_1e_r, linestyle=":", color="green")

    # label the vertical lines
    ymin, ymax = plt.ylim()

    param_txt = (
        rf"$\alpha = {a_val:.2f}i$" + "\n"
        rf"$n_{{\rm osc}} = {n_val}$" + "\n"
        r"$\omega = \dfrac{n_{\rm osc}^2\pi^2\,\alpha}{2}$"
    )

    plt.text(
        0.95 * domain_size[0],
        ymax * 0.95,
        param_txt,
        ha="right",
        va="top",
        bbox=dict(
            facecolor="white", edgecolor="black", pad=8, boxstyle="round,pad=0.5"
        ),
    )

    plt.xlabel("x")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper left")
    plt.xlim(0, domain_size[0])
    plt.savefig(
        PLOT_DIR / f"gaussian_beam_alpha{a_val:.2f}_nosc{n_val}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

elif d == 2:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(
        np.real(np.array(u0[0, ..., 0])),
        extent=(0, domain_size[0], 0, domain_size[1]),
        origin="lower",
        aspect="auto",
    )
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("General $M_0$")
    plt.subplot(1, 2, 2)
    plt.imshow(
        np.real(np.array(u1[0, ..., 0])),
        extent=(0, domain_size[0], 0, domain_size[1]),
        origin="lower",
        aspect="auto",
    )
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Diagonal $M_0$")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "gaussian_beam_M0.png", dpi=300, bbox_inches="tight")
    plt.show()

elif d == 3:
    XY_np = np.array(XY)
    u0_np = np.array(u0)

    # Create the structured grid with correct dimensions
    grid = pv.StructuredGrid(XY_np[..., 0], XY_np[..., 1], XY_np[..., 2])

    # Add the data values
    grid.point_data["u"] = u0_np[0, ..., 0].real.flatten()

    # Create a more informative visualization
    plotter = pv.Plotter()

    # Add the data as a volume with custom transfer function
    plotter.add_volume(grid, "u", cmap="viridis", opacity="sigmoid", shade=True)

    # Or visualize as an isosurface instead
    threshold = np.percentile(u0_np[0, ..., 0].real, 75)  # Adjust percentile as needed
    plotter.add_mesh(grid.contour([threshold]), opacity=0.7)

    # Add a slice for better visibility
    # slices = grid.slice_orthogonal()
    # plotter.add_mesh(slices, opacity=0.5)

    plotter.show()

else:
    raise ValueError("Invalid dimension")
