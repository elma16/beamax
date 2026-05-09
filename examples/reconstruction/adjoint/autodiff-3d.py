#!/usr/bin/env python
# coding: utf-8



"""
3D adjoint via JAX autodiff.
"""
import jax.numpy as jnp
import jax as jax
from time import time
import matplotlib.pyplot as plt
from pathlib import Path
import equinox as eqx
from beamax import geometry, plotter, utils
from beamax.decomposition import DyadicDecomposition
from beamax.transforms import MSWPT
from beamax.gb import gb_solvers
from beamax.solvers import MSGBSolver, KWaveSolver, ShardingStrategy
from beamax.plotter import use_beamax_style
import numpy as np
from matplotlib import animation

from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions

try:
    import optax
except ModuleNotFoundError:
    print("Skipping example: optax is not installed (`pip install optax`).")
    raise SystemExit(0)

try:
    from tqdm import tqdm
except ModuleNotFoundError:

    def tqdm(iterable, *args, **kwargs):
        return iterable


"""
This example shows forward and adjoint solve of a 2D wave equation using the MSGB solver.
"""

jax.config.update("jax_enable_x64", True)

pltgb = plotter.PlotHelper()

ROOT_DIR = utils.detect_root()
PLOT_DIR = Path(ROOT_DIR / "plots")
DATA_DIR = Path(ROOT_DIR / "data")
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

use_beamax_style()

d = 3
N = (32,) * d
dx = (1e-4,) * d
extent = tuple([dx[i] * N[i] for i in range(d)])
periodic = (False,) * d
box_aspect_ratio = (1,) * d
num_levels = 2
num_boxes_level = (4,) * num_levels
iters = 101
lr = 1e-1


def c(x):
    return 1 + 0 * x[..., 0]


c0 = c(jnp.zeros(N))

windowing = "rectangular_mirror"
input_type = "spatial"
output_type = "spatial"
redundancy = 2

cfl = jnp.sqrt(d) / 4  # for 2D, cfl should be <= sqrt(2)/2

domain = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)
XY, KXY = domain.generate_meshgrid()

KXY = jnp.stack(KXY, axis=-1)

ts = domain.generate_time_domain()
Nt = len(ts)
dt = ts[1] - ts[0]

t1 = time()
dyadic_decomp = DyadicDecomposition(num_levels, N, num_boxes_level, box_aspect_ratio)
wpt = MSWPT(dyadic_decomp, redundancy, windowing)
t2 = time()
print("Time to create params", t2 - t1)

# pltgb.plot_centers(dyadic_decomp.centres_ndim)

binary_mask = jnp.zeros(N)
binary_mask = binary_mask.at[0, ...].set(1)
binary_mask = binary_mask.at[..., 0].set(1)
# X, Y = XY
# R = extent[0] / 2 - dx[0]  # radius of the circle
# dx2 = dx[0] / 2  # half the grid spacing
# r = jnp.sqrt((X - R)**2 + (Y - R)**2)
# binary_mask = jnp.logical_and(
#     r >= (R - dx2),
#     r <= (R + dx2)
# ).astype(jnp.int32)

sensors = geometry.Sensor(domain, binary_mask=binary_mask)
sensor_idx = np.argwhere(np.array(binary_mask) == 1)  # shape (Ns, 2) with [iy, ix]
sensor_y = sensor_idx[:, 0]
sensor_x = sensor_idx[:, 1]

p0 = jnp.zeros(N)
# p0 = p0.at[64].set(1)

# boxidx = 6
# k = jnp.array([3]).reshape(1)
# p0 = transforms.compute_frames(
#     dyadic_decomp, boxidx, jnp.array([3]), KXY, redundancy, windowing
# )

p0 = p0.at[N[0] // 2 - 10 : N[0] // 2 + 10, N[1] // 2 - 10 : N[1] // 2 + 10].set(1)

# p0 = jax.random.normal(jax.random.PRNGKey(0), N)

# normalise p0
p0 = p0 / jnp.max(jnp.abs(p0))

dpdt = jnp.zeros_like(p0)

p0 = p0.real
dpdt = dpdt.real

# p0 = jnp.array(p0, dtype=jnp.complex128)
# dpdt = jnp.array(dpdt, dtype=jnp.complex128)

# dpdt = dpdt.at[64].set(1)

threshold = 1000
strategy = "top_n"
batch_size = 100
method = "all_real"
solver = gb_solvers.solve_hom_general

num_devices = jax.device_count()

mesh = jax.make_mesh((num_devices,), ("x",))

# Create sharding strategy
sharding_strategy = ShardingStrategy(mesh, beam_axis="x")

msgb_solver = MSGBSolver(
    thr=threshold,
    thr_strat=strategy,
    batch_size=batch_size,
    input_type="spatial",
    ode_solver=solver,
    sum_method=method,
    sharding=sharding_strategy,
)

for _ in range(2):
    t1 = time()
    sensor_data, _ = msgb_solver.forward(p0, domain, sensors, ts, wpt)
    t2 = time()
    print("Time to forward solve", t2 - t1)

sensor_data = sensor_data.real

print(f"Shape of sensor_data: {sensor_data.shape}")


def forward_sensor_data(p0, dpdt):
    sd, _ = msgb_solver.forward(p0, domain, sensors, ts, wpt)
    return sd


def mse_loss(pred, meas):
    return 0.5 * jnp.mean((pred - meas) ** 2)


# -------------- Synthetic data --------------
true_p0 = p0  # from your construction above
true_dpdt = dpdt

simulation_options = SimulationOptions(
    data_cast="single",
    smooth_p0=False,
    save_to_disk=True,
)
execution_options = SimulationExecutionOptions(
    is_gpu_simulation=False, delete_data=False, verbose_level=0, show_sim_log=False
)
kwave_solver = KWaveSolver(simulation_options, execution_options)

measurements = kwave_solver.forward(p0, domain, binary_mask, ts)

# measurements = forward_sensor_data(true_p0, true_dpdt)

# -------------- Inversion variables --------------
recon_p0 = jnp.zeros_like(true_p0) + 1
recon_dpdt = jnp.zeros_like(true_dpdt) + 1


def loss_p0(recon_p0):
    pred = forward_sensor_data(recon_p0, recon_dpdt)
    return mse_loss(pred, measurements) + 1e-6 * 0.5 * jnp.mean(recon_p0**2)


var = recon_p0
loss_fn = loss_p0

optim = optax.adam(lr)
opt_state = optim.init(var)


@eqx.filter_jit
def step(var, opt_state):
    loss, grad = jax.value_and_grad(loss_fn)(var)
    updates, opt_state = optim.update(grad, opt_state, params=var)
    var = optax.apply_updates(var, updates)
    return var, opt_state, loss


hist_p0 = []
hist_loss = []
hist_iter = []

for it in tqdm(range(iters)):
    var, opt_state, loss = step(var, opt_state)
    hist_p0.append(var)
    hist_loss.append(loss)
    hist_iter.append(it)

    if it % 10 == 0:
        rel_err = jnp.linalg.norm(var - true_p0) / jnp.linalg.norm(true_p0)
        print(it, "loss", float(loss), "rel_p0_err", float(rel_err))

true_np = np.asarray(true_p0)
hist_np = np.asarray(hist_p0)  # (F, Nx, Ny)
loss_np = np.asarray(hist_loss)
iter_np = np.asarray(hist_iter)

vmin = min(true_np.min(), hist_np.min())
vmax = max(true_np.max(), hist_np.max())
res_abs_max = np.max(np.abs(true_np - hist_np[0]))

fig = plt.figure(figsize=(6, 6))
gs = fig.add_gridspec(2, 2, height_ratios=[4, 1])
ax_recon = fig.add_subplot(gs[0, 0])
ax_resid = fig.add_subplot(gs[0, 1])
ax_loss = fig.add_subplot(gs[1, :])

im_recon = ax_recon.imshow(
    hist_np[0],
    origin="lower",
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
    interpolation="nearest",
)
ax_recon.set_title("Iteration 0 recon")
ax_recon.set_xticks([])
ax_recon.set_yticks([])

im_resid = ax_resid.imshow(
    true_np - hist_np[0],
    origin="lower",
    cmap="seismic",
    vmin=-res_abs_max,
    vmax=res_abs_max,
    interpolation="nearest",
)
ax_resid.set_title("Residual")
ax_resid.set_xticks([])
ax_resid.set_yticks([])

# Plot sensors on reconstruction panel
scatter_recon = ax_recon.scatter(
    sensor_x,
    sensor_y,
    marker="^",
    s=50,
    c="red",
    edgecolors="k",
    linewidths=0.5,
    label="sensor",
)

# Plot sensors on residual panel (optional)
scatter_resid = ax_resid.scatter(
    sensor_x, sensor_y, marker="^", s=50, c="red", edgecolors="k", linewidths=0.5
)

# Include sensors in legend if desired
handles, labels = ax_recon.get_legend_handles_labels()
ax_recon.legend(handles, labels, loc="upper right")

ax_loss.set_xlabel("Iteration")
ax_loss.set_ylabel("Loss")
ax_loss.set_yscale("log")
(line_loss,) = ax_loss.plot([iter_np[0]], [loss_np[0]], "r-")
ax_loss.set_xlim(0, iter_np[-1] if iter_np[-1] > 0 else 1)
ax_loss.set_ylim(loss_np.min() * 0.9, loss_np.max() * 1.1)

cbar1 = fig.colorbar(im_recon, ax=ax_recon, fraction=0.046, pad=0.04)
cbar1.set_label("p0")
cbar2 = fig.colorbar(im_resid, ax=ax_resid, fraction=0.046, pad=0.04)
cbar2.set_label("Residual")


def animate(k):
    im_recon.set_data(hist_np[k])
    resid = true_np - hist_np[k]
    im_resid.set_data(resid)
    im_resid.set_clim(
        -np.max(np.abs(resid)), np.max(np.abs(resid))
    )  # optional dynamic scaling
    line_loss.set_data(iter_np[: k + 1], loss_np[: k + 1])
    ax_recon.set_title(f"Iteration {iter_np[k]} recon")
    return im_recon, im_resid, line_loss, scatter_recon, scatter_resid


ani = animation.FuncAnimation(
    fig, animate, frames=hist_np.shape[0], interval=120, blit=False
)
ani.save(PLOT_DIR / "recon2d_iters.mp4", fps=10, dpi=150)
plt.close(fig)
