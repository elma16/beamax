#!/usr/bin/env python
# coding: utf-8

"""
1D time-reversal reconstruction with MSGB vs. k-Wave. Part of the CI example smoke suite.
"""
# # Time Reversal with GBs in 1D


import jax.numpy as jnp
from beamax import utils, geometry, transforms, plotter
from beamax.decomposition import DyadicDecomposition
from beamax.gb import gb_solvers
import jax
import matplotlib.pyplot as plt
from einops import rearrange
from time import time

from pathlib import Path
from beamax.solvers import MSGBSolver, ShardingStrategy

jax.config.update("jax_enable_x64", True)
ROOT_DIR = utils.detect_root()
DATA_DIR = Path(ROOT_DIR / "data")
PLOT_DIR = Path(ROOT_DIR / "plots")
DATA_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

pltgb = plotter.PlotHelper()

# ## Set up domain

d = 1
N = (512,) * d
extent = (1,) * d
dx = (1e-4,) * d
box_aspect_ratio = (1,) * d
num_levels = 3
num_boxes_levels = tuple([2 ** (level + 2) for level in range(num_levels)])

windowing = "rectangular_mirror"
redundancy = 2
num_GB_img_space = 2 * N[0]
batch_size = 1
input_type = "spatial"
output_type = "spatial"
thr_strat = "top_n"
sum_method = "scan_real"
lam = 0
cfl = 0.5

periodic = (False,) * d
solver = gb_solvers.solve_ODE_base
# solverODE_batch = gb_solvers.solve_ODE_batch_t
solverODE_batch = gb_solvers.solve_hom_TR


def c(x):
    return 1 - 0 * (
        jnp.exp(-((x[..., 0] - extent[0] / 3) ** 2) / (0.1**2))
        - jnp.exp(-((x[..., 0] - 2 * extent[0] / 3) ** 2) / (0.1**2))
    )


img_domain = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)

XY = img_domain.grid

ts = img_domain.generate_time_domain()
tmax_img = ts[-1]
Nt = len(ts)
if Nt != 4 * N[0]:
    ts = jnp.linspace(0, tmax_img, 4 * N[0])
    tmax_img = ts[-1]
    Nt = len(ts)
    dt = ts[1] - ts[0]
    cfl = jnp.min(c(XY)) * dt / min(dx)
    img_domain = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)

img_dyadic_decomp = DyadicDecomposition(
    num_levels, N, num_boxes_levels, box_aspect_ratio
)

# pltgb.plot_centers(img_dyadic_decomp.centres_ndim)

img_wpt = transforms.MSWPT(img_dyadic_decomp, redundancy, windowing)

binary_mask = jnp.zeros(N)
binary_mask = binary_mask.at[-1, ...].set(1)
sensors = geometry.Sensor(domain=img_domain, binary_mask=binary_mask)

# ## Set up initial pressure

######################################
### INITIAL PRESSURE #################
######################################

# TWO GBS

from beamax import transforms

KXY = img_dyadic_decomp.fourier_meshgrid

pltgb.plot_centers(img_dyadic_decomp.centres_ndim)

boxhf = 4
boxlf = 0

khf = jnp.array(
    [
        10,
    ]
)
klf = jnp.array(
    [
        25,
    ]
)
kerft_hf = transforms.compute_frames(
    img_dyadic_decomp, boxhf, khf, KXY, redundancy, "none"
)
kerft_lf = transforms.compute_frames(
    img_dyadic_decomp, boxlf, klf, KXY, redundancy, "none"
)
p0 = utils.unitary_ifft(kerft_hf) + utils.unitary_ifft(kerft_lf)
p0 = p0 / jnp.max(jnp.abs(p0))
p0 = p0.T
exp = 1

# p0 = jnp.zeros(N)
# p0 = p0.at[N[0] // 8 : N[0] : N[0] // 8, ...].set(1.0)
# exp = 2
# p0 = p0.at[N[0] // 8 : 7 * N[0] // 8, ...].set(1.0)  # Point source
# p0 = jax.random.normal(jax.random.PRNGKey(0), N)  # Random noise

# POINT SOURCE

# p0 = jnp.zeros(N)
# p0 = p0.at[N[0] // 2 - 10:N[0] // 2 + 10].set(1)
# # p0 = p0.at[N[0] // 4, N[1] // 2].set(1)
# exp = 2

# p0 = jax.random.normal(jax.random.PRNGKey(0), N)
# exp = 3

p0 = p0.real

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(p0)
plt.title("Initial Pressure Field")
plt.subplot(1, 2, 2)
plt.plot(c(XY))
plt.title("Sound Speed")
plt.savefig(
    PLOT_DIR / f"1d-initial-pressure-field-{exp}.png", dpi=300, bbox_inches="tight"
)
plt.close()

p0_fft = utils.unitary_fft(p0)

dpdt = jnp.zeros_like(p0)

# ## Forward Solve

########################################
## Forward solve with MSGB solvers #####
########################################

centres_img = img_dyadic_decomp.centres_ndim + jnp.array(N) // 2

num_devices = jax.device_count()

mesh = jax.make_mesh((num_devices,), ("x",))

# Create sharding strategy
sharding_strategy = ShardingStrategy(mesh, beam_axis="x")

# Create solver with sharding
msgb_solver = MSGBSolver(
    thr=num_GB_img_space,
    thr_strat=thr_strat,
    batch_size=batch_size,
    input_type=input_type,
    ode_solver=solver,
    tr_ode_solver=solverODE_batch,
    sum_method=sum_method,
    sharding=sharding_strategy,
)


t1 = time()
sensor_data_gb, params_fwd = msgb_solver.forward(
    p0,
    img_domain,
    sensors.positions,
    ts,
    img_wpt,
)
t2 = time()
print(f"Forward solve took {t2 - t1:.2f} seconds")

plt.plot(ts, sensor_data_gb, label="Sensor Data")
plt.title("MSGB Sensor Data")
plt.savefig(PLOT_DIR / f"{d}d-sensor-data-gb-{exp}.png", dpi=300, bbox_inches="tight")
plt.close()

#
# ## TR solve with MSGB solvers
#
#


def cut_out_middle(arr, size):
    mid = arr.shape[0] // 2
    return arr[mid - size // 2 : mid + size // 2]


sensor_data_fft = utils.unitary_fft(sensor_data_gb)
sensor_data_fft_cropped = cut_out_middle(sensor_data_fft, N[0])
sensor_data_cropped = utils.unitary_ifft(sensor_data_fft_cropped)

energy = jnp.linalg.norm(sensor_data_gb)
cropped_energy = jnp.linalg.norm(sensor_data_cropped)
print(
    f"Energy: {energy}, Cropped Energy: {cropped_energy}, Ratio: {cropped_energy / energy}"
)

N_rect = jnp.squeeze(sensor_data_cropped).shape
dpdt_rect = jnp.zeros(N_rect)

print(f"Shape of cropped sensor data: {sensor_data_cropped.shape}")

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.plot(jnp.log(jnp.abs(sensor_data_fft.real)))
plt.title("Sensor Data FFT")
plt.subplot(1, 3, 2)
plt.plot(jnp.log(jnp.abs(sensor_data_fft_cropped.real)))
plt.title("Cropped Sensor Data FFT")
plt.subplot(1, 3, 3)
plt.plot(jnp.log(jnp.abs(sensor_data_cropped.real)))
plt.title("Cropped Sensor Data")
plt.tight_layout()
plt.savefig(
    PLOT_DIR / f"{d}d-sensor-data-fft-cropped-{exp}.png", dpi=300, bbox_inches="tight"
)
plt.close()

Nt = N_rect[0]
ts = jnp.linspace(0, tmax_img, Nt)
dt = float(ts[1] - ts[0])
dx_rect = (dt,)
box_aspect_ratio_rect = (1,)
tmax_data = ts[-1]
assert jnp.allclose(
    tmax_img, tmax_data
), f"tmax_img {tmax_img} and tmax_data {tmax_data} are not equal."

data_domain = geometry.Domain(N=N_rect, dx=dx_rect, c=c, periodic=periodic, cfl=cfl)

data_dyadic_decomp = DyadicDecomposition(
    num_levels, N_rect, num_boxes_levels, box_aspect_ratio_rect
)

data_wpt = transforms.MSWPT(data_dyadic_decomp, redundancy, windowing)

sensor_data_gb = jnp.squeeze(sensor_data_cropped)
boundary_mask = jnp.ones(N_rect)

t1 = time()
p0_TR_msgb, params_TR = msgb_solver.time_reversal(
    data=sensor_data_gb,
    domain=img_domain,
    sensors=XY,
    sources=sensors,
    ts=ts,
    data_domain=data_domain,
    data_wpt=data_wpt,
)
t2 = time()
print(f"Time-reversal solve took {t2 - t1:.2f} seconds")

# t1 = time()
# p0_adj_msgb, params_adj = msgb_solver.adjoint(
#     data=sensor_data_gb,
#     domain=img_domain,
#     sensors=XY,
#     sources=sensors,
#     ts=ts,
#     data_domain=data_domain,
#     data_wpt=data_wpt,
# )
# t2 = time()
# print(f"Adjoint solve took {t2 - t1:.2f} seconds")

(p0_fwd, m0_fwd, x0_fwd, ws_fwd, a0_fwd, modes_fwd) = params_fwd
(p0_tr, m0_tr, x0_tr, ws_tr, a0_tr, signum_tr, ts_tr) = params_TR


def flatten_params(params):
    return tuple(rearrange(param, "a b ... -> (a b) ...") for param in params)


p0_fwd, m0_fwd, x0_fwd, ws_fwd, a0_fwd = flatten_params(
    (p0_fwd, m0_fwd, x0_fwd, ws_fwd, a0_fwd)
)
p0_tr, m0_tr, x0_tr, ws_tr, a0_tr, ts_tr, signum_tr = flatten_params(
    (p0_tr, m0_tr, x0_tr, ws_tr, a0_tr, ts_tr, signum_tr)
)

print("FWD PARAMS")
print(f"p0 fwd: {p0_fwd[:num_GB_img_space]} p0_tr: {p0_tr}")
print(f"m0 fwd: {m0_fwd[:num_GB_img_space]} m0_tr: {m0_tr}")
print(f"x0 fwd: {x0_fwd[:num_GB_img_space]} x0_tr: {x0_tr}")
print(f"ws fwd: {ws_fwd[:num_GB_img_space]} ws_tr: {ws_tr}")
print(f"a0 fwd: {a0_fwd[:num_GB_img_space]} a0_tr: {a0_tr}")
print(f"modes fwd: {modes_fwd[:num_GB_img_space]} modes tr: {signum_tr}")
print(f"ts tr: {ts_tr}")

ts_0 = jnp.array([0.0])
gb_init = msgb_solver.forward(p0, img_domain, XY, ts_0, img_wpt)[0]
diff = gb_init - p0_TR_msgb
sensor_idx = sensors.positions

# plt.plot(jnp.real(gb_init).squeeze(), label="GB Initial Pressure Field (MSGB)")
plt.plot(jnp.real(p0).squeeze(), label="Original p0")
plt.plot(jnp.real(p0_TR_msgb).squeeze(), "--", label="TR Pressure Field (MSGB)")
# plt.plot(jnp.real(p0_adj_msgb).squeeze(), ":", label="Adjoint Pressure Field (MSGB)")
plt.plot(
    sensor_idx,
    jnp.full_like(sensor_idx, 0.01, dtype=float),
    "^",
    mfc="red",
    mec="k",
    ls="None",
)[0]
plt.legend()
plt.title("GB Initial Pressure Field (MSGB)")
plt.savefig(
    PLOT_DIR / f"{d}d-time-reversed-pressure-field-beamax-{exp}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

plt.plot(
    jnp.abs(diff.squeeze()),
    label="Difference between GB Init and TR Pressure Field (MSGB)",
)
plt.savefig(
    PLOT_DIR / f"{d}d-time-reversed-pressure-field-difference-beamax-{exp}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# # ---- Time-reversal snapshots & animation (1D) ------------------------------
# import numpy as np
# import jax.numpy as jnp
# from beamax.gb import core, gb_solvers


# def tr_snapshot_at_tau(
#     params_TR,
#     sensors_XY,  # domain grid as sensors (img_domain.grid)
#     tau: float,  # movie time (t_max -> 0)
#     *,
#     c_fn,  # speed of sound function (img_domain.c)
#     lam: float = 0.0,
#     domain_size: jnp.ndarray,  # img_domain.grid_size
#     periodic: jnp.ndarray,  # jnp.array(img_domain.periodic)
#     solver_config=None,
# ):
#     """
#     Compute a single TR snapshot u(x, tau) over the whole image grid.

#     params_TR : (pT, mT, xT, wT, aT, sign, ts_pair)
#         Flattened TR params from your MSGB solver (one row per beam),
#         where ts_pair[:,0] = t_gamma and ts_pair[:,1] (originally 0) is ignored here.
#     """
#     (pT, mT, xT, wT, aT, sign, ts_pair) = params_TR

#     # Per-beam launch times
#     t_gamma = ts_pair[:, 0]  # (b,)
#     tau_vec = jnp.full_like(t_gamma, float(tau))
#     # Build per-beam [t_gamma, tau] arrays (length-2 per beam)
#     ts_batch = jnp.stack([t_gamma, tau_vec], axis=-1)  # (b, 2)

#     # Compute real-valued GB field at sensors for these times.
#     # This internally integrates each beam over [t_gamma, tau] (forward in time if tau >= t_gamma,
#     # backward if tau < t_gamma; we explicitly mask those out just below).
#     gb = core.compute_gaussian_beam_real_TR(
#         x0=xT,
#         p0=pT,
#         M0=mT,
#         a0=aT,
#         ω0=wT,
#         mode=sign,
#         c=c_fn,
#         lam=lam,
#         ts=ts_batch,
#         sensors=sensors_XY,
#         domain_size=domain_size,
#         periodic=periodic,
#         ode_solver=gb_solvers.solve_ODE_batch_t,
#         solver_config=solver_config,
#     )  # shape: (2, *S, b). Frame index 1 corresponds to "tau".

#     u_tau_beams = gb[-1, ...]  # (*S, b)

#     # Only beams with tau >= t_gamma should exist yet; others contribute 0.
#     active = (tau_vec >= t_gamma).astype(u_tau_beams.dtype)  # (b,)
#     u_tau = jnp.einsum("...b,b->...", u_tau_beams, active)  # (*S,)

#     return u_tau


# def tr_frames_global_clock(
#     params_TR,
#     sensors_XY,
#     frame_times_desc,  # e.g. ts[::-1] or jnp.linspace(tmax_img, 0, nframes)
#     *,
#     c_fn,
#     lam: float,
#     domain_size,
#     periodic,
#     solver_config=None,
#     progress: bool = True,
# ):
#     """
#     Produce an array of TR frames u(x, tau_k) for tau_k descending from t_max -> 0.

#     Returns: frames with shape (nframes, N) in 1D (or (nframes, Nx, Ny) in 2D).
#     """
#     # Ensure we iterate in Python space (avoid JAX loop overhead & recompiles per tau)
#     taus = np.array(frame_times_desc, dtype=float)
#     frames = []
#     if progress:
#         try:
#             from tqdm import tqdm

#             tau_iter = tqdm(taus, desc="TR frames")
#         except Exception:
#             tau_iter = taus
#     else:
#         tau_iter = taus

#     for tau in tau_iter:
#         u_tau = tr_snapshot_at_tau(
#             params_TR,
#             sensors_XY,
#             tau,
#             c_fn=c_fn,
#             lam=lam,
#             domain_size=domain_size,
#             periodic=periodic,
#             solver_config=solver_config,
#         )
#         frames.append(u_tau)

#     return jnp.stack(frames, axis=0)  # (nframes, N) for 1D


# You already have:
# (p0_tr, m0_tr, x0_tr, ws_tr, a0_tr, ts_tr, signum_tr) = flatten_params(...)
# img_domain, XY, lam, etc., are defined above.

# params_TR_flat = (p0_tr, m0_tr, x0_tr, ws_tr, a0_tr, signum_tr, ts_tr)

# # Choose movie times τ from t_max → 0.
# # Using the same total horizon as your forward time grid is a safe default:
# frame_times = ts[
#     :
# ]  # descending (t_max -> 0). You can also subsample: ts[::-10][::-1], etc.

# U_frames = tr_frames_global_clock(
#     params_TR_flat,
#     sensors_XY=XY,  # evaluate on full image grid (as sensors)
#     frame_times_desc=frame_times,  # τ descending
#     c_fn=img_domain.c,
#     lam=lam,
#     domain_size=img_domain.grid_size,
#     periodic=jnp.array(img_domain.periodic),
#     solver_config=None,  # or your gb_solvers.SolverConfig(...)
#     progress=True,
# )

# # Quick validation: final frame should match your TR reconstruction (up to normalization)
# rel_err = float(
#     jnp.linalg.norm(U_frames[-1] - p0_TR_msgb) / (jnp.linalg.norm(p0_TR_msgb) + 1e-30)
# )
# print(f"[check] relative L2 error between last frame and p0_TR_msgb: {rel_err:.3e}")

# U_frames = U_frames[:, ::-1] * 2

# from matplotlib import animation, pyplot as plt

# # True (time-reversed) target
# ref = np.asarray(jnp.real(p0_TR_msgb).squeeze())

# fig, ax = plt.subplots(figsize=(10, 4))

# # Plot the first frame and the reference
# (line_frame,) = ax.plot(
#     np.asarray(jnp.real(U_frames[0]).squeeze()), lw=1.5, label="U(x, τ)"
# )
# (line_ref,) = ax.plot(ref, "--", lw=1.2, color="black", label="Target (p₀_TR)")

# # Combined limits so both are visible
# ymin = float(min(U_frames.min(), ref.min()))
# ymax = float(max(U_frames.max(), ref.max()))
# ax.set_ylim(ymin, ymax)
# ax.set_xlim(0, U_frames.shape[1])
# ax.legend(loc="upper right")
# ax.set_title("Time Reversal (τ: t_max → 0)")


# # Update only the evolving frame
# def update(i):
#     line_frame.set_ydata(np.asarray(jnp.real(U_frames[i]).squeeze()))
#     return (line_frame,)


# ani = animation.FuncAnimation(
#     fig, update, frames=U_frames.shape[0], interval=50, blit=True
# )

# ani.save(PLOT_DIR / "time_reversal_1d_with_true_overlay.mp4", fps=20, dpi=150)
# plt.show()
# plt.close()
