#!/usr/bin/env python
# coding: utf-8



"""
2D time-reversal comparison between MSGB and k-Wave.
"""
import jax.numpy as jnp
from beamax import utils, geometry, transforms, plotter
from beamax.decomposition import DyadicDecomposition
from beamax.gb import gb_solvers
import jax
import matplotlib.pyplot as plt
from time import time
import numpy as np

from pathlib import Path
from beamax.solvers import KWaveSolver, MSGBSolver
from beamax.solvers import hybrid_solver_utils
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions

jax.config.update("jax_enable_x64", True)
ROOT_DIR = utils.detect_root()
DATA_DIR = Path(ROOT_DIR / "data")
PLOT_DIR = Path(ROOT_DIR / "plots")
DATA_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

"""
This example shows the application of the forward and time reversal solvers for the linear wave equation.

Step 1: Forward simulation
- Set up the domain, sensors, and initial pressure field.
- Run the forward simulation to obtain the sensor data.

Step 2: Time reversal
- Set up a new domain with the sensor data as the initial pressure field.
- Run the time reversal simulation to obtain the time-reversed pressure field.

try and match the parameters
"""

pltgb = plotter.PlotHelper()

d = 2
N = (128,) * d
dx = (1e-4,) * d
box_aspect_ratio = (1,) * d
num_levels = 2
num_boxes_levels = (4, 8)  # This is a fixed value for the example
windowing = "rectangular_mirror"
redundancy = 2
total_coeffs = jnp.prod(redundancy * jnp.array(N))
strategy = "top_n"
# num_beams = int(total_coeffs * 0.1)
num_beams = 6500
batch_size = 100
input_type = "spatial"
output_type = "spatial"
thr_strat = "top_n"
sum_method = "scan_real"

print(f"num_GB_img_space: {num_beams}, batch_size: {batch_size}")

cfl = (jnp.sqrt(2) / 4).round(3)
print("Using cfl: ", cfl)

periodic = (False,) * d
solverODE = gb_solvers.solve_ODE_base
# solverODE_batch = gb_solvers.solve_ODE_batch_t
solverODE_batch = gb_solvers.solve_hom_TR

# ode_config = SolverConfig(
#     dt0=1e-4
# )


def c(x):
    # return 1000 + 0 * x[..., 0]
    return 1 + 0 * jnp.exp(
        -((x[..., 0] - 0.5 * N[0] * dx[0]) ** 2 + (x[..., 1] - 0.5 * N[1] * dx[1]) ** 2)
        / (0.1**3)
    )


domain_img = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)
XY = domain_img.grid

ts_img = domain_img.generate_time_domain()
tmax_img = ts_img[-1]
Nt = len(ts_img)

n = 2
if Nt != n * N[0]:
    ts_img = jnp.linspace(0, tmax_img, n * N[0])
    tmax_img = ts_img[-1]
    Nt = len(ts_img)
    dt_img = ts_img[1] - ts_img[0]
    cfl = c(jnp.zeros(d)) * dt_img / min(dx)
    print("Adjusted Nt to ", Nt, " with cfl: ", cfl)
    domain_img = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)

dyadic_decomp_img = DyadicDecomposition(
    num_levels, N, num_boxes_levels, box_aspect_ratio
)

# pltgb.plot_centers(dyadic_decomp.centres_ndim)

wpt_img = transforms.MSWPT(dyadic_decomp_img, redundancy, windowing)

#######################################
### SENSORS ###########################
#######################################
binary_mask = jnp.zeros(N)
# binary_mask = binary_mask.at[0, ...].set(1)
binary_mask = binary_mask.at[-1, ...].set(1)
# binary_mask = binary_mask.at[:, 0].set(1)
# binary_mask = binary_mask.at[:, -1].set(1)
sensors = geometry.Sensor(domain=domain_img, binary_mask=binary_mask)
sensors_all = geometry.Sensor(domain=domain_img, binary_mask=jnp.ones(N))

# from beamax import transforms

# KXY = dyadic_decomp_img.fourier_meshgrid

# # pltgb.plot_centers(dyadic_decomp.centres_ndim)

# boxhf = 44
# boxlf = 10

# khf = jnp.array([19, 14])
# klf = jnp.array([10, 3])
# kerft_hf = transforms.compute_frames(
#     dyadic_decomp_img, boxhf, khf, KXY, redundancy, "none"
# )
# kerft_lf = transforms.compute_frames(
#     dyadic_decomp_img, boxlf, klf, KXY, redundancy, "none"
# )
# p0 = utils.unitary_ifft(kerft_hf) + utils.unitary_ifft(kerft_lf)
# p0 = p0 / jnp.max(jnp.abs(p0))
# p0 = p0.T
# exp = 1

# boxhf = 63  # flat
# # boxhf = 54
# khf = jnp.array([8, 8])
# kerft_hf = transforms.compute_frames(
#     dyadic_decomp_img, boxhf, khf, KXY, redundancy, "none"
# )
# p0 = utils.unitary_ifft(kerft_hf)
# p0 = p0 / jnp.max(jnp.abs(p0))
# # p0 = p0.T
# exp = 1

#######################################
### POINT SOURCE ######################
#######################################

p0 = jnp.zeros(N)
p0 = p0.at[N[0] // 4 : 3 * N[0] // 4, N[1] // 4 : 3 * N[1] // 4].set(1.0)

# p0 = jnp.zeros(N)
# p0 = p0.at[N[0] // 4, N[1] // 2].set(1)
# # p0 = p0.at[N[0] // 2, N[1] // 4].set(1)
# # # p0 = p0.at[N[1]//8::N[1]//8,N[0] // 2].set(1.0)
# p0 = p0.at[N[0] // 8 :: N[0] // 8, N[1] // 8 :: N[1] // 8].set(1.0)
exp = 2

# pth = Path(DATA_DIR / "EXAMPLE_source_one.png")
# img = jnp.sum(plt.imread(pth), axis=-1)
# img = jnp.where(img > 1.5, 0.0, img)
# p0 = jax.image.resize(img, N, method="bilinear")
# exp = 5

######################################
### CIRCLES PHANTOM ##################
######################################
# from kwave.utils.mapgen import make_disc
# from kwave.data import Vector

# expected_scale = (128, 128)
# # # create initial pressure distribution using makeDisc
# N_vec = Vector(N)  # [grid points]
# disc_magnitude = 1  # [Pa]
# disc_pos = Vector([25 // 2, 60 // 2])  # [grid points]
# disc_radius = 5  # [grid points]
# disc_2 = disc_magnitude * make_disc(N_vec, disc_pos, disc_radius)

# disc_pos = Vector([80 // 2, 50 // 2])  # [grid points]
# disc_radius = 20  # [grid points]
# disc_magnitude = 2  # [Pa]
# disc_1 = disc_magnitude * make_disc(N_vec, disc_pos, disc_radius)

# p0 = disc_1 + disc_2
# exp = 6

p0 = p0.real

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(p0, origin="lower")
plt.scatter(jnp.where(sensors.binary_mask)[1], jnp.where(sensors.binary_mask)[0], c="r")
plt.colorbar()
plt.title("Initial Pressure Field")
plt.subplot(1, 2, 2)
plt.imshow(c(XY), origin="lower")
plt.colorbar()
plt.title("Sound Speed")
plt.savefig(
    PLOT_DIR / f"2d-initial-pressure-field-{exp}.png", dpi=300, bbox_inches="tight"
)
# plt.show()
plt.close()

p0_fft = utils.unitary_fft(p0)

dpdt = jnp.zeros_like(p0)

plt.figure(figsize=(10, 10))
coeffs = wpt_img.convert_to_array(wpt_img.forward(p0, input_type))
centres2 = dyadic_decomp_img.centres_ndim * 2 + jnp.array(N)
plt.imshow(jnp.abs(coeffs.T), origin="lower")
plt.colorbar()
plt.scatter(centres2[:, 0], centres2[:, 1], c="r")
for idx, (x, y) in enumerate(centres2, start=0):
    plt.annotate(
        str(idx),
        (x, y),
        fontsize=10,
        ha="left",
        va="bottom",
        xytext=(5, 5),
        textcoords="offset points",
        color="white",
    )
plt.savefig(PLOT_DIR / f"2d-coefficients-{exp}.png", dpi=300, bbox_inches="tight")
plt.close()

centres_img = dyadic_decomp_img.centres_ndim + jnp.array(N) // 2

msgb_solver = MSGBSolver(
    thr=num_beams,
    thr_strat=thr_strat,
    batch_size=batch_size,
    input_type=input_type,
    ode_solver=solverODE,
    tr_ode_solver=solverODE_batch,
    sum_method=sum_method,
)

simulation_options = SimulationOptions(
    data_cast="double",
    smooth_p0=False,
    save_to_disk=True,
)

execution_options = SimulationExecutionOptions(
    is_gpu_simulation=False, delete_data=False, verbose_level=0, show_sim_log=False
)

kwave_solver = KWaveSolver(simulation_options, execution_options)

# hybrid_solver = HybridSolver(
#     lf_solver=kwave_solver,
#     hf_solver=msgb_solver,
#     downsample=False,
#     box_corners=box_corners,
#     cutoff_freq=cutoff_freq,
#     input_type="spatial",
#     interp_method="fourier",
#     dt_oversample=0,
#     beta=12.0,
# )

t1 = time()
sensor_data_kw = kwave_solver.forward(p0, domain_img, sensors.binary_mask, ts_img)
t2 = time()
print(f"k-Wave forward solve took {t2 - t1:.2f} seconds")

t1 = time()
sensor_data_gb, params_fwd = msgb_solver.forward(
    p0,
    domain_img,
    sensors,
    ts_img,
    wpt_img,
)
t2 = time()
print(f"MSGB forward solve took {t2 - t1:.2f} seconds")


# ## Time Reversal
def cut_out_middle(arr, size):
    mid = arr.shape[0] // 2
    return arr[mid - size // 2 : mid + size // 2]


sensor_data_fft = utils.unitary_fft(sensor_data_kw)
sensor_data_fft_cropped = cut_out_middle(
    sensor_data_fft, 2 * jnp.sqrt(N[0] * N[1]).astype(int)
)
sensor_data_cropped = utils.unitary_ifft(sensor_data_fft_cropped)

print(f"Shape of sensor_data_fft: {sensor_data_fft.shape}")
print(f"Shape of sensor_data_cropped: {sensor_data_cropped.shape}")

energy = jnp.linalg.norm(sensor_data_fft)
cropped_energy = jnp.linalg.norm(sensor_data_cropped)
print(
    f"Energy: {energy}, Cropped Energy: {cropped_energy}, Ratio: {cropped_energy / energy}"
)

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(jnp.log(jnp.abs(sensor_data_fft)), origin="lower")
plt.colorbar()
plt.title("Sensor Data FFT")
plt.subplot(1, 3, 2)
plt.imshow(jnp.log(jnp.abs(sensor_data_fft_cropped)), origin="lower")
plt.colorbar()
plt.title("Cropped Sensor Data FFT")
plt.subplot(1, 3, 3)
plt.imshow(sensor_data_cropped.real, origin="lower")
plt.colorbar()
plt.title("Cropped Sensor Data")
plt.tight_layout()
plt.savefig(
    PLOT_DIR / f"2d-sensor-data-fft-cropped-{exp}.png", dpi=300, bbox_inches="tight"
)
plt.close()

N_rect = sensor_data_cropped.shape
dpdt_rect = jnp.zeros(N_rect)

Nt = max(N_rect)
N_min = min(N_rect)
ts_data = jnp.linspace(0, tmax_img, Nt)
dt_data = float(ts_data[1] - ts_data[0])
tmax_data = ts_data[-1]

dx_rect = (dt_data,) + dx[1:]
box_aspect_ratio_rect = tuple([N_rect[i] / N_min for i in range(d)])

assert jnp.allclose(
    tmax_img, tmax_data
), f"tmax_img {tmax_img} and tmax_data {tmax_data} are not equal."
print(f"N: {N}, dx: {dx}, box_aspect_ratio: {box_aspect_ratio}")
print(
    f"N_rect: {N_rect}, dx_rect: {dx_rect}, box_aspect_ratio_rect: {box_aspect_ratio_rect}"
)
print(f"ts_img: {ts_img[-1]}, ts_data: {ts_data[-1]}")

domain_data = geometry.Domain(N=N_rect, dx=dx_rect, c=c, periodic=periodic, cfl=cfl)

dyadic_decomp_data = DyadicDecomposition(
    num_levels, N_rect, num_boxes_levels, box_aspect_ratio_rect
)

wpt_data = transforms.MSWPT(dyadic_decomp_data, redundancy, windowing)

t1 = time()
p0_TR_msgb, params_TR = msgb_solver.time_reversal(
    sensor_data_cropped, domain_img, XY, sensors, ts_img, domain_data, wpt_data
)
t2 = time()
print(f"MSGB time reversal took {t2 - t1:.2f} seconds")


t1 = time()
p0_adj_msgb, params_adj = msgb_solver.adjoint(
    data=sensor_data_gb,
    domain=domain_img,
    sensors=XY,
    sources=sensors,
    ts=ts_img,
    data_domain=domain_data,
    data_wpt=wpt_data,
)
t2 = time()
print(f"Adjoint solve took {t2 - t1:.2f} seconds")

t1 = time()
p0_TR_kw = kwave_solver.time_reversal(
    sensor_data_kw.T, domain_img, jnp.ones(N), sensors.binary_mask, ts_img
).T
t2 = time()
print(f"k-Wave time reversal took {t2 - t1:.2f} seconds")

t1 = time()
p0_adj_kw = kwave_solver.adjoint(
    sensor_data_kw.T, domain_img, jnp.ones(N), sensors.binary_mask, ts_img
).T
t2 = time()
print(f"k-Wave adjoint solve took {t2 - t1:.2f} seconds")

# ## hybrid

box_corners = jnp.array([0, 15])
# box_corners = jnp.array([16, 75])
cutoff_freq = None
downsample = False
use_pow2 = False
mask = sensors.binary_mask

pltgb.plot_centers(dyadic_decomp_data.centres_ndim)

sensor_data_fft = utils.unitary_fft(sensor_data_kw)
sensor_data_fft_cropped = cut_out_middle(sensor_data_fft, 2 * N[0])
sensor_data_cropped = utils.unitary_ifft(sensor_data_fft_cropped)

plt.imshow(sensor_data_cropped.real, origin="lower")
plt.colorbar()
plt.title("Cropped Sensor Data")
plt.savefig(
    PLOT_DIR / f"2d-cropped-sensor-data-{exp}.png", dpi=300, bbox_inches="tight"
)
# plt.show()

(
    sensor_data_HF,
    sensor_data_LF,
    ds_mask,
    ds_domain,
) = hybrid_solver_utils.split_frequency_components(
    p0=sensor_data_cropped,
    sensors_mask=mask,
    input_type="spatial",
    output_type="spatial",
    wpt=wpt_data,
    box_corners=box_corners,
    windowing=wpt_data.windowing,
    domain=domain_img,
    cutoff_freq=cutoff_freq,
    downsample=downsample,
    use_pow2=use_pow2,
)

assert jnp.allclose(
    sensor_data_cropped, sensor_data_HF + sensor_data_LF, atol=1e-6
), "HF and LF components do not sum to original data."

old_shape = sensor_data_HF.shape
new_shape = (2 * N[0], N[0])

scale_factor = jnp.sqrt(jnp.prod(jnp.array(old_shape)) / jnp.prod(jnp.array(new_shape)))

sensor_data_HF_new = (
    utils.interpolate_fourier(sensor_data_HF, new_shape, "spatial", "spatial")
    / scale_factor
)
sensor_data_LF_new = (
    utils.interpolate_fourier(sensor_data_LF, new_shape, "spatial", "spatial")
    / scale_factor
)

sensor_data_fft = utils.unitary_fft(sensor_data_HF_new)
sensor_data_fft_cropped = cut_out_middle(sensor_data_fft, 2 * N[0])
sensor_data_cropped = utils.unitary_ifft(sensor_data_fft_cropped)

hf_solve = msgb_solver.time_reversal(
    sensor_data_cropped, domain_img, XY, sensors, ts_img, domain_data, wpt_data
)[0]
lf_solve = kwave_solver.time_reversal(
    sensor_data_LF_new.T,
    domain_img,
    sensors_all.binary_mask,
    sensors.binary_mask,
    ts_img,
).T

p0_TR_hyb = hf_solve + lf_solve

hf_solve_adj = msgb_solver.adjoint(
    data=sensor_data_HF_new,
    domain=domain_img,
    sensors=XY,
    sources=sensors,
    ts=ts_img,
    data_domain=domain_data,
    data_wpt=wpt_data,
)[0]
lf_solve_adj = kwave_solver.adjoint(
    sensor_data_LF_new.T,
    domain_img,
    sensors_all.binary_mask,
    sensors.binary_mask,
    ts_img,
).T
p0_adj_hyb = hf_solve_adj + lf_solve_adj

extent = (
    0,
    domain_img.grid_size[0],
    0,
    domain_img.grid_size[1],
)  # adjust as needed based on your domain


from matplotlib import gridspec


def tr_adj_comparison_with_profile(
    p0,
    p0_TR_kw,
    p0_TR_msgb,
    p0_TR_hyb,
    p0_adj_kw,
    p0_adj_msgb,
    p0_adj_hyb=None,  # can be None for now
    extent=None,
    profile_axis="x",
    profile_pos_phys=0.0,
    sensors=None,
    cmap="RdBu_r",
):
    """
    Top row:  p0, TR k-Wave, TR MSGB, TR Hybrid
    Second:   adj k-Wave, adj MSGB, adj Hybrid (or blank if None)
    Bottom:   1D profile at the same x/y position (vertical/horizontal line).

    extent: (xmin, xmax, ymin, ymax) for imshow.
    profile_axis: 'x' → vertical line; 'y' → horizontal line.
    profile_pos_phys: physical coordinate of the line (in same units as extent).
    """
    # Collect all arrays that exist so we can set common vmin/vmax
    arrays = [p0, p0_TR_kw, p0_TR_msgb, p0_TR_hyb, p0_adj_kw, p0_adj_msgb]
    if p0_adj_hyb is not None:
        arrays.append(p0_adj_hyb)

    arrays_np = [np.asarray(jnp.real(a)) for a in arrays]
    vmin = min(a.min() for a in arrays_np)
    vmax = max(a.max() for a in arrays_np)

    Ny, Nx = p0.shape
    if extent is None:
        extent = (0.0, Nx, 0.0, Ny)

    xs = np.linspace(extent[0], extent[1], Nx)
    ys = np.linspace(extent[2], extent[3], Ny)

    # Figure + GridSpec: 3 rows (2x4 images + 1x4 profile)
    fig = plt.figure(figsize=(12, 7))
    gs = gridspec.GridSpec(
        3, 4, height_ratios=[1.0, 1.0, 0.7], hspace=0.15, wspace=0.05, figure=fig
    )

    # --- Top row: p0 + TRs ---------------------------------------------------
    titles_top = ["p0", "TR k-Wave", "TR MSGB", "TR Hybrid"]
    top_arrays = [
        p0,
        p0_TR_kw,
        p0_TR_msgb,
        p0_TR_hyb if p0_TR_hyb is not None else np.zeros_like(p0),
    ]

    axes_top = []
    ims = []
    for j in range(4):
        ax = fig.add_subplot(gs[0, j])
        arr = np.asarray(jnp.real(top_arrays[j]))
        im = ax.imshow(
            arr,
            origin="lower",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            cmap=cmap,
        )
        axes_top.append(ax)
        ims.append(im)
        ax.set_title(titles_top[j])
        ax.set_xticks([])
        ax.set_yticks([])

    # --- Second row: adjoints ------------------------------------------------
    titles_mid = ["", "Adj k-Wave", "Adj MSGB", "Adj Hybrid"]
    mid_arrays = [
        None,
        p0_adj_kw,
        p0_adj_msgb,
        p0_adj_hyb if p0_adj_hyb is not None else np.zeros_like(p0),
    ]

    axes_mid = []
    for j in range(4):
        ax = fig.add_subplot(gs[1, j])
        arr = mid_arrays[j]
        if arr is not None:
            arr = np.asarray(jnp.real(arr))
            ax.imshow(
                arr,
                origin="lower",
                extent=extent,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
                cmap=cmap,
            )
        else:
            ax.axis("off")
        ax.set_title(titles_mid[j])
        ax.set_xticks([])
        ax.set_yticks([])
        axes_mid.append(ax)

    # --- Overlay sensor locations if given ----------------------------------
    if sensors is not None:
        rr, cc = jnp.where(sensors.binary_mask)
        rr = np.asarray(rr)
        cc = np.asarray(cc)
        xs_pix = np.linspace(extent[0], extent[1], Nx)
        ys_pix = np.linspace(extent[2], extent[3], Ny)
        xs_s = xs_pix[cc]
        ys_s = ys_pix[rr]
        for ax in axes_top + axes_mid:
            if ax is not None and ax.has_data():
                ax.scatter(xs_s, ys_s, s=8, c="r")

    # --- Vertical or horizontal line in all images --------------------------
    if profile_axis.lower() == "x":
        x0 = profile_pos_phys
        for ax in axes_top + axes_mid:
            if ax is not None and ax.has_data():
                ax.axvline(x0, color="k", lw=1.0, ls="--")
        # Column index corresponding to x0
        idx = int(np.clip(np.round((x0 - extent[0]) / (xs[1] - xs[0])), 0, Nx - 1))
        abscissa = ys
        line_labels = [
            "p0",
            "TR k-Wave",
            "TR MSGB",
            "TR Hybrid",
            "Adj k-Wave",
            "Adj MSGB",
            "Adj Hybrid",
        ]
        line_data = [
            p0[:, idx],
            p0_TR_kw[:, idx],
            p0_TR_msgb[:, idx],
            (p0_TR_hyb[:, idx] if p0_TR_hyb is not None else None),
            p0_adj_kw[:, idx],
            p0_adj_msgb[:, idx],
            (p0_adj_hyb[:, idx] if p0_adj_hyb is not None else None),
        ]
        xlabel = "y [m]"
        where_str = f"x = {x0:.3g} m"
    else:
        y0 = profile_pos_phys
        for ax in axes_top + axes_mid:
            if ax is not None and ax.has_data():
                ax.axhline(y0, color="k", lw=1.0, ls="--")
        # Row index corresponding to y0
        idx = int(np.clip(np.round((y0 - extent[2]) / (ys[1] - ys[0])), 0, Ny - 1))
        abscissa = xs
        line_labels = [
            "p0",
            "TR k-Wave",
            "TR MSGB",
            "TR Hybrid",
            "Adj k-Wave",
            "Adj MSGB",
            "Adj Hybrid",
        ]
        line_data = [
            p0[idx, :],
            p0_TR_kw[idx, :],
            p0_TR_msgb[idx, :],
            (p0_TR_hyb[idx, :] if p0_TR_hyb is not None else None),
            p0_adj_kw[idx, :],
            p0_adj_msgb[idx, :],
            (p0_adj_hyb[idx, :] if p0_adj_hyb is not None else None),
        ]
        xlabel = "x [m]"
        where_str = f"y = {y0:.3g} m"

    # --- Colorbar shared across all images ----------------------------------
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.6])
    fig.colorbar(ims[0], cax=cax)

    # --- Bottom row: 1D profile plot ----------------------------------------
    ax_prof = fig.add_subplot(gs[2, :])
    for dat, lab in zip(line_data, line_labels):
        if dat is None:
            continue
        ax_prof.plot(abscissa, np.asarray(jnp.real(dat)), label=lab, lw=1.5)
    ax_prof.set_xlabel(xlabel)
    ax_prof.set_ylabel("amplitude")
    ax_prof.set_title(f"Profile at {where_str}")
    ax_prof.grid(True)
    ax_prof.legend(ncol=3, fontsize=8)

    fig.tight_layout(rect=[0.0, 0.0, 0.9, 1.0])

    return fig


extent = (
    0,
    domain_img.grid_size[0],
    0,
    domain_img.grid_size[1],
)

fig = tr_adj_comparison_with_profile(
    p0=p0,
    p0_TR_kw=p0_TR_kw,
    p0_TR_msgb=p0_TR_msgb,
    p0_TR_hyb=p0_TR_hyb,
    p0_adj_kw=p0_adj_kw,
    p0_adj_msgb=p0_adj_msgb,
    p0_adj_hyb=p0_adj_hyb,
    extent=extent,
    profile_axis="x",
    profile_pos_phys=62 * dx[0],  # same as you used before
    sensors=sensors,
)

fig.savefig(
    PLOT_DIR / f"2d-TR-AND-ADJ-comparison-{exp}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
plt.close(fig)


#### validation ###

# ts_0 = jnp.array([0.0])
# gb_init = msgb_solver.forward(p0, domain_img, XY, ts_0, wpt_img)[0]


# (p0_fwd, m0_fwd, x0_fwd, ws_fwd, a0_fwd, modes_fwd) = params_fwd
# (p0_tr, m0_tr, x0_tr, ws_tr, a0_tr, signum_tr, ts_tr) = params_TR


# def flatten_params(params):
#     return tuple(rearrange(param, "a b ... -> (a b) ...") for param in params)


# p0_fwd, m0_fwd, x0_fwd, ws_fwd, a0_fwd, modes_fwd = flatten_params(
#     (p0_fwd, m0_fwd, x0_fwd, ws_fwd, a0_fwd, modes_fwd)
# )
# p0_tr, m0_tr, x0_tr, ws_tr, a0_tr, ts_tr, signum_tr = flatten_params(
#     (p0_tr, m0_tr, x0_tr, ws_tr, a0_tr, ts_tr, signum_tr)
# )
# # === BLOCK 1 (fixed plotting): TR beam starting points + directions ===

# x0_tr_np = np.array(x0_tr)  # (B, 2)
# p0_tr_np = np.array(p0_tr)  # (B, 2)
# a0_tr_np = np.array(a0_tr).reshape(-1)

# # Normalise momenta to get direction vectors
# p_norm = np.linalg.norm(p0_tr_np, axis=1, keepdims=True) + 1e-12
# dirs = p0_tr_np / p_norm

# num_show = min(200, x0_tr_np.shape[0])
# idx_sorted = np.argsort(-np.abs(a0_tr_np))
# sel = idx_sorted[:num_show]

# x_sel = x0_tr_np[sel]
# d_sel = dirs[sel]

# dx_arr = np.array(
#     dx
# )  # dx_arr[0] → spacing along axis 0 (rows), dx_arr[1] → along axis 1 (cols)

# # IMPORTANT: imshow(p0) uses (row=i, col=j) → (axis 0, axis 1)
# row_idx = x_sel[:, 0] / dx_arr[0]  # i index (vertical)
# col_idx = x_sel[:, 1] / dx_arr[1]  # j index (horizontal)

# scale = 5.0
# u = d_sel[:, 1] * scale  # horizontal component (Δj)
# v = d_sel[:, 0] * scale  # vertical component (Δi)

# plt.figure(figsize=(8, 8))
# plt.imshow(np.array(p0), origin="lower", cmap="gray")
# plt.colorbar(label="p0")

# sens_i, sens_j = np.where(np.array(sensors.binary_mask))
# plt.scatter(sens_j, sens_i, c="red", s=10, label="Sensors")

# plt.scatter(col_idx, row_idx, c="yellow", s=5, label="TR start points")

# plt.quiver(
#     col_idx,
#     row_idx,
#     u,
#     v,
#     angles="xy",
#     scale_units="xy",
#     scale=1.0,
#     color="cyan",
#     width=0.002,
#     alpha=0.7,
#     label="TR directions",
# )

# plt.legend(loc="upper right")
# plt.title("TR beams: start points and directions")
# plt.xlabel("j (x index)")
# plt.ylabel("i (y index)")
# plt.tight_layout()
# plt.savefig(
#     PLOT_DIR / f"2d-tr-beams-start-points-{exp}.png", dpi=300, bbox_inches="tight"
# )
# plt.close()

# # === CLEAN BLOCK 2: Compare forward vs TR beam positions ===
# p0_np = np.array(p0, dtype=float)

# x0_fwd_np = np.array(x0_fwd)
# a0_fwd_np = np.array(a0_fwd).reshape(-1)

# x0_tr_np = np.array(x0_tr)
# a0_tr_np = np.array(a0_tr).reshape(-1)

# dx_arr = np.array(dx)  # [dx0, dx1] = spacing along axes 0, 1

# # Pick the strongest beams in each set
# num_show = min(200, x0_fwd_np.shape[0], x0_tr_np.shape[0])
# idx_fwd = np.argsort(-np.abs(a0_fwd_np))[:num_show]
# idx_tr = np.argsort(-np.abs(a0_tr_np))[:num_show]

# x_fwd_sel = x0_fwd_np[idx_fwd]
# x_tr_sel = x0_tr_np[idx_tr]

# # Map physical coords -> index coords for plotting
# # axis 0 → row (i), axis 1 → col (j)
# i_fwd = x_fwd_sel[:, 0] / dx_arr[0]
# j_fwd = x_fwd_sel[:, 1] / dx_arr[1]

# i_tr = x_tr_sel[:, 0] / dx_arr[0]
# j_tr = x_tr_sel[:, 1] / dx_arr[1]

# sens_i, sens_j = np.where(np.array(sensors.binary_mask))

# plt.figure(figsize=(8, 8))
# plt.imshow(p0_np, origin="lower", cmap="gray")
# plt.colorbar(label="p0")

# plt.scatter(sens_j, sens_i, c="red", s=10, label="Sensors")
# plt.scatter(j_fwd, i_fwd, c="lime", s=15, label="FWD beam $x_0$")
# plt.scatter(j_tr, i_tr, c="cyan", s=15, label="TR beam $x_0$")

# plt.legend(loc="upper right")
# plt.title("Forward vs Time-Reversal beam positions")
# plt.xlabel("j (x index)")
# plt.ylabel("i (y index)")
# plt.tight_layout()
# plt.savefig(PLOT_DIR / f"2d-tr-beams-positions-{exp}.png", dpi=300, bbox_inches="tight")
# plt.close()

# # === BLOCK A: Image comparison: true p0 vs k-Wave TR vs MSGB TR ===

# p0_np = np.array(p0, dtype=float)
# kw_np = np.array(p0_TR_kw, dtype=float)
# msgb_np = np.array(p0_TR_msgb, dtype=float)

# # Common colour scale for fair visual comparison
# vmin = min(p0_np.min(), kw_np.min(), msgb_np.min())
# vmax = max(p0_np.max(), kw_np.max(), msgb_np.max())

# # Error fields
# err_kw = kw_np - p0_np
# err_msgb = msgb_np - p0_np
# err_diff = msgb_np - kw_np  # MSGB vs k-Wave


# def rmse(a, b):
#     return np.sqrt(np.mean((a - b) ** 2))


# print("=== RMSEs ===")
# print(f"RMSE(k-Wave TR, p0) : {rmse(kw_np,   p0_np):.4e}")
# print(f"RMSE(MSGB  TR, p0)  : {rmse(msgb_np, p0_np):.4e}")
# print(f"RMSE(MSGB  TR, k-W) : {rmse(msgb_np, kw_np):.4e}")

# fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# # Row 1: reconstructions
# im0 = axes[0, 0].imshow(p0_np, origin="lower", vmin=vmin, vmax=vmax)
# axes[0, 0].set_title("True $p_0$")
# axes[0, 0].set_xlabel("j (x index)")
# axes[0, 0].set_ylabel("i (y index)")

# im1 = axes[0, 1].imshow(kw_np, origin="lower", vmin=vmin, vmax=vmax)
# axes[0, 1].set_title("k-Wave TR")

# im2 = axes[0, 2].imshow(msgb_np, origin="lower", vmin=vmin, vmax=vmax)
# axes[0, 2].set_title("MSGB TR")

# # Row 2: error maps (symmetric colour scale around 0)
# err_max = max(abs(err_kw).max(), abs(err_msgb).max(), abs(err_diff).max())
# im3 = axes[1, 0].imshow(err_kw, origin="lower", vmin=-err_max, vmax=err_max, cmap="bwr")
# axes[1, 0].set_title("k-Wave TR - $p_0$")

# im4 = axes[1, 1].imshow(
#     err_msgb, origin="lower", vmin=-err_max, vmax=err_max, cmap="bwr"
# )
# axes[1, 1].set_title("MSGB TR - $p_0$")

# im5 = axes[1, 2].imshow(
#     err_diff, origin="lower", vmin=-err_max, vmax=err_max, cmap="bwr"
# )
# axes[1, 2].set_title("MSGB TR - k-Wave TR")

# for ax in axes.ravel():
#     ax.set_xticks([])
#     ax.set_yticks([])

# fig.subplots_adjust(right=0.90)
# cbar_ax1 = fig.add_axes([0.92, 0.54, 0.015, 0.35])
# cbar_ax2 = fig.add_axes([0.92, 0.09, 0.015, 0.35])
# fig.colorbar(im0, cax=cbar_ax1, label="pressure")
# fig.colorbar(im3, cax=cbar_ax2, label="error")
# plt.suptitle("TR comparison: k-Wave vs MSGB vs true $p_0$")
# plt.tight_layout(rect=[0, 0, 0.9, 0.96])
# plt.savefig(PLOT_DIR / f"2d-tr-comparison-{exp}.png", dpi=300, bbox_inches="tight")
# plt.close()

# # === BLOCK B: 1D line profiles through the source ===
# # Source indices: you used p0[N[0]//4, N[1]//2] = 1
# src_i = N[0] // 4
# src_j = N[1] // 2

# # Vertical line at fixed j = src_j
# line_p0_v = p0_np[:, src_j]
# line_kw_v = kw_np[:, src_j]
# line_msgb_v = msgb_np[:, src_j]

# # Horizontal line at fixed i = src_i
# line_p0_h = p0_np[src_i, :]
# line_kw_h = kw_np[src_i, :]
# line_msgb_h = msgb_np[src_i, :]

# fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# # Vertical slice
# axes[0].plot(line_p0_v, label="true $p_0$", linewidth=2)
# axes[0].plot(line_kw_v, label="k-Wave TR", linestyle="--")
# axes[0].plot(line_msgb_v, label="MSGB TR", linestyle=":")
# axes[0].set_title(f"Vertical slice at j = {src_j}")
# axes[0].set_xlabel("i (y index)")
# axes[0].set_ylabel("pressure")
# axes[0].legend()

# # Horizontal slice
# axes[1].plot(line_p0_h, label="true $p_0$", linewidth=2)
# axes[1].plot(line_kw_h, label="k-Wave TR", linestyle="--")
# axes[1].plot(line_msgb_h, label="MSGB TR", linestyle=":")
# axes[1].set_title(f"Horizontal slice at i = {src_i}")
# axes[1].set_xlabel("j (x index)")
# axes[1].set_ylabel("pressure")
# axes[1].legend()

# plt.tight_layout()
# plt.savefig(PLOT_DIR / f"2d-tr-line-profiles-{exp}.png", dpi=300, bbox_inches="tight")
# plt.close()
