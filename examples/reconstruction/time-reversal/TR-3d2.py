#!/usr/bin/env python
# coding: utf-8
"""
3D Time Reversal Diagnostic Script

Comprehensive diagnostics for comparing k-Wave and MSGB time reversal reconstructions.
Plots at each critical stage to isolate issues.
"""

import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from time import time
from einops import rearrange

from beamax import utils, geometry, transforms, plotter
from beamax.decomposition import DyadicDecomposition
from beamax.gb import gb_solvers
from beamax.solvers import KWaveSolver, MSGBSolver
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions

jax.config.update("jax_enable_x64", True)

ROOT_DIR = utils.detect_root()
DATA_DIR = Path(ROOT_DIR / "data")
PLOT_DIR = Path(ROOT_DIR / "plots" / "tr_3d_diagnostics")
DATA_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True, parents=True)

pltgb = plotter.PlotHelper()

# =============================================================================
# PLOTTING UTILITIES
# =============================================================================


def mip_xyz(vol):
    """Return MIP projections along each axis: (xy, xz, yz)."""
    mip_xy = np.max(vol, axis=2)  # project along z -> (Nx, Ny)
    mip_xz = np.max(vol, axis=1)  # project along y -> (Nx, Nz)
    mip_yz = np.max(vol, axis=0)  # project along x -> (Ny, Nz)
    return mip_xy, mip_xz, mip_yz


def plot_mip_comparison(
    vol_list,
    titles,
    suptitle,
    save_name,
    cmap="viridis",
    shared_scale=True,
    symmetric=False,
):
    """
    Plot MIP projections for multiple 3D volumes side by side.

    Args:
        vol_list: List of 3D arrays
        titles: List of titles for each volume
        suptitle: Overall figure title
        save_name: Filename to save
        shared_scale: Use same colorscale across all plots
        symmetric: Use symmetric colorscale around 0 (for difference plots)
    """
    n_vols = len(vol_list)
    fig = plt.figure(figsize=(4 * n_vols, 10))
    gs = gridspec.GridSpec(3, n_vols, hspace=0.3, wspace=0.3)

    # Compute global scale if shared
    if shared_scale:
        all_data = np.concatenate([v.ravel() for v in vol_list])
        if symmetric:
            vmax = np.max(np.abs(all_data))
            vmin = -vmax
        else:
            vmin, vmax = np.min(all_data), np.max(all_data)

    proj_labels = ["XY (max over Z)", "XZ (max over Y)", "YZ (max over X)"]

    for col, (vol, title) in enumerate(zip(vol_list, titles)):
        mips = mip_xyz(np.asarray(vol))

        for row, (mip, proj_label) in enumerate(zip(mips, proj_labels)):
            ax = fig.add_subplot(gs[row, col])

            if not shared_scale:
                if symmetric:
                    vmax = np.max(np.abs(mip))
                    vmin = -vmax
                else:
                    vmin, vmax = np.min(mip), np.max(mip)

            cmap_use = "RdBu_r" if symmetric else cmap
            im = ax.imshow(
                mip.T,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap_use,
                aspect="auto",
            )

            if row == 0:
                ax.set_title(title, fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(proj_label, fontsize=10)

            ax.set_xticks([])
            ax.set_yticks([])

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PLOT_DIR / save_name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_name}")


def plot_line_profiles_3d(vol_list, titles, slice_idx, axis, suptitle, save_name):
    """
    Plot 1D line profiles through 3D volumes at specified slice.

    Args:
        vol_list: List of 3D arrays
        titles: List of titles
        slice_idx: Tuple (i, j, k) specifying where to take profiles
        axis: Which axis to plot along ('x', 'y', or 'z')
        suptitle: Figure title
        save_name: Filename
    """
    i, j, k = slice_idx

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for vol, title in zip(vol_list, titles):
        vol = np.asarray(vol)

        # Profile along x (varying first index)
        line_x = vol[:, j, k]
        axes[0].plot(line_x, label=title, linewidth=2)

        # Profile along y (varying second index)
        line_y = vol[i, :, k]
        axes[1].plot(line_y, label=title, linewidth=2)

        # Profile along z (varying third index)
        line_z = vol[i, j, :]
        axes[2].plot(line_z, label=title, linewidth=2)

    axes[0].set_title(f"X profile at y={j}, z={k}")
    axes[0].set_xlabel("X index")
    axes[1].set_title(f"Y profile at x={i}, z={k}")
    axes[1].set_xlabel("Y index")
    axes[2].set_title(f"Z profile at x={i}, y={j}")
    axes[2].set_xlabel("Z index")

    for ax in axes:
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / save_name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_name}")


def plot_sensor_data_comparison(sensor_kw, sensor_msgb, ts, N, save_name):
    """
    Compare sensor data from k-Wave forward and what would be used for TR.
    Assumes planar sensor at x=0.
    """
    Nt = len(ts)

    # Reshape if needed
    if sensor_kw.ndim == 1:
        sensor_kw = sensor_kw.reshape(Nt, -1)
    if sensor_msgb.ndim == 1:
        sensor_msgb = sensor_msgb.reshape(Nt, -1)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Row 1: Full sensor data
    vmax = max(np.max(np.abs(sensor_kw)), np.max(np.abs(sensor_msgb)))

    im0 = axes[0, 0].imshow(
        sensor_kw, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax
    )
    axes[0, 0].set_title("k-Wave sensor data")
    axes[0, 0].set_ylabel("Time index")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(
        sensor_msgb, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax
    )
    axes[0, 1].set_title("MSGB sensor data")
    plt.colorbar(im1, ax=axes[0, 1])

    diff = sensor_kw - sensor_msgb
    im2 = axes[0, 2].imshow(diff, aspect="auto", cmap="RdBu_r")
    axes[0, 2].set_title(f"Difference (RMSE={np.sqrt(np.mean(diff**2)):.2e})")
    plt.colorbar(im2, ax=axes[0, 2])

    # Row 2: FFT magnitude (to check spectral content)
    fft_kw = np.abs(np.fft.fftshift(np.fft.fft2(sensor_kw)))
    fft_msgb = np.abs(np.fft.fftshift(np.fft.fft2(sensor_msgb)))

    im3 = axes[1, 0].imshow(np.log10(fft_kw + 1e-10), aspect="auto")
    axes[1, 0].set_title("k-Wave FFT (log10)")
    axes[1, 0].set_ylabel("Freq index")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(np.log10(fft_msgb + 1e-10), aspect="auto")
    axes[1, 1].set_title("MSGB FFT (log10)")
    plt.colorbar(im4, ax=axes[1, 1])

    # Time trace at center sensor
    mid_sensor = sensor_kw.shape[1] // 2
    axes[1, 2].plot(ts, sensor_kw[:, mid_sensor], label="k-Wave", linewidth=2)
    axes[1, 2].plot(ts, sensor_msgb[:, mid_sensor], "--", label="MSGB", linewidth=2)
    axes[1, 2].set_title(f"Time trace at sensor {mid_sensor}")
    axes[1, 2].set_xlabel("Time")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / save_name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_name}")


def plot_beam_diagnostics(params_TR, domain, sensors, N, dx, save_name):
    """
    Visualize TR beam starting points and directions.
    """
    (p0_tr, m0_tr, x0_tr, ws_tr, a0_tr, signum_tr, ts_tr) = params_TR

    # Flatten batch dimensions
    def flatten(x):
        return rearrange(x, "a b ... -> (a b) ...")

    x0 = np.asarray(flatten(x0_tr))
    p0 = np.asarray(flatten(p0_tr))
    a0 = np.asarray(flatten(a0_tr)).ravel()

    # Normalize momenta for direction vectors
    p_norm = np.linalg.norm(p0, axis=1, keepdims=True) + 1e-12
    dirs = p0 / p_norm

    # Select top beams by amplitude
    num_show = min(500, len(a0))
    idx_sorted = np.argsort(-np.abs(a0))[:num_show]

    x_sel = x0[idx_sorted]
    d_sel = dirs[idx_sorted]
    a_sel = np.abs(a0[idx_sorted])

    # Convert to index coordinates
    dx_arr = np.array(dx)
    idx_coords = x_sel / dx_arr

    fig = plt.figure(figsize=(16, 5))

    # XY projection
    ax1 = fig.add_subplot(131)
    scatter = ax1.scatter(
        idx_coords[:, 1], idx_coords[:, 0], c=a_sel, s=20, cmap="hot", alpha=0.7
    )
    ax1.quiver(
        idx_coords[:, 1],
        idx_coords[:, 0],
        d_sel[:, 1],
        d_sel[:, 0],
        scale=20,
        alpha=0.5,
        color="cyan",
        width=0.003,
    )
    ax1.set_xlabel("Y index")
    ax1.set_ylabel("X index")
    ax1.set_title("XY projection of beam starts")
    ax1.set_xlim(0, N[1])
    ax1.set_ylim(0, N[0])
    plt.colorbar(scatter, ax=ax1, label="|amplitude|")

    # XZ projection
    ax2 = fig.add_subplot(132)
    scatter = ax2.scatter(
        idx_coords[:, 2], idx_coords[:, 0], c=a_sel, s=20, cmap="hot", alpha=0.7
    )
    ax2.quiver(
        idx_coords[:, 2],
        idx_coords[:, 0],
        d_sel[:, 2],
        d_sel[:, 0],
        scale=20,
        alpha=0.5,
        color="cyan",
        width=0.003,
    )
    ax2.set_xlabel("Z index")
    ax2.set_ylabel("X index")
    ax2.set_title("XZ projection of beam starts")
    ax2.set_xlim(0, N[2])
    ax2.set_ylim(0, N[0])
    plt.colorbar(scatter, ax=ax2, label="|amplitude|")

    # YZ projection
    ax3 = fig.add_subplot(133)
    scatter = ax3.scatter(
        idx_coords[:, 2], idx_coords[:, 1], c=a_sel, s=20, cmap="hot", alpha=0.7
    )
    ax3.quiver(
        idx_coords[:, 2],
        idx_coords[:, 1],
        d_sel[:, 2],
        d_sel[:, 1],
        scale=20,
        alpha=0.5,
        color="cyan",
        width=0.003,
    )
    ax3.set_xlabel("Z index")
    ax3.set_ylabel("Y index")
    ax3.set_title("YZ projection of beam starts")
    ax3.set_xlim(0, N[2])
    ax3.set_ylim(0, N[1])
    plt.colorbar(scatter, ax=ax3, label="|amplitude|")

    fig.suptitle(
        f"TR Beam diagnostics (showing {num_show} strongest beams)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(PLOT_DIR / save_name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_name}")

    # Print statistics
    print("\n  Beam statistics:")
    print(f"    Total beams: {len(a0)}")
    print(f"    x0 range: [{x0.min(axis=0)}, {x0.max(axis=0)}]")
    print(f"    |a0| range: [{np.min(np.abs(a0)):.2e}, {np.max(np.abs(a0)):.2e}]")
    print(f"    Beams on sensor plane (x≈0): {np.sum(np.abs(x0[:, 0]) < dx[0])}")


def compute_metrics(p0_true, p0_recon, name=""):
    """Compute and print reconstruction metrics."""
    p0_true = np.asarray(p0_true).real
    p0_recon = np.asarray(p0_recon).real

    # RMSE
    rmse = np.sqrt(np.mean((p0_true - p0_recon) ** 2))

    # Relative L2
    rel_l2 = np.linalg.norm(p0_true - p0_recon) / (np.linalg.norm(p0_true) + 1e-12)

    # Peak SNR
    max_val = np.max(np.abs(p0_true))
    mse = np.mean((p0_true - p0_recon) ** 2)
    psnr = 10 * np.log10(max_val**2 / (mse + 1e-12))

    # Structural similarity (simple version)
    mean_true = np.mean(p0_true)
    mean_recon = np.mean(p0_recon)
    var_true = np.var(p0_true)
    var_recon = np.var(p0_recon)
    cov = np.mean((p0_true - mean_true) * (p0_recon - mean_recon))
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mean_true * mean_recon + c1) * (2 * cov + c2)) / (
        (mean_true**2 + mean_recon**2 + c1) * (var_true + var_recon + c2)
    )

    print(f"\n  {name} Metrics:")
    print(f"    RMSE:        {rmse:.4e}")
    print(f"    Rel L2:      {rel_l2:.4f}")
    print(f"    PSNR:        {psnr:.2f} dB")
    print(f"    SSIM:        {ssim:.4f}")
    print(f"    Max(true):   {np.max(np.abs(p0_true)):.4e}")
    print(f"    Max(recon):  {np.max(np.abs(p0_recon)):.4e}")

    return {"rmse": rmse, "rel_l2": rel_l2, "psnr": psnr, "ssim": ssim}


# =============================================================================
# MAIN DIAGNOSTIC ROUTINE
# =============================================================================


def run_tr_diagnostics():
    print("=" * 70)
    print("3D TIME REVERSAL DIAGNOSTICS")
    print("=" * 70)

    # =========================================================================
    # SETUP
    # =========================================================================
    print("\n[1] Setting up domain and parameters...")

    d = 3
    N = (64,) * d
    dx = (1 / 64,) * d
    # box_aspect_ratio = (1,) * d
    num_levels = 2
    num_boxes_levels = tuple([2 ** (level + 2) for level in range(num_levels)])

    windowing = "rectangular_mirror"
    redundancy = 2
    num_GB_img_space = 10000
    batch_size = 100
    input_type = "spatial"
    thr_strat = "top_n"
    sum_method = "scan_real"

    cfl = float((jnp.sqrt(3) / 4).round(3))
    periodic = (False,) * d

    def c(x):
        return 100 + 0 * x[..., 0]

    # Create domain
    domain_img = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)
    XY = domain_img.grid

    # Time grid
    ts_img = domain_img.generate_time_domain()
    tmax_img = ts_img[-1]
    Nt = len(ts_img)

    desired_Nt = 3 * N[0]
    if Nt != desired_Nt:
        ts_img = jnp.linspace(0, tmax_img, desired_Nt)
        tmax_img = ts_img[-1]
        Nt = len(ts_img)
        dt = float(ts_img[1] - ts_img[0])
        cfl = c(jnp.zeros(d)) * dt / min(dx)
        domain_img = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)

    print(f"  N={N}, dx={dx}")
    print(f"  Nt={Nt}, tmax={tmax_img:.4e}, dt={float(ts_img[1] - ts_img[0]):.4e}")
    print(f"  cfl={cfl}")

    # Decomposition
    # dyadic_decomp_img = DyadicDecomposition(
    #     num_levels, N, num_boxes_levels, box_aspect_ratio
    # )
    # wpt_img = transforms.MSWPT(dyadic_decomp_img, redundancy, windowing)

    # =========================================================================
    # SENSORS
    # =========================================================================
    print("\n[2] Setting up sensors...")

    binary_mask = jnp.zeros(N)
    binary_mask = binary_mask.at[0, ...].set(1)  # Planar sensor at x=0
    sensors = geometry.Sensor(domain=domain_img, binary_mask=binary_mask)

    num_sensors = int(jnp.sum(binary_mask))
    print("  Sensor plane at x=0")
    print(f"  Number of sensors: {num_sensors}")

    # =========================================================================
    # INITIAL PRESSURE
    # =========================================================================
    print("\n[3] Creating initial pressure (point source)...")

    p0 = jnp.zeros(N)
    src_loc = (N[0] // 4, N[1] // 3, 3 * N[2] // 4)
    p0 = p0.at[src_loc].set(1.0)

    print(f"  Point source at index {src_loc}")
    print(f"  Physical location: {tuple(src_loc[i] * dx[i] for i in range(d))}")

    # Plot initial condition
    plot_mip_comparison(
        [p0],
        ["Initial pressure p₀"],
        "Initial Condition",
        "01_initial_pressure.png",
        cmap="hot",
    )

    # =========================================================================
    # SOLVERS
    # =========================================================================
    print("\n[4] Setting up solvers...")

    tr_solver = gb_solvers.solve_hom_TR
    # tr_solver = gb_solvers.solve_ODE_batch_t

    msgb_solver = MSGBSolver(
        thr=num_GB_img_space,
        thr_strat=thr_strat,
        batch_size=batch_size,
        input_type=input_type,
        ode_solver=gb_solvers.solve_ODE_base,
        tr_ode_solver=tr_solver,
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

    # =========================================================================
    # FORWARD SIMULATION
    # =========================================================================
    print("\n[5] Running forward simulation (k-Wave)...")

    t1 = time()
    sensor_data_kw = kwave_solver.forward(p0, domain_img, sensors.binary_mask, ts_img)
    t2 = time()
    print(f"  k-Wave forward: {t2 - t1:.2f}s")
    print(f"  sensor_data_kw shape: {sensor_data_kw.shape}")
    print(
        f"  sensor_data_kw range: [{float(jnp.min(sensor_data_kw)):.4e}, {float(jnp.max(sensor_data_kw)):.4e}]"
    )

    # =========================================================================
    # SENSOR DATA PROCESSING (CRITICAL STEP)
    # =========================================================================
    print("\n[6] Processing sensor data for TR...")

    # IMPORTANT: Understand how k-Wave returns data
    # For a planar sensor, it should be (Nt, Ny, Nz) or flattened
    print(f"  Raw sensor data shape: {sensor_data_kw.shape}")

    # Reshape to (Nt, Ny, Nz)
    sensor_shape_expected = (Nt, N[1], N[2])
    print(f"  Expected sensor shape: {sensor_shape_expected}")

    # Try to reshape correctly
    if sensor_data_kw.shape == (Nt, num_sensors):
        # Flatten format: need to reshape to (Nt, Ny, Nz)
        sensor_data_kw_block = sensor_data_kw.reshape(Nt, N[1], N[2], order="C")
        print(f"  Reshaped from (Nt, num_sensors) to {sensor_data_kw_block.shape}")
    elif sensor_data_kw.shape == (num_sensors, Nt):
        # Transposed format
        sensor_data_kw_block = sensor_data_kw.T.reshape(Nt, N[1], N[2], order="C")
        print(f"  Transposed and reshaped to {sensor_data_kw_block.shape}")
    else:
        print(f"  WARNING: Unexpected shape {sensor_data_kw.shape}")
        sensor_data_kw_block = sensor_data_kw.reshape(Nt, N[1], N[2], order="C")

    # Plot raw sensor data
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Time slice at peak
    t_peak = np.argmax(np.max(np.abs(sensor_data_kw_block), axis=(1, 2)))
    im0 = axes[0].imshow(sensor_data_kw_block[t_peak], origin="lower", cmap="RdBu_r")
    axes[0].set_title(f"Sensor data at t={t_peak} (peak time)")
    axes[0].set_xlabel("Z index")
    axes[0].set_ylabel("Y index")
    plt.colorbar(im0, ax=axes[0])

    # Y-t slice
    im1 = axes[1].imshow(
        sensor_data_kw_block[:, :, N[2] // 2],
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
    )
    axes[1].set_title(f"Y-t slice at z={N[2] // 2}")
    axes[1].set_xlabel("Y index")
    axes[1].set_ylabel("Time index")
    plt.colorbar(im1, ax=axes[1])

    # Z-t slice
    im2 = axes[2].imshow(
        sensor_data_kw_block[:, N[1] // 2, :],
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
    )
    axes[2].set_title(f"Z-t slice at y={N[1] // 2}")
    axes[2].set_xlabel("Z index")
    axes[2].set_ylabel("Time index")
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle("Raw sensor data from k-Wave forward", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_raw_sensor_data.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 02_raw_sensor_data.png")

    # =========================================================================
    # FFT CROPPING
    # =========================================================================
    print("\n[7] FFT cropping of sensor data...")

    def cut_out_middle(arr, size):
        mid = arr.shape[0] // 2
        return arr[mid - size // 2 : mid + size // 2]

    sensor_data_fft = utils.unitary_fft(sensor_data_kw_block)

    # Original: 3*N[0] for 3D. Let's also try 2*N[0] for comparison
    crop_size_3N = 3 * N[0]
    # crop_size_2N = 2 * N[0]

    sensor_data_fft_cropped_3N = cut_out_middle(sensor_data_fft, crop_size_3N)
    sensor_data_cropped_3N = utils.unitary_ifft(sensor_data_fft_cropped_3N)

    energy_full = float(jnp.linalg.norm(sensor_data_kw_block))
    energy_3N = float(jnp.linalg.norm(sensor_data_cropped_3N))

    print(f"  Original shape: {sensor_data_kw_block.shape}")
    print(f"  Cropped (3N) shape: {sensor_data_cropped_3N.shape}")
    print(f"  Energy ratio (3N crop): {energy_3N / energy_full:.6f}")

    # Plot FFT and cropping
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    fft_mag = np.log10(np.abs(sensor_data_fft[:, :, N[2] // 2]) + 1e-10)
    im0 = axes[0].imshow(fft_mag, origin="lower", aspect="auto")
    axes[0].set_title("Full FFT magnitude (log10)")
    axes[0].axhline(
        Nt // 2 - crop_size_3N // 2, color="r", linestyle="--", label="3N crop"
    )
    axes[0].axhline(Nt // 2 + crop_size_3N // 2, color="r", linestyle="--")
    axes[0].legend()
    plt.colorbar(im0, ax=axes[0])

    fft_cropped_mag = np.log10(
        np.abs(sensor_data_fft_cropped_3N[:, :, N[2] // 2]) + 1e-10
    )
    im1 = axes[1].imshow(fft_cropped_mag, origin="lower", aspect="auto")
    axes[1].set_title("Cropped FFT (3N)")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(sensor_data_cropped_3N[crop_size_3N // 2].real, origin="lower")
    axes[2].set_title("Cropped data (middle time slice)")
    plt.colorbar(im2, ax=axes[2])

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_fft_cropping.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 03_fft_cropping.png")

    # =========================================================================
    # DATA DOMAIN SETUP FOR TR
    # =========================================================================
    print("\n[8] Setting up data domain for TR...")

    N_rect = sensor_data_cropped_3N.shape
    Nt_new = max(N_rect)
    N_min = min(N_rect)

    ts_data = jnp.linspace(0, tmax_img, Nt_new)
    dt_data = float(ts_data[1] - ts_data[0])

    # CRITICAL: dx_rect should be (dt, dy, dz)
    dx_rect = (dt_data,) + dx[1:]
    box_aspect_ratio_rect = tuple([N_rect[i] / N_min for i in range(d)])

    print(f"  N_rect: {N_rect}")
    print(f"  dx_rect: {dx_rect}")
    print(f"  box_aspect_ratio_rect: {box_aspect_ratio_rect}")
    print(f"  dt_data: {dt_data:.4e}")

    domain_data = geometry.Domain(N=N_rect, dx=dx_rect, c=c, periodic=periodic, cfl=cfl)

    dyadic_decomp_data = DyadicDecomposition(
        num_levels, N_rect, num_boxes_levels, box_aspect_ratio_rect
    )
    wpt_data = transforms.MSWPT(dyadic_decomp_data, redundancy, windowing)

    # =========================================================================
    # TIME REVERSAL: k-Wave
    # =========================================================================
    print("\n[9] Running k-Wave time reversal...")

    sensors_all = jnp.ones(N)

    t1 = time()
    p0_TR_kw = kwave_solver.time_reversal(
        sensor_data_kw.T, domain_img, sensors_all, sensors.binary_mask, ts_img
    ).T
    t2 = time()
    print(f"  k-Wave TR: {t2 - t1:.2f}s")
    print(f"  p0_TR_kw shape: {p0_TR_kw.shape}")
    print(
        f"  p0_TR_kw range: [{float(jnp.min(p0_TR_kw)):.4e}, {float(jnp.max(p0_TR_kw)):.4e}]"
    )

    # =========================================================================
    # TIME REVERSAL: MSGB
    # =========================================================================
    print("\n[10] Running MSGB time reversal...")

    t1 = time()
    p0_TR_msgb, params_TR = msgb_solver.time_reversal(
        sensor_data_cropped_3N, domain_img, XY, sensors, ts_img, domain_data, wpt_data
    )
    t2 = time()
    p0_TR_msgb = p0_TR_msgb.transpose(0, 2, 1)

    print(f"  MSGB TR: {t2 - t1:.2f}s")
    print(f"  p0_TR_msgb shape: {p0_TR_msgb.shape}")
    print(
        f"  p0_TR_msgb range: [{float(jnp.min(p0_TR_msgb)):.4e}, {float(jnp.max(p0_TR_msgb)):.4e}]"
    )

    # =========================================================================
    # BEAM DIAGNOSTICS
    # =========================================================================
    print("\n[11] Analyzing TR beam parameters...")
    plot_beam_diagnostics(
        params_TR, domain_img, sensors, N, dx, "04_beam_diagnostics.png"
    )

    # =========================================================================
    # COMPARISON PLOTS
    # =========================================================================
    print("\n[12] Creating comparison plots...")

    # Normalize for comparison (optional - comment out to see raw amplitudes)
    # p0_TR_msgb_norm = p0_TR_msgb / (jnp.max(jnp.abs(p0_TR_msgb)) + 1e-12) * jnp.max(jnp.abs(p0))
    # p0_TR_kw_norm = p0_TR_kw / (jnp.max(jnp.abs(p0_TR_kw)) + 1e-12) * jnp.max(jnp.abs(p0))

    # Use amplitude-scaled MSGB result (factor of 2 from 2D script)
    p0_TR_msgb_scaled = p0_TR_msgb

    # Main comparison: p0, k-Wave TR, MSGB TR
    plot_mip_comparison(
        [p0, p0_TR_kw, p0_TR_msgb_scaled],
        ["True p₀", "k-Wave TR", "MSGB TR (×2)"],
        "Time Reversal Comparison",
        "05_tr_comparison.png",
        cmap="hot",
        shared_scale=True,
    )

    # Error comparison
    err_kw = p0_TR_kw - p0
    err_msgb = p0_TR_msgb_scaled - p0
    err_diff = p0_TR_msgb_scaled - p0_TR_kw

    plot_mip_comparison(
        [err_kw, err_msgb, err_diff],
        ["k-Wave TR - p₀", "MSGB TR - p₀", "MSGB TR - k-Wave TR"],
        "Error Comparison",
        "06_error_comparison.png",
        shared_scale=True,
        symmetric=True,
    )

    # Line profiles through source
    plot_line_profiles_3d(
        [p0, p0_TR_kw, p0_TR_msgb_scaled],
        ["True p₀", "k-Wave TR", "MSGB TR (×2)"],
        src_loc,
        "x",
        f"Line profiles through source at {src_loc}",
        "07_line_profiles.png",
    )

    # =========================================================================
    # METRICS
    # =========================================================================
    print("\n[13] Computing reconstruction metrics...")

    metrics_kw = compute_metrics(p0, p0_TR_kw, "k-Wave TR")
    metrics_msgb = compute_metrics(p0, p0_TR_msgb_scaled, "MSGB TR (×2)")
    metrics_msgb_vs_kw = compute_metrics(p0_TR_kw, p0_TR_msgb_scaled, "MSGB vs k-Wave")

    # =========================================================================
    # ADDITIONAL DIAGNOSTICS
    # =========================================================================
    print("\n[14] Additional diagnostics...")

    # Check if MSGB produces anything meaningful
    print("\n  Energy in reconstructions:")
    print(f"    True p0:    {float(jnp.linalg.norm(p0)):.4e}")
    print(f"    k-Wave TR:  {float(jnp.linalg.norm(p0_TR_kw)):.4e}")
    print(f"    MSGB TR:    {float(jnp.linalg.norm(p0_TR_msgb)):.4e}")
    print(f"    MSGB TR×2:  {float(jnp.linalg.norm(p0_TR_msgb_scaled)):.4e}")

    # Check for NaN/Inf
    print("\n  NaN check:")
    print(f"    k-Wave TR has NaN: {bool(jnp.any(jnp.isnan(p0_TR_kw)))}")
    print(f"    MSGB TR has NaN:   {bool(jnp.any(jnp.isnan(p0_TR_msgb)))}")

    # Localization check
    kw_max_loc = np.unravel_index(np.argmax(np.abs(p0_TR_kw)), N)
    msgb_max_loc = np.unravel_index(np.argmax(np.abs(p0_TR_msgb)), N)

    print("\n  Peak locations:")
    print(f"    True p0:    {src_loc}")
    print(f"    k-Wave TR:  {kw_max_loc}")
    print(f"    MSGB TR:    {msgb_max_loc}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"\nAll plots saved to: {PLOT_DIR}")
    print("\nKey findings:")
    print(f"  - k-Wave RMSE:  {metrics_kw['rmse']:.4e}")
    print(f"  - MSGB RMSE:    {metrics_msgb['rmse']:.4e}")
    print(f"  - MSGB-kW RMSE: {metrics_msgb_vs_kw['rmse']:.4e}")
    print(f"  - k-Wave localizes to: {kw_max_loc} (true: {src_loc})")
    print(f"  - MSGB localizes to:   {msgb_max_loc} (true: {src_loc})")

    return {
        "p0": p0,
        "p0_TR_kw": p0_TR_kw,
        "p0_TR_msgb": p0_TR_msgb,
        "params_TR": params_TR,
        "metrics_kw": metrics_kw,
        "metrics_msgb": metrics_msgb,
    }


if __name__ == "__main__":
    results = run_tr_diagnostics()
