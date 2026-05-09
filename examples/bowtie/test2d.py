#!/usr/bin/env python3
# coding: utf-8
"""
Planar line-sensor wave data: 2D FFT in (sensor coordinate, time) and support-energy analysis.

This script:
  1) Solves the 2D linear wave IVP with k-Wave (via beamax.KWaveSolver).
  2) Records pressure on a line sensor (binary mask).
  3) Computes the unitary 2D FFT over (sensor index, time index).
  4) Measures energy fractions inside:
       (a) bow-tie (planar range):            |omega|/c >= |k_s|
       (b) FLAT temporal bandlimit:           |omega| <= c*pi/dx_s         (often too crude)
       (c) CURVED grid cap (2D):              |omega| <= c*sqrt(k_s^2 + (pi/dx_perp)^2)
       (d) intersections: bowtie ∩ flat, bowtie ∩ curved-cap
  5) Plots spectrum with overlaid boundaries and mask visualisations.

Important:
  - Robust to sensor_data returned as (Ns,Nt) or (Nt,Ns).
  - The sensor mask used below is a *vertical* line at x=0 (all y). That means:
        sensor coordinate s ~ y  -> dx_s = dx[0]
        perpendicular coord ⟂ ~ x -> dx_perp = dx[1]
    If you instead want a horizontal line at y=0 (all x), change the mask and swap dx_s/dx_perp.

Usage:
  python3 this_script.py

Toggle CFL below to reproduce your Nt=~2N+1 vs Nt=~4N cases.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from beamax.solvers import KWaveSolver
from beamax.geometry import Domain
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions


# -----------------------------
# Experiment configuration
# -----------------------------
d = 2
N = (128,) * d
dx = (1e-4,) * d
periodic = (True,) * d

C0 = 1500.0

# Choose ONE:
CFL = float((jnp.sqrt(2) / 2).round(3))  # ~0.707  -> typically Nt ~ 2N+1
# CFL = float((jnp.sqrt(2) / 4).round(3)) # ~0.354  -> typically Nt ~ 4N

USE_TIME_HANN = False  # reduce leakage in omega
USE_GAUSSIAN_P0 = False  # smoother initial condition
GAUSS_SIGMA_PX = 2.0  # stddev in grid points

SHOW_PLOTS = True
SAVE_PLOTS = False
OUT_PREFIX = "bowtie_support"


def c_hom(x):
    return C0 + 0.0 * x[..., 0]


def unitary_fft(arr: jnp.ndarray) -> jnp.ndarray:
    """Unitary N-D FFT with centred zero-frequency component."""
    return jnp.fft.fftshift(jnp.fft.fftn(arr, norm="ortho"))


def make_p0(N, use_gaussian=False, sigma_px=2.0):
    ny, nx = N
    cy, cx = ny // 2, nx // 2

    if not use_gaussian:
        return jnp.zeros(N).at[cy, cx].set(1.0)

    yy = jnp.arange(ny)
    xx = jnp.arange(nx)
    Y, X = jnp.meshgrid(yy, xx, indexing="ij")
    sig2 = sigma_px**2
    g = jnp.exp(-0.5 * (((Y - cy) ** 2 + (X - cx) ** 2) / sig2))
    g = g / jnp.max(g)
    return g


def make_line_sensor_mask_vertical_x0(N):
    """
    Vertical line at x=0 (all y). This matches your earlier code: mask.at[:,0]=1.
    """
    ny, nx = N
    mask = jnp.zeros((ny, nx))
    mask = mask.at[0, :].set(1.0)
    return mask


def ensure_ns_nt(sensor_data, ts):
    """
    Ensure array is (Ns, Nt), robust to (Ns,Nt) vs (Nt,Ns) returns.
    """
    Nt = len(ts)
    arr = sensor_data
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D sensor data, got {arr.shape}")

    if arr.shape[1] == Nt:
        return arr
    if arr.shape[0] == Nt:
        return arr.T

    raise ValueError(
        f"Cannot infer time axis. sensor_data.shape={arr.shape}, len(ts)={Nt}"
    )


def shifted_axes(Ns, Nt, dx_s, dt):
    """Shifted k_s (rad/m) and omega (rad/s) axes matching fftshifted FFT output."""
    ks = 2 * jnp.pi * jnp.fft.fftfreq(Ns, d=dx_s)
    om = 2 * jnp.pi * jnp.fft.fftfreq(Nt, d=dt)
    return jnp.fft.fftshift(ks), jnp.fft.fftshift(om)


def compute_power_and_masks(arr_ns_nt, dx_s, dx_perp, dt, c=C0, use_time_hann=False):
    """
    Compute FFT power P=|FFT|^2 and support masks.
    """
    Ns, Nt = arr_ns_nt.shape

    arr = arr_ns_nt
    if use_time_hann:
        w = jnp.hanning(Nt)
        arr = arr * w[None, :]

    G = unitary_fft(arr)
    P = jnp.abs(G) ** 2

    ks, om = shifted_axes(Ns, Nt, dx_s=dx_s, dt=dt)

    # (a) Bow-tie lower constraint: |omega|/c >= |k_s|
    mask_bowtie = jnp.abs(om)[None, :] / c >= jnp.abs(ks)[:, None]

    # (b) Flat temporal bandlimit (crude): |omega| <= c*pi/dx_s
    omega_flat_max = c * np.pi / dx_s
    mask_flat = jnp.abs(om)[None, :] <= omega_flat_max

    # (c) Curved grid cap (2D): |omega| <= c*sqrt(k_s^2 + (pi/dx_perp)^2)
    k_perp_max = np.pi / dx_perp
    omega_cap = c * jnp.sqrt(ks[:, None] ** 2 + (k_perp_max**2))
    mask_cap = jnp.abs(om)[None, :] <= omega_cap

    # Intersections
    mask_bowtie_flat = mask_bowtie & mask_flat
    mask_bowtie_cap = mask_bowtie & mask_cap

    total = jnp.sum(P)

    def frac(mask):
        return float(jnp.sum(P * mask) / total)

    out = {
        "ks": ks,
        "omega": om,
        "P": P,
        "omega_flat_max": float(omega_flat_max),
        "k_perp_max": float(k_perp_max),
        "mask_bowtie": mask_bowtie,
        "mask_flat": mask_flat,
        "mask_cap": mask_cap,
        "mask_bowtie_flat": mask_bowtie_flat,
        "mask_bowtie_cap": mask_bowtie_cap,
        "frac_bowtie": frac(mask_bowtie),
        "frac_flat": frac(mask_flat),
        "frac_cap": frac(mask_cap),
        "frac_bowtie_flat": frac(mask_bowtie_flat),
        "frac_bowtie_cap": frac(mask_bowtie_cap),
    }
    return out


def plot_results(out, title_suffix=""):
    ks = np.array(out["ks"])
    om = np.array(out["omega"])
    P = np.array(out["P"])
    L = np.log1p(P)

    extent = [ks[0], ks[-1], om[0], om[-1]]

    # Spectrum
    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(L.T, origin="lower", aspect="auto", extent=extent)
    plt.colorbar(im, ax=ax)
    ax.set_title(f"log(1+|FFT|^2) in (k_s, omega) {title_suffix}")
    ax.set_xlabel("k_s (rad/m)")
    ax.set_ylabel("omega (rad/s)")

    # Overlays:
    # Bowtie boundary: omega = ± c |k_s|
    ks_line = np.linspace(ks[0], ks[-1], 2000)
    omega_bow = C0 * np.abs(ks_line)
    ax.plot(ks_line, +omega_bow, linewidth=1.0)
    ax.plot(ks_line, -omega_bow, linewidth=1.0)

    # Flat bandlimit: omega = ± c*pi/dx_s
    omega_flat_max = out["omega_flat_max"]
    ax.axhline(+omega_flat_max, linewidth=1.0)
    ax.axhline(-omega_flat_max, linewidth=1.0)

    # Curved cap: omega = ± c*sqrt(k_s^2 + (pi/dx_perp)^2)
    k_perp_max = out["k_perp_max"]
    omega_cap = C0 * np.sqrt(ks_line**2 + k_perp_max**2)
    ax.plot(ks_line, +omega_cap, linewidth=1.0)
    ax.plot(ks_line, -omega_cap, linewidth=1.0)

    # Masks
    for name, mask in [
        ("bowtie mask", out["mask_bowtie"]),
        ("flat mask", out["mask_flat"]),
        ("curved cap mask", out["mask_cap"]),
        ("bowtie ∩ flat", out["mask_bowtie_flat"]),
        ("bowtie ∩ cap", out["mask_bowtie_cap"]),
    ]:
        plt.figure()
        ax2 = plt.gca()
        mm = np.array(mask).astype(np.float32)
        im2 = ax2.imshow(
            mm.T, origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1
        )
        plt.colorbar(im2, ax=ax2)
        ax2.set_title(f"{name} {title_suffix}")
        ax2.set_xlabel("k_s (rad/m)")
        ax2.set_ylabel("omega (rad/s)")

    return fig


def main():
    # Domain + time grid
    domain = Domain(N, dx, periodic, CFL, c_hom)
    ts = domain.generate_time_domain()
    dt = float(ts[1] - ts[0])

    # Initial condition
    p0 = make_p0(N, use_gaussian=USE_GAUSSIAN_P0, sigma_px=GAUSS_SIGMA_PX)

    # Sensor mask: vertical line at x=0 (all y)
    binary_mask = make_line_sensor_mask_vertical_x0(N)

    # k-Wave config
    simulation_options = SimulationOptions(
        data_cast="double",
        smooth_p0=False,
        save_to_disk=True,
    )
    execution_options = SimulationExecutionOptions(
        is_gpu_simulation=False,
        delete_data=False,
        verbose_level=0,
        show_sim_log=False,
    )
    kwave_solver = KWaveSolver(simulation_options, execution_options)

    # Forward solve
    sensor_data_kw = kwave_solver.forward(p0, domain, binary_mask, ts)

    # Prints
    print("len(ts) =", len(ts))
    print("dt =", dt)
    print("sensor_data_kw.shape =", tuple(sensor_data_kw.shape))
    print("sensor_data_kw dtype =", sensor_data_kw.dtype)

    # Ensure (Ns, Nt)
    arr = ensure_ns_nt(sensor_data_kw, ts)
    Ns, Nt = arr.shape
    print("interpreting data as (Ns, Nt) =", (Ns, Nt))

    # IMPORTANT: For vertical line at x=0, sensor coordinate is y.
    dx_s = float(dx[0])
    dx_perp = float(dx[1])

    cv = C0 * dt / dx_s
    print("cv = c*dt/dx_s =", cv)

    out = compute_power_and_masks(
        arr_ns_nt=arr,
        dx_s=dx_s,
        dx_perp=dx_perp,
        dt=dt,
        c=C0,
        use_time_hann=USE_TIME_HANN,
    )

    print("energy in bowtie |omega|/c >= |k_s|:", out["frac_bowtie"])
    print("energy in flat band |omega| <= c*pi/dx_s:", out["frac_flat"])
    print(
        "energy in curved cap |omega| <= c*sqrt(k_s^2 + (pi/dx_perp)^2):",
        out["frac_cap"],
    )
    print("energy in bowtie ∩ flat:", out["frac_bowtie_flat"])
    print("energy in bowtie ∩ curved-cap:", out["frac_bowtie_cap"])

    title_suffix = (
        f"(Ns={Ns}, Nt={Nt}, CFL={CFL:.3f}, cv={cv:.3f}, "
        f"hann={USE_TIME_HANN}, gauss={USE_GAUSSIAN_P0})"
    )
    plot_results(out, title_suffix=title_suffix)

    if SAVE_PLOTS:
        for i, f in enumerate(list(map(plt.figure, plt.get_fignums()))):
            fname = f"{OUT_PREFIX}_{i:02d}.png"
            f.savefig(fname, dpi=150, bbox_inches="tight")
            print("saved", fname)

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
