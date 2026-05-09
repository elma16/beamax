#!/usr/bin/env python3
# coding: utf-8
"""
TEST3D_SUPPORT_V4

Goal:
  Diagnose why the 3D continuum cone test fails at CFL ~ sqrt(3)/2.

Main hypothesis:
  Temporal aliasing: omega_N = pi/dt is too small to represent omega_max ~ c*sqrt(3)*pi/dx,
  so high-k content folds to low omega and violates |omega|/c >= ||k_parallel||.

This script sweeps CFL values and prints:
  - cv = c*dt/dx
  - omega_N and omega_max estimate
  - energy fractions in cone, cap, and cone∩cap

Expect:
  For cv <= 1/sqrt(3) ~ 0.577, cone fraction should rise dramatically.
"""

import numpy as np
import jax.numpy as jnp

from beamax.solvers import KWaveSolver
from beamax.geometry import Domain
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions


# -----------------------------
# Config
# -----------------------------
N = (64, 64, 64)  # (Nx, Ny, Nz)
dx = (1e-4, 1e-4, 1e-4)  # (dx_x, dx_y, dx_z)
periodic = (True, True, True)

C0 = 1500.0
USE_TIME_HANN = True

# Sweep CFL values. Include your current ~0.866 and values below 1/sqrt(3) ~ 0.577.
CFL_LIST = [
    float((jnp.sqrt(3) / 4).round(3)),  # ~0.866 (your current)
    0.65,
    0.58,
    0.55,
    0.50,
    0.40,
]

# Reuse solver options across runs
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


def c_hom(x):
    return C0 + 0.0 * x[..., 0]


def unitary_fft(arr: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.fftshift(jnp.fft.fftn(arr, norm="ortho"))


def make_p0_delta(N):
    Nx, Ny, Nz = N
    return jnp.zeros(N).at[Nx // 2, Ny // 2, Nz // 2].set(1.0)


def make_plane_mask_x0(N):
    Nx, Ny, Nz = N
    mask = jnp.zeros((Nx, Ny, Nz))
    return mask.at[0, :, :].set(1.0)


def ensure_ns_nt(sensor_data, ts):
    Nt = len(ts)
    arr = sensor_data
    if arr.shape[1] == Nt:
        return arr
    if arr.shape[0] == Nt:
        return arr.T
    raise ValueError(f"Cannot infer time axis: shape={arr.shape}, Nt={Nt}")


def shifted_axes(Ny, Nz, Nt, dy, dz, dt):
    ky = 2 * jnp.pi * jnp.fft.fftfreq(Ny, d=dy)
    kz = 2 * jnp.pi * jnp.fft.fftfreq(Nz, d=dz)
    om = 2 * jnp.pi * jnp.fft.fftfreq(Nt, d=dt)
    return jnp.fft.fftshift(ky), jnp.fft.fftshift(kz), jnp.fft.fftshift(om)


def analyze_plane_fft(sensor_data, ts, Ny, Nz, dx_x, dx_y, dx_z, c=C0):
    """
    sensor_data: returned by solver.forward, shape (Nt,Ns) or (Ns,Nt) where Ns=Ny*Nz.
    Returns cone/cap fractions for continuum theory in (k_parallel, omega).
    """
    dt = float(ts[1] - ts[0])
    arr_ns_nt = ensure_ns_nt(sensor_data, ts)  # (Ns,Nt)
    Ns, Nt = arr_ns_nt.shape
    if Ns != Ny * Nz:
        raise ValueError(f"Expected Ns=Ny*Nz={Ny * Nz}, got {Ns}")

    # IMPORTANT: we don't need perfect y-z ordering to test aliasing vs Nyquist
    # if the spectrum is roughly isotropic, but we do need *some* consistent 2D grid.
    # We'll just use a C-order reshape; the key diagnostic here is how fractions change with dt.
    arr_yz_t = jnp.asarray(np.array(arr_ns_nt).reshape((Ny, Nz, Nt), order="C"))

    if USE_TIME_HANN:
        w = jnp.hanning(Nt)
        arr_yz_t = arr_yz_t * w[None, None, :]

    G = unitary_fft(arr_yz_t)
    P = jnp.abs(G) ** 2

    ky, kz, om = shifted_axes(Ny, Nz, Nt, dx_y, dx_z, dt)
    KY, KZ = jnp.meshgrid(ky, kz, indexing="ij")
    KPAR = jnp.sqrt(KY**2 + KZ**2)[:, :, None]  # (Ny,Nz,1)
    OM = jnp.abs(om)[None, None, :]  # (1,1,Nt)

    # Cone: |omega|/c >= ||k_parallel||
    mask_cone = OM / c >= KPAR

    # Curved cap: |omega| <= c*sqrt(||k_parallel||^2 + (pi/dx_x)^2)
    kx_max = np.pi / dx_x
    omega_cap = c * jnp.sqrt(KPAR**2 + (kx_max**2))
    mask_cap = OM <= omega_cap

    mask_phys = mask_cone & mask_cap

    total = jnp.sum(P)
    frac_cone = float(jnp.sum(P * mask_cone) / total)
    frac_cap = float(jnp.sum(P * mask_cap) / total)
    frac_phys = float(jnp.sum(P * mask_phys) / total)

    return dt, Nt, frac_cone, frac_cap, frac_phys


def main():
    print("\n==============================")
    print("TEST3D_SUPPORT_V4  (must appear)")
    print("==============================\n")

    Nx, Ny, Nz = N
    dx_x, dx_y, dx_z = map(float, dx)

    # Spatial max omega estimate at grid Nyquist in 3D:
    # omega_max ~ c * sqrt( (pi/dx_x)^2 + (pi/dx_y)^2 + (pi/dx_z)^2 )
    kxN = np.pi / dx_x
    kyN = np.pi / dx_y
    kzN = np.pi / dx_z
    omega_max_est = C0 * np.sqrt(kxN**2 + kyN**2 + kzN**2)

    solver = KWaveSolver(simulation_options, execution_options)
    p0 = make_p0_delta(N)
    mask = make_plane_mask_x0(N)

    print(f"Grid omega_max_est ≈ {omega_max_est:.3e} rad/s")
    print(
        f"1/sqrt(3) ≈ {1 / np.sqrt(3):.3f} (target upper bound for c*dt/dx to avoid time aliasing)\n"
    )

    for cfl in CFL_LIST:
        domain = Domain(N, dx, periodic, float(cfl), c_hom)
        ts = domain.generate_time_domain()

        sensor_data = solver.forward(p0, domain, mask, ts)

        dt, Nt, frac_cone, frac_cap, frac_phys = analyze_plane_fft(
            sensor_data=sensor_data,
            ts=ts,
            Ny=Ny,
            Nz=Nz,
            dx_x=dx_x,
            dx_y=dx_y,
            dx_z=dx_z,
            c=C0,
        )

        omega_N = np.pi / dt
        cv = C0 * dt / dx_x  # matches your printed cv_x

        print(f"CFL={cfl:.3f}  Nt={Nt:4d}  dt={dt:.3e}  cv=c*dt/dx={cv:.3f}")
        print(
            f"   omega_N = {omega_N:.3e} rad/s,  omega_max_est = {omega_max_est:.3e} rad/s,  omega_N/omega_max_est = {omega_N / omega_max_est:.3f}"
        )
        print(
            f"   frac_cone = {frac_cone:.6f},  frac_cap = {frac_cap:.6f},  frac_cone∩cap = {frac_phys:.6f}\n"
        )


if __name__ == "__main__":
    main()
