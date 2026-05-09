#!/usr/bin/env python3
# coding: utf-8
"""
TEST3D_ALIASING_V1

Goal:
  Quantify temporal aliasing as the cause of "cone violation" in 3D planar sensor FFT.

We compare:
  - Coarse run (large dt / high CFL): observed cone-violation fraction in P_coarse(ky,kz,omega)
  - Reference run (small dt / low CFL): spectrum P_ref
  - Predicted coarse spectrum by folding/aliasing P_ref into coarse Nyquist band and binning to omega_coarse grid
    => P_fold

Then we compare:
  observed_violation = mass outside cone in P_coarse
  predicted_violation = mass outside cone in P_fold

If these match well, your cone violation is basically time-aliasing.

Notes:
  - Uses the continuum cone mask: |omega|/c >= ||k_parallel|| with k_parallel=(ky,kz).
  - Uses a Hann window in time by default to reduce leakage (apply consistently to both runs).
"""

import numpy as np
import jax.numpy as jnp

from beamax.solvers import KWaveSolver
from beamax.geometry import Domain
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions


# -----------------------------
# Experiment configuration
# -----------------------------
N = (64, 64, 64)  # (Nx, Ny, Nz)
dx = (1e-4, 1e-4, 1e-4)
periodic = (True, True, True)

C0 = 1500.0

# Pick the two CFL values you want to compare.
CFL_COARSE = 0.866  # the one that shows cone violations
CFL_REF = 0.400  # small dt reference (should have omega_N >= omega_max_est)

USE_TIME_HANN = True

# If you want to speed up, you can reduce N to (48,48,48) etc.


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
    """Return (Ns,Nt) from either (Nt,Ns) or (Ns,Nt)."""
    Nt = len(ts)
    arr = sensor_data
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D sensor_data, got shape={arr.shape}")
    if arr.shape[1] == Nt:
        return arr
    if arr.shape[0] == Nt:
        return arr.T
    raise ValueError(f"Cannot infer time axis: shape={arr.shape}, len(ts)={Nt}")


def shifted_axes(Ny, Nz, Nt, dy, dz, dt):
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    kz = 2 * np.pi * np.fft.fftfreq(Nz, d=dz)
    om = 2 * np.pi * np.fft.fftfreq(Nt, d=dt)
    return np.fft.fftshift(ky), np.fft.fftshift(kz), np.fft.fftshift(om)


def compute_power_plane(sensor_data, ts, Ny, Nz, dy, dz, c=C0, use_time_hann=True):
    """
    Takes flattened plane sensor data (Ns,Nt or Nt,Ns), reshapes to (Ny,Nz,Nt) using C-order,
    computes P=|FFT_yzt|^2 and returns (P, ky, kz, omega, dt).
    """
    dt = float(ts[1] - ts[0])
    arr_ns_nt = np.array(ensure_ns_nt(sensor_data, ts))  # (Ns,Nt) as numpy
    Ns, Nt = arr_ns_nt.shape
    if Ns != Ny * Nz:
        raise ValueError(f"Expected Ns=Ny*Nz={Ny * Nz}, got Ns={Ns}")

    # Reshape plane to (Ny,Nz,Nt). Ordering isn’t the main question here (aliasing is),
    # and your earlier sweep already showed the cone fraction is stable for small CFL.
    arr_yz_t = arr_ns_nt.reshape((Ny, Nz, Nt), order="C")

    if use_time_hann:
        w = np.hanning(Nt).astype(arr_yz_t.dtype)
        arr_yz_t = arr_yz_t * w[None, None, :]

    G = np.fft.fftshift(np.fft.fftn(arr_yz_t, axes=(0, 1, 2), norm="ortho"))
    P = np.abs(G) ** 2

    ky, kz, om = shifted_axes(Ny, Nz, Nt, dy, dz, dt)
    return P, ky, kz, om, dt


def cone_masks(ky, kz, om, c=C0):
    """
    Returns masks on (Ny,Nz,Nt):
      inside_cone: |omega|/c >= ||k_parallel||
      outside_cone: logical negation
    """
    KY, KZ = np.meshgrid(ky, kz, indexing="ij")
    kpar = np.sqrt(KY**2 + KZ**2)[:, :, None]  # (Ny,Nz,1)
    OM = np.abs(om)[None, None, :]  # (1,1,Nt)
    inside = OM / c >= kpar
    outside = ~inside
    return inside, outside


def frac_mass(P, mask):
    total = P.sum()
    return float((P * mask).sum() / total)


def alias_map_omega(omega, dt_coarse):
    """
    Map a frequency omega (rad/s) to its aliased value in [-pi/dt, +pi/dt].
    Using period 2*pi/dt and Nyquist omega_N = pi/dt.
    """
    omega_N = np.pi / dt_coarse
    period = 2.0 * omega_N
    # shift by +omega_N to map to [0, 2*omega_N), mod, then shift back
    return ((omega + omega_N) % period) - omega_N


def nearest_bin_indices(values, grid):
    """
    Map each value to nearest index in sorted grid.
    values: (M,)
    grid: (K,) sorted ascending
    returns idx: (M,) in [0,K-1]
    """
    idx = np.searchsorted(grid, values, side="left")
    idx = np.clip(idx, 0, len(grid) - 1)

    # compare to left neighbor if exists
    left = np.clip(idx - 1, 0, len(grid) - 1)
    choose_left = np.abs(values - grid[left]) <= np.abs(values - grid[idx])
    return np.where(choose_left, left, idx)


def fold_reference_spectrum_to_coarse(P_ref, om_ref, om_coarse, dt_coarse):
    """
    Fold/alias reference spectrum along omega into coarse Nyquist band and bin to om_coarse grid.

    P_ref: (Ny,Nz,Nt_ref)
    om_ref: (Nt_ref,) shifted, sorted
    om_coarse: (Nt_coarse,) shifted, sorted
    dt_coarse: coarse dt

    Returns:
      P_fold: (Ny,Nz,Nt_coarse)
    """
    Ny, Nz, Nt_ref = P_ref.shape
    Nt_coarse = len(om_coarse)
    P_fold = np.zeros((Ny, Nz, Nt_coarse), dtype=np.float64)

    # Map each reference omega bin to an aliased omega, then to a coarse bin index
    om_alias = alias_map_omega(om_ref, dt_coarse)  # (Nt_ref,)
    idx = nearest_bin_indices(om_alias, om_coarse)  # (Nt_ref,)

    # Accumulate each omega-slice
    for j in range(Nt_ref):
        P_fold[:, :, idx[j]] += P_ref[:, :, j]

    return P_fold


def omega_max_estimate(dx_x, dx_y, dx_z, c=C0):
    kxN = np.pi / dx_x
    kyN = np.pi / dx_y
    kzN = np.pi / dx_z
    return c * np.sqrt(kxN**2 + kyN**2 + kzN**2)


def main():
    print("\n==============================")
    print("TEST3D_ALIASING_V1  (must appear)")
    print("==============================\n")

    Nx, Ny, Nz = N
    dx_x, dx_y, dx_z = map(float, dx)

    omega_max_est = omega_max_estimate(dx_x, dx_y, dx_z, c=C0)
    print(f"Grid omega_max_est ≈ {omega_max_est:.3e} rad/s\n")

    # k-Wave solver (reuse object; domain changes per CFL)
    sim_opts = SimulationOptions(data_cast="double", smooth_p0=False, save_to_disk=True)
    exec_opts = SimulationExecutionOptions(
        is_gpu_simulation=False,
        delete_data=False,
        verbose_level=0,
        show_sim_log=False,
    )
    solver = KWaveSolver(sim_opts, exec_opts)

    p0 = make_p0_delta(N)
    mask = make_plane_mask_x0(N)

    # -----------------------------
    # COARSE RUN
    # -----------------------------
    domain_c = Domain(N, dx, periodic, float(CFL_COARSE), c_hom)
    ts_c = domain_c.generate_time_domain()
    dt_c = float(ts_c[1] - ts_c[0])
    omega_N_c = np.pi / dt_c

    print(f"[COARSE] CFL={CFL_COARSE:.3f}  len(ts)={len(ts_c)}  dt={dt_c:.3e}")
    print(
        f"[COARSE] omega_N={omega_N_c:.3e}  omega_N/omega_max_est={omega_N_c / omega_max_est:.3f}\n"
    )

    data_c = solver.forward(p0, domain_c, mask, ts_c)
    P_c, ky_c, kz_c, om_c, dt_c2 = compute_power_plane(
        data_c, ts_c, Ny=Ny, Nz=Nz, dy=dx_y, dz=dx_z, c=C0, use_time_hann=USE_TIME_HANN
    )
    assert abs(dt_c2 - dt_c) < 1e-14

    inside_c, outside_c = cone_masks(ky_c, kz_c, om_c, c=C0)
    obs_violation = frac_mass(P_c, outside_c)
    obs_inside = frac_mass(P_c, inside_c)
    print(f"[COARSE] observed frac inside cone = {obs_inside:.6f}")
    print(f"[COARSE] observed frac OUTSIDE cone = {obs_violation:.6f}\n")

    # -----------------------------
    # REFERENCE RUN
    # -----------------------------
    domain_r = Domain(N, dx, periodic, float(CFL_REF), c_hom)
    ts_r = domain_r.generate_time_domain()
    dt_r = float(ts_r[1] - ts_r[0])
    omega_N_r = np.pi / dt_r

    print(f"[REF]    CFL={CFL_REF:.3f}  len(ts)={len(ts_r)}  dt={dt_r:.3e}")
    print(
        f"[REF]    omega_N={omega_N_r:.3e}  omega_N/omega_max_est={omega_N_r / omega_max_est:.3f}\n"
    )

    data_r = solver.forward(p0, domain_r, mask, ts_r)
    P_r, ky_r, kz_r, om_r, dt_r2 = compute_power_plane(
        data_r, ts_r, Ny=Ny, Nz=Nz, dy=dx_y, dz=dx_z, c=C0, use_time_hann=USE_TIME_HANN
    )
    assert abs(dt_r2 - dt_r) < 1e-14

    inside_r, outside_r = cone_masks(ky_r, kz_r, om_r, c=C0)
    ref_violation = frac_mass(P_r, outside_r)
    print(f"[REF]    frac OUTSIDE cone (should be ~0) = {ref_violation:.6f}")

    # How much REF power lives above coarse Nyquist (i.e., *must* alias if you sampled at dt_c)?
    above = np.abs(om_r) > omega_N_c
    frac_above = float(P_r[:, :, above].sum() / P_r.sum())
    print(f"[REF]    frac power with |omega| > omega_N_coarse = {frac_above:.6f}\n")

    # -----------------------------
    # FOLD/ALIAS reference spectrum to coarse omega grid
    # -----------------------------
    print(
        "[FOLD] Folding reference spectrum into coarse Nyquist band and binning to coarse omega grid..."
    )
    P_fold = fold_reference_spectrum_to_coarse(P_r, om_r, om_c, dt_coarse=dt_c)

    # Now measure predicted cone-violation on folded spectrum using *coarse* cone mask
    pred_violation = frac_mass(P_fold, outside_c)
    pred_inside = frac_mass(P_fold, inside_c)

    print(f"\n[PRED]  predicted frac inside cone (folded ref) = {pred_inside:.6f}")
    print(f"[PRED]  predicted frac OUTSIDE cone (folded ref) = {pred_violation:.6f}\n")

    # -----------------------------
    # Compare observed vs predicted
    # -----------------------------
    abs_err = abs(obs_violation - pred_violation)
    rel_err = abs_err / max(obs_violation, 1e-12)

    print("=== COMPARISON ===")
    print(f"Observed outside-cone (coarse):   {obs_violation:.6f}")
    print(f"Predicted outside-cone (folded):  {pred_violation:.6f}")
    print(f"Absolute error:                  {abs_err:.6f}")
    print(f"Relative error:                  {rel_err:.3%}")

    # Extra sanity: does folding preserve total mass?
    mass_ratio = float(P_fold.sum() / P_r.sum())
    print(
        f"\n[FOLD] total mass ratio P_fold / P_ref = {mass_ratio:.6f}  (should be ~1.0)\n"
    )


if __name__ == "__main__":
    main()
