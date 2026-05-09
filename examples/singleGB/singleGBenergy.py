#!/usr/bin/env python3
"""
Track energy conservation along a single Gaussian beam trajectory.
"""
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
import numpy as np

from beamax.geometry import Domain
from beamax.gb import gb_solvers, core, gb_utils

# ---------- Config (keep it simple) ----------
jax.config.update("jax_enable_x64", True)

d = 2  # dimension (use 2D to avoid trivialities)
N = (256, 256)  # grid points
dx = tuple(1.0 / n for n in N)  # physical spacings so domain size = (1,1)
periodic = (True, True)  # periodic so the grid integral is clean


def c(x):
    return 1.0 + 0.0 * x[..., 0]  # homogeneous medium


rho = 1.0  # density (only for optional physical energy)
lam = 0.0  # no absorption
Nt = 200
tmax = 0.3
ts = jnp.linspace(0.0, tmax, Nt)

domain = Domain(N=N, dx=dx, c=c, periodic=periodic)

# ---------- Single Gaussian beam parameters ----------
b = 1
mode = jnp.ones((b,))  # + branch
x0 = jnp.array([[0.5, 0.5]])  # center of box (physical coords)
p0 = jnp.array([[1.0, 0.0]])  # pointing along +x
p0 = p0 / jnp.linalg.norm(p0, axis=-1, keepdims=True)

a0 = jnp.ones((b,))  # initial amplitude

# Beam width via diagonal Im(M0) = diag(beta1, beta2) with beta>0
beta1, beta2 = 50.0, 20.0  # larger -> narrower beam
alpha0 = 1j * jnp.array([[beta1, beta2]])  # (b,d) imaginary diagonals
M0 = gb_utils.prepare_M0(alpha0, None)  # build (b,d,d) symmetric M0

# Carrier (any positive omega). Choose moderate so envelope is resolved.
omega0 = jnp.array([80.0])  # (b,)

# ---------- Evolve GB once (homogeneous closed-form solver) ----------
xt, pt, Mt, At = gb_solvers.solve_hom_diag(
    x0=x0, p0=p0, M0=M0, a0=a0, mode=mode, ts=ts, c=c
)  # xt:(b,Nt,d), pt:(b,Nt,d), Mt:(b,Nt,d,d), At:(b,Nt,1)


# ---------- Closed-form phase-averaged L2 energy of one beam ----------
def beam_l2_energy_per_t(omega, Mt, At):
    """
    E_b(t) = 2 |A|^2 π^{d/2} ω^{-d/2} / sqrt(det(Im M(t)))   (phase-averaged; no cross terms)
    """
    d_local = Mt.shape[-1]
    # |A|^2 -> (b,Nt)
    A2 = jnp.square(jnp.abs(At).reshape(At.shape[0], At.shape[1]))
    # det(Im M) -> (b,Nt)
    Mim = (Mt - Mt.conj()) / (2j)
    Mim = 0.5 * (Mim + jnp.swapaxes(Mim, -1, -2))  # symmetrize for numerical stability
    det_im = jnp.linalg.det(Mim).real
    pref = 2.0 * (jnp.pi ** (0.5 * d_local))
    omega_fac = jnp.power(omega[:, None], -0.5 * d_local)
    Eb_t = pref * A2 * omega_fac / jnp.sqrt(det_im + 1e-30)  # (b,Nt)
    return jnp.sum(Eb_t, axis=0)  # sum over beams -> (Nt,)


E_t = beam_l2_energy_per_t(omega0, Mt, At)  # (Nt,)

# ---------- Optional: numeric grid check (discrete integral of u^2) ----------
# Evaluate u on the full spatial grid using your real-beam synthesizer.
# Sensors := full grid in physical coords
spatial_mesh, _ = domain.generate_meshgrid()  # tuple of arrays
XY = jnp.stack(spatial_mesh, axis=-1)  # (*N, d)

u_ts = core.compute_gaussian_beam_real(
    x0=x0,
    p0=p0,
    M0=M0,
    a0=a0,
    ω0=omega0,
    mode=mode,
    c=c,
    lam=lam,
    ts=ts,
    sensors=XY,
    domain_size=domain.grid_size,
    periodic=jnp.array(periodic),
    ode_solver=gb_solvers.solve_hom_diag,
    solver_config=None,
)  # (Nt, *N)

dx_prod = float(np.prod(np.array(dx)))
E_num = jnp.sum(jnp.square(u_ts), axis=tuple(range(1, u_ts.ndim))) * dx_prod  # (Nt,)

# ---------- Print quick diagnostics ----------
E0, E1 = float(E_t[0]), float(E_t[-1])
En0, En1 = float(E_num[0]), float(E_num[-1])

print(
    f"Closed-form  L2 energy: E(0)={E0:.6e}, E(T)={E1:.6e}, drift={(E1 / E0 - 1):+.2e}"
)
print(
    f"Discrete-grid L2 energy: En(0)={En0:.6e}, En(T)={En1:.6e}, drift={(En1 / En0 - 1):+.2e}"
)
print(f"Closed-form vs grid at t=0: rel.err={(E0 - En0) / E0:+.2e}")

# ---------- (Optional) physical energy scaling ----------
# Time-averaged total acoustic energy ≈ ∫ p^2/(ρ c^2) dx ≈ E_L2 / (ρ c^2)
c0 = float(c(jnp.zeros((d,))))
E_phys = E_t / (rho * c0**2)
print(f"Physical energy proxy (avg total) at t=0: {float(E_phys[0]):.6e}")

import matplotlib.pyplot as plt


# --- helper: simple time-average over K frames (rectangular window)
def moving_avg(x, K):
    if K <= 1:
        return x
    # pad at both ends so the averaged series aligns in length
    pad = (K - 1) // 2
    xpad = jnp.pad(x, (pad, K - 1 - pad), mode="edge")
    c = jnp.convolve(xpad, jnp.ones(K) / K, mode="valid")
    return c


# choose an averaging window ≈ a few carrier periods
# carrier period in *time index* units ≈ 2π / (ω * dt * |∂Φ/∂t|). In this setup,
# a pragmatic choice is just K ~ 10–20; increase if ω is small.
K = 16
E_num_avg = moving_avg(E_num, K)

# --- plot
plt.figure(figsize=(7.5, 4.5))
t = np.array(ts)
plt.plot(t, np.array(E_t), lw=2, label="Closed-form (phase-avg)", zorder=3)
plt.plot(t, np.array(E_num), lw=1, alpha=0.6, label="Grid instantaneous $\\int u^2 dx$")
plt.plot(t, np.array(E_num_avg), lw=2, linestyle="--", label=f"Grid, {K}-step time-avg")
plt.xlabel("t")
plt.ylabel("$\\int u^2\\,dx$  (arbitrary units)")
plt.title("Gaussian beam L² energy: closed-form vs instantaneous vs time-avg")
plt.legend()
plt.tight_layout()
plt.show()
