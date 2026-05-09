#!/usr/bin/env python
# coding: utf-8

"""
Investigate the stiffness of the Gaussian beam ray ODEs.
"""
# # Stiffness probe for Gaussian-beam ODEs.
#
# Computes local Jacobian spectra along (x,p,M) trajectories; estimates κ(t) and stability-limited dt;
# cross-checks Tsit5 vs Kvaerno5 on a representative ray.
#
# Assumptions:
# - G(x,p) = c(x) * ||p|| (isotropic).
# - Analyse (x,p,M) only; amplitude excluded from J.
# - c ∈ C^2; JAX AD ok.
# - M treated as full complex d×d; realified to R^{2d^2}.



import jax
import jax.numpy as jnp
import diffrax
import numpy as np
import matplotlib.pyplot as plt
from beamax import utils

ROOT_DIR = utils.detect_root()
PLOT_DIR = ROOT_DIR / "plots"
DATA_DIR = ROOT_DIR / "data"
PLOT_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)
from beamax.plotter import use_beamax_style
use_beamax_style()
jax.config.update("jax_enable_x64", True)

d = 2
lengths = 1
N = jnp.array([128, 128 * lengths])
xmax = jnp.array([1.0, lengths])
x_linspace = [jnp.linspace(0.0, xmax[i], N[i]) for i in range(d)]
XY = jnp.stack(jnp.meshgrid(*x_linspace, indexing="ij"), axis=-1)


def c_maxwell_fisheye(x, c0=1.0, center=jnp.array([0.5, 0.5]), R=0.35):
    r2 = jnp.sum((x - center) ** 2, axis=-1)
    return c0 * (1.0 + r2 / (R**2))


n_rays = 90
center = jnp.array([0.5, 0.5])
r0 = 0.18
thetas = jnp.linspace(0, 2 * jnp.pi, n_rays, endpoint=False)
start_point = center + r0 * jnp.stack([jnp.cos(thetas), jnp.sin(thetas)], axis=-1)
angles = thetas + jnp.pi / 2
x0 = start_point
p0 = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)


# Initial complex M: narrow, purely imaginary (typical GB initialisation)
# If you already built M0 via beamax.gb.gb_utils.prepare_M0, import and use it instead.
def init_M0(alpha_im=1.0):
    # isotropic imaginary curvature i*alpha*I
    return (1j * alpha_im) * jnp.tile(jnp.eye(d)[None, ...], (n_rays, 1, 1))


M0 = init_M0(alpha_im=1.0)

# Time grid used by your main run
ts = jnp.linspace(0.0, 1.0, 200)


# ---------- Reconstruct RHS f(y) for (x,p,M) ----------
def _norm(p):
    return jnp.sqrt(jnp.dot(p, p) + 1e-30)


def gb_rhs_single(x, p, M, cfun):
    """RHS for one ray: returns (x_dot, p_dot, M_dot)."""
    L = _norm(p)
    c = cfun(x)  # scalar
    # G = c * L

    # derivatives of c
    grad_c = jax.grad(cfun)(x)  # shape (d,)
    hess_c = jax.jacfwd(jax.grad(cfun))(x)  # shape (d,d)

    # Hamiltonian derivatives
    G_x = L * grad_c  # (d,)
    G_xx = L * hess_c  # (d,d)
    p_hat = p / L
    eye = jnp.eye(d, dtype=x.dtype)
    G_p = c * p_hat  # (d,)
    G_pp = c / L * (eye - jnp.outer(p_hat, p_hat))  # (d,d)
    G_xp = jnp.outer(grad_c, p_hat)  # (d,d)

    # Equations
    x_dot = G_p
    p_dot = -G_x
    # M is complex; JAX handles complex algebra. Ensure types align.
    M_dot = -G_xx - M @ G_xp - G_xp.T @ M - M @ G_pp @ M
    return x_dot, p_dot, M_dot


# ---------- Realification helpers ----------
def pack_state(x, p, M):
    """Pack (x,p,M) complex state -> real 1D vector."""
    # x,p are real; M is complex
    xr = jnp.asarray(x).ravel()
    pr = jnp.asarray(p).ravel()
    Mr = jnp.concatenate([jnp.real(M).ravel(), jnp.imag(M).ravel()])
    return jnp.concatenate([xr, pr, Mr])


def unpack_state(y_vec):
    """Unpack real vector -> (x,p,M)."""
    xr = y_vec[:d]
    pr = y_vec[d : 2 * d]
    rem = y_vec[2 * d :]
    Mre = rem[: d * d].reshape(d, d)
    Mim = rem[d * d :].reshape(d, d)
    M = Mre + 1j * Mim
    return xr, pr, M


def f_real(y_vec, t, cfun):
    x, p, M = unpack_state(y_vec)
    x_dot, p_dot, M_dot = gb_rhs_single(x, p, M, cfun)
    return pack_state(x_dot, p_dot, M_dot)


# Jacobian (wrt state) at (y,t). We keep it EAGER (no jit) to avoid eigval jit headaches.
def jacobian_real(y_vec, t, cfun):
    return jax.jacfwd(lambda y: f_real(y, t, cfun))(y_vec)


# ---------- Stiffness metrics ----------
def stiffness_metrics(J):
    # eigenvalues of real Jacobian -> complex λ
    ev = jnp.linalg.eigvals(J)
    re = jnp.real(ev)
    # decay rates are -Re(λ) for λ with Re(λ) < 0
    decays = -re[re < 0.0]
    max_decay = jnp.max(decays) if decays.size > 0 else 0.0
    min_decay = jnp.min(decays) if decays.size > 0 else 0.0
    # κ = max/min over decaying modes (≥1). If <2 decaying modes, define κ=1.
    kappa = jnp.where(decays.size >= 2, max_decay / jnp.maximum(min_decay, 1e-300), 1.0)
    return max_decay, kappa, ev


def stability_limited_dt(max_decay, method_constant=2.8):
    """Crude stability-limited step for explicit RK on negative real axis."""
    return jnp.where(max_decay > 0.0, method_constant / max_decay, jnp.inf)


# ---------- Run your existing solver to get trajectories (x,p,M) ----------
# If you insist on using your msgb solver outputs:
from beamax.gb import gb_solvers

solver = gb_solvers.solve_ODE_base
solver_config = gb_solvers.SolverConfig(
    solver=diffrax.Tsit5(),
    max_steps=4096,
    rtol=1e-5,
    pcoeff=0.1,
    icoeff=0.3,
    dcoeff=0.0,
)

print("Running GB propagation (Tsit5) to obtain trajectories...")
xt, pt, mt, at = solver(
    x0,
    p0,
    M0,
    jnp.ones((n_rays, 1)) * 0.1,
    jnp.ones((n_rays, 1)),
    ts,
    c_maxwell_fisheye,
    0.0,
    None,
)
# xt: (n_rays, T, d), pt: (n_rays, T, d), mt: (n_rays, T, d, d) complex


# ---------- Probe J, κ, and h_stab along trajectories ----------
def probe_over_trajectory(ray_idx_subset=None, time_stride=1):
    if ray_idx_subset is None:
        ray_idx_subset = list(range(n_rays))
    T = ts.shape[0]
    out = {
        "max_decay": np.full((len(ray_idx_subset), T), np.nan, dtype=float),
        "kappa": np.full((len(ray_idx_subset), T), np.nan, dtype=float),
        "h_stab": np.full((len(ray_idx_subset), T), np.nan, dtype=float),
    }
    for i, ridx in enumerate(ray_idx_subset):
        for k in range(0, T, time_stride):
            xk = np.array(xt[ridx, k])
            pk = np.array(pt[ridx, k])
            Mk = np.array(mt[ridx, k])
            yk = pack_state(jnp.array(xk), jnp.array(pk), jnp.array(Mk))
            Jk = np.array(jacobian_real(yk, float(ts[k]), c_maxwell_fisheye))
            max_d, kap, _ = stiffness_metrics(jnp.asarray(Jk))
            hstab = float(stability_limited_dt(max_d))
            out["max_decay"][i, k] = float(max_d)
            out["kappa"][i, k] = float(kap)
            out["h_stab"][i, k] = hstab
    return out


# Sample fewer rays for spectra to keep it cheap
ray_subset = list(np.linspace(0, n_rays - 1, 12, dtype=int))
probe = probe_over_trajectory(ray_subset, time_stride=1)


# ---------- Cross-check: explicit vs L-stable implicit on one ray ----------
def integrate_single_ray_explicit_implicit(ridx=0, rtol=1e-5):
    # initial state from your initial conditions
    y0 = pack_state(x0[ridx], p0[ridx], M0[ridx])

    term = diffrax.ODETerm(lambda t, y, args: f_real(y, t, c_maxwell_fisheye))
    t0, t1 = float(ts[0]), float(ts[-1])
    save_steps = diffrax.SaveAt(steps=True)

    ctrl = diffrax.PIDController(rtol=rtol, atol=rtol * 1e-6)

    # explicit Tsit5
    sol_exp = diffrax.diffeqsolve(
        term,
        diffrax.Tsit5(),
        t0=t0,
        t1=t1,
        dt0=1e-3,
        y0=y0,
        saveat=save_steps,
        stepsize_controller=ctrl,
        max_steps=200000,
    )

    # L-stable implicit (Kvaerno5)
    sol_imp = diffrax.diffeqsolve(
        term,
        diffrax.Kvaerno5(),
        t0=t0,
        t1=t1,
        dt0=1e-3,
        y0=y0,
        saveat=save_steps,
        stepsize_controller=ctrl,
        max_steps=200000,
    )

    # step sizes from internal steps
    ts_exp = np.array(sol_exp.ts)
    ts_imp = np.array(sol_imp.ts)
    dts_exp = np.diff(ts_exp)
    dts_imp = np.diff(ts_imp)

    return {
        "ts_exp": ts_exp,
        "dts_exp": dts_exp,
        "nsteps_exp": len(dts_exp),
        "ts_imp": ts_imp,
        "dts_imp": dts_imp,
        "nsteps_imp": len(dts_imp),
    }


rep = integrate_single_ray_explicit_implicit(ridx=ray_subset[len(ray_subset) // 2])

# ---------- Plots ----------
# 1) Heatmap of max decay across subset
fig1, ax1 = plt.subplots(figsize=(7, 3.2))
im1 = ax1.imshow(
    probe["max_decay"],
    aspect="auto",
    origin="lower",
    extent=[float(ts[0]), float(ts[-1]), 0, len(ray_subset)],
)
ax1.set_xlabel("t")
ax1.set_ylabel("ray index (subset)")
ax1.set_title("max decay rate  max(-Re λ(J))")
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label("rate")
fig1.tight_layout()
fig1.savefig(PLOT_DIR / "gb_stiffness_max_decay.png", dpi=200)

# 2) Heatmap of κ
fig2, ax2 = plt.subplots(figsize=(7, 3.2))
im2 = ax2.imshow(
    probe["kappa"],
    aspect="auto",
    origin="lower",
    extent=[float(ts[0]), float(ts[-1]), 0, len(ray_subset)],
)
ax2.set_xlabel("t")
ax2.set_ylabel("ray index (subset)")
ax2.set_title("stiffness ratio κ(t)")
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label("κ")
fig2.tight_layout()
fig2.savefig(PLOT_DIR / "gb_stiffness_kappa.png", dpi=200)

# 3) Representative ray: h_stab(t) vs actual dt histories
rep_idx = ray_subset[len(ray_subset) // 2]
h_stab_rep = probe["h_stab"][ray_subset.index(rep_idx)]
fig3, ax3 = plt.subplots(figsize=(7, 3.0))
ax3.plot(ts, h_stab_rep, lw=2, label="h_stab(t) ≈ 2.8 / max(-Re λ)")
ax3.plot(rep["ts_exp"][:-1], rep["dts_exp"], lw=1.5, label="Tsit5 dt")
ax3.plot(rep["ts_imp"][:-1], rep["dts_imp"], lw=1.0, label="Kvaerno5 dt")
ax3.set_yscale("log")
ax3.set_xlabel("t")
ax3.set_ylabel("step size")
ax3.set_title("Stability-limited step vs actual steps")
ax3.legend()
fig3.tight_layout()
fig3.savefig(PLOT_DIR / "gb_stiffness_dt_compare.png", dpi=200)

# ---------- Textual summary ----------
# Percentiles of κ and max decay over probed rays/times
valid_mask = np.isfinite(probe["kappa"])
kap = probe["kappa"][valid_mask]
md = probe["max_decay"][np.isfinite(probe["max_decay"])]


def pct(a, q):
    return float(np.nanpercentile(a, q)) if a.size else np.nan


print("\n===== Stiffness summary =====")
print(
    f"κ percentiles (over subset/time): 50%={pct(kap, 50):.2e}, 90%={pct(kap, 90):.2e}, 99%={pct(kap, 99):.2e}"
)
print(
    f"max(-Re λ) percentiles:           50%={pct(md, 50):.2e}, 90%={pct(md, 90):.2e}, 99%={pct(md, 99):.2e}"
)
print(
    f"Representative ray steps: Tsit5 n={rep['nsteps_exp']}, Kvaerno5 n={rep['nsteps_imp']}"
)
print(
    "If Tsit5 dt tracks h_stab(t) << desired accuracy dt, the system is stiff on those intervals."
)
