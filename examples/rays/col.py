#!/usr/bin/env python
# coding: utf-8

"""
Ray-tracing diagnostic for the Gaussian beam Hamiltonian.
"""
import jax.numpy as jnp
import jax
import diffrax
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

from beamax import utils
from beamax.gb import gb_utils, gb_solvers  # your own package
from beamax.plotter import use_beamax_style

ROOT_DIR = utils.detect_root()
PLOT_DIR = ROOT_DIR / "plots"
DATA_DIR = ROOT_DIR / "data"
PROF_DIR = ROOT_DIR / "profiler"
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
PROF_DIR.mkdir(exist_ok=True)
use_beamax_style()

jax.config.update("jax_enable_x64", True)

# ------------------------------------------------------------------------------
# Utilities: π-style angle ticks and labels on colorbars/axes
# ------------------------------------------------------------------------------

PI = np.pi


def _pi_tick_labels(ticks):
    """Return mathtext labels like -π, -π/2, 0, π/2, ... for the given tick values."""
    labels = []
    for t in ticks:
        # Snap near integer multiples to avoid ugly floats
        # k = np.round(t / PI, 6)
        # Try common fractions
        frac = None
        for den in (1, 2, 3, 4, 6, 8, 12):
            num = np.round(den * t / PI)
            if np.isclose(t, (num * PI) / den, rtol=0, atol=1e-8):
                frac = (int(num), den)
                break
        if frac is None:
            # fallback to numeric with π factor
            labels.append(rf"{t / PI:.2f}$\pi$")
            continue
        num, den = frac
        if num == 0:
            labels.append("0")
            continue
        sign = "-" if num < 0 else ""
        a = abs(num)
        if den == 1:
            core = r"\pi" if a == 1 else rf"{a}\pi"
        else:
            core = rf"\frac{{{a}\pi}}{{{den}}}"
        labels.append(f"${sign}{core}$")
    return labels


def set_colorbar_pi_ticks(cbar, vmin, vmax):
    """Set reasonable π ticks for colorbar in either [-π,π] or [0,2π]."""
    rng = (float(vmin), float(vmax))
    if rng[0] >= -1e-6 and rng[1] <= 2 * PI + 1e-6:
        # [0, 2π] style
        base = np.array([0, PI / 2, PI, 3 * PI / 2, 2 * PI], dtype=float)
    else:
        # default to [-π, π]
        base = np.array([-PI, -PI / 2, 0, PI / 2, PI], dtype=float)
    ticks = base[(base >= rng[0] - 1e-9) & (base <= rng[1] + 1e-9)]
    if ticks.size < 3:  # extend if the range is odd
        extra = np.linspace(rng[0], rng[1], 5)
        ticks = np.unique(np.concatenate([ticks, extra]))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(_pi_tick_labels(ticks))


# ------------------------------------------------------------------------------
# Caustic detection and branch extraction (for 2D rays → 3D space-time curve)
# ------------------------------------------------------------------------------


def jacobian_dets(xt, ts):
    """
    Compute J = det[ ∂x/∂α , ∂x/∂t ], α = ray index (discrete).
    xt: [R, T, 2]  (R=n_rays, T=nt)
    ts: [T]
    Returns:
      J: [R-2, T-1]
      X_mid: [R-2, T-1, 2] aligned with J
    """
    X = np.asarray(xt)
    T = np.asarray(ts)
    dX_dalpha = 0.5 * (X[2:, :, :] - X[:-2, :, :])  # [R-2, T, 2]
    dt = (T[1:] - T[:-1]).reshape(1, -1, 1)  # [1, T-1, 1]
    dX_dt = (X[1:-1, 1:, :] - X[1:-1, :-1, :]) / dt  # [R-2, T-1, 2]
    dX_dalpha = dX_dalpha[:, :-1, :]  # [R-2, T-1, 2]
    J = dX_dalpha[..., 0] * dX_dt[..., 1] - dX_dalpha[..., 1] * dX_dt[..., 0]
    return J, X[1:-1, :-1, :]


def caustic_zero_crossings_alpha(J, X_mid, ts):
    """
    Extract a crisp J=0 set by sign-changes along α for each fixed time.
    Returns arrays of 3D points (x, y, t) and also α-positions (continuous index).
    """
    Jnp = np.asarray(J)
    Xnp = np.asarray(X_mid)
    Tnp = np.asarray(ts[:-1])  # aligned with J columns

    sgn = np.sign(Jnp)
    sgn[sgn == 0] = 1

    xs, ys, zs, alphas = [], [], [], []
    for j in range(Jnp.shape[1]):  # over time
        s = sgn[:, j]
        cross = np.where(s[:-1] * s[1:] < 0)[
            0
        ]  # indices i where sign flips between i and i+1
        for i in cross:
            J1, J2 = Jnp[i, j], Jnp[i + 1, j]
            # linear interpolation factor along α between i and i+1
            s_alpha = -J1 / (J2 - J1) if J2 != J1 else 0.5
            s_alpha = float(np.clip(s_alpha, 0.0, 1.0))
            P = (1.0 - s_alpha) * Xnp[i, j, :] + s_alpha * Xnp[i + 1, j, :]
            xs.append(P[0])
            ys.append(P[1])
            zs.append(Tnp[j])
            alphas.append(i + s_alpha)  # continuous α-position
    if len(xs) == 0:
        return (np.empty((0,)),) * 4
    return np.array(xs), np.array(ys), np.array(zs), np.array(alphas)


def candidate_cusps(J):
    """
    Heuristic 'cusp' flag: |J| small AND gradient magnitude small.
    Returns boolean mask with same shape as J for candidate cusp locations.
    """
    Jnp = np.asarray(J)
    # |J| small threshold via low-quantile
    q = np.quantile(np.abs(Jnp), 0.01)
    small = np.abs(Jnp) <= q
    # finite-diff gradient magnitude
    dJ_da = np.zeros_like(Jnp)
    dJ_dt = np.zeros_like(Jnp)
    dJ_da[1:-1, :] = 0.5 * (Jnp[2:, :] - Jnp[:-2, :])
    dJ_dt[:, 1:-1] = 0.5 * (Jnp[:, 2:] - Jnp[:, :-2])
    grad_norm = np.hypot(dJ_da, dJ_dt)
    gq = np.quantile(grad_norm, 0.1)
    flat = grad_norm <= gq
    return small & flat


def trace_caustic_branches(
    J, X_mid, ts, alpha_pos, x_list, y_list, z_list, alpha_link_thresh=0.75, max_gap=3
):
    """
    Greedy association of J=0 points (per time slice) into continuous branches in (α,t).
    Inputs are the outputs from caustic_zero_crossings_alpha: alpha_pos, x_list, y_list, z_list
    Returns a list of branches; each branch is dict with arrays 't','x','y','alpha'.
    """
    if len(alpha_pos) == 0:
        return []

    # Group points by time index (discretize time back to nearest ts[:-1] index)
    t_vals = np.asarray(z_list)
    Tgrid = np.asarray(ts[:-1])
    time_idx = np.searchsorted(Tgrid, t_vals)
    time_idx = np.clip(time_idx, 0, len(Tgrid) - 1)

    # pack by time step
    per_t = {}
    for a, x, y, j in zip(alpha_pos, x_list, y_list, time_idx):
        per_t.setdefault(j, []).append((a, x, y))

    branches = []  # list of dicts: { 'alpha':[], 'x':[], 'y':[], 't_idx':[] }
    last_seen = []  # how many steps since last attached

    for j in range(Tgrid.shape[0]):
        pts = sorted(per_t.get(j, []), key=lambda u: u[0])  # sort by alpha
        # mark all branches as unassigned this step
        assigned = [False] * len(branches)
        # try to attach points to nearest branch in alpha
        for a, x, y in pts:
            best_k, best_dist = None, None
            for k, br in enumerate(branches):
                if len(br["t_idx"]) == 0 or assigned[k]:
                    continue
                if br["t_idx"][-1] < j - max_gap:  # this branch has gone stale
                    continue
                dist = abs(a - br["alpha"][-1])
                if dist <= alpha_link_thresh and (
                    best_dist is None or dist < best_dist
                ):
                    best_k, best_dist = k, dist
            if best_k is None:
                # start a new branch
                branches.append({"alpha": [a], "x": [x], "y": [y], "t_idx": [j]})
                assigned.append(True)
                last_seen.append(0)
            else:
                # extend existing branch
                branches[best_k]["alpha"].append(a)
                branches[best_k]["x"].append(x)
                branches[best_k]["y"].append(y)
                branches[best_k]["t_idx"].append(j)
                assigned[best_k] = True
                last_seen[best_k] = 0
        # increment gaps
        for k in range(len(branches)):
            if not assigned[k]:
                last_seen[k] += 1

    # convert to arrays and add 't'
    out = []
    Tgrid = np.asarray(ts[:-1])
    for br in branches:
        if len(br["t_idx"]) < 3:
            continue
        idx = np.array(br["t_idx"])
        out.append(
            {
                "alpha": np.array(br["alpha"]),
                "x": np.array(br["x"]),
                "y": np.array(br["y"]),
                "t": Tgrid[idx],
            }
        )
    return out


# ------------------------------------------------------------------------------
# ===============  PART I: Trapping rays (Maxwell fisheye variant)  ============
# ------------------------------------------------------------------------------

# Domain setup
d = 2
lengths = 1
N = jnp.array([128, 128 * lengths])
xmax = jnp.array([1.0, lengths])
x_linspace = [jnp.linspace(0.0, xmax[i], int(N[i])) for i in range(d)]
XY = jnp.stack(jnp.meshgrid(*x_linspace, indexing="ij"), axis=-1)


def c_maxwell_fisheye(x, c0=1.0, center=jnp.array([0.5, 0.5]), R=0.35):
    r2 = jnp.sum((x - center) ** 2, axis=-1)
    return c0 * (1.0 + r2 / (R**2))


# Rays on a circle, tangential initial directions
n_rays = 90
center = jnp.array([0.5, 0.5])
r0 = 0.18
thetas = jnp.linspace(0, 2 * jnp.pi, n_rays, endpoint=False)
start_point = center + r0 * jnp.stack([jnp.cos(thetas), jnp.sin(thetas)], axis=-1)
angles = thetas + jnp.pi / 2
x0 = start_point
p0 = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)

# GB ancillary
mode = jnp.ones((n_rays, 1))
a0 = jnp.ones((n_rays, 1)) * 0.1
alpha0 = jnp.ones((n_rays, d)) * 1j
M0 = gb_utils.prepare_M0(alpha0, None)
lam = 0

# Time
ts = jnp.linspace(0, 1, 200)

# Solver config (kept for consistency)
solver = gb_solvers.solve_ODE_base
solver_config = gb_solvers.SolverConfig(
    solver=diffrax.Tsit5(),
    max_steps=4096,
    rtol=1e-5,
    pcoeff=0.1,
    icoeff=0.3,
    dcoeff=0.0,
)

# Run
xt, pt, mt, at = solver(x0, p0, M0, a0, mode, ts, c_maxwell_fisheye, lam, None)

# Background speed field
c_vals = jax.vmap(jax.vmap(c_maxwell_fisheye))(XY)

# Color by angle
angle_norm = mcolors.Normalize(vmin=float(angles.min()), vmax=float(angles.max()))
colormap = cm.plasma

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
# Velocity map
im1 = ax1.imshow(
    np.asarray(c_vals).T,
    extent=[0, xmax[0], 0, xmax[1]],
    origin="lower",
    cmap="viridis",
    aspect="auto",
)
ax1.set_title("$c(x)$")
ax1.set_xlabel("$x$")
ax1.set_ylabel("$y$")
cb1 = plt.colorbar(im1, ax=ax1, label="Speed of sound")

# Rays
ax2.imshow(
    np.asarray(c_vals).T,
    extent=[0, xmax[0], 0, xmax[1]],
    origin="lower",
    cmap="viridis",
    alpha=0.25,
    aspect="auto",
)
for i in range(n_rays):
    ax2.plot(
        np.asarray(xt[i, :, 0]),
        np.asarray(xt[i, :, 1]),
        color=colormap(angle_norm(float(angles[i]))),
        lw=0.8,
        alpha=0.8,
    )
ax2.set_xlim(0, xmax[0])
ax2.set_ylim(0, xmax[1])
ax2.set_xlabel("$x$")
ax2.set_ylabel("$y$")
ax2.set_title("Trapping Rays (colored by initial angle)")

sm = cm.ScalarMappable(cmap=colormap, norm=angle_norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax2, label="Ray Angle")
set_colorbar_pi_ticks(cbar, float(angles.min()), float(angles.max()))

plt.tight_layout()
plt.savefig(PLOT_DIR / "trapping_rays.png", dpi=300, bbox_inches="tight")
plt.show()

# 3D space-time view (no caustic extraction for this scenario)
fig = plt.figure(figsize=(12, 10))
ax3d = fig.add_subplot(111, projection="3d")
for i in range(n_rays):
    ax3d.plot(
        np.asarray(xt[i, :, 0]),
        np.asarray(xt[i, :, 1]),
        np.asarray(ts),
        color=colormap(angle_norm(float(angles[i]))),
        lw=1,
        alpha=0.8,
    )
ax3d.set(
    xlabel="$x$",
    ylabel="$y$",
    zlabel="$t$",
    xlim=(0, float(xmax[0])),
    ylim=(0, float(xmax[1])),
    zlim=(0, float(ts[-1])),
)
cbar3 = plt.colorbar(sm, ax=ax3d, label="Ray Angle", shrink=0.8)
set_colorbar_pi_ticks(cbar3, float(angles.min()), float(angles.max()))
ax3d.view_init(elev=22, azim=46)
plt.tight_layout()
plt.savefig(PLOT_DIR / "trapping_rays_3d.png", dpi=300, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------------------------
# ===============  PART II: 2D Caustics + 3D Caustic branches ==================
# ------------------------------------------------------------------------------

# Choose: "A2" (fold) or "A3" (cusp-like via aberration)
SCENARIO = "A3"

# Domain
d = 2
lengths = 1.0
N = jnp.array([256, int(256 * lengths)])
xmax = jnp.array([1.0, lengths])
x_linspace = [jnp.linspace(0.0, xmax[i], int(N[i])) for i in range(d)]
XY = jnp.stack(jnp.meshgrid(*x_linspace, indexing="ij"), axis=-1)

# Time
T_final = 1.10
nt = 450
ts = jnp.linspace(0.0, T_final, nt)


# Refractive index lenses → speed c = c0 / n(x)
def n_luneburg_astig(x, center=jnp.array([0.5, 0.5]), Rx=0.33, Ry=0.28):
    dx = x[..., 0] - center[0]
    dy = x[..., 1] - center[1]
    val = 2.0 - (dx / Rx) ** 2 - (dy / Ry) ** 2
    n_core = jnp.sqrt(jnp.maximum(val, 1e-12))
    return jnp.where(val >= 1.0, n_core, 1.0)


def n_luneburg_astig_cubic(x, center=jnp.array([0.5, 0.5]), Rx=0.33, Ry=0.28, C=0.06):
    dx = x[..., 0] - center[0]
    dy = x[..., 1] - center[1]
    val = 2.0 - (dx / Rx) ** 2 - (dy / Ry) ** 2
    n_core = jnp.sqrt(jnp.maximum(val, 1e-12))
    ux, uy = dx / Rx, dy / Ry
    cubic = ux**3 - 3.0 * ux * uy**2  # Re[(ux+iuy)^3]
    aberr = 1.0 + C * cubic
    n_core_ab = jnp.clip(n_core * aberr, 0.4, None)
    return jnp.where(val >= 1.0, n_core_ab, 1.0)


def c_from_n(n_fn, c0=1.0, **kwargs):
    def c_fn(x):
        return c0 / n_fn(x, **kwargs)

    return c_fn


if SCENARIO == "A2":
    c_fn = c_from_n(n_luneburg_astig, c0=1.0, Rx=0.33, Ry=0.28)
    medium_name = "A2_fold_astig_luneburg"
elif SCENARIO == "A3":
    c_fn = c_from_n(n_luneburg_astig_cubic, c0=1.0, Rx=0.33, Ry=0.28, C=0.06)
    medium_name = "A3_cusp_astig_luneburg_cubic"
else:
    raise ValueError("SCENARIO must be 'A2' or 'A3'.")

# Plane-wave like initial data
n_rays = 121
x_line = jnp.linspace(0.15, 0.85, n_rays)
y0 = jnp.full_like(x_line, 0.10)
x0 = jnp.stack([x_line, y0], axis=-1)
p0 = jnp.tile(jnp.array([0.0, 1.0]), (n_rays, 1))

# GB ancillary
mode = jnp.ones((n_rays, 1))
a0 = jnp.ones((n_rays, 1)) * 0.1
alpha0 = jnp.ones((n_rays, d)) * 1j
M0 = gb_utils.prepare_M0(alpha0, None)
lam = 0.0

solver = gb_solvers.solve_ODE_base
solver_config = gb_solvers.SolverConfig(
    solver=diffrax.Tsit5(),
    max_steps=8192,
    rtol=1e-5,
    pcoeff=0.1,
    icoeff=0.3,
    dcoeff=0.0,
)

# Run
xt, pt, mt, at = solver(x0, p0, M0, a0, mode, ts, c_fn, lam, None)

# Background speed field
c_vals = jax.vmap(jax.vmap(c_fn))(XY)
cmin = float(c_vals.min())
cmax = float(c_vals.max())

# Caustic diagnostics
J, X_mid = jacobian_dets(xt, ts)  # [R-2, T-1], [R-2, T-1, 2]
cx, cy, cz, calpha = caustic_zero_crossings_alpha(J, X_mid, ts)
cusp_mask = candidate_cusps(J)

# Greedy branch tracing in (α,t)
branches = trace_caustic_branches(J, X_mid, ts, calpha, cx, cy, cz)

# ---------------- Plot in 2D with overlays ----------------
norm = mcolors.Normalize(vmin=float(x_line.min()), vmax=float(x_line.max()))
cmap = cm.plasma

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
im = ax.imshow(
    np.asarray(c_vals).T,
    extent=[0, xmax[0], 0, xmax[1]],
    origin="lower",
    cmap="viridis",
    alpha=0.35,
    aspect="auto",
)
plt.colorbar(im, ax=ax, label="Speed c(x)")
for i in range(n_rays):
    ax.plot(
        np.asarray(xt[i, :, 0]),
        np.asarray(xt[i, :, 1]),
        color=cmap(norm(float(x_line[i]))),
        lw=0.8,
        alpha=0.9,
    )

# Scatter J=0 points (projected)
if cx.size > 0:
    ax.scatter(cx, cy, s=12, c="black", alpha=0.9, label="caustic (J=0)")
ax.set(
    xlim=(0, xmax[0]),
    ylim=(0, xmax[1]),
    xlabel="x",
    ylabel="y",
    title=f"Rays and Caustic — {SCENARIO}",
)
sm2 = cm.ScalarMappable(norm=norm, cmap=cmap)
sm2.set_array([])
plt.colorbar(sm2, ax=ax, label="initial x on entry line")
if cx.size > 0:
    ax.legend(loc="upper right", frameon=False)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"rays_caustic_{medium_name}.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------------- 3D space-time: rays + 3D caustic branches -------------------

fig = plt.figure(figsize=(12, 10))
ax3d = fig.add_subplot(111, projection="3d")

# Rays
for i in range(n_rays):
    ax3d.plot(
        np.asarray(xt[i, :, 0]),
        np.asarray(xt[i, :, 1]),
        np.asarray(ts),
        color=cmap(norm(float(x_line[i]))),
        lw=0.7,
        alpha=0.7,
    )

# Caustic branches as lines in (x,y,t)
for k, br in enumerate(branches):
    ax3d.plot(br["x"], br["y"], br["t"], lw=2.2, alpha=0.95, color="black")

# Candidate cusp markers (optional, sparse)
if cusp_mask.any():
    intensity, Jt = np.where(cusp_mask)
    # map to positions for a sparse subset
    step = max(1, len(intensity) // 300)
    intensity = intensity[::step]
    Jt = Jt[::step]
    P = np.asarray(X_mid)[intensity, Jt, :]
    ax3d.scatter(
        P[:, 0],
        P[:, 1],
        np.asarray(ts[:-1])[Jt],
        s=18,
        c="crimson",
        alpha=0.8,
        depthshade=False,
        label="candidate cusp",
    )

ax3d.set(
    xlabel="x",
    ylabel="y",
    zlabel="t",
    xlim=(0, float(xmax[0])),
    ylim=(0, float(xmax[1])),
    zlim=(0, float(ts[-1])),
    title="3D Caustic: branches (black) in space-time",
)
sm3 = cm.ScalarMappable(norm=norm, cmap=cmap)
sm3.set_array([])
plt.colorbar(sm3, ax=ax3d, label="initial x", shrink=0.8)
ax3d.view_init(elev=24, azim=50)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"caustic_3d_{medium_name}.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------------- Summary / listing of branches ----------------


def _arc_length(x, y, t):
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)
    return np.sum(np.sqrt(dx * dx + dy * dy + dt * dt))


if len(branches) == 0:
    print("[INFO] No J=0 caustic branches extracted.")
else:
    print(f"[INFO] Extracted {len(branches)} caustic branch(es):")
    for idx, br in enumerate(branches, 1):
        tmin, tmax = float(br["t"][0]), float(br["t"][-1])
        L = _arc_length(br["x"], br["y"], br["t"])
        print(
            f"  • Branch {idx}: samples={len(br['t'])}, "
            f"t∈[{tmin:.3f},{tmax:.3f}], arc≈{L:.4f}"
        )

# ---------------- Angle diagnostics with π labels (if needed) -----------------


def compute_ray_angles(p0=None, xt=None):
    """Return per-ray angles in radians."""
    if p0 is not None:
        v = jnp.asarray(p0)
    elif (xt is not None) and (xt.shape[1] >= 2):
        v = jnp.asarray(xt[:, 1, :] - xt[:, 0, :])
        norms = jnp.linalg.norm(v, axis=1)
        if bool(jnp.any(norms == 0)):
            k = min(5, int(xt.shape[1]) - 1)
            v_alt = jnp.asarray(xt[:, k, :] - xt[:, 0, :])
            v = jnp.where(norms[:, None] == 0, v_alt, v)
    else:
        raise ValueError("Provide p0 or xt with ≥2 time steps.")
    return jnp.arctan2(v[:, 1], v[:, 0])  # [-π, π]


# Example: colorbar with π ticks for angle distribution (plane-wave angles are trivial)
angles = compute_ray_angles(p0=p0)
fig, ax = plt.subplots(figsize=(6, 3))
vals = np.asarray(angles)
ax.hist(vals, bins=24, density=True)
ax.set_xlabel("Angle")
ax.set_ylabel("Density")
# Replace numeric ticks with π labels on x-axis
xticks = ax.get_xticks()
ax.set_xticklabels(_pi_tick_labels(xticks))
ax.set_title("Initial angle distribution (π-styled)")
plt.tight_layout()
plt.savefig(
    PLOT_DIR / f"angle_histogram_{medium_name}.png", dpi=300, bbox_inches="tight"
)
plt.show()
