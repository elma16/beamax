#!/usr/bin/env python
# coding: utf-8



"""
Adjoint vs. time-reversal reconstruction on a synthetic box phantom, contrasting the two inverse operators.
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from pathlib import Path
from beamax import geometry, utils
from beamax.solvers.kwave_solver import KWaveSolver
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions

try:
    import tqdm.auto as tqdm
except ModuleNotFoundError:

    class _ProgressFallback:
        def __init__(self, iterable):
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix(self, *args, **kwargs):
            return None

        def set_description(self, *args, **kwargs):
            return None

        def update(self, *args, **kwargs):
            return None

        def close(self):
            return None

    class _TqdmFallback:
        @staticmethod
        def tqdm(iterable, *args, **kwargs):
            return _ProgressFallback(iterable)

        @staticmethod
        def trange(*args, **kwargs):
            return _ProgressFallback(range(*args))

    tqdm = _TqdmFallback()

ROOT_DIR = utils.detect_root()
PLOT_DIR = Path(ROOT_DIR / "plots")
DATA_DIR = Path(ROOT_DIR / "data")
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

"""

Some silly plot inspired by Figure 2 from [1].

What happens to the reconstruction error as we increase the number of boundary sensors?

[1]: Kultima, J., Ramlau, R., Sahlström, T. and Tarvainen, T., 2025. Fast reconstruction approaches for photoacoustic tomography with smoothing Sobolev/Mat\'ern priors. arXiv preprint arXiv:2507.02401.

"""

# ------------------------------------------------------------
# Domain & source
# ------------------------------------------------------------
N = (64, 64)
d = len(N)
dx = (1e-4,) * d
cfl = 0.3
periodic = (False,) * d
p0 = jnp.zeros(N).at[10:30, 30:50].set(1)


def c(x):
    return 1500 + 0 * x[..., 0]


domain = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)
ts = domain.generate_time_domain()

sim_opts = SimulationOptions(data_cast="double", smooth_p0=False, save_to_disk=True)
exec_opts = SimulationExecutionOptions(
    is_gpu_simulation=False, delete_data=False, verbose_level=0, show_sim_log=False
)
solver = KWaveSolver(sim_opts, exec_opts)

# ------------------------------------------------------------
# Boundary‑sensor enumeration
# ------------------------------------------------------------


def boundary(shape):
    h, w = shape
    return (
        [(0, j) for j in range(w)]
        + [(i, w - 1) for i in range(1, h)]
        + [(h - 1, j) for j in range(w - 2, -1, -1)]
        + [(i, 0) for i in range(h - 2, 0, -1)]
    )


coords = boundary(N)
full_examples = os.environ.get("BEAMAX_FULL_EXAMPLES", "0") == "1"
if not full_examples and len(coords) > 24:
    stride = max(1, len(coords) // 24)
    coords = coords[::stride]
    print(
        f"Running reduced sensor sweep with {len(coords)} masks. "
        "Set BEAMAX_FULL_EXAMPLES=1 for full boundary sweep."
    )
mask = jnp.zeros(N)
all_sensors = jnp.ones(N)

masks, tr_recs, adj_recs, ae, re = [], [], [], [], []

for r, c in tqdm.tqdm(coords, desc="Simulating"):
    mask = mask.at[r, c].set(1)
    try:
        meas = solver.forward(p0, domain, mask, ts)
        tr = solver.time_reversal(meas.T, domain, all_sensors, mask, ts).T
        adj = solver.adjoint(meas.T, domain, all_sensors, mask, ts).T
    except Exception:
        tr = adj = jnp.full_like(p0, jnp.nan)
    masks.append(mask.copy())
    tr_recs.append(tr)
    adj_recs.append(adj)
    ae.append(jnp.linalg.norm(p0 - adj))
    re.append(jnp.linalg.norm(p0 - tr))

# thresholds where each edge is fully covered
t1, t2, t3 = N[1], N[1] + N[0] - 1, N[1] + N[0] + N[1] - 2

# ------------------------------------------------------------
# Global colour scaling (shared)
# ------------------------------------------------------------
recon_max = float(jnp.nanmax(jnp.stack(tr_recs + adj_recs + [p0])))
GLOBAL_MAX = recon_max if np.isfinite(recon_max) and recon_max > 0 else 1.0
finite_err = [v for v in ae + re if np.isfinite(v)]
ERR_MAX = max(finite_err) if finite_err else 1

# ------------------------------------------------------------
# Static diagnostic figure
# ------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(masks[-1], cmap="Greys", alpha=0.5)
plt.imshow(p0, cmap="hot", vmin=0, vmax=GLOBAL_MAX, alpha=0.6)
plt.title(f"Sensor mask ({len(coords)} px)")
plt.subplot(1, 2, 2)
plt.plot(ae, "o-", label="Adjoint")
plt.plot(re, "s-", label="Time rev")
for x in (t1, t2, t3):
    plt.axvline(x - 1, ls="--")
plt.ylim(0, ERR_MAX * 1.1)
plt.xlabel("# boundary sensors")
plt.ylabel("error")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("sensor_recon_diagnostics.png", dpi=150)

if not full_examples:
    print("Skipping animation in reduced mode. Set BEAMAX_FULL_EXAMPLES=1 for full run.")
    raise SystemExit(0)

# ------------------------------------------------------------
# Animation: mask | TR | Adjoint | error
# ------------------------------------------------------------
fig, axs = plt.subplots(
    1, 4, figsize=(19, 4), gridspec_kw={"width_ratios": [1, 1, 1, 1.3]}
)

# panel 0: sensor mask + p0 (both use shared scale)
im_mask = axs[0].imshow(masks[0], cmap="Greys", alpha=0.5)
axs[0].imshow(p0, cmap="hot", vmin=0, vmax=GLOBAL_MAX, alpha=0.6)
axs[0].set_title("Sensors 1")

# panels 1 & 2: reconstructions
im_tr = axs[1].imshow(tr_recs[0], cmap="hot", vmin=0, vmax=GLOBAL_MAX)
axs[1].set_title("TR")

im_adj = axs[2].imshow(adj_recs[0], cmap="hot", vmin=0, vmax=GLOBAL_MAX)
axs[2].set_title("Adjoint")

# panel 3: error curves
(line1,) = axs[3].plot([], [], "o-", label="Adjoint err")
(line2,) = axs[3].plot([], [], "s-", label="TR err")
for x in (t1, t2, t3):
    axs[3].axvline(x - 1, ls="--")
axs[3].set_xlim(0, len(coords))
axs[3].set_ylim(0, ERR_MAX * 1.1)
axs[3].set_xlabel("# sensors")
axs[3].set_ylabel("error")
axs[3].legend()

# shared colourbar for all hot‑scaled images
cbar = fig.colorbar(im_tr, ax=axs[:3], location="right", fraction=0.025, pad=0.02)
cbar.set_label("Amplitude", rotation=270, labelpad=15)


def update(i):
    im_mask.set_data(masks[i])
    im_tr.set_data(tr_recs[i])
    im_adj.set_data(adj_recs[i])
    line1.set_data(range(i + 1), ae[: i + 1])
    line2.set_data(range(i + 1), re[: i + 1])
    axs[0].set_title(f"Sensors {i + 1}")
    return im_mask, im_tr, im_adj, line1, line2


ani = animation.FuncAnimation(fig, update, frames=len(coords), interval=100, blit=False)
ani.save(PLOT_DIR / "sensor_recon_animation.mp4", fps=10, dpi=150)
