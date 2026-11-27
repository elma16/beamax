"""Compare MSGB and k-Wave pressure recordings from a 2D wave packet.

Example extras: kwave,viz-mpl
Example smoke: false
"""

# %% Setup
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from beamax import Domain, DyadicDecomposition, MSWPT
from beamax.gb.gb_solvers import solve_hom_diag
from beamax.solvers import KWaveSolver, MSGBSolver
from beamax.utils import rel_l2

jax.config.update("jax_enable_x64", False)


def timed(label, solve, *args, **kwargs):
    """Include JIT compilation on first use and wait for each result on the host."""
    for run in ("first call", "repeat"):
        start = perf_counter()
        result = np.asarray(solve(*args, **kwargs))
        print(f"{label} ({run}): {perf_counter() - start:.3f} s")
    return result


def main():
    # %% Define the source and receiver line
    domain = Domain(N=(48, 48), dx=(1e-4, 1e-4), c=1500.0, periodic=(False, False))
    x, y = domain.generate_meshgrid()[0]
    lx, ly = domain.grid_size
    u, v = x / lx - 0.42, y / ly - 0.42
    p0 = jnp.exp(-(u**2 + v**2) / (2 * 0.12**2)) * jnp.cos(18 * jnp.pi * (u + v))
    p0 = p0 / jnp.max(jnp.abs(p0))
    ts = domain.generate_time_domain()
    receivers = jnp.zeros(domain.N, dtype=bool).at[5, :].set(True)

    # %% Predict the pressure at each receiver
    tiles = DyadicDecomposition(2, domain.N, (4, 8), (1, 1))
    wpt = MSWPT(tiles, redundancy=2, windowing="rectangular_mirror")
    msgb = MSGBSolver(
        thr=512,
        thr_strat="top_n",
        batch_size=64,
        input_type="spatial",
        ode_solver=solve_hom_diag,
        sum_method="scan_real",
    )
    kwave = KWaveSolver(
        backend="python", device="cpu", pml_size=6, smooth_p0=False, quiet=True
    )
    jax.block_until_ready((p0, receivers, ts, wpt))
    print(f"MSGB device: {jax.default_backend()} | k-Wave device: cpu")
    beams = timed(
        "MSGB forward", msgb.forward, p0, domain, detectors=receivers, ts=ts, wpt=wpt
    )
    reference = timed(
        "k-Wave forward", kwave.forward, p0, domain, detectors=receivers, ts=ts
    )
    print(f"{beams.shape[0]} time samples × {beams.shape[1]} receivers")
    print(f"Relative L2 difference: {rel_l2(reference, beams):.2%}")

    # %% Plot the receiver with the strongest k-Wave signal
    receiver = int(np.argmax(np.max(np.abs(reference), axis=0)))
    fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")
    ax.plot(ts * 1e6, reference[:, receiver], label="k-Wave")
    ax.plot(ts * 1e6, beams[:, receiver], "--", label="MSGB")
    ax.set(title=f"Receiver {receiver}", xlabel="Time [µs]", ylabel="Pressure")
    ax.legend()
    out_dir = Path("plots/forward")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "2d_forward_comparison.png", dpi=150)
    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
