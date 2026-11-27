"""Compare MSGB and k-Wave time reversal images from one detector line.

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
from beamax.surfaces import DetectorSurface

jax.config.update("jax_enable_x64", False)


def timed(label, solve, *args, **kwargs):
    """Include JIT compilation on first use and wait for each result on the host."""
    for run in ("first call", "repeat"):
        start = perf_counter()
        result = np.asarray(solve(*args, **kwargs))
        print(f"{label} ({run}): {perf_counter() - start:.3f} s")
    return result


def main():
    # %% Record the pressure from two sources
    domain = Domain(N=(48, 48), dx=(1e-4, 1e-4), c=1500.0, periodic=(False, False))
    x, y = domain.generate_meshgrid()[0]
    lx, ly = domain.grid_size
    p0 = jnp.exp(-((x / lx - 0.38) ** 2 + (y / ly - 0.45) ** 2) / (2 * 0.08**2))
    p0 -= 0.7 * jnp.exp(-((x / lx - 0.62) ** 2 + (y / ly - 0.58) ** 2) / (2 * 0.09**2))
    p0 = p0 - jnp.mean(p0)
    p0 = p0 / jnp.max(jnp.abs(p0))
    detectors = DetectorSurface.plane(
        2, axis=0, value=0.0, shape=(48,), spacing=(domain.dx[1],), front=1
    )
    # A 192 × 48 record fits the data transform directly.
    ts = domain.generate_time_domain()[:192]
    kwave = KWaveSolver(
        backend="python", device="cpu", pml_size=6, smooth_p0=False, quiet=True
    )
    jax.block_until_ready((p0, ts))
    print(f"MSGB device: {jax.default_backend()} | k-Wave device: cpu")
    data = timed(
        "k-Wave acquisition", kwave.forward, p0, domain, detectors=detectors, ts=ts
    )

    # %% Apply time reversal to the same recordings
    tiles = DyadicDecomposition(2, data.shape, (4, 8), (4, 1))
    data_wpt = MSWPT(tiles, redundancy=2, windowing="rectangular_mirror")
    msgb = MSGBSolver(
        thr=512,
        thr_strat="top_n",
        batch_size=64,
        input_type="spatial",
        ode_solver=solve_hom_diag,
        sum_method="scan_real",
    )
    jax.block_until_ready(data_wpt)
    beams = timed(
        "MSGB time reversal",
        msgb.time_reversal,
        data,
        domain,
        detectors=detectors,
        ts=ts,
        data_wpt=data_wpt,
    )
    reference = timed(
        "k-Wave time reversal",
        kwave.time_reversal,
        data,
        domain,
        detectors=detectors,
        ts=ts,
    )

    # %% Compare reconstructed source locations
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")
    extent = 1e3 * (np.array([0, lx, 0, ly]) - 0.5 * np.repeat(domain.dx, 2))
    for ax, title, field in zip(
        axes,
        ["Initial pressure", "k-Wave time reversal", "MSGB time reversal"],
        [p0, reference, beams],
    ):
        image = np.asarray(field).real
        # Use each image's peak magnitude to compare shapes and preserve polarity.
        limit = np.abs(image).max()
        ax.imshow(
            image.T,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        ax.set(title=title, xlabel="x [mm]")
    axes[0].set_ylabel("y [mm]")
    out_dir = Path("plots/reconstruction")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "2d_time_reversal.png", dpi=150)
    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
