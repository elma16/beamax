#!/usr/bin/env python
"""
Compare a small 2D MSGB forward solve with a k-Wave boundary-sensor reference.

This optional example is a compact rewrite of the thesis 2D forward comparison:
a smooth pressure field is propagated to one boundary with MSGB and k-Wave, and
the resulting sensor traces are compared in one static figure.

Example category: Forward propagation
Example extras: kwave,viz-mpl
Example smoke: false
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from beamax import utils
from beamax.decomposition import DyadicDecomposition
from beamax.gb import gb_solvers
from beamax.geometry import Domain, Sensor
from beamax.solvers import MSGBSolver
from beamax.transforms import MSWPT


jax.config.update("jax_enable_x64", True)

INSTALL_HINT = 'pip install -e ".[kwave,viz-mpl]"'


def load_kwave_solver():
    """Import k-Wave lazily so base beamax installs can still import this file."""
    try:
        from beamax.solvers import KWaveSolver
    except ImportError as exc:
        print(f"Skipping optional example: k-Wave is not installed ({INSTALL_HINT}).")
        raise SystemExit(0) from exc
    return KWaveSolver


def time_first(data: np.ndarray, nt: int) -> np.ndarray:
    """Return k-Wave sensor data as ``(Nt, Ns)``."""
    arr = np.asarray(data)
    if arr.ndim == 1:
        return arr[:, None] if arr.shape[0] == nt else arr[None, :]
    if arr.shape[0] == nt:
        return arr
    if arr.shape[-1] == nt:
        return np.moveaxis(arr, -1, 0).reshape(nt, -1)
    return arr.reshape(nt, -1)


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def make_initial_pressure(domain: Domain) -> jnp.ndarray:
    """Two smooth Gaussian packets on the computational grid."""
    x, y = jnp.meshgrid(
        jnp.arange(domain.N[0]) * domain.dx[0],
        jnp.arange(domain.N[1]) * domain.dx[1],
        indexing="ij",
    )
    extent_x, extent_y = domain.grid_size
    p0 = jnp.exp(
        -((x - 0.35 * extent_x) ** 2 + (y - 0.45 * extent_y) ** 2)
        / (2.0 * (0.08 * extent_x) ** 2)
    )
    p0 -= 0.75 * jnp.exp(
        -((x - 0.62 * extent_x) ** 2 + (y - 0.58 * extent_y) ** 2)
        / (2.0 * (0.09 * extent_x) ** 2)
    )
    return p0 / jnp.max(jnp.abs(p0))


def main() -> None:
    KWaveSolver = load_kwave_solver()

    n = (32, 32)
    dx = (1.0e-4, 1.0e-4)

    def c(x):
        return 1500.0 + 0.0 * x[..., 0]

    domain = Domain(N=n, dx=dx, c=c, cfl=0.3, periodic=(False, False))
    ts = domain.generate_time_domain()
    p0 = make_initial_pressure(domain)

    sensor_mask = jnp.zeros(domain.N).at[0, :].set(1.0)
    sensors = Sensor(domain, binary_mask=sensor_mask)

    dyadic = DyadicDecomposition(
        num_levels=2,
        N=domain.N,
        num_boxes_levels=(4, 8),
        box_aspect_ratio=(1, 1),
    )
    wpt = MSWPT(dyadic, redundancy=2, windowing="rectangular_mirror")
    msgb = MSGBSolver(
        thr=min(int(wpt.total_coeffs), 1400),
        thr_strat="top_n",
        batch_size=256,
        input_type="spatial",
        ode_solver=gb_solvers.solve_ODE_base,
        sum_method="scan_real",
    )
    msgb_data, _ = msgb.forward(p0, domain, sensors, ts, wpt)
    msgb_data = np.asarray(msgb_data.block_until_ready())

    kwave = KWaveSolver(
        backend="python",
        device="cpu",
        pml_size=8,
        smooth_p0=False,
        debug=False,
    )
    kw_data = time_first(kwave.forward(p0, domain, sensor_mask, ts), len(ts))

    nt = min(msgb_data.shape[0], kw_data.shape[0])
    ns = min(msgb_data.shape[1], kw_data.shape[1])
    msgb_data = msgb_data[:nt, :ns]
    kw_data = kw_data[:nt, :ns]
    diff = msgb_data - kw_data

    print(f"Grid: {domain.N}; sensors: {ns}; time samples: {nt}")
    print(f"MSGB relative L2 vs k-Wave:   {relative_l2(msgb_data, kw_data):.3e}")
    print(f"MSGB relative Linf vs k-Wave: {np.max(np.abs(diff)) / np.max(np.abs(kw_data)):.3e}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    im0 = axes[0, 0].imshow(np.asarray(p0), origin="lower", cmap="viridis")
    axes[0, 0].set_title("initial pressure")
    fig.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(kw_data, origin="lower", aspect="auto")
    axes[0, 1].set_title("k-Wave boundary data")
    axes[0, 1].set_xlabel("sensor index")
    axes[0, 1].set_ylabel("time index")
    fig.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].imshow(diff, origin="lower", aspect="auto", cmap="RdBu_r")
    axes[1, 0].set_title("MSGB - k-Wave")
    axes[1, 0].set_xlabel("sensor index")
    axes[1, 0].set_ylabel("time index")
    fig.colorbar(im2, ax=axes[1, 0])

    sensor_index = ns // 2
    axes[1, 1].plot(np.asarray(ts[:nt]), kw_data[:, sensor_index], label="k-Wave")
    axes[1, 1].plot(np.asarray(ts[:nt]), msgb_data[:, sensor_index], "--", label="MSGB")
    axes[1, 1].set_title(f"middle boundary trace {sensor_index}")
    axes[1, 1].set_xlabel("t")
    axes[1, 1].legend()

    out_dir = Path(utils.detect_root()) / "plots" / "optional"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "2d_forward_kwave_reference.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
