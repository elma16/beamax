#!/usr/bin/env python
"""
Compare a compact 1D MSGB forward solve with a k-Wave strip reference.

This optional example mirrors the thesis forward-comparison workflow on a tiny
problem. It uses a 2D one-line k-Wave strip as a reference for a 1D pressure
profile, compares both solvers against a periodic spectral solution, and saves
one static diagnostic figure.

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


def spectral_solution(p0: jnp.ndarray, c: float, ts: jnp.ndarray, dx: float) -> np.ndarray:
    """Periodic 1D wave-equation solution for zero initial velocity."""
    p0_np = np.asarray(p0)
    t = np.asarray(ts)[:, None]
    k = 2.0 * np.pi * np.fft.fftfreq(p0_np.shape[0], d=float(dx))
    omega = c * np.abs(k)[None, :]
    u_hat = np.fft.fft(p0_np)[None, :] * np.cos(omega * t)
    return np.fft.ifft(u_hat, axis=1).real


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


def main() -> None:
    KWaveSolver = load_kwave_solver()

    n = 128
    domain = Domain(N=(n,), dx=(1.0 / n,), c=1.0, periodic=(True,))
    ts = domain.generate_time_domain()
    x = jnp.arange(n) * domain.dx[0]
    p0 = jnp.exp(-180.0 * (x - 0.35) ** 2) - 0.65 * jnp.exp(-140.0 * (x - 0.65) ** 2)
    p0 = p0 - jnp.mean(p0)

    dyadic = DyadicDecomposition(
        num_levels=2,
        N=domain.N,
        num_boxes_levels=(4, 8),
        box_aspect_ratio=(1,),
    )
    wpt = MSWPT(dyadic, redundancy=2, windowing="rectangular")
    sensors = Sensor(domain, binary_mask=jnp.ones(domain.N))
    msgb = MSGBSolver(
        thr=int(wpt.total_coeffs),
        thr_strat="top_n",
        batch_size=256,
        input_type="spatial",
        ode_solver=gb_solvers.solve_ODE_base,
        sum_method="all_real",
    )
    msgb_data, _ = msgb.forward(p0, domain, sensors, ts, wpt)
    msgb_data = np.asarray(msgb_data.block_until_ready())

    spectral = spectral_solution(p0, c=1.0, ts=ts, dx=domain.dx[0])

    strip_width = 4
    kw_domain = Domain(
        N=(n, strip_width),
        dx=(domain.dx[0], domain.dx[0]),
        c=1.0,
        periodic=(True, True),
    )
    kw_p0 = jnp.repeat(p0[:, None], strip_width, axis=1)
    kw_sensor_mask = jnp.zeros(kw_domain.N).at[:, 0].set(1.0)
    kwave = KWaveSolver(
        backend="python",
        device="cpu",
        pml_size=0,
        smooth_p0=False,
        debug=False,
    )
    kw_data = time_first(kwave.forward(kw_p0, kw_domain, kw_sensor_mask, ts), len(ts))

    nt = min(msgb_data.shape[0], spectral.shape[0], kw_data.shape[0])
    msgb_data = msgb_data[:nt]
    spectral = spectral[:nt]
    kw_data = kw_data[:nt, :n]
    ts_np = np.asarray(ts[:nt])

    print(f"Grid points: {n}; time samples: {nt}")
    print(f"MSGB relative L2 vs spectral:   {relative_l2(msgb_data, spectral):.3e}")
    print(f"k-Wave relative L2 vs spectral: {relative_l2(kw_data, spectral):.3e}")

    trace_index = n // 3
    extent = [0.0, 1.0, float(ts_np[0]), float(ts_np[-1])]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    axes[0, 0].plot(np.asarray(x), np.asarray(p0))
    axes[0, 0].set_title("initial pressure")
    axes[0, 0].set_xlabel("x")

    im0 = axes[0, 1].imshow(kw_data, extent=extent, origin="lower", aspect="auto")
    axes[0, 1].set_title("k-Wave strip reference")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("t")
    fig.colorbar(im0, ax=axes[0, 1])

    im1 = axes[1, 0].imshow(msgb_data - kw_data, extent=extent, origin="lower", aspect="auto", cmap="RdBu_r")
    axes[1, 0].set_title("MSGB - k-Wave")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("t")
    fig.colorbar(im1, ax=axes[1, 0])

    axes[1, 1].plot(ts_np, spectral[:, trace_index], label="spectral")
    axes[1, 1].plot(ts_np, msgb_data[:, trace_index], "--", label="MSGB")
    axes[1, 1].plot(ts_np, kw_data[:, trace_index], ":", label="k-Wave")
    axes[1, 1].set_title(f"trace at x[{trace_index}]")
    axes[1, 1].set_xlabel("t")
    axes[1, 1].legend()

    out_dir = Path(utils.detect_root()) / "plots" / "optional"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "1d_forward_kwave_reference.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
