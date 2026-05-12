#!/usr/bin/env python
"""
Compare lossless and absorbing single-Gaussian-beam propagation with k-Wave.

This optional example keeps the thesis absorption diagnostic small: one 1D
Gaussian beam is propagated with and without viscous damping, while a thin
k-Wave strip gives a qualitative absorbing reference from the same initial
pressure profile.

Example category: Single Gaussian beam diagnostics
Example extras: kwave,viz-mpl
Example smoke: false
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from beamax import utils
from beamax.gb import core, gb_solvers, gb_utils
from beamax.geometry import Domain


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


def c_homogeneous(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 + 0.0 * x[..., 0]


def lam_to_alpha_db(lam: float, c0: float) -> float:
    """Convert the GB damping coefficient to the matching k-Wave dB/cm scale."""
    return float(jnp.log10(jnp.e) * lam / (10.0 * c0))


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


def normalized_rms(u: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(np.asarray(u) ** 2, axis=1))
    return rms / max(float(rms[0]), np.finfo(float).eps)


def gaussian_beam_pair(domain: Domain, ts: jnp.ndarray, lam: float) -> jnp.ndarray:
    """Evaluate a right/left mode pair so the real initial field is visible."""
    b, d = 1, 1
    x0 = jnp.array([[0.35 * domain.grid_size[0]]])
    p0 = jnp.ones((b, d))
    mode = jnp.ones((b,))
    a0 = jnp.ones((b,))
    omega0 = jnp.ones((b,)) * 70.0
    alpha0 = jnp.ones((b, d)) * 1j
    m0 = gb_utils.prepare_M0(alpha0, None)
    periodic = jnp.array(domain.periodic)

    forward = core.compute_gaussian_beam_real(
        x0,
        p0,
        m0,
        a0,
        omega0,
        mode,
        domain.c_fn,
        lam,
        ts,
        domain.grid,
        domain.grid_size,
        periodic,
        gb_solvers.solve_ODE_base,
        None,
    )
    backward = core.compute_gaussian_beam_real(
        x0,
        p0,
        m0,
        a0,
        omega0,
        -mode,
        domain.c_fn,
        lam,
        ts,
        domain.grid,
        domain.grid_size,
        periodic,
        gb_solvers.solve_ODE_base,
        None,
    )
    return jnp.squeeze(forward + backward)


def main() -> None:
    KWaveSolver = load_kwave_solver()

    n = 256
    domain = Domain(N=(n,), dx=(1.0 / n,), c=c_homogeneous, cfl=0.3, periodic=(True,))
    ts = jnp.linspace(0.0, 0.45, 120)
    lam = 4.0

    gb_lossless = np.asarray(gaussian_beam_pair(domain, ts, lam=0.0))
    gb_absorbing = np.asarray(gaussian_beam_pair(domain, ts, lam=lam))

    strip_width = 4
    kw_lossless_domain = Domain(
        N=(n, strip_width),
        dx=(domain.dx[0], domain.dx[0]),
        c=1.0,
        cfl=0.3,
        periodic=(True, True),
    )
    kw_absorbing_domain = Domain(
        N=(n, strip_width),
        dx=(domain.dx[0], domain.dx[0]),
        c=1.0,
        cfl=0.3,
        periodic=(True, True),
        alpha_power=0,
        alpha_coeff=lam_to_alpha_db(lam, 1.0),
    )
    kw_p0 = jnp.repeat(jnp.asarray(gb_lossless[0])[:, None], strip_width, axis=1)
    kw_sensor_mask = jnp.zeros(kw_lossless_domain.N).at[:, 0].set(1.0)
    kwave = KWaveSolver(
        backend="python",
        device="cpu",
        pml_size=0,
        smooth_p0=False,
        debug=False,
    )
    kw_lossless = time_first(kwave.forward(kw_p0, kw_lossless_domain, kw_sensor_mask, ts), len(ts))
    kw_absorbing = time_first(kwave.forward(kw_p0, kw_absorbing_domain, kw_sensor_mask, ts), len(ts))

    gb_loss_ratio = normalized_rms(gb_lossless)
    gb_abs_ratio = normalized_rms(gb_absorbing)
    kw_loss_ratio = normalized_rms(kw_lossless)
    kw_abs_ratio = normalized_rms(kw_absorbing)

    print(f"GB final absorbing/lossless RMS ratio: {gb_abs_ratio[-1] / gb_loss_ratio[-1]:.3f}")
    print(f"k-Wave final absorbing/lossless RMS ratio: {kw_abs_ratio[-1] / kw_loss_ratio[-1]:.3f}")
    print(f"k-Wave alpha_coeff: {lam_to_alpha_db(lam, 1.0):.3f} dB/cm")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(np.asarray(domain.grid).reshape(-1), gb_lossless[0], label="lossless")
    axes[0].plot(np.asarray(domain.grid).reshape(-1), gb_absorbing[-1], label="absorbing, final")
    axes[0].set_title("Gaussian-beam profiles")
    axes[0].set_xlabel("x")
    axes[0].legend()

    t = np.asarray(ts)
    axes[1].plot(t, gb_loss_ratio, label="GB lossless")
    axes[1].plot(t, gb_abs_ratio, label="GB absorbing")
    axes[1].plot(t, kw_loss_ratio, "--", label="k-Wave lossless")
    axes[1].plot(t, kw_abs_ratio, "--", label="k-Wave absorbing")
    axes[1].set_title("normalized RMS amplitude")
    axes[1].set_xlabel("t")
    axes[1].legend()

    out_dir = Path(utils.detect_root()) / "plots" / "optional"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "single_gaussian_beam_absorption.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
