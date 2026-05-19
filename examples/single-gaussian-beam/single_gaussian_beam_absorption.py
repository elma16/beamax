#!/usr/bin/env python
"""
Compare lossless and absorbing single-Gaussian beam propagation.

The example evaluates the same 1D beam pair with two damping coefficients and
plots the resulting profiles and normalized RMS amplitudes over time.
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


def c_homogeneous(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 + 0.0 * x[..., 0]


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
    n = 256
    domain = Domain(N=(n,), dx=(1.0 / n,), c=c_homogeneous, cfl=0.3, periodic=(True,))
    ts = jnp.linspace(0.0, 0.45, 120)
    lam = 4.0

    gb_lossless = np.asarray(gaussian_beam_pair(domain, ts, lam=0.0))
    gb_absorbing = np.asarray(gaussian_beam_pair(domain, ts, lam=lam))

    loss_ratio = normalized_rms(gb_lossless)
    abs_ratio = normalized_rms(gb_absorbing)
    final_ratio = float(abs_ratio[-1] / loss_ratio[-1])

    print(f"Final absorbing/lossless RMS ratio: {final_ratio:.3f}")
    print(f"Damping coefficient lambda: {lam:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x = np.asarray(domain.grid).reshape(-1)
    axes[0].plot(x, gb_lossless[0], label="lossless, initial")
    axes[0].plot(x, gb_lossless[-1], "--", label="lossless, final")
    axes[0].plot(x, gb_absorbing[-1], label="absorbing, final")
    axes[0].set_title("Gaussian beam profiles")
    axes[0].set_xlabel("x")
    axes[0].legend()

    t = np.asarray(ts)
    axes[1].plot(t, loss_ratio, label="lossless")
    axes[1].plot(t, abs_ratio, label="absorbing")
    axes[1].set_title("normalized RMS amplitude")
    axes[1].set_xlabel("t")
    axes[1].legend()

    out_dir = Path(utils.detect_root()) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "single_gaussian_beam_absorption.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
