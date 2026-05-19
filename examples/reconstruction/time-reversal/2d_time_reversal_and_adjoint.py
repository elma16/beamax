#!/usr/bin/env python
"""
Compare k-Wave time-reversal and adjoint reconstructions on a tiny 2D phantom.

This optional example distills the thesis inverse-comparison scripts to one
small one-sided acquisition. It forwards a smooth phantom to a boundary sensor
line, reconstructs with k-Wave time reversal and adjoint backpropagation, and
prints normalized overlap metrics.

Example category: Time-reversal reconstruction
Example extras: kwave,viz-mpl
Example smoke: false
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from beamax import utils
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
    return 1500.0 + 0.0 * x[..., 0]


def make_phantom(domain: Domain) -> jnp.ndarray:
    """Two smooth inclusions with zero mean."""
    x, y = jnp.meshgrid(
        jnp.arange(domain.N[0]) * domain.dx[0],
        jnp.arange(domain.N[1]) * domain.dx[1],
        indexing="ij",
    )
    lx, ly = domain.grid_size
    p0 = jnp.exp(
        -((x - 0.38 * lx) ** 2 + (y - 0.45 * ly) ** 2) / (2.0 * (0.08 * lx) ** 2)
    )
    p0 -= 0.7 * jnp.exp(
        -((x - 0.62 * lx) ** 2 + (y - 0.58 * ly) ** 2) / (2.0 * (0.09 * lx) ** 2)
    )
    p0 = p0 - jnp.mean(p0)
    return p0 / jnp.max(jnp.abs(p0))


def match_image_shape(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Coerce k-Wave image output to the expected image shape."""
    image = np.asarray(arr)
    if image.shape == shape:
        return image
    if image.T.shape == shape:
        return image.T
    return image.reshape(shape)


def scaled_reconstruction(
    recon: np.ndarray, truth: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Scale reconstruction to the truth and return ``(scaled, overlap, relative_l2)``."""
    recon_real = np.asarray(recon).real
    truth_real = np.asarray(truth).real
    scale = float(
        np.vdot(recon_real, truth_real) / (np.vdot(recon_real, recon_real) + 1e-30)
    )
    scaled = scale * recon_real
    overlap = float(
        abs(np.vdot(scaled, truth_real))
        / (np.linalg.norm(scaled) * np.linalg.norm(truth_real) + 1e-30)
    )
    rel_l2 = float(np.linalg.norm(scaled - truth_real) / np.linalg.norm(truth_real))
    return scaled, overlap, rel_l2


def main() -> None:
    KWaveSolver = load_kwave_solver()

    n = (32, 32)
    domain = Domain(
        N=n,
        dx=(1.0e-4, 1.0e-4),
        c=c_homogeneous,
        cfl=0.3,
        periodic=(False, False),
    )
    ts = domain.generate_time_domain()
    p0 = make_phantom(domain)
    sensor_mask = jnp.zeros(n).at[0, :].set(1.0)
    image_mask = jnp.ones(n)

    kwave = KWaveSolver(
        backend="python",
        device="cpu",
        pml_size=8,
        smooth_p0=False,
        debug=False,
    )
    data = kwave.forward(p0, domain, sensor_mask, ts)
    tr = -match_image_shape(
        kwave.time_reversal(
            data=data,
            domain=domain,
            sensors=image_mask,
            sources=sensor_mask,
            ts=ts,
            data_layout="nt_ns",
        ),
        n,
    )
    adj = -match_image_shape(
        kwave.adjoint(
            data=data,
            domain=domain,
            sensors=image_mask,
            sources=sensor_mask,
            ts=ts,
            data_layout="nt_ns",
        ),
        n,
    )

    tr_scaled, tr_overlap, tr_l2 = scaled_reconstruction(tr, np.asarray(p0))
    adj_scaled, adj_overlap, adj_l2 = scaled_reconstruction(adj, np.asarray(p0))
    print(f"TR overlap={tr_overlap:.3f}, relative L2={tr_l2:.3f}")
    print(f"Adjoint overlap={adj_overlap:.3f}, relative L2={adj_l2:.3f}")

    vmax = float(np.max(np.abs(np.asarray(p0))))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    for ax, image, title in (
        (axes[0], np.asarray(p0), "initial pressure"),
        (axes[1], tr_scaled, "time reversal"),
        (axes[2], adj_scaled, "adjoint"),
    ):
        im = ax.imshow(image, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_dir = Path(utils.detect_root()) / "plots" / "optional"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "2d_time_reversal_and_adjoint.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
