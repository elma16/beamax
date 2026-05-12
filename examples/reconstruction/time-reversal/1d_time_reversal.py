#!/usr/bin/env python
"""
Run a compact 1D MSGB time-reversal smoke test.

The example creates a synthetic initial pressure from two MSWPT frame atoms,
records the outgoing wave at the right boundary, crops the recorded data in
frequency space, and propagates it backward with the MSGB time-reversal solver.
The single boundary makes the inverse problem deliberately one-sided, so the
reported metrics focus on data retention and reconstruction strength rather
than presenting this as a full-data accuracy benchmark.
"""

from pathlib import Path
from time import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from beamax import geometry, transforms, utils
from beamax.decomposition import DyadicDecomposition
from beamax.gb import gb_solvers
from beamax.plotter import use_beamax_style
from beamax.solvers import MSGBSolver


jax.config.update("jax_enable_x64", True)


def c_homogeneous(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 + 0.0 * x[..., 0]


def crop_center(arr: jnp.ndarray, size: int) -> jnp.ndarray:
    midpoint = arr.shape[0] // 2
    return arr[midpoint - size // 2 : midpoint + size // 2]


def make_initial_pressure(
    decomp: DyadicDecomposition, redundancy: int, n: tuple[int, ...]
) -> jnp.ndarray:
    kxy = decomp.fourier_meshgrid
    high = transforms.compute_frames(
        decomp, 4, jnp.array([10]), kxy, redundancy, "none"
    )
    low = transforms.compute_frames(
        decomp, 0, jnp.array([25]), kxy, redundancy, "none"
    )
    p0 = utils.unitary_ifft(high) + utils.unitary_ifft(low)
    p0 = jnp.real(p0 / jnp.max(jnp.abs(p0)))
    return p0.reshape(n)


def main() -> None:
    root_dir = utils.detect_root()
    plot_dir = Path(root_dir) / "plots"
    plot_dir.mkdir(exist_ok=True)
    use_beamax_style()

    n = (256,)
    dx = (1.0e-4,)
    cfl = 0.5
    periodic = (False,)
    num_levels = 2
    num_boxes_levels = (4, 8)
    redundancy = 2
    windowing = "rectangular_mirror"

    image_domain = geometry.Domain(
        N=n, dx=dx, c=c_homogeneous, cfl=cfl, periodic=periodic
    )
    xy = image_domain.grid
    ts_image = image_domain.generate_time_domain()
    tmax = ts_image[-1]
    ts_image = jnp.linspace(0, tmax, 4 * n[0])
    image_domain = geometry.Domain(
        N=n, dx=dx, c=c_homogeneous, cfl=cfl, periodic=periodic
    )

    image_decomp = DyadicDecomposition(
        num_levels=num_levels,
        N=n,
        num_boxes_levels=num_boxes_levels,
        box_aspect_ratio=(1,),
    )
    image_wpt = transforms.MSWPT(image_decomp, redundancy, windowing)
    p0 = make_initial_pressure(image_decomp, redundancy, n)

    sensor_mask = jnp.zeros(n).at[-1].set(1)
    sources = geometry.Sensor(domain=image_domain, binary_mask=sensor_mask)

    solver = MSGBSolver(
        thr=2 * n[0],
        thr_strat="top_n",
        batch_size=16,
        input_type="spatial",
        ode_solver=gb_solvers.solve_ODE_base,
        tr_ode_solver=gb_solvers.solve_hom_TR,
        sum_method="scan_real",
    )

    t0 = time()
    sensor_data, _ = solver.forward(
        p0,
        image_domain,
        sources.positions,
        ts_image,
        image_wpt,
    )
    forward_s = time() - t0

    sensor_fft = utils.unitary_fft(sensor_data)
    cropped_fft = crop_center(sensor_fft, n[0])
    cropped_data = jnp.squeeze(utils.unitary_ifft(cropped_fft))
    energy_ratio = float(jnp.linalg.norm(cropped_data) / jnp.linalg.norm(sensor_data))

    nt_data = cropped_data.shape[0]
    ts_data = jnp.linspace(0, tmax, nt_data)
    dt_data = float(ts_data[1] - ts_data[0])
    data_domain = geometry.Domain(
        N=(nt_data,), dx=(dt_data,), c=c_homogeneous, periodic=periodic, cfl=cfl
    )
    data_decomp = DyadicDecomposition(
        num_levels=num_levels,
        N=(nt_data,),
        num_boxes_levels=num_boxes_levels,
        box_aspect_ratio=(1,),
    )
    data_wpt = transforms.MSWPT(data_decomp, redundancy, windowing)

    t0 = time()
    reconstructed, _ = solver.time_reversal(
        data=cropped_data,
        domain=image_domain,
        sensors=xy,
        sources=sources,
        ts=ts_data,
        data_domain=data_domain,
        data_wpt=data_wpt,
    )
    tr_s = time() - t0

    truth = jnp.real(p0).reshape(-1)
    recon = jnp.real(reconstructed).reshape(-1)
    scale = jnp.vdot(recon, truth).real / (jnp.vdot(recon, recon).real + 1e-30)
    recon_scaled = scale * recon
    reconstruction_ratio = float(jnp.linalg.norm(recon) / jnp.linalg.norm(truth))
    overlap = float(
        jnp.abs(jnp.vdot(recon, truth))
        / (jnp.linalg.norm(recon) * jnp.linalg.norm(truth) + 1e-30)
    )

    x = np.arange(n[0]) * dx[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    axes[0].plot(ts_image, np.asarray(jnp.squeeze(sensor_data).real))
    axes[0].set_title("Boundary measurement")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("pressure")

    axes[1].plot(x, np.asarray(truth), label="truth")
    axes[1].plot(x, np.asarray(recon_scaled), "--", label="time reversal")
    axes[1].set_title("Initial pressure reconstruction")
    axes[1].set_xlabel("x")
    axes[1].legend()

    out_path = plot_dir / "1d_time_reversal.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Forward solve: {forward_s:.2f}s; time reversal: {tr_s:.2f}s")
    print(f"Cropped sensor-data energy ratio: {energy_ratio:.6f}")
    print(f"One-sided reconstruction norm ratio: {reconstruction_ratio:.3f}")
    print(f"One-sided reconstruction overlap: {overlap:.3f}")
    print(f"Saved 1D time-reversal plot to {out_path}")


if __name__ == "__main__":
    main()
