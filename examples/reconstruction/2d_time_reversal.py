"""Reproduce the time-reversal panels for the first face of the thesis box scanner.

Example extras: kwave,viz-mpl
Example smoke: false
"""

# %% Import the solvers and plotting functions
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from beamax import Domain, DyadicDecomposition, MSWPT, utils
from beamax.gb.gb_solvers import solve_hom_diag, solve_hom_TR
from beamax.solvers import KWaveSolver, MSGBSolver
from beamax.transforms import compute_frames
from beamax.surfaces import DetectorSurface
from beamax.plotter import plot_packet_box_scanner, save_comparison_figure


# %% Measure a first call and a repeat
# Converting the result to a NumPy array waits for the JAX device to finish.
# Pass a zero-argument function so the timer starts before the solver call.
# The first timing includes any compilation triggered by this call.
def timed(label, solve):
    """Report wall time including compilation on first use and host transfer."""
    for run in ("first call", "repeat"):
        start = perf_counter()
        result = np.asarray(solve())
        print(f"{label} ({run}): {perf_counter() - start:.3f} s")
    return result


# %% Construct a wave packet from a frequency box and a translation
def packet(tiles, box, position):
    """Synthesize one spatial packet from the selected Fourier frame."""
    spectrum = compute_frames(
        tiles, box, jnp.array(position), tiles.fourier_meshgrid, 2, "none"
    )
    return utils.unitary_ifft(spectrum)


# %% Resample a time-first pressure recording
# Fourier crop only the time axis. The two grids span the same interval, so the
# square-root factor accounts for the change in the time integration weight.
def resize_record(record, samples):
    """Thesis spectral crop or matched lift, with rectangle-rule time weights."""
    field = jnp.asarray(record)
    resized = utils.interpolate_fourier(
        field, (samples, field.shape[1]), "spatial", "spatial"
    )
    return resized.real * jnp.sqrt((samples - 1) / (field.shape[0] - 1))


def main():
    # %% Choose the numerical precision
    # MSGB uses the selected JAX device. TPU FFTs use single precision.
    jax.config.update("jax_enable_x64", jax.default_backend() != "tpu")

    # %% Define the open domain and its left detector face
    # The 128 receivers sit at x = 0, one grid step apart, and face into the domain.
    domain = Domain(
        N=(128, 128), dx=(1e-4, 1e-4), c=1500.0, cfl=0.354, periodic=(False, False)
    )
    ts = domain.generate_time_domain()
    face = DetectorSurface.plane(
        2, axis=0, value=0.0, shape=(domain.N[1],), spacing=(domain.dx[1],), front=1
    )

    # %% Build the mixed two-packet initial pressure
    # Use the Chapter 4 fine and coarse frequency boxes at their original translations.
    # The transpose puts the frame axes in the image convention used by the thesis.
    tiles = DyadicDecomposition(3, domain.N, (4, 8, 16), (1, 1))
    field = packet(tiles, 125, (11, 6)) + packet(tiles, 44, (11, 3))
    p0 = (field / jnp.max(jnp.abs(field))).T.real

    # %% Record the pressure with k-Wave
    # The 20-cell absorbing layer is outside the image. Every inverse below starts
    # with this same record, indexed by time first and receiver second.
    kwave = KWaveSolver(
        backend="python",
        device="cpu",
        pml_size=20,
        pml_inside=False,
        smooth_p0=False,
        quiet=True,
    )
    jax.block_until_ready((p0, ts))
    print(f"MSGB device: {jax.default_backend()} | k-Wave: Python on CPU")
    record = timed(
        "k-Wave acquisition",
        lambda: kwave.forward(p0, domain, detectors=face, ts=ts),
    )

    # %% Prepare the MSGB detector-data transform and solver
    # k-Wave uses the full record. MSGB analyses 256 time samples by 128 receivers
    # with a rectangular tiling. Both time grids start and end at the same instants.
    ts_data = jnp.linspace(ts[0], ts[-1], 256)
    cropped = resize_record(record, len(ts_data))
    data_tiles = DyadicDecomposition(3, cropped.shape, (4, 8, 16), (2, 1))
    wpt = MSWPT(data_tiles, redundancy=2, windowing="rectangular_mirror")
    # The 1% thesis budget retains 1,849 coefficients. The homogeneous inverse
    # propagates beams analytically. scan_real selects the standard JAX kernels.
    msgb = MSGBSolver(
        thr=1849,
        thr_strat="top_n",
        batch_size=128,
        input_type="spatial",
        ode_solver=solve_hom_diag,
        tr_ode_solver=solve_hom_TR,
        sum_method="scan_real",
    )
    jax.block_until_ready((ts_data, cropped, wpt))

    # %% Time-reverse the recording with k-Wave and MSGB
    # Back-propagate the boundary pressures to the initial time.
    tr_kw = timed(
        "k-Wave TR",
        lambda: kwave.time_reversal(record, domain, detectors=face, ts=ts),
    )
    tr_msgb = timed(
        "MSGB TR",
        lambda: msgb.time_reversal(
            cropped, domain, detectors=face, ts=ts_data, data_wpt=wpt
        ),
    )
    tr = (tr_kw, tr_msgb)

    # %% Plot the one-face fields at the thesis scale and marked column
    # Pass raw image pairs in (k-Wave, MSGB) order. The plotter applies the
    # thesis display gain to both adjoints and marks the same profile column.
    fig = plot_packet_box_scanner(
        p0,
        face.to_mask(domain).binary_mask,
        domain.grid_size,
        time_reversal=tr,
    )
    save_comparison_figure(fig, "plots/reconstruction/2d_time_reversal.png", show=True)


if __name__ == "__main__":
    main()
