"""Compare k-Wave, MSGB and Hybrid records for the thesis mixed two-packet source.

Example extras: kwave,viz-mpl
Example smoke: false
"""

# %% Import the solvers and plotting functions
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from beamax import Domain, DyadicDecomposition, MSWPT, utils
from beamax.gb.gb_solvers import solve_hom_diag
from beamax.solvers import KWaveSolver, MSGBSolver
from beamax.transforms import compute_frames
from beamax.plotter import plot_forward_records, save_comparison_figure
from beamax.solvers.hybrid_solver_utils import split_frequency_components


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


# %% Combine coarse-grid k-Wave and high-frequency MSGB recordings
# The low-frequency record is extended and tapered before Fourier interpolation.
# This matches the time period on the coarse and fine FFT grids. The two bands
# can then be added at the same receiver locations and sample times.
def hybrid_forward(p0, domain, detectors, ts, wpt, kwave, high_solver):
    """Chapter 4 Hybrid recording with matching fine and coarse FFT periods."""
    # Frequency boxes 16 and 75 delimit the low-frequency region in this tiling.
    high, low, low_detectors, coarse = split_frequency_components(
        p0,
        detectors,
        "spatial",
        "spatial",
        wpt,
        (16, 75),
        wpt.windowing,
        domain,
        downsample=True,
    )
    # The coarse step is an integer multiple of the fine step within its CFL limit.
    # Extend by 30 fine samples and round up to a complete coarse FFT period.
    dt = float(ts[1] - ts[0])
    coarse_ts = coarse.generate_time_domain()
    q = max(1, int(np.floor(float(coarse_ts[1] - coarse_ts[0]) / dt + 1e-12)))
    nt_extended = len(ts) + 30
    nt_extended += -nt_extended % q
    coarse_ts = jnp.arange(nt_extended // q) * (q * dt)
    low_record = np.asarray(
        kwave.forward(low.real, coarse, detectors=low_detectors, ts=coarse_ts)
    )

    # Only taper the extension. A periodic FFT then interpolates time and receiver position.
    tail = np.clip(
        (np.asarray(coarse_ts) - float(ts[-1])) / float(coarse_ts[-1] - ts[-1]), 0, 1
    )
    weights = np.i0(12.0 * np.sqrt(1 - tail**2)) / np.i0(12.0)
    weights[-1] = 0
    ns = int(np.count_nonzero(detectors))
    low_fine = utils.interpolate_fourier(
        jnp.asarray(low_record * weights[:, None], dtype=p0.dtype),
        (nt_extended, ns),
        "spatial",
        "spatial",
    ).real
    scale = np.sqrt((low_record.shape[1] / ns) * (nt_extended / len(coarse_ts)))
    high_record = high_solver.forward(
        high.real, domain, detectors=detectors, ts=ts, wpt=wpt
    )
    return scale * low_fine[: len(ts)] + high_record.real


def main():
    # %% Choose the numerical precision
    # MSGB uses the selected JAX device. TPU FFTs use single precision.
    jax.config.update("jax_enable_x64", jax.default_backend() != "tpu")

    # %% Define the periodic domain and receiver line
    # The thesis forward experiment lets waves wrap around the boundary. Disable
    # the k-Wave absorbing layer and record pressure along the first grid row.
    domain = Domain(
        N=(128, 128), dx=(1e-4, 1e-4), c=1500.0, cfl=0.354, periodic=(True, True)
    )
    ts = domain.generate_time_domain()
    detectors = jnp.zeros(domain.N, dtype=bool).at[0, :].set(True)

    # %% Build the mixed two-packet initial pressure
    # Use the Chapter 4 fine and coarse frequency boxes at their original translations.
    # The transpose puts the frame axes in the image convention used by the thesis.
    tiles = DyadicDecomposition(3, domain.N, (4, 8, 16), (1, 1))
    field = packet(tiles, 125, (11, 6)) + packet(tiles, 44, (11, 3))
    p0 = (field / jnp.max(jnp.abs(field))).T.real

    # %% Check the 1% synthesis budget
    # Rank the analysis coefficients by magnitude and reconstruct the retained
    # coefficients on nominal frame supports, as in the thesis calibration.
    wpt = MSWPT(tiles, redundancy=2, windowing="rectangular_mirror")
    budget = 1849
    coeffs = wpt.forward(p0, "spatial")
    nominal = MSWPT(tiles, redundancy=2, windowing="none")
    indices = jnp.argsort(-jnp.abs(coeffs), stable=True)[:budget]
    selected = jnp.zeros_like(coeffs).at[indices].set(coeffs[indices])
    error = utils.rel_l2(p0, nominal.inverse(selected, "spatial").real)
    print(f"{budget} coefficients; relative synthesis error {error:.3%}")

    # %% Configure the three forward simulations
    # Both beam solvers use the closed-form homogeneous propagator and standard
    # JAX beam summation. Hybrid uses a separate budget for its high-frequency band.
    kwave = KWaveSolver(
        backend="python", device="cpu", pml_size=0, smooth_p0=False, quiet=True
    )
    beam_options = dict(
        thr_strat="top_n",
        batch_size=128,
        input_type="spatial",
        ode_solver=solve_hom_diag,
        sum_method="scan_real",
    )
    msgb = MSGBSolver(thr=budget, **beam_options)
    high_solver = MSGBSolver(thr=940, **beam_options)

    # %% Record pressure at the 128 receivers
    jax.block_until_ready((p0, detectors, ts, wpt))
    print(f"MSGB device: {jax.default_backend()} | k-Wave: Python on CPU")
    reference = timed(
        "k-Wave forward",
        lambda: kwave.forward(p0, domain, detectors=detectors, ts=ts),
    )
    beams = timed(
        "MSGB forward",
        lambda: msgb.forward(p0, domain, detectors=detectors, ts=ts, wpt=wpt),
    )
    # The thesis Hybrid uses 940 high-frequency coefficients at the same 1% target.
    hybrid = timed(
        "Hybrid forward",
        lambda: hybrid_forward(p0, domain, detectors, ts, wpt, kwave, high_solver),
    )
    for name, record in (("MSGB", beams), ("Hybrid", hybrid)):
        print(
            f"{name} relative L2 difference from k-Wave: {utils.rel_l2(reference, record):.3%}"
        )

    # %% Compare the three recordings at the marked receiver
    fig = plot_forward_records(reference, beams, hybrid, ts)
    save_comparison_figure(
        fig, "plots/forward/2d_forward_comparison.png", fixed_canvas=True, show=True
    )


if __name__ == "__main__":
    main()
