"""Acquisition and timing helpers for the homogeneous 2D photoacoustic examples."""

from pathlib import Path
from time import perf_counter

import jax.numpy as jnp
import numpy as np

from beamax import DyadicDecomposition, MSWPT, utils
from beamax.gb.gb_solvers import solve_hom_diag
from beamax.solvers import MSGBSolver
from beamax.transforms import compute_frames


def packet_transform(shape):
    aspect = (shape[0] // shape[1], 1)
    tiles = DyadicDecomposition(3, shape, (4, 8, 16), aspect)
    return MSWPT(tiles, redundancy=2, windowing="rectangular_mirror")


def mixed_two_packets(wpt):
    """Chapter 4 source with one fine-scale packet and one coarser packet."""
    tiles = wpt.dyadic_decomp

    def packet(box, position):
        spectrum = compute_frames(
            tiles, box, jnp.array(position), tiles.fourier_meshgrid, 2, "none"
        )
        return utils.unitary_ifft(spectrum)

    field = packet(125, (11, 6)) + packet(44, (11, 3))
    return (field / jnp.max(jnp.abs(field))).T.real


def packet_synthesis_error(p0, wpt, count):
    """Check the archived 1% budget using the thesis nominal-support synthesis."""
    coeffs = wpt.forward(p0, "spatial")
    nominal = MSWPT(wpt.dyadic_decomp, redundancy=2, windowing="none")
    indices = jnp.argsort(-jnp.abs(coeffs), stable=True)[:count]
    selected = jnp.zeros_like(coeffs).at[indices].set(coeffs[indices])
    return utils.rel_l2(p0, nominal.inverse(selected, "spatial").real)


def homogeneous_msgb(count):
    return MSGBSolver(
        thr=count,
        thr_strat="top_n",
        batch_size=128,
        input_type="spatial",
        ode_solver=solve_hom_diag,
        sum_method="scan_real",
    )


def resize_record(record, samples):
    """Thesis spectral crop or matched lift, with rectangle-rule time weights."""
    field = jnp.asarray(record)
    resized = utils.interpolate_fourier(
        field, (samples, field.shape[1]), "spatial", "spatial"
    )
    return resized.real * jnp.sqrt((samples - 1) / (field.shape[0] - 1))


def timed(label, solve, *args, **kwargs):
    """Report wall time including compilation on first use and host transfer."""
    for run in ("first call", "repeat"):
        start = perf_counter()
        result = np.asarray(solve(*args, **kwargs))
        print(f"{label} ({run}): {perf_counter() - start:.3f} s")
    return result


def show_and_save(fig, path, *, fixed_canvas=False):
    from matplotlib import pyplot as plt

    from beamax._comparison_plots import _STYLE

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Tick locators also consult rcParams at save time. Display the exported
    # PNG in Colab so its inline formatter cannot change the ticks or crop.
    with plt.rc_context(_STYLE):
        bounds = fig.bbox_inches if fixed_canvas else "tight"
        fig.savefig(path, dpi=300, bbox_inches=bounds, pad_inches=0.05)
        backend = plt.get_backend().lower()
        if "inline" in backend:
            from IPython.display import Image, display

            display(Image(filename=str(path), retina=True))
        elif backend != "agg":
            plt.show()
    plt.close(fig)


def thesis_hybrid_forward(p0, domain, detectors, ts, wpt, kwave, high_solver):
    """Chapter 4 Hybrid recording with matching fine and coarse FFT periods."""
    from beamax.solvers.hybrid_solver_utils import split_frequency_components

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


def box_scanner_acquisition():
    """Acquire the left face of the Chapter 5 homogeneous box scanner."""
    import jax
    from beamax import Domain
    from beamax.solvers import KWaveSolver
    from beamax.surfaces import DetectorSurface

    domain = Domain(
        N=(128, 128), dx=(1e-4, 1e-4), c=1500.0, cfl=0.354, periodic=(False, False)
    )
    face = DetectorSurface.plane(
        2, axis=0, value=0.0, shape=(128,), spacing=(domain.dx[1],), front=1
    )
    ts = domain.generate_time_domain()
    p0 = mixed_two_packets(packet_transform(domain.N))
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
        "k-Wave acquisition", kwave.forward, p0, domain, detectors=face, ts=ts
    )
    return p0, domain, face, ts, kwave, record


def box_scanner_msgb_inputs(record, ts):
    """Resample a box-scanner recording for the thesis boundary-data transform."""
    import jax

    ts_data = jnp.linspace(ts[0], ts[-1], 256)
    cropped = resize_record(record, len(ts_data))
    wpt = packet_transform(cropped.shape)
    jax.block_until_ready((ts_data, cropped, wpt))
    return cropped, ts_data, wpt
