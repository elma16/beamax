"""Compare k-Wave, MSGB and Hybrid records for the thesis mixed two-packet source.

Example extras: kwave,viz-mpl
Example smoke: false
"""

# %% Setup
import jax
import jax.numpy as jnp

from beamax import Domain, utils
from beamax._example_pat import (
    homogeneous_msgb,
    mixed_two_packets,
    packet_synthesis_error,
    packet_transform,
    show_and_save,
    timed,
    thesis_hybrid_forward,
)
from beamax.plotter import plot_forward_records
from beamax.solvers import KWaveSolver

jax.config.update("jax_enable_x64", jax.default_backend() != "tpu")


def main():
    # %% Generate the Chapter 4 mixed two-packet source
    domain = Domain(
        N=(128, 128), dx=(1e-4, 1e-4), c=1500.0, cfl=0.354, periodic=(True, True)
    )
    ts = domain.generate_time_domain()
    detectors = jnp.zeros(domain.N, dtype=bool).at[0, :].set(True)
    wpt = packet_transform(domain.N)
    p0 = mixed_two_packets(wpt)
    budget = 1849  # Chapter 4's magnitude-ranked budget for the 1% synthesis target.
    print(
        f"{budget} coefficients; relative synthesis error {packet_synthesis_error(p0, wpt, budget):.3%}"
    )
    msgb = homogeneous_msgb(budget)
    kwave = KWaveSolver(
        backend="python", device="cpu", pml_size=0, smooth_p0=False, quiet=True
    )

    # %% Record pressure at the 128 receivers
    jax.block_until_ready((p0, detectors, ts, wpt))
    print(f"MSGB device: {jax.default_backend()} | k-Wave: Python on CPU")
    reference = timed(
        "k-Wave forward", kwave.forward, p0, domain, detectors=detectors, ts=ts
    )
    beams = timed(
        "MSGB forward", msgb.forward, p0, domain, detectors=detectors, ts=ts, wpt=wpt
    )
    # The thesis Hybrid uses 940 high-frequency coefficients at the same 1% target.
    hybrid = timed(
        "Hybrid forward",
        thesis_hybrid_forward,
        p0,
        domain,
        detectors,
        ts,
        wpt,
        kwave,
        homogeneous_msgb(940),
    )
    for name, record in (("MSGB", beams), ("Hybrid", hybrid)):
        print(
            f"{name} relative L2 difference from k-Wave: {utils.rel_l2(reference, record):.3%}"
        )

    # %% Compare the three recordings at the marked receiver
    fig = plot_forward_records(reference, beams, hybrid, ts)
    show_and_save(fig, "plots/forward/2d_forward_comparison.png", fixed_canvas=True)


if __name__ == "__main__":
    main()
