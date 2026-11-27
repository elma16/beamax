"""Reproduce the time-reversal panels for the first face of the thesis box scanner.

Example extras: kwave,viz-mpl
Example smoke: false
"""

# %% Setup
import jax

from beamax._example_pat import (
    box_scanner_acquisition,
    box_scanner_msgb_inputs,
    homogeneous_msgb,
    timed,
    show_and_save,
)
from beamax.plotter import plot_packet_box_scanner

jax.config.update("jax_enable_x64", jax.default_backend() != "tpu")


def main():
    # %% Acquire the left-face recording with k-Wave
    # The two-packet source occupies a homogeneous, open 128 × 128 domain.
    p0, domain, face, ts, kwave, record = box_scanner_acquisition()
    cropped, ts_data, wpt = box_scanner_msgb_inputs(record, ts)
    msgb = homogeneous_msgb(1849)

    # %% Time-reverse the recording with k-Wave and MSGB
    tr_kw = timed(
        "k-Wave TR",
        kwave.time_reversal,
        record,
        domain,
        detectors=face,
        ts=ts,
    )
    tr_msgb = timed(
        "MSGB TR",
        msgb.time_reversal,
        cropped,
        domain,
        detectors=face,
        ts=ts_data,
        data_wpt=wpt,
    )
    tr = (tr_kw, tr_msgb)

    # %% Plot the one-face fields at the thesis scale and marked column
    fig = plot_packet_box_scanner(
        p0,
        face.to_mask(domain).binary_mask,
        domain.grid_size,
        time_reversal=tr,
    )
    show_and_save(fig, "plots/reconstruction/2d_time_reversal.png")


if __name__ == "__main__":
    main()
