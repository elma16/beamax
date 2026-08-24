"""
Compare static MSGB memory plans against a device budget.

Example notebook: false
"""

from __future__ import annotations

import argparse

from beamax.utils import device_capabilities, estimate_msgb_memory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare static MSGB memory plans against a device budget."
    )
    parser.add_argument(
        "--budget-gib",
        type=float,
        default=16.0,
        help="Memory budget in GiB (default: 16).",
    )
    args = parser.parse_args()
    budget = int(args.budget_gib * 1024**3)

    common = dict(
        grid_shape=(128, 128, 128),
        data_shape=(512, 128, 128),
        top_n=4096,
        batch_size=128,
        budget_bytes=budget,
    )
    requests = (
        ("time_reversal", common),
        ("time_reversal", common | {"selection": "materialized"}),
        (
            "forward",
            {
                "grid_shape": (128, 128, 128),
                "top_n": 4096,
                "batch_size": 128,
                "num_times": 512,
                "num_detectors": 128 * 128,
                "budget_bytes": budget,
            },
        ),
    )

    print(device_capabilities().render())
    for operation, options in requests:
        print()
        print(estimate_msgb_memory(operation, **options).render())


if __name__ == "__main__":
    main()
