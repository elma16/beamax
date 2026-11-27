import numpy as np
import pytest

from beamax._example_pat import resize_record


@pytest.mark.parametrize("full_samples", [15, 16])
def test_detector_crop_and_lift_preserve_weighted_pairing(full_samples):
    rng = np.random.default_rng(15)
    record = rng.normal(size=(full_samples, 5))
    probe = rng.normal(size=(8, 5))
    # Both grids cover the same unit interval. The rectangle-rule weights are dt.
    data_pairing = np.vdot(resize_record(record, len(probe)), probe) / (len(probe) - 1)
    full_pairing = np.vdot(record, resize_record(probe, len(record))) / (
        len(record) - 1
    )
    np.testing.assert_allclose(data_pairing, full_pairing, rtol=1e-5, atol=1e-6)
