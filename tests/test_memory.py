"""Tests for the pre-flight memory planner."""

from __future__ import annotations

import math

import jax
import numpy as np
import pytest

import beamax.utils.memory as memory_module
from beamax.utils.memory import device_capabilities, estimate_msgb_memory

_REFERENCE_KWARGS = dict(
    grid_shape=(64, 64, 64),
    data_shape=(256, 64, 64),
    top_n=1024,
    batch_size=64,
    use_device_budget=False,
)


def test_device_capabilities_reports_and_renders_current_backend():
    caps = device_capabilities()
    assert caps.backend in {"cpu", "gpu", "tpu"}
    assert caps.device_count >= 1
    assert isinstance(caps.pallas_triton_available, bool)
    assert isinstance(caps.pallas_tpu_available, bool)
    rendered = caps.render()
    assert caps.backend in rendered
    assert caps.jax_version in rendered


def test_device_capabilities_probes_the_pallas_tpu_module(monkeypatch):
    probed = []
    monkeypatch.setattr(
        memory_module,
        "_pallas_backend_available",
        lambda name: probed.append(name) or True,
    )

    caps = device_capabilities()

    assert probed == ["triton", "tpu"]
    assert caps.pallas_tpu_available


def test_selection_grid_and_batch_size_scale_the_estimate():
    default = estimate_msgb_memory("time_reversal", **_REFERENCE_KWARGS)
    materialized = estimate_msgb_memory(
        "time_reversal",
        selection="materialized",
        **_REFERENCE_KWARGS,
    )
    larger_grid = estimate_msgb_memory(
        "time_reversal",
        **{**_REFERENCE_KWARGS, "grid_shape": (128, 128, 128)},
    )
    larger_batch = estimate_msgb_memory(
        "time_reversal",
        **{**_REFERENCE_KWARGS, "batch_size": 128},
    )

    assert default.planning_bytes == 2 * default.lower_bound_bytes
    assert materialized.lower_bound_bytes > default.lower_bound_bytes
    assert larger_grid.lower_bound_bytes > default.lower_bound_bytes
    assert larger_batch.lower_bound_bytes > default.lower_bound_bytes


def test_configurable_safety_margin_is_explicit():
    est = estimate_msgb_memory(
        "forward",
        grid_shape=(64, 64, 64),
        top_n=1024,
        batch_size=64,
        num_times=256,
        num_detectors=64 * 64,
        forward_kernel="xla",
        safety_factor=1.5,
        use_device_budget=False,
    )
    rendered = est.render()
    assert est.planning_bytes == math.ceil(1.5 * est.lower_bound_bytes)
    assert "1.5x lower bound" in rendered
    assert "not a measured peak" in rendered
    assert "no device budget known" in rendered


def test_estimate_scales_with_real_dtype_and_defaults_to_jax_x64_mode():
    float32 = estimate_msgb_memory(
        "time_reversal", real_dtype=np.float32, **_REFERENCE_KWARGS
    )
    float64 = estimate_msgb_memory(
        "time_reversal", real_dtype=np.float64, **_REFERENCE_KWARGS
    )

    assert float64.lower_bound_bytes == 2 * float32.lower_bound_bytes
    assert "float32" in float32.config_summary
    assert "float64" in float64.config_summary

    with jax.enable_x64():
        implicit = estimate_msgb_memory("time_reversal", **_REFERENCE_KWARGS)
    assert implicit.lower_bound_bytes == float64.lower_bound_bytes


def test_fused_pallas_memory_plan_requires_float32():
    kwargs = dict(
        grid_shape=(64, 64, 64),
        top_n=1024,
        batch_size=64,
        num_times=256,
        num_detectors=64 * 64,
        forward_kernel="hom_diag_3d_pallas",
        use_device_budget=False,
    )

    estimate = estimate_msgb_memory("forward", real_dtype=np.float32, **kwargs)
    assert "float32" in estimate.config_summary
    with pytest.raises(ValueError, match="requires real_dtype=float32"):
        estimate_msgb_memory("forward", real_dtype=np.float64, **kwargs)


def test_verdict_thresholds():
    est = estimate_msgb_memory(
        "time_reversal", budget_bytes=40 * 1024**3, **_REFERENCE_KWARGS
    )
    assert "fits comfortably" in est.verdict
    tight = estimate_msgb_memory(
        "time_reversal",
        budget_bytes=int(est.planning_bytes / 0.8),
        **_REFERENCE_KWARGS,
    )
    assert "tight" in tight.verdict
    over = estimate_msgb_memory(
        "time_reversal", budget_bytes=est.planning_bytes // 2, **_REFERENCE_KWARGS
    )
    assert "unlikely to fit" in over.verdict


def test_render_is_itemized_and_totalled():
    est = estimate_msgb_memory(
        "adjoint", budget_bytes=16 * 1024**3, **_REFERENCE_KWARGS
    )
    rendered = est.render()
    assert "adjoint source formation" in rendered
    assert "total" in rendered
    assert "Verdict" in rendered


def test_validation_errors():
    with pytest.raises(ValueError, match="data_shape"):
        estimate_msgb_memory(
            "time_reversal",
            grid_shape=(64, 64),
            top_n=8,
            batch_size=4,
            use_device_budget=False,
        )
    with pytest.raises(ValueError, match="num_times"):
        estimate_msgb_memory(
            "forward",
            grid_shape=(64, 64),
            top_n=8,
            batch_size=4,
            use_device_budget=False,
        )
    with pytest.raises(ValueError, match="positive"):
        estimate_msgb_memory(
            "forward",
            grid_shape=(64, 64),
            top_n=0,
            batch_size=4,
            num_times=4,
            num_detectors=4,
            use_device_budget=False,
        )
    with pytest.raises(ValueError, match="safety_factor"):
        estimate_msgb_memory(
            "forward",
            grid_shape=(64, 64),
            top_n=8,
            batch_size=4,
            num_times=4,
            num_detectors=4,
            safety_factor=0.5,
            use_device_budget=False,
        )
    with pytest.raises(ValueError, match="Unknown operation"):
        estimate_msgb_memory(
            "reconstruct",  # type: ignore[arg-type]
            grid_shape=(64, 64),
            top_n=8,
            batch_size=4,
            use_device_budget=False,
        )
    with pytest.raises(ValueError, match="selection"):
        estimate_msgb_memory(
            "forward",
            grid_shape=(64, 64),
            top_n=8,
            batch_size=4,
            num_times=4,
            num_detectors=4,
            selection="unknown",  # type: ignore[arg-type]
            use_device_budget=False,
        )


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"grid_shape": ()}, "grid_shape"),
        ({"grid_shape": (64, -1)}, "grid_shape"),
        ({"grid_shape": (64, 2.5)}, "grid_shape"),
        ({"grid_shape": (64, True)}, "grid_shape"),
        ({"data_shape": ()}, "data_shape"),
        ({"data_shape": (8, 0)}, "data_shape"),
        ({"top_n": True}, "top_n"),
        ({"batch_size": 2.5}, "batch_size"),
        ({"num_times": 2.5}, "num_times"),
        ({"num_detectors": -1}, "num_detectors"),
        ({"redundancy": 0}, "redundancy"),
        ({"total_coeffs": 0}, "total_coeffs"),
        ({"budget_bytes": 0}, "budget_bytes"),
        ({"use_device_budget": 1}, "use_device_budget"),
        ({"safety_factor": True}, "safety_factor"),
        ({"safety_factor": math.inf}, "safety_factor"),
        ({"real_dtype": np.int32}, "real_dtype"),
        ({"real_dtype": np.complex64}, "real_dtype"),
    ],
)
def test_numeric_shape_and_budget_validation(override, match):
    kwargs = dict(
        grid_shape=(64, 64),
        data_shape=(16, 8),
        top_n=8,
        batch_size=4,
        num_times=16,
        num_detectors=8,
        budget_bytes=1024,
        use_device_budget=False,
    )
    kwargs.update(override)

    with pytest.raises(ValueError, match=match):
        estimate_msgb_memory("forward", **kwargs)
