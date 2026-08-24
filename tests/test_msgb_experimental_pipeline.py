"""Focused tests for MSGB pipeline-stage policies and their defaults."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from beamax import geometry
from beamax.decomposition import DyadicDecomposition
from beamax.gb import gb_solvers
from beamax.gb.pallas_config import PallasConfig
from beamax.solvers.msgb_solvers.forward_solver_utils import compute_forward_result
from beamax.solvers.msgb_solvers.msgb_solver import (
    MSGBExperimentalConfig,
    MSGBSolver,
)
from beamax.transforms import MSWPT


def _solver(
    *,
    experimental_config=None,
    thr_strat="top_n",
    batch_size=8,
    sharding=None,
    sum_method="scan_real",
):
    return MSGBSolver(
        thr=4 if thr_strat == "top_n" else 0.1,
        thr_strat=thr_strat,
        batch_size=batch_size,
        input_type="spatial",
        ode_solver=gb_solvers.solve_hom_diag,
        tr_ode_solver=gb_solvers.solve_ODE_batch_t,
        sum_method=sum_method,
        experimental_config=experimental_config,
        sharding=sharding,
    )


def _make_1d_setup(n=8):
    domain = geometry.Domain(N=(n,), dx=(1.0 / n,), c=1.0, periodic=(False,))
    wpt = MSWPT(
        DyadicDecomposition(
            num_levels=1,
            N=domain.N,
            num_boxes_levels=(4,),
            box_aspect_ratio=(1,),
        ),
        redundancy=2,
        windowing="rectangular_mirror",
    )
    sensors = geometry.Sensor(
        domain=domain,
        binary_mask=jnp.ones(domain.N, dtype=jnp.float32),
    )
    return domain, wpt, sensors


def test_experimental_config_validation():
    with pytest.raises(ValueError, match="boxes_per_chunk"):
        MSGBExperimentalConfig(boxes_per_chunk=0)
    with pytest.raises(ValueError, match="coefficient_selection"):
        MSGBExperimentalConfig(coefficient_selection="inherit")
    with pytest.raises(ValueError, match="forward_kernel"):
        MSGBExperimentalConfig(forward_kernel="inherit")
    with pytest.raises(ValueError, match="inverse_evaluator"):
        MSGBExperimentalConfig(inverse_evaluator="inherit")


def test_all_auto_config_is_equivalent_to_none_and_allows_sharding_object():
    # All-auto is equivalent to omitting the configuration.
    solver = _solver(
        experimental_config=MSGBExperimentalConfig(),
        sharding=None,
    )
    assert solver._effective_inverse_aggregate_method() == "terminal_xla"

    with pytest.raises(ValueError, match="cannot be combined with sharding"):
        _solver(
            experimental_config=MSGBExperimentalConfig(
                inverse_evaluator="terminal_xla"
            ),
            sharding=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="cannot be combined with sharding"):
        _solver(
            experimental_config=MSGBExperimentalConfig(
                coefficient_selection="streaming_top_n"
            ),
            sharding=object(),  # type: ignore[arg-type]
        )


def test_streaming_requires_top_n():
    with pytest.raises(ValueError, match="requires\\s+thr_strat='top_n'"):
        _solver(
            thr_strat="hard",
            experimental_config=MSGBExperimentalConfig(
                coefficient_selection="streaming_top_n"
            ),
        )
    # Other threshold strategies fall back to materialized selection.
    solver = _solver(thr_strat="hard")
    _, wpt, _ = _make_1d_setup()
    assert not solver._streams_top_n(wpt, jnp.float32)


@pytest.mark.parametrize("sound_speed_kind", ["callable", "grid"])
def test_fused_forward_rejects_nonconstant_sound_speed(sound_speed_kind):
    n = 8
    sound_speed = (
        (lambda x: jnp.ones(x.shape[:-1], dtype=jnp.float32))
        if sound_speed_kind == "callable"
        else jnp.ones((n, n, n), dtype=jnp.float32)
    )
    domain = geometry.Domain(
        N=(n, n, n),
        dx=(1.0 / n,) * 3,
        c=sound_speed,
        periodic=(False, False, False),
    )
    wpt = MSWPT(
        DyadicDecomposition(
            num_levels=1,
            N=domain.N,
            num_boxes_levels=(4,),
            box_aspect_ratio=(1, 1, 1),
        ),
        redundancy=2,
        windowing="rectangular_mirror",
    )
    sensor_mask = jnp.zeros(domain.N, dtype=jnp.float32).at[:, :, 0].set(1)
    sensors = geometry.Sensor(domain=domain, binary_mask=sensor_mask)
    solver = _solver(
        batch_size=4,
        experimental_config=MSGBExperimentalConfig(forward_kernel="hom_diag_3d_pallas"),
    )

    with pytest.raises(ValueError, match="scalar constant"):
        solver.forward(
            jnp.zeros(domain.N, dtype=jnp.float32),
            domain,
            sensors,
            jnp.asarray([0.0], dtype=jnp.float32),
            wpt,
        )


def test_streamed_forward_rejects_complex_input():
    domain, wpt, sensors = _make_1d_setup()
    p0_complex = jnp.ones(domain.N, dtype=jnp.complex64)
    ts = jnp.asarray([0.0], dtype=jnp.float32)

    explicit = _solver(
        experimental_config=MSGBExperimentalConfig(
            coefficient_selection="streaming_top_n"
        ),
        sum_method="scan_complex",
    )
    with pytest.raises(ValueError, match="real float32 forward inputs"):
        explicit.forward(p0_complex, domain, sensors, ts, wpt)


def test_real_aggregation_rejects_complex_initial_data():
    domain, wpt, sensors = _make_1d_setup()
    solver = _solver(sum_method="scan_real")

    with pytest.raises(ValueError, match=r"requires an \*_complex sum_method"):
        solver.forward(
            jnp.ones(domain.N, dtype=jnp.complex64),
            domain,
            sensors,
            jnp.asarray([0.0], dtype=jnp.float32),
            wpt,
        )


def test_explicit_streaming_rejects_provided_dpdt():
    domain, wpt, sensors = _make_1d_setup()
    x = domain.grid[..., 0]
    p0 = jnp.exp(-(((x - 0.35) / 0.12) ** 2)).astype(jnp.float32)
    dpdt = (0.1 * jnp.sin(2 * jnp.pi * x)).astype(jnp.float32)
    ts = jnp.asarray([0.0, 0.02], dtype=jnp.float32)

    explicit = _solver(
        batch_size=4,
        experimental_config=MSGBExperimentalConfig(
            coefficient_selection="streaming_top_n"
        ),
    )
    with pytest.raises(ValueError, match="dpdt=None"):
        explicit.forward(p0, domain, sensors, ts, wpt, dpdt=dpdt)


def test_operation_specific_aggregate_resolution():
    explicit = _solver(
        experimental_config=MSGBExperimentalConfig(
            forward_kernel="trajectory_pallas",
            inverse_evaluator="terminal_xla",
        )
    )
    assert explicit.aggregate_method == "scan"
    assert explicit._effective_forward_aggregate_method() == "pallas"
    assert explicit._effective_inverse_aggregate_method() == "terminal_xla"

    # Only inverse aggregation defaults to terminal-only XLA.
    auto = _solver()
    assert auto._effective_forward_aggregate_method() == "scan"
    assert auto._effective_inverse_aggregate_method() == "terminal_xla"

    complex_sum = _solver(sum_method="scan_complex")
    assert complex_sum._effective_inverse_aggregate_method() == "scan"

    pallas_sum = _solver(sum_method="pallas_real")
    assert pallas_sum._effective_inverse_aggregate_method() == "pallas"


def test_public_3d_streamed_zero_velocity_is_consistent_and_wpt_roundtrips():
    n = 16
    domain = geometry.Domain(
        N=(n, n, n),
        dx=(1.0 / n,) * 3,
        c=1.0,
        cfl=float(np.sqrt(3.0) / 4.0),
        periodic=(False, False, False),
    )
    decomp = DyadicDecomposition(
        num_levels=1,
        N=domain.N,
        num_boxes_levels=(4,),
        box_aspect_ratio=(1, 1, 1),
    )
    wpt = MSWPT(decomp, redundancy=2, windowing="rectangular_mirror")
    coords = domain.grid
    p0 = jnp.exp(
        -0.5
        * (
            ((coords[..., 0] - 0.43) / 0.08) ** 2
            + ((coords[..., 1] - 0.57) / 0.11) ** 2
            + ((coords[..., 2] - 0.36) / 0.09) ** 2
        )
    ).astype(jnp.float32)
    plane = jnp.zeros(domain.N, dtype=jnp.float32).at[:, :, 0].set(1)
    sensors = geometry.Sensor(domain=domain, binary_mask=plane)
    ts = jnp.asarray([0.0, 0.02], dtype=jnp.float32)

    default_solver = _solver(batch_size=4)
    default = default_solver.forward(p0, domain, sensors, ts, wpt)
    explicit_zero = default_solver.forward(
        p0, domain, sensors, ts, wpt, dpdt=jnp.zeros_like(p0)
    )
    streamed = _solver(
        batch_size=4,
        experimental_config=MSGBExperimentalConfig(
            coefficient_selection="streaming_top_n",
        ),
    ).forward(p0, domain, sensors, ts, wpt)

    np.testing.assert_allclose(streamed, explicit_zero, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(streamed, default, rtol=2e-5, atol=2e-6)

    fused_solver = _solver(
        batch_size=4,
        experimental_config=MSGBExperimentalConfig(
            coefficient_selection="streaming_top_n",
            forward_kernel="hom_diag_3d_pallas",
        ),
    )
    fused_params = fused_solver._prepare_forward_params_real(p0, None, domain, wpt)
    fused_params = (
        fused_params[0].astype(jnp.float32),
        fused_params[1].astype(jnp.complex64),
        fused_params[2].astype(jnp.float32),
        fused_params[3].astype(jnp.float32),
        fused_params[4].astype(jnp.complex64),
        fused_params[5].astype(jnp.float32),
    )
    with pytest.raises(ValueError, match="requires ode_solver=solve_hom_diag"):
        compute_forward_result(
            params=fused_params,
            c=domain.c_fn,
            lam=domain.lam,
            ts=ts,
            ode_solver=gb_solvers.solve_hom_TR,
            sensors=sensors.positions.astype(jnp.float32),
            domain_size=domain.grid_size.astype(jnp.float32),
            periodic=jnp.asarray(domain.periodic),
            aggregate_method="pallas_fused_hom_diag_3d",
        )
    fused = compute_forward_result(
        params=fused_params,
        c=domain.c_fn,
        lam=domain.lam,
        ts=ts,
        ode_solver=gb_solvers.solve_hom_diag,
        sensors=sensors.positions.astype(jnp.float32),
        domain_size=domain.grid_size.astype(jnp.float32),
        periodic=jnp.asarray(domain.periodic),
        aggregate_method="pallas_fused_hom_diag_3d",
        pallas_config=PallasConfig(
            sensor_block_size=128,
            gpu_time_block_size=2,
            periodic_axes=domain.periodic,
        ),
    )
    np.testing.assert_allclose(fused, default, rtol=3e-5, atol=3e-6)
