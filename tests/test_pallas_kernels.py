"""Correctness tests for the experimental Pallas GB accumulator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamax.gb import core, gb_solvers
from beamax.gb.pallas_config import PallasConfig
from beamax.gb.pallas_kernels import (
    _minimum_image_displacement,
    _round_to_nearest_even_with_floor,
    sum_gaussian_beam_real_pallas,
)
from beamax.solvers.msgb_solvers.forward_solver_utils import compute_forward_result
from beamax.solvers.msgb_solvers.msgb_solver import MSGBSolver, ShardingStrategy
from beamax.solvers.msgb_solvers.tr_solver_utils import compute_TR_result


jax.config.update("jax_enable_x64", True)


def test_floor_round_emulation_matches_jax_at_positive_and_negative_ties():
    values = jnp.asarray(
        [-4.6, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 4.6],
        dtype=jnp.float32,
    )

    actual = jax.jit(_round_to_nearest_even_with_floor)(values)

    np.testing.assert_array_equal(actual, jnp.round(values))
    jaxpr = str(jax.make_jaxpr(_round_to_nearest_even_with_floor)(values))
    assert "round" not in jaxpr
    assert "floor" in jaxpr


def test_half_open_minimum_image_has_documented_tie_convention():
    displacement = jnp.asarray([-7.5, -2.5, 2.5, 7.5], dtype=jnp.float32)
    actual = _minimum_image_displacement(
        displacement,
        jnp.asarray(5.0),
        jnp.asarray(True),
        rounding_mode="half_open",
        periodic_axis=True,
    )

    np.testing.assert_array_equal(actual, jnp.asarray([-2.5, -2.5, -2.5, -2.5]))


def test_full_pallas_kernel_obeys_each_exact_half_domain_tie_convention():
    xt = jnp.zeros((1, 1, 1), dtype=jnp.float32)
    pt = jnp.ones_like(xt)
    matrix = jnp.full((1, 1, 1, 1), 0.4j, dtype=jnp.complex64)
    phase_offset = jnp.asarray(0.3, dtype=jnp.float32)
    amplitude = (0.5 * jnp.exp(1j * phase_offset)).reshape(1, 1, 1)
    omega = jnp.ones((1,), dtype=jnp.float32)
    sensors = jnp.asarray([[-1.5], [-0.5], [0.5], [1.5]], dtype=jnp.float32)
    domain_size = jnp.ones((1,), dtype=jnp.float32)
    periodic = jnp.ones((1,), dtype=bool)
    outputs = {}

    for rounding_mode in ("nearest_even", "half_open"):
        quotient = sensors[:, 0]
        image = (
            jnp.round(quotient)
            if rounding_mode == "nearest_even"
            else jnp.floor(quotient + 0.5)
        )
        delta = quotient - image
        expected = (jnp.cos(delta + phase_offset) * jnp.exp(-0.2 * delta**2))[None, :]
        outputs[rounding_mode] = sum_gaussian_beam_real_pallas(
            xt,
            pt,
            matrix,
            amplitude,
            omega,
            sensors,
            domain_size,
            periodic,
            config=PallasConfig(
                sensor_block_size=4,
                rounding_mode=rounding_mode,
                periodic_axes=(True,),
            ),
            interpret=True,
        )
        np.testing.assert_allclose(
            outputs[rounding_mode], expected, rtol=1e-6, atol=1e-6
        )

    assert not np.allclose(outputs["nearest_even"], outputs["half_open"])


def _trajectory_case(ndim, *, dtype=jnp.float32, num_sensors=11):
    num_beams, num_times = 5, 3
    keys = jax.random.split(jax.random.key(17 + ndim), 7)
    xt = jax.random.uniform(keys[0], (num_beams, num_times, ndim), dtype=dtype)
    pt = 0.2 * jax.random.normal(keys[1], (num_beams, num_times, ndim), dtype=dtype)
    Mr = 0.1 * jax.random.normal(
        keys[2], (num_beams, num_times, ndim, ndim), dtype=dtype
    )
    identity = jnp.eye(ndim, dtype=dtype)[None, None, :, :]
    Mi = jnp.broadcast_to(identity, Mr.shape) * jnp.asarray(0.3, dtype=dtype)
    Mt = Mr + 1j * Mi
    At = jax.random.normal(
        keys[3], (num_beams, num_times, 1), dtype=dtype
    ) + 1j * jax.random.normal(keys[4], (num_beams, num_times, 1), dtype=dtype)
    omega = 0.5 + jax.random.uniform(keys[5], (num_beams,), dtype=dtype)
    sensors = jax.random.uniform(keys[6], (num_sensors, ndim), dtype=dtype)
    domain_size = jnp.arange(2, ndim + 2, dtype=dtype)
    periodic = jnp.asarray([(axis % 2) == 0 for axis in range(ndim)])
    return xt, pt, Mt, At, omega, sensors, domain_size, periodic


def _reference_sum(xt, pt, Mt, At, omega, sensors, domain_size, periodic):
    delta = sensors[None, None, :, :] - xt[:, :, None, :]
    delta = delta - domain_size * jnp.round(delta / domain_size) * periodic
    linear = jnp.einsum(
        "btsd,btd->bts",
        delta,
        pt,
        precision=jax.lax.Precision.HIGHEST,
    )
    real_quadratic = 0.5 * jnp.einsum(
        "btsi,btij,btsj->bts",
        delta,
        jnp.real(Mt),
        delta,
        precision=jax.lax.Precision.HIGHEST,
    )
    imag_quadratic = 0.5 * jnp.einsum(
        "btsi,btij,btsj->bts",
        delta,
        jnp.imag(Mt),
        delta,
        precision=jax.lax.Precision.HIGHEST,
    )
    amplitudes = At[..., 0]
    contributions = (
        2.0
        * jnp.abs(amplitudes)[:, :, None]
        * jnp.cos(
            omega[:, None, None] * (linear + real_quadratic)
            + core.safe_angle_eps(amplitudes)[:, :, None]
        )
        * jnp.exp(-omega[:, None, None] * imag_quadratic)
    )
    return jnp.sum(contributions, axis=0)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_pallas_accumulator_matches_reference_with_padding(ndim):
    args = _trajectory_case(ndim, num_sensors=11)
    actual = sum_gaussian_beam_real_pallas(
        *args, config=PallasConfig(sensor_block_size=8), interpret=True
    )
    expected = _reference_sum(*args)

    assert actual.shape == (3, 11)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_pallas_accumulator_preserves_float64_accuracy():
    args = _trajectory_case(2, dtype=jnp.float64, num_sensors=5)
    actual = sum_gaussian_beam_real_pallas(
        *args, config=PallasConfig(sensor_block_size=4), interpret=True
    )
    expected = _reference_sum(*args)

    assert actual.dtype == jnp.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_xla_accumulator_requests_highest_contraction_precision_by_default():
    jaxpr = str(
        jax.make_jaxpr(core.sum_gaussian_beam_real_trajectories_xla)(
            *_trajectory_case(2, num_sensors=5)
        )
    )

    assert "precision=(Precision.HIGHEST, Precision.HIGHEST)" in jaxpr


def test_pallas_accumulator_matches_reference_for_large_physical_phases():
    args = list(_trajectory_case(2, num_sensors=17))
    args[1] = args[1] * (2.0 * jnp.pi)
    args[2] = jnp.real(args[2]) * 8.0 + 1j * jnp.imag(args[2]) * 0.2
    args[4] = 5.0 + 8.0 * (args[4] - 0.5)

    actual = sum_gaussian_beam_real_pallas(
        *args,
        config=PallasConfig(sensor_block_size=8),
        interpret=True,
    )
    expected = _reference_sum(*args)
    relative_error = jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected)

    assert float(relative_error) < 2e-5


@pytest.mark.parametrize(
    "periodic_axes",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_static_periodic_specialization_matches_generic(periodic_axes):
    args = list(_trajectory_case(2, num_sensors=11))
    args[-1] = jnp.asarray(periodic_axes)
    generic = sum_gaussian_beam_real_pallas(
        *args,
        config=PallasConfig(sensor_block_size=8),
        interpret=True,
    )
    specialized = sum_gaussian_beam_real_pallas(
        *args,
        config=PallasConfig(
            sensor_block_size=8,
            periodic_axes=periodic_axes,
        ),
        interpret=True,
    )

    np.testing.assert_allclose(specialized, generic, rtol=5e-7, atol=1e-6)


def test_pallas_accumulator_adds_nonzero_initial_field_and_can_keep_padding():
    args = _trajectory_case(2, num_sensors=11)
    initial = jnp.arange(33, dtype=jnp.float32).reshape(3, 11) / 10
    expected = _reference_sum(*args) + initial

    actual = sum_gaussian_beam_real_pallas(
        *args,
        config=PallasConfig(sensor_block_size=8),
        interpret=True,
        initial_field=initial,
    )
    padded = sum_gaussian_beam_real_pallas(
        *args,
        config=PallasConfig(sensor_block_size=8),
        interpret=True,
        initial_field=initial,
        return_padded=True,
    )

    assert padded.shape == (3, 16)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(actual, padded[:, :11])


def test_aliased_initial_field_differentiates_through_outer_scan():
    raw = _trajectory_case(2, num_sensors=5)
    xt, pt, Mt, At, omega, sensors, domain_size, periodic = raw
    batches = tuple(
        value[:4].reshape((2, 2) + value.shape[1:]) for value in (xt, pt, Mt, At, omega)
    )
    config = PallasConfig(sensor_block_size=4)

    def loss(initial):
        def body(carry, batch):
            batch_xt, batch_pt, batch_Mt, batch_At, batch_omega = batch
            result = sum_gaussian_beam_real_pallas(
                batch_xt,
                batch_pt,
                batch_Mt,
                batch_At,
                batch_omega,
                sensors,
                domain_size,
                periodic,
                config=config,
                interpret=True,
                initial_field=carry,
                return_padded=True,
            )
            return result, None

        result = jax.lax.scan(body, initial, batches)[0]
        return jnp.sum(result[:, : sensors.shape[0]])

    initial = jnp.zeros((xt.shape[1], 8), dtype=xt.dtype)
    gradient = jax.jit(jax.grad(loss))(initial)
    expected = jnp.pad(
        jnp.ones((xt.shape[1], sensors.shape[0]), dtype=xt.dtype),
        ((0, 0), (0, 8 - sensors.shape[0])),
    )
    np.testing.assert_array_equal(gradient, expected)


def test_pallas_jvp_and_vjp_match_the_streaming_reference():
    args = _trajectory_case(2, num_sensors=5)
    xt, *rest = args
    tangent = jnp.full_like(xt, 0.01)

    def pallas_fn(positions):
        return sum_gaussian_beam_real_pallas(
            positions,
            *rest,
            config=PallasConfig(
                sensor_block_size=4,
                rounding_mode="half_open",
                periodic_axes=(True, False),
            ),
            interpret=True,
        )

    def reference_fn(positions):
        return core.sum_gaussian_beam_real_trajectories_xla(positions, *rest)

    pallas_jvp = jax.jvp(pallas_fn, (xt,), (tangent,))[1]
    reference_jvp = jax.jvp(reference_fn, (xt,), (tangent,))[1]
    pallas_output, pallas_pullback = jax.vjp(pallas_fn, xt)
    reference_output, reference_pullback = jax.vjp(reference_fn, xt)
    cotangent = jnp.ones_like(pallas_output)

    np.testing.assert_allclose(pallas_jvp, reference_jvp, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        pallas_pullback(cotangent)[0],
        reference_pullback(jnp.ones_like(reference_output))[0],
        rtol=2e-5,
        atol=2e-6,
    )


def test_pallas_terminal_tr_evaluates_only_the_last_state():
    args = _trajectory_case(2, num_sensors=5)
    xt, pt, Mt, At, omega, sensors, domain_size, periodic = args

    def trajectory_solver(*unused_args):
        return xt, pt, Mt, At

    result = core.compute_gaussian_beam_real_TR_pallas_terminal(
        x0=xt[:, 0],
        p0=pt[:, 0],
        M0=Mt[:, 0],
        a0=At[:, 0, 0],
        omega0=omega,
        mode=jnp.ones((xt.shape[0],)),
        c=lambda x: jnp.ones(x.shape[:-1]),
        lam=0.0,
        ts=jnp.broadcast_to(jnp.arange(xt.shape[1]), (xt.shape[0], xt.shape[1])),
        sensors=sensors,
        domain_size=domain_size,
        periodic=periodic,
        ode_solver=trajectory_solver,
    )
    xla_terminal = core.compute_gaussian_beam_real_TR_xla_terminal(
        x0=xt[:, 0],
        p0=pt[:, 0],
        M0=Mt[:, 0],
        a0=At[:, 0, 0],
        omega0=omega,
        mode=jnp.ones((xt.shape[0],)),
        c=lambda x: jnp.ones(x.shape[:-1]),
        lam=0.0,
        ts=jnp.broadcast_to(jnp.arange(xt.shape[1]), (xt.shape[0], xt.shape[1])),
        sensors=sensors,
        domain_size=domain_size,
        periodic=periodic,
        ode_solver=trajectory_solver,
    )
    expected = _reference_sum(
        xt[:, -1:],
        pt[:, -1:],
        Mt[:, -1:],
        At[:, -1:],
        omega,
        sensors,
        domain_size,
        periodic,
    )[0]

    assert result.shape == (sensors.shape[0],)
    np.testing.assert_allclose(xla_terminal, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(result, expected, rtol=2e-5, atol=2e-6)


def test_terminal_batch_ode_solver_matches_last_full_saved_state():
    num_beams, ndim = 2, 1
    x0 = jnp.asarray([[0.2], [0.7]])
    p0 = jnp.asarray([[1.0], [-1.0]])
    M0 = jnp.full((num_beams, ndim, ndim), 0.1 + 0.3j)
    a0 = jnp.asarray([1.0 + 0.2j, 0.5 - 0.1j])
    mode = jnp.asarray([1.0, -1.0])
    ts = jnp.asarray([[0.0, 0.1], [0.0, 0.12]])
    config = gb_solvers.SolverConfig(dt0=0.01, rtol=1e-5, atol=1e-7)

    def c(x):
        return 1.0 + 0.0 * x[..., 0]

    full = gb_solvers.solve_ODE_batch_t(x0, p0, M0, a0, mode, ts, c, 0.0, config)
    terminal = gb_solvers.solve_ODE_batch_t_terminal(
        x0, p0, M0, a0, mode, ts, c, 0.0, config
    )

    for full_value, terminal_value in zip(full, terminal):
        assert terminal_value.shape[1] == 1
        np.testing.assert_allclose(
            terminal_value,
            full_value[:, -1:],
            rtol=1e-6,
            atol=1e-7,
        )


def test_pallas_sum_method_is_explicitly_selectable():
    config = PallasConfig(sensor_block_size=64)
    solver = MSGBSolver(
        thr=4,
        thr_strat="top_n",
        batch_size=4,
        input_type="spatial",
        ode_solver=gb_solvers.solve_ODE_base,
        sum_method="pallas_real",
        pallas_config=config,
    )

    assert solver.use_real
    assert solver.aggregate_method == "pallas"
    assert solver.pallas_config == config


def test_pallas_sum_method_rejects_sharding():
    sharding = ShardingStrategy(jax.sharding.Mesh(np.asarray(jax.devices()), ("x",)))
    with pytest.raises(ValueError, match="cannot be combined with sharding"):
        MSGBSolver(
            thr=4,
            thr_strat="top_n",
            batch_size=4,
            input_type="spatial",
            ode_solver=gb_solvers.solve_ODE_base,
            sum_method="pallas_real",
            sharding=sharding,
        )


def test_solver_rejects_static_periodic_config_that_disagrees_with_domain():
    solver = MSGBSolver(
        thr=4,
        thr_strat="top_n",
        batch_size=4,
        input_type="spatial",
        ode_solver=gb_solvers.solve_ODE_base,
        sum_method="pallas_real",
        pallas_config=PallasConfig(periodic_axes=(True, False)),
    )

    with pytest.raises(ValueError, match="does not match domain.periodic"):
        solver._effective_pallas_config((False, False))


def test_pallas_forward_aggregation_matches_scan_plumbing():
    num_batches, batch_size, num_times, ndim = 2, 3, 2, 2
    keys = jax.random.split(jax.random.key(91), 6)
    p0 = 0.2 * jax.random.normal(keys[0], (num_batches, batch_size, ndim))
    x0 = jax.random.uniform(keys[1], (num_batches, batch_size, ndim))
    M0_real = 0.1 * jax.random.normal(keys[2], (num_batches, batch_size, ndim, ndim))
    M0 = M0_real + 0.3j * jnp.eye(ndim)[None, None, :, :]
    omega = 0.5 + jax.random.uniform(keys[3], (num_batches, batch_size))
    a0 = jax.random.normal(keys[4], (num_batches, batch_size)) + 0.2j
    mode = jnp.ones((num_batches, batch_size))
    params = (p0, M0, x0, omega, a0, mode)
    ts = jnp.linspace(0.0, 0.2, num_times)
    sensors = jax.random.uniform(keys[5], (7, ndim))

    def trajectory_solver(x0, p0, M0, a0, mode, ts, *unused_args):
        del mode
        nt = ts.shape[-1]
        xt = x0[:, None, :] + ts.reshape((1, nt, 1)) * p0[:, None, :]
        pt = jnp.broadcast_to(p0[:, None, :], xt.shape)
        Mt = jnp.broadcast_to(M0[:, None, :, :], (x0.shape[0], nt, ndim, ndim))
        At = jnp.broadcast_to(a0[:, None, None], (x0.shape[0], nt, 1))
        return xt, pt, Mt, At

    common_args = (
        params,
        lambda x: jnp.ones(x.shape[:-1]),
        0.0,
        ts,
        trajectory_solver,
        sensors,
        jnp.array([2.0, 3.0]),
        jnp.array([True, False]),
    )
    scan_result = compute_forward_result(
        *common_args, use_real=True, aggregate_method="scan"
    )
    aliased_result = compute_forward_result(
        *common_args,
        use_real=True,
        aggregate_method="pallas",
        pallas_config=PallasConfig(
            sensor_block_size=8,
        ),
    )
    np.testing.assert_allclose(aliased_result, scan_result, rtol=2e-5, atol=2e-6)


def test_pallas_terminal_batch_carry_matches_xla_scan():
    num_batches, batch_size, ndim = 2, 3, 2
    keys = jax.random.split(jax.random.key(193), 5)
    p0 = 0.5 + jax.random.normal(keys[0], (num_batches, batch_size, ndim))
    x0 = jax.random.uniform(keys[1], (num_batches, batch_size, ndim))
    M0 = (
        0.02 * jax.random.normal(keys[2], (num_batches, batch_size, ndim, ndim))
        + 0.3j * jnp.eye(ndim)[None, None]
    )
    omega = 0.5 + jax.random.uniform(keys[3], (num_batches, batch_size))
    a0 = jax.random.normal(keys[4], (num_batches, batch_size, 1)) + 0.2j
    mode = -jnp.ones((num_batches, batch_size, 1))
    ts = jnp.broadcast_to(jnp.asarray([-0.2, 0.0]), (num_batches, batch_size, 2))
    params = (p0, M0, x0, omega, a0, mode, ts)
    sensors = jnp.linspace(0.0, 1.0, 14).reshape(7, ndim)

    def c(x):
        return jnp.ones(x.shape[:-1])

    domain_size = jnp.asarray([2.0, 3.0])
    periodic = jnp.asarray([False, False])
    expected = compute_TR_result(
        params,
        c,
        0.0,
        sensors,
        domain_size,
        periodic,
        ode_solver=gb_solvers.solve_hom_TR,
        aggregate_method="scan",
    )
    actual = compute_TR_result(
        params,
        c,
        0.0,
        sensors,
        domain_size,
        periodic,
        ode_solver=gb_solvers.solve_hom_TR,
        aggregate_method="pallas",
        pallas_config=PallasConfig(
            sensor_block_size=8,
            periodic_axes=(False, False),
        ),
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_pallas_cpu_requires_interpret_mode():
    if jax.default_backend() != "cpu":
        pytest.skip("CPU-specific backend guard")
    args = _trajectory_case(1, num_sensors=3)
    with pytest.raises(RuntimeError, match="no CPU code-generation backend"):
        sum_gaussian_beam_real_pallas(
            *args, config=PallasConfig(sensor_block_size=4), interpret=False
        )


def test_pallas_tpu_layout_matches_reference_with_beam_and_sensor_padding(
    monkeypatch,
):
    args = _trajectory_case(2, num_sensors=131)
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")

    actual = sum_gaussian_beam_real_pallas(
        *args, config=PallasConfig(sensor_block_size=128), interpret=True
    )
    expected = _reference_sum(*args)

    assert actual.shape == (3, 131)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize(
    ("ndim", "grid_order"),
    [
        (1, "time_sensor"),
        (2, "time_sensor"),
        (2, "sensor_time"),
        (3, "time_sensor"),
    ],
)
def test_pallas_tpu_time_sensor_layout_matches_reference_with_all_padding(
    monkeypatch, ndim, grid_order
):
    args = _trajectory_case(ndim, num_sensors=131)
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    config = PallasConfig(
        sensor_block_size=128,
        tpu_layout="time_sensor",
        tpu_beam_block_size=4,
        tpu_time_block_size=8,
        tpu_grid_order=grid_order,
    )

    actual = sum_gaussian_beam_real_pallas(*args, config=config, interpret=True)
    expected = _reference_sum(*args)

    assert actual.shape == (3, 131)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_pallas_tpu_time_sensor_layout_accepts_an_initial_field(monkeypatch):
    args = _trajectory_case(2, num_sensors=131)
    initial = jnp.linspace(0.0, 1.0, 3 * 131).reshape(3, 131)
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    config = PallasConfig(
        sensor_block_size=128,
        tpu_layout="time_sensor",
        tpu_beam_block_size=4,
        tpu_time_block_size=8,
        tpu_grid_order="sensor_time",
    )

    actual = sum_gaussian_beam_real_pallas(
        *args,
        config=config,
        interpret=True,
        initial_field=initial,
        return_padded=True,
    )
    expected = _reference_sum(*args) + initial

    assert actual.shape == (3, 256)
    np.testing.assert_allclose(actual[:, :131], expected, rtol=2e-5, atol=2e-6)


def test_pallas_tpu_beam_sensor_layout_accepts_initial_field_for_many_times(
    monkeypatch,
):
    args = _trajectory_case(2, num_sensors=131)
    initial = jnp.linspace(0.0, 1.0, 3 * 131).reshape(3, 131)
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")

    actual = sum_gaussian_beam_real_pallas(
        *args,
        config=PallasConfig(sensor_block_size=128),
        interpret=True,
        initial_field=initial,
    )

    np.testing.assert_allclose(
        actual, _reference_sum(*args) + initial, rtol=2e-5, atol=2e-6
    )


def test_pallas_tpu_requires_native_sensor_tile(monkeypatch):
    args = _trajectory_case(2, num_sensors=11)
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")

    with pytest.raises(ValueError, match="multiple of 128 on TPU"):
        sum_gaussian_beam_real_pallas(
            *args, config=PallasConfig(sensor_block_size=64), interpret=True
        )


def test_pallas_tpu_requires_float32(monkeypatch):
    args = _trajectory_case(2, dtype=jnp.float64, num_sensors=11)
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")

    with pytest.raises(ValueError, match="requires float32"):
        sum_gaussian_beam_real_pallas(
            *args, config=PallasConfig(sensor_block_size=128), interpret=True
        )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            PallasConfig(sensor_block_size=128, tpu_beam_block_size=4),
            "tpu_beam_block_size must be a multiple of 8",
        ),
        (
            PallasConfig(
                sensor_block_size=128,
                tpu_layout="time_sensor",
                tpu_time_block_size=4,
            ),
            "tpu_time_block_size must be a multiple of 8",
        ),
    ],
)
def test_pallas_tpu_rejects_non_native_reduction_tiles(monkeypatch, config, message):
    args = _trajectory_case(2, num_sensors=11)
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")

    with pytest.raises(ValueError, match=message):
        sum_gaussian_beam_real_pallas(*args, config=config, interpret=True)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sensor_block_size": 0}, "positive integer"),
        ({"gpu_num_warps": 3}, "gpu_num_warps"),
        ({"gpu_num_stages": 0}, "gpu_num_stages"),
        ({"rounding_mode": "unknown"}, "rounding_mode"),
        ({"tpu_layout": "unknown"}, "tpu_layout"),
        ({"tpu_grid_order": "unknown"}, "tpu_grid_order"),
        ({"periodic_axes": (True, 1)}, "periodic_axes"),
    ],
)
def test_pallas_config_rejects_invalid_static_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PallasConfig(**kwargs)


def test_pallas_static_periodic_rank_must_match_trajectory_dimension():
    args = _trajectory_case(2, num_sensors=5)
    with pytest.raises(ValueError, match="periodic_axes must have length 2"):
        sum_gaussian_beam_real_pallas(
            *args,
            config=PallasConfig(sensor_block_size=4, periodic_axes=(True,)),
            interpret=True,
        )


def test_low_level_pallas_config_must_have_the_static_config_type():
    args = _trajectory_case(2, num_sensors=5)
    with pytest.raises(TypeError, match="PallasConfig"):
        sum_gaussian_beam_real_pallas(
            *args,
            config={"sensor_block_size": 4},  # pyright: ignore[reportArgumentType]
            interpret=True,
        )


def test_forward_helper_rejects_complex_pallas_aggregation():
    dummy = jnp.zeros((1, 1), dtype=jnp.float32)
    params = (dummy, dummy, dummy, dummy, dummy, dummy)
    with pytest.raises(ValueError, match="only available for real-valued"):
        compute_forward_result(
            params,
            lambda x: jnp.ones(x.shape[:-1]),
            0.0,
            jnp.asarray([0.0]),
            gb_solvers.solve_hom_diag,
            jnp.zeros((1, 1)),
            jnp.ones((1,)),
            jnp.zeros((1,), dtype=bool),
            use_real=False,
            aggregate_method="pallas",
        )
