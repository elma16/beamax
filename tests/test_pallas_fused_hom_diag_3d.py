"""Oracle tests for fused 3D homogeneous propagation and accumulation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamax.gb import core, gb_solvers
from beamax.gb.pallas_config import PallasConfig
from beamax.gb.pallas_kernels import (
    sum_gaussian_beam_real_hom_diag_3d_pallas,
)


jax.config.update("jax_enable_x64", True)


def _initial_parameter_case(*, num_beams=3, num_times=3, num_sensors=5):
    keys = jax.random.split(jax.random.key(314), 7)
    x0 = 0.1 + 0.8 * jax.random.uniform(keys[0], (num_beams, 3), dtype=jnp.float32)
    p0 = jax.random.normal(keys[1], (num_beams, 3), dtype=jnp.float32)
    p0 = p0 + jnp.asarray([0.7, -0.2, 0.4], dtype=jnp.float32)
    alpha_imag = 0.2 + jax.random.uniform(keys[2], (num_beams, 3), dtype=jnp.float32)
    alpha_real = 0.04 * jax.random.normal(keys[3], (num_beams, 3), dtype=jnp.float32)
    alpha = alpha_real + 1j * alpha_imag
    M0 = jnp.eye(3, dtype=jnp.complex64)[None, :, :] * alpha[:, :, None]
    a0 = jax.random.normal(keys[4], (num_beams,), dtype=jnp.float32)
    a0 = a0 + 1j * jax.random.normal(keys[5], (num_beams,), dtype=jnp.float32)
    omega0 = jnp.linspace(0.7, 1.6, num_beams, dtype=jnp.float32)
    mode = jnp.where(jnp.arange(num_beams) % 2, -1.0, 1.0).astype(jnp.float32)
    ts = jnp.linspace(0.0, 0.31, num_times, dtype=jnp.float32)
    sensors = jax.random.uniform(keys[6], (num_sensors, 3), dtype=jnp.float32)
    domain_size = jnp.asarray([1.0, 1.4, 0.8], dtype=jnp.float32)
    periodic = jnp.asarray([True, False, True])
    c0 = jnp.asarray(1.3, dtype=jnp.float32)
    return (
        x0,
        p0,
        M0,
        a0,
        omega0,
        mode,
        c0,
        ts,
        sensors,
        domain_size,
        periodic,
    )


def _trajectory_oracle(*args):
    (
        x0,
        p0,
        M0,
        a0,
        omega0,
        mode,
        c0,
        ts,
        sensors,
        domain_size,
        periodic,
    ) = args
    xt, pt, Mt, At = gb_solvers.solve_hom_diag(
        x0,
        p0,
        M0,
        a0,
        mode,
        ts,
        lambda unused_position: c0,
    )
    xt = core.wrap_position(xt, domain_size, periodic)
    Mt = Mt.astype(jnp.complex64)
    return core.sum_gaussian_beam_real_trajectories_xla(
        xt,
        pt,
        Mt,
        At,
        omega0,
        sensors,
        domain_size,
        periodic,
    )


def test_fused_explicit_kernel_matches_solve_hom_diag_oracle_with_padding():
    args = _initial_parameter_case(num_beams=3, num_times=3, num_sensors=5)
    config = PallasConfig(sensor_block_size=4, gpu_time_block_size=2)

    actual = sum_gaussian_beam_real_hom_diag_3d_pallas(
        *args, config=config, interpret=True
    )
    expected = _trajectory_oracle(*args)

    assert actual.shape == (3, 5)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_fused_kernel_accumulates_batches_into_one_padded_carry():
    args = _initial_parameter_case(num_beams=4, num_times=3, num_sensors=5)
    config = PallasConfig(
        sensor_block_size=8,
        gpu_time_block_size=4,
    )
    beam_values = args[:6]
    shared_values = args[6:]
    carry = jnp.zeros((4, 8), dtype=jnp.float32)

    for start in (0, 2):
        batch_x0, batch_p0, batch_M0, batch_a0, batch_omega0, batch_mode = (
            value[start : start + 2] for value in beam_values
        )
        c0, ts, sensors, domain_size, periodic = shared_values
        carry = sum_gaussian_beam_real_hom_diag_3d_pallas(
            batch_x0,
            batch_p0,
            batch_M0,
            batch_a0,
            batch_omega0,
            batch_mode,
            c0,
            ts,
            sensors,
            domain_size,
            periodic,
            config=config,
            initial_field=carry,
            return_padded=True,
            interpret=True,
        )

    expected = _trajectory_oracle(*args)
    assert carry.shape == (4, 8)
    np.testing.assert_allclose(carry[:3, :5], expected, rtol=2e-5, atol=2e-6)


def test_fused_kernel_jvp_and_vjp_use_the_portable_physical_oracle():
    args = list(_initial_parameter_case(num_beams=2, num_times=3, num_sensors=4))
    args[-1] = jnp.zeros((3,), dtype=bool)
    x0 = args[0]
    rest = args[1:]
    config = PallasConfig(sensor_block_size=4, gpu_time_block_size=2)

    def fused(positions):
        return sum_gaussian_beam_real_hom_diag_3d_pallas(
            positions,
            *rest,
            config=config,
            interpret=True,
        )

    def reference(positions):
        return _trajectory_oracle(positions, *rest)

    tangent = jnp.full_like(x0, 0.013)
    fused_output, fused_jvp = jax.jvp(fused, (x0,), (tangent,))
    reference_output, reference_jvp = jax.jvp(reference, (x0,), (tangent,))
    _, fused_pullback = jax.vjp(fused, x0)
    _, reference_pullback = jax.vjp(reference, x0)
    cotangent = jnp.linspace(-0.2, 0.4, fused_output.size, dtype=jnp.float32).reshape(
        fused_output.shape
    )

    np.testing.assert_allclose(fused_output, reference_output, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(fused_jvp, reference_jvp, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        fused_pullback(cotangent)[0],
        reference_pullback(cotangent)[0],
        rtol=2e-5,
        atol=2e-6,
    )


def test_fused_kernel_static_periodic_specialization_matches_runtime_flags():
    args = _initial_parameter_case(num_beams=2, num_times=2, num_sensors=5)
    generic = sum_gaussian_beam_real_hom_diag_3d_pallas(
        *args,
        config=PallasConfig(sensor_block_size=4, gpu_time_block_size=2),
        interpret=True,
    )
    specialized = sum_gaussian_beam_real_hom_diag_3d_pallas(
        *args,
        config=PallasConfig(
            sensor_block_size=4,
            gpu_time_block_size=2,
            periodic_axes=(True, False, True),
        ),
        interpret=True,
    )

    np.testing.assert_allclose(specialized, generic, rtol=2e-5, atol=2e-6)


def test_fused_kernel_preserves_exact_half_domain_rounding_conventions():
    phase_offset = jnp.asarray(0.3, dtype=jnp.float32)
    x0 = jnp.zeros((1, 3), dtype=jnp.float32)
    p0 = jnp.asarray([[1.0, 0.0, 0.0]], dtype=jnp.float32)
    M0 = jnp.asarray([jnp.eye(3) * 0.4j], dtype=jnp.complex64)
    a0 = jnp.asarray([0.5 * jnp.exp(1j * phase_offset)], dtype=jnp.complex64)
    omega0 = jnp.ones((1,), dtype=jnp.float32)
    mode = jnp.ones((1,), dtype=jnp.float32)
    c0 = jnp.asarray(1.0, dtype=jnp.float32)
    ts = jnp.zeros((1,), dtype=jnp.float32)
    sensor_x = jnp.asarray([-1.5, -0.5, 0.5, 1.5], dtype=jnp.float32)
    sensors = jnp.stack(
        (sensor_x, jnp.zeros_like(sensor_x), jnp.zeros_like(sensor_x)), axis=1
    )
    domain_size = jnp.ones((3,), dtype=jnp.float32)
    periodic = jnp.asarray([True, False, False])
    outputs = {}

    for rounding_mode in ("nearest_even", "half_open"):
        image = (
            jnp.round(sensor_x)
            if rounding_mode == "nearest_even"
            else jnp.floor(sensor_x + 0.5)
        )
        displacement = sensor_x - image
        expected = (
            jnp.cos(displacement + phase_offset) * jnp.exp(-0.2 * displacement**2)
        )[None, :]
        outputs[rounding_mode] = sum_gaussian_beam_real_hom_diag_3d_pallas(
            x0,
            p0,
            M0,
            a0,
            omega0,
            mode,
            c0,
            ts,
            sensors,
            domain_size,
            periodic,
            config=PallasConfig(
                sensor_block_size=4,
                gpu_time_block_size=1,
                rounding_mode=rounding_mode,
                periodic_axes=(True, False, False),
            ),
            interpret=True,
        )
        np.testing.assert_allclose(
            outputs[rounding_mode], expected, rtol=2e-5, atol=2e-6
        )

    assert not np.allclose(outputs["nearest_even"], outputs["half_open"])


def test_fused_kernel_rejects_non_3d_and_non_float32_contracts():
    args = list(_initial_parameter_case(num_beams=2))

    with pytest.raises(ValueError, match=r"x0 must have shape \(B, 3\)"):
        sum_gaussian_beam_real_hom_diag_3d_pallas(
            args[0][:, :2], *args[1:], interpret=True
        )

    args[0] = args[0].astype(jnp.float64)
    with pytest.raises(ValueError, match="requires float32 x0"):
        sum_gaussian_beam_real_hom_diag_3d_pallas(*args, interpret=True)


def test_fused_kernel_accepts_other_gpu_models(monkeypatch):
    class FakeDevice:
        device_kind = "NVIDIA T4"

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(jax, "devices", lambda unused_backend=None: [FakeDevice()])
    result = sum_gaussian_beam_real_hom_diag_3d_pallas(
        *_initial_parameter_case(num_beams=2),
        interpret=True,
    )

    assert result.shape == (3, 5)


@pytest.mark.parametrize("gpu_time_block_size", [0, 3])
def test_pallas_config_validates_gpu_time_block_size(gpu_time_block_size):
    with pytest.raises(ValueError, match="gpu_time_block_size"):
        PallasConfig(gpu_time_block_size=gpu_time_block_size)
