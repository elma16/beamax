"""Initial velocity in a uniform periodic medium has a Fourier half-wave split."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamax import Domain, DyadicDecomposition, MSWPT
from beamax.gb import solve_hom_diag
from beamax.solvers import MSGBSolver
from beamax.utils import unitary_fft


def _periodic_problem(lam=0.0):
    n = 128
    domain = Domain((n,), (1 / n,), (True,), c=1.7, lam=lam)
    wpt = MSWPT(DyadicDecomposition(2, (n,), (4, 8), (1,)), 2, "rectangular_mirror")
    return domain, wpt


@pytest.mark.parametrize("data_space", ["spatial", "fourier"])
def test_velocity_of_off_centre_sine_has_correct_sign_and_amplitude(data_space):
    # Mode 20 lies in the packet centred at 24. Frozen polarisation would
    # leave an amplitude bias of 20/24 even at a large coefficient budget.
    with jax.enable_x64():
        domain, wpt = _periodic_problem()
        mode = 20
        spatial_wave = jnp.sin(2 * jnp.pi * mode * domain.grid[:, 0])
        omega = 2 * jnp.pi * mode * domain.c
        times = jnp.arange(3) * (jnp.pi / (4 * omega))
        expected = jnp.sin(omega * times[:, None]) * spatial_wave / omega
        velocity = (
            unitary_fft(spatial_wave) if data_space == "fourier" else spatial_wave
        )
        method = "all_complex" if data_space == "fourier" else "all_real"
        solver = MSGBSolver(1e-11, "hard", 32, data_space, solve_hom_diag, method)
        actual = solver.forward(
            jnp.zeros_like(velocity), domain, domain.grid, times, wpt, dpdt=velocity
        )
        amplitude = jnp.vdot(expected, actual) / jnp.vdot(expected, expected)
        np.testing.assert_allclose(amplitude, 1.0, atol=1e-10, rtol=0)
        # The spatial Gaussian synthesis also contributes small off-mode tails.
        assert jnp.linalg.norm(actual - expected) < 1e-5 * jnp.linalg.norm(expected)


@pytest.mark.parametrize("lam", [0.0, 0.7])
@pytest.mark.parametrize("data_space", ["spatial", "fourier"])
def test_uniform_velocity_including_damping(lam, data_space):
    with jax.enable_x64():
        domain, wpt = _periodic_problem(lam)
        times = jnp.array([0.1, 0.2])
        velocity = jnp.full(domain.N, 2.0)
        if data_space == "fourier":
            velocity = unitary_fft(velocity)
        solver = MSGBSolver(1e-8, "hard", 32, data_space, solve_hom_diag, "all_complex")
        actual = solver.forward(
            jnp.zeros_like(velocity), domain, domain.grid[:3], times, wpt, dpdt=velocity
        )
        scale = times if lam == 0 else -jnp.expm1(-lam * times) / lam
        np.testing.assert_allclose(
            actual, jnp.broadcast_to(2 * scale[:, None], (2, 3)), atol=1e-12
        )
