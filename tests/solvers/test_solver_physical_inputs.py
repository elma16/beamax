"""Physical initial data and acquisition-coordinate regressions."""

import jax.numpy as jnp
import numpy as np
import pytest

from beamax.decomposition import DyadicDecomposition
from beamax.gb.gb_solvers import solve_hom_diag
from beamax.geometry import Domain
from beamax.solvers import MSGBSolver
from beamax.transforms import MSWPT


def transform(shape, boxes=4):
    decomp = DyadicDecomposition(
        num_levels=1,
        N=shape,
        num_boxes_levels=(boxes,),
        box_aspect_ratio=(1,) * len(shape),
    )
    return MSWPT(decomp, redundancy=2, windowing="rectangular_mirror")


def solver(method, *, hard=False):
    return MSGBSolver(
        thr=0.1 if hard else 10000,
        thr_strat="hard" if hard else "top_n",
        batch_size=8,
        input_type="spatial",
        ode_solver=solve_hom_diag,
        sum_method=method,
    )


def test_real_initial_velocity_and_complex_renderer():
    d = Domain(N=(32,), dx=(1.0,), c=1.0, periodic=(True,))
    w = transform(d.N)
    t = jnp.array([0.0, 0.1, 0.2])
    v = jnp.sin(2 * jnp.pi * 6 * jnp.arange(32) / 32)
    zero = jnp.zeros_like(v)
    real = solver("all_real").forward(zero, d, d.grid, t, w, dpdt=v)
    complex_solver = solver("all_complex")
    full = complex_solver.forward(zero.astype(complex), d, d.grid, t, w, dpdt=v)
    np.testing.assert_allclose(real[0], 0.0, atol=2e-6)
    np.testing.assert_allclose(real, full.real, atol=2e-6)
    pressure = complex_solver.forward(v, d, d.grid, t, w)
    cast = complex_solver.forward(v.astype(complex), d, d.grid, t, w)
    np.testing.assert_allclose(pressure, cast, atol=2e-6)
    # A sample request beginning later still uses the physical initial time zero.
    np.testing.assert_allclose(
        complex_solver.forward(v, d, d.grid, t[1:], w), pressure[1:], atol=2e-6
    )


def test_odd_box_width_uses_complete_frame():
    d = Domain(N=(12,), dx=(1.0,), c=1.0, periodic=(True,))
    w = transform(d.N)
    assert not w.has_conjugate_centres
    p0 = jnp.ones(d.N)
    args = (d, d.grid, jnp.array([0.0]), w)
    real = solver("all_real").forward(p0, *args)
    full = solver("all_complex").forward(p0, *args)
    np.testing.assert_allclose(real, full.real, atol=2e-6)


@pytest.mark.parametrize("method", ["scan_real", "scan_complex", "pallas_real"])
def test_empty_forward_selection_preserves_output_shape(method):
    d = Domain(N=(16,), dx=(0.1,), c=1.0, periodic=(False,))
    w = transform(d.N)
    t = jnp.arange(16) * 0.1
    signal = solver(method, hard=True).forward(jnp.zeros(d.N), d, d.grid, t, w)
    assert signal.shape == (16, 16)
    np.testing.assert_array_equal(signal, 0.0)
