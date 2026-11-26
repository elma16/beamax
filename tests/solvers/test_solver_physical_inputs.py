"""Physical initial data and acquisition-coordinate regressions."""

import jax.numpy as jnp
import numpy as np
import pytest

from beamax.decomposition import DyadicDecomposition
from beamax.gb.gb_solvers import solve_hom_diag
from beamax.geometry import Domain, Sensor
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


@pytest.mark.parametrize("method", ["all_complex", "scan_complex"])
@pytest.mark.parametrize("op", ["time_reversal", "adjoint"])
def test_inverse_preserves_complex_phase(method, op):
    d = Domain(N=(16,), dx=(0.1,), c=1.0, periodic=(False,))
    w = transform(d.N, boxes=2)
    t = jnp.arange(16) * 0.1
    residual = jnp.sin(2 * jnp.pi * 4 * jnp.arange(16) / 16)
    source = Sensor(domain=d, positions=jnp.array([[0.0]]))
    operation = getattr(solver(method), op)
    # The acquisition grid may have more time samples than its spectrally cropped record.
    acquisition_times = jnp.linspace(0.0, t[-1], 2 * len(t) - 1)
    args = (d, d.grid, source, acquisition_times, d, w)
    real = operation(residual.astype(complex), *args)
    rotated = operation(1j * residual, *args)
    assert float(jnp.max(jnp.abs(real))) > 0.1
    np.testing.assert_allclose(rotated, 1j * real, atol=2e-6)


@pytest.mark.parametrize("method", ["scan_real", "scan_complex", "pallas_real"])
def test_empty_inverse_selection_preserves_output_shape(method):
    d = Domain(N=(16,), dx=(0.1,), c=1.0, periodic=(False,))
    w = transform(d.N)
    t = jnp.arange(16) * 0.1
    source = Sensor(domain=d, positions=jnp.array([[0.0]]))
    op = solver(method, hard=True)
    for value, shape in [
        (op.time_reversal(jnp.zeros(d.N), d, d.grid, source, t, d, w), d.N),
        (op.adjoint(jnp.zeros(d.N), d, d.grid, source, t, d, w), d.N),
    ]:
        assert value.shape == shape
        np.testing.assert_array_equal(value, 0.0)


def test_planar_launch_uses_acquisition_origin_and_normal_axis():
    image = Domain(N=(32, 32), dx=(1.0, 1.0), c=1.0, periodic=(False, False))
    data_domain = Domain(N=(16, 16), dx=(0.1, 1.0), c=1.0, periodic=(False, False))
    w = transform(data_domain.N, boxes=2)
    samples = jnp.arange(16.0)
    time = 0.1 * samples
    data = jnp.cos(2 * jnp.pi * (3 * samples[:, None] + samples[None, :]) / 16)
    op = MSGBSolver(1, "top_n", 1, "spatial", solve_hom_diag, "all_real")

    def launch(positions):
        detector = Sensor(domain=image, positions=positions)
        _, params = op.time_reversal_with_params(
            data, image, image.grid[:1, :1], detector, time, data_domain, w
        )
        return [np.asarray(x).squeeze() for x in params[:3]]

    positions = jnp.stack((jnp.zeros(16), samples), axis=1)
    p, hessian, x = launch(positions)
    _, _, shifted = launch(positions + jnp.array([0.0, 10.0]))
    rotated_p, rotated_hessian, rotated_x = launch(positions[:, ::-1])
    np.testing.assert_allclose(shifted - x, [0.0, 10.0], atol=2e-6)
    np.testing.assert_allclose(rotated_x, x[::-1], atol=2e-6)
    np.testing.assert_allclose(rotated_p, p[::-1], atol=2e-6)
    np.testing.assert_allclose(rotated_hessian, hessian[::-1, ::-1], atol=2e-6)


@pytest.mark.parametrize(
    "problem, message",
    [
        ("time_origin", "start at zero"),
        ("spacing", "Detector spacing"),
        ("channel_order", "Cartesian grid order"),
        ("missing_detector_axis", "Boundary data detector axes"),
    ],
)
def test_inverse_validates_acquisition_coordinates(problem, message):
    image = Domain(N=(32, 32), dx=(1.0, 1.0), c=1.0, periodic=(False, False))
    times = jnp.arange(16) * 0.1
    positions = jnp.stack((jnp.zeros(16), jnp.arange(16)), axis=1)
    spacing = 2.0 if problem == "spacing" else 1.0
    acquisition = Domain(N=(16, 16), dx=(0.1, spacing), c=1.0, periodic=(False, False))
    if problem == "time_origin":
        times = times + 0.1
    if problem == "channel_order":
        positions = positions[::-1]
    if problem == "missing_detector_axis":
        acquisition = Domain(N=(16,), dx=(0.1,), c=1.0, periodic=(False,))
    source = Sensor(domain=image, positions=positions)

    with pytest.raises(ValueError, match=message):
        solver("all_real").time_reversal(
            jnp.zeros(acquisition.N),
            image,
            image.grid[:1, :1],
            source,
            times,
            acquisition,
            transform(acquisition.N, boxes=2),
        )
