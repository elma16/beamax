import jax.numpy as jnp
import numpy as np
import pytest

from beamax import Domain, DyadicDecomposition, MSWPT, Sensor
from beamax.gb.gb_solvers import solve_hom_diag
from beamax.solvers import MSGBSolver
from beamax.surfaces import DetectorSurface


@pytest.fixture
def experiment():
    domain = Domain((16, 16), (0.125, 0.125), (False, False), c=0.75)
    wpt = MSWPT(DyadicDecomposition(1, domain.N, (4,), (1, 1)), 2, "rectangular_mirror")
    solver = MSGBSolver(
        thr=64,
        thr_strat="top_n",
        batch_size=32,
        input_type="spatial",
        ode_solver=solve_hom_diag,
        sum_method="scan_real",
    )
    p0 = jnp.exp(
        -jnp.sum((domain.grid - jnp.array([0.65, 0.875])) ** 2, axis=-1) / 0.05
    )
    ts = jnp.arange(16) * 0.0625
    return solver, domain, ts, wpt, p0


def test_msgb_same_voxel_receivers_have_distinct_traces(experiment):
    solver, domain, ts, wpt, p0 = experiment
    line = DetectorSurface.plane(
        2, value=0.9, origin=(0.13,), spacing=(0.02,), shape=(2,)
    )
    actual = solver.forward(p0, domain, detectors=line, ts=ts, wpt=wpt)
    np.testing.assert_allclose(
        actual, solver.forward(p0, domain, line.positions(), ts, wpt)
    )
    assert actual.shape == (16, 2)
    assert not np.allclose(actual[:, 0], actual[:, 1])


def test_msgb_grid_mask_order(experiment):
    solver, domain, ts, wpt, p0 = experiment
    ring = DetectorSurface.circle((1.0, 0.875), 0.5, 4)
    mapping = ring.to_mask(domain)
    actual = solver.forward(p0, domain, detectors=mapping.binary_mask, ts=ts, wpt=wpt)
    ref = solver.forward(
        p0, domain, Sensor(domain, binary_mask=mapping.binary_mask), ts, wpt
    )
    np.testing.assert_allclose(actual, ref)
    chart, _ = solver.forward_with_params(p0, domain, detectors=ring, ts=ts, wpt=wpt)
    np.testing.assert_allclose(
        chart, actual[:, mapping.mask_to_chart], rtol=2e-6, atol=5e-7
    )


def test_forward_accepts_ordered_points_and_sensor_objects(experiment):
    solver, domain, ts, wpt, p0 = experiment
    ring = DetectorSurface.circle((1.0, 0.875), 0.5, 4)
    coords = np.asarray(ring.positions())
    expected = solver.forward(p0, domain, ring.positions(), ts, wpt)
    for acquisition in (coords, Sensor(domain, positions=jnp.asarray(coords))):
        result = solver.forward(p0, domain, detectors=acquisition, ts=ts, wpt=wpt)
        np.testing.assert_allclose(result, expected)
    for invalid in (
        np.zeros((5, 3)),
        np.array([[np.nan, 0.0]]),
        DetectorSurface.point(0.125),
    ):
        with pytest.raises(ValueError):
            solver.forward(p0, domain, detectors=invalid, ts=ts, wpt=wpt)
    with pytest.raises(TypeError, match="only one"):
        solver.forward(p0, domain, ring.positions(), ts, wpt, detectors=ring)


def test_inversion_requires_an_oriented_regular_acquisition(experiment):
    solver, domain, ts, _, _ = experiment
    plane = DetectorSurface.plane(2, value=0.5, spacing=(0.125,), shape=(4,))
    wpt = MSWPT(DyadicDecomposition(1, (16, 4), (2,), (4, 1)), 2, "rectangular_mirror")
    y = np.ones((16, 4))

    def run(data=y, **kwargs):
        opts = dict(detectors=plane, ts=ts, data_wpt=wpt)
        opts.update(kwargs)
        return solver.time_reversal(data, domain, **opts)

    with pytest.raises(ValueError, match="orientation"):
        run(front=-1)
    with pytest.raises(ValueError, match="front"):
        run(detectors=plane.to_mask(domain).binary_mask)
    with pytest.raises(ValueError, match="complete planar"):
        run(detectors=np.eye(16, dtype=bool), front=1)
    irregular = np.zeros(domain.N, bool)
    irregular[4, [0, 1, 2, 5]] = True
    with pytest.raises(ValueError, match="uniform"):
        run(detectors=irregular, front=1)
    with pytest.raises(ValueError, match="nonnegative"):
        run(ts=ts - 1)
    with pytest.raises(ValueError, match="time-first"):
        run(y.T)
    with pytest.raises(ValueError, match="transform"):
        run(data_wpt=None)
    with pytest.raises(TypeError, match="acquisition"):
        run(sources=plane.to_sensor(domain))


@pytest.mark.parametrize("sensor", ["plane", "ring", "arc"])
def test_solver_time_reversal_matches_chart(experiment, sensor):
    from beamax.surfaces import surface_time_reversal

    solver, domain, ts, _, _ = experiment
    choices = dict(
        plane=DetectorSurface.plane(
            2, value=1.25, origin=(0.25,), spacing=(0.125,), shape=(4,), front=-1
        ),
        ring=DetectorSurface.circle((1.0, 0.875), 0.5, 4),
        arc=DetectorSurface.arc((1.0, 1.0), 0.5, 0.2, 1.7, 4),
    )
    surface = choices[sensor]
    dd = Domain((16, 4), (0.0625, *surface.spacing), (False, *surface.periodic))
    wpt = MSWPT(DyadicDecomposition(1, dd.N, (2,), (4, 1)), 2, "rectangular_mirror")
    y = jnp.asarray(np.random.default_rng(299).normal(size=dd.N))
    points = domain.grid[4:8, 6:9]
    opts = dict(detectors=surface, ts=ts, data_wpt=wpt, eval_points=points)
    image, diag = solver.time_reversal(y, domain, **opts, return_diagnostics=True)
    ref, expected = surface_time_reversal(
        y, dd, wpt, surface, domain, 64, eval_points=points, batch_size=32
    )
    np.testing.assert_allclose(image, ref, rtol=2e-5, atol=3e-6)
    np.testing.assert_allclose(diag["t_gamma"], expected["t_gamma"])
    if sensor == "plane":
        opts.update(detectors=surface.to_mask(domain).binary_mask, front=-1)
        np.testing.assert_allclose(
            solver.time_reversal(y, domain, **opts), image, rtol=2e-5, atol=3e-6
        )
