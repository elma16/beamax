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


def test_adjoint_pairing_units_and_invalid_measures(experiment):
    solver, domain, ts, _, _ = experiment
    chart = DetectorSurface.plane(2, value=0.5, spacing=(0.125,), shape=(4,))
    mask = chart.to_mask(domain).binary_mask
    wpt = MSWPT(DyadicDecomposition(1, (16, 4), (2,), (4, 1)), 2, "rectangular_mirror")
    data = np.random.default_rng(424).normal(size=(16, 4))
    base = dict(ts=ts, data_wpt=wpt, eval_points=domain.grid[2:4, 3:6])
    equal = solver.adjoint(data, domain, detectors=chart, pairing="element_sum", **base)
    euclidean = solver.adjoint(
        data, domain, detectors=mask, front=1, pairing="euclidean", **base
    )
    np.testing.assert_allclose(
        euclidean, equal * np.prod(domain.dx) / float(ts[1]), atol=2e-6, rtol=2e-5
    )
    for pairing, measures, pattern in [
        ("undefined", None, "Choose"),
        ("element_sum", 0.1, "only"),
        ("surface_area", None, "Supply"),
        ("surface_area", [1, 2], "one per"),
        ("surface_area", [1.0, 0.0, 2.0, 1.0], "positive"),
    ]:
        with pytest.raises(ValueError, match=pattern):
            solver.adjoint(
                data,
                domain,
                detectors=mask,
                front=1,
                pairing=pairing,
                element_measures=measures,
                **base,
            )
    with pytest.raises(ValueError, match="chart supplies"):
        solver.adjoint(data, domain, detectors=chart, element_measures=0.125, **base)


def test_solver_adjoint_preserves_pairings_and_window(experiment):
    from beamax.surfaces import surface_adjoint

    solver, domain, ts, _, _ = experiment
    surface = DetectorSurface.plane(
        2, value=1.25, origin=(0.25,), spacing=(0.125,), shape=(4,), front=-1
    )
    dd = Domain((16, 4), (0.0625, *surface.spacing), (False, False))
    wpt = MSWPT(DyadicDecomposition(1, dd.N, (2,), (4, 1)), 2, "rectangular_mirror")
    y = jnp.asarray(np.random.default_rng(81).normal(size=dd.N))
    opts = dict(
        ts=ts,
        data_wpt=wpt,
        eval_points=domain.grid[5:8, 4:7],
        window=jnp.hanning(16),
        guard=0.2,
    )
    image, diag = solver.adjoint(
        y, domain, detectors=surface, **opts, return_diagnostics=True
    )
    ref, _ = surface_adjoint(
        y,
        dd,
        wpt,
        surface,
        domain,
        64,
        window=opts["window"],
        guard=0.2,
        eval_points=opts["eval_points"],
        batch_size=32,
    )
    assert diag["adjoint_formula"] == "general"
    np.testing.assert_allclose(image, ref, rtol=2e-5, atol=2e-6)
    mask = surface.to_mask(domain).binary_mask
    np.testing.assert_allclose(
        solver.adjoint(
            y,
            domain,
            detectors=mask,
            front=-1,
            pairing="surface_area",
            element_measures=0.125,
            **opts,
        ),
        ref,
        rtol=2e-5,
        atol=2e-6,
    )
    with pytest.raises(ValueError, match="pairing"):
        solver.adjoint(y, domain, detectors=mask, front=-1, **opts)


@pytest.mark.parametrize(
    "pairing,reference_pairing",
    [("surface_area", "arc_length"), ("element_sum", "element_sum")],
)
def test_circular_adjoint_matches_specialization(
    experiment, pairing, reference_pairing
):
    from beamax.surfaces import ring_adjoint

    solver, domain, ts, _, _ = experiment
    surface = DetectorSurface.circle((1.0, 0.875), 0.5, 4, name="instrument")
    dd = Domain((16, 4), (0.0625, *surface.spacing), (False, True))
    wpt = MSWPT(DyadicDecomposition(1, dd.N, (2,), (4, 1)), 2, "rectangular_mirror")
    y = np.random.default_rng(3).normal(size=dd.N)
    taper = np.hanning(16)
    points = domain.grid[3:5, 6:9]
    image, diag = solver.adjoint(
        y,
        domain,
        detectors=surface,
        ts=ts,
        data_wpt=wpt,
        eval_points=points,
        window=taper,
        pairing=pairing,
        return_diagnostics=True,
    )
    ref, _ = ring_adjoint(
        y * taper[:, None],
        dd,
        wpt,
        domain,
        64,
        centre=(1.0, 0.875),
        radius=0.5,
        pairing=reference_pairing,
        eval_points=points,
        batch_size=32,
    )
    assert diag["adjoint_formula"] == "circular"
    np.testing.assert_allclose(image, ref, atol=2e-6, rtol=2e-5)


def test_specialization_requires_eligible_geometry_and_exterior(experiment):
    from beamax.surfaces.ring import circular_boundary_speed

    _, domain, _, _, _ = experiment
    for chart in [
        DetectorSurface.circle((1.0, 1.0), 0.5, 8, front="outside"),
        DetectorSurface.ellipse((1.0, 1.0), 0.6, 0.5, 8, name="circle"),
        DetectorSurface.arc((1.0, 1.0), 0.5, 0.0, 1.5, 8),
    ]:
        assert circular_boundary_speed(chart, domain, None) is None
    variable = Domain(
        domain.N, domain.dx, domain.periodic, c=lambda x: 1.0 + 0.2 * x[..., 0]
    )
    ring = DetectorSurface.circle((1.0, 1.0), 0.5, 8)
    assert circular_boundary_speed(ring, variable, 1.0) is None
    assert circular_boundary_speed(ring, variable, None) is None


def test_kwave_rejects_ambiguous_input_and_off_grid_image_points(experiment):
    from beamax.solvers import KWaveSolver

    solver, domain, ts, _, p0 = experiment
    kw = KWaveSolver(backend="python", pml_size=0, smooth_p0=False)
    chart = DetectorSurface.circle((1.0, 0.875), 0.5, 4)
    mask = chart.to_mask(domain).binary_mask
    data = np.zeros((len(ts), 4))
    with pytest.raises(TypeError, match="only one"):
        kw.forward(p0, domain, mask, ts, detectors=chart)
    for method in (kw.time_reversal, kw.adjoint):
        with pytest.raises(ValueError, match="time-first"):
            method(data.T, domain, detectors=chart, ts=ts)
        with pytest.raises(TypeError, match="time-first"):
            method(data, domain, detectors=chart, ts=ts, data_layout="ns_nt")
        with pytest.raises(ValueError, match="start at zero"):
            method(data, domain, detectors=chart, ts=ts + ts[1])
    for points, pattern in [
        (np.array([[0.01, 0.125]]), "grid nodes"),
        (np.array([[0.0, np.nan]]), "finite"),
    ]:
        with pytest.raises(ValueError, match=pattern):
            kw.adjoint(data, domain, detectors=chart, ts=ts, eval_points=points)


def test_kwave_surface_matches_permuted_mask_and_weighted_adjoint():
    from beamax.solvers import KWaveSolver

    kw = KWaveSolver(backend="python", pml_size=0, smooth_p0=False)
    d = Domain((12, 12), (0.125,) * 2, (False, False), c=1.0)
    ring = DetectorSurface.circle((0.75, 0.625), 0.25, 4)
    mapping = ring.to_mask(d)
    ts = np.arange(4) * 0.025
    p0 = np.random.default_rng(71).normal(size=d.N)
    chart = kw.forward(p0, d, detectors=ring, ts=ts)
    mask = kw.forward(p0, d, mapping.binary_mask, ts)
    np.testing.assert_allclose(chart, mask[:, mapping.mask_to_chart])
    y = np.random.default_rng(92).normal(size=(4, 4))
    image_mask = np.ones(d.N, bool)
    args = (y[:, mapping.chart_to_mask], d, image_mask, mapping.binary_mask, ts)
    tr = kw.time_reversal(*args, data_layout="nt_ns")
    np.testing.assert_allclose(kw.time_reversal(y, d, detectors=ring, ts=ts), tr)
    euclidean = kw.adjoint(*args, data_layout="nt_ns")
    weight = ts[1] * (2 * np.pi * 0.25 / 4) / np.prod(d.dx)
    points = d.grid[2:5, 3:6]
    result = kw.adjoint(y, d, detectors=ring, ts=ts, eval_points=points)
    np.testing.assert_allclose(
        result, weight * euclidean[2:5, 3:6], atol=2e-6, rtol=2e-5
    )
