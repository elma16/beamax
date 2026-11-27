import jax.numpy as jnp
import numpy as np
import pytest

from beamax import Domain, DyadicDecomposition, MSWPT, Sensor
from beamax.gb.gb_solvers import solve_hom_diag
from beamax.solvers import MSGBSolver
from beamax.surfaces import DetectorSurface


@pytest.fixture
def experiment():
    domain = Domain((16, 16), (0.125, 0.125), (False, False))
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
    np.testing.assert_allclose(chart, actual[:, mapping.mask_to_chart])
