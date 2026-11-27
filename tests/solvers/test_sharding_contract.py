"""Direct multi-device contract tests for MSGB beam sharding."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from beamax.solvers.msgb_solvers.msgb_solver import ShardingStrategy


@pytest.fixture
def two_device_strategy():
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires two JAX devices")
    mesh = Mesh(np.asarray(devices[:2]), ("beam",))
    return ShardingStrategy(mesh=mesh, beam_axis="beam")


def _forward_params(count):
    return (
        jnp.ones((count, 2)),
        jnp.ones((count, 2, 2), dtype=jnp.complex64),
        jnp.ones((count, 2)),
        jnp.ones((count,)),
        jnp.ones((count,), dtype=jnp.complex64),
        jnp.ones((count,)),
    )


def test_forward_params_are_really_partitioned(two_device_strategy):
    sharded = two_device_strategy.shard_beam_params(*_forward_params(4))

    assert sharded[0].sharding.spec[0] == "beam"
    assert len(sharded[0].addressable_shards) == 2


def test_nondivisible_beam_count_fails_clearly(two_device_strategy):
    with pytest.raises(ValueError, match="not divisible"):
        two_device_strategy.shard_beam_params(*_forward_params(3))


@pytest.mark.parametrize("operation", ["time_reversal", "adjoint"])
def test_complex_surface_inverse_matches_sharded_plane(two_device_strategy, operation):
    from beamax import Domain, DyadicDecomposition, MSWPT
    from beamax.gb.gb_solvers import solve_hom_diag
    from beamax.solvers import MSGBSolver
    from beamax.surfaces import DetectorSurface

    d = Domain((8, 8), (0.125, 0.125), (False, False), c=1.0)
    plane = DetectorSurface.plane(2, spacing=(0.125,), shape=(8,))
    ts = jnp.arange(8) * 0.0625
    dd = Domain((8, 8), (0.0625, 0.125), (False, False))
    wpt = MSWPT(DyadicDecomposition(1, dd.N, (4,), (1, 1)), 2, "rectangular_mirror")
    solver = MSGBSolver(
        thr=16,
        thr_strat="top_n",
        batch_size=8,
        input_type="spatial",
        ode_solver=solve_hom_diag,
        sum_method="all_complex",
        sharding=two_device_strategy,
    )
    rng = np.random.default_rng(621)
    y = jnp.asarray(rng.normal(size=dd.N) + 1j * rng.normal(size=dd.N))
    points = d.grid[1:5, 2:5]
    solve = getattr(solver, operation)
    # Legacy and chart methods must propagate both frequency halves.
    reference = solve(y, d, points, plane.to_sensor(d), ts, dd, wpt)
    image = solve(y, d, detectors=plane, ts=ts, data_wpt=wpt, eval_points=points)
    assert np.iscomplexobj(image)
    np.testing.assert_allclose(image, reference, rtol=5e-5, atol=3e-6)
