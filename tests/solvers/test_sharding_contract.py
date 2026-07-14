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
