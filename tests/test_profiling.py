import jax.numpy as jnp

from beamax.utils import profiling


class DummySync:
    def __init__(self):
        self.calls = 0

    def block_until_ready(self):
        self.calls += 1


def test_format_bytes_units():
    assert profiling.format_bytes(512) == "0.50 KB"
    assert profiling.format_bytes(5 * 1024**2) == "5.00 MB"
    assert profiling.format_bytes(3 * 1024**3) == "3.00 GB"


def test_array_info_metadata():
    arr = jnp.ones((2, 3), dtype=jnp.float32)
    info = profiling.array_info(arr, "x")

    assert info["name"] == "x"
    assert info["shape"] == (2, 3)
    assert info["dtype"] == "float32"
    assert info["size"] == 6
    assert info["memory_bytes"] == arr.size * arr.dtype.itemsize
    assert info["memory"] == "0.02 KB"


def test_profile_section_enabled_syncs(capsys):
    dummy = DummySync()
    with profiling.profile_section("unit-test", enabled=True, sync=[dummy]):
        pass

    out = capsys.readouterr().out
    assert "PROFILE START: unit-test" in out
    assert "PROFILE END: unit-test" in out
    assert dummy.calls == 1
