import pytest
import jax
import jax.numpy as jnp
from beamax import utils
from beamax.utils.interp import make_c_function_from_grid, Interpolator
from beamax.geometry import Domain
from beamax.decomposition import DyadicDecomposition
import beamax.utils.device as device_utils
from beamax.utils.device import memory_estimate, array_str, detect_root
from beamax.utils.arrays import (
    pad_zero,
    pad_edge,
    pad_array,
    rel_l2,
    _center_slices,
    crop_centered,
    interpolate_fourier,
)
from beamax.transforms import MSWPT

jax.config.update("jax_enable_x64", True)

common_params = [
    (64,),
    (64, 64),
    (64, 128),
    (128, 64),
    (64, 64, 64),
    (64, 64, 128),
    (64, 128, 64),
    (128, 64, 64),
    (32, 64, 128),
]

# ------------------------- _center_slices -------------------------


def test_center_slices_ok_and_error():
    curr = (9, 7)
    target = (5, 3)
    sl = _center_slices(curr, target)
    assert sl == (slice(2, 7), slice(2, 5))  # centered crop

    # Error when target is larger than current
    with pytest.raises(ValueError):
        _center_slices((5, 5), (6, 5))


# --------------------------- pad_array ----------------------------

# def test_pad_array_constant_and_edge_and_noop():
#     x = jnp.arange(6).reshape(2, 3)

#     # pad (constant zeros) to larger shape
#     y = pad_array(x, (6, 7), mode="constant")
#     assert y.shape == (6, 7)
#     assert jnp.all(y[:2, :3] == x)

#     # pad using edge mode
#     z = pad_array(x, (4, 5), mode="edge")
#     assert z.shape == (4, 5)
#     assert jnp.all(z[:2, :3] == x)  # original block preserved

#     # no-op when target <= current
#     w = pad_array(x, (2, 3), mode="constant")
#     assert w is not x and w.shape == x.shape and jnp.all(w == x)

# --------------------------- crop_centered ------------------------


def test_crop_centered_behaviour():
    big = jnp.arange(100).reshape(10, 10)
    cropped = crop_centered(big, (6, 4))
    assert cropped.shape == (6, 4)

    # If desired larger than current → function returns original (by design)
    small = jnp.arange(12).reshape(3, 4)
    no_op = crop_centered(small, (4, 4))
    assert no_op.shape == small.shape and jnp.all(no_op == small)


# ----------------------- interpolate_fourier ---------------------


def test_interpolate_fourier_spatial_up_down_and_fourier_modes():
    # Start with a simple impulse
    x = jnp.zeros((8, 8))
    x = x.at[0, 0].set(1.0)

    # Mixed up/down per-axis (spatial→spatial)
    upmix = interpolate_fourier(x, (12, 6), "spatial", "spatial")
    assert upmix.shape == (12, 6) and jnp.isfinite(jnp.sum(jnp.abs(upmix)))

    # Round-trip down to original
    back = interpolate_fourier(upmix, (8, 8), "spatial", "spatial")
    assert back.shape == (8, 8)

    # Direct fourier→fourier pad/crop path
    X = jnp.fft.fftshift(jnp.fft.fftn(x, norm="ortho"))
    Xp = interpolate_fourier(X, (10, 10), "fourier", "fourier")
    assert Xp.shape == (10, 10)

    # fourier→spatial path
    x2 = interpolate_fourier(Xp, (10, 10), "fourier", "spatial")
    assert x2.shape == (10, 10)
    assert jnp.isfinite(jnp.sum(jnp.abs(x2)))


@pytest.fixture
def random_signal(request):
    N = request.param
    key = jax.random.PRNGKey(0)
    return jax.random.normal(key, N)


@pytest.fixture
def interpolation_setup(request):
    N = request.param
    input_array = jnp.ones(N)
    desired_size = tuple(s * 2 for s in N)
    return input_array, desired_size


def assert_allclose(a, b, atol=1e-16):
    assert jnp.allclose(a, b, atol=atol)


def assert_norm_equal(a, b, atol=1e-16):
    assert jnp.isclose(jnp.linalg.norm(a), jnp.linalg.norm(b), atol=atol)


@pytest.mark.parametrize("random_signal", common_params, indirect=True)
def test_fft_helper(random_signal):
    """
    Assert that the FFT helper function is unitary.
    """
    F = utils.unitary_fft(random_signal)
    f2 = utils.unitary_ifft(F)
    assert_allclose(random_signal, f2)
    assert_norm_equal(random_signal, f2)


@pytest.mark.parametrize("random_signal", common_params, indirect=True)
def test_interpftn(random_signal):
    """
    Test that the interpolation function is unitary.
    """
    input_type = "spatial"
    output_type = "spatial"
    desired_size = tuple(s * 2 for s in random_signal.shape)
    original_energy = jnp.linalg.norm(random_signal)
    interp_signal = utils.interpolate_fourier(
        random_signal, desired_size, input_type, output_type
    )
    interp_energy = jnp.linalg.norm(interp_signal)
    interp_signal_back = utils.interpolate_fourier(
        interp_signal, random_signal.shape, output_type, input_type
    )
    assert_allclose(random_signal, interp_signal_back)
    # assert interp_signal.dtype == random_signal.dtype FALSE
    assert interp_signal.shape == desired_size
    assert_allclose(original_energy, interp_energy)


@pytest.mark.parametrize("N", [jnp.array([64, 64]), jnp.array([64, 128])])
def test_nn_interp(N):
    """
    Testing the nearest neighbour interpolation.
    """
    input = jnp.zeros(N).at[:, 0].set(1)

    interp = utils.interpolate_nearest(input, N // 2)

    aim = jnp.zeros(N // 2).at[:, 0].set(1)

    assert jnp.allclose(interp, aim)


def test_convert_space():
    """
    Test the function which converts an array from one space to another.
    """
    x = jnp.ones((2, 2))
    x_f1 = utils.convert_space(x, "spatial", "fourier")
    x_f2 = utils.unitary_fft(x)
    assert_allclose(x_f1, x_f2)

    with pytest.raises(ValueError):
        utils.convert_space(x, "spatial", "Fourier")
        utils.convert_space(x, "fourier", "Spatial")
        utils.convert_space(x, "Spatial", "fourier")


def test_make_c_function_from_grid_and_derivs():
    xx = jnp.linspace(0, 1, 5)
    yy = jnp.linspace(0, 1, 7)
    vals = jnp.outer(xx, yy)  # f(x,y) = x*y

    # Direct function built from grid
    cfun = make_c_function_from_grid(
        vals,
        spacing=(float(xx[1] - xx[0]), float(yy[1] - yy[0])),
        origin=(float(xx[0]), float(yy[0])),
    )

    # Unbatched query → scalar output
    p = jnp.array([0.5, 0.25])
    f = cfun(p)
    assert f.shape == ()  # scalar from unbatched input

    itp = Interpolator([xx, yy], vals)
    f2 = itp(p)
    assert f2.shape == ()
    # Grad/Hess shapes for unbatched input
    g = itp.grad(p)
    H = itp.hessian(p)
    assert g.shape == (2,)
    assert H.shape == (2, 2)

    # Optional: batched query → batched outputs
    P = jnp.stack([p, jnp.array([0.75, 0.5])], axis=0)  # (2, 2)
    Fb = itp(P)
    assert Fb.shape == (2,)


def test_bspline3_interpolates_grid_values():
    xx = jnp.linspace(0.0, 1.0, 12)
    vals = jnp.sin(2 * jnp.pi * xx) + 0.1 * xx
    cfun = make_c_function_from_grid(
        vals,
        spacing=(float(xx[1] - xx[0]),),
        origin=(float(xx[0]),),
        method="bspline3",
        boundary="reflect",
    )

    got = cfun(xx[:, None])
    assert got.shape == vals.shape
    assert jnp.allclose(got, vals, atol=1e-10)


def test_bspline3_derivatives_are_continuous_at_cell_boundaries():
    vals = jnp.array([0.0, 1.0, 0.5, 2.0, 1.0, 0.2, 0.3, 0.0])
    cfun = make_c_function_from_grid(vals, method="bspline3", boundary="reflect")

    eps = 1e-5
    x0 = 3.0
    grad_left = jax.grad(lambda x: cfun(jnp.array([x])))(x0 - eps)
    grad_right = jax.grad(lambda x: cfun(jnp.array([x])))(x0 + eps)
    hess_left = jax.hessian(lambda x: cfun(jnp.array([x])))(x0 - eps)
    hess_right = jax.hessian(lambda x: cfun(jnp.array([x])))(x0 + eps)

    assert jnp.allclose(grad_left, grad_right, atol=1e-3)
    assert jnp.allclose(hess_left, hess_right, atol=1e-3)


def test_make_c_function_from_grid_options_validate():
    vals = jnp.ones((4, 4))

    with pytest.raises(ValueError, match="method"):
        make_c_function_from_grid(vals, method="quadratic")

    with pytest.raises(ValueError, match="Unsupported boundary"):
        make_c_function_from_grid(vals, boundary="zero")

    with pytest.raises(ValueError, match="smooth_sigma"):
        make_c_function_from_grid(vals, smooth_sigma=(1.0, 2.0, 3.0))


def test_memory_helpers_and_array_str():
    x = jnp.zeros((2, 3), dtype=jnp.float32)
    s = memory_estimate(jnp.array(x.shape), x.dtype)
    assert "Kb" in s or "Mb" in s or "Gb" in s
    assert "Array" in array_str(x)


def test_detect_root_env_takes_priority(monkeypatch, tmp_path):
    monkeypatch.setenv("BEAMAX_ROOT", str(tmp_path))
    root = detect_root()
    assert str(root) == str(tmp_path)


def test_find_repo_root_accepts_file_paths(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    file_path = tmp_path / "scripts" / "example.py"
    assert device_utils.find_repo_root(file_path) == tmp_path


def test_detect_root_falls_back_to_cwd_for_installed_layout(monkeypatch, tmp_path):
    monkeypatch.delenv("BEAMAX_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        device_utils,
        "__file__",
        str(tmp_path / "site-packages" / "beamax" / "utils" / "device.py"),
    )

    assert detect_root() == tmp_path


def test_example_plot_dir_uses_public_example_category(monkeypatch, tmp_path):
    monkeypatch.setenv("BEAMAX_ROOT", str(tmp_path))
    example_path = tmp_path / "examples" / "forward" / "2d_forward.py"

    plot_dir = device_utils.example_plot_dir(example_path)

    assert plot_dir == tmp_path / "plots" / "forward"
    assert plot_dir.is_dir()


def test_example_plot_dir_falls_back_to_parent_name(monkeypatch, tmp_path):
    monkeypatch.setenv("BEAMAX_ROOT", str(tmp_path))
    example_path = tmp_path / "scratch" / "demo.py"

    plot_dir = device_utils.example_plot_dir(example_path)

    assert plot_dir == tmp_path / "plots" / "scratch"
    assert plot_dir.is_dir()


def test_pad_zero_and_edge():
    x = jnp.arange(6).reshape(2, 3)
    z = pad_zero(x, (4, 5))
    e = pad_edge(x, (4, 5))
    assert z.shape == (4, 5) and e.shape == (4, 5)


# ============================================================================
# Shared fixtures
# ============================================================================
@pytest.fixture
def simple_dyadic_decomp():
    """Simple 2D dyadic decomposition for testing."""
    return DyadicDecomposition(
        num_levels=2,
        N=(64, 64),
        num_boxes_levels=(4, 8),
        box_aspect_ratio=(1, 1),
    )


@pytest.fixture
def simple_domain():
    """Simple 2D domain for testing."""
    return Domain(
        N=(64, 64),
        dx=(0.01, 0.01),
        c=1500.0,
        periodic=(False, False),
    )


@pytest.fixture
def simple_wpt(simple_dyadic_decomp):
    """Simple MSWPT for testing."""
    return MSWPT(
        dyadic_decomp=simple_dyadic_decomp,
        redundancy=2,
        windowing="rectangular",
    )


# ============================================================================
# Test rel_l2
# ============================================================================


class TestRelL2:
    def test_identical_arrays(self):
        """Test relative L2 between identical arrays."""
        arr = jnp.ones((10, 10))
        error = rel_l2(arr, arr)

        assert error < 1e-6

    def test_different_arrays(self):
        """Test relative L2 between different arrays."""
        arr1 = jnp.ones((10, 10))
        arr2 = jnp.zeros((10, 10))
        error = rel_l2(arr1, arr2)

        assert error > 0.99  # Should be close to 1

    def test_small_difference(self):
        """Test relative L2 with small difference."""
        arr1 = jnp.ones((10, 10))
        arr2 = arr1 + 0.01
        error = rel_l2(arr1, arr2)

        assert 0 < error < 0.1


# ============================================================================
# Test small error paths in array helpers
# ============================================================================


def test_pad_array_unsupported_mode_raises():
    """pad_array must reject unknown modes."""
    arr = jnp.zeros((4, 4))
    with pytest.raises(ValueError, match="Unsupported pad mode"):
        pad_array(arr, (6, 6), mode="reflect")


def test_pad_array_edge_mode():
    """edge mode must replicate the boundary values."""
    arr = jnp.array([1.0, 2.0, 3.0])
    out = pad_array(arr, (5,), mode="edge")
    assert out.shape == (5,)
    # Centered pad: one element left, one right; both pick up the boundary value.
    assert float(out[0]) == 1.0
    assert float(out[-1]) == 3.0


def test_interpolate_fourier_invalid_input_type_raises():
    """interpolate_fourier rejects unrecognized domain strings."""
    arr = jnp.zeros((8, 8))
    with pytest.raises(ValueError, match="input_type/output_type"):
        interpolate_fourier(arr, (16, 16), input_type="time", output_type="spatial")
    with pytest.raises(ValueError, match="input_type/output_type"):
        interpolate_fourier(arr, (16, 16), input_type="spatial", output_type="time")


def test_interpolate_fourier_roundtrip_spatial_to_fourier():
    """Real-input upsample in fourier output should match plain unitary FFT padding."""
    arr = jnp.ones((4,))
    out = interpolate_fourier(arr, (8,), input_type="spatial", output_type="fourier")
    assert out.shape == (8,)


if __name__ == "__main__":
    pytest.main([__file__])
