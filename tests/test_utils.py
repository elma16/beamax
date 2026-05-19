import pytest
import jax
import jax.numpy as jnp
from beamax import utils
from beamax.utils.interp import make_c_function_from_grid, Interpolator
from beamax.geometry import Domain
from beamax.decomposition import DyadicDecomposition
import h5py
import numpy as np
import tempfile
import os
import beamax.utils.device as device_utils
from beamax.utils.oabreast import (
    load_oabreast_p0_c,
    _effective_spacing_after_shape_resample,
    _ensure_axis_order_zyx,
)
from beamax.utils.device import memory_estimate, array_str, detect_root
from beamax.utils.misc import (
    pad_zero,
    pad_edge,
    ellipsoid_superposition,
    find_closest_center_indices,
    rel_l2,
    choose_K_by_tau,
    select_levelaware_topK_indices,
    reconstruct_from_selection,
)
from beamax.transforms import MSWPT
from beamax.utils.misc import (
    _center_slices,
    _rand_rot,
    crop_centered,
    interpolate_fourier,
)

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


# --------------------------- _rand_rot ----------------------------


def test_rand_rot_returns_rotations_d1_d2_d3():
    key = jax.random.PRNGKey(0)

    # d=1 → identity for any K
    R1 = _rand_rot(key, d=1, K=3)
    assert R1.shape == (3, 1, 1)
    assert jnp.allclose(R1, jnp.ones_like(R1))

    # d=2 → orthogonal, det ~ +1
    R2 = _rand_rot(key, d=2, K=5)
    assert R2.shape == (5, 2, 2)
    I2 = jnp.einsum("bij,bjk->bik", jnp.swapaxes(R2, -1, -2), R2)
    assert jnp.allclose(I2, jnp.eye(2)[None, :, :], atol=1e-6)
    det2 = jnp.linalg.det(R2)
    assert jnp.all(det2 > 0.0)

    # d=3 → orthogonal, det ~ +1
    R3 = _rand_rot(key, d=3, K=4)
    assert R3.shape == (4, 3, 3)
    I3 = jnp.einsum("bij,bjk->bik", jnp.swapaxes(R3, -1, -2), R3)
    assert jnp.allclose(I3, jnp.eye(3)[None, :, :], atol=1e-6)
    det3 = jnp.linalg.det(R3)
    assert jnp.all(det3 > 0.0)


def test_ellipsoid_superposition_gaussian_and_indicator():
    key = jax.random.PRNGKey(123)

    # 2D gaussian
    f2_gauss, meta2 = ellipsoid_superposition(
        key, (16, 16), n_ellipses=3, profile="gaussian", nonnegative=True
    )
    assert f2_gauss.shape == (16, 16)
    assert jnp.all(f2_gauss >= 0)

    # 1D indicator
    f1_ind, meta1 = ellipsoid_superposition(
        key, (32,), n_ellipses=2, profile="indicator", nonnegative=True
    )
    assert f1_ind.shape == (32,)
    assert jnp.all((f1_ind == 0) | (f1_ind > 0))


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


def test_load_oabreast_2d_variants(tmp_path):
    # build tiny 3D labels with values {0,3,4}
    lbl = np.zeros((4, 4, 4), dtype=np.uint8)
    lbl[1:3, 1:3, 1:3] = 3
    lbl[2, 2, 2] = 4
    f = tmp_path / "ph.h5"
    with h5py.File(f, "w") as h:
        h.create_dataset("MergedPhantom", data=lbl)
    p0, c, meta = load_oabreast_p0_c(
        f,
        dim="2d",
        axis_order="ZYX",
        slice_axis=0,
        slice_policy="max_variance",
        target_shape=(8, 8),
        return_labels=False,
    )
    assert p0.shape == c.shape == (8, 8)
    assert set(meta["label_set"]).issubset({0, 3, 4})


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
    (tmp_path / "src" / "beamax").mkdir(parents=True)
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


def test_pad_zero_and_edge():
    x = jnp.arange(6).reshape(2, 3)
    z = pad_zero(x, (4, 5))
    e = pad_edge(x, (4, 5))
    assert z.shape == (4, 5) and e.shape == (4, 5)


def test_ellipsoid_superposition_shapes():
    key = jnp.array([0, 1], dtype=jnp.uint32)  # fake PRNGKey
    f2, _ = ellipsoid_superposition(key, (16, 16), n_ellipses=3, profile="gaussian")
    assert f2.shape == (16, 16)
    f1, _ = ellipsoid_superposition(key, (16,), n_ellipses=2, profile="indicator")
    assert f1.shape == (16,)


def test_find_closest_center_indices():
    centers = jnp.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    idxs = find_closest_center_indices(centers, index=0, k=2)
    assert set(map(int, idxs)) <= {0, 1, 2}


def test_axis_order_and_spacing_helpers_and_label_checks(tmp_path):
    A = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    zyx = _ensure_axis_order_zyx(A, "ZYX")
    xyz = _ensure_axis_order_zyx(A, "XYZ")
    assert zyx.shape == (2, 3, 4) and xyz.shape == (4, 3, 2)
    sp = _effective_spacing_after_shape_resample((1, 1, 1), (2, 3, 4), (4, 6, 8))
    assert sp == (0.5, 0.5, 0.5)

    good = np.zeros((2, 2, 2), dtype=np.uint8)
    good[0] = 3
    bad = np.ones((2, 2, 2), dtype=np.uint8) * 9
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        with h5py.File(f.name, "w") as h:
            h.create_dataset("MergedPhantom", data=good)
        p0, c, meta = load_oabreast_p0_c(f.name, dim="3d")
        assert p0.shape == c.shape == good.shape
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as g:
        with h5py.File(g.name, "w") as h:
            h.create_dataset("MergedPhantom", data=bad)
        with pytest.raises(ValueError):
            load_oabreast_p0_c(g.name, dim="3d")
    os.unlink(f.name)
    os.unlink(g.name)


# ============================================================================
# Test level-aware coefficient selection
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


class TestLevelAwareSelection:
    def test_select_topk_basic(self, simple_dyadic_decomp, simple_wpt):
        """Test basic top-K selection."""
        # Create random coefficients
        total_coeffs = simple_wpt.total_coeffs
        coeffs = jax.random.normal(jax.random.PRNGKey(42), (total_coeffs,))

        K = 100
        indices, values = select_levelaware_topK_indices(
            coeffs, simple_dyadic_decomp, simple_wpt, K
        )

        assert indices.shape[0] <= K
        assert values.shape[0] == indices.shape[0]

    def test_select_topk_zero(self, simple_dyadic_decomp, simple_wpt):
        """Test K=0 returns empty arrays."""
        total_coeffs = simple_wpt.total_coeffs
        coeffs = jax.random.normal(jax.random.PRNGKey(42), (total_coeffs,))

        indices, values = select_levelaware_topK_indices(
            coeffs, simple_dyadic_decomp, simple_wpt, 0
        )

        assert indices.shape[0] == 0
        assert values.shape[0] == 0

    def test_reconstruct_from_selection(self, simple_wpt):
        """Test reconstruction from selected coefficients."""
        total_coeffs = simple_wpt.total_coeffs
        coeffs = jax.random.normal(jax.random.PRNGKey(42), (total_coeffs,))

        # Select some indices
        indices = jnp.array([0, 10, 20])
        values = coeffs[indices]

        result = reconstruct_from_selection(
            coeffs, indices, values, simple_wpt, output_type="spatial"
        )

        assert result.shape == simple_wpt.dyadic_decomp.N


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
# Test choose_K_by_tau
# ============================================================================


class TestChooseKByTau:
    def test_basic_search(self, simple_wpt, simple_domain):
        """Test basic K search."""
        # Create a simple test signal
        p0 = jnp.ones(simple_domain.N)
        coeffs = simple_wpt.forward(p0, "spatial")

        # Create inverse WPT with windowing="none"
        inv_wpt = MSWPT(
            simple_wpt.dyadic_decomp, redundancy=simple_wpt.redundancy, windowing="none"
        )

        K = choose_K_by_tau(
            coeffs,
            p0,
            inv_wpt,
            simple_wpt.dyadic_decomp,
            simple_wpt,
            tau=0.1,
            Kmin=10,
            Kmax=500,
            num_steps=5,
        )

        assert 10 <= K <= 500

    def test_beam_budget(self, simple_wpt, simple_domain):
        """Test K search with beam budget."""
        p0 = jnp.ones(simple_domain.N)
        coeffs = simple_wpt.forward(p0, "spatial")

        inv_wpt = MSWPT(
            simple_wpt.dyadic_decomp, redundancy=simple_wpt.redundancy, windowing="none"
        )

        beam_budget = 100
        K = choose_K_by_tau(
            coeffs,
            p0,
            inv_wpt,
            simple_wpt.dyadic_decomp,
            simple_wpt,
            tau=0.1,
            beam_budget=beam_budget,
        )

        # K should respect beam budget
        assert K <= beam_budget // 2


if __name__ == "__main__":
    pytest.main([__file__])
