import pytest
import jax
import jax.numpy as jnp
from beamax import geometry
from beamax.decomposition import DyadicDecomposition

from beamax.transforms import MSWPT
from beamax.solvers import hybrid_solver_utils
from beamax.solvers.msgb_solvers import forward_solver_utils

from beamax.solvers.hybrid_solver_utils import (
    get_indices_with_norm_less_than,
    downsample_p0,
)

jax.config.update("jax_enable_x64", True)


def c(x):
    return 1500 + 0 * x[..., 0]


def test_compute_coefficients_dpdt_zero():
    """
    Test that if dpdt = 0, c+ == c-.
    """
    N = (128, 64)
    d = len(N)
    dx = (1e-4,) * d
    periodic = (False,) * d

    p0 = jax.random.normal(jax.random.PRNGKey(0), N)
    dpdt = jnp.zeros_like(p0)

    box_aspect_ratio = (1, 1)
    num_levels = 2
    num_boxes_level = (4, 8)

    def c(x):
        return 1500 + 0 * x[..., 0]

    input_type = "spatial"

    domain = geometry.Domain(N=N, dx=dx, c=c, periodic=periodic)

    dyadic_decomp = DyadicDecomposition(
        num_levels, N, num_boxes_level, box_aspect_ratio
    )
    wpt = MSWPT(dyadic_decomp, redundancy=2, windowing="rectangular")

    cpos, cneg = forward_solver_utils.compute_coefficients(
        p0, dpdt, input_type, domain, wpt
    )

    assert jnp.allclose(cpos, cneg)


def test_compute_coefficients_p0_zero():
    """
    Test that if p0 = 0, c+ ==  - c-.
    """
    N = (128, 64)
    d = len(N)
    dx = (1e-4,) * d
    periodic = (False,) * d

    dpdt = jax.random.normal(jax.random.PRNGKey(0), N)
    p0 = jnp.zeros_like(dpdt)

    box_aspect_ratio = (1, 1)
    num_levels = 2
    num_boxes_level = (4, 8)

    def c(x):
        return 1500 + 0 * x[..., 0]

    input_type = "spatial"

    domain = geometry.Domain(N=N, dx=dx, c=c, periodic=periodic)

    dyadic_decomp = DyadicDecomposition(
        num_levels, N, num_boxes_level, box_aspect_ratio
    )
    wpt = MSWPT(dyadic_decomp, redundancy=2, windowing="rectangular")

    cpos, cneg = forward_solver_utils.compute_coefficients(
        p0, dpdt, input_type, domain, wpt
    )

    assert jnp.allclose(cpos, -1 * cneg)


def test_coefficients_conjugate_if_p0_dpdt_real():
    """
    Test that if p0 and dpdt are real,

    conj(c^+_{l,j',k}) = c^-_{l,j,k}
    """
    N = (128, 64)
    d = len(N)
    dx = (1e-4,) * d
    periodic = (False,) * d
    redundancy = 2
    windowing = "rectangular_mirror"

    key1 = jax.random.PRNGKey(10)
    key2 = jax.random.PRNGKey(20)

    p0 = jax.random.normal(key1, N)
    dpdt = jax.random.normal(key2, N)

    box_aspect_ratio = (1,) * d
    num_levels = 2
    num_boxes_level = (4, 8)

    def c(x):
        return 1500 + 0 * x[..., 0]

    input_type = "spatial"

    domain = geometry.Domain(N=N, dx=dx, c=c, periodic=periodic)

    dyadic_decomp = DyadicDecomposition(
        num_levels, N, num_boxes_level, box_aspect_ratio
    )
    wpt = MSWPT(dyadic_decomp, redundancy=redundancy, windowing=windowing)

    cpos, cneg = forward_solver_utils.compute_coefficients(
        p0, dpdt, input_type, domain, wpt
    )

    num_boxes_level = dyadic_decomp.num_boxes_ndim_cumsum - 1
    num_boxes_level = jnp.concatenate([jnp.array([0]), num_boxes_level], axis=0)

    def to_list(c):
        return [
            c[wpt.coeffs_cumsum[level] : wpt.coeffs_cumsum[level + 1]].reshape(
                tuple(wpt.coeff_shapes[level])
            )
            for level in range(len(wpt.coeff_shapes))
        ]

    c_pos_list = to_list(cpos)
    c_neg_list = to_list(cneg)

    for level in range(num_levels):
        start_idx = wpt.boxes_cumsum[level]
        end_idx = wpt.boxes_cumsum[level + 1]
        print("c pos", c_pos_list[level].shape)
        for box in range(end_idx - start_idx):
            print(f"level: {level}, box: {box}")
            box_conj = (end_idx - start_idx) - box - 1
            assert jnp.allclose(
                c_pos_list[level][box], jnp.conj(c_neg_list[level][box_conj])
            ), f"Box {box} and {box_conj} are not conjugates"


@pytest.mark.parametrize(
    "size, max_size, expected",
    [
        (1, 16, 1),
        (3, 16, 4),
        (16, 32, 16),
        (20, 32, 32),
        (64, 64, 64),
        (128, 64, 64),
        (200, 256, 256),
        (300, 256, 256),
        (513, 512, 512),
    ],
)
def test_closest_power_of_two(size, max_size, expected):
    result = hybrid_solver_utils.closest_power_of_two(size, max_size)
    assert result == expected, f"Expected {expected}, but got {result}"


# @pytest.mark.parametrize("size, max_size", [
#     (0, 16),              # size is 0
#     (-1, 16),             # size is negative
#     (16, 0),              # max_size is 0
#     (32, -1),             # max_size is negative
# ])
# def test_closest_power_of_two_invalid_values(size, max_size):
#     with pytest.raises(ValueError):
#         solver_utils.closest_power_of_two(size, max_size)


def test_threshold_strategies_cover_all():
    c = jnp.array([0.0, 0.1, 0.5, 1.0, 2.0])
    # hard
    idx, val = forward_solver_utils.threshold_coefficients(c, 0.5, "hard")
    assert (c[idx] == val).all()
    # top_n
    idx, val = forward_solver_utils.threshold_coefficients(c, 2, "top_n")
    assert len(idx) == 2
    # percentile
    idx, val = forward_solver_utils.threshold_coefficients(c, 80, "percentile")
    assert len(idx) <= c.size
    # hard_reassign
    idx, val = forward_solver_utils.threshold_coefficients(c, 0.5, "hard_reassign")
    assert jnp.all(jnp.abs(val) >= 0.0)
    # perc_max_abs
    idx, val = forward_solver_utils.threshold_coefficients(c, 0.5, "perc_max_abs")
    assert (jnp.abs(c[idx]) > 0.5 * jnp.max(jnp.abs(c))).all()


def test_threshold_bao_energy_path():
    decomp = DyadicDecomposition(
        num_levels=1, N=(8, 8), num_boxes_levels=(2,), box_aspect_ratio=(1, 1)
    )
    wpt = MSWPT(decomp, redundancy=2, windowing="rectangular")
    coeff = jnp.linspace(0, 1, 4 * (8 * 8))  # matches function's expected length
    idx, val = forward_solver_utils.threshold_coefficients(
        coeff, 0.5, "bao_energy", wpt=wpt
    )
    assert idx.ndim == 1 and val.ndim == 1


def test_get_indices_with_norm_less_than_and_downsample():
    centers = jnp.array([[0, 0], [1, 1], [2, 2]])
    idx = get_indices_with_norm_less_than(centers, 1.5)
    assert set(map(int, idx)) == {0, 1}
    arr = jnp.ones((32, 32))
    bd = jnp.array([[8, 24], [8, 24]])
    ds = downsample_p0(arr, bd, use_power_of_two=True)
    assert ds.shape[0] == ds.shape[1] and ds.shape[0] <= 32


def test_oversample_window_no_oversample():
    """Test that dt_oversample=0 returns the original array unchanged."""
    x = jnp.ones((12, 3))
    result = hybrid_solver_utils.oversample_window(x, dt_oversample=0)
    assert jnp.array_equal(result, x)


def test_oversample_window_shape_preserved():
    """Test that all window types preserve array shape."""
    x = jnp.ones((12, 3))
    for window_type in ("cos2", "hann", "hamming", "blackman"):
        result = hybrid_solver_utils.oversample_window(
            x, dt_oversample=4, axis=0, window_type=window_type
        )
        assert result.shape == x.shape, f"Shape mismatch for {window_type}"


def test_oversample_window_applies_at_end():
    """Test that windowing is applied only to the last dt_oversample elements."""
    x = jnp.ones((12, 3))
    dt_oversample = 4
    result = hybrid_solver_utils.oversample_window(
        x, dt_oversample=dt_oversample, axis=0, window_type="cos2"
    )

    # First elements should be unchanged
    assert jnp.array_equal(result[:8], x[:8])
    # Last elements should be modified (windowed, so values < 1)
    assert jnp.all(result[-dt_oversample:] <= x[-dt_oversample:])


def test_oversample_window_different_axis():
    """Test windowing along different axes."""
    x = jnp.ones((10, 8, 6))
    dt_oversample = 3

    # Test axis=0
    result0 = hybrid_solver_utils.oversample_window(
        x, dt_oversample=dt_oversample, axis=0, window_type="cos2"
    )
    assert result0.shape == x.shape
    assert jnp.all(result0[-dt_oversample:] <= 1.0)

    # Test axis=1
    result1 = hybrid_solver_utils.oversample_window(
        x, dt_oversample=dt_oversample, axis=1, window_type="cos2"
    )
    assert result1.shape == x.shape


def test_oversample_window_invalid_type():
    """Test that invalid window type raises ValueError."""
    x = jnp.ones((12, 3))
    with pytest.raises(ValueError, match="Unsupported window type"):
        hybrid_solver_utils.oversample_window(x, dt_oversample=4, window_type="invalid")


def test_interpolate_LF_soln_spline():
    """Test spline interpolation with energy correction."""
    x = jnp.zeros((16, 8))
    x = x.at[-3:].set(1.0)  # Set last 3 rows to 1
    target_size = (32, 16)

    result = hybrid_solver_utils.interpolate_LF_soln(
        x,
        target_size,
        interpolation_method="spline",
        interp_window="cos2",
        dt_oversample=2,
        spline_order=1,
    )

    assert result.shape == target_size
    # Check that upsampling occurred
    assert result.shape[0] > x.shape[0]
    assert result.shape[1] > x.shape[1]


def test_interpolate_LF_soln_fourier():
    """Test Fourier interpolation."""
    x = jnp.zeros((16, 8))
    x = x.at[7:9, 3:5].set(1.0)  # Set a small region to 1
    target_size = (32, 16)

    result = hybrid_solver_utils.interpolate_LF_soln(
        x,
        target_size,
        interpolation_method="fourier",
        interp_window="hann",
        dt_oversample=2,
    )

    assert result.shape == target_size


def test_interpolate_LF_soln_all_window_types():
    """Test that all window types work with interpolation."""
    x = jnp.ones((8, 4))
    target_size = (16, 8)

    for window_type in ("cos2", "hann", "hamming", "blackman"):
        result = hybrid_solver_utils.interpolate_LF_soln(
            x,
            target_size,
            interpolation_method="fourier",
            interp_window=window_type,
            dt_oversample=2,
        )
        assert result.shape == target_size, f"Failed for window type: {window_type}"


def test_interpolate_LF_soln_no_oversample():
    """Test interpolation without temporal oversampling."""
    x = jnp.ones((8, 4))
    target_size = (16, 8)

    result = hybrid_solver_utils.interpolate_LF_soln(
        x, target_size, interpolation_method="fourier", dt_oversample=0
    )

    assert result.shape == target_size


def test_interpolate_LF_soln_different_spline_orders():
    """Test different spline interpolation orders."""
    x = jnp.zeros((12, 6))
    x = x.at[5:7, 2:4].set(1.0)
    target_size = (24, 12)

    for order in [0, 1, 2, 3]:
        result = hybrid_solver_utils.interpolate_LF_soln(
            x,
            target_size,
            interpolation_method="spline",
            spline_order=order,
            dt_oversample=0,
        )
        assert result.shape == target_size


def test_interpolate_LF_soln_invalid_method():
    """Test that invalid interpolation method raises ValueError."""
    x = jnp.ones((8, 4))
    target_size = (16, 8)

    with pytest.raises(ValueError, match="not supported"):
        hybrid_solver_utils.interpolate_LF_soln(
            x, target_size, interpolation_method="invalid_method"
        )


def test_interpolate_LF_soln_1d():
    """Test interpolation on 1D arrays."""
    x = jnp.ones((16,))
    target_size = (32,)

    result = hybrid_solver_utils.interpolate_LF_soln(
        x, target_size, interpolation_method="fourier", dt_oversample=0
    )

    assert result.shape == target_size


def test_interpolate_LF_soln_3d():
    """Test interpolation on 3D arrays."""
    x = jnp.ones((8, 4, 6))
    target_size = (16, 8, 12)

    result = hybrid_solver_utils.interpolate_LF_soln(
        x, target_size, interpolation_method="fourier", dt_oversample=2
    )

    assert result.shape == target_size


if __name__ == "__main__":
    pytest.main([__file__])
