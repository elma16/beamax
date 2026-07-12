import pytest
import jax
import jax.numpy as jnp
from beamax import geometry, utils
from beamax.decomposition import DyadicDecomposition

from beamax.transforms import MSWPT, compute_frames
from beamax.gb import core, gb_solvers, gb_utils
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


@pytest.mark.parametrize("redundancy", [1, 2])
def test_forward_parameter_positions_follow_local_fft_support(redundancy):
    """Packet indices map through the full redundant, anisotropic support."""
    N = (32, 16)
    dx = (0.1, 0.3)
    aspect = (2, 1)
    domain = geometry.Domain(
        N=N,
        dx=dx,
        c=lambda x: 2.0 + 0.0 * x[..., 0],
        periodic=(True, True),
    )
    decomp = DyadicDecomposition(1, N, (4,), aspect)
    wpt = MSWPT(decomp, redundancy=redundancy, windowing="rectangular")

    coeff_shape = tuple(wpt.coeff_shapes[0])
    k = jnp.asarray([coeff_shape[1] - 1, coeff_shape[2] - 1])
    flat_index = jnp.ravel_multi_index((0, k[0], k[1]), coeff_shape)
    _, _, x0s, _, _, _ = forward_solver_utils.compute_forward_parameters(
        jnp.asarray([flat_index]), wpt, domain
    )

    support_lengths = redundancy * decomp.box_lengths[0] * jnp.asarray(aspect)
    expected = k * domain.grid_size / support_lengths
    assert jnp.allclose(x0s[0], expected)
    assert jnp.all(x0s[0] < domain.grid_size)


@pytest.mark.parametrize("redundancy", [1, 2])
def test_forward_parameters_match_anisotropic_frame_phase(redundancy):
    """The beam at t=0 includes the transform's parity-recentring phase."""
    N = (64, 32)

    def c_fn(x):
        return 1.0 + 0.0 * x[..., 0]

    domain = geometry.Domain(
        N=N,
        dx=(1.0 / N[0], 1.0 / N[1]),
        c=c_fn,
        periodic=(True, True),
    )
    decomp = DyadicDecomposition(1, N, (4,), (2, 1))
    wpt = MSWPT(decomp, redundancy=redundancy, windowing="none")
    coeff_shape = tuple(wpt.coeff_shapes[0])
    local_k = jnp.asarray([coeff_shape[1] - 1, coeff_shape[2] - 1])
    coeff_idx = jnp.ravel_multi_index((0, local_k[0], local_k[1]), coeff_shape)

    expected = utils.unitary_ifft(
        compute_frames(
            decomp,
            0,
            local_k,
            decomp.fourier_meshgrid,
            redundancy,
            "none",
        )
    )
    p0s, M0s, x0s, omegas, a0s, modes = forward_solver_utils.compute_forward_parameters(
        jnp.asarray([coeff_idx]), wpt, domain
    )
    actual = core.compute_gaussian_beam(
        x0=x0s,
        p0=p0s,
        M0=M0s,
        a0=a0s,
        omega0=omegas,
        mode=modes,
        c=domain.c,
        lam=0.0,
        ts=jnp.asarray([0.0]),
        sensors=domain.grid,
        domain_size=domain.grid_size,
        periodic=jnp.asarray(domain.periodic),
        ode_solver=gb_solvers.solve_ODE_base,
        solver_config=None,
    )[0, ..., 0]

    # The unwindowed periodic Gaussian is evaluated through two numerically
    # equivalent representations; the phase error this test targets is O(1),
    # while their finite-grid tail difference is about 1e-8 here.
    assert jnp.allclose(actual, expected, rtol=1e-7, atol=1e-7)


def test_compute_coefficients_samples_speed_at_physical_packet_centres():
    """The initial-velocity split uses the same packet centres as propagation."""
    N = (32, 16)
    dx = (0.1, 0.3)
    aspect = (2, 1)
    redundancy = 2

    def heterogeneous_speed(x):
        return 2.0 + 0.3 * x[..., 0] + 0.07 * x[..., 1]

    domain = geometry.Domain(
        N=N,
        dx=dx,
        c=heterogeneous_speed,
        periodic=(True, True),
    )
    decomp = DyadicDecomposition(1, N, (4,), aspect)
    wpt = MSWPT(decomp, redundancy=redundancy, windowing="rectangular")
    p0 = jnp.zeros(N)
    dpdt = jnp.arange(N[0] * N[1], dtype=float).reshape(N) / (N[0] * N[1])

    cpos, _ = forward_solver_utils.compute_coefficients(
        p0, dpdt, "spatial", domain, wpt
    )

    # Recreate the coefficient block after the donated forward-call buffers.
    dpdt_expected = jnp.arange(N[0] * N[1], dtype=float).reshape(N) / (N[0] * N[1])
    b_coeff = wpt.forward(dpdt_expected, "spatial")
    shapes = utils.compute_coeff_shapes(decomp, redundancy, jnp.arange(1))
    nn_level, nn_idx = utils.find_tensor_and_multiindex(
        jnp.arange(wpt.total_coeffs), shapes
    )
    support_lengths = (
        decomp.box_lengths[nn_level, None] * jnp.asarray(aspect)[None, :] * redundancy
    )
    x_expected = nn_idx[1:, :].T * domain.grid_size / support_lengths
    box_offsets = jnp.r_[0, jnp.cumsum(decomp.num_boxes_ndim)]
    box_idx = nn_idx[0, :] + box_offsets[nn_level]
    centres = decomp.centres_ndim[box_idx] / domain.grid_size
    momenta = 2 * jnp.pi * centres
    hamiltonian = gb_utils.vmap_g(
        x_expected,
        momenta,
        jnp.ones(wpt.total_coeffs),
        domain.c_fn,
    )
    expected_cpos = 0.5j * b_coeff / hamiltonian

    assert jnp.all(x_expected >= 0.0)
    assert jnp.all(x_expected < domain.grid_size)
    assert jnp.allclose(cpos, expected_cpos, rtol=1e-11, atol=1e-11)


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


def test_interpolate_LF_soln_fourier_preserves_legacy_planar_amplitude():
    """Legacy 2-D-volume/1-D-sensor normalization cancels crop inflation."""
    # Cropping an 8x8 unitary spectrum to 4x4 inflates a linear LF prediction
    # by sqrt((8/4)^2) = 2. The historical helper then resizes the one retained
    # sensor axis from 4 to 8.
    coarse_sensor_data = 2.0 * jnp.ones((3, 4))
    reconstructed = hybrid_solver_utils.interpolate_LF_soln(
        coarse_sensor_data,
        (3, 8),
        interpolation_method="fourier",
        dt_oversample=0,
    )

    assert jnp.allclose(reconstructed, 1.0, atol=5e-7)


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
