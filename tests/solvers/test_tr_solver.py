import jax.numpy as jnp
import jax as jax
import pytest
from beamax.solvers.msgb_solvers import tr_solver_utils
from beamax.decomposition import DyadicDecomposition
from beamax import utils
from beamax.solvers import MSGBSolver, ShardingStrategy
from beamax import geometry, transforms
from beamax.gb import gb_solvers


jax.config.update("jax_enable_x64", True)


def c(x):
    return 1 + 0 * x[..., 0]


def is_symmetric(matrix, atol=1e-16):
    """Check if a matrix is symmetric within tolerance."""
    return jnp.allclose(matrix, matrix.T, atol=atol)


def is_positive_definite(matrix):
    """Check if a matrix is positive definite."""
    return jnp.all(jnp.linalg.eigvals(matrix).real > 0)


def create_symmetric_matrix_with_positive_definite_imag(n, key=None):
    """
    Create an n x n complex symmetric matrix with positive definite imaginary part.

    Args:
        n: Size of the matrix
        key: JAX PRNG key (optional)

    Returns:
        A complex symmetric matrix with positive definite imaginary part
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # For the real part, we can use any symmetric matrix
    key, subkey = jax.random.split(key)
    real_part_raw = jax.random.normal(subkey, (n, n))
    real_part = (real_part_raw + real_part_raw.T) / 2  # Make symmetric

    # For the imaginary part, we need a positive definite matrix
    # We can create it as B*B.T + epsilon*I for some matrix B and small epsilon
    key, subkey = jax.random.split(key)
    B = jax.random.normal(subkey, (n, n))
    imag_part = jnp.matmul(B, B.T) + 0.1 * jnp.eye(n)

    # Create the complex matrix
    matrix = real_part + 1j * imag_part
    return matrix


def create_batched_symmetric_matrices_with_positive_definite_imag(
    batch_size, n, key=None
):
    """
    Create a batch of different n×n complex symmetric matrices, each with positive definite imaginary part.

    Args:
        batch_size: Number of matrices to create
        n: Size of each matrix
        key: JAX PRNG key (optional)

    Returns:
        A batch of complex symmetric matrices with positive definite imaginary parts
        Shape: (batch_size, n, n)
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    matrices = []
    for i in range(batch_size):
        key, subkey = jax.random.split(key)
        matrix = create_symmetric_matrix_with_positive_definite_imag(n, subkey)
        matrices.append(matrix)

    return jnp.stack(matrices)


vmap_is_symmetric = jax.vmap(is_symmetric)
vmap_is_pos_def = jax.vmap(is_positive_definite)


def _const_c(x):
    return 1.0 + 0.0 * x[..., 0]


def _make_small_domain_and_wpt():
    N = (16,)
    dx = (0.1,)
    domain = geometry.Domain(N=N, dx=dx, c=_const_c, periodic=(False,))
    decomp = DyadicDecomposition(
        num_levels=1, N=N, num_boxes_levels=(2,), box_aspect_ratio=(1,)
    )
    wpt = transforms.MSWPT(decomp, redundancy=2, windowing="rectangular")
    return domain, wpt


def _make_2d_data_wpt(c_fn, data_dx=(1.0, 1.0)):
    N = (16, 16)
    domain = geometry.Domain(N=N, dx=data_dx, c=c_fn, periodic=(False, False))
    decomp = DyadicDecomposition(
        num_levels=1, N=N, num_boxes_levels=(2,), box_aspect_ratio=(1, 1)
    )
    wpt = transforms.MSWPT(decomp, redundancy=2, windowing="rectangular")
    return domain, wpt


def _assert_inward_normal(mask_idx: int, expected_sign: float):
    domain, wpt = _make_small_domain_and_wpt()
    mask = jnp.zeros(domain.N)
    mask = mask.at[mask_idx].set(1)
    sensors = geometry.Sensor(domain=domain, binary_mask=mask)

    coeffs = jnp.arange(min(8, wpt.total_coeffs))
    pts, *_ = tr_solver_utils.compute_TR_parameters(coeffs, domain, wpt, sensors)
    normal_comp = pts[:, 0]

    finite = jnp.abs(normal_comp) > 1e-8  # ignore grazing packets
    assert jnp.any(finite), "expected at least one non-grazing packet"
    assert jnp.all(expected_sign * normal_comp[finite] >= 0)


def test_tr_beam_angles_point_inward_left_boundary():
    _assert_inward_normal(mask_idx=0, expected_sign=1.0)


def test_tr_beam_angles_point_inward_right_boundary():
    _assert_inward_normal(mask_idx=-1, expected_sign=-1.0)


def test_tr_beam_angles_use_spatial_domain_for_detector_side():
    def c_fn(x):
        return 0.01 + 0.0 * x[..., 0]

    data_domain, wpt = _make_2d_data_wpt(c_fn, data_dx=(1.0, 0.1))
    spatial_domain = geometry.Domain(
        N=(16, 16), dx=(0.1, 0.1), c=c_fn, periodic=(False, False)
    )
    mask = jnp.zeros(spatial_domain.N)
    mask = mask.at[-1, :].set(1)
    sensors = geometry.Sensor(domain=spatial_domain, binary_mask=mask)

    coeffs = jnp.arange(min(64, wpt.total_coeffs))
    pts, *_ = tr_solver_utils.compute_TR_parameters(
        coeffs, data_domain, wpt, sensors
    )
    normal_comp = pts[:, 0]

    decomp = wpt.dyadic_decomp
    shapes = utils.compute_coeff_shapes(
        decomp, wpt.redundancy, jnp.arange(decomp.num_levels)
    )
    cumsum_boxes = jnp.r_[0, jnp.cumsum(decomp.num_boxes_ndim)]
    nn_level, nn_idx = utils.find_tensor_and_multiindex(coeffs, shapes)
    box_idx = nn_idx[0, :] + cumsum_boxes[nn_level]
    centres_hat = decomp.centres_ndim[box_idx, :] / jnp.array(data_domain.grid_size)
    tau = centres_hat[:, 0]
    k_tan_sq = jnp.sum(centres_hat[:, 1:] ** 2, axis=1)
    hyperbolic = tau**2 - (0.01**2) * k_tan_sq > 1e-10

    assert jnp.any(hyperbolic), "expected at least one non-grazing packet"
    assert jnp.all(normal_comp[hyperbolic] <= 0)


def test_tr_temporal_carrier_sets_tangential_phase():
    def c_fn(x):
        return 0.3 + 0.0 * x[..., 0]

    data_domain, wpt = _make_2d_data_wpt(c_fn)
    spatial_domain = geometry.Domain(
        N=(16, 16), dx=(1.0, 1.0), c=c_fn, periodic=(False, False)
    )
    mask = jnp.zeros(spatial_domain.N)
    mask = mask.at[0, :].set(1)
    sensors = geometry.Sensor(domain=spatial_domain, binary_mask=mask)

    coeffs = jnp.arange(min(64, wpt.total_coeffs))
    pts, _, _, omegas, *_ = tr_solver_utils.compute_TR_parameters(
        coeffs, data_domain, wpt, sensors
    )

    decomp = wpt.dyadic_decomp
    shapes = utils.compute_coeff_shapes(
        decomp, wpt.redundancy, jnp.arange(decomp.num_levels)
    )
    cumsum_boxes = jnp.r_[0, jnp.cumsum(decomp.num_boxes_ndim)]
    nn_level, nn_idx = utils.find_tensor_and_multiindex(coeffs, shapes)
    box_idx = nn_idx[0, :] + cumsum_boxes[nn_level]
    centres_hat = decomp.centres_ndim[box_idx, :] / jnp.array(data_domain.grid_size)

    tau = centres_hat[:, 0]
    k_tan = centres_hat[:, 1:]
    usable = (jnp.abs(tau) > 1e-8) & (jnp.linalg.norm(k_tan, axis=1) > 1e-8)

    assert jnp.any(usable), "expected packets with nonzero tau and tangential carrier"
    assert jnp.allclose(omegas[usable], jnp.abs(tau[usable]))
    assert jnp.allclose(
        omegas[usable, None] * pts[usable, 1:],
        2.0 * jnp.pi * k_tan[usable],
    )


@pytest.mark.parametrize("d", [1, 2, 3])
def test_linear_system(d):
    """
    Test the linear system used in the TR solver.

    1. Generate a batch of symmetric matrices with positive definite imaginary parts.
    2. Check if the matrices are symmetric and have positive definite imaginary parts.
    3. Compute the linear system using the matrices.
    4. Check if the resulting matrix is symmetric and has positive definite imaginary parts.
    5. Reconstruct the original matrix from the linear system and check if it matches the original matrix.
    """
    b = 4  # batch size
    xt = jnp.ones((b, d))
    pt = jnp.ones((b, d))
    mode = jnp.ones((b,))

    key = jax.random.PRNGKey(42)
    mt_data = create_batched_symmetric_matrices_with_positive_definite_imag(b, d, key)

    # Verify each matrix in the batch is symmetric with positive definite imaginary part
    assert jnp.all(vmap_is_symmetric(mt_data))
    assert jnp.all(vmap_is_pos_def(jnp.imag(mt_data)))

    mt_img = tr_solver_utils.compute_mT_linear_system(xt, pt, None, mt_data, mode, c)
    assert jnp.all(vmap_is_symmetric(mt_img))
    assert jnp.all(vmap_is_pos_def(jnp.imag(mt_img)))

    mt_data_recon = tr_solver_utils.compute_mT_linear_system(
        xt, pt, mt_img, None, mode, c
    )
    assert jnp.allclose(mt_data, mt_data_recon, atol=1e-16)


def test_msgb_forward_then_time_reversal_1d():
    """
    1-D MSGB forward → time-reversal smoke with a small grid.
    - homogeneous c(x)
    - two simple wavepackets synthesized via compute_frames
    - one sensor at the left boundary
    - crop sensor data in frequency to N and run MSGB TR
    - assert correlation with ground-truth p0 is high and rel-L2 is reasonable
    """
    d = 1
    N = (512,) * d
    extent = (1, 1)
    dx = tuple([extent[i] / N[i] for i in range(d)])
    box_aspect_ratio = (1,) * d
    num_levels = 3
    num_boxes_levels = tuple([2 ** (level + 2) for level in range(num_levels)])

    windowing = "rectangular_mirror"
    redundancy = 2
    num_GB_img_space = N[0] * 2
    batch_size = 128
    input_type = "spatial"
    thr_strat = "top_n"
    sum_method = "scan_real"
    cfl = 0.5

    periodic = (False,) * d
    solver = gb_solvers.solve_ODE_base

    def c(x):
        return 1500 - 0 * (
            jnp.exp(-((x[..., 0] - extent[0] / 3) ** 2) / (0.1**2))
            - jnp.exp(-((x[..., 0] - 2 * extent[0] / 3) ** 2) / (0.1**2))
        )

    img_domain = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)

    XY = img_domain.grid

    ts = img_domain.generate_time_domain()
    tmax_img = ts[-1]
    Nt = len(ts)
    if Nt != 4 * N[0]:
        ts = jnp.linspace(0, tmax_img, 4 * N[0])
        tmax_img = ts[-1]
        Nt = len(ts)
        dt = ts[1] - ts[0]
        cfl = jnp.min(c(XY)) * dt / min(dx)
        img_domain = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)

    img_dyadic_decomp = DyadicDecomposition(
        num_levels, N, num_boxes_levels, box_aspect_ratio
    )

    # pltgb.plot_centers(img_dyadic_decomp.centres_ndim)

    img_wpt = transforms.MSWPT(img_dyadic_decomp, redundancy, windowing)

    binary_mask = jnp.zeros(N)
    binary_mask = binary_mask.at[0, ...].set(1)
    sensors = geometry.Sensor(domain=img_domain, binary_mask=binary_mask)
    sensors_all = geometry.Sensor(
        domain=img_domain, binary_mask=jnp.ones_like(binary_mask)
    )

    # ## Set up initial pressure

    ######################################
    ### INITIAL PRESSURE #################
    ######################################

    # TWO GBS

    KXY = img_dyadic_decomp.fourier_meshgrid

    # pltgb.plot_centers(img_dyadic_decomp.centres_ndim)

    boxhf = 4
    boxlf = 0

    khf = jnp.array(
        [
            10,
        ]
    )
    klf = jnp.array(
        [
            25,
        ]
    )
    kerft_hf = transforms.compute_frames(
        img_dyadic_decomp, boxhf, khf, KXY, redundancy, "none"
    )
    kerft_lf = transforms.compute_frames(
        img_dyadic_decomp, boxlf, klf, KXY, redundancy, "none"
    )
    p0 = utils.unitary_ifft(kerft_hf) + utils.unitary_ifft(kerft_lf)
    p0 = p0 / jnp.max(jnp.abs(p0))
    p0 = p0.T

    p0 = p0.real

    jnp.zeros_like(p0)

    img_dyadic_decomp.centres_ndim + jnp.array(N) // 2

    num_devices = jax.device_count()

    mesh = jax.make_mesh((num_devices,), ("x",))

    # Create sharding strategy
    sharding_strategy = ShardingStrategy(mesh, beam_axis="x")

    # Create solver with sharding
    msgb_solver = MSGBSolver(
        thr=num_GB_img_space,
        thr_strat=thr_strat,
        batch_size=batch_size,
        input_type=input_type,
        ode_solver=solver,
        tr_ode_solver=gb_solvers.solve_ODE_batch_t,
        sum_method=sum_method,
        sharding=sharding_strategy,
    )

    sensor_data_gb = msgb_solver.forward(
        p0,
        img_domain,
        sensors.positions,
        ts,
        img_wpt,
    )[0]

    def cut_out_middle(arr, size):
        mid = arr.shape[0] // 2
        return arr[mid - size // 2 : mid + size // 2]

    sensor_data_fft = utils.unitary_fft(sensor_data_gb)
    sensor_data_fft_cropped = cut_out_middle(sensor_data_fft, N[0])
    sensor_data_cropped = utils.unitary_ifft(sensor_data_fft_cropped)

    N_rect = jnp.squeeze(sensor_data_cropped).shape
    Nt = N_rect[0]
    ts = jnp.linspace(0, tmax_img, Nt)
    dt = float(ts[1] - ts[0])
    dx_rect = (dt,)
    box_aspect_ratio_rect = (1,)
    tmax_data = ts[-1]
    assert jnp.allclose(
        tmax_img, tmax_data
    ), f"tmax_img {tmax_img} and tmax_data {tmax_data} are not equal."

    data_domain = geometry.Domain(N=N_rect, dx=dx_rect, c=c, periodic=periodic, cfl=cfl)

    data_dyadic_decomp = DyadicDecomposition(
        num_levels, N_rect, num_boxes_levels, box_aspect_ratio_rect
    )

    data_wpt = transforms.MSWPT(data_dyadic_decomp, redundancy, windowing)

    sensor_data_gb = jnp.squeeze(sensor_data_cropped)

    p0_TR_msgb = msgb_solver.time_reversal(
        data=sensor_data_gb,
        domain=img_domain,
        sensors=sensors_all,
        sources=sensors,
        ts=ts,
        data_domain=data_domain,
        data_wpt=data_wpt,
    )[0]

    assert jnp.max(p0 - p0_TR_msgb) < 0.2


if __name__ == "__main__":
    pytest.main([__file__])
