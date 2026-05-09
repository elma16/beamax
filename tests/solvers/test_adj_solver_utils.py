import jax
import jax.numpy as jnp

from beamax import geometry, transforms, utils
from beamax.decomposition import DyadicDecomposition
from beamax.solvers.msgb_solvers import adjoint_solver_utils, tr_solver_utils


jax.config.update("jax_enable_x64", True)


def _make_2d_setup(c_fn):
    N = (16, 16)
    dx = (1.0, 1.0)
    domain_data = geometry.Domain(N=N, dx=dx, c=c_fn, periodic=(False, False))
    decomp = DyadicDecomposition(
        num_levels=1, N=N, num_boxes_levels=(2,), box_aspect_ratio=(1, 1)
    )
    wpt_data = transforms.MSWPT(decomp, redundancy=2, windowing="rectangular")

    domain_spatial = geometry.Domain(N=N, dx=dx, c=c_fn, periodic=(False, False))
    mask = jnp.zeros(N)
    mask = mask.at[0, :].set(1)
    sensors = geometry.Sensor(domain=domain_spatial, binary_mask=mask)
    return domain_data, wpt_data, sensors


def _make_1d_setup(c_fn):
    N = (16,)
    dx = (1.0,)
    domain_data = geometry.Domain(N=N, dx=dx, c=c_fn, periodic=(False,))
    decomp = DyadicDecomposition(
        num_levels=1, N=N, num_boxes_levels=(2,), box_aspect_ratio=(1,)
    )
    wpt_data = transforms.MSWPT(decomp, redundancy=2, windowing="rectangular")

    domain_spatial = geometry.Domain(N=N, dx=dx, c=c_fn, periodic=(False,))
    mask = jnp.zeros(N)
    mask = mask.at[0].set(1)
    sensors = geometry.Sensor(domain=domain_spatial, binary_mask=mask)
    return domain_data, wpt_data, sensors


def _expected_binv(coeff_indices, domain_data, wpt_data, xts):
    decomp = wpt_data.dyadic_decomp
    shapes = utils.compute_coeff_shapes(
        decomp, wpt_data.redundancy, jnp.arange(decomp.num_levels)
    )
    cumsum_boxes = jnp.r_[0, jnp.cumsum(decomp.num_boxes_ndim)]
    nn_level, nn_idx = utils.find_tensor_and_multiindex(coeff_indices, shapes)
    box_idx = nn_idx[0, :] + cumsum_boxes[nn_level]
    centres_hat = decomp.centres_ndim[box_idx, :] / jnp.array(domain_data.grid_size)

    tau = centres_hat[:, 0]
    k_tan = centres_hat[:, 1:]
    omega = jnp.abs(tau)

    if k_tan.shape[1] > 0:
        k_tan_sq = jnp.sum(k_tan**2, axis=1)
    else:
        k_tan_sq = jnp.zeros_like(omega)

    c_flat = domain_data.c(xts).reshape(-1)
    rad = omega**2 - (c_flat**2) * k_tan_sq
    rad = jnp.maximum(rad, 0.0)
    gamma = jnp.sqrt(rad)
    eps = 1e-6 * (1.0 + jnp.max(gamma))
    binv = jnp.where(gamma > eps, 1j / (2.0 * c_flat * gamma), 0.0)
    return binv.reshape(-1, 1)


def test_compute_adj_parameters_matches_tr_geometry_and_binv():
    def c_fn(x):
        return 0.3 + 0.01 * x[..., 1]

    domain_data, wpt_data, sensors = _make_2d_setup(c_fn)

    coeff_indices = jnp.array([0, 5, 256, 261])

    tr_params = tr_solver_utils.compute_TR_parameters(
        coeff_indices, domain_data, wpt_data, sensors
    )
    adj_params = adjoint_solver_utils.compute_adj_parameters(
        coeff_indices, domain_data, wpt_data, sensors
    )

    pts_tr, Mts_tr, xts_tr, omegas_tr, ats_geom, signum_tr, ts_tr = tr_params
    pts_adj, Mts_adj, xts_adj, omegas_adj, ats_adj, signum_adj, ts_adj = adj_params

    assert jnp.allclose(pts_tr, pts_adj)
    assert jnp.allclose(Mts_tr, Mts_adj)
    assert jnp.allclose(xts_tr, xts_adj)
    assert jnp.allclose(omegas_tr, omegas_adj)
    assert jnp.allclose(signum_tr, signum_adj)
    assert jnp.allclose(ts_tr, ts_adj)

    expected_binv = _expected_binv(coeff_indices, domain_data, wpt_data, xts_adj)
    assert jnp.allclose(ats_adj, ats_geom * expected_binv)


def test_compute_adj_parameters_zero_for_evanescent_packets():
    def c_fn(x):
        return 10.0 + 0.0 * x[..., 0]

    domain_data, wpt_data, sensors = _make_2d_setup(c_fn)

    coeff_indices = jnp.array([0, 256, 512, 768])
    _, _, xts, _, ats_geom, _, _ = tr_solver_utils.compute_TR_parameters(
        coeff_indices, domain_data, wpt_data, sensors
    )
    _, _, _, _, ats_adj, _, _ = adjoint_solver_utils.compute_adj_parameters(
        coeff_indices, domain_data, wpt_data, sensors
    )

    expected_binv = _expected_binv(coeff_indices, domain_data, wpt_data, xts)
    assert jnp.allclose(expected_binv, 0.0)
    assert jnp.allclose(ats_adj, ats_geom * expected_binv)


def test_compute_adj_parameters_1d_no_tangential_modes():
    def c_fn(x):
        return 0.5 + 0.0 * x[..., 0]

    domain_data, wpt_data, sensors = _make_1d_setup(c_fn)

    coeff_indices = jnp.array([0, 16])
    tr_params = tr_solver_utils.compute_TR_parameters(
        coeff_indices, domain_data, wpt_data, sensors
    )
    adj_params = adjoint_solver_utils.compute_adj_parameters(
        coeff_indices, domain_data, wpt_data, sensors
    )

    expected_binv = _expected_binv(coeff_indices, domain_data, wpt_data, adj_params[2])
    assert jnp.allclose(adj_params[4], tr_params[4] * expected_binv)
