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


def _expected_binv(
    coeff_indices, domain_data, wpt_data, xts, sources, relative_guard=5e-2
):
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
    c_flat = sources.domain.c_fn(xts).reshape(-1)
    # The production TR builder is an advanced (terminal-value) propagator, so
    # its acquisition-time multiplier is the negative retarded symbol.
    binv = -adjoint_solver_utils.principal_b_inverse(
        tau, k_tan, c_flat, relative_guard=relative_guard
    )
    return binv.reshape(-1, 1)


def test_principal_b_inverse_has_correct_normal_incidence_sign_and_conjugacy():
    tau = jnp.array([3.0, -3.0])
    k_tan = jnp.array([[0.0], [0.0]])
    c = jnp.array([2.0, 2.0])

    binv = adjoint_solver_utils.principal_b_inverse(tau, k_tan, c)
    expected_positive = -1j / (4.0 * jnp.pi * c[0] * tau[0])

    assert jnp.allclose(binv[0], expected_positive)
    assert jnp.allclose(binv[1], jnp.conj(binv[0]))


def test_principal_b_inverse_removes_evanescent_and_grazing_modes():
    near_grazing_ratio = 5e-7
    tau = jnp.array([1.0, 2.0, 1.0, 0.0])
    k_tan = jnp.array(
        [
            [1.0],  # evanescent for c=2
            [1.0],  # exactly grazing for c=2
            [jnp.sqrt(1.0 - near_grazing_ratio**2)],
            [0.0],  # zero temporal frequency
        ]
    )
    c = jnp.array([2.0, 2.0, 1.0, 1.0])

    binv = adjoint_solver_utils.principal_b_inverse(tau, k_tan, c)

    assert jnp.allclose(binv, 0.0)


def test_principal_b_inverse_relative_guard_is_frequency_local():
    incidence_ratio = 1e-3
    tau = jnp.array([1.0, 1e6])
    k_tan = (tau * jnp.sqrt(1.0 - incidence_ratio**2))[:, None]
    c = jnp.ones_like(tau)

    binv = adjoint_solver_utils.principal_b_inverse(tau, k_tan, c)

    assert jnp.all(jnp.abs(binv) > 0.0)
    assert jnp.allclose(binv[0] / binv[1], tau[1] / tau[0], rtol=1e-6)


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

    expected_binv = _expected_binv(
        coeff_indices, domain_data, wpt_data, xts_adj, sensors
    )
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

    expected_binv = _expected_binv(coeff_indices, domain_data, wpt_data, xts, sensors)
    assert jnp.allclose(expected_binv, 0.0)
    assert jnp.allclose(ats_adj, ats_geom * expected_binv)


def test_zero_b_inverse_replaces_geometry_before_ode(monkeypatch):
    """A guarded packet must not carry its singular Hessian into propagation."""

    def c_fn(x):
        return 0.3 + 0.0 * x[..., 0]

    domain_data, wpt_data, sensors = _make_2d_setup(c_fn)
    coeff_indices = jnp.array([0, 5])

    monkeypatch.setattr(
        adjoint_solver_utils,
        "principal_b_inverse",
        lambda tau, k_tan, c, relative_guard: jnp.zeros_like(tau, dtype=jnp.complex128),
    )
    pts, Mts, _, _, ats, _, _ = adjoint_solver_utils.compute_adj_parameters(
        coeff_indices, domain_data, wpt_data, sensors
    )

    expected_m = 1j * jnp.eye(2)[None, :, :]
    assert jnp.allclose(pts, 1.0)
    assert jnp.allclose(Mts, expected_m)
    assert jnp.allclose(ats, 0.0)


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

    expected_binv = _expected_binv(
        coeff_indices, domain_data, wpt_data, adj_params[2], sensors
    )
    assert jnp.allclose(adj_params[4], tr_params[4] * expected_binv)
