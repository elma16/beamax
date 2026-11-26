import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamax import Domain, DyadicDecomposition, MSWPT
from beamax.gb.gb_solvers import solve_hom_diag, solve_hom_TR
from beamax.solvers import MSGBSolver
from beamax.surfaces import (
    DetectorSurface,
    match_packets,
    reduced_hessian,
    surface_time_reversal,
)


def test_mask_permutation_and_duplicate_grid_cells():
    domain = Domain((128, 128), (1e-4,) * 2, (False,) * 2)
    sparse = DetectorSurface.circle((6.4e-3, 6.4e-3), 6e-3, 32)
    positions = np.asarray(sparse.positions())
    cells = np.rint(positions / np.asarray(domain.dx)).astype(int)
    field = np.arange(np.prod(domain.N)).reshape(domain.N)
    data = field[np.asarray(sparse.to_sensor(domain).binary_mask, bool)]
    np.testing.assert_array_equal(
        data[sparse.mask_to_chart(domain)], field[tuple(cells.T)]
    )
    dense = DetectorSurface.circle((6.4e-3, 6.4e-3), 6e-3, 512)
    with pytest.raises(ValueError, match="distinct"):
        dense.to_sensor(domain)


def test_open_arc_padding_masks_packet_centres_and_retains_original_midpoint():
    surface = DetectorSurface.arc((0.0, 0.0), 1.0, 0.1, 1.7, 16)
    data, padded = surface.prepare_open_data(
        jnp.ones((8, 16)), padding=(8,), taper=(3,)
    )
    assert data.shape == (8, 32) and padded.shape == (32,)
    assert np.all(np.asarray(data)[:, :9] == 0) and np.all(
        np.asarray(data)[:, -9:] == 0
    )
    np.testing.assert_allclose(np.asarray(data)[:, 11:21], 1.0)
    vth, valid = padded.active_parameters(padded.params_flat())
    np.testing.assert_array_equal(
        valid, np.r_[np.zeros(8), np.ones(16), np.zeros(8)].astype(bool)
    )
    np.testing.assert_allclose(padded.points(vth[8:24]), surface.positions(), atol=2e-6)
    with pytest.raises(ValueError, match="Periodic"):
        DetectorSurface.circle((0.0, 0.0), 1.0, 16).prepare_open_data(
            jnp.ones((8, 16)), padding=(8,), taper=(0,)
        )

    with pytest.raises(ValueError, match="already has padded"):
        padded.prepare_open_data(data, padding=(2,), taper=(1,))
    with pytest.raises(ValueError, match="leave an interior"):
        surface.prepare_open_data(jnp.ones((8, 16)), padding=(0,), taper=(8,))
    with pytest.raises(ValueError, match="nonnegative integer"):
        surface.prepare_open_data(jnp.ones((8, 16)), padding=(-1,), taper=(0,))
    with pytest.raises(ValueError, match="data must have shape"):
        surface.prepare_open_data(jnp.ones((16, 8)), padding=(0,), taper=(0,))


def test_acquisition_grid_checks_match_chart():
    domain = Domain((16,), (0.1,), (False,))
    circle = DetectorSurface.circle((0.0, 0.0), 1.0, 16)
    with pytest.raises(ValueError, match="axes"):
        circle.check_data_domain(domain)
    with pytest.raises(ValueError, match="sensor axes"):
        circle.check_data_domain(Domain((16, 8), (0.1, *circle.spacing), (False, True)))
    with pytest.raises(ValueError, match="sensor spacing"):
        circle.check_data_domain(Domain((16, 16), (0.1, 1.0), (False, True)))


@pytest.fixture(
    params=["point", "oblique", "ellipse", "polar", "sphere", "cylinder", "torus"]
)
def surface_chart(request):
    make = {
        "point": lambda: DetectorSurface.point(0.2, front=-1),
        "oblique": lambda: DetectorSurface.parametric(
            lambda t: jnp.stack([t[0], t[1], 0.2 * t[0] + t[1]]),
            3,
            (0.0, 0.0),
            (0.1, 0.1),
            (8, 8),
            front_vector=(0.0, 0.0, 1.0),
        ),
        "ellipse": lambda: DetectorSurface.ellipse((0.0, 0.0), 3.0, 2.0, 16),
        "polar": lambda: DetectorSurface.polar(
            (0.0, 0.0), lambda t: 2.0 + 0.1 * jnp.cos(3 * t), 16
        ),
        "sphere": lambda: DetectorSurface.sphere_band(
            (0.0, 0.0, 0.0), 2.0, 0.5, 2.0, 8, 16
        ),
        "cylinder": lambda: DetectorSurface.cylinder(
            (0.0, 0.0, 0.0), 2.0, -1.0, 1.0, 8, 8
        ),
        "torus": lambda: DetectorSurface.torus((0.0, 0.0, 0.0), 3.0, 1.0, 8, 8),
    }
    return make[request.param]()


def test_chart_differential_geometry(surface_chart):
    s = surface_chart
    th = s.params_flat()[:: max(1, s.num_sensors // 4)]
    assert np.all(s.area_element(th) > 0)
    np.testing.assert_allclose(jnp.linalg.norm(s.normal(th), axis=-1), 1, atol=1e-6)
    if s.m:
        lhs = s.hessian(th)
        rhs = jnp.einsum(
            "bki,biac->bkac", s.jacobian(th), s.christoffel(th)
        ) + jnp.einsum("bk,bac->bkac", s.normal(th), s.second_fundamental_form(th))
        np.testing.assert_allclose(lhs, rhs, atol=1e-5)
        k = np.sort(np.linalg.eigvals(np.asarray(s.shape_operator(th))).real, axis=-1)
        np.testing.assert_allclose(s.principal_curvatures(th), k, atol=1e-5)


def test_float32_grazing_packet_has_finite_placeholder():
    surface = DetectorSurface.plane(
        2, axis=0, value=0.0, origin=(0.0,), spacing=(1.0,), shape=(8,)
    )
    with jax.enable_x64(False):
        tau = jnp.array([1320387.875])
        eta = jnp.nextafter(tau / 1500.0, 0.0)[:, None]
        launch = match_packets(
            surface,
            jnp.zeros((1, 1)),
            tau,
            eta,
            1j * jnp.eye(2)[None],
            lambda x: 1500.0 + 0 * x[0],
        )
        assert not bool(launch["hyperbolic"][0])
        assert np.isfinite(launch["M"]).all()


def test_curved_trace_hessian_round_trip(surface_chart):
    s = surface_chart
    th = s.params_flat()[:: max(1, s.num_sensors // 4)]
    count = len(th)
    tau = jnp.ones(count) * 1e6
    eta = jnp.zeros((count, s.m))
    mt = jnp.broadcast_to(1j * jnp.eye(s.ndim), (count, s.ndim, s.ndim))
    b = match_packets(s, th, tau, eta, mt, lambda x: 1.1 + 0.02 * x[0])
    matched = reduced_hessian(s, th, b["p"], b["M"], b["xdot"], b["pdot"])
    assert np.isfinite(b["M"]).all() and np.all(b["hyperbolic"])
    np.testing.assert_allclose(matched, mt, atol=3e-6)


def test_time_reversal_rejects_mismatched_coordinates_and_complex_traces():
    domain = Domain((16,), (0.1,), (False,))
    wpt = MSWPT(DyadicDecomposition(1, domain.N, (4,), (1,)), 2, "rectangular_mirror")
    surface = DetectorSurface.point()
    data = np.zeros(16)
    args = (domain, wpt, surface, domain, 8)
    with pytest.raises(ValueError, match="same shape"):
        surface_time_reversal(data[:-1], *args)
    with pytest.raises(ValueError, match="real-valued"):
        surface_time_reversal(data.astype(complex), *args)
    with pytest.raises(ValueError, match="coordinate axis"):
        surface_time_reversal(data, *args, eval_points=np.zeros((8, 2)))
    with pytest.raises(ValueError, match="aggregate_method"):
        surface_time_reversal(data, *args, aggregate_method="unknown")
    with pytest.raises(ValueError, match="top_n"):
        surface_time_reversal(data, domain, wpt, surface, domain, 3.5)
    periodic = Domain(domain.N, domain.dx, (True,))
    with pytest.raises(ValueError, match="free-space"):
        surface_time_reversal(data, domain, wpt, surface, periodic, 8)


@pytest.mark.parametrize("n,window", [(12, "rectangular_mirror"), (16, "rectangular")])
def test_unpaired_frame_surface_image_matches_full_complex_planar_reference(n, window):
    domain = Domain((n,), (0.1,), (False,), c=1.0)
    surface = DetectorSurface.point(0.0)
    wpt = MSWPT(DyadicDecomposition(1, domain.N, (4,), (1,)), 2, window)
    data = jnp.asarray(np.random.default_rng(893).normal(size=n))
    points = domain.grid
    full = MSGBSolver(
        thr=wpt.total_coeffs,
        thr_strat="top_n",
        batch_size=8,
        input_type="spatial",
        ode_solver=solve_hom_diag,
        tr_ode_solver=solve_hom_TR,
        sum_method="all_complex",
    )
    reference = full.time_reversal(
        data.copy().astype(complex),
        domain,
        points,
        surface.to_sensor(domain),
        jnp.arange(n) * 0.1,
        domain,
        wpt,
    )
    actual, diag = surface_time_reversal(
        data, domain, wpt, surface, domain, wpt.total_coeffs, batch_size=8
    )
    assert diag["coefficients"].shape == (wpt.total_coeffs,)
    np.testing.assert_allclose(actual, reference.real, rtol=2e-5, atol=2e-5)


def test_surface_time_reversal_roi_matches_full_grid():
    centre, radius = (0.0016, 0.0016), 0.001
    surf = DetectorSurface.circle(centre, radius, 32)
    dom = Domain((32, 32), (1e-4, 1e-4), (False, False))
    dd = Domain((32, 32), (5e-8, *surf.spacing), (False, True))
    wpt = MSWPT(DyadicDecomposition(1, dd.N, (4,), (1, 1)), 2, "rectangular_mirror")
    data = np.random.default_rng(10).normal(size=dd.N)
    kw = dict(batch_size=16)
    points = dom.grid[11:14, 18:22]
    full, _ = surface_time_reversal(data, dd, wpt, surf, dom, 64, **kw)
    roi, _ = surface_time_reversal(
        data, dd, wpt, surf, dom, 64, eval_points=points, **kw
    )
    np.testing.assert_allclose(roi, full[11:14, 18:22], atol=1e-5, rtol=1e-4)
    for method in ("terminal_xla", "all"):
        alternate, _ = surface_time_reversal(
            data,
            dd,
            wpt,
            surf,
            dom,
            64,
            eval_points=points,
            aggregate_method=method,
            **kw,
        )
        np.testing.assert_allclose(alternate, roi, atol=1e-5, rtol=1e-4)
