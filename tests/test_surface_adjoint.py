import jax.numpy as jnp
import numpy as np
import pytest

from beamax import Domain, DyadicDecomposition, MSWPT
from beamax.gb.gb_solvers import (
    SolverConfig,
    solve_hom_diag,
    solve_hom_TR,
    solve_ODE_batch_t,
)
from beamax.solvers import MSGBSolver
from beamax.surfaces import DetectorSurface, surface_adjoint


def _transform(domain, boxes=4):
    decomposition = DyadicDecomposition(1, domain.N, (boxes,), (1,) * domain.ndim)
    return MSWPT(decomposition, 2, "rectangular_mirror")


def test_point_adjoint_matches_one_dimensional_wave_equation():
    # For initial pressure supported in x > 0, d'Alembert gives P f(t) = f(ct)/2.
    # The continuous adjoint under dx and dt is therefore r(x/c)/(2c).
    n, c = 128, 1.6
    dt = 1.0 / n
    acquisition = Domain((n,), (dt,), (False,))
    domain = Domain((n,), (c * dt,), (False,), c=c)
    wpt = _transform(acquisition, boxes=8)
    times = np.arange(n) * dt
    residual = np.cos(2 * np.pi * 40 * times)
    window = np.exp(-(((times - 0.5) / 0.13) ** 2))
    image, _ = surface_adjoint(
        residual,
        acquisition,
        wpt,
        DetectorSurface.point(),
        domain,
        wpt.total_coeffs,
        window=window,
        batch_size=32,
    )
    expected = window * residual / (2 * c)
    # The inverse symbol is evaluated at the packet carrier across its bandwidth.
    assert np.linalg.norm(image - expected) < 0.04 * np.linalg.norm(expected)


@pytest.mark.parametrize("heterogeneous", [False, True])
def test_surface_area_on_a_plane_matches_the_planar_msgb_adjoint(heterogeneous):
    n, dt, ds = 16, 0.07, 0.16

    def speed(x):
        # Varies along the detector line and between that line and the image.
        return 1.4 + 0.18 * x[..., 0] + 0.23 * x[..., 1]

    domain = Domain((n, n), (ds, ds), (False, False), c=speed if heterogeneous else 1.4)
    surface = DetectorSurface.plane(2, spacing=(ds,), shape=(n,))
    acquisition = Domain((n, n), (dt, ds), (False, False))
    wpt = _transform(acquisition)
    data = jnp.asarray(np.random.default_rng(614).normal(size=acquisition.N))
    window = jnp.asarray(np.hanning(n))
    points = domain.grid[1:4, 3:6]
    ode_config = SolverConfig.from_precision(dt0=0.001) if heterogeneous else None
    solver = MSGBSolver(
        thr=32,
        thr_strat="top_n",
        batch_size=8,
        input_type="spatial",
        ode_solver=solve_hom_diag,
        tr_ode_solver=solve_ODE_batch_t if heterogeneous else solve_hom_TR,
        ode_config=ode_config,
        sum_method="scan_real",
    )
    expected = solver.adjoint(
        data,
        domain,
        points,
        surface.to_sensor(domain),
        jnp.arange(n) * dt,
        acquisition,
        wpt,
        window=window,
    )
    image, _ = surface_adjoint(
        data,
        acquisition,
        wpt,
        surface,
        domain,
        32,
        window=window,
        eval_points=points,
        batch_size=8,
        solver_config=ode_config,
    )
    assert np.linalg.norm(expected) > 0
    np.testing.assert_allclose(image, expected, atol=2e-6, rtol=2e-5)


def test_ellipse_element_pairing_uses_each_detectors_arc_length():
    n, a, b = 16, 1.3, 0.65
    centre = (1.5, 1.5)
    surface = DetectorSurface.ellipse(centre, a, b, n)
    acquisition = Domain((n, n), (0.1, *surface.spacing), (False, True))
    domain = Domain((n, n), (0.2,) * 2, (False, False), c=1.0)
    data = np.random.default_rng(918).normal(size=acquisition.N)
    theta = np.arange(n) * 2 * np.pi / n
    ds = np.hypot(a * np.sin(theta), b * np.cos(theta)) * 2 * np.pi / n
    args = (acquisition, _transform(acquisition), surface, domain, 48)
    kwargs = dict(eval_points=domain.grid[5:9, 6:11], batch_size=16)
    by_area, _ = surface_adjoint(data, *args, pairing="surface_area", **kwargs)
    by_element, _ = surface_adjoint(data * ds, *args, pairing="element_sum", **kwargs)
    assert np.linalg.norm(by_area) > 0
    np.testing.assert_allclose(by_element, by_area, rtol=3e-5, atol=2e-6)


def test_skew_curved_patch_symbol_uses_the_full_metric_and_grazing_guard():
    # X(u,v) = (u+s*v, v, u²/2) has G_uu=1+u², G_uv=s, G_vv=1+s².
    n, shear, c, guard = 8, 0.6, 1.3, 0.75
    surface = DetectorSurface.parametric(
        lambda t: jnp.stack([t[0] + shear * t[1], t[1], 0.5 * t[0] ** 2]),
        3,
        (0.0, 0.0),
        (0.16, 0.16),
        (n, n),
        front_vector=(0.0, 0.0, 1.0),
    )
    acquisition = Domain((n, n, n), (0.1, *surface.spacing), (False,) * 3)
    domain = Domain((n, n, n), (0.2,) * 3, (False,) * 3, c=c)
    residual = np.random.default_rng(378).normal(size=acquisition.N)
    image, diag = surface_adjoint(
        residual,
        acquisition,
        _transform(acquisition),
        surface,
        domain,
        192,
        guard=guard,
        eval_points=jnp.array([[0.7, 0.3, 0.2]]),
        batch_size=32,
    )
    tau = np.asarray(diag["tau"])
    u = np.asarray(diag["vth"])[:, 0]
    ku, kv = np.asarray(diag["eta"]).T
    det = 1 + (1 + shear**2) * u**2
    tangential_sq = (
        (1 + shear**2) * ku**2 - 2 * shear * ku * kv + (1 + u**2) * kv**2
    ) / det
    rad = tau**2 - c**2 * tangential_sq
    valid = rad > (guard * tau) ** 2
    expected = np.zeros_like(tau, dtype=complex)
    expected[valid] = 1j * np.sign(tau[valid]) / (4 * np.pi * c * np.sqrt(rad[valid]))
    assert np.any(valid) and np.any((rad > 0) & ~valid) and np.any(rad < 0)
    np.testing.assert_allclose(
        diag["boundary_multiplier"], expected, rtol=2e-6, atol=1e-8
    )
    assert np.isfinite(image).all()


def test_padded_open_chart_excludes_data_and_packets_outside_the_aperture():
    # This chart is regular for all measured u > 0 but undefined for negative u.
    surface = DetectorSurface.parametric(
        lambda u: jnp.stack([u[0], jnp.sqrt(u[0])]),
        2,
        (0.1,),
        (0.1,),
        (8,),
        front_vector=(0.0, 1.0),
    )
    data, padded = surface.prepare_open_data(
        np.random.default_rng(44).normal(size=(16, 8)), padding=(4,), taper=(0,)
    )
    acquisition = Domain(data.shape, (0.08, *padded.spacing), (False, False))
    domain = Domain((16, 16), (0.1, 0.1), (False, False), c=1.2)
    args = (acquisition, _transform(acquisition), padded, domain, 96)
    kw = dict(eval_points=domain.grid[2:5, 7:10], pairing="element_sum", batch_size=16)
    expected, _ = surface_adjoint(data, *args, **kw)
    image, diag = surface_adjoint(data.at[:, :4].set(1.0e4), *args, **kw)
    outside = ~np.asarray(diag["inside_chart"])
    assert np.any(outside) and np.linalg.norm(expected) > 0
    np.testing.assert_array_equal(np.asarray(diag["boundary_multiplier"])[outside], 0)
    np.testing.assert_allclose(image, expected, atol=1e-7)


def test_surface_adjoint_rejects_invalid_pairing_guard_and_absorption():
    domain = Domain((16,), (0.1,), (False,), c=2.0)
    damped = Domain(domain.N, domain.dx, domain.periodic, c=2.0, lam=0.4)
    args = (jnp.zeros(domain.N), domain, _transform(domain), DetectorSurface.point())
    with pytest.raises(ValueError, match="pairing"):
        surface_adjoint(*args, domain, 8, pairing="unknown")
    with pytest.raises(ValueError, match="guard"):
        surface_adjoint(*args, domain, 8, guard=1.0)
    with pytest.raises(NotImplementedError, match="lossless"):
        surface_adjoint(*args, damped, 8)
    with pytest.raises(ValueError, match="real"):
        surface_adjoint(1j * args[0], *args[1:], domain, 8)
    with pytest.raises(ValueError, match="window"):
        surface_adjoint(*args, domain, 8, window=np.full(domain.N, np.nan))
