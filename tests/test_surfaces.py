import jax.numpy as jnp
import numpy as np
import pytest

from beamax import Domain
from beamax.surfaces import DetectorSurface


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
