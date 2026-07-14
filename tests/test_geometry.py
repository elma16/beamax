import pytest
from beamax.geometry import Domain, Sensor
import jax
import jax.numpy as jnp
import sys


jax.config.update("jax_enable_x64", True)


def constant_one(x):
    return 1.0 + 0.0 * x[..., 0]


@pytest.mark.parametrize(
    "N, dx",
    [
        (
            (64,),
            (0.1,),
        ),
        (
            (64, 32),
            (0.1, 0.1),
        ),
        (
            (64, 32, 64),
            (0.1, 0.1, 0.1),
        ),
    ],
)
def test_domain_generate_meshgrid(N, dx):
    cfl = 0.3
    ndim = len(N)
    periodic = (False,) * ndim

    domain = Domain(N=N, dx=dx, c=constant_one, periodic=periodic, cfl=cfl)

    # check that i can compute the gradient and jit compile a function that uses it
    @jax.jit
    def f(x, domain):
        return x + domain.grid_size

    for i in range(ndim):
        f(i, domain)
        jax.jacobian(f, allow_int=True)(i, domain)

    assert domain.ndim == len(N)
    assert domain.cfl == cfl

    spatial_meshgrid, fourier_meshgrid = domain.generate_meshgrid()

    assert len(spatial_meshgrid) == len(N)
    assert len(fourier_meshgrid) == len(N)

    for i in range(len(N)):
        assert spatial_meshgrid[i].shape == tuple(N)
        assert fourier_meshgrid[i].shape == tuple(N)


def test_domain_compute_max_freq():
    N = (64, 128)
    dx = (0.1, 0.1)
    ndim = len(N)

    def c(x):
        return 2.0 + 0.0 * x[..., 0]

    periodic = (False,) * ndim

    domain = Domain(N=N, dx=dx, c=c, periodic=periodic, cfl=0.3)
    max_freq = domain.compute_max_freq()

    assert max_freq == 10.0


def test_domain_material_arrays_include_absorption():
    domain = Domain(
        N=(4, 4),
        dx=(0.1, 0.1),
        c=2.0,
        density=1.1,
        alpha_coeff=0.01,
        alpha_power=lambda x: 1.5 + 0.0 * x[..., 0],
        periodic=(False, False),
    )

    assert domain.sound_speed_array.shape == domain.N
    assert jnp.allclose(domain.density_array, 1.1)
    assert jnp.allclose(domain.alpha_coeff_array, 0.01)
    assert jnp.allclose(domain.alpha_power_array, 1.5)


def test_sensor():
    N = (128, 128)
    dx = (1 / N[0], 1 / N[1])
    ndim = len(N)

    def c(x):
        return jnp.exp(jnp.sin(x[0])) + 1.0

    periodic = (False,) * ndim
    domain = Domain(N=N, dx=dx, c=c, periodic=periodic, cfl=0.3)

    x = jnp.array([0.5, 0.5])
    sensor = Sensor(domain, positions=x)

    @jax.jit
    def f(x, sensor):
        return x + sensor.positions

    for i in range(ndim):
        f(i, sensor)
        jax.jacobian(f, allow_int=True)(i, sensor)

    guess_binary_mask = jnp.zeros(N)
    guess_binary_mask = guess_binary_mask.at[N[0] // 2, N[1] // 2].set(1)

    assert jnp.allclose(sensor.positions, x, atol=1e-16)
    assert jnp.allclose(sensor.binary_mask, guess_binary_mask, atol=1e-16)


def test_sensor_validation_errors():
    domain = Domain(N=(8, 8), dx=(0.1, 0.1), c=1.0, periodic=(False, False))

    with pytest.raises(ValueError, match="Either positions"):
        Sensor(domain)

    with pytest.raises(ValueError, match="Cannot provide both"):
        Sensor(domain, positions=jnp.array([0.1, 0.1]), binary_mask=jnp.ones(domain.N))

    with pytest.raises(ValueError, match="shape"):
        Sensor(domain, positions=jnp.array([0.1, 0.1, 0.1]))

    with pytest.raises(ValueError, match="inside"):
        Sensor(domain, positions=jnp.array([0.8, 0.1]))

    with pytest.raises(ValueError, match="shape"):
        Sensor(domain, binary_mask=jnp.ones((8,)))

    with pytest.raises(ValueError, match="positive"):
        Sensor(domain, binary_mask=jnp.zeros(domain.N))


def test_sensor_positions_clip_to_grid_boundary():
    domain = Domain(N=(8,), dx=(0.1,), c=1.0, periodic=(False,))
    sensor = Sensor(domain, positions=jnp.array([0.79]))

    expected = jnp.zeros(domain.N).at[7].set(1)
    assert jnp.array_equal(sensor.binary_mask, expected)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_geom_c(d):
    """
    Test the geometry module with different input c
    """
    N = (128,) * d
    dx = (1 / N[0],) * d
    periodic = (False,) * d

    c = 2

    domain = Domain(N=N, dx=dx, c=c, periodic=periodic, cfl=0.3)

    assert jnp.allclose(domain.compute_max_speed(), c)
    assert jnp.allclose(domain.compute_min_speed(), c)

    c = jnp.ones(N) * 2.0

    domain = Domain(N=N, dx=dx, c=c, periodic=periodic, cfl=0.3)

    assert jnp.allclose(domain.compute_max_speed(), 2.0)
    assert jnp.allclose(domain.compute_min_speed(), 2.0)

    def c(x):
        return 1 + 0.0 * x[..., 0]

    domain = Domain(N=N, dx=dx, c=c, periodic=periodic, cfl=0.3)

    assert jnp.allclose(domain.compute_max_speed(), 1.0)
    assert jnp.allclose(domain.compute_min_speed(), 1.0)


def test_grid_valued_sound_speed_is_interpolated_at_ray_points():
    x = jnp.arange(4) * 0.1
    y = jnp.arange(5) * 0.2
    values = 1.0 + x[:, None] + 2.0 * y[None, :]
    domain = Domain(
        N=values.shape,
        dx=(0.1, 0.2),
        c=values,
        periodic=(False, False),
    )

    points = jnp.array([[0.15, 0.30], [0.05, 0.10]])
    assert jnp.allclose(domain.c_fn(points), 1.0 + points[:, 0] + 2.0 * points[:, 1])


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"N": (4.5,), "dx": (0.1,), "periodic": (False,)}, "integers"),
        ({"N": (4,), "dx": (0.0,), "periodic": (False,)}, "positive"),
        ({"N": (4,), "dx": (0.1, 0.2), "periodic": (False,)}, "length"),
        ({"N": (4,), "dx": (0.1,), "periodic": ()}, "length"),
        ({"N": (4,), "dx": (0.1,), "periodic": (False,), "c": -1}, "positive"),
        (
            {
                "N": (4,),
                "dx": (0.1,),
                "periodic": (False,),
                "density": jnp.ones((5,)),
            },
            "shape",
        ),
    ],
)
def test_domain_rejects_invalid_geometry_and_medium(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Domain(**kwargs)


def test_sensor_rejects_nonbinary_masks_and_quantised_duplicates():
    domain = Domain(N=(4,), dx=(0.1,), c=1.0, periodic=(False,))
    with pytest.raises(ValueError, match="binary values"):
        Sensor(domain, binary_mask=jnp.array([0.0, 0.5, 0.0, 1.0]))
    with pytest.raises(ValueError, match="distinct grid points"):
        Sensor(domain, positions=jnp.array([[0.01], [0.02]]))


if __name__ == "__main__":
    pytest.main(sys.argv)
