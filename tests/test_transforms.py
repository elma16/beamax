import pytest
import jax
import jax.numpy as jnp
from beamax.decomposition import DyadicDecomposition
from beamax import transforms, utils

jax.config.update("jax_enable_x64", True)


redundancy = 2


def generate_test_params():
    params = []
    for num_levels in range(1, 3):
        for N in [
            (128,),
            (128, 128),
            (256, 128),
            (128, 256),
        ]:
            num_boxes_outer_level = tuple(
                [2 ** (level + 2) for level in range(num_levels)]
            )
            box_aspect_ratio = (1,) * len(N)
            for windowing in ["rectangular", "rectangular_mirror"]:
                params.append(
                    (num_levels, N, num_boxes_outer_level, box_aspect_ratio, windowing)
                )

    # rectangular params
    for num_levels in range(1, 3):
        N = (128, 128)
        num_boxes_outer_level = tuple([2 ** (level + 2) for level in range(num_levels)])
        for box_aspect_ratio in [
            (1, 1),
            (2, 1),
            (4, 1),
            (1, 2),
            (1, 4),
        ]:
            for windowing in ["rectangular", "rectangular_mirror"]:
                params.append(
                    (num_levels, N, num_boxes_outer_level, box_aspect_ratio, windowing)
                )

    return params


all_params = generate_test_params()


@pytest.fixture
def setup_transform(request):
    num_levels, N, num_boxes_outer_level, box_aspect_ratio, windowing = request.param
    N = tuple([N[i] * box_aspect_ratio[i] for i in range(len(N))])

    dyadic_decomp = DyadicDecomposition(
        num_levels, N, num_boxes_outer_level, box_aspect_ratio
    )
    wpt = transforms.MSWPT(dyadic_decomp, redundancy, windowing)
    return wpt, N, dyadic_decomp


@pytest.fixture
def random_input(setup_transform):
    _, N, _ = setup_transform
    key = jax.random.PRNGKey(0)
    return jax.random.normal(key, N) + 1j * jax.random.normal(key, N)


@pytest.mark.parametrize("setup_transform", all_params, indirect=True)
def test_inv_fwd_is_f(setup_transform, random_input):
    """Test that the inverse transform of the forward transform is the original function"""
    wpt, _, _ = setup_transform
    p0 = random_input
    input_type = "spatial"

    coeffs = wpt.forward(p0, input_type)
    p0_recon = wpt.inverse(coeffs, input_type)

    assert jnp.allclose(p0, p0_recon, atol=1e-16)


@pytest.mark.parametrize("setup_transform", all_params, indirect=True)
def test_inv_fwd_inv_is_f(setup_transform):
    """Test that the inverse, and the forward applied to the inverse give the same result."""
    wpt, N, dyadic_decomp = setup_transform
    key = jax.random.PRNGKey(0)

    coeffs = jax.random.normal(
        key, (2 ** (dyadic_decomp.ndim) * jnp.prod(jnp.array(N)),)
    )

    f_rec = wpt.inverse(coeffs, "spatial")
    coeffs_rec = wpt.forward(f_rec, "spatial")
    f_rec_rec = wpt.inverse(coeffs_rec, "spatial")

    assert jnp.allclose(f_rec, f_rec_rec, atol=1e-16)


@pytest.mark.parametrize("setup_transform", all_params, indirect=True)
def test_fwd_is_linear_levels(setup_transform):
    """
    Test that the forward transform is linear
    """
    wpt, N, dyadic_decomp = setup_transform
    num_levels = dyadic_decomp.num_levels
    key = jax.random.PRNGKey(0)

    coeffs = jax.random.normal(
        key, (2 ** (dyadic_decomp.ndim) * jnp.prod(jnp.array(N)),)
    )

    coeff_shapes = utils.compute_coeff_shapes(
        dyadic_decomp, redundancy, jnp.arange(num_levels)
    )
    coeffs_cumsum = jnp.concatenate(
        [jnp.array([0]), jnp.cumsum(jnp.prod(coeff_shapes, axis=1))]
    )
    coeffs_level = [jnp.zeros_like(coeffs) for _ in range(num_levels)]
    for level in range(num_levels):
        coeff_idx_prev, coeff_idx_next = coeffs_cumsum[level], coeffs_cumsum[level + 1]
        coeffs_level[level] = (
            coeffs_level[level]
            .at[coeff_idx_prev:coeff_idx_next]
            .set(coeffs[coeff_idx_prev:coeff_idx_next])
        )

    f = wpt.inverse(coeffs, "spatial")
    f_sum = sum(
        wpt.inverse(coeffs_level[level], "spatial") for level in range(num_levels)
    )

    assert jnp.allclose(f, f_sum, atol=1e-16)


@pytest.mark.parametrize("setup_transform", all_params, indirect=True)
def test_fwd_transform_linear(setup_transform):
    """
    Test that the forward transform is linear

    i.e: F(c1 + c2) = F(c1) + F(c2)
     and F(a * c1) = a * F(c1)
    """
    wpt, N, _ = setup_transform
    key = jax.random.PRNGKey(0)

    input_type = "spatial"
    f1 = jax.random.normal(key, N)
    f2 = jax.random.normal(key, N)
    a = jax.random.normal(key)

    c1 = wpt.forward(f1, input_type)
    c2 = wpt.forward(f2, input_type)
    c_sum = wpt.forward(f1 + f2, input_type)
    c_scaled = wpt.forward(a * f1, input_type)

    assert jnp.allclose(c_sum, c1 + c2, atol=1e-16)
    assert jnp.allclose(c_scaled, a * c1, atol=1e-16)


@pytest.mark.parametrize("setup_transform", all_params, indirect=True)
def test_inv_transform_linear(setup_transform):
    wpt, N, _ = setup_transform
    key = jax.random.PRNGKey(0)

    total_coeffs = jnp.prod(redundancy * jnp.array(N))

    input_type = "spatial"
    c1 = jax.random.normal(key, (total_coeffs,))
    c2 = jax.random.normal(key, (total_coeffs,))
    a = jax.random.normal(key)

    f1 = wpt.inverse(c1, input_type)
    f2 = wpt.inverse(c2, input_type)
    f_sum = wpt.inverse(c1 + c2, input_type)
    f_scaled = wpt.inverse(a * c1, input_type)

    assert jnp.allclose(f_sum, f1 + f2, atol=1e-16)
    assert jnp.allclose(f_scaled, a * f1, atol=1e-16)


if __name__ == "__main__":
    pytest.main([__file__])
