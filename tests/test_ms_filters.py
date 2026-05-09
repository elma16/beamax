import pytest
import jax.numpy as jnp
import jax
from beamax import transforms, utils
from beamax.decomposition import DyadicDecomposition

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
            for windowing in ["rectangular", "rectangular_mirror"]:
                num_boxes_outer_level = tuple(
                    [2 ** (level + 2) for level in range(num_levels)]
                )
                box_aspect_ratio = (1,) * len(N)
                params.append(
                    (num_levels, N, num_boxes_outer_level, box_aspect_ratio, windowing)
                )

    # Add rectangular params
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
def setup_dyadic_decomposition(request):
    num_levels, N, num_boxes_outer_level, box_aspect_ratio, windowing = request.param
    N = tuple([N[i] * box_aspect_ratio[i] for i in range(len(N))])
    dyadic_decomp = DyadicDecomposition(
        num_levels, N, num_boxes_outer_level, box_aspect_ratio
    )
    gfilt, hfilt = transforms.compute_gh_filters(dyadic_decomp, redundancy, windowing)
    return dyadic_decomp, gfilt, hfilt


def assert_partition_of_unity(gfilt, hfilt):
    """
    Test that the gfilter and its conjugate provide a partition of unity.
    """
    assert jnp.allclose(jnp.sum(gfilt * hfilt, axis=0), 1, atol=1e-16)


def assert_gfilt_admissable(gfilt):
    """
    Test that the gfilter is bounded between 0 and 1

    1. 0 <= g_lj <= 1
    2. There exists C_n s.t for all xi |{(l,j) : g_lj(xi) > 0}| <= C_n
    3. There exists C_v s.t for all xi, there exists (l,j) s.t g_lj(xi) > C_v

    From the definition of the filter in Qian and Ying 2010.

    FAST MULTISCALE GAUSSIAN WAVEPACKET TRANSFORMS AND MULTISCALE GAUSSIAN BEAMS FOR THE WAVE EQUATION
    """
    assert jnp.all(gfilt >= 0) and jnp.all(
        gfilt <= 1
    ), "gfilter values are not bounded between 0 and 1"
    assert jnp.isfinite(jnp.max(jnp.sum((gfilt > 0), axis=0)))
    assert jnp.min(jnp.where(gfilt > 0, gfilt, jnp.inf)) > 0


@pytest.mark.parametrize("setup_dyadic_decomposition", all_params, indirect=True)
def test_filters(setup_dyadic_decomposition):
    _, gfilt, hfilt = setup_dyadic_decomposition
    assert_partition_of_unity(gfilt, hfilt)
    assert_gfilt_admissable(gfilt)


def test_hfilt_have_mirror_pairs():
    """
    Check that each hfilter has a mirror pair in the decomposition.
    """
    num_levels = 2
    N = (128, 64)
    box_aspect_ratio = (1, 1)
    num_boxes_outer_level = tuple([2 ** (level + 2) for level in range(num_levels)])
    windowing = "rectangular_mirror"
    redundancy = 2

    N = tuple([N[i] * box_aspect_ratio[i] for i in range(len(N))])
    decomp = DyadicDecomposition(num_levels, N, num_boxes_outer_level, box_aspect_ratio)
    gfilt, hfilt = transforms.compute_gh_filters(decomp, redundancy, windowing)
    num_boxes = decomp.num_boxes_ndim
    num_boxes_level = decomp.num_boxes_ndim_cumsum - 1

    for idx in range(num_boxes[0]):
        level = utils.find_level(decomp, idx)
        boxes_level = num_boxes_level[level]
        h1 = hfilt[idx]
        h2 = hfilt[boxes_level - idx]

        shape = h2.shape
        ndim = h2.ndim
        indices = jnp.indices(shape)
        sym_indices = tuple((shape[i] - indices[i]) % shape[i] for i in range(ndim))
        h2_flipped = h2[sym_indices]

        assert jnp.allclose(h1, h2_flipped)

        g1 = gfilt[idx]
        g2 = gfilt[boxes_level - idx]
        g2_flipped = g2[sym_indices]
        assert jnp.allclose(g1, g2_flipped)


if __name__ == "__main__":
    pytest.main([__file__])
