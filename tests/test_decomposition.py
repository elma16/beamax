from beamax.decomposition import DyadicDecomposition, validate_params
from beamax import utils
import jax.numpy as jnp
import pytest
import sys
import jax
from jax import tree_util

jax.config.update("jax_enable_x64", True)

common_params3d = [
    (
        num_levels,
        N,
        tuple([2 ** (level + 2) for level in range(num_levels)]),
        (1,) * len(N),
    )
    for num_levels in range(1, 3)
    for N in [
        (128,),
        (128, 128),
        (256, 128),
        (128, 256),
        (128, 128, 128),
        (256, 128, 128),
        (128, 256, 128),
        (128, 128, 256),
    ]
]


def test_rect_sqr_decomp_diff():
    """
    Test that the decomposition for a square domain with square boxes
    and a rectangular domain with rectangular boxes
    (such that the domain aspect ratio and the box aspect ratio are the same)

    1) the same number of boxes

    2) centres in different positions
    """
    N_sqr = (128, 128)
    num_levels = 2
    num_boxes_levels = (4, 8)
    box_aspect_ratio_sqr = (1, 1)
    dyadic_decomp_sqr = DyadicDecomposition(
        num_levels, N_sqr, num_boxes_levels, box_aspect_ratio_sqr
    )

    N_rect = (128, 256)
    box_aspect_ratio_rect = tuple([N_rect[i] / N_sqr[i] for i in range(len(N_sqr))])
    dyadic_decomp_rect = DyadicDecomposition(
        num_levels, N_rect, num_boxes_levels, box_aspect_ratio_rect
    )

    assert jnp.all(
        dyadic_decomp_sqr.num_boxes_ndim == dyadic_decomp_rect.num_boxes_ndim
    )
    assert not jnp.allclose(
        dyadic_decomp_sqr.centres_ndim, dyadic_decomp_rect.centres_ndim, atol=1e-16
    )


@pytest.mark.parametrize(
    "num_levels, N, num_boxes_outer_level, box_aspect_ratio", common_params3d
)
def test_init_valid_params(num_levels, N, num_boxes_outer_level, box_aspect_ratio):
    """
    Test that the DyadicDecomposition class can be initialized with valid parameters.
    """
    decomp = DyadicDecomposition(num_levels, N, num_boxes_outer_level, box_aspect_ratio)
    assert decomp.num_levels == num_levels


@pytest.mark.parametrize(
    "num_levels, N, num_boxes_outer_level, box_aspect_ratio", common_params3d
)
def test_dyadic_decomp_is_pytree(
    num_levels, N, num_boxes_outer_level, box_aspect_ratio
):
    """
    Test that the DyadicDecomposition class is a pytree.
    """
    decomp = DyadicDecomposition(num_levels, N, num_boxes_outer_level, box_aspect_ratio)
    flat, aux = tree_util.tree_flatten(decomp)
    reconstructed = tree_util.tree_unflatten(aux, flat)

    assert isinstance(reconstructed, DyadicDecomposition)
    assert jnp.all(reconstructed.scaling == decomp.scaling)
    assert jnp.all(reconstructed.box_lengths == decomp.box_lengths)
    assert jnp.all(reconstructed.num_boxes_ndim == decomp.num_boxes_ndim)
    assert reconstructed.total_num_boxes == decomp.total_num_boxes
    assert jnp.all(reconstructed.num_boxes_ndim_cumsum == decomp.num_boxes_ndim_cumsum)
    assert jnp.all(reconstructed.centres_ndim == decomp.centres_ndim)


@pytest.mark.parametrize(
    "num_levels, N, num_boxes_outer_level, box_aspect_ratio", common_params3d
)
def test_decomp_is_dyadic(num_levels, N, num_boxes_outer_level, box_aspect_ratio):
    """
    Test that the DyadicDecomposition class is dyadic. This means that we expect the length scales of boxes to Double, when moving from one scale to a higher one.
    """

    decomp = DyadicDecomposition(num_levels, N, num_boxes_outer_level, box_aspect_ratio)

    for i in range(1, num_levels):
        assert jnp.all(
            decomp.box_lengths[i] == 2 * decomp.box_lengths[i - 1]
        ), f"Box lengths at level {i} are not dyadic."


# @pytest.mark.parametrize(
#     "num_levels, N, num_boxes_outer_level, box_aspect_ratio", common_params3d
# )
# def test_dyadic_decomp_is_diff(num_levels, N, num_boxes_outer_level, box_aspect_ratio):
#     params = ()

#     def loss_fn(params):
#         decomp = DyadicDecomposition(params)
#         return (
#             jnp.sum(decomp.scaling)
#             + jnp.sum(decomp.box_lengths)
#             + jnp.sum(decomp.centres_ndim)
#         )

#     def compute_vjp(params):
#         _, vjp_fn = vjp(loss_fn, params)
#         return vjp_fn(jnp.ones_like(loss_fn(params)))

#     grads = compute_vjp(params)

#     flat_grads, _ = tree_util.tree_flatten(grads)
#     for g in flat_grads:
#         assert jnp.all(jnp.isfinite(g)), "Gradient contains non-finite values"


@pytest.mark.parametrize(
    "num_levels, N, num_boxes_outer_level, box_aspect_ratio", common_params3d
)
def test_dyadic_decomp_all_centres_filled(
    num_levels, N, num_boxes_outer_level, box_aspect_ratio
):
    """
    Check that none of the centres are all zeros, which would suggest incomplete initialization.
    """

    decomp = DyadicDecomposition(num_levels, N, num_boxes_outer_level, box_aspect_ratio)

    assert not jnp.any(jnp.all(decomp.centres_ndim == 0, axis=-1))


@pytest.mark.parametrize(
    "num_levels, N, num_boxes_outer_level, box_aspect_ratio", common_params3d
)
def test_centres_have_mirror_pairs(
    num_levels, N, num_boxes_outer_level, box_aspect_ratio
):
    """
    Check that each subband centre xi has a matching subband centre at (-xi mod N).
    """
    decomp = DyadicDecomposition(num_levels, N, num_boxes_outer_level, box_aspect_ratio)
    centres = decomp.centres_ndim

    num_boxes_level = decomp.num_boxes_ndim_cumsum - 1
    num_boxes_level = jnp.concatenate([jnp.array([0]), num_boxes_level], axis=0)

    for box in range(decomp.total_num_boxes):
        level = utils.find_level(decomp, box)
        box_conj = num_boxes_level[level + 1] - box + num_boxes_level[level]
        if level > 0:
            box_conj = box_conj + 1
        centre1 = centres[box]
        centre2 = centres[box_conj]
        assert jnp.allclose(centre1, -centre2)


def test_doubled_domain():
    d = 2
    N = (128,) * d
    N2 = tuple([2 * n for n in N])
    num_levels = 2
    num_boxes_outer_level = (4, 8)
    box_aspect_ratio = (1, 1)
    dyadic_decomp = DyadicDecomposition(
        num_levels, N, num_boxes_outer_level, box_aspect_ratio
    )
    dyadic_decomp_2N = DyadicDecomposition(
        num_levels, N2, num_boxes_outer_level, box_aspect_ratio
    )

    assert jnp.allclose(dyadic_decomp.centres_ndim * 2, dyadic_decomp_2N.centres_ndim)


def test_boxes_per_dim_override_rectangular_counts():
    d = DyadicDecomposition(
        num_levels=2,
        N=(512, 128),
        num_boxes_levels=(4, 8),
        box_aspect_ratio=(2, 1),
        boxes_per_dim_levels=((4, 4), (16, 8)),
    )
    assert d.num_boxes_ndim.tolist() == [16, 124]
    assert d.total_num_boxes == 16 + 124
    assert d.centres_ndim.shape == (d.total_num_boxes, 2)


def test_validate_params_error_branches():
    with pytest.raises(ValueError):
        validate_params(0, (16, 16), (2, 4), (1, 1))
    with pytest.raises(ValueError):
        validate_params(1, (15, 16), (2,), (1, 1))
    with pytest.raises(ValueError):
        validate_params(2, (16, 16), (2,), (1, 1))
    with pytest.raises(ValueError):
        validate_params(1, (16, 16), (2,), (2, 2))  # no 1 in aspect
    with pytest.raises(ValueError):
        validate_params(1, (16,), (2,), (2,))  # 1D aspect must be (1,)
    with pytest.raises(ValueError):
        validate_params(5, (16, 16), (50, 60, 70, 80, 90), (1, 1))  # base>N_ref


if __name__ == "__main__":
    pytest.main(sys.argv)
