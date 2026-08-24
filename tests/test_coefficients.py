from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from beamax import utils
from beamax.decomposition import DyadicDecomposition
from beamax.coefficients import streamed_top_n_coefficients
from beamax.transforms import MSWPT


def _make_wpt(
    shape: tuple[int, ...],
    windowing: str,
    *,
    levels: int = 1,
) -> MSWPT:
    boxes = (4,) if levels == 1 else (4, 8)
    decomp = DyadicDecomposition(
        levels,
        shape,
        boxes,
        (1,) * len(shape),
    )
    return MSWPT(decomp, redundancy=2, windowing=windowing)


def _eligible_indices(wpt: MSWPT, positive_half: bool) -> np.ndarray:
    indices: list[int] = []
    for start, end in zip(wpt.coeffs_cumsum[:-1], wpt.coeffs_cumsum[1:]):
        stop = start + (end - start) // 2 if positive_half else end
        indices.extend(range(start, stop))
    return np.asarray(indices, dtype=np.int32)


def _materialized_reference(
    data: jax.Array,
    wpt: MSWPT,
    *,
    top_n: int,
    input_type: str,
    positive_half: bool,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.asarray(scale * wpt.forward(data, input_type))
    eligible = _eligible_indices(wpt, positive_half)
    count = min(top_n, eligible.size)

    # Selection is descending magnitude with the lower flat index winning
    # exact ties.  Downstream order is ascending magnitude, then index.
    winners = sorted(
        eligible.tolist(),
        key=lambda index: (-float(abs(coefficients[index])), index),
    )[:count]
    winners = sorted(
        winners,
        key=lambda index: (float(abs(coefficients[index])), index),
    )
    winner_array = np.asarray(winners, dtype=np.int32)
    return winner_array, coefficients[winner_array]


@pytest.mark.parametrize(
    ("shape", "windowing", "input_type", "positive_half"),
    [
        ((16,), "rectangular", "spatial", False),
        ((16,), "rectangular_mirror", "fourier", True),
        ((8, 8), "rectangular", "fourier", True),
        ((8, 8), "rectangular_mirror", "spatial", False),
        ((8, 8, 8), "rectangular", "spatial", False),
        ((8, 8, 8), "rectangular_mirror", "fourier", True),
    ],
)
def test_streamed_top_n_matches_materialized_mswpt(
    shape, windowing, input_type, positive_half
):
    wpt = _make_wpt(shape, windowing)
    spatial = jax.random.normal(jax.random.PRNGKey(len(shape) * 17), shape)
    data = utils.unitary_fft(spatial) if input_type == "fourier" else spatial

    indices, values = streamed_top_n_coefficients(
        data,
        wpt,
        top_n=11,
        input_type=input_type,
        positive_half=positive_half,
        scale=0.5,
        boxes_per_chunk=3,
    )
    expected_indices, expected_values = _materialized_reference(
        data,
        wpt,
        top_n=11,
        input_type=input_type,
        positive_half=positive_half,
        scale=0.5,
    )

    assert np.array_equal(np.asarray(indices), expected_indices)
    assert np.allclose(np.asarray(values), expected_values, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    ("shape", "half_support"),
    [
        ((12,), (3,)),
        ((16,), (4,)),
        ((12, 12), (3, 3)),
        ((16, 16), (4, 4)),
    ],
)
def test_streamed_and_materialized_per_box_analysis_is_exact_for_support_parities(
    shape, half_support
):
    wpt = _make_wpt(shape, "rectangular")
    assert wpt._box_shapes == [half_support]

    real_key, imag_key = jax.random.split(jax.random.PRNGKey(sum(shape)))
    data = jax.random.normal(real_key, shape) + 1j * jax.random.normal(imag_key, shape)

    indices, values = streamed_top_n_coefficients(
        data,
        wpt,
        top_n=wpt.total_coeffs,
        input_type="fourier",
        boxes_per_chunk=1,
    )
    streamed = jnp.zeros_like(values).at[indices].set(values)
    materialized = wpt.forward(data, "fourier")

    assert jnp.array_equal(streamed, materialized)


def test_streamed_selection_merges_candidates_across_levels_and_partial_chunks():
    wpt = _make_wpt((32, 32), "rectangular_mirror", levels=2)
    data = jax.random.normal(jax.random.PRNGKey(33), (32, 32))

    indices, values = streamed_top_n_coefficients(
        data,
        wpt,
        top_n=31,
        positive_half=True,
        scale=0.5,
        boxes_per_chunk=5,
    )
    expected_indices, expected_values = _materialized_reference(
        data,
        wpt,
        top_n=31,
        input_type="spatial",
        positive_half=True,
        scale=0.5,
    )

    levels = np.searchsorted(
        np.asarray(wpt.coeffs_cumsum[1:]), np.asarray(indices), side="right"
    )
    assert np.array_equal(np.unique(levels), np.asarray([0, 1]))
    assert np.array_equal(np.asarray(indices), expected_indices)
    assert np.allclose(np.asarray(values), expected_values, rtol=2e-6, atol=2e-6)


def test_exact_zero_ties_keep_lower_eligible_flat_indices():
    wpt = _make_wpt((32,), "rectangular", levels=2)
    eligible = _eligible_indices(wpt, positive_half=False)

    indices, values = streamed_top_n_coefficients(
        jnp.zeros((32,)),
        wpt,
        top_n=20,
        positive_half=False,
        boxes_per_chunk=3,
    )

    assert np.array_equal(np.asarray(indices), eligible[:20])
    assert np.array_equal(np.asarray(values), np.zeros(20, dtype=np.complex64))


def test_top_n_is_clamped_to_positive_half_capacity():
    wpt = _make_wpt((16,), "rectangular_mirror")
    eligible = _eligible_indices(wpt, positive_half=True)

    indices, values = streamed_top_n_coefficients(
        jnp.zeros((16,)),
        wpt,
        top_n=wpt.total_coeffs + 10,
        positive_half=True,
        boxes_per_chunk=7,
    )

    assert indices.shape == values.shape == (eligible.size,)
    assert np.array_equal(np.asarray(indices), eligible)


def test_streamed_selected_values_have_the_materialized_jvp_away_from_ties():
    wpt = _make_wpt((16,), "rectangular")
    data = jax.random.normal(jax.random.PRNGKey(41), (16,))
    tangent = jax.random.normal(jax.random.PRNGKey(42), (16,))

    indices, _ = streamed_top_n_coefficients(
        data,
        wpt,
        top_n=7,
        positive_half=False,
        scale=0.5,
        boxes_per_chunk=3,
    )

    def streamed_values(field):
        return streamed_top_n_coefficients(
            field,
            wpt,
            top_n=7,
            positive_half=False,
            scale=0.5,
            boxes_per_chunk=3,
        )[1]

    def materialized_values(field):
        return (0.5 * wpt.forward(field, "spatial"))[indices]

    streamed_primal, streamed_tangent = jax.jvp(streamed_values, (data,), (tangent,))
    reference_primal, reference_tangent = jax.jvp(
        materialized_values, (data,), (tangent,)
    )

    assert jnp.allclose(streamed_primal, reference_primal, rtol=2e-6, atol=2e-6)
    assert jnp.allclose(streamed_tangent, reference_tangent, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("top_n", [0, True, 1.5])
def test_invalid_top_n_is_rejected(top_n):
    wpt = _make_wpt((16,), "rectangular")
    with pytest.raises(ValueError, match="top_n must be a positive integer"):
        streamed_top_n_coefficients(jnp.ones((16,)), wpt, top_n=top_n)


@pytest.mark.parametrize("boxes_per_chunk", [0, True, 1.5])
def test_invalid_chunk_size_is_rejected(boxes_per_chunk):
    wpt = _make_wpt((16,), "rectangular")
    with pytest.raises(ValueError, match="boxes_per_chunk must be a positive integer"):
        streamed_top_n_coefficients(
            jnp.ones((16,)),
            wpt,
            top_n=1,
            boxes_per_chunk=boxes_per_chunk,
        )


def test_other_invalid_arguments_are_rejected():
    wpt = _make_wpt((16,), "rectangular")

    with pytest.raises(ValueError, match="input_type"):
        streamed_top_n_coefficients(
            jnp.ones((16,)), wpt, top_n=1, input_type="frequency"
        )
    with pytest.raises(ValueError, match="positive_half"):
        streamed_top_n_coefficients(
            jnp.ones((16,)), wpt, top_n=1, positive_half=cast(Any, 1)
        )
    with pytest.raises(ValueError, match="data shape"):
        streamed_top_n_coefficients(jnp.ones((8,)), wpt, top_n=1)
    with pytest.raises(ValueError, match="scale must be a scalar"):
        streamed_top_n_coefficients(jnp.ones((16,)), wpt, top_n=1, scale=jnp.ones((2,)))

    synthesis_only = _make_wpt((16,), "none")
    with pytest.raises(ValueError, match="windowing='none'"):
        streamed_top_n_coefficients(jnp.ones((16,)), synthesis_only, top_n=1)
