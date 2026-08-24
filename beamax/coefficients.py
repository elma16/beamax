"""Memory-bounded exact top-k selection for MSGB's MSWPT analysis."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from beamax.transforms import MSWPT, _analyse_box
from beamax.utils.fft import convert_space


Scalar = Union[int, float, complex, jax.Array, np.number]


def _eligible_capacity(wpt: MSWPT, *, positive_half: bool) -> int:
    """Return the static number of coefficients visited by the selector."""
    if not positive_half:
        return int(wpt.total_coeffs)
    return sum(
        (end - start) // 2
        for start, end in zip(wpt.coeffs_cumsum[:-1], wpt.coeffs_cumsum[1:])
    )


def _merge_top_k(
    retained_magnitudes: jax.Array,
    retained_indices: jax.Array,
    retained_values: jax.Array,
    candidate_magnitudes: jax.Array,
    candidate_indices: jax.Array,
    candidate_values: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Merge an ascending-index coefficient chunk into the retained top-k.

    Index ordering makes ``lax.top_k`` ties favor the lower global index.
    """
    magnitudes = jnp.concatenate([retained_magnitudes, candidate_magnitudes])
    indices = jnp.concatenate([retained_indices, candidate_indices])
    values = jnp.concatenate([retained_values, candidate_values])

    retained_count = retained_magnitudes.shape[0]
    top_magnitudes, positions = jax.lax.top_k(magnitudes, retained_count)
    top_indices = indices[positions]
    top_values = values[positions]

    index_order = jnp.argsort(top_indices, stable=True)
    return (
        top_magnitudes[index_order],
        top_indices[index_order],
        top_values[index_order],
    )


@eqx.filter_jit
def _streamed_top_n_coefficients_jit(
    data: jax.Array,
    wpt: MSWPT,
    *,
    selected_count: int,
    input_type: str,
    positive_half: bool,
    scale: Scalar,
    boxes_per_chunk: int,
) -> tuple[jax.Array, jax.Array]:
    ft_data = convert_space(data, input_type, "fourier")
    ft_sum_gsquare = ft_data / wpt.sum_gsquare
    scaled = jnp.asarray(scale)

    real_dtype = jnp.abs(jnp.zeros((), dtype=wpt.complex_dtype)).dtype
    retained_magnitudes = jnp.full((selected_count,), -jnp.inf, dtype=real_dtype)
    sentinel_index = int(wpt.total_coeffs)
    retained_indices = jnp.full((selected_count,), sentinel_index, dtype=jnp.int32)
    retained_values = jnp.zeros((selected_count,), dtype=wpt.complex_dtype)

    grid_shape = jnp.asarray(wpt.dyadic_decomp.N, dtype=jnp.int32)
    centres = wpt.dyadic_decomp.centres_ndim + grid_shape // 2
    for level in range(wpt.dyadic_decomp.num_levels):
        box_start = int(wpt.boxes_cumsum[level])
        box_end = int(wpt.boxes_cumsum[level + 1])
        box_count = box_end - box_start
        coefficient_start = int(wpt.coeffs_cumsum[level])
        coefficient_end = int(wpt.coeffs_cumsum[level + 1])
        coefficient_count = coefficient_end - coefficient_start

        support_shape = wpt._support_shapes[level]
        coefficients_per_box = math.prod(support_shape)
        eligible_coefficient_count = (
            coefficient_count // 2 if positive_half else coefficient_count
        )
        visited_box_count = math.ceil(eligible_coefficient_count / coefficients_per_box)
        half_support = jnp.asarray(wpt._box_shapes[level], dtype=jnp.int32)
        packed_filter = wpt.gfilts_packed[level]
        centres_level = centres[box_start:box_end]
        chunks = math.ceil(visited_box_count / boxes_per_chunk)
        coefficient_offsets = jnp.arange(coefficients_per_box, dtype=jnp.int32)

        def analyse_box(centre: jax.Array) -> jax.Array:
            return _analyse_box(
                ft_sum_gsquare,
                packed_filter,
                support_shape,
                centre,
                grid_shape,
                half_support,
            )

        def process_chunk(
            chunk_number: jax.Array,
            retained: tuple[jax.Array, jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            box_offsets = chunk_number * boxes_per_chunk + jnp.arange(
                boxes_per_chunk, dtype=jnp.int32
            )
            valid_boxes = box_offsets < visited_box_count
            safe_box_offsets = jnp.minimum(box_offsets, box_count - 1)
            chunk_centres = centres_level[safe_box_offsets]

            coefficients = jax.vmap(analyse_box)(chunk_centres)
            coefficients = coefficients.reshape((boxes_per_chunk, coefficients_per_box))
            coefficients = (scaled * coefficients).reshape(-1)

            level_offsets = (
                box_offsets[:, None] * coefficients_per_box
                + coefficient_offsets[None, :]
            )
            valid = jnp.broadcast_to(
                valid_boxes[:, None], (boxes_per_chunk, coefficients_per_box)
            )
            if positive_half:
                # Some layouts split the half-frame boundary within a box.
                valid = valid & (level_offsets < eligible_coefficient_count)

            flat_indices = coefficient_start + level_offsets
            flat_indices = flat_indices.reshape(-1)
            valid = valid.reshape(-1)
            values = jnp.where(valid, coefficients, 0)
            magnitudes = jnp.where(valid, jnp.abs(coefficients), -jnp.inf)
            indices = jnp.where(valid, flat_indices, sentinel_index).astype(jnp.int32)

            return _merge_top_k(
                *retained,
                magnitudes,
                indices,
                values,
            )

        retained_magnitudes, retained_indices, retained_values = jax.lax.fori_loop(
            0,
            chunks,
            process_chunk,
            (retained_magnitudes, retained_indices, retained_values),
        )

    # Public top-n order is ascending magnitude, then global index.
    index_order = jnp.argsort(retained_indices, stable=True)
    index_sorted_magnitudes = retained_magnitudes[index_order]
    magnitude_order = jnp.argsort(index_sorted_magnitudes, stable=True)
    final_order = index_order[magnitude_order]
    return retained_indices[final_order], retained_values[final_order]


def streamed_top_n_coefficients(
    data: jax.Array,
    wpt: MSWPT,
    *,
    top_n: int,
    input_type: str = "spatial",
    positive_half: bool = False,
    scale: Scalar = 1.0,
    boxes_per_chunk: int = 8,
) -> tuple[jax.Array, jax.Array]:
    r"""Select exact top-n scaled MSWPT coefficients without materializing them.

    Parameters
    ----------
    data : jax.Array, shape (*N,)
        Spatial data, or an already centred Fourier array when
        ``input_type="fourier"``.
    wpt : MSWPT
        Transform whose filters, coefficient layout, and canonical dual are
        used for the analysis.
    top_n : int
        Requested number of retained coefficients.  It is clamped to the
        selected frame capacity, matching the configured top-n threshold.
    input_type : {"spatial", "fourier"}
        Domain of ``data``.
    positive_half : bool
        Visit the first half of every level's coefficient segment rather than
        the complete signed frame.
    scale : scalar
        Scalar applied before ranking and returning coefficient values.
    boxes_per_chunk : int
        Static number of local Fourier boxes analysed together.

    Returns
    -------
    indices, values : tuple[jax.Array, jax.Array]
        Global flat coefficient indices and scaled complex values.  Rows are
        ordered by ascending ``(abs(value), index)``.  At the top-n cutoff,
        equal magnitudes retain the lower global flat index.

    Notes
    -----
    Selection state is $\mathcal{O}(\mathtt{top\_n})$ beyond the Fourier
    array and one box chunk.
    """
    if not isinstance(wpt, MSWPT):
        raise TypeError(f"wpt must be an MSWPT; got {type(wpt).__name__}.")
    if isinstance(top_n, (bool, np.bool_)) or not isinstance(top_n, Integral):
        raise ValueError("top_n must be a positive integer.")
    if int(top_n) <= 0:
        raise ValueError("top_n must be a positive integer.")
    if input_type not in {"spatial", "fourier"}:
        raise ValueError(
            f"input_type must be 'spatial' or 'fourier'; got {input_type!r}."
        )
    if not isinstance(positive_half, (bool, np.bool_)):
        raise ValueError("positive_half must be a boolean.")
    if isinstance(boxes_per_chunk, (bool, np.bool_)) or not isinstance(
        boxes_per_chunk, Integral
    ):
        raise ValueError("boxes_per_chunk must be a positive integer.")
    if int(boxes_per_chunk) <= 0:
        raise ValueError("boxes_per_chunk must be a positive integer.")
    if tuple(data.shape) != tuple(wpt.dyadic_decomp.N):
        raise ValueError(
            "data shape must match the MSWPT grid: "
            f"{tuple(data.shape)} != {wpt.dyadic_decomp.N}."
        )
    if wpt.windowing == "none":
        raise ValueError(
            "streamed MSWPT analysis does not support windowing='none'; "
            "use 'rectangular' or 'rectangular_mirror'."
        )
    if jnp.asarray(scale).ndim != 0:
        raise ValueError("scale must be a scalar.")

    capacity = _eligible_capacity(wpt, positive_half=bool(positive_half))
    selected_count = min(int(top_n), capacity)
    return _streamed_top_n_coefficients_jit(
        data,
        wpt,
        selected_count=selected_count,
        input_type=input_type,
        positive_half=bool(positive_half),
        scale=scale,
        boxes_per_chunk=int(boxes_per_chunk),
    )


__all__ = ["streamed_top_n_coefficients"]
