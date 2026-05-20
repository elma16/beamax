"""
Coefficient indexing utilities used across the MSWPT transform and solvers.

These helpers translate between flat coefficient indices and (level,
multi-index) coordinates, and batch leading-axis-batched beam parameter
arrays for `scan`/`vmap` aggregation.
"""

from __future__ import annotations

from functools import partial
from typing import Tuple

import jax.numpy as jnp
from jax import jit, vmap

from beamax.decomposition import DyadicDecomposition


__all__ = [
    "batch_data",
    "find_level",
    "find_tensor_and_multiindex",
    "compute_coeff_shapes",
]


def batch_data(*args, batch_size, zero_padded_args=()):
    """
    Batch leading beam axis into `(num_batches, batch_size, ...)`.

    Parameters
    ----------
    *args : Tuple[jnp.ndarray, ...]
        Arrays with leading dimension `b`.
    batch_size : int
    zero_padded_args : Tuple[int, ...]
        Indices into `args` that should be zero-padded in the last batch;
        others repeat their last entry.

    Returns
    -------
    Tuple[jnp.ndarray, ...]
        Batched arrays (with padding if needed).
    """
    b = args[0].shape[0]
    num_batches = (b + batch_size - 1) // batch_size

    def _pad(arr, zero_pad=True):
        pad_size = num_batches * batch_size - arr.shape[0]
        if pad_size == 0:
            return arr
        if zero_pad:
            padding = jnp.zeros((pad_size,) + arr.shape[1:], dtype=arr.dtype)
        else:
            last_entry = arr[-1:]
            padding = jnp.broadcast_to(last_entry, (pad_size,) + arr.shape[1:])
        return jnp.concatenate([arr, padding], axis=0)

    batched_args = [
        jnp.reshape(
            _pad(arg, zero_pad=(i in zero_padded_args)),
            (num_batches, batch_size) + arg.shape[1:],
        )
        for i, arg in enumerate(args)
    ]
    return tuple(batched_args)


def find_level(dyadic_decomp: DyadicDecomposition, box_num: int) -> int:
    """
    Map global box index → dyadic level.

    Parameters
    ----------
    dyadic_decomp : DyadicDecomposition
    box_num : int

    Returns
    -------
    int
        Level `ℓ` such that cumulative boxes up to `ℓ` exceed `box_num`.
    """
    return jnp.searchsorted(dyadic_decomp.num_boxes_ndim_cumsum, box_num, side="right")


def find_tensor_and_multiindex(
    flat_indices: jnp.ndarray, shapes: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Decode flat indices over concatenated tensors.

    Parameters
    ----------
    flat_indices : jnp.ndarray, shape (m,)
    shapes : jnp.ndarray, shape (L, k)
        Shapes of each tensor (per level).

    Returns
    -------
    array_indices : jnp.ndarray, shape (m,)
        Which tensor each flat index belongs to.
    multidimensional_indices : jnp.ndarray, shape (m, k)
        Unravelled indices within that tensor.
    """
    shapes_prod = jnp.prod(shapes, axis=1)
    cumsum_prods = jnp.cumsum(shapes_prod)
    array_indices = jnp.searchsorted(cumsum_prods, flat_indices, side="right")
    local_indices = jnp.where(
        array_indices > 0,
        flat_indices - jnp.take(cumsum_prods, array_indices - 1),
        flat_indices,
    )

    def unravel_index(local_index, shape):
        return jnp.unravel_index(local_index, shape)

    unravel_index_jit = jit(unravel_index)
    multidimensional_indices = jnp.array(
        vmap(unravel_index_jit)(local_indices, shapes[array_indices])
    )

    return array_indices, multidimensional_indices


@partial(vmap, in_axes=(None, None, 0))
def compute_coeff_shapes(
    dyadic_decomp: DyadicDecomposition, redundancy: int, level: int
) -> jnp.ndarray:
    """
    Per-level coefficient tensor shapes.

    Parameters
    ----------
    dyadic_decomp : DyadicDecomposition
    redundancy : int
        1 (basis) or 2 (frame).
    level : int
        Vectorized: function is `vmap`ped over `level`.

    Returns
    -------
    jnp.ndarray, shape (k,)
        `(num_boxes_level, *support_shape)`.
    """
    shape = (dyadic_decomp.num_boxes_ndim[level],) + tuple(
        jnp.array(dyadic_decomp.box_aspect_ratio)
        * jnp.array(
            dyadic_decomp.ndim * (redundancy * dyadic_decomp.box_lengths[level],)
        )
    )
    return jnp.array(shape, dtype=int)
