"""
Array shape and resampling utilities.

Centered pad / crop, Fourier-based resampling, nearest-neighbour resampling,
and a wrap-around centered-box extractor. These helpers operate on raw arrays
and have no MSWPT/MSGB-specific dependencies.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
import numpy as np
from scipy.ndimage import zoom


__all__ = [
    "interpolate_nearest",
    "pad_array",
    "pad_zero",
    "pad_edge",
    "crop_centered",
    "interpolate_fourier",
    "extract_centered_box",
    "rel_l2",
]


def interpolate_nearest(array: jnp.ndarray, new_shape: Tuple) -> jnp.ndarray:
    """
    Nearest-neighbour resampling to `new_shape`.

    Parameters
    ----------
    array : jnp.ndarray
    new_shape : Tuple[int, ...]

    Returns
    -------
    jnp.ndarray
        Resampled array.
    """
    zoom_factors = tuple(
        [new_dim / old_dim for new_dim, old_dim in zip(new_shape, array.shape)]
    )
    return zoom(array, zoom_factors, order=0)


def pad_zero(array: jnp.ndarray, desired_size: Tuple) -> jnp.ndarray:
    """
    Zero-pad to centered target size.

    Parameters
    ----------
    array : jnp.ndarray
    desired_size : Tuple[int, ...]

    Returns
    -------
    jnp.ndarray
    """
    return pad_array(array, desired_size, mode="constant")


def pad_edge(array: jnp.ndarray, desired_size: Tuple) -> jnp.ndarray:
    """
    Edge-pad (replicate border) to centered target size.

    Parameters
    ----------
    array : jnp.ndarray
    desired_size : Tuple[int, ...]

    Returns
    -------
    jnp.ndarray
    """
    return pad_array(array, desired_size, mode="edge")


def _center_slices(curr: Tuple[int, ...], target: Tuple[int, ...]):
    """
    Compute centered slices for cropping one shape to another.

    Parameters
    ----------
    curr : Tuple[int, ...]
        Current shape.
    target : Tuple[int, ...]
        Target crop shape.

    Returns
    -------
    Tuple[slice, ...]
        Per-axis centered slices.

    Raises
    ------
    ValueError
        If any target dimension is larger than the current dimension.
    """
    sl = []
    for c, t in zip(curr, target):
        if t > c:
            raise ValueError("target larger than current in _center_slices")
        start = (c - t) // 2
        sl.append(slice(start, start + t))
    return tuple(sl)


def pad_array(
    array: jnp.ndarray, desired_size: Tuple[int, ...], mode: str = "constant"
) -> jnp.ndarray:
    """
    Centered pad per axis to reach `desired_size`.

    Parameters
    ----------
    array : jnp.ndarray
    desired_size : Tuple[int, ...]
    mode : {"constant", "edge"}

    Returns
    -------
    jnp.ndarray
    """
    curr = array.shape
    pads = []
    for c, d in zip(curr, desired_size):
        if d <= c:
            pads.append((0, 0))
        else:
            total = d - c
            left = total // 2
            right = total - left
            pads.append((left, right))
    if mode == "constant":
        return jnp.pad(array, pads, mode="constant")
    elif mode == "edge":
        return jnp.pad(array, pads, mode="edge")
    else:
        raise ValueError(f"Unsupported pad mode: {mode}")


def crop_centered(array: jnp.ndarray, desired_size: Tuple[int, ...]) -> jnp.ndarray:
    """
    Centered crop per axis. No-op if any target dim exceeds current.

    Parameters
    ----------
    array : jnp.ndarray
    desired_size : Tuple[int, ...]

    Returns
    -------
    jnp.ndarray
    """
    curr = array.shape
    if any(d > c for d, c in zip(desired_size, curr)):
        return array  # caller should pad first; keep function pure
    return array[_center_slices(curr, desired_size)]


def interpolate_fourier(
    array: jnp.ndarray,
    desired_size: Tuple[int, ...],
    input_type: str,
    output_type: str,
) -> jnp.ndarray:
    """
    Unitary FFT-based resampling (pad when upsampling, crop when downsampling).

    Parameters
    ----------
    array : jnp.ndarray
    desired_size : Tuple[int, ...]
    input_type : {"spatial", "fourier"}
    output_type : {"spatial", "fourier"}

    Returns
    -------
    jnp.ndarray
        Resampled array in `output_type` domain.

    Notes
    -----
    Assumes periodic boundaries. Uses `pad_array` then `crop_centered` in frequency.
    """
    if input_type not in {"spatial", "fourier"} or output_type not in {
        "spatial",
        "fourier",
    }:
        raise ValueError(
            f"input_type/output_type must be 'spatial' or 'fourier'; "
            f"got {input_type}->{output_type}"
        )

    x = array
    if input_type == "spatial":
        x = jnp.fft.fftshift(jnp.fft.fftn(x, norm="ortho"))

    curr = x.shape
    larger = tuple(max(d, c) for d, c in zip(desired_size, curr))
    x = pad_array(x, larger, mode="constant")
    x = crop_centered(x, desired_size)

    if output_type == "spatial":
        x = jnp.fft.ifftn(jnp.fft.ifftshift(x), norm="ortho")
    return x


def extract_centered_box(arr, box_shape_tuple, center):
    """
    Wrap-around extraction of a centered N-D box (JIT/static-friendly).

    Parameters
    ----------
    arr : jnp.ndarray, shape S = (N1, ..., Nd)
    box_shape_tuple : Tuple[int, ...]
        Static Python tuple of ints (box sizes per axis).
    center : jnp.ndarray, shape (d,)
        Center index in the same index space as `arr` (0..Ni-1).

    Returns
    -------
    jnp.ndarray, shape `box_shape_tuple`
    """
    assert isinstance(box_shape_tuple, tuple) and all(
        isinstance(m, (int, np.integer)) for m in box_shape_tuple
    ), "box_shape_tuple must be a tuple of ints"

    S = arr.shape
    d = arr.ndim
    c = jnp.asarray(center)
    half = jnp.asarray([m // 2 for m in box_shape_tuple])
    starts = (c - half) % jnp.asarray(S)

    out = arr
    for axis in range(d):
        size_i = int(box_shape_tuple[axis])
        idx_i = (starts[axis] + jnp.arange(size_i)) % S[axis]
        out = jnp.take(out, idx_i, axis=axis, mode="wrap")
    return out


def rel_l2(a, b):
    """
    Compute relative L2 error between two arrays.

    Parameters
    ----------
    a : array-like
        Reference array.
    b : array-like
        Comparison array.

    Returns
    -------
    float
        ``||a - b||_2 / (||a||_2 + 1e-30)``.
    """
    num = jnp.linalg.norm(a - b)
    den = jnp.linalg.norm(a) + 1e-30
    return float(num / den)
