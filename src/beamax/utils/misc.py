from __future__ import annotations

import jax
from jax import jit, vmap
import jax.numpy as jnp
from functools import partial
from typing import Tuple

from beamax.decomposition import DyadicDecomposition
from scipy.ndimage import zoom


import numpy as np


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

    def pad_array(arr, zero_pad=True):
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
            pad_array(arg, zero_pad=(i in zero_padded_args)),
            (num_batches, batch_size) + arg.shape[1:],
        )
        for i, arg in enumerate(args)
    ]
    return tuple(batched_args)


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


def find_closest_center_indices(centers, index, k):
    """
    k nearest centres to a reference centre (Euclidean on indices).

    Parameters
    ----------
    centers : jnp.ndarray, shape (B, d)
    index : int
    k : int

    Returns
    -------
    jnp.ndarray, shape (k,)
        Indices of nearest neighbours (including the reference).
    """
    # Get the reference center
    reference = centers[index]

    # Calculate distances from the reference center to all centers
    distances = jnp.sum((centers - reference) ** 2, axis=1)

    # Get the indices of the k closest centers
    closest_indices = jnp.argsort(distances)[:k]

    return closest_indices


def _rand_rot(key, d, K):
    """
    Generate random rotation matrices in d dimensions.

    Parameters
    ----------
    key : jax.random.PRNGKey
                JAX random key for generating random numbers.
    d : int
                Dimension of the rotation (1, 2, or 3).
        K : int
                Number of rotation matrices to generate.

        Returns
        -------
        jnp.ndarray
                A tensor of shape (K, d, d) containing K random rotation matrices.

    """
    if d == 1:
        return jnp.ones((K, 1, 1))
    if d == 2:
        th = jax.random.uniform(key, (K,)) * jnp.pi
        c, s = jnp.cos(th), jnp.sin(th)
        R = jnp.stack([jnp.stack([c, -s], -1), jnp.stack([s, c], -1)], -2)
        return R
    A = jax.random.normal(key, (K, 3, 3))
    Q, _ = jnp.linalg.qr(A)
    det = jnp.linalg.det(Q)
    s = jnp.where(det < 0, -1.0, 1.0)[:, None]
    Q = Q.at[:, :, 0].set(Q[:, :, 0] * s)
    return Q


# @partial(jax.jit, static_argnames=("N", "n_ellipses", "profile"))
def ellipsoid_superposition(
    key,
    N,
    n_ellipses=8,
    amp_range=(0.5, 1.0),
    size_range=(0.1, 0.35),
    center_max=0.8,
    profile="gaussian",
    nonnegative=True,
    dtype=jnp.float32,
):
    """
    Synthetic field: sum of random ellipsoids (Gaussian or indicator).

    Parameters
    ----------
    key : jax.random.PRNGKey
    N : Tuple[int, ...]
        Output grid shape.
    n_ellipses : int
    amp_range : Tuple[float, float]
    size_range : Tuple[float, float]
    center_max : float
        Centres sampled in [-center_max, center_max]^d (normalized coords).
    profile : {"gaussian", "indicator"}
    nonnegative : bool
    dtype : jnp.dtype

    Returns
    -------
    f : jnp.ndarray, shape N
    pars : dict
        Dict with 'amps', 'centers', 'sizes'.
    """
    N = tuple(N)
    d = len(N)
    K = n_ellipses
    axes = [jnp.linspace(-1, 1, n, dtype=dtype) for n in N]
    coords = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), 0)  # (d,*N)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    amps = jax.random.uniform(
        k1, (K,), minval=amp_range[0], maxval=amp_range[1]
    ).astype(dtype)
    centers = jax.random.uniform(
        k2, (K, d), minval=-center_max, maxval=center_max
    ).astype(dtype)
    sizes = jax.random.uniform(
        k3, (K, d), minval=size_range[0], maxval=size_range[1]
    ).astype(dtype)
    R = _rand_rot(k4, d, K).astype(dtype)
    ones = (1,) * d
    x = coords[None, ...]
    c = centers.reshape(K, d, *ones)
    y = jnp.einsum("kij,kj...->ki...", jnp.swapaxes(R, 1, 2), x - c)
    a = sizes.reshape(K, d, *ones)
    quad = jnp.sum((y / a) ** 2, 1)
    if profile == "gaussian":
        contrib = amps.reshape(K, *ones) * jnp.exp(-0.5 * quad)
    elif profile == "indicator":
        contrib = amps.reshape(K, *ones) * (quad <= 1).astype(dtype)
    else:
        raise ValueError("profile must be gaussian or indicator")
    f = jnp.sum(contrib, 0)
    if nonnegative:
        f = jnp.clip(f, 0)
    return f.astype(dtype), {"amps": amps, "centers": centers, "sizes": sizes}


def _center_slices(curr: Tuple[int, ...], target: Tuple[int, ...]):
    """Compute centered slice for cropping `curr -> target` per axis."""
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
            f"input_type/output_type must be 'spatial' or 'fourier'; got {input_type}->{output_type}"
        )

    x = array
    if input_type == "spatial":
        x = jnp.fft.fftshift(jnp.fft.fftn(x, norm="ortho"))

    curr = x.shape
    # Compute target in one pass by padding then cropping as needed
    # First pad where target > curr
    larger = tuple(max(d, c) for d, c in zip(desired_size, curr))
    x = pad_array(x, larger, mode="constant")
    # Then crop to desired
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
    # All sizes we take from Python tuple to keep them STATIC for JIT.
    assert isinstance(box_shape_tuple, tuple) and all(
        isinstance(m, (int, np.integer)) for m in box_shape_tuple
    ), "box_shape_tuple must be a tuple of ints"

    S = arr.shape  # Python tuple of ints (static)
    d = arr.ndim
    c = jnp.asarray(center)  # tracer OK
    half = jnp.asarray(
        [m // 2 for m in box_shape_tuple]
    )  # small vector, static lengths

    # Compute wrapped start indices per axis as tracers (fine)
    starts = (c - half) % jnp.asarray(S)

    out = arr
    for axis in range(d):
        size_i = int(box_shape_tuple[axis])  # STATIC length for arange
        idx_i = (starts[axis] + jnp.arange(size_i)) % S[axis]
        out = jnp.take(out, idx_i, axis=axis, mode="wrap")
    return out


# ---------- Level-aware Top-K selection + optional K search ----------


def _per_level_meta(dyadic, wpt):
    """Return shapes, level sizes, and flat offsets for coeff layout."""
    shapes = compute_coeff_shapes(
        dyadic, wpt.redundancy, jnp.arange(dyadic.num_levels)
    )  # (L, d+1)
    sizes = jnp.prod(shapes, axis=1)  # (L,)
    offs = jnp.r_[0, jnp.cumsum(sizes)]  # (L+1,)
    return shapes, sizes, offs


def _split_by_level(coeffs, sizes, offs):
    """Yield (lvl, flat_slice_view_of_level)."""
    views = []
    for lvl in range(len(sizes)):
        sl = slice(int(offs[lvl]), int(offs[lvl + 1]))
        views.append((lvl, coeffs[sl]))
    return views


def _allocate_K_by_energy(K, coeffs, sizes, offs):
    """K_l ∝ energy_l with rounding + correction so sum K_l == K and K_l ≤ size_l."""
    per_level = _split_by_level(coeffs, sizes, offs)
    e = jnp.array([jnp.sum(jnp.abs(v) ** 2) for _, v in per_level])
    e_sum = float(e.sum())
    if e_sum == 0:
        raw = jnp.zeros_like(e)
    else:
        raw = e / e_sum * float(K)
    k_floor = jnp.minimum(jnp.floor(raw), sizes).astype(int)
    k = int(k_floor.sum())
    # distribute leftovers by largest fractional parts, but do not exceed sizes
    need = int(K) - k
    if need > 0:
        frac = raw - jnp.floor(raw)
        order = jnp.argsort(-frac)  # descending
        k_list = list(map(int, k_floor))
        for idx in map(int, order):
            if need == 0:
                break
            if k_list[idx] < int(sizes[idx]):
                k_list[idx] += 1
                need -= 1
        k_floor = jnp.array(k_list, dtype=int)
    return k_floor  # (L,)


def select_levelaware_topK_indices(coeffs, dyadic, wpt, K: int):
    """Return (indices, values) of selected coefs (level-aware)."""
    shapes, sizes, offs = _per_level_meta(dyadic, wpt)
    K = int(max(0, min(K, int(offs[-1]))))
    if K == 0:
        return jnp.array([], dtype=jnp.int32), jnp.array([], dtype=coeffs.dtype)

    K_per_level = _allocate_K_by_energy(K, coeffs, sizes, offs)
    idx_list = []
    val_list = []
    # small L; Python loop is fine and JIT-safety not needed here
    for lvl in range(len(sizes)):
        k = int(K_per_level[lvl])
        if k == 0:
            continue
        start = int(offs[lvl])
        end = int(offs[lvl + 1])
        seg = coeffs[start:end]
        # top-k by magnitude
        absseg = jnp.abs(seg)
        # jnp.argpartition is O(n), but we want sorted top-k—use argpartition then sort that subset
        if k < seg.shape[0]:
            part = jnp.argpartition(absseg, seg.shape[0] - k)[-k:]
        else:
            part = jnp.arange(seg.shape[0])
        # sort selected in descending magnitude
        order = part[jnp.argsort(-absseg[part])]
        idx_global = start + order
        idx_list.append(idx_global.astype(jnp.int32))
        val_list.append(seg[order])
    if not idx_list:
        return jnp.array([], dtype=jnp.int32), jnp.array([], dtype=coeffs.dtype)
    indices = jnp.concatenate(idx_list)
    values = jnp.concatenate(val_list)
    return indices, values


def reconstruct_from_selection(coeffs, indices, values, inv_wpt, output_type="spatial"):
    """Make a sparse coeff vector and invert."""
    thr = jnp.zeros_like(coeffs)
    thr = thr.at[indices].set(values)
    return inv_wpt.inverse(thr, output_type=output_type)


def rel_l2(a, b):
    num = jnp.linalg.norm(a - b)
    den = jnp.linalg.norm(a) + 1e-30
    return float(num / den)


def find_min_K_for_target_error(
    coeffs,
    p0,
    inv_wpt,
    tau=0.01,
    Kmin=128,
    Kmax=None,
    verbose=True,
    force_K=None,
):
    """
    Binary search for the minimal K coefficients to hit a target rel-L2 error.

    Parameters
    ----------
    coeffs : jnp.ndarray
        Flattened coefficient vector.
    p0 : jnp.ndarray
        Ground-truth image/volume.
    inv_wpt : MSWPT
        Transform object supporting ``inverse``.
    tau : float
        Target relative L2 error (fraction).
    Kmin : int
        Minimum K to consider.
    Kmax : int | None
        Optional cap on K; defaults to len(coeffs).
    verbose : bool
        Print search progress.
    force_K : int | None
        If provided, skip the binary search entirely and return ``(force_K,
        top-K indices, top-K values)`` directly. Useful when ``K`` is
        structurally known (e.g. 3D inverse at ``tau=1%`` lands on the full
        dictionary), which lets a single CPU-side call replace ~log₂(total)
        expensive inverse-WPT evaluations on the GPU.

    Returns
    -------
    (best_K, indices, values)
        best_K : int
            Minimal K meeting the error target.
        indices : jnp.ndarray
            Indices of selected coefficients (sorted by magnitude desc).
        values : jnp.ndarray
            Corresponding coefficient values.

    Notes
    -----
    Even without ``force_K``, the function tests ``K=Kmax`` first. If that
    fails, no smaller K can succeed (the error monotonically decreases in K
    for top-K selection), so we return ``best_K=Kmax`` immediately.  This
    short-circuits the common "full dictionary needed" case from
    ``log₂(total)`` inverse-WPT calls down to one.
    """
    total = int(coeffs.shape[0])
    if Kmax is None:
        Kmax = total
    Kmin = max(1, min(int(Kmin), int(Kmax)))

    abs_coeffs = jnp.abs(coeffs)

    def select_topK_simple(K):
        """Top-K by magnitude, descending."""
        K = min(int(K), total)
        if K < total:
            part = jnp.argpartition(abs_coeffs, total - K)[-K:]
        else:
            part = jnp.arange(total)
        order = part[jnp.argsort(-abs_coeffs[part])]
        return order.astype(jnp.int32), coeffs[order]

    def test_K(K):
        idx, vals = select_topK_simple(K)
        recon = reconstruct_from_selection(
            coeffs, idx, vals, inv_wpt, output_type="spatial"
        )
        err = jnp.linalg.norm(p0 - recon.real) / (jnp.linalg.norm(p0) + 1e-30)
        return float(err)

    # Force-K fast path: skip the search entirely.
    if force_K is not None:
        K = max(1, min(int(force_K), int(Kmax)))
        if verbose:
            print(f"\nfind_min_K_for_target_error: force_K={K}, skipping search.")
        idx, vals = select_topK_simple(K)
        return K, idx, vals

    if verbose:
        print(f"\nSearching for minimum K to achieve {tau * 100:.1f}% error...")
        print(
            f"{'K':<10} {'Actual K':<10} {'Error %':<10} {'Target %':<10} {'Status':<10}"
        )
        print("-" * 60)

    # Short-circuit: test K=Kmax first.  Top-K error decreases monotonically
    # in K, so if K=Kmax already fails the target there is no point in any
    # binary-search step (every smaller K fails too).  Conversely the very
    # common "full dictionary needed" outcome lands here in one test instead
    # of ~log₂(total) tests.
    err_max = test_K(Kmax)
    if verbose:
        status = "✓ Pass" if err_max <= tau else "✗ Fail"
        print(
            f"{Kmax:<10} {Kmax:<10} {err_max * 100:<10.2f} {tau * 100:<10.2f} {status:<10}"
        )
    if err_max > tau:
        if verbose:
            print("-" * 60)
            print(
                f"✗ Even K=Kmax={Kmax} fails target tau={tau * 100:.1f}%. "
                "Returning K=Kmax."
            )
        idx, vals = select_topK_simple(Kmax)
        return int(Kmax), idx, vals

    left, right = Kmin, int(Kmax) - 1
    best_K = int(Kmax)
    while left <= right:
        mid = (left + right) // 2
        err = test_K(mid)
        if verbose:
            status = "✓ Pass" if err <= tau else "✗ Fail"
            print(
                f"{mid:<10} {mid:<10} {err * 100:<10.2f} {tau * 100:<10.2f} {status:<10}"
            )
        if err <= tau:
            best_K = mid
            right = mid - 1
        else:
            left = mid + 1

    idx, vals = select_topK_simple(best_K)
    if verbose:
        final_error = test_K(best_K)
        print("-" * 60)
        print(f"✓ Found minimum K = {best_K} (error = {final_error * 100:.2f}%)")

    return best_K, idx, vals


def choose_K_by_tau(
    coeffs,
    p0,
    inv_wpt,
    dyadic,
    wpt,
    tau=0.02,
    Kmin=256,
    Kmax=None,
    num_steps=8,
    beam_budget=None,
):
    """
    Find minimal K s.t. rel-L2 <= tau using a geometric sweep.

    coeffs: flattened coefficients (1D).
    p0: original image (array).
    inv_wpt: MSWPT instance with windowing="none" for clean inverse.
    dyadic, wpt: decomposition objects.
    tau: target relative L2 error.
    beam_budget: optional cap on beams; real path ≈ 2 beams / coef → K ≤ floor(beam_budget/2).
    """
    total = int(coeffs.shape[0])
    if Kmax is None:
        Kmax = total
    if beam_budget is not None:
        Kmax = min(Kmax, int(beam_budget) // 2)
    Kmin = max(1, min(Kmin, Kmax))

    # make a geometric ladder between Kmin and Kmax
    ratios = np.linspace(0, 1, num=num_steps)
    Ks = (Kmin * (Kmax / Kmin) ** ratios).astype(int)
    Ks = np.unique(np.clip(Ks, 1, Kmax))

    best = Kmax
    for K in Ks:
        idx, vals = select_levelaware_topK_indices(coeffs, dyadic, wpt, int(K))
        recon = reconstruct_from_selection(
            coeffs, idx, vals, inv_wpt, output_type="spatial"
        )
        err = rel_l2(p0, recon.real)
        # print(f"K={K}, rel-L2={err:.3%}")
        if err <= tau:
            best = int(K)
            break
    return best


def add_centered_box_periodic(
    dest: jnp.ndarray, patch: jnp.ndarray, center_ndim: jnp.ndarray
) -> jnp.ndarray:
    """
    Periodic scatter-add of a small 'patch' into full-size 'dest',
    centered at integer 'center_ndim' given in Fourier index coords
    (i.e. in [-N/2, N/2-1] like dyadic_decomp.centres_ndim).
    """
    N = jnp.array(dest.shape)
    S = jnp.array(patch.shape)
    d = patch.ndim

    # move center to [0, N)
    c0 = (center_ndim + N // 2) % N
    # starts = center - half-support in [0, N)
    starts = (c0 - S // 2) % N

    # build per-axis modular indices for the support window
    idx_axes = [
        (starts[i] + jnp.arange(S[i], dtype=jnp.int32)) % N[i] for i in range(d)
    ]

    # meshgrid and scatter-add
    grids = jnp.meshgrid(*idx_axes, indexing="ij")
    return dest.at[tuple(grids)].add(patch)
