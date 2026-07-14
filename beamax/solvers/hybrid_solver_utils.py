import logging
import math
import jax
from jax import vmap
import jax.numpy as jnp
from jax.lax import fori_loop
from typing import Tuple, Optional
from scipy.ndimage import zoom
from einops import rearrange
import warnings

from beamax import utils
from beamax.decomposition import DyadicDecomposition
from beamax.transforms import single_filter_idx, MSWPT
from beamax.geometry import Domain

logger = logging.getLogger(__name__)


def gh_lowpass_filter(
    p0: jnp.ndarray,
    input_type: str,
    wpt: MSWPT,
    boxes_include: jnp.ndarray,
    windowing: str = "rectangular",
    gh: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Split into LF/HF via `g,h` frame filters in Fourier domain.

    Parameters
    ----------
    p0 : jnp.ndarray
    input_type : {"spatial","fourier"}
    wpt : MSWPT
    boxes_include : jnp.ndarray
        Indices of low-frequency boxes to include.
    windowing : str
    gh : jnp.ndarray, optional
        Precomputed LF-projection filter from :func:`compute_gh_filter`.
        Pass this when the same ``(wpt, boxes_include, windowing)`` is
        reused across many inputs to skip the (data-independent) filter
        computation.

    Returns
    -------
    (p0_HF_ft, p0_LF_ft) : Tuple[jnp.ndarray, jnp.ndarray]
        Fourier-domain HF and LF components.
    """
    # Work in Fourier domain
    p0_ft = utils.convert_space(p0, input_type, "fourier")

    if gh is None:
        gh = compute_gh_filter(wpt, boxes_include, windowing)

    # Low- and high-frequency parts in Fourier domain
    p0_LF_ft = p0_ft * gh
    p0_HF_ft = p0_ft - p0_LF_ft

    return p0_HF_ft, p0_LF_ft


def compute_gh_filter(
    wpt: MSWPT,
    boxes_include: jnp.ndarray,
    windowing: str = "rectangular",
) -> jnp.ndarray:
    """
    Compute the LF-projection filter ``gh = (Σ_{b∈LF} g_b^2) / Σ_b g_b^2``.

    This is the data-independent piece of :func:`gh_lowpass_filter` and is
    therefore cacheable across calls that share ``(wpt, boxes_include,
    windowing)``.

    Implementation
    --------------
    Uses ``lax.fori_loop`` so that only one ``(*N,)``-shape filter is
    materialised at a time, mirroring the pattern used by
    :meth:`MSWPT.sum_gsquare`. Avoids both the per-iter eager-trace overhead
    of a Python ``for`` loop and the ``(num_boxes, *N)`` peak memory of a
    pure ``vmap``.

    Parameters
    ----------
    wpt : MSWPT
    boxes_include : jnp.ndarray of int32
        LF box indices.
    windowing : str

    Returns
    -------
    jnp.ndarray, shape (*N,), real dtype
    """
    boxes_include = jnp.asarray(boxes_include, dtype=jnp.int32)
    sum_gsq = wpt.sum_gsquare

    def body(i, gh):
        """
        Add one selected filter contribution to the low-pass accumulator.

        Parameters
        ----------
        i : int
            Index into ``boxes_include``.
        gh : jnp.ndarray, shape (*N,)
            Current accumulated low-pass filter.

        Returns
        -------
        jnp.ndarray, shape (*N,)
            Updated accumulated filter.
        """
        idx = boxes_include[i]
        g_b = single_filter_idx(
            idx,
            wpt.dyadic_decomp.fourier_meshgrid,
            wpt.dyadic_decomp,
            wpt.redundancy,
            windowing,
        )
        return gh + (g_b * g_b) / sum_gsq

    gh0 = jnp.zeros_like(sum_gsq)
    return fori_loop(0, boxes_include.shape[0], body, gh0)


def get_indices_between_two_opposing_corners(
    centers: jnp.ndarray, corner1_idx: int, corner2_idx: int
) -> jnp.ndarray:
    """
    Get the indices in the box defined by the two corners.

    Parameters
    ----------
    centers : jnp.ndarray, shape (num_centers, ndim)
        Box centre coordinates.
    corner1_idx : int
        Index of the first corner.
    corner2_idx : int
        Index of the opposing corner.

    Returns
    -------
    jnp.ndarray
        Indices of centres inside the closed axis-aligned box.

    Raises
    ------
    ValueError
        If both corner indices refer to the same centre.
    """
    corner1 = centers[corner1_idx]
    corner2 = centers[corner2_idx]

    if jnp.all(corner1 == corner2):
        raise ValueError("corner1 and corner2 must differ to define a box.")

    mins = jnp.minimum(corner1, corner2)
    maxs = jnp.maximum(corner1, corner2)

    mask = jnp.all((centers >= mins) & (centers <= maxs), axis=1)

    indices_in_box = jnp.where(mask)[0]

    return indices_in_box


def get_indices_with_norm_less_than(
    centers: jnp.ndarray, norm: float, inclusive: bool = True
) -> jnp.ndarray:
    """
    Get the indices of the boxes with a norm less than (or equal to) the given value.

    Parameters
    ----------
    centers : jnp.ndarray, shape (num_centers, ndim)
        Box centre coordinates.
    norm : float
        L-infinity norm threshold.
    inclusive : bool, default=True
        If ``True``, use ``<=``; otherwise use ``<``.

    Returns
    -------
    jnp.ndarray
        Indices whose centre norm satisfies the threshold.
    """
    norms = jnp.linalg.norm(centers, axis=1, ord=jnp.inf)
    if inclusive:
        indices = jnp.where(norms <= norm)[0]
    else:
        indices = jnp.where(norms < norm)[0]
    return indices


def find_bounding_corner_indices(
    centers: jnp.ndarray, idx_box: jnp.ndarray
) -> Tuple[int, int]:
    """
    Find actual corner indices from a set of selected frequency indices.

    Given a set of selected center indices, finds two opposing corners
    that exist in the centers array (rather than computing component-wise
    min/max which may not correspond to actual centers).

    Parameters
    ----------
    centers : jnp.ndarray, shape (num_centers, ndim)
        All centre coordinates.
    idx_box : jnp.ndarray
        Indices of selected centres.

    Returns
    -------
    corner1_idx : int
        Index of one selected bounding corner.
    corner2_idx : int
        Index of the opposing selected bounding corner.

    Raises
    ------
    ValueError
        If ``idx_box`` is empty.
    """
    if idx_box.size == 0:
        raise ValueError("Cannot find corners from empty index set")

    if idx_box.size == 1:
        # Single point - return same index for both corners
        # (caller should handle this edge case)
        return int(idx_box[0]), int(idx_box[0])

    selected_centers = centers[idx_box]

    # Compute component-wise min and max of selected centers
    comp_min = jnp.min(selected_centers, axis=0)
    comp_max = jnp.max(selected_centers, axis=0)

    # Find the selected center closest to comp_min (L2 distance)
    dist_to_min = jnp.linalg.norm(selected_centers - comp_min, axis=1)
    corner1_local = jnp.argmin(dist_to_min)
    corner1_idx = idx_box[corner1_local]

    # Find the selected center closest to comp_max (L2 distance)
    dist_to_max = jnp.linalg.norm(selected_centers - comp_max, axis=1)
    corner2_local = jnp.argmin(dist_to_max)
    corner2_idx = idx_box[corner2_local]

    # If both corners ended up the same (e.g., 1D case), pick the furthest point
    if corner1_idx == corner2_idx:
        # Find the point furthest from corner1
        dist_from_corner1 = jnp.linalg.norm(
            selected_centers - centers[corner1_idx], axis=1
        )
        corner2_local = jnp.argmax(dist_from_corner1)
        corner2_idx = idx_box[corner2_local]

    return int(corner1_idx), int(corner2_idx)


def are_opposing(corner1: int, corner2: int) -> bool:
    """
    Check two corners are opposing.

    Parameters
    ----------
    corner1 : int
        First corner index.
    corner2 : int
        Second corner index.

    Returns
    -------
    bool
        Whether the corner indices differ.
    """
    return corner1 != corner2


def get_bounds(
    dyadic_decomp: DyadicDecomposition, domain: Domain, corner1: int, corner2: int
) -> jnp.ndarray:
    """
    Get the bounds of the filter banks required, using the opposite corners of the box.

    Parameters
    ----------
    dyadic_decomp : DyadicDecomposition
        Dyadic decomposition.
    domain : Domain
        Physical domain.
    corner1 : int
        First corner index.
    corner2 : int
        Opposing corner index.

    Returns
    -------
    jnp.ndarray, shape (ndim, 2)
        Inclusive/exclusive bounds in grid coordinates.
    """
    box_corners = jnp.array([corner1, corner2])

    def box_bounds(box_idx):
        """
        Compute grid bounds for one dyadic box.

        Parameters
        ----------
        box_idx : int
            Global box index.

        Returns
        -------
        bounds_min : jnp.ndarray, shape (ndim,)
            Minimum Fourier-index bound.
        bounds_max : jnp.ndarray, shape (ndim,)
            Maximum Fourier-index bound.
        """
        level = utils.find_level(dyadic_decomp, box_idx)
        box_length = dyadic_decomp.box_lengths[level]
        center = dyadic_decomp.centres_ndim[box_idx]
        bounds_min = center - box_length
        bounds_max = center + box_length - 1
        return bounds_min, bounds_max

    bounds_min, bounds_max = vmap(box_bounds)(box_corners)

    global_bounds_min = jnp.min(bounds_min, axis=0)
    global_bounds_max = jnp.max(bounds_max, axis=0)
    bounds_per_dim = jnp.stack((global_bounds_min, global_bounds_max + 1), axis=-1)  # ?

    nn = jnp.array(domain.N)
    nn = rearrange(nn, "d -> d 1")
    bounds_coords = bounds_per_dim + nn // 2
    return bounds_coords


def closest_power_of_two(size: int, max_size: int) -> int:
    """
    Returns the closest power of two to the given size.

    Parameters
    ----------
    size : int
        Input size.
    max_size : int
        Upper bound for the returned size.

    Returns
    -------
    int
        Smallest power of two greater than or equal to ``size``, capped at
        ``max_size``.
    """

    if size <= 0 or max_size <= 0:
        raise ValueError("size and max_size must be positive.")
    power_of_two = 1 << (int(size) - 1).bit_length()
    return min(power_of_two, int(max_size))


def downsample_p0(
    p0_LF: jnp.ndarray, bd: jnp.ndarray, use_power_of_two: bool = False
) -> jnp.ndarray:
    """
    Downsample p0 after applying the low pass filter to it.

    Parameters
    ----------
    p0_LF : jnp.ndarray
        Low-pass filtered field.
    bd : jnp.ndarray, shape (ndim, 2)
        Bounds of the selected low-frequency support.
    use_power_of_two : bool, default=False
        Whether to round the crop size up to a power of two.

    Returns
    -------
    jnp.ndarray
        Centred crop of ``p0_LF``.
    """
    nonzero_size = int(jnp.min(jnp.array([int(stop - start) for start, stop in bd])))

    if use_power_of_two:
        slice_size = closest_power_of_two(nonzero_size, min(p0_LF.shape))
    else:
        slice_size = nonzero_size

    starts = jnp.maximum(0, (jnp.array(p0_LF.shape) - slice_size) // 2)
    stops = starts + slice_size

    slices = tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops))

    return p0_LF[slices]


def downsample_domain(domain: Domain, p0_LF_downsampled: jnp.ndarray) -> Domain:
    """
    Downsample the domain after downsampling the p0.

    Parameters
    ----------
    domain : Domain
        Original domain.
    p0_LF_downsampled : jnp.ndarray
        Downsampled low-frequency field.

    Returns
    -------
    Domain
        Domain with shape and spacing adjusted to ``p0_LF_downsampled``.

    Notes
    -----
    The domain is downsampled to match the downsampled field.
    """
    N_resized = p0_LF_downsampled.shape
    resize_factor = tuple([domain.N[i] / N_resized[i] for i in range(len(N_resized))])
    #    dx_resized = resize_factor * domain.dx
    dx_resized = tuple([resize_factor[i] * domain.dx[i] for i in range(len(N_resized))])

    # assert jnp.allclose(dx_resized * (N_resized), domain.dx * (domain.N))

    def resize_field(field):
        if field is None or callable(field):
            return field
        values = jnp.asarray(field)
        if values.ndim == 0:
            return field
        return jax.image.resize(values, N_resized, method="linear")

    domain_downsampled = Domain(
        N=N_resized,
        dx=dx_resized,
        c=resize_field(domain.c),
        density=resize_field(domain.density),
        alpha_coeff=resize_field(domain.alpha_coeff),
        lam=domain.lam,
        alpha_power=resize_field(domain.alpha_power),
        periodic=domain.periodic,
        cfl=domain.cfl,
    )

    return domain_downsampled


def split_frequency_components(
    p0: jnp.ndarray,
    sensors_mask: jnp.ndarray,
    input_type: str,
    output_type: str,
    wpt: MSWPT,
    box_corners: Optional[jnp.ndarray],
    windowing: str,
    domain: Domain,
    cutoff_freq: Optional[float] = None,
    downsample: bool = False,
    use_pow2: bool = False,
    gh: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Domain]:
    """
    Split input into high- and low-frequency components.

    Parameters
    ----------
    p0 : jnp.ndarray
        Input field.
    sensors_mask : jnp.ndarray
        Sensor mask aligned with ``p0``.
    input_type : {"spatial", "fourier"}
        Domain of ``p0``.
    output_type : {"spatial", "fourier"}
        Domain for returned components.
    wpt : MSWPT
        Wave-packet transform defining the dyadic boxes.
    box_corners : jnp.ndarray, optional
        Pair of box indices defining the low-frequency region.
    windowing : str
        Windowing type passed to the filter construction.
    domain : Domain
        Physical domain.
    cutoff_freq : float, optional
        Frequency-radius alternative to ``box_corners``.
    downsample : bool, default=False
        Whether to downsample the low-frequency part.
    use_pow2 : bool, default=False
        Whether downsampled sizes should be powers of two.
    gh : jnp.ndarray, optional
        Precomputed low-pass filter.

    Returns
    -------
    p0_HF : jnp.ndarray
        High-frequency component in `output_type` space.
    p0_LF : jnp.ndarray
        Low-frequency component in `output_type` space.
    sensors_mask_ds : jnp.ndarray
        Possibly-downsampled sensors mask (same shape as p0_LF/p0_HF).
    dom_downsample : Domain
        Possibly-downsampled domain matching p0_LF.

    Notes
    -----
    If the low-frequency index set is empty (no bins fall inside the requested
    box / cutoff), we:
      - return p0_LF = 0 (in `output_type`),
      - return p0_HF = p0 (in `output_type`),
      - leave sensors_mask and domain unchanged,
      - skip downsampling entirely.

    This avoids shape, interpolation, and domain-consistency errors.
    """
    centers = wpt.dyadic_decomp.centres_ndim

    # Determine the low-frequency index set (idx_box) and box_corners
    if cutoff_freq is not None and box_corners is None:
        # Use L-infinity norm with inclusive boundary to match box_corners behavior
        idx_box = get_indices_with_norm_less_than(centers, cutoff_freq, inclusive=True)

        # Find actual corner indices from the selected set
        if idx_box.size > 0:
            corner1_idx, corner2_idx = find_bounding_corner_indices(centers, idx_box)
            box_corners = jnp.array([corner1_idx, corner2_idx])

    elif box_corners is not None and cutoff_freq is None:
        # Use provided box_corners directly
        idx_box = get_indices_between_two_opposing_corners(
            centers, int(box_corners[0]), int(box_corners[1])
        )
    else:
        raise ValueError("Exactly one of cutoff_freq or box_corners must be provided.")

    # If idx_box is empty, short-circuit safely.
    if idx_box.size == 0:
        warnings.warn(
            "split_frequency_components: low-frequency selection is empty; "
            "returning p0_HF = p0 and p0_LF = 0 with unchanged domain.",
            RuntimeWarning,
        )
        # Convert p0 to the requested output space.
        p0_out = utils.convert_space(p0, input_type, output_type)
        # Create a zero LF of the same shape and dtype.
        p0_LF = jnp.zeros_like(p0_out)
        p0_HF = p0_out
        # No downsampling possible/needed.
        return p0_HF, p0_LF, sensors_mask, domain

    # Normal path: compute LF/HF in Fourier domain via g/h filters.
    p0_HF_ft, p0_LF_ft = gh_lowpass_filter(
        p0, input_type, wpt, idx_box, windowing, gh=gh
    )

    # If desired, downsample the LF path and align sensors.
    sensors_mask_ds = sensors_mask
    dom_downsample = domain

    if downsample:
        assert box_corners is not None
        # Compute target bounds in spatial grid for LF.
        bounds = get_bounds(
            wpt.dyadic_decomp,
            domain,
            int(box_corners[0]),
            int(box_corners[1]),
        )

        # Downsample LF Fourier data (implementation defines axis/layout).
        p0_LF_ft = downsample_p0(p0_LF_ft, bounds, use_pow2)

        # Interpolate sensors mask to the new spatial shape that corresponds to p0_LF_ft.
        sensors_mask_ds = utils.interpolate_nearest(sensors_mask_ds, p0_LF_ft.shape)
        sensors_mask_ds = sensors_mask_ds.astype(jnp.float32)

        # Build a consistent downsampled domain for LF.
        dom_downsample = downsample_domain(domain, p0_LF_ft)

    # Convert both components to the requested output space.
    p0_HF = utils.convert_space(p0_HF_ft, "fourier", output_type)
    p0_LF = utils.convert_space(p0_LF_ft, "fourier", output_type)

    return p0_HF, p0_LF, sensors_mask_ds, dom_downsample


def oversample_window(
    array: jnp.ndarray, dt_oversample: int = 0, axis: int = 0, window_type: str = "cos2"
) -> jnp.ndarray:
    """
    Apply a windowing function to the array and oversample it in the temporal domain.

    Parameters
    ----------
    array : jnp.ndarray
        Input array to be windowed
    dt_oversample : int, default=0
        Number of points to oversample
    axis : int, default=0
        Axis along which to apply the window
    window_type : str, default='cos2'
        Type of window to apply. Options: 'cos2', 'hann', 'hamming', 'blackman'

    Returns
    -------
    jnp.ndarray
        Windowed array
    """
    if dt_oversample == 0:
        return array

    if window_type == "cos2":
        window = jnp.cos(jnp.linspace(0, jnp.pi / 2, dt_oversample)) ** 2
    elif window_type == "hann":
        window = jnp.hanning(2 * dt_oversample)[-dt_oversample:]
    elif window_type == "hamming":
        window = jnp.hamming(2 * dt_oversample)[-dt_oversample:]
    elif window_type == "blackman":
        window = jnp.blackman(2 * dt_oversample)[-dt_oversample:]
    else:
        raise ValueError(f"Unsupported window type: {window_type}")

    shape = [1] * array.ndim
    shape[axis] = dt_oversample
    window = window.reshape(shape)

    slice_obj = [slice(None)] * array.ndim
    slice_obj[axis] = slice(-dt_oversample, None)

    # JAX-compatible: use .at[] API instead of in-place assignment
    windowed_section = array[tuple(slice_obj)] * window
    result = array.at[tuple(slice_obj)].set(windowed_section)

    return result


def interpolate_LF_soln(
    lf_downsampled: jnp.ndarray,
    target_size: Tuple,
    interpolation_method: str = "spline",
    interp_window: str = "cos2",
    dt_oversample: int = 0,
    spline_order: int = 3,
) -> jnp.ndarray:
    """
    Interpolates a downsampled solution from a LF wave solver, to match the desired size.

    Parameters
    ----------
    lf_downsampled : jnp.ndarray
        Downsampled low-frequency solver output.
    target_size : Tuple[int, ...]
        Desired output shape.
    interpolation_method : {"spline", "fourier"}, default="spline"
        Interpolation method. The Fourier branch retains the historical
        two-dimensional planar-sensor normalization; new code should prefer
        :class:`beamax.solvers.HybridSolver`, which has the domain context
        required for dimension-general normalization.
    interp_window : {"cos2", "hann", "hamming", "blackman"}, default="cos2"
        Temporal taper to apply before interpolation.
    dt_oversample : int, default=0
        Number of oversampled time steps in the taper region.
    spline_order : int, default=3
        Spline order for ``scipy.ndimage.zoom``.

    Returns
    -------
    jnp.ndarray
        Interpolated low-frequency solution.
    """
    lf_windowed = oversample_window(
        lf_downsampled, dt_oversample, axis=0, window_type=interp_window
    )

    if interpolation_method == "spline":
        input_size = lf_downsampled.shape
        new_shape = tuple(
            [target_size[i] / input_size[i] for i in range(len(target_size))]
        )
        # Spline `zoom` is sample-value preserving by construction (it
        # evaluates the spline interpolant at the new grid points), so no
        # post-correction is needed; an energy-renormalisation here would
        # actively un-preserve sample values.
        lf_upsampled = zoom(lf_windowed, new_shape, order=spline_order)

    elif interpolation_method == "fourier":
        lf_upsampled = utils.interpolate_fourier(
            lf_windowed, target_size, "spatial", "spatial"
        )
        # Historical convention for a 2-D volume cropped equally in both
        # axes and observed on a 1-D planar sensor: the initial unitary crop
        # inflates amplitudes by the spatial ratio, while the bare sensor-grid
        # resize cancels only its square root. This second square-root factor
        # completes the cancellation. The helper lacks enough domain metadata
        # to generalise this rule to arbitrary dimensions/geometries.
        scale = math.sqrt(
            math.prod(
                input_len / output_len
                for input_len, output_len in zip(lf_windowed.shape, target_size)
            )
        )
        lf_upsampled = lf_upsampled * scale
    else:
        raise ValueError(f"Interpolation method {interpolation_method} not supported.")

    return jnp.asarray(lf_upsampled)
