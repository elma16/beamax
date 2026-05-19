from __future__ import annotations

import jax.numpy as jnp
from typing import Tuple, Union, Optional, Dict, Literal

from scipy.ndimage import zoom

from pathlib import Path
import h5py
import numpy as np

# ---- OA-BREAST label codes: {0,2,3,4,5} = {bg, fibro, fat, skin, vessel}
VALID_LABELS = {0, 2, 3, 4, 5}
VESSEL_LABEL = 5

# ---- Default acoustic/optical properties
# Lucka-style speeds (m/s) as a sane default baseline
DEFAULT_SOS = {0: 1500.0, 2: 1515.0, 3: 1470.0, 4: 1650.0, 5: 1584.0}
# Crude μa [1/m]; replace with wavelength-specific numbers if you have them
DEFAULT_MUA = {0: 0.10, 2: 0.15, 3: 0.05, 4: 0.20, 5: 2.00}


# ------------------------------ HDF5 helpers ------------------------------ #


def _first_dataset(f: h5py.File) -> np.ndarray:
    """
    Return the first dataset found in an HDF5 file or first-level group.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file.

    Returns
    -------
    np.ndarray
        Dataset values loaded into memory.

    Raises
    ------
    ValueError
        If no dataset is found.
    """
    for _, obj in f.items():
        if isinstance(obj, h5py.Dataset):
            return np.array(obj)
        if isinstance(obj, h5py.Group):
            for __, oo in obj.items():
                if isinstance(oo, h5py.Dataset):
                    return np.array(oo)
    raise ValueError("No datasets found in HDF5.")


def _find_labels(f: h5py.File) -> np.ndarray:
    """
    Locate a label dataset in an OA-BREAST HDF5 file.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file.

    Returns
    -------
    np.ndarray
        Label volume.
    """
    for key in ("MergedPhantom", "merged_phantom", "phantom", "labels", "tissueType"):
        if key in f and isinstance(f[key], h5py.Dataset):
            return np.array(f[key])
    return _first_dataset(f)


def _ensure_axis_order_zyx(arr: np.ndarray, axis_order: str) -> np.ndarray:
    """
    Convert a label volume to ``(Z, Y, X)`` axis order.

    Parameters
    ----------
    arr : np.ndarray
        Input label volume.
    axis_order : {"ZYX", "XYZ", "YZX"}
        Axis order of ``arr``.

    Returns
    -------
    np.ndarray
        Volume in ``(Z, Y, X)`` order.

    Raises
    ------
    ValueError
        If ``axis_order`` is unsupported.
    """
    ao = axis_order.upper()
    if ao == "ZYX":
        return arr
    if ao == "XYZ":
        return np.transpose(arr, (2, 1, 0))
    if ao == "YZX":
        return np.transpose(arr, (1, 0, 2))
    raise ValueError(f"Unsupported axis_order={axis_order}. Implement a transpose.")


def _check_labels(lbl: np.ndarray):
    """
    Validate that labels are within the OA-BREAST label set.

    Parameters
    ----------
    lbl : np.ndarray
        Label array.

    Raises
    ------
    ValueError
        If labels outside :data:`VALID_LABELS` are present.
    """
    u = set(np.unique(lbl).tolist())
    if not u.issubset(VALID_LABELS):
        raise ValueError(
            f"Unexpected labels {sorted(u)}; expected subset of {sorted(VALID_LABELS)}."
        )


def _resample_labels_to_shape(
    labels_zyx: np.ndarray, target_shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Resample labels to a target shape with nearest-neighbour interpolation.

    Parameters
    ----------
    labels_zyx : np.ndarray
        Label volume in ``(Z, Y, X)`` order.
    target_shape : Tuple[int, int, int]
        Desired output shape.

    Returns
    -------
    np.ndarray
        Resampled label volume.

    Raises
    ------
    ValueError
        If any target dimension is non-positive.
    """
    src_shape = np.array(labels_zyx.shape, dtype=float)
    tgt_shape = np.array(target_shape, dtype=float)
    if (tgt_shape <= 0).any():
        raise ValueError("Invalid target_shape.")
    zf = tuple((tgt_shape / src_shape))
    try:
        return zoom(labels_zyx, zf, order=0, grid_mode=True)
    except TypeError:
        return zoom(labels_zyx, zf, order=0)


def _resample_labels_to_spacing(
    labels_zyx: np.ndarray,
    src_spacing_mm: Tuple[float, float, float],
    tgt_spacing_mm: Tuple[float, float, float],
) -> np.ndarray:
    """
    Resample labels from source spacing to target spacing.

    Parameters
    ----------
    labels_zyx : np.ndarray
        Label volume in ``(Z, Y, X)`` order.
    src_spacing_mm : Tuple[float, float, float]
        Source spacing ``(dz, dy, dx)`` in millimetres.
    tgt_spacing_mm : Tuple[float, float, float]
        Target spacing ``(dz, dy, dx)`` in millimetres.

    Returns
    -------
    np.ndarray
        Resampled label volume.

    Raises
    ------
    ValueError
        If any spacing is non-positive.
    """
    sz, sy, sx = map(float, src_spacing_mm)
    tz, ty, tx = map(float, tgt_spacing_mm)
    if min(tz, ty, tx) <= 0 or min(sz, sy, sx) <= 0:
        raise ValueError("Bad spacing (must be positive).")
    zf = (sz / tz, sy / ty, sx / tx)
    try:
        return zoom(labels_zyx, zf, order=0, grid_mode=True)
    except TypeError:
        return zoom(labels_zyx, zf, order=0)


def _effective_spacing_after_shape_resample(src_spacing_mm, src_shape, tgt_shape):
    """
    Compute effective spacing after resampling to a target shape.

    Parameters
    ----------
    src_spacing_mm : Tuple[float, float, float]
        Source spacing in millimetres.
    src_shape : Tuple[int, int, int]
        Source volume shape.
    tgt_shape : Tuple[int, int, int]
        Target volume shape.

    Returns
    -------
    Tuple[float, float, float]
        Effective target spacing in millimetres.
    """
    src_shape = np.array(src_shape, dtype=float)
    tgt_shape = np.array(tgt_shape, dtype=float)
    zf = tgt_shape / src_shape
    return tuple((np.array(src_spacing_mm, dtype=float) / zf).tolist())


def _slice_axis_and_spacing(
    labels_zyx: np.ndarray,
    spacing_zyx_mm: Tuple[float, float, float],
    slice_axis: int,
    slice_idx: Optional[int],
    policy: Literal["middle", "max_variance", None],
):
    """
    Extract a 2D slice and corresponding spacing from a 3D label volume.

    Parameters
    ----------
    labels_zyx : np.ndarray
        Label volume in ``(Z, Y, X)`` order.
    spacing_zyx_mm : Tuple[float, float, float]
        Source spacing in ``(Z, Y, X)`` order.
    slice_axis : int
        Axis to slice along.
    slice_idx : int, optional
        Explicit slice index. If ``None``, ``policy`` selects one.
    policy : {"middle", "max_variance", None}
        Slice-selection policy used when ``slice_idx`` is ``None``.

    Returns
    -------
    lbl2d : np.ndarray
        Extracted 2D label image.
    slice_idx : int
        Chosen slice index.
    sp2d : Tuple[float, float]
        Spacing of the returned 2D image.

    Raises
    ------
    ValueError
        If ``slice_axis`` is invalid for ``max_variance`` selection.
    """
    Z, Y, X = labels_zyx.shape
    if slice_idx is None:
        if policy == "max_variance":
            if slice_axis == 0:
                stats = np.var(labels_zyx, axis=(1, 2))
            elif slice_axis == 1:
                stats = np.var(labels_zyx, axis=(0, 2))
            elif slice_axis == 2:
                stats = np.var(labels_zyx, axis=(0, 1))
            else:
                raise ValueError("slice_axis must be 0,1,2")
            slice_idx = int(np.argmax(stats))
        else:
            slice_idx = [Z // 2, Y // 2, X // 2][slice_axis]

    slc = [slice(None)] * 3
    slc[slice_axis] = slice_idx
    lbl2d = labels_zyx[tuple(slc)]
    sp = list(spacing_zyx_mm)
    sp2d = (
        tuple(sp[1:])
        if slice_axis == 0
        else (sp[0], sp[2])
        if slice_axis == 1
        else tuple(sp[:2])
    )
    return lbl2d, slice_idx, sp2d


# ------------------------- New mapping/fill helpers ------------------------ #


def _make_p0_vessels_only(
    labels: np.ndarray,
    label_to_mua: Dict[int, float],
    gruneisen: float,
    normalize: bool,
) -> np.ndarray:
    """
    Build a vessel-only initial pressure map.

    Parameters
    ----------
    labels : np.ndarray
        Tissue label array.
    label_to_mua : Dict[int, float]
        Absorption coefficient per label.
    gruneisen : float
        Gruneisen coefficient.
    normalize : bool
        Whether to normalize nonzero pressure values to maximum one.

    Returns
    -------
    np.ndarray
        Vessel mask scaled by ``gruneisen * mua[vessel]``.
    """
    if VESSEL_LABEL not in np.unique(labels):
        # no vessels present; return zeros
        p0 = np.zeros_like(labels, dtype=np.float32)
    else:
        val = float(label_to_mua.get(VESSEL_LABEL, label_to_mua[0]))
        p0 = (labels == VESSEL_LABEL).astype(np.float32) * (gruneisen * val)
    if normalize and p0.max() > 0:
        p0 = p0 / p0.max()
    return p0.astype(np.float32)


def _labels_to_c_map(labels: np.ndarray, label_to_sos: Dict[int, float]) -> np.ndarray:
    """
    Map tissue labels to sound speed values.

    Parameters
    ----------
    labels : np.ndarray
        Tissue label array.
    label_to_sos : Dict[int, float]
        Sound speed per label.

    Returns
    -------
    np.ndarray
        Sound-speed map.
    """
    c = np.empty(labels.shape, dtype=np.float32)
    uniq = np.unique(labels)
    for lab in uniq:
        c[labels == lab] = float(label_to_sos.get(int(lab), label_to_sos[0]))
    return c


def _fill_line_nearest_non_vessel(
    c_line: np.ndarray, lab_line: np.ndarray, default_speed: float
) -> np.ndarray:
    """
    In 1D, replace vessel positions by the nearest non-vessel speed along the line.

    Parameters
    ----------
    c_line : np.ndarray, shape (L,)
        Sound-speed line.
    lab_line : np.ndarray, shape (L,)
        Label line aligned with ``c_line``.
    default_speed : float
        Fallback speed if the full line is vessel-labelled.

    Returns
    -------
    np.ndarray, shape (L,)
        Filled sound-speed line.

    Notes
    -----
    If a line is all vessels, fill with default_speed.
    """
    L = c_line.shape[0]
    non_vessel = lab_line != VESSEL_LABEL
    if np.all(~non_vessel):
        return np.full_like(c_line, default_speed, dtype=np.float32)

    idx = np.arange(L)

    # nearest to the left
    left_idx = np.where(non_vessel, idx, -1)
    left_idx = np.maximum.accumulate(left_idx)

    # nearest to the right
    right_idx_rev = np.where(non_vessel[::-1], idx, -1)
    right_idx_rev = np.maximum.accumulate(right_idx_rev)
    right_idx = np.where(right_idx_rev >= 0, L - 1 - right_idx_rev, -1)

    out = c_line.copy()
    vessel_pos = np.where(~non_vessel)[0]
    for i in vessel_pos:
        li = left_idx[i]
        ri = right_idx[i]
        dl = np.inf if li < 0 else (i - li)
        dr = np.inf if ri < 0 else (ri - i)
        if dl == np.inf and dr == np.inf:
            out[i] = default_speed
        elif dl <= dr:
            out[i] = c_line[li]
        else:
            out[i] = c_line[ri]
    return out


def _fill_vessels_in_c_by_nearest_along_axis(
    c_map: np.ndarray,
    labels: np.ndarray,
    axis: int,
    default_speed: float,
) -> np.ndarray:
    """
    Replace vessel-labelled voxels with nearest non-vessel speeds.

    Parameters
    ----------
    c_map : np.ndarray
        Sound-speed map.
    labels : np.ndarray
        Label array aligned with ``c_map``.
    axis : int
        Axis along which nearest non-vessel values are searched.
    default_speed : float
        Fallback speed for all-vessel lines.

    Returns
    -------
    np.ndarray
        Filled sound-speed map.

    Raises
    ------
    ValueError
        If ``labels`` is not 2D/3D or if ``axis`` is invalid.

    Notes
    -----
    The fill is column-wise and one-dimensional along the specified axis.
    """
    if labels.ndim not in (2, 3):
        raise ValueError("Expected 2D or 3D labels for nearest-along-axis fill.")

    out = c_map.copy()

    if labels.ndim == 2:
        H, W = labels.shape
        if axis == 0:
            for x in range(W):
                out[:, x] = _fill_line_nearest_non_vessel(
                    out[:, x], labels[:, x], default_speed
                )
        elif axis == 1:
            for y in range(H):
                out[y, :] = _fill_line_nearest_non_vessel(
                    out[y, :], labels[y, :], default_speed
                )
        else:
            raise ValueError("axis must be 0 or 1 for 2D.")
        return out.astype(np.float32)

    # 3D
    Z, Y, X = labels.shape
    if axis == 0:
        for y in range(Y):
            for x in range(X):
                out[:, y, x] = _fill_line_nearest_non_vessel(
                    out[:, y, x], labels[:, y, x], default_speed
                )
    elif axis == 1:
        for z in range(Z):
            for x in range(X):
                out[z, :, x] = _fill_line_nearest_non_vessel(
                    out[z, :, x], labels[z, :, x], default_speed
                )
    elif axis == 2:
        for z in range(Z):
            for y in range(Y):
                out[z, y, :] = _fill_line_nearest_non_vessel(
                    out[z, y, :], labels[z, y, :], default_speed
                )
    else:
        raise ValueError("axis must be 0,1,2 for 3D.")
    return out.astype(np.float32)


def _max_intensity_proj_vessels(labels_zyx: np.ndarray, axis: int) -> np.ndarray:
    """
    Compute a boolean maximum-intensity projection of vessel labels.

    Parameters
    ----------
    labels_zyx : np.ndarray
        Label volume in ``(Z, Y, X)`` order.
    axis : int
        Axis along which to project.

    Returns
    -------
    np.ndarray
        Two-dimensional vessel mask as ``float32``.
    """
    vmax = (labels_zyx == VESSEL_LABEL).max(axis=axis).astype(np.float32)
    return vmax


# ----------------------- Public mapping (reworked) ------------------------ #


def load_oabreast_p0_c(
    path: Union[str, Path],
    dim: Literal["2d", "3d"],
    *,
    axis_order: str = "ZYX",  # axis order in HDF5 dataset
    source_spacing_mm: Optional[Tuple[float, float, float]] = None,
    target_shape: Optional[Tuple[int, ...]] = None,  # (Z,Y,X) if 3D, (H,W) if 2D
    target_spacing_mm: Optional[Tuple[float, float, float]] = None,  # 3D only
    slice_axis: int = 0,  # 2D: axis to slice or MIP along
    slice_idx: Optional[int] = None,  # 2D: explicit index if not using policy
    slice_policy: Literal["middle", "max_variance", None] = "middle",
    vessel_only_p0: bool = True,
    c_exclude_vessels: bool = True,
    c_fill_strategy: Literal[
        "nearest_along_axis", "background", "keep"
    ] = "nearest_along_axis",
    c_fill_axis: int = 0,  # line direction for 'nearest_along_axis' (2D/3D)
    background_speed: float = DEFAULT_SOS[0],
    # 2D option: use a maximum-intensity projection along slice_axis for p0.
    vessels_mip_2d: bool = False,
    # Optical parameters
    label_to_sos: Optional[Dict[int, float]] = None,
    label_to_mua: Optional[Dict[int, float]] = None,
    gruneisen: float = 0.2,
    normalize_p0: bool = False,
    # Diagnostics
    return_labels: bool = False,
):
    """
    Load a user-supplied OA-Breast HDF5 label volume and map labels to ``(p0, c)``.

    beamax does not ship OA-Breast phantom data. Download the dataset separately
    and pass the local HDF5 path here.

    Supports:
      - p0 from vessels only (default),
      - c(x) from all tissues except vessels (default),
      - optional 2D vessel MIP for p0 along ``slice_axis``.

    Parameters
    ----------
    path : str | Path
    dim : {"2d","3d"}
    axis_order : {"ZYX","XYZ","YZX"}
        Axis order of the labels dataset in the file.
    source_spacing_mm : (dz,dy,dx) or None
        Physical spacing in mm of the source volume if known.
    target_shape : 3D→(Z,Y,X) or 2D→(H,W)
        If given, resample labels to this size (nearest).
    target_spacing_mm : (dz,dy,dx) or None
        For 3D: resample to desired spacing (needs source_spacing_mm).
    slice_axis : int
        For 2D: which axis to slice (or MIP) along (0=Z,1=Y,2=X).
    slice_idx : int | None
        For 2D slicing (ignored if vessels_mip_2d=True). If None use policy.
    slice_policy : {"middle","max_variance",None}
        Heuristic to pick a slice when slice_idx is None (2D only).
    vessel_only_p0 : bool
        If True, p0 is nonzero only on vessel label (5).
    c_exclude_vessels : bool
        If True, replace vessel-labelled c-values using c_fill_strategy.
    c_fill_strategy : {"nearest_along_axis","background","keep"}
        Strategy to fill c where labels==vessel.
    c_fill_axis : int
        Axis along which to search for nearest non-vessel when using "nearest_along_axis".
        For 2D: 0 or 1. For 3D: 0,1,2.
    background_speed : float
        Fallback speed used by "background" strategy (default maps label 0).
    vessels_mip_2d : bool
        If True (2D), compute p0 from vessel MIP along `slice_axis` instead of a single slice.
        c is still computed from a single slice (with vessels excluded if requested).
    label_to_sos, label_to_mua, gruneisen, normalize_p0 : standard optical/acoustic params
    return_labels : bool
        If True, also return the label array used (2D or 3D).

    Returns
    -------
    If dim='3d': (p0_3d, c_3d[, labels_3d], meta)
    If dim='2d': (p0_2d, c_2d[, labels_2d], meta)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # 1) Load labels and enforce ZYX ordering
    with h5py.File(path, "r") as f:
        labels = _find_labels(f)
    labels = labels.astype(np.uint8, copy=False)
    labels = _ensure_axis_order_zyx(labels, axis_order)
    _check_labels(labels)

    # Default maps
    sos_map = (
        dict(DEFAULT_SOS) if label_to_sos is None else {**DEFAULT_SOS, **label_to_sos}
    )
    mua_map = (
        dict(DEFAULT_MUA) if label_to_mua is None else {**DEFAULT_MUA, **label_to_mua}
    )

    meta = {
        "file": str(path),
        "dim": dim,
        "axis_order_in_file": axis_order,
        "vol_shape_in": tuple(labels.shape),
        "options": {
            "vessel_only_p0": vessel_only_p0,
            "c_exclude_vessels": c_exclude_vessels,
            "c_fill_strategy": c_fill_strategy,
            "c_fill_axis": int(c_fill_axis),
            "vessels_mip_2d": vessels_mip_2d if dim == "2d" else False,
        },
    }

    if dim == "3d":
        labels_3d = labels
        spacing_out = source_spacing_mm  # may be None

        # Resample by target_shape or target_spacing
        if target_shape is not None:
            if len(target_shape) != 3:
                raise ValueError("target_shape must be (Z,Y,X) for dim='3d'.")
            labels_3d = _resample_labels_to_shape(labels_3d, target_shape)
            spacing_out = _effective_spacing_after_shape_resample(
                source_spacing_mm if source_spacing_mm else (1.0, 1.0, 1.0),
                labels.shape,
                labels_3d.shape,
            )
        elif (target_spacing_mm is not None) and (source_spacing_mm is not None):
            labels_3d = _resample_labels_to_spacing(
                labels_3d, source_spacing_mm, target_spacing_mm
            )
            spacing_out = target_spacing_mm

        # p0: vessels only (default) or classic map
        if vessel_only_p0:
            p0_np = _make_p0_vessels_only(labels_3d, mua_map, gruneisen, normalize_p0)
        else:
            # classic p0 = Γ * μa(label)
            p0_np = np.empty_like(labels_3d, dtype=np.float32)
            for lab in np.unique(labels_3d):
                p0_np[labels_3d == lab] = gruneisen * float(
                    mua_map.get(int(lab), mua_map[0])
                )
            if normalize_p0 and p0_np.max() > 0:
                p0_np = p0_np / p0_np.max()

        # c: map then optionally exclude vessels
        c_np = _labels_to_c_map(labels_3d, sos_map)
        if c_exclude_vessels:
            if c_fill_strategy == "nearest_along_axis":
                c_np = _fill_vessels_in_c_by_nearest_along_axis(
                    c_np, labels_3d, c_fill_axis, background_speed
                )
            elif c_fill_strategy == "background":
                c_np[labels_3d == VESSEL_LABEL] = float(background_speed)
            elif c_fill_strategy == "keep":
                pass
            else:
                raise ValueError(f"Unknown c_fill_strategy '{c_fill_strategy}'.")

        meta.update(
            {
                "shape_out": tuple(labels_3d.shape),
                "spacing_out_mm": tuple(spacing_out)
                if spacing_out is not None
                else None,
                "label_set": [int(x) for x in np.unique(labels_3d)],
            }
        )
        out = (jnp.array(p0_np), jnp.array(c_np))
        if return_labels:
            out += (labels_3d,)
        out += (meta,)
        return out

    # ------------------------------- 2D path ------------------------------- #

    if source_spacing_mm is None:
        spacing_zyx = (1.0, 1.0, 1.0)  # unit spacing when physical spacing is unknown
    else:
        spacing_zyx = source_spacing_mm

    # -- Compute p0 (slice or MIP for vessels), and c from a slice
    if vessels_mip_2d:
        # p0 from MIP along slice_axis; c from the "middle"/policy slice (rest-only)
        vessel_mip = _max_intensity_proj_vessels(labels, axis=slice_axis)
        # optional 2D resample
        if target_shape is not None:
            if len(target_shape) != 2:
                raise ValueError("target_shape must be (H,W) for dim='2d'.")
            src_shape = np.array(vessel_mip.shape, dtype=float)
            tgt_shape = np.array(target_shape, dtype=float)
            zf2 = tuple(tgt_shape / src_shape)
            try:
                vessel_mip = zoom(vessel_mip, zf2, order=0, grid_mode=True)
            except TypeError:
                vessel_mip = zoom(vessel_mip, zf2, order=0)
        # scale to Γ * μa[vessel]
        p0_2d = vessel_mip.astype(np.float32) * (
            gruneisen * float(mua_map.get(VESSEL_LABEL, mua_map[0]))
        )
        if normalize_p0 and p0_2d.max() > 0:
            p0_2d = p0_2d / p0_2d.max()

        # For c: pick a representative slice and exclude vessels there
        lbl2d, chosen_idx, spacing2d = _slice_axis_and_spacing(
            labels, spacing_zyx, slice_axis, None, slice_policy
        )

        if target_shape is not None:
            # Bring c-slice to same 2D shape as p0_2d
            src_shape = np.array(lbl2d.shape, dtype=float)
            tgt_shape = np.array(p0_2d.shape, dtype=float)
            zf2 = tuple(tgt_shape / src_shape)
            try:
                lbl2d = zoom(lbl2d, zf2, order=0, grid_mode=True)
            except TypeError:
                lbl2d = zoom(lbl2d, zf2, order=0)
            spacing2d = tuple((np.array(spacing2d) * (src_shape / tgt_shape)).tolist())

        c2d = _labels_to_c_map(lbl2d, sos_map)
        if c_exclude_vessels:
            if c_fill_strategy == "nearest_along_axis":
                # Choose a sensible axis in 2D; reuse c_fill_axis
                c2d = _fill_vessels_in_c_by_nearest_along_axis(
                    c2d, lbl2d, axis=c_fill_axis, default_speed=background_speed
                )
            elif c_fill_strategy == "background":
                c2d[lbl2d == VESSEL_LABEL] = float(background_speed)
            elif c_fill_strategy == "keep":
                pass

        meta.update(
            {
                "slice_axis": int(slice_axis),
                "slice_index_for_c": int(chosen_idx),
                "shape_out": tuple(p0_2d.shape),
                "spacing_out_mm": tuple(spacing2d),
                "label_set": [int(x) for x in np.unique(lbl2d)],
                "p0_from_vessel_MIP": True,
            }
        )
        out = (jnp.array(p0_2d), jnp.array(c2d))
        if return_labels:
            out += (lbl2d,)
        out += (meta,)
        return out

    # --- Standard 2D slice workflow (no vessel MIP) ---
    lbl2d, chosen_idx, spacing2d = _slice_axis_and_spacing(
        labels, spacing_zyx, slice_axis, slice_idx, slice_policy
    )

    # Optional 2D resample to (H,W)
    if target_shape is not None:
        if len(target_shape) != 2:
            raise ValueError("target_shape must be (H,W) for dim='2d'.")
        src_shape = np.array(lbl2d.shape, dtype=float)
        tgt_shape = np.array(target_shape, dtype=float)
        zf2 = tuple(tgt_shape / src_shape)
        try:
            lbl2d = zoom(lbl2d, zf2, order=0, grid_mode=True)
        except TypeError:
            lbl2d = zoom(lbl2d, zf2, order=0)
        spacing2d = tuple((np.array(spacing2d) * (src_shape / tgt_shape)).tolist())

    # p0 (vessels-only or classic) on this slice
    if vessel_only_p0:
        p0_np = _make_p0_vessels_only(lbl2d, mua_map, gruneisen, normalize_p0)
    else:
        p0_np = np.empty_like(lbl2d, dtype=np.float32)
        for lab in np.unique(lbl2d):
            p0_np[lbl2d == lab] = gruneisen * float(mua_map.get(int(lab), mua_map[0]))
        if normalize_p0 and p0_np.max() > 0:
            p0_np = p0_np / p0_np.max()

    # c on this slice, excluding vessels if requested
    c_np = _labels_to_c_map(lbl2d, sos_map)
    if c_exclude_vessels:
        if c_fill_strategy == "nearest_along_axis":
            c_np = _fill_vessels_in_c_by_nearest_along_axis(
                c_np, lbl2d, axis=c_fill_axis, default_speed=background_speed
            )
        elif c_fill_strategy == "background":
            c_np[lbl2d == VESSEL_LABEL] = float(background_speed)
        elif c_fill_strategy == "keep":
            pass

    meta.update(
        {
            "slice_axis": int(slice_axis),
            "slice_index": int(chosen_idx),
            "shape_out": tuple(lbl2d.shape),
            "spacing_out_mm": tuple(spacing2d),
            "label_set": [int(x) for x in np.unique(lbl2d)],
            "p0_from_vessel_MIP": False,
        }
    )
    out = (jnp.array(p0_np), jnp.array(c_np))
    if return_labels:
        out += (lbl2d,)
    out += (meta,)
    return out
