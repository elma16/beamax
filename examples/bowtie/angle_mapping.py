#!/usr/bin/env python3
# coding: utf-8
"""
Angle-mapping experiment for planar line-sensor PAT data.

For each transform (MSWPT, curvelets, wavelets), this script:
  1) Builds a set of representative packets (one per "box"/wedge/subband).
  2) Uses each packet as an initial condition p0.
  3) Solves the 2D wave equation with k-Wave on a line sensor.
  4) Decomposes the data in (s, t) with the same transform.
  5) Records the dominant output box and its angle.

Angles are measured from Fourier-domain centers (wavefront normals). In data space,
angles are computed in the scaled coordinates (k_s, omega/c0) so both axes have
units of 1/m.

Dependencies:
  - k-wave-python
  - optional: curvelops, PyWavelets
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from beamax import geometry, utils
from beamax.decomposition import DyadicDecomposition
from beamax.transforms import MSWPT
from beamax.solvers import KWaveSolver
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions

try:
    import curvelops as cl
except ModuleNotFoundError:
    cl = None

try:
    import pywt
except ModuleNotFoundError:
    pywt = None

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

ROOT_DIR = utils.detect_root()
PLOT_DIR = Path(ROOT_DIR / "plots")
DATA_DIR = Path(ROOT_DIR / "data")
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

N = (128, 128)
DX = (1e-4, 1e-4)
C0 = 1500.0
CFL = float(np.round(np.sqrt(2) / 4, 3))
PERIODIC = (False, False)

SENSOR_AXIS = 0
SENSOR_INDEX = 0

MAX_PACKETS = 400

NUM_LEVELS = 3
NUM_BOXES_LEVELS = tuple(2 ** (i + 2) for i in range(NUM_LEVELS))
BOX_ASPECT_RATIO = (1, 1)
REDUNDANCY = 2
WINDOWING = "rectangular_mirror"

WAVELET_NAME = "db4"
WAVELET_LEVEL = 4
WAVELET_MODE = "periodization"

CURVELET_NUM_SCALES = NUM_LEVELS
CURVELET_ANGLES_COARSE = 8
CURVELET_SAMPLES_PER_SCALE = 48
CURVELET_ANGLE_TOL_DEG = 7.5

MIN_CENTER_RADIUS = 1e-6
IMAGE_HALF_PLANE_AXIS = 0
DATA_HALF_PLANE_AXIS = 1

# Plot/diagnostic controls
ENERGY_KEEP_FRACTION = None  # e.g., 0.5 keeps top 50% per transform
PLOT_ABS_ANGLES = False

RUN_TRANSFORMS = ("mswpt", "curvelet", "wavelet")
# RUN_TRANSFORMS = ("curvelet", "wavelet")


def c_hom(x: jnp.ndarray) -> jnp.ndarray:
    return C0 + 0.0 * x[..., 0]


def make_line_sensor_mask(shape: Tuple[int, int], axis: int, index: int) -> jnp.ndarray:
    mask = jnp.zeros(shape)
    slicer = [slice(None), slice(None)]
    slicer[axis] = index
    return mask.at[tuple(slicer)].set(1.0)


def ensure_ns_nt(sensor_data: np.ndarray, ts: np.ndarray) -> np.ndarray:
    Nt = len(ts)
    arr = np.asarray(sensor_data)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D sensor data, got shape {arr.shape}.")
    if arr.shape[1] == Nt:
        return arr
    if arr.shape[0] == Nt:
        return arr.T
    raise ValueError(
        f"Cannot infer time axis. sensor_data.shape={arr.shape}, len(ts)={Nt}"
    )


def compute_even_slices(Ns: int, Nt: int) -> Tuple[slice, slice]:
    s_stop = Ns - (Ns % 2)
    if s_stop <= 0:
        raise ValueError("Sensor dimension must be positive and evenable.")
    t_stop = Nt - (Nt % 2)
    if t_stop <= 0:
        raise ValueError("Time dimension must be positive and evenable.")
    return slice(0, s_stop), slice(0, t_stop)


def preprocess_sensor_data(
    sensor_data: np.ndarray,
    ts: np.ndarray,
    s_slice: slice,
    t_slice: slice,
) -> np.ndarray:
    arr = ensure_ns_nt(sensor_data, ts)
    arr = np.asarray(arr)[s_slice, t_slice]
    return arr


def estimate_frequency_center(
    atom: np.ndarray, *, half_plane_axis: int, thresh_frac: float = 0.85
) -> Optional[np.ndarray]:
    if atom.ndim != 2:
        raise ValueError(f"Only 2D atoms supported, got {atom.shape}.")
    spec = np.abs(np.fft.fftshift(np.fft.fftn(atom, norm="ortho")))
    n0, n1 = atom.shape
    k0 = np.arange(-n0 // 2, n0 // 2)
    k1 = np.arange(-n1 // 2, n1 // 2)
    K0, K1 = np.meshgrid(k0, k1, indexing="ij")

    if half_plane_axis == 0:
        mask = (K0 > 0) | ((K0 == 0) & (K1 >= 0))
    else:
        mask = (K1 > 0) | ((K1 == 0) & (K0 >= 0))
    spec = np.where(mask, spec, 0.0)

    max_val = spec.max()
    if max_val <= 0:
        return None

    thresh = max_val * float(thresh_frac)
    weights = np.where(spec >= thresh, spec, 0.0)
    total = weights.sum()
    if total > 0:
        c0 = float((weights * K0).sum() / total)
        c1 = float((weights * K1).sum() / total)
        return np.array([c0, c1], dtype=float)

    idx = int(np.argmax(spec))
    i0, i1 = np.unravel_index(idx, spec.shape)
    return np.array([float(k0[i0]), float(k1[i1])])


def _center_idx_to_freq(
    center_idx: np.ndarray, shape: Tuple[int, int], spacing: Tuple[float, float]
) -> np.ndarray:
    center_idx = np.asarray(center_idx, dtype=float)
    shape_arr = np.asarray(shape, dtype=float)
    spacing_arr = np.asarray(spacing, dtype=float)
    return center_idx / (shape_arr * spacing_arr)


def _half_plane_mask(centers: np.ndarray, axis: int) -> np.ndarray:
    if axis == 0:
        return (centers[:, 0] > 0) | ((centers[:, 0] == 0) & (centers[:, 1] >= 0))
    return (centers[:, 1] > 0) | ((centers[:, 1] == 0) & (centers[:, 0] >= 0))


def _data_center_metrics(
    center_idx: Optional[np.ndarray],
    shape: Tuple[int, int],
    spacing: Tuple[float, float],
) -> Tuple[float, float, float]:
    if center_idx is None:
        return float("nan"), float("nan"), float("nan")
    freq = _center_idx_to_freq(center_idx, shape, spacing)
    k_s = float(freq[0])
    omega_over_c = float(np.abs(freq[1]))
    ratio = abs(k_s) / max(omega_over_c, 1e-12)
    return k_s, omega_over_c, ratio


def theta_from_center_image(
    center_idx: Optional[np.ndarray],
    shape: Tuple[int, int],
    spacing: Tuple[float, float],
    min_radius: float,
) -> float:
    if center_idx is None:
        return float("nan")
    freq = _center_idx_to_freq(center_idx, shape, spacing)
    if SENSOR_AXIS == 0:
        k_perp = float(np.abs(freq[0]))
        k_s = float(freq[1])
    else:
        k_perp = float(np.abs(freq[1]))
        k_s = float(freq[0])
    if np.hypot(k_perp, k_s) < min_radius:
        return float("nan")
    return float(np.arctan2(k_s, k_perp))


def beta_from_center_data(
    center_idx: Optional[np.ndarray],
    shape: Tuple[int, int],
    spacing: Tuple[float, float],
    min_radius: float,
) -> float:
    if center_idx is None:
        return float("nan")
    freq = _center_idx_to_freq(center_idx, shape, spacing)
    k_s = float(freq[0])
    omega_over_c = float(np.abs(freq[1]))
    if np.hypot(omega_over_c, k_s) < min_radius:
        return float("nan")
    return float(np.arctan2(k_s, omega_over_c))


def normalize_max(arr: np.ndarray) -> np.ndarray:
    max_abs = float(np.max(np.abs(arr)))
    return arr if max_abs == 0 else arr / max_abs


def data_axis_spacing(
    dx: Tuple[float, float], dt: float, c0: float
) -> Tuple[float, float]:
    if SENSOR_AXIS == 0:
        dx_s = float(dx[1])
    else:
        dx_s = float(dx[0])
    return dx_s, float(dt * c0)


def safe_num_levels(num_levels: int, n_ref: int, base: int) -> int:
    if n_ref < base:
        return 1
    max_levels = int(np.floor(np.log2(n_ref / base)) + 1)
    return int(min(num_levels, max_levels))


@dataclass(frozen=True)
class MSWPTPacket:
    box_idx: int
    coeff_idx: int
    angle_rad: float
    level: int
    label: str


def mswpt_coeff_index(wpt: MSWPT, box_idx: int) -> int:
    level = int(utils.find_level(wpt.dyadic_decomp, box_idx))
    local_box = int(box_idx - wpt.boxes_cumsum[level])
    shape = wpt.coeff_shapes[level]
    support = shape[1:]
    center = (local_box,) + tuple(int(s // 2) for s in support)
    flat = int(np.ravel_multi_index(center, shape))
    return int(wpt.coeffs_cumsum[level] + flat)


def mswpt_box_angles_image(
    dyadic: DyadicDecomposition, spacing: Tuple[float, float], min_radius: float
) -> np.ndarray:
    centers = np.asarray(dyadic.centres_ndim, dtype=float)
    mask = _half_plane_mask(centers, IMAGE_HALF_PLANE_AXIS)
    shape_arr = np.asarray(dyadic.N, dtype=float)
    spacing_arr = np.asarray(spacing, dtype=float)
    freq = centers / (shape_arr * spacing_arr)

    if SENSOR_AXIS == 0:
        k_perp = np.abs(freq[:, 0])
        k_s = freq[:, 1]
    else:
        k_perp = np.abs(freq[:, 1])
        k_s = freq[:, 0]

    norms = np.hypot(k_perp, k_s)
    angles = np.arctan2(k_s, k_perp)
    angles = np.where(mask & (norms >= min_radius), angles, np.nan)
    return angles


def mswpt_box_angles_data(
    dyadic: DyadicDecomposition, spacing: Tuple[float, float], min_radius: float
) -> np.ndarray:
    centers = np.asarray(dyadic.centres_ndim, dtype=float)
    mask = _half_plane_mask(centers, DATA_HALF_PLANE_AXIS)
    shape_arr = np.asarray(dyadic.N, dtype=float)
    spacing_arr = np.asarray(spacing, dtype=float)
    freq = centers / (shape_arr * spacing_arr)

    k_s = freq[:, 0]
    omega_over_c = np.abs(freq[:, 1])

    norms = np.hypot(omega_over_c, k_s)
    angles = np.arctan2(k_s, omega_over_c)
    angles = np.where(mask & (norms >= min_radius), angles, np.nan)
    return angles


def build_mswpt_packets(
    wpt: MSWPT,
    spacing: Tuple[float, float],
    levels: Optional[Iterable[int]],
    min_radius: float,
) -> List[MSWPTPacket]:
    angles = mswpt_box_angles_image(wpt.dyadic_decomp, spacing, min_radius)
    packets: List[MSWPTPacket] = []
    num_levels = wpt.dyadic_decomp.num_levels
    box_starts = wpt.boxes_cumsum
    level_list = list(range(num_levels)) if levels is None else list(levels)
    for level in level_list:
        start = int(box_starts[level])
        end = int(box_starts[level + 1])
        for box_idx in range(start, end):
            angle = float(angles[box_idx])
            if np.isnan(angle):
                continue
            coeff_idx = mswpt_coeff_index(wpt, box_idx)
            label = f"L{level}_B{box_idx}"
            packets.append(
                MSWPTPacket(
                    box_idx=box_idx,
                    coeff_idx=coeff_idx,
                    angle_rad=angle,
                    level=level,
                    label=label,
                )
            )
    return packets


def mswpt_packet_p0(wpt: MSWPT, coeff_idx: int) -> np.ndarray:
    coeffs = jnp.zeros((wpt.total_coeffs,), dtype=wpt.complex_dtype)
    coeffs = coeffs.at[coeff_idx].set(1.0)
    p0 = wpt.inverse(coeffs, output_type="spatial").real
    return np.asarray(p0)


def mswpt_box_energy(coeffs: np.ndarray, wpt: MSWPT) -> np.ndarray:
    energies = np.zeros((wpt.dyadic_decomp.total_num_boxes,), dtype=float)
    for level in range(wpt.dyadic_decomp.num_levels):
        c_lo = int(wpt.coeffs_cumsum[level])
        c_hi = int(wpt.coeffs_cumsum[level + 1])
        shape = wpt.coeff_shapes[level]
        coeffs_lvl = np.asarray(coeffs[c_lo:c_hi]).reshape(shape)
        energy = np.sum(np.abs(coeffs_lvl) ** 2, axis=tuple(range(1, coeffs_lvl.ndim)))
        b_lo = int(wpt.boxes_cumsum[level])
        b_hi = int(wpt.boxes_cumsum[level + 1])
        energies[b_lo:b_hi] = energy
    return energies


def analyze_mswpt(
    data: np.ndarray,
    wpt: MSWPT,
    box_angles: np.ndarray,
    spacing: Tuple[float, float],
) -> Tuple[int, float, float, float, float, float]:
    coeffs = wpt.forward(jnp.asarray(data), input_type="spatial")
    energies = mswpt_box_energy(np.asarray(coeffs), wpt)
    box_idx = int(np.argmax(energies))
    center_idx = np.asarray(wpt.dyadic_decomp.centres_ndim[box_idx], dtype=float)
    k_s, omega_over_c, ratio = _data_center_metrics(
        center_idx, wpt.dyadic_decomp.N, spacing
    )
    return (
        box_idx,
        float(box_angles[box_idx]),
        float(energies[box_idx]),
        k_s,
        omega_over_c,
        ratio,
    )


@dataclass(frozen=True)
class WaveletPacket:
    coeff_index: int
    band: str
    level: int
    angle_rad: float
    label: str
    center_idx: Optional[np.ndarray] = None


def wavelet_max_level(shape: Tuple[int, int], wavelet: str) -> int:
    min_dim = int(min(shape))
    return int(pywt.dwt_max_level(min_dim, pywt.Wavelet(wavelet).dec_len))


def wavelet_packet_atom(
    shape: Tuple[int, int],
    wavelet: str,
    mode: str,
    level: int,
    coeff_index: int,
    band: str,
) -> np.ndarray:
    coeffs = pywt.wavedec2(
        np.zeros(shape, dtype=float), wavelet=wavelet, mode=mode, level=level
    )
    coeffs = list(coeffs)
    coeffs[0] = np.zeros_like(coeffs[0])
    for i in range(1, len(coeffs)):
        cH, cV, cD = coeffs[i]
        coeffs[i] = (np.zeros_like(cH), np.zeros_like(cV), np.zeros_like(cD))

    if band == "A":
        arr = coeffs[0]
        center = tuple(int(s // 2) for s in arr.shape)
        arr[center] = 1.0
        coeffs[0] = arr
    else:
        cH, cV, cD = coeffs[coeff_index]
        if band == "H":
            arr = np.zeros_like(cH)
            arr[tuple(int(s // 2) for s in arr.shape)] = 1.0
            coeffs[coeff_index] = (arr, cV, cD)
        elif band == "V":
            arr = np.zeros_like(cV)
            arr[tuple(int(s // 2) for s in arr.shape)] = 1.0
            coeffs[coeff_index] = (cH, arr, cD)
        elif band == "D":
            arr = np.zeros_like(cD)
            arr[tuple(int(s // 2) for s in arr.shape)] = 1.0
            coeffs[coeff_index] = (cH, cV, arr)
        else:
            raise ValueError(f"Unknown wavelet band: {band}")

    atom = pywt.waverec2(coeffs, wavelet=wavelet, mode=mode)
    return np.asarray(atom)


def build_wavelet_packets(
    shape: Tuple[int, int],
    wavelet: str,
    mode: str,
    level: int,
    spacing: Tuple[float, float],
    min_radius: float,
    half_plane_axis: int,
    angle_kind: str,
) -> List[WaveletPacket]:
    if angle_kind not in ("image", "data"):
        raise ValueError("angle_kind must be 'image' or 'data'.")

    def _angle(center_idx: Optional[np.ndarray]) -> float:
        if angle_kind == "image":
            return theta_from_center_image(center_idx, shape, spacing, min_radius)
        return beta_from_center_data(center_idx, shape, spacing, min_radius)

    packets: List[WaveletPacket] = []
    atom = wavelet_packet_atom(shape, wavelet, mode, level, 0, "A")
    center_idx = estimate_frequency_center(atom, half_plane_axis=half_plane_axis)
    angle = _angle(center_idx)
    packets.append(WaveletPacket(0, "A", level, angle, f"L{level}A", center_idx))

    for i in range(1, level + 1):
        scale = level - i + 1
        for band in ("H", "V", "D"):
            atom = wavelet_packet_atom(shape, wavelet, mode, level, i, band)
            center_idx = estimate_frequency_center(
                atom, half_plane_axis=half_plane_axis
            )
            angle = _angle(center_idx)
            label = f"L{scale}{band}"
            packets.append(WaveletPacket(i, band, scale, angle, label, center_idx))
    return packets


def wavelet_band_energies(coeffs: List, level: int) -> Dict[str, float]:
    energies: Dict[str, float] = {}
    energies[f"L{level}A"] = float(np.sum(np.abs(coeffs[0]) ** 2))
    for i in range(1, level + 1):
        scale = level - i + 1
        cH, cV, cD = coeffs[i]
        energies[f"L{scale}H"] = float(np.sum(np.abs(cH) ** 2))
        energies[f"L{scale}V"] = float(np.sum(np.abs(cV) ** 2))
        energies[f"L{scale}D"] = float(np.sum(np.abs(cD) ** 2))
    return energies


def analyze_wavelet(
    data: np.ndarray,
    wavelet: str,
    mode: str,
    level: int,
    spacing: Tuple[float, float],
    band_packets: Dict[str, WaveletPacket],
) -> Tuple[str, float, float, float, float, float]:
    coeffs = pywt.wavedec2(data, wavelet=wavelet, mode=mode, level=level)
    energies = wavelet_band_energies(coeffs, level)
    band = max(energies, key=energies.get)
    packet = band_packets[band]
    k_s, omega_over_c, ratio = _data_center_metrics(
        packet.center_idx, data.shape, spacing
    )
    return (
        band,
        float(packet.angle_rad),
        float(energies[band]),
        k_s,
        omega_over_c,
        ratio,
    )


@dataclass(frozen=True)
class CurveletPacket:
    coeff_idx: int
    scale_idx: int
    angle_rad: float
    label: str


def _fdct2d_kwargs(num_scales: Optional[int], num_angles_coarse: Optional[int]) -> dict:
    kwargs = {}
    try:
        sig = getattr(cl, "FDCT2D", None)
        if sig is None:
            return kwargs
        sig = __import__("inspect").signature(sig)
    except Exception:
        return kwargs

    if num_scales is not None and "nbscales" in sig.parameters:
        kwargs["nbscales"] = int(num_scales)
    if num_angles_coarse is not None and "nbangles_coarse" in sig.parameters:
        kwargs["nbangles_coarse"] = int(num_angles_coarse)
    return kwargs


def curvelet_partition(F, coeff_len: int) -> List[int]:
    part = getattr(F, "partition", None)
    if part is not None:
        return [int(x) for x in part]
    base = coeff_len // 3
    return [base, base, coeff_len - 2 * base]


def curvelet_packets(
    F,
    shape: Tuple[int, int],
    spacing: Tuple[float, float],
    samples_per_scale: int,
    angle_tol_deg: float,
    min_radius: float,
) -> List[CurveletPacket]:
    coeff_probe = F @ np.zeros(shape)
    coeff_len = int(np.asarray(coeff_probe).size)
    partition = curvelet_partition(F, coeff_len)
    center = np.array([(shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0])

    packets: List[CurveletPacket] = []
    for scale_idx, length in enumerate(partition):
        start = int(sum(partition[:scale_idx]))
        length = int(length)
        if length <= 0:
            continue
        num = min(samples_per_scale, length)
        offsets = np.linspace(0, max(0, length - 1), num=num, dtype=int)
        candidates = []
        for off in offsets:
            idx = start + int(off)
            coeffs = np.zeros(coeff_len, dtype=complex)
            coeffs[idx] = 1.0
            atom = np.asarray(F.H @ coeffs).real
            center_idx = estimate_frequency_center(
                atom, half_plane_axis=IMAGE_HALF_PLANE_AXIS
            )
            angle = theta_from_center_image(center_idx, shape, spacing, min_radius)
            if np.isnan(angle):
                continue
            w = np.abs(atom)
            total = w.sum()
            if total > 0:
                grid = np.indices(atom.shape)
                centroid = np.array(
                    [(w * grid[0]).sum() / total, (w * grid[1]).sum() / total]
                )
                dist = float(np.linalg.norm(centroid - center))
            else:
                dist = float("inf")
            candidates.append((angle, dist, idx))

        bins: Dict[int, Tuple[float, float, int]] = {}
        for angle, dist, idx in candidates:
            key = int(np.round(np.degrees(angle) / angle_tol_deg))
            prev = bins.get(key)
            if prev is None or dist < prev[1]:
                bins[key] = (angle, dist, idx)

        for angle, _, idx in bins.values():
            label = f"S{scale_idx}_i{idx}"
            packets.append(
                CurveletPacket(
                    coeff_idx=int(idx),
                    scale_idx=int(scale_idx),
                    angle_rad=float(angle),
                    label=label,
                )
            )
    packets.sort(key=lambda p: (p.scale_idx, p.angle_rad))
    return packets


def curvelet_packet_p0(F, coeff_idx: int) -> np.ndarray:
    coeffs = np.zeros(F.shape[0], dtype=complex)
    coeffs[int(coeff_idx)] = 1.0
    return np.asarray(F.H @ coeffs).real


def analyze_curvelet(
    data: np.ndarray,
    F,
    spacing: Tuple[float, float],
    angle_cache: Dict[int, float],
    center_cache: Dict[int, Optional[np.ndarray]],
) -> Tuple[int, float, float, float, float, float]:
    coeffs = np.asarray(F @ data)
    idx = int(np.argmax(np.abs(coeffs)))
    if idx not in angle_cache:
        coeffs_unit = np.zeros_like(coeffs)
        coeffs_unit[idx] = 1.0
        atom = np.asarray(F.H @ coeffs_unit).real
        center_idx = estimate_frequency_center(
            atom, half_plane_axis=DATA_HALF_PLANE_AXIS
        )
        angle_cache[idx] = beta_from_center_data(
            center_idx, data.shape, spacing, MIN_CENTER_RADIUS
        )
        center_cache[idx] = center_idx
    center_idx = center_cache.get(idx)
    k_s, omega_over_c, ratio = _data_center_metrics(center_idx, data.shape, spacing)
    return (
        idx,
        float(angle_cache[idx]),
        float(np.abs(coeffs[idx]) ** 2),
        k_s,
        omega_over_c,
        ratio,
    )


def save_results_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_mapping(
    results: List[Dict],
    outpath: Path,
    *,
    energy_keep_fraction: Optional[float],
    abs_angles: bool,
) -> None:
    if not results:
        return
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    colors = {
        "mswpt": "#1f77b4",
        "curvelet": "#ff7f0e",
        "wavelet": "#2ca02c",
    }
    for name in sorted({r["transform"] for r in results}):
        pts = [r for r in results if r["transform"] == name]
        if energy_keep_fraction is not None and pts:
            energies = np.array([p["output_energy"] for p in pts], dtype=float)
            thresh = np.quantile(energies, 1.0 - float(energy_keep_fraction))
            pts = [p for p in pts if p["output_energy"] >= thresh]
        x = [r["input_angle_deg"] for r in pts]
        y = [r["output_angle_deg"] for r in pts]
        if abs_angles:
            x = np.abs(x)
            y = np.abs(y)
        ax.scatter(x, y, s=20, alpha=0.7, color=colors.get(name, "gray"), label=name)

    theta_min, theta_max = (-90.0, 90.0) if not abs_angles else (0.0, 90.0)
    theta_deg = np.linspace(theta_min, theta_max, 721)
    theta_rad = np.deg2rad(theta_deg)
    beta_rad = np.arctan(np.sin(theta_rad))
    beta_deg = np.rad2deg(beta_rad)
    ax.plot(
        theta_deg,
        beta_deg,
        "k--",
        lw=1.2,
        alpha=0.7,
        label=r"$\beta=\arctan(\sin\theta)$",
    )

    ax.set_xlim(theta_min, theta_max)
    ax.set_ylim((-45.0, 45.0) if not abs_angles else (0.0, 45.0))
    ax.set_xlabel(("|θ|" if abs_angles else "θ") + " (deg)")
    ax.set_ylabel(("|β|" if abs_angles else "β") + " (deg)")
    ax.set_title("Angle mapping: input vs output")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)


def main() -> None:
    if "curvelet" in RUN_TRANSFORMS and cl is None:
        print("curvelops not installed; skipping curvelet analysis.")
    if "wavelet" in RUN_TRANSFORMS and pywt is None:
        print("PyWavelets not installed; skipping wavelet analysis.")

    domain = geometry.Domain(N=N, dx=DX, periodic=PERIODIC, cfl=CFL, c=c_hom)
    ts = np.asarray(domain.generate_time_domain())
    dt = float(ts[1] - ts[0])

    binary_mask = make_line_sensor_mask(N, axis=SENSOR_AXIS, index=SENSOR_INDEX)

    simulation_options = SimulationOptions(
        data_cast="double",
        smooth_p0=False,
        save_to_disk=True,
    )
    execution_options = SimulationExecutionOptions(
        is_gpu_simulation=False,
        delete_data=False,
        verbose_level=0,
        show_sim_log=False,
    )
    kwave_solver = KWaveSolver(simulation_options, execution_options)

    Ns = int(np.sum(np.asarray(binary_mask)))
    s_slice, t_slice = compute_even_slices(Ns, len(ts))
    data_shape = (s_slice.stop - s_slice.start, t_slice.stop - t_slice.start)

    spacing_data = data_axis_spacing(DX, dt, C0)

    dyadic_img = DyadicDecomposition(NUM_LEVELS, N, NUM_BOXES_LEVELS, BOX_ASPECT_RATIO)
    wpt_img = MSWPT(dyadic_img, REDUNDANCY, WINDOWING)

    base_boxes = NUM_BOXES_LEVELS[0]
    data_levels = safe_num_levels(NUM_LEVELS, min(data_shape), base_boxes)
    data_boxes_levels = tuple(2 ** (i + 2) for i in range(data_levels))
    dyadic_data = DyadicDecomposition(
        data_levels, data_shape, data_boxes_levels, BOX_ASPECT_RATIO
    )
    wpt_data = MSWPT(dyadic_data, REDUNDANCY, WINDOWING)
    mswpt_data_angles = mswpt_box_angles_data(
        dyadic_data, spacing_data, MIN_CENTER_RADIUS
    )

    wavelet_level_img = None
    wavelet_level_data = None
    wavelet_band_packets_data: Dict[str, WaveletPacket] = {}
    if pywt is not None and "wavelet" in RUN_TRANSFORMS:
        wavelet_level_img = min(WAVELET_LEVEL, wavelet_max_level(N, WAVELET_NAME))
        wavelet_level_data = min(
            WAVELET_LEVEL, wavelet_max_level(data_shape, WAVELET_NAME)
        )
        wavelet_packets = build_wavelet_packets(
            N,
            WAVELET_NAME,
            WAVELET_MODE,
            wavelet_level_img,
            DX,
            MIN_CENTER_RADIUS,
            IMAGE_HALF_PLANE_AXIS,
            angle_kind="image",
        )
        wavelet_packets_data = build_wavelet_packets(
            data_shape,
            WAVELET_NAME,
            WAVELET_MODE,
            wavelet_level_data,
            spacing_data,
            MIN_CENTER_RADIUS,
            DATA_HALF_PLANE_AXIS,
            angle_kind="data",
        )
        wavelet_band_packets_data = {p.label: p for p in wavelet_packets_data}
    else:
        wavelet_packets = []

    curvelet_packets_img: List[CurveletPacket] = []
    F_curvelet_img = None
    F_curvelet_data = None
    curvelet_angle_cache: Dict[int, float] = {}
    curvelet_center_cache: Dict[int, Optional[np.ndarray]] = {}
    if cl is not None and "curvelet" in RUN_TRANSFORMS:
        kwargs = _fdct2d_kwargs(CURVELET_NUM_SCALES, CURVELET_ANGLES_COARSE)
        F_curvelet_img = cl.FDCT2D(dims=N, **kwargs)
        F_curvelet_data = cl.FDCT2D(dims=data_shape, **kwargs)
        curvelet_packets_img = curvelet_packets(
            F_curvelet_img,
            N,
            DX,
            CURVELET_SAMPLES_PER_SCALE,
            CURVELET_ANGLE_TOL_DEG,
            MIN_CENTER_RADIUS,
        )

    mswpt_packets = build_mswpt_packets(
        wpt_img, DX, levels=None, min_radius=MIN_CENTER_RADIUS
    )

    packets_by_transform: Dict[str, List] = {
        "mswpt": mswpt_packets,
        "curvelet": curvelet_packets_img,
        "wavelet": wavelet_packets,
    }

    results: List[Dict] = []

    for transform in RUN_TRANSFORMS:
        if transform == "curvelet" and F_curvelet_img is None:
            continue
        if transform == "wavelet" and pywt is None:
            continue

        packets = packets_by_transform.get(transform, [])
        if MAX_PACKETS is not None:
            packets = packets[: int(MAX_PACKETS)]

        print(f"\n[{transform}] packets: {len(packets)}")

        for i, packet in enumerate(packets, start=1):
            if transform == "mswpt":
                p0 = mswpt_packet_p0(wpt_img, packet.coeff_idx)
                input_angle = packet.angle_rad
                input_label = packet.label
                input_box = packet.box_idx
            elif transform == "wavelet":
                p0 = wavelet_packet_atom(
                    N,
                    WAVELET_NAME,
                    WAVELET_MODE,
                    wavelet_level_img,
                    packet.coeff_index,
                    packet.band,
                )
                input_angle = packet.angle_rad
                input_label = packet.label
                input_box = packet.label
            else:
                p0 = curvelet_packet_p0(F_curvelet_img, packet.coeff_idx)
                input_angle = packet.angle_rad
                input_label = packet.label
                input_box = packet.coeff_idx

            p0 = normalize_max(p0)

            sensor_data = kwave_solver.forward(p0, domain, binary_mask, ts)
            data = preprocess_sensor_data(sensor_data, ts, s_slice, t_slice)

            if transform == "mswpt":
                (
                    out_box,
                    out_angle,
                    out_energy,
                    out_k_s,
                    out_omega_over_c,
                    out_ratio,
                ) = analyze_mswpt(data, wpt_data, mswpt_data_angles, spacing_data)
                output_box = out_box
            elif transform == "wavelet":
                (
                    out_box,
                    out_angle,
                    out_energy,
                    out_k_s,
                    out_omega_over_c,
                    out_ratio,
                ) = analyze_wavelet(
                    data,
                    WAVELET_NAME,
                    WAVELET_MODE,
                    wavelet_level_data,
                    spacing_data,
                    wavelet_band_packets_data,
                )
                output_box = out_box
            else:
                (
                    out_box,
                    out_angle,
                    out_energy,
                    out_k_s,
                    out_omega_over_c,
                    out_ratio,
                ) = analyze_curvelet(
                    data,
                    F_curvelet_data,
                    spacing_data,
                    curvelet_angle_cache,
                    curvelet_center_cache,
                )
                output_box = out_box

            results.append(
                {
                    "transform": transform,
                    "packet_label": input_label,
                    "input_box": input_box,
                    "output_box": output_box,
                    "input_angle_deg": np.degrees(input_angle)
                    if not np.isnan(input_angle)
                    else np.nan,
                    "output_angle_deg": np.degrees(out_angle)
                    if not np.isnan(out_angle)
                    else np.nan,
                    "output_energy": out_energy,
                    "output_k_s": out_k_s,
                    "output_omega_over_c": out_omega_over_c,
                    "output_ratio": out_ratio,
                }
            )

            if i % max(1, min(10, len(packets))) == 0:
                print(f"  {i}/{len(packets)} done")

    out_csv = DATA_DIR / "angle_mapping_results.csv"
    save_results_csv(results, out_csv)
    print(f"\nSaved results to {out_csv}")

    out_plot = PLOT_DIR / "angle_mapping_scatter.png"
    plot_mapping(
        results,
        out_plot,
        energy_keep_fraction=ENERGY_KEEP_FRACTION,
        abs_angles=False,
    )
    print(f"Saved plot to {out_plot}")

    if PLOT_ABS_ANGLES:
        out_plot_abs = PLOT_DIR / "angle_mapping_scatter_abs.png"
        plot_mapping(
            results,
            out_plot_abs,
            energy_keep_fraction=ENERGY_KEEP_FRACTION,
            abs_angles=True,
        )
        print(f"Saved plot to {out_plot_abs}")


if __name__ == "__main__":
    main()
