"""Device capability and memory planning for MSGB solves."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Literal, Optional, Sequence, Tuple, cast

import jax
import numpy as np

__all__ = [
    "DeviceCapabilities",
    "MemoryEstimate",
    "device_capabilities",
    "estimate_msgb_memory",
]

Operation = Literal["forward", "time_reversal", "adjoint"]
Selection = Literal["streaming_top_n", "materialized"]
ForwardKernel = Literal["xla", "trajectory_pallas", "hom_diag_3d_pallas"]


def _format_bytes(count: float) -> str:
    count = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if count < 1024.0 or unit == "TiB":
            return f"{count:,.1f} {unit}" if unit != "B" else f"{count:,.0f} B"
        count /= 1024.0
    raise AssertionError("unreachable")


def _pallas_backend_available(name: str) -> bool:
    try:
        __import__(f"jax.experimental.pallas.{name}")
    except Exception:
        return False
    return True


@dataclass(frozen=True)
class DeviceCapabilities:
    """Snapshot of the local JAX environment relevant to MSGB planning."""

    backend: str
    device_kind: str
    device_count: int
    memory_limit_bytes: Optional[int]
    x64_enabled: bool
    pallas_triton_available: bool
    pallas_tpu_available: bool
    jax_version: str

    def render(self) -> str:
        budget = (
            _format_bytes(self.memory_limit_bytes)
            if self.memory_limit_bytes is not None
            else "unknown"
        )
        lines = [
            "Device capabilities",
            f"  backend ................. {self.backend}",
            f"  device kind ............. {self.device_kind}",
            f"  visible devices ......... {self.device_count}",
            f"  device memory budget .... {budget}",
            f"  jax version ............. {self.jax_version}",
            f"  x64 enabled ............. {self.x64_enabled}",
            f"  Pallas Triton module .... {self.pallas_triton_available}",
            f"  Pallas Mosaic TPU module . {self.pallas_tpu_available}",
        ]
        return "\n".join(lines)


def device_capabilities() -> DeviceCapabilities:
    """Inspect JAX and read the default device's memory limit when available."""
    devices = jax.devices()
    default = devices[0]
    stats: dict[Any, Any] = {}
    memory_stats = getattr(default, "memory_stats", None)
    if callable(memory_stats):
        try:
            stats = cast(dict[Any, Any], memory_stats() or {})
        except Exception:
            stats = {}
    limit = stats.get("bytes_limit", stats.get(b"bytes_limit"))
    return DeviceCapabilities(
        backend=jax.default_backend(),
        device_kind=str(getattr(default, "device_kind", default.platform)),
        device_count=len(devices),
        memory_limit_bytes=int(limit) if limit else None,
        x64_enabled=bool(getattr(jax.config, "x64_enabled", False)),
        pallas_triton_available=_pallas_backend_available("triton"),
        pallas_tpu_available=_pallas_backend_available("tpu"),
        jax_version=jax.__version__,
    )


@dataclass(frozen=True)
class MemoryEstimate:
    """Itemized lower bound, planning margin, and budget verdict."""

    operation: Operation
    policy: str
    config_summary: str
    items: Tuple[Tuple[str, int], ...]
    lower_bound_bytes: int
    safety_factor: float
    budget_bytes: Optional[int]
    notes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def planning_bytes(self) -> int:
        r"""Return $\lceil sL\rceil$ for safety factor $s$ and lower bound $L$."""
        return math.ceil(self.safety_factor * self.lower_bound_bytes)

    @property
    def verdict(self) -> str:
        if self.budget_bytes is None:
            return "no device budget known — pass budget_bytes for a verdict"
        fraction = self.planning_bytes / self.budget_bytes
        if fraction < 0.7:
            return f"fits comfortably (~{fraction:.0%} of budget)"
        if fraction < 0.95:
            return f"tight (~{fraction:.0%} of budget) — leave headroom"
        return f"unlikely to fit (~{fraction:.0%} of budget)"

    def render(self) -> str:
        width = max(len(label) for label, _ in self.items) + 2
        lines = [
            f"MSGB memory plan — {self.operation}",
            "=" * (20 + len(self.operation)),
            self.config_summary,
            f"policy: {self.policy}",
            "",
            "Dominant allocations (lower bounds)",
        ]
        for label, size in self.items:
            lines.append(
                f"  {label} ".ljust(width + 2, ".") + f" {_format_bytes(size):>11}"
            )
        lines.append(
            "  total ".ljust(width + 2, ".")
            + f" {_format_bytes(self.lower_bound_bytes):>11}"
        )
        lines.append("")
        lines.append(f"Planning estimate ({self.safety_factor:g}x lower bound)")
        lines.append(
            "  estimate ".ljust(width + 2, ".")
            + f" {_format_bytes(self.planning_bytes):>11}"
        )
        lines.append("")
        budget = (
            _format_bytes(self.budget_bytes)
            if self.budget_bytes is not None
            else "unknown"
        )
        lines.append(f"Verdict vs {budget} budget: {self.verdict}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def _dominant_items(
    operation: Operation,
    *,
    grid_shape: Sequence[int],
    top_n: int,
    batch_size: int,
    num_times: int,
    num_detectors: int,
    data_shape: Optional[Sequence[int]],
    selection: Selection,
    forward_kernel: ForwardKernel,
    redundancy: int,
    total_coeffs: Optional[int],
    float_bytes: int,
    complex_bytes: int,
) -> list[tuple[str, int]]:
    """Itemize dominant allocation lower bounds, excluding XLA temporaries."""
    ndim = len(grid_shape)
    n_grid = math.prod(grid_shape)
    items: list[tuple[str, int]] = []

    if operation == "forward":
        selection_cells = n_grid
    else:
        if data_shape is None:
            raise ValueError(
                "time_reversal/adjoint estimates need data_shape "
                "(the acquired (Nt, *detector) record)."
            )
        selection_cells = math.prod(data_shape)

    coeff_count = (
        total_coeffs
        if total_coeffs is not None
        else (redundancy**ndim) * selection_cells
    )
    if selection == "materialized":
        items.append(("materialized coefficient vector", coeff_count * complex_bytes))
        items.append(("transform FFT workspace", selection_cells * complex_bytes))
    else:
        items.append(("streaming FFT workspace", selection_cells * complex_bytes))
        items.append(("top-n selection state", top_n * 3 * complex_bytes))

    trajectory_row = (
        2 * ndim * float_bytes + ndim * ndim * complex_bytes + complex_bytes
    )

    if operation == "forward":
        items.append(("initial pressure field", n_grid * float_bytes))
        record = num_times * num_detectors * float_bytes
        items.append(("sensor record (output)", record))
        items.append(("field accumulator + batch contribution", 2 * record))
        if forward_kernel != "hom_diag_3d_pallas":
            items.append(
                (
                    "per-batch trajectories",
                    batch_size * num_times * trajectory_row,
                )
            )
    else:
        items.append(
            ("acquired data record", math.prod(data_shape or ()) * float_bytes)
        )
        if operation == "adjoint":
            items.append(
                (
                    "adjoint source formation (complex FFT)",
                    2 * math.prod(data_shape or ()) * complex_bytes,
                )
            )
        items.append(("image output", n_grid * float_bytes))
        # TR/adjoint beams carry two ODE endpoints, not a long trajectory.
        items.append(
            ("per-batch endpoint trajectories", batch_size * 2 * trajectory_row)
        )
        items.append(("terminal accumulator + batch field", 2 * n_grid * float_bytes))
    return items


def _integer(name: str, value: Any, *, positive: bool) -> int:
    """Validate a host-side integer used by the static planner."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer; got {value!r}.")
    result = int(value)
    if (positive and result <= 0) or (not positive and result < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer; got {value!r}.")
    return result


def _shape(name: str, value: Sequence[int]) -> tuple[int, ...]:
    """Validate a non-empty shape containing positive integer dimensions."""
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a non-empty sequence of positive integers.")
    try:
        dimensions = tuple(value)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be a non-empty sequence of positive integers."
        ) from exc
    if not dimensions:
        raise ValueError(f"{name} must contain at least one dimension.")
    return tuple(
        _integer(f"{name}[{axis}]", dimension, positive=True)
        for axis, dimension in enumerate(dimensions)
    )


def _real_dtype(dtype: Any | None) -> np.dtype[Any]:
    """Resolve the planner's real dtype, following JAX's x64 mode by default."""
    if dtype is None:
        dtype = np.float64 if jax.config.x64_enabled else np.float32
    try:
        resolved = np.dtype(dtype)
    except TypeError as exc:
        raise ValueError(
            f"real_dtype must be float32 or float64; got {dtype!r}."
        ) from exc
    if resolved not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"real_dtype must be float32 or float64; got {dtype!r}.")
    return resolved


def estimate_msgb_memory(
    operation: Operation,
    *,
    grid_shape: Sequence[int],
    top_n: int,
    batch_size: int,
    num_times: int = 0,
    num_detectors: int = 0,
    data_shape: Optional[Sequence[int]] = None,
    selection: Selection = "streaming_top_n",
    forward_kernel: ForwardKernel = "xla",
    redundancy: int = 2,
    total_coeffs: Optional[int] = None,
    real_dtype: Any | None = None,
    safety_factor: float = 2.0,
    budget_bytes: Optional[int] = None,
    use_device_budget: bool = True,
) -> MemoryEstimate:
    r"""Estimate device memory needs for one MSGB operation.

    Pass ``total_coeffs`` for an exact materialized-vector size. ``real_dtype``
    defaults to float64 when JAX x64 is enabled and float32 otherwise. The
    planning estimate is $sL$, where $L$ is the allocation lower bound and $s$
    is the explicit ``safety_factor``. The fused homogeneous 3-D Pallas kernel
    requires float32. This estimate is not a measured allocator peak.
    """
    if operation not in ("forward", "time_reversal", "adjoint"):
        raise ValueError(f"Unknown operation {operation!r}.")
    if selection not in ("streaming_top_n", "materialized"):
        raise ValueError(f"Unknown coefficient selection {selection!r}.")
    if forward_kernel not in ("xla", "trajectory_pallas", "hom_diag_3d_pallas"):
        raise ValueError(f"Unknown forward kernel {forward_kernel!r}.")
    grid_shape = _shape("grid_shape", grid_shape)
    if data_shape is not None:
        data_shape = _shape("data_shape", data_shape)
    top_n = _integer("top_n", top_n, positive=True)
    batch_size = _integer("batch_size", batch_size, positive=True)
    num_times = _integer("num_times", num_times, positive=False)
    num_detectors = _integer("num_detectors", num_detectors, positive=False)
    redundancy = _integer("redundancy", redundancy, positive=True)
    if total_coeffs is not None:
        total_coeffs = _integer("total_coeffs", total_coeffs, positive=True)
    if budget_bytes is not None:
        budget_bytes = _integer("budget_bytes", budget_bytes, positive=True)
    if not isinstance(use_device_budget, (bool, np.bool_)):
        raise ValueError(
            f"use_device_budget must be boolean; got {use_device_budget!r}."
        )
    if (
        isinstance(safety_factor, (bool, np.bool_))
        or not isinstance(safety_factor, Real)
        or not math.isfinite(float(safety_factor))
        or safety_factor < 1
    ):
        raise ValueError("safety_factor must be finite and at least 1.")
    if operation == "forward" and (num_times <= 0 or num_detectors <= 0):
        raise ValueError("forward estimates need num_times and num_detectors.")
    dtype = _real_dtype(real_dtype)
    if (
        operation == "forward"
        and forward_kernel == "hom_diag_3d_pallas"
        and dtype != np.dtype(np.float32)
    ):
        raise ValueError(
            "forward_kernel='hom_diag_3d_pallas' requires real_dtype=float32."
        )
    float_bytes = dtype.itemsize
    complex_bytes = 2 * float_bytes

    items = _dominant_items(
        operation,
        grid_shape=grid_shape,
        top_n=top_n,
        batch_size=batch_size,
        num_times=num_times,
        num_detectors=num_detectors,
        data_shape=data_shape,
        selection=selection,
        forward_kernel=forward_kernel,
        redundancy=redundancy,
        total_coeffs=total_coeffs,
        float_bytes=float_bytes,
        complex_bytes=complex_bytes,
    )
    lower_bound = sum(size for _, size in items)

    if budget_bytes is None and use_device_budget:
        budget_bytes = device_capabilities().memory_limit_bytes
        if budget_bytes is not None:
            budget_bytes = _integer("device budget", budget_bytes, positive=True)

    notes: list[str] = []
    if selection == "materialized":
        notes.append(
            "selection='streaming_top_n' removes the materialized coefficient "
            "vector (the default for fixed top_n)"
        )
    notes.append("halving batch_size halves the per-batch terms")
    notes.append(
        "the planning margin is not a measured peak; benchmark the target "
        "device for capacity decisions"
    )

    ndim_summary = "x".join(str(v) for v in grid_shape)
    data_summary = (
        " · data " + "x".join(str(v) for v in data_shape) if data_shape else ""
    )
    detector_summary = (
        f" · {num_times} times · {num_detectors} detectors"
        if operation == "forward"
        else ""
    )
    policy = f"{selection} selection · " + (
        f"{forward_kernel} forward kernel"
        if operation == "forward"
        else "terminal inverse evaluation"
    )
    return MemoryEstimate(
        operation=operation,
        policy=policy,
        config_summary=(
            f"grid {ndim_summary}{data_summary} · {dtype.name} · top_n {top_n} · "
            f"batch {batch_size}{detector_summary}"
        ),
        items=tuple(items),
        lower_bound_bytes=lower_bound,
        safety_factor=float(safety_factor),
        budget_bytes=budget_bytes,
        notes=tuple(notes),
    )
