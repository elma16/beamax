"""Static configuration for Beamax Pallas kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


RoundingMode = Literal["nearest_even", "half_open"]
TPULayout = Literal["beam_sensor", "time_sensor"]
TPUGridOrder = Literal["time_sensor", "sensor_time"]


@dataclass(frozen=True)
class PallasConfig:
    """Compilation-static launch and wrapping options."""

    sensor_block_size: int = 128
    gpu_num_warps: int | None = 4
    gpu_num_stages: int | None = 2
    rounding_mode: RoundingMode = "nearest_even"
    periodic_axes: tuple[bool, ...] | None = None
    tpu_layout: TPULayout = "beam_sensor"
    tpu_beam_block_size: int = 8
    tpu_time_block_size: int = 8
    tpu_grid_order: TPUGridOrder = "time_sensor"
    gpu_time_block_size: int = 1

    def __post_init__(self) -> None:
        integer_fields = {
            "sensor_block_size": self.sensor_block_size,
            "gpu_time_block_size": self.gpu_time_block_size,
            "tpu_beam_block_size": self.tpu_beam_block_size,
            "tpu_time_block_size": self.tpu_time_block_size,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer; got {value!r}.")

        if self.gpu_time_block_size not in {1, 2, 4, 8}:
            raise ValueError("gpu_time_block_size must be one of 1, 2, 4, or 8.")
        if self.gpu_num_warps is not None and (
            isinstance(self.gpu_num_warps, bool)
            or self.gpu_num_warps not in {1, 2, 4, 8}
        ):
            raise ValueError("gpu_num_warps must be one of 1, 2, 4, 8, or None.")
        if self.gpu_num_stages is not None and (
            isinstance(self.gpu_num_stages, bool)
            or not isinstance(self.gpu_num_stages, int)
            or self.gpu_num_stages <= 0
        ):
            raise ValueError("gpu_num_stages must be a positive integer or None.")
        if self.rounding_mode not in {"nearest_even", "half_open"}:
            raise ValueError(
                "rounding_mode must be 'nearest_even' or 'half_open'; got "
                f"{self.rounding_mode!r}."
            )
        if self.tpu_layout not in {"beam_sensor", "time_sensor"}:
            raise ValueError(
                "tpu_layout must be 'beam_sensor' or 'time_sensor'; got "
                f"{self.tpu_layout!r}."
            )
        if self.tpu_grid_order not in {"time_sensor", "sensor_time"}:
            raise ValueError(
                "tpu_grid_order must be 'time_sensor' or 'sensor_time'; got "
                f"{self.tpu_grid_order!r}."
            )
        if self.periodic_axes is not None and (
            not isinstance(self.periodic_axes, tuple)
            or not all(isinstance(value, bool) for value in self.periodic_axes)
        ):
            raise ValueError("periodic_axes must be a tuple of bool values or None.")


__all__ = ["PallasConfig"]
