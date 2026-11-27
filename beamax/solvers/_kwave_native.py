"""Prepare pressure sources and read pressure records for native k-Wave solvers."""

from contextlib import nullcontext
from copy import copy
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory, mkdtemp

import h5py
from kwave.kspaceFirstOrder import kspaceFirstOrder
import numpy as np


NATIVE_PRESSURE_RECORDS = (
    "p",
    "p_final",
    "p_min",
    "p_max",
    "p_rms",
    "p_min_all",
    "p_max_all",
)


def _dataset(file: h5py.File, name: str) -> h5py.Dataset:
    dataset = file.get(name)
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"Expected a k-Wave HDF5 dataset named {name!r}.")
    return dataset


def _pressure_source(kgrid, medium, source):
    """Return a physical source for the writer and native split-density samples."""
    shape = tuple(int(n) for n in kgrid.N)
    mask = np.asarray(source.p_mask, dtype=bool)
    points = np.nonzero(mask)  # Beamax waveforms follow C-order mask points.
    signals = np.atleast_2d(np.asarray(source.p))
    if (
        mask.shape != shape
        or not points[0].size
        or signals.ndim != 2
        or signals.shape[0] not in (1, points[0].size)
        or not signals.shape[1]
    ):
        raise ValueError("Expected one pressure waveform or one per source mask point.")

    speeds = np.broadcast_to(medium.sound_speed, shape)[points].astype(float)
    if signals.shape[0] == 1 and np.all(speeds == speeds[0]):
        speeds = speeds[:1]  # Keep the native shared-waveform representation.
    ndim = len(shape)
    if source.p_mode == "dirichlet":
        scale = 1 / (ndim * speeds**2)
    elif source.p_mode == "additive-no-correction":
        spacing = np.asarray(kgrid.spacing)
        if not np.allclose(spacing, spacing[0], rtol=1e-12, atol=0):
            raise ValueError(
                "Native additive pressure sources require isotropic spacing."
            )
        scale = 2 * kgrid.dt / (ndim * speeds * spacing[0])
    else:
        raise ValueError(f"Unsupported inverse pressure source mode {source.p_mode!r}.")

    physical = copy(source)
    physical.p = np.broadcast_to(signals, (scale.size, signals.shape[1]))
    density = np.asarray(physical.p * scale[:, None], dtype=np.float32)
    if density.shape[0] > 1:
        native_order = np.argsort(np.ravel_multi_index(points, shape, order="F"))
        density = density[native_order]
    return physical, density


def sensor_channels_to_c_order(mask):
    """Return the native channel indices in NumPy mask order."""
    mask = np.asarray(mask, dtype=bool)
    c_points_in_f = np.ravel_multi_index(np.nonzero(mask), mask.shape, order="F")
    native_indices = np.flatnonzero(mask.ravel(order="F"))
    return np.searchsorted(native_indices, c_points_in_f)


def _read_pressure(outputs, record, shape, pml, mask, nt):
    """Read native (z, y, x) fields or (1, time, sensor) samples in Beamax order."""
    field = _dataset(outputs, record)
    if record in ("p_final", "p_min_all", "p_max_all"):
        padded = tuple(n + 2 * p for n, p in zip(shape, pml, strict=True))
        native_shape = ((1,) if len(shape) == 2 else ()) + padded[::-1]
        if field.shape != native_shape:
            raise ValueError(
                f"Expected native {record} shape {native_shape}, got {field.shape}."
            )
        slices = tuple(slice(p, p + n) for n, p in zip(shape, pml, strict=True))
        selection = ((0,) if len(shape) == 2 else ()) + slices[::-1]
        return np.asarray(field[selection]).T

    to_c_order = sensor_channels_to_c_order(mask)
    expected = (1, nt if record == "p" else 1, len(to_c_order))
    if field.shape != expected:
        raise ValueError(
            f"Expected native {record} shape {expected}, got {field.shape}."
        )
    if record == "p":
        return np.asarray(field[0]).T[to_c_order, :]
    return np.asarray(field[0, 0])[to_c_order]


def run_native_simulation(kgrid, medium, source, sensor, *, binary_path, **options):
    """Run an OMP, CUDA, or Metal binary and return the requested pressure record.

    An explicit data_path retains a unique run directory with both HDF5 files.
    """
    records = sensor.record or ["p"]
    if len(records) != 1 or records[0] not in NATIVE_PRESSURE_RECORDS:
        raise ValueError(
            f"Unsupported native k-Wave record {records!r}. Choose one of "
            f"{', '.join(NATIVE_PRESSURE_RECORDS)} or use backend='python'."
        )
    record = records[0]
    density = None
    if source.p is not None:
        source, density = _pressure_source(kgrid, medium, source)
    sensor = copy(sensor)
    sensor.record = None  # Pass the recording options directly to the executable.
    prefix = "beamax-forward-" if density is None else "beamax-inverse-"
    root = options.pop("data_path", None)
    if root is None:
        work = TemporaryDirectory(prefix=prefix)
    else:
        root = Path(root).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        work = nullcontext(mkdtemp(prefix=prefix, dir=root))

    with work as directory:
        options.update(backend="cpp", save_only=True, data_path=directory)
        paths = kspaceFirstOrder(kgrid, medium, source, sensor, **options)
        input_path = paths["input_file"]
        output_path = paths["output_file"]
        if density is not None:
            with h5py.File(input_path, "r+") as inputs:
                # The writer omits an all-zero source. For active sources, supply the
                # binary's density units and native row order regardless of writer version.
                if "p_source_input" in inputs or np.any(density):
                    dataset = _dataset(inputs, "p_source_input")
                    if dataset.shape != (1, *density.T.shape):
                        raise ValueError(
                            f"Unexpected p_source_input shape {dataset.shape}."
                        )
                    dataset[0] = density.T
                    _dataset(inputs, "p_source_many")[...] = int(density.shape[0] > 1)

        command = [
            str(Path(binary_path).resolve()),
            "-i",
            input_path,
            "-o",
            output_path,
            "--p_raw" if record == "p" else f"--{record}",
        ]
        if options.get("num_threads") is not None:
            command.extend(["-t", str(options["num_threads"])])
        if options.get("device_num") is not None:
            command.extend(["-g", str(options["device_num"])])
        quiet = options.get("quiet", False)
        if quiet or options.get("debug"):
            command.extend(["--verbose", "0" if quiet else "2"])
        subprocess.run(command, check=True, capture_output=quiet, text=True)

        shape = tuple(int(n) for n in kgrid.N)
        pml = paths.get("pml_size", (0,) * len(shape))
        with h5py.File(output_path, "r") as outputs:
            return {
                record: _read_pressure(
                    outputs, record, shape, pml, sensor.mask, kgrid.Nt
                )
            }
