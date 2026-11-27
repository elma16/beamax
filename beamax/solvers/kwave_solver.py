from __future__ import annotations

from typing import Any, Literal, Union
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from beamax.solvers._detectors import grid_image
from beamax.geometry import Domain
from beamax.install_kwave_metal import (
    METAL_BINARY_NAME,
    install as install_kwave_metal,
    metal_binary_path,
    metal_supported,
)
from beamax.solvers._kwave_native import (
    NATIVE_PRESSURE_RECORDS,
    run_native_simulation,
    sensor_channels_to_c_order,
)

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.compat import options_to_kwargs


_KWAVE_BINARY_ENV = "BEAMAX_KWAVE_BINARY_PATH"


def _uniform_time_step(ts: Any) -> float:
    """Validate a representably uniform time grid and return its spacing.

    Comparing against the endpoint-defined grid avoids cancellation in
    adjacent float32 differences at PAT time scales.
    """
    ts_array = np.asarray(ts)
    if ts_array.ndim != 1 or ts_array.size < 2:
        raise ValueError("ts must be one-dimensional with at least two points.")
    if np.iscomplexobj(ts_array):
        raise ValueError("ts must be real-valued.")
    if not np.all(np.isfinite(ts_array)):
        raise ValueError("ts must contain only finite values.")

    differences = np.diff(ts_array)
    real_dtype = np.dtype(ts_array.dtype)
    precision_dtype = (
        real_dtype if np.issubdtype(real_dtype, np.floating) else np.dtype(np.float64)
    )
    epsilon = float(np.finfo(precision_dtype).eps)
    start = float(ts_array[0])
    stop = float(ts_array[-1])
    ideal = np.linspace(start, stop, ts_array.size)
    spacing = (stop - start) / (ts_array.size - 1)
    magnitude = max(
        float(np.max(np.abs(ts_array))),
        abs(spacing),
        float(np.finfo(precision_dtype).tiny),
    )
    rounding_atol = 8.0 * epsilon * magnitude
    if (
        spacing <= 0.0
        or np.any(differences <= 0.0)
        or not np.allclose(
            np.asarray(ts_array, dtype=np.float64),
            ideal,
            rtol=0.0,
            atol=rounding_atol,
        )
    ):
        raise ValueError(
            "ts must be finite, strictly increasing, and uniformly spaced."
        )
    return float(spacing)


def _binary_name(device: str) -> str:
    if device == "metal":
        return METAL_BINARY_NAME
    name = "kspaceFirstOrder-CUDA" if device == "gpu" else "kspaceFirstOrder-OMP"
    if sys.platform == "win32":
        name += ".exe"
    return name


def _normalize_kwave_binary_path(path: Union[str, os.PathLike], *, device: str) -> Path:
    candidate = Path(path).expanduser()
    binary_name = _binary_name(device)

    if candidate.is_dir():
        candidate = candidate / binary_name

    if not candidate.exists():
        raise FileNotFoundError(
            f"k-Wave binary override {candidate} does not exist. "
            f"Pass a {binary_name!r} file or a directory containing it."
        )
    if not candidate.is_file():
        raise FileNotFoundError(f"k-Wave binary override {candidate} is not a file.")

    return candidate


def _default_kwave_binary_path(device: str) -> Path:
    import kwave

    return Path(kwave.BINARY_PATH) / _binary_name(device)


def _select_cpp_binary_path(kwargs: dict) -> tuple[Path, bool]:
    """
    Resolve the C++ binary path.

    Return the path and whether k-wave-python needs an explicit override.
    """
    device = kwargs.get("device", "cpu")
    if device == "metal" and not metal_supported():
        raise RuntimeError(
            "The k-Wave Metal solver requires Apple silicon, native arm64 Python, "
            "and macOS 15 or later."
        )
    explicit = kwargs.get("binary_path")
    if explicit:
        return _normalize_kwave_binary_path(explicit, device=device), True

    env_override = os.environ.get(_KWAVE_BINARY_ENV)
    if env_override:
        return _normalize_kwave_binary_path(env_override, device=device), True

    if device == "metal":
        # Saving HDF5 inputs does not launch the executable.
        if kwargs.get("save_only"):
            return metal_binary_path(), True
        return install_kwave_metal(), True

    return _default_kwave_binary_path(device), False


def _configure_cpp_binary_kwargs(kwargs: dict) -> Path:
    binary_path, should_forward = _select_cpp_binary_path(kwargs)
    if kwargs.get("device") == "metal":
        # k-wave-python accepts "gpu" and runs the explicitly selected executable.
        kwargs["device"] = "gpu"

    if should_forward:
        kwargs["binary_path"] = str(binary_path)
    else:
        kwargs.pop("binary_path", None)

    return binary_path


def _remove_stale_cpp_input(kwargs: dict) -> None:
    """Remove k-Wave's fixed input filename when reusing ``data_path``."""
    data_path = kwargs.get("data_path")
    if data_path is not None:
        (Path(data_path).expanduser() / "kwave_input.h5").unlink(missing_ok=True)


Array = Union[np.ndarray, jnp.ndarray]


class KWaveSolver:
    """
    :mod:`k-wave-python` wrapper (2D/3D) for forward, time-reversal, and adjoint solves.

    Used as the reference solver in examples and as the low-frequency leg of
    :class:`HybridSolver`.

    Parameters
    ----------
    simulation_options : kwave.options.SimulationOptions, optional
        Legacy simulation-options object. It cannot be mixed with unified
        keyword options.
    execution_options : kwave.options.SimulationExecutionOptions, optional
        Legacy execution-options object. It cannot be mixed with unified
        keyword options.
    backend : {"cpp", "python", "mac"}, optional
        Solver for forward propagation, time reversal and the adjoint. The
        default is native C++ on the CPU when no legacy options are supplied.
    **kwargs
        Unified k-Wave arguments forwarded to
        :func:`kwave.kspaceFirstOrder`. If neither legacy options nor kwargs
        are supplied, sensible CPU defaults are used.

    Notes
    -----
    - Requires the ``[kwave]`` extra (``pip install 'beamax[kwave]'``).
    - Boundary handling uses a PML; :attr:`Domain.periodic` flags control
      PML-inside vs. PML-outside placement via the k-Wave options.
    - ``backend="cpp"`` (default) uses the native OpenMP CPU solver,
      ``backend="python"`` uses k-wave-python, and ``backend="mac"``
      uses the native Apple Metal solver. This selection applies to forward,
      time reversal, and adjoint calls on the instance.
    - ``binary_path`` can be supplied directly or via
      ``BEAMAX_KWAVE_BINARY_PATH``. Direct kwargs take precedence.
    - ``backend="mac"`` is shorthand for ``backend="cpp", device="metal"``.
      On Apple silicon with macOS 15 or later, the first Metal run downloads
      and verifies the executable. For native CUDA use ``backend="cpp",
      device="gpu"`` on a supported host.
    - Regular C++ solves with the legacy default ``data_path`` use k-Wave's
      per-run scratch directory. Supply a custom path to retain the HDF5 files.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from beamax import Domain, Sensor
    >>> from beamax.solvers import KWaveSolver
    >>> domain = Domain(N=(64, 64), dx=(1e-3, 1e-3), c=1500.0, periodic=(False, False))  # doctest: +SKIP
    >>> solver = KWaveSolver(pml_size=10, device="cpu")  # doctest: +SKIP
    """

    def __init__(
        self,
        simulation_options=None,
        execution_options=None,
        *,
        backend: Literal["cpp", "python", "mac"] | None = None,
        **kwargs,
    ):
        """
        Initialize k-Wave solver options.

        Parameters
        ----------
        simulation_options : kwave.options.SimulationOptions, optional
            Legacy simulation options object.
        execution_options : kwave.options.SimulationExecutionOptions, optional
            Legacy execution options object.
        backend : {"cpp", "python", "mac"}, optional
            Select native C++, k-wave-python, or Apple Metal for all solves.
        **kwargs
            Unified keyword options forwarded to
            :func:`kwave.kspaceFirstOrder`.
        """
        if backend is not None:
            kwargs["backend"] = backend
        uses_legacy_options = (
            simulation_options is not None or execution_options is not None
        )
        if uses_legacy_options and kwargs:
            raise TypeError(
                "Legacy simulation/execution options cannot be mixed with "
                "unified keyword options. Convert them to keyword options first."
            )

        defaults = {
            "pml_inside": False,
            "pml_size": 20,
            "smooth_p0": False,
            "backend": "cpp",
            "device": "cpu",
            "debug": False,
        }
        supplied = (
            options_to_kwargs(simulation_options, execution_options)
            if uses_legacy_options
            else dict(kwargs)
        )
        # Legacy options supply a shared temp root. Let k-Wave allocate per-run
        # scratch for regular solves and keep the output path for save-only runs.
        if (
            uses_legacy_options
            and not supplied.get("save_only")
            and supplied.get("data_path") == tempfile.gettempdir()
        ):
            supplied.pop("data_path")
        self._solver_kwargs = {**defaults, **supplied}
        backend = self._solver_kwargs["backend"]
        device = self._solver_kwargs["device"]
        if backend == "mac":
            if "device" in supplied and device not in ("gpu", "metal"):
                raise ValueError(
                    "backend='mac' uses the Apple GPU. Omit device or use device='metal'."
                )
            self._solver_kwargs.update(backend="cpp", device="metal")
        elif backend not in ("cpp", "python"):
            raise ValueError("backend must be 'cpp', 'python', or 'mac'.")
        elif device == "metal" and backend != "cpp":
            raise ValueError("device='metal' requires backend='cpp'.")
        elif device not in ("cpu", "gpu", "metal"):
            raise ValueError("device must be 'cpu', 'gpu', or 'metal'.")
        self._explicit_solver_options = set(supplied)

    def _create_kgrid(self, domain: Domain, ts: Array) -> kWaveGrid:
        """
        Build and configure k-Wave grid from `Domain` and time vector.

        Parameters
        ----------
        domain : Domain
        ts : np.ndarray, shape (Nt,)

        Returns
        -------
        kWaveGrid
        """

        x64_enabled = bool(getattr(jax.config, "x64_enabled", False))
        dtype = np.float64 if x64_enabled else np.float32
        ts = np.asarray(ts, dtype=dtype)
        dt = _uniform_time_step(ts)
        kgrid = kWaveGrid(N=domain.N, spacing=domain.dx)

        kgrid.setTime(len(ts), dt)
        return kgrid

    def _kwargs_for_domain(self, domain: Domain) -> dict[str, Any]:
        """Return solver options with safe domain-derived PML defaults."""
        kwargs = dict(self._solver_kwargs)
        if "pml_inside" not in self._explicit_solver_options:
            kwargs["pml_inside"] = all(domain.periodic) or any(n == 1 for n in domain.N)
        if "pml_size" not in self._explicit_solver_options:
            max_pml = 20
            pml_sizes = [
                0 if p or n == 1 else max(0, min(max_pml, n // 2 - 1))
                for p, n in zip(domain.periodic, domain.N)
            ]
            kwargs["pml_size"] = (
                pml_sizes[0] if len(set(pml_sizes)) == 1 else tuple(pml_sizes)
            )
        return kwargs

    @staticmethod
    def _alpha_power_for_kwave(domain: Domain) -> float | None:
        """Return k-Wave's spatially constant absorption exponent.

        ``Domain`` permits medium fields to be scalar, array-valued, or
        callable. k-Wave permits a heterogeneous ``alpha_coeff`` but requires
        ``alpha_power`` to be one scalar. Accept a field representation only
        when it is constant, and collapse it before constructing
        :class:`kWaveMedium`.
        """
        values = domain.alpha_power_array
        if values is None:
            return None
        array = np.asarray(values)
        first = float(array.reshape(-1)[0])
        if not np.allclose(array, first, rtol=0.0, atol=0.0):
            raise ValueError(
                "k-Wave requires domain.alpha_power to be spatially constant."
            )
        return first

    @staticmethod
    def _validate_mask(mask: np.ndarray, domain: Domain, *, name: str) -> np.ndarray:
        """Validate a grid-aligned binary k-Wave mask."""
        mask = np.asarray(mask)
        if tuple(mask.shape) != domain.N:
            raise ValueError(f"{name} must have shape {domain.N}; got {mask.shape}.")
        if not np.all(np.isfinite(mask)) or not np.all((mask == 0) | (mask == 1)):
            raise ValueError(f"{name} must contain only finite binary values 0 and 1.")
        if not np.any(mask == 1):
            raise ValueError(f"{name} must contain at least one active point.")
        return mask

    def _run_simulation(
        self,
        domain: Domain,
        ts: Array,
        source: kSource,
        sensor: kSensor,
        *,
        inverse: bool = False,
        extra_time_step: bool = False,
    ) -> dict[str, Any]:
        """
        Run K-Wave simulation with current configuration.

        Parameters
        ----------
        domain : Domain
            Physical domain and medium.
        ts : np.ndarray, shape (Nt,)
            Time grid.
        source : kSource
            k-Wave source object.
        sensor : kSensor
            k-Wave sensor object.
        inverse : bool, default=False
            Propagate the time-varying pressure source for time reversal or
            the adjoint. Beamax prepares these sources for native executables.
        extra_time_step : bool, default=False
            Extend the validated grid by one time step at the same spacing.

        Returns
        -------
        dict
            k-Wave outputs keyed by record name.
        """
        kgrid = self._create_kgrid(domain, ts)
        if extra_time_step:
            kgrid.setTime(len(ts) + 1, kgrid.dt)

        kwargs = self._kwargs_for_domain(domain)
        if kwargs.get("save_only"):
            raise ValueError(
                "Beamax solver methods return simulation data and require save_only=False. "
                "To prepare HDF5 inputs without running a solve, use k-wave-python's "
                "kspaceFirstOrder(..., save_only=True)."
            )
        medium = kWaveMedium(
            sound_speed=np.asarray(domain.sound_speed_array),
            density=(
                None
                if domain.density_array is None
                else np.asarray(domain.density_array)
            ),
            alpha_coeff=(
                None
                if domain.alpha_coeff_array is None
                else np.asarray(domain.alpha_coeff_array)
            ),
            alpha_power=(self._alpha_power_for_kwave(domain)),
        )

        if kwargs["backend"] == "cpp":
            if not (kwargs.get("use_sg", True) and kwargs.get("use_kspace", True)):
                raise ValueError(
                    "Native k-Wave requires use_sg=True and use_kspace=True. "
                    "Select backend='python' to disable either option."
                )
            if len(domain.N) not in (2, 3) or min(domain.N) < 2:
                raise ValueError(
                    "Native k-Wave requires a 2D or 3D grid with at least two points per axis."
                )
            if len(domain.N) == 2 and domain.N[0] == 2:
                raise ValueError(
                    "k-wave-python's native input writer interprets a two-row 2D mask as Cartesian coordinates."
                )
            records = sensor.record
            if records and records[0] not in NATIVE_PRESSURE_RECORDS:
                raise ValueError(
                    f"Unsupported native k-Wave record {records[0]!r}. "
                    f"Choose one of {', '.join(NATIVE_PRESSURE_RECORDS)}."
                )
            binary = _configure_cpp_binary_kwargs(kwargs)
            if inverse or records:
                kwargs["binary_path"] = binary
                return run_native_simulation(kgrid, medium, source, sensor, **kwargs)
            _remove_stale_cpp_input(kwargs)
        elif (
            kwargs["pml_inside"]
            and getattr(sensor, "record", None) == ["p_final"]
            and any(p != 0 for p in np.atleast_1d(kwargs["pml_size"]))
        ):
            # k-wave-python 0.6.2 crops p_final even for an interior PML.
            # Recording one sample everywhere keeps the original grid extent.
            final_sample = kSensor(record=["p"])
            final_sample.record_start_index = int(kgrid.Nt)
            result = kspaceFirstOrder(kgrid, medium, source, final_sample, **kwargs)
            return {"p_final": np.asarray(result["p"]).reshape(domain.N)}

        return kspaceFirstOrder(
            kgrid,
            medium,
            source,
            sensor,
            **kwargs,
        )

    def _detector_inverse_inputs(
        self,
        data,
        domain,
        detectors,
        ts,
        *,
        adjoint=False,
        pairing=None,
        element_measures=None,
    ):
        from beamax.solvers._detectors import grid_detectors, detector_measure

        mask, mapping = grid_detectors(domain, detectors)
        count = int(mask.sum())
        data = np.asarray(data)
        expected = (len(ts), count)
        if mapping is not None and data.shape == (len(ts), *detectors.shape):
            data = data.reshape(expected)
        if data.shape != expected:
            raise ValueError(
                f"Expected time-first detector data with shape {expected}."
            )
        if adjoint:
            pairing, measure = detector_measure(
                detectors, pairing, element_measures, count
            )
            if pairing != "euclidean":
                data = data * (_uniform_time_step(ts) * measure / np.prod(domain.dx))
        if mapping is not None:
            data = data[:, mapping.chart_to_mask]
        return data, mask

    def forward(
        self,
        p0: Union[np.ndarray, jnp.ndarray],
        domain: Domain,
        sensors=None,
        ts=None,
        *,
        detectors=None,
        record: str = "p",
    ) -> np.ndarray:
        """
        Forward k-Wave simulation for linear wave equation.

        Parameters
        ----------
        p0 : array
            Initial pressure.
        domain : Domain
        sensors : array
            Sensor mask or positions (solver expects mask).
        ts : array, shape (Nt,)
        record : {"p", ...}
            Quantity to record.

        Returns
        -------
        np.ndarray
            Pressure time series ``(Nt, Ns)`` for ``record="p"``. A final
            pressure or whole-grid extremum has shape ``domain.N``. A
            per-sensor pressure statistic has shape ``(Ns,)``. Sensor
            channels follow NumPy C mask order on all backends.
        """
        if ts is None:
            raise TypeError("k-Wave requires the acquisition times.")
        if (
            detectors is not None
            and np.asarray(ts).size
            and np.asarray(ts).flat[0] != 0
        ):
            raise ValueError("k-Wave acquisition times must start at zero.")
        p0_array = np.asarray(p0)
        if tuple(p0_array.shape) != domain.N:
            raise ValueError(f"p0 must have shape {domain.N}; got {p0_array.shape}.")
        if not np.all(np.isfinite(p0_array)):
            raise ValueError("p0 must contain only finite values.")
        source = kSource()
        source.p0 = p0_array

        from beamax.solvers._detectors import grid_detectors

        from beamax.surfaces import DetectorSurface

        backend = str(self._solver_kwargs.get("backend", "cpp")).lower()
        mapping = None
        cartesian = False
        if detectors is not None:
            if sensors is not None:
                raise TypeError("Give only one of sensors and detectors.")
            if isinstance(detectors, DetectorSurface):
                if detectors.ndim != domain.ndim:
                    raise ValueError("Detector and propagation dimensions must match.")
                try:
                    sensors, mapping = grid_detectors(domain, detectors)
                except ValueError as error:
                    if backend != "python":
                        raise ValueError(
                            "Native receivers require distinct grid nodes. Use Python forward or explicitly snap the surface."
                        ) from error
                    positions = np.asarray(detectors.positions(), dtype=float)
                    spacing = np.asarray(domain.dx)
                    maximum = (np.asarray(domain.N) - 1) * spacing
                    if (
                        not np.all(np.isfinite(positions))
                        or np.any(positions < 0)
                        or np.any(positions > maximum)
                    ):
                        raise ValueError(
                            "Cartesian receivers must lie within the sampled grid."
                        ) from error
                    sensors = (positions - (np.asarray(domain.N) // 2) * spacing).T
                    cartesian = True
            else:
                sensors, mapping = grid_detectors(domain, detectors)
        sensor_mask = (
            np.asarray(sensors)
            if cartesian
            else self._validate_mask(np.asarray(sensors), domain, name="sensors")
        )
        sensor = (
            kSensor(mask=sensor_mask)
            if backend == "cpp" and record == "p"
            else kSensor(mask=sensor_mask, record=[record])
        )

        nt = np.size(ts)
        ns = sensor_mask.shape[1] if cartesian else int(np.count_nonzero(sensor_mask))
        simulation_nt = nt
        if backend == "cpp" and record == "p" and nt == ns:
            # k-Wave can mistake the time axis of a square C++ trace for the
            # sensor axis. One more step makes the output shape unambiguous.
            result = self._run_simulation(
                domain, ts, source, sensor, extra_time_step=True
            )
            simulation_nt += 1
        else:
            result = self._run_simulation(domain, ts, source, sensor)

        if record not in result:
            raise ValueError(
                f"The selected k-Wave backend did not return record {record!r}."
            )
        out = np.asarray(result[record])
        if record.endswith(("_final", "_all")):
            return out
        if record == "p" and out.ndim == 1 and ns == 1:
            out = out[:, None]
        if out.ndim == 2 and out.shape == (ns, simulation_nt):
            out = out.T

        # Return the requested time samples with sensor channels in C order.
        if backend == "cpp" and out.ndim == 2 and out.shape == (simulation_nt, ns):
            out = out[:nt, sensor_channels_to_c_order(sensor_mask)]

        if mapping is not None:
            out = out[..., mapping.mask_to_chart]
        return out

    @staticmethod
    def _coerce_sensor_data_layout(
        data: Array,
        source_mask: np.ndarray,
        *,
        data_layout: str,
        op_name: str,
    ) -> np.ndarray:
        """
        Normalize sensor time series to (Ns, Nt), where Ns is source points.

        Parameters
        ----------
        data : np.ndarray
            Sensor data with shape (Ns, Nt), (Nt, Ns), or (Nt,) for Ns=1.
        source_mask : np.ndarray
            Binary source mask used to determine Ns.
        data_layout : {"auto", "ns_nt", "nt_ns"}
            Explicit or inferred layout for `data`.
        op_name : str
            Operation label used in error messages.
        """
        if data_layout not in {"auto", "ns_nt", "nt_ns"}:
            raise ValueError(
                f"Invalid data_layout='{data_layout}' for {op_name}. "
                "Use one of {'auto', 'ns_nt', 'nt_ns'}."
            )

        ns = int(np.count_nonzero(source_mask))
        if ns <= 0:
            raise ValueError(f"{op_name} requires at least one active source point.")

        sensor_data = np.array(data)
        if sensor_data.ndim == 1:
            if ns != 1:
                raise ValueError(
                    f"{op_name} received 1D data of shape {sensor_data.shape}, "
                    f"but source mask has {ns} points."
                )
            sensor_data = sensor_data[None, :]
        elif sensor_data.ndim != 2:
            raise ValueError(
                f"{op_name} expects 2D sensor data, got shape {sensor_data.shape}."
            )

        if data_layout == "ns_nt":
            if sensor_data.shape[0] != ns:
                raise ValueError(
                    f"{op_name} expected data layout (Ns, Nt) with Ns={ns}, "
                    f"got shape {sensor_data.shape}."
                )
            return sensor_data

        if data_layout == "nt_ns":
            if sensor_data.shape[1] != ns:
                raise ValueError(
                    f"{op_name} expected data layout (Nt, Ns) with Ns={ns}, "
                    f"got shape {sensor_data.shape}."
                )
            return sensor_data.T

        rows_match = sensor_data.shape[0] == ns
        cols_match = sensor_data.shape[1] == ns
        if rows_match and not cols_match:
            return sensor_data
        if cols_match and not rows_match:
            return sensor_data.T
        if rows_match and cols_match:
            raise ValueError(
                f"{op_name} received ambiguous square data with Ns=Nt={ns}; "
                "pass data_layout='ns_nt' or 'nt_ns'."
            )

        raise ValueError(
            f"{op_name} could not infer data layout for shape {sensor_data.shape} "
            f"with Ns={ns}. Pass data_layout='ns_nt' or 'nt_ns'."
        )

    @staticmethod
    def _build_adjoint_source(sensor_data_ns_nt: np.ndarray) -> np.ndarray:
        """
        Build additive adjoint source from sensor data in (Ns, Nt) layout.

        Parameters
        ----------
        sensor_data_ns_nt : np.ndarray, shape (Ns, Nt)
            Sensor data with sensors along the first axis and time along the
            second axis.

        Returns
        -------
        np.ndarray, shape (Ns, Nt)
            Additive adjoint source for k-Wave.

        Notes
        -----
        This matches the updated k-Wave MATLAB adjoint example:
            p_adj = [r, 0] + [0, r]
            p_adj(:, end-1) = p_adj(:, end-1) + p_adj(:, end)
            p_src = p_adj(:, 1:end-1)
        where r is the time-reversed measurement residual.
        """
        sensor_data_ns_nt = np.asarray(sensor_data_ns_nt)
        if sensor_data_ns_nt.ndim != 2 or sensor_data_ns_nt.shape[1] < 2:
            raise ValueError(
                "Adjoint source data must have shape (Ns, Nt) with Nt >= 2."
            )
        s_rev = np.flip(sensor_data_ns_nt, axis=1)
        zeros_col = np.zeros((s_rev.shape[0], 1), dtype=s_rev.dtype)
        p_adj = np.concatenate([s_rev, zeros_col], axis=1) + np.concatenate(
            [zeros_col, s_rev], axis=1
        )
        p_adj[:, -2] = p_adj[:, -2] + p_adj[:, -1]
        return p_adj[:, :-1]

    def time_reversal(
        self,
        data: Union[np.ndarray, jnp.ndarray],
        domain,
        sensors=None,
        sources=None,
        ts=None,
        *,
        detectors=None,
        eval_points=None,
        record: str = "p_final",
        data_layout: str = "auto",
    ) -> np.ndarray:
        r"""
        Run classic k-Wave time reversal.

        Parameters
        ----------
        data : np.ndarray or jnp.ndarray, shape (Ns, Nt) or (Nt, Ns)
            Sensor measurements with one sample per time point in ``ts``.
        domain : Domain
            Reconstruction domain.
        sensors : array-like
            Sensor mask.
        sources : array-like
            Source mask where the reversed data are injected.
        ts : np.ndarray or jnp.ndarray, shape (Nt,)
            Time grid.
        record : str, default="p_final"
            k-Wave field to return.
        data_layout : {"auto", "ns_nt", "nt_ns"}, default="auto"
            Sensor-data layout interpretation.

        Returns
        -------
        np.ndarray
            Requested k-Wave record.

        Notes
        -----
        Enforces $p(\mathbf{x}_s,t)=d(t,\mathbf{x}_s)$ as a Dirichlet source,
        where $d$ denotes ``sensor_data``.
        """
        if ts is None:
            raise TypeError("k-Wave requires the acquisition times.")
        if (
            detectors is not None
            and np.asarray(ts).size
            and np.asarray(ts).flat[0] != 0
        ):
            raise ValueError("k-Wave acquisition times must start at zero.")
        if detectors is not None:
            if (
                sensors is not None
                or sources is not None
                or data_layout not in ("auto", "nt_ns")
            ):
                raise TypeError(
                    "detectors requires time-first data and replaces sources and sensors."
                )
            data, sources = self._detector_inverse_inputs(data, domain, detectors, ts)
            sensors = np.ones(domain.N, dtype=bool)
            data_layout = "nt_ns"
        elif eval_points is not None:
            raise TypeError("These options require detectors.")

        sensor_mask = self._validate_mask(np.asarray(sensors), domain, name="sensors")
        source_mask = self._validate_mask(np.asarray(sources), domain, name="sources")
        sensor_data = self._coerce_sensor_data_layout(
            data,
            source_mask,
            data_layout=data_layout,
            op_name="time_reversal",
        )
        nt = np.size(ts)
        if sensor_data.shape[1] != nt:
            raise ValueError(
                f"Expected {nt} time samples to match ts, got {sensor_data.shape[1]}."
            )

        sensor_data_rev = np.flip(sensor_data, axis=1)

        src = kSource()
        setattr(src, "p", sensor_data_rev)
        setattr(src, "p_mask", source_mask)
        src.p_mode = "dirichlet"

        sensor = kSensor(mask=sensor_mask, record=[record])

        out = self._run_simulation(domain, ts, src, sensor, inverse=True)
        if record not in out:
            raise ValueError(
                f"The selected k-Wave backend did not return record {record!r}."
            )
        if detectors is not None and eval_points is not None and record != "p_final":
            raise ValueError("eval_points requires a terminal pressure record.")
        return grid_image(out[record], domain, eval_points)

    def adjoint(
        self,
        data: Union[np.ndarray, jnp.ndarray],
        domain,
        sensors=None,
        sources=None,
        ts=None,
        *,
        detectors=None,
        eval_points=None,
        pairing=None,
        element_measures=None,
        record: str = "p_final",
        data_layout: str = "auto",
    ) -> np.ndarray:
        r"""
        Discrete k-Wave adjoint following Arridge et al., Appendix B.

        Parameters
        ----------
        data : array, shape (Ns, Nt) or (Nt, Ns)
            Sensor measurements with one sample per time point in ``ts``.
        domain : Domain
        sensors : array
            Sensor mask.
        sources : array
            Source mask.
        ts : array, shape (Nt,)
        record : {"p_final"}
            Terminal pressure field. Other records cannot be converted to the
            initial-pressure transpose and are therefore rejected.
        data_layout : {"auto", "ns_nt", "nt_ns"}, default="auto"
            Sensor-data layout interpretation.

        Returns
        -------
        np.ndarray
            Euclidean discrete adjoint image $A^{\mathsf T}d$
            (``A.T @ data``) with shape ``domain.N``.

        Notes
        -----
        This implementation requires a lossless medium and uses unweighted
        discrete sums. With an active absorbing boundary layer (PML),
        damping can introduce a discrepancy from the exact transpose of the
        finite-grid forward simulation.
        To represent a one-cell planar detector under the continuous surface
        and volume rectangle rules of Arridge et al. (2016), multiply the
        result by $\Delta t/\Delta x_{\mathrm{normal}}$.

        The source is Eq. (B.2) of Arridge et al. (2016). k-Wave's additive
        pressure-source preprocessing multiplies a user source by
        $2\Delta t/(d\,c\,\Delta x)$ before adding it to each split density
        field.
        Consequently the user source must be

        $$
        \frac{\rho_{\mathrm{source}}c_{\mathrm{source}}\Delta x}
        {4\Delta t}\,\beta,
        $$

        and the terminal pressure must be divided pointwise by
        $c^2\rho$. ``additive-no-correction`` is intentional: the optional
        cosine k-space filter on the injected source is not part of Eq. (B.2).
        This mode still applies k-Wave's additive-source amplitude scaling and
        does not disable the sinc k-space correction used by the propagation
        operators.

        The current k-wave-python Python backend uses a different pressure
        scaling for each spatial axis on anisotropic grids. A single scalar
        pressure signal cannot then create the equal split-density increments
        required by Eq. (B.2), so this implementation requires isotropic grid
        spacing.
        """
        if ts is None:
            raise TypeError("k-Wave requires the acquisition times.")
        if (
            detectors is not None
            and np.asarray(ts).size
            and np.asarray(ts).flat[0] != 0
        ):
            raise ValueError("k-Wave acquisition times must start at zero.")
        if detectors is not None:
            if (
                sensors is not None
                or sources is not None
                or data_layout not in ("auto", "nt_ns")
            ):
                raise TypeError(
                    "detectors requires time-first data and replaces sources and sensors."
                )
            data, sources = self._detector_inverse_inputs(
                data,
                domain,
                detectors,
                ts,
                adjoint=True,
                pairing=pairing,
                element_measures=element_measures,
            )
            sensors = np.ones(domain.N, dtype=bool)
            data_layout = "nt_ns"
        elif (
            eval_points is not None
            or pairing is not None
            or element_measures is not None
        ):
            raise TypeError("These options require detectors.")

        if record != "p_final":
            raise ValueError(
                f"The scaled k-Wave adjoint requires record='p_final'; got {record!r}."
            )
        if bool(self._kwargs_for_domain(domain).get("smooth_p0", True)):
            raise ValueError(
                "The discrete k-Wave adjoint requires smooth_p0=False. "
                "k-Wave's restore-max p0 smoothing is nonlinear and therefore "
                "does not have the transpose implemented here."
            )

        spacings = np.asarray(domain.dx, dtype=float)
        if not np.allclose(spacings, spacings[0], rtol=1e-12, atol=0.0):
            raise NotImplementedError(
                "The scaled k-Wave adjoint currently requires isotropic grid "
                f"spacing; got domain.dx={domain.dx}."
            )
        if domain.alpha_coeff is not None and np.any(
            np.asarray(domain.alpha_coeff_array) != 0
        ):
            raise NotImplementedError(
                "The discrete k-Wave adjoint requires a lossless medium (alpha_coeff=0)."
            )

        dt = _uniform_time_step(ts)

        source_mask = self._validate_mask(np.asarray(sources), domain, name="sources")
        sensor_mask = self._validate_mask(np.asarray(sensors), domain, name="sensors")
        sensor_data = self._coerce_sensor_data_layout(
            data,
            source_mask,
            data_layout=data_layout,
            op_name="adjoint",
        )
        nt = np.size(ts)
        if sensor_data.shape[1] != nt:
            raise ValueError(
                f"Expected {nt} time samples to match ts, got {sensor_data.shape[1]}."
            )
        beta = self._build_adjoint_source(sensor_data)

        sound_speed = np.asarray(domain.sound_speed_array, dtype=float)
        density_array = domain.density_array
        density = (
            np.full(domain.N, 1000.0, dtype=float)
            if density_array is None
            else np.asarray(density_array, dtype=float)
        )
        source_points = source_mask.astype(bool)
        source_scale = (
            density[source_points]
            * sound_speed[source_points]
            * spacings[0]
            / (4.0 * dt)
        )
        p_src = source_scale[:, None] * beta

        src = kSource()
        setattr(src, "p", p_src)
        setattr(src, "p_mask", source_mask)
        src.p_mode = "additive-no-correction"

        sensor = kSensor(mask=sensor_mask, record=[record])

        out = self._run_simulation(domain, ts, src, sensor, inverse=True)
        terminal_pressure = np.asarray(out[record])
        if tuple(terminal_pressure.shape) != domain.N:
            raise ValueError(
                "k-Wave returned an adjoint field with unexpected shape "
                f"{terminal_pressure.shape}; expected {domain.N}."
            )
        return grid_image(
            terminal_pressure / (sound_speed**2 * density), domain, eval_points
        )
