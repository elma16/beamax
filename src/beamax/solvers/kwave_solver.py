from __future__ import annotations

from typing import Optional, Union, Tuple
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from io import StringIO
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from beamax.geometry import Domain
from beamax.solvers.solverbase import Solver

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.compat import options_to_kwargs


_KWAVE_BINARY_ENV = "BEAMAX_KWAVE_BINARY_PATH"
_BAD_DARWIN_OMP_VERSIONS = ("v0.3.0rc3", "v1.4.0")
_BAD_DARWIN_OMP_SHA256 = {
    # v0.3.0rc3: old Darwin OMP build silently mishandles absorption.
    "d6bb759dd6addcfaaee9333be61b4d31793d5d6ed3c77d384d77217e5aaee32e",
    # v1.4.0: CMake release asset was built with fast-math before the v1.4.1 fix.
    "fcc5adc84266379be4d4a576640197fdfcd8c308f058abc8a89a5b81d06078fb",
}


def _patch_cpp_simulation_stale_hdf5():
    """
    Patch stale HDF5 handling in k-wave-python.

    Notes
    -----
    Some k-wave-python versions do not remove stale HDF5 files before
    writing, causing "name already exists" errors when the same temp path is
    reused.
    """
    from kwave.solvers.cpp_simulation import CppSimulation

    if getattr(CppSimulation._write_hdf5, "_beamax_stale_hdf5_patch", False):
        return

    _orig_write = CppSimulation._write_hdf5

    def _patched_write(self, filepath):
        """
        Remove an existing HDF5 file before delegating to k-Wave.

        Parameters
        ----------
        self : kwave.solvers.cpp_simulation.CppSimulation
            Simulation instance.
        filepath : str or path-like
            HDF5 output path.
        """
        if os.path.exists(filepath):
            os.remove(filepath)
        _orig_write(self, filepath)

    _patched_write._beamax_stale_hdf5_patch = True
    CppSimulation._write_hdf5 = _patched_write


def _binary_name(device: str) -> str:
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

    Returns `(binary_path, should_forward)` where `should_forward` means the
    path came from Beamax/user configuration and must be passed to
    k-wave-python.
    """
    device = kwargs.get("device", "cpu")
    explicit = kwargs.get("binary_path")
    if explicit:
        return _normalize_kwave_binary_path(explicit, device=device), True

    env_override = os.environ.get(_KWAVE_BINARY_ENV)
    if env_override:
        return _normalize_kwave_binary_path(env_override, device=device), True

    return _default_kwave_binary_path(device), False


def _configure_cpp_binary_kwargs(kwargs: dict) -> Path:
    binary_path, should_forward = _select_cpp_binary_path(kwargs)

    if should_forward:
        kwargs["binary_path"] = str(binary_path)
    else:
        kwargs.pop("binary_path", None)

    return binary_path


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _metadata_marks_bad_darwin_omp(binary_path: Path) -> bool:
    metadata_path = binary_path.with_name(f"{binary_path.name}_metadata.json")
    if not metadata_path.exists():
        return False

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    version = str(metadata.get("version", ""))
    url = str(metadata.get("url", ""))
    return any(
        bad_version in version or bad_version in url
        for bad_version in _BAD_DARWIN_OMP_VERSIONS
    )


def _domain_has_nonzero_absorption(domain: Domain) -> bool:
    alpha_coeff = domain.alpha_coeff
    if alpha_coeff is None:
        return False
    if callable(alpha_coeff):
        alpha_coeff = domain._eval(alpha_coeff)

    try:
        return bool(np.any(np.asarray(alpha_coeff) != 0))
    except (TypeError, ValueError):
        return bool(alpha_coeff)


def _reject_known_bad_darwin_absorption_binary(
    binary_path: Path, domain: Domain
) -> None:
    if sys.platform != "darwin" or binary_path.name != "kspaceFirstOrder-OMP":
        return
    if not _domain_has_nonzero_absorption(domain):
        return
    if not binary_path.exists():
        return

    is_known_bad = _metadata_marks_bad_darwin_omp(binary_path)
    if not is_known_bad:
        try:
            is_known_bad = _file_sha256(binary_path) in _BAD_DARWIN_OMP_SHA256
        except OSError:
            is_known_bad = False

    if not is_known_bad:
        return

    raise RuntimeError(
        "The configured macOS k-Wave OMP binary is a known-bad Darwin build "
        "for power-law absorption. Install beamax[kwave] with "
        "k-wave-python>=0.6.2, or point BEAMAX_KWAVE_BINARY_PATH at the "
        "k-wave-omp-darwin v1.4.1 kspaceFirstOrder-OMP release asset."
    )


_patch_cpp_simulation_stale_hdf5()


Array = Union[np.ndarray, jnp.ndarray]


class KWaveSolver(Solver):
    """
    :mod:`k-wave-python` wrapper (2D/3D) for forward, time-reversal, and adjoint solves.

    Provides a :class:`beamax.solvers.Solver`-compatible interface backed by
    the k-Wave pseudo-spectral time-domain solver. Used as the "reference" in
    examples and as the low-frequency leg of :class:`HybridSolver`.

    Parameters
    ----------
    simulation_options : kwave.options.SimulationOptions, optional
        Legacy simulation-options object. If provided with
        ``execution_options``, they are converted to the unified kwargs dict.
    execution_options : kwave.options.SimulationExecutionOptions, optional
        Legacy execution-options object. See ``simulation_options``.
    **kwargs
        Unified k-Wave arguments forwarded to
        :func:`kwave.kspaceFirstOrder`. If neither legacy options nor kwargs
        are supplied, sensible CPU defaults are used.

    Notes
    -----
    - Requires the ``[kwave]`` extra (``pip install 'beamax[kwave]'``).
    - Boundary handling uses a PML; :attr:`Domain.periodic` flags control
      PML-inside vs. PML-outside placement via the k-Wave options.
    - Forward solves run on the C++ backend by default; time-reversal and
      adjoint currently fall back to the pure-Python backend until the
      upstream ``CppSimulation`` path ships source preprocessing for
      time-varying pressure sources.
    - On macOS the class transparently patches the ``libhdf5.310`` →
      ``libhdf5.320`` linkage mismatch so recent Homebrew HDF5 installs work.
    - ``binary_path`` can be supplied directly or via
      ``BEAMAX_KWAVE_BINARY_PATH``. Direct kwargs take precedence.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from beamax import Domain, Sensor
    >>> from beamax.solvers import KWaveSolver
    >>> domain = Domain(N=(64, 64), dx=(1e-3, 1e-3), c=1500.0, periodic=(False, False))  # doctest: +SKIP
    >>> solver = KWaveSolver(pml_size=10, device="cpu")  # doctest: +SKIP
    """

    _hdf5_compat_paths: set[str] = set()

    def __init__(
        self,
        simulation_options=None,
        execution_options=None,
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
        **kwargs
            Unified keyword options forwarded to
            :func:`kwave.kspaceFirstOrder`.
        """
        if simulation_options is not None:
            # Legacy path: convert old option objects to unified kwargs
            self._solver_kwargs = options_to_kwargs(
                simulation_options, execution_options
            )
        elif kwargs:
            self._solver_kwargs = kwargs
        else:
            self._solver_kwargs = dict(
                pml_inside=False,
                pml_size=20,
                smooth_p0=False,
                backend="cpp",
                device="cpu",
                debug=True,
            )

    @classmethod
    def _ensure_macos_hdf5_compat(cls, binary_path: Optional[Path] = None) -> None:
        """
        Create DYLD symlinks for macOS HDF5 version compatibility.

        Notes
        -----
        The k-Wave OMP binary may be linked against ``libhdf5.310`` while a
        system only ships ``libhdf5.320``. This method creates temporary
        compatibility symlinks when possible.
        """
        if sys.platform != "darwin":
            return

        if binary_path is None:
            binary_path = _default_kwave_binary_path("cpu")
        if not binary_path.exists():
            return

        try:
            cache_key = str(binary_path.resolve())
        except OSError:
            cache_key = str(binary_path)
        if cache_key in cls._hdf5_compat_paths:
            return

        try:
            out = subprocess.run(
                ["otool", "-L", str(binary_path)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (subprocess.SubprocessError, OSError):
            cls._hdf5_compat_paths.add(cache_key)
            return

        refs = re.findall(
            r"^\s+(/.*libhdf5(?:_hl)?\.310\.dylib)\s", out, flags=re.MULTILINE
        )
        if not refs:
            cls._hdf5_compat_paths.add(cache_key)
            return

        compat_links = {}
        for ref in refs:
            ref_path = Path(ref)
            if ref_path.exists():
                continue
            # Try .320 first, then unversioned
            for replacement in [
                ref_path.name.replace(".310.", ".320."),
                ref_path.name.replace(".310", ""),
            ]:
                candidate = ref_path.with_name(replacement)
                if candidate.exists():
                    compat_links[ref_path.name] = candidate
                    break

        if not compat_links:
            cls._hdf5_compat_paths.add(cache_key)
            return

        compat_dir = Path(tempfile.gettempdir()) / "beamax_kwave_hdf5_compat"
        compat_dir.mkdir(parents=True, exist_ok=True)

        for link_name, target in compat_links.items():
            link_path = compat_dir / link_name
            if link_path.is_symlink() or link_path.exists():
                try:
                    if link_path.resolve() == target.resolve():
                        continue
                except OSError:
                    pass
                link_path.unlink()
            link_path.symlink_to(target)

        current = os.environ.get("DYLD_LIBRARY_PATH", "")
        compat_str = str(compat_dir)
        if compat_str not in current.split(":"):
            os.environ["DYLD_LIBRARY_PATH"] = (
                f"{compat_str}:{current}" if current else compat_str
            )
        cls._hdf5_compat_paths.add(cache_key)

    def _create_kgrid(self, domain: Domain, ts: np.ndarray) -> kWaveGrid:
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

        dtype = ts.dtype if jax.config.x64_enabled else np.float64
        ts = np.array(ts, dtype=dtype)
        kgrid = kWaveGrid(N=domain.N, spacing=domain.dx)

        # pml related to periodicity
        # NB: pml_inside = False fails in the case where one dimension has size 1.
        self._solver_kwargs["pml_inside"] = all(domain.periodic)
        # Cap PML size so that 2*pml < N per dimension (v0.6.1 validation)
        max_pml = 20
        pml_sizes = [
            0 if p else min(max_pml, n // 2 - 1)
            for p, n in zip(domain.periodic, domain.N)
        ]
        self._solver_kwargs["pml_size"] = (
            pml_sizes[0] if len(set(pml_sizes)) == 1 else tuple(pml_sizes)
        )

        dt = ts[1] - ts[0]
        kgrid.setTime(len(ts), dt)
        return kgrid

    def _run_simulation(
        self,
        domain: Domain,
        ts: np.ndarray,
        source: kSource,
        sensor: kSensor,
        *,
        force_python: bool = False,
    ) -> np.ndarray:
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
        force_python : bool, default=False
            Force the Python backend even when C++ is configured. Used for
            time-varying source simulations where the C++ backend lacks
            source-term preprocessing.

        Returns
        -------
        np.ndarray
            Raw k-Wave result object/array returned by
            :func:`kwave.kspaceFirstOrder`.
        """
        kgrid = self._create_kgrid(domain, ts)

        kwargs = dict(self._solver_kwargs)
        if force_python:
            kwargs["backend"] = "python"

        medium = kWaveMedium(
            sound_speed=domain.sound_speed_array,
            density=domain.density_array,
            alpha_coeff=domain.alpha_coeff,
            alpha_power=domain.alpha_power,
        )

        if kwargs.get("backend") == "cpp":
            binary_path = _configure_cpp_binary_kwargs(kwargs)
            self._ensure_macos_hdf5_compat(binary_path)
            _reject_known_bad_darwin_absorption_binary(binary_path, domain)

        result = kspaceFirstOrder(
            kgrid,
            medium,
            source,
            sensor,
            **kwargs,
        )

        return result

    def forward(
        self,
        p0: Union[np.ndarray, jnp.ndarray],
        domain: Domain,
        sensors: Union[np.ndarray, jnp.ndarray],
        ts: Union[np.ndarray, jnp.ndarray],
        *,
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
            Sensor time series `(Nt, Ns)` or solver-specific shape.
        """
        source = kSource()
        source.p0 = np.array(p0)

        sensor_mask = np.array(sensors)
        sensor = kSensor(mask=sensor_mask, record=[record])

        result = self._run_simulation(domain, ts, source, sensor)

        out = np.array(result[record])
        nt = len(ts)
        ns = int(np.count_nonzero(sensor_mask))
        if out.ndim == 2 and out.shape == (ns, nt):
            out = out.T

        return out

    @staticmethod
    def _coerce_sensor_data_layout(
        data: np.ndarray,
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
            # Ambiguous square case (Ns == Nt): keep backward-compatible default.
            return sensor_data

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
        sensors,
        sources,
        ts: Union[np.ndarray, jnp.ndarray],
        *,
        record: str = "p_final",
        data_layout: str = "auto",
    ) -> np.ndarray:
        """
        Run classic k-Wave time reversal.

        Parameters
        ----------
        data : np.ndarray or jnp.ndarray, shape (Ns, Nt) or (Nt, Ns)
            Sensor measurements.
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
        Enforces ``p(x_s, t) = sensor_data(t, x_s)`` as a Dirichlet source.
        """
        sensor_mask = np.array(sensors)
        source_mask = np.array(sources)
        sensor_data = self._coerce_sensor_data_layout(
            data,
            source_mask,
            data_layout=data_layout,
            op_name="time_reversal",
        )

        sensor_data_rev = np.flip(sensor_data, axis=1)

        src = kSource()
        src.p = sensor_data_rev
        src.p_mask = source_mask
        src.p_mode = "dirichlet"

        sensor = kSensor(mask=sensor_mask, record=[record])

        # v0.6.1 cpp backend lacks source-term scaling for time-varying
        # sources; force python backend until upstream fix.
        out = self._run_simulation(domain, ts, src, sensor, force_python=True)
        return out[record]

    def adjoint(
        self,
        data: Union[np.ndarray, jnp.ndarray],
        domain,
        sensors,
        sources,
        ts: Union[np.ndarray, jnp.ndarray],
        *,
        record: str = "p_final",
        data_layout: str = "auto",
    ) -> np.ndarray:
        """
        k-Wave adjoint following MATLAB demo convention.

        Parameters
        ----------
        data : array, shape (Ns, Nt) or (Nt, Ns)
            Sensor measurements.
        domain : Domain
        sensors : array
            Sensor mask.
        sources : array
            Source mask.
        ts : array, shape (Nt,)
        record : str
            What to record at the end (e.g., "p_final").

        Returns
        -------
        np.ndarray
            Adjoint image `(N...)`.

        Based off the original MATLAB k-Wave adjoint example: https://github.com/ucl-bug/k-wave/blob/main/k-Wave/examples/example_pr_2D_adjoint.m
        """
        source_mask = np.array(sources)
        sensor_mask = np.array(sensors)
        sensor_data = self._coerce_sensor_data_layout(
            data,
            source_mask,
            data_layout=data_layout,
            op_name="adjoint",
        )
        p_src = self._build_adjoint_source(sensor_data)

        src = kSource()
        src.p = p_src
        src.p_mask = source_mask
        src.p_mode = "additive"

        sensor = kSensor(mask=sensor_mask, record=[record])

        # v0.6.1 cpp backend lacks source-term scaling for time-varying
        # sources; force python backend until upstream fix.
        out = self._run_simulation(domain, ts, src, sensor, force_python=True)
        return out[record]


class TimedKWaveSolver(KWaveSolver):
    """
    k-Wave wrapper that also returns a timing.

    Timing modes
    ------------
    mode="stdout" (default):
        Parse k-Wave's own "Total execution time: <Xs>" line from stdout.
        Intended to reflect *simulation kernel time*, not wrapper overhead.
        Fallback to wall-clock if parsing fails.

    mode="wall":
        Wall-clock around the internal `_run()` call (includes whatever k-Wave
        does internally, excludes your pre/post Python work).
    """

    _RE = re.compile(r"Total execution time:\s*([\d.]+)s")

    def __init__(self, *args, mode: str = "stdout", **kwargs):
        """
        Initialize timed k-Wave solver.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :class:`KWaveSolver`.
        mode : {"stdout", "wall"}, default="stdout"
            Timing mode. ``"stdout"`` parses k-Wave output; ``"wall"`` uses
            Python wall-clock timing.
        **kwargs
            Keyword arguments forwarded to :class:`KWaveSolver`.

        Raises
        ------
        ValueError
            If ``mode`` is not ``"stdout"`` or ``"wall"``.
        """
        super().__init__(*args, **kwargs)
        if mode not in {"stdout", "wall"}:
            raise ValueError("mode must be 'stdout' or 'wall'")
        self._mode = mode

    def _time_call(self, fn, *a, **k) -> Tuple[np.ndarray, float]:
        """
        Execute a solver call and measure elapsed time.

        Parameters
        ----------
        fn : Callable
            Solver function to call.
        *a
            Positional arguments forwarded to ``fn``.
        **k
            Keyword arguments forwarded to ``fn``.

        Returns
        -------
        result : np.ndarray
            Solver result.
        seconds : float
            Parsed kernel time or wall-clock fallback.
        """
        if self._mode == "wall":
            t0 = time.perf_counter()
            out = fn(*a, **k)
            t1 = time.perf_counter()
            return out, (t1 - t0)

        # stdout parsing mode
        old = sys.stdout
        buf = StringIO()
        try:
            sys.stdout = buf
            t0 = time.perf_counter()
            out = fn(*a, **k)
            t1 = time.perf_counter()
        finally:
            sys.stdout = old
        txt = buf.getvalue()
        m = self._RE.search(txt)
        if m:
            return out, float(m.group(1))
        # fallback
        return out, (t1 - t0)

    # ---- override public API to return (result, seconds) ----

    def forward(
        self,
        p0,
        domain,
        sensors,
        ts,
        *,
        record: str = "p",
        **solver_kwargs,
    ) -> Tuple[np.ndarray, float]:
        """
        Same as KWaveSolver.forward, but returns (result, seconds).

        Parameters
        ----------
        p0 : array-like
            Initial pressure.
        domain : Domain
            Computational domain.
        sensors : array-like
            Sensor mask or positions.
        ts : array-like, shape (Nt,)
            Time grid.
        record : str, default="p"
            k-Wave field to record.
        **solver_kwargs
            Additional solver keyword arguments.

        Returns
        -------
        result : np.ndarray
            Forward simulation result.
        seconds : float
            Execution time.
        """
        return self._time_call(
            super().forward,
            p0=p0,
            domain=domain,
            sensors=sensors,
            ts=ts,
            record=record,
            **solver_kwargs,
        )

    def time_reversal(
        self,
        data,
        domain,
        sensors,
        sources,
        ts,
        *,
        record: str = "p_final",
        data_layout: str = "auto",
        **solver_kwargs,
    ) -> Tuple[np.ndarray, float]:
        """
        Same as KWaveSolver.time_reversal, but returns (result, seconds).

        Parameters
        ----------
        data : array-like
            Sensor measurements.
        domain : Domain
            Reconstruction domain.
        sensors : array-like
            Sensor mask.
        sources : array-like
            Source mask.
        ts : array-like, shape (Nt,)
            Time grid.
        record : str, default="p_final"
            k-Wave field to return.
        data_layout : {"auto", "ns_nt", "nt_ns"}, default="auto"
            Sensor-data layout interpretation.
        **solver_kwargs
            Additional solver keyword arguments.

        Returns
        -------
        result : np.ndarray
            Time-reversal result.
        seconds : float
            Execution time.
        """
        return self._time_call(
            super().time_reversal,
            data=data,
            domain=domain,
            sensors=sensors,
            sources=sources,
            ts=ts,
            record=record,
            data_layout=data_layout,
            **solver_kwargs,
        )

    def adjoint(
        self,
        data,
        domain,
        sensors,
        sources,
        ts,
        *,
        record: str = "p_final",
        data_layout: str = "auto",
        **solver_kwargs,
    ) -> Tuple[np.ndarray, float]:
        """
        Same as KWaveSolver.adjoint, but returns (result, seconds).

        Parameters
        ----------
        data : array-like
            Sensor measurements.
        domain : Domain
            Reconstruction domain.
        sensors : array-like
            Sensor mask.
        sources : array-like
            Source mask.
        ts : array-like, shape (Nt,)
            Time grid.
        record : str, default="p_final"
            k-Wave field to return.
        data_layout : {"auto", "ns_nt", "nt_ns"}, default="auto"
            Sensor-data layout interpretation.
        **solver_kwargs
            Additional solver keyword arguments.

        Returns
        -------
        result : np.ndarray
            Adjoint result.
        seconds : float
            Execution time.
        """
        return self._time_call(
            super().adjoint,
            data=data,
            domain=domain,
            sensors=sensors,
            sources=sources,
            ts=ts,
            record=record,
            data_layout=data_layout,
            **solver_kwargs,
        )
