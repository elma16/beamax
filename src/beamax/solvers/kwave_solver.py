from __future__ import annotations

from typing import Union, Tuple
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


def _patch_cpp_simulation_stale_hdf5():
    """
    Work around k-wave-python CppSimulation._write_hdf5 not removing stale
    HDF5 files before writing, causing "name already exists" errors when the
    same temp path is reused.
    """
    from kwave.solvers.cpp_simulation import CppSimulation

    _orig_write = CppSimulation._write_hdf5

    def _patched_write(self, filepath):
        if os.path.exists(filepath):
            os.remove(filepath)
        _orig_write(self, filepath)

    CppSimulation._write_hdf5 = _patched_write


def _patch_kwave_binary_path():
    """
    Allow overriding the k-wave-python bundled binary via the
    ``BEAMAX_KWAVE_BINARY_PATH`` env var.

    The pip-installed darwin binary at ``kwave.BINARY_PATH/kspaceFirstOrder-OMP``
    silently ignores ``alpha_coeff``/``alpha_power`` (no absorption applied).
    A locally built binary from upstream k-wave-omp source applies absorption
    correctly. ``cpp_simulation._execute`` (k-wave-python ≤ current) hardcodes
    ``kwave.BINARY_PATH`` and ignores the ``binary_path`` option, so the only
    way to redirect it from user code is to mutate ``kwave.BINARY_PATH`` at
    import time. Should be removed once k-wave-python honours the option.
    """
    override = os.environ.get("BEAMAX_KWAVE_BINARY_PATH")
    if not override:
        return
    override_path = Path(override)
    if not (override_path / "kspaceFirstOrder-OMP").exists():
        return
    import kwave

    kwave.BINARY_PATH = override_path


_patch_cpp_simulation_stale_hdf5()
_patch_kwave_binary_path()


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

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from beamax import Domain, Sensor
    >>> from beamax.solvers import KWaveSolver
    >>> domain = Domain(N=(64, 64), dx=(1e-3, 1e-3), c=1500.0, periodic=(False, False))  # doctest: +SKIP
    >>> solver = KWaveSolver(pml_size=10, device="cpu")  # doctest: +SKIP
    """

    _hdf5_compat_done = False

    def __init__(
        self,
        simulation_options=None,
        execution_options=None,
        **kwargs,
    ):
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
    def _ensure_macos_hdf5_compat(cls) -> None:
        """
        Create DYLD symlinks so the k-Wave OMP binary (linked against
        libhdf5.310) can load on systems that only ship libhdf5.320.
        """
        if cls._hdf5_compat_done or sys.platform != "darwin":
            cls._hdf5_compat_done = True
            return

        import kwave

        binary_path = kwave.BINARY_PATH / "kspaceFirstOrder-OMP"
        if not binary_path.exists():
            cls._hdf5_compat_done = True
            return

        try:
            out = subprocess.run(
                ["otool", "-L", str(binary_path)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (subprocess.SubprocessError, OSError):
            cls._hdf5_compat_done = True
            return

        refs = re.findall(
            r"^\s+(/.*libhdf5(?:_hl)?\.310\.dylib)\s", out, flags=re.MULTILINE
        )
        if not refs:
            cls._hdf5_compat_done = True
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
            cls._hdf5_compat_done = True
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
        cls._hdf5_compat_done = True

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

        Args:
            domain: Domain object.
            ts: Time steps.
            source: Source object.
            sensor: Sensor object.
            force_python: Force the python backend even when cpp is configured.
                Used for time-varying source simulations (TR/adjoint) where
                the v0.6.1 cpp backend has missing source-term preprocessing.

        Returns:
            Pressure field.
        """
        kgrid = self._create_kgrid(domain, ts)

        kwargs = dict(self._solver_kwargs)
        if force_python:
            kwargs["backend"] = "python"

        if kwargs.get("backend") == "cpp":
            self._ensure_macos_hdf5_compat()

        medium = kWaveMedium(
            sound_speed=domain.sound_speed_array,
            density=domain.density_array,
            alpha_coeff=domain.alpha_coeff,
            alpha_power=domain.alpha_power,
        )

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

        Classic k-Wave TR: enforce p(x_s,t) = sensor_data(t,x_s) (Dirichlet).
        Accepts either (Ns, Nt) or (Nt, Ns) input and returns `record`.
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
        super().__init__(*args, **kwargs)
        if mode not in {"stdout", "wall"}:
            raise ValueError("mode must be 'stdout' or 'wall'")
        self._mode = mode

    def _time_call(self, fn, *a, **k) -> Tuple[np.ndarray, float]:
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
