from __future__ import annotations

from typing import Any, Union, Tuple
import os
import re
import sys
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


def _patch_cpp_simulation_stale_hdf5() -> None:
    """
    Patch stale HDF5 handling in k-wave-python.

    Notes
    -----
    Even at k-wave-python>=0.6.2, ``CppSimulation._write_hdf5`` doesn't remove
    a pre-existing file before writing. Reusing the same temp path across
    multiple forward calls — which happens whenever a single ``KWaveSolver``
    runs forward twice in a process, e.g. inside ``HybridSolver`` — then
    fails with "name already exists". This wrapper deletes the stale file
    first and delegates otherwise.

    Idempotent: a guard attribute prevents re-patching the same class twice.
    """
    from kwave.solvers.cpp_simulation import CppSimulation

    if getattr(CppSimulation._write_hdf5, "_beamax_stale_hdf5_patch", False):
        return

    _orig_write = CppSimulation._write_hdf5

    def _patched_write(self, filepath):
        """Remove an existing HDF5 file before delegating to k-Wave."""
        if os.path.exists(filepath):
            os.remove(filepath)
        _orig_write(self, filepath)

    _patched_write._beamax_stale_hdf5_patch = True
    CppSimulation._write_hdf5 = _patched_write


_patch_cpp_simulation_stale_hdf5()


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
    path came from beamax/user configuration and must be passed to
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
    - Forward solves run on the C++ backend by default. Time-reversal falls
      back to the pure-Python backend until the upstream ``CppSimulation``
      path ships source preprocessing for time-varying pressure sources.
      The adjoint also uses the Python backend and applies the Appendix-B
      source and terminal-field scalings needed for the discrete transpose.
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
        self._solver_kwargs: dict[str, Any]
        if simulation_options is not None:
            # Legacy path: convert old option objects to unified kwargs
            self._solver_kwargs = options_to_kwargs(
                simulation_options, execution_options
            )
            self._explicit_solver_options = set(self._solver_kwargs)
        elif kwargs:
            self._solver_kwargs = dict(kwargs)
            self._explicit_solver_options = set(kwargs)
        else:
            self._solver_kwargs = dict(
                pml_inside=False,
                pml_size=20,
                smooth_p0=False,
                backend="cpp",
                device="cpu",
                debug=True,
            )
            self._explicit_solver_options = set()

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
        if ts.ndim != 1 or ts.size < 2:
            raise ValueError("ts must be one-dimensional with at least two points.")
        dt_values = np.diff(ts)
        if (
            not np.all(np.isfinite(ts))
            or np.any(dt_values <= 0)
            or not np.allclose(dt_values, dt_values[0], rtol=1e-6, atol=0.0)
        ):
            raise ValueError(
                "ts must be finite, strictly increasing, and uniformly spaced."
            )
        kgrid = kWaveGrid(N=domain.N, spacing=domain.dx)

        kgrid.setTime(len(ts), dt_values[0])
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
        force_python: bool = False,
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

        kwargs = self._kwargs_for_domain(domain)
        if force_python:
            kwargs["backend"] = "python"

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
            alpha_power=(
                None
                if domain.alpha_power_array is None
                else np.asarray(domain.alpha_power_array)
            ),
        )

        if kwargs.get("backend") == "cpp":
            _configure_cpp_binary_kwargs(kwargs)

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
            Sensor time series ``(Nt, Ns)``. Sensor channels follow NumPy C
            mask order for both the Python and standalone C++/CUDA backends.
        """
        p0_array = np.asarray(p0)
        if tuple(p0_array.shape) != domain.N:
            raise ValueError(f"p0 must have shape {domain.N}; got {p0_array.shape}.")
        if not np.all(np.isfinite(p0_array)):
            raise ValueError("p0 must contain only finite values.")
        source = kSource()
        source.p0 = p0_array

        sensor_mask = self._validate_mask(np.asarray(sensors), domain, name="sensors")
        backend = str(self._solver_kwargs.get("backend", "cpp")).lower()
        if backend == "cpp" and record != "p":
            raise ValueError("The standalone C++ backend supports record='p' only.")
        sensor = (
            kSensor(mask=sensor_mask)
            if backend == "cpp"
            else kSensor(mask=sensor_mask, record=[record])
        )

        result = self._run_simulation(domain, ts, source, sensor)

        out = np.array(result[record])
        nt = len(ts)
        ns = int(np.count_nonzero(sensor_mask))
        if out.ndim == 2 and out.shape == (ns, nt):
            out = out.T

        # The standalone C++/CUDA binaries enumerate mask points using
        # MATLAB/Fortran linear order, whereas the Python backend and
        # ``Sensor.positions`` use NumPy C order.  Leaving the C++ channels in
        # Fortran order is invisible for a 2D detector line, but swaps the two
        # tangential axes of a 3D detector plane when those data are injected
        # by the Python TR/adjoint backend or consumed by MSGB.  Expose one
        # canonical (C-order) channel convention from the public wrapper.
        if backend == "cpp" and out.ndim == 2 and out.shape == (nt, ns):
            out = out[:, self._cpp_sensor_channels_to_c_order(sensor_mask)]

        return out

    @staticmethod
    def _cpp_sensor_channels_to_c_order(sensor_mask: np.ndarray) -> np.ndarray:
        """Return indices that reorder C++ mask channels into NumPy C order.

        k-Wave's standalone binaries follow MATLAB/Fortran linear indexing for
        mask points.  The returned permutation ``perm`` is intended for
        ``data[..., perm]`` when the final data axis is currently in that
        Fortran order.
        """
        mask = np.asarray(sensor_mask) != 0
        coords_c = np.argwhere(mask)
        flat_f = np.flatnonzero(mask.ravel(order="F"))
        coords_f = np.column_stack(np.unravel_index(flat_f, mask.shape, order="F"))
        f_lookup = {tuple(coord): idx for idx, coord in enumerate(coords_f)}
        return np.asarray([f_lookup[tuple(coord)] for coord in coords_c], dtype=int)

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
        sensor_mask = self._validate_mask(np.asarray(sensors), domain, name="sensors")
        source_mask = self._validate_mask(np.asarray(sources), domain, name="sources")
        sensor_data = self._coerce_sensor_data_layout(
            data,
            source_mask,
            data_layout=data_layout,
            op_name="time_reversal",
        )

        sensor_data_rev = np.flip(sensor_data, axis=1)

        src = kSource()
        setattr(src, "p", sensor_data_rev)
        setattr(src, "p_mask", source_mask)
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
        Discrete k-Wave adjoint following Arridge et al., Appendix B.

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
        record : {"p_final"}
            Terminal pressure field. Other records cannot be converted to the
            initial-pressure transpose and are therefore rejected.

        Returns
        -------
        np.ndarray
            Euclidean discrete adjoint image ``A.T @ data`` with shape
            ``domain.N``.

        Notes
        -----
        The returned field is the algebraic transpose under unweighted
        discrete sums. To represent a one-cell planar detector under the
        thesis's continuous surface and volume rectangle rules, multiply this
        result by ``dt / dx_normal``.

        The source is Eq. (B.2) of Arridge et al. (2016). k-Wave's additive
        pressure-source preprocessing multiplies a user source by
        ``2*dt/(d*c*dx)`` before adding it to each split density field.
        Consequently the user source must be

        ``rho_source*c_source*dx/(4*dt) * beta``,

        and the terminal pressure must be divided pointwise by
        ``c**2*rho``. ``additive-no-correction`` is intentional: the optional
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

        ts_array = np.asarray(ts, dtype=float)
        if ts_array.ndim != 1 or ts_array.size < 2:
            raise ValueError("ts must be one-dimensional with at least two points.")
        dt_values = np.diff(ts_array)
        if (
            not np.all(np.isfinite(ts_array))
            or np.any(dt_values <= 0.0)
            or not np.allclose(dt_values, dt_values[0], rtol=1e-6, atol=0.0)
        ):
            raise ValueError(
                "ts must be finite, strictly increasing, and uniformly spaced."
            )
        dt = float(dt_values[0])

        source_mask = self._validate_mask(np.asarray(sources), domain, name="sources")
        sensor_mask = self._validate_mask(np.asarray(sensors), domain, name="sensors")
        sensor_data = self._coerce_sensor_data_layout(
            data,
            source_mask,
            data_layout=data_layout,
            op_name="adjoint",
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

        # k-wave-python 0.6.2's unified C++ path writes source.p directly to
        # HDF5 and does not request p_final. The Python backend performs the
        # documented time-varying source preprocessing and returns the
        # terminal field needed by the Appendix-B construction.
        out = self._run_simulation(domain, ts, src, sensor, force_python=True)
        terminal_pressure = np.asarray(out[record])
        if tuple(terminal_pressure.shape) != domain.N:
            raise ValueError(
                "k-Wave returned an adjoint field with unexpected shape "
                f"{terminal_pressure.shape}; expected {domain.N}."
            )
        return terminal_pressure / (sound_speed**2 * density)


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
