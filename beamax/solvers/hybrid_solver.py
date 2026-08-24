"""
Hybrid solver for combining low- and high-frequency propagation backends.

The high-frequency component is typically handled by MSGB, while the
low-frequency component is supplied through a small adapter callable API.
"""

import warnings
import math
import numpy as np
import jax.numpy as jnp
import equinox as eqx
from scipy.signal.windows import kaiser
from scipy.ndimage import zoom
from typing import Any, Literal, Optional, Tuple, Callable
from dataclasses import dataclass

from beamax.solvers.hybrid_solver_utils import split_frequency_components
from beamax.geometry import Domain, Sensor
from beamax.solvers.msgb_solvers.msgb_solver import MSGBSolver
from beamax import utils
from beamax.transforms import MSWPT


HybridOperationName = Literal["forward", "time_reversal", "adjoint"]
HybridOperation = Callable[[jnp.ndarray, "HybridContext"], Any]
_HYBRID_OPERATION_NAMES: tuple[HybridOperationName, ...] = (
    "forward",
    "time_reversal",
    "adjoint",
)


__all__ = ["HybridBackend", "HybridContext", "HybridSolver", "HybridSolverConfig"]


@dataclass(frozen=True)
class HybridSolverConfig:
    r"""
    Configuration for hybrid solver.

    Parameters
    ----------
    box_corners : jnp.ndarray, optional
        Indices defining LF/HF split region in Fourier space
    cutoff_freq : float, optional
        Alternative to box_corners - split by frequency magnitude
    downsample : bool
        Whether to solve LF on downsampled grid
    use_pow2 : bool
        Round downsampled size to power of 2
    input_type : str
        "spatial" or "fourier" - domain of input data
    interp_method : str
        "fourier" (periodic domains) or "zoom" (non-periodic)
    dt_oversample : int
        Number of extra time steps for windowing (forward only)
    beta : float
        Kaiser window shape parameter. Larger values sharpen the transition.
    order : int
        Spline order for zoom interpolation; one of $0,1,\ldots,5$.
    window_type : str
        "kaiser" or "tukey" - windowing function type
    use_windowing : bool
        Whether to apply windowing (typically True for forward, False for TR)
    use_time_extension : bool
        Whether to extend time array (typically True for forward, False for TR)
    """

    box_corners: Optional[jnp.ndarray] = None
    cutoff_freq: Optional[float] = None
    downsample: bool = True
    use_pow2: bool = True
    input_type: str = "spatial"
    interp_method: str = "fourier"
    dt_oversample: int = 30
    beta: float = 12.0
    order: int = 3
    window_type: str = "kaiser"
    use_windowing: bool = True
    use_time_extension: bool = True

    def __post_init__(self):
        r"""
        Validate mutually exclusive split options and interpolation settings.

        Raises
        ------
        ValueError
            If neither or both frequency split definitions are provided, if
            ``order`` is not one of $0,1,\ldots,5$, or if ``window_type`` is
            unknown.
        """
        has_corners = self.box_corners is not None
        has_freq = self.cutoff_freq is not None
        if not (has_corners or has_freq):
            raise ValueError("Must provide either box_corners or cutoff_freq")
        if has_corners and has_freq:
            raise ValueError("Provide only one of box_corners or cutoff_freq")
        if self.box_corners is not None:
            corners = np.asarray(self.box_corners)
            if corners.shape != (2,) or not np.issubdtype(corners.dtype, np.integer):
                raise ValueError(
                    "box_corners must contain exactly two integer indices."
                )
            if np.any(corners < 0) or corners[0] > corners[1]:
                raise ValueError("box_corners must be non-negative and ordered.")
        if self.cutoff_freq is not None and (
            not math.isfinite(float(self.cutoff_freq)) or self.cutoff_freq <= 0
        ):
            raise ValueError("cutoff_freq must be finite and positive.")
        if self.input_type not in {"spatial", "fourier"}:
            raise ValueError("input_type must be 'spatial' or 'fourier'.")
        if self.interp_method not in {"fourier", "zoom"}:
            raise ValueError("interp_method must be 'fourier' or 'zoom'.")
        if (
            isinstance(self.dt_oversample, bool)
            or not isinstance(self.dt_oversample, int)
            or self.dt_oversample < 0
        ):
            raise ValueError("dt_oversample must be a non-negative integer.")
        if not math.isfinite(float(self.beta)) or self.beta <= 0:
            raise ValueError("beta must be finite and positive.")
        if (
            isinstance(self.order, bool)
            or not isinstance(self.order, int)
            or not 0 <= self.order <= 5
        ):
            raise ValueError(f"order must be 0-5, got {self.order}")
        if self.window_type not in ("kaiser", "tukey"):
            raise ValueError(
                f"window_type must be 'kaiser' or 'tukey', got {self.window_type!r}"
            )


@dataclass(frozen=True)
class HybridContext:
    """Runtime inputs passed to a low-frequency hybrid backend.

    :class:`HybridSolver` owns splitting, optional downsampling, windowing,
    interpolation, and merging. The context contains only the geometry and
    shapes needed by the backend adapter.

    Attributes
    ----------
    domain : Domain
        Full-resolution output domain. For ``forward`` this is the physical
        simulation domain; for inverse operations this is the reconstruction
        domain.
    component_domain : Domain
        Domain for the LF component passed to the backend.
    component_sensor_mask : jnp.ndarray
        Sensor mask aligned to ``component_domain``.
    ts : jnp.ndarray
        Time grid passed to the LF operation.
    target_shape : tuple[int, ...]
        Shape the LF result will be interpolated/truncated to before merging.
    sources : object, optional
        Source geometry for inverse operations.
    """

    domain: Domain
    component_domain: Domain
    component_sensor_mask: jnp.ndarray
    ts: jnp.ndarray
    target_shape: Tuple[int, ...]
    sources: Any = None


@dataclass(frozen=True)
class HybridBackend:
    """
    Adapter for a low-frequency backend used by :class:`HybridSolver`.

    Parameters
    ----------
    forward, time_reversal, adjoint : callable, optional
        Operation callables with signature ``callable(component_array, context)
        -> array``. At least one operation must be provided.
    name : str
        Human-readable backend name used in error messages.
    """

    forward: Optional[HybridOperation] = None
    time_reversal: Optional[HybridOperation] = None
    adjoint: Optional[HybridOperation] = None
    name: str = "low-frequency backend"

    def __post_init__(self) -> None:
        """Validate that the backend exposes at least one operation."""
        if not any(getattr(self, op) is not None for op in _HYBRID_OPERATION_NAMES):
            raise ValueError(
                "HybridBackend requires at least one operation: "
                "forward, time_reversal, or adjoint."
            )

    @property
    def operations(self) -> Tuple[str, ...]:
        """Operation names implemented by this backend."""
        return tuple(
            op for op in _HYBRID_OPERATION_NAMES if getattr(self, op) is not None
        )

    def supports(self, operation: HybridOperationName) -> bool:
        """Return whether this backend implements ``operation``."""
        self._validate_operation_name(operation)
        return getattr(self, operation) is not None

    def require(self, operation: HybridOperationName) -> HybridOperation:
        """
        Return an operation callable or raise a clear missing-operation error.

        Parameters
        ----------
        operation : {"forward", "time_reversal", "adjoint"}
            Operation required by the hybrid solve.
        """
        self._validate_operation_name(operation)
        solver_op = getattr(self, operation)
        if solver_op is None:
            available = ", ".join(self.operations) or "none"
            raise NotImplementedError(
                f"{self.name} does not implement LF operation {operation!r}. "
                f"Available operations: {available}."
            )
        return solver_op

    @staticmethod
    def _validate_operation_name(operation: str) -> None:
        if operation not in _HYBRID_OPERATION_NAMES:
            allowed = ", ".join(_HYBRID_OPERATION_NAMES)
            raise ValueError(
                f"Unknown hybrid operation {operation!r}; expected {allowed}."
            )

    @classmethod
    def from_beamax_solver(
        cls, solver: Any, *, name: Optional[str] = None
    ) -> "HybridBackend":
        """
        Wrap a beamax-style solver object as a low-frequency backend.

        The wrapped solver may implement any subset of ``forward``,
        ``time_reversal``, and ``adjoint``. Each method is called with the
        current component-domain objects from :class:`HybridContext`.
        """
        backend_name = name or solver.__class__.__name__
        operations: dict[str, HybridOperation] = {}

        if callable(getattr(solver, "forward", None)):

            def forward(component: jnp.ndarray, context: HybridContext) -> Any:
                return solver.forward(
                    component,
                    context.component_domain,
                    context.component_sensor_mask,
                    context.ts,
                )

            operations["forward"] = forward

        if callable(getattr(solver, "time_reversal", None)):

            def time_reversal(component: jnp.ndarray, context: HybridContext) -> Any:
                return solver.time_reversal(
                    component,
                    context.component_domain,
                    context.component_sensor_mask,
                    context.sources,
                    context.ts,
                )

            operations["time_reversal"] = time_reversal

        if callable(getattr(solver, "adjoint", None)):

            def adjoint(component: jnp.ndarray, context: HybridContext) -> Any:
                return solver.adjoint(
                    component,
                    context.component_domain,
                    context.component_sensor_mask,
                    context.sources,
                    context.ts,
                )

            operations["adjoint"] = adjoint

        return cls(
            forward=operations.get("forward"),
            time_reversal=operations.get("time_reversal"),
            adjoint=operations.get("adjoint"),
            name=backend_name,
        )


def _interpolate_fourier(
    data: jnp.ndarray, target_shape: Tuple[int, ...]
) -> jnp.ndarray:
    """Resize spatial data with unitary Fourier padding or cropping."""
    if len(target_shape) != data.ndim:
        raise ValueError(
            f"target_shape must have length {data.ndim}, got {target_shape}."
        )
    return utils.interpolate_fourier(
        data, target_shape, input_type="spatial", output_type="spatial"
    ).real


def _interpolate_zoom(
    data: jnp.ndarray, target_shape: Tuple[int, ...], *, order: int
) -> jnp.ndarray:
    """Resize spatial data with spline interpolation."""
    if len(target_shape) != data.ndim:
        raise ValueError(
            f"target_shape must have length {data.ndim}, got {target_shape}."
        )
    zoom_factors = tuple(
        output / input_ for output, input_ in zip(target_shape, data.shape)
    )
    return jnp.asarray(zoom(data, zoom_factors, order=order))


class HybridSolver(eqx.Module):
    """
    Hybrid MSGB / low-frequency backend solver.

    Splits the input pressure in frequency: a fast high-frequency solver
    (typically :class:`beamax.solvers.MSGBSolver`) handles the sparse, highly
    oscillatory content, while a configurable low-frequency backend handles
    the smooth residual, optionally on a coarser grid. The two results are
    added back together on the target grid.

    Parameters
    ----------
    hf_solver : MSGBSolver or object
        Solver used for the high-frequency component.
    lf_backend : HybridBackend
        Low-frequency backend adapter. Each operation callable receives the
        LF component and a :class:`HybridContext`.
    config : HybridSolverConfig
        Frequency-split and windowing configuration.

    Notes
    -----
    The LF backend only needs to provide at least one operation through
    :class:`HybridBackend`. Missing operations fail when called, not at hybrid
    construction time.
    """

    lf_backend: HybridBackend = eqx.field(static=True)
    hf_solver: Any
    config: HybridSolverConfig = eqx.field(static=True)

    def __init__(
        self,
        *,
        hf_solver: Any,
        lf_backend: HybridBackend,
        config: Optional[HybridSolverConfig] = None,
        **config_kwargs,
    ):
        """
        Initialize hybrid solver.

        Parameters
        ----------
        hf_solver : MSGBSolver or object
            High-frequency solver.
        lf_backend : HybridBackend
            Low-frequency backend adapter.
        config : HybridSolverConfig, optional
            Configuration object.
        **config_kwargs
            Alternative: pass config parameters as kwargs.
        """
        self.lf_backend = lf_backend
        self.hf_solver = hf_solver

        if config is not None and config_kwargs:
            raise ValueError("Pass either config or keyword configuration, not both.")
        if config is None:
            config = HybridSolverConfig(**config_kwargs)
        self.config = config

    @staticmethod
    def create_with_domain(
        *,
        hf_solver: Any,
        lf_backend: HybridBackend,
        domain: Domain,
        config: Optional[HybridSolverConfig] = None,
        **config_kwargs,
    ) -> "HybridSolver":
        """
        Factory method with automatic interpolation selection based on domain.

        Uses Fourier interpolation for periodic domains, spline interpolation
        for non-periodic domains.

        Parameters
        ----------
        hf_solver : MSGBSolver or object
            High-frequency solver.
        lf_backend : HybridBackend
            Low-frequency backend adapter.
        domain : Domain
            Domain whose periodic flags determine interpolation choice.
        config : HybridSolverConfig, optional
            Explicit configuration.
        **config_kwargs
            Configuration options used when ``config`` is ``None``.

        Returns
        -------
        HybridSolver
            Configured hybrid solver.
        """
        if config is not None and config_kwargs:
            raise ValueError("Pass either config or keyword configuration, not both.")
        if config is None:
            config_dict = config_kwargs.copy()

            if "interp_method" not in config_dict:
                if all(domain.periodic):
                    config_dict["interp_method"] = "fourier"
                else:
                    config_dict["interp_method"] = "zoom"
                    if "order" not in config_dict:
                        config_dict["order"] = 3

            config = HybridSolverConfig(**config_dict)

        return HybridSolver(
            hf_solver=hf_solver,
            lf_backend=lf_backend,
            config=config,
        )

    def _extend_time(self, ts: jnp.ndarray) -> jnp.ndarray:
        """
        Extend time array for oversampling when configured.

        Parameters
        ----------
        ts : jnp.ndarray, shape (Nt,)
            Original time grid.

        Returns
        -------
        jnp.ndarray
            Original or extended time grid.
        """
        if (
            not self.config.use_time_extension
            or not self.config.downsample
            or self.config.dt_oversample == 0
        ):
            return ts

        if ts.ndim != 1 or ts.shape[0] < 2:
            raise ValueError(
                "ts must be a one-dimensional array with at least two points."
            )
        dt = ts[1] - ts[0]
        if (
            not bool(jnp.isfinite(dt))
            or not bool(dt > 0)
            or not bool(jnp.allclose(jnp.diff(ts), dt))
        ):
            raise ValueError(
                "ts must be finite, strictly increasing, and uniformly spaced."
            )
        Nt = len(ts)
        Nt_extended = Nt + self.config.dt_oversample
        return ts[0] + jnp.arange(0, Nt_extended) * dt

    def _apply_window(self, data: jnp.ndarray) -> jnp.ndarray:
        """
        Apply a window taper to the end of a time series when configured.

        Parameters
        ----------
        data : jnp.ndarray
            Time-leading array.

        Returns
        -------
        jnp.ndarray
            Windowed data, or ``data`` unchanged when windowing is disabled.
        """
        if (
            not self.config.use_windowing
            or not self.config.downsample
            or self.config.dt_oversample == 0
        ):
            return data

        if self.config.window_type == "kaiser":
            return self._apply_kaiser_window(data)
        elif self.config.window_type == "tukey":
            return self._apply_tukey_window(data)
        else:
            return data

    def _apply_kaiser_window(self, data: jnp.ndarray) -> jnp.ndarray:
        """
        Apply a Kaiser taper to the final time samples.

        Parameters
        ----------
        data : jnp.ndarray
            Time-leading array.

        Returns
        -------
        jnp.ndarray
            Tapered data.
        """
        Nt = data.shape[0]
        if self.config.dt_oversample <= 0:
            return data

        taper_len = int(jnp.minimum(self.config.dt_oversample, Nt))
        if taper_len == 0:
            return data

        window = jnp.ones(Nt)
        kaiser_tail = kaiser(2 * taper_len, self.config.beta)[taper_len:]
        window = window.at[-taper_len:].set(kaiser_tail)

        if data.ndim > 1:
            return data * window.reshape((Nt,) + (1,) * (data.ndim - 1))
        return data * window

    def _apply_tukey_window(self, data: jnp.ndarray, alpha: float = 0.5) -> jnp.ndarray:
        """
        Apply a Tukey taper to the final time samples.

        Parameters
        ----------
        data : jnp.ndarray
            Time-leading array.
        alpha : float, default=0.5
            Fraction of ``dt_oversample`` used for the taper length.

        Returns
        -------
        jnp.ndarray
            Tapered data.
        """
        Nt = data.shape[0]
        if self.config.dt_oversample <= 0:
            return data

        taper_len = int(jnp.minimum(max(1, int(alpha * self.config.dt_oversample)), Nt))
        if taper_len == 0:
            return data

        window = jnp.ones(Nt)
        taper_indices = jnp.arange(taper_len)
        taper = 0.5 * (1 + jnp.cos(jnp.pi * taper_indices / taper_len))
        window = window.at[-taper_len:].set(taper)

        if data.ndim > 1:
            return data * window.reshape((Nt,) + (1,) * (data.ndim - 1))
        return data * window

    def _run_lf_backend(
        self,
        operation: HybridOperationName,
        lf_data: jnp.ndarray,
        context: HybridContext,
        apply_windowing: bool = True,
        apply_interpolation: bool = True,
    ) -> jnp.ndarray:
        """
        Run an LF backend operation and apply hybrid-owned post-processing.

        Parameters
        ----------
        operation : {"forward", "time_reversal", "adjoint"}
            Backend operation to dispatch.
        lf_data : jnp.ndarray
            Low-frequency input data.
        context : HybridContext
            Stable adapter context.
        apply_windowing : bool
            Whether to apply windowing (typically True for forward, False for TR).
        apply_interpolation : bool
            Whether to interpolate to ``context.target_shape`` when downsampled.

        Returns
        -------
        jnp.ndarray
            LF result post-processed for merging with the HF result.
        """
        solver_method = self.lf_backend.require(operation)
        lf_result = jnp.asarray(solver_method(lf_data, context))

        if apply_windowing and self.config.downsample:
            lf_result = self._apply_window(lf_result)

        if apply_interpolation and self.config.downsample:
            source_shape = tuple(int(n) for n in lf_result.shape)
            if self.config.interp_method == "fourier":
                lf_result = _interpolate_fourier(lf_result, context.target_shape)
            else:
                lf_result = _interpolate_zoom(
                    lf_result, context.target_shape, order=self.config.order
                )
            if operation == "forward" and self.config.interp_method == "fourier":
                # Correct unitary-crop inflation when the field and detector
                # records have different numbers of spatial axes.
                output_ratio = math.prod(
                    target / source
                    for source, target in zip(source_shape, context.target_shape)
                )
                domain_ratio = math.prod(
                    full / coarse
                    for full, coarse in zip(
                        context.domain.N, context.component_domain.N
                    )
                )
                lf_result = lf_result * math.sqrt(output_ratio / domain_ratio)

        return lf_result

    def _split_frequencies(
        self,
        data: jnp.ndarray,
        mask: jnp.ndarray,
        wpt: MSWPT,
        domain: Domain,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Domain]:
        """
        Split data into high- and low-frequency components.

        Parameters
        ----------
        data : jnp.ndarray
            Input array to split.
        mask : jnp.ndarray
            Sensor mask aligned with ``data``.
        wpt : MSWPT
            Wave-packet transform used for the split.
        domain : Domain
            Physical domain.

        Returns
        -------
        p0_HF : jnp.ndarray
            High-frequency component.
        p0_LF : jnp.ndarray
            Low-frequency component.
        ds_mask : jnp.ndarray
            Possibly downsampled sensor mask.
        ds_domain : Domain
            Possibly downsampled domain.
        """
        return split_frequency_components(
            p0=data,
            sensors_mask=mask,
            input_type=self.config.input_type,
            output_type="spatial",
            wpt=wpt,
            box_corners=self.config.box_corners,
            windowing=wpt.windowing,
            domain=domain,
            cutoff_freq=self.config.cutoff_freq,
            downsample=self.config.downsample,
            use_pow2=self.config.use_pow2,
        )

    def _solve_hf(
        self,
        data: jnp.ndarray,
        domain: Domain,
        sensors: Sensor,
        mask: jnp.ndarray,
        ts: jnp.ndarray,
        wpt: MSWPT,
        method: str = "forward",
        **kwargs,
    ) -> jnp.ndarray:
        """
        Solve HF component with appropriate solver and method.

        Parameters
        ----------
        data : jnp.ndarray
            HF data
        domain : Domain
        sensors : Sensor
        mask : jnp.ndarray
        ts : jnp.ndarray
        wpt : MSWPT
        method : str
            Solver method name: "forward", "time_reversal", or "adjoint"
        **kwargs
            Method-specific parameters
        """
        if isinstance(self.hf_solver, MSGBSolver):
            if method == "forward":
                return self.hf_solver.forward(data, domain, sensors, ts, wpt, **kwargs)
            if method == "time_reversal":
                return self.hf_solver.time_reversal(
                    data,
                    domain,
                    sensors,
                    kwargs["sources"],
                    ts,
                    kwargs["data_domain"],
                    kwargs["data_wpt"],
                )
            if method == "adjoint":
                return self.hf_solver.adjoint(
                    data,
                    domain,
                    sensors,
                    kwargs["sources"],
                    ts,
                    kwargs["data_domain"],
                    kwargs["data_wpt"],
                )
            raise ValueError(f"Unknown HF method: {method!r}")

        solver_fn = getattr(self.hf_solver, method)
        if method == "forward":
            return solver_fn(data, domain, mask, ts, **kwargs)
        if method in ("time_reversal", "adjoint"):
            return solver_fn(data, domain, mask, kwargs["sources"], ts)
        raise ValueError(f"Unknown HF method: {method!r}")

    def _validate_configuration(self, domain: Domain):
        """
        Warn if the hybrid configuration may cause boundary artifacts.

        Parameters
        ----------
        domain : Domain
            Domain whose periodicity is checked against interpolation method.
        """
        if (
            self.config.interp_method == "fourier"
            and self.config.downsample
            and not all(domain.periodic)
        ):
            warnings.warn(
                "Using Fourier interpolation with non-periodic domain may cause "
                "boundary artifacts. Consider interp_method='zoom' or use "
                "HybridSolver.create_with_domain() for automatic selection.",
                UserWarning,
                stacklevel=3,
            )

    def forward(
        self,
        p0: jnp.ndarray,
        domain: Domain,
        sensors: Sensor,
        ts: jnp.ndarray,
        wpt: MSWPT,
    ) -> jnp.ndarray:
        """
        Forward solve with hybrid approach.

        Parameters
        ----------
        p0 : jnp.ndarray
            Initial pressure field
        domain : Domain
            Computational domain
        sensors : Sensor
            Sensor geometry
        ts : jnp.ndarray
            Time points
        wpt : MSWPT
            Wavelet packet transform for frequency splitting

        Returns
        -------
        jnp.ndarray
            Sensor data, shape (Nt, Ns)
        """
        self._validate_configuration(domain)

        mask = sensors.binary_mask if isinstance(sensors, Sensor) else sensors

        p0_HF, p0_LF, ds_mask, ds_domain = self._split_frequencies(
            p0, mask, wpt, domain
        )
        p0_HF, p0_LF = p0_HF.real, p0_LF.real

        ts_extended = self._extend_time(ts)
        original_Nt = len(ts)

        hf_result = self._solve_hf(
            p0_HF, domain, sensors, mask, ts_extended, wpt, method="forward"
        )

        hf_result = self._apply_window(hf_result)

        context = HybridContext(
            domain=domain,
            component_domain=ds_domain,
            component_sensor_mask=ds_mask,
            ts=ts_extended,
            target_shape=tuple(hf_result.shape),
        )

        lf_result = self._run_lf_backend(
            "forward",
            p0_LF,
            context,
            apply_windowing=True,
            apply_interpolation=True,
        )

        if self.config.downsample and self.config.use_time_extension:
            hf_result = hf_result[:original_Nt, ...]
            lf_result = lf_result[:original_Nt, ...]

        return hf_result + lf_result

    def time_reversal(
        self,
        data: jnp.ndarray,
        domain: Domain,
        sensors: Sensor,
        sources: Sensor,
        ts: jnp.ndarray,
        data_domain: Domain,
        data_wpt: MSWPT,
        img_wpt: MSWPT,
    ) -> jnp.ndarray:
        """
        Time reversal with hybrid approach.

        Parameters
        ----------
        data : jnp.ndarray
            Recorded sensor data
        domain : Domain
            Reconstruction domain
        sensors : Sensor
            Sensor geometry
        sources : Sensor
            Source positions for data injection
        ts : jnp.ndarray
            Time points
        data_domain : Domain
            Domain where data was recorded
        data_wpt : MSWPT
            Wavelet transform for data domain
        img_wpt : MSWPT
            Wavelet transform for reconstruction domain

        Returns
        -------
        jnp.ndarray
            Reconstructed initial pressure field

        Notes
        -----
        Time extension/windowing not used for time reversal (applies only to forward).
        """
        self._validate_configuration(domain)

        mask = sensors.binary_mask if isinstance(sensors, Sensor) else sensors

        data_HF, data_LF, ds_mask, ds_domain = self._split_frequencies(
            data, mask, data_wpt, data_domain
        )
        data_HF, data_LF = data_HF.real, data_LF.real

        hf_result = self._solve_hf(
            data_HF,
            domain,
            sensors,
            mask,
            ts,
            data_wpt,
            method="time_reversal",
            sources=sources,
            data_domain=data_domain,
            data_wpt=data_wpt,
        )

        context = HybridContext(
            domain=domain,
            component_domain=ds_domain,
            component_sensor_mask=ds_mask,
            ts=ts,
            target_shape=tuple(hf_result.shape),
            sources=sources,
        )

        lf_result = self._run_lf_backend(
            "time_reversal",
            data_LF,
            context,
            apply_windowing=False,
            apply_interpolation=True,
        )

        return jnp.asarray(hf_result + lf_result)

    def adjoint(
        self,
        data: jnp.ndarray,
        domain: Domain,
        sensors: Sensor,
        sources: Sensor,
        ts: jnp.ndarray,
        data_domain: Domain,
        data_wpt: MSWPT,
        img_wpt: MSWPT,
    ) -> jnp.ndarray:
        """
        Adjoint solve with hybrid approach.

        Similar to time_reversal but uses adjoint method.

        Parameters
        ----------
        data : jnp.ndarray
            Recorded sensor data
        domain : Domain
            Reconstruction domain
        sensors : Sensor
            Sensor geometry
        sources : Sensor
            Source positions for data injection
        ts : jnp.ndarray
            Time points
        data_domain : Domain
            Domain where data was recorded
        data_wpt : MSWPT
            Wavelet transform for data domain
        img_wpt : MSWPT
            Wavelet transform for reconstruction domain

        Returns
        -------
        jnp.ndarray
            Adjoint reconstruction
        """
        self._validate_configuration(domain)

        mask = sensors.binary_mask if isinstance(sensors, Sensor) else sensors

        data_HF, data_LF, ds_mask, ds_domain = self._split_frequencies(
            data, mask, data_wpt, data_domain
        )
        data_HF, data_LF = data_HF.real, data_LF.real

        hf_result = self._solve_hf(
            data_HF,
            domain,
            sensors,
            mask,
            ts,
            img_wpt,
            method="adjoint",
            sources=sources,
            data_domain=data_domain,
            data_wpt=data_wpt,
        )

        context = HybridContext(
            domain=domain,
            component_domain=ds_domain,
            component_sensor_mask=ds_mask,
            ts=ts,
            target_shape=tuple(hf_result.shape),
            sources=sources,
        )

        lf_result = self._run_lf_backend(
            "adjoint",
            data_LF,
            context,
            apply_windowing=False,
            apply_interpolation=True,
        )

        return hf_result + lf_result
