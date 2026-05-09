"""
Generic hybrid solver supporting forward, time_reversal, and adjoint.

Key improvements:
1. _solve_lf_component accepts **kwargs for solver-specific parameters
2. Windowing/interpolation/time-extension are optional
3. Unified pattern for all solver operations
"""

import warnings
import math
import jax.numpy as jnp
import equinox as eqx
from scipy.signal.windows import kaiser
from scipy.ndimage import zoom
from typing import Union, Optional, Tuple, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass

from beamax.solvers.hybrid_solver_utils import split_frequency_components
from beamax.geometry import Domain, Sensor
from beamax.solvers.solverbase import Solver
from beamax.solvers.msgb_solvers.msgb_solver import MSGBSolver
from beamax import utils
from beamax.transforms import MSWPT


__all__ = ["HybridSolver", "HybridSolverConfig"]


@dataclass(frozen=True)
class HybridSolverConfig:
    """
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
        Kaiser window shape parameter (higher = sharper transition)
    order : int
        Spline order for zoom interpolation (0-5)
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
        has_corners = self.box_corners is not None
        has_freq = self.cutoff_freq is not None
        if not (has_corners or has_freq):
            raise ValueError("Must provide either box_corners or cutoff_freq")
        if has_corners and has_freq:
            raise ValueError("Provide only one of box_corners or cutoff_freq")
        if not 0 <= self.order <= 5:
            raise ValueError(f"order must be 0-5, got {self.order}")
        if self.window_type not in ("kaiser", "tukey"):
            raise ValueError(
                f"window_type must be 'kaiser' or 'tukey', got {self.window_type!r}"
            )


class InterpolationStrategy(ABC):
    """Abstract base for spatial interpolation strategies."""

    @abstractmethod
    def interpolate(self, data: jnp.ndarray, target_shape: Tuple) -> jnp.ndarray:
        """Interpolate data to target_shape."""
        pass


class FourierInterpolation(InterpolationStrategy):
    """Fourier-based interpolation (periodic boundaries assumed)."""

    def interpolate(self, data: jnp.ndarray, target_shape: Tuple) -> jnp.ndarray:
        resampled = utils.interpolate_fourier(
            data, target_shape, input_type="spatial", output_type="spatial"
        ).real
        # `utils.interpolate_fourier` uses unitary FFTs, which preserve the
        # discrete L2 norm. For solver outputs we need sample values on the
        # target grid, so compensate each resized axis by sqrt(N_in / N_out).
        scale = math.sqrt(
            math.prod(
                input_len / output_len
                for input_len, output_len in zip(data.shape, target_shape)
            )
        )
        return resampled * scale


class ZoomInterpolation(InterpolationStrategy):
    """Spline-based interpolation (handles non-periodic domains)."""

    def __init__(self, order: int = 3):
        self.order = order

    def interpolate(self, data: jnp.ndarray, target_shape: Tuple) -> jnp.ndarray:
        zoom_factors = tuple(o / i for o, i in zip(target_shape, data.shape))
        return zoom(data, zoom_factors, order=self.order)


class HybridSolver(eqx.Module):
    """
    Hybrid MSGB / grid-based wave solver.

    Splits the input pressure in frequency: a fast high-frequency solver
    (typically :class:`beamax.solvers.MSGBSolver`) handles the sparse, highly
    oscillatory content, while a conventional low-frequency solver
    (typically :class:`beamax.solvers.KWaveSolver`) handles the smooth
    residual — optionally on a coarser grid. The two results are added back
    together on the target grid.

    Parameters
    ----------
    lf_solver : Solver
        Solver used for the low-frequency component. Must implement the
        :class:`beamax.solvers.Solver` interface.
    hf_solver : MSGBSolver or Solver
        Solver used for the high-frequency component.
    config : HybridSolverConfig
        Frequency-split and windowing configuration.

    Notes
    -----
    Both solvers must agree on the global time grid. The LF solver is
    invoked on a possibly-downsampled domain (per ``config``); output is
    interpolated back to the target grid via Fourier or spline interpolation
    depending on whether the domain is periodic.
    """

    lf_solver: Solver
    hf_solver: Union[MSGBSolver, Solver]
    config: HybridSolverConfig = eqx.field(static=True)
    _interpolator: InterpolationStrategy = eqx.field(static=True)

    def __init__(
        self,
        lf_solver: Solver,
        hf_solver: Union[MSGBSolver, Solver],
        config: Optional[HybridSolverConfig] = None,
        **config_kwargs,
    ):
        """
        Initialize hybrid solver.

        Parameters
        ----------
        lf_solver : Solver
            Low-frequency solver
        hf_solver : Solver
            High-frequency solver
        config : HybridSolverConfig, optional
            Configuration object
        **config_kwargs
            Alternative: pass config parameters as kwargs
        """
        self.lf_solver = lf_solver
        self.hf_solver = hf_solver

        if config is None:
            config = HybridSolverConfig(**config_kwargs)
        self.config = config

        # Setup interpolation strategy
        if config.interp_method == "fourier":
            self._interpolator = FourierInterpolation()
        elif config.interp_method == "zoom":
            self._interpolator = ZoomInterpolation(config.order)
        else:
            raise ValueError(
                f"Unknown interp_method: {config.interp_method}. "
                f"Must be 'fourier' or 'zoom'"
            )

    @staticmethod
    def create_with_domain(
        lf_solver: Solver,
        hf_solver: Union[MSGBSolver, Solver],
        domain: Domain,
        config: Optional[HybridSolverConfig] = None,
        **config_kwargs,
    ) -> "HybridSolver":
        """
        Factory method with automatic interpolation selection based on domain.

        Uses Fourier interpolation for periodic domains, spline interpolation
        for non-periodic domains.
        """
        if config is None:
            config_dict = config_kwargs.copy()

            # Auto-select interpolation method if not specified
            if "interp_method" not in config_dict:
                if all(domain.periodic):
                    config_dict["interp_method"] = "fourier"
                else:
                    config_dict["interp_method"] = "zoom"
                    if "order" not in config_dict:
                        config_dict["order"] = 3

            config = HybridSolverConfig(**config_dict)

        return HybridSolver(lf_solver, hf_solver, config)

    def _extend_time(self, ts: jnp.ndarray) -> jnp.ndarray:
        """Extend time array for oversampling (if enabled)."""
        if (
            not self.config.use_time_extension
            or not self.config.downsample
            or self.config.dt_oversample == 0
        ):
            return ts

        dt = ts[1] - ts[0]
        Nt = len(ts)
        Nt_extended = Nt + self.config.dt_oversample
        return jnp.arange(0, Nt_extended) * dt

    def _apply_window(self, data: jnp.ndarray) -> jnp.ndarray:
        """Apply window taper to end of time series (if enabled)."""
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
        """Apply Kaiser window taper."""
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
            return data * window[:, None]
        return data * window

    def _apply_tukey_window(self, data: jnp.ndarray, alpha: float = 0.5) -> jnp.ndarray:
        """Apply Tukey (tapered cosine) window."""
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
            return data * window[:, None]
        return data * window

    def _solve_lf_component(
        self,
        lf_data: jnp.ndarray,
        solver_method: Callable,
        target_shape: Tuple,
        domain: Domain,
        sensors,
        ts: jnp.ndarray,
        apply_windowing: bool = True,
        apply_interpolation: bool = True,
        **solver_kwargs,
    ) -> jnp.ndarray:
        """
        Generic LF component solver with configurable processing.

        Parameters
        ----------
        lf_data : jnp.ndarray
            Low-frequency input data
        solver_method : callable
            Solver method (e.g., lf_solver.forward, lf_solver.time_reversal)
        target_shape : Tuple
            Shape to interpolate to
        domain : Domain
            LF domain (possibly downsampled)
        sensors : array
            LF sensor mask (possibly downsampled)
        ts : jnp.ndarray
            Time array
        apply_windowing : bool
            Whether to apply windowing (typically True for forward, False for TR)
        apply_interpolation : bool
            Whether to interpolate spatially (typically True if downsampled)
        **solver_kwargs
            Method-specific parameters (e.g., sources, data_domain for TR)

        Returns
        -------
        jnp.ndarray
            LF result with shape=target_shape
        """
        # Solve with appropriate parameters
        lf_result = solver_method(lf_data, domain, sensors, ts, **solver_kwargs)

        # Apply windowing if enabled and downsampled
        if apply_windowing and self.config.downsample:
            lf_result = self._apply_window(lf_result)

        # Apply interpolation if enabled and downsampled
        if apply_interpolation and self.config.downsample:
            lf_result = self._interpolator.interpolate(lf_result, target_shape)

        return lf_result

    def _split_frequencies(
        self,
        data: jnp.ndarray,
        mask: jnp.ndarray,
        wpt: MSWPT,
        domain: Domain,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Domain]:
        """Split data into HF/LF components."""
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
            solver_fn = getattr(self.hf_solver, method)
            return solver_fn(data, domain, sensors, ts, wpt, **kwargs)[0]
        else:
            solver_fn = getattr(self.hf_solver, method)
            return solver_fn(data, domain, mask, ts, **kwargs)

    def _validate_configuration(self, domain: Domain):
        """Warn if configuration may cause issues."""
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

        mask = sensors.binary_mask if hasattr(sensors, "binary_mask") else sensors

        # Split into HF/LF
        p0_HF, p0_LF, ds_mask, ds_domain = self._split_frequencies(
            p0, mask, wpt, domain
        )
        p0_HF, p0_LF = p0_HF.real, p0_LF.real

        # Extend time if configured
        ts_extended = self._extend_time(ts)
        original_Nt = len(ts)

        # Solve HF with extended time
        hf_result = self._solve_hf(
            p0_HF, domain, sensors, mask, ts_extended, wpt, method="forward"
        )

        # Apply window to HF result
        hf_result = self._apply_window(hf_result)

        # Solve LF with extended time (windowing + interpolation inside)
        lf_result = self._solve_lf_component(
            p0_LF,
            self.lf_solver.forward,
            hf_result.shape,
            ds_domain,
            ds_mask,
            ts_extended,
            apply_windowing=True,
            apply_interpolation=True,
        )

        # Truncate both to original time length
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

        mask = sensors.binary_mask if hasattr(sensors, "binary_mask") else sensors

        # Split recorded data into HF/LF
        data_HF, data_LF, ds_mask, ds_domain = self._split_frequencies(
            data, mask, data_wpt, data_domain
        )
        data_HF, data_LF = data_HF.real, data_LF.real

        # Solve HF (in reconstruction domain)
        if isinstance(self.hf_solver, MSGBSolver):
            hf_result = self.hf_solver.time_reversal(
                data_HF,
                domain,
                sensors,
                sources,
                ts,
                data_domain,
                data_wpt,
                img_wpt,
            )[0]
        else:
            hf_result = self.hf_solver.time_reversal(
                data_HF, domain, ds_mask, sources, ts
            )

        # Solve LF (in reconstruction domain, no windowing for TR)
        lf_result = self._solve_lf_component(
            data_LF,
            self.lf_solver.time_reversal,
            hf_result.shape,
            ds_domain,
            ds_mask,
            ts,
            apply_windowing=False,  # No windowing for TR
            apply_interpolation=True,  # Still need interpolation
            sources=sources,  # Pass TR-specific parameter
        )

        return hf_result + lf_result

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

        mask = sensors.binary_mask if hasattr(sensors, "binary_mask") else sensors

        # Split recorded data into HF/LF
        data_HF, data_LF, ds_mask, ds_domain = self._split_frequencies(
            data, mask, data_wpt, data_domain
        )
        data_HF, data_LF = data_HF.real, data_LF.real

        # Solve HF adjoint
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

        # Solve LF adjoint (no windowing)
        lf_result = self._solve_lf_component(
            data_LF,
            self.lf_solver.adjoint,
            hf_result.shape,
            ds_domain,
            ds_mask,
            ts,
            apply_windowing=False,
            apply_interpolation=True,
            sources=sources,
        )

        return hf_result + lf_result
