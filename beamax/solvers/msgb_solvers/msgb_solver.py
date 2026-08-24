import math
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Literal, Union, Optional, Tuple
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec, Mesh

from beamax.solvers.msgb_solvers.forward_solver_utils import (
    compute_coefficients,
    threshold_coefficients,
    compute_forward_parameters,
    compute_forward_result,
)
from beamax.solvers.msgb_solvers.tr_solver_utils import (
    compute_TR_result,
    compute_TR_parameters,
)
from beamax.geometry import Domain, Sensor
from beamax.transforms import MSWPT
from beamax import utils
from beamax.gb.gb_solvers import SolverFn, SolverConfig, solve_hom_diag
from beamax.gb.pallas_config import PallasConfig
from beamax.solvers.msgb_solvers.adjoint_solver_utils import compute_adj_parameters
from beamax.coefficients import streamed_top_n_coefficients


__all__ = [
    "MSGBExperimentalConfig",
    "MSGBSolver",
    "ShardingStrategy",
    "apply_adjoint_image_weight",
    "form_adjoint_source",
]

complex_dtypes = (jnp.complex64, jnp.complex128)

CoefficientSelection = Literal["auto", "streaming_top_n"]
ForwardKernel = Literal["auto", "trajectory_pallas", "hom_diag_3d_pallas"]
InverseEvaluator = Literal["auto", "terminal_xla", "terminal_pallas"]

DEFAULT_BOXES_PER_CHUNK = 8


@dataclass(frozen=True)
class MSGBExperimentalConfig:
    """Experimental per-stage execution overrides.

    ``"auto"`` selects supported defaults. Streaming top-n requires real,
    zero-velocity, fixed-size input. Generic Pallas supports GPU and TPU;
    ``hom_diag_3d_pallas`` requires a GPU.
    """

    coefficient_selection: CoefficientSelection = "auto"
    boxes_per_chunk: int = DEFAULT_BOXES_PER_CHUNK
    forward_kernel: ForwardKernel = "auto"
    inverse_evaluator: InverseEvaluator = "auto"

    def __post_init__(self) -> None:
        if self.coefficient_selection not in {"auto", "streaming_top_n"}:
            raise ValueError(
                "coefficient_selection must be 'auto' or 'streaming_top_n'; "
                f"got {self.coefficient_selection!r}."
            )
        if (
            isinstance(self.boxes_per_chunk, bool)
            or not isinstance(self.boxes_per_chunk, int)
            or self.boxes_per_chunk <= 0
        ):
            raise ValueError(
                "boxes_per_chunk must be a positive integer; got "
                f"{self.boxes_per_chunk!r}."
            )
        if self.forward_kernel not in {
            "auto",
            "trajectory_pallas",
            "hom_diag_3d_pallas",
        }:
            raise ValueError(
                "forward_kernel must be 'auto', 'trajectory_pallas', or "
                f"'hom_diag_3d_pallas'; got {self.forward_kernel!r}."
            )
        if self.inverse_evaluator not in {"auto", "terminal_xla", "terminal_pallas"}:
            raise ValueError(
                "inverse_evaluator must be 'auto', 'terminal_xla', or "
                f"'terminal_pallas'; got {self.inverse_evaluator!r}."
            )


# Data-dependent selection stays eager; fixed-shape propagation is compiled.
_compute_forward_result_jit = eqx.filter_jit(compute_forward_result)
_compute_tr_result_jit = eqx.filter_jit(compute_TR_result)


def _validate_time_grid(
    ts: jnp.ndarray,
    *,
    allow_singleton: bool = False,
) -> Optional[float]:
    """Validate a finite, increasing, uniform time grid and return ``dt``.

    A singleton grid is useful for evaluating a forward solution only at
    $t=0$. In that case there is no time step to return.
    """
    ts_np = np.asarray(ts)
    if ts_np.ndim != 1 or ts_np.size == 0:
        raise ValueError("ts must be a non-empty one-dimensional array.")
    if np.iscomplexobj(ts_np):
        raise ValueError("ts must be real-valued.")
    if not np.all(np.isfinite(ts_np)):
        raise ValueError("ts must contain only finite values.")
    if ts_np.size == 1:
        if allow_singleton:
            return None
        raise ValueError("ts must be one-dimensional with at least two points.")
    diffs = np.diff(ts_np)
    # Endpoint-defined grids avoid cancellation in adjacent float32 differences.
    real_dtype = np.dtype(ts_np.dtype)
    precision_dtype = (
        real_dtype if np.issubdtype(real_dtype, np.floating) else np.dtype(np.float64)
    )
    epsilon = float(np.finfo(precision_dtype).eps)
    ideal = np.linspace(float(ts_np[0]), float(ts_np[-1]), ts_np.size)
    ideal_spacing = abs(float(ideal[-1] - ideal[0])) / (ts_np.size - 1)
    magnitude = max(
        float(np.max(np.abs(ts_np))),
        ideal_spacing,
        float(np.finfo(precision_dtype).tiny),
    )
    rounding_atol = 8.0 * epsilon * magnitude
    if np.any(diffs <= 0) or not np.allclose(
        np.asarray(ts_np, dtype=np.float64),
        ideal,
        rtol=0.0,
        atol=rounding_atol,
    ):
        raise ValueError(
            "ts must be finite, strictly increasing, and uniformly spaced."
        )
    return float((float(ts_np[-1]) - float(ts_np[0])) / (ts_np.size - 1))


def form_adjoint_source(
    data: jnp.ndarray,
    dt: float,
    c_at_sources: jnp.ndarray,
    window: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    r"""Form the acquisition-time source for the unweighted wave equation.

    For detector residual $r(s,\mathbf{x}_s)$, this returns

    $$
    -c(\mathbf{x}_s)^2\,\partial_s
    \left[w(s,\mathbf{x}_s)r(s,\mathbf{x}_s)\right].
    $$

    It is the acquisition-time representation of the time-reversed source in
    the continuous PAT adjoint.  A one-dimensional window is interpreted as a
    time window and broadcast over detector axes.
    """
    if window is None:
        windowed_data = data
    else:
        if window.ndim == 1 and data.ndim > 1:
            window = window.reshape((window.shape[0],) + (1,) * (data.ndim - 1))
        windowed_data = window * data

    # Spectral differentiation preserves the high frequencies used by the
    # principal-symbol approximation; the endpoint/taper contract permits a
    # periodic extension.
    frequencies = jnp.fft.fftfreq(windowed_data.shape[0], d=dt).astype(
        windowed_data.real.dtype
    )
    multiplier_shape = (frequencies.shape[0],) + (1,) * (windowed_data.ndim - 1)
    multiplier = (2j * jnp.pi * frequencies).reshape(multiplier_shape)
    derivative = jnp.fft.ifft(multiplier * jnp.fft.fft(windowed_data, axis=0), axis=0)
    if not jnp.issubdtype(windowed_data.dtype, jnp.complexfloating):
        derivative = derivative.real

    # Restore an unflattened detector grid before broadcasting sound speeds.
    c_at_sources = jnp.asarray(c_at_sources)
    detector_shape = derivative.shape[1:]
    if c_at_sources.ndim == 1 and c_at_sources.shape != detector_shape:
        if c_at_sources.size != math.prod(detector_shape):
            raise ValueError(
                f"c_at_sources has {c_at_sources.size} detector samples, which "
                f"does not match the data detector grid {detector_shape}."
            )
        c_at_sources = c_at_sources.reshape(detector_shape)
    return -(c_at_sources**2) * derivative


def apply_adjoint_image_weight(
    terminal_field: jnp.ndarray, c_at_image: jnp.ndarray
) -> jnp.ndarray:
    r"""Apply the $c^{-2}$ weight for an unweighted image-space pairing."""
    return terminal_field / (c_at_image**2)


@dataclass(frozen=True)
class ShardingStrategy:
    """
    Strategy for sharding beam parameters across devices.

    Attributes
    ----------
    mesh : Mesh
        JAX device mesh for multi-device parallelization.
    beam_axis : str
        Mesh axis used to shard beams.
    """

    mesh: Mesh
    beam_axis: str = "x"

    def _validate_beam_count(self, count: int) -> None:
        """Require an even partition over the configured device axis."""
        devices = int(self.mesh.shape[self.beam_axis])
        if count % devices != 0:
            raise ValueError(
                f"Beam count {count} is not divisible by {devices} devices on "
                f"mesh axis {self.beam_axis!r}. Adjust top_n or use batching."
            )

    def _beam_sharding_spec(self, ndim: int, *, is_batched: bool) -> PartitionSpec:
        """
        Build a partition spec for beam parameters.

        Parameters
        ----------
        ndim : int
            Number of dimensions of the parameter array.
        is_batched : bool
            Whether the parameter array has a leading batch axis.

        Returns
        -------
        PartitionSpec
            Sharding specification for the parameter array.

        Raises
        ------
        ValueError
            If a scalar or malformed batched array is requested.

        Notes
        -----
        For batched tensors ``(num_batches, batch_size, ...)`` used by scanned
        aggregation, we keep all axes replicated. This avoids unsupported
        sharding interactions inside nested `scan`/`diffrax` transforms.
        """
        if ndim < 1:
            raise ValueError("Cannot shard scalar tensors.")

        if is_batched:
            if ndim < 2:
                raise ValueError(
                    "Batched beam parameters must have at least two dimensions."
                )
            return PartitionSpec(*([None] * ndim))

        return PartitionSpec(self.beam_axis, *([None] * (ndim - 1)))

    def shard_beam_params(
        self,
        p0: jnp.ndarray,
        M0: jnp.ndarray,
        x0: jnp.ndarray,
        omega: jnp.ndarray,
        a0: jnp.ndarray,
        modes: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, ...]:
        """
        Shard forward beam parameters along the beam dimension.

        Parameters
        ----------
        p0 : jnp.ndarray
            Beam momenta.
        M0 : jnp.ndarray
            Beam Hessian matrices.
        x0 : jnp.ndarray
            Beam positions.
        omega : jnp.ndarray
            Beam frequencies.
        a0 : jnp.ndarray
            Beam amplitudes.
        modes : jnp.ndarray
            Beam branch signs.

        Returns
        -------
        Tuple[jnp.ndarray, ...]
            Device-placed arrays with sharding specifications applied.
        """
        is_batched = p0.ndim == 3
        if not is_batched:
            self._validate_beam_count(p0.shape[0])

        return (
            jax.device_put(
                p0,
                NamedSharding(
                    self.mesh, self._beam_sharding_spec(p0.ndim, is_batched=is_batched)
                ),
            ),
            jax.device_put(
                M0,
                NamedSharding(
                    self.mesh, self._beam_sharding_spec(M0.ndim, is_batched=is_batched)
                ),
            ),
            jax.device_put(
                x0,
                NamedSharding(
                    self.mesh, self._beam_sharding_spec(x0.ndim, is_batched=is_batched)
                ),
            ),
            jax.device_put(
                omega,
                NamedSharding(
                    self.mesh,
                    self._beam_sharding_spec(omega.ndim, is_batched=is_batched),
                ),
            ),
            jax.device_put(
                a0,
                NamedSharding(
                    self.mesh, self._beam_sharding_spec(a0.ndim, is_batched=is_batched)
                ),
            ),
            jax.device_put(
                modes,
                NamedSharding(
                    self.mesh,
                    self._beam_sharding_spec(modes.ndim, is_batched=is_batched),
                ),
            ),
        )

    def shard_tr_params(
        self,
        pts: jnp.ndarray,
        Mts: jnp.ndarray,
        xts: jnp.ndarray,
        omega_ts: jnp.ndarray,
        ats: jnp.ndarray,
        signum: jnp.ndarray,
        ts: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, ...]:
        """
        Shard time-reversal beam parameters along the beam dimension.

        Parameters
        ----------
        pts : jnp.ndarray
            Beam momenta at the boundary.
        Mts : jnp.ndarray
            Beam Hessians at the boundary.
        xts : jnp.ndarray
            Boundary positions.
        omega_ts : jnp.ndarray
            Beam frequencies.
        ats : jnp.ndarray
            Beam amplitudes.
        signum : jnp.ndarray
            Beam branch signs.
        ts : jnp.ndarray
            Per-beam time intervals.

        Returns
        -------
        Tuple[jnp.ndarray, ...]
            Device-placed arrays with sharding specifications applied.
        """
        is_batched = pts.ndim == 3
        if not is_batched:
            self._validate_beam_count(pts.shape[0])

        return (
            jax.device_put(
                pts,
                NamedSharding(
                    self.mesh, self._beam_sharding_spec(pts.ndim, is_batched=is_batched)
                ),
            ),
            jax.device_put(
                Mts,
                NamedSharding(
                    self.mesh, self._beam_sharding_spec(Mts.ndim, is_batched=is_batched)
                ),
            ),
            jax.device_put(
                xts,
                NamedSharding(
                    self.mesh, self._beam_sharding_spec(xts.ndim, is_batched=is_batched)
                ),
            ),
            jax.device_put(
                omega_ts,
                NamedSharding(
                    self.mesh,
                    self._beam_sharding_spec(omega_ts.ndim, is_batched=is_batched),
                ),
            ),
            jax.device_put(
                ats,
                NamedSharding(
                    self.mesh, self._beam_sharding_spec(ats.ndim, is_batched=is_batched)
                ),
            ),
            jax.device_put(
                signum,
                NamedSharding(
                    self.mesh,
                    self._beam_sharding_spec(signum.ndim, is_batched=is_batched),
                ),
            ),
            jax.device_put(
                ts,
                NamedSharding(
                    self.mesh, self._beam_sharding_spec(ts.ndim, is_batched=is_batched)
                ),
            ),
        )


class MSGBSolver(eqx.Module):
    r"""
    Multiscale Gaussian Beam solver for the linear wave equation.

    Implements forward, time-reversal, and adjoint operators by:

    1. Decomposing the initial pressure into wave-packet coefficients via
       :class:`beamax.transforms.MSWPT`.
    2. Thresholding to retain only significant coefficients.
    3. Integrating a small Hamiltonian ODE per retained beam.
    4. Summing (or scanning) the beam contributions at the sensor positions.

    Parameters
    ----------
    thr : int or float
        Absolute magnitude threshold or top-k count, depending on
        ``thr_strat``.
    thr_strat : str
        Thresholding strategy: ``"hard"`` or ``"top_n"``.
    batch_size : int
        Batch size along the beam axis for ODE integration. Tune to fit
        device memory; larger values amortise kernel launches.
    input_type : {"spatial", "fourier"}
        Domain the caller provides ``p0`` in.
    ode_solver : SolverFn
        Forward-time ODE integrator for beam dynamics (typically one of
        :mod:`beamax.gb.gb_solvers`).
    sum_method : str
        Method for summing beam contributions. One of ``"all_real"``,
        ``"scan_real"``, ``"all_complex"``, ``"scan_complex"``, or the
        experimental accelerator mode ``"pallas_real"``.
    tr_ode_solver : SolverFn, optional
        ODE integrator for the time-reversal dynamics. Falls back to
        ``ode_solver`` when ``None``.
    sharding : ShardingStrategy, optional
        Multi-device sharding strategy. ``None`` runs on a single device.
    ode_config : SolverConfig, optional
        Numerical configuration passed through to the ODE integrator. Falls
        back to ``SolverConfig.from_precision()``.
    adjoint_relative_guard : float, default=5e-2
        Dimensionless near-grazing exclusion $\Gamma/|\tau|$ used by the
        principal-symbol adjoint. Tune together with the ODE configuration.
    pallas_config : PallasConfig, optional
        Experimental static kernel configuration used by ``pallas_real`` and
        any explicitly selected Pallas stage.
    experimental_config : MSGBExperimentalConfig, optional
        Per-stage pipeline overrides. Omitting it is equivalent to passing a
        configuration with every field set to ``"auto"``: streamed top-n
        selection and terminal-only inverse evaluation apply whenever they
        are output-equivalent, and Pallas kernels stay off.
    """

    thr: Union[int, float] = eqx.field()
    thr_strat: str = eqx.field()
    batch_size: int = eqx.field(static=True)
    input_type: str = eqx.field(static=True)
    ode_solver: SolverFn = eqx.field()
    tr_ode_solver: SolverFn = eqx.field()
    use_real: bool = eqx.field(static=True)
    aggregate_method: str = eqx.field(static=True)
    ode_config: SolverConfig = eqx.field(static=True)
    experimental_config: MSGBExperimentalConfig = eqx.field(static=True)
    sharding: Optional[ShardingStrategy] = eqx.field(default=None, static=True)
    adjoint_relative_guard: float = eqx.field(default=5e-2, static=True)
    pallas_config: Optional[PallasConfig] = eqx.field(default=None, static=True)

    def __init__(
        self,
        thr: Union[int, float],
        thr_strat: str,
        batch_size: int,
        input_type: str,
        ode_solver: SolverFn,
        sum_method: str,
        tr_ode_solver: Optional[SolverFn] = None,
        sharding: Optional[ShardingStrategy] = None,
        ode_config: Optional[SolverConfig] = None,
        adjoint_relative_guard: float = 5e-2,
        pallas_config: Optional[PallasConfig] = None,
        experimental_config: Optional[MSGBExperimentalConfig] = None,
    ):
        r"""
        Initialize the MSGB solver.

        Parameters
        ----------
        thr : int or float
            Threshold value for coefficient selection.
        thr_strat : str
            Thresholding strategy name.
        batch_size : int
            Number of beams per batch for scanned aggregation.
        input_type : {"spatial", "fourier"}
            Domain of inputs supplied to the solver.
        ode_solver : SolverFn
            Forward ODE solver.
        sum_method : str
            Aggregation mode string. Must be one of the values listed in the
            class-level parameter documentation.
        tr_ode_solver : SolverFn, optional
            ODE solver for time reversal. Defaults to ``ode_solver``.
        sharding : ShardingStrategy, optional
            Multi-device sharding strategy.
        ode_config : SolverConfig, optional
            Numerical ODE solver configuration. Defaults to
            ``SolverConfig.from_precision()``.
        adjoint_relative_guard : float, default=5e-2
            Dimensionless near-grazing exclusion for the adjoint. Must lie in
            $[0,1)$.
        pallas_config : PallasConfig, optional
            Experimental static Pallas layout and launch configuration.
        experimental_config : MSGBExperimentalConfig, optional
            Per-stage overrides: force the streamed selector, choose the
            inverse evaluator, or opt in to the experimental Pallas
            forward/inverse kernels. Omitting it selects the supported
            defaults for every stage.
        """
        valid_thresholds = {"hard", "top_n"}
        if thr_strat not in valid_thresholds:
            allowed = ", ".join(sorted(valid_thresholds))
            raise ValueError(f"thr_strat must be one of {allowed}; got {thr_strat!r}.")
        if isinstance(thr, bool) or not isinstance(thr, Real):
            raise ValueError(f"thr must be a finite real scalar; got {thr!r}.")
        if not math.isfinite(float(thr)):
            raise ValueError(f"thr must be finite; got {thr!r}.")
        if thr_strat == "top_n":
            if not isinstance(thr, Integral) or int(thr) <= 0:
                raise ValueError("top_n threshold must be a positive integer.")
            thr = int(thr)
        else:
            if float(thr) < 0:
                raise ValueError(f"{thr_strat} threshold must be non-negative.")
            thr = float(thr)
        if input_type not in {"spatial", "fourier"}:
            raise ValueError(
                f"input_type must be 'spatial' or 'fourier'; got {input_type!r}."
            )
        if (
            isinstance(batch_size, (bool, np.bool_))
            or not isinstance(batch_size, Integral)
            or batch_size <= 0
        ):
            raise ValueError(
                f"batch_size must be a positive integer; got {batch_size!r}."
            )
        batch_size = int(batch_size)
        if not 0.0 <= adjoint_relative_guard < 1.0:
            raise ValueError(
                "adjoint_relative_guard must lie in [0, 1); got "
                f"{adjoint_relative_guard}."
            )
        if pallas_config is not None and not isinstance(pallas_config, PallasConfig):
            raise TypeError(
                "pallas_config must be a PallasConfig instance or None; got "
                f"{type(pallas_config).__name__}."
            )
        if experimental_config is None:
            experimental_config = MSGBExperimentalConfig()
        elif not isinstance(experimental_config, MSGBExperimentalConfig):
            raise TypeError(
                "experimental_config must be an MSGBExperimentalConfig "
                f"instance or None; got {type(experimental_config).__name__}."
            )
        if (
            experimental_config.coefficient_selection == "streaming_top_n"
            and thr_strat != "top_n"
        ):
            raise ValueError(
                "coefficient_selection='streaming_top_n' requires "
                "thr_strat='top_n'; use 'auto' to stream only when possible."
            )

        valid_sum_methods = {
            "all_real",
            "scan_real",
            "pallas_real",
            "all_complex",
            "scan_complex",
        }
        if sum_method not in valid_sum_methods:
            allowed = ", ".join(sorted(valid_sum_methods))
            raise ValueError(
                f"sum_method must be one of {allowed}; got {sum_method!r}."
            )
        if "pallas" in sum_method and sharding is not None:
            raise ValueError(
                "pallas_real is a single-device experimental backend and "
                "cannot be combined with sharding. Use scan_real on one "
                "device or an all_* method with sharding."
            )
        requests_experimental_stage = (
            experimental_config.coefficient_selection == "streaming_top_n"
            or experimental_config.forward_kernel != "auto"
            or experimental_config.inverse_evaluator
            in {"terminal_xla", "terminal_pallas"}
        )
        if requests_experimental_stage and sharding is not None:
            raise ValueError(
                "Explicit MSGBExperimentalConfig stage overrides are "
                "currently single-device and cannot be combined with "
                "sharding."
            )
        if (
            experimental_config.forward_kernel != "auto"
            or experimental_config.inverse_evaluator
            in {"terminal_xla", "terminal_pallas"}
        ) and "real" not in sum_method:
            raise ValueError(
                "Explicit forward-kernel and terminal-inverse overrides "
                "require a real sum_method."
            )
        if (
            experimental_config.forward_kernel == "hom_diag_3d_pallas"
            and ode_solver is not solve_hom_diag
        ):
            raise ValueError(
                "forward_kernel='hom_diag_3d_pallas' requires "
                "ode_solver=solve_hom_diag."
            )

        if ode_config is None:
            ode_config = SolverConfig.from_precision()

        self.thr = thr
        self.thr_strat = thr_strat
        self.batch_size = batch_size
        self.input_type = input_type
        self.ode_solver = ode_solver
        self.tr_ode_solver = ode_solver if tr_ode_solver is None else tr_ode_solver
        self.sharding = sharding
        self.ode_config = ode_config
        self.adjoint_relative_guard = float(adjoint_relative_guard)
        self.pallas_config = pallas_config
        self.experimental_config = experimental_config

        self.use_real = "real" in sum_method
        if "pallas" in sum_method:
            self.aggregate_method = "pallas"
        elif "scan" in sum_method:
            self.aggregate_method = "scan"
        else:
            self.aggregate_method = "all"

    def _effective_forward_aggregate_method(self) -> str:
        """Resolve the static aggregation policy for a forward call."""
        config = self.experimental_config
        if config.forward_kernel == "auto":
            return self.aggregate_method
        if config.forward_kernel == "trajectory_pallas":
            return "pallas"
        return "pallas_fused_hom_diag_3d"

    def _effective_inverse_aggregate_method(self) -> str:
        """Resolve the static aggregation policy for TR/adjoint calls.

        ``"auto"`` uses terminal XLA for unsharded real aggregation.
        """
        config = self.experimental_config
        requested = config.inverse_evaluator
        if requested == "terminal_pallas":
            return "pallas"
        if requested == "terminal_xla":
            return "terminal_xla"
        if (
            self.use_real
            and self.sharding is None
            and "pallas" not in self.aggregate_method
        ):
            return "terminal_xla"
        return self.aggregate_method

    def _streams_top_n(self, wpt: MSWPT, data_dtype) -> bool:
        """Return whether exact streamed top-n applies.

        ``"auto"`` requires real, windowed, unsharded top-n input.
        """
        config = self.experimental_config
        requested = config.coefficient_selection
        if requested == "streaming_top_n":
            return True
        return (
            self.thr_strat == "top_n"
            and self.sharding is None
            and wpt.windowing != "none"
            and not jnp.issubdtype(data_dtype, jnp.complexfloating)
        )

    def _boxes_per_chunk(self) -> int:
        """Chunk size for the streamed selector."""
        return self.experimental_config.boxes_per_chunk

    def _effective_pallas_config(
        self,
        periodic: Tuple[bool, ...],
        *,
        aggregate_method: Optional[str] = None,
    ):
        """Return a domain-specialized static config for Pallas calls."""
        method = self.aggregate_method if aggregate_method is None else aggregate_method
        if "pallas" not in method:
            return self.pallas_config
        periodic_axes = tuple(bool(value) for value in periodic)
        config = self.pallas_config or PallasConfig()
        if config.periodic_axes is None:
            return replace(config, periodic_axes=periodic_axes)
        if config.periodic_axes != periodic_axes:
            raise ValueError(
                "pallas_config.periodic_axes does not match domain.periodic: "
                f"{config.periodic_axes} != {periodic_axes}."
            )
        return config

    def _effective_top_n(
        self, wpt: MSWPT, *, half_frame: bool = False
    ) -> Union[int, float]:
        """Clamp a top-n request to the static coefficient capacity."""
        if self.thr_strat != "top_n":
            return self.thr
        if half_frame:
            capacity = sum(
                (end - start) // 2
                for start, end in zip(wpt.coeffs_cumsum[:-1], wpt.coeffs_cumsum[1:])
            )
        else:
            capacity = wpt.total_coeffs
        return min(int(self.thr), int(capacity))

    def _replicate_array(self, arr: jnp.ndarray) -> jnp.ndarray:
        """
        Materialize an array with fully replicated sharding on the active mesh.

        Parameters
        ----------
        arr : jnp.ndarray
            Array to replicate.

        Returns
        -------
        jnp.ndarray
            Replicated array, or ``arr`` unchanged if no sharding is active.
        """
        if self.sharding is None:
            return arr
        replicated = PartitionSpec(*([None] * arr.ndim))
        return jax.device_put(arr, NamedSharding(self.sharding.mesh, replicated))

    def _prepare_forward_params_real(
        self,
        p0: jnp.ndarray,
        dpdt: Optional[jnp.ndarray],
        domain: Domain,
        wpt: MSWPT,
    ) -> Tuple[jnp.ndarray, ...]:
        """
        Prepare beam parameters for a real-valued forward solve.

        Parameters
        ----------
        p0 : jnp.ndarray
            Initial pressure field.
        dpdt : jnp.ndarray
            Initial pressure time derivative.
        domain : Domain
            Physical domain.
        wpt : MSWPT
            Wave-packet transform.

        Returns
        -------
        Tuple[jnp.ndarray, ...]
            Beam parameters ``(p0s, M0s, x0s, omegas, a0s, modes)``.
        """
        threshold = self.thr
        if self.thr_strat == "top_n":
            # Positive-half analysis cannot exceed half of each level.
            threshold = self._effective_top_n(wpt, half_frame=True)

        if dpdt is None:
            # Zero initial velocity gives c+ = 0.5 WPT(p0).
            if self._streams_top_n(wpt, p0.dtype):
                coeff_pos_idx, max_pos_coeffs = streamed_top_n_coefficients(
                    p0,
                    wpt,
                    top_n=int(threshold),
                    input_type=self.input_type,
                    positive_half=True,
                    scale=0.5,
                    boxes_per_chunk=self._boxes_per_chunk(),
                )
            else:
                c_pos = 0.5 * wpt.forward(p0, self.input_type) * wpt.half_mask
                coeff_pos_idx, max_pos_coeffs = threshold_coefficients(
                    c_pos, threshold, self.thr_strat
                )
        else:
            c_pos = compute_coefficients(
                p0,
                dpdt,
                self.input_type,
                domain,
                wpt,
                mode="pos_only",
            )
            coeff_pos_idx, max_pos_coeffs = threshold_coefficients(
                c_pos, threshold, self.thr_strat
            )

        p0s, M0s, x0s, ωs, a0s, modes = compute_forward_parameters(
            coeff_pos_idx, wpt, domain
        )
        a0s = a0s * max_pos_coeffs

        params_to_concat = (p0s, M0s, x0s, ωs, a0s)
        p0s, M0s, x0s, ωs, a0s = tuple(
            jnp.concatenate([p, p]) for p in params_to_concat
        )
        modes = jnp.concatenate([modes, -modes])

        forward_aggregate = self._effective_forward_aggregate_method()
        if forward_aggregate in {
            "scan",
            "pallas",
            "pallas_fused_hom_diag_3d",
        }:
            p0s, M0s, x0s, ωs, a0s, modes = utils.batch_data(
                p0s,
                M0s,
                x0s,
                ωs,
                a0s,
                modes,
                batch_size=self.batch_size,
                zero_padded_args=(4,),
            )
        return (p0s, M0s, x0s, ωs, a0s, modes)

    def _prepare_forward_params_complex(
        self, p0: jnp.ndarray, dpdt: jnp.ndarray, domain: Domain, wpt: MSWPT
    ) -> Tuple[jnp.ndarray, ...]:
        """
        Prepare beam parameters for a complex-valued forward solve.

        Parameters
        ----------
        p0 : jnp.ndarray
            Initial pressure field.
        dpdt : jnp.ndarray
            Initial pressure time derivative.
        domain : Domain
            Physical domain.
        wpt : MSWPT
            Wave-packet transform.

        Returns
        -------
        Tuple[jnp.ndarray, ...]
            Beam parameters ``(p0s, M0s, x0s, omegas, a0s, modes)``.
        """
        c_pos, c_neg = compute_coefficients(
            p0, dpdt, self.input_type, domain, wpt, mode="both"
        )

        (coeff_pos_idx, max_pos_coeffs), (coeff_neg_idx, max_neg_coeffs) = (
            threshold_coefficients(c_pos, self._effective_top_n(wpt), self.thr_strat),
            threshold_coefficients(c_neg, self._effective_top_n(wpt), self.thr_strat),
        )

        p0s, M0s, x0s, ωs, a0s, modes = compute_forward_parameters(
            (coeff_pos_idx, coeff_neg_idx), wpt, domain
        )
        max_coeffs = jnp.concatenate([max_pos_coeffs, max_neg_coeffs])
        a0s = a0s * max_coeffs

        if self.aggregate_method in ["scan", "pallas"]:
            p0s, M0s, x0s, ωs, a0s, modes = utils.batch_data(
                p0s,
                M0s,
                x0s,
                ωs,
                a0s,
                modes,
                batch_size=self.batch_size,
                zero_padded_args=(4,),
            )

        return (p0s, M0s, x0s, ωs, a0s, modes)

    def _prepare_tr_params(
        self,
        data: jnp.ndarray,
        data_domain: Domain,
        data_wpt: MSWPT,
        sources,
        ts: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, ...]:
        """
        Prepare beam parameters for a time-reversal solve.

        Parameters
        ----------
        data : jnp.ndarray
            Sensor time-series data.
        data_domain : Domain
            Domain describing ``data``.
        data_wpt : MSWPT
            Wave-packet transform for ``data``.
        sources : Sensor
            Boundary source geometry.

        Returns
        -------
        Tuple[jnp.ndarray, ...]
            Time-reversal beam parameters
            ``(pts, Mts, xts, omegas, ats, signum, ts)``.
        """
        self._validate_boundary_data(data, data_domain, sources, ts=ts)
        threshold = self._effective_top_n(data_wpt)
        if self._streams_top_n(data_wpt, data.dtype):
            coeff_idx, max_coeffs = streamed_top_n_coefficients(
                data,
                data_wpt,
                top_n=int(threshold),
                input_type=self.input_type,
                positive_half=False,
                scale=0.5,
                boxes_per_chunk=self._boxes_per_chunk(),
            )
        else:
            # Zero initial derivative gives c+ = 0.5 WPT(data).
            c_pos = 0.5 * data_wpt.forward(data, self.input_type)
            coeff_idx, max_coeffs = threshold_coefficients(
                c_pos, threshold, self.thr_strat
            )

        pts, Mts, xts, ωts, ats, signum, beam_times = compute_TR_parameters(
            coeff_idx, data_domain, data_wpt, sources
        )

        ats = ats * max_coeffs[:, None]

        # Only these aggregators consume an explicit batch axis.
        inverse_aggregate = self._effective_inverse_aggregate_method()
        if inverse_aggregate in {"scan", "pallas", "terminal_xla"}:
            pts, Mts, xts, ωts, ats, signum, beam_times = utils.batch_data(
                pts,
                Mts,
                xts,
                ωts,
                ats,
                signum,
                beam_times,
                batch_size=self.batch_size,
                zero_padded_args=(4,),
            )
        return pts, Mts, xts, ωts, ats, signum, beam_times

    def _infer_planar_surface(self, sensor_positions: jnp.ndarray, eps: float = 1e-9):
        """
        Infer a planar detector surface from a constant sensor coordinate.

        Parameters
        ----------
        sensor_positions : jnp.ndarray, shape (Ns, d)
            Sensor coordinates.
        eps : float, default=1e-9
            Maximum standard deviation allowed on the inferred normal axis.

        Returns
        -------
        surface : Callable[[jnp.ndarray], jnp.ndarray]
            Implicit surface function.
        axis : int
            Inferred normal axis.
        coord : float
            Constant coordinate value on that axis.

        Raises
        ------
        ValueError
            If no nearly constant sensor-position axis is found.
        """
        stds = jnp.std(sensor_positions, axis=0)
        axis = int(jnp.argmin(stds))
        if stds[axis] > eps:
            raise ValueError(
                "Cannot infer planar surface from sensor positions; please provide `surface`."
            )
        coord = float(sensor_positions[0, axis])

        def surface(x):
            """
            Evaluate the inferred planar surface function.

            Parameters
            ----------
            x : jnp.ndarray, shape (d,)
                Query coordinate.

            Returns
            -------
            jnp.ndarray
                Signed distance-like residual ``x[axis] - coord``.
            """
            return x[axis] - coord

        return surface, axis, coord

    def _validate_boundary_data(
        self,
        data: jnp.ndarray,
        data_domain: Domain,
        sources: Sensor,
        *,
        ts: Optional[jnp.ndarray] = None,
    ) -> None:
        """Validate the regular axis-aligned detector grid assumed by MSGB."""
        if tuple(data.shape) != data_domain.N:
            raise ValueError(
                f"Boundary data must have shape {data_domain.N}; got {data.shape}."
            )
        if ts is not None:
            _validate_time_grid(ts)
            data_span = (data.shape[0] - 1) * data_domain.dx[0]
            acquisition_span = float(np.asarray(ts)[-1] - np.asarray(ts)[0])
            if not np.isclose(acquisition_span, data_span, rtol=1e-6, atol=1e-12):
                raise ValueError(
                    f"ts spans {acquisition_span}, but data_domain spans {data_span}."
                )
        if not isinstance(sources, Sensor):
            raise ValueError("sources must be a Sensor on a regular planar grid.")

        positions = np.asarray(sources.positions)
        stds = np.std(positions, axis=0)
        normal_axis = int(np.argmin(stds))
        tolerance = 1e-7 * max(1.0, float(np.max(sources.domain.grid_size)))
        if stds[normal_axis] > tolerance:
            raise ValueError("sources must lie on an axis-aligned planar surface.")

        tangential_axes = [ax for ax in range(positions.shape[1]) if ax != normal_axis]
        counts = []
        for axis in tangential_axes:
            values = np.unique(positions[:, axis])
            if values.size > 2 and not np.allclose(
                np.diff(values), np.diff(values)[0], rtol=1e-6, atol=tolerance
            ):
                raise ValueError(
                    "sources must be uniformly spaced on the detector plane."
                )
            counts.append(int(values.size))
        expected_detector_shape = tuple(counts) if counts else (1,)
        actual_detector_shape = tuple(data.shape[1:]) or (1,)
        if actual_detector_shape != expected_detector_shape:
            raise ValueError(
                "Boundary data detector axes do not match the source grid: "
                f"got {actual_detector_shape}, expected {expected_detector_shape}."
            )
        if math.prod(expected_detector_shape) != positions.shape[0]:
            raise ValueError("sources must form a complete Cartesian detector grid.")

    def forward(
        self,
        p0: jnp.ndarray,
        domain: Domain,
        sensors: Union[Sensor, jnp.ndarray],
        ts: jnp.ndarray,
        wpt: MSWPT,
        *,
        dpdt: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        r"""
        Solve the forward wave equation with MSGB:

        $$
        \partial_t^2u-c(\mathbf{x})^2\Delta u=0,
        \qquad
        u(0,\mathbf{x})=p_0(\mathbf{x}),
        \qquad
        \partial_tu(0,\mathbf{x})=\dot p_0(\mathbf{x}).
        $$

        The ``dpdt`` argument supplies $\dot p_0$ and is zero for standard
        photoacoustic tomography.

        Parameters
        ----------
        p0 : jnp.ndarray, shape (*N,)
            Initial pressure field. Real or complex; dtype selects the
            underlying beam formulation.
        domain : Domain
            Computational domain and medium.
        sensors : Sensor or jnp.ndarray
            Sensor geometry. Either a :class:`Sensor` or an array of
            positions in physical units, shape ``(Ns, ndim)``.
        ts : jnp.ndarray, shape (Nt,)
            Time grid.
        wpt : MSWPT
            Wave-packet transform used to build the beam decomposition.
        dpdt : jnp.ndarray, optional
            Initial time derivative. Defaults to zeros (standard PAT).

        Returns
        -------
        jnp.ndarray, shape (Nt, Ns)
            Pressure at each sensor over time.
        """
        sensor_data, _ = self.forward_with_params(
            p0, domain, sensors, ts, wpt, dpdt=dpdt
        )
        return sensor_data

    def forward_with_params(
        self,
        p0: jnp.ndarray,
        domain: Domain,
        sensors: Union[Sensor, jnp.ndarray],
        ts: jnp.ndarray,
        wpt: MSWPT,
        *,
        dpdt: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
        """
        Forward MSGB solve plus diagnostic beam parameters.

        This is the explicit diagnostic variant of :meth:`forward`. Most users
        should call :meth:`forward`, which returns only sensor data.

        Returns
        -------
        sensor_data : jnp.ndarray, shape (Nt, Ns)
            Pressure at each sensor over time.
        params : tuple of jnp.ndarray
            Beam parameters used in the solve:
            ``(p0s, M0s, x0s, omegas, a0s, modes)``.
        """
        dpdt_was_omitted = dpdt is None
        if tuple(p0.shape) != domain.N or (
            dpdt is not None and tuple(dpdt.shape) != domain.N
        ):
            raise ValueError(
                f"p0 and dpdt must both have shape {domain.N}; got "
                f"{p0.shape} and {None if dpdt is None else dpdt.shape}."
            )
        if wpt.dyadic_decomp.N != domain.N:
            raise ValueError("wpt grid shape must match domain.N.")
        _validate_time_grid(ts, allow_singleton=True)

        forward_aggregate = self._effective_forward_aggregate_method()
        use_sharding = self.sharding is not None and forward_aggregate == "all"

        sensor_positions = (
            sensors.positions
            if isinstance(sensors, Sensor)
            else sensors
            if isinstance(sensors, jnp.ndarray)
            else None
        )
        if sensor_positions is None:
            raise ValueError("Unsupported sensor type")

        forward_override = self.experimental_config.forward_kernel != "auto"
        explicit_streaming = (
            self.experimental_config.coefficient_selection == "streaming_top_n"
        )
        dpdt_is_complex = dpdt is not None and dpdt.dtype in complex_dtypes
        input_is_complex = p0.dtype in complex_dtypes or dpdt_is_complex
        if input_is_complex and self.use_real:
            raise ValueError(
                "Complex p0 or dpdt requires an *_complex sum_method; "
                "the *_real methods use the conjugate-pair formulation for "
                "real-valued initial data."
            )
        if input_is_complex and (forward_override or explicit_streaming):
            raise ValueError(
                "Streamed selection and forward-kernel overrides support "
                "real float32 forward inputs only."
            )
        if explicit_streaming and not dpdt_was_omitted:
            raise ValueError(
                "coefficient_selection='streaming_top_n' requires a forward "
                "solve with dpdt=None (zero initial velocity); use 'auto' to "
                "stream only when possible."
            )
        if forward_aggregate == "pallas_fused_hom_diag_3d":
            if domain.ndim != 3:
                raise ValueError(
                    "hom_diag_3d_pallas requires a three-dimensional domain."
                )
            if callable(domain.c) or np.asarray(domain.c).ndim != 0:
                raise ValueError(
                    "hom_diag_3d_pallas requires Domain.c to be a scalar "
                    "constant; callable and grid-valued sound speeds must use "
                    "the trajectory-based forward path."
                )
            if p0.dtype != jnp.float32 or (
                dpdt is not None and dpdt.dtype != jnp.float32
            ):
                raise ValueError("hom_diag_3d_pallas requires float32 p0 and dpdt.")
            if bool(getattr(jax.config, "x64_enabled", False)):
                raise RuntimeError(
                    "hom_diag_3d_pallas requires jax_enable_x64=False so all "
                    "derived beam parameters remain float32/complex64."
                )
            if jax.default_backend() != "gpu":
                raise RuntimeError(
                    "hom_diag_3d_pallas requires a GPU backend. Use "
                    "the low-level interpret=True entry point for CPU tests."
                )
            positions_np = np.asarray(sensor_positions)
            if positions_np.ndim != 2 or positions_np.shape[1] != 3:
                raise ValueError(
                    "hom_diag_3d_pallas expects sensor positions of shape (Ns, 3)."
                )

        if input_is_complex:
            effective_dpdt = jnp.zeros_like(p0) if dpdt is None else dpdt
            params = self._prepare_forward_params_complex(
                p0, effective_dpdt, domain, wpt
            )
        else:
            params = self._prepare_forward_params_real(
                p0,
                None if dpdt_was_omitted else dpdt,
                domain,
                wpt,
            )

        if use_sharding:
            assert self.sharding is not None  # implied by `use_sharding`
            params = self.sharding.shard_beam_params(*params)

        sensor_data = _compute_forward_result_jit(
            params=params,
            c=domain.c_fn,
            lam=domain.lam,
            ts=ts,
            ode_solver=self.ode_solver,
            sensors=sensor_positions,
            domain_size=domain.grid_size,
            periodic=jnp.array(domain.periodic),
            use_real=self.use_real,
            aggregate_method=forward_aggregate,
            solver_config=self.ode_config,
            pallas_config=self._effective_pallas_config(
                domain.periodic, aggregate_method=forward_aggregate
            ),
        )

        if use_sharding:
            sensor_data = self._replicate_array(sensor_data)

        return sensor_data, params

    def time_reversal(
        self,
        data: jnp.ndarray,
        domain: Domain,
        sensors: Sensor,
        sources: Sensor,
        ts,
        data_domain: Domain,
        data_wpt: MSWPT,
    ) -> jnp.ndarray:
        """
        MSGB time-reversal reconstruction.

        Parameters
        ----------
        data : jnp.ndarray, shape (Nt, Ns)
            Sensor time series to time-reverse.
        domain : Domain
            Reconstruction domain. Must have ``periodic`` all False —
            time-reversal here assumes free-space boundaries.
        sensors : Sensor
            Sensor geometry corresponding to ``data``.
        sources : Sensor
            Source positions used to seed the TR beams (often the same
            boundary as ``sensors``).
        ts : jnp.ndarray, shape (Nt,)
            Time grid corresponding to ``data``.
        data_domain : Domain
            Domain on which ``data`` was acquired (may differ from ``domain``
            under downsampling).
        data_wpt : MSWPT
            Wave-packet transform on ``data_domain`` used to analyse ``data``.

        Returns
        -------
        jnp.ndarray, shape (*N,)
            Reconstructed initial pressure. Scaled by 2 to match the
            standard full-field time-reversal convention.

        Raises
        ------
        ValueError
            If any axis of ``domain`` is periodic.

        Notes
        -----
        Time reversal here uses per-beam time intervals (via
        :func:`beamax.gb.solve_ODE_batch_t`) regardless of the forward
        integrator, to accommodate the variable emission time of each beam.
        """
        p0_recon, _ = self.time_reversal_with_params(
            data, domain, sensors, sources, ts, data_domain, data_wpt
        )
        return p0_recon

    def time_reversal_with_params(
        self,
        data: jnp.ndarray,
        domain: Domain,
        sensors: Sensor,
        sources: Sensor,
        ts,
        data_domain: Domain,
        data_wpt: MSWPT,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
        """
        Time-reversal MSGB solve plus diagnostic beam parameters.

        This is the explicit diagnostic variant of :meth:`time_reversal`. Most
        users should call :meth:`time_reversal`, which returns only the
        reconstructed field.

        Returns
        -------
        p0_recon : jnp.ndarray, shape (*N,)
            Reconstructed initial pressure.
        params : tuple of jnp.ndarray
            Beam parameters used in the solve.
        """
        if any(domain.periodic):
            raise ValueError(
                "The MSGB time reversal solver only supports free space boundary conditions."
            )
        if data.dtype in complex_dtypes and self.use_real:
            raise ValueError(
                "Complex time-reversal data requires an *_complex sum_method; "
                "the *_real methods require real-valued boundary data."
            )
        inverse_aggregate = self._effective_inverse_aggregate_method()
        use_sharding = self.sharding is not None and inverse_aggregate == "all"

        if use_sharding:
            data = self._replicate_array(data)

        sensor_positions = (
            sensors.positions
            if isinstance(sensors, Sensor)
            else sensors
            if isinstance(sensors, jnp.ndarray)
            else None
        )
        if sensor_positions is None:
            raise ValueError("Unsupported sensor type")

        params = self._prepare_tr_params(data, data_domain, data_wpt, sources, ts)

        if use_sharding:
            assert self.sharding is not None  # implied by `use_sharding`
            params = self.sharding.shard_tr_params(*params)

        p0_recon = _compute_tr_result_jit(
            params=params,
            c=domain.c_fn,
            lam=domain.lam,
            sensors=sensor_positions,
            domain_size=domain.grid_size,
            periodic=jnp.array(domain.periodic),
            ode_solver=self.tr_ode_solver,
            aggregate_method=inverse_aggregate,
            solver_config=self.ode_config,
            pallas_config=self._effective_pallas_config(
                domain.periodic, aggregate_method=inverse_aggregate
            ),
        )

        p0_recon = p0_recon * 2

        if use_sharding:
            p0_recon = self._replicate_array(p0_recon)

        return p0_recon, params

    def _prepare_adj_params(
        self,
        source: jnp.ndarray,
        data_domain: Domain,
        data_wpt: MSWPT,
        sources: Sensor,
    ) -> Tuple[jnp.ndarray, ...]:
        """
        Prepare beam parameters for the principal-symbol adjoint backprojection.

        Parameters
        ----------
        source : jnp.ndarray
            Boundary source density appearing on the RHS of the unweighted
            second-order wave equation. It is already windowed, differentiated,
            and expressed on the time grid used by the backpropagator.
        data_domain : Domain
            Domain describing the (t, x_s) grid of `source`.
        data_wpt : MSWPT
            Transform used to analyse `source` in the MSWPT frame.
        sources : Sensor
            Geometry of the injection boundary (usually the same as `sensors`).

        Returns
        -------
        Tuple of beam parameters suitable for `compute_TR_result`.
        """
        # Direct source analysis avoids the IVP half-wave factor; B^{-1} needs
        # the signed temporal boxes.
        threshold = self._effective_top_n(data_wpt)
        if self._streams_top_n(data_wpt, source.dtype):
            coeff_idx, max_coeffs = streamed_top_n_coefficients(
                source,
                data_wpt,
                top_n=int(threshold),
                input_type=self.input_type,
                positive_half=False,
                scale=1.0,
                boxes_per_chunk=self._boxes_per_chunk(),
            )
        else:
            source_coeffs = data_wpt.forward(source, self.input_type)
            coeff_idx, max_coeffs = threshold_coefficients(
                source_coeffs,
                threshold,
                self.thr_strat,
            )

        pts, Mts, xts, omegas, ats, signum, ts = compute_adj_parameters(
            coeff_idx,
            data_domain,
            data_wpt,
            sources,
            relative_guard=self.adjoint_relative_guard,
        )

        ats = ats * max_coeffs[:, None]

        inverse_aggregate = self._effective_inverse_aggregate_method()
        if inverse_aggregate in {"scan", "pallas", "terminal_xla"}:
            pts, Mts, xts, omegas, ats, signum, ts = utils.batch_data(
                pts,
                Mts,
                xts,
                omegas,
                ats,
                signum,
                ts,
                batch_size=self.batch_size,
                zero_padded_args=(4,),  # ts is the zero-padded arg
            )

        return pts, Mts, xts, omegas, ats, signum, ts

    def adjoint(
        self,
        data: jnp.ndarray,
        domain: Domain,
        sensors: Union[Sensor, jnp.ndarray],
        sources: Sensor,
        ts: jnp.ndarray,
        data_domain: Domain,
        data_wpt: MSWPT,
        *,
        window: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        r"""
        Principal-symbol MSGB approximation of the continuous PAT adjoint.

        Parameters
        ----------
        data : jnp.ndarray
            Boundary residual $r(t,\mathbf{x}_s)$ on $\Gamma$. Shape (Nt, Ns)
            or (Nt,), with time along axis 0.
        domain : Domain
            Reconstruction domain for $q(T,\mathbf{x})$.
        sensors : Sensor or jnp.ndarray
            Adjoint evaluation locations. Image reconstruction typically uses
            ``domain.grid`` to evaluate $q_T$ on the full grid.
        sources : Sensor
            Source geometry on $\Gamma$, used to construct the boundary
            beam parameters (same role as in time reversal).
        ts : jnp.ndarray
            Time grid, shape (Nt,). Reserved for interface consistency.
        data_domain : Domain
            Domain describing the $(t,\mathbf{x}_s)$ grid of the boundary
            data. Its `dx[0]` is used as the time step $\Delta t$.
        data_wpt : MSWPT
            MSWPT instance for analysing the boundary data / source.

        window : jnp.ndarray, optional
            Sampled acquisition window. It may have the same shape as ``data``
            or shape (Nt,), in which case it is broadcast over sensors. The
            default is one on the sampled acquisition array and assumes the
            residual is negligible at the temporal endpoints; otherwise pass a
            taper that vanishes there.

        Returns
        -------
        jnp.ndarray
            Principal-symbol approximation to $P^*r$ for unweighted image and
            data $L^2$ pairings. This is not the exact transpose of the
            thresholded discrete MSGB forward solver.
        """
        q_T, _ = self.adjoint_with_params(
            data,
            domain,
            sensors,
            sources,
            ts,
            data_domain,
            data_wpt,
            window=window,
        )
        return q_T

    def adjoint_with_params(
        self,
        data: jnp.ndarray,
        domain: Domain,
        sensors: Union[Sensor, jnp.ndarray],
        sources: Sensor,
        ts: jnp.ndarray,
        data_domain: Domain,
        data_wpt: MSWPT,
        *,
        window: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
        r"""
        Adjoint MSGB solve plus diagnostic beam parameters.

        This is the explicit diagnostic variant of :meth:`adjoint`. Most users
        should call :meth:`adjoint`, which returns only the adjoint field.

        Returns
        -------
        q_T : jnp.ndarray
            Principal-symbol approximation to $P^*r$ on the reconstruction
            domain under unweighted $L^2$ pairings.
        params : tuple of jnp.ndarray
            Beam parameters used internally.
        """
        if any(domain.periodic):
            raise ValueError(
                "MSGBSolver.adjoint currently assumes non-periodic spatial "
                "boundaries in the reconstruction domain."
            )
        if data.dtype in complex_dtypes and self.use_real:
            raise ValueError(
                "Complex adjoint data requires an *_complex sum_method; "
                "the *_real methods require real-valued boundary data."
            )

        sensor_positions = (
            sensors.positions
            if isinstance(sensors, Sensor)
            else sensors
            if isinstance(sensors, jnp.ndarray)
            else None
        )
        if sensor_positions is None:
            raise ValueError("Unsupported sensor type for `sensors` in adjoint().")

        self._validate_boundary_data(data, data_domain, sources, ts=ts)
        dt = data_domain.dx[0]
        # Boundary sound speed belongs to the acquisition geometry.
        c_at_sources = sources.domain.c_fn(sources.positions)
        source = form_adjoint_source(data, dt, c_at_sources, window)

        inverse_aggregate = self._effective_inverse_aggregate_method()
        params = self._prepare_adj_params(source, data_domain, data_wpt, sources)
        if self.sharding is not None and inverse_aggregate == "all":
            params = self.sharding.shard_tr_params(*params)

        q_T = _compute_tr_result_jit(
            params=params,
            c=domain.c_fn,
            lam=domain.lam,
            sensors=sensor_positions,
            domain_size=domain.grid_size,
            periodic=jnp.array(domain.periodic),
            ode_solver=self.tr_ode_solver,
            aggregate_method=inverse_aggregate,
            solver_config=self.ode_config,
            pallas_config=self._effective_pallas_config(
                domain.periodic, aggregate_method=inverse_aggregate
            ),
        )

        q_T = apply_adjoint_image_weight(q_T, domain.c_fn(sensor_positions))

        return q_T, params
