"""Experimental Pallas kernels for real Gaussian beam accumulation.

GPU and TPU use tiled kernels; CPU supports tests through interpret mode only.
"""

from __future__ import annotations

import math

import jax
from jax import lax
import jax.numpy as jnp
from beamax.gb.pallas_config import PallasConfig, RoundingMode


def _load_pallas():
    try:
        from jax.experimental import pallas
    except ImportError as error:  # pragma: no cover - version dependent
        raise RuntimeError(
            "This JAX build does not expose Pallas; "
            f"Beamax is running with JAX {jax.__version__}."
        ) from error
    return pallas


def _triton_compiler_params(config: PallasConfig):
    try:
        from jax.experimental.pallas import triton
    except ImportError as error:  # pragma: no cover - version dependent
        raise RuntimeError(
            "This JAX build does not expose the Pallas Triton backend; "
            f"Beamax is running with JAX {jax.__version__}."
        ) from error
    return triton.CompilerParams(
        num_warps=config.gpu_num_warps,
        num_stages=config.gpu_num_stages,
    )


def _tpu_compiler_params():
    try:
        from jax.experimental.pallas import tpu
    except ImportError as error:  # pragma: no cover - version dependent
        raise RuntimeError(
            "This JAX build does not expose the Mosaic TPU Pallas backend; "
            f"Beamax is running with JAX {jax.__version__}."
        ) from error
    return tpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary"))


def _pallas_call_with_optional_initial(
    pl,
    *,
    kernel,
    accumulate_kernel,
    output_shape,
    grid,
    base_specs,
    output_spec,
    initial_operand_index: int,
    initial_field_was_none: bool,
    interpret: bool,
    compiler_params,
    name: str,
):
    """Build the no-carry or aliased-carry Pallas call."""
    if initial_field_was_none:
        call = pl.pallas_call(
            kernel,
            out_shape=output_shape,
            grid=grid,
            in_specs=base_specs,
            out_specs=output_spec,
            interpret=interpret,
            compiler_params=compiler_params,
            name=name,
        )

        def call_without_dummy_initial(*operands):
            return call(*operands[:-1])

        return call_without_dummy_initial

    return pl.pallas_call(
        accumulate_kernel,
        out_shape=output_shape,
        grid=grid,
        in_specs=[*base_specs, output_spec],
        out_specs=output_spec,
        input_output_aliases={initial_operand_index: 0},
        interpret=interpret,
        compiler_params=compiler_params,
        name=f"{name}_accumulate",
    )


def _round_to_nearest_even_with_floor(value):
    r"""Express ``round`` using primitives supported by Pallas Triton.

    Correct $\lfloor x+\tfrac{1}{2}\rfloor$ at half-integers to match JAX's
    ties-to-even rule.
    """
    upper = jnp.floor(value + 0.5)
    upper_is_odd = upper - 2.0 * jnp.floor(0.5 * upper) == 1.0
    is_half_integer = upper - value == 0.5
    return jnp.where(is_half_integer & upper_is_odd, upper - 1.0, upper)


def _round_for_minimum_image(value, rounding_mode: RoundingMode):
    if rounding_mode == "half_open":
        # Map exact ties into the half-open interval [-L/2, L/2).
        return jnp.floor(value + 0.5)
    return _round_to_nearest_even_with_floor(value)


def _minimum_image_displacement(
    displacement,
    domain_size,
    periodic,
    *,
    rounding_mode: RoundingMode = "nearest_even",
    periodic_axis: bool | None = None,
):
    """Map periodic displacements to the nearest domain image."""
    if periodic_axis is False:
        return displacement
    image_index = _round_for_minimum_image(displacement / domain_size, rounding_mode)
    wrapped = displacement - domain_size * image_index
    if periodic_axis is True:
        return wrapped
    return displacement - domain_size * image_index * periodic


def _validate_inputs(
    xt: jax.Array,
    pt: jax.Array,
    Mt: jax.Array,
    At: jax.Array,
    omega0: jax.Array,
    sensors: jax.Array,
    domain_size: jax.Array,
    periodic: jax.Array,
    config: PallasConfig,
) -> tuple[int, int, int, int]:
    """Validate the static shape contract needed by ``BlockSpec`` objects."""
    if xt.ndim != 3:
        raise ValueError(f"xt must have shape (B, Nt, d); got {xt.shape}.")

    num_beams, num_times, ndim = xt.shape
    if num_beams == 0:
        raise ValueError("xt must contain at least one beam.")
    if num_times == 0:
        raise ValueError("xt must contain at least one time sample.")
    if not 1 <= ndim <= 3:
        raise ValueError(f"Pallas GB accumulation supports d in [1, 3]; got {ndim}.")
    if pt.shape != xt.shape:
        raise ValueError(f"pt must have shape {xt.shape}; got {pt.shape}.")
    if Mt.shape != (num_beams, num_times, ndim, ndim):
        raise ValueError(
            "Mt must have shape (B, Nt, d, d); got "
            f"{Mt.shape} for B={num_beams}, Nt={num_times}, d={ndim}."
        )
    if At.shape not in {(num_beams, num_times), (num_beams, num_times, 1)}:
        raise ValueError(f"At must have shape (B, Nt) or (B, Nt, 1); got {At.shape}.")
    if omega0.shape != (num_beams,):
        raise ValueError(f"omega0 must have shape ({num_beams},); got {omega0.shape}.")
    if sensors.ndim != 2 or sensors.shape[1] != ndim:
        raise ValueError(f"sensors must have shape (Ns, {ndim}); got {sensors.shape}.")
    if sensors.shape[0] == 0:
        raise ValueError("sensors must contain at least one position.")
    if domain_size.shape != (ndim,):
        raise ValueError(
            f"domain_size must have shape ({ndim},); got {domain_size.shape}."
        )
    if periodic.shape != (ndim,):
        raise ValueError(f"periodic must have shape ({ndim},); got {periodic.shape}.")
    if not jnp.issubdtype(xt.dtype, jnp.floating):
        raise ValueError(f"xt must have a floating dtype; got {xt.dtype}.")
    if config.periodic_axes is not None and len(config.periodic_axes) != ndim:
        raise ValueError(
            f"periodic_axes must have length {ndim}; got {config.periodic_axes}."
        )

    return num_beams, num_times, sensors.shape[0], ndim


def sum_gaussian_beam_real_pallas(
    xt: jax.Array,
    pt: jax.Array,
    Mt: jax.Array,
    At: jax.Array,
    omega0: jax.Array,
    sensors: jax.Array,
    domain_size: jax.Array,
    periodic: jax.Array,
    *,
    interpret: bool | None = None,
    config: PallasConfig | None = None,
    initial_field: jax.Array | None = None,
    return_padded: bool = False,
) -> jax.Array:
    """Evaluate and sum real Gaussian beams with an output-stationary kernel.

    Parameters
    ----------
    xt, pt : array, shape (B, Nt, d)
        Beam centres and momenta along the trajectories.
    Mt : complex array, shape (B, Nt, d, d)
        Beam Hessians along the trajectories.
    At : complex array, shape (B, Nt) or (B, Nt, 1)
        Beam amplitudes along the trajectories.
    omega0 : array, shape (B,)
        Beam angular frequencies.
    sensors : array, shape (Ns, d)
        Evaluation positions.
    domain_size, periodic : array, shape (d,)
        Periodic wrapping parameters.
    config : PallasConfig, optional
        Static kernel layout, launch, wrapping, and batch-accumulation
        configuration. Defaults to :class:`PallasConfig`.
    initial_field : array, shape (Nt, Ns) or (Nt, padded_Ns), optional
        Existing field to include. GPU lowering may alias it to the output.
    return_padded : bool, default=False
        Retain detector padding for internal batched scans.
    interpret : bool, optional
        Run the Pallas HLO interpreter.  Defaults to ``True`` on CPU and
        ``False`` on accelerators.

    Returns
    -------
    array, shape (Nt, Ns)
        Real field summed over the beam axis.

    Notes
    -----
    Reduction order differs from vectorized ``jnp.sum``.
    """
    pl = _load_pallas()

    if config is None:
        config = PallasConfig()
    elif not isinstance(config, PallasConfig):
        raise TypeError(
            "config must be a PallasConfig instance or None; got "
            f"{type(config).__name__}."
        )
    if not isinstance(return_padded, bool):
        raise ValueError("return_padded must be a bool.")

    sensor_block_size = config.sensor_block_size
    xt = jnp.asarray(xt)
    pt = jnp.asarray(pt)
    Mt = jnp.asarray(Mt)
    At = jnp.asarray(At)
    omega0 = jnp.asarray(omega0)
    sensors = jnp.asarray(sensors)
    domain_size = jnp.asarray(domain_size)
    periodic = jnp.asarray(periodic)

    num_beams, num_times, num_sensors, ndim = _validate_inputs(
        xt,
        pt,
        Mt,
        At,
        omega0,
        sensors,
        domain_size,
        periodic,
        config,
    )

    backend = jax.default_backend()
    if interpret is None:
        interpret = backend == "cpu"
    if backend == "cpu" and not interpret:
        raise RuntimeError(
            "Pallas has no CPU code-generation backend; use interpret=True for "
            "correctness testing or select a JAX accelerator backend."
        )
    if backend not in {"cpu", "gpu", "tpu"}:
        raise RuntimeError(f"Unsupported Pallas backend {backend!r}.")
    if backend == "tpu":
        if xt.dtype != jnp.float32:
            raise ValueError(
                "The TPU Pallas accumulator currently requires float32 inputs; "
                f"got {xt.dtype}."
            )
        if sensor_block_size % 128:
            raise ValueError(
                "sensor_block_size must be a multiple of 128 on TPU; got "
                f"{sensor_block_size}."
            )
        if config.tpu_layout == "beam_sensor" and config.tpu_beam_block_size % 8:
            raise ValueError(
                "tpu_beam_block_size must be a multiple of 8 for the "
                f"beam_sensor layout; got {config.tpu_beam_block_size}."
            )
        if config.tpu_layout == "time_sensor" and config.tpu_time_block_size % 8:
            raise ValueError(
                "tpu_time_block_size must be a multiple of 8 for the "
                f"time_sensor layout; got {config.tpu_time_block_size}."
            )
    elif backend == "gpu" and not interpret:
        if sensor_block_size & (sensor_block_size - 1):
            raise ValueError(
                "sensor_block_size must be a power of two for the Triton GPU "
                f"lowering; got {sensor_block_size}."
            )

    # Convert sensor-invariant complex values before entering the tiled kernel.
    real_dtype = xt.dtype
    pt = pt.astype(real_dtype)
    Mr = jnp.real(Mt).astype(real_dtype)
    Mi = jnp.imag(Mt).astype(real_dtype)
    At_flat = At[..., 0] if At.ndim == 3 else At
    amplitude = (2.0 * jnp.abs(At_flat)).astype(real_dtype)
    at_real = jnp.real(At_flat)
    at_imag = jnp.imag(At_flat)
    angle_real = jnp.where(
        (at_real == 0) & (at_imag == 0),
        jnp.asarray(1e-12, dtype=at_real.dtype),
        at_real,
    )
    angle = jnp.arctan2(at_imag, angle_real).astype(real_dtype)
    omega0 = omega0.astype(real_dtype)
    sensors = sensors.astype(real_dtype)
    domain_size = domain_size.astype(real_dtype)
    periodic = periodic.astype(real_dtype)

    padded_sensors = math.ceil(num_sensors / sensor_block_size) * sensor_block_size
    sensors = jnp.pad(sensors, ((0, padded_sensors - num_sensors), (0, 0)))

    initial_field_was_none = initial_field is None
    if initial_field is None:
        # Preserve the custom-JVP signature without an output-sized zero input.
        initial_field = jnp.zeros((1,), dtype=real_dtype)
    else:
        initial_field = jnp.asarray(initial_field, dtype=real_dtype)
        if initial_field.shape == (num_times, num_sensors):
            initial_field = jnp.pad(
                initial_field,
                ((0, 0), (0, padded_sensors - num_sensors)),
            )
        elif initial_field.shape != (num_times, padded_sensors):
            raise ValueError(
                "initial_field must have shape "
                f"({num_times}, {num_sensors}) or "
                f"({num_times}, {padded_sensors}); got {initial_field.shape}."
            )

    if backend == "tpu":
        # Beam tiles share output windows and must remain the ordered grid axis.
        if config.tpu_grid_order == "time_sensor":

            def unpack_grid(first_idx, second_idx, beam_idx):
                return first_idx, second_idx, beam_idx

            def make_grid(time_blocks, sensor_blocks, beam_blocks):
                return time_blocks, sensor_blocks, beam_blocks

        else:

            def unpack_grid(first_idx, second_idx, beam_idx):
                return second_idx, first_idx, beam_idx

            def make_grid(time_blocks, sensor_blocks, beam_blocks):
                return sensor_blocks, time_blocks, beam_blocks

        compiler_params = _tpu_compiler_params()

        if config.tpu_layout == "beam_sensor":
            # Terminal layout maps the native 8 x 128 tile to beams x sensors.
            beam_block_size = config.tpu_beam_block_size
            padded_beams = math.ceil(num_beams / beam_block_size) * beam_block_size
            beam_padding = padded_beams - num_beams

            def pad_beams(value):
                return jnp.pad(
                    value,
                    ((0, beam_padding),) + ((0, 0),) * (value.ndim - 1),
                )

            xt = pad_beams(xt)
            pt = pad_beams(pt)
            Mr = pad_beams(Mr)
            Mi = pad_beams(Mi)
            amplitude = pad_beams(amplitude)
            angle = pad_beams(angle)
            omega0 = pad_beams(omega0)

            def tpu_beam_sensor_kernel(
                xt_ref,
                pt_ref,
                Mr_ref,
                Mi_ref,
                amplitude_ref,
                angle_ref,
                omega_ref,
                sensors_ref,
                domain_ref,
                periodic_ref,
                initial_ref,
                output_ref,
            ):
                delta = []
                for axis in range(ndim):
                    displacement = (
                        sensors_ref[axis, :][None, :] - xt_ref[0, :, axis][:, None]
                    )
                    displacement = _minimum_image_displacement(
                        displacement,
                        domain_ref[axis],
                        periodic_ref[axis],
                        rounding_mode=config.rounding_mode,
                        periodic_axis=(
                            None
                            if config.periodic_axes is None
                            else config.periodic_axes[axis]
                        ),
                    )
                    delta.append(displacement)

                phase_linear = sum(
                    delta[axis] * pt_ref[0, :, axis][:, None] for axis in range(ndim)
                )
                phase_quadratic = 0.5 * sum(
                    delta[row] * Mr_ref[0, :, row * ndim + col][:, None] * delta[col]
                    for row in range(ndim)
                    for col in range(ndim)
                )
                decay_quadratic = 0.5 * sum(
                    delta[row] * Mi_ref[0, :, row * ndim + col][:, None] * delta[col]
                    for row in range(ndim)
                    for col in range(ndim)
                )

                omega = omega_ref[:, 0][:, None]
                phase = (
                    omega * (phase_linear + phase_quadratic)
                    + angle_ref[0, :, 0][:, None]
                )
                contribution = (
                    amplitude_ref[0, :, 0][:, None]
                    * jnp.cos(phase)
                    * jnp.exp(-omega * decay_quadratic)
                )
                partial_sum = jnp.sum(contribution, axis=0)

                @pl.when(pl.program_id(2) == 0)
                def initialize_output():
                    if initial_field_was_none:
                        output_ref[...] = jnp.zeros_like(output_ref[...])
                    else:
                        output_ref[0, 0, :] = initial_ref[0, 0, :]

                output_ref[0, 0, :] += partial_sum

            def trajectory_index(first_idx, second_idx, beam_idx):
                time_idx, _, beam_idx = unpack_grid(first_idx, second_idx, beam_idx)
                return time_idx, beam_idx, 0

            def beam_index(first_idx, second_idx, beam_idx):
                del first_idx, second_idx
                return beam_idx, 0

            def sensor_index(first_idx, second_idx, beam_idx):
                _, sensor_idx, _ = unpack_grid(first_idx, second_idx, beam_idx)
                return 0, sensor_idx

            def spatial_index(first_idx, second_idx, beam_idx):
                del first_idx, second_idx, beam_idx
                return (0,)

            def output_index(first_idx, second_idx, beam_idx):
                time_idx, sensor_idx, _ = unpack_grid(first_idx, second_idx, beam_idx)
                return time_idx, 0, sensor_idx

            def initial_index(first_idx, second_idx, beam_idx):
                time_idx, sensor_idx, _ = unpack_grid(first_idx, second_idx, beam_idx)
                return time_idx, 0, sensor_idx

            trajectory_spec = pl.BlockSpec((1, beam_block_size, ndim), trajectory_index)
            matrix_spec = pl.BlockSpec(
                (1, beam_block_size, ndim * ndim), trajectory_index
            )
            beam_time_spec = pl.BlockSpec((1, beam_block_size, 1), trajectory_index)
            beam_spec = pl.BlockSpec((beam_block_size, 1), beam_index)
            sensor_spec = pl.BlockSpec((ndim, sensor_block_size), sensor_index)
            spatial_parameter_spec = pl.BlockSpec((ndim,), spatial_index)
            initial_spec = (
                pl.BlockSpec((1,), spatial_index)
                if initial_field_was_none
                else pl.BlockSpec((1, 1, sensor_block_size), initial_index)
            )
            output_spec = pl.BlockSpec((1, 1, sensor_block_size), output_index)

            call = pl.pallas_call(
                tpu_beam_sensor_kernel,
                out_shape=jax.ShapeDtypeStruct(
                    (num_times, 1, padded_sensors), dtype=real_dtype
                ),
                grid=make_grid(
                    num_times,
                    padded_sensors // sensor_block_size,
                    padded_beams // beam_block_size,
                ),
                in_specs=[
                    trajectory_spec,
                    trajectory_spec,
                    matrix_spec,
                    matrix_spec,
                    beam_time_spec,
                    beam_time_spec,
                    beam_spec,
                    sensor_spec,
                    spatial_parameter_spec,
                    spatial_parameter_spec,
                    initial_spec,
                ],
                out_specs=output_spec,
                interpret=interpret,
                compiler_params=compiler_params,
                name="beamax_gaussian_beam_real_sum_tpu_beam_sensor",
            )

            def beam_sensor_primal_call(
                xt,
                pt,
                Mr,
                Mi,
                amplitude,
                angle,
                omega0,
                sensors,
                domain_size,
                periodic,
                initial_field,
            ):
                result = call(
                    jnp.swapaxes(xt, 0, 1),
                    jnp.swapaxes(pt, 0, 1),
                    jnp.swapaxes(Mr, 0, 1).reshape(
                        (num_times, padded_beams, ndim * ndim)
                    ),
                    jnp.swapaxes(Mi, 0, 1).reshape(
                        (num_times, padded_beams, ndim * ndim)
                    ),
                    jnp.swapaxes(amplitude, 0, 1)[..., None],
                    jnp.swapaxes(angle, 0, 1)[..., None],
                    omega0[:, None],
                    jnp.swapaxes(sensors, 0, 1),
                    domain_size,
                    periodic,
                    (
                        initial_field
                        if initial_field_was_none
                        else initial_field[:, None, :]
                    ),
                )
                return result[:, 0, :]

            primal_call = beam_sensor_primal_call

        else:
            # Forward layout maps the native tile to time x sensors.
            beam_block_size = config.tpu_beam_block_size
            time_block_size = config.tpu_time_block_size
            padded_beams = math.ceil(num_beams / beam_block_size) * beam_block_size
            padded_times = math.ceil(num_times / time_block_size) * time_block_size
            beam_padding = padded_beams - num_beams
            time_padding = padded_times - num_times

            def pad_beams_and_times(value):
                return jnp.pad(
                    value,
                    ((0, beam_padding), (0, time_padding))
                    + ((0, 0),) * (value.ndim - 2),
                )

            xt = pad_beams_and_times(xt)
            pt = pad_beams_and_times(pt)
            Mr = pad_beams_and_times(Mr)
            Mi = pad_beams_and_times(Mi)
            amplitude = pad_beams_and_times(amplitude)
            angle = pad_beams_and_times(angle)
            omega0 = jnp.pad(omega0, ((0, beam_padding),))
            if not initial_field_was_none:
                initial_field = jnp.pad(initial_field, ((0, time_padding), (0, 0)))

            def tpu_time_sensor_kernel(
                xt_ref,
                pt_ref,
                Mr_ref,
                Mi_ref,
                amplitude_ref,
                angle_ref,
                omega_ref,
                sensors_ref,
                domain_ref,
                periodic_ref,
                initial_ref,
                output_ref,
            ):
                delta = []
                for axis in range(ndim):
                    displacement = (
                        sensors_ref[axis, :][None, None, :]
                        - xt_ref[:, :, axis][:, :, None]
                    )
                    displacement = _minimum_image_displacement(
                        displacement,
                        domain_ref[axis],
                        periodic_ref[axis],
                        rounding_mode=config.rounding_mode,
                        periodic_axis=(
                            None
                            if config.periodic_axes is None
                            else config.periodic_axes[axis]
                        ),
                    )
                    delta.append(displacement)

                phase_linear = sum(
                    delta[axis] * pt_ref[:, :, axis][:, :, None] for axis in range(ndim)
                )
                phase_quadratic = 0.5 * sum(
                    delta[row] * Mr_ref[:, :, row * ndim + col][:, :, None] * delta[col]
                    for row in range(ndim)
                    for col in range(ndim)
                )
                decay_quadratic = 0.5 * sum(
                    delta[row] * Mi_ref[:, :, row * ndim + col][:, :, None] * delta[col]
                    for row in range(ndim)
                    for col in range(ndim)
                )

                omega = omega_ref[:, 0, 0][:, None, None]
                phase = (
                    omega * (phase_linear + phase_quadratic)
                    + angle_ref[:, :, 0][:, :, None]
                )
                contribution = (
                    amplitude_ref[:, :, 0][:, :, None]
                    * jnp.cos(phase)
                    * jnp.exp(-omega * decay_quadratic)
                )
                partial_sum = jnp.sum(contribution, axis=0)

                @pl.when(pl.program_id(2) == 0)
                def initialize_output():
                    if initial_field_was_none:
                        output_ref[...] = jnp.zeros_like(output_ref[...])
                    else:
                        output_ref[...] = initial_ref[...]

                output_ref[...] += partial_sum

            def time_trajectory_index(first_idx, second_idx, beam_idx):
                time_idx, _, beam_idx = unpack_grid(first_idx, second_idx, beam_idx)
                return beam_idx, time_idx, 0

            def time_beam_index(first_idx, second_idx, beam_idx):
                del first_idx, second_idx
                return beam_idx, 0, 0

            def time_sensor_index(first_idx, second_idx, beam_idx):
                _, sensor_idx, _ = unpack_grid(first_idx, second_idx, beam_idx)
                return 0, sensor_idx

            def time_spatial_index(first_idx, second_idx, beam_idx):
                del first_idx, second_idx, beam_idx
                return (0,)

            def time_output_index(first_idx, second_idx, beam_idx):
                time_idx, sensor_idx, _ = unpack_grid(first_idx, second_idx, beam_idx)
                return time_idx, sensor_idx

            trajectory_spec = pl.BlockSpec(
                (beam_block_size, time_block_size, ndim), time_trajectory_index
            )
            matrix_spec = pl.BlockSpec(
                (beam_block_size, time_block_size, ndim * ndim),
                time_trajectory_index,
            )
            beam_time_spec = pl.BlockSpec(
                (beam_block_size, time_block_size, 1), time_trajectory_index
            )
            beam_spec = pl.BlockSpec((beam_block_size, 1, 1), time_beam_index)
            sensor_spec = pl.BlockSpec((ndim, sensor_block_size), time_sensor_index)
            spatial_parameter_spec = pl.BlockSpec((ndim,), time_spatial_index)
            output_spec = pl.BlockSpec(
                (time_block_size, sensor_block_size), time_output_index
            )
            initial_spec = (
                pl.BlockSpec((1,), time_spatial_index)
                if initial_field_was_none
                else output_spec
            )

            call = pl.pallas_call(
                tpu_time_sensor_kernel,
                out_shape=jax.ShapeDtypeStruct(
                    (padded_times, padded_sensors), dtype=real_dtype
                ),
                grid=make_grid(
                    padded_times // time_block_size,
                    padded_sensors // sensor_block_size,
                    padded_beams // beam_block_size,
                ),
                in_specs=[
                    trajectory_spec,
                    trajectory_spec,
                    matrix_spec,
                    matrix_spec,
                    beam_time_spec,
                    beam_time_spec,
                    beam_spec,
                    sensor_spec,
                    spatial_parameter_spec,
                    spatial_parameter_spec,
                    initial_spec,
                ],
                out_specs=output_spec,
                interpret=interpret,
                compiler_params=compiler_params,
                name="beamax_gaussian_beam_real_sum_tpu_time_sensor",
            )

            def time_sensor_primal_call(
                xt,
                pt,
                Mr,
                Mi,
                amplitude,
                angle,
                omega0,
                sensors,
                domain_size,
                periodic,
                initial_field,
            ):
                return call(
                    xt,
                    pt,
                    Mr.reshape((padded_beams, padded_times, ndim * ndim)),
                    Mi.reshape((padded_beams, padded_times, ndim * ndim)),
                    amplitude[..., None],
                    angle[..., None],
                    omega0[:, None, None],
                    jnp.swapaxes(sensors, 0, 1),
                    domain_size,
                    periodic,
                    initial_field,
                )

            primal_call = time_sensor_primal_call

    else:

        def gpu_partial_sum(
            xt_ref,
            pt_ref,
            Mr_ref,
            Mi_ref,
            amplitude_ref,
            angle_ref,
            omega_ref,
            sensors_ref,
            domain_ref,
            periodic_ref,
        ):
            def add_beam(beam_idx, accumulator):
                delta = []
                for axis in range(ndim):
                    displacement = sensors_ref[:, axis] - xt_ref[beam_idx, 0, axis]
                    displacement = _minimum_image_displacement(
                        displacement,
                        domain_ref[axis],
                        periodic_ref[axis],
                        rounding_mode=config.rounding_mode,
                        periodic_axis=(
                            None
                            if config.periodic_axes is None
                            else config.periodic_axes[axis]
                        ),
                    )
                    delta.append(displacement)

                phase_linear = sum(
                    delta[axis] * pt_ref[beam_idx, 0, axis] for axis in range(ndim)
                )
                phase_quadratic = 0.5 * sum(
                    delta[row] * Mr_ref[beam_idx, 0, row, col] * delta[col]
                    for row in range(ndim)
                    for col in range(ndim)
                )
                decay_quadratic = 0.5 * sum(
                    delta[row] * Mi_ref[beam_idx, 0, row, col] * delta[col]
                    for row in range(ndim)
                    for col in range(ndim)
                )

                omega = omega_ref[beam_idx]
                phase = (
                    omega * (phase_linear + phase_quadratic) + angle_ref[beam_idx, 0]
                )
                contribution = (
                    amplitude_ref[beam_idx, 0]
                    * jnp.cos(phase)
                    * jnp.exp(-omega * decay_quadratic)
                )
                return accumulator + contribution

            zero = jnp.zeros((sensor_block_size,), dtype=real_dtype)
            return lax.fori_loop(0, num_beams, add_beam, zero)

        def gpu_kernel(
            xt_ref,
            pt_ref,
            Mr_ref,
            Mi_ref,
            amplitude_ref,
            angle_ref,
            omega_ref,
            sensors_ref,
            domain_ref,
            periodic_ref,
            output_ref,
        ):
            output_ref[0, :] = gpu_partial_sum(
                xt_ref,
                pt_ref,
                Mr_ref,
                Mi_ref,
                amplitude_ref,
                angle_ref,
                omega_ref,
                sensors_ref,
                domain_ref,
                periodic_ref,
            )

        def gpu_accumulate_kernel(
            xt_ref,
            pt_ref,
            Mr_ref,
            Mi_ref,
            amplitude_ref,
            angle_ref,
            omega_ref,
            sensors_ref,
            domain_ref,
            periodic_ref,
            initial_ref,
            output_ref,
        ):
            partial_sum = gpu_partial_sum(
                xt_ref,
                pt_ref,
                Mr_ref,
                Mi_ref,
                amplitude_ref,
                angle_ref,
                omega_ref,
                sensors_ref,
                domain_ref,
                periodic_ref,
            )
            output_ref[0, :] = initial_ref[0, :] + partial_sum

        trajectory_spec = pl.BlockSpec(
            (num_beams, 1, ndim), lambda time_idx, sensor_idx: (0, time_idx, 0)
        )
        matrix_spec = pl.BlockSpec(
            (num_beams, 1, ndim, ndim),
            lambda time_idx, sensor_idx: (0, time_idx, 0, 0),
        )
        beam_time_spec = pl.BlockSpec(
            (num_beams, 1), lambda time_idx, sensor_idx: (0, time_idx)
        )
        beam_spec = pl.BlockSpec((num_beams,), lambda time_idx, sensor_idx: (0,))
        sensor_spec = pl.BlockSpec(
            (sensor_block_size, ndim),
            lambda time_idx, sensor_idx: (sensor_idx, 0),
        )
        spatial_parameter_spec = pl.BlockSpec(
            (ndim,), lambda time_idx, sensor_idx: (0,)
        )
        output_spec = pl.BlockSpec(
            (1, sensor_block_size),
            lambda time_idx, sensor_idx: (time_idx, sensor_idx),
        )

        compiler_params = None
        if backend == "gpu" and not interpret:
            compiler_params = _triton_compiler_params(config)

        output_shape = jax.ShapeDtypeStruct(
            (num_times, padded_sensors), dtype=real_dtype
        )
        grid = (num_times, padded_sensors // sensor_block_size)
        base_specs = [
            trajectory_spec,
            trajectory_spec,
            matrix_spec,
            matrix_spec,
            beam_time_spec,
            beam_time_spec,
            beam_spec,
            sensor_spec,
            spatial_parameter_spec,
            spatial_parameter_spec,
        ]
        primal_call = _pallas_call_with_optional_initial(
            pl,
            kernel=gpu_kernel,
            accumulate_kernel=gpu_accumulate_kernel,
            output_shape=output_shape,
            grid=grid,
            base_specs=base_specs,
            output_spec=output_spec,
            initial_operand_index=10,
            initial_field_was_none=initial_field_was_none,
            interpret=interpret,
            compiler_params=compiler_params,
            name="beamax_gaussian_beam_real_sum_gpu",
        )

    def reference_call(*operands):
        from beamax.gb.core import _sum_gaussian_beam_real_components_xla

        return _sum_gaussian_beam_real_components_xla(
            *operands[:-1],
            rounding_mode=config.rounding_mode,
            periodic_axes=config.periodic_axes,
            initial_field=None if initial_field_was_none else operands[-1],
        )

    @jax.custom_jvp
    def differentiable_call(*operands):
        return primal_call(*operands)

    @differentiable_call.defjvp
    def differentiable_call_jvp(primals, tangents):
        primal_output = differentiable_call(*primals)
        _, tangent_output = jax.jvp(
            reference_call,
            primals,
            tangents,
        )
        return primal_output, tangent_output

    result = differentiable_call(
        xt,
        pt,
        Mr,
        Mi,
        amplitude,
        angle,
        omega0,
        sensors,
        domain_size,
        periodic,
        initial_field,
    )
    sensor_limit = padded_sensors if return_padded else num_sensors
    return result[:num_times, :sensor_limit]


def _complex_multiply(ar, ai, br, bi):
    """Multiply complex numbers represented by real/imaginary pairs."""
    return ar * br - ai * bi, ar * bi + ai * br


def _complex_reciprocal(real, imag):
    denominator = real * real + imag * imag
    return real / denominator, -imag / denominator


def _complex_divide(ar, ai, br, bi):
    reciprocal_real, reciprocal_imag = _complex_reciprocal(br, bi)
    return _complex_multiply(ar, ai, reciprocal_real, reciprocal_imag)


def _complex_square_root(real, imag):
    """Principal complex square root using Pallas-supported real primitives."""
    magnitude = jnp.sqrt(real * real + imag * imag)
    root_real = jnp.sqrt(jnp.maximum(0.5 * (magnitude + real), 0.0))
    root_imag_abs = jnp.sqrt(jnp.maximum(0.5 * (magnitude - real), 0.0))
    root_imag = jnp.where(imag < 0.0, -root_imag_abs, root_imag_abs)
    return root_real, root_imag


def _validate_hom_diag_3d_inputs(
    x0: jax.Array,
    p0: jax.Array,
    M0: jax.Array,
    a0: jax.Array,
    omega0: jax.Array,
    mode: jax.Array,
    c0: jax.Array,
    ts: jax.Array,
    sensors: jax.Array,
    domain_size: jax.Array,
    periodic: jax.Array,
    config: PallasConfig,
) -> tuple[int, int, int]:
    """Validate the static contract for the fused 3D homogeneous kernel."""
    if x0.ndim != 2 or x0.shape[1] != 3:
        raise ValueError(f"x0 must have shape (B, 3); got {x0.shape}.")
    num_beams = x0.shape[0]
    if num_beams == 0:
        raise ValueError("x0 must contain at least one beam.")
    if p0.shape != x0.shape:
        raise ValueError(f"p0 must have shape {x0.shape}; got {p0.shape}.")
    if M0.shape != (num_beams, 3, 3):
        raise ValueError(f"M0 must have shape ({num_beams}, 3, 3); got {M0.shape}.")
    if a0.shape != (num_beams,):
        raise ValueError(f"a0 must have shape ({num_beams},); got {a0.shape}.")
    if omega0.shape != (num_beams,):
        raise ValueError(f"omega0 must have shape ({num_beams},); got {omega0.shape}.")
    if mode.shape != (num_beams,):
        raise ValueError(f"mode must have shape ({num_beams},); got {mode.shape}.")
    if c0.ndim != 0:
        raise ValueError(f"c0 must be a scalar; got shape {c0.shape}.")
    if ts.ndim != 1 or ts.shape[0] == 0:
        raise ValueError(f"ts must have non-empty shape (Nt,); got {ts.shape}.")
    if sensors.ndim != 2 or sensors.shape[1] != 3 or sensors.shape[0] == 0:
        raise ValueError(
            f"sensors must have non-empty shape (Ns, 3); got {sensors.shape}."
        )
    if domain_size.shape != (3,):
        raise ValueError(f"domain_size must have shape (3,); got {domain_size.shape}.")
    if periodic.shape != (3,):
        raise ValueError(f"periodic must have shape (3,); got {periodic.shape}.")
    if config.periodic_axes is not None and len(config.periodic_axes) != 3:
        raise ValueError(
            "periodic_axes must have length 3 for the fused kernel; got "
            f"{config.periodic_axes}."
        )

    real_inputs = {
        "x0": x0,
        "p0": p0,
        "omega0": omega0,
        "mode": mode,
        "c0": c0,
        "ts": ts,
        "sensors": sensors,
        "domain_size": domain_size,
    }
    for name, value in real_inputs.items():
        if value.dtype != jnp.float32:
            raise ValueError(
                "The fused homogeneous-diagonal Pallas kernel requires "
                f"float32 {name}; got {value.dtype}."
            )
    if M0.dtype != jnp.complex64:
        raise ValueError(
            "The fused homogeneous-diagonal Pallas kernel requires complex64 "
            f"M0; got {M0.dtype}."
        )
    if a0.dtype != jnp.complex64:
        raise ValueError(
            "The fused homogeneous-diagonal Pallas kernel requires complex64 "
            f"a0; got {a0.dtype}."
        )

    return num_beams, ts.shape[0], sensors.shape[0]


def _sum_gaussian_beam_real_hom_diag_3d_reference(
    x0,
    p0,
    alpha_real,
    alpha_imag,
    a0_real,
    a0_imag,
    omega0,
    mode,
    c0,
    ts,
    sensors,
    domain_size,
    periodic,
    initial_field,
    *,
    rounding_mode: RoundingMode,
    periodic_axes: tuple[bool, ...] | None,
    initial_field_was_none: bool,
):
    """Portable HIGHEST-precision oracle and derivative rule for the fusion."""
    from beamax.gb.gb_solvers import solve_hom_diag

    alpha = alpha_real + 1j * alpha_imag
    M0 = jnp.eye(3, dtype=alpha.dtype)[None, :, :] * alpha[:, :, None]
    a0 = a0_real + 1j * a0_imag

    xt, pt, Mt, At = solve_hom_diag(
        x0,
        p0,
        M0,
        a0,
        mode,
        ts,
        lambda unused_position: c0,
    )
    wrapped_axes = []
    for axis in range(3):
        wrapped = xt[..., axis] - domain_size[axis] * jnp.floor(
            xt[..., axis] / domain_size[axis]
        )
        if periodic_axes is None:
            wrapped = jnp.where(periodic[axis], wrapped, xt[..., axis])
        elif not periodic_axes[axis]:
            wrapped = xt[..., axis]
        wrapped_axes.append(wrapped)
    xt = jnp.stack(wrapped_axes, axis=-1)

    At_flat = At[..., 0]
    trajectory_dtype = x0.dtype
    xt = xt.astype(trajectory_dtype)
    pt = pt.astype(trajectory_dtype)
    matrix_real = jnp.real(Mt).astype(trajectory_dtype)
    matrix_imag = jnp.imag(Mt).astype(trajectory_dtype)
    At_flat = At_flat.astype(jnp.complex64)
    angle_real = jnp.where(
        (jnp.real(At_flat) == 0) & (jnp.imag(At_flat) == 0),
        jnp.asarray(1e-12, dtype=At_flat.real.dtype),
        jnp.real(At_flat),
    )
    from beamax.gb.core import _sum_gaussian_beam_real_components_xla

    return _sum_gaussian_beam_real_components_xla(
        xt,
        pt,
        matrix_real,
        matrix_imag,
        (2.0 * jnp.abs(At_flat)).astype(trajectory_dtype),
        jnp.arctan2(jnp.imag(At_flat), angle_real).astype(trajectory_dtype),
        omega0,
        sensors,
        domain_size,
        periodic,
        rounding_mode=rounding_mode,
        periodic_axes=periodic_axes,
        initial_field=None if initial_field_was_none else initial_field,
    )


def sum_gaussian_beam_real_hom_diag_3d_pallas(
    x0: jax.Array,
    p0: jax.Array,
    M0: jax.Array,
    a0: jax.Array,
    omega0: jax.Array,
    mode: jax.Array,
    c0: jax.Array,
    ts: jax.Array,
    sensors: jax.Array,
    domain_size: jax.Array,
    periodic: jax.Array,
    *,
    config: PallasConfig | None = None,
    initial_field: jax.Array | None = None,
    return_padded: bool = False,
    interpret: bool | None = None,
) -> jax.Array:
    """Fuse homogeneous 3D propagation and real sensor accumulation.

    The primal evaluates ``solve_hom_diag`` within each output tile instead of
    materializing trajectories. Differentiation uses portable XLA.
    """
    pl = _load_pallas()

    if config is None:
        config = PallasConfig()
    elif not isinstance(config, PallasConfig):
        raise TypeError(
            "config must be a PallasConfig instance or None; got "
            f"{type(config).__name__}."
        )
    if not isinstance(return_padded, bool):
        raise ValueError("return_padded must be a bool.")

    x0 = jnp.asarray(x0)
    p0 = jnp.asarray(p0)
    M0 = jnp.asarray(M0)
    a0 = jnp.asarray(a0)
    omega0 = jnp.asarray(omega0)
    mode = jnp.asarray(mode)
    c0 = jnp.asarray(c0)
    ts = jnp.asarray(ts)
    sensors = jnp.asarray(sensors)
    domain_size = jnp.asarray(domain_size)
    periodic = jnp.asarray(periodic)
    num_beams, num_times, num_sensors = _validate_hom_diag_3d_inputs(
        x0,
        p0,
        M0,
        a0,
        omega0,
        mode,
        c0,
        ts,
        sensors,
        domain_size,
        periodic,
        config,
    )

    backend = jax.default_backend()
    if interpret is None:
        interpret = backend == "cpu"
    if backend == "cpu" and not interpret:
        raise RuntimeError(
            "Pallas has no CPU code-generation backend; use interpret=True for "
            "correctness testing or select a GPU backend."
        )
    if backend not in {"cpu", "gpu"}:
        raise RuntimeError(
            "The fused homogeneous-diagonal kernel supports a GPU or "
            f"CPU interpret mode; got backend {backend!r}."
        )
    if backend == "gpu" and not interpret:
        for name, value in {
            "sensor_block_size": config.sensor_block_size,
            "gpu_time_block_size": config.gpu_time_block_size,
        }.items():
            if value & (value - 1):
                raise ValueError(
                    f"{name} must be a power of two for the GPU lowering; got {value}."
                )

    time_block_size = config.gpu_time_block_size
    sensor_block_size = config.sensor_block_size
    padded_times = math.ceil(num_times / time_block_size) * time_block_size
    padded_sensors = math.ceil(num_sensors / sensor_block_size) * sensor_block_size
    ts = jnp.pad(ts, ((0, padded_times - num_times),))
    sensors = jnp.pad(sensors, ((0, padded_sensors - num_sensors), (0, 0)))

    initial_field_was_none = initial_field is None
    if initial_field is None:
        initial_field = jnp.zeros((1,), dtype=jnp.float32)
    else:
        initial_field = jnp.asarray(initial_field)
        if initial_field.dtype != jnp.float32:
            raise ValueError(
                "The fused homogeneous-diagonal Pallas kernel requires a "
                f"float32 initial_field; got {initial_field.dtype}."
            )
        if initial_field.shape == (num_times, num_sensors):
            initial_field = jnp.pad(
                initial_field,
                (
                    (0, padded_times - num_times),
                    (0, padded_sensors - num_sensors),
                ),
            )
        elif initial_field.shape != (padded_times, padded_sensors):
            raise ValueError(
                "initial_field must have shape "
                f"({num_times}, {num_sensors}) or "
                f"({padded_times}, {padded_sensors}); got "
                f"{initial_field.shape}."
            )

    alpha = jnp.diagonal(M0, axis1=1, axis2=2)
    alpha_real = jnp.real(alpha).astype(jnp.float32)
    alpha_imag = jnp.imag(alpha).astype(jnp.float32)
    a0_real = jnp.real(a0).astype(jnp.float32)
    a0_imag = jnp.imag(a0).astype(jnp.float32)
    periodic = periodic.astype(jnp.float32)

    def partial_sum(
        x0_ref,
        p0_ref,
        alpha_real_ref,
        alpha_imag_ref,
        a0_real_ref,
        a0_imag_ref,
        omega_ref,
        mode_ref,
        c_ref,
        ts_ref,
        sensors_ref,
        domain_ref,
        periodic_ref,
    ):
        times = ts_ref[:]

        def add_mode(beam_idx, signed_mode, accumulator):
            px = p0_ref[beam_idx, 0]
            py = p0_ref[beam_idx, 1]
            pz = p0_ref[beam_idx, 2]
            normp = jnp.sqrt(px * px + py * py + pz * pz)
            signed_c = c_ref[...] * signed_mode

            delta = []
            for axis, momentum in enumerate((px, py, pz)):
                centre = x0_ref[beam_idx, axis] + signed_c * momentum * times / normp
                wrapped = centre - domain_ref[axis] * jnp.floor(
                    centre / domain_ref[axis]
                )
                if config.periodic_axes is None:
                    centre = jnp.where(periodic_ref[axis] != 0.0, wrapped, centre)
                elif config.periodic_axes[axis]:
                    centre = wrapped
                displacement = sensors_ref[:, axis][None, :] - centre[:, None]
                displacement = _minimum_image_displacement(
                    displacement,
                    domain_ref[axis],
                    periodic_ref[axis],
                    rounding_mode=config.rounding_mode,
                    periodic_axis=(
                        None
                        if config.periodic_axes is None
                        else config.periodic_axes[axis]
                    ),
                )
                delta.append(displacement)

            inverse_normp = 1.0 / normp
            propagation = signed_c * times * inverse_normp
            u_scale = -signed_c * times * inverse_normp**3
            z_real = []
            z_imag = []
            q_real = []
            q_imag = []
            u = []
            for axis, momentum in enumerate((px, py, pz)):
                ar = alpha_real_ref[beam_idx, axis]
                ai = alpha_imag_ref[beam_idx, axis]
                inverse_real, inverse_imag = _complex_reciprocal(
                    1.0 + propagation * ar,
                    propagation * ai,
                )
                zr, zi = _complex_multiply(ar, ai, inverse_real, inverse_imag)
                z_real.append(zr)
                z_imag.append(zi)
                q_real.append(momentum * zr)
                q_imag.append(momentum * zi)
                u.append(u_scale * momentum)

            scalar_real = jnp.ones_like(times)
            scalar_imag = jnp.zeros_like(times)
            for axis in range(3):
                scalar_real += q_real[axis] * u[axis]
                scalar_imag += q_imag[axis] * u[axis]
            diagonal_real = sum(
                delta[axis] * delta[axis] * z_real[axis][:, None] for axis in range(3)
            )
            diagonal_imag = sum(
                delta[axis] * delta[axis] * z_imag[axis][:, None] for axis in range(3)
            )
            left_real = sum(
                delta[axis] * (z_real[axis] * u[axis])[:, None] for axis in range(3)
            )
            left_imag = sum(
                delta[axis] * (z_imag[axis] * u[axis])[:, None] for axis in range(3)
            )
            right_real = sum(delta[axis] * q_real[axis][:, None] for axis in range(3))
            right_imag = sum(delta[axis] * q_imag[axis][:, None] for axis in range(3))
            correction_real, correction_imag = _complex_multiply(
                left_real, left_imag, right_real, right_imag
            )
            correction_real, correction_imag = _complex_divide(
                correction_real,
                correction_imag,
                scalar_real[:, None],
                scalar_imag[:, None],
            )
            quadratic_real = diagonal_real - correction_real
            quadratic_imag = diagonal_imag - correction_imag

            alpha_products = []
            for left_axis, right_axis in ((1, 2), (0, 2), (0, 1)):
                alpha_products.append(
                    _complex_multiply(
                        alpha_real_ref[beam_idx, left_axis],
                        alpha_imag_ref[beam_idx, left_axis],
                        alpha_real_ref[beam_idx, right_axis],
                        alpha_imag_ref[beam_idx, right_axis],
                    )
                )
            denominator_sum_real = jnp.zeros_like(times)
            denominator_sum_imag = jnp.zeros_like(times)
            momentum_squared = (px * px, py * py, pz * pz)
            complement_axes = ((1, 2), (0, 2), (0, 1))
            for axis in range(3):
                left_axis, right_axis = complement_axes[axis]
                product_real, product_imag = alpha_products[axis]
                term_real = (
                    signed_c * product_real * times
                    + (
                        alpha_real_ref[beam_idx, left_axis]
                        + alpha_real_ref[beam_idx, right_axis]
                    )
                    * normp
                )
                term_imag = (
                    signed_c * product_imag * times
                    + (
                        alpha_imag_ref[beam_idx, left_axis]
                        + alpha_imag_ref[beam_idx, right_axis]
                    )
                    * normp
                )
                denominator_sum_real += momentum_squared[axis] * term_real
                denominator_sum_imag += momentum_squared[axis] * term_imag
            time_scale = signed_c * times / normp**4
            denominator_real = 1.0 + time_scale * denominator_sum_real
            denominator_imag = time_scale * denominator_sum_imag
            root_real, root_imag = _complex_square_root(
                denominator_real, denominator_imag
            )
            amplitude_real, amplitude_imag = _complex_divide(
                a0_real_ref[beam_idx],
                a0_imag_ref[beam_idx],
                root_real,
                root_imag,
            )
            phase_linear = sum(
                delta[axis] * momentum for axis, momentum in enumerate((px, py, pz))
            )
            omega = omega_ref[beam_idx]
            phase = omega * (phase_linear + 0.5 * quadratic_real)
            contribution = (
                2.0
                * (
                    amplitude_real[:, None] * jnp.cos(phase)
                    - amplitude_imag[:, None] * jnp.sin(phase)
                )
                * jnp.exp(-0.5 * omega * quadratic_imag)
            )
            return accumulator + contribution

        def add_beam(beam_idx, accumulator):
            return add_mode(beam_idx, mode_ref[beam_idx], accumulator)

        return lax.fori_loop(
            0,
            num_beams,
            add_beam,
            jnp.zeros((time_block_size, sensor_block_size), dtype=jnp.float32),
        )

    def kernel(
        x0_ref,
        p0_ref,
        alpha_real_ref,
        alpha_imag_ref,
        a0_real_ref,
        a0_imag_ref,
        omega_ref,
        mode_ref,
        c_ref,
        ts_ref,
        sensors_ref,
        domain_ref,
        periodic_ref,
        output_ref,
    ):
        output_ref[:, :] = partial_sum(
            x0_ref,
            p0_ref,
            alpha_real_ref,
            alpha_imag_ref,
            a0_real_ref,
            a0_imag_ref,
            omega_ref,
            mode_ref,
            c_ref,
            ts_ref,
            sensors_ref,
            domain_ref,
            periodic_ref,
        )

    def accumulate_kernel(
        x0_ref,
        p0_ref,
        alpha_real_ref,
        alpha_imag_ref,
        a0_real_ref,
        a0_imag_ref,
        omega_ref,
        mode_ref,
        c_ref,
        ts_ref,
        sensors_ref,
        domain_ref,
        periodic_ref,
        initial_ref,
        output_ref,
    ):
        output_ref[:, :] = initial_ref[:, :] + partial_sum(
            x0_ref,
            p0_ref,
            alpha_real_ref,
            alpha_imag_ref,
            a0_real_ref,
            a0_imag_ref,
            omega_ref,
            mode_ref,
            c_ref,
            ts_ref,
            sensors_ref,
            domain_ref,
            periodic_ref,
        )

    def beam_matrix_index(unused_time_idx, unused_sensor_idx):
        return 0, 0

    def beam_index(unused_time_idx, unused_sensor_idx):
        return (0,)

    def scalar_index(unused_time_idx, unused_sensor_idx):
        return ()

    def time_index(time_idx, unused_sensor_idx):
        return (time_idx,)

    def sensor_index(unused_time_idx, sensor_idx):
        return sensor_idx, 0

    def spatial_index(unused_time_idx, unused_sensor_idx):
        return (0,)

    def output_index(time_idx, sensor_idx):
        return time_idx, sensor_idx

    beam_matrix_spec = pl.BlockSpec((num_beams, 3), beam_matrix_index)
    beam_spec = pl.BlockSpec((num_beams,), beam_index)
    scalar_spec = pl.BlockSpec((), scalar_index)
    time_spec = pl.BlockSpec((time_block_size,), time_index)
    sensor_spec = pl.BlockSpec((sensor_block_size, 3), sensor_index)
    spatial_spec = pl.BlockSpec((3,), spatial_index)
    output_spec = pl.BlockSpec((time_block_size, sensor_block_size), output_index)
    base_specs = [
        beam_matrix_spec,
        beam_matrix_spec,
        beam_matrix_spec,
        beam_matrix_spec,
        beam_spec,
        beam_spec,
        beam_spec,
        beam_spec,
        scalar_spec,
        time_spec,
        sensor_spec,
        spatial_spec,
        spatial_spec,
    ]
    compiler_params = None
    if backend == "gpu" and not interpret:
        compiler_params = _triton_compiler_params(config)

    output_shape = jax.ShapeDtypeStruct(
        (padded_times, padded_sensors), dtype=jnp.float32
    )
    grid = (
        padded_times // time_block_size,
        padded_sensors // sensor_block_size,
    )
    primal_call = _pallas_call_with_optional_initial(
        pl,
        kernel=kernel,
        accumulate_kernel=accumulate_kernel,
        output_shape=output_shape,
        grid=grid,
        base_specs=base_specs,
        output_spec=output_spec,
        initial_operand_index=13,
        initial_field_was_none=initial_field_was_none,
        interpret=interpret,
        compiler_params=compiler_params,
        name="beamax_hom_diag_3d_real_sum_gpu",
    )

    def reference_call(*operands):
        return _sum_gaussian_beam_real_hom_diag_3d_reference(
            *operands,
            rounding_mode=config.rounding_mode,
            periodic_axes=config.periodic_axes,
            initial_field_was_none=initial_field_was_none,
        )

    @jax.custom_jvp
    def differentiable_call(*operands):
        return primal_call(*operands)

    @differentiable_call.defjvp
    def differentiable_call_jvp(primals, tangents):
        primal_output = differentiable_call(*primals)
        _, tangent_output = jax.jvp(reference_call, primals, tangents)
        return primal_output, tangent_output

    result = differentiable_call(
        x0,
        p0,
        alpha_real,
        alpha_imag,
        a0_real,
        a0_imag,
        omega0,
        mode,
        c0,
        ts,
        sensors,
        domain_size,
        periodic,
        initial_field,
    )
    time_limit = padded_times if return_padded else num_times
    sensor_limit = padded_sensors if return_padded else num_sensors
    return result[:time_limit, :sensor_limit]


__all__ = [
    "sum_gaussian_beam_real_hom_diag_3d_pallas",
    "sum_gaussian_beam_real_pallas",
]
