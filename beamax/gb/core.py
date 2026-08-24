from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Complex, Float, Num

from beamax.gb.gb_solvers import SolverConfig, SolverFn

if TYPE_CHECKING:
    from beamax.gb.pallas_config import PallasConfig

__all__ = [
    "compute_gaussian_beam",
    "compute_gaussian_beam_real",
    "compute_gaussian_beam_real_pallas",
    "compute_gaussian_beam_real_TR",
    "compute_gaussian_beam_real_TR_xla_terminal",
    "compute_gaussian_beam_real_TR_pallas_terminal",
    "sum_gaussian_beam_real_trajectories_xla",
]


def wrap_position(
    position: Float[Array, "b t d"],
    domain_size: Float[Array, " d"],
    periodic: Bool[Array, " d"],
) -> Float[Array, "b t d"]:
    """
    Apply periodic wrapping selectively per axis.

    Parameters
    ----------
    position : jnp.ndarray, shape (b, t, d)
    domain_size : jnp.ndarray, shape (d,)
    periodic : jnp.ndarray, shape (d,), bool

    Returns
    -------
    jnp.ndarray, shape (b, t, d)
        Wrapped where `periodic` is True; unchanged elsewhere.
    """
    wrapped = position % domain_size
    return jnp.where(periodic, wrapped, position)


def compute_phase(
    xt: Float[Array, "b t d"],
    pt: Float[Array, "b t d"],
    mt: Complex[Array, "b t d d"],
    sensors: Float[Array, "*S d"],
    domain_size: Float[Array, " d"],
    periodic: Bool[Array, " d"],
) -> Complex[Array, "b t *S"]:
    r"""
    GB phase at sensors:

    $$
    \Phi=p\mathbin{\cdot}\Delta x
    +\frac{1}{2}\Delta x^{\mathsf T}M\Delta x.
    $$

    Parameters
    ----------
    xt : jnp.ndarray, shape (b, t, d)
    pt : jnp.ndarray, shape (b, t, d)
    mt : jnp.ndarray, shape (b, t, d, d), complex
    sensors : jnp.ndarray, shape (*S, d)
    domain_size : jnp.ndarray, shape (d,)
    periodic : jnp.ndarray, shape (d,), bool

    Returns
    -------
    jnp.ndarray, shape (b, t, *S), complex
        Phase values (real part used in oscillatory term).
    """
    diff = sensors[None, None, ...] - jnp.expand_dims(
        xt, axis=tuple(range(2, 2 + sensors.ndim - 1))
    )
    diff = jnp.where(periodic, diff - domain_size * jnp.round(diff / domain_size), diff)

    phase = jnp.einsum(
        "btd,bt...d->bt...",
        pt,
        diff,
        precision=jax.lax.Precision.HIGHEST,
    ) + 0.5 * jnp.einsum(
        "btij,bt...i,bt...j->bt...",
        mt,
        diff,
        diff,
        precision=jax.lax.Precision.HIGHEST,
    )

    return phase


def compute_diff(
    xt: Float[Array, "b t d"],
    sensors: Float[Array, "*S d"],
    domain_size: Float[Array, " d"],
    periodic: Bool[Array, " d"],
) -> Float[Array, "b t *S d"]:
    r"""
    Sensor–ray displacement with periodic wrap.

    Parameters
    ----------
    xt : jnp.ndarray, shape (b, t, d)
    sensors : jnp.ndarray, shape (*S, d)
    domain_size : jnp.ndarray, shape (d,)
    periodic : jnp.ndarray, shape (d,), bool

    Returns
    -------
    jnp.ndarray, shape (b, t, *S, d)
        The displacement $\Delta x=x_s-x_t$, broadcast over sensors and
        wrapped as needed.
    """
    diff = sensors[None, None, ...] - jnp.expand_dims(
        xt, axis=tuple(range(2, 2 + sensors.ndim - 1))
    )
    diff = jnp.where(periodic, diff - domain_size * jnp.round(diff / domain_size), diff)

    return diff


def compute_gaussian_beam(
    x0: Float[Array, "b d"],
    p0: Float[Array, "b d"],
    M0: Complex[Array, "b d d"],
    a0: Complex[Array, " b"],
    omega0: Float[Array, " b"],
    mode: Num[Array, " b"],
    c: Callable[[Float[Array, "... d"]], Float[Array, "..."]],
    lam: float,
    ts: Float[Array, " Nt"],
    sensors: Float[Array, "*S d"],
    domain_size: Float[Array, " d"],
    periodic: Bool[Array, " d"],
    ode_solver: SolverFn,
    solver_config: Optional[SolverConfig] = None,
) -> Complex[Array, "Nt *S b"]:
    r"""
    Complex GB field at sensors, keeping beam axis.

    Parameters
    ----------
    x0 : jnp.ndarray, shape (b, d)
    p0 : jnp.ndarray, shape (b, d)
    M0 : jnp.ndarray, shape (b, d, d), complex
    a0 : jnp.ndarray, shape (b,), complex
    omega0 : jnp.ndarray, shape (b,)
        Angular frequencies scaled by $\lVert p\rVert$.
    mode : jnp.ndarray, shape (b,)
        Branch value $\pm1$ per beam.
    c : Callable[[jnp.ndarray], jnp.ndarray]
        Speed of sound.
    lam : float
        Absorption parameter (affects amplitude ODE if enabled).
    ts : jnp.ndarray, shape (Nt,)
    sensors : jnp.ndarray, shape (*S, d)
    domain_size : jnp.ndarray, shape (d,)
    periodic : jnp.ndarray, shape (d,), bool
    ode_solver : SolverFn
        Integrator returning (xt, pt, Mt, At).
    solver_config : Optional[SolverConfig]

    Returns
    -------
    jnp.ndarray, shape (Nt, *S, b), complex
        Field contributions per beam (not summed).

    Notes
    -----
    - Calls `ode_solver` once; wraps positions; phases from `xt, pt, Mt`.
    - Overall factor $A_t\exp(i\omega_0\Phi)$.
    """
    xt, pt, Mt, At = ode_solver(x0, p0, M0, a0, mode, ts, c, lam, solver_config)

    xt = wrap_position(xt, domain_size, periodic)

    phase = compute_phase(xt, pt, Mt, sensors, domain_size, periodic)

    gb = jnp.einsum(
        "bt1,bt...->t...b",
        At,
        jnp.exp(jnp.einsum("b,bt...->bt...", 1j * omega0, phase)),
    )

    return gb


def safe_angle_eps(z, eps=1e-12):
    r"""
    Phase angle with zero-safe branch for `(0+0j)`.

    Parameters
    ----------
    z : jnp.ndarray
        Complex array.
    eps : float
        Substitute real part when both real/imag are exactly zero.

    Returns
    -------
    jnp.ndarray
        $\operatorname{atan2}\!\left(\operatorname{Im}z,
        \operatorname{Re}z_{\mathrm{safe}}\right)$.
    """
    re = jnp.real(z)
    im = jnp.imag(z)
    # Map 0 + 0j to angle zero with finite derivatives.
    re = jnp.where((re == 0) & (im == 0), eps, re)
    return jnp.arctan2(im, re)


def _sum_gaussian_beam_real_components_xla(
    xt,
    pt,
    matrix_real,
    matrix_imag,
    amplitude,
    angle,
    omega0,
    sensors,
    domain_size,
    periodic,
    *,
    precision=jax.lax.Precision.HIGHEST,
    rounding_mode: str = "nearest_even",
    periodic_axes: tuple[bool, ...] | None = None,
    initial_field=None,
):
    """Sum preprocessed real beam components with XLA."""
    if rounding_mode not in {"nearest_even", "half_open"}:
        raise ValueError(f"Unsupported rounding mode {rounding_mode!r}.")

    num_times = xt.shape[1]
    sensors_flat = sensors.reshape((-1, sensors.shape[-1]))
    num_sensors = sensors_flat.shape[0]
    dom = domain_size.astype(xt.dtype)
    pmask = periodic.astype(xt.dtype)

    def minimum_image(delta):
        def image_index(value):
            if rounding_mode == "half_open":
                return jnp.floor(value + 0.5)
            return jnp.round(value)

        if periodic_axes is None:
            return delta - dom * image_index(delta / dom) * pmask
        return jnp.stack(
            [
                (
                    delta[..., axis]
                    - dom[axis] * image_index(delta[..., axis] / dom[axis])
                    if periodic_axes[axis]
                    else delta[..., axis]
                )
                for axis in range(delta.shape[-1])
            ],
            axis=-1,
        )

    def add_beam(accumulator, beam):
        xi, pi, mr, mi, amp, phase_offset, frequency = beam
        delta = minimum_image(sensors_flat[None, :, :] - xi[:, None, :])
        phase_linear = jnp.einsum("tsd,td->ts", delta, pi, precision=precision)
        phase_quadratic = 0.5 * jnp.einsum(
            "tsd,tde,tse->ts",
            delta,
            mr,
            delta,
            precision=precision,
        )
        decay_quadratic = 0.5 * jnp.einsum(
            "tsd,tde,tse->ts",
            delta,
            mi,
            delta,
            precision=precision,
        )
        contribution = (
            amp[:, None]
            * jnp.cos(
                frequency * (phase_linear + phase_quadratic) + phase_offset[:, None]
            )
            * jnp.exp(-frequency * decay_quadratic)
        )
        return accumulator + contribution, None

    initial = (
        jnp.zeros((num_times, num_sensors), dtype=xt.dtype)
        if initial_field is None
        else initial_field
    )
    result, _ = jax.lax.scan(
        add_beam,
        initial,
        (xt, pt, matrix_real, matrix_imag, amplitude, angle, omega0),
    )
    return result


def sum_gaussian_beam_real_trajectories_xla(
    xt: Float[Array, "b Nt d"],
    pt: Float[Array, "b Nt d"],
    Mt: Complex[Array, "b Nt d d"],
    At: Complex[Array, "b Nt 1"],
    omega0: Float[Array, " b"],
    sensors: Float[Array, "*S d"],
    domain_size: Float[Array, " d"],
    periodic: Bool[Array, " d"],
    *,
    precision=jax.lax.Precision.HIGHEST,
) -> Float[Array, "Nt *S"]:
    """Sum trajectory fields with output-sized XLA accumulation.

    A singleton time axis provides terminal-only evaluation.
    """
    num_times = xt.shape[1]
    amplitude = At[..., 0]
    result = _sum_gaussian_beam_real_components_xla(
        xt,
        pt,
        jnp.real(Mt),
        jnp.imag(Mt),
        2.0 * jnp.abs(amplitude),
        safe_angle_eps(amplitude),
        omega0,
        sensors,
        domain_size,
        periodic,
        precision=precision,
    )
    return result.reshape((num_times,) + sensors.shape[:-1])


def compute_gaussian_beam_real(
    x0: Float[Array, "b d"],
    p0: Float[Array, "b d"],
    M0: Complex[Array, "b d d"],
    a0: Complex[Array, " b"],
    omega0: Float[Array, " b"],
    mode: Num[Array, " b"],
    c: Callable[[Float[Array, "... d"]], Float[Array, "..."]],
    lam: float,
    ts: Float[Array, " Nt"],
    sensors: Float[Array, "*S d"],
    domain_size: Float[Array, " d"],
    periodic: Bool[Array, " d"],
    ode_solver: SolverFn,
    solver_config: Optional[SolverConfig] = None,
) -> Float[Array, "Nt *S"]:
    r"""
    Real-valued streaming GB: scan over beams, vmap over time.

    Parameters
    ----------
    x0 : jnp.ndarray, shape (b, d)
        Initial beam positions.
    p0 : jnp.ndarray, shape (b, d)
        Initial momenta.
    M0 : jnp.ndarray, shape (b, d, d)
        Initial complex Hessian matrices.
    a0 : jnp.ndarray, shape (b,)
        Initial amplitudes.
    omega0 : jnp.ndarray, shape (b,)
        Beam angular frequencies.
    mode : jnp.ndarray, shape (b,)
        Hamiltonian branch signs.
    c : Callable
        Sound-speed function.
    lam : float
        Absorption parameter passed to ``ode_solver``.
    ts : jnp.ndarray, shape (Nt,)
        Time grid.
    sensors : jnp.ndarray, shape (*S, d)
        Sensor positions.
    domain_size : jnp.ndarray, shape (d,)
        Physical domain size.
    periodic : jnp.ndarray, shape (d,)
        Per-axis periodicity flags.
    ode_solver : SolverFn
        ODE integrator returning ``(xt, pt, Mt, At)``.
    solver_config : SolverConfig, optional
        Optional numerical configuration for ``ode_solver``.

    Returns
    -------
    jnp.ndarray, shape (Nt, *S)
        Real-valued summed field at the sensor positions.

    Notes
    -----
    Uses $\mathcal{O}(N_tS)$ memory instead of materializing the beam axis
    with $\mathcal{O}(bN_tS)$ storage.
    """
    xt, pt, Mt, At = ode_solver(x0, p0, M0, a0, mode, ts, c, lam, solver_config)
    if xt.shape[1] != ts.shape[0]:
        # Prevent terminal-only states from broadcasting across the time grid.
        raise ValueError(
            f"ode_solver saved {xt.shape[1]} time steps but {ts.shape[0]} "
            "were requested; forward evaluation needs the full time grid."
        )
    xt = wrap_position(xt, domain_size, periodic)
    return sum_gaussian_beam_real_trajectories_xla(
        xt,
        pt,
        Mt,
        At,
        omega0,
        sensors,
        domain_size,
        periodic,
    )


def compute_gaussian_beam_real_pallas(
    x0: Float[Array, "b d"],
    p0: Float[Array, "b d"],
    M0: Complex[Array, "b d d"],
    a0: Complex[Array, " b"],
    omega0: Float[Array, " b"],
    mode: Num[Array, " b"],
    c: Callable[[Float[Array, "... d"]], Float[Array, "..."]],
    lam: float,
    ts: Float[Array, " Nt"],
    sensors: Float[Array, "*S d"],
    domain_size: Float[Array, " d"],
    periodic: Bool[Array, " d"],
    ode_solver: SolverFn,
    solver_config: Optional[SolverConfig] = None,
    pallas_config: PallasConfig | None = None,
    initial_field: jax.Array | None = None,
    return_padded: bool = False,
) -> Float[Array, "Nt *S"]:
    """Evaluate a real GB field with XLA trajectories and Pallas reduction."""
    from beamax.gb.pallas_kernels import sum_gaussian_beam_real_pallas

    xt, pt, Mt, At = ode_solver(x0, p0, M0, a0, mode, ts, c, lam, solver_config)
    xt = wrap_position(xt, domain_size, periodic)

    d = sensors.shape[-1]
    sensors_flat = sensors.reshape((-1, d))
    result = sum_gaussian_beam_real_pallas(
        xt=xt,
        pt=pt,
        Mt=Mt,
        At=At,
        omega0=omega0,
        sensors=sensors_flat,
        domain_size=domain_size,
        periodic=periodic,
        config=pallas_config,
        initial_field=initial_field,
        return_padded=return_padded,
    )
    if return_padded:
        return result
    return result.reshape((ts.shape[0],) + sensors.shape[:-1])


def compute_gaussian_beam_real_TR(
    x0: Float[Array, "b d"],
    p0: Float[Array, "b d"],
    M0: Complex[Array, "b d d"],
    a0: Complex[Array, " b"],
    omega0: Float[Array, " b"],
    mode: Num[Array, " b"],
    c: Callable[[Float[Array, "... d"]], Float[Array, "..."]],
    lam: float,
    ts: Float[Array, "b Nt"],
    sensors: Float[Array, "*S d"],
    domain_size: Float[Array, " d"],
    periodic: Bool[Array, " d"],
    ode_solver: SolverFn,
    solver_config: Optional[SolverConfig] = None,
) -> Float[Array, "Nt *S b"]:
    r"""
    Compute a collection of Gaussian Beams in n-dimensions, assuming the resulting field is real.

    $$
    u+\overline{u}
    =2|A|\cos\!\left(
      \omega\left[p\mathbin{\cdot}\Delta x
      +\frac{1}{2}\Delta x^{\mathsf T}M_R\Delta x\right]
      +\arg A
    \right)
    \exp\!\left(
      -\frac{\omega}{2}\Delta x^{\mathsf T}M_I\Delta x
    \right).
    $$

    This should require 2 times less memory than the complex version, and should be faster.

    Parameters
    ----------
    x0 : jnp.ndarray, shape (b, d)
        Initial beam positions.
    p0 : jnp.ndarray, shape (b, d)
        Initial momenta.
    M0 : jnp.ndarray, shape (b, d, d)
        Initial complex Hessian matrices.
    a0 : jnp.ndarray, shape (b,)
        Initial amplitudes.
    omega0 : jnp.ndarray, shape (b,)
        Beam angular frequencies.
    mode : jnp.ndarray, shape (b,)
        Hamiltonian branch signs.
    c : Callable
        Sound-speed function.
    lam : float
        Absorption parameter passed to ``ode_solver``.
    ts : jnp.ndarray, shape (Nt,)
        Time grid.
    sensors : jnp.ndarray, shape (*S, d)
        Sensor positions.
    domain_size : jnp.ndarray, shape (d,)
        Physical domain size.
    periodic : jnp.ndarray, shape (d,)
        Per-axis periodicity flags.
    ode_solver : SolverFn
        ODE integrator returning ``(xt, pt, Mt, At)``.
    solver_config : SolverConfig, optional
        Optional numerical configuration for ``ode_solver``.

    Returns
    -------
    jnp.ndarray, shape (Nt, *S, b)
        Real-valued beam field with the beam axis retained.
    """
    xt, pt, Mt, At = ode_solver(x0, p0, M0, a0, mode, ts, c, lam, solver_config)

    xt = wrap_position(xt, domain_size, periodic)
    diff = compute_diff(xt, sensors, domain_size, periodic)

    xp_term = jnp.einsum(
        "btd,bt...d->bt...",
        pt,
        diff,
        precision=jax.lax.Precision.HIGHEST,
    )

    Mr = jnp.real(Mt)
    Mi = jnp.imag(Mt)

    temp = jnp.einsum(
        "btij,bt...i->btj...",
        Mr,
        diff,
        precision=jax.lax.Precision.HIGHEST,
    )
    xMrx_term = 0.5 * jnp.einsum(
        "btj...,bt...j->bt...",
        temp,
        diff,
        precision=jax.lax.Precision.HIGHEST,
    )
    temp = jnp.einsum(
        "btij,bt...i->btj...",
        Mi,
        diff,
        precision=jax.lax.Precision.HIGHEST,
    )
    xMix_term = 0.5 * jnp.einsum(
        "btj...,bt...j->bt...",
        temp,
        diff,
        precision=jax.lax.Precision.HIGHEST,
    )

    real_phase = xp_term + xMrx_term

    amplitude = jnp.abs(At)

    real_ω = jnp.einsum(
        "b,bt...->bt...",
        omega0,
        real_phase,
        precision=jax.lax.Precision.HIGHEST,
    )
    num_sensor_dims = real_phase.ndim - 3
    angle = safe_angle_eps(At).reshape(At.shape + (1,) * num_sensor_dims)
    phase_angle = real_ω + angle

    damping = jnp.exp(
        -jnp.einsum(
            "b,bt...->bt...",
            omega0,
            xMix_term,
            precision=jax.lax.Precision.HIGHEST,
        )
    )

    gb_real = jnp.einsum(
        "bt1,bt...,bt...->t...b", 2 * amplitude, jnp.cos(phase_angle), damping
    )

    return gb_real


def compute_gaussian_beam_real_TR_xla_terminal(
    x0: Float[Array, "b d"],
    p0: Float[Array, "b d"],
    M0: Complex[Array, "b d d"],
    a0: Complex[Array, " b"],
    omega0: Float[Array, " b"],
    mode: Num[Array, " b"],
    c: Callable[[Float[Array, "... d"]], Float[Array, "..."]],
    lam: float,
    ts: Float[Array, "b Nt"],
    sensors: Float[Array, "*S d"],
    domain_size: Float[Array, " d"],
    periodic: Bool[Array, " d"],
    ode_solver: SolverFn,
    solver_config: Optional[SolverConfig] = None,
) -> Float[Array, "*S"]:
    """Evaluate the terminal TR field with output-sized XLA accumulation."""
    xt, pt, Mt, At = ode_solver(x0, p0, M0, a0, mode, ts, c, lam, solver_config)
    xt = wrap_position(xt, domain_size, periodic)
    result = sum_gaussian_beam_real_trajectories_xla(
        xt[:, -1:, :],
        pt[:, -1:, :],
        Mt[:, -1:, :, :],
        At[:, -1:, :],
        omega0,
        sensors,
        domain_size,
        periodic,
    )
    return result[0]


def compute_gaussian_beam_real_TR_pallas_terminal(
    x0: Float[Array, "b d"],
    p0: Float[Array, "b d"],
    M0: Complex[Array, "b d d"],
    a0: Complex[Array, " b"],
    omega0: Float[Array, " b"],
    mode: Num[Array, " b"],
    c: Callable[[Float[Array, "... d"]], Float[Array, "..."]],
    lam: float,
    ts: Float[Array, "b Nt"],
    sensors: Float[Array, "*S d"],
    domain_size: Float[Array, " d"],
    periodic: Bool[Array, " d"],
    ode_solver: SolverFn,
    solver_config: Optional[SolverConfig] = None,
    pallas_config: PallasConfig | None = None,
    initial_field: jax.Array | None = None,
    return_padded: bool = False,
) -> Float[Array, "*S"]:
    """Evaluate the terminal TR field with Pallas beam reduction."""
    from beamax.gb.pallas_kernels import sum_gaussian_beam_real_pallas

    xt, pt, Mt, At = ode_solver(x0, p0, M0, a0, mode, ts, c, lam, solver_config)
    xt = wrap_position(xt, domain_size, periodic)

    d = sensors.shape[-1]
    sensors_flat = sensors.reshape((-1, d))
    result = sum_gaussian_beam_real_pallas(
        xt=xt[:, -1:, :],
        pt=pt[:, -1:, :],
        Mt=Mt[:, -1:, :, :],
        At=At[:, -1:, :],
        omega0=omega0,
        sensors=sensors_flat,
        domain_size=domain_size,
        periodic=periodic,
        config=pallas_config,
        initial_field=(
            None if initial_field is None else jnp.reshape(initial_field, (1, -1))
        ),
        return_padded=return_padded,
    )
    if return_padded:
        return result[0]
    return result[0].reshape(sensors.shape[:-1])
