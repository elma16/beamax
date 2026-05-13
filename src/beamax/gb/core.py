import jax
import jax.numpy as jnp
import warnings
from typing import Callable, Optional
from beamax.gb.gb_solvers import SolverFn, SolverConfig

warnings.filterwarnings("ignore", module="equinox")

__all__ = [
    "compute_gaussian_beam",
    "compute_gaussian_beam_real",
    "compute_gaussian_beam_real_TR",
]


def wrap_position(
    position: jnp.ndarray, domain_size: jnp.ndarray, periodic: jnp.ndarray
) -> jnp.ndarray:
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
    xt: jnp.ndarray,
    pt: jnp.ndarray,
    mt: jnp.ndarray,
    sensors: jnp.ndarray,
    domain_size: jnp.ndarray,
    periodic: jnp.ndarray,
) -> jnp.ndarray:
    """
    GB phase at sensors: `(p · Δx) + 0.5 Δxᵀ M Δx`.

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

    phase = jnp.einsum("btd,bt...d->bt...", pt, diff) + 0.5 * jnp.einsum(
        "btij,bt...i,bt...j->bt...", mt, diff, diff
    )

    return phase


def compute_diff(
    xt: jnp.ndarray,
    sensors: jnp.ndarray,
    domain_size: jnp.ndarray,
    periodic: jnp.ndarray,
) -> jnp.ndarray:
    """
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
        Δx = sensors - xt (broadcast), wrapped as needed.
    """
    diff = sensors[None, None, ...] - jnp.expand_dims(
        xt, axis=tuple(range(2, 2 + sensors.ndim - 1))
    )
    diff = jnp.where(periodic, diff - domain_size * jnp.round(diff / domain_size), diff)

    return diff


def compute_gaussian_beam(
    x0: jnp.ndarray,
    p0: jnp.ndarray,
    M0: jnp.ndarray,
    a0: jnp.ndarray,
    omega0: jnp.ndarray,
    mode: jnp.ndarray,
    c: Callable,
    lam: float,
    ts: jnp.ndarray,
    sensors: jnp.ndarray,
    domain_size: jnp.ndarray,
    periodic: jnp.ndarray,
    ode_solver: SolverFn,
    solver_config: Optional[SolverConfig] = None,
) -> jnp.ndarray:
    """
    Complex GB field at sensors, keeping beam axis.

    Parameters
    ----------
    x0 : jnp.ndarray, shape (b, d)
    p0 : jnp.ndarray, shape (b, d)
    M0 : jnp.ndarray, shape (b, d, d), complex
    a0 : jnp.ndarray, shape (b,), complex
    omega0 : jnp.ndarray, shape (b,)
        Angular frequencies (|p| scaled).
    mode : jnp.ndarray, shape (b,)
        ±1 per beam.
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
    - Overall factor `At * exp(i omega0 * phase)`.
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
    """
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
        `atan2(Im(z), Re(z_safe))`.
    """
    re = jnp.real(z)
    im = jnp.imag(z)
    # Replace exactly-zero pair by (eps,0) so atan2 returns 0, finite derivatives
    re = jnp.where((re == 0) & (im == 0), eps, re)
    return jnp.arctan2(im, re)


def compute_gaussian_beam_real(
    x0,
    p0,
    M0,
    a0,
    omega0,
    mode,
    c,
    lam,
    ts,
    sensors,
    domain_size,
    periodic,
    ode_solver,
    solver_config=None,
):
    """
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
    Uses ``O(Nt * S)`` memory instead of materializing the beam axis with
    ``O(b * Nt * S)`` storage.
    """
    # Solve ODEs for all beams (this is fine, ODEs are small)
    xt, pt, Mt, At = ode_solver(x0, p0, M0, a0, mode, ts, c, lam, solver_config)
    xt = wrap_position(xt, domain_size, periodic)

    d = sensors.shape[-1]
    sensors_flat = sensors.reshape((-1, d))
    S = sensors_flat.shape[0]
    Nt = ts.shape[0]

    dom = domain_size.astype(xt.dtype)
    pmask = periodic.astype(xt.dtype)

    # Vectorize computation over time for a single beam
    def single_beam_all_times(xi, pi, Mi, Ai, wi):
        """
        Process one beam at all time points.

        Parameters
        ----------
        xi : jnp.ndarray, shape (Nt, d)
            Beam centre over time.
        pi : jnp.ndarray, shape (Nt, d)
            Beam momentum over time.
        Mi : jnp.ndarray, shape (Nt, d, d)
            Complex Hessian over time.
        Ai : jnp.ndarray, shape (Nt,)
            Complex amplitude over time.
        wi : jnp.ndarray
            Scalar beam frequency.

        Returns
        -------
        jnp.ndarray, shape (Nt, S)
            Real beam contribution at flattened sensor positions.
        """
        Mr = jnp.real(Mi)  # (Nt, d, d)
        Mi_im = jnp.imag(Mi)  # (Nt, d, d)

        # Broadcast: xi (Nt,1,d), sensors_flat (S,d) -> delta (Nt,S,d)
        delta = sensors_flat[None, :, :] - xi[:, None, :]
        delta = delta - dom * jnp.round(delta / dom) * pmask

        # Vectorized over (Nt, S)
        phi_lin = jnp.einsum("tsd,td->ts", delta, pi)
        phi_quad = 0.5 * jnp.einsum("tsd,tde,tse->ts", delta, Mr, delta)
        chi = 0.5 * jnp.einsum("tsd,tde,tse->ts", delta, Mi_im, delta)

        phase = wi * (phi_lin + phi_quad) + safe_angle_eps(Ai[:, None])

        return 2.0 * jnp.abs(Ai[:, None]) * jnp.cos(phase) * jnp.exp(-wi * chi)

    # Scan over beams to accumulate without materializing (b, Nt, S)
    def scan_fn(acc, beam_data):
        """
        Accumulate one beam contribution into the streaming field.

        Parameters
        ----------
        acc : jnp.ndarray, shape (Nt, S)
            Running field sum.
        beam_data : Tuple[jnp.ndarray, ...]
            Per-beam tuple ``(xi, pi, Mi, Ai, wi)``.

        Returns
        -------
        (jnp.ndarray, None)
            Updated accumulator and empty scan output.
        """
        xi, pi, Mi, Ai, wi = beam_data
        contrib = single_beam_all_times(xi, pi, Mi, Ai[:, 0], wi)
        return acc + contrib, None

    init = jnp.zeros((Nt, S), dtype=xt.dtype)
    result, _ = jax.lax.scan(scan_fn, init, (xt, pt, Mt, At, omega0))

    return result.reshape((Nt,) + sensors.shape[:-1])


def compute_gaussian_beam_real_TR(
    x0: jnp.ndarray,
    p0: jnp.ndarray,
    M0: jnp.ndarray,
    a0: jnp.ndarray,
    omega0: jnp.ndarray,
    mode: jnp.ndarray,
    c: Callable,
    lam: float,
    ts: jnp.ndarray,
    sensors: jnp.ndarray,
    domain_size: jnp.ndarray,
    periodic: jnp.ndarray,
    ode_solver: SolverFn,
    solver_config: Optional[SolverConfig] = None,
) -> jnp.ndarray:
    """
    Compute a collection of Gaussian Beams in n-dimensions, assuming the resulting field is real.

    u + bar(u) = 2|A|cos(ω(x p + 0.5 x Mr x) + angle(A))exp(-ω/2 x Mi x).

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

    # Pre-compute these terms
    xp_term = jnp.einsum("btd,bt...d->bt...", pt, diff)

    Mr = jnp.real(Mt)
    Mi = jnp.imag(Mt)

    temp = jnp.einsum("btij,bt...i->btj...", Mr, diff)
    xMrx_term = 0.5 * jnp.einsum("btj...,bt...j->bt...", temp, diff)
    temp = jnp.einsum("btij,bt...i->btj...", Mi, diff)
    xMix_term = 0.5 * jnp.einsum("btj...,bt...j->bt...", temp, diff)

    # Combine into real_phase
    real_phase = xp_term + xMrx_term

    # Calculate field
    amplitude = jnp.abs(At)

    real_ω = jnp.einsum("b,bt...->bt...", omega0, real_phase)
    num_sensor_dims = real_phase.ndim - 3
    angle = safe_angle_eps(At).reshape(At.shape + (1,) * num_sensor_dims)
    # angle = jnp.angle(At).reshape(At.shape + (1,) * num_sensor_dims)
    phase_angle = real_ω + angle

    damping = jnp.exp(-jnp.einsum("b,bt...->bt...", omega0, xMix_term))

    gb_real = jnp.einsum(
        "bt1,bt...,bt...->t...b", 2 * amplitude, jnp.cos(phase_angle), damping
    )

    return gb_real
