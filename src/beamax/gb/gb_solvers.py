import jax
import jax.numpy as jnp
from jax import vmap, grad, hessian

import diffrax
from functools import partial
import warnings
from einops import rearrange
from dataclasses import dataclass
from typing import Tuple, Callable, Protocol, Optional
import optimistix


warnings.filterwarnings("ignore", module="equinox")

__all__ = [
    "SolverFn",
    "solve_hom_diag",
    "solve_hom_general",
    "solve_ODE_base",
    "solve_ODE_batch_t",
    "solve_ODE_QP_base",
]


class SolverFn(Protocol):
    """
    Protocol for Gaussian beam ODE integrators.

    Implementations integrate beam initial data and return beam positions,
    momenta, Hessians, and amplitudes over time.

    Notes
    -----
    Expected signature is ``(x0, p0, M0, a0, mode, ts, c, *args, **kwargs)``
    returning ``(xt, pt, Mt, At)``. Standard shapes are:

    - ``x0, p0``: ``(b, d)``
    - ``M0``: ``(b, d, d)``
    - ``a0, mode``: ``(b,)``
    - ``ts``: ``(Nt,)``
    - ``xt, pt``: ``(b, Nt, d)``
    - ``Mt``: ``(b, Nt, d, d)``
    - ``At``: ``(b, Nt, 1)``
    """

    def __call__(
        self,
        x0: jnp.ndarray,
        p0: jnp.ndarray,
        M0: jnp.ndarray,
        a0: jnp.ndarray,
        mode: jnp.ndarray,
        ts: jnp.ndarray,
        c: Callable,
        *args,
        **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Integrate Gaussian beam ODE state over the requested times.

        Parameters
        ----------
        x0 : jnp.ndarray, shape (b, d)
            Initial positions.
        p0 : jnp.ndarray, shape (b, d)
            Initial momenta.
        M0 : jnp.ndarray, shape (b, d, d)
            Initial complex Hessian matrices.
        a0 : jnp.ndarray, shape (b,)
            Initial amplitudes.
        mode : jnp.ndarray, shape (b,)
            Hamiltonian branch signs.
        ts : jnp.ndarray, shape (Nt,)
            Time grid.
        c : Callable
            Sound-speed function.
        *args
            Additional integrator-specific positional arguments.
        **kwargs
            Additional integrator-specific keyword arguments.

        Returns
        -------
        xt : jnp.ndarray, shape (b, Nt, d)
            Beam positions over time.
        pt : jnp.ndarray, shape (b, Nt, d)
            Beam momenta over time.
        Mt : jnp.ndarray, shape (b, Nt, d, d)
            Beam Hessians over time.
        At : jnp.ndarray, shape (b, Nt, 1)
            Beam amplitudes over time.
        """
        ...


def create_p_perp(
    p0: jnp.ndarray, normp_sq: jnp.ndarray, eye: jnp.ndarray
) -> jnp.ndarray:
    """
    Projector perpendicular to `p0`.

    Parameters
    ----------
    p0 : jnp.ndarray, shape (..., d, 1)
    normp_sq : jnp.ndarray, shape (..., 1, 1)
    eye : jnp.ndarray, shape (..., d, d)

    Returns
    -------
    jnp.ndarray, shape (..., d, d)
        I - p pᵀ / ||p||².
    """
    return eye - jnp.matmul(p0, jnp.swapaxes(p0, -1, -2)) / normp_sq


def compute_amp_hom_gen(
    p0: jnp.ndarray,
    m0: jnp.ndarray,
    c0: jnp.ndarray,
    ts: jnp.ndarray,
    a0: jnp.ndarray,
) -> jnp.ndarray:
    """
    Amplitude for general homogeneous GB (no diagonal assumption).

    Parameters
    ----------
    p0 : (b, d)
    m0 : (b, d, d)
    c0 : (b, 1)
    ts : (Nt,)
    a0 : (b,)

    Returns
    -------
    jnp.ndarray, shape (b, Nt, 1)
        a(t) = a0 / sqrt(det(I + c0 t P_perp M0 / ||p||)).
    """
    d = p0.shape[-1]
    id = rearrange(jnp.eye(d), "i j -> 1 1 i j")
    p0 = rearrange(p0, "b d -> b 1 d 1")
    normp = jnp.linalg.norm(p0, axis=-2, keepdims=True)
    c0 = rearrange(c0, "b 1 -> b 1 1 1")
    m0 = rearrange(m0, "b d1 d2 -> b 1 d1 d2")
    ts = rearrange(ts, "nt -> 1 nt 1 1")

    p_perp = id - jnp.einsum("btia,btjb->btij", p0, p0) / normp**2

    interior = id + c0 * ts * jnp.einsum("btij,btjk->btik", p_perp, m0) / normp
    det = jnp.sqrt(jnp.linalg.det(interior))
    det = rearrange(det, "b t -> b t 1 1")
    a0 = rearrange(a0, "b -> b 1 1 1")
    at = a0 / det
    at = rearrange(at, "b t 1 1 -> b t 1")

    return at


def compute_m_hom_gen(
    p0: jnp.ndarray,
    m0: jnp.ndarray,
    c0: jnp.ndarray,
    ts: jnp.ndarray,
) -> jnp.ndarray:
    """
    M(t) for general homogeneous GB.

    Parameters
    ----------
    p0 : (b, d)
    m0 : (b, d, d)
    c0 : (b, 1)
    ts : (Nt,)

    Returns
    -------
    jnp.ndarray, shape (b, Nt, d, d)
        M(t) = M0 (I + c0 t P_perp M0 / ||p||)^(-1).
    """
    d = p0.shape[-1]
    id = rearrange(jnp.eye(d), "i j -> 1 1 i j")
    p0 = rearrange(p0, "b d -> b 1 d 1")
    normp = jnp.linalg.norm(p0, axis=-2, keepdims=True)
    c0 = rearrange(c0, "b 1 -> b 1 1 1")
    m0 = rearrange(m0, "b d1 d2 -> b 1 d1 d2")
    ts = rearrange(ts, "nt -> 1 nt 1 1")

    p_perp = id - jnp.einsum("btia,btjb->btij", p0, p0) / normp**2

    interior = id + c0 * ts * jnp.einsum("btij,btjk->btik", p_perp, m0) / normp
    return m0 @ jnp.linalg.inv(interior)


def compute_amp_hom_diag_2d(
    p0: jnp.ndarray,
    normp: jnp.ndarray,
    alpha0: jnp.ndarray,
    c0: jnp.ndarray,
    ts: jnp.ndarray,
    a0: jnp.ndarray,
) -> jnp.ndarray:
    """
    Amplitude for diagonal M0 in 2D anisotropy.

    Parameters
    ----------
    p0 : (b, 2)
    normp : (b, 1)
    alpha0 : (b, 2)   # Im-positive expected
    c0 : (b, 1)
    ts : (Nt,)
    a0 : (b,)

    Returns
    -------
    jnp.ndarray, shape (b, Nt)
        a(t) = a0 / (1 + c0 t * <p², alpha0[::-1]> / ||p||³)^{(d-1)/2}.
    """
    d = p0.shape[-1]

    αp_normp = (
        jnp.sum(jnp.square(p0) * alpha0[:, ::-1], axis=1, keepdims=True) / normp**3
    )

    return a0[..., None] / (1 + c0 * ts[None, :] * αp_normp) ** ((d - 1) / 2)


def compute_amp_hom_diag_3d(
    p0: jnp.ndarray,
    normp: jnp.ndarray,
    alpha0: jnp.ndarray,
    c0: jnp.ndarray,
    ts: jnp.ndarray,
    a0: jnp.ndarray,
) -> jnp.ndarray:
    """
    Amplitude for diagonal M0 in 3D anisotropy.

    Parameters
    ----------
    p0 : (b, 3)
    normp, alpha0, c0, ts, a0 : as above

    Returns
    -------
    jnp.ndarray, shape (b, Nt)
        Closed-form 3D expression combining axis terms (see source).
    """
    p1, p2, p3 = p0[:, 0] ** 2, p0[:, 1] ** 2, p0[:, 2] ** 2
    a1 = alpha0[:, 0][:, jnp.newaxis]
    a2 = alpha0[:, 1][:, jnp.newaxis]
    a3 = alpha0[:, 2][:, jnp.newaxis]

    ts_scaled = c0 * ts / normp**4
    term1 = c0 * a2 * a3 * ts + (a2 + a3) * normp
    term2 = c0 * a1 * a3 * ts + (a1 + a3) * normp
    term3 = c0 * a1 * a2 * ts + (a1 + a2) * normp

    denominator = 1 + ts_scaled * (
        p1[:, jnp.newaxis] * term1
        + p2[:, jnp.newaxis] * term2
        + p3[:, jnp.newaxis] * term3
    )
    return a0[:, jnp.newaxis] / jnp.sqrt(denominator)


def compute_amp_hom_diag(
    p0: jnp.ndarray,
    normp: jnp.ndarray,
    alpha0: jnp.ndarray,
    c0: jnp.ndarray,
    ts: jnp.ndarray,
    a0: jnp.ndarray,
) -> jnp.ndarray:
    """
    Dispatch amplitude formula for diagonal M0 (2D/3D).

    Parameters
    ----------
    p0 : jnp.ndarray, shape (b, d)
        Initial momenta.
    normp : jnp.ndarray, shape (b, 1)
        Momentum norms.
    alpha0 : jnp.ndarray, shape (b, d)
        Diagonal entries of ``M0``.
    c0 : jnp.ndarray, shape (b, 1)
        Signed homogeneous sound speed.
    ts : jnp.ndarray, shape (Nt,)
        Time grid.
    a0 : jnp.ndarray, shape (b,)
        Initial amplitudes.

    Returns
    -------
    jnp.ndarray, shape (b, Nt)
    """
    d = p0.shape[-1]
    if d == 3:
        return compute_amp_hom_diag_3d(p0, normp, alpha0, c0, ts, a0)
    elif d < 3:
        return compute_amp_hom_diag_2d(p0, normp, alpha0, c0, ts, a0)
    else:
        raise ValueError("Currently only 1D, 2D and 3D are supported.")


def compute_m_hom_diag(
    p0: jnp.ndarray,
    normp: jnp.ndarray,
    alpha0: jnp.ndarray,
    c0: jnp.ndarray,
    ts: jnp.ndarray,
) -> jnp.ndarray:
    """
    M(t) with diagonal M0 via Sherman–Morrison.

    Parameters
    ----------
    p0 : (b, d)
    normp : (b, 1)
    alpha0 : (b, d)
    c0 : (b, 1)
    ts : (Nt,)

    Returns
    -------
    jnp.ndarray, shape (b, Nt, d, d)
        N0 (A + u vᵀ)^(-1) with diagonal A, rank-1 update from ray direction.
    """

    d = p0.shape[1]
    p0 = rearrange(p0, "b d -> b 1 d 1")
    c0 = rearrange(c0, "b 1 -> b 1 1 1")
    alpha0 = rearrange(alpha0, "b d -> b 1 d 1")
    normp = rearrange(normp, "b 1 -> b 1 1 1")
    ts = rearrange(ts, "nt -> 1 nt 1 1")
    eye = rearrange(jnp.eye(d), "i j -> 1 1 i j")

    N0 = alpha0 * eye
    A_diag = 1 + c0 * ts * alpha0 / normp
    A_inv_diag = 1 / A_diag
    A_inv = A_inv_diag * eye

    u = -c0 * ts * p0 / normp**3  # shape (b, nt, d, 1)
    vT = jnp.einsum("btij,btik->btjk", p0, N0)  # shape (b, nt, d, d)

    A_inv_u = jnp.matmul(A_inv, u)  # shape (b, nt, d, 1)
    vT_A_inv = jnp.matmul(vT, A_inv)  # shape (b, nt, 1, d)

    scalar = 1 + jnp.matmul(vT_A_inv, u)  # shape (b, nt, 1, 1)

    Yt_inv = A_inv - jnp.matmul(A_inv_u, vT_A_inv) / scalar  # shape (b, nt, d, d)

    return jnp.matmul(N0, Yt_inv)


def solve_hom_diag(
    x0: jnp.ndarray,
    p0: jnp.ndarray,
    M0: jnp.ndarray,
    a0: jnp.ndarray,
    mode: jnp.ndarray,
    ts: jnp.ndarray,
    c: Callable,
    lam=None,
    config=None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Solver for homogeneous media with simplified equations.

    Parameters
    ----------
    x0 : jnp.ndarray, shape (b, d)
        Initial beam positions.
    p0 : jnp.ndarray, shape (b, d)
        Initial momenta.
    M0 : jnp.ndarray, shape (b, d, d)
        Initial Hessian matrices. Only diagonal entries are used.
    a0 : jnp.ndarray, shape (b,)
        Initial amplitudes.
    mode : jnp.ndarray, shape (b,)
        Hamiltonian branch signs.
    ts : jnp.ndarray, shape (Nt,)
        Time grid.
    c : Callable
        Homogeneous sound-speed function.
    lam : Any, optional
        Ignored compatibility argument.
    config : Any, optional
        Ignored compatibility argument.

    Returns
    -------
    xt : jnp.ndarray, shape (b, Nt, d)
        Beam positions over time.
    pt : jnp.ndarray, shape (b, Nt, d)
        Beam momenta over time.
    Mt : jnp.ndarray, shape (b, Nt, d, d)
        Beam Hessians over time.
    At : jnp.ndarray, shape (b, Nt, 1)
        Beam amplitudes over time.

    Notes
    -----
    Assumes ``c(x)`` is homogeneous, ``M0`` is diagonal, and ``d`` is 1, 2,
    or 3. The diagonal and dimensionality assumptions are relaxed by
    :func:`solve_hom_general`.
    """
    d = p0.shape[-1]
    nt = ts.shape[0]

    c0 = c(jnp.zeros((d,))) * mode[..., None]
    normp = jnp.linalg.norm(p0, axis=-1, keepdims=True)

    xt = x0[:, None, :] + c0[:, None, :] * (p0 / normp)[:, None, :] * ts[None, :, None]
    pt = p0[:, None, :].repeat(nt, axis=1)

    alpha0 = jnp.diagonal(M0, axis1=1, axis2=2)

    Mt = compute_m_hom_diag(p0, normp, alpha0, c0, ts)
    At = compute_amp_hom_diag(p0, normp, alpha0, c0, ts, a0)[:, :, None]

    return xt, pt, Mt, At


def solve_hom_general(
    x0: jnp.ndarray,
    p0: jnp.ndarray,
    m0: jnp.ndarray,
    a0: jnp.ndarray,
    mode: jnp.ndarray,
    ts: jnp.ndarray,
    c: Callable,
    lam=None,
    config=None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Solver for homogeneous media with simplified equations.

    Parameters
    ----------
    x0 : jnp.ndarray, shape (b, d)
        Initial beam positions.
    p0 : jnp.ndarray, shape (b, d)
        Initial momenta.
    m0 : jnp.ndarray, shape (b, d, d)
        Initial Hessian matrices.
    a0 : jnp.ndarray, shape (b,)
        Initial amplitudes.
    mode : jnp.ndarray, shape (b,)
        Hamiltonian branch signs.
    ts : jnp.ndarray, shape (Nt,)
        Time grid.
    c : Callable
        Homogeneous sound-speed function.
    lam : Any, optional
        Ignored compatibility argument.
    config : Any, optional
        Ignored compatibility argument.

    Returns
    -------
    xt : jnp.ndarray, shape (b, Nt, d)
        Beam positions over time.
    pt : jnp.ndarray, shape (b, Nt, d)
        Beam momenta over time.
    Mt : jnp.ndarray, shape (b, Nt, d, d)
        Beam Hessians over time.
    At : jnp.ndarray, shape (b, Nt, 1)
        Beam amplitudes over time.
    """
    d = p0.shape[-1]
    c0 = c(jnp.zeros((d,))) * mode[..., None]
    normp = jnp.linalg.norm(p0, axis=-1, keepdims=True)

    xt = x0[:, None, :] + c0[:, None, :] * (p0 / normp)[:, None, :] * ts[None, :, None]
    pt = p0[:, None, :].repeat(ts.shape[0], axis=1)

    Mt = compute_m_hom_gen(p0, m0, c0, ts)
    At = compute_amp_hom_gen(p0, m0, c0, ts, a0)

    return xt, pt, Mt, At


def solve_hom_TR(
    xT: jnp.ndarray,
    pT: jnp.ndarray,
    mT: jnp.ndarray,
    aT: jnp.ndarray,
    mode: jnp.ndarray,
    ts: jnp.ndarray,
    c: Callable,
    lam=None,
    config=None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Time-reversal solver for homogeneous media.

    Parameters
    ----------
    xT : jnp.ndarray, shape (b, d)
        Beam positions at the reference final time.
    pT : jnp.ndarray, shape (b, d)
        Beam momenta at the reference final time.
    mT : jnp.ndarray, shape (b, d, d)
        Beam Hessians at the reference final time.
    aT : jnp.ndarray, shape (b,) or (b, 1)
        Beam amplitudes at the reference final time.
    mode : jnp.ndarray, shape (b,) or (b, 1)
        Hamiltonian branch signs.
    ts : jnp.ndarray
        Per-beam or shared time grid.
    c : Callable
        Homogeneous sound-speed function.
    lam : Any, optional
        Ignored compatibility argument.
    config : Any, optional
        Ignored compatibility argument.

    Returns
    -------
    x0 : jnp.ndarray, shape (b, Nt, d)
        Time-reversed beam positions.
    p0_time : jnp.ndarray, shape (b, Nt, d)
        Time-reversed beam momenta.
    m0 : jnp.ndarray, shape (b, Nt, d, d)
        Time-reversed Hessians.
    a0 : jnp.ndarray, shape (b, Nt, 1)
        Time-reversed amplitudes.
    """

    d = pT.shape[-1]
    b = pT.shape[0]

    # Normalise input shapes to be beam-aligned.
    mode = jnp.asarray(mode)
    if mode.ndim > 1:
        mode = mode.reshape((mode.shape[0],))

    aT = jnp.asarray(aT)
    if aT.ndim == 1:
        aT = aT[:, None]
    elif aT.ndim > 2:
        aT = aT.reshape((aT.shape[0], -1))

    ts_arr = jnp.asarray(ts)
    if ts_arr.ndim == 0:
        ts_beams = jnp.broadcast_to(ts_arr[None], (b, 1))
    elif ts_arr.ndim == 1:
        # Ambiguous between per-beam scalars vs common timeline; treat length==b as per-beam.
        if ts_arr.shape[0] == b and ts_arr.size == b:
            ts_beams = ts_arr[:, None]
        else:
            ts_beams = jnp.broadcast_to(ts_arr[None, :], (b, ts_arr.shape[0]))
    else:  # ndim >= 2
        if ts_arr.shape[0] == b:
            ts_beams = ts_arr
        elif ts_arr.ndim == 2 and ts_arr.shape[1] == b:
            ts_beams = jnp.swapaxes(ts_arr, 0, 1)
        else:
            flat = ts_arr.reshape(-1)
            ts_beams = jnp.broadcast_to(flat[None, :], (b, flat.shape[0]))

    id = rearrange(jnp.eye(d), "i j -> 1 1 i j")
    c0 = c(jnp.zeros((d,))) * mode[:, None]  # (b, 1)
    p0 = pT

    normp = jnp.linalg.norm(p0, axis=-1, keepdims=True)  # (b, 1)

    # Time offsets relative to the reference point (ts_beams[:, 0])
    dt = ts_beams - ts_beams[:, :1]  # (b, nt)
    dirn = (p0 / normp)[:, None, :]  # (b, 1, d)

    # Positions along the ray: x(t) = xT + c * p̂ * (t - t_ref)
    x0 = xT[:, None, :] + c0[:, None, :] * dirn * dt[:, :, None]  # (b, nt, d)
    p0_time = jnp.broadcast_to(p0[:, None, :], x0.shape)  # (b, nt, d)

    mT_b = rearrange(mT, "b i j -> b 1 i j")  # (b, 1, d, d)
    p_perp = id - (p0[:, None, :, None] * p0[:, None, None, :]) / rearrange(
        normp**2, "b 1 -> b 1 1 1"
    )  # (b, 1, d, d)

    c0_b = rearrange(c0, "b 1 -> b 1 1 1")
    normp_b = rearrange(normp, "b 1 -> b 1 1 1")
    dt_b = dt[:, :, None, None]  # (b, nt, 1, 1)

    interior = id + c0_b * dt_b * jnp.einsum("btij,btjk->btik", p_perp, mT_b) / normp_b
    interior_inv = jnp.linalg.inv(interior)
    m0 = jnp.einsum(
        "...ij,...jk->...ik", jnp.broadcast_to(mT_b, interior.shape), interior_inv
    )

    a0 = aT[:, None, :] / jnp.sqrt(jnp.linalg.det(interior))[..., None]  # (b, nt, 1)

    return x0, p0_time, m0, a0


@dataclass
class SolverConfig:
    """
    Configuration for ODE solver settings.
    """

    solver: diffrax.AbstractSolver = diffrax.Tsit5()
    max_steps: int = 4096
    rtol: float = 1e-7
    atol: float = 1e-9
    pcoeff: float = 0.0
    icoeff: float = 1.0
    dcoeff: float = 0.0
    dt0: float | None = None

    @classmethod
    def from_precision(
        cls, use_x64: bool = None, solver: diffrax.AbstractSolver = None, **overrides
    ):
        """
        Create config with precision-appropriate tolerances.

        Parameters
        ----------
        use_x64 : bool | None
            If None, auto-detects from jax.config.x64_enabled
        solver : diffrax.AbstractSolver | None
            Override default solver (Tsit5)
        **overrides
            Override any other config fields (max_steps, rtol, etc.)

        Examples
        --------
        # Auto-detect precision, use defaults
        config = SolverConfig.from_precision()

        # Force float32 tolerances, but increase max_steps
        config = SolverConfig.from_precision(use_x64=False, max_steps=8192)

        # Auto precision, custom solver and tolerances
        config = SolverConfig.from_precision(
            solver=diffrax.Dopri5(),
            rtol=1e-5,
            max_steps=10000
        )

        # Everything custom
        config = SolverConfig.from_precision(
            use_x64=False,
            solver=diffrax.Dopri8(),
            max_steps=2048,
            rtol=1e-3,
            atol=1e-5,
            pcoeff=0.3
        )
        """

        if use_x64 is None:
            use_x64 = jax.config.x64_enabled

        # Precision-appropriate defaults
        if use_x64:
            defaults = {
                "rtol": 1e-7,
                "atol": 1e-9,
                "max_steps": 4096,
            }
        else:
            defaults = {
                "rtol": 1e-4,
                "atol": 1e-6,
                "max_steps": 4096,
            }

        # Apply user overrides (they take precedence)
        defaults.update(overrides)

        # Handle solver separately since it's not a simple type
        if solver is not None:
            defaults["solver"] = solver
        elif "solver" not in defaults:
            defaults["solver"] = diffrax.Tsit5()

        return cls(**defaults)


def ode_solver_setup(
    coupled_rhs: Callable,
    y0: jnp.ndarray,
    t0: float,
    t1: float,
    dt0: float,
    ts: jnp.ndarray,
    args: Tuple,
    config: Optional[SolverConfig] = None,
    cond_fn: Optional[Callable] = None,
    saveat: Optional[diffrax.SaveAt] = None,
):
    """
    Setup the ODE solver for the coupled system of ODEs for the GB motion.

    Parameters
    ----------
    coupled_rhs : Callable
        Right-hand-side function passed to :class:`diffrax.ODETerm`.
    y0 : jnp.ndarray
        Initial state vector.
    t0 : float
        Initial time.
    t1 : float
        Final time.
    dt0 : float
        Initial time step.
    ts : jnp.ndarray
        Save times.
    args : Tuple
        Extra ODE arguments.
    config : SolverConfig, optional
        Numerical solver configuration.
    cond_fn : Callable, optional
        Event condition function.
    saveat : diffrax.SaveAt, optional
        Custom save specification. Defaults to ``SaveAt(ts=ts)``.

    Returns
    -------
    diffrax.Solution
        Diffrax solution object.
    """
    if config is None:
        config = SolverConfig()

    if saveat is None:
        saveat = diffrax.SaveAt(ts=ts)

    stepsize_controller = diffrax.PIDController(
        rtol=config.rtol,
        atol=config.atol,
        pcoeff=config.pcoeff,
        icoeff=config.icoeff,
        dcoeff=config.dcoeff,
    )

    event = None
    if cond_fn is not None:
        event = diffrax.Event(
            cond_fn, optimistix.Newton(1e-9, 1e-9, optimistix.rms_norm)
        )

    solution = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(coupled_rhs),
        solver=config.solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=config.max_steps,
        event=event,
    )
    return solution


def riccati_rhs(M, x, p, mode, c):
    """
    Riccati equation for Hessian evolution Ṁ.

    Parameters
    ----------
    M : jnp.ndarray, shape (d, d)
    x : jnp.ndarray, shape (d,)
    p : jnp.ndarray, shape (d,)
    mode : scalar
    c : Callable

    Returns
    -------
    jnp.ndarray, shape (d, d)
        Ṁ = -(Gxx + Gxp M + M Gxpᵀ + M Gpp M).

        With (Gxp)_ij = ∂²G/(∂x_i ∂p_j), this is the standard textbook form
        (Berra–de Hoop–Romero 2017 eq 2.13; Červený 2007 eq 66).
    """
    d = x.shape[0]
    normp = jnp.linalg.norm(p)
    c_val = c(x)
    grad_c = grad(c)(x)
    hess_c = hessian(c)(x)

    Gpp = mode * c_val * (jnp.eye(d) / normp - jnp.outer(p, p) / normp**3)
    Gxp = mode * jnp.outer(grad_c, p) / normp
    Gxx = mode * hess_c * normp

    return -(Gxx + Gxp @ M + M @ Gxp.T + M @ Gpp @ M)


def coupled_rhs_absorption(t, y, args) -> jnp.ndarray:
    """
    Full GB ODE system with absorption `lam`.

    State layout
    ------------
    y = concat(x (d), p (d), vec(M) (d²), A (1))

    Parameters
    ----------
    t : float
    y : jnp.ndarray, shape (d+d+d²+1,)
    args : Tuple[mode, c, d, lam]

    Returns
    -------
    jnp.ndarray, same shape as `y`
    """
    mode, c, d, lam = args

    x = y[:d].real
    p = y[d : 2 * d].real
    M = rearrange(y[2 * d : 2 * d + d**2], "(d1 d2) -> d1 d2", d1=d, d2=d)
    A = y[2 * d + d**2 :]

    norm_p = jnp.linalg.norm(p, axis=-1)

    c_val = c(x)
    grad_c = grad(c)(x)

    G_val = mode * c_val * norm_p
    Gx_val = mode * grad_c * norm_p
    Gp_val = mode * c_val * p / norm_p

    dx = Gp_val
    dp = -Gx_val
    dM = riccati_rhs(M, x, p, mode, c)
    dA = (
        -A
        * (
            c(x) ** 2 * jnp.trace(M)
            - Gx_val @ Gp_val
            - Gp_val.T @ M @ Gp_val
            + lam * G_val
        )
        / (2 * G_val)
    )
    return jnp.concatenate([dx.ravel(), dp.ravel(), dM.ravel(), dA.ravel()])


def coupled_rhs(t, y, args) -> jnp.ndarray:
    """
    GB ODE system without absorption.

    Parameters
    ----------
    t : float
    y : jnp.ndarray, shape (d+d+d²+1,)
    args : Tuple[mode, c, d]

    Returns
    -------
    jnp.ndarray, same shape as `y`
    """
    mode, c, d = args

    x = y[:d].real
    p = y[d : 2 * d].real
    M = rearrange(y[2 * d : 2 * d + d**2], "(d1 d2) -> d1 d2", d1=d, d2=d)
    A = y[2 * d + d**2 :]

    norm_p = jnp.linalg.norm(p, axis=-1)

    c_val = c(x)
    grad_c = grad(c)(x)

    G_val = mode * c_val * norm_p
    Gx_val = mode * grad_c * norm_p
    Gp_val = mode * c_val * p / norm_p

    dx = Gp_val
    dp = -Gx_val
    dM = riccati_rhs(M, x, p, mode, c)
    dA = (
        -A
        * (c(x) ** 2 * jnp.trace(M) - Gx_val @ Gp_val - Gp_val.T @ M @ Gp_val)
        / (2 * G_val)
    )
    return jnp.concatenate([dx.ravel(), dp.ravel(), dM.ravel(), dA.ravel()])


def format_solution(ys, d):
    """
    Format the solution of the ODEs.

    Parameters
    ----------
    ys : jnp.ndarray, shape (Nt, d + d + d**2 + 1)
        Flat ODE state trajectory.
    d : int
        Spatial dimension.

    Returns
    -------
    xt : jnp.ndarray, shape (Nt, d)
        Beam positions.
    pt : jnp.ndarray, shape (Nt, d)
        Beam momenta.
    Mt : jnp.ndarray, shape (Nt, d, d)
        Beam Hessians.
    At : jnp.ndarray, shape (Nt, 1)
        Beam amplitudes.
    """
    xt = ys[..., :d].real
    pt = ys[..., d : 2 * d].real
    mt = rearrange(ys[..., 2 * d : 2 * d + d**2], "t (d1 d2) -> t d1 d2", d1=d, d2=d)
    at = ys[..., 2 * d + d**2 :]
    return xt, pt, mt, at


@partial(vmap, in_axes=(0, 0, 0, 0, 0, None, None, None, None))
def solve_ODE_base(
    x0: jnp.ndarray,
    p0: jnp.ndarray,
    M0: jnp.ndarray,
    a0: jnp.ndarray,
    mode: jnp.ndarray,
    ts: jnp.ndarray,
    c: Callable,
    lam: float = 0.0,
    solver_config: Optional[SolverConfig] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Solve the coupled system of ODEs for the GB with configurable solver settings.

    Parameters
    ----------
    x0 : jnp.ndarray, shape (d,)
        Initial beam position for one vmapped beam.
    p0 : jnp.ndarray, shape (d,)
        Initial momentum for one vmapped beam.
    M0 : jnp.ndarray, shape (d, d)
        Initial Hessian for one vmapped beam.
    a0 : jnp.ndarray, shape (1,) or scalar
        Initial amplitude.
    mode : jnp.ndarray
        Hamiltonian branch sign.
    ts : jnp.ndarray, shape (Nt,)
        Time grid.
    c : Callable
        Sound-speed function.
    lam : float, default=0.0
        Absorption coefficient.
    solver_config : SolverConfig, optional
        Numerical solver configuration.

    Returns
    -------
    xt : jnp.ndarray, shape (Nt, d)
        Beam positions.
    pt : jnp.ndarray, shape (Nt, d)
        Beam momenta.
    Mt : jnp.ndarray, shape (Nt, d, d)
        Beam Hessians.
    At : jnp.ndarray, shape (Nt, 1)
        Beam amplitudes.
    """
    t0 = ts[0]
    t1 = ts[-1]

    if solver_config is not None and solver_config.dt0 is not None:
        dt0 = solver_config.dt0
    else:
        dt0 = ts[1] - ts[0]

    d = x0.shape[-1]
    y0 = jnp.concatenate([x0.ravel(), p0.ravel(), M0.ravel(), a0.ravel()])
    args_ode = (mode, c, d, lam)

    solution = ode_solver_setup(
        coupled_rhs_absorption,
        y0,
        t0,
        t1,
        dt0,
        ts,
        args_ode,
        solver_config,
        cond_fn=None,
        saveat=None,
    )

    xt, pt, Mt, At = format_solution(solution.ys, d)

    return xt, pt, Mt, At


def solve_ODE_batch_t(
    x0: jnp.ndarray,
    p0: jnp.ndarray,
    M0: jnp.ndarray,
    A0: jnp.ndarray,
    mode: jnp.ndarray,
    ts: jnp.ndarray,
    c: Callable,
    lam: float = None,
    solver_config: Optional[SolverConfig] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Solve the coupled system of ODEs for the GB motion with per-batch time points.

    Parameters
    ----------
    x0 : jnp.ndarray, shape (b, d)
        Initial beam positions.
    p0 : jnp.ndarray, shape (b, d)
        Initial momenta.
    M0 : jnp.ndarray, shape (b, d, d)
        Initial Hessian matrices.
    A0 : jnp.ndarray, shape (b,)
        Initial amplitudes.
    mode : jnp.ndarray, shape (b,)
        Hamiltonian branch signs.
    ts : jnp.ndarray, shape (b, Nt)
        Per-beam time grids, commonly ``(t0, t1)`` intervals.
    c : Callable
        Sound-speed function.
    lam : float, optional
        Absorption coefficient. Currently unused by this no-absorption RHS.
    solver_config : SolverConfig, optional
        Numerical solver configuration.

    Returns
    -------
    xt : jnp.ndarray, shape (b, Nt, d)
        Beam positions.
    pt : jnp.ndarray, shape (b, Nt, d)
        Beam momenta.
    Mt : jnp.ndarray, shape (b, Nt, d, d)
        Beam Hessians.
    At : jnp.ndarray, shape (b, Nt, 1)
        Beam amplitudes.
    """
    d = x0.shape[-1]

    def single_solve(args):
        """
        Solve one beam with its own time grid.

        Parameters
        ----------
        args : Tuple[jnp.ndarray, ...]
            Tuple ``(x0_i, p0_i, M0_i, A0_i, pol_i, ts_i)``.

        Returns
        -------
        xt : jnp.ndarray, shape (Nt, d)
            Beam positions.
        pt : jnp.ndarray, shape (Nt, d)
            Beam momenta.
        Mt : jnp.ndarray, shape (Nt, d, d)
            Beam Hessians.
        At : jnp.ndarray, shape (Nt, 1)
            Beam amplitudes.
        """
        x0_i, p0_i, M0_i, A0_i, pol_i, ts_i = args
        t0 = ts_i[0]
        t1 = ts_i[-1]

        if solver_config is not None and solver_config.dt0 is not None:
            dt0 = solver_config.dt0
        else:
            dt0 = ts_i[1] - ts_i[0]

        y0 = jnp.concatenate([x0_i.ravel(), p0_i.ravel(), M0_i.ravel(), A0_i.ravel()])
        args_ode = (pol_i, c, d)
        solution = ode_solver_setup(
            coupled_rhs,
            y0,
            t0,
            t1,
            dt0,
            ts_i,
            args_ode,
            solver_config,
            cond_fn=None,
            saveat=None,
        )
        xt, pt, Mt, At = format_solution(solution.ys, d)
        return xt, pt, Mt, At

    xt, pt, Mt, At = vmap(single_solve)((x0, p0, M0, A0, mode, ts))

    return xt, pt, Mt, At


@partial(vmap, in_axes=(0, 0, 0, 0, 0, None, None, None, None, None))
def solve_ODE_intersection(
    x0: jnp.ndarray,
    p0: jnp.ndarray,
    M0: jnp.ndarray,
    a0: jnp.ndarray,
    mode: jnp.ndarray,
    ts: jnp.ndarray,
    c: Callable,
    lam: float,
    surface: Callable,
    solver_config: Optional[SolverConfig] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Solve the ODE for the Gaussian beam and find the intersection time with the surface.

    Parameters
    ----------
    x0 : jnp.ndarray, shape (d,)
        Initial beam position for one vmapped beam.
    p0 : jnp.ndarray, shape (d,)
        Initial momentum for one vmapped beam.
    M0 : jnp.ndarray, shape (d, d)
        Initial Hessian.
    a0 : jnp.ndarray
        Initial amplitude.
    mode : jnp.ndarray
        Hamiltonian branch sign.
    ts : jnp.ndarray, shape (Nt,)
        Time grid.
    c : Callable
        Sound-speed function.
    lam : float
        Absorption coefficient.
    surface : Callable
        Implicit surface function whose zero defines the target surface.
    solver_config : SolverConfig, optional
        Numerical solver configuration.

    Returns
    -------
    xt : jnp.ndarray, shape (Nt, d)
        Beam positions.
    pt : jnp.ndarray, shape (Nt, d)
        Beam momenta.
    Mt : jnp.ndarray, shape (Nt, d, d)
        Beam Hessians.
    At : jnp.ndarray, shape (Nt, 1)
        Beam amplitudes.
    t_int : jnp.ndarray
        Intersection time, or ``inf`` when the root solve fails.

    Notes
    -----
    First solves the beam ODE with dense output, then solves a scalar root
    problem for ``surface(x(t))``.
    """
    t0 = ts[0]
    t1 = ts[-1]
    dt0 = ts[1] - ts[0]
    d = x0.shape[-1]
    y0 = jnp.concatenate([x0.ravel(), p0.ravel(), M0.ravel(), a0.ravel()])
    args_ode = (mode, c, d, lam)

    sol = ode_solver_setup(
        coupled_rhs=coupled_rhs_absorption,
        y0=y0,
        t0=t0,
        t1=t1,
        dt0=dt0,
        ts=ts,
        args=args_ode,
        config=solver_config,
        cond_fn=None,
        saveat=diffrax.SaveAt(dense=True, ts=ts),
    )

    def surface_root(t, _=None):
        """
        Evaluate the surface function along the dense ODE solution.

        Parameters
        ----------
        t : float
            Candidate time.
        _ : Any, optional
            Ignored argument accepted for Optimistix compatibility.

        Returns
        -------
        jnp.ndarray
            Surface residual at ``x(t)``.
        """
        y_t = sol.evaluate(t)
        xt = y_t[:d].real
        return surface(xt)

    # Initial guess for the root (midpoint of the time interval)
    t_init = (t0 + t1) / 2

    # Use Optimistix's Newton method for root finding
    solver = optimistix.Newton(rtol=1e-9, atol=1e-9)
    result = optimistix.root_find(surface_root, solver, t_init, throw=False)
    is_successful = result.result == optimistix.RESULTS.successful
    t_int = jnp.where(is_successful, result.value, jnp.inf)

    xt, pt, Mt, At = format_solution(sol.ys, d)

    return xt, pt, Mt, At, t_int


def solve_ODE_first_hit(
    x0: jnp.ndarray,
    p0: jnp.ndarray,
    M0: jnp.ndarray,
    a0: jnp.ndarray,
    mode: jnp.ndarray,
    ts: jnp.ndarray,
    c: Callable,
    lam: float,
    surface: Callable[[jnp.ndarray], float],
    solver_config: Optional[SolverConfig] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float, bool]:
    """
    Integrate a single beam until it hits `surface(x)=0` (or reaches t1).

    Returns the state at the first hit time (beam axis first, then time axis).

    Parameters
    ----------
    x0, p0, M0, a0 : jnp.ndarray
        Initial GB parameters for one beam.
    mode : jnp.ndarray
        Polarisation (+/-1) for this beam (shape `(1,)` or scalar).
    ts : jnp.ndarray
        Global time grid; assumed uniform. Integration stops at `ts[-1]` if no hit.
    c : Callable
        Sound speed function.
    lam : float
        Absorption parameter.
    surface : Callable[[jnp.ndarray], float]
        Implicit surface function; root at zero triggers a hit.
    solver_config : SolverConfig | None

    Returns
    -------
    (xt, pt, Mt, At, t_hit, hit)
        xt, pt : (1, 1, d)
        Mt     : (1, 1, d, d)
        At     : (1, 1, 1)
        t_hit  : float
        hit    : bool (True if event occurred before ts[-1])
    """
    d = x0.shape[-1]
    t0, t1 = ts[0], ts[-1]
    dt0 = ts[1] - ts[0] if ts.shape[0] > 1 else 1e-3
    y0 = jnp.concatenate([x0.ravel(), p0.ravel(), M0.ravel(), a0.ravel()])
    args_ode = (mode, c, d, lam)

    def cond_fn(t, y, *_, **__):
        """
        Event condition for first surface hit.

        Parameters
        ----------
        t : float
            Current integration time.
        y : jnp.ndarray
            Current flat ODE state.
        *_ : tuple
            Ignored positional event arguments.
        **__ : dict
            Ignored keyword event arguments.

        Returns
        -------
        jnp.ndarray
            Surface residual for the current beam position.
        """
        return surface(y[:d].real)

    event = diffrax.Event(
        cond_fn=cond_fn,
        root_finder=optimistix.Newton(rtol=1e-9, atol=1e-9, norm=optimistix.rms_norm),
    )

    sol = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(coupled_rhs_absorption),
        solver=solver_config.solver if solver_config else diffrax.Tsit5(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=args_ode,
        saveat=diffrax.SaveAt(t1=True, dense=True),
        stepsize_controller=diffrax.PIDController(
            rtol=solver_config.rtol if solver_config else 1e-4,
            atol=solver_config.atol if solver_config else 1e-6,
            pcoeff=solver_config.pcoeff if solver_config else 0.3,
            icoeff=solver_config.icoeff if solver_config else 0.3,
            dcoeff=solver_config.dcoeff if solver_config else 0.0,
        ),
        max_steps=solver_config.max_steps if solver_config else 4096,
        event=event,
    )

    t_hit = sol.ts[-1]
    hit = bool(t_hit < t1 - 1e-9)

    xt, pt, Mt, At = format_solution(sol.ys, d)
    # Keep only the final (hit) state and add a beam axis
    xt = xt[-1:][None, ...]
    pt = pt[-1:][None, ...]
    Mt = Mt[-1:][None, ...]
    At = At[-1:][None, ...]

    return xt, pt, Mt, At, t_hit, hit


def coupled_rhs_QP_absorption(t, y, args) -> jnp.ndarray:
    """
    GB ODE system in (x, p, Q, P, A) coordinates with absorption.

    State layout
    ------------
    y = concat(x (d),
               p (d),
               vec(Q) (d²),
               vec(P) (d²),
               A (1))

    Parameters
    ----------
    t : float
    y : jnp.ndarray, shape (d + d + d² + d² + 1,)
    args : Tuple[mode, c, d, lam]

    Returns
    -------
    jnp.ndarray, same shape as `y`
    """
    mode, c, d, lam = args

    x = y[:d].real
    p = y[d : 2 * d].real
    Q_flat = y[2 * d : 2 * d + d**2]
    P_flat = y[2 * d + d**2 : 2 * d + 2 * d**2]
    A = y[2 * d + 2 * d**2 :]

    Q = rearrange(Q_flat, "(d1 d2) -> d1 d2", d1=d, d2=d)
    P = rearrange(P_flat, "(d1 d2) -> d1 d2", d1=d, d2=d)

    norm_p = jnp.linalg.norm(p)

    c_val = c(x)
    grad_c = grad(c)(x)
    hess_c = hessian(c)(x)

    # Hamiltonian G(x,p) = mode * c(x) * |p|
    G_val = mode * c_val * norm_p
    Gx_val = mode * grad_c * norm_p  # dG/dx
    Gp_val = mode * c_val * p / norm_p  # dG/dp

    # Second derivatives
    eye_d = jnp.eye(d)
    Gpp = mode * c_val * (eye_d / norm_p - jnp.outer(p, p) / norm_p**3)
    Gxp = mode * jnp.outer(grad_c, p) / norm_p  # d²G/dx dp
    Gxx = mode * hess_c * norm_p  # d²G/dx²
    Gpx = Gxp.T  # d²G/dp dx

    # Ray equations
    dx = Gp_val
    dp = -Gx_val

    # Correct linearised Hamiltonian system for Q,P:
    #   dQ/dt = G_xp Q + G_pp P
    #   dP/dt = -G_xx Q - G_xp^T P
    dQ = Gxp @ Q + Gpp @ P
    dP = -Gxx @ Q - Gpx @ P

    # Width matrix M = P Q^{-1} (small d, so direct inverse is fine)
    # Optionally regularise Q a bit if needed:
    Q_inv = jnp.linalg.inv(Q)
    M = P @ Q_inv

    # Amplitude equation in terms of M (same formula you had)
    dA = (
        -A
        * (
            c_val**2 * jnp.trace(M)
            - Gx_val @ Gp_val
            - Gp_val.T @ M @ Gp_val
            + lam * G_val
        )
        / (2 * G_val)
    )

    return jnp.concatenate([dx.ravel(), dp.ravel(), dQ.ravel(), dP.ravel(), dA.ravel()])


def format_solution_QP(ys, d):
    """
    Format the solution of the ODEs in (x, p, Q, P, A) into (x, p, M, A).

    Parameters
    ----------
    ys : jnp.ndarray, shape (Nt, d + d + d**2 + d**2 + 1)
        Flat ODE state trajectory.
    d : int
        Spatial dimension.

    Returns
    -------
    xt : jnp.ndarray, shape (Nt, d)
        Beam positions.
    pt : jnp.ndarray, shape (Nt, d)
        Beam momenta.
    Mt : jnp.ndarray, shape (Nt, d, d)
        Hessians reconstructed as ``P @ inv(Q)``.
    At : jnp.ndarray, shape (Nt, 1)
        Beam amplitudes.
    """
    xt = ys[..., :d].real
    pt = ys[..., d : 2 * d].real

    Q_flat = ys[..., 2 * d : 2 * d + d**2]
    P_flat = ys[..., 2 * d + d**2 : 2 * d + 2 * d**2]
    At = ys[..., 2 * d + 2 * d**2 :]

    Qt = rearrange(Q_flat, "t (d1 d2) -> t d1 d2", d1=d, d2=d)
    Pt = rearrange(P_flat, "t (d1 d2) -> t d1 d2", d1=d, d2=d)

    def _MK(Q, P):
        """
        Reconstruct the Hessian matrix from Q/P variables.

        Parameters
        ----------
        Q : jnp.ndarray, shape (d, d)
            Q block of the linearised Hamiltonian system.
        P : jnp.ndarray, shape (d, d)
            P block of the linearised Hamiltonian system.

        Returns
        -------
        jnp.ndarray, shape (d, d)
            ``P @ inv(Q)``.
        """
        return P @ jnp.linalg.inv(Q)

    Mt = jax.vmap(_MK)(Qt, Pt)

    return xt, pt, Mt, At


@partial(vmap, in_axes=(0, 0, 0, 0, 0, None, None, None, None))
def solve_ODE_QP_base(
    x0: jnp.ndarray,
    p0: jnp.ndarray,
    M0: jnp.ndarray,
    a0: jnp.ndarray,
    mode: jnp.ndarray,
    ts: jnp.ndarray,
    c: Callable,
    lam: float = 0.0,
    solver_config: Optional[SolverConfig] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Solve the GB ODEs using (Q,P) instead of M directly.

    Parameters
    ----------
    x0 : jnp.ndarray, shape (d,)
        Initial beam position for one vmapped beam.
    p0 : jnp.ndarray, shape (d,)
        Initial beam momentum.
    M0 : jnp.ndarray, shape (d, d)
        Initial Hessian matrix.
    a0 : jnp.ndarray
        Initial amplitude.
    mode : jnp.ndarray
        Hamiltonian branch sign.
    ts : jnp.ndarray, shape (Nt,)
        Time grid.
    c : Callable
        Sound-speed function.
    lam : float, default=0.0
        Absorption coefficient.
    solver_config : SolverConfig, optional
        Numerical solver configuration.

    Returns
    -------
    xt : jnp.ndarray, shape (Nt, d)
        Beam positions.
    pt : jnp.ndarray, shape (Nt, d)
        Beam momenta.
    Mt : jnp.ndarray, shape (Nt, d, d)
        Reconstructed Hessian matrices.
    At : jnp.ndarray, shape (Nt, 1)
        Beam amplitudes.

    Notes
    -----
    Uses initial condition ``Q(0) = I`` and ``P(0) = M0`` so that
    ``M(0) = M0``.
    """
    t0 = ts[0]
    t1 = ts[-1]
    dt0 = ts[1] - ts[0]
    d = x0.shape[-1]

    # Q(0) = I, P(0) = M0
    Q0 = jnp.eye(d, dtype=M0.dtype)
    P0 = M0

    y0 = jnp.concatenate([x0.ravel(), p0.ravel(), Q0.ravel(), P0.ravel(), a0.ravel()])
    args_ode = (mode, c, d, lam)

    solution = ode_solver_setup(
        coupled_rhs_QP_absorption,
        y0,
        t0,  # <-- no float() here
        t1,
        dt0,
        ts,
        args_ode,
        solver_config,
        cond_fn=None,
        saveat=None,
    )

    xt, pt, Mt, At = format_solution_QP(solution.ys, d)
    return xt, pt, Mt, At
