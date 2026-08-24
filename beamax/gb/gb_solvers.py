import jax
import jax.numpy as jnp
from jax import vmap, grad, hessian
import math

import diffrax
from functools import partial
from einops import rearrange
from dataclasses import dataclass
from typing import Tuple, Callable, Protocol, Optional


__all__ = [
    "SolverFn",
    "SolverConfig",
    "solve_hom_diag",
    "solve_hom_general",
    "solve_hom_TR",
    "solve_ODE_base",
    "solve_ODE_batch_t",
    "solve_ODE_batch_t_terminal",
]


def _matmul_highest(left, right):
    """Multiply with stable float32 precision across backends."""
    return jnp.matmul(left, right, precision=jax.lax.Precision.HIGHEST)


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
        /,
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


def compute_amp_hom_gen(
    p0: jnp.ndarray,
    m0: jnp.ndarray,
    c0: jnp.ndarray,
    ts: jnp.ndarray,
    a0: jnp.ndarray,
) -> jnp.ndarray:
    r"""
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
        $$
        a(t)=\frac{a_0}{
        \sqrt{\det\!\left(
        I+\frac{c_0t}{\lVert p_0\rVert}P_\perp M_0
        \right)}}.
        $$
    """
    d = p0.shape[-1]
    id = rearrange(jnp.eye(d), "i j -> 1 1 i j")
    p0 = rearrange(p0, "b d -> b 1 d 1")
    normp = jnp.linalg.norm(p0, axis=-2, keepdims=True)
    c0 = rearrange(c0, "b 1 -> b 1 1 1")
    m0 = rearrange(m0, "b d1 d2 -> b 1 d1 d2")
    ts = rearrange(ts, "nt -> 1 nt 1 1")

    p_perp = (
        id
        - jnp.einsum(
            "btia,btjb->btij",
            p0,
            p0,
            precision=jax.lax.Precision.HIGHEST,
        )
        / normp**2
    )

    interior = (
        id
        + c0
        * ts
        * jnp.einsum(
            "btij,btjk->btik",
            p_perp,
            m0,
            precision=jax.lax.Precision.HIGHEST,
        )
        / normp
    )
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
    r"""
    $M(t)$ for general homogeneous GB.

    Parameters
    ----------
    p0 : (b, d)
    m0 : (b, d, d)
    c0 : (b, 1)
    ts : (Nt,)

    Returns
    -------
    jnp.ndarray, shape (b, Nt, d, d)
        $$
        M(t)=M_0\left(
        I+\frac{c_0t}{\lVert p_0\rVert}P_\perp M_0
        \right)^{-1}.
        $$
    """
    d = p0.shape[-1]
    id = rearrange(jnp.eye(d), "i j -> 1 1 i j")
    p0 = rearrange(p0, "b d -> b 1 d 1")
    normp = jnp.linalg.norm(p0, axis=-2, keepdims=True)
    c0 = rearrange(c0, "b 1 -> b 1 1 1")
    m0 = rearrange(m0, "b d1 d2 -> b 1 d1 d2")
    ts = rearrange(ts, "nt -> 1 nt 1 1")

    p_perp = (
        id
        - jnp.einsum(
            "btia,btjb->btij",
            p0,
            p0,
            precision=jax.lax.Precision.HIGHEST,
        )
        / normp**2
    )

    interior = (
        id
        + c0
        * ts
        * jnp.einsum(
            "btij,btjk->btik",
            p_perp,
            m0,
            precision=jax.lax.Precision.HIGHEST,
        )
        / normp
    )
    return jnp.matmul(
        m0,
        jnp.linalg.inv(interior),
        precision=jax.lax.Precision.HIGHEST,
    )


def compute_amp_hom_diag_2d(
    p0: jnp.ndarray,
    normp: jnp.ndarray,
    alpha0: jnp.ndarray,
    c0: jnp.ndarray,
    ts: jnp.ndarray,
    a0: jnp.ndarray,
) -> jnp.ndarray:
    r"""
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
        $$
        a(t)=a_0\left(
        1+\frac{c_0t}{\lVert p_0\rVert^3}
        \left\langle
        p_0^{\odot2},\operatorname{rev}(\alpha_0)
        \right\rangle
        \right)^{-(d-1)/2}.
        $$
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
    r"""
    $M(t)$ with diagonal M0 via Sherman–Morrison.

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
        $N_0(A+uv^{\mathsf T})^{-1}$ with diagonal $A$ and a rank-1 update
        from the ray direction.
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

    u = -c0 * ts * p0 / normp**3
    vT = jnp.einsum(
        "btij,btik->btjk",
        p0,
        N0,
        precision=jax.lax.Precision.HIGHEST,
    )

    A_inv_u = jnp.matmul(A_inv, u, precision=jax.lax.Precision.HIGHEST)
    vT_A_inv = jnp.matmul(vT, A_inv, precision=jax.lax.Precision.HIGHEST)

    scalar = 1 + jnp.matmul(vT_A_inv, u, precision=jax.lax.Precision.HIGHEST)

    Yt_inv = (
        A_inv
        - jnp.matmul(A_inv_u, vT_A_inv, precision=jax.lax.Precision.HIGHEST) / scalar
    )

    return jnp.matmul(N0, Yt_inv, precision=jax.lax.Precision.HIGHEST)


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
    r"""
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
    Assumes $c(x)$ is homogeneous, ``M0`` is diagonal, and
    $d\in\{1,2,3\}$. The diagonal and dimensionality assumptions are relaxed by
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
        # A length-b vector denotes one time per beam.
        if ts_arr.shape[0] == b and ts_arr.size == b:
            ts_beams = ts_arr[:, None]
        else:
            ts_beams = jnp.broadcast_to(ts_arr[None, :], (b, ts_arr.shape[0]))
    else:
        if ts_arr.shape[0] == b:
            ts_beams = ts_arr
        elif ts_arr.ndim == 2 and ts_arr.shape[1] == b:
            ts_beams = jnp.swapaxes(ts_arr, 0, 1)
        else:
            flat = ts_arr.reshape(-1)
            ts_beams = jnp.broadcast_to(flat[None, :], (b, flat.shape[0]))

    id = rearrange(jnp.eye(d), "i j -> 1 1 i j")
    c0 = c(jnp.zeros((d,))) * mode[:, None]
    p0 = pT

    normp = jnp.linalg.norm(p0, axis=-1, keepdims=True)

    dt = ts_beams - ts_beams[:, :1]
    dirn = (p0 / normp)[:, None, :]

    # x(t) = xT + c p̂ (t - t_ref)
    x0 = xT[:, None, :] + c0[:, None, :] * dirn * dt[:, :, None]
    p0_time = jnp.broadcast_to(p0[:, None, :], x0.shape)

    mT_b = rearrange(mT, "b i j -> b 1 i j")
    p_perp = id - (p0[:, None, :, None] * p0[:, None, None, :]) / rearrange(
        normp**2, "b 1 -> b 1 1 1"
    )

    c0_b = rearrange(c0, "b 1 -> b 1 1 1")
    normp_b = rearrange(normp, "b 1 -> b 1 1 1")
    dt_b = dt[:, :, None, None]

    interior = (
        id
        + c0_b
        * dt_b
        * jnp.einsum(
            "btij,btjk->btik",
            p_perp,
            mT_b,
            precision=jax.lax.Precision.HIGHEST,
        )
        / normp_b
    )
    interior_inv = jnp.linalg.inv(interior)
    m0 = jnp.einsum(
        "...ij,...jk->...ik",
        jnp.broadcast_to(mT_b, interior.shape),
        interior_inv,
        precision=jax.lax.Precision.HIGHEST,
    )

    a0 = aT[:, None, :] / jnp.sqrt(jnp.linalg.det(interior))[..., None]

    return x0, p0_time, m0, a0


@dataclass(frozen=True)
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

    def __post_init__(self) -> None:
        """Validate integration limits and tolerances."""
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int):
            raise ValueError("max_steps must be a positive integer.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be a positive integer.")
        for name in ("rtol", "atol"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("pcoeff", "icoeff", "dcoeff"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.dt0 is not None and (
            not math.isfinite(float(self.dt0)) or float(self.dt0) <= 0
        ):
            raise ValueError("dt0 must be finite and positive when provided.")

    @classmethod
    def from_precision(
        cls,
        use_x64: Optional[bool] = None,
        solver: Optional[diffrax.AbstractSolver] = None,
        **overrides,
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
            # Pyright cannot see JAX's dynamically attached x64 flag.
            use_x64 = bool(getattr(jax.config, "x64_enabled", False))

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

        defaults.update(overrides)

        if solver is not None:
            defaults["solver"] = solver
        elif "solver" not in defaults:
            defaults["solver"] = diffrax.Tsit5()

        return cls(**defaults)


def ode_solver_setup(
    coupled_rhs: Callable,
    y0: jnp.ndarray,
    # vmap may supply these as scalar arrays rather than Python floats.
    t0,
    t1,
    dt0,
    ts: jnp.ndarray,
    args: Tuple,
    config: Optional[SolverConfig] = None,
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
    )
    return solution


def riccati_rhs(M, x, p, mode, c):
    r"""
    Riccati equation for Hessian evolution $\dot M$.

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
        $$
        \dot M=-\left(
        G_{xx}+G_{xp}M+MG_{xp}^{\mathsf T}+MG_{pp}M
        \right).
        $$

        With
        $(G_{xp})_{ij}=\partial^2G/(\partial x_i\,\partial p_j)$.
    """
    d = x.shape[0]
    normp = jnp.linalg.norm(p)
    c_val = c(x)
    grad_c = grad(c)(x)
    hess_c = hessian(c)(x)

    Gpp = mode * c_val * (jnp.eye(d) / normp - jnp.outer(p, p) / normp**3)
    Gxp = mode * jnp.outer(grad_c, p) / normp
    Gxx = mode * hess_c * normp

    return -(
        Gxx
        + _matmul_highest(Gxp, M)
        + _matmul_highest(M, Gxp.T)
        + _matmul_highest(_matmul_highest(M, Gpp), M)
    )


def coupled_rhs_absorption(t, y, args) -> jnp.ndarray:
    r"""
    Full GB ODE system with absorption `lam`.

    State layout
    ------------
    $\mathbf y=\operatorname{concat}(x,p,\operatorname{vec}M,A)$.

    Parameters
    ----------
    t : float
    y : jnp.ndarray, shape (2 * d + d**2 + 1,)
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
            - _matmul_highest(Gx_val, Gp_val)
            - _matmul_highest(_matmul_highest(Gp_val.T, M), Gp_val)
            + lam * G_val
        )
        / (2 * G_val)
    )
    return jnp.concatenate([dx.ravel(), dp.ravel(), dM.ravel(), dA.ravel()])


def coupled_rhs(t, y, args) -> jnp.ndarray:
    r"""
    GB ODE system without absorption.

    Parameters
    ----------
    t : float
    y : jnp.ndarray, shape (2 * d + d**2 + 1,)
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
        * (
            c(x) ** 2 * jnp.trace(M)
            - _matmul_highest(Gx_val, Gp_val)
            - _matmul_highest(_matmul_highest(Gp_val.T, M), Gp_val)
        )
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
        saveat=None,
    )

    xt, pt, Mt, At = format_solution(solution.ys, d)

    return xt, pt, Mt, At


def _solve_ODE_batch_t(
    x0: jnp.ndarray,
    p0: jnp.ndarray,
    M0: jnp.ndarray,
    A0: jnp.ndarray,
    mode: jnp.ndarray,
    ts: jnp.ndarray,
    c: Callable,
    lam: Optional[float] = None,
    solver_config: Optional[SolverConfig] = None,
    *,
    terminal_only: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Shared per-beam integration for full-grid and terminal saves."""
    del lam
    d = x0.shape[-1]
    saveat = diffrax.SaveAt(t1=True) if terminal_only else None

    def single_solve(args):
        x0_i, p0_i, M0_i, A0_i, pol_i, ts_i = args
        t0 = ts_i[0]
        t1 = ts_i[-1]
        dt0 = (
            solver_config.dt0
            if solver_config is not None and solver_config.dt0 is not None
            else ts_i[1] - ts_i[0]
        )
        y0 = jnp.concatenate([x0_i.ravel(), p0_i.ravel(), M0_i.ravel(), A0_i.ravel()])
        solution = ode_solver_setup(
            coupled_rhs,
            y0,
            t0,
            t1,
            dt0,
            ts_i,
            (pol_i, c, d),
            solver_config,
            saveat=saveat,
        )
        return format_solution(solution.ys, d)

    return vmap(single_solve)((x0, p0, M0, A0, mode, ts))


def solve_ODE_batch_t(
    x0: jnp.ndarray,
    p0: jnp.ndarray,
    M0: jnp.ndarray,
    A0: jnp.ndarray,
    mode: jnp.ndarray,
    ts: jnp.ndarray,
    c: Callable,
    lam: Optional[float] = None,
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
    return _solve_ODE_batch_t(
        x0,
        p0,
        M0,
        A0,
        mode,
        ts,
        c,
        lam,
        solver_config,
        terminal_only=False,
    )


def solve_ODE_batch_t_terminal(
    x0: jnp.ndarray,
    p0: jnp.ndarray,
    M0: jnp.ndarray,
    A0: jnp.ndarray,
    mode: jnp.ndarray,
    ts: jnp.ndarray,
    c: Callable,
    lam: Optional[float] = None,
    solver_config: Optional[SolverConfig] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve per-beam intervals and save each terminal state.

    The return values retain a singleton time axis.

    Returns
    -------
    xt, pt : jnp.ndarray, shape (b, 1, d)
        Terminal beam positions and momenta.
    Mt : jnp.ndarray, shape (b, 1, d, d)
        Terminal complex Hessians.
    At : jnp.ndarray, shape (b, 1, 1)
        Terminal complex amplitudes.
    """
    return _solve_ODE_batch_t(
        x0,
        p0,
        M0,
        A0,
        mode,
        ts,
        c,
        lam,
        solver_config,
        terminal_only=True,
    )
