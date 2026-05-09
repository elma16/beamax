from jax import grad, jit, vmap
import jax.numpy as jnp
from typing import Optional, Callable
from functools import partial
import warnings

warnings.filterwarnings("ignore", module="equinox")

__all__ = ["G", "Gx", "Gp", "check_M0", "prepare_M0", "is_diagonal"]


@partial(jit, static_argnames="c")
def G(x: jnp.ndarray, p: jnp.ndarray, mode: jnp.ndarray, c: Callable) -> jnp.ndarray:
    """
    Hamiltonian for acoustics: `G(x,p) = mode * c(x) * ||p||`.

    Parameters
    ----------
    x : jnp.ndarray, shape (..., d)
        Positions.
    p : jnp.ndarray, shape (..., d)
        Momenta.
    mode : jnp.ndarray, shape (...,)
        ±1 branch selector.
    c : Callable[[jnp.ndarray], jnp.ndarray]
        Speed of sound `c(x)`.

    Returns
    -------
    jnp.ndarray, shape (...,)
        Hamiltonian values. dtype follows `c(x)`/inputs.

    Notes
    -----
    JIT-compiled; pure; vectorize with `vmap` as needed.
    """
    return mode * c(x) * jnp.linalg.norm(p, axis=-1)


vmap_g = vmap(G, in_axes=(0, 0, 0, None))


@partial(jit, static_argnames="c")
def Gx(x: jnp.ndarray, p: jnp.ndarray, mode: jnp.ndarray, c: Callable) -> jnp.ndarray:
    """
    ∂G/∂x for `G(x,p) = mode * c(x) * ||p||`.

    Parameters
    ----------
    x : jnp.ndarray, shape (..., d)
    p : jnp.ndarray, shape (..., d)
    mode : jnp.ndarray, shape (...,)
    c : Callable

    Returns
    -------
    jnp.ndarray, shape (..., d)
        mode * ∇c(x) * ||p||.
    """
    grad_c = grad(c)(x)
    return mode * grad_c * jnp.linalg.norm(p, axis=-1)


vmap_gx = vmap(Gx, in_axes=(0, 0, 0, None))


@partial(jit, static_argnames="c")
def Gp(x: jnp.ndarray, p: jnp.ndarray, mode: jnp.ndarray, c: Callable) -> jnp.ndarray:
    """
    ∂G/∂p for `G(x,p) = mode * c(x) * ||p||`.

    Parameters
    ----------
    x : jnp.ndarray, shape (..., d)
    p : jnp.ndarray, shape (..., d)
    mode : jnp.ndarray, shape (...,)
    c : Callable

    Returns
    -------
    jnp.ndarray, shape (..., d)
        mode * c(x) * p / ||p||.
    """

    return mode * c(x) * p / jnp.linalg.norm(p, axis=-1)


vmap_gp = vmap(Gp, in_axes=(0, 0, 0, None))


def check_M0(M0: jnp.ndarray) -> None:
    """
    Validate initial Hessian `M0`: symmetric and Im(M0) ≻ 0.

    Parameters
    ----------
    M0 : jnp.ndarray, shape (b, d, d), complex

    Raises
    ------
    ValueError
        If symmetry fails or Im(M0) is not positive definite (per batch).
    """
    if not jnp.allclose(M0, jnp.transpose(M0, (0, 2, 1))):
        raise ValueError("M0 must be symmetric.")

    if not jnp.all(jnp.linalg.eigh(jnp.imag(M0).real)[0] > 0):
        raise ValueError("Imaginary part of M0 must be positive definite.")


def prepare_M0(alpha0: Optional[jnp.ndarray], M0: Optional[jnp.ndarray]) -> jnp.ndarray:
    """
    Construct/validate initial Hessian.

    Parameters
    ----------
    alpha0 : jnp.ndarray | None, shape (b, d), complex
        If given, produces diagonal M0 = diag(alpha0). Im(alpha0) should be > 0.
    M0 : jnp.ndarray | None, shape (b, d, d), complex
        If given, must be symmetric with Im(M0) ≻ 0.

    Returns
    -------
    jnp.ndarray, shape (b, d, d), complex
        Validated/constructed Hessian.

    Raises
    ------
    ValueError
        If both or neither of `alpha0` and `M0` are provided.
    """
    if (M0 is None) == (alpha0 is None):
        raise ValueError("Provide either alpha0 or M0, but not both.")

    if M0 is None:
        d = alpha0.shape[-1]
        # assert jnp.all(alpha0.imag > 0), "Imaginary part of alpha0 must be positive."
        M0 = jnp.einsum("bd,dj->bdj", alpha0, jnp.eye(d))

    # check_M0(M0)
    return M0


def is_diagonal(M0: jnp.ndarray) -> bool:
    """
    Test whether `M0` is diagonal (per batch).

    Parameters
    ----------
    M0 : jnp.ndarray, shape (b, d, d)

    Returns
    -------
    bool
    """
    d = M0.shape[-1]
    diag_elements = jnp.diagonal(M0, axis1=1, axis2=2)
    reconstructed_M0 = diag_elements[:, :, None] * jnp.eye(d)
    return jnp.allclose(M0, reconstructed_M0)
