import jax.numpy as jnp
from jax import jit


@jit
def unitary_fft(arr: jnp.ndarray) -> jnp.ndarray:
    """
    Unitary N-D FFT with centred zero-frequency component.

    Parameters
    ----------
    arr : jnp.ndarray
        Real or complex array in the spatial domain.

    Returns
    -------
    jnp.ndarray
        Fourier transform with ``norm="ortho"`` and `fftshift` applied.
    """
    return jnp.fft.fftshift(jnp.fft.fftn(arr, norm="ortho"))


@jit
def unitary_ifft(arr: jnp.ndarray) -> jnp.ndarray:
    """
    Unitary inverse FFT matching :func:`unitary_fft`.

    Parameters
    ----------
    arr : jnp.ndarray
        Fourier-domain array (already shifted).

    Returns
    -------
    jnp.ndarray
        Spatial-domain array with unitary scaling.
    """
    return jnp.fft.ifftn(jnp.fft.ifftshift(arr), norm="ortho")


def convert_space(
    array: jnp.ndarray, input_space: str, target_space: str
) -> jnp.ndarray:
    """
    Convert an array between spatial and Fourier domains.

    Parameters
    ----------
    array : jnp.ndarray
    input_space : {"spatial", "fourier"}
        Declares the domain of ``array``.
    target_space : {"spatial", "fourier"}
        Desired output domain.

    Returns
    -------
    jnp.ndarray
        Array in the requested domain.

    Raises
    ------
    ValueError
        If an unsupported conversion is requested.
    """
    if input_space == target_space:
        return array
    elif input_space == "spatial" and target_space == "fourier":
        return unitary_fft(array)
    elif input_space == "fourier" and target_space == "spatial":
        return unitary_ifft(array)
    else:
        raise ValueError("Invalid conversion.")
