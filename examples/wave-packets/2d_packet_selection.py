"""Recover a 2D pressure field from its strongest wave-packet coefficients."""

# %% Import the wave-packet transform
import jax.numpy as jnp

from beamax import DyadicDecomposition, MSWPT
from beamax.utils import rel_l2


def main() -> None:
    # %% Make a modulated Gaussian initial pressure
    n = 32
    axis = (jnp.arange(n) - n // 2) / n
    x, y = jnp.meshgrid(axis, axis, indexing="ij")
    envelope = jnp.exp(-(x**2 + y**2) / (2 * 0.12**2))
    p0 = envelope * jnp.cos(2 * jnp.pi * (6 * x + 3 * y))

    # %% Analyse the field and rank its coefficients
    tiling = DyadicDecomposition(2, p0.shape, (4, 8), (1, 1))
    transform = MSWPT(tiling, redundancy=2, windowing="rectangular_mirror")
    coeffs = transform.forward(p0.copy(), input_type="spatial")
    by_magnitude = jnp.argsort(jnp.abs(coeffs), descending=True)

    # %% Reconstruct with 16, 64 and all coefficients
    for count in (16, 64, coeffs.size):
        keep = by_magnitude[:count]
        selected = jnp.zeros_like(coeffs).at[keep].set(coeffs[keep])
        recovered = transform.inverse(selected, output_type="spatial")
        error = rel_l2(p0, recovered)
        print(f"{count:4d} coefficients     relative L2 error {error:.2e}")


if __name__ == "__main__":
    main()
