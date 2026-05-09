"""
Ray-tracing diagnostic for the Gaussian beam Hamiltonian.
"""
import jax.numpy as jnp

###########################
# --- 2D Velocity Maps ---#
###########################


def constant_2d(x, c0=1.0):
    """Constant speed of sound."""
    return c0 * jnp.ones_like(x[..., 0])


def gaussian_bump_2d(x, center=(0.5, 1.0), amplitude=0.5, width=20.0, c0=1.0):
    """Gaussian bump/dip velocity field."""
    dist_sq = jnp.sum((x - jnp.array(center)) ** 2, axis=-1)
    return c0 - amplitude * jnp.exp(-width * dist_sq)


def vertical_gradient_2d(x, c0=1.0, gradient=1.0 / 3.0):
    """Linearly increasing speed with x-coordinate."""
    return c0 + gradient * x[..., 0]


def sinusoidal_2d(x, c0=1.0, amp=0.2, freq_x=4.0, freq_y=2.0):
    """Sinusoidal variation in 2D."""
    return c0 + amp * jnp.sin(freq_x * jnp.pi * x[..., 0]) * jnp.sin(
        freq_y * jnp.pi * x[..., 1]
    )


def peaks_function_2d(x):
    """
    MATLAB's peaks function scaled for [0,1]x[0,2] domain.

    Ref: https://uk.mathworks.com/help/matlab/ref/peaks.html#d126e1286246
    """
    # Scale x and y to approx [-3, 3] range for peaks compatibility if needed
    sx = 6 * x[..., 0] - 3
    sy = 3 * x[..., 1] - 3  # Scaled for [0, 2] height
    z = (
        3 * (1 - sx) ** 2 * jnp.exp(-(sx**2) - (sy + 1) ** 2)
        - 10 * (sx / 5 - sx**3 - sy**5) * jnp.exp(-(sx**2) - sy**2)
        - 1 / 3 * jnp.exp(-((sx + 1) ** 2) - sy**2)
    )
    # Scale output to a reasonable speed range (e.g., 1.0 +/- variation)
    return 1.5 + z / 10.0


def checkerboard_2d(x, c1=1.0, c2=1.5, grid_size=0.2):
    """Checkerboard pattern."""
    idx_x = jnp.floor(x[..., 0] / grid_size)
    idx_y = jnp.floor(x[..., 1] / grid_size)
    is_c1 = (idx_x + idx_y) % 2 == 0
    return jnp.where(is_c1, c1, c2)


# Lenses (adapted from rays2d.py) - Ensure parameters (center, radius, c0) match domain
def sos_maxwell_fish_eye_2d(x, center=(0.5, 0.5), radius=0.25, c0=2.0):
    r_sq = jnp.sum((x - jnp.array(center)) ** 2, axis=-1)
    r_scaled_sq = r_sq / (radius**2)
    n = 2 / (1 + r_scaled_sq)  # Refractive index based on scaled radius squared
    c = c0 / n
    # Apply lens effect only within the radius
    return jnp.where(r_sq <= radius**2, c, c0)


def sos_eaton_lens_2d(x, center=(0.5, 0.5), radius=0.25, c0=2.0):
    r = jnp.sqrt(jnp.sum((x - jnp.array(center)) ** 2, axis=-1))
    r_scaled = r / radius
    n = jnp.sqrt(jnp.maximum(1e-9, 2 / r_scaled - 1))  # Add epsilon for safety at r=0
    c = c0 / n
    return jnp.where(r_scaled <= 1.0, c, c0)


def sos_mikaelian_lens(x, center=(0.5, 0.5), radius=0.25, c0=2):
    r = jnp.sqrt(jnp.sum((x - jnp.array(center)) ** 2, axis=-1)) / radius
    n = jnp.cosh(r)
    c = c0 / n
    return jnp.where(r <= radius, c, c0)


def sos_exponential_lens(x, center=(0.5, 1.8), radius=0.25, c0=2, alpha=1):
    r = jnp.sqrt(jnp.sum((x - jnp.array(center)) ** 2, axis=-1)) / radius
    n = jnp.exp(alpha * r)
    c = c0 / n
    return jnp.where(r <= radius, c, c0)


def sos_hyperbolic_secant(x, center=(0.5, 0.5), width=0.25, c0=2):
    y_dist = jnp.abs(x[..., 1] - center[1])
    n = 1 / jnp.cosh(y_dist / width)
    return c0 * (1 + 0.5 * n)


def sos_circular_caustic(x, center=(0.5, 0.5), radius=0.25, c0=2):
    r = jnp.sqrt(jnp.sum((x - jnp.array(center)) ** 2, axis=-1))
    n = 1 + (r / radius) ** 2
    return c0 / n


def gaussian_bump(x):
    center = jnp.array([0.5, 1.0])
    return 1.0 - 0.5 * jnp.exp(-20 * jnp.sum((x - center) ** 2, axis=-1))


def vertical_gradient(x):
    return 1.0 + x[..., 0] / 3


def sinusoidal(x):
    return 1.0 + 0.2 * jnp.sin(4 * jnp.pi * x[..., 0]) * jnp.sin(2 * jnp.pi * x[..., 1])


def gaussian_dips_2d(x):
    center1 = jnp.array([0.35, 0.5])
    center2 = jnp.array([0.65, 0.5])
    return (
        2
        - 0.8 * jnp.exp(-50 * jnp.sum((x - center1) ** 2, axis=-1))
        - 1.2 * jnp.exp(-50 * jnp.sum((x - center2) ** 2, axis=-1))
    )


###########################
# --- 3D Velocity Maps ---#
###########################


def constant_3d(x, c0=1.0):
    """Constant speed of sound in 3D."""
    return c0 + 0 * x[..., 0]


def gaussian_bump_3d(x, center=(0.5, 0.5, 0.5), amplitude=0.5, width=20.0, c0=1.0):
    """Gaussian bump/dip velocity field in 3D."""
    dist_sq = jnp.sum((x - jnp.array(center)) ** 2, axis=-1)
    return c0 + amplitude * jnp.exp(-width * dist_sq)  # Use +amplitude for bump


def gradient_3d(x, c0=1.0, grad_vec=(0.1, 0.05, -0.02)):
    """Linear gradient in 3D."""
    g = jnp.array(grad_vec)
    return c0 + jnp.dot(x, g)  # Assumes x has shape (..., 3)


def layered_medium_3d(x, c_layers=(1.0, 1.5, 1.2), boundaries=(0.4, 0.7), axis=2):
    """Layered medium along a specified axis (default z)."""
    coord = x[..., axis]
    c0 = c_layers[0]
    c1 = c_layers[1]
    c2 = c_layers[2]
    b1 = boundaries[0]
    b2 = boundaries[1]
    return jnp.where(coord < b1, c0, jnp.where(coord < b2, c1, c2))


def spherical_shell_3d(
    x,
    center=(0.5, 0.5, 0.5),
    r_inner=0.2,
    r_outer=0.4,
    c_inside=1.0,
    c_shell=1.5,
    c_outside=1.0,
):
    """Spherical shell with different speeds."""
    dist = jnp.sqrt(jnp.sum((x - jnp.array(center)) ** 2, axis=-1))
    is_inside = dist < r_inner
    is_shell = (dist >= r_inner) & (dist < r_outer)
    return jnp.where(is_inside, c_inside, jnp.where(is_shell, c_shell, c_outside))


def checkerboard_3d(x, c1=1.0, c2=1.5, grid_size=0.2):
    """3D Checkerboard pattern."""
    idx_x = jnp.floor(x[..., 0] / grid_size)
    idx_y = jnp.floor(x[..., 1] / grid_size)
    idx_z = jnp.floor(x[..., 2] / grid_size)
    is_c1 = (idx_x + idx_y + idx_z) % 2 == 0
    return jnp.where(is_c1, c1, c2)


def velocity_field_3d(x):
    # Example: Gaussian bump in 3D
    center = jnp.array([0.5, 0.5, 0.5])
    return 1.0 + 0.5 * jnp.exp(-20 * jnp.sum((x - center) ** 2, axis=-1))


def parabolic_mirror_3d(x, focus=(0.5, 0.5, 0.5), a=1.0, c0=2.0):
    """Parabolic mirror in 3D."""
    y_dist = x[..., 1] - focus[1]
    x_dist = x[..., 0] - focus[0]
    z_dist = x[..., 2] - focus[2]
    r = jnp.sqrt(x_dist**2 + y_dist**2 + z_dist**2)
    n = 1 + a * r**2
    return c0 / n


def maxwell_fish_eye_3d(x, center=(0.5, 0.5, 0.5), radius=0.25, c0=2.0):
    """Maxwell's fish-eye lens in 3D."""
    r = jnp.sqrt(jnp.sum((x - jnp.array(center)) ** 2, axis=-1)) / radius
    n = 2 / (1 + (r / radius) ** 2)
    c = c0 / n
    return jnp.where(r <= radius, c, c0)


def eaton_lens_3d(x, center=(0.5, 0.5, 0.5), radius=0.25, c0=2.0):
    """Eaton lens in 3D."""
    r = jnp.sqrt(jnp.sum((x - jnp.array(center)) ** 2, axis=-1)) / radius
    n = jnp.sqrt(2 / r - 1)
    c = c0 / n
    return jnp.where(r <= 1, c, c0)


def grin_lens_3d(x, center=(0.5, 0.5, 0.5), radius=0.25, c0=2.0, n0=1.5):
    """GRIN lens in 3D."""
    r = jnp.sqrt(jnp.sum((x - jnp.array(center)) ** 2, axis=-1)) / radius
    n = n0 * (1 - 0.5 * (r / radius) ** 2)
    c = c0 / n
    return jnp.where(r <= radius, c, c0)
