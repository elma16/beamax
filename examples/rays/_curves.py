"""
Ray-tracing diagnostic for the Gaussian beam Hamiltonian.
"""
import jax.numpy as jnp

#################
# 2D curves
#################


def circle_curve(t, radius=0.5, center=(0.5, 0.5)):
    if radius == 0:
        return jnp.tile(jnp.array(center), (len(t), 1))
    return jnp.stack(
        [
            center[0] + radius * jnp.cos(2 * jnp.pi * t),
            center[1] + radius * jnp.sin(2 * jnp.pi * t),
        ],
        axis=-1,
    )


def line_curve(t, start=(0, 0), end=(1, 1)):
    return jnp.stack(
        [start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t],
        axis=-1,
    )


def spiral_curve(t, max_radius=0.5, num_turns=2, center=(0.5, 0.5)):
    angle = 2 * jnp.pi * num_turns * t
    radius = max_radius * t
    return jnp.stack(
        [center[0] + radius * jnp.cos(angle), center[1] + radius * jnp.sin(angle)],
        axis=-1,
    )


def zigzag_curve(t, amplitude=0.5, frequency=2, start=(0, 0), end=(1, 1)):
    x = start[0] + (end[0] - start[0]) * t
    y = (
        start[1]
        + (end[1] - start[1]) * t
        + amplitude * jnp.sin(2 * jnp.pi * frequency * t)
    )
    return jnp.stack([x, y], axis=-1)


def box_curve(t, width=0.8, height=0.6, center=(0.5, 0.5)):
    x_center, y_center = center
    x = x_center - width / 2
    y = y_center - height / 2
    perimeter = 2 * (width + height)
    distance = t * perimeter

    # Calculate coordinates for each side of the box
    side1 = jnp.stack(
        [x + jnp.minimum(distance, width), y * jnp.ones_like(distance)], axis=-1
    )
    side2 = jnp.stack(
        [
            (x + width) * jnp.ones_like(distance),
            y + jnp.minimum(jnp.maximum(distance - width, 0), height),
        ],
        axis=-1,
    )
    side3 = jnp.stack(
        [
            x + width - jnp.minimum(jnp.maximum(distance - width - height, 0), width),
            (y + height) * jnp.ones_like(distance),
        ],
        axis=-1,
    )
    side4 = jnp.stack(
        [
            x * jnp.ones_like(distance),
            y
            + height
            - jnp.minimum(jnp.maximum(distance - 2 * width - height, 0), height),
        ],
        axis=-1,
    )

    # Create masks for each side
    mask1 = distance < width
    mask2 = (distance >= width) & (distance < width + height)
    mask3 = (distance >= width + height) & (distance < 2 * width + height)
    mask4 = distance >= 2 * width + height

    # Combine the sides using the masks
    result = (
        side1 * mask1[:, jnp.newaxis]
        + side2 * mask2[:, jnp.newaxis]
        + side3 * mask3[:, jnp.newaxis]
        + side4 * mask4[:, jnp.newaxis]
    )

    return result


def arc_curve(t, center=(0.5, 0.5), radius=0.3, start_angle=0, end_angle=jnp.pi):
    """
    Generate points along an arc.

    Parameters:
    t : array-like
        Parameter values from 0 to 1.
    center : tuple
        (x, y) coordinates of the circle's center.
    radius : float
        Radius of the circle.
    start_angle : float
        Starting angle of the arc in radians.
    end_angle : float
        Ending angle of the arc in radians.

    Returns:
    array-like
        Array of (x, y) coordinates along the arc.
    """
    angles = start_angle + (end_angle - start_angle) * t
    x = center[0] + radius * jnp.cos(angles)
    y = center[1] + radius * jnp.sin(angles)
    return jnp.stack([x, y], axis=-1)


def circle_surface(x, y, radius=0.5, center=(0.5, 0.5)):
    """Implicit function for a circle."""
    return (x - center[0]) ** 2 + (y - center[1]) ** 2 - radius**2


def line_surface(x, y, slope=1, intercept=0):
    """Implicit function for a line."""
    return y - (slope * x + intercept)


#################
# 3D curves
#################


# Define 3D initial conditions
def sphere_points(n_points, radius=0.01, center=(0.5, 0.5, 0.5)):
    # Generate evenly distributed points on a sphere
    indices = jnp.arange(0, n_points, dtype=float) + 0.5
    phi = jnp.arccos(1 - 2 * indices / n_points)
    theta = jnp.pi * (1 + 5**0.5) * indices

    x = center[0] + radius * jnp.sin(phi) * jnp.cos(theta)
    y = center[1] + radius * jnp.sin(phi) * jnp.sin(theta)
    z = center[2] + radius * jnp.cos(phi)

    return jnp.stack([x, y, z], axis=-1)
