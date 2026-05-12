#!/usr/bin/env python
"""
Trace a small fan of 2D rays through a smooth speed field.

The Gaussian-beam ray equations bend trajectories toward gradients in the
Hamiltonian `G(x, p) = c(x) |p|`. This example solves those ODEs for a few
parallel rays, overlays the paths on the speed map, and reports compact
diagnostics for the amount of bending.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from beamax import utils
from beamax.gb import gb_solvers
from beamax.plotter import use_beamax_style


jax.config.update("jax_enable_x64", True)


def speed_field(x: jnp.ndarray) -> jnp.ndarray:
    """Smooth 2D sound-speed map used by the ray example."""
    lens_center = jnp.array([0.46, 0.52])
    lens = jnp.exp(-35.0 * jnp.sum((x - lens_center) ** 2, axis=-1))
    vertical_gradient = 0.18 * (x[..., 1] - 0.5)
    return 1.05 + vertical_gradient - 0.28 * lens


def solve_rays() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve a compact bundle of initially parallel rays."""
    n_rays = 60
    ts = jnp.linspace(0.0, 0.75, 48)
    y0 = jnp.linspace(0.18, 0.82, n_rays)
    x0 = jnp.stack([jnp.full((n_rays,), 0.08), y0], axis=-1)
    p0 = jnp.tile(jnp.array([1.0, 0.0]), (n_rays, 1))
    m0 = 1j * jnp.eye(2)[None, :, :].repeat(n_rays, axis=0)
    a0 = jnp.ones((n_rays,))
    mode = jnp.ones((n_rays,))

    solver_config = gb_solvers.SolverConfig(
        rtol=1e-4,
        atol=1e-6,
        max_steps=1024,
    )
    xt, pt, _, _ = gb_solvers.solve_ODE_base(
        x0,
        p0,
        m0,
        a0,
        mode,
        ts,
        speed_field,
        0.0,
        solver_config,
    )
    return xt, pt, x0, ts


def main() -> None:
    root_dir = utils.detect_root()
    plot_dir = Path(root_dir) / "plots"
    plot_dir.mkdir(exist_ok=True)
    use_beamax_style()

    xt, pt, x0, ts = solve_rays()
    final = xt[:, -1, :]
    direction_angles = jnp.arctan2(pt[:, -1, 1], pt[:, -1, 0])
    initial_angles = jnp.arctan2(pt[:, 0, 1], pt[:, 0, 0])

    grid_n = 128
    x = jnp.linspace(0.0, 1.0, grid_n)
    y = jnp.linspace(0.0, 1.0, grid_n)
    xy = jnp.stack(jnp.meshgrid(x, y, indexing="ij"), axis=-1)
    c_values = speed_field(xy)

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    im = ax.imshow(
        np.asarray(c_values.T),
        extent=[0.0, 1.0, 0.0, 1.0],
        origin="lower",
        cmap="viridis",
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, label="c(x)")

    ray_color = "#d94801"
    for ray in np.asarray(xt):
        ax.plot(ray[:, 0], ray[:, 1], color=ray_color, lw=1.6, alpha=0.88)
    ax.scatter(np.asarray(x0[:, 0]), np.asarray(x0[:, 1]), s=26, color="white", edgecolor="black", zorder=3)
    ax.scatter(np.asarray(final[:, 0]), np.asarray(final[:, 1]), s=24, color=ray_color, edgecolor="black", zorder=3)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D ray bending in a smooth speed field")
    fig.tight_layout()

    out_path = plot_dir / "2d_ray_bending.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    mean_lateral_shift = float(jnp.mean(jnp.abs(final[:, 1] - x0[:, 1])))
    max_angle_change = float(jnp.max(jnp.abs(direction_angles - initial_angles)))
    print(f"Rays solved: {xt.shape[0]}, time samples: {ts.shape[0]}")
    print(f"Speed range on plot grid: [{float(c_values.min()):.3f}, {float(c_values.max()):.3f}]")
    print(f"Mean lateral displacement: {mean_lateral_shift:.3f}")
    print(f"Max direction change: {max_angle_change:.3f} rad")
    print(f"Saved ray-bending plot to {out_path}")


if __name__ == "__main__":
    main()
