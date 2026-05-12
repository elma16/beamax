#!/usr/bin/env python
"""
Optimize a neural sound-speed field so initially parallel rays focus.

The source line and launch directions stay fixed. A tiny Equinox neural field
represents `c(x)`, and a few explicit gradient steps update the field
parameters through autodiff of the Gaussian-beam ray ODE.
"""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from beamax import utils
from beamax.gb import gb_solvers
from beamax.plotter import use_beamax_style


jax.config.update("jax_enable_x64", True)


class NeuralSpeed(eqx.Module):
    """Radial-basis neural field constrained to positive speeds."""

    weights: jnp.ndarray
    bias: jnp.ndarray
    centers: tuple[tuple[float, float], ...] = eqx.field(static=True)
    sharpness: float = eqx.field(static=True)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        centers = jnp.asarray(self.centers, dtype=x.dtype)
        r2 = jnp.sum((x[..., None, :] - centers) ** 2, axis=-1)
        features = jnp.exp(-self.sharpness * r2)
        raw = jnp.sum(features * self.weights, axis=-1) + self.bias
        return 1.0 + 0.25 * jnp.tanh(raw)


def initial_model() -> NeuralSpeed:
    return NeuralSpeed(
        weights=jnp.zeros((5,)),
        bias=jnp.array(0.0),
        centers=((0.40, 0.25), (0.60, 0.25), (0.50, 0.38), (0.42, 0.52), (0.58, 0.52)),
        sharpness=30.0,
    )


def ray_setup():
    n_rays = 30
    x0 = jnp.stack(
        [jnp.linspace(0.35, 0.65, n_rays), jnp.full((n_rays,), 0.12)],
        axis=-1,
    )
    p0 = jnp.tile(jnp.array([0.0, 1.0]), (n_rays, 1))
    m0 = 1j * jnp.eye(2)[None, :, :].repeat(n_rays, axis=0)
    a0 = jnp.ones((n_rays,))
    mode = jnp.ones((n_rays,))
    ts = jnp.linspace(0.0, 0.62, 24)
    focus = jnp.array([0.50, 0.68])
    return x0, p0, m0, a0, mode, ts, focus


def solve_rays(model: NeuralSpeed, x0, p0, m0, a0, mode, ts) -> jnp.ndarray:
    solver_config = gb_solvers.SolverConfig(rtol=3e-3, atol=1e-5, max_steps=1024)
    xt, _, _, _ = gb_solvers.solve_ODE_base(
        x0, p0, m0, a0, mode, ts, model, 0.0, solver_config
    )
    return xt


def focusing_loss(model: NeuralSpeed, x0, p0, m0, a0, mode, ts, focus):
    xt = solve_rays(model, x0, p0, m0, a0, mode, ts)
    distance = jnp.linalg.norm(xt - focus[None, None, :], axis=-1)
    min_distance = jnp.min(distance, axis=1)
    focus_time = jnp.argmin(jnp.mean(distance, axis=0))
    spread = jnp.mean(jnp.std(xt[:, focus_time, :], axis=0))
    smoothness = 1e-3 * jnp.sum(model.weights**2)
    return jnp.mean(min_distance) + 0.6 * spread + smoothness


def clipped_gradient_update(grads: NeuralSpeed, step_size: float, max_norm: float):
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(grads) if leaf is not None]
    global_norm = jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in leaves))
    scale = step_size * jnp.minimum(1.0, max_norm / (global_norm + 1e-12))
    return jax.tree_util.tree_map(
        lambda grad: -scale * grad if grad is not None else None,
        grads,
    )


def speed_map(model: NeuralSpeed, grid_n: int = 100):
    x = jnp.linspace(0.0, 1.0, grid_n)
    y = jnp.linspace(0.0, 1.0, grid_n)
    xy = jnp.stack(jnp.meshgrid(x, y, indexing="ij"), axis=-1)
    return model(xy)


def plot_panel(ax, c_map, xt, x0, focus, title, vmin, vmax):
    im = ax.imshow(
        np.asarray(c_map.T),
        extent=[0.0, 1.0, 0.0, 1.0],
        origin="lower",
        cmap="viridis",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    for ray in np.asarray(xt):
        ax.plot(ray[:, 0], ray[:, 1], color="white", lw=1.2, alpha=0.75)
    ax.scatter(np.asarray(x0[:, 0]), np.asarray(x0[:, 1]), s=28, color="#d7301f", edgecolor="black", zorder=3)
    ax.scatter(float(focus[0]), float(focus[1]), s=110, marker="*", color="#d7301f", edgecolor="black", zorder=4)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def main() -> None:
    root_dir = utils.detect_root()
    plot_dir = Path(root_dir) / "plots"
    plot_dir.mkdir(exist_ok=True)
    use_beamax_style()

    x0, p0, m0, a0, mode, ts, focus = ray_setup()
    model = initial_model()
    initial_xt = solve_rays(model, x0, p0, m0, a0, mode, ts)
    initial_loss = float(focusing_loss(model, x0, p0, m0, a0, mode, ts, focus))

    step_size = 0.8
    max_grad_norm = 1.0
    loss_history = [initial_loss]

    for _ in range(8):
        loss_value, grads = eqx.filter_value_and_grad(focusing_loss)(
            model, x0, p0, m0, a0, mode, ts, focus
        )
        updates = clipped_gradient_update(grads, step_size, max_grad_norm)
        model = eqx.apply_updates(model, updates)
        loss_history.append(float(loss_value))

    final_xt = solve_rays(model, x0, p0, m0, a0, mode, ts)
    final_loss = float(focusing_loss(model, x0, p0, m0, a0, mode, ts, focus))
    initial_c = speed_map(initial_model())
    final_c = speed_map(model)
    vmin = float(jnp.minimum(initial_c.min(), final_c.min()))
    vmax = float(jnp.maximum(initial_c.max(), final_c.max()))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    plot_panel(axes[0], initial_c, initial_xt, x0, focus, "initial c(x)", vmin, vmax)
    im = plot_panel(axes[1], final_c, final_xt, x0, focus, "optimized c(x)", vmin, vmax)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="c(x)", shrink=0.88)

    out_path = plot_dir / "neural_sound_speed_autofocus.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Optimization steps: {len(loss_history) - 1}")
    print(f"Focusing loss: {initial_loss:.4e} -> {final_loss:.4e}")
    print(f"Final speed range: [{float(final_c.min()):.3f}, {float(final_c.max()):.3f}]")
    print(f"Saved neural-speed autofocus plot to {out_path}")


if __name__ == "__main__":
    main()
