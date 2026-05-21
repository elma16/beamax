#!/usr/bin/env python
"""
Differentiate through 2D Gaussian beam rays.

This example ports the thesis ray-focusing setup to the public gallery. A
small neural field represents `c(x)`, and autodiff through the Gaussian beam
ray ODE optimizes the medium so a fan of rays focuses at a target point.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from beamax import utils
from beamax.gb import gb_solvers
from beamax.plotter import use_beamax_style


jax.config.update("jax_enable_x64", True)

XMIN, XMAX = -10.0, 10.0
YMIN, YMAX = -10.0, 10.0
EXTENT = [XMIN, XMAX, YMIN, YMAX]


class NeuralC:
    """Small MLP parametrization of the sound-speed field."""

    def __init__(self, hidden_dim: int = 32, base_c: float = 1.0):
        self.base_c = base_c
        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)
        self.params = {
            "w1": 0.1 * jax.random.normal(k1, (2, hidden_dim)),
            "b1": jnp.zeros(hidden_dim),
            "w2": 0.1 * jax.random.normal(k2, (hidden_dim, hidden_dim)),
            "b2": jnp.zeros(hidden_dim),
            "w3": 0.1 * jax.random.normal(k3, (hidden_dim, 1)),
            "b3": jnp.zeros(1),
        }

    def __call__(self, x: jnp.ndarray, params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        x_norm = jnp.stack(
            [
                2.0 * (x[..., 0] - XMIN) / (XMAX - XMIN) - 1.0,
                2.0 * (x[..., 1] - YMIN) / (YMAX - YMIN) - 1.0,
            ],
            axis=-1,
        )
        h = jnp.tanh(x_norm @ params["w1"] + params["b1"])
        h = jnp.tanh(h @ params["w2"] + params["b2"])
        delta_c = 0.3 * jnp.tanh(h @ params["w3"] + params["b3"])[..., 0]
        return self.base_c + delta_c


def ray_setup():
    """Build the source line, upward launches, target focus, and time grid."""
    n_rays = 20
    source_x = jnp.linspace(-5.0, 5.0, n_rays)
    source_y = -7.0
    x0 = jnp.stack([source_x, jnp.full(n_rays, source_y)], axis=-1)
    p0 = jnp.stack([jnp.zeros(n_rays), jnp.ones(n_rays)], axis=-1)
    focus = jnp.array([0.0, 5.0])

    d = 2
    alpha0 = jnp.ones((n_rays, d))
    m0 = 1j * jnp.einsum("bd,dj->bdj", alpha0, jnp.eye(d))
    a0 = jnp.ones((n_rays, 1))
    mode = jnp.ones((n_rays, 1))
    ts = jnp.linspace(0.0, 15.0, 300)
    return x0, p0, m0, a0, mode, ts, focus


def solve_rays(
    params: dict[str, jnp.ndarray],
    param_c: NeuralC,
    x0: jnp.ndarray,
    p0: jnp.ndarray,
    m0: jnp.ndarray,
    a0: jnp.ndarray,
    mode: jnp.ndarray,
    ts: jnp.ndarray,
) -> jnp.ndarray:
    def c_fn(x):
        return param_c(x, params)

    xt, _, _, _ = gb_solvers.solve_ODE_base(x0, p0, m0, a0, mode, ts, c_fn, 0.0, None)
    return xt


def focusing_loss(
    params: dict[str, jnp.ndarray],
    param_c: NeuralC,
    x0: jnp.ndarray,
    p0: jnp.ndarray,
    m0: jnp.ndarray,
    a0: jnp.ndarray,
    mode: jnp.ndarray,
    ts: jnp.ndarray,
    focus: jnp.ndarray,
):
    """Penalize distance to the focus, spread at focus time, and field roughness."""
    xt = solve_rays(params, param_c, x0, p0, m0, a0, mode, ts)
    dist_to_focus = jnp.linalg.norm(xt - focus[None, None, :], axis=-1)
    min_dist = jnp.min(dist_to_focus, axis=1)
    focus_loss = jnp.mean(min_dist)

    mean_dist = jnp.mean(dist_to_focus, axis=0)
    focus_time_idx = jnp.argmin(mean_dist)
    rays_at_focus = xt[:, focus_time_idx, :]
    spread_loss = jnp.mean(jnp.std(rays_at_focus, axis=0))
    smooth_loss = 0.01 * sum(jnp.mean(value**2) for value in params.values())

    total_loss = focus_loss + 0.5 * spread_loss + smooth_loss
    return total_loss, {
        "focus": focus_loss,
        "spread": spread_loss,
        "smooth": smooth_loss,
        "xt": xt,
    }


def tree_global_norm(tree) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in leaves))


def clip_by_global_norm(tree, max_norm: float):
    norm = tree_global_norm(tree)
    scale = jnp.minimum(1.0, max_norm / (norm + 1.0e-12))
    return jax.tree_util.tree_map(lambda leaf: leaf * scale, tree)


def adam_step(params, grads, state, step: int, learning_rate: float):
    """Minimal Adam update so the example does not depend on Optax."""
    beta1 = 0.9
    beta2 = 0.999
    eps = 1.0e-8
    grads = clip_by_global_norm(grads, max_norm=1.0)
    m, v = state
    m = jax.tree_util.tree_map(
        lambda m_leaf, g: beta1 * m_leaf + (1 - beta1) * g, m, grads
    )
    v = jax.tree_util.tree_map(
        lambda v_leaf, g: beta2 * v_leaf + (1 - beta2) * (g**2),
        v,
        grads,
    )
    bias_step = step + 1
    m_hat = jax.tree_util.tree_map(lambda m_leaf: m_leaf / (1 - beta1**bias_step), m)
    v_hat = jax.tree_util.tree_map(lambda v_leaf: v_leaf / (1 - beta2**bias_step), v)
    updates = jax.tree_util.tree_map(
        lambda m_leaf, v_leaf: -learning_rate * m_leaf / (jnp.sqrt(v_leaf) + eps),
        m_hat,
        v_hat,
    )
    params = jax.tree_util.tree_map(lambda p, update: p + update, params, updates)
    return params, (m, v)


def learning_rate(step: int) -> float:
    return 0.02 * (0.9 ** (step / 50.0))


def speed_map(
    param_c: NeuralC,
    params: dict[str, jnp.ndarray],
    nx: int = 200,
    ny: int = 200,
) -> jnp.ndarray:
    xg = jnp.linspace(XMIN, XMAX, nx)
    yg = jnp.linspace(YMIN, YMAX, ny)
    xx, yy = jnp.meshgrid(xg, yg, indexing="xy")
    grid_points = jnp.stack([xx, yy], axis=-1)
    return param_c(grid_points, params)


def plot_rays_before_after(
    out_path,
    c_init_map,
    c_opt_map,
    xt_init,
    xt_final,
    x0,
    focus,
) -> None:
    vmin = float(jnp.minimum(jnp.min(c_init_map), jnp.min(c_opt_map)))
    vmax = float(jnp.maximum(jnp.max(c_init_map), jnp.max(c_opt_map)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    panels = (
        (ax1, c_init_map, xt_init, r"$c_{\mathrm{init}}(\mathbf{x})$"),
        (ax2, c_opt_map, xt_final, r"$c_{\mathrm{opt}}(\mathbf{x})$"),
    )
    for ax, c_map, xt, title in panels:
        im = ax.imshow(
            np.asarray(c_map.T),
            extent=EXTENT,
            origin="lower",
            cmap="viridis",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        for ray in np.asarray(xt):
            ax.plot(ray[:, 0], ray[:, 1], "w-", lw=1.0, alpha=0.5)
        ax.scatter(
            np.asarray(x0[:, 0]),
            np.asarray(x0[:, 1]),
            s=20,
            c="red",
            marker="o",
            label="sources",
        )
        ax.scatter(
            float(focus[0]),
            float(focus[1]),
            s=100,
            c="red",
            marker="*",
            label="focus",
        )
        ax.set_title(title)
        ax.legend(frameon=True, fancybox=True, loc="lower right")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_loss(out_path, loss_history: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_history)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\mathcal{L}$")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_speed_delta(out_path, c_init_map, c_opt_map, focus) -> None:
    diff_map = c_opt_map - c_init_map
    delta_max = float(jnp.max(jnp.abs(diff_map)))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        np.asarray(diff_map.T),
        extent=EXTENT,
        origin="lower",
        cmap="RdBu_r",
        aspect="equal",
        vmin=-delta_max,
        vmax=delta_max,
    )
    ax.scatter(float(focus[0]), float(focus[1]), s=100, c="black", marker="*")
    ax.set_title(r"$\Delta c(\mathbf{x})$")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_dir = utils.example_plot_dir(__file__)
    use_beamax_style()

    x0, p0, m0, a0, mode, ts, focus = ray_setup()
    param_c = NeuralC(hidden_dim=32)
    params = param_c.params

    loss_grad = jax.jit(
        jax.value_and_grad(
            lambda current_params: focusing_loss(
                current_params,
                param_c,
                x0,
                p0,
                m0,
                a0,
                mode,
                ts,
                focus,
            ),
            has_aux=True,
        )
    )

    xt_init = solve_rays(params, param_c, x0, p0, m0, a0, mode, ts)
    m_state = jax.tree_util.tree_map(jnp.zeros_like, params)
    v_state = jax.tree_util.tree_map(jnp.zeros_like, params)
    opt_state = (m_state, v_state)

    loss_history: list[float] = []
    best_loss = float("inf")
    best_params = params
    best_xt = xt_init

    num_iters = 300
    print("Optimizing speed of sound field for ray focusing...")
    for step in range(num_iters):
        (loss_value, aux), grads = loss_grad(params)
        loss_float = float(loss_value)
        params, opt_state = adam_step(
            params,
            grads,
            opt_state,
            step,
            learning_rate(step),
        )
        loss_history.append(loss_float)

        # Mirrors the thesis script: the best ray trajectory is the one used
        # to evaluate the loss, while the displayed medium is after the update.
        if loss_float < best_loss:
            best_loss = loss_float
            best_params = params
            best_xt = aux["xt"]

        if step % 50 == 0:
            print(
                f"iter {step:03d} | loss {loss_float:.6f} | "
                f"focus {float(aux['focus']):.4f} | spread {float(aux['spread']):.4f}"
            )

    c_init_map = speed_map(param_c, param_c.params)
    c_opt_map = speed_map(param_c, best_params)

    rays_path = plot_dir / "focusing_rays_before_after.png"
    loss_path = plot_dir / "focusing_loss_convergence.png"
    delta_path = plot_dir / "focusing_sound_speed_delta.png"

    plot_rays_before_after(
        rays_path,
        c_init_map,
        c_opt_map,
        xt_init,
        best_xt,
        x0,
        focus,
    )
    plot_loss(loss_path, loss_history)
    plot_speed_delta(delta_path, c_init_map, c_opt_map, focus)

    print("Optimization complete.")
    print(f"Final focusing loss: {best_loss:.6f}")
    print(f"Saved figures to {plot_dir.resolve()}")


if __name__ == "__main__":
    main()
