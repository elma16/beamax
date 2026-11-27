"""Render the 6.6-cycle Beamax bee with the viz-mpl extra installed."""

import argparse
from io import StringIO
from pathlib import Path

import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D

from beamax.gb import compute_gaussian_beam, solve_hom_diag


INK = "#25231F"
HONEY = "#F6B51B"
WING = [
    (0.60, 0.00),
    (0.47, 0.16),
    (0.24, 0.54),
    (0.30, 0.83),
    (0.36, 1.00),
    (0.68, 0.90),
    (0.76, 0.62),
    (0.81, 0.39),
    (0.72, 0.14),
    (0.60, 0.00),
]


def waveform():
    """Sample the real field of one Gaussian beam at launch."""
    x = np.linspace(-1, 1, 301)
    omega = 2 * np.pi * 3.3  # 6.6 cycles across the two-unit interval
    field = compute_gaussian_beam(
        x0=jnp.array([[0.0]]),
        p0=jnp.array([[1.0]]),
        M0=jnp.array([[[5.6j / omega]]]),
        a0=jnp.array([1.0]),
        omega0=jnp.array([omega]),
        mode=jnp.array([1.0]),
        c=lambda position: jnp.ones(position.shape[:-1]),
        lam=0.0,
        ts=jnp.array([0.0]),
        sensors=jnp.asarray(x[:, None]),
        domain_size=jnp.array([2.0]),
        periodic=jnp.array([False]),
        ode_solver=solve_hom_diag,
    )
    trace = np.asarray(jnp.real(field[0, :, 0]))
    return MplPath(np.column_stack((-0.13 + 0.92 * x, 0.50 * trace)))


@mpl.rc_context({"path.simplify": False, "svg.hashsalt": "beamax-bee-studies"})
def render(destination):
    fig = Figure(figsize=(5.12, 5.12), dpi=100, facecolor="none")
    ax = fig.add_axes((0, 0, 1, 1), xlim=(0, 512), ylim=(0, 512), aspect="equal")
    ax.set_axis_off()
    rotation = Affine2D().rotate_deg(32)
    parts = []

    def add(path, stroke, width, fill="none"):
        parts.append((rotation.transform_path(path), stroke, width, fill))

    codes = [MplPath.MOVETO] + [MplPath.CURVE4] * 9 + [MplPath.CLOSEPOLY]
    for side in (-1, 1):
        vertices = np.asarray([*WING, WING[0]]) * (1, side)
        add(MplPath(vertices, codes), INK, 0.055)
    trace = waveform()
    add(trace, HONEY, 0.185)
    add(trace, INK, 0.070)
    add(MplPath([(0.89, 0.11), (1.28, 0.43)]), INK, 0.073)
    add(MplPath([(0.94, -0.04), (1.43, -0.20)]), INK, 0.073)
    head = Affine2D().scale(0.172, 0.158).translate(0.86, 0.037)
    add(head.transform_path(MplPath.unit_circle()), "none", 0, INK)

    bounds = np.array(
        [
            path.get_extents().extents + np.array([-1, -1, 1, 1]) * width / 2
            for path, _, width, _ in parts
        ]
    )
    lo, hi = bounds[:, :2].min(axis=0), bounds[:, 2:].max(axis=0)
    scale = 460 / max(hi - lo)
    placement = Affine2D().translate(*(-(lo + hi) / 2)).scale(scale).translate(256, 256)
    for path, stroke, width, fill in parts:
        ax.add_patch(
            PathPatch(
                placement.transform_path(path),
                facecolor=fill,
                edgecolor=stroke,
                lw=width * scale * 72 / fig.dpi,
                capstyle="round",
                joinstyle="round",
            )
        )

    destination.mkdir(parents=True, exist_ok=True)
    svg = StringIO()
    fig.savefig(svg, format="svg", transparent=True, metadata={"Date": None})
    (destination / "beamax-logo.svg").write_text(
        "".join(f"{line.rstrip()}\n" for line in svg.getvalue().splitlines()),
        encoding="utf-8",
    )
    # The pale tile keeps the thin charcoal strokes visible on coloured headers.
    ax.add_patch(
        FancyBboxPatch(
            (0, 0),
            512,
            512,
            boxstyle="round,pad=0,rounding_size=68",
            linewidth=0,
            facecolor="#FAF9F5",
            zorder=-1,
        )
    )
    fig.savefig(destination / "beamax-logo-tile.png", transparent=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/assets",
    )
    render(parser.parse_args().output)
