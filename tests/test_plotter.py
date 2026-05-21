import numpy as np
import pytest

from beamax.decomposition import DyadicDecomposition


def _red_highlight_bounds(ax):
    for patch in ax.patches:
        edge_rgba = patch.get_edgecolor()
        if np.allclose(edge_rgba[:3], (1.0, 0.0, 0.0)):
            x0, y0 = patch.get_xy()
            return np.array([x0, y0, patch.get_width(), patch.get_height()])
    raise AssertionError("red highlight rectangle was not drawn")


def test_plot_mswpt_coeffs_box_corners_use_solver_selection():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from beamax import plotter

    dyadic = DyadicDecomposition(
        num_levels=3,
        N=(128, 128),
        num_boxes_levels=(4, 8, 16),
        box_aspect_ratio=(1, 1),
    )
    coeffs = np.ones(dyadic.N)

    fig, ax = plt.subplots()
    try:
        plotter.plot_mswpt_coeffs(
            ax,
            coeffs,
            dyadic,
            box_corners=(16, 75),
            log_scale=True,
        )
        bounds = _red_highlight_bounds(ax)
    finally:
        plt.close(fig)

    # These are opposing level-1 corners. The LF solver receives the
    # geometric set between them, whose plotted support is the central square.
    assert np.allclose(bounds, np.array([-16.0, -16.0, 32.0, 32.0]))


def test_plot_mswpt_coeffs_3d_box_corners_use_solver_selection():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from beamax import plotter

    dyadic = DyadicDecomposition(
        num_levels=2,
        N=(64, 64, 64),
        num_boxes_levels=(4, 8),
        box_aspect_ratio=(1, 1, 1),
    )
    coeffs_mip = np.ones(dyadic.N[:2])

    fig, ax = plt.subplots()
    try:
        plotter.plot_mswpt_coeffs_3d(
            ax,
            coeffs_mip,
            dyadic,
            box_corners=(0, 63),
        )
        bounds = _red_highlight_bounds(ax)
    finally:
        plt.close(fig)

    # Opposing level-0 corners select all level-0 boxes used by the 3D LF
    # solve. The projection is the central square in the coefficient MIP.
    assert np.allclose(bounds, np.array([-8.0, -8.0, 16.0, 16.0]))
