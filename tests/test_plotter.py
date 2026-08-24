import subprocess
import sys

import numpy as np
import pytest

from beamax.decomposition import DyadicDecomposition


def test_plotter_import_does_not_load_solver_modules():
    pytest.importorskip("matplotlib")
    code = """
import sys
import beamax.plotter

loaded = [
    name for name in sys.modules
    if name.startswith(("beamax.solvers", "beamax.gb.pallas"))
]
if loaded:
    raise AssertionError(loaded)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_plotter_public_surface_is_minimal():
    pytest.importorskip("matplotlib")
    from beamax import plotter

    assert plotter.__all__ == [
        "plot_mswpt_coeffs",
        "plot_mswpt_coeffs_3d",
        "use_beamax_style",
    ]


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


def test_plot_mswpt_coeffs_rectangular_normalized_axes_follow_transpose():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from beamax import plotter

    dyadic = DyadicDecomposition(
        num_levels=3,
        N=(256, 128),
        num_boxes_levels=(4, 8, 16),
        box_aspect_ratio=(2, 1),
    )
    coeffs = np.ones((512, 256))

    fig, ax = plt.subplots()
    try:
        image = plotter.plot_mswpt_coeffs(
            ax,
            coeffs,
            dyadic,
            asymptote=True,
            asymptote_slope=1.25,
            normalized_frequency_axes=True,
        )
        assert np.allclose(image.get_extent(), (-0.5, 0.5, -0.5, 0.5))
        assert image.get_array().shape == coeffs.T.shape
        positive_cone = ax.lines[0]
        assert np.allclose(positive_cone.get_ydata(), 1.25 * positive_cone.get_xdata())
    finally:
        plt.close(fig)


def test_plot_mswpt_coeffs_rectangular_bin_axes_preserve_sample_counts():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from beamax import plotter

    dyadic = DyadicDecomposition(
        num_levels=3,
        N=(256, 128),
        num_boxes_levels=(4, 8, 16),
        box_aspect_ratio=(2, 1),
    )
    coeffs = np.ones((512, 256))

    fig, ax = plt.subplots()
    try:
        image = plotter.plot_mswpt_coeffs(
            ax,
            coeffs,
            dyadic,
            asymptote=True,
            asymptote_slope=0.625,
        )
        assert np.allclose(image.get_extent(), (-128.0, 128.0, -64.0, 64.0))
        positive_cone = ax.lines[0]
        assert np.allclose(positive_cone.get_ydata(), 0.625 * positive_cone.get_xdata())
    finally:
        plt.close(fig)


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
