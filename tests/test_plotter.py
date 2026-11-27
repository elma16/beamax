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


def test_plotter_public_exports():
    pytest.importorskip("matplotlib")
    from beamax import plotter

    assert plotter.__all__ == [
        "plot_forward_records",
        "plot_mswpt_coeffs",
        "plot_mswpt_coeffs_3d",
        "plot_packet_box_scanner",
        "save_comparison_figure",
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


def test_plot_mswpt_coeffs_array_order_matches_imshow_orientation():
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
    coeffs = np.zeros((512, 256))
    coeffs[401, 37] = 3.0

    fig, ax = plt.subplots()
    try:
        image = plotter.plot_mswpt_coeffs(
            ax,
            coeffs,
            dyadic,
            display_order="array",
        )
        assert image.get_array().shape == coeffs.shape
        assert np.unravel_index(np.argmax(image.get_array()), coeffs.shape) == (401, 37)
        assert np.allclose(image.get_extent(), (-64.0, 64.0, -128.0, 128.0))
    finally:
        plt.close(fig)


def test_plot_mswpt_coeffs_array_order_swaps_rectangular_box_geometry():
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
        plotter.plot_mswpt_coeffs(
            ax,
            coeffs,
            dyadic,
            box_corners=(16, 75),
            display_order="array",
        )
        bounds = _red_highlight_bounds(ax)
    finally:
        plt.close(fig)

    # In array order, decomposition axis 1 is horizontal and axis 0 vertical.
    assert np.allclose(bounds, np.array([-16.0, -32.0, 32.0, 64.0]))


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


@pytest.mark.parametrize("array_order", [False, True], ids=["default", "array"])
def test_plot_mswpt_coeffs_3d_rectangular_projection(array_order):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from beamax import plotter

    dyadic = DyadicDecomposition(
        num_levels=2,
        N=(64, 32, 32),
        num_boxes_levels=(4, 8),
        box_aspect_ratio=(2, 1, 1),
    )
    coeffs_mip = np.zeros((128, 64))
    coeffs_mip[101, 13] = 2.0

    if array_order:
        plot_kwargs = {"display_order": "array"}
        expected_image = coeffs_mip
        expected_extent = (-16.0, 16.0, -32.0, 32.0)
        expected_highlight = (-4.0, -8.0, 8.0, 16.0)
    else:
        plot_kwargs = {}
        expected_image = coeffs_mip.T
        expected_extent = (-32.0, 32.0, -16.0, 16.0)
        expected_highlight = (-8.0, -4.0, 16.0, 8.0)

    fig, ax = plt.subplots()
    try:
        image = plotter.plot_mswpt_coeffs_3d(
            ax,
            coeffs_mip,
            dyadic,
            box_corners=(0, 63),
            **plot_kwargs,
        )
        np.testing.assert_allclose(image.get_array(), np.maximum(expected_image, 0.02))
        np.testing.assert_allclose(image.get_extent(), expected_extent)
        np.testing.assert_allclose(_red_highlight_bounds(ax), expected_highlight)
    finally:
        plt.close(fig)


def test_forward_thesis_residuals_share_scale_and_receiver(tmp_path):
    plt = pytest.importorskip("matplotlib.pyplot")
    from beamax.plotter import plot_forward_records, save_comparison_figure

    kw = np.arange(60.0).reshape(5, 12)
    ms, hybrid = kw.copy(), kw.copy()
    ms[:, 0] += 30  # Excluded edge.
    ms[:, 5] += 3
    hybrid[:, 7] -= 5  # The second residual chooses the shared profile.
    ts = np.arange(5) * 0.01
    fig = plot_forward_records(kw, ms, hybrid, ts)
    try:
        panels = [ax for ax in fig.axes if ax.images]
        assert len(panels) == 3
        # The tight-cropped Chapter 4 plate and its panel borders, in the
        # original 300 dpi export. A default tight_layout moves these axes.
        width, height = fig.get_size_inches() * 300
        np.testing.assert_allclose((width, height), (1560, 1222))
        for ax, left in zip(panels, (214, 616, 1018), strict=True):
            bounds = ax.get_position()
            np.testing.assert_allclose(
                (bounds.x0 * width, (1 - bounds.y1) * height),
                (left, 73),
                atol=1,
            )
        png = tmp_path / "plots" / "forward.png"
        save_comparison_figure(fig, png, fixed_canvas=True)
        assert plt.imread(png).shape[:2] == (1222, 1560)
        assert plt.fignum_exists(fig.number)
        assert panels[1].images[0].norm is panels[2].images[0].norm
        assert panels[0].images[0].get_clim() == (
            min(a.min() for a in (kw, ms, hybrid)),
            max(a.max() for a in (kw, ms, hybrid)),
        )
        for ax, field in zip(panels, (kw, kw - ms, kw - hybrid), strict=True):
            np.testing.assert_array_equal(ax.images[0].get_array(), field)
            np.testing.assert_allclose(ax.lines[0].get_xdata(), 7 / 11)
            np.testing.assert_allclose(ax.images[0].get_extent(), (0, 1, ts[-1], ts[0]))
        assert panels[2].images[0].get_clim() == (-30, 30)
        for line, field in zip(fig.axes[-1].lines, (kw, ms, hybrid), strict=True):
            np.testing.assert_array_equal(line.get_ydata(), field[:, 7])
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "operations", [("time_reversal",), ("adjoint",), ("time_reversal", "adjoint")]
)
def test_box_scanner_excerpt_preserves_thesis_calibration(operations):
    plt = pytest.importorskip("matplotlib.pyplot")
    from beamax.plotter import plot_packet_box_scanner

    p0 = np.random.default_rng(5).normal(size=(128, 128))
    mask = np.zeros_like(p0, dtype=bool)
    mask[0] = True
    fields = {op: (p0 / 5, p0 / 7) for op in operations}
    fig = plot_packet_box_scanner(p0, mask, (0.0128, 0.0128), **fields)
    try:
        panels = [ax for ax in fig.axes if ax.images]
        expected = [p0]
        for op in operations:
            for field in fields[op]:
                expected.append(field * (2296.5277637882564 if op == "adjoint" else 1))
        for i, (ax, field) in enumerate(zip(panels, expected, strict=True)):
            im = ax.images[0]
            np.testing.assert_allclose(im.get_array(), field)
            assert im.norm is panels[0].images[0].norm
            assert im.get_clim() == (-1.0806082602212956, 1.1509535463142675)
            np.testing.assert_allclose(ax.lines[0].get_xdata(), 87 * 0.0128 / 127)
            if i:
                assert ax.collections[0].get_offsets().shape == (128, 2)
        for i, ax in enumerate(fig.axes[-len(operations) :]):
            for line, field in zip(
                ax.lines, [p0, *expected[2 * i + 1 : 2 * i + 3]], strict=True
            ):
                np.testing.assert_allclose(line.get_ydata(), field[:, 87])
    finally:
        plt.close(fig)
