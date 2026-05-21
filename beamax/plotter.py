"""
Plotting helpers for beamax workflows (matplotlib/pyvista).

This module collects visualization utilities built on the optional `viz`
dependencies: matplotlib for standard figures and pyvista for selected 3D
views. Most functions accept NumPy/JAX arrays and are intended for diagnostics
rather than production rendering.
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from beamax.decomposition import DyadicDecomposition
from beamax.solvers import hybrid_solver_utils

# since pyvista is only used for 3d stuff, it's not essential
try:
    import pyvista as pv
except ImportError:
    pass

from beamax import utils
from beamax.geometry import Domain


def use_beamax_style() -> None:
    """
    Apply a lightweight matplotlib style for example figures.

    Notes
    -----
    This function updates a small set of ``rcParams`` directly so examples
    remain portable in editable checkouts, wheels, and notebooks.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": "medium",
            "axes.labelsize": "medium",
            "legend.frameon": False,
            "image.cmap": "viridis",
        }
    )


class PlotHelper:
    """
    Convenience wrapper around common plotting patterns.

    Parameters
    ----------
    figsize : Tuple[int, int], default=(12, 5)
        Figure size passed to matplotlib.
    font_size : int, default=12
        Base font size for axes labels.
    title_font_size : int, default=14
        Title font size.
    colormap : str, default="viridis"
        Name of a matplotlib colormap to use for heatmaps/surfaces.
    """

    def __init__(
        self, figsize=(12, 5), font_size=12, title_font_size=14, colormap="viridis"
    ):
        """
        Store global style defaults for plots.

        Parameters
        ----------
        figsize : Tuple[int, int], default=(12, 5)
            Figure size passed to matplotlib.
        font_size : int, default=12
            Base font size for labels.
        title_font_size : int, default=14
            Title font size.
        colormap : str, default="viridis"
            Matplotlib colormap name.
        """
        self.figsize = figsize
        self.font_size = font_size
        self.title_font_size = title_font_size
        self.colormap = colormap
        self.pv_background_color = "white"
        self.pv_colormap = colormap

    def setup_figure(self, title=None, projection=None):
        """
        Create a figure/axes pair with optional projection and title.

        Parameters
        ----------
        title : str | None
        projection : str | None
            e.g. ``"3d"`` for surface plots.

        Returns
        -------
        (matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        fig = plt.figure(figsize=self.figsize)
        ax = (
            fig.add_subplot(111, projection=projection)
            if projection
            else fig.add_subplot(111)
        )
        if title:
            ax.set_title(title, fontsize=self.title_font_size)
        return fig, ax

    def set_labels(self, ax, xlabel=None, ylabel=None, zlabel=None):
        """
        Apply axis labels if provided.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes.
        xlabel : str, optional
            X-axis label.
        ylabel : str, optional
            Y-axis label.
        zlabel : str, optional
            Z-axis label for 3D axes.
        """
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.font_size)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.font_size)
        if zlabel and hasattr(ax, "set_zlabel"):
            ax.set_zlabel(zlabel, fontsize=self.font_size)

    def save_plot(self, filename):
        """
        Save the current matplotlib figure and close it.

        Parameters
        ----------
        filename : str or path-like
            Output path passed to :func:`matplotlib.pyplot.savefig`.
        """
        plt.tight_layout()
        plt.savefig(filename, transparent=True)
        plt.close()

    def _finalize_plot(self, fig, filename):
        """
        Save or display a figure depending on ``filename``.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to finalize.
        filename : str or path-like, optional
            Output path. If ``None``, the figure is shown interactively.
        """
        if filename:
            self.save_plot(filename)
        else:
            plt.show()

    # ------------------------------------------------------------------
    # Coefficient heatmaps with dyadic box overlays
    # ------------------------------------------------------------------
    def plot_coeffs_with_boxes(
        self,
        coeffs,
        dyadic_decomp,
        *,
        ax=None,
        plane: str = "xy",
        slice_index: int | None = None,
        cmap: str | None = None,
        title: str | None = None,
        show_colorbar: bool = True,
        extent=None,
    ):
        """
        Plot transform coefficients with dyadic box overlays (1D/2D/3D).

        Parameters
        ----------
        coeffs : array-like
            Coefficients arranged on a regular grid (1D/2D/3D).
        dyadic_decomp : beamax.decomposition.DyadicDecomposition
            Decomposition whose boxes will be overlaid.
        ax : matplotlib Axes | None
            Axes to draw on; if None a new figure/axes is created.
        plane : {"xy","xz","yz"}, default "xy"
            Projection plane for 3D coefficients. Ignored for 1D/2D.
        slice_index : int | None
            Slice to show for 3D; defaults to central slice along the
            axis orthogonal to `plane`.
        cmap : str | None
            Matplotlib colormap name.
        title : str | None
            Title for the axes.
        show_colorbar : bool, default True
            Whether to attach a colorbar (if a new figure is created).
        extent : tuple | None
            Optional imshow extent to use for 2D/3D projections.

        Returns
        -------
        (fig, ax)
        """
        arr = jnp.asarray(coeffs)
        d = arr.ndim
        cmap = cmap or self.colormap

        if ax is None:
            fig, ax = self.setup_figure(title=title)
        else:
            fig = ax.figure
            if title:
                ax.set_title(title, fontsize=self.title_font_size)

        if d == 1:
            x = jnp.arange(arr.shape[0])
            ax.plot(x, jnp.abs(arr), lw=1.5)
            self._draw_boxes_1d(ax, dyadic_decomp, color="tab:gray")
            self.set_labels(ax, xlabel="index", ylabel="|coeff|")
        elif d in (2, 3):
            view = self._coeff_slice(arr, plane=plane, slice_index=slice_index)
            extent = extent or self._default_extent(
                dyadic_decomp, plane=plane, arr_shape=arr.shape
            )
            im = ax.imshow(
                view,
                origin="lower",
                cmap=cmap,
                interpolation="nearest",
                extent=extent,
            )
            self._draw_boxes_projected(ax, dyadic_decomp, plane=plane)
            ax.set_xticks([])
            ax.set_yticks([])
            if title:
                ax.set_title(title, fontsize=self.title_font_size)
            if show_colorbar and fig is not None:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        else:
            raise ValueError(f"Unsupported coefficient ndim={d}")

        return fig, ax

    def _coeff_slice(self, arr, plane="xy", slice_index=None):
        """
        Return a 2D view from 2D or 3D coefficient arrays.

        Parameters
        ----------
        arr : jnp.ndarray
            Coefficient array.
        plane : {"xy", "xz", "yz"}, default="xy"
            Projection plane for 3D arrays.
        slice_index : int, optional
            Slice index orthogonal to ``plane``.

        Returns
        -------
        jnp.ndarray
            Absolute-value 2D coefficient view.

        Raises
        ------
        ValueError
            If ``arr`` is not 2D/3D or ``plane`` is invalid.
        """
        if arr.ndim == 2:
            return jnp.abs(arr)

        if arr.ndim != 3:
            raise ValueError("Only 2D or 3D coefficient arrays are supported.")

        plane = plane.lower()
        if plane not in {"xy", "xz", "yz"}:
            raise ValueError("plane must be one of {'xy','xz','yz'}")

        if plane == "xy":
            sl = slice_index if slice_index is not None else arr.shape[2] // 2
            return jnp.abs(arr[:, :, sl])
        if plane == "xz":
            sl = slice_index if slice_index is not None else arr.shape[1] // 2
            return jnp.abs(arr[:, sl, :])
        # plane == "yz"
        sl = slice_index if slice_index is not None else arr.shape[0] // 2
        return jnp.abs(arr[sl, :, :])

    def _default_extent(self, dyadic_decomp, plane="xy", arr_shape=None):
        """
        Compute a symmetric extent centered at zero for a projection plane.

        Parameters
        ----------
        dyadic_decomp : DyadicDecomposition
            Decomposition providing default grid shape.
        plane : {"xy", "xz", "yz"}, default="xy"
            Projection plane.
        arr_shape : Tuple[int, ...], optional
            Shape to use instead of ``dyadic_decomp.N``.

        Returns
        -------
        list[int]
            Matplotlib ``imshow`` extent.

        Raises
        ------
        ValueError
            If ``plane`` is invalid.
        """
        if arr_shape is None:
            arr_shape = dyadic_decomp.N
        dshape = tuple(int(s) for s in arr_shape)
        plane = plane.lower()
        if plane not in {"xy", "xz", "yz"}:
            raise ValueError("plane must be one of {'xy','xz','yz'}")
        if plane == "xy":
            nx, ny = dshape[0], dshape[1]
            return [-nx // 2, nx // 2, -ny // 2, ny // 2]
        if plane == "xz":
            nx, nz = dshape[0], dshape[2]
            return [-nx // 2, nx // 2, -nz // 2, nz // 2]
        ny, nz = dshape[1], dshape[2]
        return [-ny // 2, ny // 2, -nz // 2, nz // 2]

    def _draw_boxes_1d(self, ax, dyadic_decomp, color="gray", alpha=0.4):
        """
        Draw 1D dyadic boxes as shaded spans.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes.
        dyadic_decomp : DyadicDecomposition
            Decomposition providing centres and box lengths.
        color : str, default="gray"
            Span color.
        alpha : float, default=0.4
            Span opacity.
        """
        cumsum_boxes = jnp.r_[0, jnp.cumsum(dyadic_decomp.num_boxes_ndim)]
        lengths = dyadic_decomp.box_lengths
        aspect = jnp.array(dyadic_decomp.box_aspect_ratio)
        for level in range(dyadic_decomp.num_levels):
            start = int(cumsum_boxes[level])
            end = int(cumsum_boxes[level + 1])
            centers = dyadic_decomp.centres_ndim[start:end]
            width = float(lengths[level] * aspect[0])
            for center in centers:
                x0 = float(center[0] - width / 2)
                x1 = float(center[0] + width / 2)
                ax.axvspan(x0, x1, color=color, alpha=alpha, linewidth=0)

    def _draw_boxes_projected(
        self,
        ax,
        dyadic_decomp,
        *,
        plane="xy",
        alpha=0.5,
        linewidth=1.0,
        colors=None,
    ):
        """
        Overlay dyadic boxes projected onto a 2D plane.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes.
        dyadic_decomp : DyadicDecomposition
            Decomposition providing centres and box sizes.
        plane : {"xy", "xz", "yz"}, default="xy"
            Projection plane.
        alpha : float, default=0.5
            Rectangle opacity.
        linewidth : float, default=1.0
            Rectangle line width.
        colors : list[str], optional
            Per-level colors. Defaults to white.

        Raises
        ------
        ValueError
            If ``plane`` is invalid.
        """
        if colors is None:
            colors = ["white"] * max(1, dyadic_decomp.num_levels)
        plane = plane.lower()
        if plane not in {"xy", "xz", "yz"}:
            raise ValueError("plane must be one of {'xy','xz','yz'}")

        cumsum_boxes = jnp.r_[0, jnp.cumsum(dyadic_decomp.num_boxes_ndim)]
        lengths = dyadic_decomp.box_lengths
        aspect = jnp.array(dyadic_decomp.box_aspect_ratio)

        if plane == "xy":
            axes = (0, 1)
        elif plane == "xz":
            axes = (0, 2)
        else:  # yz
            axes = (1, 2)

        for level in range(dyadic_decomp.num_levels):
            start = int(cumsum_boxes[level])
            end = int(cumsum_boxes[level + 1])
            centers = dyadic_decomp.centres_ndim[start:end]
            box_length = float(lengths[level])

            # widths along the projected axes, mimicking legacy integer-centered boxes
            width_a = (
                box_length * float(aspect[axes[0]])
                if axes[0] < len(aspect)
                else box_length
            )
            width_b = (
                box_length * float(aspect[axes[1]])
                if axes[1] < len(aspect)
                else box_length
            )

            color = colors[level % len(colors)]
            for center in centers:
                cx = float(center[axes[0]])
                cy = float(center[axes[1]])

                x_min = cx - width_a // 2
                x_max = cx + width_a // 2
                y_min = cy - width_b // 2
                y_max = cy + width_b // 2

                rect = Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=linewidth,
                    edgecolor=color,
                    facecolor="none",
                    linestyle=":",
                    alpha=alpha,
                )
                ax.add_patch(rect)

    def plot_wavefield(self, data, complex=False, *args, **kwargs):
        """
        Dispatch to the appropriate real/complex wavefield plotter.

        Parameters
        ----------
        data : jnp.ndarray
            Wavefield with shape `(Nx, ...)` (1D/2D/3D).
        complex : bool, default=False
            If True, splits into real/imaginary panels.
        *args, **kwargs :
            Passed through to the underlying plotting helper
            (e.g. titles, axes labels, sensor markers).
        """
        ndim = data.ndim
        if complex:
            if ndim == 1:
                self._plot_complex_wavefield_1d(data, *args, **kwargs)
            elif ndim == 2:
                self._plot_complex_wavefield_2d(data, *args, **kwargs)
            elif ndim == 3:
                self._plot_complex_wavefield_3d(data, *args, **kwargs)
        elif not complex:
            if ndim == 1:
                self._plot_wavefield_1d(data, *args, **kwargs)
            elif ndim == 2:
                self._plot_wavefield_2d(data, *args, **kwargs)
            elif ndim == 3:
                self._plot_wavefield_3d(data, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported data dimension: {ndim}")

    def _plot_wavefield_1d(
        self,
        Y,
        X=None,
        title=None,
        xlabel="X-axis",
        ylabel="Y-axis",
        filename=None,
        sensors=None,
    ):
        """
        Plot a one-dimensional wavefield trace.

        Parameters
        ----------
        Y : jnp.ndarray, shape (Nx,)
            Values to plot.
        X : jnp.ndarray, optional
            Coordinates for ``Y``. Defaults to integer indices.
        title : str, optional
            Figure title.
        xlabel : str, default="X-axis"
            X-axis label.
        ylabel : str, default="Y-axis"
            Y-axis label.
        filename : str, optional
            Output path. If ``None``, the figure is shown.
        sensors : jnp.ndarray, optional
            Sensor locations to overlay.
        """
        fig, ax = self.setup_figure(title=title)
        if X is None:
            X = jnp.arange(len(Y))
        ax.plot(X, Y)
        if sensors is not None:
            ax.scatter(
                sensors,
                jnp.zeros_like(sensors),
                marker="^",
                color="red",
                s=100,
                label="Sensors",
            )
            ax.legend(fontsize=self.font_size)
        self.set_labels(ax, xlabel=xlabel, ylabel=ylabel)
        self._finalize_plot(fig, filename)

    def _plot_wavefield_2d(
        self,
        Z,
        X=None,
        Y=None,
        title=None,
        xlabel="X",
        ylabel="Y",
        filename=None,
        sensors=None,
        plot_type="pcolor",  # New parameter to choose plot type
    ):
        """
        Plot a 2D scalar field either as a heatmap or surface.

        Parameters
        ----------
        Z : jnp.ndarray
            2D array of samples.
        X, Y : jnp.ndarray | None
            Optional coordinate grids. If ``None`` index grids are used.
        title : str | None
        xlabel, ylabel : str
        filename : str | None
            If provided, saves instead of showing.
        sensors : jnp.ndarray | None
            Array of sensor coordinates to overlay.
        plot_type : {"pcolor", "surface"}
            Choose between 2D heatmap and 3D surface render.
        """
        if plot_type not in ["pcolor", "surface"]:
            raise ValueError("plot_type must be either 'pcolor' or 'surface'")

        # For surface plot, we need 3D projection
        if plot_type == "surface":
            fig, ax = self.setup_figure(title=title, projection="3d")
        else:
            fig, ax = self.setup_figure(title=title)

        if X is None or Y is None:
            Y, X = jnp.indices(Z.shape)

        if plot_type == "pcolor":
            # Original pcolor plot
            pcm = ax.pcolormesh(X, Y, Z, cmap=self.colormap)
            fig.colorbar(pcm, ax=ax)

            # Add sensors if provided
            if sensors is not None:
                ax.scatter(
                    sensors[:, 0],
                    sensors[:, 1],
                    marker="^",
                    color="red",
                    s=10,
                    label="Sensors",
                )
                ax.legend(fontsize=self.font_size)

            self.set_labels(ax, xlabel=xlabel, ylabel=ylabel)

        else:  # surface plot
            # Create meshgrid if X and Y are 1D
            if X.ndim == 1 and Y.ndim == 1:
                X, Y = jnp.meshgrid(X, Y)

            # Create surface plot
            surf = ax.plot_surface(X, Y, Z, cmap=self.colormap)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

            # Add sensors if provided (need to get Z coordinates for 3D plot)
            if sensors is not None:
                sensor_z = jnp.zeros_like(sensors[:, 0])  # or interpolate Z values
                ax.scatter(
                    sensors[:, 0],
                    sensors[:, 1],
                    sensor_z,
                    marker="^",
                    color="red",
                    s=10,
                    label="Sensors",
                )
                ax.legend(fontsize=self.font_size)

            self.set_labels(ax, xlabel=xlabel, ylabel=ylabel, zlabel="Amplitude")

            # Optional: set the viewing angle for better visualization
            ax.view_init(elev=30, azim=45)

        self._finalize_plot(fig, filename)

    def _plot_wavefield_3d(
        self, values, X=None, Y=None, Z=None, title=None, filename=None
    ):
        """
        Volume render for 3D scalar fields using PyVista (optional dependency).

        Parameters
        ----------
        values : jnp.ndarray
            Scalar field with shape `(Nx, Ny, Nz)`.
        X, Y, Z : jnp.ndarray | None
            Optional coordinate grids; defaults to indices if omitted.
        title : str | None
        filename : str | None
            If provided, saves a screenshot instead of opening an interactive window.

        Raises
        ------
        ImportError
            If PyVista is not installed.
        """
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError("PyVista is required for 3D plotting.")
        if X is None or Y is None or Z is None:
            X, Y, Z = jnp.mgrid[: values.shape[0], : values.shape[1], : values.shape[2]]
        grid = pv.StructuredGrid(X, Y, Z)
        grid["values"] = values.flatten(order="F")
        plotter = pv.Plotter()
        plotter.add_volume(grid, scalars="values", cmap=self.colormap)
        plotter.add_scalar_bar("Values")
        plotter.set_background("white")
        if title:
            plotter.add_title(title)
        if filename:
            plotter.screenshot(filename)
        else:
            plotter.show()

    def _plot_complex_wavefield_1d(
        self, y_complex, title=None, xlabel="X-axis", ylabel="Y-axis", filename=None
    ):
        """
        Plot real and imaginary parts of a one-dimensional complex field.

        Parameters
        ----------
        y_complex : jnp.ndarray, shape (Nx,)
            Complex-valued trace.
        title : str, optional
            Figure title.
        xlabel : str, default="X-axis"
            X-axis label.
        ylabel : str, default="Y-axis"
            Y-axis label.
        filename : str, optional
            Output path. If ``None``, the figure is shown.
        """
        real_part = jnp.real(y_complex)
        imag_part = jnp.imag(y_complex)
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.plot(real_part)
        ax1.set_title("Real Part", fontsize=self.font_size)
        self.set_labels(ax1, xlabel=xlabel, ylabel=ylabel)

        ax2.plot(imag_part)
        ax2.set_title("Imaginary Part", fontsize=self.font_size)
        self.set_labels(ax2, xlabel=xlabel, ylabel=ylabel)

        fig.suptitle(title or "Complex Data Plot", fontsize=self.title_font_size)
        if filename:
            self.save_plot(filename)
        else:
            plt.show()

    def _plot_complex_wavefield_2d(self, Z_complex, X, Y, title=None, filename=None):
        """
        Plot real and imaginary parts of a two-dimensional complex field.

        Parameters
        ----------
        Z_complex : jnp.ndarray, shape (Ny, Nx)
            Complex-valued field.
        X, Y : jnp.ndarray
            Coordinate grids.
        title : str, optional
            Figure title.
        filename : str, optional
            Output path. If ``None``, the figure is shown.
        """
        real_part = jnp.real(Z_complex)
        imag_part = jnp.imag(Z_complex)
        fig, (ax1, ax2) = plt.subplots(1, 2)

        pcm1 = ax1.pcolormesh(X, Y, real_part, cmap=self.colormap)
        fig.colorbar(pcm1, ax=ax1)
        ax1.set_title("Real Part", fontsize=self.font_size)

        pcm2 = ax2.pcolormesh(X, Y, imag_part, cmap=self.colormap)
        fig.colorbar(pcm2, ax=ax2)
        ax2.set_title("Imaginary Part", fontsize=self.font_size)

        fig.suptitle(title or "Complex Array Plot", fontsize=self.title_font_size)
        if filename:
            self.save_plot(filename)
        else:
            plt.show()

    def plot_pressure_time(self, p, t, title=None, filename=None):
        """
        Plot time traces or space–time image of pressure data.

        Parameters
        ----------
        p : jnp.ndarray
            Pressure array shaped `(Nt,)` or `(Nt, Ns)`.
        t : jnp.ndarray
            Time samples aligned with the first dimension of ``p``.
        title : str | None
        filename : str | None
            If provided, save instead of displaying interactively.
        """
        fig, ax = self.setup_figure(title=title)
        ndim = p.ndim
        tmax = t[-1]

        if ndim == 1:
            ax.plot(t, p)
            ax.set_xlabel("Time")
            ax.set_ylabel("Pressure")
            ax.set_title(title or "Pressure vs. Time")
        elif ndim == 2:
            im = ax.imshow(
                p.T,
                aspect="auto",
                cmap=self.colormap,
                extent=[0, tmax, 0, p.shape[1]],
                origin="lower",
            )
            self.set_labels(ax, xlabel="Time", ylabel="spatial Index")
            fig.colorbar(im, ax=ax, label="Pressure")
        self._finalize_plot(fig, filename)

    def _plot_centers_1d(self, centers, title=None, filename=None):
        """
        Plot labelled one-dimensional centre locations.

        Parameters
        ----------
        centers : jnp.ndarray, shape (n, 1)
            Centre coordinates.
        title : str, optional
            Figure title.
        filename : str, optional
            Output path. If ``None``, the figure is shown.
        """
        fig, ax = self.setup_figure(title=title)
        ax.scatter(
            centers[:, 0], jnp.zeros_like(centers[:, 0]), c="blue", s=50, zorder=5
        )
        for idx, x in enumerate(centers[:, 0]):
            ax.annotate(
                str(idx),
                (x, 0),
                fontsize=self.font_size,
                ha="center",
                va="bottom",
                xytext=(0, 5),
                textcoords="offset points",
            )
        self.set_labels(ax, xlabel="X-axis")
        ax.set_yticks([])
        if filename:
            self.save_plot(filename)
        else:
            plt.show()

    def _plot_centers_2d(self, centers, title=None, filename=None):
        """
        Plot labelled two-dimensional centre locations.

        Parameters
        ----------
        centers : jnp.ndarray, shape (n, 2)
            Centre coordinates.
        title : str, optional
            Figure title.
        filename : str, optional
            Output path. If ``None``, the figure is shown.
        """
        fig, ax = self.setup_figure(title=title)
        ax.scatter(centers[:, 0], centers[:, 1], c="blue", s=50, zorder=5)
        for idx, (x, y) in enumerate(centers):
            ax.annotate(
                str(idx),
                (x, y),
                fontsize=self.font_size,
                ha="left",
                va="bottom",
                xytext=(5, 5),
                textcoords="offset points",
            )
        self.set_labels(ax, xlabel="X-axis", ylabel="Y-axis")
        ax.set_aspect("equal", adjustable="box")
        if filename:
            self.save_plot(filename)
        else:
            plt.show()

    def _plot_centers_3d(self, centers, title=None, filename=None):
        """
        Plot labelled three-dimensional centre locations.

        Parameters
        ----------
        centers : jnp.ndarray, shape (n, 3)
            Centre coordinates.
        title : str, optional
            Figure title.
        filename : str, optional
            Output path. If ``None``, the figure is shown.
        """
        # Create figure with 3D projection
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        if title:
            ax.set_title(title, fontsize=self.font_size + 2)

        # Scatter plot for 3D points
        ax.scatter(
            centers[:, 0], centers[:, 1], centers[:, 2], c="blue", s=50, zorder=5
        )

        # Annotate each point with its index
        for idx, (x, y, z) in enumerate(centers):
            ax.text(x, y, z, str(idx), fontsize=self.font_size, ha="left", va="bottom")

        # Set axis labels
        ax.set_xlabel("X-axis", fontsize=self.font_size)
        ax.set_ylabel("Y-axis", fontsize=self.font_size)
        ax.set_zlabel("Z-axis", fontsize=self.font_size)

        # Optional: Set equal aspect ratio (though this can be tricky in 3D)
        # You may want to comment this out if it causes issues
        ax.set_box_aspect([1, 1, 1])

        if filename:
            self.save_plot(filename)
        else:
            plt.show()

    def plot_centers(self, centers, title=None, filename=None):
        """
        Visualize dyadic box centres in 1D/2D/3D.

        Parameters
        ----------
        centers : jnp.ndarray
            Array of integer centre coordinates with shape `(B, d)`.
        title : str | None
        filename : str | None
            Optional path to save the figure instead of showing.
        """
        ndim = centers.shape[1]
        if ndim == 1:
            self._plot_centers_1d(centers, title, filename)
        elif ndim == 2:
            self._plot_centers_2d(centers, title, filename)
        elif ndim == 3:
            self._plot_centers_3d(centers, title, filename)
        else:
            raise ValueError(f"Unsupported number of dimensions: {ndim}")

    def plot_centers_slice_grid(
        self,
        centers,
        axis="z",
        slice_values=None,
        cols=3,
        title=None,
        font_size=None,
        figsize=None,
        save_path=None,
        show=True,
    ):
        """
        Plot labelled centers as 2D scatter slices along a chosen axis.

        Parameters
        ----------
        centers : array-like, shape (n_points, 3)
            3D coordinates.
        axis : {"x","y","z",0,1,2}
            Which axis to slice along.
        slice_values : iterable | None
            Specific coordinate values to plot. If None, uses all unique values.
        cols : int
            Number of subplot columns.
        title : str | None
            Figure title.
        font_size : int | None
            Base font size for labels (defaults to self.font_size).
        figsize : tuple | None
            Optional matplotlib figsize. Defaults to `(cols*4, rows*4)`.
        save_path : str | Path | None
            If provided, path to save the PNG.
        show : bool
            Whether to call ``plt.show()``.
        """

        centers = np.asarray(centers)
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError(f"`centers` must have shape (n, 3); got {centers.shape}")

        fs = self.font_size if font_size is None else font_size
        axis_map = {"x": 0, "y": 1, "z": 2}
        axis_idx = axis_map.get(axis.lower(), axis) if isinstance(axis, str) else axis
        if axis_idx not in (0, 1, 2):
            raise ValueError(f"axis must be x/y/z or 0/1/2; got {axis}")

        axis_labels = ["X", "Y", "Z"]
        if slice_values is None:
            slice_values = np.unique(centers[:, axis_idx])
        slice_values = list(slice_values)
        if len(slice_values) == 0:
            raise ValueError("No slice values to plot.")

        cols = max(1, int(cols))
        rows = int(np.ceil(len(slice_values) / cols))
        if figsize is None:
            figsize = (cols * 4, rows * 4)

        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        other_axes = [ax for ax in (0, 1, 2) if ax != axis_idx]

        for ax, val in zip(axes.flat, slice_values):
            mask = centers[:, axis_idx] == val
            pts = centers[mask]
            idxs = np.nonzero(mask)[0]

            ax.scatter(pts[:, other_axes[0]], pts[:, other_axes[1]], c="tab:blue", s=40)
            for idx, (x, y) in zip(idxs, pts[:, other_axes]):
                ax.annotate(
                    str(idx),
                    (x, y),
                    xytext=(6, 0),
                    textcoords="offset points",
                    fontsize=fs,
                    ha="left",
                    va="center",
                    color="darkred",
                )

            ax.set_title(f"{axis_labels[axis_idx]} = {val}", fontsize=fs)
            ax.set_xlabel(f"{axis_labels[other_axes[0]]}-axis", fontsize=fs)
            ax.set_ylabel(f"{axis_labels[other_axes[1]]}-axis", fontsize=fs)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3)

        # Hide any unused subplots
        for ax in axes.flat[len(slice_values) :]:
            ax.axis("off")

        fig.suptitle(
            title or f"Centers by {axis_labels[axis_idx]} slices", fontsize=fs + 2
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if save_path is not None:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()

        return fig

    def plot_coefficients(
        self,
        coeffs: jnp.ndarray,
        wpt,
        sort_data=False,
        log_scale=False,
        filename=None,
    ):
        """
        Scatter-plot MSWPT coefficients coloured by dyadic level.

        Parameters
        ----------
        coeffs : jnp.ndarray
            Flattened coefficient vector.
        wpt : MSWPT
            Transform object (used to infer level layout).
        sort_data : bool, default=False
            If True, sort by magnitude descending.
        log_scale : bool, default=False
            Plot ``log10(|c|)`` instead of ``|c|``.
        filename : str | None
            Optional path to save the figure.
        """
        dyadic_decomp = wpt.dyadic_decomp
        redundancy = wpt.redundancy
        num_levels = dyadic_decomp.num_levels
        coeff_shapes = utils.compute_coeff_shapes(
            dyadic_decomp, redundancy, jnp.arange(num_levels)
        )
        cumulative_lengths = jnp.cumsum(jnp.prod(coeff_shapes, axis=1))
        flat_indices = jnp.arange(len(coeffs))
        tensor_indices = jnp.searchsorted(
            cumulative_lengths, flat_indices, side="right"
        )

        vals = coeffs
        cols = tensor_indices

        if num_levels > 1:
            cmap = colormaps.get_cmap(self.colormap)
            level_colors = ["brown", "navy"] + list(
                cmap(jnp.linspace(0, 1, num_levels - 2))
            )
        else:
            level_colors = ["brown"]

        sorted_indices = (
            jnp.argsort(-jnp.abs(vals)) if sort_data else jnp.arange(len(vals))
        )

        coeff_values = vals[sorted_indices]
        coeff_levels = cols[sorted_indices]

        y_values = (
            jnp.log10(jnp.abs(coeff_values)) if log_scale else jnp.abs(coeff_values)
        )

        marker_size = 10
        fig, ax = self.setup_figure()
        ax.scatter(
            range(len(y_values)),
            y_values,
            s=marker_size,
            color=[level_colors[level] for level in coeff_levels],
            alpha=0.3,
        )

        self.set_labels(
            ax,
            xlabel="Coefficient index",
            ylabel="|cₗ|" if not log_scale else "log|cₗ|",
        )
        ax.set_title("Plot of Coefficients", fontsize=self.title_font_size)

        for level, color in enumerate(level_colors):
            ax.plot([], [], "o", color=color, label=f"Level {level + 1}")

        ax.legend(fontsize=self.font_size)

        if filename is not None:
            self.save_plot(filename)
        plt.show()

    def plot_speed_of_sound(
        self,
        c: jnp.ndarray,
        domain: Domain,
        title: str = "Speed of Sound Map",
        filename: str = None,
        num_contours: int = 20,
    ):
        """
        Plot a speed-of-sound field in 2D/3D.

        Parameters
        ----------
        c : jnp.ndarray
            Scalar field sampled on `domain.N`.
        domain : Domain
            Provides grid spacing and dimensionality.
        title : str, default="Speed of Sound Map"
        filename : str | None
            Save path (optional).
        num_contours : int, default=20
            Number of contour levels for 2D plots.

        Raises
        ------
        ValueError
            If ``domain.ndim`` is not 2 or 3.
        """
        spatial_meshgrid, _ = domain.generate_meshgrid()

        if domain.ndim == 2:
            fig, ax = self.setup_figure()
            X, Y = spatial_meshgrid

            # Create filled contour plot
            contour = ax.contourf(X, Y, c, levels=num_contours, cmap=self.colormap)

            # Add contour lines
            ax.contour(
                X, Y, c, levels=num_contours, colors="k", linewidths=0.5, alpha=0.5
            )

            self.set_labels(ax, xlabel="x", ylabel="y")
            ax.set_title(title, fontsize=self.title_font_size)
            fig.colorbar(contour, ax=ax, label="Speed of Sound")

            if filename is not None:
                self.save_plot(filename)
            plt.show()

        elif domain.ndim == 3:
            X, Y, Z = spatial_meshgrid
            grid = pv.StructuredGrid(X, Y, Z)
            grid["speed_of_sound"] = c.flatten(order="F")

            p = pv.Plotter()
            p.add_mesh(grid, scalars="speed_of_sound", cmap=self.pv_colormap)
            p.add_scalar_bar("Speed of Sound")
            p.set_background(self.pv_background_color)
            p.add_title(title)

            if filename is not None:
                p.screenshot(filename)
            p.show()

        else:
            raise ValueError(f"Unsupported number of dimensions: {domain.ndim}")

    def plot_coeff_surf(self, coeffs, N, title=None, filename=None):
        """
        Surface plot for a flattened 4N² coefficient vector.

        Parameters
        ----------
        coeffs : jnp.ndarray
            1D array expected to be length ``4 * N**2``.
        N : int
            Reshape factor (resulting grid is ``(2N, 2N)``).
        title : str | None
        filename : str | None
        """
        # Reshape the 1D array into a 2N x 2N grid
        Z = coeffs.reshape((2 * N, 2 * N))

        # Create x and y coordinates
        x = jnp.arange(0, 2 * N, 1)
        y = jnp.arange(0, 2 * N, 1)
        X, Y = jnp.meshgrid(x, y)

        # Create the 3D plot
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Create the surface plot
        surf = ax.plot_surface(X, Y, Z, cmap=self.colormap)

        # Add a color bar
        fig.colorbar(surf)

        # Set labels and title
        ax.set_xlabel("X axis", fontsize=self.font_size)
        ax.set_ylabel("Y axis", fontsize=self.font_size)
        ax.set_zlabel("Coefficient value", fontsize=self.font_size)
        if title:
            ax.set_title(title, fontsize=self.title_font_size)
        else:
            ax.set_title("Surface Plot of Coefficients", fontsize=self.title_font_size)

        # Save the plot if filename is provided
        if filename:
            plt.savefig(filename)

        # Show the plot
        plt.show()


def animate_wavefield_2d(data, skip_frames=1, cmap="viridis", label="", timesteps=None):
    """
    Animate a time-series of 2D fields with matplotlib.

    Parameters
    ----------
    data : jnp.ndarray
        Array shaped `(Nt, Ny, Nx)`.
    skip_frames : int, default=1
        Step between frames to thin out long sequences.
    cmap : str, default="viridis"
        Colormap name.
    label : str, default=""
        Colorbar label.
    timesteps : jnp.ndarray | None
        Optional time stamps used in the frame title.
    """
    frames = data.shape[0] // skip_frames

    fig, ax = plt.subplots()

    # Create the initial plot
    im = ax.imshow(data[0], animated=True, cmap=cmap, aspect="auto", origin="lower")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label)

    # Keep a title artist for frame updates.
    title = ax.set_title("")

    def update_plot(frame):
        """
        Update one frame of the 2D wavefield animation.

        Parameters
        ----------
        frame : int
            Animation frame index after applying ``skip_frames``.

        Returns
        -------
        list
            Matplotlib artists updated for this frame.
        """
        im.set_array(data[frame * skip_frames])
        if timesteps is not None:
            current_time = timesteps[frame * skip_frames]
            title.set_text(f"t = {current_time:.2f}")
        return [im, title]

    # Create the animation
    ani = FuncAnimation(fig, update_plot, frames=frames, interval=50, blit=False)  # noqa
    plt.show()


def animate_wavefield_1d(data, skip_frames=1):
    """
    Animate 1D traces over time.

    Parameters
    ----------
    data : jnp.ndarray
        Array shaped `(Nt, Nx)`.
    skip_frames : int, default=1
        Step between frames to thin out long sequences.
    """
    frames = data.shape[0] // skip_frames
    fig, ax = plt.subplots()
    (line,) = ax.plot(data[0])

    # Set axis limits
    ax.set_ylim(data.min(), data.max())
    ax.set_xlim(0, data.shape[1])

    def update_plot(frame):
        """
        Update one frame of the 1D wavefield animation.

        Parameters
        ----------
        frame : int
            Animation frame index after applying ``skip_frames``.

        Returns
        -------
        tuple
            Updated line artist.
        """
        line.set_ydata(data[frame * skip_frames])
        return (line,)

    ani = FuncAnimation(fig, update_plot, frames=frames, interval=50, blit=True)  # noqa
    plt.show()


def animate_wavefield(data, skip_frames=1, cmap="viridis"):
    """
    Dispatch animation to 1D/2D helpers based on data dimensionality.

    Parameters
    ----------
    data : jnp.ndarray
        Array shaped `(Nt, Nx)` or `(Nt, Ny, Nx)`.
    skip_frames : int, default=1
    cmap : str, default="viridis"
    """
    ndim = len(data.shape)
    if len(data.shape) == 2:
        animate_wavefield_1d(data, skip_frames)
    elif len(data.shape) == 3:
        animate_wavefield_2d(data, skip_frames, cmap)
    else:
        raise ValueError(f"Unsupported data dimension: {ndim}")


def plot_wavefields_interactive(
    reference_all,
    p0_msgb_all,
    ts,
    domain: Domain,
    XY,
    plot_type="imshow",
    fixed_diff_colorbar=False,
):
    """
    Interactive comparison between Reference and MSGB wavefields.

    Parameters
    ----------
    reference_all : jnp.ndarray
        Wavefield sequence from the Reference solver `(Nt, ...)`.
    p0_msgb_all : jnp.ndarray
        Wavefield sequence from the MSGB solver `(Nt, ...)`.
    ts : jnp.ndarray
        Time points associated with frames.
    domain : Domain
        Provides spatial coordinates.
    XY : jnp.ndarray
        Meshgrid used for plotting.
    plot_type : {"imshow", "surf"}, default="imshow"
        Choose between image and 3D surface rendering for 2D data.
    fixed_diff_colorbar : bool, default=False
        If True, keep the difference plot color limits fixed across frames.
    """
    fig = plt.figure(figsize=(18, 12))

    if len(reference_all.shape) == 2:
        X = domain.generate_meshgrid()[0][0]
    else:
        X, Y = domain.generate_meshgrid()[0]
    time_index = 0
    difference = reference_all[time_index] - p0_msgb_all[time_index]

    # Calculate global min and max for the differences across all time points if fixed_diff_colorbar is True
    if fixed_diff_colorbar:
        global_min_diff = jnp.min(reference_all - p0_msgb_all)
        global_max_diff = jnp.max(reference_all - p0_msgb_all)

    if len(reference_all.shape) == 2:
        axs = [fig.add_subplot(1, 3, i + 1) for i in range(3)]

        (im_reference,) = axs[0].plot(
            X.flatten(), reference_all[time_index], color="blue"
        )
        (im_msgb,) = axs[1].plot(X.flatten(), p0_msgb_all[time_index], color="orange")
        (im_diff,) = axs[2].plot(X.flatten(), difference, color="red")

    else:
        if plot_type == "surf":
            axs = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
        else:
            axs = [fig.add_subplot(1, 3, i + 1) for i in range(3)]

        if plot_type == "imshow":
            im_reference = axs[0].imshow(
                reference_all[time_index], extent=[X.min(), X.max(), Y.min(), Y.max()]
            )
            plt.colorbar(im_reference, ax=axs[0], orientation="horizontal", pad=0.15)

            im_msgb = axs[1].imshow(
                p0_msgb_all[time_index], extent=[X.min(), X.max(), Y.min(), Y.max()]
            )
            plt.colorbar(im_msgb, ax=axs[1], orientation="horizontal", pad=0.15)

            im_diff = axs[2].imshow(
                difference, extent=[X.min(), X.max(), Y.min(), Y.max()], cmap="coolwarm"
            )
            diff_colorbar = plt.colorbar(
                im_diff, ax=axs[2], orientation="horizontal", pad=0.15
            )
            # Set the colorbar limits if fixed_diff_colorbar is True
            if fixed_diff_colorbar:
                im_diff.set_clim(vmin=global_min_diff, vmax=global_max_diff)

        else:
            surf_reference = axs[0].plot_surface(
                X, Y, reference_all[time_index], cmap="viridis"
            )
            fig.colorbar(surf_reference, ax=axs[0], orientation="horizontal", pad=0.15)

            surf_msgb = axs[1].plot_surface(
                X, Y, p0_msgb_all[time_index], cmap="viridis"
            )
            fig.colorbar(surf_msgb, ax=axs[1], orientation="horizontal", pad=0.15)

            surf_diff = axs[2].plot_surface(X, Y, difference, cmap="coolwarm")
            diff_colorbar = fig.colorbar(
                surf_diff, ax=axs[2], orientation="horizontal", pad=0.15
            )
            if fixed_diff_colorbar:
                axs[2].set_zlim(global_min_diff, global_max_diff)
                surf_diff.set_clim(vmin=global_min_diff, vmax=global_max_diff)
                diff_colorbar.update_normal(surf_diff)

        axs[0].set_title(f"Reference Solution at t={ts[time_index]:.2e}s")
        axs[1].set_title(f"MSGB Solution at t={ts[time_index]:.2e}s")
        axs[2].set_title(f"Difference\nat t={ts[time_index]:.2e}s")

        for ax in axs[:2]:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

    plt.subplots_adjust(bottom=0.25)

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    slider = Slider(ax_slider, "Time", 0, len(ts) - 1, valinit=0, valfmt="%0.0f")

    ax_button_prev = plt.axes([0.1, 0.025, 0.1, 0.04])
    button_prev = Button(ax_button_prev, "-dt")

    ax_button_next = plt.axes([0.8, 0.025, 0.1, 0.04])
    button_next = Button(ax_button_next, "+dt")

    def update(val):
        """
        Respond to slider changes in the interactive comparison view.

        Parameters
        ----------
        val : float
            Slider value representing the time index.
        """
        time_index = int(slider.val)
        update_plots(time_index)

    def update_plots(time_index):
        """
        Redraw all comparison panels for a time index.

        Parameters
        ----------
        time_index : int
            Index into ``ts`` and both wavefield sequences.
        """
        difference = reference_all[time_index] - p0_msgb_all[time_index]
        if len(reference_all.shape) == 2:  # Handle 1D data
            im_reference.set_ydata(reference_all[time_index])
            im_msgb.set_ydata(p0_msgb_all[time_index])
            im_diff.set_ydata(difference)

            axs[0].set_ylim(
                reference_all[time_index].min(), reference_all[time_index].max()
            )
            axs[1].set_ylim(
                p0_msgb_all[time_index].min(), p0_msgb_all[time_index].max()
            )
            if fixed_diff_colorbar:
                axs[2].set_ylim(global_min_diff, global_max_diff)
            else:
                axs[2].set_ylim(difference.min(), difference.max())

        elif plot_type == "imshow":
            im_reference.set_data(reference_all[time_index])
            im_msgb.set_data(p0_msgb_all[time_index])

            im_diff.set_data(difference)
            if fixed_diff_colorbar:
                im_diff.set_clim(vmin=global_min_diff, vmax=global_max_diff)
            else:
                im_diff.set_clim(difference.min(), difference.max())

        else:
            axs[0].clear()
            axs[1].clear()
            axs[2].clear()

            axs[0].plot_surface(X, Y, reference_all[time_index], cmap="viridis")
            axs[1].plot_surface(X, Y, p0_msgb_all[time_index], cmap="viridis")

            surf_diff = axs[2].plot_surface(X, Y, difference, cmap="coolwarm")

            if fixed_diff_colorbar:
                axs[2].set_zlim(global_min_diff, global_max_diff)
                surf_diff.set_clim(vmin=global_min_diff, vmax=global_max_diff)
                diff_colorbar.update_normal(surf_diff)
            else:
                axs[2].set_zlim(difference.min(), difference.max())

        axs[0].set_title(f"Reference Solution\nat t={ts[time_index]:.2e}s")
        axs[1].set_title(f"MSGB Solution\nat t={ts[time_index]:.2e}s")
        axs[2].set_title(f"Difference\nat t={ts[time_index]:.2e}s")

        for ax in axs[:2]:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

        fig.canvas.draw_idle()

    def next_step(event):
        """
        Advance the interactive comparison by one time step.

        Parameters
        ----------
        event : matplotlib.backend_bases.Event
            Button-click event.
        """
        current_time_index = int(slider.val)
        if current_time_index < len(ts) - 1:
            slider.set_val(current_time_index + 1)

    def prev_step(event):
        """
        Move the interactive comparison back by one time step.

        Parameters
        ----------
        event : matplotlib.backend_bases.Event
            Button-click event.
        """
        current_time_index = int(slider.val)
        if current_time_index > 0:
            slider.set_val(current_time_index - 1)

    slider.on_changed(update)
    button_next.on_clicked(next_step)
    button_prev.on_clicked(prev_step)

    plt.show()


def plot_wavefields_interactive_wpt(
    reference_all,
    p0_msgb_all,
    ts,
    domain: Domain,
    XY,
    wpt,
    dyadic_decomp,
    plot_type="imshow",
    fixed_diff_colorbar=False,
):
    """
    Interactive wavefield viewer that also shows MSWPT coefficients.

    Parameters
    ----------
    reference_all : jnp.ndarray
        Wavefield sequence from the Reference solver `(Nt, ...)`.
    p0_msgb_all : jnp.ndarray
        Wavefield sequence from the MSGB solver `(Nt, ...)`.
    ts : jnp.ndarray
        Time points associated with frames.
    domain : Domain
    XY : jnp.ndarray
        Meshgrid used for plotting.
    wpt : MSWPT
        Transform used to compute coefficients.
    dyadic_decomp : DyadicDecomposition
        Associated dyadic tiling used for labelling levels.
    plot_type : {"imshow", "surf"}, default="imshow"
    fixed_diff_colorbar : bool, default=False
        If True, keep the difference plot color limits fixed across frames.
    """
    fig = plt.figure(figsize=(18, 12))

    # Create subplots for wavefields (reference, msgb, difference, wpt coefficients)
    ax_wavefield_reference = fig.add_subplot(2, 3, 1)
    ax_wavefield_msgb = fig.add_subplot(2, 3, 2)
    ax_wavefield_wpt = fig.add_subplot(2, 3, 3)

    if len(reference_all.shape) == 2:
        X = domain.generate_meshgrid()[0][0]
    else:
        X, Y = domain.generate_meshgrid()[0]
    time_index = 0

    if len(reference_all.shape) == 2:  # 1D case
        im_reference = ax_wavefield_reference.plot(
            X.flatten(), reference_all[time_index], color="blue"
        )[0]
        im_msgb = ax_wavefield_msgb.plot(
            X.flatten(), p0_msgb_all[time_index], color="orange"
        )[0]
    else:  # 2D case
        im_reference = ax_wavefield_reference.imshow(
            reference_all[time_index], extent=[X.min(), X.max(), Y.min(), Y.max()]
        )
        im_msgb = ax_wavefield_msgb.imshow(
            p0_msgb_all[time_index], extent=[X.min(), X.max(), Y.min(), Y.max()]
        )

        plt.colorbar(im_reference, ax=ax_wavefield_reference)
        plt.colorbar(im_msgb, ax=ax_wavefield_msgb)

    # Set titles for wavefield plots
    ax_wavefield_reference.set_title(f"Reference Solution at t={ts[time_index]:.2e}s")
    ax_wavefield_msgb.set_title(f"MSGB Solution at t={ts[time_index]:.2e}s")
    ax_wavefield_wpt.set_title(f"MSWPT Coefficients at t={ts[time_index]:.2e}s")

    # Add slider and buttons for interactivity
    plt.subplots_adjust(bottom=0.25)
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    slider = Slider(ax_slider, "Time", 0, len(ts) - 1, valinit=0, valfmt="%0.0f")

    ax_button_prev = plt.axes([0.1, 0.025, 0.1, 0.04])
    button_prev = Button(ax_button_prev, "-dt")

    ax_button_next = plt.axes([0.8, 0.025, 0.1, 0.04])
    button_next = Button(ax_button_next, "+dt")

    def update(val):
        """
        Respond to slider changes in the WPT comparison view.

        Parameters
        ----------
        val : float
            Slider value representing the time index.
        """
        time_index = int(slider.val)
        update_plots(time_index)

    def update_plots(time_index):
        """
        Redraw WPT comparison panels for a time index.

        Parameters
        ----------
        time_index : int
            Index into ``ts`` and both wavefield sequences.
        """
        if len(reference_all.shape) == 2:  # 1D case
            im_reference.set_ydata(reference_all[time_index])
            im_msgb.set_ydata(p0_msgb_all[time_index])
        else:  # 2D case
            im_reference.set_data(reference_all[time_index])
            im_msgb.set_data(p0_msgb_all[time_index])

    def next_step(event):
        """
        Advance the WPT comparison by one time step.

        Parameters
        ----------
        event : matplotlib.backend_bases.Event
            Button-click event.
        """
        current_time_index = int(slider.val)
        if current_time_index < len(ts) - 1:
            slider.set_val(current_time_index + 1)

    def prev_step(event):
        """
        Move the WPT comparison back by one time step.

        Parameters
        ----------
        event : matplotlib.backend_bases.Event
            Button-click event.
        """
        current_time_index = int(slider.val)
        if current_time_index > 0:
            slider.set_val(current_time_index - 1)

    slider.on_changed(update)
    button_next.on_clicked(next_step)
    button_prev.on_clicked(prev_step)

    plt.show()


def plot_GB_ellipse(x0, p0, a0, m0, figsize=(10, 10), scale_factor=1.0):
    """
    Visualize Gaussian beams as ellipses with arrows.

    Parameters
    ----------
    x0 : jnp.ndarray
        Beam positions, shape `(b, d)`.
    p0 : jnp.ndarray
        Propagation directions, shape `(b, d)`.
    a0 : jnp.ndarray
        Complex amplitudes, shape `(b,)`.
    m0 : jnp.ndarray
        Hessian matrices, shape `(b, d, d)`.
    figsize : Tuple[int, int], default=(10, 10)
    scale_factor : float, default=1.0
        Scales ellipse size for legibility.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize amplitudes for visualization
    max_amplitude = jnp.max(jnp.abs(a0))
    normalized_amplitudes = jnp.abs(a0) / max_amplitude

    for i in range(len(x0)):
        eigenvals, eigenvecs = jnp.linalg.eigh(m0[i])

        width = (
            scale_factor
            * jnp.sqrt(jnp.abs(1 / eigenvals[0]))
            * normalized_amplitudes[i]
        )
        height = (
            scale_factor
            * jnp.sqrt(jnp.abs(1 / eigenvals[1]))
            * normalized_amplitudes[i]
        )

        angle = jnp.degrees(jnp.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))

        ellipse = Ellipse(
            xy=x0[i],
            width=width,
            height=height,
            angle=angle,
            alpha=0.5,
            facecolor="blue",
        )
        ax.add_patch(ellipse)

        # Add arrow for propagation direction
        # Normalize p0 vector and scale it by the ellipse size
        p0_norm = p0[i] / jnp.linalg.norm(p0[i])
        arrow_length = max(width, height) * 0.5
        ax.arrow(
            x0[i, 0],
            x0[i, 1],
            p0_norm[0] * arrow_length,
            p0_norm[1] * arrow_length,
            head_width=arrow_length * 0.2,
            head_length=arrow_length * 0.3,
            fc="red",
            ec="red",
        )

    # Set equal aspect ratio and adjust limits
    ax.set_aspect("equal")

    # Add some padding to the plot
    all_coords = jnp.vstack((x0 + p0, x0 - p0))
    x_min, y_min = jnp.min(all_coords, axis=0) - scale_factor
    x_max, y_max = jnp.max(all_coords, axis=0) + scale_factor
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Add grid and labels
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Gaussian Beam Decomposition")

    return fig, ax


def GB_point_line_plot(X, Y, p0, p0_b, x0_b):
    """
    Overlay beam start points and directions on an initial pressure field.

    Parameters
    ----------
    X, Y : jnp.ndarray
        Meshgrid arrays for plotting the pressure field.
    p0 : jnp.ndarray
        Initial pressure map (same shape as ``X``/``Y``).
    p0_b : jnp.ndarray
        Beam directions, shape `(b, d)`.
    x0_b : jnp.ndarray
        Beam start positions, shape `(b, d)`.
    """

    plt.figure()

    # Your existing pcolormesh
    plt.pcolormesh(X, Y, p0)
    plt.colorbar()
    plt.title("Initial Pressure")

    # Normalize p0_b row-wise to unit vectors
    p0_b_unit = p0_b / jnp.linalg.norm(p0_b, axis=1, keepdims=True) / 10

    # Scatter points for each slice of x0_b and plot arrows for p0_b
    for i in range(x0_b.shape[0]):
        plt.scatter(x0_b[i, 0], x0_b[i, 1], color="red")
        plt.quiver(
            x0_b[i, 0],
            x0_b[i, 1],
            p0_b_unit[i, 0],
            p0_b_unit[i, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="red",
        )

    plt.axis("equal")
    plt.show()


def plot_coeffs(data, dyadic_decomp, wpt):
    """
    Quick visualizer for MSWPT coefficients and Fourier-space centres.

    Parameters
    ----------
    data : jnp.ndarray
        Spatial-domain input field.
    dyadic_decomp : DyadicDecomposition
    wpt : MSWPT
        Transform instance; uses ``forward`` with spatial input.

    Notes
    -----
    Only tested for 2D layouts; higher dimensions will need refinements.
    """
    coeffs = wpt.forward(data, "spatial", False)
    centres2 = dyadic_decomp.centres_ndim * 2 + jnp.array(dyadic_decomp.N)
    plt.imshow(jnp.abs(coeffs))
    plt.scatter(centres2[:, 0], centres2[:, 1], c="r")
    for idx, (x, y) in enumerate(centres2, start=0):
        plt.annotate(
            str(idx),
            (x, y),
            fontsize=10,
            ha="left",
            va="bottom",
            xytext=(5, 5),
            textcoords="offset points",
            color="white",
        )
    plt.colorbar()
    plt.show()


def _mswpt_highlight_indices(
    dyadic_decomp: DyadicDecomposition,
    cutoff_freq: float | None,
    box_corners: np.ndarray | jnp.ndarray | tuple[int, int] | None,
):
    """Return coefficient-box indices using the same LF selection as hybrid solves."""
    if cutoff_freq is not None:
        idx = hybrid_solver_utils.get_indices_with_norm_less_than(
            dyadic_decomp.centres_ndim, cutoff_freq
        )
        return np.asarray(idx)
    if box_corners is None:
        return None

    bc = np.asarray(box_corners, dtype=int)
    if bc.shape != (2,):
        raise ValueError("box_corners must contain exactly two box indices.")

    idx = hybrid_solver_utils.get_indices_between_two_opposing_corners(
        dyadic_decomp.centres_ndim,
        int(bc[0]),
        int(bc[1]),
    )
    return np.asarray(idx)


def plot_mswpt_coeffs(
    ax: plt.Axes,
    coeffs_array: jnp.ndarray,
    dyadic_decomp: DyadicDecomposition,
    cutoff_freq: float | None = None,
    box_corners: np.ndarray | jnp.ndarray | tuple[int, int] | None = None,
    asymptote: bool = False,
    log_scale: bool = False,
):
    """
    Plot MSWPT coefficients with dyadic boxes, highlights, and asymptotes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    coeffs_array : jnp.ndarray
        Two-dimensional coefficient image.
    dyadic_decomp : DyadicDecomposition
        Decomposition providing box centres and sizes.
    cutoff_freq : float, optional
        Highlight boxes whose centre norm is below this threshold.
    box_corners : array-like, optional
        Pair of opposing global box indices defining the highlighted
        low-frequency region.
    asymptote : bool, default=False
        Whether to overlay diagonal asymptote guides.
    log_scale : bool, default=False
        Whether to display coefficient magnitudes with logarithmic
        normalization.

    Returns
    -------
    matplotlib.image.AxesImage
        Image handle returned by ``imshow``.
    """
    # --- Data Prep ---
    # Transpose to match (Time, Space) orientation usually desired in plots
    coeffs_mag = np.asarray(np.abs(coeffs_array)).T

    N0, N1 = dyadic_decomp.N
    # Extent logic: x-axis is original 2nd dim, y-axis is original 1st dim
    extent_coeffs = [-N1 // 2, N1 // 2, -N0 // 2, N0 // 2]

    # --- Log Normalization ---
    if log_scale:
        c_max = float(coeffs_mag.max())
        if c_max == 0:
            c_max = 1.0
        c_min = c_max * 1e-2  # Dynamic range: 2 orders of magnitude

        coeffs_mag_plot = np.maximum(coeffs_mag, c_min)
        w_norm = mcolors.LogNorm(vmin=c_min, vmax=c_max)
        title = r"$\log |c_{\ell,j,k}|$"
    else:
        coeffs_mag_plot = coeffs_mag
        w_norm = None
        title = r"$|c_{\ell,j,k}|$"

    # --- Plotting ---
    im = ax.imshow(  # noqa
        coeffs_mag_plot,
        origin="lower",
        extent=extent_coeffs,
        aspect="auto",
        norm=w_norm,
    )
    ax.set_title(title)

    # --- Dyadic Boxes ---
    cumsum_boxes = jnp.r_[0, jnp.cumsum(dyadic_decomp.num_boxes_ndim)]
    box_lengths = dyadic_decomp.box_lengths
    box_aspect = dyadic_decomp.box_aspect_ratio
    colors = ["gray", "darkgray", "silver", "lightgray"]

    for level in range(dyadic_decomp.num_levels):
        start_idx = cumsum_boxes[level]
        end_idx = cumsum_boxes[level + 1]
        centers = dyadic_decomp.centres_ndim[start_idx:end_idx]
        box_length = box_lengths[level]

        # Swapped dimensions for transposed plot
        box_width_x = box_length * box_aspect[1]
        box_width_y = box_length * box_aspect[0]
        color = colors[level % len(colors)]

        for center in centers:
            # Swapped centers for transposed plot
            cx, cy = float(center[1]), float(center[0])
            rect = patches.Rectangle(
                (cx - box_width_x / 2, cy - box_width_y / 2),
                box_width_x,
                box_width_y,
                linewidth=1.0,
                edgecolor=color,
                facecolor="none",
                linestyle=":",
                alpha=0.6,
            )
            ax.add_patch(rect)

    # --- Highlights ---
    selector_indices = _mswpt_highlight_indices(
        dyadic_decomp,
        cutoff_freq,
        box_corners,
    )

    if selector_indices is not None and selector_indices.size > 0:
        bounds_x, bounds_y = [], []
        for global_idx in selector_indices:
            level = utils.find_level(dyadic_decomp, int(global_idx))
            center = dyadic_decomp.centres_ndim[int(global_idx)]
            box_length = float(box_lengths[level])

            # Swapped dims
            box_width_x = float(box_length * box_aspect[1])
            box_width_y = float(box_length * box_aspect[0])
            cx, cy = float(center[1]), float(center[0])

            bounds_x.extend([cx - box_width_x / 2, cx + box_width_x / 2])
            bounds_y.extend([cy - box_width_y / 2, cy + box_width_y / 2])

        rect = patches.Rectangle(
            (min(bounds_x), min(bounds_y)),
            max(bounds_x) - min(bounds_x),
            max(bounds_y) - min(bounds_y),
            linewidth=2.0,
            edgecolor="red",
            facecolor="none",
            linestyle="--",
            zorder=5,
        )
        ax.add_patch(rect)

    if asymptote:
        # --- Diagonal Asymptotes ---
        # Gradient = 2 relative to the axes (steeper slope).
        # F-O1: brighter colour and thicker line so the cone is visible
        # against the dark log-magnitude background.
        limit = abs(extent_coeffs[1])  # x_max
        # Plot y = 2x (approx)
        ax.plot(
            [-limit / 2, limit / 2],
            [-2 * limit, 2 * limit],
            color="#ffa500",  # orange
            linestyle="-.",
            linewidth=2.5,
            zorder=6,
        )
        # Plot y = -2x
        ax.plot(
            [-limit / 2, limit / 2],
            [2 * limit, -2 * limit],
            color="#ffa500",  # orange
            linestyle="-.",
            linewidth=2.5,
            zorder=6,
        )

    return im


def plot_mswpt_coeffs_3d(
    ax: plt.Axes,
    coeffs_array: jnp.ndarray,
    dyadic_decomp: DyadicDecomposition,
    cutoff_freq: float | None = None,
    box_corners: np.ndarray | jnp.ndarray | tuple[int, int] | None = None,
    asymptote: bool = False,
):
    """
    Plot a 2D projection (e.g., MIP) of 3D MSWPT coefficients with projected dyadic boxes.

    The input ``coeffs_array`` is expected to be 2D (already projected), and boxes are
    projected onto the (x, y) plane.

    Returns
    -------
    im : matplotlib.image.AxesImage
        The imshow image handle (useful for colorbars).
    """
    coeffs_mag = np.asarray(np.abs(coeffs_array)).T

    if len(dyadic_decomp.N) < 2:
        raise ValueError(
            "dyadic_decomp.N must have at least two dimensions for 3D plots."
        )

    N0, N1 = dyadic_decomp.N[:2]
    extent_coeffs = [-N1 // 2, N1 // 2, -N0 // 2, N0 // 2]

    c_max = float(coeffs_mag.max())
    if c_max == 0:
        c_max = 1.0
    c_min = c_max * 1e-2

    coeffs_mag_plot = np.maximum(coeffs_mag, c_min)
    w_norm = mcolors.LogNorm(vmin=c_min, vmax=c_max)

    im = ax.imshow(
        coeffs_mag_plot,
        origin="lower",
        extent=extent_coeffs,
        aspect="auto",
        norm=w_norm,
    )

    cumsum_boxes = jnp.r_[0, jnp.cumsum(dyadic_decomp.num_boxes_ndim)]
    box_lengths = dyadic_decomp.box_lengths
    box_aspect = dyadic_decomp.box_aspect_ratio
    colors = ["gray", "darkgray", "silver", "lightgray"]

    for level in range(dyadic_decomp.num_levels):
        start_idx = cumsum_boxes[level]
        end_idx = cumsum_boxes[level + 1]
        centers = dyadic_decomp.centres_ndim[start_idx:end_idx]
        box_length = box_lengths[level]

        box_width_x = box_length * box_aspect[1]
        box_width_y = box_length * box_aspect[0]
        color = colors[level % len(colors)]

        for center in centers:
            cx, cy = float(center[1]), float(center[0])
            rect = patches.Rectangle(
                (cx - box_width_x / 2, cy - box_width_y / 2),
                box_width_x,
                box_width_y,
                linewidth=1.0,
                edgecolor=color,
                facecolor="none",
                linestyle=":",
                alpha=0.6,
            )
            ax.add_patch(rect)

    selector_indices = _mswpt_highlight_indices(
        dyadic_decomp,
        cutoff_freq,
        box_corners,
    )

    if selector_indices is not None and selector_indices.size > 0:
        bounds_x, bounds_y = [], []
        for global_idx in selector_indices:
            level = utils.find_level(dyadic_decomp, int(global_idx))
            center = dyadic_decomp.centres_ndim[int(global_idx)]
            box_length = float(box_lengths[level])

            box_width_x = float(box_length * box_aspect[1])
            box_width_y = float(box_length * box_aspect[0])
            cx, cy = float(center[1]), float(center[0])

            bounds_x.extend([cx - box_width_x / 2, cx + box_width_x / 2])
            bounds_y.extend([cy - box_width_y / 2, cy + box_width_y / 2])

        rect = patches.Rectangle(
            (min(bounds_x), min(bounds_y)),
            max(bounds_x) - min(bounds_x),
            max(bounds_y) - min(bounds_y),
            linewidth=2.0,
            edgecolor="red",
            facecolor="none",
            linestyle="--",
            zorder=5,
        )
        ax.add_patch(rect)

    if asymptote:
        limit = abs(extent_coeffs[1])
        ax.plot(
            [-limit / 2, limit / 2],
            [-2 * limit, 2 * limit],
            color="orange",
            linestyle="-.",
            linewidth=1.5,
            zorder=6,
        )
        ax.plot(
            [-limit / 2, limit / 2],
            [2 * limit, -2 * limit],
            color="orange",
            linestyle="-.",
            linewidth=1.5,
            zorder=6,
        )

    return im
