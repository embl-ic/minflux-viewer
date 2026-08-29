"""
``mfv.plot`` -- plot windows for script and plugin output.

Implemented by extension-layer **track C**.

Built on pyqtgraph through ``ui/plot_format.py``, so plots match the rest of
the application rather than looking like a debug window. Every constructor
returns a :class:`PlotHandle` whose methods are chainable::

    ctx.plot.line(r, excess).labels(x="distance (nm)", y="obs / null").show()
"""

from __future__ import annotations

import math
import weakref
from typing import Any

import numpy as np
import pyqtgraph as pg

from ..colormaps import canonical_colormap_name, map_to_rgba
from ._base import ApiError, Namespace

#: Line styles, MATLAB-style, matching ``ui/plot_style_dialog.py``.
LINE_STYLES = ("-", "--", ":", "-.")

#: Marker symbols pyqtgraph accepts.
SYMBOLS = ("o", "s", "t", "d", "+", "x", "p", "h", "star")


class PlotHandle:
    """One plot window. Every method returns self except :meth:`save_png`."""

    def __init__(self, namespace: Plot, *, title: str | None, image: bool = False) -> None:
        from ..ui.script_plot import ScriptPlotWindow

        self._namespace = namespace
        self._closed = False
        self._shown = False
        self._window = ScriptPlotWindow(title=title or "Script Plot", image=image)
        handle_ref = weakref.ref(self)

        def _destroyed(*_args) -> None:
            handle = handle_ref()
            if handle is not None:
                handle._on_destroyed()

        self._window.destroyed.connect(_destroyed)
        self._namespace._retain(self)
        if title is not None:
            self._window.set_title(title)

    def _require_open(self) -> None:
        if self._closed:
            raise ApiError("This script plot has been closed.")

    def _on_destroyed(self) -> None:
        self._window = None
        self._closed = True
        self._namespace._release(self)

    def _add_series(
        self,
        x: Any,
        y: Any = None,
        *,
        label: str | None = None,
        color: Any = None,
        style: str = "-",
        width: float = 1.0,
        symbol: str | None = None,
        symbol_size: float = 6.0,
        line: bool = True,
        symbol_brush: Any = None,
    ) -> PlotHandle:
        self._require_open()
        x_values, y_values = _xy_values(x, y)
        if style not in LINE_STYLES:
            raise ApiError(f"Unknown line style {style!r}; use one of {LINE_STYLES}.")
        if symbol is not None and symbol not in SYMBOLS:
            raise ApiError(f"Unknown marker symbol {symbol!r}; use one of {SYMBOLS}.")
        width = _positive_number(width, "series width")
        symbol_size = _positive_number(symbol_size, "symbol size")
        try:
            if color is not None:
                pg.mkColor(color)
        except Exception:
            raise ApiError(f"Could not interpret plot color {color!r}.") from None
        self._window.add_series(
            x_values,
            y_values,
            label=None if label is None else str(label),
            color=color,
            style=style,
            width=width,
            symbol=symbol,
            symbol_size=symbol_size,
            line=line,
            symbol_brush=symbol_brush,
        )
        return self

    def series(
        self,
        x: Any,
        y: Any = None,
        *,
        label: str | None = None,
        color: Any = None,
        style: str = "-",
        width: float = 1.0,
        symbol: str | None = None,
        symbol_size: float = 6.0,
    ) -> PlotHandle:
        """
        Add a data series. Call repeatedly for a multi-series plot.

        *y* may be omitted to plot *x* against its own index.
        """
        return self._add_series(
            x,
            y,
            label=label,
            color=color,
            style=style,
            width=width,
            symbol=symbol,
            symbol_size=symbol_size,
        )

    def labels(
        self,
        *,
        x: str | None = None,
        y: str | None = None,
        title: str | None = None,
    ) -> PlotHandle:
        """Set axis labels and the plot title. Include units in the label text."""
        self._require_open()
        self._window.set_labels(x=x, y=y, title=title)
        return self

    def legend(self, show: bool = True) -> PlotHandle:
        """Show or hide the legend. Series without a *label* are omitted from it."""
        self._require_open()
        self._window.show_legend(show)
        return self

    def log(self, *, x: bool = False, y: bool = False) -> PlotHandle:
        """Set logarithmic axes."""
        self._require_open()
        self._window.set_log_mode(x=x, y=y)
        return self

    def grid(self, show: bool = True, *, alpha: float = 0.3) -> PlotHandle:
        """Show or hide grid lines."""
        self._require_open()
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ApiError("Grid alpha must be between 0 and 1.")
        self._window.show_grid(show, alpha=alpha)
        return self

    def range(
        self,
        *,
        x: tuple[float, float] | None = None,
        y: tuple[float, float] | None = None,
    ) -> PlotHandle:
        """Pin an axis range. Omitted axes keep auto-ranging."""
        self._require_open()
        x_range = _axis_range(x, "x")
        y_range = _axis_range(y, "y")
        self._window.set_ranges(x=x_range, y=y_range)
        return self

    def clear(self) -> PlotHandle:
        """Remove every series, keeping the window and its labels."""
        self._require_open()
        self._window.clear_series()
        return self

    def show(self) -> PlotHandle:
        """Show and raise the plot window."""
        self._require_open()
        from ..ui.modeless import show_modeless

        owner = self._namespace._main_window()
        self._window.set_owner(owner)
        if not self._shown:
            show_modeless(self._window, owner)
            self._shown = True
        else:
            self._window.show()
            self._window.raise_()
            self._window.activateWindow()
        return self

    def close(self) -> None:
        """Close the window."""
        self._dispose()

    def _dispose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._namespace._release(self)
        window, self._window = self._window, None
        try:
            if window is not None:
                window.close()
        except RuntimeError:
            pass

    def save_png(self, path: str, *, width: int | None = None) -> str:
        """Write the plot to *path* as a PNG. Returns the path written."""
        self._require_open()
        if width is not None:
            try:
                parsed_width = int(width)
            except (TypeError, ValueError):
                raise ApiError("PNG width must be a positive integer.") from None
            if isinstance(width, bool) or parsed_width != width or parsed_width < 1:
                raise ApiError("PNG width must be a positive integer.")
        else:
            parsed_width = None
        return self._window.save_png(path, width=parsed_width)


class Plot(Namespace):
    """Create plot windows."""

    name = "plot"

    def __init__(self, facade: Any) -> None:
        super().__init__(facade)
        self._handles: list[PlotHandle] = []

    def _retain(self, handle: PlotHandle) -> None:
        self._handles.append(handle)

    def _release(self, handle: PlotHandle) -> None:
        try:
            self._handles.remove(handle)
        except ValueError:
            pass

    def line(
        self,
        x: Any,
        y: Any = None,
        *,
        label: str | None = None,
        title: str | None = None,
    ) -> PlotHandle:
        """A line plot. Add further series with :meth:`PlotHandle.series`."""
        x_values, y_values = _xy_values(x, y)
        handle = PlotHandle(self, title=title)
        try:
            handle.series(x_values, y_values, label=label)
        except Exception:
            handle._dispose()
            raise
        return handle

    def scatter(
        self,
        x: Any,
        y: Any,
        *,
        color_by: Any = None,
        label: str | None = None,
        title: str | None = None,
        symbol: str = "o",
        size: float = 5.0,
    ) -> PlotHandle:
        """A scatter plot, optionally coloured by a third value."""
        x_values, y_values = _xy_values(x, y)
        brushes = None
        if color_by is not None:
            color_values = _one_dimensional(color_by, "scatter color_by")
            if color_values.size != x_values.size:
                raise ApiError("scatter color_by must have the same length as x and y.")
            finite = np.isfinite(color_values)
            normalised = np.zeros(color_values.shape, dtype=float)
            if np.any(finite):
                lo = float(np.min(color_values[finite]))
                hi = float(np.max(color_values[finite]))
                if hi > lo:
                    normalised[finite] = (color_values[finite] - lo) / (hi - lo)
                else:
                    normalised[finite] = 0.5
            rgba = map_to_rgba("viridis", normalised)
            rgba[~finite, 3] = 0
            brushes = [pg.mkBrush(*map(int, color)) for color in rgba]
        handle = PlotHandle(self, title=title)
        try:
            handle._add_series(
                x_values,
                y_values,
                label=label,
                symbol=symbol,
                symbol_size=size,
                line=False,
                symbol_brush=brushes,
            )
        except Exception:
            handle._dispose()
            raise
        return handle

    def hist(
        self,
        values: Any,
        *,
        bins: int | None = None,
        bin_width: float | None = None,
        label: str | None = None,
        title: str | None = None,
    ) -> PlotHandle:
        """
        A histogram. Give ``bins`` or ``bin_width``, not both; with neither, the
        bin count is chosen from the data.
        """
        if bins is not None and bin_width is not None:
            raise ApiError("Give histogram bins or bin_width, not both.")
        data = _one_dimensional(values, "histogram values")
        data = data[np.isfinite(data)]
        if data.size == 0:
            raise ApiError("Histogram values contain no finite numbers.")
        if bins is not None:
            if isinstance(bins, bool) or int(bins) != bins or int(bins) < 1:
                raise ApiError("Histogram bins must be a positive integer.")
            edges = np.histogram_bin_edges(data, bins=int(bins))
        elif bin_width is not None:
            width = _positive_number(bin_width, "histogram bin width")
            lo, hi = float(np.min(data)), float(np.max(data))
            if hi == lo:
                edges = np.asarray([lo - width / 2.0, lo + width / 2.0])
            else:
                count = max(1, int(math.ceil((hi - lo) / width)))
                edges = lo + np.arange(count + 1, dtype=float) * width
                if edges[-1] < hi:
                    edges = np.append(edges, edges[-1] + width)
        else:
            edges = np.histogram_bin_edges(data, bins="auto")
        counts, edges = np.histogram(data, bins=edges)
        handle = PlotHandle(self, title=title)
        try:
            handle._window.add_histogram(edges, counts, label=label)
        except Exception:
            handle._dispose()
            raise
        return handle

    def image(
        self,
        array: Any,
        *,
        extent: tuple[float, float, float, float] | None = None,
        colormap: str = "hot",
        title: str | None = None,
    ) -> PlotHandle:
        """
        A 2-D image with the application colormaps and a level histogram.

        *extent* is ``(x0, y0, x1, y1)`` in nm, so the image lands in data
        coordinates rather than pixel indices.
        """
        data = np.asarray(array)
        if data.ndim != 2 or not data.size:
            raise ApiError("plot.image requires a non-empty two-dimensional array.")
        if not np.issubdtype(data.dtype, np.number):
            raise ApiError("plot.image requires numeric array values.")
        image_extent = _image_extent(extent)
        try:
            cmap = canonical_colormap_name(colormap)
        except (KeyError, ValueError):
            raise ApiError(f"Unknown application colormap {colormap!r}.") from None
        handle = PlotHandle(self, title=title, image=True)
        try:
            handle._window.set_image(
                np.asarray(data, dtype=float),
                extent=image_extent,
                colormap=cmap,
            )
        except Exception:
            handle._dispose()
            raise
        return handle

    def close_all(self) -> None:
        """Close every plot window this session opened."""
        handles = list(self._handles)
        self._handles.clear()
        for handle in handles:
            handle._dispose()


def _one_dimensional(values: Any, what: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ApiError(f"{what} must be a numeric sequence.") from None
    if array.ndim != 1:
        raise ApiError(f"{what} must be one-dimensional.")
    return array


def _xy_values(x: Any, y: Any = None) -> tuple[np.ndarray, np.ndarray]:
    if y is None:
        y_values = _one_dimensional(x, "series values")
        x_values = np.arange(y_values.size, dtype=float)
    else:
        x_values = _one_dimensional(x, "series x")
        y_values = _one_dimensional(y, "series y")
    if x_values.size != y_values.size:
        raise ApiError("Plot x and y must have the same length.")
    return x_values, y_values


def _positive_number(value: Any, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ApiError(f"{what} must be a positive number.") from None
    if not math.isfinite(number) or number <= 0:
        raise ApiError(f"{what} must be a positive number.")
    return number


def _axis_range(
    value: tuple[float, float] | None,
    axis: str,
) -> tuple[float, float] | None:
    if value is None:
        return None
    try:
        size = len(value)
    except TypeError:
        raise ApiError(f"The {axis} range must contain exactly two values.") from None
    if size != 2:
        raise ApiError(f"The {axis} range must contain exactly two values.")
    try:
        lo, hi = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        raise ApiError(f"The {axis} range must contain finite numbers.") from None
    if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
        raise ApiError(f"The {axis} range must be finite and increasing.")
    return (lo, hi)


def _image_extent(
    extent: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    if extent is None:
        return None
    try:
        size = len(extent)
    except TypeError:
        raise ApiError("Image extent must be (x0, y0, x1, y1).") from None
    if size != 4:
        raise ApiError("Image extent must be (x0, y0, x1, y1).")
    try:
        x0, y0, x1, y1 = (float(value) for value in extent)
    except (TypeError, ValueError):
        raise ApiError("Image extent must contain finite numbers.") from None
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        raise ApiError("Image extent must contain finite numbers.")
    if x1 <= x0 or y1 <= y0:
        raise ApiError("Image extent must increase in both x and y.")
    return (x0, y0, x1, y1)
