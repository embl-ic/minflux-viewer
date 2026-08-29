"""
``mfv.plot`` -- plot windows for script and plugin output.

Implemented by extension-layer **track C**.

Built on pyqtgraph through ``ui/plot_format.py``, so plots match the rest of
the application rather than looking like a debug window. Every constructor
returns a :class:`PlotHandle` whose methods are chainable::

    ctx.plot.line(r, excess).labels(x="distance (nm)", y="obs / null").show()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from ._base import Namespace, _todo

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

_TRACK = "C"

#: Line styles, MATLAB-style, matching ``ui/plot_style_dialog.py``.
LINE_STYLES = ("-", "--", ":", "-.")

#: Marker symbols pyqtgraph accepts.
SYMBOLS = ("o", "s", "t", "d", "+", "x", "p", "h", "star")


class PlotHandle:
    """One plot window. Every method returns self except :meth:`save_png`."""

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
    ) -> "PlotHandle":
        """
        Add a data series. Call repeatedly for a multi-series plot.

        *y* may be omitted to plot *x* against its own index.
        """
        raise _todo(_TRACK, "PlotHandle.series()")

    def labels(
        self,
        *,
        x: str | None = None,
        y: str | None = None,
        title: str | None = None,
    ) -> "PlotHandle":
        """Set axis labels and the plot title. Include units in the label text."""
        raise _todo(_TRACK, "PlotHandle.labels()")

    def legend(self, show: bool = True) -> "PlotHandle":
        """Show or hide the legend. Series without a *label* are omitted from it."""
        raise _todo(_TRACK, "PlotHandle.legend()")

    def log(self, *, x: bool = False, y: bool = False) -> "PlotHandle":
        """Set logarithmic axes."""
        raise _todo(_TRACK, "PlotHandle.log()")

    def grid(self, show: bool = True, *, alpha: float = 0.3) -> "PlotHandle":
        """Show or hide grid lines."""
        raise _todo(_TRACK, "PlotHandle.grid()")

    def range(
        self,
        *,
        x: tuple[float, float] | None = None,
        y: tuple[float, float] | None = None,
    ) -> "PlotHandle":
        """Pin an axis range. Omitted axes keep auto-ranging."""
        raise _todo(_TRACK, "PlotHandle.range()")

    def clear(self) -> "PlotHandle":
        """Remove every series, keeping the window and its labels."""
        raise _todo(_TRACK, "PlotHandle.clear()")

    def show(self) -> "PlotHandle":
        """Show and raise the plot window."""
        raise _todo(_TRACK, "PlotHandle.show()")

    def close(self) -> None:
        """Close the window."""
        raise _todo(_TRACK, "PlotHandle.close()")

    def save_png(self, path: str, *, width: int | None = None) -> str:
        """Write the plot to *path* as a PNG. Returns the path written."""
        raise _todo(_TRACK, "PlotHandle.save_png()")


class Plot(Namespace):
    """Create plot windows."""

    name = "plot"

    def line(
        self,
        x: Any,
        y: Any = None,
        *,
        label: str | None = None,
        title: str | None = None,
    ) -> PlotHandle:
        """A line plot. Add further series with :meth:`PlotHandle.series`."""
        raise _todo(_TRACK, "plot.line()")

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
        raise _todo(_TRACK, "plot.scatter()")

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
        raise _todo(_TRACK, "plot.hist()")

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
        raise _todo(_TRACK, "plot.image()")

    def close_all(self) -> None:
        """Close every plot window this session opened."""
        raise _todo(_TRACK, "plot.close_all()")
