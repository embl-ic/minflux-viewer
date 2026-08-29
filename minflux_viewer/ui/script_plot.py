"""Modeless pyqtgraph windows for the public ``mfv.plot`` API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from ..colormaps import make_colormap
from .plot_format import ScientificAxisItem, plot_widget

_PEN_STYLES = {
    "-": Qt.PenStyle.SolidLine,
    "--": Qt.PenStyle.DashLine,
    ":": Qt.PenStyle.DotLine,
    "-.": Qt.PenStyle.DashDotLine,
}


class _ScriptAxisItem(ScientificAxisItem):
    """Scientific app axis that also honours pyqtgraph's logarithmic ticks."""

    def tickStrings(self, values, scale, spacing):  # noqa: N802 - pyqtgraph API
        if self.logMode:
            return self.logTickStrings(values, scale, spacing)
        return super().tickStrings(values, scale, spacing)


class ScriptPlotWindow(QWidget):
    """One non-owned plot or image window, created for a script/plugin."""

    def __init__(self, *, title: str = "Script Plot", image: bool = False) -> None:
        super().__init__(None)
        self._owner = None
        self._image_mode = bool(image)
        self._series: list[Any] = []
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.resize(760, 560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        if self._image_mode:
            axes = {
                "bottom": _ScriptAxisItem("bottom"),
                "left": _ScriptAxisItem("left"),
            }
            self.plot_item = pg.PlotItem(axisItems=axes)
            # ImageView inspects the supplied ImageItem during construction, so
            # seed it with a real array rather than leaving ``image`` as None.
            image_item = pg.ImageItem(np.zeros((1, 1)), axisOrder="row-major")
            self.image_view = pg.ImageView(view=self.plot_item, imageItem=image_item)
            self.plot_widget = None
            layout.addWidget(self.image_view)
        else:
            axes = {
                "bottom": _ScriptAxisItem("bottom"),
                "left": _ScriptAxisItem("left"),
            }
            self.plot_widget = plot_widget(background="w", axisItems=axes)
            self.plot_item = self.plot_widget.getPlotItem()
            self.image_view = None
            layout.addWidget(self.plot_widget)

        self.legend_item = self.plot_item.addLegend()
        self.legend_item.hide()

    def set_owner(self, owner: QWidget | None) -> None:
        """Remember the lifetime owner without making it a QWidget parent."""
        self._owner = owner

    def set_title(self, title: str | None) -> None:
        if title is None:
            return
        text = str(title)
        self.setWindowTitle(text or "Script Plot")
        self.plot_item.setTitle(text)

    def add_series(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        label: str | None,
        color: Any,
        style: str,
        width: float,
        symbol: str | None,
        symbol_size: float,
        line: bool = True,
        symbol_brush: Any = None,
    ):
        if color is None:
            color = pg.intColor(len(self._series), hues=12)
        pen = pg.mkPen(color=color, width=width)
        pen.setStyle(_PEN_STYLES[style])
        brush = symbol_brush if symbol_brush is not None else pg.mkBrush(color)
        item = self.plot_item.plot(
            x,
            y,
            name=label,
            pen=pen if line else None,
            symbol=symbol,
            symbolSize=symbol_size,
            symbolPen=None if symbol is None else pg.mkPen(color),
            symbolBrush=None if symbol is None else brush,
        )
        self._series.append(item)
        return item

    def add_histogram(
        self,
        edges: np.ndarray,
        counts: np.ndarray,
        *,
        label: str | None,
    ) -> None:
        color = pg.intColor(len(self._series), hues=12)
        item = self.plot_item.plot(
            edges,
            counts,
            name=label,
            stepMode="center",
            fillLevel=0,
            pen=pg.mkPen(color, width=1.5),
            brush=pg.mkBrush(color.red(), color.green(), color.blue(), 90),
        )
        self._series.append(item)

    def set_image(
        self,
        array: np.ndarray,
        *,
        extent: tuple[float, float, float, float] | None,
        colormap: str,
    ) -> None:
        if self.image_view is None:
            raise RuntimeError("This script plot is not an image window.")
        self.image_view.setImage(array, autoRange=True, autoLevels=True)
        self.image_view.setColorMap(make_colormap(colormap))
        if extent is not None:
            x0, y0, x1, y1 = extent
            self.image_view.getImageItem().setRect(QRectF(x0, y0, x1 - x0, y1 - y0))
            self.plot_item.enableAutoRange()
            self.plot_item.autoRange()

    def set_labels(
        self,
        *,
        x: str | None,
        y: str | None,
        title: str | None,
    ) -> None:
        if x is not None:
            self.plot_item.setLabel("bottom", str(x))
        if y is not None:
            self.plot_item.setLabel("left", str(y))
        self.set_title(title)

    def show_legend(self, show: bool) -> None:
        self.legend_item.setVisible(bool(show))

    def set_log_mode(self, *, x: bool, y: bool) -> None:
        self.plot_item.setLogMode(x=bool(x), y=bool(y))

    def show_grid(self, show: bool, *, alpha: float) -> None:
        self.plot_item.showGrid(x=bool(show), y=bool(show), alpha=float(alpha))

    def set_ranges(
        self,
        *,
        x: tuple[float, float] | None,
        y: tuple[float, float] | None,
    ) -> None:
        if x is not None:
            self.plot_item.setXRange(*x, padding=0)
        if y is not None:
            self.plot_item.setYRange(*y, padding=0)

    def clear_series(self) -> None:
        for item in self._series:
            try:
                self.plot_item.removeItem(item)
            except RuntimeError:
                pass
        self._series.clear()
        try:
            self.legend_item.clear()
        except (AttributeError, RuntimeError):
            pass

    def save_png(self, path: str, *, width: int | None = None) -> str:
        target = Path(path).expanduser()
        exporter = pyqtgraph.exporters.ImageExporter(self.plot_item)
        if width is not None:
            exporter.parameters()["width"] = int(width)
        exporter.export(str(target))
        return str(target)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self.image_view is not None:
            from .qt_lifecycle import close_image_views

            close_image_views(self.image_view)
        elif self.plot_widget is not None:
            from .qt_lifecycle import close_plot_widgets

            close_plot_widgets(self.plot_widget)
        super().closeEvent(event)
