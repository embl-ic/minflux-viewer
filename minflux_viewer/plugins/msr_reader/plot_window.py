'Plot window'

from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from minflux_viewer.ui.plot_style_dialog import PlotStyleDialog


class PlotWindow(QDialog):
    def __init__(self, data, title: str = "Plot", parent=None, label: str | None = None, mode: str | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(950, 700)
        self._datasets: list[dict[str, Any]] = []
        self._owner = parent
        self._controls: dict[int, dict[str, Any]] = {}

        import pyqtgraph as pg

        self.plot = pg.PlotWidget(background="w")
        self.plot.showGrid(x=True, y=True, alpha=0.22)
        self.plot.getAxis("bottom").setPen("k")
        self.plot.getAxis("left").setPen("k")
        self.plot.getAxis("bottom").setTextPen("k")
        self.plot.getAxis("left").setTextPen("k")
        self.mode = QComboBox(self)
        self.mode.addItems(["line", "scatter", "histogram"])
        if mode in {"line", "scatter", "histogram"}:
            self.mode.setCurrentText(mode)
        self.mode.currentIndexChanged.connect(self._draw)

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Mode:"))
        top.addWidget(self.mode)
        top.addStretch(1)
        layout.addLayout(top)
        layout.addWidget(self.plot)
        self.layer_controls = QGroupBox("Plot layers")
        self.layer_controls_layout = QVBoxLayout(self.layer_controls)
        self.layer_controls_layout.setContentsMargins(6, 6, 6, 6)
        self.layer_controls_layout.setSpacing(4)
        layout.addWidget(self.layer_controls)
        self.add_data(data, label or title, redraw=True)

    @staticmethod
    def _to_series_matrix(a: np.ndarray) -> np.ndarray:
        if a.ndim == 1:
            return a.reshape(-1, 1)
        if a.ndim > 2:
            return a.reshape(a.shape[0], -1)
        r, c = a.shape
        if r in (2, 3) and c not in (2, 3):
            return a.T
        return a

    @staticmethod
    def _finite_bounds(series: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray] | None:
        m = np.asarray(series, dtype=float)
        if m.size == 0:
            return None
        if mode == "scatter" and m.ndim == 2 and m.shape[1] >= 2:
            vals = m[:, :2]
        elif mode == "line":
            vals = m.reshape(m.shape[0], -1)
            vals = np.column_stack([np.arange(vals.shape[0]), np.nanmin(vals, axis=1)])
        else:
            vals = m.reshape(-1, 1)
        vals = vals[np.all(np.isfinite(vals), axis=1)]
        if vals.size == 0:
            return None
        return vals.min(axis=0), vals.max(axis=0)

    def add_data(self, data, label: str, redraw: bool = True) -> None:
        import pyqtgraph as pg

        series = self._to_series_matrix(np.asarray(data))
        idx = len(self._datasets)
        color = pg.intColor(idx, hues=max(idx + 2, 2))
        self._datasets.append({
            "label": label,
            "series": series,
            "visible": True,
            "symbol": "o",
            "size": 6,
            "color": (color.red(), color.green(), color.blue()),
            "alpha": 120,
        })
        self._add_layer_control(idx)
        if redraw:
            self._draw()

    def _add_layer_control(self, idx: int) -> None:
        ds = self._datasets[idx]
        row = QWidget(self.layer_controls)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        visible = QCheckBox(ds["label"])
        visible.setChecked(True)
        visible.toggled.connect(lambda checked, i=idx: self._set_layer_visible(i, checked))
        layout.addWidget(visible, stretch=1)
        style_button = QPushButton("plot style")
        style_button.clicked.connect(lambda _checked=False, i=idx: self._edit_layer_style(i))
        layout.addWidget(style_button)
        self.layer_controls_layout.addWidget(row)
        self._controls[idx] = {"row": row, "visible": visible, "style": style_button}

    def _set_layer_visible(self, idx: int, visible: bool) -> None:
        if 0 <= idx < len(self._datasets):
            self._datasets[idx]["visible"] = bool(visible)
            self._draw()

    def _edit_layer_style(self, idx: int) -> None:
        if not (0 <= idx < len(self._datasets)):
            return
        dlg = PlotStyleDialog(self._datasets[idx], self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._datasets[idx].update(dlg.result_payload())
            self._draw()

    def can_overlay(self, data) -> tuple[bool, str]:
        """Return whether *data* is visually compatible with this plot window."""
        if not self._datasets:
            return True, ""
        mode = self.mode.currentText()
        new_series = self._to_series_matrix(np.asarray(data))
        existing_bounds = [
            self._finite_bounds(ds["series"], mode)
            for ds in self._datasets
        ]
        existing_bounds = [b for b in existing_bounds if b is not None]
        new_bounds = self._finite_bounds(new_series, mode)
        if not existing_bounds or new_bounds is None:
            return True, ""
        old_min = np.min(np.vstack([b[0] for b in existing_bounds]), axis=0)
        old_max = np.max(np.vstack([b[1] for b in existing_bounds]), axis=0)
        new_min, new_max = new_bounds
        old_span = np.maximum(old_max - old_min, 1e-12)
        combined_min = np.minimum(old_min, new_min)
        combined_max = np.maximum(old_max, new_max)
        combined_span = combined_max - combined_min
        if np.any(combined_span > old_span * 100.0):
            return False, "new data range is more than 100x wider than the active plot range"
        new_center = (new_min + new_max) / 2.0
        old_center = (old_min + old_max) / 2.0
        if np.any(np.abs(new_center - old_center) > old_span * 50.0):
            return False, "new data range is far away from the active plot range"
        return True, ""

    def focusInEvent(self, event):
        super().focusInEvent(event)
        if hasattr(self._owner, "_on_plot_window_focused"):
            self._owner._on_plot_window_focused(self)

    def closeEvent(self, event):
        from minflux_viewer.ui.qt_lifecycle import close_plot_widgets

        close_plot_widgets(self.plot)
        if hasattr(self._owner, "_on_plot_window_closed"):
            self._owner._on_plot_window_closed(self)
        super().closeEvent(event)

    def _draw(self):
        import pyqtgraph as pg

        self.plot.clear()
        legend = getattr(self.plot.plotItem, "legend", None)
        if legend is None:
            self.plot.addLegend(offset=(10, 10))
        else:
            legend.clear()
        self.plot.setAspectLocked(False)
        mode = self.mode.currentText()
        if mode == "line":
            for ds_idx, ds in enumerate(self._datasets):
                if not ds.get("visible", True):
                    continue
                m = ds["series"]
                cols = m.shape[1] if m.ndim == 2 else 1
                x = np.arange(m.shape[0])
                for i in range(cols):
                    r, g, b = ds["color"]
                    color = pg.mkColor(r, g, b, min(int(ds["alpha"]) + 70, 255))
                    name = ds["label"] if cols == 1 else f"{ds['label']} [{i}]"
                    self.plot.plot(x, m[:, i], pen=pg.mkPen(color, width=1.5), name=name)
            return
        if mode == "scatter":
            self.plot.setLabel("bottom", "x")
            self.plot.setLabel("left", "y")
            self.plot.setAspectLocked(True)
            for ds_idx, ds in enumerate(self._datasets):
                if not ds.get("visible", True):
                    continue
                m = ds["series"]
                r, g, b = ds["color"]
                brush = pg.mkBrush(r, g, b, int(ds["alpha"]))
                pen = pg.mkPen(r, g, b, min(int(ds["alpha"]) + 70, 255))
                if m.shape[1] >= 2:
                    x, y = m[:, 0], m[:, 1]
                else:
                    x, y = np.arange(m.shape[0]), m[:, 0]
                self.plot.plot(
                    x, y, pen=None, symbol=ds["symbol"], symbolSize=int(ds["size"]),
                    symbolBrush=brush, symbolPen=pen, name=ds["label"],
                )
            return
        self.plot.setLabel("bottom", "value")
        self.plot.setLabel("left", "count")
        for ds_idx, ds in enumerate(self._datasets):
            if not ds.get("visible", True):
                continue
            m = ds["series"]
            cols = m.shape[1] if m.ndim == 2 else 1
            for i in range(cols):
                values = m[:, i]
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                h, edges = np.histogram(values, bins=50)
                x = 0.5 * (edges[:-1] + edges[1:])
                r, g, b = ds["color"]
                color = pg.mkColor(r, g, b, int(ds["alpha"]))
                name = ds["label"] if cols == 1 else f"{ds['label']} [{i}]"
                self.plot.plot(
                    x, h, stepMode=False, fillLevel=0,
                    brush=pg.mkBrush(color), pen=pg.mkPen(color, width=1.2), name=name,
                )
