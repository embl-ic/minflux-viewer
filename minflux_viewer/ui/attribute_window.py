"""
minflux_viewer.ui.attribute_window
=====================================
Attribute plot window — port of ``plot_attribute.m``.

Plots two numeric attributes against each other (X vs Y), with optional
colouring by a third attribute.  The most common usage is plotting one
attribute over time (X = ``tim``) or index (X = ``idx``).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..core.app_state import AppState
from ..core.attributes import plot_attribute_names
from ..core.roi_selection import rectangle_mask


class AttributeWindow(QWidget):
    """Plot any two numeric attributes against each other."""

    TAG = "attribute_window"

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._view_state_key = "attribute_plot_state"

        self.setWindowTitle("Attribute Plot")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(700, 400)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._roi_overlay = None

        self._build_ui()
        self._refresh()

        state.active_changed.connect(self._on_active_changed)
        state.filter_changed.connect(self._on_filter_changed)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── Control row ───────────────────────────────────────────
        bar = QHBoxLayout()
        bar.setSpacing(8)

        bar.addWidget(QLabel("X:"))
        self._x_combo = QComboBox()
        self._x_combo.setMinimumWidth(100)
        self._x_combo.currentTextChanged.connect(self._draw)
        bar.addWidget(self._x_combo)

        bar.addWidget(QLabel("Y:"))
        self._y_combo = QComboBox()
        self._y_combo.setMinimumWidth(100)
        self._y_combo.currentTextChanged.connect(self._draw)
        bar.addWidget(self._y_combo)

        self._lines_chk = QCheckBox("Lines")
        self._lines_chk.setChecked(False)
        self._lines_chk.stateChanged.connect(self._draw)
        bar.addWidget(self._lines_chk)

        self._filter_chk = QCheckBox("Filtered only")
        self._filter_chk.setChecked(True)
        self._filter_chk.stateChanged.connect(self._draw)
        bar.addWidget(self._filter_chk)

        bar.addStretch()
        root.addLayout(bar)

        # ── Plot ─────────────────────────────────────────────────
        pg.setConfigOptions(antialias=True)
        self._plot = pg.PlotWidget(background="w")
        self._plot.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._plot.showGrid(x=True, y=True, alpha=0.2)

        self._curve  = self._plot.plot(pen=pg.mkPen("steelblue", width=1))
        self._points = pg.ScatterPlotItem(size=3, pen=None, brush="steelblue")
        self._plot.addItem(self._points)
        from .roi_overlay import RoiOverlayController
        self._roi_overlay = RoiOverlayController(
            self._state.rois,
            self,
            self._plot,
            self._plot.getPlotItem(),
            coordinate_space="plot",
        )

        root.addWidget(self._plot)

        # ── Info ─────────────────────────────────────────────────
        self._info = QLabel("")
        self._info.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._info)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        ds = self._state.active_dataset
        if ds is None:
            self.setWindowTitle("Attribute Plot")
            self._curve.setData([], [])
            self._points.setData([], [])
            return

        self.setWindowTitle(f"Attribute Plot  —  {ds.name}")

        numeric = plot_attribute_names(ds, self._state.prefs)
        saved = ds.state.get(self._view_state_key, {})

        for combo in (self._x_combo, self._y_combo):
            old = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(numeric)
            if old in numeric:
                combo.setCurrentText(old)
            combo.blockSignals(False)

        x_default = saved.get("x", "idx")
        y_default = saved.get("y", "efo")
        self._lines_chk.blockSignals(True)
        self._filter_chk.blockSignals(True)
        self._lines_chk.setChecked(bool(saved.get("lines", False)))
        self._filter_chk.setChecked(bool(saved.get("filtered_only", True)))
        self._lines_chk.blockSignals(False)
        self._filter_chk.blockSignals(False)
        if x_default in numeric:
            self._x_combo.setCurrentText(x_default)
        elif "idx" in numeric:
            self._x_combo.setCurrentText("idx")
        elif self._x_combo.currentText() == "" and "tim" in numeric:
            self._x_combo.setCurrentText("tim")
        if y_default in numeric:
            self._y_combo.setCurrentText(y_default)
        elif "efo" in numeric:
            self._y_combo.setCurrentText("efo")

        self._draw()

    def _draw(self) -> None:
        ds = self._state.active_dataset
        if ds is None or self._x_combo.count() == 0:
            return

        x_name = self._x_combo.currentText()
        y_name = self._y_combo.currentText()
        if not x_name or not y_name:
            return
        ds.state[self._view_state_key] = {
            "x": x_name,
            "y": y_name,
            "lines": self._lines_chk.isChecked(),
            "filtered_only": self._filter_chk.isChecked(),
        }

        ftr = ds.filter_mask if self._filter_chk.isChecked() else np.ones(ds.prop.num_loc, bool)

        x = np.asarray(ds.attr.get(x_name, np.empty(0))).ravel().astype(float)[ftr]
        y = np.asarray(ds.attr.get(y_name, np.empty(0))).ravel().astype(float)[ftr]

        n = min(x.size, y.size)
        if n == 0:
            self._curve.setData([], [])
            self._points.setData([], [])
            return

        x, y = x[:n], y[:n]

        use_lines = self._lines_chk.isChecked()
        if use_lines:
            self._curve.setData(x, y)
            self._points.setData([], [])
        else:
            self._curve.setData([], [])
            # Downsample for performance if very large
            if n > 50_000:
                step = n // 50_000
                self._points.setData(x[::step], y[::step])
            else:
                self._points.setData(x, y)

        self._plot.setLabel("bottom", x_name)
        self._plot.setLabel("left", y_name)
        self._info.setText(f"{n:,} points")

    def compute_roi_selection(self, record):
        if record.type != "rectangle":
            return None
        ds = self._state.active_dataset
        if ds is None:
            return None
        x_name = self._x_combo.currentText()
        y_name = self._y_combo.currentText()
        if not x_name or not y_name:
            return None
        x = np.asarray(ds.attr.get(x_name, np.empty(0))).ravel().astype(float)
        y = np.asarray(ds.attr.get(y_name, np.empty(0))).ravel().astype(float)
        n = min(x.size, y.size, ds.prop.num_loc)
        if n == 0:
            return None
        base = np.ones(n, dtype=bool)
        if self._filter_chk.isChecked():
            ftr = np.asarray(ds.filter_mask, dtype=bool).ravel()
            if ftr.size == n:
                base = ftr.copy()
        mask = rectangle_mask(x[:n], y[:n], record, base_mask=base)
        if mask.size != ds.prop.num_loc:
            full = np.zeros(ds.prop.num_loc, dtype=bool)
            full[:mask.size] = mask
            mask = full
        context = {
            "source_view": "attribute",
            "dataset_idx": self._state.active_idx,
            "x_attr": x_name,
            "y_attr": y_name,
            "filtered_only": self._filter_chk.isChecked(),
        }
        return ds, mask, context

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_active_changed(self, _idx: int) -> None:
        self._refresh()

    def _on_filter_changed(self, idx: int) -> None:
        if idx == self._state.active_idx:
            self._draw()

    def focusInEvent(self, event) -> None:
        if self._roi_overlay is not None:
            self._roi_overlay.activate()
        super().focusInEvent(event)

    def changeEvent(self, event) -> None:
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            if self._roi_overlay is not None:
                self._roi_overlay.activate()
        super().changeEvent(event)
