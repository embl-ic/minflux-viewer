"""
minflux_viewer.ui.brightness_contrast_dialog
=============================================
Fiji-style Brightness & Contrast dialog for rendered scalar tiles.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


_SLIDER_RES = 1000


class HistogramPreview(QWidget):
    """Small ImageJ/Fiji-like histogram and transfer-function preview."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._hist = np.zeros(128, dtype=float)
        self._data_min = 0.0
        self._data_max = 1.0
        self._lo = 0.0
        self._hi = 1.0
        self.setMinimumSize(150, 70)

    def set_state(
        self,
        hist: np.ndarray,
        data_min: float,
        data_max: float,
        lo: float,
        hi: float,
    ) -> None:
        self._hist = np.asarray(hist, dtype=float).ravel()
        self._data_min = float(data_min)
        self._data_max = float(data_max if data_max > data_min else data_min + 1.0)
        self._lo = float(lo)
        self._hi = float(hi)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        rect = self.rect().adjusted(1, 1, -2, -2)
        painter.fillRect(rect, QColor("white"))
        painter.setPen(QPen(QColor("black"), 1))
        painter.drawRect(rect)

        if self._hist.size:
            hist = np.log1p(np.maximum(self._hist, 0.0))
            peak = float(hist.max()) if hist.size else 0.0
            if peak > 0.0:
                w = max(rect.width() - 2, 1)
                h = max(rect.height() - 2, 1)
                x0 = rect.left() + 1
                y_base = rect.bottom() - 1
                painter.setPen(QPen(QColor(255, 0, 0), 1))
                for i, value in enumerate(hist):
                    x = x0 + int(round(i * w / max(hist.size - 1, 1)))
                    y = y_base - int(round((float(value) / peak) * h))
                    painter.drawLine(x, y_base, x, y)

        span = self._data_max - self._data_min
        lo_frac = np.clip((self._lo - self._data_min) / span, 0.0, 1.0)
        hi_frac = np.clip((self._hi - self._data_min) / span, 0.0, 1.0)
        x_lo = rect.left() + int(round(lo_frac * rect.width()))
        x_hi = rect.left() + int(round(hi_frac * rect.width()))
        painter.setPen(QPen(QColor("black"), 1))
        painter.drawLine(x_lo, rect.bottom(), x_hi, rect.top())


class BrightnessContrastDialog(QDialog):
    """
    Brightness/contrast dialog with live Min/Max controls.

    Callbacks:
      on_levels_changed(lo, hi) - called live whenever levels change
      on_auto()                 - called when Auto is clicked
      on_reset()                - called when Reset is clicked
    """

    def __init__(
        self,
        on_levels_changed: Callable[[float, float], None],
        on_auto: Callable[[], None] | None = None,
        on_reset: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_levels = on_levels_changed
        self._on_auto = on_auto
        self._on_reset = on_reset

        self._pixels = np.empty(0, dtype=float)
        self._hist = np.zeros(128, dtype=float)
        self._data_min = 0.0
        self._data_max = 1.0
        self._lo = 0.0
        self._hi = 1.0
        self._levels_initialized = False

        self.setWindowTitle("Brightness / Contrast")
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.resize(230, 230)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(4)

        self._histogram = HistogramPreview(self)
        root.addWidget(self._histogram)

        range_row = QHBoxLayout()
        self._data_min_label = QLabel("0")
        self._data_max_label = QLabel("1")
        range_row.addWidget(self._data_min_label)
        range_row.addStretch()
        range_row.addWidget(self._data_max_label)
        root.addLayout(range_row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(2)

        self._min_slider = QSlider(Qt.Orientation.Horizontal)
        self._min_slider.setRange(0, _SLIDER_RES)
        self._min_slider.valueChanged.connect(self._on_slider_changed)
        self._min_spin = self._make_spin()
        self._min_spin.editingFinished.connect(self._on_spin_changed)
        grid.addWidget(QLabel("Minimum"), 0, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self._min_slider, 1, 0)
        grid.addWidget(self._min_spin, 1, 1)

        self._max_slider = QSlider(Qt.Orientation.Horizontal)
        self._max_slider.setRange(0, _SLIDER_RES)
        self._max_slider.valueChanged.connect(self._on_slider_changed)
        self._max_spin = self._make_spin()
        self._max_spin.editingFinished.connect(self._on_spin_changed)
        grid.addWidget(QLabel("Maximum"), 2, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self._max_slider, 3, 0)
        grid.addWidget(self._max_spin, 3, 1)
        root.addLayout(grid)

        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._info_label)

        buttons = QGridLayout()
        self._auto_btn = QPushButton("Auto")
        self._auto_btn.setCheckable(True)
        self._auto_btn.clicked.connect(self._on_auto_clicked)
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._on_reset_clicked)
        set_btn = QPushButton("Set")
        set_btn.clicked.connect(self._on_set_clicked)
        buttons.addWidget(self._auto_btn, 0, 0)
        buttons.addWidget(reset_btn, 0, 1)
        buttons.addWidget(set_btn, 1, 0, 1, 2)
        root.addLayout(buttons)

    def _make_spin(self) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(4)
        spin.setRange(-1e12, 1e12)
        spin.setKeyboardTracking(False)
        return spin

    def set_data(self, pixels: np.ndarray) -> None:
        values = np.asarray(pixels, dtype=float).ravel()
        values = values[np.isfinite(values)]
        self._pixels = values
        if values.size == 0:
            self.set_data_range(0.0, 1.0)
            return
        self.set_data_range(float(values.min()), float(values.max()))

    def set_data_range(self, data_min: float, data_max: float) -> None:
        if data_max <= data_min:
            data_max = data_min + 1.0
        self._data_min = float(data_min)
        self._data_max = float(data_max)
        if self._pixels.size:
            self._hist, _ = np.histogram(
                self._pixels,
                bins=128,
                range=(self._data_min, self._data_max),
            )
        else:
            self._hist = np.zeros(128, dtype=float)

        span = self._data_max - self._data_min
        for spin in (self._min_spin, self._max_spin):
            spin.setRange(self._data_min - span, self._data_max + span)
            spin.setSingleStep(span / 100.0 if span > 0 else 1.0)
        if not self._levels_initialized or self._hi <= self._lo:
            self._lo, self._hi = self._data_min, self._data_max
        self._update_controls_from_levels()

    def set_levels(self, lo: float, hi: float) -> None:
        if hi <= lo:
            hi = lo + 1.0
        self._lo, self._hi = float(lo), float(hi)
        self._levels_initialized = True
        self._update_controls_from_levels()

    def set_auto_state(self, checked: bool) -> None:
        self._auto_btn.blockSignals(True)
        self._auto_btn.setChecked(bool(checked))
        self._auto_btn.blockSignals(False)

    def _update_controls_from_levels(self) -> None:
        span = self._data_max - self._data_min
        frac_lo = (self._lo - self._data_min) / span if span > 0 else 0.0
        frac_hi = (self._hi - self._data_min) / span if span > 0 else 1.0

        for widget in (self._min_slider, self._max_slider, self._min_spin, self._max_spin):
            widget.blockSignals(True)
        self._min_slider.setValue(int(round(np.clip(frac_lo, 0.0, 1.0) * _SLIDER_RES)))
        self._max_slider.setValue(int(round(np.clip(frac_hi, 0.0, 1.0) * _SLIDER_RES)))
        self._min_spin.setValue(self._lo)
        self._max_spin.setValue(self._hi)
        for widget in (self._min_slider, self._max_slider, self._min_spin, self._max_spin):
            widget.blockSignals(False)

        self._data_min_label.setText(f"{self._data_min:.4g}")
        self._data_max_label.setText(f"{self._data_max:.4g}")
        self._info_label.setText(f"levels [{self._lo:.4g}, {self._hi:.4g}]")
        self._histogram.set_state(
            self._hist,
            self._data_min,
            self._data_max,
            self._lo,
            self._hi,
        )

    def _on_slider_changed(self, _value: int) -> None:
        span = self._data_max - self._data_min
        lo = self._data_min + (self._min_slider.value() / _SLIDER_RES) * span
        hi = self._data_min + (self._max_slider.value() / _SLIDER_RES) * span
        if hi <= lo:
            step = max(span / _SLIDER_RES, np.finfo(float).eps)
            if self.sender() is self._min_slider:
                lo = hi - step
            else:
                hi = lo + step
        self._apply_levels(lo, hi)

    def _on_spin_changed(self) -> None:
        lo = float(self._min_spin.value())
        hi = float(self._max_spin.value())
        if hi <= lo:
            hi = lo + np.finfo(float).eps
        self._apply_levels(lo, hi)

    def _apply_levels(self, lo: float, hi: float) -> None:
        self._lo, self._hi = float(lo), float(hi)
        self._levels_initialized = True
        self._update_controls_from_levels()
        self._on_levels(self._lo, self._hi)

    def _on_auto_clicked(self) -> None:
        if self._on_auto is not None:
            self._on_auto()

    def _on_reset_clicked(self) -> None:
        if self._on_reset is not None:
            self._on_reset()

    def _on_set_clicked(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Set Brightness / Contrast")
        layout = QVBoxLayout(dialog)
        grid = QGridLayout()
        layout.addLayout(grid)
        min_spin = self._make_spin()
        max_spin = self._make_spin()
        span = self._data_max - self._data_min
        for spin, value in ((min_spin, self._lo), (max_spin, self._hi)):
            spin.setRange(self._data_min - span, self._data_max + span)
            spin.setValue(float(value))
        grid.addWidget(QLabel("Minimum"), 0, 0)
        grid.addWidget(min_spin, 0, 1)
        grid.addWidget(QLabel("Maximum"), 1, 0)
        grid.addWidget(max_spin, 1, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            lo = float(min_spin.value())
            hi = float(max_spin.value())
            if hi <= lo:
                hi = lo + np.finfo(float).eps
            self._apply_levels(lo, hi)
