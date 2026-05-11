"""
minflux_viewer.ui.brightness_contrast_dialog
=============================================
ImageJ/Fiji-style Brightness & Contrast dialog.

Two sliders control the low and high display levels that map to the
colormap limits. Moving them live-updates the image via the
``on_levels_changed(lo, hi)`` callback.
"""

from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


# Slider resolution (integer range); mapped linearly onto the data range.
_SLIDER_RES = 1000


class BrightnessContrastDialog(QDialog):
    """
    Fiji-style B&C dialog with Min / Max sliders.

    Callbacks:
      on_levels_changed(lo, hi) — called live whenever sliders move
      on_auto()                 — called when 'Auto' is clicked
      on_reset()                — called when 'Reset' is clicked
    """

    def __init__(
        self,
        on_levels_changed: Callable[[float, float], None],
        on_auto: Callable[[bool], None] | None = None,
        on_reset: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_levels = on_levels_changed
        self._on_auto   = on_auto
        self._on_reset  = on_reset

        self._data_min = 0.0
        self._data_max = 1.0
        self._lo       = 0.0
        self._hi       = 1.0

        self.setWindowTitle("Brightness / Contrast")
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.resize(360, 170)
        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)

        # Min row
        grid.addWidget(QLabel("Min"), 0, 0)
        self._min_slider = QSlider(Qt.Orientation.Horizontal)
        self._min_slider.setRange(0, _SLIDER_RES)
        self._min_slider.setValue(0)
        self._min_slider.valueChanged.connect(self._on_slider_changed)
        grid.addWidget(self._min_slider, 0, 1)

        self._min_spin = QDoubleSpinBox()
        self._min_spin.setDecimals(3)
        self._min_spin.setRange(-1e12, 1e12)
        self._min_spin.setKeyboardTracking(False)
        self._min_spin.editingFinished.connect(self._on_spin_changed)
        grid.addWidget(self._min_spin, 0, 2)

        # Max row
        grid.addWidget(QLabel("Max"), 1, 0)
        self._max_slider = QSlider(Qt.Orientation.Horizontal)
        self._max_slider.setRange(0, _SLIDER_RES)
        self._max_slider.setValue(_SLIDER_RES)
        self._max_slider.valueChanged.connect(self._on_slider_changed)
        grid.addWidget(self._max_slider, 1, 1)

        self._max_spin = QDoubleSpinBox()
        self._max_spin.setDecimals(3)
        self._max_spin.setRange(-1e12, 1e12)
        self._max_spin.setKeyboardTracking(False)
        self._max_spin.editingFinished.connect(self._on_spin_changed)
        grid.addWidget(self._max_spin, 1, 2)

        root.addLayout(grid)

        # Status
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._info_label)

        # Buttons
        btns = QHBoxLayout()
        self._auto_btn = QPushButton("Auto")
        self._auto_btn.setCheckable(True)
        self._auto_btn.setToolTip(
            "Toggle dynamic auto brightness/contrast (0.35 % saturation, updates with view)"
        )
        self._auto_btn.clicked.connect(self._on_auto_clicked)
        btns.addWidget(self._auto_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Revert to automatic levels on every render")
        reset_btn.clicked.connect(self._on_reset_clicked)
        btns.addWidget(reset_btn)

        btns.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btns.addWidget(close_btn)

        root.addLayout(btns)

    # ------------------------------------------------------------------
    # Public API — called from RenderWindow
    # ------------------------------------------------------------------

    def set_data_range(self, data_min: float, data_max: float) -> None:
        """Tell the dialog the current image's data range."""
        if data_max <= data_min:
            data_max = data_min + 1e-9
        self._data_min = float(data_min)
        self._data_max = float(data_max)
        # Keep spin box range generous so user can push beyond the data range
        span = self._data_max - self._data_min
        self._min_spin.setRange(self._data_min - span, self._data_max + span)
        self._max_spin.setRange(self._data_min - span, self._data_max + span)
        self._min_spin.setSingleStep(span / 100.0 if span > 0 else 1.0)
        self._max_spin.setSingleStep(span / 100.0 if span > 0 else 1.0)
        # If levels are unset, default them to the full data range
        if self._lo == 0.0 and self._hi == 1.0:
            self.set_levels(self._data_min, self._data_max)
        else:
            self._update_controls_from_levels()

    def set_levels(self, lo: float, hi: float) -> None:
        """Externally set the current levels (e.g. from Auto)."""
        if hi <= lo:
            hi = lo + 1e-9
        self._lo, self._hi = float(lo), float(hi)
        self._update_controls_from_levels()

    def set_auto_state(self, checked: bool) -> None:
        """Update the Auto button's checked state without firing the callback."""
        self._auto_btn.blockSignals(True)
        self._auto_btn.setChecked(checked)
        self._auto_btn.blockSignals(False)

    # ------------------------------------------------------------------
    # Internal — slider/spin synchronisation
    # ------------------------------------------------------------------

    def _update_controls_from_levels(self) -> None:
        """Push current (_lo, _hi) to sliders AND spin boxes, suppressing signals."""
        span = self._data_max - self._data_min
        frac_lo = (self._lo - self._data_min) / span if span > 0 else 0.0
        frac_hi = (self._hi - self._data_min) / span if span > 0 else 1.0

        for w in (self._min_slider, self._max_slider, self._min_spin, self._max_spin):
            w.blockSignals(True)

        self._min_slider.setValue(int(round(frac_lo * _SLIDER_RES)))
        self._max_slider.setValue(int(round(frac_hi * _SLIDER_RES)))
        self._min_spin.setValue(self._lo)
        self._max_spin.setValue(self._hi)

        for w in (self._min_slider, self._max_slider, self._min_spin, self._max_spin):
            w.blockSignals(False)

        self._update_info()

    def _update_info(self) -> None:
        self._info_label.setText(
            f"data [{self._data_min:.3g}, {self._data_max:.3g}]   "
            f"levels [{self._lo:.3g}, {self._hi:.3g}]"
        )

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_slider_changed(self, _value: int) -> None:
        span = self._data_max - self._data_min
        lo = self._data_min + (self._min_slider.value() / _SLIDER_RES) * span
        hi = self._data_min + (self._max_slider.value() / _SLIDER_RES) * span
        # Keep lo < hi
        if hi <= lo:
            if self.sender() is self._min_slider:
                lo = max(self._data_min, hi - span / _SLIDER_RES)
            else:
                hi = min(self._data_max, lo + span / _SLIDER_RES)
        self._lo, self._hi = lo, hi
        self._min_spin.blockSignals(True); self._min_spin.setValue(lo); self._min_spin.blockSignals(False)
        self._max_spin.blockSignals(True); self._max_spin.setValue(hi); self._max_spin.blockSignals(False)
        self._update_info()
        self._disable_auto_if_active()
        self._on_levels(lo, hi)

    def _on_spin_changed(self) -> None:
        lo, hi = self._min_spin.value(), self._max_spin.value()
        if hi <= lo:
            hi = lo + 1e-9
            self._max_spin.blockSignals(True); self._max_spin.setValue(hi); self._max_spin.blockSignals(False)
        self._lo, self._hi = lo, hi
        self._update_controls_from_levels()   # re-sync sliders
        self._disable_auto_if_active()
        self._on_levels(lo, hi)

    def _disable_auto_if_active(self) -> None:
        """Uncheck Auto and notify parent when the user manually overrides levels."""
        if self._auto_btn.isChecked():
            self._auto_btn.setChecked(False)
            if self._on_auto is not None:
                self._on_auto(False)

    def _on_auto_clicked(self, checked: bool) -> None:
        if self._on_auto is not None:
            self._on_auto(checked)

    def _on_reset_clicked(self) -> None:
        if self._on_reset is not None:
            self._on_reset()
