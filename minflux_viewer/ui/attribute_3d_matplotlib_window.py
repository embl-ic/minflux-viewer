"""
Experimental matplotlib-backed 3D attribute plot.

This intentionally mirrors the regular Attribute Plot controls, but uses a
matplotlib Qt canvas and a 3D axes for testing MATLAB-like ``plot3`` behavior.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, ScalarFormatter
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
from ..core.loader import attr_values_1d

_MAX_MPL_POINTS = 50_000


class _OneDecimalScalarFormatter(ScalarFormatter):
    """ScalarFormatter with math-text scientific notation and 1 decimal."""

    def _set_format(self) -> None:
        self.format = "%1.1f"
        if self._usetex or self._useMathText:
            self.format = rf"$\mathdefault{{{self.format}}}$"


def _axis_formatter() -> _OneDecimalScalarFormatter:
    formatter = _OneDecimalScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 4))
    formatter.set_useOffset(False)
    return formatter


class Attribute3DMatplotlibWindow(QWidget):
    """Plot three numeric attributes as an experimental matplotlib 3D scatter."""

    TAG = "attribute_3d_matplotlib_window"

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._view_state_key = "attribute_plot_3d_matplotlib_state"

        self.setWindowTitle("Attribute Plot 3D (Matplotlib)")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(780, 620)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._build_ui()
        self._refresh()

        state.active_changed.connect(self._on_active_changed)
        state.filter_changed.connect(self._on_filter_changed)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

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

        bar.addWidget(QLabel("Z:"))
        self._z_combo = QComboBox()
        self._z_combo.setMinimumWidth(100)
        self._z_combo.currentTextChanged.connect(self._draw)
        bar.addWidget(self._z_combo)

        self._filter_chk = QCheckBox("Filtered only")
        self._filter_chk.setChecked(True)
        self._filter_chk.stateChanged.connect(self._draw)
        bar.addWidget(self._filter_chk)

        bar.addStretch()
        root.addLayout(bar)

        self._figure = Figure(figsize=(6, 5), constrained_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        root.addWidget(self._toolbar)
        root.addWidget(self._canvas, stretch=1)

        self._info = QLabel("")
        self._info.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._info)

    def _refresh(self) -> None:
        ds = self._state.active_dataset
        if ds is None:
            self.setWindowTitle("Attribute Plot 3D (Matplotlib)")
            self._figure.clear()
            self._canvas.draw_idle()
            return

        self.setWindowTitle(f"Attribute Plot 3D (Matplotlib)  —  {ds.name}")
        numeric = plot_attribute_names(ds, self._state.prefs)
        saved = ds.state.get(self._view_state_key, {})

        for combo in (self._x_combo, self._y_combo, self._z_combo):
            old = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(numeric)
            if old in numeric:
                combo.setCurrentText(old)
            combo.blockSignals(False)

        self._filter_chk.blockSignals(True)
        self._filter_chk.setChecked(bool(saved.get("filtered_only", True)))
        self._filter_chk.blockSignals(False)

        self._set_combo_default(self._x_combo, saved.get("x", "idx"), numeric, fallback="idx")
        self._set_combo_default(self._y_combo, saved.get("y", "efo"), numeric, fallback="efo")
        self._set_combo_default(self._z_combo, saved.get("z", "cfr"), numeric, fallback="cfr")
        self._draw()

    def _draw(self) -> None:
        ds = self._state.active_dataset
        if ds is None or self._x_combo.count() == 0:
            return

        x_name = self._x_combo.currentText()
        y_name = self._y_combo.currentText()
        z_name = self._z_combo.currentText()
        if not x_name or not y_name or not z_name:
            return

        ds.state[self._view_state_key] = {
            "x": x_name,
            "y": y_name,
            "z": z_name,
            "filtered_only": self._filter_chk.isChecked(),
        }

        ftr = ds.filter_mask if self._filter_chk.isChecked() else np.ones(ds.prop.num_loc, dtype=bool)
        def _vals(name):
            v = attr_values_1d(ds, name)
            return np.empty(0) if v is None else np.asarray(v).ravel().astype(float)
        x, y, z = _vals(x_name), _vals(y_name), _vals(z_name)
        n = min(x.size, y.size, z.size, ftr.size)
        if n == 0:
            return

        ftr = np.asarray(ftr[:n], dtype=bool)
        x, y, z = x[:n][ftr], y[:n][ftr], z[:n][ftr]
        keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x, y, z = x[keep], y[keep], z[keep]
        n_visible = x.size
        if n_visible > _MAX_MPL_POINTS:
            step = int(np.ceil(n_visible / _MAX_MPL_POINTS))
            x, y, z = x[::step], y[::step], z[::step]

        self._figure.clear()
        ax = self._figure.add_subplot(111, projection="3d")
        ax.scatter(x, y, z, s=2.0, c="#1f77b4", alpha=0.65, depthshade=False)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_zlabel(z_name)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_formatter(_axis_formatter())
            axis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(True, alpha=0.25)
        self._canvas.draw_idle()

        if n_visible > x.size:
            self._info.setText(f"showing {x.size:,} / {n_visible:,} filtered points")
        else:
            self._info.setText(f"{n_visible:,} filtered points")

    @staticmethod
    def _set_combo_default(combo: QComboBox, preferred: str, names: list[str], *, fallback: str) -> None:
        if preferred in names:
            combo.setCurrentText(preferred)
        elif fallback in names:
            combo.setCurrentText(fallback)
        elif combo.count() > 0:
            combo.setCurrentIndex(0)

    def _on_active_changed(self, _idx: int) -> None:
        self._refresh()

    def _on_filter_changed(self, idx: int) -> None:
        if idx == self._state.active_idx:
            self._draw()
