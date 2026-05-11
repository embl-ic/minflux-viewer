"""
minflux_viewer.ui.histogram_window
====================================
Attribute histogram window.

Shows a histogram of any numeric attribute. This window is intentionally
view-only: histogram range controls do not filter the dataset.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..core.app_state import AppState

_TRACE_AGG_MODES = [
    "trace mean",
    "trace median",
    "trace min",
    "trace max",
    "trace stdev",
    "trace range",
]


class HistogramWindow(QWidget):
    """Interactive attribute histogram without dataset filtering."""

    TAG = "histogram_window"

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._vals: np.ndarray = np.empty(0)
        self._auto_bin_width: float | None = None
        self._resetting_plot = False
        self._last_log_warning_key: tuple | None = None

        self.setWindowTitle("Histogram")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(700, 500)
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

        bar.addWidget(QLabel("Attribute:"))
        self._attr_combo = QComboBox()
        self._attr_combo.setMinimumWidth(110)
        self._attr_combo.currentTextChanged.connect(self._on_attr_changed)
        bar.addWidget(self._attr_combo)

        bar.addWidget(QLabel("As:"))
        self._agg_combo = QComboBox()
        self._agg_combo.currentTextChanged.connect(self._on_attr_changed)
        bar.addWidget(self._agg_combo)

        bar.addWidget(QLabel("Bin size:"))
        self._bin_spin = QDoubleSpinBox()
        self._bin_spin.setDecimals(6)
        self._bin_spin.setRange(1e-12, 1e12)
        self._bin_spin.valueChanged.connect(self._on_bin_changed)
        bar.addWidget(self._bin_spin)

        self._zero_chk = QCheckBox("Hide zeros")
        self._zero_chk.stateChanged.connect(self._on_attr_changed)
        bar.addWidget(self._zero_chk)

        self._log_chk = QCheckBox("Log(data)")
        self._log_chk.setToolTip("Plot the histogram of natural log(data); values <= 0 are removed.")
        self._log_chk.stateChanged.connect(self._on_attr_changed)
        bar.addWidget(self._log_chk)

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._reset_histogram)
        bar.addWidget(reset_btn)

        bar.addStretch()
        root.addLayout(bar)

        pg.setConfigOptions(antialias=True)
        self._plot = pg.PlotWidget(background="w")
        self._plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._plot.setLabel("left", "count")
        self._plot.showGrid(x=False, y=True, alpha=0.2)

        self._hist_item = pg.BarGraphItem(x=[], height=[], width=1, brush="steelblue")
        self._plot.addItem(self._hist_item)
        root.addWidget(self._plot)

        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._info_label)

    def _refresh(self) -> None:
        ds = self._state.active_dataset
        if ds is None:
            self.setWindowTitle("Histogram")
            self._hist_item.setOpts(x=[], height=[], width=1)
            return

        self.setWindowTitle(f"Histogram  —  {ds.name}")
        self._populate_agg_modes()

        old = self._attr_combo.currentText()
        self._attr_combo.blockSignals(True)
        self._attr_combo.clear()
        numeric = [
            k for k in ds.prop.attr_names
            if k not in ("ftr", "idx")
            and np.issubdtype(np.asarray(ds.attr[k]).dtype, np.number)
            and np.asarray(ds.attr[k]).ndim == 1
        ]
        self._attr_combo.addItems(numeric)
        if old in numeric:
            self._attr_combo.setCurrentText(old)
        elif "efo" in numeric:
            self._attr_combo.setCurrentText("efo")
        self._attr_combo.blockSignals(False)

        self._auto_bin_width = None
        self._draw()

    def _populate_agg_modes(self) -> None:
        old = self._agg_combo.currentText()
        enabled = self._state.prefs.get("plot", {}).get("histogram_values", ["trace mean"])
        modes = ["per loc"] + [mode for mode in _TRACE_AGG_MODES if mode in enabled]
        self._agg_combo.blockSignals(True)
        self._agg_combo.clear()
        self._agg_combo.addItems(modes)
        self._agg_combo.setCurrentText(old if old in modes else "per loc")
        self._agg_combo.blockSignals(False)

    def _draw(self) -> None:
        ds = self._state.active_dataset
        if ds is None or self._attr_combo.count() == 0:
            return

        attr_name = self._attr_combo.currentText()
        agg_mode = self._agg_combo.currentText()
        raw = np.asarray(ds.attr.get(attr_name, np.empty(0))).ravel().astype(float)
        if raw.size == 0:
            return

        vals = self._aggregate(raw, ds.filter_mask, agg_mode, ds)
        vals = vals[np.isfinite(vals)]
        original_value_count = vals.size
        if self._log_chk.isChecked():
            positive_mask = vals > 0.0
            removed = int(vals.size - np.count_nonzero(positive_mask))
            vals = vals[positive_mask]
            if removed:
                self._warn_log_filtered_values(ds.name, attr_name, agg_mode, removed, original_value_count)
            vals = np.log(vals)
        if self._zero_chk.isChecked():
            vals = vals[vals != 0.0]
        if vals.size == 0:
            self._hist_item.setOpts(x=[], height=[], width=1)
            self._info_label.setText("No histogram values.")
            return

        self._vals = vals
        if self._auto_bin_width is None:
            bin_width = self._default_bin_width(vals)
            self._set_bin_spin(bin_width)
        else:
            bin_width = float(self._bin_spin.value())

        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax <= vmin:
            vmax = vmin + max(bin_width, 1.0)
        n_bins = max(1, min(int(np.ceil((vmax - vmin) / max(bin_width, 1e-12))), 4096))
        counts, edges = np.histogram(vals, bins=n_bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]

        self._hist_item.setOpts(x=centers, height=counts, width=width * 0.95)
        x_label = f"log({attr_name})" if self._log_chk.isChecked() else attr_name
        self._plot.setLabel("bottom", f"{x_label}  [{agg_mode}]")
        self._info_label.setText(
            f"{int(ds.filter_mask.sum()):,} / {ds.prop.num_loc:,} localisations  |  "
            f"{vals.size:,} histogram values  |  bin size {bin_width:.6g}"
        )

    def _warn_log_filtered_values(
        self,
        dataset_name: str,
        attr_name: str,
        agg_mode: str,
        removed: int,
        total: int,
    ) -> None:
        key = (self._state.active_idx, dataset_name, attr_name, agg_mode, removed, total)
        if key == self._last_log_warning_key:
            return
        self._last_log_warning_key = key
        msg = (
            f"Histogram log(data) removed {removed:,} non-positive value(s) "
            f"from '{attr_name}' [{agg_mode}] in dataset '{dataset_name}' "
            f"before applying the natural logarithm."
        )
        from .console_window import ConsoleWindow
        ConsoleWindow.write_app_message(f"WARN: {msg}", is_err=True, show_on_error=True)
        self._state.log(msg, "WARN")

    def _aggregate(self, raw: np.ndarray, ftr: np.ndarray, mode: str, ds) -> np.ndarray:
        if mode == "per loc":
            return raw[ftr]

        ti = ds.prop.trace_idx
        n_tr = ds.prop.num_traces

        def trace_agg(fn) -> np.ndarray:
            return np.array([fn(raw[ti[i, 0] : ti[i, 1] + 1]) for i in range(n_tr)])

        if mode == "trace mean":
            return trace_agg(np.nanmean)
        if mode == "trace median":
            return trace_agg(np.nanmedian)
        if mode == "trace min":
            return trace_agg(np.nanmin)
        if mode == "trace max":
            return trace_agg(np.nanmax)
        if mode == "trace stdev":
            return trace_agg(np.nanstd)
        if mode == "trace range":
            return trace_agg(lambda a: float(np.nanmax(a)) - float(np.nanmin(a)))
        return raw[ftr]

    def _default_bin_width(self, vals: np.ndarray) -> float:
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            return 1.0
        q25, q75 = np.percentile(vals, [25, 75])
        iqr = float(q75 - q25)
        if iqr > 0:
            width = 2.0 * iqr / np.cbrt(vals.size)
        else:
            width = float(np.ptp(vals)) / max(np.sqrt(vals.size), 1.0)
        return max(width, np.finfo(float).eps)

    def _set_bin_spin(self, value: float) -> None:
        self._bin_spin.blockSignals(True)
        self._bin_spin.setValue(float(value))
        self._bin_spin.setSingleStep(max(float(value) / 10.0, 1e-12))
        self._bin_spin.blockSignals(False)

    def _reset_histogram(self) -> None:
        self._auto_bin_width = None
        self._draw()
        self._plot.autoRange()

    def _reset_for_new_data(self) -> None:
        if self._resetting_plot:
            return
        self._resetting_plot = True
        try:
            self._auto_bin_width = None
            self._draw()
            self._plot.autoRange()
        finally:
            self._resetting_plot = False

    def _on_bin_changed(self) -> None:
        self._auto_bin_width = float(self._bin_spin.value())
        self._draw()

    def _on_attr_changed(self) -> None:
        self._reset_for_new_data()

    def _on_active_changed(self, _idx: int) -> None:
        self._refresh()

    def _on_filter_changed(self, idx: int) -> None:
        if idx == self._state.active_idx:
            self._reset_for_new_data()
