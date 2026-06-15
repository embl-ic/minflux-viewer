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
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..core.app_state import AppState
from ..core.attributes import (
    aggregation_description,
    attribute_description,
    is_trace_wise_attribute,
    plot_attribute_names,
)
from ..core.iteration import iteration_labels, ordinal, parse_iteration_label
from ..core.loader import attr_matches_selection, attr_values_1d, mfx_filter_mask, mfx_get
from ..core.roi_selection import rectangle_bounds, value_range_mask
from .filter_dialog import SmartBoundsSpinBox, _filter_spinner_values
from .plot_format import plot_widget

# Distinct per-iteration colours for the "all iterations" overlay.
_ITER_COLORS = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207),
]


def _iter_color(k: int) -> tuple[int, int, int]:
    return _ITER_COLORS[k % len(_ITER_COLORS)]

_TRACE_AGG_MODES = [
    "trace mean",
    "trace median",
    "trace min",
    "trace max",
    "trace stdev",
    "trace range",
]

_COLOR_NAMES = {
    "yellow": QColor(255, 221, 0),
    "red": QColor(220, 40, 40),
    "green": QColor(0, 160, 0),
    "cyan": QColor(0, 180, 220),
    "magenta": QColor(210, 60, 210),
    "white": QColor(255, 255, 255),
    "black": QColor(0, 0, 0),
}


def _named_color(name: str) -> QColor:
    return QColor(_COLOR_NAMES.get(name.strip().lower(), QColor(0, 160, 0)))


def _format_filter_report_number(value: float) -> str:
    if not np.isfinite(value):
        return str(value)
    if value == 0:
        return "0.00"
    abs_value = abs(value)
    if 1e-2 <= abs_value < 1e4:
        return f"{value:.2f}"
    exponent = int(np.floor(np.log10(abs_value)))
    mantissa = value / (10 ** exponent)
    return f"{mantissa:.2f} x 10^{exponent}"


class HistogramWindow(QWidget):
    """Interactive attribute histogram without dataset filtering."""

    TAG = "histogram_window"

    def __init__(self, state: AppState, parent: QWidget | None = None, *, dataset_idx: int | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._dataset_idx = dataset_idx if dataset_idx is not None else state.active_idx
        self._vals: np.ndarray = np.empty(0)
        self._auto_bin_width: float | None = None
        self._resetting_plot = False
        self._last_log_warning_key: tuple | None = None
        self._roi_overlay = None
        self._view_state_key = "histogram_plot_state"
        self._filter_edit: dict | None = None
        self._last_histogram_bounds: tuple[float, float, float, float] | None = None
        self._original_auto_range = None
        self._last_attr_name = ""
        self._last_agg_mode = ""
        self._last_log_data = False

        self.setWindowTitle("Histogram")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(700, 500)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._build_ui()
        self._refresh()

        state.filter_changed.connect(self._on_filter_changed)

    @property
    def dataset_idx(self) -> int | None:
        return self._dataset_idx

    def _dataset(self):
        if self._dataset_idx is None:
            return None
        if not (0 <= self._dataset_idx < len(self._state.datasets)):
            return None
        return self._state.datasets[self._dataset_idx]

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        bar = QHBoxLayout()
        bar.setSpacing(8)

        bar.addWidget(QLabel("Attribute:"))
        self._attr_combo = QComboBox()
        self._attr_combo.setMinimumWidth(110)
        self._attr_combo.currentTextChanged.connect(self._on_histogram_attribute_changed)
        self._attr_combo.currentTextChanged.connect(
            lambda text: self._attr_combo.setToolTip(attribute_description(text))
        )
        bar.addWidget(self._attr_combo)

        bar.addWidget(QLabel("As:"))
        self._agg_combo = QComboBox()
        self._agg_combo.currentTextChanged.connect(self._on_histogram_aggregation_changed)
        self._agg_combo.currentTextChanged.connect(
            lambda text: self._agg_combo.setToolTip(aggregation_description(text))
        )
        bar.addWidget(self._agg_combo)

        self._iter_label = QLabel("Iter:")
        bar.addWidget(self._iter_label)
        self._iter_combo = QComboBox()
        self._iter_combo.setMinimumWidth(96)
        self._iter_combo.currentTextChanged.connect(self._on_histogram_selection_changed)
        bar.addWidget(self._iter_combo)

        self._valid_chk = QCheckBox("Valid only")
        self._valid_chk.setChecked(True)
        self._valid_chk.setToolTip("Show only vld=True localizations. Uncheck to include invalid ones.")
        self._valid_chk.stateChanged.connect(self._on_histogram_selection_changed)
        bar.addWidget(self._valid_chk)

        bar.addWidget(QLabel("Bin size:"))
        self._bin_spin = QDoubleSpinBox()
        self._bin_spin.setDecimals(6)
        self._bin_spin.setRange(1e-12, 1e12)
        self._bin_spin.valueChanged.connect(self._on_bin_changed)
        bar.addWidget(self._bin_spin)

        self._zero_chk = QCheckBox("Hide zeros")
        self._zero_chk.setToolTip("Hide zero value bin (useful when 0 is dominant in data).")
        self._zero_chk.stateChanged.connect(self._on_histogram_display_changed)
        bar.addWidget(self._zero_chk)

        self._log_chk = QCheckBox("Log(data)")
        self._log_chk.setToolTip("Plot the histogram of natural log(data); values <= 0 are removed.")
        self._log_chk.stateChanged.connect(self._on_histogram_log_changed)
        bar.addWidget(self._log_chk)

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._reset_histogram)
        bar.addWidget(reset_btn)

        bar.addStretch()
        root.addLayout(bar)

        pg.setConfigOptions(antialias=True)
        self._plot = plot_widget(background="w")
        self._plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._plot.setLabel("left", "count")
        self._plot.showGrid(x=False, y=True, alpha=0.2)
        self._plot.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self._plot.getPlotItem().setMenuEnabled(False)
        view_box = self._plot.getPlotItem().vb
        try:
            view_box.setMenuEnabled(False)
        except Exception:
            pass
        self._original_auto_range = view_box.autoRange
        view_box.autoRange = lambda *args, **kwargs: self._fit_histogram_view()
        view_box.sigRangeChanged.connect(lambda *_args: self._update_filter_edit_labels())

        self._hist_item = pg.BarGraphItem(x=[], height=[], width=1, brush="steelblue")
        self._plot.addItem(self._hist_item)
        # Per-iteration overlay curves for the raw "all iterations" view.
        self._raw_items: list = []
        self._raw_legend = None
        from .roi_overlay import RoiOverlayController
        self._roi_overlay = RoiOverlayController(
            self._state.rois,
            self,
            self._plot,
            self._plot.getPlotItem(),
            coordinate_space="plot",
        )
        root.addWidget(self._plot)

        self._bottom_bar = QHBoxLayout()
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: gray; font-size: 11px;")
        self._bottom_bar.addWidget(self._info_label, stretch=1)
        self._filter_update_btn = QPushButton("Update Filter")
        self._filter_update_btn.setVisible(False)
        self._filter_update_btn.clicked.connect(self._update_filter_edit)
        self._bottom_bar.addWidget(self._filter_update_btn)
        self._filter_finish_btn = QPushButton("Finish Edit")
        self._filter_finish_btn.setVisible(False)
        self._filter_finish_btn.clicked.connect(self._finish_filter_edit)
        self._bottom_bar.addWidget(self._filter_finish_btn)
        self._filter_cancel_btn = QPushButton("Cancel Edit")
        self._filter_cancel_btn.setVisible(False)
        self._filter_cancel_btn.clicked.connect(self._cancel_filter_edit)
        self._bottom_bar.addWidget(self._filter_cancel_btn)
        root.addLayout(self._bottom_bar)

    def _refresh(self) -> None:
        ds = self._dataset()
        if ds is None:
            self.setWindowTitle("Histogram")
            self._hist_item.setOpts(x=[], height=[], width=1)
            self._last_histogram_bounds = None
            return

        self.setWindowTitle(f"Histogram  —  {ds.name}")
        self._populate_agg_modes()
        saved = ds.state.get(self._view_state_key, {})

        old = self._attr_combo.currentText()
        self._attr_combo.blockSignals(True)
        self._attr_combo.clear()
        numeric = plot_attribute_names(ds, self._state.prefs, exclude=("ftr",))
        if not numeric:
            numeric = [
                key for key in ds.attr.keys()
                if key != "ftr"
                and np.asarray(ds.attr[key]).ndim == 1
                and np.issubdtype(np.asarray(ds.attr[key]).dtype, np.number)
            ]
        self._attr_combo.addItems(numeric)
        self._apply_attribute_combo_tooltips(self._attr_combo)
        attr_default = saved.get("attribute", old)
        if attr_default in numeric:
            self._attr_combo.setCurrentText(attr_default)
        elif old in numeric:
            self._attr_combo.setCurrentText(old)
        elif "cfr" in numeric:
            self._attr_combo.setCurrentText("cfr")
        self._attr_combo.blockSignals(False)
        self._attr_combo.setToolTip(attribute_description(self._attr_combo.currentText()))

        self._agg_combo.blockSignals(True)
        agg_default = saved.get("aggregation", "per loc")
        if self._agg_combo.findText(agg_default) >= 0:
            self._agg_combo.setCurrentText(agg_default)
        else:
            self._agg_combo.setCurrentText("per loc")
        self._agg_combo.blockSignals(False)
        self._agg_combo.setToolTip(aggregation_description(self._agg_combo.currentText()))
        self._enforce_trace_aggregation()

        iter_opts = self._iter_labels(ds)
        self._iter_combo.blockSignals(True)
        self._iter_combo.clear()
        self._iter_combo.addItems(iter_opts)
        default_label = self._default_iter_label(ds)
        saved_label = str(saved.get("iter", "") or "")
        self._iter_combo.setCurrentText(saved_label if saved_label in iter_opts else default_label)
        self._iter_combo.blockSignals(False)
        # Nothing to browse for single-iteration data.
        has_iters = bool(iter_opts)
        self._iter_combo.setVisible(has_iters)
        self._iter_label.setVisible(has_iters)

        self._zero_chk.blockSignals(True)
        self._log_chk.blockSignals(True)
        self._valid_chk.blockSignals(True)
        self._zero_chk.setChecked(bool(saved.get("hide_zeros", False)))
        self._log_chk.setChecked(bool(saved.get("log_data", False)))
        self._valid_chk.setChecked(bool(saved.get("valid_only", True)))
        self._zero_chk.blockSignals(False)
        self._log_chk.blockSignals(False)
        self._valid_chk.blockSignals(False)

        self._auto_bin_width = None
        self._draw()

    def _populate_agg_modes(self) -> None:
        old = self._agg_combo.currentText()
        enabled = self._state.prefs.get("plot", {}).get("histogram_values", ["trace mean"])
        modes = ["per loc"] + [mode for mode in _TRACE_AGG_MODES if mode in enabled]
        self._agg_combo.blockSignals(True)
        self._agg_combo.clear()
        self._agg_combo.addItems(modes)
        self._apply_aggregation_combo_tooltips(self._agg_combo)
        self._agg_combo.setCurrentText(old if old in modes else "per loc")
        self._agg_combo.blockSignals(False)

    def _apply_attribute_combo_tooltips(self, combo: QComboBox) -> None:
        for i in range(combo.count()):
            combo.setItemData(i, attribute_description(combo.itemText(i)), Qt.ItemDataRole.ToolTipRole)

    def _apply_aggregation_combo_tooltips(self, combo: QComboBox) -> None:
        for i in range(combo.count()):
            combo.setItemData(i, aggregation_description(combo.itemText(i)), Qt.ItemDataRole.ToolTipRole)

    # ------------------------------------------------------------------
    # Iteration / validity (Stage B) helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _num_itr(ds) -> int:
        return max(1, int(ds.metadata.get("raw_num_itr", ds.prop.num_itr or 1)))

    def _iter_labels(self, ds) -> list[str]:
        return iteration_labels(self._num_itr(ds))

    def _default_iter_label(self, ds) -> str:
        labels = self._iter_labels(ds)
        return f"last ({ordinal(self._num_itr(ds))})" if labels else ""

    def _selection(self) -> tuple:
        """Return (itr_selector, render_mode) for the current label."""
        return parse_iteration_label(self._iter_combo.currentText())

    def _is_raw_mode(self) -> bool:
        """True unless the view is the ``last`` single series whose rows ARE
        the materialized store.

        The default (ds.attr) path requires ds.attr row alignment with the
        current selection: last+valid for normal loads, last+all-validity for
        ``only_valid_locs=False`` loads (so filter-edit/ROI work there too).
        For all-iteration loads even a ``last`` selection must come from the
        raw store.
        """
        itr_sel, render = self._selection()
        if not (itr_sel == "last" and render == "single"):
            return True
        ds = self._dataset()
        return ds is not None and not attr_matches_selection(
            ds, itr="last", vld_only=self._valid_chk.isChecked(),
        )

    def _clear_raw_items(self) -> None:
        for item in self._raw_items:
            try:
                self._plot.removeItem(item)
            except Exception:
                pass
        self._raw_items = []
        if self._raw_legend is not None:
            try:
                self._raw_legend.scene().removeItem(self._raw_legend)
            except Exception:
                pass
            self._raw_legend = None
        # PlotItem caches its legend; clearing the attribute is required or a
        # later addLegend() returns the detached (invisible) old one.
        try:
            self._plot.getPlotItem().legend = None
        except Exception:
            pass

    def _raw_values(self, ds, attr_name: str, sel, vld_only: bool, agg_mode: str):
        """Return histogram values + unevaluable filter attrs for one iteration."""
        from ..utils.filters import raw_trace_aggregate
        vals = mfx_get(ds, attr_name, itr=sel, vld_only=vld_only)
        if vals is None:
            return np.empty(0), []
        vals = np.asarray(vals).ravel().astype(float)
        unevaluable: list[str] = []
        if agg_mode == "per loc":
            res = mfx_filter_mask(ds, itr=sel, vld_only=vld_only)
            if res is not None:
                fmask, unevaluable = res
                if fmask.shape[0] == vals.shape[0]:
                    vals = vals[fmask]
        else:
            tid = mfx_get(ds, "tid", itr=sel, vld_only=vld_only)
            vals = raw_trace_aggregate(vals, tid, agg_mode)
        return vals, unevaluable

    def _transform_hist_values(self, vals: np.ndarray) -> np.ndarray:
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if self._log_chk.isChecked():
            vals = vals[vals > 0.0]
            vals = np.log(vals)
        if self._zero_chk.isChecked():
            vals = vals[vals != 0.0]
        return vals

    def _draw_raw_mode(self) -> None:
        ds = self._dataset()
        if ds is None:
            return
        # Filter editing / ROI bands require the materialized store; not valid here.
        if self._filter_edit:
            self._clear_filter_edit(restore=False)

        attr_name = self._attr_combo.currentText()
        self._enforce_trace_aggregation()
        agg_mode = self._agg_combo.currentText()
        itr_sel, render = self._selection()
        vld_only = self._valid_chk.isChecked()

        ds.state[self._view_state_key] = {
            "attribute": attr_name,
            "aggregation": agg_mode,
            "hide_zeros": self._zero_chk.isChecked(),
            "log_data": self._log_chk.isChecked(),
            "iter": self._iter_combo.currentText() or "",
            "valid_only": vld_only,
        }

        self._clear_raw_items()

        # Derived attrs (den/dst/...) have no per-iteration value.
        probe_sel = 0 if render == "stacked" else itr_sel
        if mfx_get(ds, attr_name, itr=probe_sel, vld_only=vld_only) is None:
            self._hist_item.setOpts(x=[], height=[], width=1)
            self._last_histogram_bounds = None
            self._info_label.setText(
                f"{attr_name} has no per-iteration values (computed from the last valid iteration)."
            )
            self._fit_histogram_view()
            return

        if render == "stacked":
            self._draw_stacked_histogram(ds, attr_name, agg_mode, vld_only)
        else:
            self._draw_single_raw_histogram(ds, attr_name, agg_mode, itr_sel, vld_only, render)

    def _bin_edges_for(self, vals: np.ndarray) -> tuple[np.ndarray, float]:
        bin_width = (
            float(self._bin_spin.value())
            if self._auto_bin_width is not None
            else self._default_bin_width(vals)
        )
        if self._auto_bin_width is None:
            self._set_bin_spin(bin_width)
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax <= vmin:
            vmax = vmin + max(bin_width, 1.0)
        n_bins = max(1, min(int(np.ceil((vmax - vmin) / max(bin_width, 1e-12))), 4096))
        return np.linspace(vmin, vmax, n_bins + 1), bin_width

    def _draw_single_raw_histogram(self, ds, attr_name, agg_mode, itr_sel, vld_only, render) -> None:
        """One filled-bar series — same look as the default view, different rows."""
        sel = "all" if render == "flatten" else itr_sel
        vals, uneval = self._raw_values(ds, attr_name, sel, vld_only, agg_mode)
        vals = self._transform_hist_values(vals)
        if vals.size == 0:
            self._hist_item.setOpts(x=[], height=[], width=1)
            self._last_histogram_bounds = None
            self._info_label.setText("No histogram values for this selection.")
            self._fit_histogram_view()
            return

        edges, bin_width = self._bin_edges_for(vals)
        counts, edges = np.histogram(vals, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]
        self._hist_item.setOpts(x=centers, height=counts, width=width * 0.95, brush="steelblue")
        max_count = float(counts.max()) if counts.size else 1.0
        self._last_histogram_bounds = (float(edges[0]), float(edges[-1]), 0.0, max(max_count, 1.0))

        x_label = f"log({attr_name})" if self._log_chk.isChecked() else attr_name
        self._plot.setLabel("bottom", f"{x_label}  [{agg_mode}]")
        sel_label = self._iter_combo.currentText()
        note = f"{vals.size:,} values  |  {sel_label}  |  {attr_name} [{agg_mode}]  |  bin {bin_width:.6g}"
        if not vld_only:
            note += "  |  incl. invalid"
        if uneval:
            note += f"  |  filter on {', '.join(uneval)} not applied"
        self._info_label.setText(note)
        self._remember_histogram_controls()
        self._fit_histogram_view()

    def _draw_stacked_histogram(self, ds, attr_name, agg_mode, vld_only) -> None:
        """One transparent filled step-histogram per iteration, with a legend."""
        n_itr = self._num_itr(ds)
        self._hist_item.setOpts(x=[], height=[], width=1)   # hide the single-series bars

        series: list[tuple[int, np.ndarray]] = []
        uneval: list[str] = []
        for k in range(n_itr):
            vals, un = self._raw_values(ds, attr_name, k, vld_only, agg_mode)
            uneval = un or uneval
            vals = self._transform_hist_values(vals)
            if vals.size:
                series.append((k, vals))

        if not series:
            self._last_histogram_bounds = None
            self._info_label.setText("No histogram values for this selection.")
            self._fit_histogram_view()
            return

        pooled = np.concatenate([v for _k, v in series])
        edges, bin_width = self._bin_edges_for(pooled)

        # More overlaid iterations -> more transparent so overlaps stay readable.
        alpha = int(np.clip(round(255.0 / max(len(series), 1) * 1.4), 45, 200))
        self._raw_legend = self._plot.addLegend(offset=(-10, 10))
        max_count = 1.0
        for k, vals in series:
            counts, _ = np.histogram(vals, bins=edges)
            max_count = max(max_count, float(counts.max()) if counts.size else 1.0)
            r, g, b = _iter_color(k)
            # self._plot.plot(name=...) registers a legend sample reliably.
            item = self._plot.plot(
                edges, counts, stepMode="center", fillLevel=0,
                brush=(r, g, b, alpha), pen=pg.mkPen(r, g, b, width=1),
                name=ordinal(k + 1),
            )
            self._raw_items.append(item)

        self._last_histogram_bounds = (float(edges[0]), float(edges[-1]), 0.0, max(max_count, 1.0))
        x_label = f"log({attr_name})" if self._log_chk.isChecked() else attr_name
        self._plot.setLabel("bottom", f"{x_label}  [{agg_mode}]")
        total = int(sum(v.size for _k, v in series))
        note = (
            f"{total:,} values across {len(series)} iterations  |  all [stacked]  |  "
            f"{attr_name} [{agg_mode}]  |  bin {bin_width:.6g}"
        )
        if not vld_only:
            note += "  |  incl. invalid"
        if uneval:
            note += f"  |  filter on {', '.join(uneval)} not applied"
        self._info_label.setText(note)
        self._remember_histogram_controls()
        self._fit_histogram_view()

    def _on_histogram_selection_changed(self, *_args) -> None:
        self._auto_bin_width = None
        self._reset_for_new_data()
        self._remember_histogram_controls()

    def _draw(self, *, preserve_histogram_frame: bool = False) -> None:
        ds = self._dataset()
        if ds is None or self._attr_combo.count() == 0:
            return

        # Raw-mode selection (non-final iteration or invalid included) takes a
        # separate, additive path. The default (last + valid) path below is
        # left byte-for-byte unchanged so filter-edit and ROI keep working.
        if self._is_raw_mode():
            self._draw_raw_mode()
            return
        self._clear_raw_items()

        previous_bounds = self._last_histogram_bounds

        attr_name = self._attr_combo.currentText()
        self._enforce_trace_aggregation()
        agg_mode = self._agg_combo.currentText()
        ds.state[self._view_state_key] = {
            "attribute": attr_name,
            "aggregation": agg_mode,
            "hide_zeros": self._zero_chk.isChecked(),
            "log_data": self._log_chk.isChecked(),
            "iter": self._iter_combo.currentText() or "",
            "valid_only": True,
        }
        raw = attr_values_1d(ds, attr_name)
        raw = np.empty(0) if raw is None else np.asarray(raw).ravel().astype(float)
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
            if preserve_histogram_frame and previous_bounds is not None:
                bin_width = float(self._bin_spin.value())
                edges = self._histogram_edges_from_frame(previous_bounds, bin_width)
                counts = np.zeros(max(edges.size - 1, 0), dtype=int)
                centers = 0.5 * (edges[:-1] + edges[1:])
                width = edges[1] - edges[0] if edges.size > 1 else 1.0
                self._hist_item.setOpts(x=centers, height=counts, width=width * 0.95)
                self._last_histogram_bounds = previous_bounds
                self._info_label.setText(
                    f"{int(ds.filter_mask.sum()):,} / {ds.prop.num_loc:,} localisations  |  "
                    f"0 histogram values  |  {attr_name} [{agg_mode}]  |  "
                    f"bin size {bin_width:.6g}"
                )
                self._remember_histogram_controls()
                self._fit_histogram_view()
                self._update_filter_edit_labels()
                return
            self._hist_item.setOpts(x=[], height=[], width=1)
            self._last_histogram_bounds = None
            self._info_label.setText("No histogram values.")
            return

        self._vals = vals
        if preserve_histogram_frame and previous_bounds is not None:
            bin_width = float(self._bin_spin.value())
        elif self._auto_bin_width is None:
            bin_width = self._default_bin_width(vals)
            self._set_bin_spin(bin_width)
        else:
            bin_width = float(self._bin_spin.value())

        data_vmin, data_vmax = float(vals.min()), float(vals.max())
        if preserve_histogram_frame and previous_bounds is not None:
            edges = self._histogram_edges_from_frame(previous_bounds, bin_width)
            counts, edges = np.histogram(vals, bins=edges)
        else:
            vmin, vmax = data_vmin, data_vmax
            if vmax <= vmin:
                vmax = vmin + max(bin_width, 1.0)
            n_bins = max(1, min(int(np.ceil((vmax - vmin) / max(bin_width, 1e-12))), 4096))
            counts, edges = np.histogram(vals, bins=n_bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]

        self._hist_item.setOpts(x=centers, height=counts, width=width * 0.95)
        max_count = float(np.max(counts)) if counts.size else 1.0
        if preserve_histogram_frame and previous_bounds is not None:
            self._last_histogram_bounds = previous_bounds
        else:
            self._last_histogram_bounds = (
                float(edges[0]),
                float(edges[-1]),
                0.0,
                max(max_count, 1.0),
            )
        x_label = f"log({attr_name})" if self._log_chk.isChecked() else attr_name
        self._plot.setLabel("bottom", f"{x_label}  [{agg_mode}]")
        self._info_label.setText(
            f"{int(ds.filter_mask.sum()):,} / {ds.prop.num_loc:,} localisations  |  "
            f"{vals.size:,} histogram values  |  {attr_name} [{agg_mode}]  |  "
            f"bin size {bin_width:.6g}  |  min {data_vmin:.6g}  |  max {data_vmax:.6g}"
        )
        self._remember_histogram_controls()
        self._fit_histogram_view()
        self._update_filter_edit_labels()

    def _fit_histogram_view(self) -> None:
        """Show the complete histogram and any active filter bounds."""
        bounds = self._last_histogram_bounds
        if bounds is None:
            if self._original_auto_range is not None:
                self._original_auto_range()
            return
        x0, x1, y0, y1 = bounds
        if not np.isfinite([x0, x1, y0, y1]).all():
            return
        if x1 <= x0:
            pad = max(abs(x0) * 0.01, 1.0)
            x0 -= pad
            x1 += pad
        y_top = max(y1, 1.0)
        y_bottom = 0.0
        if self._filter_edit:
            region = self._filter_edit.get("region")
            if region is not None:
                lo, hi = region.getRegion()
                x0 = min(x0, float(lo))
                x1 = max(x1, float(hi))
            y_pad = max(y_top * 0.01, 1.0)
            y_bottom = -y_pad
            y_top += y_pad
        self._plot.getPlotItem().vb.setRange(
            xRange=(x0, x1),
            yRange=(y_bottom, y_top),
            padding=0,
        )
        self._update_filter_edit_labels()

    def normalize_roi_record(self, record):
        """Histograms treat rectangle ROIs as x-range bands."""
        if record.type != "rectangle":
            return record
        try:
            (x0, x1), (_y0, _y1) = self._plot.getPlotItem().vb.viewRange()
        except Exception:
            return record
        bounds = rectangle_bounds(record)
        if bounds is None:
            return record
        rx0, rx1, _ry0, _ry1 = bounds
        y0, y1 = _y0, _y1
        record.geometry = {"bounds": [rx0, y0, rx1 - rx0, y1 - y0]}
        record.context = {**record.context, "histogram_band": True, "view_x_range": [x0, x1]}
        return record

    def compute_roi_selection(self, record):
        if record.type != "rectangle":
            return None
        # ROI bands map onto the materialized store; not defined for a raw
        # (non-final iteration / invalid-included) view.
        if self._is_raw_mode():
            return None
        ds = self._dataset()
        bounds = rectangle_bounds(record)
        if ds is None or bounds is None:
            return None
        attr_name = self._attr_combo.currentText()
        agg_mode = self._agg_combo.currentText()
        raw = attr_values_1d(ds, attr_name)
        raw = np.empty(0) if raw is None else np.asarray(raw).ravel().astype(float)
        if raw.size == 0:
            return None

        lo, hi = bounds[0], bounds[1]
        if agg_mode == "per loc":
            mask = self._per_loc_histogram_mask(ds, raw, lo, hi)
        else:
            mask = self._trace_histogram_mask(ds, raw, agg_mode, lo, hi)
        context = {
            "source_view": "histogram",
            "dataset_idx": self._dataset_idx,
            "attribute": attr_name,
            "aggregation": agg_mode,
            "x_range": [lo, hi],
            "log_data": self._log_chk.isChecked(),
            "hide_zeros": self._zero_chk.isChecked(),
        }
        return ds, mask, context

    def _per_loc_histogram_mask(self, ds, raw: np.ndarray, lo: float, hi: float) -> np.ndarray:
        n = min(raw.size, ds.prop.num_loc)
        values = raw[:n].astype(float, copy=True)
        base = np.asarray(ds.filter_mask, dtype=bool).ravel()
        if base.size != n:
            base = np.ones(n, dtype=bool)
        display_mask = base & np.isfinite(values)
        if self._log_chk.isChecked():
            positive = values > 0.0
            display_mask &= positive
            values = np.where(positive, np.log(values), np.nan)
        if self._zero_chk.isChecked():
            display_mask &= values != 0.0
        mask = value_range_mask(values, lo, hi, base_mask=display_mask)
        if mask.size != ds.prop.num_loc:
            full = np.zeros(ds.prop.num_loc, dtype=bool)
            full[:mask.size] = mask
            mask = full
        return mask

    def _trace_histogram_mask(self, ds, raw: np.ndarray, agg_mode: str, lo: float, hi: float) -> np.ndarray:
        vals = self._aggregate(raw, ds.filter_mask, agg_mode, ds).astype(float, copy=True)
        display_mask = np.isfinite(vals)
        if self._log_chk.isChecked():
            positive = vals > 0.0
            display_mask &= positive
            vals = np.where(positive, np.log(vals), np.nan)
        if self._zero_chk.isChecked():
            display_mask &= vals != 0.0
        selected_traces = value_range_mask(vals, lo, hi, base_mask=display_mask)
        mask = np.zeros(ds.prop.num_loc, dtype=bool)
        trace_idx = np.asarray(ds.prop.trace_idx, dtype=int)
        for trace_id in np.flatnonzero(selected_traces):
            if trace_id >= trace_idx.shape[0]:
                continue
            start, stop = trace_idx[trace_id]
            start = max(int(start), 0)
            stop = min(int(stop), ds.prop.num_loc - 1)
            if stop >= start:
                mask[start : stop + 1] = True
        ftr = np.asarray(ds.filter_mask, dtype=bool).ravel()
        if ftr.size == mask.size:
            mask &= ftr
        return mask

    def _warn_log_filtered_values(
        self,
        dataset_name: str,
        attr_name: str,
        agg_mode: str,
        removed: int,
        total: int,
    ) -> None:
        key = (self._dataset_idx, dataset_name, attr_name, agg_mode, removed, total)
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

    def _histogram_edges_from_frame(
        self,
        bounds: tuple[float, float, float, float],
        bin_width: float,
    ) -> np.ndarray:
        x0, x1, _y0, _y1 = bounds
        if not np.isfinite([x0, x1, bin_width]).all() or bin_width <= 0 or x1 <= x0:
            return np.array([0.0, 1.0])
        n_bins = max(1, min(int(np.ceil((x1 - x0) / max(bin_width, 1e-12))), 4096))
        return np.linspace(float(x0), float(x1), n_bins + 1)

    def _set_bin_spin(self, value: float) -> None:
        self._bin_spin.blockSignals(True)
        self._bin_spin.setValue(float(value))
        self._bin_spin.setSingleStep(max(float(value) / 10.0, 1e-12))
        self._bin_spin.blockSignals(False)

    def _reset_histogram(self) -> None:
        self._auto_bin_width = None
        self._draw()
        self._fit_histogram_view()

    def _reset_for_new_data(self) -> None:
        if self._resetting_plot:
            return
        self._resetting_plot = True
        try:
            self._auto_bin_width = None
            self._draw()
            self._fit_histogram_view()
        finally:
            self._resetting_plot = False

    def start_filter_edit(
        self,
        *,
        attr: str,
        mode: str,
        lo: float,
        hi: float,
        on_update=None,
        on_finish=None,
        on_cancel=None,
        restore_view_on_finish: bool = True,
        restore_view_on_cancel: bool = True,
    ) -> None:
        """Display and edit a filter range on top of this histogram."""
        self._clear_filter_edit(restore=False)
        # Filter editing operates on the materialized (last-iteration, valid)
        # store. Force the selection back to default if the user had switched.
        if self._is_raw_mode():
            self._iter_combo.blockSignals(True)
            self._valid_chk.blockSignals(True)
            ds0 = self._dataset()
            if ds0 is not None:
                self._iter_combo.setCurrentText(self._default_iter_label(ds0))
            self._valid_chk.setChecked(True)
            self._iter_combo.blockSignals(False)
            self._valid_chk.blockSignals(False)
            self._clear_raw_items()
        ds = self._dataset()
        saved_state = dict((ds.state.get(self._view_state_key, {}) if ds is not None else {}))
        if self._attr_combo.findText(attr) >= 0:
            self._attr_combo.setCurrentText(attr)
        if self._agg_combo.findText(mode) >= 0:
            self._agg_combo.setCurrentText(mode)
        self._reset_for_new_data()
        self._filter_edit = {
            "saved_state": saved_state,
            "attr": attr,
            "mode": mode,
            "on_update": on_update,
            "on_finish": on_finish,
            "on_cancel": on_cancel,
            "restore_view_on_finish": restore_view_on_finish,
            "restore_view_on_cancel": restore_view_on_cancel,
        }

        # The incoming bounds are real data values. When the histogram plots
        # log(data) the region lives in log space, so place it at log(bounds).
        region_lo, region_hi = float(min(lo, hi)), float(max(lo, hi))
        if self._log_chk.isChecked():
            if region_lo > 0.0 and region_hi > 0.0:
                region_lo, region_hi = float(np.log(region_lo)), float(np.log(region_hi))
            else:
                fallback = self._filter_log_data_range()
                if fallback is not None:
                    region_lo, region_hi = fallback

        range_color, range_alpha, bounds_color = self._filter_style()
        region = pg.LinearRegionItem(
            values=(region_lo, region_hi),
            orientation="vertical",
            brush=pg.mkBrush(range_color.red(), range_color.green(), range_color.blue(), range_alpha),
            pen=pg.mkPen(bounds_color, width=4),
            hoverPen=pg.mkPen(bounds_color, width=5),
            movable=True,
        )
        region.setZValue(20)
        self._plot.addItem(region)
        report_label = pg.TextItem(
            "",
            color=bounds_color,
            anchor=(1.0, 0.0),
            fill=pg.mkBrush(255, 255, 255, 220),
            border=pg.mkPen(bounds_color, width=1),
        )
        report_font = report_label.textItem.font()
        report_font.setBold(True)
        base_size = report_font.pointSizeF()
        if base_size <= 0:
            base_size = 10.0
        report_font.setPointSizeF(base_size * 1.1)
        report_label.textItem.setFont(report_font)
        report_label.setZValue(21)
        self._plot.addItem(report_label)
        region.sigRegionChanged.connect(self._update_filter_edit_labels)
        region.sigRegionChangeFinished.connect(self._on_filter_region_changed)
        region.scene().sigMouseClicked.connect(self._on_filter_scene_clicked)
        self._filter_edit.update({
            "region": region,
            "report_label": report_label,
            "inclusive_min": True,
            "inclusive_max": True,
        })
        self._set_filter_edit_buttons_visible(True)
        self._fit_histogram_view()
        self._update_filter_edit_labels()

    def _on_filter_region_changed(self) -> None:
        edit = self._filter_edit or {}
        region = edit.get("region")
        if region is None:
            return
        lo, hi = region.getRegion()
        if hi < lo:
            region.setRegion((hi, lo))
        self._fit_histogram_view()
        self._update_filter_edit_labels()

    def _update_filter_edit_labels(self) -> None:
        edit = self._filter_edit
        if not edit:
            return
        region = edit.get("region")
        report_label = edit.get("report_label")
        if region is None or report_label is None:
            return
        lo, hi = region.getRegion()
        try:
            xrange, yrange = self._plot.getPlotItem().vb.viewRange()
            x0, x1 = float(xrange[0]), float(xrange[1])
            y0, y1 = float(yrange[0]), float(yrange[1])
        except Exception:
            x0, x1, y0, y1 = 0.0, 1.0, 0.0, 1.0
        x_span = max(abs(x1 - x0), 1e-12)
        y_span = max(abs(y1 - y0), 1e-12)
        attr_name = self._attr_combo.currentText()
        x_label = f"log({attr_name})" if self._log_chk.isChecked() else attr_name
        report_label.setText(
            f"filtering {x_label} as {self._agg_combo.currentText()}\n"
            f"loc in filter {self._filter_count_text(lo, hi)}\n"
            f"min: {_format_filter_report_number(float(lo))}\n"
            f"max: {_format_filter_report_number(float(hi))}"
        )
        report_label.setPos(x1 - 0.02 * x_span, y1 - 0.03 * y_span)

    def _filter_count_text(self, lo: float, hi: float) -> str:
        ds = self._dataset()
        if ds is None:
            return ""
        attr_name = self._attr_combo.currentText()
        agg_mode = self._agg_combo.currentText()
        raw = attr_values_1d(ds, attr_name)
        raw = np.empty(0) if raw is None else np.asarray(raw).ravel().astype(float)
        if raw.size == 0:
            return f"0 / {ds.prop.num_loc:,}"
        try:
            if agg_mode == "per loc":
                mask = self._per_loc_histogram_mask(ds, raw, lo, hi)
            else:
                mask = self._trace_histogram_mask(ds, raw, agg_mode, lo, hi)
            selected = int(np.count_nonzero(mask))
        except Exception:
            selected = 0
        return f"{selected:,} / {ds.prop.num_loc:,}"

    def _filter_style(self) -> tuple[QColor, int, QColor]:
        prefs = self._state.prefs.get("plot", {})
        range_color = _named_color(str(prefs.get("filter_range_color", "Green")))
        bounds_color = _named_color(str(prefs.get("filter_bounds_color", "Green")))
        alpha_pct = int(prefs.get("filter_range_alpha", 45))
        alpha = round(alpha_pct * 255 / 100)
        return range_color, max(0, min(alpha, 255)), bounds_color

    def _on_filter_scene_clicked(self, event) -> None:
        if not event.double():
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        edit = self._filter_edit
        if not edit:
            return
        region = edit.get("region")
        if region is None:
            return
        pos = event.scenePos()
        try:
            view_pos = self._plot.getPlotItem().vb.mapSceneToView(pos)
        except Exception:
            return
        lo, hi = region.getRegion()
        span = max(abs(hi - lo), 1e-12)
        if lo - 0.05 * span <= view_pos.x() <= hi + 0.05 * span:
            event.accept()
            self._show_filter_bounds_dialog()

    def _show_filter_bounds_dialog(self) -> None:
        edit = self._filter_edit
        if not edit:
            return
        region = edit.get("region")
        if region is None:
            return
        lo, hi = region.getRegion()
        dlg = QDialog(self)
        dlg.setWindowTitle("Filter bounds")
        form = QFormLayout(dlg)
        lo_spin = SmartBoundsSpinBox(dlg)
        hi_spin = SmartBoundsSpinBox(dlg)
        ds = self._dataset()
        attr = str(edit.get("attr", self._attr_combo.currentText()))
        mode = str(edit.get("mode", self._agg_combo.currentText()))
        values = None
        data_min = None
        data_max = None
        if ds is not None and attr in ds.attr:
            values, range_values = _filter_spinner_values(ds, attr, mode)
            finite = range_values.astype(float, copy=False)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                data_min = float(np.nanmin(finite))
                data_max = float(np.nanmax(finite))
        for spin, value in ((lo_spin, lo), (hi_spin, hi)):
            spin.configure(value=float(value), values=values, data_min=data_min, data_max=data_max, mode=mode)
        lo_inc = QCheckBox("Inclusive", dlg)
        hi_inc = QCheckBox("Inclusive", dlg)
        lo_inc.setChecked(bool(edit.get("inclusive_min", True)))
        hi_inc.setChecked(bool(edit.get("inclusive_max", True)))
        form.addRow("Min", lo_spin)
        form.addRow("Min bound", lo_inc)
        form.addRow("Max", hi_spin)
        form.addRow("Max bound", hi_inc)
        buttons = QHBoxLayout()
        ok = QPushButton("OK", dlg)
        cancel = QPushButton("Cancel", dlg)
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        buttons.addStretch()
        buttons.addWidget(ok)
        buttons.addWidget(cancel)
        form.addRow(buttons)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_lo = min(lo_spin.value(), hi_spin.value())
            new_hi = max(lo_spin.value(), hi_spin.value())
            region.setRegion((new_lo, new_hi))
            edit["inclusive_min"] = lo_inc.isChecked()
            edit["inclusive_max"] = hi_inc.isChecked()
            self._update_filter_edit_labels()

    def _filter_log_data_range(self) -> tuple[float, float] | None:
        """log(min..max) of the current attribute's positive values, or None.

        Used to place the filter region when the incoming real bounds can't be
        log-transformed (non-positive) under the log(data) view.
        """
        ds = self._dataset()
        if ds is None:
            return None
        raw = attr_values_1d(ds, self._attr_combo.currentText())
        if raw is None:
            return None
        vals = np.asarray(raw).ravel().astype(float)
        vals = vals[np.isfinite(vals) & (vals > 0.0)]
        if vals.size == 0:
            return None
        return float(np.log(vals.min())), float(np.log(vals.max()))

    def _current_filter_edit_values(self) -> tuple[float, float, bool, bool]:
        edit = self._filter_edit
        if not edit:
            return 0.0, 1.0, True, True
        region = edit.get("region")
        lo, hi = region.getRegion() if region is not None else (0.0, 1.0)
        lo, hi = float(lo), float(hi)
        if self._log_chk.isChecked():
            # The histogram plots log(data), so the region bounds live in log
            # space; convert back to real data values for the filter, which
            # operates on the un-transformed data.
            lo, hi = float(np.exp(lo)), float(np.exp(hi))
        return lo, hi, bool(edit.get("inclusive_min", True)), bool(edit.get("inclusive_max", True))

    def _update_filter_edit(self) -> None:
        edit = self._filter_edit
        if not edit:
            return
        callback = edit.get("on_update")
        if callable(callback):
            callback(*self._current_filter_edit_values())

    def _finish_filter_edit(self) -> None:
        edit = self._filter_edit
        if not edit:
            return
        callback = edit.get("on_finish")
        values = self._current_filter_edit_values()
        self._clear_filter_edit(restore=bool(edit.get("restore_view_on_finish", True)))
        if callable(callback):
            callback(*values)

    def _cancel_filter_edit(self) -> None:
        edit = self._filter_edit
        if not edit:
            return
        callback = edit.get("on_cancel")
        self._clear_filter_edit(restore=bool(edit.get("restore_view_on_cancel", True)))
        if callable(callback):
            callback()

    def _set_filter_edit_buttons_visible(self, visible: bool) -> None:
        for btn in (self._filter_update_btn, self._filter_finish_btn, self._filter_cancel_btn):
            btn.setVisible(visible)

    def _clear_filter_edit(self, *, restore: bool) -> None:
        edit = self._filter_edit
        if not edit:
            self._set_filter_edit_buttons_visible(False)
            return
        for key in ("region", "report_label"):
            item = edit.get(key)
            if item is not None:
                try:
                    self._plot.removeItem(item)
                except Exception:
                    pass
        saved = edit.get("saved_state")
        self._filter_edit = None
        self._set_filter_edit_buttons_visible(False)
        ds = self._dataset()
        if restore and saved and ds is not None:
            ds.state[self._view_state_key] = saved
            self._refresh()
        else:
            self._fit_histogram_view()

    def _on_bin_changed(self) -> None:
        self._auto_bin_width = float(self._bin_spin.value())
        self._draw()

    def _on_histogram_attribute_changed(self, *_args) -> None:
        self._enforce_trace_aggregation()
        self._reset_for_new_data()
        if self._filter_edit:
            self._reset_filter_region_to_current_data()
            self._fit_histogram_view()
        self._remember_histogram_controls()

    def _on_histogram_log_changed(self, *_args) -> None:
        old_log = bool(self._last_log_data)
        new_log = self._log_chk.isChecked()
        transformed = True
        if self._filter_edit:
            transformed = self._transform_filter_region_for_log_change(old_log, new_log)
        self._enforce_trace_aggregation()
        self._reset_for_new_data()
        if self._filter_edit:
            if not transformed:
                self._reset_filter_region_to_current_data()
            self._fit_histogram_view()
        self._remember_histogram_controls()

    def _on_histogram_aggregation_changed(self, *_args) -> None:
        self._enforce_trace_aggregation()
        self._reset_for_new_data()
        if self._filter_edit and self._should_reset_filter_for_aggregation(self._agg_combo.currentText()):
            self._reset_filter_region_to_current_data()
            self._fit_histogram_view()
        self._remember_histogram_controls()

    def _on_histogram_display_changed(self, *_args) -> None:
        self._enforce_trace_aggregation()
        self._reset_for_new_data()
        self._remember_histogram_controls()

    def _remember_histogram_controls(self) -> None:
        self._last_attr_name = self._attr_combo.currentText()
        self._last_agg_mode = self._agg_combo.currentText()
        self._last_log_data = self._log_chk.isChecked()

    def _current_histogram_x_bounds(self) -> tuple[float, float] | None:
        if self._last_histogram_bounds is None:
            return None
        x0, x1, _y0, _y1 = self._last_histogram_bounds
        if not np.isfinite([x0, x1]).all() or x1 <= x0:
            return None
        return float(x0), float(x1)

    def _reset_filter_region_to_current_data(self) -> None:
        edit = self._filter_edit or {}
        region = edit.get("region")
        bounds = self._current_histogram_x_bounds()
        if region is None or bounds is None:
            return
        region.setRegion(bounds)
        self._update_filter_edit_labels()

    def _transform_filter_region_for_log_change(self, old_log: bool, new_log: bool) -> bool:
        if old_log == new_log:
            return True
        edit = self._filter_edit or {}
        region = edit.get("region")
        if region is None:
            return True
        lo, hi = region.getRegion()
        lo, hi = float(lo), float(hi)
        try:
            if not old_log and new_log:
                if lo <= 0.0 or hi <= 0.0:
                    return False
                region.setRegion((np.log(lo), np.log(hi)))
            elif old_log and not new_log:
                if max(lo, hi) > 700.0:
                    return False
                region.setRegion((np.exp(lo), np.exp(hi)))
        except Exception:
            return False
        self._update_filter_edit_labels()
        return True

    def _should_reset_filter_for_aggregation(self, mode: str) -> bool:
        return mode not in {"per loc", "trace mean", "trace median"}

    def _enforce_trace_aggregation(self) -> None:
        trace_wise = is_trace_wise_attribute(self._attr_combo.currentText())
        self._agg_combo.blockSignals(True)
        if trace_wise:
            self._agg_combo.setCurrentText("trace mean")
            self._agg_combo.setEnabled(False)
            self._agg_combo.setToolTip("This is a track-level attribute expanded per localization; histogram uses trace mean.")
        else:
            self._agg_combo.setEnabled(True)
            self._agg_combo.setToolTip("")
        self._agg_combo.blockSignals(False)

    def _on_filter_changed(self, idx: int) -> None:
        if idx == self._dataset_idx:
            if self._filter_edit:
                self._reset_for_new_data()
            else:
                self._draw(preserve_histogram_frame=True)
                self._fit_histogram_view()

    def focusInEvent(self, event) -> None:
        if self._dataset_idx is not None and 0 <= self._dataset_idx < len(self._state.datasets):
            self._state.set_active(self._dataset_idx)
        if self._roi_overlay is not None:
            self._roi_overlay.activate()
        super().focusInEvent(event)

    def changeEvent(self, event) -> None:
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            if self._dataset_idx is not None and 0 <= self._dataset_idx < len(self._state.datasets):
                self._state.set_active(self._dataset_idx)
            if self._roi_overlay is not None:
                self._roi_overlay.activate()
        super().changeEvent(event)
