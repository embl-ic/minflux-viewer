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
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..colors import component_colors, viewer_color
from ..core.app_state import AppState
from ..core.attributes import (
    aggregation_description,
    attribute_description,
    is_trace_wise_attribute,
    plot_attribute_names,
)
from ..core.iteration import (
    iteration_bold_flags,
    iteration_labels,
    iteration_selector_label,
    ordinal,
    parse_iteration_label,
)
from ..core.loader import (
    attr_matches_selection,
    attr_values_for_selection,
    effective_iteration_for_attr,
    effective_iterations_for_attr,
    is_value_pool_selector,
    mfx_filter_mask,
    mfx_get,
)
from ..core.roi_selection import rectangle_bounds, value_range_mask
from ..utils.filters import TRACE_AGG_FUNCS, trace_agg_func
from .filter_dialog import SmartBoundsSpinBox, _filter_spinner_values
from .plot_format import plot_widget


def _iter_color(k: int, prefs: dict | None = None) -> tuple[int, int, int, int]:
    colors = list(component_colors(prefs, "functions", "Iteration series").values())
    return colors[k % len(colors)] if colors else (70, 130, 180, 255)

#: Trace read-outs offered in the "As" dropdown. Order and membership come from
#: the shared registry so the histogram, the filter and the raw path can never
#: disagree; which of them actually appear is gated by
#: ``prefs["plot"]["histogram_values"]``.
_TRACE_AGG_MODES = list(TRACE_AGG_FUNCS)

#: Upper bound on bins across the full data range (mirrors ``_bin_edges_for``).
_MAX_HISTOGRAM_BINS = 4096
#: Bin-count targeting for a zoomed x span: roughly this many values per bin,
#: clamped to a readable number of bars across the window.
_ZOOM_VALUES_PER_BIN = 4
_ZOOM_MIN_BINS = 10
_ZOOM_MAX_BINS = 60
#: Headroom above the tallest visible bar when a zoom re-fits the height.
_ZOOM_Y_HEADROOM = 0.05

_ZOOM_TOOLTIPS = {
    "horizontal": (
        "Drag a horizontal guide at the mouse; X remaps to the drawn span.\n"
        "The bin size is refined for the new span and the height is\n"
        "re-fitted to the re-binned bars."
    ),
    "vertical": (
        "Drag a vertical guide at the mouse; Y (count) remaps to the drawn\n"
        "span, clamped at zero. X and the bin size are unchanged."
    ),
    "unconstrained": (
        "Drag a rectangle; X remaps to it and the bin size is refined.\n"
        "The height is re-fitted to the re-binned bars."
    ),
}

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
        # Right-click Zoom tool: None = off, else the armed drag mode.
        self._zoom_mode: str | None = None
        self._zoom_drag_start = None
        self._zoom_preview = None
        self._original_auto_range = None
        self._last_attr_name = ""
        self._last_agg_mode = ""
        self._last_log_data = False
        # Set while a filter row that targets one specific iteration is being
        # edited: that iteration's values are then gathered onto the
        # materialized rows so the region drag stays available (see
        # _current_value_itr / _is_raw_mode).
        self._edit_itr: int | None = None

        self.setWindowTitle("Histogram")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(700, 500)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._build_ui()
        self._refresh()

        state.filter_changed.connect(self._on_filter_changed)
        state.attributes_changed.connect(self._on_attributes_changed)

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
        self._plot.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._plot.customContextMenuRequested.connect(self._show_plot_context_menu)
        self._plot.getPlotItem().setMenuEnabled(False)
        plot_item = self._plot.getPlotItem()
        view_box = plot_item.vb
        self._view_box = view_box
        try:
            view_box.setMenuEnabled(False)
        except Exception:
            pass
        self._original_auto_range = view_box.autoRange
        view_box.autoRange = lambda *args, **kwargs: self._fit_histogram_view()
        # The floating "A" button calls PlotItem.autoBtnClicked(), which turns on
        # pyqtgraph's *continuous* auto-range (enableAutoRange) rather than going
        # through vb.autoRange — so the patch above never saw it. Continuous
        # auto-range fits every item including the view-anchored filter report
        # label, and repositioning that label on sigRangeChanged grew the bounds
        # again, so each click zoomed further out. Route "A" to the same
        # deterministic fit the Reset button uses.
        plot_item.autoBtnClicked = lambda *args, **kwargs: self._fit_histogram_view()
        view_box.sigRangeChanged.connect(lambda *_args: self._update_filter_edit_labels())
        self._original_mouse_drag_event = view_box.mouseDragEvent
        view_box.mouseDragEvent = self._zoom_mouse_drag_event
        self._apply_plot_colors()

        self._hist_item = pg.BarGraphItem(
            x=[], height=[], width=1,
            brush=pg.mkBrush(*viewer_color(self._state.prefs, "histogram_data")),
        )
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

        self._eff_iter_cache = {}                    # dataset (re)loaded → drop cache
        self._populate_iteration_modes(ds, str(saved.get("iter", "") or ""))

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

    def _populate_iteration_modes(self, ds, preferred: str = "") -> None:
        """Fill the ``Iter`` dropdown, honouring the Preferences pooled-mode set.

        *preferred* is the label to re-select when it is still on offer (the
        saved view state on a refresh, the live selection on a preference
        change); otherwise the attribute's default label is used.
        """
        iter_opts = self._iter_labels(ds)
        self._iter_combo.blockSignals(True)
        self._iter_combo.clear()
        self._iter_combo.addItems(iter_opts)
        default_label = self._default_iter_label_for(ds, self._attr_combo.currentText())
        self._iter_combo.setCurrentText(
            preferred if preferred in iter_opts else default_label
        )
        self._iter_combo.blockSignals(False)
        # Nothing to browse for single-iteration data.
        has_iters = bool(iter_opts)
        self._iter_combo.setVisible(has_iters)
        self._iter_label.setVisible(has_iters)
        self._style_iteration_boldness()             # bold the useful iterations

    def refresh_preferences(self) -> None:
        """Re-read Preferences > Appearance > Histogram Plot.

        Broadcast by ``main_window._refresh_plot_preferences`` after Preferences
        is accepted, so the "As" / "Iter" dropdowns pick up the enabled trace
        read-outs and pooled modes without reopening the window.
        """
        ds = self._dataset()
        if ds is None:
            return
        agg_before = self._agg_combo.currentText()
        iter_before = self._iter_combo.currentText()
        self._populate_agg_modes()
        self._populate_iteration_modes(ds, iter_before)
        self._enforce_trace_aggregation()
        if (self._agg_combo.currentText() != agg_before
                or self._iter_combo.currentText() != iter_before):
            # The plotted quantity changed because the old choice is gone.
            self._reset_for_new_data()
        self._remember_histogram_controls()
        self.refresh_colors()

    def _apply_plot_colors(self) -> None:
        background = viewer_color(self._state.prefs, "histogram_background")
        self._plot.setBackground(QColor(*background))
        luminance = 0.2126 * background[0] + 0.7152 * background[1] + 0.0722 * background[2]
        foreground = QColor(25, 25, 25) if luminance >= 145 else QColor(235, 235, 235)
        for name in ("bottom", "left"):
            axis = self._plot.getPlotItem().getAxis(name)
            axis.setPen(foreground)
            axis.setTextPen(foreground)

    def refresh_colors(self) -> None:
        self._apply_plot_colors()
        self._draw()
        self._restyle_filter_edit()

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
        # Which pooled modes are offered is a preference; a missing key means
        # "all of them", an empty list means the user turned them all off.
        allowed = self._state.prefs.get("plot", {}).get("histogram_iterations", None)
        return iteration_labels(self._num_itr(ds), allowed=allowed)

    def _default_iter_label(self, ds) -> str:
        labels = self._iter_labels(ds)
        return f"last ({ordinal(self._num_itr(ds))})" if labels else ""

    def _effective_iter_label(self, ds, attr: str) -> "str | None":
        """Dropdown label for *attr*'s effective iteration (cfr/efc), else None."""
        eff = effective_iteration_for_attr(ds, attr)
        if eff is None:
            return None
        label = ordinal(eff + 1)
        return label if label in self._iter_labels(ds) else None

    def _default_iter_label_for(self, ds, attr: str) -> str:
        """Preferred default label for *attr*: its effective iteration (cfr/efc)
        when defined, otherwise the last-iteration default."""
        return self._effective_iter_label(ds, attr) or self._default_iter_label(ds)

    def _style_iteration_boldness(self) -> None:
        """Bold the iteration-dropdown entries that hold real values for the
        current attribute, so the useful iterations stand out while browsing
        (e.g. cfr → only its measured iteration(s); dcr → every iteration)."""
        from PyQt6.QtGui import QFont

        ds = self._dataset()
        if ds is None or self._iter_combo.count() == 0:
            return
        attr = self._attr_combo.currentText()
        cache = getattr(self, "_eff_iter_cache", None)
        if cache is None:
            cache = self._eff_iter_cache = {}
        if attr not in cache:
            try:
                cache[attr] = effective_iterations_for_attr(ds, attr)
            except Exception:
                cache[attr] = None
        eff = cache[attr]
        labels = [self._iter_combo.itemText(i) for i in range(self._iter_combo.count())]
        flags = (iteration_bold_flags(labels, eff, self._num_itr(ds))
                 if eff is not None else [False] * len(labels))
        for i, bold in enumerate(flags):
            font = QFont(self._iter_combo.font())
            font.setBold(bool(bold))
            self._iter_combo.setItemData(i, font, Qt.ItemDataRole.FontRole)

    def _enforce_effective_iteration(self) -> None:
        """Auto-select the effective iteration when the attribute is cfr/efc.

        cfr/efc are only measured at one loop iteration (zero/NaN at `last`), so
        switching to one of them jumps the Iter dropdown to that iteration. The
        dropdown stays enabled, so the user can still browse other iterations.
        Switching *away* to a normal attribute restores the default view when the
        dropdown is still sitting on the previous attribute's effective iteration
        (a deliberate manual browse of another iteration is preserved).
        """
        ds = self._dataset()
        if ds is None:
            return
        label = self._effective_iter_label(ds, self._attr_combo.currentText())
        if label is None:
            prev_label = self._effective_iter_label(ds, getattr(self, "_last_attr_name", "") or "")
            if prev_label is None or self._iter_combo.currentText() != prev_label:
                return
            label = self._default_iter_label(ds)
        if label and self._iter_combo.currentText() != label and self._iter_combo.findText(label) >= 0:
            self._iter_combo.blockSignals(True)
            self._iter_combo.setCurrentText(label)
            self._iter_combo.blockSignals(False)

    def _selection(self) -> tuple:
        """Return (itr_selector, render_mode) for the current label."""
        return parse_iteration_label(self._iter_combo.currentText())

    def _value_label(self, attr_name: str) -> str:
        """Axis/report name for the plotted quantity (log made explicit)."""
        return f"log({attr_name})" if self._log_chk.isChecked() else attr_name

    def _current_value_itr(self) -> "str | int":
        """Iteration selector the materialized path reads values at.

        ``"auto"`` is the historical materialized value (cfr/efc at their
        effective iteration, everything else last-valid). The value-pooling
        modes pass straight through, and while a filter row targeting one
        specific iteration is being edited that iteration is used, so the plot
        shows the values the filter actually tests.
        """
        itr_sel, render = self._selection()
        if is_value_pool_selector(itr_sel):
            return itr_sel
        if (
            self._edit_itr is not None
            and render == "single"
            and isinstance(itr_sel, (int, np.integer))
            and int(itr_sel) == self._edit_itr
        ):
            return int(itr_sel)
        return "auto"

    def _materialized_values(self, ds, attr_name: str) -> "np.ndarray | None":
        """Per-loc values on the default (``ds.attr``-aligned) path.

        Honours the Iter dropdown's value-pooling modes (``all [sum]`` /
        ``all [average]``, which yield one value per localization just like the
        default view). Everything else keeps the materialized value, so this is
        a drop-in for the previous ``attr_values_1d`` call.
        """
        if ds is None:
            return None
        return attr_values_for_selection(ds, attr_name, itr=self._current_value_itr())

    def _is_raw_mode(self) -> bool:
        """True unless the current selection yields one value per localization
        on rows that ARE the materialized store.

        The default (ds.attr) path requires ds.attr row alignment with the
        current selection: last+valid for normal loads, last+all-validity for
        ``only_valid_locs=False`` loads (so filter-edit/ROI work there too).
        For all-iteration loads even a ``last`` selection must come from the
        raw store.

        Selections that stay on the materialized path — all one value per
        localization, which is what keeps filter editing and ROI bands alive:

        * plain ``last``;
        * cfr/efc at their **effective** iteration — the canonical per-loc value
          ``attr_values_for_selection`` returns, so the Iter label matches the
          displayed values (any other iteration browses the genuine raw values);
        * the value-pooling modes ``all [sum]`` / ``all [average]``, which pool
          onto the same rows as ``last``;
        * while a filter row is being edited, the single iteration that row
          filters on (gathered onto the materialized rows).
        """
        itr_sel, render = self._selection()
        ds = self._dataset()
        if ds is None:
            return not (itr_sel == "last" and render == "single")
        aligned = attr_matches_selection(
            ds, itr="last", vld_only=self._valid_chk.isChecked(),
        )
        if is_value_pool_selector(itr_sel):
            return not aligned
        if render == "single" and isinstance(itr_sel, (int, np.integer)):
            eff = effective_iteration_for_attr(ds, self._attr_combo.currentText())
            on_effective = eff is not None and int(itr_sel) == int(eff)
            on_edit_iter = self._edit_itr is not None and int(itr_sel) == self._edit_itr
            return not aligned if (on_effective or on_edit_iter) else True
        if not (itr_sel == "last" and render == "single"):
            return True
        # cfr/efc on `last`: the materialized value is their effective one, so
        # the plotted numbers would not match the label -> browse raw instead.
        if effective_iteration_for_attr(ds, self._attr_combo.currentText()) is not None:
            return True
        return not aligned

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
        """Return histogram values + unevaluable filter attrs for one iteration.

        A value-pooling selector yields one value per localization laid on the
        ``last`` rows, so the companion tid / filter mask must be taken at
        ``last`` — pooling those would be meaningless (a summed ``tid``).
        """
        from ..utils.filters import raw_trace_aggregate
        vals = mfx_get(ds, attr_name, itr=sel, vld_only=vld_only)
        if vals is None:
            return np.empty(0), []
        vals = np.asarray(vals).ravel().astype(float)
        row_sel = "last" if is_value_pool_selector(sel) else sel
        unevaluable: list[str] = []
        if agg_mode == "per loc":
            res = mfx_filter_mask(ds, itr=row_sel, vld_only=vld_only)
            if res is not None:
                fmask, unevaluable = res
                if fmask.shape[0] == vals.shape[0]:
                    vals = vals[fmask]
        else:
            tid = mfx_get(ds, "tid", itr=row_sel, vld_only=vld_only)
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
        # Keep an active filter edit alive while browsing raw iterations.  A
        # region is a user-owned edit session; changing the histogram view is
        # not an implicit Finish/Cancel action.  Concrete iterations selected
        # during an edit are normally routed through the materialized path by
        # ``_on_histogram_selection_changed``.  This guard also preserves the
        # region for selections that genuinely need the raw renderer.

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
            self._update_filter_edit_labels()
            return

        if render == "stacked":
            self._draw_stacked_histogram(ds, attr_name, agg_mode, vld_only)
        else:
            self._draw_single_raw_histogram(ds, attr_name, agg_mode, itr_sel, vld_only, render)
        self._update_filter_edit_labels()

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
        self._hist_item.setOpts(
            x=centers, height=counts, width=width * 0.95,
            brush=pg.mkBrush(*viewer_color(self._state.prefs, "histogram_data")),
        )
        max_count = float(counts.max()) if counts.size else 1.0
        self._last_histogram_bounds = (float(edges[0]), float(edges[-1]), 0.0, max(max_count, 1.0))

        x_label = self._value_label(attr_name)
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
            r, g, b, color_alpha = _iter_color(k, self._state.prefs)
            series_alpha = int(round(alpha * color_alpha / 255.0))
            # self._plot.plot(name=...) registers a legend sample reliably.
            item = self._plot.plot(
                edges, counts, stepMode="center", fillLevel=0,
                brush=(r, g, b, series_alpha), pen=pg.mkPen(r, g, b, color_alpha, width=1),
                name=ordinal(k + 1),
            )
            self._raw_items.append(item)

        self._last_histogram_bounds = (float(edges[0]), float(edges[-1]), 0.0, max(max_count, 1.0))
        x_label = self._value_label(attr_name)
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
        if self._filter_edit:
            itr_sel, render = self._selection()
            # A concrete iteration can be gathered onto the materialized
            # rows, which keeps the editable region usable while the user
            # browses away from an attribute's default/effective iteration.
            # The value is deliberately updated only for a single iteration;
            # pooled/stacked selections retain their own raw rendering rules.
            if render == "single" and isinstance(itr_sel, (int, np.integer)):
                self._edit_itr = int(itr_sel)
            else:
                self._edit_itr = None
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
        raw = self._materialized_values(ds, attr_name)
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
        x_label = self._value_label(attr_name)
        self._plot.setLabel("bottom", f"{x_label}  [{agg_mode}]")
        iter_note = (
            f"  |  {self._iter_combo.currentText()}"
            if is_value_pool_selector(self._selection()[0]) else ""
        )
        self._info_label.setText(
            f"{int(ds.filter_mask.sum()):,} / {ds.prop.num_loc:,} localisations  |  "
            f"{vals.size:,} histogram values  |  {attr_name} [{agg_mode}]"
            f"{iter_note}  |  "
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

    # ------------------------------------------------------------------
    # Right-click menu: Zoom + Reset View
    # ------------------------------------------------------------------

    #: Zoom drag modes, in the order they appear in the context menu.
    ZOOM_MODES = ("horizontal", "vertical", "unconstrained")

    def _show_plot_context_menu(self, pos) -> None:
        """Plot right-click menu; defers to a ROI right-click like the render view."""
        overlay = self._roi_overlay
        if overlay is not None:
            try:
                view_pos = self._view_box.mapSceneToView(
                    self._plot.mapToScene(pos)
                )
                if overlay._record_at(view_pos) is not None:
                    return
            except Exception:
                pass

        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        zoom_menu = menu.addMenu("Zoom")
        zoom_menu.setToolTipsVisible(True)
        zoom_menu.menuAction().setToolTip(
            "Arm a zoom drag. The tool releases itself after one drag,\n"
            "so the next left-drag pans as usual. Esc cancels."
        )
        for mode in self.ZOOM_MODES:
            action = zoom_menu.addAction(mode)
            action.setCheckable(True)
            action.setChecked(self._zoom_mode == mode)
            action.setToolTip(_ZOOM_TOOLTIPS[mode])
            action.triggered.connect(
                lambda _checked=False, value=mode: self._toggle_zoom_mode(value)
            )
        menu.addSeparator()
        reset_action = menu.addAction("Reset View")
        reset_action.setToolTip(
            "Re-bin over the full data range and fit the plot to it\n"
            "(the same as the Reset button)."
        )
        reset_action.triggered.connect(self._reset_view)
        menu.exec(self._plot.mapToGlobal(pos))

    def _toggle_zoom_mode(self, mode: str) -> None:
        """Arm a zoom drag mode; selecting the armed mode again disarms it.

        The armed mode is one-shot — ``_zoom_mouse_drag_event`` releases it once
        the drag finishes.
        """
        if mode not in self.ZOOM_MODES:
            return
        self._set_zoom_mode(None if self._zoom_mode == mode else mode)

    def _set_zoom_mode(self, mode: "str | None") -> None:
        self._zoom_mode = mode
        self._clear_zoom_preview()
        # Left-drag normally pans, so an armed zoom needs a visible affordance;
        # the menu check-mark alone is not on screen while dragging.
        try:
            viewport = self._plot.viewport()
            if mode:
                viewport.setCursor(Qt.CursorShape.CrossCursor)
            else:
                viewport.unsetCursor()
        except Exception:
            pass

    def _reset_view(self) -> None:
        """Context-menu Reset View — identical to the Reset button."""
        self._set_zoom_mode(None)
        self._reset_histogram()

    def keyPressEvent(self, event):                      # noqa: D102 - Qt override
        if event.key() == Qt.Key.Key_Escape and self._zoom_mode:
            self._set_zoom_mode(None)
            event.accept()
            return
        super().keyPressEvent(event)

    def _zoom_mouse_drag_event(self, event, axis=None) -> None:
        if not self._zoom_mode or event.button() != Qt.MouseButton.LeftButton:
            self._original_mouse_drag_event(event, axis=axis)
            return
        event.accept()
        if event.isStart():
            self._zoom_drag_start = self._view_box.mapSceneToView(
                event.buttonDownScenePos(Qt.MouseButton.LeftButton)
            )
            self._clear_zoom_preview()
            self._zoom_preview = pg.PlotDataItem(
                pen=pg.mkPen((30, 120, 220), width=1.5, style=Qt.PenStyle.DashLine)
            )
            self._zoom_preview.setZValue(30)
            self._plot.addItem(self._zoom_preview, ignoreBounds=True)
        if self._zoom_drag_start is None:
            return
        current = self._view_box.mapSceneToView(event.scenePos())
        self._update_zoom_preview(self._zoom_drag_start, current)
        if event.isFinish():
            self._apply_zoom_drag(self._zoom_drag_start, current)
            self._zoom_drag_start = None
            # One-shot: finishing the gesture releases the tool, so the next
            # left-drag pans as usual instead of starting another zoom. Pick the
            # mode again from the right-click menu to zoom once more.
            self._set_zoom_mode(None)

    def _update_zoom_preview(self, start, current) -> None:
        """Draw the rubber band: an 'H' for horizontal, an 'I' for vertical,
        a rectangle for unconstrained.

        The guide tracks the **cursor** on its free axis (the horizontal guide
        rides at the mouse's y, the vertical one at the mouse's x) rather than
        sitting at the middle of the view, so it can be lined up against the
        bars it is about to zoom into.
        """
        if self._zoom_preview is None:
            return
        x0, x1 = float(start.x()), float(current.x())
        y0, y1 = float(start.y()), float(current.y())
        (vx0, vx1), (vy0, vy1) = self._view_box.viewRange()
        if self._zoom_mode == "horizontal":
            cap = (vy1 - vy0) * 0.08
            self._zoom_preview.setData(
                [x0, x1, np.nan, x0, x0, np.nan, x1, x1],
                [y1, y1, np.nan, y1 - cap, y1 + cap, np.nan, y1 - cap, y1 + cap],
            )
        elif self._zoom_mode == "vertical":
            cap = (vx1 - vx0) * 0.08
            self._zoom_preview.setData(
                [x1, x1, np.nan, x1 - cap, x1 + cap, np.nan, x1 - cap, x1 + cap],
                [y0, y1, np.nan, y0, y0, np.nan, y1, y1],
            )
        else:
            self._zoom_preview.setData([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0])

    def _apply_zoom_drag(self, start, current) -> None:
        x0, x1 = sorted((float(start.x()), float(current.x())))
        y0, y1 = sorted((float(start.y()), float(current.y())))
        (vx0, vx1), (vy0, vy1) = self._view_box.viewRange()
        min_dx = abs(vx1 - vx0) * 1e-6
        min_dy = abs(vy1 - vy0) * 1e-6
        if self._zoom_mode == "horizontal":
            if (x1 - x0) > min_dx:
                # X remaps to the drawn span; Y is re-fitted because re-binning
                # changed the counts (see _zoom_to).
                self._zoom_to(x0, x1, None, None, rebin=True)
        elif self._zoom_mode == "vertical":
            # Counts are never negative, so a guide dragged below the axis only
            # buys empty space — clamp it away.
            y0 = max(y0, 0.0)
            if (y1 - y0) > min_dy:
                # Y only — the bin size (and so the counts) stay put.
                self._zoom_to(vx0, vx1, y0, y1, rebin=False)
        elif (x1 - x0) > min_dx and (y1 - y0) > min_dy:
            # The drawn box sets X; Y is re-fitted for the same reason as above,
            # falling back to the drawn height if the fit has nothing to go on.
            self._zoom_to(x0, x1, max(y0, 0.0), y1, rebin=True, auto_y=True)

    def _zoom_to(
        self,
        x0: float,
        x1: float,
        y0: "float | None",
        y1: "float | None",
        *,
        rebin: bool,
        auto_y: bool = False,
    ) -> None:
        """Apply a zoom rectangle, optionally re-binning for the new x span.

        Re-binning goes through ``_draw()``, which re-bins over the **full** data
        range and then re-fits the view, so the requested range is re-applied
        afterwards.

        A ``None`` y bound (or ``auto_y``) means "fit the height to the re-binned
        bars now visible": a finer bin width splits each bar's counts, so the
        peak the user zoomed into lands at a different height and the pre-zoom
        (or drawn) y range would leave it squashed or off-screen.
        """
        if rebin:
            width = self._zoom_bin_width(x0, x1)
            if width is not None:
                self._auto_bin_width = width
                self._set_bin_spin(width)
                self._draw()
        if y0 is None or y1 is None or auto_y:
            fitted = self._auto_y_for_x_range(x0, x1)
            if fitted is not None:
                y0, y1 = fitted
        if y0 is None or y1 is None:
            _vx, (y0, y1) = self._view_box.viewRange()
        # Never zoom into the empty band below zero.
        y0 = max(float(y0), 0.0)
        y1 = float(y1)
        if y1 <= y0:
            y1 = y0 + max(abs(y0) * 0.01, 1.0)
        self._view_box.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0.0)
        self._update_filter_edit_labels()

    def _auto_y_for_x_range(self, x0: float, x1: float) -> "tuple[float, float] | None":
        """``(0, peak)`` over the bars currently visible in ``[x0, x1]``.

        Read off the drawn bars rather than the source values so it reflects the
        bin width actually in use. ``None`` when there is nothing to measure
        (empty plot, or a multi-series ``all [stacked]`` view that does not use
        the bar item) — the caller then leaves the height alone.
        """
        opts = getattr(self._hist_item, "opts", {}) or {}
        x = np.asarray(opts.get("x", []), dtype=float).ravel()
        heights = np.asarray(opts.get("height", []), dtype=float).ravel()
        if x.size == 0 or heights.size != x.size:
            return None
        # A bar counts as visible when any part of it overlaps the span.
        try:
            half = abs(float(opts.get("width", 0.0) or 0.0)) / 2.0
        except (TypeError, ValueError):
            half = 0.0
        inside = (x + half >= x0) & (x - half <= x1)
        if not inside.any():
            return None
        visible = heights[inside]
        visible = visible[np.isfinite(visible)]
        if visible.size == 0:
            return None
        top = float(np.max(visible))
        if not np.isfinite(top) or top <= 0.0:
            return None
        return 0.0, top * (1.0 + _ZOOM_Y_HEADROOM)

    def _zoom_bin_width(self, x0: float, x1: float) -> "float | None":
        """A bin width that resolves detail inside the zoomed x span.

        Aims for a bin *count* across the zoomed span, scaled by how many values
        actually land there (``_ZOOM_VALUES_PER_BIN`` each, clamped to
        ``_ZOOM_MIN_BINS``…``_ZOOM_MAX_BINS``), so the window always shows a
        readable number of bars.

        Freedman-Diaconis is deliberately **not** consulted. It is the right rule
        for a whole distribution but the wrong one for a zoom window: it trades
        span for sample size (a 10x zoom refines only ~4.6x, leaving the window
        emptier than before), and on a window holding a sharp concentration its
        IQR collapses and it asks for thousands of sub-pixel bars, nearly all of
        count 0 or 1 — which renders as one solid block of equal-height bars.

        ``None`` when there is nothing to re-bin from.
        """
        span = float(x1) - float(x0)
        if not np.isfinite([x0, x1]).all() or span <= 0:
            return None
        vals = np.asarray(self._vals, dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        inside = vals[(vals >= x0) & (vals <= x1)]
        target = int(np.clip(
            inside.size // _ZOOM_VALUES_PER_BIN, _ZOOM_MIN_BINS, _ZOOM_MAX_BINS,
        ))
        width = span / target
        # _bin_edges_for caps the full-range bin count, so a finer width than
        # that would be silently clamped and the spin box would lie. This is what
        # limits how far a zoom can keep refining (see _MAX_HISTOGRAM_BINS).
        full_span = float(np.max(vals)) - float(np.min(vals))
        if full_span > 0:
            width = max(width, full_span / _MAX_HISTOGRAM_BINS)
        return width if np.isfinite(width) and width > 0 else None

    def _clear_zoom_preview(self) -> None:
        if self._zoom_preview is not None:
            try:
                self._plot.removeItem(self._zoom_preview)
            except Exception:
                pass
            self._zoom_preview = None

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
        raw = self._materialized_values(ds, attr_name)
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
        """One value per trace (or the filtered per-loc values for ``per loc``).

        NB: a trace read-out is computed from **every** localization in the
        trace — *ftr* is deliberately not applied. A trace's mean/median/... is a
        property of the trace itself, so it does not shift as the user tunes an
        unrelated filter. (``per loc`` does honour *ftr*.)
        """
        if mode == "per loc":
            return raw[ftr]

        ti = ds.prop.trace_idx
        n_tr = ds.prop.num_traces
        fn = trace_agg_func(mode)
        with np.errstate(all="ignore"):
            return np.array([fn(raw[ti[i, 0] : ti[i, 1] + 1]) for i in range(n_tr)])

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
        itr: "str | int | None" = None,
        on_update=None,
        on_finish=None,
        on_cancel=None,
        restore_view_on_finish: bool = True,
        restore_view_on_cancel: bool = True,
    ) -> None:
        """Display and edit a filter range on top of this histogram.

        ``itr`` mirrors the filter row's own value space onto the plot, so the
        region the user drags is over exactly the numbers the filter tests.
        ``None`` keeps the histogram's current setting.
        """
        self._clear_filter_edit(restore=False)
        # Set the attribute first so cfr/efc can pick their effective iteration.
        if self._attr_combo.findText(attr) >= 0:
            self._attr_combo.blockSignals(True)
            self._attr_combo.setCurrentText(attr)
            self._attr_combo.blockSignals(False)
        if self._agg_combo.findText(mode) >= 0:
            self._agg_combo.blockSignals(True)
            self._agg_combo.setCurrentText(mode)
            self._agg_combo.blockSignals(False)
        applied_itr = self._apply_filter_iteration(itr, attr)
        if not applied_itr:
            self._edit_itr = None
            # cfr/efc: edit on their effective-iteration view (its canonical
            # per-loc value); a materialized selection, so filter-edit works.
            self._enforce_effective_iteration()
        # Filter editing needs the materialized store. If still in raw mode (a
        # normal attribute on a non-last / all-iteration selection), fall back
        # to the default last-iteration view.
        if self._is_raw_mode():
            self._iter_combo.blockSignals(True)
            self._valid_chk.blockSignals(True)
            ds0 = self._dataset()
            if ds0 is not None:
                self._iter_combo.setCurrentText(self._default_iter_label_for(ds0, attr))
            self._valid_chk.setChecked(True)
            self._iter_combo.blockSignals(False)
            self._valid_chk.blockSignals(False)
            self._clear_raw_items()
        ds = self._dataset()
        saved_state = dict((ds.state.get(self._view_state_key, {}) if ds is not None else {}))
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

        range_color, range_alpha, bounds_color, text_color = self._filter_style()
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
            color=text_color,
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
        # The label is anchored to the current view corner and re-positioned on
        # every range change, so letting it contribute to auto-range bounds
        # creates a feedback loop that grows the view without bound.
        self._plot.addItem(report_label, ignoreBounds=True)
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

    def _apply_filter_iteration(self, itr: "str | int | None", attr: str) -> bool:
        """Point the Iter dropdown at a filter row's selector; True when applied.

        Accepts the persisted spec tokens (``"last"``, ``"effective"``, an int,
        ``"sum"``/``"average"``). ``"effective"`` resolves against *attr*.
        Returns False when there is nothing to apply, so the caller falls back
        to its own default.
        """
        if itr is None or self._iter_combo.count() == 0:
            return False
        ds = self._dataset()
        sel = itr
        if isinstance(sel, str) and sel.strip().lower() in ("effective", "auto", "browse"):
            eff = effective_iteration_for_attr(ds, attr) if ds is not None else None
            sel = int(eff) if eff is not None else "last"
        label = iteration_selector_label(sel, self._num_itr(ds) if ds is not None else 1)
        if not label or self._iter_combo.findText(label) < 0:
            return False
        self._iter_combo.blockSignals(True)
        self._iter_combo.setCurrentText(label)
        self._iter_combo.blockSignals(False)
        # Remember a concrete iteration so it keeps the materialized path (and
        # therefore the draggable region) while this edit is open.
        self._edit_itr = int(sel) if isinstance(sel, (int, np.integer)) else None
        return True

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
        x_label = self._value_label(attr_name)
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
        raw = self._materialized_values(ds, attr_name)
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

    def _filter_style(self) -> tuple[QColor, int, QColor, QColor]:
        range_color = QColor(*viewer_color(self._state.prefs, "filter_range"))
        bounds_color = QColor(*viewer_color(self._state.prefs, "filter_bounds"))
        text_color = QColor(*viewer_color(self._state.prefs, "filter_text"))
        return range_color, range_color.alpha(), bounds_color, text_color

    def _restyle_filter_edit(self) -> None:
        edit = self._filter_edit or {}
        region = edit.get("region")
        report = edit.get("report_label")
        if region is None and report is None:
            return
        range_color, range_alpha, bounds_color, text_color = self._filter_style()
        if region is not None:
            region.setBrush(pg.mkBrush(
                range_color.red(), range_color.green(), range_color.blue(), range_alpha
            ))
            for line in getattr(region, "lines", ()):
                line.setPen(pg.mkPen(bounds_color, width=4))
                try:
                    line.setHoverPen(pg.mkPen(bounds_color, width=5))
                except Exception:
                    pass
        if report is not None:
            report.setColor(text_color)
            try:
                report.setBorder(pg.mkPen(bounds_color, width=1))
            except Exception:
                pass

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
            values, range_values = _filter_spinner_values(
                ds, attr, mode, itr=self._current_value_itr(),
            )
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
        raw = self._materialized_values(ds, self._attr_combo.currentText())
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

    def current_filter_edit_state(self) -> dict:
        """Return the histogram controls currently associated with an edit.

        The Filter Dialog uses this snapshot when Update/Finish is pressed so
        the row records the state the user actually edited, including a manual
        iteration browse away from cfr/efc's effective iteration.
        """
        itr_sel, render = self._selection()
        state = {
            "attribute": self._attr_combo.currentText(),
            "aggregation": self._agg_combo.currentText(),
            "itr": itr_sel,
            "iteration_label": self._iter_combo.currentText(),
            "render_mode": render,
        }
        if self._filter_edit is not None:
            self._filter_edit["attr"] = state["attribute"]
            self._filter_edit["mode"] = state["aggregation"]
            self._filter_edit["itr"] = state["itr"]
        return state

    def _update_filter_edit(self) -> None:
        edit = self._filter_edit
        if not edit:
            return
        callback = edit.get("on_update")
        if callable(callback):
            callback(*self._current_filter_edit_values(), self.current_filter_edit_state())

    def _finish_filter_edit(self) -> None:
        edit = self._filter_edit
        if not edit:
            return
        callback = edit.get("on_finish")
        values = self._current_filter_edit_values()
        state = self.current_filter_edit_state()
        self._clear_filter_edit(restore=bool(edit.get("restore_view_on_finish", True)))
        if callable(callback):
            callback(*values, state)

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
        self._edit_itr = None
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
        self._enforce_effective_iteration()
        self._style_iteration_boldness()             # re-bold for the new attribute
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
            # "trace mean" can be switched off in Preferences, and setCurrentText
            # is a no-op for a missing entry — which would leave a track-level
            # attribute on a per-loc read-out while the combo says otherwise.
            forced = "trace mean"
            if self._agg_combo.findText(forced) < 0:
                forced = next(
                    (m for m in _TRACE_AGG_MODES if self._agg_combo.findText(m) >= 0),
                    "per loc",
                )
            self._agg_combo.setCurrentText(forced)
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

    def _on_attributes_changed(self, idx: int) -> None:
        if idx == self._dataset_idx:
            self._refresh()

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
