"""
minflux_viewer.ui.attribute_separation_dialog
==============================================
Attribute-agnostic **channel separation** dialog — the redesigned tool behind
*Process › Channel › Separate Channel by DCR* and the foundation of the future
generic *Convert Dataset to Multi-Channel Overlay*.

It shows the distribution of a chosen MINFLUX attribute (DCR is the first
instance), lets you place **channel windows** on that axis three ways —

* fit a mixture distribution (:mod:`analysis.distribution_fit`) and split at the
  Bayes boundaries (*Fit* / *Auto*),
* *Place evenly*,
* drag the LUT-coloured region on the histogram / edit start–end in the table,

— and assigns each **trace** to a channel by mean / median / majority vote
(optionally on photon-weighted DCR). *Apply* builds one dataset per channel plus
a hidden *unassigned* channel and combines them as a render overlay via
``main_window.apply_channel_separation``.

Borrowed from the Histogram/Filter UI (row 1): iteration selector, aggregation,
bin size, Log(data), Reset. The draggable regions are **bounded** to a 5%-padded
view so they can't be dragged off-screen (the reported run-away-drag issue).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..analysis.attribute_channels import (
    Channel,
    assign_traces,
    channels_from_fit,
    place_evenly,
    pooled_dcr_per_loc,
)
from ..analysis.distribution_fit import (
    DISTRIBUTION_LABELS,
    DISTRIBUTIONS,
    auto_fit,
    fit_mixture,
)
from ..core.iteration import FLATTEN_LABEL, iteration_labels, ordinal, parse_iteration_label
from ..core.loader import attr_values_1d, mfx_get
from ..core.overlay import CHANNEL_LUTS, PURE_COLOR_RGB
from ..utils.filters import raw_trace_aggregate

_DISPLAY_AGG = ["per loc", "trace mean", "trace median"]
_DECISION_MODES = ["trace mean", "trace median", "trace majority vote"]
_CHANNEL_COUNTS = [str(n) for n in range(2, 8)]        # 2..7


def _rgb_for_lut(lut: str) -> tuple[int, int, int]:
    return PURE_COLOR_RGB.get(lut, (120, 120, 120))


class AttributeSeparationDialog(QDialog):
    """Separate one dataset into a multi-channel overlay by an attribute's
    distribution. Modeless; one instance per (dataset, attribute)."""

    def __init__(self, state, dataset_idx: int, *, attribute: str = "dcr",
                 title: str | None = None, default_distribution: str = "gaussian",
                 allow_photon_weight: bool = False, pick_attribute: bool = False,
                 owner=None) -> None:
        super().__init__(None)
        self._state = state
        self._idx = dataset_idx
        self._attribute = attribute
        self._owner = owner
        self._pick_attribute = bool(pick_attribute)
        self._default_distribution = default_distribution if default_distribution in DISTRIBUTIONS else "gaussian"
        self._allow_photon_weight = bool(allow_photon_weight)
        self.setWindowTitle(title or f"Separate Channels by {attribute.upper()}")
        self.resize(1160, 680)                              # wide, normal height (3.1)

        self._values = np.empty(0)                          # transformed display values (fit basis)
        self._bin_width = 0.01
        self._rows: list[dict] = []                         # channel table rows
        self._fit_result = None                             # last MixtureResult (for overlay)
        self._fit_channel_luts: list[str] = []              # LUT per fit component (overlay colour)
        self._synchronizing = False
        self._suspend = False

        self._closing = False
        self._build_ui()
        self._recompute_values(reset_bin=True)
        # Seed instantly with evenly-placed channels (no heavy fit on the
        # construction path), then refine with the default distribution fit once
        # the event loop is running — keeps the dialog snappy and off the
        # sklearn call during construction.
        self._seed_default_channels()
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, self._initial_fit)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        ds = self._dataset()
        self._title = QLabel()
        self._title.setWordWrap(True)
        root.addWidget(self._title)

        # Attribute picker (generic "by attribute" mode). Hidden for the fixed
        # single-attribute entries (e.g. by-DCR), which lock to their attribute.
        if self._pick_attribute:
            arow = QHBoxLayout()
            arow.addWidget(QLabel("Separate by attribute:"))
            self._attr_combo = QComboBox()
            self._attr_combo.addItems(self._attribute_candidates())
            if self._attr_combo.findText(self._attribute) >= 0:
                self._attr_combo.setCurrentText(self._attribute)
            elif self._attr_combo.count():
                self._attribute = self._attr_combo.currentText()
            self._attr_combo.currentTextChanged.connect(self._on_attribute_changed)
            arow.addWidget(self._attr_combo)
            arow.addStretch(1)
            root.addLayout(arow)
        else:
            self._attr_combo = None
        self._update_title()

        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", self._attribute.upper())
        self._plot.setLabel("left", "count")
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        self._plot.setMinimumHeight(240)
        root.addWidget(self._plot, 1)

        # --- Row 1: borrowed histogram/filter controls ---------------------
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Iteration:"))
        self._iter_combo = QComboBox()
        self._iter_combo.currentTextChanged.connect(lambda *_: self._on_basis_changed())
        r1.addWidget(self._iter_combo)
        r1.addWidget(QLabel("Values:"))
        self._agg_combo = QComboBox()
        self._agg_combo.addItems(_DISPLAY_AGG)
        self._agg_combo.currentTextChanged.connect(lambda *_: self._on_basis_changed())
        r1.addWidget(self._agg_combo)
        self._photon_chk = QCheckBox("Photon-weighted DCR")
        self._photon_chk.setToolTip(
            "Pool DCR over the final-scale iterations weighted by photons (eco): "
            "Σ dcr·eco / Σ eco. Averages out DCR fluctuations for cleaner peaks.")
        self._photon_chk.toggled.connect(lambda *_: self._on_photon_toggled())
        self._photon_chk.setVisible(self._allow_photon_weight and self._attribute == "dcr")
        r1.addWidget(self._photon_chk)
        r1.addWidget(QLabel("Bin size:"))
        self._bin_spin = QDoubleSpinBox()
        self._bin_spin.setDecimals(4)
        self._bin_spin.setRange(1e-4, 1e9)
        self._bin_spin.valueChanged.connect(self._on_bin_changed)
        r1.addWidget(self._bin_spin)
        self._log_chk = QCheckBox("Log(data)")
        self._log_chk.toggled.connect(lambda *_: self._on_basis_changed(reset_bin=True))
        r1.addWidget(self._log_chk)
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._reset)
        r1.addWidget(reset_btn)
        r1.addStretch(1)
        root.addLayout(r1)

        # --- Row 2: channel ops + trace decision --------------------------
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Channels:"))
        self._nch_combo = QComboBox()
        self._nch_combo.addItems(_CHANNEL_COUNTS)
        r2.addWidget(self._nch_combo)
        even_btn = QPushButton("Place evenly")
        even_btn.clicked.connect(self._place_evenly)
        r2.addWidget(even_btn)
        add_btn = QPushButton("Add channel")
        add_btn.clicked.connect(self._add_channel)
        r2.addWidget(add_btn)
        rm_btn = QPushButton("Remove selected")
        rm_btn.clicked.connect(self._remove_selected)
        r2.addWidget(rm_btn)
        r2.addStretch(1)
        r2.addWidget(QLabel("Decide trace by:"))
        self._decision_combo = QComboBox()
        self._decision_combo.addItems(_DECISION_MODES)
        self._decision_combo.currentTextChanged.connect(lambda *_: self._refresh_counts())
        r2.addWidget(self._decision_combo)
        r2.addWidget(QLabel("Min %:"))
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setDecimals(0)
        self._conf_spin.setRange(50, 100)
        self._conf_spin.setValue(50)
        self._conf_spin.setToolTip(
            "Majority vote only: required agreement fraction of a trace's per-loc "
            "votes; below it the trace is unassigned.")
        self._conf_spin.valueChanged.connect(lambda *_: self._refresh_counts())
        r2.addWidget(self._conf_spin)
        root.addLayout(r2)

        # --- Row 3: fit + residual ----------------------------------------
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Fit:"))
        self._fit_combo = QComboBox()
        for key in DISTRIBUTIONS:
            self._fit_combo.addItem(DISTRIBUTION_LABELS[key], key)
        self._fit_combo.setCurrentIndex(DISTRIBUTIONS.index(self._default_distribution))
        r3.addWidget(self._fit_combo)
        r3.addWidget(QLabel("Components:"))
        self._comp_combo = QComboBox()
        self._comp_combo.addItems(_CHANNEL_COUNTS)
        r3.addWidget(self._comp_combo)
        fit_btn = QPushButton("Fit")
        fit_btn.clicked.connect(self._run_fit)
        r3.addWidget(fit_btn)
        auto_btn = QPushButton("Auto")
        auto_btn.setToolTip("Pick the best distribution + component count by BIC.")
        auto_btn.clicked.connect(self._auto_fit)
        r3.addWidget(auto_btn)
        r3.addStretch(1)
        self._residual_label = QLabel("residual: —")
        r3.addWidget(self._residual_label)
        r3.addWidget(QLabel("Fit residual:"))
        self._res_fit_combo = QComboBox()
        for key in DISTRIBUTIONS:
            self._res_fit_combo.addItem(DISTRIBUTION_LABELS[key], key)
        r3.addWidget(self._res_fit_combo)
        res_fit_btn = QPushButton("Fit")
        res_fit_btn.setToolTip("Fit the chosen distribution to the currently-unassigned "
                               "localizations and add channels for it.")
        res_fit_btn.clicked.connect(self._fit_residual)
        r3.addWidget(res_fit_btn)
        res_auto_btn = QPushButton("Auto")
        res_auto_btn.clicked.connect(lambda: self._fit_residual(auto=True))
        r3.addWidget(res_auto_btn)
        root.addLayout(r3)

        # --- Channel table -------------------------------------------------
        self._table = QTableWidget(0, 5, self)
        self._table.setHorizontalHeaderLabels(["Channel name", "Start", "End", "LUT / color", "Locs"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.setMinimumHeight(150)
        root.addWidget(self._table)

        # --- buttons -------------------------------------------------------
        btns = QHBoxLayout()
        btns.addStretch(1)
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._apply)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.close)
        btns.addWidget(self._apply_btn)
        btns.addWidget(cancel_btn)
        root.addLayout(btns)

        # iteration options
        n_itr = int(ds.metadata.get("raw_num_itr", 1)) if ds else 1
        labels = iteration_labels(n_itr)
        self._iter_combo.blockSignals(True)
        if labels:
            self._iter_combo.addItems(labels)
            self._iter_combo.setCurrentText(FLATTEN_LABEL)
            self._iter_combo.setVisible(True)
        else:
            self._iter_combo.setVisible(False)
        self._iter_combo.blockSignals(False)

    # ------------------------------------------------------------ data path
    def _dataset(self):
        if 0 <= self._idx < len(self._state.datasets):
            return self._state.datasets[self._idx]
        return None

    def _update_title(self) -> None:
        ds = self._dataset()
        self._title.setText(
            f"<b>{ds.name if ds else '(no dataset)'}</b> — separate channels by "
            f"<b>{self._attribute.upper()}</b>")

    def _attribute_candidates(self) -> list[str]:
        """Numeric per-loc attributes worth separating by (variable, not id/flag)."""
        ds = self._dataset()
        if ds is None:
            return [self._attribute]
        from ..core.attributes import plot_attribute_names
        skip = {"tid", "vld", "idx", "itr"}
        out: list[str] = []
        for name in plot_attribute_names(ds, self._state.prefs, numeric_only=True, exclude=skip):
            v = attr_values_1d(ds, name)
            if v is None:
                continue
            v = np.asarray(v, dtype=float).ravel()
            finite = v[np.isfinite(v)]
            if finite.size and np.unique(finite).size > 2:      # varies enough to split
                out.append(name)
        # keep the requested attribute selectable if it is actually available
        if self._attribute and self._attribute not in out and attr_values_1d(ds, self._attribute) is not None:
            out.insert(0, self._attribute)
        return out or [self._attribute]

    def _on_attribute_changed(self, name: str) -> None:
        if self._suspend or not name:
            return
        self._attribute = name
        self._photon_chk.setVisible(self._allow_photon_weight and name == "dcr")
        if name != "dcr":
            self._photon_chk.setChecked(False)
        self._plot.setLabel("bottom", name.upper())
        self._update_title()
        self._fit_result = None
        self._recompute_values(reset_bin=True)
        self._seed_default_channels()
        self._refresh_counts()

    def _selection(self):
        return parse_iteration_label(self._iter_combo.currentText())

    def _photon_weight_on(self) -> bool:
        return self._allow_photon_weight and self._attribute == "dcr" and self._photon_chk.isChecked()

    def _log_on(self) -> bool:
        return self._log_chk.isChecked()

    def _transform_drop(self, vals) -> np.ndarray:
        """Finite (and >0 if Log) values — for the histogram / fit basis."""
        vals = np.asarray(vals, dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        if self._log_on():
            vals = vals[vals > 0.0]
            vals = np.log(vals)
        return vals

    def _transform_keep(self, vals) -> np.ndarray:
        """Same transform but length-preserving (NaN where dropped) — for
        assignment, so labels stay aligned to num_loc rows."""
        vals = np.asarray(vals, dtype=float).ravel()
        if self._log_on():
            return np.where(vals > 0.0, np.log(vals), np.nan)
        return vals

    def _per_loc_raw(self):
        """Per-localization attribute values (num_loc), un-transformed."""
        ds = self._dataset()
        if ds is None:
            return None
        if self._photon_weight_on():
            v = pooled_dcr_per_loc(ds)
            if v is not None:
                return np.asarray(v, dtype=float).ravel()
        v = attr_values_1d(ds, self._attribute)
        return None if v is None else np.asarray(v, dtype=float).ravel()

    def _display_values(self) -> np.ndarray:
        """Transformed values driving the histogram + fit."""
        ds = self._dataset()
        if ds is None:
            return np.empty(0)
        if self._photon_weight_on():
            return self._transform_drop(self._per_loc_raw())
        itr_sel, _ = self._selection()
        vals = mfx_get(ds, self._attribute, itr=itr_sel, vld_only=True)
        if vals is None:
            return np.empty(0)
        vals = np.asarray(vals).ravel().astype(float)
        agg = self._agg_combo.currentText()
        if agg != "per loc":
            tid = mfx_get(ds, "tid", itr=itr_sel, vld_only=True)
            if tid is not None:
                vals = raw_trace_aggregate(vals, np.asarray(tid).ravel(), agg)
        return self._transform_drop(vals)

    def _assign_basis(self):
        """(values, tid) per num_loc localization in channel (display) space."""
        ds = self._dataset()
        raw = self._per_loc_raw()
        if ds is None or raw is None:
            return None, None
        v = self._transform_keep(raw)
        tid = attr_values_1d(ds, "tid")
        tid = np.arange(v.size) if tid is None else np.asarray(tid).ravel()
        return v, tid

    def _data_range(self):
        """(lo, hi) padded 5% on each side of the display data (view + region bounds)."""
        v = self._values
        if v.size == 0:
            return 0.0, 1.0
        lo, hi = float(np.min(v)), float(np.max(v))
        span = hi - lo
        if span <= 0:
            span = abs(hi) or 1.0
        pad = 0.05 * span
        return lo - pad, hi + pad

    # ------------------------------------------------------------- compute
    def _recompute_values(self, *, reset_bin: bool = False) -> None:
        self._values = self._display_values()
        if reset_bin:
            self._auto_bin()
        self._apply_region_bounds()
        self._redraw()

    def _auto_bin(self) -> None:
        v = self._values
        if v.size < 2:
            return
        lo, hi = float(v.min()), float(v.max())
        span = hi - lo
        if span <= 0:
            return
        iqr = float(np.subtract(*np.percentile(v, [75, 25]))) * -1.0
        bw = 2.0 * abs(iqr) / (v.size ** (1.0 / 3.0)) if iqr else span / 60.0
        if not np.isfinite(bw) or bw <= 0:
            bw = span / 60.0
        bw = float(np.clip(bw, span / 250.0, span / 15.0))
        self._bin_width = bw
        self._suspend = True
        self._bin_spin.setValue(bw)
        self._suspend = False

    def _on_basis_changed(self, *, reset_bin: bool = False) -> None:
        if self._suspend:
            return
        self._fit_result = None                # display basis changed → old fit curve is stale
        self._recompute_values(reset_bin=reset_bin)
        self._refresh_counts()

    def _on_photon_toggled(self) -> None:
        on = self._photon_weight_on()
        # eco-weighted DCR is inherently per-loc; iteration/aggregation don't apply.
        self._iter_combo.setEnabled(not on)
        self._agg_combo.setEnabled(not on)
        self._on_basis_changed(reset_bin=True)

    def _on_bin_changed(self, value: float) -> None:
        if self._suspend:
            return
        self._bin_width = float(value)
        self._redraw()

    def _reset(self) -> None:
        self._suspend = True
        self._log_chk.setChecked(False)
        self._agg_combo.setCurrentText("per loc")
        if self._photon_chk.isVisible():
            self._photon_chk.setChecked(False)
        if self._iter_combo.isVisible():
            self._iter_combo.setCurrentText(FLATTEN_LABEL)
        self._iter_combo.setEnabled(True)
        self._agg_combo.setEnabled(True)
        self._conf_spin.setValue(50)
        self._decision_combo.setCurrentText("trace mean")
        self._suspend = False
        self._recompute_values(reset_bin=True)
        self._run_fit()
        try:
            self._plot.getViewBox().enableAutoRange(y=True)
        except Exception:
            pass

    # ------------------------------------------------------------- drawing
    def _redraw(self) -> None:
        self._plot.clear()
        self._reset_legend()
        v = self._values
        plo, phi = self._data_range()
        if v.size == 0:
            self._readd_regions(plo, phi)
            return
        lo, hi = float(v.min()), float(v.max())
        bw = max(self._bin_width, (hi - lo) / 1000.0 or 1e-6)
        nbins = int(np.clip(np.ceil((hi - lo) / bw), 1, 2000))
        edges = lo + bw * np.arange(nbins + 1)

        _, render = self._selection()
        if render == "stacked" and not self._photon_weight_on() and self._num_itr() > 1:
            # all [stacked]: one translucent series per iteration + legend (like the
            # Histogram window). The fit / channels stay on the pooled distribution;
            # the per-iteration series are a display aid, so the fit curve is omitted
            # here (it is pooled-scale and would dwarf the individual iterations).
            self._draw_stacked_series(edges)
        else:
            counts, _ = np.histogram(v, bins=edges)
            centers = 0.5 * (edges[:-1] + edges[1:])
            self._plot.addItem(pg.BarGraphItem(
                x=centers, height=counts, width=bw * 0.92, brush=(150, 150, 150, 150), pen=None))
            if self._fit_result is not None:
                self._draw_fit_overlay(lo, hi, bw)

        self._readd_regions(plo, phi)

    def _draw_fit_overlay(self, lo: float, hi: float, bw: float) -> None:
        """Fitted mixture overlay (per-component, coloured by its channel LUT)."""
        xs = np.linspace(lo, hi, 512)
        comp = self._fit_result.component_pdfs(xs)
        scale = self._values.size * bw
        for k in range(comp.shape[0]):
            rgb = (_rgb_for_lut(self._fit_channel_luts[k])
                   if k < len(self._fit_channel_luts) else (200, 200, 200))
            self._plot.addItem(pg.PlotDataItem(xs, comp[k] * scale, pen=pg.mkPen(rgb, width=2)))
        self._plot.addItem(pg.PlotDataItem(
            xs, comp.sum(axis=0) * scale,
            pen=pg.mkPen((160, 160, 160), width=1, style=Qt.PenStyle.DashLine)))

    def _draw_stacked_series(self, edges: np.ndarray) -> None:
        """One translucent step-histogram per iteration, coloured + with a legend."""
        from .histogram_window import _iter_color

        series = []
        for k in range(self._num_itr()):
            vals = self._values_for_itr(k)
            if vals.size:
                series.append((k, vals))
        if not series:
            return
        alpha = int(np.clip(round(255.0 / max(len(series), 1) * 1.4), 45, 200))
        self._plot.addLegend(offset=(-10, 10))
        for k, vals in series:
            counts, _ = np.histogram(vals, bins=edges)
            r, g, b = _iter_color(k)
            self._plot.plot(
                edges, counts, stepMode="center", fillLevel=0,
                brush=(r, g, b, alpha), pen=pg.mkPen(r, g, b, width=1),
                name=ordinal(k + 1))
        try:                                        # let the per-iteration series fit
            self._plot.getViewBox().enableAutoRange(y=True)
        except Exception:
            pass

    def _values_for_itr(self, k: int) -> np.ndarray:
        """Transformed values of the attribute at a single iteration *k* (for the
        stacked per-iteration display)."""
        ds = self._dataset()
        if ds is None:
            return np.empty(0)
        vals = mfx_get(ds, self._attribute, itr=int(k), vld_only=True)
        if vals is None:
            return np.empty(0)
        vals = np.asarray(vals).ravel().astype(float)
        agg = self._agg_combo.currentText()
        if agg != "per loc":
            tid = mfx_get(ds, "tid", itr=int(k), vld_only=True)
            if tid is not None:
                vals = raw_trace_aggregate(vals, np.asarray(tid).ravel(), agg)
        return self._transform_drop(vals)

    def _num_itr(self) -> int:
        ds = self._dataset()
        if ds is None:
            return 1
        return max(1, int(ds.metadata.get("raw_num_itr", getattr(ds.prop, "num_itr", 1) or 1)))

    def _reset_legend(self) -> None:
        """Drop any cached legend so a later ``addLegend`` shows (PlotItem caches
        ``legend``; ``clear()`` leaves the stale attribute behind)."""
        try:
            pi = self._plot.getPlotItem()
            if pi.legend is not None:
                pi.legend.scene().removeItem(pi.legend)
            pi.legend = None
        except Exception:
            pass

    def _readd_regions(self, plo: float, phi: float) -> None:
        for row in self._rows:
            self._plot.addItem(row["region"])
            row["region"].setBounds((plo, phi))
        try:
            self._plot.getViewBox().setXRange(plo, phi, padding=0)
        except Exception:
            pass

    def _apply_region_bounds(self) -> None:
        plo, phi = self._data_range()
        for row in self._rows:
            row["region"].setBounds((plo, phi))

    # ------------------------------------------------------- channel table
    def _clear_rows(self) -> None:
        for row in self._rows:
            try:
                self._plot.removeItem(row["region"])
            except Exception:
                pass
        self._rows.clear()
        self._table.setRowCount(0)

    def _set_channels(self, channels: list[Channel]) -> None:
        self._clear_rows()
        for ch in channels:
            self._append_row(ch)
        if channels:
            self._nch_combo.blockSignals(True)
            self._nch_combo.setCurrentText(str(min(max(len(channels), 2), 7)))
            self._nch_combo.blockSignals(False)
        self._refresh_counts()

    def _append_row(self, ch: Channel) -> None:
        r = self._table.rowCount()
        self._table.insertRow(r)
        name_edit = QLineEdit(ch.name)
        start_spin = self._spin(ch.lo)
        end_spin = self._spin(ch.hi)
        lut_combo = QComboBox()
        lut_combo.addItems(CHANNEL_LUTS)
        if lut_combo.findText(ch.lut) < 0:
            lut_combo.addItem(ch.lut)
        lut_combo.setCurrentText(ch.lut)
        count_item = QTableWidgetItem("0")
        count_item.setFlags(count_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        count_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._table.setCellWidget(r, 0, name_edit)
        self._table.setCellWidget(r, 1, start_spin)
        self._table.setCellWidget(r, 2, end_spin)
        self._table.setCellWidget(r, 3, lut_combo)
        self._table.setItem(r, 4, count_item)

        rgb = _rgb_for_lut(ch.lut)
        plo, phi = self._data_range()
        region = pg.LinearRegionItem(
            values=(ch.lo, ch.hi), orientation=pg.LinearRegionItem.Vertical, movable=True,
            brush=(*rgb, 45), pen=pg.mkPen(rgb, width=2), hoverPen=pg.mkPen(rgb, width=3),
            bounds=(plo, phi))
        region.setZValue(10 + r)
        self._plot.addItem(region)

        row = {"name": name_edit, "start": start_spin, "end": end_spin,
               "lut": lut_combo, "count": count_item, "region": region}
        self._rows.append(row)
        name_edit.textChanged.connect(self._refresh_counts)
        start_spin.valueChanged.connect(lambda _v, item=row: self._spin_changed(item))
        end_spin.valueChanged.connect(lambda _v, item=row: self._spin_changed(item))
        lut_combo.currentTextChanged.connect(lambda _v, item=row: self._lut_changed(item))
        region.sigRegionChanged.connect(lambda _r, item=row: self._region_changed(item))

    @staticmethod
    def _spin(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(4)
        spin.setRange(-1e12, 1e12)
        spin.setSingleStep(0.01)
        spin.setValue(float(value))
        spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        return spin

    def _spin_changed(self, row: dict) -> None:
        if self._synchronizing:
            return
        self._synchronizing = True
        row["region"].setRegion((row["start"].value(), row["end"].value()))
        self._synchronizing = False
        self._fit_result = None                # manual edit → drop stale fit overlay
        self._refresh_counts()

    def _region_changed(self, row: dict) -> None:
        if self._synchronizing:
            return
        lo, hi = sorted(float(v) for v in row["region"].getRegion())
        self._synchronizing = True
        row["start"].setValue(lo)
        row["end"].setValue(hi)
        self._synchronizing = False
        self._fit_result = None
        self._refresh_counts()

    def _lut_changed(self, row: dict) -> None:
        rgb = _rgb_for_lut(row["lut"].currentText())
        row["region"].setBrush((*rgb, 45))
        row["region"].setPen(pg.mkPen(rgb, width=2))
        row["region"].setHoverPen(pg.mkPen(rgb, width=3))

    def _current_channels(self) -> list[Channel]:
        out = []
        for row in self._rows:
            lo, hi = sorted((row["start"].value(), row["end"].value()))
            out.append(Channel(name=row["name"].text().strip() or "channel",
                               lo=float(lo), hi=float(hi), lut=row["lut"].currentText()))
        return out

    # ------------------------------------------------------------- actions
    def _base_name(self) -> str:
        ds = self._dataset()
        return ds.name if ds else "channel"

    def _place_evenly(self) -> None:
        n = int(self._nch_combo.currentText())
        lo, hi = (float(self._values.min()), float(self._values.max())) if self._values.size else (0.0, 1.0)
        self._fit_result = None
        self._set_channels(place_evenly(lo, hi, n, base_name=self._base_name(),
                                        attribute=self._attribute, luts=self._lut_cycle()))
        self._redraw()

    def _lut_cycle(self):
        return CHANNEL_LUTS

    def _seed_default_channels(self) -> None:
        """Place evenly (no fit) so the dialog opens with channels immediately."""
        lo, hi = (float(self._values.min()), float(self._values.max())) if self._values.size else (0.0, 1.0)
        self._set_channels(place_evenly(lo, hi, 2, base_name=self._base_name(),
                                        attribute=self._attribute, luts=self._lut_cycle()))
        self._redraw()

    def _initial_fit(self) -> None:
        """Deferred default fit run once the event loop is live (off the
        construction path). Guarded against a dataset/dialog torn down first."""
        if self._closing or self._dataset() is None:
            return
        self._run_fit()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._closing = True
        super().closeEvent(event)

    def _run_fit(self) -> None:
        if self._closing:
            return
        dist = self._fit_combo.currentData()
        n_comp = int(self._comp_combo.currentText())
        if self._values.size < max(2, n_comp):
            return
        try:
            res = fit_mixture(self._values, dist, n_comp)
        except Exception as exc:
            self._state.log(f"Channel fit failed: {exc}", "WARN")
            return
        self._apply_fit_result(res)

    def _auto_fit(self) -> None:
        if self._values.size < 2:
            return
        try:
            res = auto_fit(self._values, max_components=3)
        except Exception as exc:
            self._state.log(f"Auto fit failed: {exc}", "WARN")
            return
        # reflect the winner in the Fit / Components pickers
        self._suspend = True
        i = self._fit_combo.findData(res.distribution)
        if i >= 0:
            self._fit_combo.setCurrentIndex(i)
        self._comp_combo.setCurrentText(str(min(max(res.n_components, 2), 7)))
        self._suspend = False
        self._apply_fit_result(res)

    def _apply_fit_result(self, res) -> None:
        lo, hi = (float(self._values.min()), float(self._values.max())) if self._values.size else res.domain
        channels = channels_from_fit(res, data_range=(lo, hi), base_name=self._base_name(),
                                     attribute=self._attribute, luts=self._lut_cycle())
        self._fit_result = res
        self._fit_channel_luts = [c.lut for c in channels]
        self._set_channels(channels)
        self._comp_combo.blockSignals(True)
        self._comp_combo.setCurrentText(str(min(max(res.n_components, 2), 7)))
        self._comp_combo.blockSignals(False)
        self._redraw()

    def _add_channel(self) -> None:
        if not self._rows:
            self._place_evenly()
            return
        sel = self._table.currentRow()
        index = sel if 0 <= sel < len(self._rows) else len(self._rows) - 1
        row = self._rows[index]
        lo = row["start"].value()
        hi = row["end"].value()
        mid = 0.5 * (lo + hi)
        self._synchronizing = True
        row["end"].setValue(mid)
        row["region"].setRegion((lo, mid))
        self._synchronizing = False
        new_i = len(self._rows)
        self._append_row(Channel(
            name=f"{self._base_name()} [{self._attribute} {new_i + 1}]",
            lo=float(mid), hi=float(hi), lut=CHANNEL_LUTS[new_i % len(CHANNEL_LUTS)]))
        self._fit_result = None
        self._refresh_counts()

    def _remove_selected(self) -> None:
        selected = sorted({i.row() for i in self._table.selectionModel().selectedRows()})
        if not selected and self._table.currentRow() >= 0:
            selected = [self._table.currentRow()]
        if len(self._rows) - len(selected) < 1:
            QMessageBox.information(self, "Channels", "At least one channel is required.")
            return
        for i in reversed(selected):
            try:
                self._plot.removeItem(self._rows[i]["region"])
            except Exception:
                pass
            self._rows.pop(i)
            self._table.removeRow(i)
        self._fit_result = None
        self._refresh_counts()

    def _fit_residual(self, auto: bool = False) -> None:
        """Fit the (currently unassigned) residual and append channels for it."""
        labels = self._current_labels()
        v, _tid = self._assign_basis()
        if labels is None or v is None:
            return
        res_vals = v[labels == -1]
        res_vals = res_vals[np.isfinite(res_vals)]
        if res_vals.size < 2:
            self._state.log("No residual localizations to fit.", "INFO")
            return
        try:
            if auto:
                res = auto_fit(res_vals, max_components=3)
                i = self._res_fit_combo.findData(res.distribution)
                if i >= 0:
                    self._res_fit_combo.setCurrentIndex(i)
            else:
                res = fit_mixture(res_vals, self._res_fit_combo.currentData(), 2)
        except Exception as exc:
            self._state.log(f"Residual fit failed: {exc}", "WARN")
            return
        extra = channels_from_fit(res, data_range=(float(res_vals.min()), float(res_vals.max())),
                                  base_name=self._base_name(), attribute=f"{self._attribute} residual",
                                  luts=CHANNEL_LUTS[len(self._rows):] + CHANNEL_LUTS)
        for ch in extra:
            self._append_row(ch)
        self._fit_result = None
        self._refresh_counts()

    # ------------------------------------------------------------- counts
    def _current_labels(self):
        v, tid = self._assign_basis()
        channels = self._current_channels()
        if v is None or not channels:
            return None
        mode = self._decision_combo.currentText()
        conf = float(self._conf_spin.value()) / 100.0
        return assign_traces(v, tid, channels, mode=mode, min_confidence=conf)

    def _refresh_counts(self) -> None:
        labels = self._current_labels()
        if labels is None:
            self._residual_label.setText("residual: —")
            self._apply_btn.setEnabled(False)
            return
        ds = self._dataset()
        tid = attr_values_1d(ds, "tid")
        tid = np.arange(labels.size) if tid is None else np.asarray(tid).ravel()

        def n_traces(mask):
            return int(np.unique(tid[mask]).size) if mask.any() else 0

        for k, row in enumerate(self._rows):
            m = labels == k
            row["count"].setText(f"{int(m.sum()):,} / {n_traces(m)} tr")
        res = labels == -1
        self._residual_label.setText(
            f"residual: {int(res.sum()):,} locs / {n_traces(res)} traces")
        assigned = int((labels >= 0).sum())
        self._apply_btn.setEnabled(assigned > 0)

    def _apply(self) -> None:
        labels = self._current_labels()
        channels = self._current_channels()
        if labels is None or not channels:
            return
        if self._owner is None or not hasattr(self._owner, "apply_channel_separation"):
            return
        ok = self._owner.apply_channel_separation(
            self._idx, labels, channels, attribute=self._attribute,
            method_label=f"{self._attribute} channel separation")
        if ok:
            self.close()
