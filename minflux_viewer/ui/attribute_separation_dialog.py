"""
minflux_viewer.ui.attribute_separation_dialog
==============================================
Attribute-agnostic **channel separation** dialog — *Process › Channel › Separate
Channel by DCR* and *Convert to Multi-Channel Overlay (by attribute)*.

It shows the distribution of a chosen MINFLUX attribute and builds channels four
ways:

* **Place evenly** — equal windows across the value range,
* **Detect peak** — the N most prominent peaks, cut at the valleys between them
  (:mod:`analysis.peak_channels`), which also reports when the data supports
  fewer peaks than asked for,
* **Fit** / **Auto** — a mixture fit split at its Bayes boundaries
  (:mod:`analysis.distribution_fit`; *Auto* picks distribution + count by BIC),
* **add channel from ROI / from filter** — a channel whose membership is a
  *selection* rather than a value window, via :mod:`core.channel_labels`.

A channel is therefore one of two kinds, and the dialog keeps both in one list:
a **window** channel (start–end on the attribute axis, draggable on the
histogram) or a **mask** channel (a ROI's in-region rows, or a filter row's
passing rows). Assignment is **per localization**; two policies settle the rest —
what to do where channels overlap, and what to do with rows no channel claims.

*Apply* hands one mask per channel to ``main_window.apply_channel_separation``,
which builds a dataset per channel and combines them as a render overlay.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
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
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..analysis.attribute_channels import (
    Channel,
    channels_from_boundaries,
    channels_from_fit,
    place_evenly,
)
from ..analysis.distribution_fit import (
    DISTRIBUTION_LABELS,
    DISTRIBUTIONS,
    auto_fit,
    fit_mixture,
)
from ..colormaps import channel_colormap_names, representative_rgb
from ..colors import is_solid_color, rgba_hex, solid_color_rgb
from ..core.iteration import FLATTEN_LABEL, iteration_labels, ordinal, parse_iteration_label
from ..core.loader import attr_values_1d, is_value_pool_selector, mfx_get
from ..utils.filters import raw_trace_aggregate
from .attribute_help import apply_attribute_tooltips

_DISPLAY_AGG = ["per loc", "trace mean", "trace median"]

#: Channel colours, in the order channels are created: R, G, B, M, C, Y. Solid
#: colours rather than colormaps — a channel is one population, so one hue reads
#: better in an overlay than a gradient — and they are applied with transparency
#: (see :data:`CHANNEL_ALPHA`) so overlapping channels stay legible.
_CHANNEL_SOLIDS = ("Red", "Green", "Blue", "Magenta", "Cyan", "Yellow")
CHANNEL_ALPHA = 200

_MAX_CHANNELS = 12

#: Where channels claim the same localization.
OVERLAP_KEEP_BOTH = "keep in both channels"
OVERLAP_BY_WEIGHT = "force assign to one channel by weight"
OVERLAP_DISCARD = "discard"
_OVERLAP_MODES = (OVERLAP_KEEP_BOTH, OVERLAP_BY_WEIGHT, OVERLAP_DISCARD)

#: Where no channel claims a localization.
UNASSIGNED_KEEP = "keep in an additional channel"
UNASSIGNED_DISCARD = "discard"
_UNASSIGNED_MODES = (UNASSIGNED_KEEP, UNASSIGNED_DISCARD)

#: Paint budget for the separation preview (it is a thumbnail, not a view).
_PREVIEW_MAX_POINTS = 20_000
_PREVIEW_DELAY_MS = 120


def _rgb_for_lut(lut: str) -> tuple[int, int, int]:
    if is_solid_color(lut):
        return solid_color_rgb(lut)
    if str(lut).startswith("solid:custom:"):
        text = str(lut).split(":")[-1].lstrip("#")
        try:
            return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            return 120, 120, 120
    try:
        return tuple(
            int(round(channel * 255.0)) for channel in representative_rgb(lut)
        )
    except (KeyError, ValueError):
        return 120, 120, 120


def encoded_channel_lut(lut: str, *, alpha: int = CHANNEL_ALPHA) -> str:
    """A channel's LUT as the overlay stores it.

    A solid colour becomes ``solid:custom:#RRGGBBAA`` — the encoding render and
    scatter read the alpha from — so the channel is a transparent solid colour
    rather than an opaque one. Anything else (a colormap a user picked) is kept.
    """
    if is_solid_color(lut):
        return f"solid:custom:{rgba_hex((*solid_color_rgb(lut), int(alpha)))}"
    return lut


class _ChannelRegion(pg.LinearRegionItem):
    """A channel's window on the histogram, with a right-click menu.

    pyqtgraph's region has no context menu of its own, and the channel it stands
    for has to be deletable where the user is looking at it.
    """

    def __init__(self, *args, on_context=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._on_context = on_context

    def mouseClickEvent(self, ev) -> None:                     # noqa: N802 - pyqtgraph API
        if ev.button() == Qt.MouseButton.RightButton and self._on_context is not None:
            ev.accept()
            self._on_context(ev)
            return
        super().mouseClickEvent(ev)


class AttributeSeparationDialog(QDialog):
    """Separate one dataset into a multi-channel overlay by an attribute's
    distribution, by ROIs, or by filters. Modeless; one instance per dataset."""

    def __init__(self, state, dataset_idx: int, *, attribute: str = "dcr",
                 title: str | None = None, default_distribution: str = "gaussian",
                 pick_attribute: bool = False, owner=None) -> None:
        super().__init__(None)
        self._state = state
        self._idx = dataset_idx
        self._attribute = attribute
        self._owner = owner
        self._pick_attribute = bool(pick_attribute)
        self._default_distribution = default_distribution if default_distribution in DISTRIBUTIONS else "gaussian"
        self.setWindowTitle(title or f"Separate Channels by {attribute.upper()}")
        self.resize(1160, 720)

        self._values = np.empty(0)                          # transformed display values (fit basis)
        self._bin_width = 0.01
        self._rows: list[dict] = []                          # channel rows (window or mask)
        self._fit_result = None                              # last MixtureResult (for overlay + weights)
        self._fit_channel_luts: list[str] = []
        self._synchronizing = False
        self._suspend = False
        self._closing = False

        self._build_ui()
        self._recompute_values(reset_bin=True)
        # Seed instantly with evenly-placed channels (no heavy fit on the
        # construction path), then refine with the default distribution fit once
        # the event loop is running — keeps the dialog snappy.
        self._seed_default_channels()
        QTimer.singleShot(0, self._initial_fit)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        ds = self._dataset()
        self._title = QLabel()
        self._title.setWordWrap(True)
        root.addWidget(self._title)

        # --- Row 1: attribute + "add channel from" sources -----------------
        r1 = QHBoxLayout()
        if self._pick_attribute:
            r1.addWidget(QLabel("Separate by attribute:"))
            self._attr_combo = QComboBox()
            self._attr_combo.addItems(self._attribute_candidates())
            apply_attribute_tooltips(self._attr_combo)
            if self._attr_combo.findText(self._attribute) >= 0:
                self._attr_combo.setCurrentText(self._attribute)
            elif self._attr_combo.count():
                self._attribute = self._attr_combo.currentText()
            self._attr_combo.currentTextChanged.connect(self._on_attribute_changed)
            r1.addWidget(self._attr_combo)
        else:
            self._attr_combo = None
        r1.addSpacing(12)
        r1.addWidget(QLabel("add channel from ROI"))
        self._roi_combo = QComboBox()
        self._roi_combo.setMinimumWidth(150)
        self._roi_combo.setToolTip(
            "ROIs of this dataset in the ROI Manager. The channel is the "
            "localizations inside the ROI, not a value window.")
        self._roi_combo.activated.connect(self._on_roi_source_picked)
        r1.addWidget(self._roi_combo)
        r1.addWidget(QLabel(", from filter"))
        self._filter_combo = QComboBox()
        self._filter_combo.setMinimumWidth(150)
        self._filter_combo.setToolTip(
            "Rows of this dataset's current filter. The channel is the "
            "localizations that row keeps.")
        self._filter_combo.activated.connect(self._on_filter_source_picked)
        r1.addWidget(self._filter_combo)
        r1.addStretch(1)
        root.addLayout(r1)
        self._update_title()

        # --- Histogram + separation preview --------------------------------
        split = QSplitter(Qt.Orientation.Horizontal)
        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", self._attribute.upper())
        self._plot.setLabel("left", "count")
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        self._plot.setMinimumHeight(240)
        split.addWidget(self._plot)

        self._preview = pg.PlotWidget()
        self._preview.setMinimumWidth(220)
        self._preview.setAspectLocked(True)
        self._preview.hideAxis("left")
        self._preview.hideAxis("bottom")
        self._preview.setMouseEnabled(x=False, y=False)
        self._preview.setMenuEnabled(False)
        self._preview.setToolTip(
            "Separation preview: the localizations coloured by channel "
            "(downsampled). Unassigned rows are grey.")
        split.addWidget(self._preview)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        root.addWidget(split, 1)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(_PREVIEW_DELAY_MS)
        self._preview_timer.timeout.connect(self._refresh_preview)

        # --- Row 2: channels + how to place them ---------------------------
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Channels:"))
        self._nch_spin = QSpinBox()
        self._nch_spin.setRange(1, _MAX_CHANNELS)
        self._nch_spin.setValue(2)
        r2.addWidget(self._nch_spin)
        even_btn = QPushButton("Place evenly")
        even_btn.clicked.connect(self._place_evenly)
        r2.addWidget(even_btn)
        peak_btn = QPushButton("Detect peak")
        peak_btn.setToolTip(
            "Keep the most prominent peaks of the distribution and cut at the "
            "valleys between them. Says so when the data supports fewer peaks "
            "than the channel count.")
        peak_btn.clicked.connect(self._detect_peaks)
        r2.addWidget(peak_btn)
        fit_btn = QPushButton("Fit")
        fit_btn.setToolTip("Fit the selected distribution with one component per channel.")
        fit_btn.clicked.connect(self._run_fit)
        r2.addWidget(fit_btn)
        self._fit_combo = QComboBox()
        for key in DISTRIBUTIONS:
            self._fit_combo.addItem(DISTRIBUTION_LABELS[key], key)
        self._fit_combo.setCurrentIndex(DISTRIBUTIONS.index(self._default_distribution))
        r2.addWidget(self._fit_combo)
        auto_btn = QPushButton("Auto")
        auto_btn.setToolTip(
            "Suggest the separation: best distribution and channel count by BIC, "
            "then update the channels on the histogram.")
        auto_btn.clicked.connect(self._auto_fit)
        r2.addWidget(auto_btn)
        r2.addStretch(1)
        root.addLayout(r2)

        # --- Row 3: distribution display controls --------------------------
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Iteration:"))
        self._iter_combo = QComboBox()
        self._iter_combo.currentTextChanged.connect(lambda *_: self._on_basis_changed())
        r3.addWidget(self._iter_combo)
        r3.addWidget(QLabel("Values:"))
        self._agg_combo = QComboBox()
        self._agg_combo.addItems(_DISPLAY_AGG)
        self._agg_combo.currentTextChanged.connect(lambda *_: self._on_basis_changed())
        r3.addWidget(self._agg_combo)
        r3.addWidget(QLabel("Bin size:"))
        self._bin_spin = QDoubleSpinBox()
        self._bin_spin.setDecimals(4)
        self._bin_spin.setRange(1e-4, 1e9)
        self._bin_spin.valueChanged.connect(self._on_bin_changed)
        r3.addWidget(self._bin_spin)
        self._log_chk = QCheckBox("Log(data)")
        self._log_chk.toggled.connect(lambda *_: self._on_basis_changed(reset_bin=True))
        r3.addWidget(self._log_chk)
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._reset)
        r3.addWidget(reset_btn)
        r3.addStretch(1)
        root.addLayout(r3)

        # --- Row 4: what to do with contested and unclaimed rows -----------
        r4 = QHBoxLayout()
        r4.addWidget(QLabel("Overlapped region:"))
        self._overlap_combo = QComboBox()
        self._overlap_combo.addItems(_OVERLAP_MODES)
        self._overlap_combo.setToolTip(
            "Localizations claimed by more than one channel.\n"
            "• keep in both channels — the row goes into every channel that claims it\n"
            "• force assign by weight — the fitted component with the highest "
            "posterior wins; without a matching fit (e.g. ROI/filter channels) the "
            "earlier channel wins\n"
            "• discard — the row is left unassigned")
        self._overlap_combo.currentTextChanged.connect(lambda *_: self._refresh_counts())
        r4.addWidget(self._overlap_combo)
        r4.addSpacing(16)
        r4.addWidget(QLabel("Data without assigned channel:"))
        self._unassigned_combo = QComboBox()
        self._unassigned_combo.addItems(_UNASSIGNED_MODES)
        self._unassigned_combo.setToolTip(
            "Localizations no channel claims: keep them as an extra (hidden) "
            "channel so nothing is lost, or leave them out of the overlay.")
        self._unassigned_combo.currentTextChanged.connect(lambda *_: self._refresh_counts())
        r4.addWidget(self._unassigned_combo)
        r4.addStretch(1)
        self._status_label = QLabel("—")
        r4.addWidget(self._status_label)
        root.addLayout(r4)

        # --- Channel table -------------------------------------------------
        self._table = QTableWidget(0, 5, self)
        self._table.setHorizontalHeaderLabels(
            ["Channel name", "Start", "End", "Color", "Locs"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_table_menu)
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

        self._populate_sources()
        # A ROI added or a filter applied while this dialog is open must show up
        # in the dropdowns.
        try:
            self._state.rois.changed.connect(self._populate_sources)
            self._state.filter_changed.connect(lambda *_: self._populate_sources())
        except Exception:
            pass

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
        self._plot.setLabel("bottom", name.upper())
        self._update_title()
        self._fit_result = None
        self._recompute_values(reset_bin=True)
        self._seed_default_channels()
        self._refresh_counts()

    def _selection(self):
        return parse_iteration_label(self._iter_combo.currentText())

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
        assignment, so masks stay aligned to num_loc rows."""
        vals = np.asarray(vals, dtype=float).ravel()
        if self._log_on():
            return np.where(vals > 0.0, np.log(vals), np.nan)
        return vals

    def _per_loc_raw(self):
        """Per-localization attribute values (num_loc), un-transformed."""
        ds = self._dataset()
        if ds is None:
            return None
        v = attr_values_1d(ds, self._attribute)
        return None if v is None else np.asarray(v, dtype=float).ravel()

    def _display_values(self) -> np.ndarray:
        """Transformed values driving the histogram + fit."""
        ds = self._dataset()
        if ds is None:
            return np.empty(0)
        itr_sel, _ = self._selection()
        vals = mfx_get(ds, self._attribute, itr=itr_sel, vld_only=True)
        if vals is None:
            return np.empty(0)
        vals = np.asarray(vals).ravel().astype(float)
        agg = self._agg_combo.currentText()
        if agg != "per loc":
            # tid identifies a ROW, so a value-pooling selector uses the `last`
            # rows its pooled values are laid on (a summed tid is meaningless).
            row_sel = "last" if is_value_pool_selector(itr_sel) else itr_sel
            tid = mfx_get(ds, "tid", itr=row_sel, vld_only=True)
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

    def _num_loc(self) -> int:
        ds = self._dataset()
        return int(getattr(getattr(ds, "prop", None), "num_loc", 0) or 0)

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

    def _tune_bin_spin(self, span: float) -> None:
        """Step / decimals / range from the data being plotted.

        A fixed 0.01 step is unusable on an attribute spanning 10^5 (efo) and far
        too coarse on one spanning 0.01, so the step is a power of ten two orders
        below the span — the same rule the Filter dialog's bounds spinners use.
        """
        if not np.isfinite(span) or span <= 0:
            return
        step = 10.0 ** (np.floor(np.log10(span)) - 2.0)
        decimals = int(np.clip(-np.floor(np.log10(step)) + 1, 0, 8))
        self._suspend = True
        try:
            self._bin_spin.setDecimals(decimals)
            self._bin_spin.setRange(step / 100.0, max(span * 2.0, step * 10.0))
            self._bin_spin.setSingleStep(step)
        finally:
            self._suspend = False

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
        self._tune_bin_spin(span)
        self._suspend = True
        self._bin_spin.setValue(bw)
        self._suspend = False

    def _on_basis_changed(self, *, reset_bin: bool = False) -> None:
        if self._suspend:
            return
        self._fit_result = None                # display basis changed → old fit curve is stale
        self._recompute_values(reset_bin=reset_bin)
        self._refresh_counts()

    def _on_bin_changed(self, value: float) -> None:
        if self._suspend:
            return
        self._bin_width = float(value)
        self._redraw()

    def _reset(self) -> None:
        self._suspend = True
        self._log_chk.setChecked(False)
        self._agg_combo.setCurrentText("per loc")
        if self._iter_combo.isVisible():
            self._iter_combo.setCurrentText(FLATTEN_LABEL)
        self._overlap_combo.setCurrentText(OVERLAP_KEEP_BOTH)
        self._unassigned_combo.setCurrentText(UNASSIGNED_KEEP)
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
        if render == "stacked" and self._num_itr() > 1:
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

    def refresh_colors(self) -> None:
        choices = channel_colormap_names()
        for row in self._rows:
            combo = row["lut"]
            current = combo.currentText()
            blocked = combo.blockSignals(True)
            try:
                combo.clear()
                combo.addItems(choices)
                if current in choices:
                    combo.setCurrentText(current)
            finally:
                combo.blockSignals(blocked)
        self._redraw()

    def _draw_fit_overlay(self, lo: float, hi: float, bw: float) -> None:
        """Fitted mixture overlay (per-component, colored by its channel LUT)."""
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
        """One translucent step-histogram per iteration, colored + with a legend."""
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
            r, g, b, color_alpha = _iter_color(k, self._state.prefs)
            series_alpha = int(round(alpha * color_alpha / 255.0))
            self._plot.plot(
                edges, counts, stepMode="center", fillLevel=0,
                brush=(r, g, b, series_alpha),
                pen=pg.mkPen(r, g, b, color_alpha, width=1),
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
            region = row.get("region")
            if region is None:
                continue
            self._plot.addItem(region)
            region.setBounds((plo, phi))
        try:
            self._plot.getViewBox().setXRange(plo, phi, padding=0)
        except Exception:
            pass

    def _apply_region_bounds(self) -> None:
        plo, phi = self._data_range()
        for row in self._rows:
            region = row.get("region")
            if region is not None:
                region.setBounds((plo, phi))

    # ----------------------------------------------------- preview (scatter)
    def _queue_preview(self) -> None:
        if not self._closing:
            self._preview_timer.start()

    def _refresh_preview(self) -> None:
        """A downsampled scatter of the localizations coloured by channel.

        Cheap by construction: the thumbnail is capped at
        :data:`_PREVIEW_MAX_POINTS` points *per channel share*, and it is redrawn
        on a coalescing timer, so dragging a region does not re-render the cloud
        on every mouse move.
        """
        if self._closing:
            return
        from ..core.roi_crop import display_coords

        self._preview.clear()
        ds = self._dataset()
        masks, _overlap, unassigned = self._resolve_assignment()
        if ds is None or masks is None:
            return
        xy = display_coords(ds)
        if xy.ndim != 2 or xy.shape[0] == 0:
            return
        n = min(xy.shape[0], self._num_loc() or xy.shape[0])
        budget = max(1, _PREVIEW_MAX_POINTS // max(len(masks) + 1, 1))

        def draw(mask: np.ndarray, rgb, alpha: int) -> int:
            idx = np.flatnonzero(mask[:n])
            if idx.size == 0:
                return 0
            kept = idx[::int(np.ceil(idx.size / budget))] if idx.size > budget else idx
            finite = np.all(np.isfinite(xy[kept, :2]), axis=1)
            kept = kept[finite]
            if kept.size == 0:
                return 0
            self._preview.addItem(pg.ScatterPlotItem(
                x=xy[kept, 0], y=xy[kept, 1], size=2.5, pen=None,
                brush=pg.mkBrush(*rgb, alpha)))
            return int(idx.size)

        if unassigned is not None and self._unassigned_combo.currentText() == UNASSIGNED_KEEP:
            draw(unassigned, (130, 130, 130), 90)
        for row, mask in zip(self._rows, masks):
            draw(mask, _rgb_for_lut(row["lut"].currentText()), 170)
        try:
            self._preview.getViewBox().autoRange()
        except Exception:
            pass

    # ------------------------------------------------- channel sources (row 1)
    def _roi_candidates(self) -> list:
        """Region ROIs of this dataset that are in the ROI Manager."""
        from ..core.roi_scope import roi_dataset_indices
        from ..core.roi_selection import REGION_ROI_TYPES

        out = []
        for record in getattr(self._state.rois, "records", []) or []:
            if str(getattr(record, "type", "")) not in REGION_ROI_TYPES:
                continue
            indices = roi_dataset_indices(record)
            if indices is None or self._idx in indices:
                out.append(record)
        return out

    def _filter_candidates(self) -> list[dict]:
        """This dataset's current filter rows."""
        ds = self._dataset()
        if ds is None:
            return []
        specs = ds.state.get("filter_specs") or []
        return [spec for spec in specs if isinstance(spec, dict) and spec.get("attribute")]

    @staticmethod
    def _filter_label(spec: dict) -> str:
        return (f"{spec.get('attribute', '?')} [{float(spec.get('lo', 0.0)):g}, "
                f"{float(spec.get('hi', 0.0)):g}]")

    def _populate_sources(self, *_args) -> None:
        """Fill the ROI / filter dropdowns; an empty one reads ``-``."""
        for combo, items in (
            (self._roi_combo, [str(getattr(r, "name", "") or r.id[:8]) for r in self._roi_candidates()]),
            (self._filter_combo, [self._filter_label(s) for s in self._filter_candidates()]),
        ):
            blocked = combo.blockSignals(True)
            try:
                combo.clear()
                if items:
                    combo.addItem("—")              # placeholder, index 0
                    combo.addItems(items)
                    combo.setEnabled(True)
                else:
                    combo.addItem("-")
                    combo.setEnabled(False)
                combo.setCurrentIndex(0)
            finally:
                combo.blockSignals(blocked)

    def _on_roi_source_picked(self, index: int) -> None:
        if index <= 0:
            return
        records = self._roi_candidates()
        self._roi_combo.setCurrentIndex(0)
        if not (0 <= index - 1 < len(records)):
            return
        record = records[index - 1]
        from ..core.channel_labels import roi_mask_for_record

        ds = self._dataset()
        mask = roi_mask_for_record(ds, record) if ds is not None else None
        if mask is None or not mask.any():
            self._state.log(
                f"ROI '{getattr(record, 'name', '?')}' selects no localization of "
                f"this dataset, so no channel was added.", "WARN")
            return
        self._add_mask_channel(mask, f"ROI {getattr(record, 'name', '?')}", kind="roi")

    def _on_filter_source_picked(self, index: int) -> None:
        if index <= 0:
            return
        specs = self._filter_candidates()
        self._filter_combo.setCurrentIndex(0)
        if not (0 <= index - 1 < len(specs)):
            return
        spec = specs[index - 1]
        from ..core.channel_labels import mask_from_filter_specs

        ds = self._dataset()
        if ds is None:
            return
        mask, skipped = mask_from_filter_specs(ds, [spec])
        for reason in skipped:
            self._state.log(f"Filter channel: {reason}", "WARN")
        if not mask.any():
            self._state.log("That filter row keeps no localization, so no channel "
                            "was added.", "WARN")
            return
        self._add_mask_channel(mask, f"filter {self._filter_label(spec)}", kind="filter")

    def _add_mask_channel(self, mask: np.ndarray, label: str, *, kind: str) -> None:
        luts = self._lut_cycle()
        index = len(self._rows)
        self._append_row(
            Channel(name=f"{self._base_name()} [{label}]", lo=float("nan"),
                    hi=float("nan"), lut=luts[index % len(luts)]),
            kind=kind, mask=np.asarray(mask, dtype=bool).ravel())
        self._refresh_counts()

    # ------------------------------------------------------- channel table
    def _clear_rows(self) -> None:
        for row in self._rows:
            region = row.get("region")
            if region is None:
                continue
            try:
                self._plot.removeItem(region)
            except Exception:
                pass
        self._rows.clear()
        self._table.setRowCount(0)

    def _set_channels(self, channels: list[Channel]) -> None:
        """Replace the **window** channels; mask channels (ROI/filter) survive."""
        kept = [row for row in self._rows if row["kind"] != "window"]
        kept_specs = [(row["kind"], row["mask"], row["name"].text(), row["lut"].currentText())
                      for row in kept]
        self._clear_rows()
        for ch in channels:
            self._append_row(ch)
        for kind, mask, name, lut in kept_specs:
            self._append_row(Channel(name=name, lo=float("nan"), hi=float("nan"), lut=lut),
                             kind=kind, mask=mask)
        if channels:
            self._nch_spin.blockSignals(True)
            self._nch_spin.setValue(int(np.clip(len(channels), 1, _MAX_CHANNELS)))
            self._nch_spin.blockSignals(False)
        self._refresh_counts()

    def _append_row(self, ch: Channel, *, kind: str = "window",
                    mask: np.ndarray | None = None) -> None:
        r = self._table.rowCount()
        self._table.insertRow(r)
        name_edit = QLineEdit(ch.name)
        lut_combo = QComboBox()
        lut_combo.addItems(channel_colormap_names())
        if lut_combo.findText(ch.lut) < 0:
            lut_combo.addItem(ch.lut)
        lut_combo.setCurrentText(ch.lut)
        count_item = QTableWidgetItem("0")
        count_item.setFlags(count_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        count_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._table.setCellWidget(r, 0, name_edit)
        self._table.setCellWidget(r, 3, lut_combo)
        self._table.setItem(r, 4, count_item)

        row = {"kind": kind, "mask": mask, "name": name_edit, "lut": lut_combo,
               "count": count_item, "start": None, "end": None, "region": None}

        if kind == "window":
            start_spin = self._spin(ch.lo)
            end_spin = self._spin(ch.hi)
            self._table.setCellWidget(r, 1, start_spin)
            self._table.setCellWidget(r, 2, end_spin)
            rgb = _rgb_for_lut(ch.lut)
            plo, phi = self._data_range()
            region = _ChannelRegion(
                values=(ch.lo, ch.hi), orientation=pg.LinearRegionItem.Vertical,
                movable=True, brush=(*rgb, 45), pen=pg.mkPen(rgb, width=2),
                hoverPen=pg.mkPen(rgb, width=3), bounds=(plo, phi),
                on_context=lambda _ev, item=row: self._show_region_menu(item))
            region.setZValue(10 + r)
            self._plot.addItem(region)
            row.update(start=start_spin, end=end_spin, region=region)
            start_spin.valueChanged.connect(lambda _v, item=row: self._spin_changed(item))
            end_spin.valueChanged.connect(lambda _v, item=row: self._spin_changed(item))
            region.sigRegionChanged.connect(lambda _r, item=row: self._region_changed(item))
        else:
            # A selection has no start/end on this axis; say so rather than
            # showing numbers that do not mean anything.
            for column in (1, 2):
                item = QTableWidgetItem("—")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(r, column, item)

        self._rows.append(row)
        name_edit.textChanged.connect(self._refresh_counts)
        lut_combo.currentTextChanged.connect(lambda _v, item=row: self._lut_changed(item))

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
        if self._synchronizing or row.get("region") is None:
            return
        self._synchronizing = True
        row["region"].setRegion((row["start"].value(), row["end"].value()))
        self._synchronizing = False
        self._fit_result = None                # manual edit → drop stale fit overlay
        self._refresh_counts()

    def _region_changed(self, row: dict) -> None:
        if self._synchronizing or row.get("region") is None:
            return
        lo, hi = sorted(float(v) for v in row["region"].getRegion())
        self._synchronizing = True
        row["start"].setValue(lo)
        row["end"].setValue(hi)
        self._synchronizing = False
        self._fit_result = None
        self._refresh_counts()

    def _lut_changed(self, row: dict) -> None:
        region = row.get("region")
        if region is not None:
            rgb = _rgb_for_lut(row["lut"].currentText())
            region.setBrush((*rgb, 45))
            region.setPen(pg.mkPen(rgb, width=2))
            region.setHoverPen(pg.mkPen(rgb, width=3))
        self._queue_preview()

    def _row_index(self, row: dict) -> int:
        try:
            return self._rows.index(row)
        except ValueError:
            return -1

    def _show_region_menu(self, row: dict) -> None:
        """Right-click on a channel's window: act on that channel."""
        index = self._row_index(row)
        if index < 0:
            return
        menu = QMenu(self)
        menu.addAction(f"Delete channel '{row['name'].text()}'",
                       lambda: self._delete_rows([index]))
        menu.addAction("Split channel here", lambda: self._split_row(index))
        menu.exec(self.cursor().pos())

    def _show_table_menu(self, pos) -> None:
        selected = sorted({i.row() for i in self._table.selectionModel().selectedRows()})
        clicked = self._table.indexAt(pos).row()
        if clicked >= 0 and clicked not in selected:
            selected = [clicked]
        menu = QMenu(self)
        if selected:
            menu.addAction(f"Delete {len(selected)} channel(s)",
                           lambda: self._delete_rows(selected))
        menu.addAction("Add channel", self._add_channel)
        menu.exec(self._table.viewport().mapToGlobal(pos))

    def _delete_rows(self, indices) -> None:
        indices = sorted({int(i) for i in indices if 0 <= int(i) < len(self._rows)})
        if not indices:
            return
        if len(self._rows) - len(indices) < 1:
            QMessageBox.information(self, "Channels", "At least one channel is required.")
            return
        for i in reversed(indices):
            region = self._rows[i].get("region")
            if region is not None:
                try:
                    self._plot.removeItem(region)
                except Exception:
                    pass
            self._rows.pop(i)
            self._table.removeRow(i)
        self._fit_result = None
        self._refresh_counts()

    def _split_row(self, index: int) -> None:
        """Halve a window channel, the other half becoming a new channel."""
        if not (0 <= index < len(self._rows)) or self._rows[index]["kind"] != "window":
            return
        row = self._rows[index]
        lo, hi = row["start"].value(), row["end"].value()
        mid = 0.5 * (lo + hi)
        self._synchronizing = True
        row["end"].setValue(mid)
        row["region"].setRegion((lo, mid))
        self._synchronizing = False
        luts = self._lut_cycle()
        new_i = len(self._rows)
        self._append_row(Channel(
            name=f"{self._base_name()} [{self._attribute} {new_i + 1}]",
            lo=float(mid), hi=float(hi), lut=luts[new_i % len(luts)]))
        self._fit_result = None
        self._refresh_counts()

    def _current_channels(self) -> list[Channel]:
        """Carriers for the overlay: name + LUT (encoded with transparency)."""
        out = []
        for row in self._rows:
            if row["kind"] == "window":
                lo, hi = sorted((row["start"].value(), row["end"].value()))
            else:
                lo, hi = float("nan"), float("nan")
            out.append(Channel(name=row["name"].text().strip() or "channel",
                               lo=float(lo), hi=float(hi),
                               lut=encoded_channel_lut(row["lut"].currentText())))
        return out

    # ------------------------------------------------------------- actions
    def _base_name(self) -> str:
        ds = self._dataset()
        return ds.name if ds else "channel"

    def _lut_cycle(self):
        """R, G, B, M, C, Y — solid colours, cycled."""
        return list(_CHANNEL_SOLIDS)

    def _window_span(self):
        if self._values.size:
            return float(self._values.min()), float(self._values.max())
        return 0.0, 1.0

    def _place_evenly(self) -> None:
        lo, hi = self._window_span()
        self._fit_result = None
        self._set_channels(place_evenly(lo, hi, int(self._nch_spin.value()),
                                        base_name=self._base_name(),
                                        attribute=self._attribute, luts=self._lut_cycle()))
        self._redraw()

    def _detect_peaks(self) -> None:
        """Channels from the most prominent peaks, cut at the valleys between."""
        from ..analysis.peak_channels import peak_channel_boundaries

        wanted = int(self._nch_spin.value())
        if self._values.size < 2:
            return
        bounds, found = peak_channel_boundaries(self._values, wanted)
        if found == 0:
            self._state.log(
                f"Detect peak: no peak stands out in {self._attribute}.", "WARN")
            return
        lo, hi = self._window_span()
        self._fit_result = None
        self._set_channels(channels_from_boundaries(
            bounds, lo, hi, base_name=self._base_name(),
            attribute=self._attribute, luts=self._lut_cycle()))
        if found < wanted:
            self._state.log(
                f"Detect peak: {self._attribute} supports {found} peak(s), not "
                f"{wanted} — placed {found} channel(s).", "INFO")
        self._redraw()

    def _seed_default_channels(self) -> None:
        """Place evenly (no fit) so the dialog opens with channels immediately."""
        lo, hi = self._window_span()
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
        from .qt_lifecycle import dispose_plot_widgets

        self._preview_timer.stop()
        # Retire the pyqtgraph plots last, after this window's own
        # teardown, so nothing here touches an already-inert plot.
        dispose_plot_widgets(self)
        super().closeEvent(event)

    def _run_fit(self) -> None:
        if self._closing:
            return
        dist = self._fit_combo.currentData()
        n_comp = int(self._nch_spin.value())
        if self._values.size < max(2, n_comp):
            return
        try:
            res = fit_mixture(self._values, dist, n_comp)
        except Exception as exc:
            self._state.log(f"Channel fit failed: {exc}", "WARN")
            return
        self._apply_fit_result(res)

    def _auto_fit(self) -> None:
        """Suggest the separation: best distribution + channel count by BIC."""
        if self._values.size < 2:
            return
        try:
            res = auto_fit(self._values, max_components=3)
        except Exception as exc:
            self._state.log(f"Auto fit failed: {exc}", "WARN")
            return
        self._suspend = True
        i = self._fit_combo.findData(res.distribution)
        if i >= 0:
            self._fit_combo.setCurrentIndex(i)
        self._suspend = False
        self._state.log(
            f"Auto: {res.n_components} x {DISTRIBUTION_LABELS.get(res.distribution, res.distribution)} "
            f"fits {self._attribute} best (BIC {res.bic:.0f}).", "INFO")
        self._apply_fit_result(res)

    def _apply_fit_result(self, res) -> None:
        lo, hi = (self._window_span() if self._values.size else res.domain)
        channels = channels_from_fit(res, data_range=(lo, hi), base_name=self._base_name(),
                                     attribute=self._attribute, luts=self._lut_cycle())
        self._fit_result = res
        self._fit_channel_luts = [c.lut for c in channels]
        self._set_channels(channels)
        self._redraw()

    def _add_channel(self) -> None:
        if not self._rows:
            self._place_evenly()
            return
        index = self._table.currentRow()
        windows = [i for i, row in enumerate(self._rows) if row["kind"] == "window"]
        if not windows:
            self._place_evenly()
            return
        self._split_row(index if index in windows else windows[-1])

    # --------------------------------------------------- assignment + counts
    def _channel_masks(self):
        """One boolean mask per channel row, before the overlap policy."""
        n = self._num_loc()
        if n == 0 or not self._rows:
            return None
        v, _tid = self._assign_basis()
        masks: list[np.ndarray] = []
        for row in self._rows:
            if row["kind"] == "window":
                if v is None:
                    masks.append(np.zeros(n, dtype=bool))
                    continue
                lo, hi = sorted((row["start"].value(), row["end"].value()))
                value = v[:n] if v.size >= n else np.full(n, np.nan)
                mask = np.isfinite(value) & (value >= lo) & (value <= hi)
            else:
                raw = np.asarray(row["mask"], dtype=bool).ravel()
                mask = np.zeros(n, dtype=bool)
                keep = min(n, raw.size)
                mask[:keep] = raw[:keep]
            masks.append(mask)
        return masks

    def _component_posteriors(self, values: np.ndarray):
        """Per-channel posterior of the current fit, or ``None``.

        Only meaningful while the channels still *are* the fit's components: one
        window channel per component, in order. After an edit, or with ROI/filter
        channels in the list, there is no component to weigh a row against.
        """
        res = self._fit_result
        if res is None or len(self._rows) != res.n_components:
            return None
        if any(row["kind"] != "window" for row in self._rows):
            return None
        try:
            return res.responsibilities(np.nan_to_num(values, nan=0.0))
        except Exception:
            return None

    def _resolve_assignment(self):
        """``(masks, overlap_count, unassigned_mask)`` after both policies."""
        masks = self._channel_masks()
        if masks is None:
            return None, 0, None
        n = masks[0].size
        stack = np.vstack(masks) if masks else np.zeros((0, n), dtype=bool)
        claims = stack.sum(axis=0)
        contested = claims > 1
        overlap_count = int(np.count_nonzero(contested))
        mode = self._overlap_combo.currentText()
        if overlap_count and mode != OVERLAP_KEEP_BOTH:
            if mode == OVERLAP_DISCARD:
                stack[:, contested] = False
            else:                                   # force assign by weight
                columns = np.flatnonzero(contested)
                v, _tid = self._assign_basis()
                posteriors = self._component_posteriors(
                    v[:n] if v is not None and v.size >= n else np.zeros(n))
                if posteriors is not None:
                    weights = posteriors[columns].T.copy()          # (k, m)
                    weights[~stack[:, columns]] = -np.inf           # only claiming channels
                    winners = np.argmax(weights, axis=0)
                else:
                    winners = np.argmax(stack[:, columns], axis=0)   # earlier channel wins
                stack[:, columns] = False
                stack[winners, columns] = True
            masks = [stack[k] for k in range(stack.shape[0])]
        union = stack.any(axis=0) if stack.size else np.zeros(n, dtype=bool)
        return masks, overlap_count, ~union

    def _refresh_counts(self) -> None:
        masks, overlap, unassigned = self._resolve_assignment()
        if masks is None:
            self._status_label.setText("—")
            self._apply_btn.setEnabled(False)
            return
        ds = self._dataset()
        tid = attr_values_1d(ds, "tid") if ds is not None else None
        tid = (np.arange(masks[0].size) if tid is None
               else np.asarray(tid).ravel()[:masks[0].size])

        def n_traces(mask):
            return int(np.unique(tid[mask[:tid.size]]).size) if mask.any() else 0

        assigned = 0
        for row, mask in zip(self._rows, masks):
            count = int(mask.sum())
            assigned += count
            row["count"].setText(f"{count:,} / {n_traces(mask)} tr")
        n_unassigned = int(unassigned.sum()) if unassigned is not None else 0
        self._status_label.setText(
            f"unassigned: {n_unassigned:,} locs  |  overlapped: {overlap:,} locs")
        self._apply_btn.setEnabled(assigned > 0)
        self._queue_preview()

    def _apply(self) -> None:
        masks, _overlap, _unassigned = self._resolve_assignment()
        channels = self._current_channels()
        if masks is None or not channels:
            return
        if self._owner is None or not hasattr(self._owner, "apply_channel_separation"):
            return
        keep_unassigned = self._unassigned_combo.currentText() == UNASSIGNED_KEEP
        ok = self._owner.apply_channel_separation(
            self._idx, None, channels, attribute=self._attribute,
            method_label=f"{self._attribute} channel separation",
            masks=masks, include_unassigned=keep_unassigned)
        if ok:
            self.close()
