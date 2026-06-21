"""
minflux_viewer.ui.channel_separation_window
============================================
"Separate Channel by DCR" tool (Process › Channel… › Separate Channel by DCR).

Shows the active dataset's **DCR** histogram, fits a two-component Gaussian
mixture by EM (:mod:`minflux_viewer.analysis.dcr_em`) and overlays the two
components — **red = lower-DCR label (channel 1)**, **green = higher (channel 2)**
— plus a legend with the fit parameters and goodness of fit.

The fit is computed on the displayed values (defaulting to *per-loc* / *all
[flatten]* — the maximum-statistics basis for picking the cut). Channels are
assigned at the **trace** level: each trace's mean/median DCR is compared to the
fitted mixture, so a whole trace stays in one channel. Traces below the
confidence threshold land in a third **unassigned** channel.

Apply splits the dataset into real truncated copies (via
:func:`minflux_viewer.core.roi_crop.subset_dataset`) and combines them as a
red / green / (unassigned) render overlay.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..analysis.dcr_em import (
    assign_per_trace,
    component_pdfs,
    decision_boundary,
    fit_two_component_gaussian,
    goodness_of_fit,
)
from ..core.iteration import iteration_labels, parse_iteration_label
from ..core.loader import attr_values_1d, mfx_get
from ..core.overlay import PURE_COLOR_RGB
from ..utils.filters import raw_trace_aggregate

_RED = PURE_COLOR_RGB["Red"]
_GREEN = PURE_COLOR_RGB["Green"]
_GRAY = PURE_COLOR_RGB["Gray"]
_DECISION_MODES = ["trace mean", "trace median"]
_DISPLAY_AGG = ["per loc", "trace mean", "trace median"]


class ChannelSeparationWindow(QDialog):
    """Modeless DCR two-colour separation tool for one dataset."""

    def __init__(self, state, dataset_idx: int, owner=None) -> None:
        super().__init__(None)
        self._state = state
        self._idx = dataset_idx
        self._owner = owner
        self.setWindowTitle("Separate Channel by DCR")
        self.resize(720, 620)

        self._values = np.empty(0)          # transformed DCR values driving fit + histogram
        self._fit_gmm = None                # raw EM result
        self._gmm = None                    # fit_gmm possibly with user-overridden means
        self._bin_width = 0.01
        self._suspend = False               # guard to avoid feedback loops

        self._build_ui()
        self._recompute(refit=True, reset_bin=True)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        ds = self._dataset()
        title = QLabel(f"<b>{ds.name if ds else '(no dataset)'}</b> — DCR channel separation")
        title.setWordWrap(True)
        root.addWidget(title)

        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "DCR")
        self._plot.setLabel("left", "count")
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        self._legend = pg.TextItem(anchor=(1, 0))
        self._legend.setZValue(100)
        # ignoreBounds: the legend is pinned to the view corner on every range
        # change, so it must NOT feed back into auto-range (else the view runs
        # away, zooming out without end).
        self._plot.addItem(self._legend, ignoreBounds=True)
        self._plot.getViewBox().sigRangeChanged.connect(lambda *_: self._place_legend())
        root.addWidget(self._plot, 1)

        # --- control row (mirrors the Histogram window) -------------------
        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("Iteration:"))
        self._iter_combo = QComboBox()
        self._iter_combo.currentTextChanged.connect(lambda *_: self._recompute(refit=True))
        ctl.addWidget(self._iter_combo)
        ctl.addWidget(QLabel("Values:"))
        self._agg_combo = QComboBox()
        self._agg_combo.addItems(_DISPLAY_AGG)
        self._agg_combo.currentTextChanged.connect(lambda *_: self._recompute(refit=True))
        ctl.addWidget(self._agg_combo)
        ctl.addWidget(QLabel("Bin size:"))
        self._bin_spin = QDoubleSpinBox()
        self._bin_spin.setDecimals(4)
        self._bin_spin.setRange(1e-4, 1e6)
        self._bin_spin.setValue(self._bin_width)
        self._bin_spin.valueChanged.connect(self._on_bin_changed)
        ctl.addWidget(self._bin_spin)
        self._log_chk = QCheckBox("Log(data)")
        self._log_chk.toggled.connect(lambda *_: self._recompute(refit=True, reset_bin=True))
        ctl.addWidget(self._log_chk)
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._reset)
        ctl.addWidget(reset_btn)
        ctl.addStretch(1)
        root.addLayout(ctl)

        # --- separation row ----------------------------------------------
        # --- editable Gaussian parameters (centre + width) ----------------
        gauss = QHBoxLayout()
        gauss.addWidget(QLabel("<span style='color:#dc2828'>Red</span> center:"))
        self._mu1_spin = self._make_param_spin()
        self._mu1_spin.valueChanged.connect(self._on_params_edited)
        gauss.addWidget(self._mu1_spin)
        gauss.addWidget(QLabel("σ:"))
        self._sigma1_spin = self._make_param_spin(positive=True)
        self._sigma1_spin.valueChanged.connect(self._on_params_edited)
        gauss.addWidget(self._sigma1_spin)
        gauss.addSpacing(12)
        gauss.addWidget(QLabel("<span style='color:#14aa46'>Green</span> center:"))
        self._mu2_spin = self._make_param_spin()
        self._mu2_spin.valueChanged.connect(self._on_params_edited)
        gauss.addWidget(self._mu2_spin)
        gauss.addWidget(QLabel("σ:"))
        self._sigma2_spin = self._make_param_spin(positive=True)
        self._sigma2_spin.valueChanged.connect(self._on_params_edited)
        gauss.addWidget(self._sigma2_spin)
        gauss.addStretch(1)
        root.addLayout(gauss)

        # --- separation / assignment row ----------------------------------
        sep = QHBoxLayout()
        sep.addWidget(QLabel("Decide trace by:"))
        self._decision_combo = QComboBox()
        self._decision_combo.addItems(_DECISION_MODES)
        self._decision_combo.currentTextChanged.connect(lambda *_: self._update_counts())
        sep.addWidget(self._decision_combo)
        sep.addWidget(QLabel("Min confidence %:"))
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setDecimals(0)
        self._conf_spin.setRange(50, 100)
        self._conf_spin.setValue(90)
        self._conf_spin.setToolTip(
            "Assign a trace to a channel only when it is at least this confident; "
            "below it the trace goes to the unassigned channel. 50% = keep all.")
        self._conf_spin.valueChanged.connect(lambda *_: self._update_counts())
        sep.addWidget(self._conf_spin)
        sep.addStretch(1)
        root.addLayout(sep)

        self._count_label = QLabel("")
        self._count_label.setStyleSheet("color: #1a6;")
        root.addWidget(self._count_label)

        # --- buttons ------------------------------------------------------
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
            from ..core.iteration import FLATTEN_LABEL
            self._iter_combo.setCurrentText(FLATTEN_LABEL)   # default: all [flatten]
            self._iter_combo.setVisible(True)
        else:
            self._iter_combo.setVisible(False)
        self._iter_combo.blockSignals(False)

    def _make_param_spin(self, *, positive: bool = False) -> QDoubleSpinBox:
        """3-decimal spin box for a Gaussian centre (μ) or width (σ)."""
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(0.001 if positive else -100.0, 100.0)
        spin.setSingleStep(0.001)
        return spin

    # ------------------------------------------------------------ data path
    def _dataset(self):
        if 0 <= self._idx < len(self._state.datasets):
            return self._state.datasets[self._idx]
        return None

    def _selection(self):
        return parse_iteration_label(self._iter_combo.currentText())

    def _transform(self, vals: np.ndarray) -> np.ndarray:
        """Apply Log(data) like the Histogram window (drops non-positive first)."""
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if self._log_chk.isChecked():
            vals = vals[vals > 0.0]
            vals = np.log(vals)
        return vals

    def _decision_transform(self):
        if self._log_chk.isChecked():
            return lambda v: np.log(np.where(np.asarray(v, float) > 0.0, v, np.nan))
        return None

    def _display_values(self) -> np.ndarray:
        ds = self._dataset()
        if ds is None:
            return np.empty(0)
        itr_sel, _render = self._selection()
        vals = mfx_get(ds, "dcr", itr=itr_sel, vld_only=True)
        if vals is None:
            return np.empty(0)
        vals = np.asarray(vals).ravel().astype(float)
        agg = self._agg_combo.currentText()
        if agg != "per loc":
            tid = mfx_get(ds, "tid", itr=itr_sel, vld_only=True)
            if tid is not None:
                vals = raw_trace_aggregate(vals, np.asarray(tid).ravel(), agg)
        return self._transform(vals)

    # ------------------------------------------------------------- compute
    def _recompute(self, *, refit: bool, reset_bin: bool = False) -> None:
        if self._suspend:
            return
        self._values = self._display_values()
        if reset_bin:
            self._auto_bin()
        if refit:
            self._fit()
        self._redraw()
        self._update_counts()

    def _auto_bin(self) -> None:
        v = self._values
        if v.size < 2:
            return
        lo, hi = float(v.min()), float(v.max())
        span = hi - lo
        if span <= 0:
            return
        # Freedman–Diaconis, clamped to a sane number of bins
        iqr = float(np.subtract(*np.percentile(v, [75, 25]))) * -1.0
        bw = 2.0 * abs(iqr) / (v.size ** (1.0 / 3.0)) if iqr else span / 60.0
        if not np.isfinite(bw) or bw <= 0:
            bw = span / 60.0
        bw = float(np.clip(bw, span / 250.0, span / 15.0))
        self._bin_width = bw
        self._suspend = True
        self._bin_spin.setValue(bw)
        self._suspend = False

    def _fit(self) -> None:
        if self._values.size < 2:
            self._fit_gmm = self._gmm = None
            return
        try:
            self._fit_gmm = fit_two_component_gaussian(self._values)
        except ValueError:
            self._fit_gmm = self._gmm = None
            return
        self._gmm = self._fit_gmm
        self._suspend = True
        self._mu1_spin.setValue(round(self._gmm.mu1, 3))
        self._mu2_spin.setValue(round(self._gmm.mu2, 3))
        self._sigma1_spin.setValue(round(self._gmm.sigma1, 3))
        self._sigma2_spin.setValue(round(self._gmm.sigma2, 3))
        self._suspend = False

    def _on_bin_changed(self, value: float) -> None:
        if self._suspend:
            return
        self._bin_width = float(value)
        self._redraw()

    def _on_params_edited(self) -> None:
        """Rebuild the mixture from the user-edited centres and widths."""
        if self._suspend or self._fit_gmm is None:
            return
        self._gmm = self._fit_gmm.with_params(
            self._mu1_spin.value(), self._mu2_spin.value(),
            self._sigma1_spin.value(), self._sigma2_spin.value())
        self._redraw()
        self._update_counts()

    def _reset(self) -> None:
        self._suspend = True
        self._log_chk.setChecked(False)
        self._agg_combo.setCurrentText("per loc")
        from ..core.iteration import FLATTEN_LABEL
        if self._iter_combo.isVisible():
            self._iter_combo.setCurrentText(FLATTEN_LABEL)
        self._conf_spin.setValue(90)
        self._decision_combo.setCurrentText("trace mean")
        self._suspend = False
        self._recompute(refit=True, reset_bin=True)
        # Restore the view to fit the data (undo any pan/zoom).
        try:
            self._plot.getViewBox().enableAutoRange(x=True, y=True)
            self._plot.getViewBox().autoRange()
        except Exception:
            pass

    # ------------------------------------------------------------- drawing
    def _redraw(self) -> None:
        self._plot.clear()
        self._plot.addItem(self._legend, ignoreBounds=True)
        v = self._values
        if v.size == 0:
            self._legend.setText("no DCR data")
            return
        lo, hi = float(v.min()), float(v.max())
        bw = max(self._bin_width, (hi - lo) / 1000.0 or 1e-6)
        nbins = int(np.clip(np.ceil((hi - lo) / bw), 1, 2000))
        edges = lo + bw * np.arange(nbins + 1)
        counts, _ = np.histogram(v, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        self._plot.addItem(pg.BarGraphItem(
            x=centers, height=counts, width=bw * 0.92,
            brush=(150, 150, 150, 150), pen=None))

        if self._gmm is not None:
            xs = np.linspace(lo, hi, 512)
            c0, c1 = component_pdfs(xs, self._gmm)
            scale = v.size * bw
            self._plot.addItem(pg.PlotDataItem(xs, c0 * scale, pen=pg.mkPen(_RED, width=2)))
            self._plot.addItem(pg.PlotDataItem(xs, c1 * scale, pen=pg.mkPen(_GREEN, width=2)))
            self._plot.addItem(pg.PlotDataItem(xs, (c0 + c1) * scale,
                                               pen=pg.mkPen((150, 150, 150), width=1, style=Qt.PenStyle.DashLine)))
            b = decision_boundary(self._gmm)
            # Explicit, labelled channel cutoff. ignoreBounds: a vertical
            # InfiniteLine has unbounded y-extent and would break auto-range.
            self._plot.addItem(pg.InfiniteLine(
                pos=b, angle=90,
                pen=pg.mkPen((235, 235, 235), width=1.6, style=Qt.PenStyle.DashLine),
                label=f"cutoff = {b:.3f}",
                labelOpts={"position": 0.96, "color": (235, 235, 235),
                           "fill": (0, 0, 0, 120), "movable": False}),
                ignoreBounds=True)
            self._draw_channel_side_labels(b, lo, hi, float(counts.max()) if counts.size else 1.0)
            self._set_legend_text(b)

    def _draw_channel_side_labels(self, boundary: float, lo: float, hi: float, ymax: float) -> None:
        """Annotate which side of the cutoff goes to which channel on Apply."""
        red_left = self._gmm.mu1 <= self._gmm.mu2     # red (component 0) is the lower peak?
        left = ("ch1 (red)", _RED) if red_left else ("ch2 (green)", _GREEN)
        right = ("ch2 (green)", _GREEN) if red_left else ("ch1 (red)", _RED)
        y = 0.88 * (ymax or 1.0)
        for (text, color), x, anchor in (
            (left, 0.5 * (lo + boundary), (0.5, 0.5)),
            (right, 0.5 * (boundary + hi), (0.5, 0.5)),
        ):
            ti = pg.TextItem(text, color=color, anchor=anchor)
            ti.setPos(x, y)
            self._plot.addItem(ti, ignoreBounds=True)
        self._place_legend()

    def _set_legend_text(self, boundary: float) -> None:
        g = self._gmm
        gof = goodness_of_fit(self._values, g)
        unit = "ln(DCR)" if self._log_chk.isChecked() else "DCR"
        html = (
            f"<div style='color:#bbb'><b>2-component Gaussian (EM)</b><br>"
            f"<span style='color:rgb{_RED}'>● ch1 μ={g.mu1:.3f} σ={g.sigma1:.3f} w={g.weight1:.2f}</span><br>"
            f"<span style='color:rgb{_GREEN}'>● ch2 μ={g.mu2:.3f} σ={g.sigma2:.3f} w={g.weight2:.2f}</span><br>"
            f"boundary={boundary:.3f} {unit}<br>"
            f"nRMSE={gof['nrmse']:.3f}  BIC={gof['bic']:.0f}"
            f"{'' if g.converged else '  (not converged)'}</div>"
        )
        self._legend.setHtml(html)

    def _place_legend(self) -> None:
        try:
            (x0, x1), (y0, y1) = self._plot.getViewBox().viewRange()
            self._legend.setPos(x1, y1)
        except Exception:
            pass

    # ------------------------------------------------------- separation
    def _per_loc_labels(self):
        """Channel label per materialized localization (0 red, 1 green, -1 = unassigned)."""
        ds = self._dataset()
        if ds is None or self._gmm is None:
            return None
        dcr = attr_values_1d(ds, "dcr")
        if dcr is None:
            return None
        dcr = np.asarray(dcr, dtype=float).ravel()
        tid = attr_values_1d(ds, "tid")
        tid = np.arange(dcr.size) if tid is None else np.asarray(tid).ravel()
        conf = float(self._conf_spin.value()) / 100.0
        return assign_per_trace(
            dcr, tid, self._gmm,
            mode=self._decision_combo.currentText(),
            min_confidence=conf,
            transform=self._decision_transform(),
        )

    def _update_counts(self) -> None:
        labels = self._per_loc_labels()
        if labels is None:
            self._count_label.setText("")
            self._apply_btn.setEnabled(False)
            return
        ds = self._dataset()
        tid = attr_values_1d(ds, "tid")
        tid = np.arange(labels.size) if tid is None else np.asarray(tid).ravel()

        def n_traces(mask):
            return int(np.unique(tid[mask]).size) if mask.any() else 0

        r, g, u = labels == 0, labels == 1, labels == -1
        self._count_label.setText(
            f"channel 1 (red): {n_traces(r)} traces / {int(r.sum())} locs   ·   "
            f"channel 2 (green): {n_traces(g)} traces / {int(g.sum())} locs   ·   "
            f"unassigned: {n_traces(u)} traces / {int(u.sum())} locs")
        self._apply_btn.setEnabled(bool(r.any() or g.any()))
        self._last_labels = labels

    def _apply(self) -> None:
        labels = self._per_loc_labels()
        if labels is None:
            return
        if self._owner is None or not hasattr(self._owner, "apply_dcr_channel_separation"):
            return
        ok = self._owner.apply_dcr_channel_separation(self._idx, labels)
        if ok:
            self.close()
