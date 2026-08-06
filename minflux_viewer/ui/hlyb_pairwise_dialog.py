"""
minflux_viewer.ui.hlyb_pairwise_dialog
======================================
Parameter picker and result window for **Pair-distance model fit (3D)**
(:mod:`minflux_viewer.analysis.hlyb_pairwise`).

The result window deliberately leads with the model-independent observable —
the measured pair-distance profile against an envelope-preserving null — and
presents the forward-model fit underneath it, with the sensitivity of the model
comparison stated rather than hidden.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..analysis.hlyb_pairwise import PairFitConfig, pairs_in_band
from .text_select import make_labels_selectable

_HYPOTHESIS_LABELS = {
    "dimer_gaussian": "dimer, Gaussian spread",
    "dimer_uniform": "dimer, flat band (elastic)",
    "dimer_lognormal": "dimer, log-normal spread",
    "trimer_six_site": "published six-site trimer",
    "no_structure": "no structure",
    # legacy keys, so a result from an earlier run still labels cleanly
    "six_site": "six-site HlyB complex",
    "dimer_only": "dimer distance only",
}

#: Display cap for raw localizations; deterministic thinning, display only.
_MAX_RAW_POINTS = 100_000
#: Display cap for drawn pair links, so a wide band cannot freeze the view.
_MAX_LINKS = 20_000
_VIEW_AXES = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}
_AXIS_LABELS = {"XY": ("X", "Y"), "XZ": ("X", "Z"), "YZ": ("Y", "Z")}


class HlyBPairwiseDialog(QDialog):
    """Modal parameter picker for the ensemble pair-distance analysis."""

    def __init__(self, parent=None, defaults: PairFitConfig | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("HlyB Pair-Distance Model Fit (3D)")
        d = defaults or PairFitConfig()

        root = QVBoxLayout(self)
        intro = QLabel(
            "Measure the pair-distance distribution of trace centroids without "
            "merging them, compare it with an envelope-preserving null, and fit "
            "the distribution of inter-subunit distances.\n\n"
            "Unlike template matching this never imposes a merge radius, so no "
            "distance range is removed and no artificial peak is created. The "
            "distance is fitted, not assumed: the published trimer geometry is a "
            "reference architecture that need not survive sample preparation, so "
            "it is entered as one candidate shape alongside a single distance "
            "with a Gaussian or log-normal spread and a fully elastic flat band. "
            "The shapes are ranked by AIC, so the data decide."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        form = QFormLayout()
        self._min_loc = QSpinBox()
        self._min_loc.setRange(1, 100000)
        self._min_loc.setValue(int(d.min_loc_per_trace))
        self._min_loc.setToolTip(
            "Traces with fewer localizations than this are dropped. It sets the "
            "centroid precision, which in turn sets the smallest resolvable "
            "distance.")
        form.addRow("Min loc per trace:", self._min_loc)

        self._zscale = self._dspin(0.1, 2.0, d.z_scaling_factor, 4, 0.01, "")
        self._zscale.setToolTip(
            "Factor applied to raw z before analysis (z_nm = raw_z × this). "
            "Defaults to the dataset's current RIMF.")
        form.addRow("Z scaling (RIMF):", self._zscale)

        self._r_max = self._dspin(10.0, 500.0, d.r_max_nm, 0, 5.0, " nm")
        self._r_max.setToolTip("Largest pair distance included in the profile.")
        form.addRow("Max pair distance:", self._r_max)

        self._bin = self._dspin(0.1, 5.0, d.bin_nm, 2, 0.1, " nm")
        form.addRow("Bin width:", self._bin)

        self._gap = self._dspin(0.01, 10.0, d.repeat_gap_s, 2, 0.05, " s")
        self._gap.setToolTip(
            "Consecutive traces separated by less than this in time are treated "
            "as the same molecule re-acquired, and calibrate the short-range "
            "kernel. Selection is on time only, never on distance.")
        form.addRow("Repeat time gap:", self._gap)

        self._null_reps = QSpinBox()
        self._null_reps.setRange(1, 100)
        self._null_reps.setValue(int(d.null_replicates))
        self._null_reps.setToolTip(
            "Surrogate replicates used for the null band. More replicates give "
            "a smoother reference at proportionally more compute.")
        form.addRow("Null replicates:", self._null_reps)

        self._label_spread = self._dspin(0.0, 10.0, d.label_spread_nm, 2, 0.1, " nm")
        self._label_spread.setToolTip(
            "Spread a pair distance acquires from the two antibody displacements.\n"
            "Combined in quadrature with the measured centroid precision, this "
            "sets the\npositional blur, which is NOT fitted — a free blur absorbs "
            "the very width\nthe analysis is meant to measure.\n\n"
            "It matters: on the reference dataset the fitted median distance moves "
            "from\n12.0 nm at 0 nm allowance to 8.3 nm at 5 nm. Which shape wins "
            "does not change.")
        form.addRow("Labelling allowance:", self._label_spread)

        self._dimer_lo = self._dspin(0.5, 100.0, d.dimer_distance_bounds_nm[0], 1, 0.5, " nm")
        self._dimer_hi = self._dspin(1.0, 200.0, d.dimer_distance_bounds_nm[1], 1, 0.5, " nm")
        for spin in (self._dimer_lo, self._dimer_hi):
            spin.setToolTip(
                "Range over which the inter-subunit distance is searched. It is "
                "fitted, not\nassumed: the published geometry is a reference "
                "architecture that need not\nsurvive sample preparation.")
        row = QHBoxLayout()
        row.addWidget(self._dimer_lo)
        row.addWidget(QLabel("to"))
        row.addWidget(self._dimer_hi)
        holder = QWidget()
        holder.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        form.addRow("Distance search range:", holder)

        root.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        make_labels_selectable(self)

    @staticmethod
    def _dspin(lo, hi, value, decimals, step, suffix) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(lo, hi)
        box.setDecimals(decimals)
        box.setSingleStep(step)
        box.setValue(value)
        if suffix:
            box.setSuffix(suffix)
        return box

    def config(self) -> PairFitConfig:
        base = PairFitConfig()
        return PairFitConfig(
            min_loc_per_trace=int(self._min_loc.value()),
            z_scaling_factor=float(self._zscale.value()),
            r_max_nm=float(self._r_max.value()),
            bin_nm=float(self._bin.value()),
            repeat_gap_s=float(self._gap.value()),
            null_replicates=int(self._null_reps.value()),
            label_spread_nm=float(self._label_spread.value()),
            dimer_distance_bounds_nm=(
                min(float(self._dimer_lo.value()), float(self._dimer_hi.value())),
                max(float(self._dimer_lo.value()), float(self._dimer_hi.value())),
            ),
            repeat_max_nm=base.repeat_max_nm,
            fit_r_max_nm=min(base.fit_r_max_nm, float(self._r_max.value())),
        )


class HlyBPairwiseWindow(QDialog):
    """Modeless result window: profile + null + fitted components, and a report."""

    def __init__(self, result: dict, *, title: str = "", owner=None) -> None:
        super().__init__(None)
        self._result = result
        self._owner = owner
        self.setWindowTitle(f"HlyB Pair-Distance Model Fit — {title}" if title
                            else "HlyB Pair-Distance Model Fit")
        self.resize(1040, 780)

        self.resize(1180, 900)
        self._band_lo = 8.0
        self._band_hi = 14.0
        self._scatter_pages: dict[str, dict] = {}
        self._current_view: str | None = None
        self._black_background = True
        self._band_region = None

        root = QVBoxLayout(self)
        root.addWidget(self._summary_label())

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._build_scatter())
        splitter.addWidget(self._build_plot())
        splitter.addWidget(self._build_report())
        for i in range(3):
            splitter.setCollapsible(i, False)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        root.addWidget(splitter, 1)
        self._splitter = splitter
        QTimer.singleShot(0, lambda: splitter.setSizes([400, 290, 210]))
        make_labels_selectable(self)

    # -- summary ------------------------------------------------------

    def _summary_label(self) -> QLabel:
        r = self._result
        sem = np.asarray(r.get("centroid_sem_nm", [np.nan] * 3), dtype=float)
        sem_text = " / ".join(f"{v:.2f}" for v in sem) if np.isfinite(sem).all() else "n/a"
        best = r.get("best_hypothesis", "")
        label = QLabel(
            f"Traces: {r['n_traces_total']} total, {r['n_traces_used']} used   |   "
            f"Centroid error x/y/z: {sem_text} nm   |   "
            f"Excess over null out to {r['excess_outer_nm']:.1f} nm   |   "
            f"Best model: {_HYPOTHESIS_LABELS.get(best, best or 'n/a')}"
        )
        label.setWordWrap(True)
        return label

    # -- spatial view -------------------------------------------------

    def _build_scatter(self) -> QWidget:
        """Spatial view of the measurement.

        Deliberately NOT a port of the template-matching scatter.  That view
        draws a fitted template and the pair links of an accepted complex; this
        method never fits a template to an individual complex and never assigns
        a pair to a distance class, so drawing either would display a result
        that was not computed.  What is shown instead is exactly what the
        measurement contains: the raw localizations, the trace centroids it
        operates on, and every centroid pair whose separation falls in a chosen
        band of the profile below.
        """
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)

        row = QHBoxLayout()
        row.addWidget(QLabel("View:"))
        self._view_combo = QComboBox()
        self._view_combo.addItems(["XY", "XZ", "YZ", "3D"])
        self._view_combo.setCurrentText("XY")
        self._view_combo.currentTextChanged.connect(self._show_view)
        row.addWidget(self._view_combo)

        self._raw_check = QCheckBox("raw loc")
        self._raw_check.setChecked(True)
        self._raw_check.setToolTip("Show the raw localizations.")
        self._centroid_check = QCheckBox("trace centroid")
        self._centroid_check.setChecked(True)
        self._centroid_check.setToolTip(
            "Show the trace centroids the analysis operates on.\n"
            "These are NOT sub-unit centres: several centroids may belong to\n"
            "one labelled site, which is the population the short-range kernel\n"
            "describes. Nothing is merged.")
        self._band_check = QCheckBox("pair link")
        self._band_check.setChecked(True)
        self._band_check.setToolTip(
            "Draw every centroid pair whose separation falls in the selected\n"
            "band. This asserts no assignment of a pair to a complex or to a\n"
            "distance class — the measurement makes none.")
        self._repeat_check = QCheckBox("repeat link")
        self._repeat_check.setChecked(False)
        self._repeat_check.setToolTip(
            "Draw the consecutive-in-time trace pairs that calibrate the\n"
            "short-range kernel, so the calibration set can be judged rather\n"
            "than taken on trust.")
        for box in (self._raw_check, self._centroid_check,
                    self._band_check, self._repeat_check):
            box.toggled.connect(lambda _v: self._refresh_scatter())
            row.addWidget(box)
        row.addStretch(1)
        outer.addLayout(row)

        band_row = QHBoxLayout()
        band_row.addWidget(QLabel("Link band:"))
        self._band_lo_spin = HlyBPairwiseDialog._dspin(0.0, 200.0, self._band_lo, 1, 0.5, " nm")
        self._band_hi_spin = HlyBPairwiseDialog._dspin(0.0, 200.0, self._band_hi, 1, 0.5, " nm")
        for spin in (self._band_lo_spin, self._band_hi_spin):
            spin.setToolTip(
                "Distance window whose pairs are drawn. Drag the shaded region "
                "on the profile below to change it.")
            spin.valueChanged.connect(self._on_band_spin_changed)
            band_row.addWidget(spin)
        self._band_count = QLabel("")
        band_row.addWidget(self._band_count)
        band_row.addStretch(1)
        self._bg_button = QPushButton("White background")
        self._bg_button.clicked.connect(self._toggle_background)
        band_row.addWidget(self._bg_button)
        reset = QPushButton("Reset view")
        reset.clicked.connect(self._reset_scatter_view)
        band_row.addWidget(reset)
        outer.addLayout(band_row)

        self._view_stack = QStackedWidget()
        self._view_pages: dict[str, int] = {}
        outer.addWidget(self._view_stack, 1)
        self._show_view(self._view_combo.currentText())
        return container

    # -- scatter plumbing ---------------------------------------------

    def _display_points(self) -> np.ndarray:
        pts = np.asarray(self._result.get("points_nm", np.empty((0, 3))), dtype=float)
        if pts.shape[0] > _MAX_RAW_POINTS:
            step = int(np.ceil(pts.shape[0] / _MAX_RAW_POINTS))
            pts = pts[::step]
        return pts

    def _centroids(self) -> np.ndarray:
        return np.asarray(self._result.get("centroids_nm", np.empty((0, 3))), dtype=float)

    def _band_pairs(self) -> np.ndarray:
        cent = self._centroids()
        if cent.shape[0] < 2:
            return np.empty((0, 2), dtype=np.int64)
        pairs = pairs_in_band(cent, self._band_lo, self._band_hi)
        if pairs.shape[0] > _MAX_LINKS:
            step = int(np.ceil(pairs.shape[0] / _MAX_LINKS))
            pairs = pairs[::step]
        return pairs

    def _on_band_spin_changed(self, *_args) -> None:
        lo = float(self._band_lo_spin.value())
        hi = float(self._band_hi_spin.value())
        if hi < lo:
            lo, hi = hi, lo
        self._band_lo, self._band_hi = lo, hi
        if getattr(self, "_band_region", None) is not None:
            self._band_region.blockSignals(True)
            self._band_region.setRegion((lo, hi))
            self._band_region.blockSignals(False)
        self._refresh_scatter()

    def _on_band_region_changed(self) -> None:
        lo, hi = self._band_region.getRegion()
        for spin, value in ((self._band_lo_spin, lo), (self._band_hi_spin, hi)):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)
        self._band_lo, self._band_hi = float(lo), float(hi)
        self._refresh_scatter()

    def _toggle_background(self) -> None:
        self._black_background = not self._black_background
        self._bg_button.setText("White background" if self._black_background
                                else "Black background")
        for page in self._scatter_pages.values():
            if page["kind"] == "2d":
                page["widget"].setBackground("k" if self._black_background else "w")
            else:
                page["widget"].setBackgroundColor(
                    "k" if self._black_background else "w")
        self._refresh_scatter()

    def _reset_scatter_view(self) -> None:
        page = self._scatter_pages.get(self._current_view or "")
        if not page:
            return
        if page["kind"] == "2d":
            page["widget"].getPlotItem().enableAutoRange()
        else:
            page["widget"].setCameraPosition(distance=page.get("span", 1000.0))

    def _show_view(self, view: str) -> None:
        if view not in self._view_pages:
            if view == "3D":
                try:
                    widget = self._build_gl_view()
                except Exception:
                    # OpenGL unavailable: fall back to the flat XY projection
                    self._view_combo.blockSignals(True)
                    self._view_combo.setCurrentText("XY")
                    self._view_combo.blockSignals(False)
                    return self._show_view("XY")
            else:
                widget = self._build_2d_view(view)
            self._view_pages[view] = self._view_stack.addWidget(widget)
        self._view_stack.setCurrentIndex(self._view_pages[view])
        self._current_view = view
        self._refresh_scatter()

    def _build_2d_view(self, view: str) -> QWidget:
        widget = pg.PlotWidget()
        widget.setBackground("k" if self._black_background else "w")
        item = widget.getPlotItem()
        xl, yl = _AXIS_LABELS[view]
        item.setLabel("bottom", f"{xl} (nm)")
        item.setLabel("left", f"{yl} (nm)")
        item.setAspectLocked(True)
        raw = pg.ScatterPlotItem(pxMode=True, size=2.0, pen=None)
        # PlotCurveItem, not PlotDataItem: only the curve honours an explicit
        # per-segment `connect` array, which is what draws thousands of
        # DISJOINT links in one item instead of one long polyline.
        links = pg.PlotCurveItem(pen=pg.mkPen(70, 170, 255, 190, width=1))
        repeats = pg.PlotCurveItem(pen=pg.mkPen(60, 220, 120, 190, width=1))
        cent = pg.ScatterPlotItem(pxMode=True, size=5.0, pen=None)
        for entry in (raw, links, repeats, cent):
            item.addItem(entry)
        self._scatter_pages[view] = {
            "kind": "2d", "widget": widget, "axes": _VIEW_AXES[view],
            "raw": raw, "links": links, "repeats": repeats, "cent": cent,
        }
        return widget

    def _build_gl_view(self) -> QWidget:
        import pyqtgraph.opengl as gl

        view = gl.GLViewWidget()
        view.setBackgroundColor("k" if self._black_background else "w")
        anchor = self._centroids()
        anchor = anchor.mean(axis=0) if anchor.shape[0] else np.zeros(3)
        raw = gl.GLScatterPlotItem(pxMode=True, size=2.0)
        links = gl.GLLinePlotItem(mode="lines", width=1.0, antialias=False)
        repeats = gl.GLLinePlotItem(mode="lines", width=1.0, antialias=False)
        cent = gl.GLScatterPlotItem(pxMode=True, size=6.0)
        for entry in (raw, links, repeats, cent):
            view.addItem(entry)
        ref = self._centroids()
        span = float(np.linalg.norm(np.ptp(ref, axis=0))) if ref.shape[0] else 1000.0
        view.setCameraPosition(distance=max(span * 0.6, 100.0))
        self._scatter_pages["3D"] = {
            "kind": "3d", "widget": view, "gl": gl, "anchor": anchor,
            "raw": raw, "links": links, "repeats": repeats, "cent": cent,
            "span": max(span * 0.6, 100.0),
        }
        return view

    @staticmethod
    def _segments(points: np.ndarray, pairs: np.ndarray) -> np.ndarray:
        """Interleave pair endpoints into a flat 'lines' vertex array."""
        if pairs.shape[0] == 0:
            return np.empty((0, points.shape[1]))
        out = np.empty((pairs.shape[0] * 2, points.shape[1]), dtype=float)
        out[0::2] = points[pairs[:, 0]]
        out[1::2] = points[pairs[:, 1]]
        return out

    def _refresh_scatter(self) -> None:
        view = self._current_view
        page = self._scatter_pages.get(view or "")
        if not page:
            return
        cent = self._centroids()
        pairs = self._band_pairs() if self._band_check.isChecked() else \
            np.empty((0, 2), dtype=np.int64)
        repeats = np.asarray(self._result.get("repeat_pairs", np.empty((0, 2))),
                             dtype=np.int64) if self._repeat_check.isChecked() else \
            np.empty((0, 2), dtype=np.int64)
        total_band = pairs_in_band(cent, self._band_lo, self._band_hi).shape[0] \
            if cent.shape[0] > 1 else 0
        shown = pairs.shape[0]
        note = "" if shown == total_band else f" (showing {shown:,})"
        self._band_count.setText(
            f"{total_band:,} pair(s) in {self._band_lo:.1f}–{self._band_hi:.1f} nm{note}")

        raw_col = (200, 200, 200, 90) if self._black_background else (60, 60, 60, 70)
        cent_col = (255, 205, 60, 220) if self._black_background else (190, 120, 0, 230)
        link_col = (70, 170, 255, 190)
        rep_col = (60, 220, 120, 190)

        if page["kind"] == "2d":
            ax, ay = page["axes"]
            pts = self._display_points() if self._raw_check.isChecked() else np.empty((0, 3))
            page["raw"].setData(
                x=pts[:, ax] if pts.shape[0] else [],
                y=pts[:, ay] if pts.shape[0] else [],
                brush=pg.mkBrush(*raw_col), pen=None, size=2.0)
            page["cent"].setData(
                x=cent[:, ax] if (cent.shape[0] and self._centroid_check.isChecked()) else [],
                y=cent[:, ay] if (cent.shape[0] and self._centroid_check.isChecked()) else [],
                brush=pg.mkBrush(*cent_col), pen=None, size=5.0)
            for key, idx, colour in (("links", pairs, link_col),
                                     ("repeats", repeats, rep_col)):
                if idx.shape[0]:
                    seg = self._segments(cent[:, [ax, ay]], idx)
                    # connect[i] == 1 joins vertex i to i+1, so alternating 1/0
                    # draws each pair as its own segment
                    connect = np.tile(np.array([1, 0], dtype=np.uint8), idx.shape[0])
                    page[key].setData(x=seg[:, 0], y=seg[:, 1], connect=connect,
                                      pen=pg.mkPen(*colour, width=1))
                else:
                    page[key].setData(x=np.empty(0), y=np.empty(0))
        else:
            anchor = page["anchor"]
            pts = self._display_points() if self._raw_check.isChecked() else np.empty((0, 3))
            page["raw"].setData(pos=pts - anchor if pts.shape[0] else np.empty((0, 3)),
                                color=tuple(c / 255 for c in raw_col), size=2.0)
            show_cent = cent.shape[0] and self._centroid_check.isChecked()
            page["cent"].setData(
                pos=(cent - anchor) if show_cent else np.empty((0, 3)),
                color=tuple(c / 255 for c in cent_col), size=6.0)
            for key, idx, colour in (("links", pairs, link_col),
                                     ("repeats", repeats, rep_col)):
                seg = self._segments(cent - anchor, idx) if idx.shape[0] else np.empty((0, 3))
                page[key].setData(pos=seg, color=tuple(c / 255 for c in colour))

    # -- plot ---------------------------------------------------------

    def _build_plot(self) -> QWidget:
        r = self._result
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        controls = QHBoxLayout()
        self._show_components = QCheckBox("show fitted components")
        self._show_components.setChecked(True)
        self._show_components.toggled.connect(lambda _v: self._redraw())
        controls.addWidget(self._show_components)
        self._show_null = QCheckBox("show null band")
        self._show_null.setChecked(True)
        self._show_null.toggled.connect(lambda _v: self._redraw())
        controls.addWidget(self._show_null)
        self._show_excess = QCheckBox("plot excess over null")
        self._show_excess.setChecked(False)
        self._show_excess.setToolTip(
            "Subtract the null instead of overlaying it. This is the "
            "model-independent view of the measurement.")
        self._show_excess.toggled.connect(lambda _v: self._redraw())
        controls.addWidget(self._show_excess)
        controls.addStretch()
        reset = QPushButton("Reset view")
        reset.clicked.connect(lambda: self._plot.enableAutoRange())
        controls.addWidget(reset)
        layout.addLayout(controls)

        widget = pg.PlotWidget()
        self._plot = widget.getPlotItem()
        self._plot.setLabel("bottom", "pair distance (nm)")
        self._plot.setLabel("left", f"pairs per {float(np.diff(r['edges_nm']).mean()):.2g} nm bin")
        self._plot.addLegend(offset=(-10, 10))
        layout.addWidget(widget, 1)

        # Dragging this region selects which pairs the spatial view draws, so
        # a feature of the distribution can be located in the cell directly.
        self._band_region = pg.LinearRegionItem(
            values=(self._band_lo, self._band_hi),
            brush=pg.mkBrush(70, 170, 255, 45),
            pen=pg.mkPen(70, 170, 255, 160))
        self._band_region.setZValue(-10)
        self._band_region.sigRegionChanged.connect(self._on_band_region_changed)
        self._plot.addItem(self._band_region, ignoreBounds=True)
        self._redraw()
        return panel

    def _redraw(self) -> None:
        r = self._result
        plot = self._plot
        plot.clear()
        if plot.legend is not None:
            plot.legend.clear()
        c = np.asarray(r["centres_nm"], dtype=float)
        fit = r.get("best_fit") or {}
        excess_mode = bool(self._show_excess.isChecked())
        base = np.asarray(r["null_mean"], dtype=float) if excess_mode else np.zeros_like(c)

        if self._show_null.isChecked() and not excess_mode:
            lo = np.asarray(r["null_mean"]) - 2 * np.asarray(r["null_sd"])
            hi = np.asarray(r["null_mean"]) + 2 * np.asarray(r["null_sd"])
            band = pg.FillBetweenItem(
                pg.PlotDataItem(c, lo), pg.PlotDataItem(c, hi),
                brush=pg.mkBrush(150, 150, 150, 110))
            plot.addItem(band, ignoreBounds=True)
            plot.plot(c, np.asarray(r["null_mean"]), pen=pg.mkPen(120, 120, 120, 200),
                      name="null (envelope-preserving)")
        elif excess_mode:
            sd = 2 * np.asarray(r["null_sd"], dtype=float)
            band = pg.FillBetweenItem(
                pg.PlotDataItem(c, -sd), pg.PlotDataItem(c, sd),
                brush=pg.mkBrush(150, 150, 150, 110))
            plot.addItem(band, ignoreBounds=True)

        plot.plot(c, np.asarray(r["counts"], dtype=float) - base,
                  pen=pg.mkPen(20, 20, 20, 220, width=2), name="observed")

        if fit:
            model = np.asarray(fit["model"], dtype=float)
            plot.plot(c, model - base, pen=pg.mkPen(210, 40, 40, 235, width=2),
                      name="fit (total)")
            if self._show_components.isChecked():
                plot.plot(c, np.asarray(fit["repeat_component"]),
                          pen=pg.mkPen(30, 150, 60, 220, width=2,
                                       style=Qt.PenStyle.DashLine),
                          name="same-site short range")
                plot.plot(c, np.asarray(fit.get("structure_component",
                                                fit.get("complex_component"))),
                          pen=pg.mkPen(50, 90, 220, 230, width=2,
                                       style=Qt.PenStyle.DashLine),
                          name="inter-subunit distances")
                if not excess_mode:
                    plot.plot(c, np.asarray(fit["background_component"]),
                              pen=pg.mkPen(230, 150, 30, 220, width=2,
                                           style=Qt.PenStyle.DotLine),
                              name="unrelated pairs")
                # Mark the FITTED distance distribution, not the tabulated
                # classes: the geometry is an outcome here, not an input.
                summary = fit.get("distance_summary") or {}
                if summary:
                    plot.addItem(pg.InfiniteLine(
                        pos=float(summary.get("median_nm", 0.0)), angle=90,
                        pen=pg.mkPen(50, 90, 220, 200, width=2)), ignoreBounds=True)
                    band = pg.LinearRegionItem(
                        values=(float(summary.get("p16_nm", 0.0)),
                                float(summary.get("p84_nm", 0.0))),
                        brush=pg.mkBrush(50, 90, 220, 28), movable=False)
                    band.setZValue(-20)
                    plot.addItem(band, ignoreBounds=True)
        # clear() drops every item, so the band selector has to be put back
        if self._band_region is not None:
            plot.addItem(self._band_region, ignoreBounds=True)
        plot.enableAutoRange()

    # -- report -------------------------------------------------------

    def _build_report(self) -> QWidget:
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(self.report_text())
        font = text.font()
        font.setFamily("monospace")
        text.setFont(font)
        return text

    def report_text(self) -> str:
        return pairwise_report(self._result)


def pairwise_report(result: dict) -> str:
    """Plain-text account of an ensemble pair-distance analysis."""
    r = result
    lines: list[str] = []
    add = lines.append

    sem = np.asarray(r.get("centroid_sem_nm", [np.nan] * 3), dtype=float)
    add("MEASUREMENT")
    add(f"  traces total / used            : {r['n_traces_total']} / {r['n_traces_used']}")
    if np.isfinite(sem).all():
        add(f"  centroid error x / y / z (nm)  : "
            f"{sem[0]:.2f} / {sem[1]:.2f} / {sem[2]:.2f}")
    add(f"  pair-blur floor (nm)           : {r.get('sigma_floor_nm', float('nan')):.2f}"
        "   (sqrt(2) x centroid error; no fitted blur may go below it)")
    add(f"  null replicates                : {r.get('null_replicates', 0)}")
    add(f"  excess above null (z>3) out to : {r['excess_outer_nm']:.1f} nm")
    add("")

    k = r.get("repeat_kernel", {})
    add("SAME-SITE SHORT-RANGE KERNEL")
    add(f"  source                         : {k.get('source', 'n/a')}")
    add(f"  calibration pairs              : {k.get('n_pairs', 0)}")
    add(f"  median separation (nm)         : {k.get('median_nm', float('nan')):.2f}")
    far = k.get("rejected_far_fraction", float("nan"))
    if np.isfinite(far):
        add(f"  time-selected pairs discarded  : {far:.1%} (moved to a new molecule)")
    add("  This population lumps together a molecule re-acquired as several")
    add("  traces, the two fluorophores carried by one FluoTag-X2, and drift")
    add("  between re-acquisitions. It is modelled, not deleted, which is what")
    add("  removes the merge-radius artifact of template matching.")
    add("")

    add("DIMER DISTANCE  (fitted distribution of true inter-subunit distances)")
    best_fit = r.get("best_fit") or {}
    summary = best_fit.get("distance_summary") or {}
    if summary:
        add(f"  preferred shape                : "
            f"{_HYPOTHESIS_LABELS.get(r.get('best_hypothesis', ''), '?')}")
        add(f"  median true distance           : {summary.get('median_nm', float('nan')):.2f} nm")
        add(f"  central 68% of the population  : "
            f"{summary.get('p16_nm', float('nan')):.2f} to "
            f"{summary.get('p84_nm', float('nan')):.2f} nm "
            f"(half-width {summary.get('spread_nm', float('nan')):.2f} nm)")
        add(f"  mode / mean                    : {summary.get('mode_nm', float('nan')):.2f} / "
            f"{summary.get('mean_nm', float('nan')):.2f} nm")
        add(f"  fitted shape parameters        : {best_fit.get('structure_description', '')}")
        ref = r.get("reference_dimer_nm")
        if ref:
            add(f"  reference diagram value        : {ref:.2f} nm "
                f"(a reference architecture, NOT a constraint here)")
    scan = r.get("distance_scan") or {}
    if scan.get("available"):
        lo, hi = scan.get("ci68_nm", (float('nan'),) * 2)
        lo95, hi95 = scan.get("ci95_nm", (float('nan'),) * 2)
        add(f"  likelihood scan of the centre  : best {scan.get('best_nm', float('nan')):.2f} nm, "
            f"68% [{lo:.2f}, {hi:.2f}], 95% [{lo95:.2f}, {hi95:.2f}]")
        if not scan.get("constrained", True):
            add("    The scan never rises by 1.92, so the data do NOT localize the")
            add("    centre: only the width of the population is being measured.")
        elif scan.get("ci68_below_scan_step"):
            add(f"    The 68% interval is not wider than the {scan.get('step_nm', 0):.2f} nm scan")
            add("    step, so it is unresolved rather than tight.")
    add("  The distribution is summarised by percentiles rather than by raw shape")
    add("  parameters, so the shapes are comparable and a broad population is not")
    add("  misread as a precise distance.")
    add("")

    add("SHAPE COMPARISON  (short-range kernel pinned at its measured width)")
    add(f"  {'hypothesis':<28} {'dAIC':>9} {'pairs':>9} {'median':>8} {'68% range':>16}")
    for name, fit in sorted(r.get("fits", {}).items(),
                            key=lambda kv: kv[1].get("delta_aic", 0.0)):
        s = fit.get("distance_summary") or {}
        rng = (f"{s['p16_nm']:.1f}-{s['p84_nm']:.1f} nm" if s else "-")
        med = f"{s['median_nm']:.2f}" if s else "-"
        add(f"  {_HYPOTHESIS_LABELS.get(name, name):<28} {fit.get('delta_aic', 0.0):9.1f} "
            f"{fit.get('n_structure_pairs', fit.get('n_complex_pairs', 0.0)):9.0f} "
            f"{med:>8} {rng:>16}")
    add("  Lower dAIC is better; 0 marks the preferred shape. The published trimer")
    add("  geometry is one candidate here, not the model: it may not survive sample")
    add("  preparation, and imposing it would assume the answer. The positional")
    add("  blur is fitted once and shared, so no shape can win by claiming extra")
    add("  blur that another is denied.")
    add("")

    relaxed = r.get("fits_relaxed_kernel", {})
    if relaxed:
        add("SENSITIVITY  (short-range kernel allowed to broaden)")
        add(f"  {'hypothesis':<26} {'dAIC':>10} {'kernel stretch':>16}")
        for name, fit in relaxed.items():
            add(f"  {_HYPOTHESIS_LABELS.get(name, name):<26} {fit.get('delta_aic', 0.0):10.1f} "
                f"{fit.get('repeat_scale', 1.0):16.2f}")
        add("  Releasing the kernel width describes the 5-9 nm region better but")
        add("  weakens the structural discrimination. Both are reported because")
        add("  the strength of the evidence depends on that choice.")
        add("")

    best = r.get("best_fit") or {}
    bounds = best.get("parameters_at_bounds") or []
    fits = r.get("fits", {})
    add("INTERPRETATION")
    trimer = fits.get("trimer_six_site")
    if trimer is not None and r.get("best_hypothesis", "").startswith("dimer"):
        add(f"  A single broad inter-subunit distance describes the data better than")
        add(f"  the published six-site trimer by {trimer.get('delta_aic', 0.0):.0f} AIC units.")
        add("  That is consistent with the trimer not surviving sample preparation,")
        add("  leaving dimers whose separation is neither fixed at the tabulated")
        add("  value nor sharp.")
    elif r.get("best_hypothesis") == "trimer_six_site":
        add("  The published six-site trimer geometry describes the data better than")
        add("  any of the single-distance shapes tested.")
    if summary:
        add(f"  The population spans {summary.get('p16_nm', float('nan')):.1f}-"
            f"{summary.get('p84_nm', float('nan')):.1f} nm at 68%, which is broad "
            f"relative to the")
        add(f"  {best.get('sigma_nm', float('nan')):.2f} nm measurement blur, so the width is "
            f"a property of the sample")
        add("  rather than of the microscope: the linkage is flexible, or several")
        add("  conformations are present. A flat band is a legitimate outcome and is")
        add("  offered as an explicit hypothesis.")
    if bounds:
        add(f"  Parameters resting on a bound (limits, not estimates): "
            f"{', '.join(bounds)}.")
    add("  No individual distance class is resolved and none is claimed. What is")
    add("  measured is the distribution of inter-subunit distances, its width, and")
    add("  which shape reproduces it.")
    add("  The spatial view therefore shows no fitted template and no per-complex")
    add("  assignment: this method fits the ensemble, not individual complexes,")
    add("  so drawing either would display a result that was not computed. Its")
    add("  links are simply the centroid pairs falling in the selected band.")
    return "\n".join(lines)
