"""UI for the staged model-independent HlyB 3-D short-range workflow."""

from __future__ import annotations

import colorsys

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..analysis.hlyb_staged import Staged3DConfig
from .text_select import make_labels_selectable

_VIEW_AXES = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}
_AXIS_LABELS = {"XY": ("X", "Y"), "XZ": ("X", "Z"), "YZ": ("Y", "Z")}
_MAX_RAW_POINTS = 100_000


class HlyBStagedDialog(QDialog):
    """Parameter picker for the staged 3-D population analysis."""

    def __init__(self, parent=None, defaults: Staged3DConfig | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("HlyB Staged Short-Range Population Analysis (3D)")
        d = defaults or Staged3DConfig()
        root = QVBoxLayout(self)
        intro = QLabel(
            "Test for a population-level excess of inferred label-site pairs without "
            "an HlyB template or a fitted molecular distance. Traces are conservatively "
            "consolidated below the dimer scale, spatial components are analyzed "
            "independently, and the observed profile is compared with a conditional "
            "rod-surface randomization that preserves exact site count, axial density "
            "and observed membrane support.\n\n"
            "The result describes evidence for a short-range population. Its excess "
            "centroid is a distribution descriptor, not an HlyB dimer-distance estimate."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        form = QFormLayout()
        self._min_loc = self._ispin(1, 100000, d.min_loc_per_trace)
        self._min_loc.setToolTip("Traces below this localization count are excluded.")
        form.addRow("Min loc per trace:", self._min_loc)

        self._zscale = self._dspin(0.1, 2.0, d.z_scaling_factor, 4, 0.01, "")
        self._zscale.setToolTip(
            "Factor applied once to raw z. The project default is the fixed RIMF 0.67.")
        form.addRow("Z scaling (RIMF):", self._zscale)

        self._merge = self._dspin(1.0, 8.0, d.site_merge_nm, 1, 0.5, " nm")
        self._merge.setToolTip(
            "Hard maximum diameter for consolidating repeated trace centroids into one "
            "label-site estimate. Complete-link constraints prevent chaining. The "
            "3/4/5 nm sensitivity audit is enabled below.")
        form.addRow("Same-site max diameter:", self._merge)

        self._cell_link = self._dspin(20.0, 1000.0, d.cell_link_nm, 0, 10.0, " nm")
        self._cell_link.setToolTip(
            "Coarse neighbor link used only to separate spatial/cell components. "
            "Pairs are never formed between components.")
        form.addRow("Spatial-component link:", self._cell_link)

        self._min_sites = self._ispin(3, 10000, d.min_sites_per_component)
        form.addRow("Min sites per component:", self._min_sites)

        self._r_max = self._dspin(20.0, 500.0, d.r_max_nm, 0, 5.0, " nm")
        form.addRow("Max displayed distance:", self._r_max)
        self._bin = self._dspin(0.1, 5.0, d.bin_nm, 2, 0.1, " nm")
        form.addRow("Bin width:", self._bin)

        band = QHBoxLayout()
        self._short_lo = self._dspin(0.0, 100.0, d.short_range_lo_nm, 1, 0.5, " nm")
        self._short_hi = self._dspin(1.0, 200.0, d.short_range_hi_nm, 1, 0.5, " nm")
        self._short_lo.setToolTip(
            "Primary test lower bound. The default 8 nm is twice the default same-site "
            "diameter, separating the test from unresolved repeat-site splitting.")
        band.addWidget(self._short_lo)
        band.addWidget(QLabel("to"))
        band.addWidget(self._short_hi)
        holder = QWidget()
        holder.setLayout(band)
        band.setContentsMargins(0, 0, 0, 0)
        form.addRow("Pre-declared short range:", holder)

        self._stratum = self._ispin(4, 10000, d.null_stratum_sites)
        self._stratum.setToolTip(
            "Number of axial-neighbor sites per local permutation stratum. The "
            "32/64/128-site sensitivity audit reports dependence on this scale.")
        form.addRow("Null axial stratum:", self._stratum)
        self._null_reps = self._ispin(9, 9999, d.null_replicates)
        self._null_reps.setToolTip(
            "Conditional-randomization replicates. 99 gives a minimum empirical "
            "one-sided p-value of 0.01.")
        form.addRow("Null replicates:", self._null_reps)

        self._sensitivity = QCheckBox(
            "run site-radius, surface-null, and ±25% component-link audit")
        self._sensitivity.setChecked(bool(d.run_sensitivity))
        form.addRow("Sensitivity:", self._sensitivity)
        root.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        make_labels_selectable(self)

    @staticmethod
    def _dspin(lo, hi, value, decimals, step, suffix) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(lo, hi)
        box.setDecimals(decimals)
        box.setSingleStep(step)
        box.setValue(float(value))
        box.setSuffix(suffix)
        return box

    @staticmethod
    def _ispin(lo, hi, value) -> QSpinBox:
        box = QSpinBox()
        box.setRange(int(lo), int(hi))
        box.setValue(int(value))
        return box

    def _accept_if_valid(self) -> None:
        if self._short_hi.value() > self._r_max.value():
            self._r_max.setValue(self._short_hi.value())
        if self._short_lo.value() >= self._short_hi.value():
            self._short_lo.setValue(max(0.0, self._short_hi.value() - 1.0))
        self.accept()

    def config(self) -> Staged3DConfig:
        base = Staged3DConfig()
        return Staged3DConfig(
            min_loc_per_trace=int(self._min_loc.value()),
            z_scaling_factor=float(self._zscale.value()),
            site_merge_nm=float(self._merge.value()),
            site_sigma_factor=base.site_sigma_factor,
            site_precision_floor_nm=base.site_precision_floor_nm,
            cell_link_nm=float(self._cell_link.value()),
            min_sites_per_component=int(self._min_sites.value()),
            r_max_nm=float(self._r_max.value()),
            bin_nm=float(self._bin.value()),
            short_range_lo_nm=float(self._short_lo.value()),
            short_range_hi_nm=float(self._short_hi.value()),
            null_stratum_sites=int(self._stratum.value()),
            null_replicates=int(self._null_reps.value()),
            rng_seed=base.rng_seed,
            bootstrap_replicates=base.bootstrap_replicates,
            run_sensitivity=bool(self._sensitivity.isChecked()),
            sensitivity_replicates=base.sensitivity_replicates,
            sensitivity_site_merge_nm=base.sensitivity_site_merge_nm,
            sensitivity_stratum_sites=base.sensitivity_stratum_sites,
            sensitivity_cell_link_factors=base.sensitivity_cell_link_factors,
        )


class HlyBStagedWindow(QDialog):
    """Modeless result window for the staged analysis."""

    def __init__(self, result: dict, *, title: str = "", owner=None,
                 prefs: dict | None = None) -> None:
        super().__init__(None)
        self._result = result
        self._owner = owner
        self._prefs = prefs or {}
        self.setWindowTitle(
            f"HlyB Staged Short-Range Population (3D) — {title}" if title
            else "HlyB Staged Short-Range Population (3D)")
        self.resize(1150, 900)
        root = QVBoxLayout(self)
        root.addWidget(self._summary_label())

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._build_spatial_view())
        splitter.addWidget(self._build_profile_view())
        splitter.addWidget(self._build_report())
        for i in range(3):
            splitter.setCollapsible(i, False)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 2)
        root.addWidget(splitter, 1)
        splitter.setSizes([280, 390, 220])
        make_labels_selectable(self)

    @staticmethod
    def _p_text(value: float, replicates: int) -> str:
        floor = 1.0 / (max(int(replicates), 0) + 1)
        if np.isfinite(value) and value <= floor + 1e-12:
            return f"≤ {floor:.3g}"
        return f"{value:.3g}" if np.isfinite(value) else "n/a"

    def _summary_label(self) -> QLabel:
        r = self._result
        s = r["summary"]
        reps = int(np.asarray(r.get("null_profiles", [])).shape[0])
        robust = r.get("robust_short_range_excess")
        robustness = ("passes all sensitivity variants" if robust is True else
                      "does NOT pass every sensitivity variant" if robust is False else
                      "sensitivity audit not run")
        label = QLabel(
            f"Traces: {r['n_traces_used']:,}/{r['n_traces_total']:,} used  |  "
            f"Inferred sites: {r['n_sites']:,} ({r['n_sites_used']:,} in "
            f"{r['n_components']} component(s))  |  "
            f"{r['config'].short_range_lo_nm:g}–{r['config'].short_range_hi_nm:g} nm "
            f"excess ratio: {s['band_ratio']:.2f}  |  empirical p {self._p_text(s['band_p'], reps)}  |  "
            f"positive-excess centroid: {s['positive_excess_centroid_nm']:.2f} nm  |  "
            f"{robustness}"
        )
        label.setWordWrap(True)
        label.setToolTip(
            "The excess centroid describes the positive observed-minus-null population. "
            "It is not a fitted or assigned HlyB dimer distance.")
        return label

    def _build_spatial_view(self) -> QWidget:
        holder = QWidget()
        root = QVBoxLayout(holder)
        root.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Projection:"))
        self._view_combo = QComboBox()
        self._view_combo.addItems(["XY", "XZ", "YZ"])
        self._view_combo.currentTextChanged.connect(self._refresh_spatial)
        row.addWidget(self._view_combo)
        self._raw_check = QCheckBox("raw loc")
        self._raw_check.setChecked(False)
        self._trace_check = QCheckBox("trace centroid")
        self._trace_check.setChecked(True)
        self._site_check = QCheckBox("inferred label site")
        self._site_check.setChecked(True)
        self._null_check = QCheckBox("one surface-null draw")
        self._null_check.setChecked(False)
        for box in (self._raw_check, self._trace_check, self._site_check, self._null_check):
            box.toggled.connect(self._refresh_spatial)
            row.addWidget(box)
        row.addStretch(1)
        root.addLayout(row)
        self._spatial_plot = pg.PlotWidget(background="k")
        self._spatial_plot.showGrid(x=True, y=True, alpha=0.15)
        self._spatial_plot.getViewBox().setAspectLocked(True)
        root.addWidget(self._spatial_plot, 1)
        self._refresh_spatial()
        return holder

    @staticmethod
    def _component_brush(component: int):
        if component < 0:
            return pg.mkBrush(120, 120, 120, 180)
        hue = (0.61803398875 * component) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
        return pg.mkBrush(*(int(255 * value) for value in rgb), 230)

    def _refresh_spatial(self, *_args) -> None:
        if not hasattr(self, "_spatial_plot"):
            return
        plot = self._spatial_plot
        plot.clear()
        view = self._view_combo.currentText() if hasattr(self, "_view_combo") else "XY"
        a, b = _VIEW_AXES[view]
        xlab, ylab = _AXIS_LABELS[view]
        plot.setLabel("bottom", xlab, units="nm")
        plot.setLabel("left", ylab, units="nm")

        if self._raw_check.isChecked():
            points = np.asarray(self._result.get("points_nm", np.empty((0, 3))), dtype=float)
            if points.shape[0] > _MAX_RAW_POINTS:
                points = points[::int(np.ceil(points.shape[0] / _MAX_RAW_POINTS))]
            plot.addItem(pg.ScatterPlotItem(
                x=points[:, a], y=points[:, b], size=2,
                pen=None, brush=pg.mkBrush(90, 90, 90, 90), pxMode=True))
        if self._trace_check.isChecked():
            traces = np.asarray(self._result.get("trace_centroids_nm", np.empty((0, 3))))
            plot.addItem(pg.ScatterPlotItem(
                x=traces[:, a], y=traces[:, b], size=3,
                pen=None, brush=pg.mkBrush(255, 150, 40, 120), pxMode=True))
        if self._null_check.isChecked():
            null = np.asarray(self._result.get("null_preview_sites_nm", np.empty((0, 3))))
            plot.addItem(pg.ScatterPlotItem(
                x=null[:, a], y=null[:, b], size=5,
                pen=pg.mkPen(120, 190, 255, 180), brush=None, pxMode=True))
        if self._site_check.isChecked():
            sites = np.asarray(self._result.get("site_centers_nm", np.empty((0, 3))))
            labels = np.asarray(self._result.get("component_labels", np.full(sites.shape[0], -1)))
            spots = [{"pos": (float(p[a]), float(p[b])), "brush": self._component_brush(int(c))}
                     for p, c in zip(sites, labels)]
            plot.addItem(pg.ScatterPlotItem(
                spots=spots, size=7, pen=pg.mkPen(230, 230, 230, 90), pxMode=True))
        plot.enableAutoRange()

    def _build_profile_view(self) -> QWidget:
        holder = QWidget()
        root = QHBoxLayout(holder)
        root.setContentsMargins(0, 0, 0, 0)
        centers = np.asarray(self._result["centers_nm"], dtype=float)
        observed = np.asarray(self._result["observed"], dtype=float)
        mean = np.asarray(self._result["null_mean"], dtype=float)
        lo = np.asarray(self._result["null_lo"], dtype=float)
        hi = np.asarray(self._result["null_hi"], dtype=float)
        cfg = self._result["config"]

        profile = pg.PlotWidget(background="w")
        profile.setLabel("bottom", "Site-pair distance", units="nm")
        profile.setLabel("left", "Pair count per bin")
        profile.showGrid(x=True, y=True, alpha=0.18)
        profile.addLegend()
        upper = pg.PlotCurveItem(centers, hi, pen=pg.mkPen(120, 170, 230, 80))
        lower = pg.PlotCurveItem(centers, lo, pen=pg.mkPen(120, 170, 230, 80))
        profile.addItem(upper)
        profile.addItem(lower)
        profile.addItem(pg.FillBetweenItem(
            upper, lower, brush=pg.mkBrush(120, 170, 230, 55)))
        profile.plot(centers, mean, pen=pg.mkPen(45, 105, 190, width=2), name="surface null")
        profile.plot(centers, observed, pen=pg.mkPen(30, 30, 30, width=2), name="observed")
        region = pg.LinearRegionItem(
            values=(cfg.short_range_lo_nm, cfg.short_range_hi_nm), movable=False,
            brush=pg.mkBrush(255, 180, 50, 35), pen=pg.mkPen(220, 130, 20, 100))
        profile.addItem(region)
        root.addWidget(profile, 1)

        excess = pg.PlotWidget(background="w")
        excess.setLabel("bottom", "Site-pair distance", units="nm")
        excess.setLabel("left", "Observed − null count")
        excess.showGrid(x=True, y=True, alpha=0.18)
        excess.addLine(y=0, pen=pg.mkPen(100, 100, 100, style=Qt.PenStyle.DashLine))
        excess.plot(
            centers, np.asarray(self._result["summary"]["excess_counts"]),
            pen=pg.mkPen(20, 145, 75, width=2), fillLevel=0,
            brush=pg.mkBrush(20, 145, 75, 45))
        excess.addItem(pg.LinearRegionItem(
            values=(cfg.short_range_lo_nm, cfg.short_range_hi_nm), movable=False,
            brush=pg.mkBrush(255, 180, 50, 25), pen=pg.mkPen(220, 130, 20, 80)))
        root.addWidget(excess, 1)
        return holder

    def _build_report(self) -> QTextEdit:
        r = self._result
        s = r["summary"]
        b = r.get("bootstrap", {})
        cfg = r["config"]
        lines = [
            "INTERPRETATION",
            "A positive result supports a short-range population relative to the "
            "conditional surface null. It does not identify pair membership and does "
            "not estimate a molecular dimer distance.",
            "",
            "PRIMARY RESULT",
            f"Band: {cfg.short_range_lo_nm:g}–{cfg.short_range_hi_nm:g} nm",
            f"Observed pairs: {s['band_observed_pairs']:,}",
            f"Null expectation: {s['band_null_mean_pairs']:.1f} ± "
            f"{s['band_null_sd_pairs']:.1f}",
            f"Observed/null ratio: {s['band_ratio']:.3f}",
            f"Empirical one-sided p: {self._p_text(s['band_p'], cfg.null_replicates)}",
            f"Positive-excess peak / centroid / median: {s['peak_nm']:.2f} / "
            f"{s['positive_excess_centroid_nm']:.2f} / "
            f"{s['positive_excess_median_nm']:.2f} nm",
            "",
            "SITE AND COMPONENT DIAGNOSTICS",
            f"Repeated-site consolidation: {r['n_traces_consolidated']:,} trace(s) "
            f"collapsed across {r['n_repeated_sites']:,} site(s)",
            f"Median within-site RMS: {r['median_within_site_rms_nm']:.2f} nm",
            f"Components: {r['n_components']} retained / {r['n_components_all']} total; "
            f"{r['n_excluded_sites']} site(s) explicitly excluded",
            f"Rod-like PCA diagnostic: {r['n_rod_like_components']} of "
            f"{r['n_components']} retained component(s)",
        ]
        if b.get("available"):
            ci = b.get("centroid_ci95_nm", [np.nan, np.nan])
            ri = b.get("band_ratio_ci95", [np.nan, np.nan])
            lines += [
                "",
                "COMPONENT BOOTSTRAP",
                f"Positive-excess centroid 95% interval: {ci[0]:.2f}–{ci[1]:.2f} nm",
                f"Band-ratio 95% interval: {ri[0]:.2f}–{ri[1]:.2f}",
            ]
        else:
            lines += ["", "COMPONENT BOOTSTRAP", f"Unavailable: {b.get('reason', 'n/a')}"]

        sensitivity = r.get("sensitivity") or []
        if sensitivity:
            lines += [
                "", "SENSITIVITY AUDIT",
                f"Primary claim passes {r.get('sensitivity_passes', 0)} of "
                f"{r.get('sensitivity_valid_variants', 0)} valid variants; "
                f"robust = {bool(r.get('robust_short_range_excess'))}.",
                "merge nm | link nm | stratum sites | sites used | components | ratio | p | excess centroid nm",
            ]
            for row in sensitivity:
                lines.append(
                    f"{row['site_merge_nm']:7.1f} | {row['cell_link_nm']:7.0f} | "
                    f"{row['null_stratum_sites']:14d} | "
                    f"{row['n_sites_used']:10d} | {row['n_components']:10d} | "
                    f"{row['band_ratio']:5.2f} | {row['band_p']:.3g} | "
                    f"{row['positive_excess_centroid_nm']:7.2f}")
        lines += ["", "LIMITATIONS"] + [f"• {item}" for item in r.get("limitations", [])]
        report = QTextEdit()
        report.setReadOnly(True)
        report.setPlainText("\n".join(lines))
        return report
