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
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..analysis.hlyb_pairwise import PairFitConfig
from .text_select import make_labels_selectable

_HYPOTHESIS_LABELS = {
    "six_site": "six-site HlyB complex",
    "dimer_only": "dimer distance only",
    "no_structure": "no structure",
}


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
            "a forward model whose components are the same-site short-range "
            "population, the six HlyB labelling sites, and unrelated pairs.\n\n"
            "Unlike template matching this never imposes a merge radius, so no "
            "distance range is removed and no artificial peak is created. The "
            "HlyB class weights are fixed by the structure, so the fit is a test "
            "of the model rather than a flexible curve fit."
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

        self._fit_delta = QCheckBox("fit the antibody displacement (δ)")
        self._fit_delta.setChecked(bool(d.fit_label_offset))
        self._fit_delta.setToolTip(
            "The label displacement lengthens every observed distance. Fitting "
            "it decides from the data whether the label sits isotropically "
            "(δ ≈ 1 nm) or radially outward (δ ≈ 4 nm).")
        form.addRow("", self._fit_delta)

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
            fit_label_offset=bool(self._fit_delta.isChecked()),
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

        root = QVBoxLayout(self)
        root.addWidget(self._summary_label())

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._build_plot())
        splitter.addWidget(self._build_report())
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)
        QTimer.singleShot(0, lambda: splitter.setSizes([460, 300]))
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
                plot.plot(c, np.asarray(fit["complex_component"]),
                          pen=pg.mkPen(50, 90, 220, 230, width=2,
                                       style=Qt.PenStyle.DashLine),
                          name="HlyB complex")
                if not excess_mode:
                    plot.plot(c, np.asarray(fit["background_component"]),
                              pen=pg.mkPen(230, 150, 30, 220, width=2,
                                           style=Qt.PenStyle.DotLine),
                              name="unrelated pairs")
                delta = float(fit.get("label_offset_nm", 0.0))
                for d_nm in r.get("class_distances_nm", []):
                    line = pg.InfiniteLine(pos=float(d_nm) + delta, angle=90,
                                           pen=pg.mkPen(50, 90, 220, 90, width=1))
                    plot.addItem(line, ignoreBounds=True)
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

    add("MODEL COMPARISON  (short-range kernel pinned at its measured width)")
    add(f"  {'hypothesis':<26} {'dAIC':>10} {'complex pairs':>15} {'delta nm':>10} {'sigma nm':>10}")
    for name, fit in r.get("fits", {}).items():
        add(f"  {_HYPOTHESIS_LABELS.get(name, name):<26} {fit.get('delta_aic', 0.0):10.1f} "
            f"{fit.get('n_complex_pairs', 0.0):15.0f} {fit.get('label_offset_nm', 0.0):10.2f} "
            f"{fit.get('sigma_nm', 0.0):10.2f}")
    add("  Lower dAIC is better; 0 marks the preferred model. The HlyB class")
    add("  weights are fixed at 1/5 each, because every class holds three of the")
    add("  fifteen pairs and all scale as p^2 with labelling efficiency, so their")
    add("  ratio does not depend on it.")
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
    add("INTERPRETATION")
    delta = best.get("label_offset_nm")
    if delta is not None:
        add(f"  Fitted antibody displacement delta = {delta:.2f} nm.")
        if "label_offset_nm" in bounds:
            add("    This sits at a bound, so it is a limit rather than an estimate:")
            add("    the data prefer distances at or below the protein-domain values")
            add("    and give no support for a radially outward label (which would")
            add("    lengthen every class by 3-4 nm).")
        elif delta < 1.8:
            add("    Consistent with an isotropically oriented label.")
        else:
            add("    Consistent with a radially outward label, i.e. the tabulated")
            add("    distances that include 2 nm per antibody at each endpoint.")
    if bounds:
        add(f"  Parameters resting on a bound: {', '.join(bounds)}.")
    add("  Individual distance classes are NOT resolved: the three short classes")
    add("  span 2.1 nm against a pair blur of several nm. What is measured is the")
    add("  ensemble envelope and its outer cutoff, not a single distance.")
    return "\n".join(lines)
