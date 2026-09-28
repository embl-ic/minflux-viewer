"""UI for the staged model-independent HlyB 3-D short-range workflow."""

from __future__ import annotations

import colorsys

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCursor
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
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from ..analysis.hlyb_staged import Staged3DConfig
from .plot_format import apply_spatial_y_direction
from .text_select import make_labels_selectable

_VIEW_AXES = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}
_AXIS_LABELS = {"XY": ("X", "Y"), "XZ": ("X", "Z"), "YZ": ("Y", "Z")}
_MAX_RAW_POINTS = 100_000

# A pair link is one line segment, so the drawn set is capped rather than
# allowed to grow with the square of the site count.  The read-out always
# states when the cap is in force -- a silently thinned overlay would
# misreport how many pairs the selected bins actually hold.
_MAX_PAIR_LINKS = 20_000
# Distance labels follow the pair-analysis window's rules: only links long
# enough on screen to carry a number, longest first, and never more than can
# be read at once.
_MAX_DISTANCE_LABELS = 100
_PAIR_LABEL_MIN_PIXELS = 42.0
# One event-loop turn's worth of coalescing, so dragging a band selector
# re-draws once per settled position instead of once per mouse move.
_BAND_REDRAW_MS = 30

# Two independent highlight bands, so two populations -- say one near 14 nm and
# one near 25 nm -- can be marked and compared at once. The colours have to work
# on the white profile plot *and* as links and rings on the black site view, and
# they differ on the blue-yellow axis as well as in luminance, so they stay
# separable under the common red-green colour vision deficiencies. Neither is
# the orange of the pre-declared band or the green of the excess curve.
_BAND_STYLES = (
    ("A", (80, 195, 255)),          # cyan-blue
    ("B", (245, 120, 215)),         # magenta
)


class _Band:
    """One highlight selection: its region, its mirror and its drawn items.

    Built in three places -- the definition here, the site-view items in
    ``_build_spatial_items`` and the regions in ``_build_profile_view`` -- so
    the fields start empty and are filled as each view is constructed.
    """

    __slots__ = ("index", "key", "rgb", "region", "mirror",
                 "link_item", "glow_item", "enabled")

    def __init__(self, index: int, key: str, rgb: tuple) -> None:
        self.index = int(index)
        self.key = str(key)
        self.rgb = tuple(int(c) for c in rgb)
        self.region = None
        self.mirror = None
        self.link_item = None
        self.glow_item = None
        # Band A is always on; a second band is opt-in so the default view is
        # exactly the single-selection one.
        self.enabled = index == 0

    @property
    def label(self) -> str:
        return f"band {self.key}"

    def pen(self, alpha: int = 230, width: float = 2.0):
        return pg.mkPen(*self.rgb, alpha, width=width)

    def brush(self, alpha: int = 60):
        return pg.mkBrush(*self.rgb, alpha)

_COMPONENT_MODES = (
    ("Neighbour link (single-linkage)", "link"),
    ("Rod cell detection (XY projection)", "rod"),
)


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
            "Factor applied once to raw z. The project default is the fixed Z scaling factor 0.67.")
        form.addRow("Z scaling (Z scaling factor):", self._zscale)

        self._merge = self._dspin(1.0, 8.0, d.site_merge_nm, 1, 0.5, " nm")
        self._merge.setToolTip(
            "Hard maximum diameter for consolidating repeated trace centroids into one "
            "label-site estimate. Complete-link constraints prevent chaining. The "
            "3/4/5 nm sensitivity audit is enabled below.")
        form.addRow("Same-site max diameter:", self._merge)

        self._form = form
        self._component_mode = QComboBox()
        for label, key in _COMPONENT_MODES:
            self._component_mode.addItem(label, key)
        self._component_mode.setCurrentIndex(
            max(self._component_mode.findData(d.component_mode), 0))
        self._component_mode.setToolTip(
            "How the field is separated into cells. Pairs are never formed "
            "between components, and the null fits one local axis per "
            "component.\n\n"
            "Neighbour link groups sites by distance. Every pair closer than "
            "the link distance is linked directly, so the short-range pair "
            "counts are unaffected by how it carves the field up — but a "
            "fragment or a clump still gets an unreliable local axis, and "
            "nothing reports that it happened.\n\n"
            "Rod cell detection delineates rod-shaped cells of a stated width "
            "in the XY projection, supplies each cell's measured long axis to "
            "the null, and rejects objects that are the wrong size or shape "
            "instead of analysing them.")
        self._component_mode.currentIndexChanged.connect(self._sync_component_mode)
        form.addRow("Spatial components:", self._component_mode)

        self._cell_link = self._dspin(20.0, 1000.0, d.cell_link_nm, 0, 10.0, " nm")
        self._cell_link.setToolTip(
            "Coarse neighbor link used only to separate spatial/cell components. "
            "Pairs are never formed between components.")
        form.addRow("Spatial-component link:", self._cell_link)

        self._min_sites = self._ispin(3, 10000, d.min_sites_per_component)
        form.addRow("Min sites per component:", self._min_sites)

        self._rod_width = self._range_row(
            "_rod_min_width", "_rod_max_width", d.rod_min_width_nm,
            d.rod_max_width_nm, 100.0, 5000.0, 25.0)
        self._rod_width.setToolTip(
            "Width of the cells to detect — for E. coli typically 800–1100 nm.\n\n"
            "This is the width of the structure. It is measured on the "
            "smoothed density mask, which envelopes the cell somewhat wider, "
            "so the gate is automatically widened by twice the smoothing "
            "length. Every region's measured width is listed in the report, "
            "rejected ones included, so a window that is merely slightly off "
            "shows up immediately.")
        form.addRow("Cell width:", self._rod_width)

        self._rod_length = self._range_row(
            "_rod_min_length", "_rod_max_length", d.rod_min_length_nm,
            d.rod_max_length_nm, 100.0, 50000.0, 100.0)
        self._rod_length.setToolTip(
            "Length of the cells to detect. The upper bound is what rejects "
            "two cells merged end to end, which would otherwise pass every "
            "other gate as one long cell of exactly the right width.")
        form.addRow("Cell length:", self._rod_length)

        self._rod_smooth = self._dspin(
            0.0, 1000.0, max(d.rod_smooth_nm, 0.0), 0, 10.0, " nm")
        self._rod_smooth.setSpecialValueText("auto")
        self._rod_smooth.setToolTip(
            "How far apart two labelled positions may be and still land in one "
            "cell body. 0 = auto, derived from the measured spacing of the "
            "inferred sites.\n\n"
            "This is the setting that decides whether one cell is found as one "
            "cell. It is governed by the labelling sparsity, not by the optics: "
            "a sparsely labelled cell needs a longer bridging length, and too "
            "short a value shatters it into fragments. The value actually used "
            "is reported in the result.")
        form.addRow("Bridging length:", self._rod_smooth)

        self._rod_pixel = self._dspin(2.0, 200.0, d.rod_pixel_size_nm, 0, 5.0, " nm")
        self._rod_pixel.setToolTip(
            "Pixel size of the detection image. Must be at most an eighth of "
            "the minimum cell width, or the mask and its distance transform "
            "cannot resolve the cell across its width.")
        form.addRow("Detection pixel:", self._rod_pixel)

        self._rod_split = QCheckBox("cut thin bridges between cells")
        self._rod_split.setChecked(bool(d.rod_split_touching))
        self._rod_split.setToolTip(
            "Separate cell bodies joined by a constriction — nearly touching "
            "caps bridged by the morphological closing, a dividing cell, a "
            "spurious filament. Cells that overlap in projection while running "
            "parallel cannot be separated by any 2-D method; those are "
            "rejected by the width gate instead.")
        form.addRow("", self._rod_split)

        self._rod_axis = QCheckBox("null axis from the fitted cell axis")
        self._rod_axis.setChecked(bool(d.rod_use_axis))
        self._rod_axis.setToolTip(
            "Give the conditional randomization each cell's measured long axis "
            "instead of the component's own principal axis. The principal axis "
            "is the part that goes wrong on a fragment or a clump, so this is "
            "the main reason to prefer rod detection.")
        form.addRow("", self._rod_axis)

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
            "one-sided p-value of 0.01 — the p-value is censored there, so the "
            "reported evidence is the band ratio scored against the spread of "
            "these replicates, which does not saturate.")
        form.addRow("Null replicates:", self._null_reps)

        self._sensitivity = QCheckBox(
            "run site-radius, surface-null, ±25% component-link audit and "
            "stratification profile")
        self._sensitivity.setChecked(bool(d.run_sensitivity))
        form.addRow("Sensitivity:", self._sensitivity)
        root.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self._sync_component_mode()
        make_labels_selectable(self)

    def _range_row(self, lo_attr, hi_attr, lo_value, hi_value,
                   lo_limit, hi_limit, step) -> QWidget:
        """A ``<from> to <to>`` pair of nm spin boxes on one form row."""
        lo_spin = self._dspin(lo_limit, hi_limit, lo_value, 0, step, " nm")
        hi_spin = self._dspin(lo_limit, hi_limit, hi_value, 0, step, " nm")
        setattr(self, lo_attr, lo_spin)
        setattr(self, hi_attr, hi_spin)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(lo_spin)
        row.addWidget(QLabel("to"))
        row.addWidget(hi_spin)
        holder = QWidget()
        holder.setLayout(row)
        return holder

    def component_mode(self) -> str:
        return str(self._component_mode.currentData())

    def _sync_component_mode(self, *_args) -> None:
        """Grey out the knobs the selected component mode does not use."""
        rod = self.component_mode() == "rod"
        self._set_row_enabled(self._cell_link, not rod)
        for widget in (self._rod_width, self._rod_length, self._rod_smooth,
                       self._rod_pixel, self._rod_split, self._rod_axis):
            self._set_row_enabled(widget, rod)

    def _set_row_enabled(self, widget: QWidget, enabled: bool) -> None:
        widget.setEnabled(enabled)
        label = self._form.labelForField(widget)
        if label is not None:
            label.setEnabled(enabled)

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
        if self.component_mode() == "rod":
            for lo, hi in ((self._rod_min_width, self._rod_max_width),
                           (self._rod_min_length, self._rod_max_length)):
                if lo.value() > hi.value():
                    lo.setValue(hi.value())
            # The analysis rejects a pixel too coarse to resolve the width;
            # clamp here so that never surfaces as a failed run.
            coarsest = self._rod_min_width.value() / 8.0
            if self._rod_pixel.value() > coarsest:
                self._rod_pixel.setValue(coarsest)
        self.accept()

    def config(self) -> Staged3DConfig:
        base = Staged3DConfig()
        return Staged3DConfig(
            min_loc_per_trace=int(self._min_loc.value()),
            z_scaling_factor=float(self._zscale.value()),
            site_merge_nm=float(self._merge.value()),
            site_sigma_factor=base.site_sigma_factor,
            site_precision_floor_nm=base.site_precision_floor_nm,
            component_mode=self.component_mode(),
            cell_link_nm=float(self._cell_link.value()),
            min_sites_per_component=int(self._min_sites.value()),
            rod_min_width_nm=float(self._rod_min_width.value()),
            rod_max_width_nm=float(self._rod_max_width.value()),
            rod_min_length_nm=float(self._rod_min_length.value()),
            rod_max_length_nm=float(self._rod_max_length.value()),
            rod_pixel_size_nm=float(self._rod_pixel.value()),
            # 0 in the spin box means "auto"; the analysis spells that -1.
            rod_smooth_nm=(float(self._rod_smooth.value())
                           if self._rod_smooth.value() > 0 else -1.0),
            rod_close_nm=base.rod_close_nm,
            rod_split_touching=bool(self._rod_split.isChecked()),
            rod_use_axis=bool(self._rod_axis.isChecked()),
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
            sensitivity_rod_width_factors=base.sensitivity_rod_width_factors,
            # The stratification profile rides on the sensitivity switch: both
            # are robustness reporting rather than the primary computation.
            run_stratum_profile=bool(self._sensitivity.isChecked()),
            stratum_profile_sites=base.stratum_profile_sites,
            stratum_profile_ratio_tolerance=base.stratum_profile_ratio_tolerance,
            calibrated_ratio_z=base.calibrated_ratio_z,
        )


class HlyBStagedWindow(QDialog):
    """Modeless result window for the staged analysis."""

    def __init__(self, result: dict, *, title: str = "", owner=None,
                 prefs: dict | None = None) -> None:
        super().__init__(None)
        self._result = result
        self._owner = owner
        self._prefs = prefs or {}
        # Built lazily by _pair_table(); assigned here so the spatial view can
        # be drawn before the profile view that owns the band selector exists.
        self._pairs = None
        self._bands = [_Band(i, key, rgb)
                       for i, (key, rgb) in enumerate(_BAND_STYLES)]
        self._band_label = None
        self._band_check = None
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
        robust = r.get("robust_short_range_excess_calibrated")
        robustness = ("passes all sensitivity variants" if robust is True else
                      "does NOT pass every sensitivity variant" if robust is False else
                      "sensitivity audit not run")
        # The ratio is quoted against its own null spread rather than the
        # empirical p, which is censored at 1/(replicates + 1).
        null_sd = s.get("null_band_ratio_sd", float("nan"))
        ratio_text = f"excess ratio: {s['band_ratio']:.2f}"
        if np.isfinite(null_sd) and null_sd > 0:
            ratio_text += (f" vs null {s.get('null_band_ratio_mean', 1.0):.2f}"
                           f"±{null_sd:.2f} ({s.get('band_ratio_z', float('nan')):.0f}σ)")
        span = r.get("centroid_sensitivity_range_nm") or []
        centroid_text = (f"positive-excess centroid: "
                         f"{s['positive_excess_centroid_nm']:.2f} nm")
        if len(span) == 2 and np.isfinite(span[0]) and np.isfinite(span[1]):
            centroid_text += f" (sensitivity {span[0]:.1f}–{span[1]:.1f})"
        label = QLabel(
            f"Traces: {r['n_traces_used']:,}/{r['n_traces_total']:,} used  |  "
            f"Inferred sites: {r['n_sites']:,} ({r['n_sites_used']:,} in "
            f"{r['n_components']} component(s))  |  "
            f"{r['config'].short_range_lo_nm:g}–{r['config'].short_range_hi_nm:g} nm "
            f"{ratio_text}  |  {centroid_text}  |  {robustness}"
        )
        label.setWordWrap(True)
        label.setToolTip(
            "The excess centroid describes the positive observed-minus-null population. "
            "It is not a fitted or assigned HlyB dimer distance.\n\n"
            "The ratio is quoted against the spread of the null replicates because "
            "the empirical p-value cannot fall below 1/(replicates + 1). The ratio's "
            "magnitude is conditional on the null stratification scale; see the "
            "stratum profile in the report.")
        return label

    def _build_spatial_view(self) -> QWidget:
        holder = QWidget()
        root = QVBoxLayout(holder)
        root.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Projection:"))
        self._view_combo = QComboBox()
        self._view_combo.addItems(["XY", "XZ", "YZ"])
        self._view_combo.currentTextChanged.connect(self._on_projection_changed)
        row.addWidget(self._view_combo)
        self._raw_check = QCheckBox("raw loc")
        self._raw_check.setChecked(False)
        self._raw_check.setToolTip(
            "Every localization of every used trace, thinned for drawing.")
        self._trace_check = QCheckBox("trace centroid")
        self._trace_check.setChecked(True)
        self._trace_check.setToolTip(
            "One point per trace: the sub-unit position before repeated visits "
            "to the same label are consolidated into a site.")
        self._merge_check = QCheckBox("consolidation link")
        self._merge_check.setChecked(False)
        has_map = self._trace_site_map().size > 0
        self._merge_check.setEnabled(has_map)
        self._merge_check.setToolTip(
            "Joins each trace centroid to the label site it was consolidated "
            "into, so which repeats were merged is visible. A site built from a "
            "single trace draws a zero-length link."
            if has_map else
            "This result carries no trace-to-site map, so which traces were "
            "consolidated cannot be drawn.")
        self._site_check = QCheckBox("inferred label site")
        self._site_check.setChecked(True)
        self._site_check.setToolTip(
            "The consolidated label positions the pair distances are measured "
            "between. Colour is the spatial component; hover for details.")
        self._pair_check = QCheckBox("selected pair link")
        self._pair_check.setChecked(True)
        self._pair_check.setToolTip(
            "Draws a line between the two sites of every pair counted in the "
            "selected histogram bins, with its distance where the line is long "
            "enough on screen to carry a number. Drag the band on the pair "
            "profile below to change the selection.")
        self._null_check = QCheckBox("one null draw")
        self._null_check.setChecked(False)
        self._null_check.setToolTip(
            "One randomization from the conditional null, for visual "
            "comparison with the inferred sites.")
        self._rod_check = QCheckBox("detected cell")
        has_rods = (self._result.get("cell_detection") is not None
                    or self._result.get("rod_detection") is not None)
        self._rod_check.setChecked(has_rods)
        self._rod_check.setEnabled(has_rods)
        self._rod_check.setToolTip(
            "Outlines of the detected cells (XY projection only). Shape-prior "
            "cells are fitted as full capsules; legacy rod-detector regions "
            "rejected by size or shape gates are dashed."
            if has_rods else
            "Available when spatial components come from automatic cell detection.")
        for box in (self._raw_check, self._trace_check, self._merge_check,
                    self._site_check, self._pair_check, self._null_check,
                    self._rod_check):
            box.toggled.connect(self._refresh_spatial)
            row.addWidget(box)
        row.addStretch(1)
        root.addLayout(row)

        self._spatial_plot = pg.PlotWidget(background="k")
        self._spatial_plot.showGrid(x=True, y=True, alpha=0.15)
        self._spatial_plot.getViewBox().setAspectLocked(True)
        root.addWidget(self._spatial_plot, 1)
        self._build_spatial_items()
        # A label is placed by how long its link is *on screen*, so the set is
        # recomputed when the view moves, not only when the data changes.
        self._spatial_plot.getViewBox().sigRangeChanged.connect(
            self._schedule_label_refresh)
        self._refresh_spatial(refit=True)
        return holder

    def _build_spatial_items(self) -> None:
        """Create the plot items once and keep them.

        Rebuilding the scene on every toggle would auto-range it again, which
        throws away the pan and zoom the user chose \u2014 and with a draggable
        band selector driving this view, that would happen continuously.
        """
        plot = self._spatial_plot
        self._cell_items: list = []
        self._label_items: dict[int, pg.TextItem] = {}
        self._raw_item = pg.ScatterPlotItem(
            size=2, pen=None, brush=pg.mkBrush(90, 90, 90, 90), pxMode=True)
        self._null_item = pg.ScatterPlotItem(
            size=5, pen=pg.mkPen(120, 190, 255, 180), brush=None, pxMode=True)
        self._merge_item = pg.PlotDataItem()
        self._trace_item = pg.ScatterPlotItem(
            size=3, pen=None, brush=pg.mkBrush(255, 150, 40, 120), pxMode=True)
        self._site_item = pg.ScatterPlotItem(
            size=7, pen=pg.mkPen(230, 230, 230, 90), pxMode=True)
        # One link and one ring layer per band, so two selections can be shown
        # at once in their own colours. A single item takes a single pen.
        layers: list = [self._raw_item, self._null_item, self._merge_item]
        for offset, band in enumerate(self._bands):
            band.link_item = pg.PlotDataItem()
            # Rings are sized apart so two bands sharing a site both stay
            # visible instead of one hiding inside the other.
            band.glow_item = pg.ScatterPlotItem(
                size=13 + 5 * offset, pen=pg.mkPen(*band.rgb, 220),
                brush=None, pxMode=True)
            layers.append(band.link_item)
        layers.append(self._trace_item)
        layers.extend(band.glow_item for band in self._bands)
        layers.append(self._site_item)
        for z, item in enumerate(layers):
            item.setZValue(z)
            plot.addItem(item)
        self._site_item.sigHovered.connect(self._on_site_hovered)
        self._trace_item.sigHovered.connect(self._on_trace_hovered)

    def _on_projection_changed(self, *_args) -> None:
        """A different projection is a different picture, so it is re-fitted."""
        self._refresh_spatial(refit=True)

    @staticmethod
    def _component_brush(component: int):
        if component < 0:
            return pg.mkBrush(120, 120, 120, 180)
        hue = (0.61803398875 * component) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
        return pg.mkBrush(*(int(255 * value) for value in rgb), 230)

    def _pair_table(self) -> dict:
        """Every within-component pair the profile counts, computed once."""
        if getattr(self, "_pairs", None) is None:
            from ..analysis.hlyb_staged import within_component_pairs

            sites = np.asarray(
                self._result.get("site_centers_nm", np.empty((0, 3))), dtype=float)
            labels = np.asarray(
                self._result.get("component_labels", np.full(sites.shape[0], -1)),
                dtype=np.int64)
            if sites.shape[0] < 2 or labels.size != sites.shape[0]:
                self._pairs = {
                    "pairs": np.empty((0, 2), dtype=np.int64),
                    "distances_nm": np.empty(0, dtype=float),
                    "component": np.empty(0, dtype=np.int64),
                }
            else:
                self._pairs = within_component_pairs(
                    sites, labels, r_max_nm=float(self._result["config"].r_max_nm))
        return self._pairs

    def _default_band_range(self, band: int) -> tuple[float, float]:
        """Where a band opens before the user has moved it.

        Band A opens on the pre-declared test range, so the window starts by
        reproducing the reported band counts. A second band opens just above it
        rather than on top of it -- two selections at the same place would look
        like one.
        """
        cfg = self._result["config"]
        lo = float(cfg.short_range_lo_nm)
        hi = float(cfg.short_range_hi_nm)
        if band == 0:
            return lo, hi
        width = max(hi - lo, float(cfg.bin_nm))
        r_max = float(cfg.r_max_nm)
        start = min(hi, max(0.0, r_max - width))
        return start, min(r_max, start + width)

    def _enabled_bands(self) -> list:
        return [band for band in self._bands if band.enabled]

    def _band_range(self, band: int = 0) -> tuple[float, float]:
        region = self._bands[band].region
        if region is None:
            return self._default_band_range(band)
        lo, hi = sorted(float(value) for value in region.getRegion())
        return lo, hi

    def selected_bin_mask(self, band: int = 0) -> np.ndarray:
        """Histogram bins currently inside *band*'s selector."""
        from ..analysis.hlyb_staged import bin_mask_for_range

        lo, hi = self._band_range(band)
        return bin_mask_for_range(
            np.asarray(self._result["edges_nm"], dtype=float), lo, hi)

    def selected_pair_mask(self, band: int = 0) -> np.ndarray:
        """Pairs counted by *band*'s bins \u2014 the same pairs, not a re-cut."""
        from ..analysis.hlyb_staged import pairs_in_bins

        return pairs_in_bins(
            self._pair_table()["distances_nm"],
            np.asarray(self._result["edges_nm"], dtype=float),
            self.selected_bin_mask(band))

    def _refresh_spatial(self, *_args, refit: bool = False) -> None:
        if not hasattr(self, "_spatial_plot"):
            return
        plot = self._spatial_plot
        view = self._view_combo.currentText() if hasattr(self, "_view_combo") else "XY"
        a, b = _VIEW_AXES[view]
        xlab, ylab = _AXIS_LABELS[view]
        plot.setLabel("bottom", xlab, units="nm")
        plot.setLabel("left", ylab, units="nm")
        apply_spatial_y_direction(
            plot,
            vertical_coordinate=ylab,
            prefs=self._prefs,
            preference_key="scatter_xy_origin",
        )

        points = np.asarray(
            self._result.get("points_nm", np.empty((0, 3))), dtype=float)
        if self._raw_check.isChecked() and points.shape[0]:
            if points.shape[0] > _MAX_RAW_POINTS:
                points = points[::int(np.ceil(points.shape[0] / _MAX_RAW_POINTS))]
            self._raw_item.setData(x=points[:, a], y=points[:, b])
        else:
            self._raw_item.setData(x=[], y=[])
        self._raw_item.setVisible(self._raw_check.isChecked())

        traces = np.asarray(
            self._result.get("trace_centroids_nm", np.empty((0, 3))), dtype=float)
        if self._trace_check.isChecked() and traces.shape[0]:
            self._trace_item.setData(
                x=traces[:, a], y=traces[:, b],
                data=np.arange(traces.shape[0], dtype=int),
                hoverable=True, tip=None)
        else:
            self._trace_item.setData(x=[], y=[], hoverable=True, tip=None)
        self._trace_item.setVisible(self._trace_check.isChecked())

        null = np.asarray(
            self._result.get("null_preview_sites_nm", np.empty((0, 3))), dtype=float)
        if self._null_check.isChecked() and null.shape[0]:
            self._null_item.setData(x=null[:, a], y=null[:, b])
        else:
            self._null_item.setData(x=[], y=[])
        self._null_item.setVisible(self._null_check.isChecked())

        self._refresh_detected_cells(view)
        self._refresh_merge_links(a, b)
        self._refresh_pair_links(a, b)
        self._refresh_sites(a, b)
        if refit:
            plot.getViewBox().autoRange()
        self._refresh_pair_labels()
        self._refresh_band_readout()

    def _refresh_sites(self, a: int, b: int) -> None:
        sites = np.asarray(
            self._result.get("site_centers_nm", np.empty((0, 3))), dtype=float)
        labels = np.asarray(
            self._result.get("component_labels", np.full(sites.shape[0], -1)),
            dtype=np.int64)
        show = self._site_check.isChecked() and sites.shape[0] > 0
        if show:
            self._site_item.setData(
                x=sites[:, a], y=sites[:, b],
                data=np.arange(sites.shape[0], dtype=int),
                brush=[self._component_brush(int(c)) for c in labels],
                hoverable=True, tip=None)
        else:
            self._site_item.setData(x=[], y=[], hoverable=True, tip=None)
        self._site_item.setVisible(show)

        # The sites taking part in a selected pair, ringed in their band's own
        # colour so a highlighted distance can be traced back to the labels
        # that produced it -- and so two bands stay told apart.
        links_on = self._pair_check.isChecked()
        for band in self._bands:
            involved = (self._selected_site_indices(band.index)
                        if band.enabled else np.empty(0, dtype=np.int64))
            glow = show and links_on and band.enabled and involved.size > 0
            if glow:
                band.glow_item.setData(
                    x=sites[involved, a], y=sites[involved, b])
            else:
                band.glow_item.setData(x=[], y=[])
            band.glow_item.setVisible(glow)

    def _selected_site_indices(self, band: int = 0) -> np.ndarray:
        pairs = self._pair_table()["pairs"]
        if pairs.shape[0] == 0:
            return np.empty(0, dtype=np.int64)
        return np.unique(pairs[self.selected_pair_mask(band)])

    def _selected_pair_rows(self, band: int = 0) -> np.ndarray:
        """Indices of *band*'s drawn pairs, longest first when the cap applies."""
        table = self._pair_table()
        rows = np.flatnonzero(self.selected_pair_mask(band))
        if rows.size > _MAX_PAIR_LINKS:
            order = np.argsort(table["distances_nm"][rows])[::-1]
            rows = np.sort(rows[order[:_MAX_PAIR_LINKS]])
        return rows

    def _refresh_pair_links(self, a: int, b: int) -> None:
        show = self._pair_check.isChecked()
        sites = np.asarray(
            self._result.get("site_centers_nm", np.empty((0, 3))), dtype=float)
        table = self._pair_table()
        for band in self._bands:
            item = band.link_item
            if not show or not band.enabled:
                item.setData(x=[], y=[])
                item.setVisible(False)
                continue
            rows = self._selected_pair_rows(band.index)
            if rows.size == 0 or sites.shape[0] == 0:
                item.setData(x=[], y=[])
            else:
                left = sites[table["pairs"][rows, 0]]
                right = sites[table["pairs"][rows, 1]]
                gap = np.full(rows.size, np.nan)
                xs = np.column_stack([left[:, a], right[:, a], gap]).ravel()
                ys = np.column_stack([left[:, b], right[:, b], gap]).ravel()
                item.setData(xs, ys, connect="finite",
                             pen=band.pen(190, width=1.2))
            item.setVisible(True)

    def _trace_site_map(self) -> np.ndarray:
        """Trace-to-site map, empty when this result does not carry one.

        The pooled entry point once exported ``None`` here, so the array is
        normalized in one place rather than at each reader.
        """
        traces = np.asarray(
            self._result.get("trace_centroids_nm", np.empty((0, 3))), dtype=float)
        mapping = self._result.get("trace_to_site")
        if mapping is None:
            return np.empty(0, dtype=np.int64)
        mapping = np.asarray(mapping, dtype=np.int64).ravel()
        return mapping if mapping.size == traces.shape[0] else np.empty(
            0, dtype=np.int64)

    def _refresh_merge_links(self, a: int, b: int) -> None:
        """Trace centroid to the label site it was consolidated into."""
        if not self._merge_check.isChecked():
            self._merge_item.setData(x=[], y=[])
            self._merge_item.setVisible(False)
            return
        traces = np.asarray(
            self._result.get("trace_centroids_nm", np.empty((0, 3))), dtype=float)
        sites = np.asarray(
            self._result.get("site_centers_nm", np.empty((0, 3))), dtype=float)
        mapping = self._trace_site_map()
        if (mapping.size == traces.shape[0] and sites.shape[0] > 0
                and traces.shape[0] > 0):
            valid = (mapping >= 0) & (mapping < sites.shape[0])
            src = traces[valid]
            dst = sites[mapping[valid]]
            gap = np.full(src.shape[0], np.nan)
            xs = np.column_stack([src[:, a], dst[:, a], gap]).ravel()
            ys = np.column_stack([src[:, b], dst[:, b], gap]).ravel()
            self._merge_item.setData(
                xs, ys, connect="finite",
                pen=pg.mkPen(255, 150, 40, 110, width=0.9))
        else:
            self._merge_item.setData(x=[], y=[])
        self._merge_item.setVisible(True)

    def _refresh_detected_cells(self, view: str) -> None:
        """Capsule outlines of the detected cells \u2014 an XY-plane geometry."""
        for item in self._cell_items:
            self._spatial_plot.removeItem(item)
        self._cell_items = []
        detection = (self._result.get("cell_detection")
                     or self._result.get("rod_detection"))
        if detection is None or view != "XY" or not self._rod_check.isChecked():
            return
        if hasattr(detection, "instances"):
            from ..analysis.shape_segmentation import instance_outline
            outlined = [(instance_outline(instance), True)
                        for instance in detection.instances]
        else:
            from ..analysis.rod_segmentation import capsule_outline
            outlined = [(capsule_outline(rod), bool(rod.accepted))
                        for rod in detection.rods]

        for outline, accepted in outlined:
            pen = (pg.mkPen(90, 235, 150, width=2) if accepted else
                   pg.mkPen(235, 120, 95, width=1, style=Qt.PenStyle.DashLine))
            item = pg.PlotCurveItem(outline[:, 0], outline[:, 1], pen=pen)
            item.setZValue(-1)
            self._spatial_plot.addItem(item)
            self._cell_items.append(item)

    def _schedule_label_refresh(self, *_args) -> None:
        timer = getattr(self, "_label_timer", None)
        if timer is None:
            timer = self._label_timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._refresh_pair_labels)
        timer.start(_BAND_REDRAW_MS)

    def _refresh_pair_labels(self) -> None:
        """Distances on the links long enough on screen to carry a number."""
        if not hasattr(self, "_spatial_plot"):
            return
        existing = self._label_items
        if not self._pair_check.isChecked():
            for item in existing.values():
                item.setVisible(False)
            return
        sites = np.asarray(
            self._result.get("site_centers_nm", np.empty((0, 3))), dtype=float)
        table = self._pair_table()
        a, b = _VIEW_AXES[self._view_combo.currentText()]
        (x0, x1), (y0, y1) = self._spatial_plot.getPlotItem().vb.viewRange()
        width = max(int(self._spatial_plot.viewport().width()), 1)
        height = max(int(self._spatial_plot.viewport().height()), 1)
        # Candidates are pooled across the bands and capped once, so turning a
        # second band on does not double the number of numbers on screen.
        pool: list[tuple] = []
        for band in self._enabled_bands():
            rows = self._selected_pair_rows(band.index)
            if not rows.size or not sites.shape[0] or x1 <= x0 or y1 <= y0:
                continue
            left = sites[table["pairs"][rows, 0]]
            right = sites[table["pairs"][rows, 1]]
            dx = np.abs(right[:, a] - left[:, a]) / (x1 - x0) * width
            dy = np.abs(right[:, b] - left[:, b]) / (y1 - y0) * height
            pixels = np.hypot(dx, dy)
            mid_x = 0.5 * (left[:, a] + right[:, a])
            mid_y = 0.5 * (left[:, b] + right[:, b])
            on_screen = ((mid_x >= x0) & (mid_x <= x1)
                         & (mid_y >= y0) & (mid_y <= y1))
            for local in np.flatnonzero(
                    on_screen & (pixels >= _PAIR_LABEL_MIN_PIXELS)):
                pool.append((
                    float(pixels[local]), (band.index, int(rows[local])),
                    float(mid_x[local]), float(mid_y[local]),
                    float(table["distances_nm"][rows[local]]), band.rgb))
        pool.sort(key=lambda row: row[0], reverse=True)
        selected = {row[1]: row[2:] for row in pool[:_MAX_DISTANCE_LABELS]}
        for key, item in existing.items():
            if key in selected:
                item.setPos(selected[key][0], selected[key][1])
                item.setVisible(True)
            else:
                item.setVisible(False)
        # Hidden items are kept so panning back and forth does not rebuild
        # them, but only up to a bound: the pair count grows with the square of
        # the site count, and every pair that was ever labelled would otherwise
        # keep a QGraphicsItem for the life of the window.
        if len(existing) > 4 * _MAX_DISTANCE_LABELS:
            for key in [k for k in existing if k not in selected]:
                self._spatial_plot.removeItem(existing.pop(key))
        for key, (x, y, distance, rgb) in selected.items():
            if key in existing:
                continue
            # Coloured by its band, so a number can be attributed to the
            # selection it came from without tracing its line.
            item = pg.TextItem(
                f"{distance:.1f}", color=rgb, anchor=(0.5, 0.5),
                fill=pg.mkBrush(0, 0, 0, 165))
            item.setZValue(20)
            item.setPos(x, y)
            self._spatial_plot.addItem(item, ignoreBounds=True)
            existing[key] = item

    def _on_site_hovered(self, _item, points, _event) -> None:
        if points is None or len(points) == 0:
            QToolTip.hideText()
            return
        text = self._site_tooltip(points[0])
        if text:
            QToolTip.showText(QCursor.pos(), text, self)

    def _site_tooltip(self, point) -> str:
        sites = np.asarray(
            self._result.get("site_centers_nm", np.empty((0, 3))), dtype=float)
        index = point.data()
        if index is None or not (0 <= int(index) < sites.shape[0]):
            return ""
        index = int(index)
        labels = np.asarray(
            self._result.get("component_labels", np.full(sites.shape[0], -1)),
            dtype=np.int64)
        component = (f"{labels[index] + 1}" if labels[index] >= 0
                     else "\u2014 (excluded from pairing)")
        position = sites[index]
        lines = [
            f"Inferred label site {index + 1}",
            f"X {position[0]:.2f} nm   Y {position[1]:.2f} nm   "
            f"Z {position[2]:.2f} nm",
            f"Component {component}",
        ]
        mapping = self._trace_site_map()
        if mapping.size:
            lines.append(
                f"Traces consolidated: {int(np.sum(mapping == index))}")
        pairs = self._pair_table()["pairs"]
        if pairs.shape[0]:
            member = np.any(pairs == index, axis=1)
            # Named per band, so a site ringed twice says which selection
            # each of its pairs belongs to.
            within = (f" of {int(member.sum())} within "
                      f"{self._result['config'].r_max_nm:g} nm")
            for band in self._enabled_bands():
                hits = int(np.sum(member & self.selected_pair_mask(band.index)))
                lines.append(f"Pairs in {band.label}: {hits}{within}")
                within = ""
        return "\n".join(lines)

    def _on_trace_hovered(self, _item, points, _event) -> None:
        if points is None or len(points) == 0:
            QToolTip.hideText()
            return
        traces = np.asarray(
            self._result.get("trace_centroids_nm", np.empty((0, 3))), dtype=float)
        index = points[0].data()
        if index is None or not (0 <= int(index) < traces.shape[0]):
            return
        index = int(index)
        position = traces[index]
        mapping = self._trace_site_map()
        site = (f"site {int(mapping[index]) + 1}"
                if index < mapping.size and mapping[index] >= 0 else "no site")
        QToolTip.showText(
            QCursor.pos(),
            f"Trace centroid {index + 1}\n"
            f"X {position[0]:.2f} nm   Y {position[1]:.2f} nm   "
            f"Z {position[2]:.2f} nm\n"
            f"Consolidated into {site}",
            self)

    def _build_profile_view(self) -> QWidget:
        holder = QWidget()
        outer = QVBoxLayout(holder)
        outer.setContentsMargins(0, 0, 0, 0)
        root = QHBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        centers = np.asarray(self._result["centers_nm"], dtype=float)
        observed = np.asarray(self._result["observed"], dtype=float)
        mean = np.asarray(self._result["null_mean"], dtype=float)
        lo = np.asarray(self._result["null_lo"], dtype=float)
        hi = np.asarray(self._result["null_hi"], dtype=float)
        cfg = self._result["config"]
        null_name = ("ROI-conditioned null" if self._result.get("is_2d")
                     else "surface null")

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
        profile.plot(centers, mean, pen=pg.mkPen(45, 105, 190, width=2),
                     name=null_name)
        profile.plot(centers, observed, pen=pg.mkPen(30, 30, 30, width=2),
                     name="observed")
        # The pre-declared band is marked by its two edges rather than by a
        # second shaded region: shading both made the fixed test range and the
        # movable selection read as one band, which is exactly the distinction
        # that has to stay visible.
        self._declared_lines = []
        for position in (cfg.short_range_lo_nm, cfg.short_range_hi_nm):
            line = pg.InfiniteLine(
                pos=float(position), angle=90,
                pen=pg.mkPen(220, 130, 20, 200, width=2,
                             style=Qt.PenStyle.DashLine))
            line.setZValue(-10)
            line.setToolTip(
                f"Pre-declared test band: "
                f"{cfg.short_range_lo_nm:g}–{cfg.short_range_hi_nm:g} nm. "
                f"The reported result is this range; the blue selection only "
                f"chooses what the site view highlights.")
            profile.addItem(line, ignoreBounds=True)
            self._declared_lines.append(line)
        for band in self._bands:
            band.region = pg.LinearRegionItem(
                values=self._default_band_range(band.index), movable=True,
                brush=band.brush(60), pen=band.pen(230),
                hoverBrush=band.brush(95))
            band.region.setZValue(10 + band.index)
            band.region.setToolTip(
                f"Highlight {band.label}. Drag to choose which histogram bins "
                f"it covers; the pairs counted in them are drawn as links in "
                f"that colour in the site view above. The orange dashed lines "
                f"mark the pre-declared test range, which no selector moves.")
            band.region.sigRegionChanged.connect(
                lambda *_a, _i=band.index: self._on_band_changed(_i))
            band.region.sigRegionChangeFinished.connect(
                lambda *_a, _i=band.index: self._snap_band_to_bins(_i))
            profile.addItem(band.region)
            band.region.setVisible(band.enabled)
        self._profile_plot = profile
        root.addWidget(profile, 1)

        excess = pg.PlotWidget(background="w")
        excess.setLabel("bottom", "Site-pair distance", units="nm")
        excess.setLabel("left", "Observed \u2212 null count")
        excess.showGrid(x=True, y=True, alpha=0.18)
        excess.addLine(y=0, pen=pg.mkPen(100, 100, 100, style=Qt.PenStyle.DashLine))
        excess.plot(
            centers, np.asarray(self._result["summary"]["excess_counts"]),
            pen=pg.mkPen(20, 145, 75, width=2), fillLevel=0,
            brush=pg.mkBrush(20, 145, 75, 45))
        excess.addItem(pg.LinearRegionItem(
            values=(cfg.short_range_lo_nm, cfg.short_range_hi_nm), movable=False,
            brush=pg.mkBrush(255, 180, 50, 25), pen=pg.mkPen(220, 130, 20, 80)))
        # Read-only mirrors, so both panels show the same selections without
        # giving the user two places to drag each one from.
        for band in self._bands:
            band.mirror = pg.LinearRegionItem(
                values=self._band_range(band.index), movable=False,
                brush=band.brush(35), pen=band.pen(140, width=1.0))
            excess.addItem(band.mirror)
            band.mirror.setVisible(band.enabled)
        root.addWidget(excess, 1)
        outer.addLayout(root, 1)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        second = self._bands[1]
        self._band_check = QCheckBox(f"second highlight ({second.label})")
        self._band_check.setChecked(second.enabled)
        self._band_check.setToolTip(
            "Add a second, independently coloured selection, so two "
            "populations \u2014 say one near 14 nm and one near 25 nm \u2014 "
            "can be marked and compared in the site view at the same time.")
        self._band_check.toggled.connect(self._on_second_band_toggled)
        row.addWidget(self._band_check)
        self._band_label = QLabel()
        self._band_label.setWordWrap(True)
        self._band_label.setToolTip(
            "The selected bins, and the pairs they count. The pair count is "
            "the same number the observed curve shows over those bins, because "
            "both come from one enumeration of the within-component pairs.")
        row.addWidget(self._band_label, 1)
        outer.addLayout(row)
        self._refresh_band_readout()
        return holder

    def _on_second_band_toggled(self, checked: bool) -> None:
        """Show or hide the second selection, in every panel at once."""
        band = self._bands[1]
        band.enabled = bool(checked)
        if band.region is not None:
            band.region.setVisible(band.enabled)
        if band.mirror is not None:
            band.mirror.setVisible(band.enabled)
        self._refresh_spatial()

    def _on_band_changed(self, band: int = 0) -> None:
        """Coalesce a drag into one redraw per settled position."""
        mirror = self._bands[band].mirror
        if mirror is not None:
            mirror.setRegion(self._band_range(band))
        self._refresh_band_readout()
        timer = getattr(self, "_band_timer", None)
        if timer is None:
            timer = self._band_timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._refresh_spatial)
        timer.start(_BAND_REDRAW_MS)

    def _snap_band_to_bins(self, band: int = 0) -> None:
        """Settle a selector on the bin edges it actually selected.

        The selection is a set of whole bins, so leaving the handles between
        edges would show a range that does not match the pairs highlighted
        from it.
        """
        edges = np.asarray(self._result["edges_nm"], dtype=float)
        mask = self.selected_bin_mask(band)
        if not mask.any():
            return
        inside = np.flatnonzero(mask)
        snapped = (float(edges[inside[0]]), float(edges[inside[-1] + 1]))
        region = self._bands[band].region
        if region is not None and not np.allclose(snapped, region.getRegion()):
            region.setRegion(snapped)

    def band_readout(self, band: int = 0) -> str:
        """One line describing what *band* currently selects."""
        observed = np.asarray(self._result["observed"], dtype=float)
        mean = np.asarray(self._result["null_mean"], dtype=float)
        mask = self.selected_bin_mask(band)
        lo, hi = self._band_range(band)
        name = self._bands[band].label
        if not mask.any():
            return (f"{name} {lo:.2f}\u2013{hi:.2f} nm: no histogram bin falls "
                    f"in this range.")
        drawn = self._selected_pair_rows(band).size
        total = int(round(float(observed[mask].sum())))
        expected = float(mean[mask].sum())
        ratio = (f"{total / expected:.2f}" if expected > 0 else "n/a")
        sites = self._selected_site_indices(band).size
        capped = ("" if drawn >= total else
                  f"  \u00b7  drawing the {drawn:,} longest of them")
        return (f"{name} {lo:.2f}\u2013{hi:.2f} nm ({int(mask.sum())} bin(s)): "
                f"{total:,} pair(s) over {sites:,} site(s)  \u00b7  null expects "
                f"{expected:,.0f}  \u00b7  ratio {ratio}{capped}")

    def _refresh_band_readout(self) -> None:
        label = getattr(self, "_band_label", None)
        if label is None:
            return
        # One line per enabled band, so a second selection is described rather
        # than merely drawn.
        label.setText("\n".join(
            self.band_readout(band.index) for band in self._enabled_bands()))

    def _rod_report_lines(self) -> list[str]:
        """Detection block: what was found, what was rejected, and on what number.

        The measured width of every region is listed, rejected ones included,
        so a width window that is merely slightly off reads as such instead of
        as an empty result.
        """
        summary = self._result.get("rod_segmentation")
        if not summary:
            return []
        cfg = self._result["config"]
        window = summary.get("width_window_nm") or [
            cfg.rod_min_width_nm, cfg.rod_max_width_nm]
        lines = [
            "",
            "ROD CELL DETECTION",
            f"Accepted {summary.get('n_accepted', 0)} of "
            f"{summary.get('n_regions', 0)} region(s); "
            f"{summary.get('n_components_kept', 0)} kept as component(s) after the "
            f"minimum-site cut",
            f"Width window: {window[0]:g}–{window[1]:g} nm "
            f"(±{summary.get('width_tolerance_nm', 0.0):.0f} nm mask tolerance) at "
            f"{summary.get('pixel_size_nm', 0.0):g} nm/pixel",
            f"Bridging length: {summary.get('smooth_nm', 0.0):.0f} nm"
            + (" (auto, from the measured site spacing)"
               if summary.get("smooth_is_auto") else " (set manually)")
            + f", closing {summary.get('close_nm', 0.0):.0f} nm",
        ]
        if summary.get("n_split"):
            lines.append(
                f"Thin bridges cut: {summary['n_split']} region(s) separated that "
                "the density mask had joined")
        accepted = [rod for rod in summary.get("rods", []) if rod.get("accepted")]
        if accepted:
            lines.append(
                "Accepted cells (width × length nm, sites): " + ", ".join(
                    f"{rod['width_nm']:.0f}×{rod['length_nm']:.0f} ({rod['n_sites']})"
                    for rod in accepted))
        rejections = summary.get("rejections") or {}
        if rejections:
            lines.append("Rejected: " + ", ".join(
                f"{count} {reason}" for reason, count in sorted(rejections.items())))
        widths = [float(w) for w in (summary.get("region_widths_nm") or [])]
        if widths:
            lines.append(
                f"Measured widths of all regions: {min(widths):.0f}–{max(widths):.0f} nm "
                "— widen the window if genuine cells sit just outside it")
        return lines

    def _shape_report_lines(self) -> list[str]:
        summary = self._result.get("cell_segmentation") or {}
        if summary.get("mode") != "capsule_shape_prior":
            return []
        instances = summary.get("instances") or []
        sizes = ", ".join(
            f"{item.get('width_nm', float('nan')):.0f}×"
            f"{item.get('length_nm', float('nan')):.0f}"
            for item in instances)
        components = summary.get("components") or []
        multi = sum(int(row.get("chosen_k", 0)) > 1 for row in components)
        lines = [
            "",
            "AUTOMATIC E. COLI DETECTION",
            f"Fitted {summary.get('n_instances', len(instances))} capsule(s); "
            f"{summary.get('n_components_kept', 0)} kept after the minimum-site cut",
            f"Connected footprints explained by multiple cells: {multi}",
            f"Instance penalty: {summary.get('instance_cost', 0.0):g}; "
            f"detection pixel: {summary.get('detection_pixel_nm', 0.0):g} nm",
        ]
        if sizes:
            lines.append("Fitted cells (width × length nm): " + sizes)
        return lines

    def _build_report(self) -> QTextEdit:
        r = self._result
        s = r["summary"]
        b = r.get("bootstrap", {})
        cfg = r["config"]
        # Name the null that was actually run: the 2-D ROI mode randomizes
        # inside the drawn outline and models no membrane surface, so calling
        # it a surface null in its own report would misdescribe the result.
        null_name = ("ROI-conditioned 2-D null" if r.get("is_2d")
                     else "conditional surface null")
        projection = (" Distances are XY projections, so a Z separation is not "
                      "recoverable." if r.get("is_2d") else "")
        lines = [
            "INTERPRETATION",
            f"A positive result supports a short-range population relative to the "
            f"{null_name}. It does not identify pair membership and does "
            f"not estimate a molecular dimer distance.{projection}",
            "",
            "PRIMARY RESULT",
            f"Band: {cfg.short_range_lo_nm:g}–{cfg.short_range_hi_nm:g} nm",
            f"Observed pairs: {s['band_observed_pairs']:,}",
            f"Null expectation: {s['band_null_mean_pairs']:.1f} ± "
            f"{s['band_null_sd_pairs']:.1f}",
            f"Observed/null ratio: {s['band_ratio']:.3f}  "
            f"(null {s.get('null_band_ratio_mean', float('nan')):.3f} ± "
            f"{s.get('null_band_ratio_sd', float('nan')):.3f}, "
            f"{s.get('band_ratio_z', float('nan')):.1f}σ) "
            f"at stratum {cfg.null_stratum_sites}",
            f"Empirical one-sided p: {self._p_text(s['band_p'], cfg.null_replicates)}  "
            "— censored at this resolution and anti-conservative; quote the ratio "
            "and its σ instead",
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
        lines += self._rod_report_lines()
        lines += self._shape_report_lines()
        span = r.get("centroid_sensitivity_range_nm") or []
        if len(span) == 2 and np.isfinite(span[0]) and np.isfinite(span[1]):
            lines += [
                "",
                "UNCERTAINTY ON THE EXCESS LOCATION",
                f"Sensitivity spread (preferred): {span[0]:.2f}–{span[1]:.2f} nm "
                "across the audited parameter choices",
            ]
        if b.get("available"):
            ci = b.get("centroid_ci95_nm", [np.nan, np.nan])
            ri = b.get("band_ratio_ci95", [np.nan, np.nan])
            note = (" — only %d component(s); narrower than the true "
                    "between-cell variance" % b.get("n_components", 0)
                    if b.get("narrow_ci_warning") else "")
            lines += [
                "",
                "COMPONENT BOOTSTRAP",
                f"Positive-excess centroid 95% interval: {ci[0]:.2f}–{ci[1]:.2f} nm{note}",
                f"Band-ratio 95% interval: {ri[0]:.2f}–{ri[1]:.2f}",
            ]
        else:
            lines += ["", "COMPONENT BOOTSTRAP", f"Unavailable: {b.get('reason', 'n/a')}"]

        profile = r.get("stratum_profile") or {}
        if profile.get("rows"):
            lo, hi = profile.get("band_ratio_range", [np.nan, np.nan])
            clo, chi = profile.get("centroid_range_nm", [np.nan, np.nan])
            verdict = ("the ratio is conditional on this scale and must be quoted "
                       "with it; the excess location is the stable descriptor"
                       if profile.get("band_ratio_is_stratum_conditional")
                       else "the ratio is stable across this scale")
            lines += [
                "", "NULL STRATIFICATION PROFILE",
                "Varying the randomization scale alone, with the inferred sites and "
                "components held fixed:",
                f"ratio {lo:.2f}–{hi:.2f} ({profile.get('band_ratio_spread', float('nan')):.1f}×), "
                f"excess centroid {clo:.2f}–{chi:.2f} nm — {verdict}.",
                "stratum sites | ratio | ratio σ | excess centroid nm",
            ]
            for row in profile["rows"]:
                lines.append(
                    f"{row['null_stratum_sites']:13d} | {row['band_ratio']:5.2f} | "
                    f"{row['band_ratio_z']:7.1f} | "
                    f"{row['positive_excess_centroid_nm']:7.2f}")

        sensitivity = r.get("sensitivity") or []
        if sensitivity:
            lines += [
                "", "SENSITIVITY AUDIT",
                f"Primary claim passes {r.get('sensitivity_calibrated_passes', 0)} of "
                f"{r.get('sensitivity_valid_variants', 0)} valid variants on the "
                f"calibrated criterion (ratio > 1 and ≥ {cfg.calibrated_ratio_z:g}σ); "
                f"robust = {bool(r.get('robust_short_range_excess_calibrated'))}.",
                f"On the nominal p ≤ 0.05 criterion: "
                f"{r.get('sensitivity_passes', 0)} of "
                f"{r.get('sensitivity_valid_variants', 0)}; "
                f"robust = {bool(r.get('robust_short_range_excess'))}.",
                "merge nm | link nm | stratum sites | sites used | components | ratio | ratio σ | p | excess centroid nm",
            ]
            for row in sensitivity:
                lines.append(
                    f"{row['site_merge_nm']:7.1f} | {row['cell_link_nm']:7.0f} | "
                    f"{row['null_stratum_sites']:14d} | "
                    f"{row['n_sites_used']:10d} | {row['n_components']:10d} | "
                    f"{row['band_ratio']:5.2f} | {row.get('band_ratio_z', float('nan')):7.1f} | "
                    f"{row['band_p']:.3g} | "
                    f"{row['positive_excess_centroid_nm']:7.2f}")
        lines += ["", "LIMITATIONS"] + [f"• {item}" for item in r.get("limitations", [])]
        report = QTextEdit()
        report.setReadOnly(True)
        report.setPlainText("\n".join(lines))
        return report

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        """Stop the pending redraws, then retire this window's plots."""
        for name in ("_band_timer", "_label_timer"):
            timer = getattr(self, name, None)
            if timer is not None:
                timer.stop()
        from .qt_lifecycle import dispose_plot_widgets

        dispose_plot_widgets(self)
        super().closeEvent(event)
