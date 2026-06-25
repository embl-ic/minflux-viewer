"""Parameter dialog + result window for the HlyB sub-unit pair analysis
(Analyze › Clustering › HlyB subunit pair analysis).

``HlyBClusteringDialog`` is a small modal parameter picker; ``HlyBResultWindow``
is a modeless result window with a 3-D scatter of the detected sub-units / HlyB
structures / measured pair links and a histogram of all pair distances.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..analysis.hlyb_clustering import HlyBConfig

# Cap GL text labels so a busy field of view stays legible.
_MAX_DISTANCE_LABELS = 60
_MAX_STRUCTURE_LABELS = 200
_MAX_RAW_POINTS = 100_000


class HlyBClusteringDialog(QDialog):
    """Modal parameter picker for the HlyB sub-unit pair analysis."""

    def __init__(self, parent=None, *, defaults: HlyBConfig | None = None, mode: str = "3D") -> None:
        super().__init__(parent)
        mode_text = str(mode).upper()
        if "TEMPLATE" in mode_text:
            self._mode = "TEMPLATE3D"
            mode_label = "Template matching (3D)"
        else:
            self._mode = "2D" if mode_text == "2D" else "3D"
            mode_label = self._mode
        self.setWindowTitle(f"HlyB Subunit Pair Analysis ({mode_label})")
        self.setModal(True)
        d = defaults or HlyBConfig()

        root = QVBoxLayout(self)
        intro_text = (
            "Detect protein sub-units from MINFLUX traces, cluster them into HlyB "
            "structures, and measure the sub-unit pair distances within each structure."
        )
        if self._mode == "2D":
            intro_text += (
                " A per-E.coli mask is built and eroded inward; traces in that border "
                "margin (where 2-D distances are unreliable) are excluded."
            )
        elif self._mode == "TEMPLATE3D":
            intro_text += (
                " Candidate sub-units are then assigned to partial six-site 3-D HlyB "
                "distance templates, allowing missing sub-units while rejecting over-merged groups."
            )
        intro = QLabel(intro_text)
        intro.setWordWrap(True)
        root.addWidget(intro)

        form = QFormLayout()
        self._min_loc = QSpinBox()
        self._min_loc.setRange(1, 100000)
        self._min_loc.setValue(int(d.min_loc_per_trace))
        form.addRow("Min loc per trace:", self._min_loc)

        self._zscale = None
        self._border = None
        self._mask_px = None
        if self._mode in {"3D", "TEMPLATE3D"}:
            self._zscale = self._dspin(0.1, 2.0, d.z_scaling_factor, 4, 0.01, "")
            form.addRow("Z scaling (RIMF):", self._zscale)
        else:
            self._border = self._dspin(0.0, 2000.0, d.border_size_nm, 1, 10.0, " nm")
            self._border.setToolTip("Shrink each cell mask inward by this amount; "
                                    "traces in the margin are excluded.")
            form.addRow("Border erosion:", self._border)
            self._mask_px = self._dspin(1.0, 200.0, d.mask_pixel_size_nm, 1, 1.0, " nm")
            self._mask_px.setToolTip("Pixel size of the coarse histogram used to build the cell mask.")
            form.addRow("Mask pixel size:", self._mask_px)

        self._pixel = self._dspin(0.25, 50.0, d.unit_render_pixel_size, 2, 0.5, " nm")
        form.addRow("Unit render pixel size:", self._pixel)

        self._unit_size = self._dspin(0.0, 500.0, d.basic_unit_size_nm, 2, 1.0, " nm")
        self._unit_size.setSpecialValueText("auto")
        self._unit_size.setToolTip("Basic sub-unit diameter (Dunit). 0 = auto from trace size.")
        form.addRow("Basic unit size:", self._unit_size)

        self._min_units = QSpinBox()
        self._min_units.setRange(1, 100)
        self._min_units.setValue(int(d.min_unit_count_per_HlyB))
        if self._mode == "TEMPLATE3D":
            self._min_units.setToolTip("Minimum observed sub-units for an accepted partial HlyB match.")
            form.addRow("Min observed sub-units:", self._min_units)
        else:
            self._min_units.setToolTip("DBSCAN minPts: smallest number of sub-units forming an HlyB.")
            form.addRow("Min units per HlyB:", self._min_units)

        self._d1a1b = self._dspin(1.0, 200.0, d.diameter_distance_1a1b_nm, 1, 1.0, " nm")
        self._d1a2a = self._dspin(1.0, 200.0, d.diameter_distance_1a2a_nm, 1, 1.0, " nm")
        self._d1b2b = self._dspin(1.0, 200.0, d.diameter_distance_1b2b_nm, 1, 1.0, " nm")
        self._d1b2b.setToolTip("Sets the HlyB clustering radius (2·d/√3).")
        form.addRow("Distance 1a–1b (prior):", self._d1a1b)
        form.addRow("Distance 1a–2a (prior):", self._d1a2a)
        form.addRow("Distance 1b–2b (prior):", self._d1b2b)

        self._neighbor = None
        self._cross = None
        self._template_tol = None
        self._template_rms = None
        self._template_max_resid = None
        self._template_match_frac = None
        if self._mode == "TEMPLATE3D":
            self._neighbor = self._dspin(1.0, 200.0, d.neighboring_domain_distance_nm, 1, 1.0, " nm")
            self._cross = self._dspin(1.0, 200.0, d.cross_domain_distance_nm, 1, 1.0, " nm")
            form.addRow("Neighboring domain prior:", self._neighbor)
            form.addRow("Cross-domain prior:", self._cross)

            self._template_tol = self._dspin(0.0, 50.0, d.model_pair_tolerance_nm, 1, 0.5, " nm")
            self._template_tol.setSpecialValueText("auto")
            self._template_tol.setToolTip("Pair-distance residual tolerance. 0 = max(5 nm, auto Dunit).")
            form.addRow("Template pair tolerance:", self._template_tol)

            self._template_rms = self._dspin(0.0, 50.0, d.model_rms_threshold_nm, 1, 0.5, " nm")
            self._template_rms.setSpecialValueText("auto")
            self._template_rms.setToolTip("Accepted-template RMS residual. 0 = 0.8 × pair tolerance.")
            form.addRow("Template RMS threshold:", self._template_rms)

            self._template_max_resid = self._dspin(0.0, 80.0, d.model_max_residual_nm, 1, 0.5, " nm")
            self._template_max_resid.setSpecialValueText("auto")
            self._template_max_resid.setToolTip("Maximum single-pair residual. 0 = 1.6 × pair tolerance.")
            form.addRow("Max pair residual:", self._template_max_resid)

            self._template_match_frac = self._dspin(0.0, 1.0, d.min_pair_match_fraction, 2, 0.05, "")
            self._template_match_frac.setToolTip("Fraction of pairs that must be within tolerance.")
            form.addRow("Min match fraction:", self._template_match_frac)
        root.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Analyze")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    @staticmethod
    def _dspin(lo, hi, val, decimals, step, suffix) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setSingleStep(step)
        s.setValue(float(val))
        if suffix:
            s.setSuffix(suffix)
        return s

    def config(self) -> HlyBConfig:
        defaults = HlyBConfig()
        return HlyBConfig(
            min_loc_per_trace=int(self._min_loc.value()),
            z_scaling_factor=(float(self._zscale.value()) if self._zscale is not None else 1.0),
            unit_render_pixel_size=float(self._pixel.value()),
            basic_unit_size_nm=float(self._unit_size.value()),
            min_unit_count_per_HlyB=int(self._min_units.value()),
            diameter_distance_1a1b_nm=float(self._d1a1b.value()),
            diameter_distance_1a2a_nm=float(self._d1a2a.value()),
            diameter_distance_1b2b_nm=float(self._d1b2b.value()),
            neighboring_domain_distance_nm=(
                float(self._neighbor.value()) if self._neighbor is not None
                else defaults.neighboring_domain_distance_nm
            ),
            cross_domain_distance_nm=(
                float(self._cross.value()) if self._cross is not None
                else defaults.cross_domain_distance_nm
            ),
            min_observed_subunits_per_HlyB=(
                int(self._min_units.value()) if self._mode == "TEMPLATE3D"
                else defaults.min_observed_subunits_per_HlyB
            ),
            model_pair_tolerance_nm=(
                float(self._template_tol.value()) if self._template_tol is not None
                else defaults.model_pair_tolerance_nm
            ),
            model_rms_threshold_nm=(
                float(self._template_rms.value()) if self._template_rms is not None
                else defaults.model_rms_threshold_nm
            ),
            model_max_residual_nm=(
                float(self._template_max_resid.value()) if self._template_max_resid is not None
                else defaults.model_max_residual_nm
            ),
            min_pair_match_fraction=(
                float(self._template_match_frac.value()) if self._template_match_frac is not None
                else defaults.min_pair_match_fraction
            ),
            border_size_nm=(float(self._border.value()) if self._border is not None else defaults.border_size_nm),
            mask_pixel_size_nm=(float(self._mask_px.value()) if self._mask_px is not None else defaults.mask_pixel_size_nm),
        )


class HlyBResultWindow(QDialog):
    """Modeless result window: 3-D sub-unit/pair scatter + pair-distance histogram."""

    def __init__(self, result: dict, cfg: HlyBConfig, *, title: str = "", owner=None,
                 prefer_2d: bool | None = None) -> None:
        super().__init__(None)
        self._owner = owner
        self._result = result
        self._cfg = cfg
        # 2-D data (E.coli border-removal path) → use the flat XY scatter.
        self._prefer_2d = bool(result.get("is_2d", False)) if prefer_2d is None else bool(prefer_2d)
        self.setWindowTitle(f"HlyB Subunit Pair Analysis — {title}" if title else
                            "HlyB Subunit Pair Analysis")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(1100, 900)

        root = QVBoxLayout(self)
        root.addWidget(self._summary_label())

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._build_scatter())
        splitter.addWidget(self._build_histogram())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

    # -- summary --------------------------------------------------------

    def _summary_label(self) -> QLabel:
        r = self._result
        pd = r["all_pair_distances"]
        med = float(np.median(pd)) if pd.size else float("nan")
        text = ""
        if r.get("is_2d") and "n_total_traces" in r:
            text += (f"Traces: {r['n_total_traces']} total, {r['n_border_traces']} border-excluded, "
                     f"{r['n_traces']} interior   |   ")
        else:
            text += f"Traces: {r['n_traces']}   |   "
        if r.get("template_matching"):
            qc = r.get("match_qc", {})
            text += (
                f"Sub-units: {r['n_subunits']}   |   "
                f"Template matches: {r['n_structures']}   |   Pairs: {pd.size}   |   "
                f"median pair distance: {med:.2f} nm   |   "
                f"unit Ø {r['dunit_nm']:.1f} nm, tolerance {r['model_pair_tolerance_nm']:.1f} nm   |   "
                f"tested {qc.get('n_candidates_tested', 0)}, "
                f"passed {qc.get('n_candidates_passed_thresholds', 0)}, "
                f"overlap-rejected {qc.get('n_overlap_rejected', 0)}"
            )
        else:
            text += (
                f"Sub-units: {r['n_subunits']}   |   "
                f"HlyB structures: {r['n_structures']}   |   Pairs: {pd.size}   |   "
                f"median pair distance: {med:.2f} nm   |   "
                f"unit Ø {r['dunit_nm']:.1f} nm, HlyB radius {r['hlyb_diameter_nm']:.1f} nm"
            )
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lbl.setStyleSheet("font-weight: bold;")
        return lbl

    # -- scatter --------------------------------------------------------

    def _build_scatter(self) -> QWidget:
        if self._prefer_2d:
            return self._build_2d_scatter()
        try:
            return self._build_gl_scatter()
        except Exception as exc:  # noqa: BLE001 - fall back to a flat 2-D scatter
            return self._build_2d_scatter(str(exc))

    def _structure_colors(self) -> np.ndarray:
        """RGBA per sub-unit: structure colour, or gray for unclustered."""
        centers = self._result["subunit_centers"]
        colors = np.tile(np.array([0.6, 0.6, 0.6, 0.85], dtype=np.float32), (centers.shape[0], 1))
        structs = self._result["structures"]
        for k, st in enumerate(structs):
            c = pg.intColor(k, hues=max(len(structs), 9))
            colors[st["unit_indices"]] = [c.red() / 255.0, c.green() / 255.0, c.blue() / 255.0, 1.0]
        return colors

    def _pair_segments(self) -> tuple[np.ndarray, list[tuple[np.ndarray, float]]]:
        """Flat (2P,3) line-segment endpoints + (midpoint, distance) per pair."""
        seg = []
        labels = []
        for st in self._result["structures"]:
            uc = st["unit_centers"]
            pair_idx = st.get("pair_index")
            if pair_idx is None:
                pair_idx = np.array([(i, j) for i in range(uc.shape[0] - 1) for j in range(i + 1, uc.shape[0])])
            pair_dist = st.get("pair_distances")
            for p, (i, j) in enumerate(np.asarray(pair_idx, dtype=int)):
                p1, p2 = uc[i], uc[j]
                seg.append(p1)
                seg.append(p2)
                dist = float(pair_dist[p]) if pair_dist is not None and p < len(pair_dist) else float(np.linalg.norm(p2 - p1))
                labels.append(((p1 + p2) / 2.0, dist))
        return (np.array(seg, dtype=np.float64) if seg else np.empty((0, 3))), labels

    def _build_gl_scatter(self) -> QWidget:
        import pyqtgraph.opengl as gl

        points = np.asarray(self._result["points_nm"], dtype=np.float64)
        centers = np.asarray(self._result["subunit_centers"], dtype=np.float64)
        seg, labels = self._pair_segments()

        anchor = centers.mean(axis=0) if centers.shape[0] else (
            points.mean(axis=0) if points.shape[0] else np.zeros(3))

        view = gl.GLViewWidget()
        view.setBackgroundColor("k")

        if points.shape[0]:
            raw = points
            if raw.shape[0] > _MAX_RAW_POINTS:
                step = int(np.ceil(raw.shape[0] / _MAX_RAW_POINTS))
                raw = raw[::step]
            view.addItem(gl.GLScatterPlotItem(
                pos=(raw - anchor).astype(np.float32),
                color=(0.4, 0.4, 0.4, 0.5), size=2.0, pxMode=True))

        if centers.shape[0]:
            view.addItem(gl.GLScatterPlotItem(
                pos=(centers - anchor).astype(np.float32),
                color=self._structure_colors(), size=10.0, pxMode=True))

        if seg.shape[0]:
            view.addItem(gl.GLLinePlotItem(
                pos=(seg - anchor).astype(np.float32),
                color=(0.3, 0.6, 1.0, 0.9), width=1.5, mode="lines", antialias=False))

        if len(labels) <= _MAX_DISTANCE_LABELS:
            for mid, dist in labels:
                self._gl_text(gl, view, f"{dist:.1f}", mid - anchor, (210, 210, 210, 255))
        structs = self._result["structures"]
        if len(structs) <= _MAX_STRUCTURE_LABELS:
            for k, st in enumerate(structs):
                c = pg.intColor(k, hues=max(len(structs), 9))
                prefix = "T" if self._result.get("template_matching") else "HlyB"
                self._gl_text(gl, view, f"{prefix} {st['id']}", st["centroid"] - anchor,
                              (c.red(), c.green(), c.blue(), 255))

        ref = centers if centers.shape[0] else points
        if ref.shape[0]:
            span = float(np.ptp(ref - anchor)) or 100.0
            view.setCameraPosition(distance=span * 1.6)
            grid = gl.GLGridItem()
            grid.setSize(span * 1.5, span * 1.5)
            grid.setSpacing(max(span / 10.0, 1.0), max(span / 10.0, 1.0))
            view.addItem(grid)
        return view

    @staticmethod
    def _gl_text(gl, view, text, pos, color) -> None:
        try:
            view.addItem(gl.GLTextItem(pos=np.asarray(pos, dtype=float), text=text, color=color))
        except Exception:
            pass

    def _build_2d_scatter(self, reason: str = "") -> QWidget:
        plot = pg.PlotWidget(background="w")
        plot.setAspectLocked(True)
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.setLabel("bottom", "x", units="nm")
        plot.setLabel("left", "y", units="nm")
        if reason:
            plot.setTitle(f"3-D view unavailable ({reason}); showing XY")
        elif self._result.get("is_2d"):
            plot.setTitle("Interior (gray) vs border-excluded (orange) localizations; "
                          "sub-unit centres and measured pairs")
        # Border-excluded localizations (2-D path), drawn faint orange beneath.
        border = self._result.get("border_points_nm")
        if border is not None:
            border = np.asarray(border, dtype=np.float64)
            if border.shape[0]:
                plot.addItem(pg.ScatterPlotItem(
                    border[:, 0], border[:, 1], size=2, pen=None,
                    brush=pg.mkBrush(230, 160, 40, 90)))
        points = np.asarray(self._result["points_nm"], dtype=np.float64)
        centers = np.asarray(self._result["subunit_centers"], dtype=np.float64)
        if points.shape[0]:
            plot.addItem(pg.ScatterPlotItem(
                points[:, 0], points[:, 1], size=2, pen=None,
                brush=pg.mkBrush(150, 150, 150, 120)))
        seg, labels = self._pair_segments()
        for a in range(0, seg.shape[0], 2):
            plot.plot(seg[a:a + 2, 0], seg[a:a + 2, 1], pen=pg.mkPen((40, 110, 230), width=1.2))
        if len(labels) <= _MAX_DISTANCE_LABELS:
            for mid, dist in labels:
                t = pg.TextItem(f"{dist:.1f}", color=(40, 40, 40), anchor=(0.5, 0.5))
                t.setPos(float(mid[0]), float(mid[1]))
                plot.addItem(t)
        if centers.shape[0]:
            colors = self._structure_colors()
            brushes = [pg.mkBrush(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), 255) for c in colors]
            plot.addItem(pg.ScatterPlotItem(
                centers[:, 0], centers[:, 1], size=10, brush=brushes, pen=pg.mkPen("k", width=0.5)))
        return plot

    # -- histogram ------------------------------------------------------

    def _build_histogram(self) -> QWidget:
        plot = pg.PlotWidget(background="w")
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.setLabel("bottom", "sub-unit pair distance", units="nm")
        plot.setLabel("left", "count")
        plot.setTitle("Distance among sub-unit pairs (within each HlyB structure)")
        pd = self._result["all_pair_distances"]
        if pd.size:
            counts, edges = np.histogram(pd, bins="auto")
            width = float(edges[1] - edges[0]) if edges.size > 1 else 1.0
            plot.addItem(pg.BarGraphItem(
                x=0.5 * (edges[:-1] + edges[1:]), height=counts, width=width * 0.95,
                brush=pg.mkBrush(80, 130, 200, 180), pen=pg.mkPen(60, 100, 160)))
            med = float(np.median(pd))
            plot.addItem(pg.InfiniteLine(
                pos=med, angle=90, pen=pg.mkPen("r", width=2, style=Qt.PenStyle.DashLine),
                label=f"median {med:.2f} nm",
                labelOpts={"color": "r", "position": 0.9}))
            if self._result.get("template_matching") and "model" in self._result:
                priors = [
                    (f"{v:g} nm", float(v), pg.intColor(i, hues=7))
                    for i, v in enumerate(self._result["model"]["prior_distances_nm"])
                ]
                prior_lines = [
                    (label, val, (color.red(), color.green(), color.blue()))
                    for label, val, color in priors
                ]
            else:
                prior_lines = [
                    ("1a-1b", self._cfg.diameter_distance_1a1b_nm, (0, 150, 0)),
                    ("1a-2a", self._cfg.diameter_distance_1a2a_nm, (200, 120, 0)),
                    ("1b-2b", self._cfg.diameter_distance_1b2b_nm, (150, 0, 150)),
                ]
            # HlyB sub-unit distance priors for reference.
            for label, val, color in prior_lines:
                plot.addItem(pg.InfiniteLine(
                    pos=float(val), angle=90,
                    pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DotLine),
                    label=f"{label} {val:g}", labelOpts={"color": color, "position": 0.7}))
        else:
            plot.setTitle("No sub-unit pairs found")
        return plot
