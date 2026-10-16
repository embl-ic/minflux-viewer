"""Result dialogs and launchers for numerical analysis."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..analysis import localization_precision as _calculation
from ..analysis.localization_precision import (
    _SQRT2,
    CRLB_DEFAULT_L_NM,
    CRLB_DEFAULT_SIGMA_Q_NM,
    MIN_LOCS_PER_TRACE,
    CRLBResult,
    FRCResult,
)
from ..core.mfx_sequence import DEFAULT_FINAL_GEO_FACTOR, PATTERN_SCALE_NM


def run_crlb(parent, state) -> None:
    """Compute the MINFLUX CRB precision for the active dataset, show a dialog."""
    ds = state.active_dataset
    if ds is None:
        return
    try:
        inputs = _calculation.crlb_inputs_for_dataset(ds, prefs=getattr(state, "prefs", None))
    except ValueError as exc:
        _show_info_dialog(parent, "CRLB — invalid inputs", str(exc))
        return
    if inputs is None or inputs["xy"]["eco"] is None:
        _show_info_dialog(
            parent, "CRLB — Cramér-Rao bound",
            "Dataset has no photon-count attribute (<b>eco</b>), which the "
            "MINFLUX Cramér-Rao bound requires. Load a MINFLUX dataset that "
            "carries effective counts.")
        return

    try:
        result = _calculation.crlb_from_inputs(inputs)
        measured = _calculation.crlb_matched_comparison(result, inputs)
    except ValueError as exc:
        _show_info_dialog(parent, "CRLB — invalid inputs", str(exc))
        return
    _try_journal(state, _calculation.crlb_log_line(result, measured))

    from .modeless import show_modeless
    show_modeless(CRLBResultDialog(result, inputs, measured=measured), parent)


def run_frc(parent, state) -> None:
    """Compute the FRC resolution of the active dataset and show a dialog."""
    ds = state.active_dataset
    if ds is None:
        return
    role, reason = _calculation._precision_role(ds, getattr(state, "prefs", None))
    if role == "Tracking":
        _show_info_dialog(
            parent,
            "FRC — unavailable for a tracking channel",
            "FRC compares two independent reconstructions of the same static "
            "structure. A moving trajectory does not satisfy that assumption, "
            f"so no FRC value was computed.<br><br>Role basis: {reason}.",
        )
        return
    from ..core.dataset_kind import missing_reason, require
    miss = require(ds, "loc")
    if miss:
        _show_info_dialog(parent, "FRC — Fourier Ring Correlation",
                          f"This analysis requires {missing_reason(miss)}.")
        return

    from ..core.loader import attr_values_1d
    loc_x = attr_values_1d(ds, "loc_x")
    loc_y = attr_values_1d(ds, "loc_y")

    x = np.asarray(loc_x, dtype=float) * 1.0e9    # m → nm
    y = np.asarray(loc_y, dtype=float) * 1.0e9
    tid = ds.attr.get("tid")
    tid = None if tid is None else np.asarray(tid)
    ftr = np.asarray(ds.filter_mask, dtype=bool)
    if ftr.shape[0] == x.shape[0]:
        x, y = x[ftr], y[ftr]
        if tid is not None:
            tid = tid[ftr]

    with state.task(f"Computing FRC of dataset {ds.name}…") as t:
        result = _calculation.frc_resolution(
            x, y, tid, progress=lambda done, total: t.update(done / total)
        )

    if np.isfinite(result.resolution_nm):
        _try_journal(
            state,
            f"Localization precision (FRC): resolution = "
            f"{result.resolution_nm:.2f} nm (1/7 threshold, {result.mode}, "
            f"{result.n_points:,} points, pixel {result.pixel_size_nm:.2f} nm)."
        )

    # Modeless result window (project convention for analysis output).
    from .modeless import show_modeless
    show_modeless(FRCResultDialog(result, x, y, tid), parent)


def run_stddev_per_trace(parent, state) -> None:
    """Compute σ per trace for the active dataset and show a results dialog."""
    ds = state.active_dataset
    if ds is None:
        return
    role, reason = _calculation._precision_role(ds, getattr(state, "prefs", None))
    if role == "Tracking":
        _show_info_dialog(
            parent,
            "StdDev per trace — unavailable for a tracking channel",
            "For a moving emitter, scatter about the trace mean is dominated "
            "by the trajectory's excursion and is not localization precision. "
            "Use the tracking MSD result's explicitly labelled apparent "
            f"short-lag intercept instead.<br><br>Role basis: {reason}.",
        )
        return
    from ..core.dataset_kind import missing_reason, require
    miss = require(ds, "loc", "traces")
    if miss:
        _show_info_dialog(
            parent, "Localization Precision — StdDev per trace",
            f"This analysis requires {missing_reason(miss)}.")
        return

    from ..core.loader import attr_values_1d
    loc_x = np.asarray(attr_values_1d(ds, "loc_x"))
    loc_y = np.asarray(attr_values_1d(ds, "loc_y"))
    # Raw z — no Z scaling factor applied. The precision is a property of the measurement,
    # so it follows Ostersehlt et al. and uses the uncorrected coordinate.
    _z = attr_values_1d(ds, "loc_z")
    loc_z = np.zeros_like(loc_x) if _z is None else np.asarray(_z)
    _tid = attr_values_1d(ds, "tid")
    tid   = np.arange(loc_x.size) if _tid is None else np.asarray(_tid)
    ftr   = np.asarray(ds.filter_mask, dtype=bool)

    loc_xyz = np.column_stack([loc_x[ftr], loc_y[ftr], loc_z[ftr]])
    tid_f   = tid[ftr]

    result = _calculation.stddev_per_trace(loc_xyz, tid_f)

    # Stash the per-trace σ on the dataset for re-use
    try:
        ds.attr["sigma_per_trace_nm"] = result["per_trace_sigma_xyz"]
        ds.attr["sigma_trace_ids"]    = result["trace_ids"]
        ds.derived["sigma_per_trace_nm"] = result["per_trace_sigma_xyz"]
        ds.derived["sigma_trace_ids"]    = result["trace_ids"]
    except Exception:
        pass

    _try_journal(
        state,
        f"Localization precision (StdDev per trace): "
        f"combined (n-weighted) sigma_r = {result['combined_sigma_r']:.2f} nm, "
        f"sigma_z = {result['combined_sigma_xyz'][2]:.2f} nm "
        f"over {result['n_traces_used']:,} of {result['n_traces_total']:,} traces "
        f"(>={MIN_LOCS_PER_TRACE} loc/trace, raw z)."
    )

    # Modeless result window (project convention for analysis output).
    from .modeless import show_modeless
    show_modeless(StdDevResultDialog(result), parent)


def _precision_color(name: str) -> QColor:
    """One plot color from COLOR ▸ Components ▸ Localization precision."""
    from ..colors import runtime_component_colors

    colors = runtime_component_colors("functions", "Localization precision")
    return QColor(*colors[name])


class StdDevResultDialog(QDialog):
    """Localization-precision report (Schmidt et al.), styled like anisotropy.

    Reports the n-weighted combined precision σ_r (lateral) and σ_z (axial),
    with per-trace σ_r / σ_z distributions and a per-axis X/Y/R(xy)/Z table.
    """

    def __init__(self, result: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Localization Precision — StdDev per trace")
        self.setModal(False)
        self.resize(880, 640)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        n_used  = result["n_traces_used"]
        n_total = result["n_traces_total"]
        med = result["median_sigma_xyz"]
        mean = result["mean_sigma_xyz"]

        if n_used == 0:
            root.addWidget(QLabel(
                f"<b>No traces had at least {MIN_LOCS_PER_TRACE} localizations.</b><br><br>"
                f"Total traces in selection: {n_total:,}<br><br>"
                "Try widening the filter, or increase the number of repeats per trace."
            ))
            bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            bb.rejected.connect(self.reject)
            bb.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.close)
            root.addWidget(bb)
            return

        comb   = result["combined_sigma_xyz"]
        med_r  = result["median_sigma_r"]
        mean_r = result["mean_sigma_r"]
        comb_r = result["combined_sigma_r"]

        info = QLabel(
            f"<b>{n_used:,}</b> of {n_total:,} traces used "
            f"(≥ {MIN_LOCS_PER_TRACE} loc/trace, raw coordinates)"
        )
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(info)

        headline = QLabel(
            "<span style='font-size:18px'><b>localization precision: "
            f"σ<sub>r</sub> = {comb_r:.2f} nm, "
            f"σ<sub>z</sub> = {comb[2]:.2f} nm</b></span>"
            "<span style='color:gray'> &nbsp;(n-weighted)</span>"
        )
        headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        headline.setToolTip(
            "σ_r = sqrt((σ_x² + σ_y²)/2);  reported value = mean over traces of σ/sqrt(n)"
        )
        root.addWidget(headline)

        # Per-axis table — X / Y / R(xy) / Z, with the combined row reported.
        grp = QGroupBox("Localization precision across traces (nm) — combined (n-weighted) is reported")
        grid = QGridLayout(grp)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(4)
        for col, h in enumerate(("", "X", "Y", "R(xy)", "Z")):
            lbl = QLabel(f"<b>{h}</b>")
            lbl.setAlignment(
                Qt.AlignmentFlag.AlignLeft if col == 0 else Qt.AlignmentFlag.AlignCenter
            )
            grid.addWidget(lbl, 0, col)
        rows = (
            ("Median",      [med[0],  med[1],  med_r,  med[2]],  False),
            ("Mean",        [mean[0], mean[1], mean_r, mean[2]], False),
            ("Combined ★",  [comb[0], comb[1], comb_r, comb[2]], True),
        )
        for r, (row_lbl, vals, bold) in enumerate(rows, start=1):
            name = QLabel(f"<b>{row_lbl}:</b>" if bold else f"{row_lbl}:")
            name.setAlignment(Qt.AlignmentFlag.AlignLeft)
            grid.addWidget(name, r, 0)
            for col, v in enumerate(vals):
                t = f"{v:.2f} nm"
                cell = QLabel(f"<b>{t}</b>" if bold else t)
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                grid.addWidget(cell, r, col + 1)
        for col in range(1, 5):
            grid.setColumnStretch(col, 1)
        root.addWidget(grp)

        # Per-trace distributions — σ_r (lateral) and σ_z (axial), median marked.
        try:
            from .trace_analysis_dialogs import _value_hist_plot
            hist_group = QGroupBox("Per-trace σ distributions: R(xy) and Z")
            row = QHBoxLayout(hist_group)
            row.setContentsMargins(8, 8, 8, 8)
            row.addWidget(_value_hist_plot(
                "σ<sub>r</sub> per trace (lateral)",
                result["per_trace_sigma_r"], _precision_color("Lateral sigma"),
                marker=float(med_r), marker_label=f"median = {med_r:.2f} nm",
                x_label="σ<sub>r</sub> (nm)", y_label="traces",
            ))
            row.addWidget(_value_hist_plot(
                "σ<sub>z</sub> per trace (axial)",
                result["per_trace_sigma_xyz"][:, 2], _precision_color("Axial sigma"),
                marker=float(med[2]), marker_label=f"median = {med[2]:.2f} nm",
                x_label="σ<sub>z</sub> (nm)", y_label="traces",
            ))
            root.addWidget(hist_group, stretch=1)
        except Exception as exc:
            warn = QLabel(f"Histogram report could not be created: {exc}")
            warn.setStyleSheet("color: #a66;")
            root.addWidget(warn)

        note = QLabel(
            "σ<sub>x</sub>, σ<sub>y</sub>, σ<sub>z</sub> — sample standard deviation "
            "(ddof = 1) of each <b>raw</b> coordinate over the localizations of one "
            "trace (nm)<br>"
            "σ<sub>r</sub> = √( (σ<sub>x</sub>² + σ<sub>y</sub>²) ∕ 2 ) — combined "
            "lateral spread of a trace<br>"
            f"n — localizations in a trace; only traces with n ≥ {MIN_LOCS_PER_TRACE} "
            "are used<br>"
            "reported precision = n-weighted standard error of the mean position, "
            "averaged over the M traces:<br>"
            "&nbsp;&nbsp;σ<sub>r</sub> = ⟨ σ<sub>r</sub> ∕ √n ⟩ , &nbsp; "
            "σ<sub>z</sub> = ⟨ σ<sub>z</sub> ∕ √n ⟩ &nbsp; ( ⟨·⟩ = mean over traces )<br>"
            "z uses the raw coordinate (no Z scaling factor applied)."
            + _references_html(
                "<a href='https://www.nature.com/articles/s41592-022-01577-1'>"
                "Ostersehlt <i>et al.</i>, <i>Nat. Methods</i> 19:1072 (2022)</a>",
            )
        )
        note.setWordWrap(True)
        note.setOpenExternalLinks(True)
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        note.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(note)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.close)
        root.addWidget(bb)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        """Retire this window's pyqtgraph plots before Qt deletes them."""
        from .qt_lifecycle import dispose_plot_widgets

        dispose_plot_widgets(self)
        super().closeEvent(event)


def _frc_curve_plot(result: FRCResult):
    """Anisotropy-style panel of the FRC curve vs spatial frequency."""
    import pyqtgraph as pg

    pw = pg.PlotWidget(background="#222")
    pw.setMinimumHeight(320)
    pw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    pw.showGrid(x=True, y=True, alpha=0.22)
    pw.setTitle("Fourier Ring Correlation")
    pw.setLabel("bottom", "spatial frequency q (nm<sup>-1</sup>)")
    pw.setLabel("left", "FRC")
    if result.freq_inv_nm.size < 2:
        return pw

    # Skip the DC bin (FRC ≈ 1 by construction).
    freq = result.freq_inv_nm[1:]
    frc = result.frc_mean[1:]

    # ± std band across the random splits.
    if result.frc_std.size == result.frc_mean.size and result.n_repeats > 1:
        std = result.frc_std[1:]
        hi = pw.plot(freq, frc + std, pen=None)
        lo = pw.plot(freq, frc - std, pen=None)
        band = pg.FillBetweenItem(hi, lo, brush=pg.mkBrush(74, 144, 255, 45))
        pw.addItem(band)

    pw.plot(freq, frc, pen=pg.mkPen(_precision_color("FRC curve"), width=2.0))

    # 1/7 threshold (horizontal).
    thr = pg.InfiniteLine(
        pos=result.threshold, angle=0,
        pen=pg.mkPen("#bbbbbb", width=1.3, style=Qt.PenStyle.DashLine),
        label=f"1/7 = {result.threshold:.3f}",
        labelOpts={"position": 0.06, "color": "#bbbbbb", "fill": (20, 20, 20, 200)},
    )
    pw.addItem(thr)

    # Resolution crossing (vertical).
    if np.isfinite(result.crossing_freq) and np.isfinite(result.resolution_nm):
        cl = pg.InfiniteLine(
            pos=result.crossing_freq, angle=90,
            pen=pg.mkPen(_precision_color("FRC resolution"), width=1.7, style=Qt.PenStyle.DashLine),
            label=f"resolution = {result.resolution_nm:.2f} nm",
            labelOpts={"position": 0.9, "color": _precision_color("FRC resolution"), "fill": (20, 20, 20, 220)},
        )
        pw.addItem(cl)

    pw.setYRange(-0.15, 1.05, padding=0)
    pw.setXRange(0.0, float(result.freq_inv_nm[-1]), padding=0)
    return pw


class FRCResultDialog(QDialog):
    """Fourier Ring Correlation resolution report, anisotropy-plugin style.

    Holds the source 2-D coordinates so the user can recompute with a
    different super-resolution pixel size from the dialog.
    """

    def __init__(self, result: FRCResult, x_nm: np.ndarray, y_nm: np.ndarray,
                 tid: np.ndarray | None = None, parent=None) -> None:
        super().__init__(parent)
        self._x, self._y, self._tid = x_nm, y_nm, tid
        self.setWindowTitle("Localization Precision — FRC resolution")
        self.setModal(False)
        self.resize(820, 700)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        self._root = root
        self._build(result)

    # -- (re)build the body for a freshly computed result -----------------
    def _build(self, result: FRCResult) -> None:
        # Clear any previously built widgets (on recompute), sub-layouts included.
        _clear_layout(self._root)
        root = self._root
        res = result.resolution_nm

        if result.mode == "combined":
            pts_txt = (f"<b>{result.n_points:,}</b> combined positions "
                       f"(from {result.n_locs:,} localizations)")
            mode_txt = "per-trace combined"
        else:
            pts_txt = f"<b>{result.n_locs:,}</b> localizations"
            mode_txt = "per-localization"
        info = QLabel(
            f"{pts_txt} &nbsp;|&nbsp; {result.n_repeats}× averaged "
            f"&nbsp;|&nbsp; {result.image_px}² px @ {result.pixel_size_nm:.2f} nm/px"
        )
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(info)

        headline = QLabel(
            f"<span style='font-size:18px'><b>FRC resolution = {_res(res)}</b></span>"
            f"<span style='color:gray'> &nbsp;({mode_txt}, 1/7 threshold)</span>"
        )
        headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(headline)

        if result.status:
            warn = QLabel(result.status)
            warn.setWordWrap(True)
            warn.setAlignment(Qt.AlignmentFlag.AlignCenter)
            warn.setStyleSheet("color: #d2a24a; font-size: 11px;")
            root.addWidget(warn)

        try:
            grp = QGroupBox("FRC curve")
            box = QVBoxLayout(grp)
            box.setContentsMargins(8, 8, 8, 8)
            box.addWidget(_frc_curve_plot(result))
            root.addWidget(grp, stretch=1)
        except Exception as exc:
            warn = QLabel(f"FRC plot could not be created: {exc}")
            warn.setStyleSheet("color: #a66;")
            root.addWidget(warn)

        note = QLabel(
            "The localizations are split into two independent images by a random "
            "50/50 per-localization assignment, each rendered as a ~1 nm-pixel "
            "histogram.<br>"
            "FRC(q) = Σ<sub>ring</sub> Re{ F<sub>1</sub>(q)·F<sub>2</sub>*(q) } ∕ "
            "√( Σ<sub>ring</sub>|F<sub>1</sub>(q)|² · Σ<sub>ring</sub>|F<sub>2</sub>(q)|² )<br>"
            "where F<sub>1</sub>, F<sub>2</sub> are the 2-D Fourier transforms of the "
            "half-images, summed over each ring of constant spatial frequency q. The "
            "curve is averaged over the random splits, Savitzky-Golay smoothed, and the "
            "high-frequency tail cut at 0.8·q<sub>max</sub>.<br>"
            "resolution = 1 ∕ q<sub>c</sub> , with q<sub>c</sub> the frequency where "
            "FRC first drops below 1/7.<br>"
            "<i>FRC mode</i>: <b>per localization</b> uses every localization and is "
            "kept as the default for continuity, but can be optimistic when emitters "
            "are localized many times; <b>per-trace combined</b> first collapses each "
            "trace to its mean position, so it measures the resolution of the molecule "
            "positions themselves (one point per emitter) — the value Ostersehlt "
            "<i>et al.</i> report beside the per-localization one."
            + _references_html(
                "<a href='https://www.nature.com/articles/s41592-019-0688-0'>Gwosch "
                "<i>et al.</i>, <i>Nat. Methods</i> 17:217 (2020)</a>",
                "<a href='https://www.nature.com/articles/s41592-022-01577-1'>Ostersehlt "
                "<i>et al.</i>, <i>Nat. Methods</i> 19:1072 (2022)</a>",
            )
        )
        note.setWordWrap(True)
        note.setOpenExternalLinks(True)
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        note.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(note)

        # -- controls: FRC mode + editable pixel size + recompute ---------
        btn_row = QHBoxLayout()
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #6c6;")
        btn_row.addWidget(self._status_lbl)
        btn_row.addStretch()
        btn_row.addWidget(QLabel("FRC mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("per localization (default)", "localization")
        self._mode_combo.addItem("per-trace combined", "combined")
        has_tid = self._tid is not None
        self._mode_combo.model().item(1).setEnabled(has_tid)
        idx = self._mode_combo.findData(result.mode if has_tid else "localization")
        self._mode_combo.setCurrentIndex(max(0, idx))
        self._mode_combo.setToolTip(
            "Choose the point set used for FRC. Per localization is the default; "
            "per-trace combined uses one mean position per trace and needs trace IDs.")
        btn_row.addWidget(self._mode_combo)
        btn_row.addSpacing(8)
        btn_row.addWidget(QLabel("pixel (nm):"))
        self._px_spin = QDoubleSpinBox()
        self._px_spin.setRange(0.1, 1000.0)
        self._px_spin.setDecimals(2)
        self._px_spin.setSingleStep(0.5)
        self._px_spin.setValue(result.pixel_size_nm if result.pixel_size_nm > 0 else 1.0)
        self._px_spin.setToolTip("Super-resolution render pixel size for the FRC images.")
        btn_row.addWidget(self._px_spin)
        recompute = QPushButton("Recompute")
        recompute.setToolTip("Re-run FRC with the chosen mode and pixel size.")
        recompute.clicked.connect(self._recompute)
        btn_row.addWidget(recompute)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    def _recompute(self) -> None:
        pixel = float(self._px_spin.value())
        mode = self._mode_combo.currentData()
        try:
            result = _calculation.frc_resolution(self._x, self._y, self._tid,
                                    mode=mode, pixel_size_nm=pixel)
        except Exception as exc:
            self._status_lbl.setText(f"failed: {exc}")
            return
        self._build(result)
        self._status_lbl.setText(f"recomputed ({result.mode}) @ {pixel:.2f} nm/px")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        """Retire this window's pyqtgraph plots before Qt deletes them."""
        from .qt_lifecycle import dispose_plot_widgets

        dispose_plot_widgets(self)
        super().closeEvent(event)


def _res(val: float) -> str:
    if val is None or not np.isfinite(val):
        return "—"
    return f"{val:.2f} nm"


def _references_html(*refs: str) -> str:
    """Standard reference block for result/UI dialogs.

    Project convention: a dialog's reference section is preceded by one blank
    row and lists each reference on its own line (vertically stacked). Pass the
    fully-formatted reference strings (HTML, including any ``<a>`` links); the
    body text this is appended to must NOT end with a trailing ``<br>``.
    """
    label = "Reference" if len(refs) == 1 else "References"
    items = "<br>".join(refs)
    return f"<br><br><b>{label}</b><br>{items}"


def _clear_layout(layout) -> None:
    """Recursively remove and delete every widget/sub-layout from *layout*.

    A plain ``takeAt``/``setParent(None)`` loop misses widgets nested inside
    child layouts (e.g. a controls ``QHBoxLayout``), so they accumulate on each
    rebuild. This walks sub-layouts too.
    """
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.setParent(None)
            w.deleteLater()
            continue
        child = item.layout()
        if child is not None:
            _clear_layout(child)
            child.deleteLater()


def _pattern_text(pattern: str, centre_ratio: float, counts_centre: bool) -> str:
    """'hexagon + centre', 'square', … for the info line ('' when unknown)."""
    if not pattern:
        return "with centre exposure" if counts_centre else ""
    return f"{pattern} + centre" if (centre_ratio > 0 or counts_centre) else pattern


class CRLBResultDialog(QDialog):
    """MINFLUX Cramér-Rao bound report, anisotropy-plugin style.

    Holds the per-channel inputs (:func:`crlb_inputs_for_dataset`) so the user
    can recompute with a different L / σ_q / L_z — the axial photon count and
    background are kept, not replaced by the lateral ones.
    """

    def __init__(self, result: CRLBResult, inputs: dict, *,
                 measured: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        self._inputs = inputs
        self._num_dim = int(inputs.get("num_dim", 2) or 2)
        # Matched per-trace comparison, or None for tracking/ineligible data.
        self._measured = measured
        self.setWindowTitle("Localization Precision — CRLB (MINFLUX)")
        self.setModal(False)
        self.resize(1120 if self._num_dim >= 3 else 880, 720)

        outer = QVBoxLayout(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget()
        scroll.setWidget(content)
        outer.addWidget(scroll)
        root = QVBoxLayout(content)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        self._root = root
        self._build(result)

    def _build(self, result: CRLBResult) -> None:
        from html import escape

        from .qt_lifecycle import dispose_plot_widgets

        # Recompute replaces the plots; retire their ViewBoxes before deletion.
        dispose_plot_widgets(self)
        _clear_layout(self._root)
        root = self._root

        if result.status:
            warn = QLabel(f"<b>{escape(result.status)}</b>")
            warn.setWordWrap(True)
            root.addWidget(warn)

        n_def = "eco + ecc" if result.counts_centre else "eco"
        if result.has_z and result.per_dimension_photons and np.isfinite(result.median_n_z):
            n_txt = (f"median N<sub>xy</sub> = <b>{result.median_n:.0f}</b>, "
                     f"N<sub>z</sub> = <b>{result.median_n_z:.0f}</b> photons ({n_def})")
        else:
            n_value = f"{result.median_n:.0f}" if np.isfinite(result.median_n) else "unavailable"
            n_txt = f"median N = <b>{n_value}</b> input counts ({n_def})"
        pat = _pattern_text(result.pattern, result.centre_dwell_ratio, result.counts_centre)
        pat_txt = f", {pat}" if pat else ""
        info = QLabel(
            f"<b>{result.n_locs:,}</b> localizations &nbsp;|&nbsp; {n_txt} &nbsp;|&nbsp; "
            f"L = <b>{result.L_nm:.1f}</b> nm ({result.L_source}{pat_txt}) &nbsp;|&nbsp; "
            f"σ<sub>q</sub> = <b>{result.sigma_q_nm:.0f}</b> nm"
        )
        info.setWordWrap(True)
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(info)

        primary = result.primary_sigma
        def bound_value(value):
            return _res(value) if np.isfinite(value) else "unavailable"

        if result.has_z:
            head_value = (f"σ<sub>xy</sub> = {bound_value(primary)}, "
                          f"σ<sub>z</sub> = {bound_value(result.primary_sigma_z)}")
        else:
            head_value = bound_value(primary)
        headline = QLabel(
            f"<span style='font-size:18px'><b>Conditional model bound (median σ): "
            f"{head_value}</b></span>"
        )
        headline.setWordWrap(True)
        headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(headline)
        self._add_validity_report(root, result)

        # Caveats the user must see before reading the numbers.
        caveats = [
            "Centred-emitter model bound, conditional on the input conventions and "
            "background model; not a measured localization precision or image resolution. "
            "σ<sub>xy</sub> is per-coordinate RMS, √[(σ<sub>x</sub>² + σ<sub>y</sub>²)/2], "
            "not a radial displacement uncertainty."
        ]
        if result.count_convention != "detected":
            caveats.append(
                "Input count convention unknown: the displayed numbers assume detected "
                "orbit counts. Do not interpret them as calibrated experimental precision "
                "or automatically restore background counts.")
        ideal_axes = [name for name, status in (
            ("lateral", result.axis_status),
            ("axial", result.axis_status_z if result.has_z else ""))
            if status == "ideal_only"]
        if ideal_axes:
            caveats.append(
                "No usable background estimate for " + " and ".join(ideal_axes)
                + ": only a hypothetical zero-background bound is available; "
                "missing background is not evidence of zero background.")
        refs_used = [r for r in (result.centre_reference,
                                 result.centre_reference_z if result.has_z else "") if r]
        if refs_used:
            caveats.append(
                "The photon iteration does not probe the centre, so the background "
                "rate is the centre exposure of " + " and ".join(dict.fromkeys(refs_used))
                + " — assuming the effective background scales with excitation power. "
                "This is a model assumption, not an independently calibrated background.")
        caveats.extend(escape(str(note)) for note in result.background_notes)
        caveats.extend(escape(str(issue)) for issue in result.issues)
        if "assumed" in result.L_source or (result.has_z and "assumed" in result.L_z_source):
            caveats.append(
                f"The pattern size was not recorded in this file; L = "
                f"{CRLB_DEFAULT_L_NM:.1f} nm (patGeoFactor {DEFAULT_FINAL_GEO_FACTOR} × "
                f"{PATTERN_SCALE_NM:.0f} nm, the final pattern of Abberior's standard "
                "imaging sequences) is assumed. Edit L below if the acquisition used "
                "another sequence — the bound scales linearly with it.")
        if self._num_dim >= 3 and result.has_z:
            caveats.append(
                "3-D: σ<sub>xy</sub> uses the photons, background and pattern of the final "
                "lateral iteration; σ<sub>z</sub> those of the final axial iteration.")
        if result.pattern == "octahedron":
            caveats.append(
                "Octahedral result: isotropic quadratic-minimum approximation; real axial "
                "and lateral optical curvature can differ and is not calibrated here.")
        for text in dict.fromkeys(caveats):
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: #d2a24a; font-size: 11px;")
            root.addWidget(lbl)

        # σ_CRB table — ideal vs background-limited (per channel for 3-D).
        grp = QGroupBox("Conditional Cramér-Rao bounds across localizations  (nm)")
        grid = QGridLayout(grp)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(4)
        for col, h in enumerate(("", "hypothetical zero background", "estimated-background model")):
            lbl = QLabel(f"<b>{h}</b>")
            lbl.setWordWrap(True)
            lbl.setAlignment(Qt.AlignmentFlag.AlignLeft if col == 0
                             else Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(lbl, 0, col)
        if result.has_z:
            rows = (
                ("xy Median ★", result.median_sigma_ideal, result.median_sigma_bg, True),
                ("xy Mean", result.mean_sigma_ideal, result.mean_sigma_bg, False),
                ("z Median ★", result.median_sigma_z_ideal, result.median_sigma_z_bg, True),
                ("z Mean", result.mean_sigma_z_ideal, result.mean_sigma_z_bg, False),
            )
        else:
            rows = (
                ("Median ★", result.median_sigma_ideal, result.median_sigma_bg, True),
                ("Mean", result.mean_sigma_ideal, result.mean_sigma_bg, False),
            )
        for r, (row_lbl, v_ideal, v_bg, bold) in enumerate(rows, start=1):
            name = QLabel(f"<b>{row_lbl}:</b>" if bold else f"{row_lbl}:")
            grid.addWidget(name, r, 0)
            for col, v in enumerate((v_ideal, v_bg), start=1):
                t = _res(v) if np.isfinite(v) else "n/a"
                cell = QLabel(f"<b>{t}</b>" if bold else t)
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                grid.addWidget(cell, r, col)
        for col in (1, 2):
            grid.setColumnStretch(col, 1)
        root.addWidget(grp)

        self._add_matched_comparison(root, result)

        # Distributions — σ_xy [· σ_z for 3-D] · photon count N, in one row.
        try:
            from .trace_analysis_dialogs import _value_hist_plot
            hist_group = QGroupBox("Per-localization distributions")
            row = QHBoxLayout(hist_group)
            row.setContentsMargins(8, 8, 8, 8)
            sigma = result.per_loc_sigma_bg if result.has_background else result.per_loc_sigma_ideal
            if not np.isfinite(primary):
                sigma = np.empty(0)
            row.addWidget(_value_hist_plot(
                "Conditional σ<sub>xy</sub> bound" if np.isfinite(primary)
                else "Lateral bound unavailable", sigma, _precision_color("CRLB lateral"),
                marker=float(primary), marker_label=f"median = {primary:.2f} nm",
                x_label="σ<sub>xy</sub> (nm)", y_label="localizations",
                clip_percentile=99.0,
            ))
            if result.has_z:
                sigma_z = (result.per_loc_sigma_z_bg if result.has_background_z
                           else result.per_loc_sigma_z_ideal)
                z_primary = result.primary_sigma_z
                if not np.isfinite(z_primary):
                    sigma_z = np.empty(0)
                row.addWidget(_value_hist_plot(
                    "Conditional σ<sub>z</sub> bound" if np.isfinite(z_primary)
                    else "Axial bound unavailable", sigma_z, _precision_color("CRLB axial"),
                    marker=float(z_primary), marker_label=f"median = {z_primary:.2f} nm",
                    x_label="σ<sub>z</sub> (nm)", y_label="localizations",
                clip_percentile=99.0,
                ))
            row.addWidget(_value_hist_plot(
                "Input count N per localization", result.per_loc_n_total, _precision_color("Photon count"),
                marker=float(result.median_n), marker_label=f"median = {result.median_n:.0f}",
                x_label="N (photons)", y_label="localizations",
            ))
            root.addWidget(hist_group, stretch=1)
        except Exception as exc:
            warn = QLabel(f"Histogram report could not be created: {exc}")
            warn.setStyleSheet("color: #a66;")
            root.addWidget(warn)

        z_formula = (
            "&nbsp;&nbsp;σ<sub>z</sub> = L<sub>z</sub> ∕ ( 4·√N ) · F &nbsp;&nbsp;(axial, "
            "1-D quadratic minimum; Eq. 29 × the background factor of Eq. 28)<br>"
        ) if result.has_z else ""
        z_param = (
            f"&nbsp;&nbsp;L<sub>z</sub> — axial pattern size &nbsp;({result.L_z_source})<br>"
        ) if result.has_z else ""

        # References describe the model, not an attribution of observed errors.
        refs = [
            "<a href='https://arxiv.org/abs/2410.12427'>Marin &amp; Ries, "
            "<i>arXiv</i>:2410.12427 (2024)</a> (Eq. 44/45 lateral, Eq. 28/29 axial)",
            "<a href='https://doi.org/10.1126/science.aak9913'>Balzarotti "
            "<i>et al.</i>, <i>Science</i> 355:606 (2017)</a>",
            # The parameters below name both of its conventions: L = patGeoFactor
            # × 360 nm and the default σ_q of its analytic 2-D donut.
            "<a href='https://doi.org/10.1038/s41467-025-66952-w'>Marin &amp; Ries, "
            "<i>Nat. Commun.</i> 17:246 (2026)</a> (SimuFLUX: pattern scale, donut width)",
        ]
        if result.donut_envelope and abs(result.geometry_factor - _SQRT2) < 1e-9:
            xy_formula = (
                "&nbsp;&nbsp;σ<sub>xy</sub> = L ∕ √(8N) · ( 1 − L² ∕ (8σ<sub>q</sub>²) )"
                "<sup>−1</sup> · F &nbsp;&nbsp;(lateral; Eq. 45 × the background factor of "
                "Eq. 44)<br>")
        else:
            xy_formula = (
                f"&nbsp;&nbsp;σ<sub>xy</sub> = L · G ∕ (4√N) · F, G = "
                f"{result.geometry_factor:.3f} &nbsp;&nbsp;({result.pattern or 'pattern'}: "
                "G² = K·(L∕2)² ∕ Σ p<sub>axis</sub>² over its exposure positions — the same "
                "Fisher information as Eq. 26/45, photons shared by every axis it "
                "localizes)<br>")
            if result.has_z:
                z_formula = (f"&nbsp;&nbsp;σ<sub>z</sub> = L<sub>z</sub> · G<sub>z</sub> ∕ "
                             f"(4√N) · F, G<sub>z</sub> = {result.geometry_factor_z:.3f}<br>")
        note = QLabel(
            "<b>Formulas</b> (emitter at the pattern centre):<br>"
            + xy_formula
            + z_formula +
            "&nbsp;&nbsp;F = √[ (1 + ρ)(1 + ρ(1 + c)) ] = √[ (1 + 1∕SBR)(1 + w∕SBR) ], "
            "w = 1∕(1 + c) &nbsp;(F = 1 without background)<br>"
            "<b>Parameters</b><br>"
            f"&nbsp;&nbsp;N — input counts assumed to be detected photons over all "
            f"exposures of the iteration = <i>{n_def}</i><br>"
            "&nbsp;&nbsp;ρ — background ÷ signal rate at a pattern position, "
            "b ∕ (<i>efo</i> − b) with the background rate b from the centre exposure "
            "(<i>efc</i>, treated as effective background under the centred-emitter "
            "assumption) — this iteration's, else the "
            "latest earlier one scaled by the laser power (<i>pwrFactor</i>) — else "
            "<i>fbg</i> when background sensing was on; b and <i>efo</i> are the medians of "
            "each trace when trace ids are known<br>"
            "&nbsp;&nbsp;c — centre dwell ÷ whole-orbit dwell (<i>ctrDwellFactor</i>, or "
            "(<i>ecc</i>∕<i>eco</i>) ∕ (<i>efc</i>∕<i>efo</i>) from the counts; 0 without a "
            "centre exposure)<br>"
            f"&nbsp;&nbsp;L — lateral pattern diameter of the final iteration "
            f"&nbsp;({result.L_source}; patGeoFactor × {PATTERN_SCALE_NM:.0f} nm)<br>"
            "&nbsp;&nbsp;σ<sub>q</sub> — donut width of the PSF model; default "
            f"{CRLB_DEFAULT_SIGMA_Q_NM:.1f} nm = SimuFLUX's analytic donut (FWHM 310 nm "
            "∕ √(8 ln 2)) &nbsp;(user input)<br>"
            + z_param +
            "A per-coordinate bound at the pattern centre; motion, offsets and "
            "uncalibrated optics are not modelled."
            + _references_html(*refs)
        )
        note.setWordWrap(True)
        note.setOpenExternalLinks(True)
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        note.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(note)

        self._add_controls(root, result)

    def _add_validity_report(self, root, result: CRLBResult) -> None:
        """Keep missing information distinct from contradictory measurements."""
        from html import escape

        names = {
            "ideal_only": "ideal only — no usable background estimate",
            "background_estimated": "background estimated",
            "background_inconsistent": "background ≥ orbit rate — finite bound unavailable",
            "invalid_input": "invalid input — bound unavailable",
            "unsupported_geometry": "unsupported geometry — bound unavailable",
        }
        axes = [("Lateral", result.axis_status, result.background_source,
                 result.background_coverage, result.invalid_background_fraction)]
        if result.has_z:
            axes.append(("Axial", result.axis_status_z, result.background_source_z,
                         result.background_coverage_z, result.invalid_background_fraction_z))
        lines = []
        for name, status, source, coverage, invalid in axes:
            line = f"<b>{name}:</b> {escape(names.get(status, status))}"
            line += f"; usable background coverage {100.0 * coverage:.1f}%"
            if source != "none":
                line += f"; source: {escape(source)}"
            if invalid > 0:
                line += (f"; {100.0 * invalid:.1f}% with background ≥ orbit rate "
                         "(excluded, not treated as zero background)")
            lines.append(line)
        if result.background_pooled:
            lines.append("Rates pooled by trace median; assumes approximately constant "
                         "effective background within each trace.")
        if result.input_rows or result.excluded_photon_rows:
            lines.append(f"Input rows: {result.input_rows:,}; excluded photon rows: "
                         f"{result.excluded_photon_rows:,}.")
        label = QLabel("<br>".join(lines))
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(label)

    def _add_matched_comparison(self, root, result: CRLBResult) -> None:
        """Describe matched spread/bound summaries without decomposing errors."""
        m = self._measured
        if not m:
            reason = result.provenance.get("comparison_reason") or self._inputs.get("comparison_reason")
            if reason:
                from html import escape

                label = QLabel("Trace comparison omitted: " + escape(str(reason)))
                label.setWordWrap(True)
                root.addWidget(label)
            return
        rows = []
        for axis, label in (("xy", "lateral (xy)"), ("z", "axial (z)")):
            values = m.get(axis)
            if axis == "z" and not result.has_z:
                continue
            if not isinstance(values, dict):
                continue
            measured = values.get("measured_nm")
            bound = values.get("bound_nm")
            if measured is None or bound is None or not np.isfinite([measured, bound]).all():
                continue
            primary = result.primary_sigma_z if axis == "z" else result.primary_sigma
            if np.isfinite(primary):
                rows.append((label, values))
        if not rows:
            return

        grp = QGroupBox("Matched trace comparison — descriptive, not an error decomposition")
        grid = QGridLayout(grp)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(4)
        for col, h in enumerate(("", "median measured SD", "median trace RMS bound",
                                 "measured / bound", "matched sample")):
            lbl = QLabel(f"<b>{h}</b>")
            lbl.setWordWrap(True)
            lbl.setAlignment(Qt.AlignmentFlag.AlignLeft if col == 0
                             else Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(lbl, 0, col)
        for r, (name, values) in enumerate(rows, start=1):
            grid.addWidget(QLabel(f"<b>{name}:</b>"), r, 0)
            for col, v in enumerate((values["measured_nm"], values["bound_nm"]), start=1):
                cell = QLabel(_res(v))
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                grid.addWidget(cell, r, col)
            ratio = values.get("ratio")
            ratio_text = f"{ratio:.2f}" if ratio is not None and np.isfinite(ratio) else "unavailable"
            grid.addWidget(QLabel(ratio_text), r, 3)
            sample = QLabel(
                f"{values.get('n_traces', 0):,} traces; {values.get('n_locs', 0):,} locs; "
                f"{100.0 * values.get('coverage', 0.0):.1f}% coverage")
            sample.setWordWrap(True)
            grid.addWidget(sample, r, 4)
        for col in (1, 2, 3, 4):
            grid.setColumnStretch(col, 1)
        root.addWidget(grp)
        note = (
            f"Each trace has ≥ {MIN_LOCS_PER_TRACE} eligible localizations. Its measured SD "
            "and RMS model bound use the same coordinates and localizations; both columns "
            "are then summarized by the median across the same traces (raw, unscaled z). "
            "These summaries do not identify a physical error component or establish "
            "photon-limited performance.")
        below = [name for name, values in rows if values["measured_nm"] < values["bound_nm"]]
        if below:
            note += (" Measured spread below the conditional bound (" + ", ".join(below)
                     + "): unexplained disagreement; input conventions and model assumptions "
                     "require investigation. No cause is assigned.")
        if any(values.get("bound_kind") in {"ideal_only", "ideal only", "ideal", "background-free"}
               for _, values in rows):
            note += " The ideal-only comparison assumes zero background, not measured zero background."
        tip = QLabel(note)
        tip.setWordWrap(True)
        root.addWidget(tip)

    def _add_controls(self, root, result: CRLBResult) -> None:
        """Editable L + σ_q (+ L_z) + Recompute row (also shown on the error path)."""
        btn_row = QHBoxLayout()
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #6c6;")
        btn_row.addWidget(self._status_lbl)
        btn_row.addStretch()
        btn_row.addWidget(QLabel("pattern L (nm):"))
        self._l_spin = QDoubleSpinBox()
        self._l_spin.setRange(1.0, 1000.0)
        self._l_spin.setDecimals(1)
        self._l_spin.setSingleStep(5.0)
        self._l_spin.setValue(result.L_nm)
        self._l_spin.setToolTip(
            "Lateral pattern diameter L of the final iteration. Read from the "
            "acquisition sequence (patGeoFactor × 360 nm) when the file records it.")
        btn_row.addWidget(self._l_spin)
        btn_row.addSpacing(8)
        btn_row.addWidget(QLabel("σ_q (nm):"))
        self._sq_spin = QDoubleSpinBox()
        self._sq_spin.setRange(1.0, 2000.0)
        self._sq_spin.setDecimals(0)
        self._sq_spin.setSingleStep(10.0)
        self._sq_spin.setValue(result.sigma_q_nm)
        self._sq_spin.setToolTip(
            "Donut width parameter σ_q of the Marin & Ries PSF model, not its FWHM. "
            f"Default {CRLB_DEFAULT_SIGMA_Q_NM:.1f} nm = 310 nm / √(8 ln 2). "
            "Requires L < 2√2·σ_q for the planar donut envelope.")
        btn_row.addWidget(self._sq_spin)
        # Axial pattern size L_z — only for 3-D (when an axial bound is computed).
        self._lz_spin = None
        if self._inputs.get("geometry_z") is not None:
            btn_row.addSpacing(8)
            btn_row.addWidget(QLabel("L_z (nm):"))
            self._lz_spin = QDoubleSpinBox()
            self._lz_spin.setRange(1.0, 2000.0)
            self._lz_spin.setDecimals(1)
            self._lz_spin.setSingleStep(5.0)
            self._lz_spin.setValue(result.L_z_nm if result.L_z_nm > 0
                                   else self._inputs["geometry_z"].L_nm)
            self._lz_spin.setToolTip("Axial pattern size L_z of the final axial iteration.")
            btn_row.addWidget(self._lz_spin)
        recompute = QPushButton("Recompute")
        recompute.clicked.connect(self._recompute)
        btn_row.addWidget(recompute)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    def _recompute(self) -> None:
        L = float(self._l_spin.value())
        sq = float(self._sq_spin.value())
        lz = float(self._lz_spin.value()) if self._lz_spin is not None else None
        try:
            result = _calculation.crlb_from_inputs(self._inputs, L_nm=L, sigma_q_nm=sq, L_z_nm=lz)
            self._measured = _calculation.crlb_matched_comparison(result, self._inputs)
        except Exception as exc:
            self._status_lbl.setText(f"failed: {exc}")
            return
        self._build(result)
        msg = f"recomputed @ L = {L:.1f} nm, σ_q = {sq:.0f} nm"
        if lz is not None:
            msg += f", L_z = {lz:.1f} nm"
        self._status_lbl.setText(msg)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        """Retire this window's pyqtgraph plots before Qt deletes them."""
        from .qt_lifecycle import dispose_plot_widgets

        dispose_plot_widgets(self)
        super().closeEvent(event)


class InfoDialog(QDialog):
    """Small non-blocking rich-text information dialog with clickable links."""

    def __init__(self, title: str, html: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.resize(620, 360)

        root = QVBoxLayout(self)
        label = QLabel(html)
        label.setWordWrap(True)
        label.setOpenExternalLinks(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(label)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.close)
        root.addWidget(bb)


def _show_info_dialog(parent, title: str, html: str) -> None:
    dlg = InfoDialog(title, html, parent=parent)
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    if parent is not None:
        parent._loc_precision_info_dialog = dlg


def _try_journal(state, text: str) -> None:
    """Append a journal entry if the state has a processing journal."""
    journal = getattr(state, "journal", None)
    if journal is not None:
        try:
            journal.add("analysis", text)
        except Exception:
            pass
    try:
        state.log(text, "INFO")
    except Exception:
        pass
