"""
minflux_viewer.plugins.drift_correction.drift_correction_window
================================================================
Modeless window for **drift correction**, with two complementary methods:

* **MBM beads** — the instrument's fiducial beads, computed exactly as Imspector
  does (``lnc`` shifted by the consensus of the selected bead tracks; see
  :mod:`minflux_viewer.analysis.bead_drift`). The default selection is the one
  the file records, so with no change the result *is* the file's own correction
  (and the window says how closely it is reproduced). Per-bead quality, a
  suggested selection and a median consensus go beyond Imspector.
* **Time windows** — fiducial-free cross-correlation of the localizations
  between time windows (Ostersehlt et al. 2022; see
  :mod:`minflux_viewer.analysis.drift_correction`). It sees drift the beads do
  not, at a time resolution of minutes.
* **Beads, then the residual from localizations** — both, in that order: the
  beads remove the stage drift at their 4 s cadence, the time windows the slow
  residual of the structure relative to the beads.

The estimate runs off the UI thread (it appears in Help ▸ Task Monitor and can
be stopped); *Create corrected dataset* adds a new dataset and keeps the
original for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ...analysis import bead_drift as bd
from ...analysis import drift_correction as dc
from ...colors import component_colors
from ...ui.dataset_binding import event_index

METHOD_BEADS = "beads"
METHOD_WINDOWS = "windows"
METHOD_BOTH = "both"
METHOD_LABELS = {
    METHOD_BEADS: "MBM beads (as Imspector)",
    METHOD_WINDOWS: "Localizations: time-window cross-correlation",
    METHOD_BOTH: "MBM beads, then the residual from localizations",
}
_METHOD_TIPS = {
    METHOD_BEADS: (
        "Shift the uncorrected positions (lnc) by the consensus of the selected\n"
        "fiducial beads' smoothed tracks — Imspector's own drift correction.\n"
        "With the recorded bead set this reproduces the file's coordinates."),
    METHOD_WINDOWS: (
        "Estimate drift from the localizations themselves: cross-correlate\n"
        "time windows of the acquisition (Ostersehlt et al. 2022). No fiducials\n"
        "needed; time resolution is the window length."),
    METHOD_BOTH: (
        "Bead correction first, then the time-window estimate of the drift\n"
        "that remains — the structure moving relative to the beads."),
}
_CONSENSUS_LABELS = {"mean": "mean (as Imspector)", "median": "median (robust)"}
_POOL_NAME = "drift-correction"


@dataclass
class DriftEstimate:
    """What *Estimate* produced, for *Create corrected dataset*."""
    method: str
    dataset_key: object                    # the DatasetKey estimated (never written out)
    dims: int
    coords_nm: np.ndarray                  # (n, dims) corrected display coordinates
    beads: bd.BeadCorrection | None = None
    windows: dc.DriftResult | None = None
    bead_names: list = field(default_factory=list)
    n_beads_usable: int = 0
    recorded: bool = False                 # bead selection == the file's recorded one
    # The active filter the estimate was made under. ``row_mask`` selects the
    # rows of ``coords_nm`` out of the last-valid rows (None = every row passed);
    # the corrected dataset holds exactly those rows.
    row_mask: np.ndarray | None = None
    n_source: int = 0                      # last-valid rows before the filter
    filter_attrs: list = field(default_factory=list)
    filter_skipped: list = field(default_factory=list)   # specs that could not be evaluated
    filter_signature: str = ""

    @property
    def n_kept(self) -> int:
        return int(self.coords_nm.shape[0])

    @property
    def filtered(self) -> bool:
        return self.row_mask is not None


class DriftCorrectionWindow(QDialog):
    """Estimate and apply drift correction (beads and/or time windows)."""

    def __init__(self, state, owner=None) -> None:
        super().__init__(None)                 # non-owned, modeless
        self._state = state
        self._owner = owner
        self._estimate: DriftEstimate | None = None
        self._task = None
        self._bead_inputs = None               # (dataset key, BeadInputs | None, reason)
        self._suspend = False

        self.setWindowTitle("Drift Correction")
        self.resize(720, 820)
        self._build_ui()

        try:
            state.dataset_events.active_changed.connect(self._on_active_changed)
            state.dataset_events.filter_changed.connect(self._on_filter_changed)
        except Exception:
            pass
        self._refresh_dataset_info()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        self._info = QLabel("")
        self._info.setWordWrap(True)
        self._info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self._info)

        row = QHBoxLayout()
        row.addWidget(QLabel("Method:"))
        self._method_combo = QComboBox()
        for key in (METHOD_BEADS, METHOD_WINDOWS, METHOD_BOTH):
            self._method_combo.addItem(METHOD_LABELS[key], key)
            self._method_combo.setItemData(
                self._method_combo.count() - 1, _METHOD_TIPS[key], Qt.ItemDataRole.ToolTipRole)
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        row.addWidget(self._method_combo, 1)
        root.addLayout(row)

        # --- fiducial beads ---------------------------------------------
        self._beads_box = QGroupBox("Fiducial (MBM) beads")
        bl = QVBoxLayout(self._beads_box)
        self._bead_table = QTableWidget(0, 6)
        self._bead_table.setHorizontalHeaderLabels(
            ["Use", "Bead", "Samples", "Drift p-p x / y / z (nm)", "Deviation (nm)", "Recorded"])
        self._bead_table.horizontalHeaderItem(4).setToolTip(
            "Drift-only deviation of this bead from the median of all beads (rms,\n"
            "constant offset removed). A bead far above the others moved on its own.")
        self._bead_table.horizontalHeaderItem(5).setToolTip(
            "In the selection recorded in the file — the beads its current\n"
            "correction was computed with.")
        hdr = self._bead_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setStretchLastSection(True)
        self._bead_table.verticalHeader().setVisible(False)
        self._bead_table.setMinimumHeight(150)
        self._bead_table.itemChanged.connect(self._on_bead_toggled)
        bl.addWidget(self._bead_table)
        sel_row = QHBoxLayout()
        for text, slot, tip in (
                ("Recorded selection", self._select_recorded,
                 "The beads the file's current correction used."),
                ("Suggested selection", self._select_suggested,
                 "Drop beads whose deviation exceeds 2.5x the median deviation\n"
                 "(an objective outlier rule; it needs three beads to judge)."),
                ("All beads", self._select_all, "Every bead with a usable track.")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            sel_row.addWidget(b)
        sel_row.addStretch(1)
        bl.addLayout(sel_row)
        opt = QGridLayout()
        opt.addWidget(QLabel("Consensus:"), 0, 0)
        self._consensus_combo = QComboBox()
        for key in bd.CONSENSUS_MODES:
            self._consensus_combo.addItem(_CONSENSUS_LABELS[key], key)
        self._consensus_combo.setToolTip(
            "How the selected bead tracks are combined. The mean is Imspector's\n"
            "computation; the median resists one deviating bead among three or more.")
        opt.addWidget(self._consensus_combo, 0, 1)
        opt.addWidget(QLabel("Smoothing (bead samples):"), 0, 2)
        self._smooth_spin = QDoubleSpinBox()
        self._smooth_spin.setRange(0.0, 20.0)
        self._smooth_spin.setDecimals(2)
        self._smooth_spin.setSingleStep(0.1)
        self._smooth_spin.setValue(bd.mbm_drift.DEFAULT_SMOOTH_SAMPLES)
        self._smooth_spin.setToolTip(
            "Gaussian smoothing σ of each bead track, in bead samples (1.2 ≈ 5 s at\n"
            "the usual 4 s cadence — the value that reproduces Imspector).")
        opt.addWidget(self._smooth_spin, 0, 3)
        bl.addLayout(opt)
        root.addWidget(self._beads_box)

        # --- time windows -------------------------------------------------
        self._windows_box = QGroupBox("Time windows (fiducial-free)")
        grid = QGridLayout(self._windows_box)
        grid.addWidget(QLabel("Analysis dimensions:"), 0, 0)
        self._dims_combo = QComboBox()
        self._dims_combo.addItems(["Auto (from dataset)", "2D (x, y)", "3D (x, y, z)"])
        self._dims_combo.currentIndexChanged.connect(lambda *_: self._refresh_dataset_info())
        grid.addWidget(self._dims_combo, 0, 1)
        grid.addWidget(QLabel("Render resolution (nm):"), 1, 0)
        self._res_spin = QDoubleSpinBox()
        self._res_spin.setRange(0.5, 100.0)
        self._res_spin.setDecimals(2)
        self._res_spin.setSingleStep(1.0)
        self._res_spin.setValue(5.0)
        self._res_spin.setToolTip(
            "Pixel/voxel size of the window images (nm); each localization is a\n"
            "Gaussian of FWHM 3 pixels. Finer resolves smaller drift but costs\n"
            "memory and time.")
        grid.addWidget(self._res_spin, 1, 1)
        grid.addWidget(QLabel("Time window (s):"), 2, 0)
        tw = QHBoxLayout()
        self._auto_T = QCheckBox("Auto")
        self._auto_T.setChecked(True)
        self._auto_T.setToolTip(
            f"The most windows (at most {dc.AUTO_MAX_WINDOWS}) for which 80 % of them\n"
            f"hold at least {dc.AUTO_MIN_LOCS_PER_WINDOW[2]:,} localizations (2-D) or "
            f"{dc.AUTO_MIN_LOCS_PER_WINDOW[3]:,} (3-D).\nUntick to set the window yourself.")
        self._auto_T.toggled.connect(self._on_auto_T_toggled)
        self._T_spin = QDoubleSpinBox()
        self._T_spin.setRange(1.0, 1e7)
        self._T_spin.setDecimals(1)
        self._T_spin.setSingleStep(60.0)
        self._T_spin.setValue(600.0)
        self._T_spin.setEnabled(False)
        self._T_spin.setToolTip("Length of each time window (s); at least two are needed.")
        tw.addWidget(self._auto_T)
        tw.addWidget(self._T_spin, 1)
        grid.addLayout(tw, 2, 1)
        root.addWidget(self._windows_box)

        self._plot = pg.PlotWidget()
        self._plot.setBackground("w")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setLabel("bottom", "time", units="s")
        self._plot.setLabel("left", "drift removed", units="nm")
        self._plot.addLegend(offset=(-10, 10))         # top right, clear of the curves
        root.addWidget(self._plot, 1)

        self._stats = QLabel("")
        self._stats.setWordWrap(True)
        self._stats.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self._stats)

        btns = QHBoxLayout()
        self._estimate_btn = QPushButton("Estimate drift")
        self._estimate_btn.clicked.connect(self._on_estimate)
        btns.addWidget(self._estimate_btn)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        btns.addWidget(self._stop_btn)
        self._apply_btn = QPushButton("Create corrected dataset")
        self._apply_btn.clicked.connect(self._on_apply)
        self._apply_btn.setEnabled(False)
        btns.addWidget(self._apply_btn)
        btns.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        btns.addWidget(close)
        root.addLayout(btns)

        refs = QLabel(
            "<span style='color:gray;font-size:11px'>"
            "Beads: the drift correction the acquisition software applies, recomputed "
            "from the uncorrected positions (lnc) and the bead record; fiducial "
            "stabilisation as in Schmidt <i>et&nbsp;al.</i>, <i>Nat. Commun.</i> 12:1478 "
            "(2021).<br>Time windows: Ostersehlt <i>et&nbsp;al.</i>, <i>DNA-PAINT MINFLUX "
            "nanoscopy</i>, <i>Nat. Methods</i> 19:1072 (2022), re-derived from its "
            "reference implementation (MIT license).</span>")
        refs.setWordWrap(True)
        refs.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(refs)

    # --------------------------------------------------------------- helpers
    def _active(self):
        return self._state.active_dataset

    def _method(self) -> str:
        return str(self._method_combo.currentData())

    def _chosen_dims(self, ds) -> int:
        idx = self._dims_combo.currentIndex()
        if idx == 1:
            return 2
        if idx == 2:
            return 3
        return 3 if int(getattr(ds.prop, "num_dim", 2)) >= 3 else 2

    def _validate(self, ds):
        if ds is None:
            return "No active dataset. Load a MINFLUX dataset and select it."
        from ...core.dataset_kind import is_minflux
        if not is_minflux(ds):
            return ("The active dataset is not a MINFLUX dataset (no trace ids), "
                    "so drift correction cannot use the time information.")
        from minflux_viewer.core.loader_raw_access import mfx_get
        try:
            t = mfx_get(ds, "tim", itr="last", vld_only=True)
        except Exception:
            t = None
        if t is None or np.asarray(t).size == 0:
            return "The active dataset has no time information (tim)."
        t = np.asarray(t, dtype=float)
        if not np.isfinite(t).any() or np.ptp(t[np.isfinite(t)]) <= 0:
            return ("The active dataset's time points do not span a range, so "
                    "drift over time cannot be estimated.")
        if self._method() in (METHOD_BEADS, METHOD_BOTH):
            inputs, reason = self._beads_for(ds)
            if inputs is None:
                return f"Bead correction is not possible: {reason}."
        return None

    def _beads_for(self, ds):
        """Cached :func:`bead_drift.bead_inputs_for_dataset` of the active dataset."""
        if ds is None:
            return None, "no dataset"
        key = self._state.key_of(ds)
        if self._bead_inputs is None or self._bead_inputs[0] != key:
            try:
                inputs, reason = bd.bead_inputs_for_dataset(ds)
            except Exception as exc:                       # noqa: BLE001
                inputs, reason = None, str(exc)
            self._bead_inputs = (key, inputs, reason)
        return self._bead_inputs[1], self._bead_inputs[2]

    def _gather(self, ds, dims):
        """Last-valid display coordinates (nm) and times (s)."""
        from minflux_viewer.core.loader_raw_access import mfx_get
        cols = [mfx_get(ds, k, itr="last", vld_only=True) for k in ("xnm", "ynm", "znm")[:dims]]
        t = mfx_get(ds, "tim", itr="last", vld_only=True)
        return [np.asarray(c, dtype=float) for c in cols], np.asarray(t, dtype=float)

    @staticmethod
    def _filter_signature(ds) -> str:
        """Identity of the active filter, to tell whether an estimate is stale."""
        return repr(ds.state.get("filter_specs") or [])

    @staticmethod
    def _filter_rows(ds, n: int):
        """The active filter over the last-valid rows :meth:`_gather` reads.

        Returns ``(keep, attrs, skipped)``: the row mask, the filtered attributes
        and those whose spec could not be evaluated (reported, not hidden). The
        specs are re-evaluated exactly as the snapshot export does, each at its
        own iteration (cfr/efc at the one they are measured at).
        """
        from minflux_viewer.core.loader_raw_access import mfx_filter_mask
        specs = ds.state.get("filter_specs") or []
        if not specs:
            return np.ones(n, dtype=bool), [], []
        fm = mfx_filter_mask(ds, itr="last", vld_only=True)
        if fm is None:
            raise ValueError("the active filter cannot be evaluated: the dataset "
                             "has no trace ids at its last valid iteration")
        keep = np.asarray(fm[0], dtype=bool).ravel()
        if keep.shape[0] != n:
            raise ValueError(f"the active filter covers {keep.shape[0]:,} localizations "
                             f"but the data has {n:,}")
        skipped = list(fm[1])
        attrs = []
        for spec in specs:
            name = str(spec.get("attribute", "") or "")
            if name and name not in attrs and name not in skipped:
                attrs.append(name)
        return keep, attrs, skipped

    def _selected_gris(self) -> list[int]:
        out = []
        for r in range(self._bead_table.rowCount()):
            item = self._bead_table.item(r, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                out.append(int(item.data(Qt.ItemDataRole.UserRole)))
        return out

    def _set_selection(self, gris) -> None:
        gris = {int(g) for g in gris}
        self._suspend = True
        try:
            for r in range(self._bead_table.rowCount()):
                item = self._bead_table.item(r, 0)
                gri = int(item.data(Qt.ItemDataRole.UserRole))
                item.setCheckState(Qt.CheckState.Checked if gri in gris
                                   else Qt.CheckState.Unchecked)
        finally:
            self._suspend = False
        self._invalidate()

    # --------------------------------------------------------------- events
    def _on_active_changed(self, *_a) -> None:
        self._invalidate()
        self._refresh_dataset_info()

    def _on_filter_changed(self, idx) -> None:
        idx = event_index(self._state, idx)
        if idx is None:
            return
        # The estimate used the filtered rows, so another filter needs another estimate.
        if idx != self._state.active_idx:
            return
        if self._estimate is not None:
            self._invalidate()
            self._stats.setText("The filter changed: estimate the drift again.")
        self._refresh_dataset_info()

    def _on_method_changed(self, *_a) -> None:
        m = self._method()
        self._beads_box.setVisible(m in (METHOD_BEADS, METHOD_BOTH))
        self._windows_box.setVisible(m in (METHOD_WINDOWS, METHOD_BOTH))
        self._invalidate()
        self._refresh_dataset_info()

    def _on_auto_T_toggled(self, checked: bool) -> None:
        self._T_spin.setEnabled(not checked)
        self._refresh_dataset_info()

    def _on_bead_toggled(self, *_a) -> None:
        if not self._suspend:
            self._invalidate()

    def _select_recorded(self) -> None:
        inputs, _ = self._beads_for(self._active())
        if inputs is not None:
            self._set_selection(inputs.recorded_gris)

    def _select_suggested(self) -> None:
        inputs, _ = self._beads_for(self._active())
        if inputs is not None:
            self._set_selection(bd.suggest_beads(inputs.beads))

    def _select_all(self) -> None:
        inputs, _ = self._beads_for(self._active())
        if inputs is not None:
            self._set_selection(inputs.usable_gris)

    def _invalidate(self) -> None:
        """A parameter changed: the last estimate no longer describes it."""
        if self._estimate is not None:
            self._estimate = None
            self._apply_btn.setEnabled(False)

    def _refresh_dataset_info(self) -> None:
        ds = self._active()
        if ds is None:
            self._info.setText("No active dataset.")
            self._estimate_btn.setEnabled(False)
            self._apply_btn.setEnabled(False)
            return
        self._estimate_btn.setEnabled(self._task is None)

        inputs, reason = self._beads_for(ds)
        bead_ok = inputs is not None
        for key in (METHOD_BEADS, METHOD_BOTH):
            i = self._method_combo.findData(key)
            item = self._method_combo.model().item(i)
            if item is not None:
                item.setEnabled(bead_ok)
            self._method_combo.setItemData(
                i, _METHOD_TIPS[key] if bead_ok else f"Not available: {reason}.",
                Qt.ItemDataRole.ToolTipRole)
        if not bead_ok and self._method() != METHOD_WINDOWS:
            self._method_combo.setCurrentIndex(self._method_combo.findData(METHOD_WINDOWS))
            return                                   # _on_method_changed refreshes
        self._fill_bead_table(inputs)
        self._beads_box.setVisible(bead_ok and self._method() in (METHOD_BEADS, METHOD_BOTH))
        self._windows_box.setVisible(self._method() in (METHOD_WINDOWS, METHOD_BOTH))

        dims = self._chosen_dims(ds)
        n_loc = int(getattr(ds.prop, "num_loc", 0))
        n_tr = int(getattr(ds.prop, "num_traces", 0))
        span_txt = auto_txt = bead_txt = filter_txt = ""
        try:
            _, t = self._gather(ds, 2)
            try:
                keep, attrs, skipped = self._filter_rows(ds, t.size)
            except Exception as exc:                       # noqa: BLE001
                keep, attrs, skipped = None, [], []
                filter_txt = (f"<br><span style='color:#a33'>Active filter: {exc}.</span>")
            if keep is not None:
                if not keep.all():
                    filter_txt = (f"<br>Active filter ({', '.join(attrs) or 'none evaluable'}): "
                                  f"{int(keep.sum()):,} of {keep.size:,} localizations pass "
                                  "and are used.")
                    t = t[keep]
                if skipped:
                    filter_txt += (f"<br><span style='color:#a33'>The filter on "
                                   f"{', '.join(skipped)} cannot be evaluated here and "
                                   "will not be applied.</span>")
            ft = t[np.isfinite(t)]
            if ft.size:
                span_txt = f", {np.ptp(ft):.0f} s acquisition"
                if self._auto_T.isChecked():
                    aw = dc.auto_time_window(ft, dims)
                    self._T_spin.blockSignals(True)
                    self._T_spin.setValue(aw.time_window_s)
                    self._T_spin.blockSignals(False)
                    auto_txt = (f" · auto window {aw.time_window_s:.0f} s "
                                f"({aw.n_windows} windows"
                                + (", sparse" if aw.sparse else "") + ")")
        except Exception:
            pass
        if bead_ok:
            bead_txt = (f"<br>{len(inputs.beads)} usable beads; recorded selection: "
                        f"{len(inputs.recorded_gris)} ({inputs.recorded_source})")
        else:
            bead_txt = f"<br>Bead correction unavailable: {reason}."
        self._info.setText(
            f"<b>{ds.name}</b><br>{n_loc:,} valid localizations · {n_tr:,} traces · "
            f"{int(getattr(ds.prop, 'num_dim', 2))}-D data · time windows in {dims}-D"
            f"{span_txt}{auto_txt}{filter_txt}{bead_txt}")

    def _fill_bead_table(self, inputs) -> None:
        self._suspend = True
        try:
            keep = set(self._selected_gris()) if self._bead_table.rowCount() else None
            self._bead_table.setRowCount(0)
            if inputs is None:
                return
            chosen = set(inputs.recorded_gris) if not keep else keep
            for b in inputs.beads:
                r = self._bead_table.rowCount()
                self._bead_table.insertRow(r)
                use = QTableWidgetItem()
                use.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                use.setCheckState(Qt.CheckState.Checked if b.gri in chosen
                                  else Qt.CheckState.Unchecked)
                use.setData(Qt.ItemDataRole.UserRole, int(b.gri))
                self._bead_table.setItem(r, 0, use)
                cells = (f"{b.name}  (gri {b.gri})", f"{b.n_samples:,}",
                         " / ".join(f"{v:.0f}" for v in b.ptp_nm),
                         f"{b.deviation_nm:.1f}" if np.isfinite(b.deviation_nm) else "—",
                         "yes" if b.recorded else "")
                for c, text in enumerate(cells, start=1):
                    it = QTableWidgetItem(text)
                    it.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    self._bead_table.setItem(r, c, it)
        finally:
            self._suspend = False

    # --------------------------------------------------------------- estimate
    def _on_estimate(self) -> None:
        ds = self._active()
        err = self._validate(ds)
        if err:
            QMessageBox.information(self, "Drift Correction", err)
            return
        method = self._method()
        dims = self._chosen_dims(ds)
        bead_inputs = None
        gris: list[int] = []
        if method in (METHOD_BEADS, METHOD_BOTH):
            bead_inputs, _ = self._beads_for(ds)
            gris = self._selected_gris()
            if not gris:
                QMessageBox.information(self, "Drift Correction", "Select at least one bead.")
                return
        try:
            cols, t = self._gather(ds, dims)
        except Exception as exc:
            QMessageBox.warning(self, "Drift Correction", f"Could not read data:\n{exc}")
            return
        # Only the localizations that pass the active filter are estimated from
        # and corrected: filtering out poor localizations is meant to happen
        # before the drift estimate, not after it.
        n_source = int(t.size)
        try:
            keep, f_attrs, f_skipped = self._filter_rows(ds, n_source)
        except Exception as exc:
            QMessageBox.warning(self, "Drift Correction", f"Could not apply the filter:\n{exc}")
            return
        if not keep.any():
            QMessageBox.information(
                self, "Drift Correction",
                "The active filter removes every localization of this dataset, so there "
                "is nothing to estimate the drift from.")
            return
        filtered = not keep.all()
        if filtered:
            cols = [c[keep] for c in cols]
            t = t[keep]
        f_signature = self._filter_signature(ds)
        consensus = str(self._consensus_combo.currentData())
        smooth = float(self._smooth_spin.value())
        res_nm = float(self._res_spin.value())
        T = None if self._auto_T.isChecked() else float(self._T_spin.value())
        ds_key = self._state.key_of(ds)

        def work(report):
            coords = np.column_stack(cols)
            bead_corr = None
            if bead_inputs is not None:
                report("Drift correction: bead consensus")
                bead_corr = bd.correct_with_beads(bead_inputs, gris, consensus=consensus,
                                                  smooth_samples=smooth)
                disp = bd.display_coordinates_nm(bead_corr.positions_m,
                                                 bead_inputs.z_scaling_factor)
                if disp.shape[0] != n_source:
                    raise ValueError(
                        f"the bead correction covers {disp.shape[0]:,} localizations "
                        f"but the data has {n_source:,}")
                # The bead consensus does not depend on the filter (it is the
                # beads' drift); only the rows it is applied to do.
                coords = disp[keep, :dims]
            windows = None
            if method in (METHOD_WINDOWS, METHOD_BOTH):
                def progress(frac):
                    report(f"Drift correction: time windows  {100.0 * frac:.0f}%")
                windows = dc.estimate_drift(
                    coords[:, 0], coords[:, 1], coords[:, 2] if dims == 3 else None, t,
                    resolution_nm=res_nm, time_window_s=T, dims=dims, progress=progress)
                shift = np.column_stack([windows.dx, windows.dy]
                                        + ([windows.dz] if dims == 3 else []))
                coords = coords - shift
            return bead_corr, windows, coords

        from ...ui.background_tasks import BackgroundTask, shared_thread_pool

        task = BackgroundTask(work, description="Drift correction", category="analysis")
        task.signals.stage.connect(lambda text: self._state.status_progress(text))
        filter_info = {"row_mask": keep if filtered else None, "n_source": n_source,
                       "filter_attrs": f_attrs, "filter_skipped": f_skipped,
                       "filter_signature": f_signature}
        task.signals.done.connect(
            lambda result: self._on_estimated(result, method, ds_key, dims, gris, bead_inputs,
                                              filter_info))
        task.signals.failed.connect(self._on_estimate_failed)
        task.signals.cancelled.connect(self._on_estimate_cancelled)
        task.signals.finished.connect(self._on_task_finished)
        self._task = task
        self._estimate_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._apply_btn.setEnabled(False)
        self._stats.setText("Estimating …")
        shared_thread_pool(_POOL_NAME).start(task)

    def _on_stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._stats.setText("Stopping …")

    def _on_task_finished(self) -> None:
        self._task = None
        self._stop_btn.setEnabled(False)
        self._estimate_btn.setEnabled(self._active() is not None)

    def _on_estimate_failed(self, message: str) -> None:
        self._state.status_progress("Drift estimation failed")
        self._state.log(f"Drift correction failed: {message}", "ERROR")
        self._stats.setText(f"<span style='color:#a33'>Failed: {message}</span>")

    def _on_estimate_cancelled(self) -> None:
        self._state.status_progress("Drift estimation stopped")
        self._stats.setText("Stopped.")

    def _on_estimated(self, result, method, ds_key, dims, gris, bead_inputs,
                      filter_info) -> None:
        bead_corr, windows, coords = result
        ds = self._active()
        if ds is None or self._state.key_of(ds) != ds_key:
            self._stats.setText("The active dataset changed during the estimate; "
                                "estimate again.")
            return
        if self._filter_signature(ds) != filter_info["filter_signature"]:
            self._stats.setText("The filter changed during the estimate; estimate again.")
            return
        names = []
        recorded = False
        if bead_inputs is not None:
            names = [bead_inputs.names.get(g, f"gri {g}") for g in sorted(gris)]
            recorded = sorted(gris) == sorted(bead_inputs.recorded_gris)
        self._estimate = DriftEstimate(
            method=method, dataset_key=ds_key, dims=dims, coords_nm=coords,
            beads=bead_corr, windows=windows, bead_names=names,
            n_beads_usable=len(bead_inputs.beads) if bead_inputs is not None else 0,
            recorded=recorded, **filter_info)
        self._state.status_progress("Drift estimated", 1.0)
        self._apply_btn.setEnabled(True)
        self._plot_estimate(self._estimate)
        self._report_estimate(ds, self._estimate)

    def _plot_estimate(self, est: DriftEstimate) -> None:
        self._plot.clear()
        legend = self._plot.plotItem.legend
        if legend is not None:
            legend.clear()
        colors = component_colors(self._state.prefs, "plugins", "Drift Correction")
        pens = [QColor(*colors["X drift"]), QColor(*colors["Y drift"]), QColor(*colors["Z drift"])]
        axes = "xyz"[:est.dims]
        if est.beads is not None and est.beads.curve_t.size:
            for k, ax in enumerate(axes):
                self._plot.plot(est.beads.curve_t, est.beads.curve_nm[:, k],
                                pen=pg.mkPen(pens[k], width=2), name=f"{ax} beads")
        if est.windows is not None:
            w = est.windows
            traj = [w.dxt, w.dyt, w.dzt]
            for k, ax in enumerate(axes):
                if w.curve.size:
                    self._plot.plot(w.curve_t, w.curve[:, k],
                                    pen=pg.mkPen(pens[k], width=2,
                                                 style=Qt.PenStyle.DashLine
                                                 if est.beads is not None else Qt.PenStyle.SolidLine),
                                    name=(f"{ax} residual" if est.beads is not None
                                          else f"{ax} drift"))
                self._plot.plot(w.ti, traj[k], pen=None, symbol="o", symbolSize=6,
                                symbolBrush=pens[k])
        self._plot.enableAutoRange()
        self._plot.autoRange()

    def _report_estimate(self, ds, est: DriftEstimate) -> None:
        lines = []
        idx = self._state.active_idx
        if est.filtered:
            lines.append(
                f"Active filter ({', '.join(est.filter_attrs)}): {est.n_kept:,} of "
                f"{est.n_source:,} localizations "
                + ("estimated from and " if est.windows is not None else "")
                + "kept in the corrected dataset.")
        if est.filter_skipped:
            skipped = ", ".join(est.filter_skipped)
            lines.append(f"<span style='color:#a33'>The filter on {skipped} could not be "
                         "evaluated and was not applied.</span>")
            self._state.log(
                f"Drift correction on '{ds.name}': the filter on {skipped} could not be "
                "evaluated at the last valid iteration and was not applied.",
                "WARN", dataset_idx=idx)
        if est.beads is not None:
            b = est.beads
            ptp = np.ptp(b.curve_nm, axis=0) if b.curve_nm.size else np.zeros(3)
            p_txt = "/".join(f"{v:.1f}" for v in ptp[:est.dims])
            if est.recorded:
                rep = ("reproduces the file's recorded correction to "
                       + "/".join(f"{v:.2f}" for v in b.reproduction_nm[:est.dims]) + " nm rms")
            else:
                moved = b.shift_from_file_nm[np.isfinite(b.shift_from_file_nm)]
                rep = (f"moves the file's coordinates by median {np.median(moved):.2f} nm, "
                       f"at most {moved.max():.2f} nm" if moved.size else "")
            lines.append(f"Beads: {len(b.gris)} of {est.n_beads_usable} "
                         f"({', '.join(est.bead_names)}), {b.consensus}; drift removed "
                         f"p-p {'/'.join('xyz'[:est.dims])} = {p_txt} nm; {rep}.")
            self._state.log(
                f"Drift correction (MBM beads) on '{ds.name}': {len(b.gris)} of "
                f"{est.n_beads_usable} beads ({', '.join(est.bead_names)}), {b.consensus} "
                f"consensus, smoothing {b.smooth_samples:g} bead samples; drift removed "
                f"peak-to-peak {'/'.join('xyz'[:est.dims])} = {p_txt} nm; {rep}.",
                dataset_idx=idx)
        if est.windows is not None:
            w = est.windows
            traj = [w.dxt, w.dyt, w.dzt][:w.dims]
            p_txt = "/".join(f"{np.ptp(v):.1f}" for v in traj)
            r_txt = "/".join(f"{v:.2f}" for v in w.rms)
            what = "residual drift" if est.beads is not None else "drift"
            mode = "automatic" if w.auto_window else "set"
            lines.append(f"Time windows: {w.dims}-D, {w.n_windows} × {w.time_window_s:.0f} s "
                         f"({mode}), render {w.resolution_nm:g} nm; {what} p-p "
                         f"{'/'.join('xyz'[:w.dims])} = {p_txt} nm (rms {r_txt} nm).")
            filt = (f"; estimated from {est.n_kept:,} of {est.n_source:,} localizations "
                    f"passing the active filter on {', '.join(est.filter_attrs)}"
                    if est.filtered else "")
            self._state.log(
                f"Drift correction (time windows) on '{ds.name}': {w.dims}-D, "
                f"{w.n_windows} windows of {w.time_window_s:.0f} s ({mode}), render "
                f"{w.resolution_nm:g} nm; {what} peak-to-peak "
                f"{'/'.join('xyz'[:w.dims])} = {p_txt} nm, rms {r_txt} nm{filt}.",
                dataset_idx=idx)
        self._stats.setText("<br>".join(lines))

    # --------------------------------------------------------------- apply
    def _on_apply(self) -> None:
        ds = self._active()
        est = self._estimate
        if ds is None or est is None or self._state.key_of(ds) != est.dataset_key:
            QMessageBox.information(self, "Drift Correction",
                                    "Estimate the drift of the active dataset first.")
            return
        if self._filter_signature(ds) != est.filter_signature:
            self._invalidate()
            QMessageBox.information(self, "Drift Correction",
                                    "The filter changed since the estimate; estimate the "
                                    "drift again.")
            return
        try:
            self._build_corrected_dataset(ds, est)
        except Exception as exc:
            self._state.log(f"Drift correction apply failed: {exc}", "ERROR")
            QMessageBox.warning(self, "Drift Correction",
                                f"Could not create corrected dataset:\n{exc}")

    def _build_corrected_dataset(self, ds, est: DriftEstimate):
        from minflux_viewer.core.loader_raw_access import last_valid_attr_values, mfx_get

        from ...core.dataset import build_localization_dataset

        coords = np.asarray(est.coords_nm, dtype=float)
        n = coords.shape[0]
        rows = est.row_mask                     # None: every last-valid row

        def column(arr):
            """A last-valid column cut to the estimated rows, or None if misaligned."""
            if arr is None:
                return None
            arr = np.asarray(arr)
            if arr.ndim != 1 or arr.shape[0] != (est.n_source or n):
                return None
            return arr[rows] if rows is not None else arr

        tid = column(mfx_get(ds, "tid", itr="last", vld_only=True))
        tim = column(mfx_get(ds, "tim", itr="last", vld_only=True))
        if tid is None or tim is None or tid.shape[0] != n:
            raise ValueError("the trace ids / times no longer align with the estimate; "
                             "estimate the drift again")
        # Carry the canonical per-localization quality attributes, aligned with
        # the coordinates; cfr/efc from the iteration they are measured at (at the
        # last one they read 0 / NaN). Derived attributes are recomputed.
        carry = {}
        for name in ("efo", "cfr", "dcr", "eco", "ecc", "efc", "fbg",
                     "lcx", "lcy", "lcz", "ext", "sta"):
            arr = column(last_valid_attr_values(ds, name))
            if arr is not None:
                carry[name] = arr

        new = build_localization_dataset(
            name=f"{ds.name} (drift corrected)",
            x_nm=coords[:, 0], y_nm=coords[:, 1],
            z_nm=coords[:, 2] if est.dims == 3 else None,
            tid=tid, tim=np.asarray(tim, dtype=float),
            attrs=carry or None,
            source_version=str(ds.metadata.get("source_version", "imported")),
            prefs=self._state.prefs)
        # Coordinates are final (drift removed, Z scaling factor applied in znm):
        # pin the factor to 1.0 and suppress the post-load estimate.
        try:
            new.set_z_scaling_factor(1.0, source="processed (drift corrected, coordinates final)")
            new.derived["z_scaling_factor"] = np.asarray([1.0], dtype=float)
        except Exception:
            pass
        import uuid
        new.file.folder = f"<drift-corrected>/{uuid.uuid4().hex}"
        record = {"method": est.method, "dims": est.dims, "source": ds.name}
        if est.beads is not None:
            record["beads"] = {"gris": est.beads.gris, "names": est.bead_names,
                               "consensus": est.beads.consensus,
                               "smooth_samples": est.beads.smooth_samples,
                               "recorded_selection": est.recorded}
        if est.windows is not None:
            w = est.windows
            record["windows"] = {"time_window_s": w.time_window_s, "n_windows": w.n_windows,
                                 "automatic": w.auto_window, "resolution_nm": w.resolution_nm,
                                 "rms_nm": list(w.rms)}
            # flat keys kept for older readers of this record
            record.update(resolution_nm=w.resolution_nm, time_window_s=w.time_window_s,
                          n_windows=w.n_windows, rms_nm=list(w.rms))
        if est.filtered or est.filter_skipped:
            record["filter"] = {"attributes": list(est.filter_attrs),
                                "n_kept": est.n_kept, "n_source": est.n_source,
                                "not_applied": list(est.filter_skipped)}
        new.metadata["drift_corrected"] = True
        new.metadata["drift_correction"] = record

        idx = self._state.add_dataset(new)
        kept = (f"; {est.n_kept:,} of {est.n_source:,} localizations passing the active "
                f"filter on {', '.join(est.filter_attrs)}" if est.filtered else "")
        self._state.log(
            f"Created drift-corrected dataset '{new.name}' from '{ds.name}' "
            f"({METHOD_LABELS[est.method]}{kept}).", dataset_idx=idx)
        return idx

    # --------------------------------------------------------------- close
    def closeEvent(self, event) -> None:
        events = self._state.dataset_events
        for signal, slot in ((events.active_changed, self._on_active_changed),
                             (events.filter_changed, self._on_filter_changed)):
            try:
                signal.disconnect(slot)
            except Exception:
                pass
        if self._task is not None:
            from ...ui.background_tasks import retire_background_tasks
            retire_background_tasks([self._task])
            self._task = None
        from ...ui.qt_lifecycle import dispose_plot_widgets

        # Retire the pyqtgraph plots last, after this window's own teardown.
        dispose_plot_widgets(self)
        super().closeEvent(event)
