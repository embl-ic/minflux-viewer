"""
minflux_viewer.plugins.drift_correction.drift_correction_window
================================================================
Modeless UI for time-window auto-correlation **drift correction**.

The user picks a render resolution and a time window, clicks *Estimate drift*
to see the recovered drift trajectory (dx, dy, dz vs time), and *Create
corrected dataset* to add a new, drift-corrected dataset to the viewer (the
original is kept, so before/after can be compared). The estimation itself is a
faithful reimplementation of pyMINFLUX's ``drift_correction_time_windows_2d/3d``
(see :mod:`minflux_viewer.analysis.drift_correction`).

Parameters read from the MINFLUX data (last-valid localizations only):
    x, y[, z]  display coordinates in nm  (``xnm``/``ynm``/``znm``)
    t          time points in seconds     (``tim``)
    tid        trace ids                   (``tid``) — used for the auto window
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ...analysis import drift_correction as dc

_X_COLOR = "#ff5a5a"
_Y_COLOR = "#4caf50"
_Z_COLOR = "#5a9bff"


class DriftCorrectionWindow(QDialog):
    """Estimate and apply time-window auto-correlation drift correction."""

    def __init__(self, state, owner=None) -> None:
        super().__init__(None)                 # non-owned, modeless
        self._state = state
        self._owner = owner
        self._result: "dc.DriftResult | None" = None
        self._result_key = None                # (id(ds), dims, res, T) the result is for

        self.setWindowTitle("Drift Correction")
        self.resize(560, 640)
        self._build_ui()

        try:
            state.active_changed.connect(self._on_active_changed)
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

        params = QGroupBox("Parameters")
        grid = QGridLayout(params)

        grid.addWidget(QLabel("Analysis dimensions:"), 0, 0)
        self._dims_combo = QComboBox()
        self._dims_combo.addItems(["Auto (from dataset)", "2D (x, y)", "3D (x, y, z)"])
        self._dims_combo.currentIndexChanged.connect(self._on_dims_changed)
        grid.addWidget(self._dims_combo, 0, 1)

        grid.addWidget(QLabel("Render resolution (nm):"), 1, 0)
        self._res_spin = QDoubleSpinBox()
        self._res_spin.setRange(0.5, 100.0)
        self._res_spin.setDecimals(2)
        self._res_spin.setSingleStep(1.0)
        self._res_spin.setValue(5.0)
        self._res_spin.setToolTip(
            "Pixel/voxel size of the auto-correlation render (nm). Finer values "
            "resolve smaller drift but cost more memory/time.")
        grid.addWidget(self._res_spin, 1, 1)

        grid.addWidget(QLabel("Time window (s):"), 2, 0)
        tw_row = QHBoxLayout()
        self._auto_T = QCheckBox("Auto")
        self._auto_T.setChecked(True)
        self._auto_T.toggled.connect(self._on_auto_T_toggled)
        self._T_spin = QDoubleSpinBox()
        self._T_spin.setRange(1.0, 1e6)
        self._T_spin.setDecimals(1)
        self._T_spin.setSingleStep(60.0)
        self._T_spin.setValue(600.0)
        self._T_spin.setEnabled(False)
        self._T_spin.setToolTip(
            "Length of each time window (seconds). The acquisition is split into "
            "windows of this length; at least two are required.")
        tw_row.addWidget(self._auto_T)
        tw_row.addWidget(self._T_spin, 1)
        grid.addLayout(tw_row, 2, 1)

        root.addWidget(params)

        # Drift trajectory plot.
        self._plot = pg.PlotWidget()
        self._plot.setBackground("w")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setLabel("bottom", "time", units="s")
        self._plot.setLabel("left", "drift", units="nm")
        self._plot.addLegend(offset=(10, 10))
        root.addWidget(self._plot, 1)

        self._stats = QLabel("")
        self._stats.setWordWrap(True)
        self._stats.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self._stats)

        btns = QHBoxLayout()
        self._estimate_btn = QPushButton("Estimate drift")
        self._estimate_btn.clicked.connect(self._on_estimate)
        btns.addWidget(self._estimate_btn)
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
            "Method: time-window auto-correlation drift correction "
            "(reimplemented from pyMINFLUX, <i>pyminflux.correct</i>).<br>"
            "Ostersehlt et&nbsp;al., <i>DNA-PAINT MINFLUX nanoscopy</i>, "
            "Nat Methods 19, 1072-1075 (2022).</span>")
        refs.setWordWrap(True)
        refs.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(refs)

    # --------------------------------------------------------------- helpers
    def _active(self):
        return self._state.active_dataset

    def _chosen_dims(self, ds) -> int:
        idx = self._dims_combo.currentIndex()
        if idx == 1:
            return 2
        if idx == 2:
            return 3
        return 3 if int(getattr(ds.prop, "num_dim", 2)) >= 3 else 2

    def _gather(self, ds, dims):
        """Last-valid display coordinates (nm), time (s) and trace ids."""
        from ...core.loader import mfx_get
        x = mfx_get(ds, "xnm", itr="last", vld_only=True)
        y = mfx_get(ds, "ynm", itr="last", vld_only=True)
        z = mfx_get(ds, "znm", itr="last", vld_only=True) if dims == 3 else None
        t = mfx_get(ds, "tim", itr="last", vld_only=True)
        tid = mfx_get(ds, "tid", itr="last", vld_only=True)
        return x, y, z, t, tid

    def _validate(self, ds):
        if ds is None:
            return "No active dataset. Load a MINFLUX dataset and select it."
        from ...core.dataset_kind import is_minflux
        if not is_minflux(ds):
            return ("The active dataset is not a MINFLUX dataset (no trace ids), "
                    "so drift correction cannot use the time information.")
        t = None
        try:
            from ...core.loader import mfx_get
            t = mfx_get(ds, "tim", itr="last", vld_only=True)
        except Exception:
            pass
        if t is None or np.asarray(t).size == 0:
            return "The active dataset has no time information (tim)."
        t = np.asarray(t, dtype=float)
        if not np.isfinite(t).any() or np.ptp(t[np.isfinite(t)]) <= 0:
            return ("The active dataset's time points do not span a range, so "
                    "drift over time cannot be estimated.")
        return None

    # --------------------------------------------------------------- events
    def _on_active_changed(self, *_a) -> None:
        self._refresh_dataset_info()

    def _on_dims_changed(self, *_a) -> None:
        self._refresh_dataset_info()

    def _on_auto_T_toggled(self, checked: bool) -> None:
        self._T_spin.setEnabled(not checked)
        if checked:
            self._refresh_dataset_info()

    def _refresh_dataset_info(self) -> None:
        ds = self._active()
        if ds is None:
            self._info.setText("No active dataset.")
            self._estimate_btn.setEnabled(False)
            self._apply_btn.setEnabled(False)
            return
        self._estimate_btn.setEnabled(True)

        dims = self._chosen_dims(ds)
        n_dim = int(getattr(ds.prop, "num_dim", 2))
        n_loc = int(getattr(ds.prop, "num_loc", 0))
        n_tr = int(getattr(ds.prop, "num_traces", 0))

        span_txt = ""
        auto_T_txt = ""
        try:
            from ...core.loader import mfx_get
            t = np.asarray(mfx_get(ds, "tim", itr="last", vld_only=True), dtype=float)
            tid = np.asarray(mfx_get(ds, "tid", itr="last", vld_only=True))
            x = np.asarray(mfx_get(ds, "xnm", itr="last", vld_only=True), dtype=float)
            y = np.asarray(mfx_get(ds, "ynm", itr="last", vld_only=True), dtype=float)
            finite = np.isfinite(t)
            if finite.any():
                span = float(np.ptp(t[finite]))
                span_txt = f", {span:.0f}s acquisition"
                if self._auto_T.isChecked() and tid.size and x.size:
                    T = dc.default_time_window(
                        np.sort(t[finite]), tid[finite],
                        (float(np.nanmin(x)), float(np.nanmax(x))),
                        (float(np.nanmin(y)), float(np.nanmax(y))))
                    self._T_spin.blockSignals(True)
                    self._T_spin.setValue(T)
                    self._T_spin.blockSignals(False)
                    n_win = int(np.floor(span / T)) if T > 0 else 0
                    auto_T_txt = f" · auto window ≈ {T:.0f}s ({n_win} windows)"
        except Exception:
            pass

        self._info.setText(
            f"<b>{ds.name}</b><br>{n_loc:,} valid localizations · {n_tr:,} traces · "
            f"{n_dim}-D data · correcting in {dims}-D{span_txt}{auto_T_txt}")

    def _on_estimate(self) -> None:
        ds = self._active()
        err = self._validate(ds)
        if err:
            QMessageBox.information(self, "Drift Correction", err)
            return
        dims = self._chosen_dims(ds)
        res_nm = float(self._res_spin.value())
        T = None if self._auto_T.isChecked() else float(self._T_spin.value())

        try:
            x, y, z, t, tid = self._gather(ds, dims)
        except Exception as exc:
            QMessageBox.warning(self, "Drift Correction", f"Could not read data:\n{exc}")
            return
        if x is None or y is None or t is None or tid is None:
            QMessageBox.warning(self, "Drift Correction", "Could not read localizations.")
            return

        self._state.status_progress("Estimating drift")
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = dc.estimate_drift(
                x, y, z, t, tid, resolution_nm=res_nm, time_window_s=T, dims=dims)
        except Exception as exc:
            QGuiApplication.restoreOverrideCursor()
            self._state.status_progress("Drift estimation failed")
            self._state.log(f"Drift correction failed: {exc}", "ERROR")
            QMessageBox.warning(self, "Drift Correction", f"Drift estimation failed:\n{exc}")
            return
        QGuiApplication.restoreOverrideCursor()
        self._state.status_progress("Drift estimated", 1.0)

        self._result = result
        self._result_key = (id(ds), dims, res_nm, result.time_window_s)
        self._apply_btn.setEnabled(True)
        self._plot_result(result)
        self._report_result(ds, result)

    def _plot_result(self, result: "dc.DriftResult") -> None:
        self._plot.clear()
        legend = self._plot.plotItem.legend
        if legend is not None:
            legend.clear()
        ti = np.asarray(result.ti, dtype=float)

        def _series(dt, color, name):
            self._plot.plot(ti, np.asarray(dt, dtype=float),
                            pen=pg.mkPen(color, width=2),
                            symbol="o", symbolSize=6, symbolBrush=color,
                            name=name)

        _series(result.dxt, _X_COLOR, "x drift")
        _series(result.dyt, _Y_COLOR, "y drift")
        if result.dims == 3 and result.dzt is not None:
            _series(result.dzt, _Z_COLOR, "z drift")
        self._plot.enableAutoRange()
        self._plot.autoRange()

    def _report_result(self, ds, result: "dc.DriftResult") -> None:
        rms = result.rms
        if result.dims == 3:
            rms_txt = (f"RMS drift: x={rms[0]:.2f}, y={rms[1]:.2f}, z={rms[2]:.2f} nm")
            pp = (f"peak-to-peak: x={np.ptp(result.dxt):.1f}, "
                  f"y={np.ptp(result.dyt):.1f}, z={np.ptp(result.dzt):.1f} nm")
        else:
            rms_txt = f"RMS drift: x={rms[0]:.2f}, y={rms[1]:.2f} nm"
            pp = (f"peak-to-peak: x={np.ptp(result.dxt):.1f}, "
                  f"y={np.ptp(result.dyt):.1f} nm")
        self._stats.setText(
            f"{result.dims}-D · resolution {result.resolution_nm:g} nm · "
            f"time window {result.time_window_s:.0f} s · {result.n_windows} windows<br>"
            f"{rms_txt} · {pp}")
        self._state.log(
            f"Estimated {result.dims}-D drift for '{ds.name}': "
            f"{result.n_windows} time windows of {result.time_window_s:.0f}s, "
            f"resolution {result.resolution_nm:g} nm; {rms_txt}.",
            dataset_idx=self._state.active_idx)

    def _on_apply(self) -> None:
        ds = self._active()
        err = self._validate(ds)
        if err:
            QMessageBox.information(self, "Drift Correction", err)
            return
        dims = self._chosen_dims(ds)
        res_nm = float(self._res_spin.value())
        T = None if self._auto_T.isChecked() else float(self._T_spin.value())

        # Re-estimate if params/dataset changed since the last estimate.
        key = (id(ds), dims, res_nm, None if T is None else T)
        need = (self._result is None or self._result_key is None
                or self._result_key[:3] != key[:3])
        if need:
            self._on_estimate()
            if self._result is None:
                return
        result = self._result

        try:
            x, y, z, t, tid = self._gather(ds, dims)
            new_idx = self._build_corrected_dataset(ds, dims, result, x, y, z, t, tid)
        except Exception as exc:
            self._state.log(f"Drift correction apply failed: {exc}", "ERROR")
            QMessageBox.warning(self, "Drift Correction",
                                f"Could not create corrected dataset:\n{exc}")
            return
        if new_idx is None:
            QMessageBox.warning(self, "Drift Correction",
                                "No localizations remained after correction.")

    def _build_corrected_dataset(self, ds, dims, result, x, y, z, t, tid):
        from ...core.dataset import build_localization_dataset
        from ...core.loader import mfx_get

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x_corr = x - result.dx
        y_corr = y - result.dy
        if dims == 3:
            z = np.asarray(z, dtype=float)
            z_corr = z - result.dz
        else:
            z_corr = None

        n = x_corr.size
        # Carry over the canonical per-localization quality attributes (last-valid,
        # aligned with the coordinates). Derived attributes are recomputed by the
        # builder / post-load chain.
        carry = {}
        for name in ("efo", "cfr", "dcr", "eco", "ecc", "efc", "fbg",
                     "lcx", "lcy", "lcz", "ext", "sta"):
            try:
                arr = mfx_get(ds, name, itr="last", vld_only=True)
            except Exception:
                arr = None
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim == 1 and arr.shape[0] == n:
                carry[name] = arr

        new = build_localization_dataset(
            name=f"{ds.name} (drift corrected)",
            x_nm=x_corr, y_nm=y_corr, z_nm=z_corr,
            tid=np.asarray(tid), tim=np.asarray(t, dtype=float),
            attrs=carry or None,
            source_version=str(ds.metadata.get("source_version", "imported")),
            prefs=self._state.prefs)

        # Coordinates are final (drift baked, RIMF already applied in znm) — pin
        # RIMF to 1.0 and suppress the post-load auto-anisotropy estimate.
        try:
            new.set_rimf(1.0, source="processed (drift corrected, coordinates final)")
            new.derived["rimf"] = np.asarray([1.0], dtype=float)
        except Exception:
            pass
        import uuid
        new.file.folder = f"<drift-corrected>/{uuid.uuid4().hex}"
        new.metadata["drift_corrected"] = True
        new.metadata["drift_correction"] = {
            "dims": result.dims,
            "resolution_nm": result.resolution_nm,
            "time_window_s": result.time_window_s,
            "n_windows": result.n_windows,
            "rms_nm": list(result.rms),
            "source": ds.name,
        }

        idx = self._state.add_dataset(new)
        rms = result.rms
        rms_txt = (", ".join(f"{ax}={v:.2f}" for ax, v in zip("xyz", rms)))
        self._state.log(
            f"Created drift-corrected dataset '{new.name}' from '{ds.name}' "
            f"({result.dims}-D, {result.n_windows} windows of "
            f"{result.time_window_s:.0f}s, resolution {result.resolution_nm:g} nm; "
            f"RMS drift {rms_txt} nm).",
            dataset_idx=idx)
        return idx

    # --------------------------------------------------------------- close
    def closeEvent(self, event) -> None:
        try:
            self._state.active_changed.disconnect(self._on_active_changed)
        except Exception:
            pass
        super().closeEvent(event)
