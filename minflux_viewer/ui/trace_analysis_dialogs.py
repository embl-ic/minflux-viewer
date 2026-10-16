"""Result dialogs and launchers for numerical analysis."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..analysis import trace_analysis as _calculation
from ..analysis.trace_analysis import _AXIS_COLORS, _DEFAULT_MODE, _TRACE_SIZE_COLORS, SizeFitResult
from ..core.coordinates import coordinate_view


def show_trace_size_dialog(parent: QWidget, ds, state) -> None:
    """Run trace size estimation on *ds* and display a result dialog."""

    # --- collect data ---
    try:
        loc = coordinate_view(ds)   # (N,3) in nm
    except Exception as exc:
        QMessageBox.warning(parent, "Trace size", f"Cannot read localisations:\n{exc}")
        return

    tid_raw = ds.attr.get("tid")
    if tid_raw is None:
        QMessageBox.warning(parent, "Trace size",
                            "Dataset has no trace ID (tid) attribute.\n"
                            "Load a MINFLUX dataset with trace information.")
        return

    mask = np.asarray(ds.filter_mask, dtype=bool)
    if mask.shape[0] == loc.shape[0]:
        loc = loc[mask]
        tid = np.asarray(tid_raw, dtype=np.int64).ravel()[mask]
    else:
        tid = np.asarray(tid_raw, dtype=np.int64).ravel()

    n_traces = int(np.unique(tid[tid != 0]).size)
    if n_traces == 0:
        QMessageBox.warning(parent, "Trace size", "No valid traces found (all IDs are 0).")
        return

    # --- compute (same log-distance method as the anisotropy plugin) ---
    # XY  = lateral distance to the trace's XY centroid (loc x/y only).
    # Z   = axial offset using the CALIBRATED z (ds.loc_nm already applied Z scaling factor).
    # XYZ = full 3-D Euclidean distance.
    try:
        result = {
            "xy":  _calculation.estimate_size_nd_details(loc[:, [0, 1]], tid),
            "z":   _calculation.estimate_size_nd_details(loc[:, 2], tid),
            "xyz": _calculation.estimate_size_nd_details(loc[:, :3], tid),
            "mode": _DEFAULT_MODE,
        }
    except Exception as exc:
        QMessageBox.critical(parent, "Trace size", f"Computation failed:\n{exc}")
        return

    # Modeless result window (project convention for analysis output): does
    # not block the app and does not sit permanently in front of the viewers.
    from .modeless import show_modeless
    show_modeless(_TraceSizeDialog(None, result, n_traces, loc.shape[0]), parent)


def show_anisotropy_dialog(parent: QWidget, ds, state=None) -> None:
    """Estimate a Z scaling factor from trace anisotropy and show the result.

    Uses the **raw** last-valid-iteration ``loc`` coordinates (nm), not the
    Z-scaled ``ds.loc_nm`` and not the interactive filter — so the
    estimate is reproducible and not contaminated by a previously applied
    z-scaling. ``state`` (AppState) enables the dialog's "Apply to data"
    button to set the dataset's Z scaling factor and refresh open views.
    """
    from ..core.dataset_kind import is_3d

    if not is_3d(ds):
        QMessageBox.warning(
            parent, "Z Scaling Factor",
            "Z scaling factor estimation requires a 3-D dataset (non-degenerate Z).",
        )
        return
    if ds.attr.get("tid") is None and ds.mfx_raw.get("tid") is None:
        QMessageBox.warning(parent, "Z Scaling Factor",
                            "Dataset has no trace ID (tid) attribute.")
        return

    data = _calculation.raw_last_valid_loc_nm(ds)
    if data is None:
        QMessageBox.warning(
            parent, "Z Scaling Factor", "Cannot read localisations (loc x/y/z)."
        )
        return
    loc, tid = data

    if loc.shape[1] < 3:
        QMessageBox.warning(parent, "Z Scaling Factor",
                            "Z scaling factor estimation requires 3-D data (x, y, z).")
        return

    n_traces = int(np.unique(tid[tid != 0]).size)
    if n_traces == 0:
        QMessageBox.warning(parent, "Z Scaling Factor", "No valid traces found.")
        return

    # Check for actual finite Z variation as a defensive guard for raw data
    # containing missing Z measurements.
    finite_z = loc[:, 2][np.isfinite(loc[:, 2])]
    if finite_z.size == 0 or float(np.ptp(finite_z)) < 1.0:
        QMessageBox.warning(parent, "Z Scaling Factor",
                            "Z range is < 1 nm — this appears to be 2-D data.\n"
                            "Z scaling factor estimation requires 3-D acquisitions.")
        return

    try:
        result = _calculation.estimate_anisotropy_details(loc, tid)
    except Exception as exc:
        QMessageBox.critical(parent, "Z Scaling Factor", f"Computation failed:\n{exc}")
        return

    # Modeless result window (see show_trace_size_dialog).
    from .modeless import show_modeless
    show_modeless(_AnisotropyDialog(None, result, n_traces, ds=ds, state=state), parent)


def _nm(val: float) -> str:
    if not np.isfinite(val):
        return "—"
    return f"{val:.2f} nm"


class _TraceSizeDialog(QDialog):
    def __init__(
        self,
        parent: QWidget,
        result: dict,
        n_traces: int,
        n_locs: int,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Estimated Average Trace Size")
        self.resize(980, 640)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        mode = int(result.get("mode", _DEFAULT_MODE))
        fit_xy, fit_z, fit_xyz = result["xy"], result["z"], result["xyz"]
        size_xyz = float(fit_xyz.sizes[mode]) if mode < fit_xyz.sizes.size else np.nan

        info = QLabel(f"<b>{n_traces}</b> traces &nbsp;|&nbsp; <b>{n_locs:,}</b> localizations")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(info)

        headline = QLabel(
            f"<span style='font-size:18px'><b>average trace size = {_nm(size_xyz)}</b></span>"
            "<span style='color:gray'> &nbsp;(Gaussian fit, 3-D)</span>"
        )
        headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(headline)

        # Per-mode size table — rows = modes, columns = XY / Z / XYZ.
        grp = QGroupBox("Trace size per mode (nm) — Gaussian fit is the final value")
        grid = QGridLayout(grp)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(4)
        for col, h in enumerate(("", "XY", "Z", "XYZ")):
            lbl = QLabel(f"<b>{h}</b>")
            lbl.setAlignment(
                Qt.AlignmentFlag.AlignLeft if col == 0 else Qt.AlignmentFlag.AlignCenter
            )
            grid.addWidget(lbl, 0, col)
        row_labels = ["Mean", "Median", "Max-bin", "Gaussian fit ★"]
        for i, row_lbl in enumerate(row_labels):
            bold = i == _DEFAULT_MODE
            name = QLabel(f"<b>{row_lbl}:</b>" if bold else f"{row_lbl}:")
            name.setAlignment(Qt.AlignmentFlag.AlignLeft)
            grid.addWidget(name, i + 1, 0)
            cells = [_nm(fit_xy.sizes[i]), _nm(fit_z.sizes[i]), _nm(fit_xyz.sizes[i])]
            for col, text in enumerate(cells, start=1):
                cell = QLabel(f"<b>{text}</b>" if bold else text)
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                grid.addWidget(cell, i + 1, col)
        for col in range(1, 4):
            grid.setColumnStretch(col, 1)
        root.addWidget(grp)

        # Log-distance histograms with Gaussian fits — XY / Z / XYZ in one row.
        try:
            hist_group = QGroupBox("Log-distance histograms with Gaussian fits")
            row = QHBoxLayout(hist_group)
            row.setContentsMargins(8, 8, 8, 8)
            for axis, fit in (("XY", fit_xy), ("Z", fit_z), ("XYZ", fit_xyz)):
                row.addWidget(_anisotropy_hist_plot(axis, fit, _TRACE_SIZE_COLORS[axis], mode))
            root.addWidget(hist_group, stretch=1)
        except Exception as exc:
            warn = QLabel(f"Histogram report could not be created: {exc}")
            warn.setStyleSheet("color: #a66;")
            root.addWidget(warn)

        note = QLabel(
            "d — each localization's offset from its trace centroid (Δ = coord − "
            "centroid), in nm:<br>"
            "&nbsp;&nbsp;XY — radial distance in the XY plane: d = √( Δx² + Δy² )<br>"
            "&nbsp;&nbsp;Z — axial offset using the calibrated z (Z scaling factor applied): "
            "d = | Δz |<br>"
            "&nbsp;&nbsp;XYZ — full 3-D Euclidean distance: "
            "d = √( Δx² + Δy² + Δz² )<br>"
            "size — peak of a log-normal fit to the distribution of d; "
            "the Gaussian fit (★) is the reported value."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(note)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

        from .text_select import make_labels_selectable
        make_labels_selectable(self)   # let users copy the reported values

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        """Retire this window's pyqtgraph plots before Qt deletes them."""
        from .qt_lifecycle import dispose_plot_widgets

        dispose_plot_widgets(self)
        super().closeEvent(event)


class _AnisotropyDialog(QDialog):
    def __init__(self, parent: QWidget, result: dict, n_traces: int,
                 *, ds=None, state=None) -> None:
        super().__init__(parent)
        self._ds = ds
        self._state = state
        self.setWindowTitle("Estimated Z Scaling Factor")
        self.resize(980, 800)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        z_scaling_factor = float(result.get("z_scaling_factor", np.nan))
        self._z_scaling_factor = z_scaling_factor
        z_scaling_factor_by_mode = result.get("z_scaling_factor_by_mode", [np.nan, np.nan, np.nan, z_scaling_factor])
        sizes_x = result["x"].sizes
        sizes_y = result["y"].sizes
        sizes_z = result["z"].sizes

        info = QLabel(f"<b>{n_traces}</b> traces used for estimation")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(info)

        # Abnormal when outside the physically expected [0.5, 1.0] window.
        abnormal = bool(np.isfinite(z_scaling_factor)) and not (0.5 <= z_scaling_factor <= 1.0)
        val_txt = f"{z_scaling_factor:.4f}" if np.isfinite(z_scaling_factor) else "—"
        val_color = "#ff5a5a" if abnormal else "palette(text)"
        z_scaling_factor_lbl = QLabel(
            f"<span style='font-size:18px; color:{val_color}'>"
            f"<b>Z scaling factor = {val_txt}</b></span>"
            "<span style='color:gray'> &nbsp;(Gaussian fit)</span>"
        )
        z_scaling_factor_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        z_scaling_factor_lbl.setToolTip(
            "Z scaling factor = geomean(σx, σy) / σz = √(σx·σy) / σz\n"
            "Applied as: z_calibrated = z_raw × z_scaling_factor"
        )
        root.addWidget(z_scaling_factor_lbl)

        if abnormal:
            warn_lbl = QLabel("abnormal value, check the data!")
            warn_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            warn_lbl.setStyleSheet("color: #ff5a5a; font-size: 11px;")
            root.addWidget(warn_lbl)

        grp = QGroupBox(
            "Per-axis size estimates (nm) and Z scaling factor per mode — "
            "Gaussian fit is the final value"
        )
        # A real grid keeps the X / Y / Z / factor columns aligned
        # vertically with their value cells (a QFormLayout's field column would
        # not line the header row up with the value rows).
        grid = QGridLayout(grp)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(4)

        col_headers = ["", "X", "Y", "Z", "Z scaling factor"]
        for col, h in enumerate(col_headers):
            lbl = QLabel(f"<b>{h}</b>")
            lbl.setAlignment(
                Qt.AlignmentFlag.AlignLeft if col == 0 else Qt.AlignmentFlag.AlignCenter
            )
            grid.addWidget(lbl, 0, col)

        row_labels = ["Mean", "Median", "Max-bin", "Gaussian fit ★"]
        for i, (row_lbl, vx, vy, vz) in enumerate(zip(row_labels, sizes_x, sizes_y, sizes_z)):
            bold = i == _DEFAULT_MODE
            rmode = float(z_scaling_factor_by_mode[i]) if i < len(z_scaling_factor_by_mode) else np.nan
            name = QLabel(f"<b>{row_lbl}:</b>" if bold else f"{row_lbl}:")
            name.setAlignment(Qt.AlignmentFlag.AlignLeft)
            grid.addWidget(name, i + 1, 0)
            cells = [_nm(vx), _nm(vy), _nm(vz),
                     f"{rmode:.4f}" if np.isfinite(rmode) else "—"]
            for col, text in enumerate(cells, start=1):
                cell = QLabel(f"<b>{text}</b>" if bold else text)
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                grid.addWidget(cell, i + 1, col)
        for col in range(1, 5):
            grid.setColumnStretch(col, 1)
        root.addWidget(grp)

        try:
            hist_group = QGroupBox("Log-distance histograms with Gaussian fits")
            grid = QGridLayout(hist_group)
            grid.setContentsMargins(8, 8, 8, 8)
            fits = [
                ("X", result["x"], _AXIS_COLORS["X"]),
                ("Y", result["y"], _AXIS_COLORS["Y"]),
                ("Z", result["z"], _AXIS_COLORS["Z"]),
            ]
            for pos, (axis, fit, color) in enumerate(fits):
                grid.addWidget(_anisotropy_hist_plot(axis, fit, color, result["mode"]), pos // 2, pos % 2)
            grid.addWidget(_anisotropy_overlay_plot(fits, result["mode"]), 1, 1)
            root.addWidget(hist_group, stretch=1)
        except Exception as exc:
            warn = QLabel(f"Histogram report could not be created: {exc}")
            warn.setStyleSheet("color: #a66;")
            root.addWidget(warn)

        failed = [
            axis for axis, fit in (("X", result["x"]), ("Y", result["y"]), ("Z", result["z"]))
            if fit.fit_params is None
        ]
        fit_note = ""
        if failed:
            fit_note = (
                f"<br>Gaussian fit fallback used for: {', '.join(failed)} "
                "(reported Gaussian value equals max-bin estimate)."
            )
        note = QLabel(
            "d — each localization's offset from its trace centroid along one axis "
            "(d = | coord − centroid |), in nm, for X, Y, or Z<br>"
            "size σ — peak of a log-normal fit to the distribution of d over all "
            "localizations<br>"
            "Z scaling factor — geomean(σ<sub>x</sub>, σ<sub>y</sub>) ∕ σ<sub>z</sub> "
            "= √( σ<sub>x</sub> · σ<sub>y</sub> ) ∕ σ<sub>z</sub><br>"
            "<br>"
            "Use the Gaussian-fit (★) estimate. This empirical factor is derived "
            "from trace anisotropy; it is not calculated from refractive indices."
            f"{fit_note}"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(note)

        # ── Apply row: editable Z scaling factor (defaults to the computed value) + Apply ──
        btn_row = QHBoxLayout()
        self._apply_status = QLabel("")
        self._apply_status.setStyleSheet("color: #6c6;")
        btn_row.addWidget(self._apply_status)
        btn_row.addStretch()
        btn_row.addWidget(QLabel("Z scaling factor:"))
        from .z_scaling_widgets import (
            Z_SCALING_ARROW_STEP,
            Z_SCALING_WHEEL_STEP,
            ZScalingFactorSpinBox,
        )

        self._z_scaling_factor_spin = ZScalingFactorSpinBox(minimum=0.0, maximum=100.0)
        self._z_scaling_factor_spin.setValue(z_scaling_factor if np.isfinite(z_scaling_factor) else 1.0)
        self._z_scaling_factor_spin.setToolTip(
            "Dimensionless multiplier applied to this dataset's raw z coordinates "
            "(defaults to the trace-anisotropy estimate; editable). "
            f"Arrows step by {Z_SCALING_ARROW_STEP:g}, the mouse wheel by "
            f"{Z_SCALING_WHEEL_STEP:g}."
        )
        btn_row.addWidget(self._z_scaling_factor_spin)
        self._apply_btn = QPushButton("Apply to data")
        self._apply_btn.setToolTip(
            "Set this dataset's Z scaling factor and refresh open 3-D views. "
            "Raw z is not modified."
        )
        self._apply_btn.setEnabled(self._ds is not None and np.isfinite(z_scaling_factor))
        self._apply_btn.clicked.connect(self._apply_to_data)
        btn_row.addWidget(self._apply_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

        from .text_select import make_labels_selectable
        make_labels_selectable(self)   # let users copy the factor and size values

    def _apply_to_data(self) -> None:
        from .z_scaling_widgets import format_z_scaling_factor

        if self._ds is None:
            return
        value = float(self._z_scaling_factor_spin.value())
        self._ds.set_z_scaling_factor(
            value, source="manual (trace-anisotropy dialog)"
        )
        if self._state is not None:
            idx = next(
                (i for i, d in enumerate(self._state.datasets) if d is self._ds),
                None,
            )
            if idx is not None:
                # Refresh render, scatter, 3-D, and the dataset-info Z scaling factor text
                # (they re-pull the Z-scaled loc_nm / invalidate caches).
                self._state.notify_calibration_changed(idx)
            try:
                self._state.log(
                    f"Applied Z scaling factor = "
                    f"{format_z_scaling_factor(value)} to "
                    f"'{self._ds.name}' (trace-anisotropy dialog)."
                )
            except Exception:
                pass
        self._apply_status.setText(
            f"Applied Z scaling factor = {format_z_scaling_factor(value)}")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        """Retire this window's pyqtgraph plots before Qt deletes them."""
        from .qt_lifecycle import dispose_plot_widgets

        dispose_plot_widgets(self)
        super().closeEvent(event)


def _plot_color(color) -> QColor:
    """Normalize any colour spec these plot helpers are handed to a ``QColor``.

    ⚠ They used to build their translucent fill by **string concatenation**
    (``color + "66"``), which silently required every caller to pass a hex
    string. When ``localization_precision._precision_color`` moved to the global
    colour registry it began returning a ``QColor``, so ``QColor + str`` raised
    ``TypeError`` and the CRLB / StdDev histogram groups were replaced by their
    "Histogram report could not be created" fallback label. Accepting the three
    forms that actually occur — hex string, ``QColor``, RGB(A) tuple — is what
    stops the alpha from being a property of the caller's colour *encoding*.
    """
    from PyQt6.QtGui import QColor

    if isinstance(color, QColor):
        return QColor(color)
    if isinstance(color, (tuple, list)):
        return QColor(*color)
    return QColor(str(color))


def _fill_brush(color, alpha: int):
    """A translucent fill brush of *color* at *alpha* (0-255)."""
    import pyqtgraph as pg

    c = _plot_color(color)
    c.setAlpha(int(alpha))
    return pg.mkBrush(c)


def _anisotropy_hist_plot(axis: str, fit: SizeFitResult, color: str, mode: int):
    import pyqtgraph as pg

    pw = pg.PlotWidget(background="#222")
    pw.setMinimumHeight(230)
    pw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    pw.showGrid(x=True, y=True, alpha=0.22)
    pw.setTitle(f"{axis}: log(distance)")
    pw.setLabel("bottom", "log(distance in nm)")
    pw.setLabel("left", "count")
    if fit.counts.size == 0:
        return pw
    pw.plot(
        fit.bin_centers,
        fit.counts,
        stepMode=False,
        fillLevel=0,
        brush=_fill_brush(color, 0x66),
        pen=pg.mkPen(_plot_color(color), width=1.2),
    )
    _plot_gaussian_fit(pw, fit, color)
    _plot_size_marker(pw, fit, mode, color, axis)
    return pw


def _anisotropy_overlay_plot(fits: list[tuple[str, SizeFitResult, str]], mode: int):
    import pyqtgraph as pg

    pw = pg.PlotWidget(background="#222")
    pw.setMinimumHeight(230)
    pw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    pw.showGrid(x=True, y=True, alpha=0.22)
    pw.setTitle("Overlay, Z' = geomean( X , Y )")
    pw.setLabel("bottom", "log(distance in nm)")
    pw.setLabel("left", "count")
    # No legend / values: the per-axis colors match the individual plots and
    # the markers are labelled X/Y/Z. The individual plots already show the
    # histograms; here we overlay only the smooth Gaussian curves (filled).
    peaks: list[dict] = []          # per-axis: position / height / width
    bin_spans: list[float] = []
    sizes_by_axis: dict[str, float] = {}
    y_ref = 0.0
    for axis, fit, color in fits:
        if fit.counts.size == 0:
            continue
        _plot_gaussian_fit(pw, fit, color, fill=True)
        # Markers without values (X/Y/Z letters only).
        _plot_size_marker(pw, fit, mode, color, axis, label_value=False)

        bin_spans.append(float(fit.bin_centers[-1] - fit.bin_centers[0]))
        height = abs(float(fit.fit_params[0])) if fit.fit_params is not None else float(fit.counts.max())
        y_ref = max(y_ref, height)
        val = float(fit.sizes[mode]) if mode < fit.sizes.size else np.nan
        if np.isfinite(val) and val > 0:
            sizes_by_axis[axis] = val
            width = abs(float(fit.fit_params[2])) if fit.fit_params is not None else 0.3
            peaks.append({"axis": axis, "pos": float(np.log(val)), "height": height, "width": width})

    # Z' = geomean(size_x, size_y): the effective in-plane size used in
    # Z scaling factor = Z' / size_z. Distinct orange dash-dot line so it stands apart.
    sx, sy = sizes_by_axis.get("X"), sizes_by_axis.get("Y")
    if sx and sy and sx > 0 and sy > 0:
        z_prime = float(np.exp((np.log(sx) + np.log(sy)) / 2.0))
        zp_line = pg.InfiniteLine(
            pos=float(np.log(z_prime)),
            angle=90,
            pen=pg.mkPen("#ff9a33", width=1.6, style=Qt.PenStyle.DashDotLine),
            label=f"Z' = {z_prime:.2f} nm",
            labelOpts={"position": 0.82, "color": "#ff9a33", "fill": (20, 20, 20, 200)},
        )
        pw.addItem(zp_line)

    if len(peaks) < 2 or y_ref <= 0:
        return pw

    # Match the individual plots' data aspect (Δx : Δy) so the Gaussian tops
    # are not distorted — they share this widget's pixel size. Use a robust
    # x reference (median bin span) so an outlier tail on one axis does not
    # skew it. Then zoom to the peak tops at that aspect.
    aspect = float(np.median(bin_spans)) / y_ref

    def zoom_box(sel: list[dict]) -> tuple[float, float, float, float]:
        positions = [p["pos"] for p in sel]
        # ~0.6 Gaussian widths of context shows the rounded tops, not flat tips.
        half_w = (max(positions) - min(positions)) / 2.0 + 0.6 * max(p["width"] for p in sel)
        center = 0.5 * (min(positions) + max(positions))
        height = (2.0 * half_w) / aspect                   # aspect-preserving
        y_top = 1.06 * max(p["height"] for p in sel)
        return center - half_w, center + half_w, max(0.0, y_top - height), y_top

    # All three first; if a peak top falls outside the aspect-preserving box,
    # the peaks are too spread — keep Z plus whichever of X/Y is closer to it.
    shown = peaks
    box = zoom_box(shown)
    if not all(box[0] <= p["pos"] <= box[1] and p["height"] >= box[2] for p in shown):
        z = next((p for p in peaks if p["axis"] == "Z"), peaks[-1])
        others = [p for p in peaks if p is not z]
        nearer = min(others, key=lambda p: abs(p["pos"] - z["pos"])) if others else None
        shown = [z] + ([nearer] if nearer else [])
        box = zoom_box(shown)

    pw.setXRange(box[0], box[1], padding=0)
    pw.setYRange(box[2], box[3], padding=0)
    return pw


def _plot_gaussian_fit(pw, fit: SizeFitResult, color: str, *, name: str | None = None,
                       fill: bool = False) -> None:
    if fit.fit_params is None or fit.edges.size < 2:
        return
    import pyqtgraph as pg

    x = np.linspace(float(fit.edges[0]), float(fit.edges[-1]), 400)
    y = _calculation._gauss1(x, *fit.fit_params)
    pw.plot(
        x, y,
        pen=pg.mkPen(_plot_color(color), width=2.0),
        fillLevel=0.0 if fill else None,
        brush=_fill_brush(color, 0x22) if fill else None,
        name=name,
    )


def _plot_size_marker(
    pw,
    fit: SizeFitResult,
    mode: int,
    color: str,
    axis: str,
    *,
    alpha: int = 220,
    label_value: bool = True,
) -> None:
    val = float(fit.sizes[mode]) if mode < fit.sizes.size else np.nan
    if not np.isfinite(val) or val <= 0:
        return
    import pyqtgraph as pg

    label = f"{axis} = {val:.2f} nm" if label_value else axis
    line = pg.InfiniteLine(
        pos=float(np.log(val)),
        angle=90,
        pen=pg.mkPen(color, width=1.4, style=Qt.PenStyle.DashLine),
        label=label,
        labelOpts={"position": 0.92, "color": color, "fill": (20, 20, 20, alpha)},
    )
    pw.addItem(line)


def _value_hist_plot(
    title: str,
    values: np.ndarray,
    color: str,
    *,
    marker: float | None = None,
    marker_label: str | None = None,
    x_label: str = "value (nm)",
    y_label: str = "count",
    n_bins: int = 40,
    clip_percentile: float | None = None,
):
    """Anisotropy-style histogram panel of a value distribution (linear axis).

    Used by reports that summarize a distribution with a single statistic
    (e.g. localization precision = median per-trace sigma) rather than a
    log-normal fit. Optionally draws a dashed marker at ``marker``. ``y_label``
    must match what each row of ``values`` is (e.g. "traces" vs "localizations").
    ``clip_percentile`` bins only up to that percentile, so a long sparse tail
    cannot squeeze the distribution into one bar; the title says how much is
    shown.
    """
    import pyqtgraph as pg

    pw = pg.PlotWidget(background="#222")
    pw.setMinimumHeight(230)
    pw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    pw.showGrid(x=True, y=True, alpha=0.22)
    pw.setTitle(title)
    pw.setLabel("bottom", x_label)
    pw.setLabel("left", y_label)

    vals = np.asarray(values, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return pw
    hist_range = None
    if clip_percentile is not None and vals.size > 1:
        hi = float(np.percentile(vals, clip_percentile))
        lo = float(vals.min())
        if hi > lo and hi < float(vals.max()):
            hist_range = (lo, hi)
            pw.setTitle(f"{title} ({clip_percentile:g}% shown)")
    counts, edges = np.histogram(vals, bins=n_bins, range=hist_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pw.plot(
        centers, counts,
        stepMode=False, fillLevel=0,
        brush=_fill_brush(color, 0x66), pen=pg.mkPen(_plot_color(color), width=1.2),
    )
    if marker is not None and np.isfinite(marker):
        line = pg.InfiniteLine(
            pos=float(marker), angle=90,
            pen=pg.mkPen(color, width=1.4, style=Qt.PenStyle.DashLine),
            label=marker_label or f"{marker:.2f} nm",
            labelOpts={"position": 0.92, "color": color, "fill": (20, 20, 20, 220)},
        )
        pw.addItem(line)
    return pw


def _widget_from_layout(layout: QHBoxLayout) -> QWidget:
    w = QWidget()
    w.setLayout(layout)
    return w
