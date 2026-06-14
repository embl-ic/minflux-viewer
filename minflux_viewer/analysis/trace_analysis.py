"""
minflux_viewer.analysis.trace_analysis
=======================================
Trace-level size and anisotropy estimation.

Python port of the MATLAB functions ``estimate_size_ND`` and
``estimate_anisotropy`` (Wanlu Liu / Abberior group).

Public API
----------
estimate_size_nd(data, ids, ...)
    Fit a log-normal distribution to intra-cluster distances and return
    four size estimates (mean, median, max-bin, Gaussian-fit peak).

estimate_anisotropy(loc_nm, tid, ...)
    Estimate the z-scaling factor (rimf) from per-axis size ratios.

show_trace_size_dialog(parent, ds, state)
show_anisotropy_dialog(parent, ds)
    Convenience wrappers that pull data from a MinfluxDataset and display
    a result dialog.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
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


# ---------------------------------------------------------------------------
# Core computation — estimate_size_nd
# ---------------------------------------------------------------------------

@dataclass
class SizeFitResult:
    sizes: np.ndarray
    logdist: np.ndarray
    counts: np.ndarray
    edges: np.ndarray
    bin_centers: np.ndarray
    fit_params: tuple[float, float, float] | None


def _gauss1(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """MATLAB ``fit(..., 'gauss1')`` model: a*exp(-((x-b)/c)^2)."""
    return float(a) * np.exp(-((x - float(b)) / max(abs(float(c)), 1e-9)) ** 2)


def estimate_size_nd_details(
    data: np.ndarray,
    ids: np.ndarray,
    *,
    bin_width: float = 0.1,
    exclude_zero_id: bool = True,
) -> SizeFitResult:
    """Detailed version of :func:`estimate_size_nd` with histogram/fit data."""
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    ids = np.asarray(ids, dtype=np.int64).ravel()

    uid = np.unique(ids)
    if exclude_zero_id:
        uid = uid[uid != 0]

    empty = SizeFitResult(
        sizes=np.full(4, np.nan),
        logdist=np.array([]),
        counts=np.array([]),
        edges=np.array([]),
        bin_centers=np.array([]),
        fit_params=None,
    )
    if uid.size == 0:
        return empty

    parts: list[np.ndarray] = []
    for cid in uid:
        mask = ids == cid
        pts = data[mask]
        if len(pts) < 2:
            continue
        centroid = np.median(pts, axis=0)
        parts.append(np.linalg.norm(pts - centroid, axis=1))

    if not parts:
        return empty

    dist = np.concatenate(parts)
    dist = dist[np.isfinite(dist) & (dist > 0)]
    if dist.size < 3:
        return empty

    logdist = np.log(dist)
    sizes = np.full(4, np.nan)
    sizes[0] = np.exp(np.mean(logdist))
    sizes[1] = np.exp(np.median(logdist))

    lo = float(np.nanmin(logdist))
    hi = float(np.nanmax(logdist))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return SizeFitResult(sizes, logdist, np.array([]), np.array([]), np.array([]), None)

    # MATLAB ``histcounts(logdist, 'BinWidth', bw)`` aligns bin edges to integer
    # multiples of the bin width; reproduce that exactly so the max-bin and
    # Gaussian-fit estimates match ``estimate_size_ND.m``.
    e_lo = np.floor(lo / bin_width) * bin_width
    e_hi = np.ceil(hi / bin_width) * bin_width
    edges = np.arange(e_lo, e_hi + bin_width / 2.0, bin_width, dtype=float)
    if edges.size < 3:
        edges = np.linspace(e_lo, e_hi + bin_width, 10)
    counts, edges = np.histogram(logdist, bins=edges)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    if counts.size == 0:
        return SizeFitResult(sizes, logdist, counts, edges, bin_centers, None)

    max_idx = int(np.argmax(counts))
    sizes[2] = np.exp((edges[max_idx] + edges[max_idx + 1]) / 2.0)

    fit_params = None
    try:
        # MATLAB ``fit(bincenter, counts, 'gauss1')`` fits ALL bins, including
        # the zero-count tails — do the same.
        fit_x = bin_centers
        fit_y = counts.astype(float)
        p0 = [
            float(counts[max_idx]),
            float(bin_centers[max_idx]),
            max(float(np.nanstd(logdist)), 0.1),
        ]
        popt, _ = curve_fit(_gauss1, fit_x, fit_y, p0=p0, maxfev=10000)
        fit_params = (float(popt[0]), float(popt[1]), float(popt[2]))
        sizes[3] = np.exp(fit_params[1])
    except Exception:
        sizes[3] = sizes[2]

    return SizeFitResult(sizes, logdist, counts, edges, bin_centers, fit_params)


def estimate_size_nd(
    data: np.ndarray,
    ids: np.ndarray,
    *,
    bin_width: float = 0.1,
    exclude_zero_id: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate cluster / trace size from intra-cluster distance distribution.

    Port of MATLAB ``estimate_size_ND``.

    Parameters
    ----------
    data : array_like, shape (N,) or (N, M)
        Coordinates in nm. Pass the full N×3 array for 3-D size, or a
        single column for per-axis size.
    ids : array_like, shape (N,)
        Cluster / trace ID per row.  IDs equal to 0 are excluded by default.
    bin_width : float
        Histogram bin width for the log-distance distribution.
    exclude_zero_id : bool
        If True (default), rows whose ID is 0 are ignored.

    Returns
    -------
    sizes : ndarray of shape (4,)
        ``[mean, median, max_bin, gaussian_fit]`` in nm.
    logdist : ndarray
        Log-transformed distances used for the fit (useful for plotting).
    """
    result = estimate_size_nd_details(
        data,
        ids,
        bin_width=bin_width,
        exclude_zero_id=exclude_zero_id,
    )
    return result.sizes, result.logdist


# ---------------------------------------------------------------------------
# Core computation — estimate_anisotropy
# ---------------------------------------------------------------------------

def estimate_anisotropy(
    loc_nm: np.ndarray,
    tid: np.ndarray,
    *,
    mode: int = 3,
    bin_width: float = 0.1,
    exclude_zero_id: bool = False,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the z-scaling factor (rimf) from per-axis trace sizes.

    Port of MATLAB ``estimate_anisotropy``.

    Parameters
    ----------
    loc_nm : (N, 3) ndarray
        Localizations in **nanometres** (x, y, z columns).
    tid : (N,) ndarray
        Trace / cluster IDs.
    mode : int
        Which size estimate to use: 0=mean, 1=median, 2=max-bin, 3=Gauss-fit.
    bin_width, exclude_zero_id : passed through to estimate_size_nd.

    Returns
    -------
    rimf : float
        z-scaling factor = geomean(size_x, size_y) / size_z.
    sizes_x, sizes_y, sizes_z : ndarray (4,)
        The four size estimates for each axis.
    """
    loc_nm = np.asarray(loc_nm, dtype=float)
    tid = np.asarray(tid).ravel()

    result = estimate_anisotropy_details(
        loc_nm,
        tid,
        mode=mode,
        bin_width=bin_width,
        exclude_zero_id=exclude_zero_id,
    )
    return (
        result["rimf"],
        result["x"].sizes,
        result["y"].sizes,
        result["z"].sizes,
    )


def estimate_anisotropy_details(
    loc_nm: np.ndarray,
    tid: np.ndarray,
    *,
    mode: int = 3,
    bin_width: float = 0.1,
    exclude_zero_id: bool = False,
) -> dict:
    """Detailed anisotropy result including per-axis log-histogram fits.

    ``exclude_zero_id`` defaults to ``False`` to match ``estimate_anisotropy.m``
    / ``estimate_size_ND.m``, which take ``unique(ID)`` without dropping id 0.
    """
    loc_nm = np.asarray(loc_nm, dtype=float)
    tid = np.asarray(tid).ravel()

    kw = dict(bin_width=bin_width, exclude_zero_id=exclude_zero_id)
    fit_x = estimate_size_nd_details(loc_nm[:, 0], tid, **kw)
    fit_y = estimate_size_nd_details(loc_nm[:, 1], tid, **kw)
    fit_z = estimate_size_nd_details(loc_nm[:, 2], tid, **kw)

    # rimf for every size-estimate mode (mean, median, max-bin, Gaussian fit);
    # the Gaussian-fit value (mode 3) is the default/final rimf.
    rimf_by_mode = [
        _rimf_from_sizes(
            float(fit_x.sizes[m]), float(fit_y.sizes[m]), float(fit_z.sizes[m])
        )
        for m in range(4)
    ]
    rimf = rimf_by_mode[mode]

    z_corrected = estimate_size_nd_details(loc_nm[:, 2] * rimf, tid, **kw) if np.isfinite(rimf) else None
    return {
        "rimf": rimf,
        "rimf_by_mode": rimf_by_mode,
        "mode": int(mode),
        "bin_width": float(bin_width),
        "x": fit_x,
        "y": fit_y,
        "z": fit_z,
        "z_corrected": z_corrected,
    }


def _rimf_from_sizes(sx: float, sy: float, sz: float) -> float:
    """z-scaling factor = geomean(size_x, size_y) / size_z (NaN if undefined)."""
    if (not np.isfinite(sx) or not np.isfinite(sy) or not np.isfinite(sz)
            or sx <= 0 or sy <= 0 or sz <= 0):
        return float("nan")
    return float(np.exp((np.log(sx) + np.log(sy)) / 2.0) / sz)


# ---------------------------------------------------------------------------
# Result dialogs
# ---------------------------------------------------------------------------

_MODE_LABELS = ["Mean", "Median", "Max-bin", "Gaussian fit"]
_DEFAULT_MODE = 3  # Gaussian fit (same default as MATLAB)

# Per-axis plot colours (vivid on the dark plot background).
_AXIS_COLORS = {"X": "#35d07f", "Y": "#4a90ff", "Z": "#ff5a5a"}  # green / blue / red
_TRACE_SIZE_COLORS = {"XY": "#35d07f", "Z": "#ff5a5a", "XYZ": "#4a90ff"}  # green / red / blue


def show_trace_size_dialog(parent: QWidget, ds, state) -> None:
    """Run trace size estimation on *ds* and display a result dialog."""
    from ..core.dataset import MinfluxDataset

    # --- collect data ---
    try:
        loc = np.asarray(ds.loc_nm, dtype=float)   # (N,3) in nm
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
    # Z   = axial offset using the CALIBRATED z (ds.loc_nm already applied RIMF).
    # XYZ = full 3-D Euclidean distance.
    try:
        result = {
            "xy":  estimate_size_nd_details(loc[:, [0, 1]], tid),
            "z":   estimate_size_nd_details(loc[:, 2], tid),
            "xyz": estimate_size_nd_details(loc[:, :3], tid),
            "mode": _DEFAULT_MODE,
        }
    except Exception as exc:
        QMessageBox.critical(parent, "Trace size", f"Computation failed:\n{exc}")
        return

    # Modeless result window (project convention for analysis output): does
    # not block the app and does not sit permanently in front of the viewers.
    from ..ui.modeless import show_modeless
    show_modeless(_TraceSizeDialog(None, result, n_traces, loc.shape[0]), parent)


def raw_last_valid_loc_nm(ds) -> tuple[np.ndarray, np.ndarray] | None:
    """Raw last-valid-iteration ``loc`` in **nanometres** plus trace ids.

    Reads the raw ``loc_x``/``loc_y``/``loc_z`` (metres) from the last valid
    iteration via :func:`mfx_get` and scales to nm. The raw coordinates are
    used deliberately and unconditionally: anisotropy / RIMF estimation must
    NOT see the RIMF-corrected z that ``ds.loc_nm`` applies (that would make
    the estimate circular and dependent on the dataset's current RIMF state).
    Returns ``None`` if coordinates are unavailable.
    """
    from ..core.loader import mfx_get

    def _col(name: str) -> np.ndarray | None:
        v = mfx_get(ds, name, itr="last", vld_only=True)
        if v is None:
            v = ds.attr.get(name)
        return None if v is None else np.asarray(v, dtype=float).ravel()

    x, y, z = _col("loc_x"), _col("loc_y"), _col("loc_z")
    if x is None or y is None or z is None:
        return None
    n = min(x.size, y.size, z.size)
    if n == 0:
        return None
    loc_nm = np.column_stack([x[:n], y[:n], z[:n]]) * 1e9

    tid = mfx_get(ds, "tid", itr="last", vld_only=True)
    if tid is None:
        tid = ds.attr.get("tid")
    if tid is None:
        return None
    tid = np.asarray(tid, dtype=np.int64).ravel()[:n]
    return loc_nm, tid


def estimate_anisotropy_for_dataset(ds, *, mode: int = 3) -> dict | None:
    """Anisotropy estimate from a dataset's RAW last-valid coordinates.

    Always reads the raw ``loc`` (never the RIMF-corrected ``ds.loc_nm``), so
    the result is independent of the dataset's current RIMF state — running it
    twice gives the same value. Returns the detailed result dict, or ``None``
    when 3-D coordinates / traces are unavailable.
    """
    data = raw_last_valid_loc_nm(ds)
    if data is None:
        return None
    loc, tid = data
    if loc.shape[1] < 3 or int(np.unique(tid[tid != 0]).size) == 0:
        return None
    return estimate_anisotropy_details(loc, tid, mode=mode)


def show_anisotropy_dialog(parent: QWidget, ds, state=None) -> None:
    """Run anisotropy estimation on *ds* and display a result dialog.

    Uses the **raw** last-valid-iteration ``loc`` coordinates (nm), not the
    RIMF-corrected ``ds.loc_nm`` and not the interactive filter — so the
    estimate is reproducible and not contaminated by a previously applied
    z-scaling. ``state`` (AppState) enables the dialog's "Apply to data"
    button to set the dataset's RIMF and refresh open views.
    """
    if ds.attr.get("tid") is None and ds.mfx_raw.get("tid") is None:
        QMessageBox.warning(parent, "Anisotropy",
                            "Dataset has no trace ID (tid) attribute.")
        return

    data = raw_last_valid_loc_nm(ds)
    if data is None:
        QMessageBox.warning(parent, "Anisotropy", "Cannot read localisations (loc x/y/z).")
        return
    loc, tid = data

    if loc.shape[1] < 3:
        QMessageBox.warning(parent, "Anisotropy",
                            "Anisotropy estimation requires 3-D data (x, y, z).")
        return

    n_traces = int(np.unique(tid[tid != 0]).size)
    if n_traces == 0:
        QMessageBox.warning(parent, "Anisotropy", "No valid traces found.")
        return

    # Check for actual z variation
    z_range = float(loc[:, 2].max() - loc[:, 2].min())
    if z_range < 1.0:
        QMessageBox.warning(parent, "Anisotropy",
                            "Z range is < 1 nm — this appears to be 2-D data.\n"
                            "Anisotropy estimation requires 3-D acquisitions.")
        return

    try:
        result = estimate_anisotropy_details(loc, tid)
    except Exception as exc:
        QMessageBox.critical(parent, "Anisotropy", f"Computation failed:\n{exc}")
        return

    # Modeless result window (see show_trace_size_dialog).
    from ..ui.modeless import show_modeless
    show_modeless(_AnisotropyDialog(None, result, n_traces, ds=ds, state=state), parent)


# ---------------------------------------------------------------------------
# Private dialog classes
# ---------------------------------------------------------------------------

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
            "&nbsp;&nbsp;Z — axial offset using the calibrated z (RIMF applied): "
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


class _AnisotropyDialog(QDialog):
    def __init__(self, parent: QWidget, result: dict, n_traces: int,
                 *, ds=None, state=None) -> None:
        super().__init__(parent)
        self._ds = ds
        self._state = state
        self.setWindowTitle("Estimated Anisotropy Factor (RIMF)")
        self.resize(980, 800)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        rimf = float(result.get("rimf", np.nan))
        self._rimf = rimf
        rimf_by_mode = result.get("rimf_by_mode", [np.nan, np.nan, np.nan, rimf])
        sizes_x = result["x"].sizes
        sizes_y = result["y"].sizes
        sizes_z = result["z"].sizes

        info = QLabel(f"<b>{n_traces}</b> traces used for estimation")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(info)

        # Abnormal when outside the physically expected [0.5, 1.0] window.
        abnormal = bool(np.isfinite(rimf)) and not (0.5 <= rimf <= 1.0)
        val_txt = f"{rimf:.4f}" if np.isfinite(rimf) else "—"
        val_color = "#ff5a5a" if abnormal else "palette(text)"
        rimf_lbl = QLabel(
            f"<span style='font-size:18px; color:{val_color}'>"
            f"<b>anisotropy factor = {val_txt}</b></span>"
            "<span style='color:gray'> &nbsp;(Gaussian fit)</span>"
        )
        rimf_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rimf_lbl.setToolTip(
            "anisotropy factor = geomean(σx, σy) / σz = √(σx·σy) / σz\n"
            "Applied as the RIMF (refractive index mismatch factor): "
            "z_corrected = z × RIMF"
        )
        root.addWidget(rimf_lbl)

        if abnormal:
            warn_lbl = QLabel("abnormal value, check the data!")
            warn_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            warn_lbl.setStyleSheet("color: #ff5a5a; font-size: 11px;")
            root.addWidget(warn_lbl)

        grp = QGroupBox(
            "Per-axis size estimates (nm) and anisotropy factor per mode — "
            "Gaussian fit is the final value"
        )
        # A real grid keeps the X / Y / Z / anisotropy-factor columns aligned
        # vertically with their value cells (a QFormLayout's field column would
        # not line the header row up with the value rows).
        grid = QGridLayout(grp)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(4)

        col_headers = ["", "X", "Y", "Z", "anisotropy factor"]
        for col, h in enumerate(col_headers):
            lbl = QLabel(f"<b>{h}</b>")
            lbl.setAlignment(
                Qt.AlignmentFlag.AlignLeft if col == 0 else Qt.AlignmentFlag.AlignCenter
            )
            grid.addWidget(lbl, 0, col)

        row_labels = ["Mean", "Median", "Max-bin", "Gaussian fit ★"]
        for i, (row_lbl, vx, vy, vz) in enumerate(zip(row_labels, sizes_x, sizes_y, sizes_z)):
            bold = i == _DEFAULT_MODE
            rmode = float(rimf_by_mode[i]) if i < len(rimf_by_mode) else np.nan
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
            "anisotropy factor — geomean(σ<sub>x</sub>, σ<sub>y</sub>) ∕ σ<sub>z</sub> "
            "= √( σ<sub>x</sub> · σ<sub>y</sub> ) ∕ σ<sub>z</sub><br>"
            "<br>"
            "Use the Gaussian-fit (★) estimate. The anisotropy factor is applied as the "
            "RIMF z-scaling factor."
            f"{fit_note}"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(note)

        # ── Apply row: editable RIMF (defaults to the computed value) + Apply ──
        btn_row = QHBoxLayout()
        self._apply_status = QLabel("")
        self._apply_status.setStyleSheet("color: #6c6;")
        btn_row.addWidget(self._apply_status)
        btn_row.addStretch()
        btn_row.addWidget(QLabel("RIMF:"))
        self._rimf_spin = QDoubleSpinBox()
        self._rimf_spin.setRange(0.0, 100.0)
        self._rimf_spin.setDecimals(4)
        self._rimf_spin.setSingleStep(0.01)
        self._rimf_spin.setValue(rimf if np.isfinite(rimf) else 1.0)
        self._rimf_spin.setToolTip(
            "RIMF value to apply to this dataset's z (defaults to the computed "
            "anisotropy factor; editable)."
        )
        btn_row.addWidget(self._rimf_spin)
        self._apply_btn = QPushButton("Apply to data")
        self._apply_btn.setToolTip(
            "Set this dataset's RIMF (z-scaling) and refresh open 3-D views. "
            "Raw z is not modified."
        )
        self._apply_btn.setEnabled(self._ds is not None and np.isfinite(rimf))
        self._apply_btn.clicked.connect(self._apply_to_data)
        btn_row.addWidget(self._apply_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    def _apply_to_data(self) -> None:
        if self._ds is None:
            return
        value = float(self._rimf_spin.value())
        self._ds.set_rimf(value, source="manual (anisotropy plugin)")
        if self._state is not None:
            idx = next(
                (i for i, d in enumerate(self._state.datasets) if d is self._ds),
                None,
            )
            if idx is not None:
                # Refresh render, scatter, 3-D, and the dataset-info RIMF text
                # (they re-pull the RIMF-corrected loc_nm / invalidate caches).
                self._state.notify_calibration_changed(idx)
            try:
                self._state.log(
                    f"Applied RIMF = {value:.4g} to '{self._ds.name}' (anisotropy plugin)."
                )
            except Exception:
                pass
        self._apply_status.setText(f"Applied RIMF = {value:.4g}")


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
        brush=pg.mkBrush(color + "66"),
        pen=pg.mkPen(color, width=1.2),
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
    # No legend / values: the per-axis colours match the individual plots and
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
    # RIMF = Z' / size_z. Distinct orange dash-dot line so it stands apart.
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
    y = _gauss1(x, *fit.fit_params)
    pw.plot(
        x, y,
        pen=pg.mkPen(color, width=2.0),
        fillLevel=0.0 if fill else None,
        brush=pg.mkBrush(color + "22") if fill else None,
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
):
    """Anisotropy-style histogram panel of a value distribution (linear axis).

    Used by reports that summarize a distribution with a single statistic
    (e.g. localization precision = median per-trace sigma) rather than a
    log-normal fit. Optionally draws a dashed marker at ``marker``. ``y_label``
    must match what each row of ``values`` is (e.g. "traces" vs "localizations").
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
    counts, edges = np.histogram(vals, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pw.plot(
        centers, counts,
        stepMode=False, fillLevel=0,
        brush=pg.mkBrush(color + "66"), pen=pg.mkPen(color, width=1.2),
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
