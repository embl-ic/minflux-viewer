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

import numpy as np
from scipy.optimize import curve_fit
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Core computation — estimate_size_nd
# ---------------------------------------------------------------------------

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
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    ids = np.asarray(ids, dtype=np.int64).ravel()

    uid = np.unique(ids)
    if exclude_zero_id:
        uid = uid[uid != 0]

    if uid.size == 0:
        return np.full(4, np.nan), np.array([])

    parts: list[np.ndarray] = []
    for cid in uid:
        mask = ids == cid
        pts = data[mask]
        if len(pts) < 2:
            continue
        centroid = np.median(pts, axis=0)
        d = np.linalg.norm(pts - centroid, axis=1)
        parts.append(d)

    if not parts:
        return np.full(4, np.nan), np.array([])

    dist = np.concatenate(parts)
    dist = dist[dist > 0]

    if dist.size < 3:
        return np.full(4, np.nan), np.array([])

    logdist = np.log(dist)

    sizes = np.full(4, np.nan)
    sizes[0] = np.exp(np.mean(logdist))
    sizes[1] = np.exp(np.median(logdist))

    lo = logdist.min()
    hi = logdist.max() + bin_width
    edges = np.arange(lo, hi, bin_width)
    if edges.size < 3:
        edges = np.linspace(lo, hi, max(int((hi - lo) / 0.05), 10))
    counts, edges = np.histogram(logdist, bins=edges)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0

    max_idx = int(np.argmax(counts))
    sizes[2] = np.exp((edges[max_idx] + edges[max_idx + 1]) / 2.0)

    # Gaussian fit: f(x) = a * exp(-((x - b) / c)^2)
    try:
        def _gauss(x, a, b, c):
            return a * np.exp(-((x - b) / max(abs(c), 1e-9)) ** 2)

        p0 = [float(counts[max_idx]), float(bin_centers[max_idx]),
              max(float(logdist.std()), 0.1)]
        popt, _ = curve_fit(
            _gauss, bin_centers, counts.astype(float),
            p0=p0, maxfev=5000,
        )
        sizes[3] = np.exp(float(popt[1]))
    except Exception:
        sizes[3] = sizes[2]  # fallback: max-bin estimate

    return sizes, logdist


# ---------------------------------------------------------------------------
# Core computation — estimate_anisotropy
# ---------------------------------------------------------------------------

def estimate_anisotropy(
    loc_nm: np.ndarray,
    tid: np.ndarray,
    *,
    mode: int = 3,
    bin_width: float = 0.1,
    exclude_zero_id: bool = True,
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

    kw = dict(bin_width=bin_width, exclude_zero_id=exclude_zero_id)
    sizes_x, _ = estimate_size_nd(loc_nm[:, 0], tid, **kw)
    sizes_y, _ = estimate_size_nd(loc_nm[:, 1], tid, **kw)
    sizes_z, _ = estimate_size_nd(loc_nm[:, 2], tid, **kw)

    sx = float(sizes_x[mode])
    sy = float(sizes_y[mode])
    sz = float(sizes_z[mode])

    if sz <= 0 or not np.isfinite(sz) or not np.isfinite(sx) or not np.isfinite(sy):
        return float("nan"), sizes_x, sizes_y, sizes_z

    xy_geo = np.exp((np.log(sx) + np.log(sy)) / 2.0)
    rimf = xy_geo / sz

    return float(rimf), sizes_x, sizes_y, sizes_z


# ---------------------------------------------------------------------------
# Result dialogs
# ---------------------------------------------------------------------------

_MODE_LABELS = ["Mean", "Median", "Max-bin", "Gaussian fit"]
_DEFAULT_MODE = 3  # Gaussian fit (same default as MATLAB)


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

    # --- compute ---
    try:
        sizes, logdist = estimate_size_nd(loc, tid)
    except Exception as exc:
        QMessageBox.critical(parent, "Trace size", f"Computation failed:\n{exc}")
        return

    _TraceSizeDialog(parent, sizes, n_traces, loc.shape[0]).exec()


def show_anisotropy_dialog(parent: QWidget, ds) -> None:
    """Run anisotropy estimation on *ds* and display a result dialog."""
    try:
        loc = np.asarray(ds.loc_nm, dtype=float)
    except Exception as exc:
        QMessageBox.warning(parent, "Anisotropy", f"Cannot read localisations:\n{exc}")
        return

    if loc.shape[1] < 3:
        QMessageBox.warning(parent, "Anisotropy",
                            "Anisotropy estimation requires 3-D data (x, y, z).")
        return

    tid_raw = ds.attr.get("tid")
    if tid_raw is None:
        QMessageBox.warning(parent, "Anisotropy",
                            "Dataset has no trace ID (tid) attribute.")
        return

    mask = np.asarray(ds.filter_mask, dtype=bool)
    if mask.shape[0] == loc.shape[0]:
        loc = loc[mask]
        tid = np.asarray(tid_raw, dtype=np.int64).ravel()[mask]
    else:
        tid = np.asarray(tid_raw, dtype=np.int64).ravel()

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
        rimf, sx, sy, sz = estimate_anisotropy(loc, tid)
    except Exception as exc:
        QMessageBox.critical(parent, "Anisotropy", f"Computation failed:\n{exc}")
        return

    _AnisotropyDialog(parent, rimf, sx, sy, sz, n_traces).exec()


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
        sizes: np.ndarray,
        n_traces: int,
        n_locs: int,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Estimated Average Trace Size")
        self.setMinimumWidth(340)

        root = QVBoxLayout(self)

        info = QLabel(
            f"<b>{n_traces}</b> traces &nbsp;|&nbsp; <b>{n_locs:,}</b> localisations"
        )
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(info)

        grp = QGroupBox("Size estimates  (3-D Euclidean radius, nm)")
        form = QFormLayout(grp)
        labels = [
            ("Mean of log-distances",   sizes[0]),
            ("Median of log-distances", sizes[1]),
            ("Max histogram bin",       sizes[2]),
            ("Gaussian fit peak",       sizes[3]),
        ]
        for text, val in labels:
            lbl = QLabel(f"<b>{_nm(val)}</b>")
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            form.addRow(text + ":", lbl)
        root.addWidget(grp)

        note = QLabel(
            "Distances are log-transformed before fitting — assumes a log-normal\n"
            "distribution of intra-trace spread."
        )
        note.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(note)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)


class _AnisotropyDialog(QDialog):
    def __init__(
        self,
        parent: QWidget,
        rimf: float,
        sizes_x: np.ndarray,
        sizes_y: np.ndarray,
        sizes_z: np.ndarray,
        n_traces: int,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Estimated Anisotropy (rimf / z-scaling factor)")
        self.setMinimumWidth(420)

        root = QVBoxLayout(self)

        info = QLabel(f"<b>{n_traces}</b> traces used for estimation")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(info)

        # Rimf headline
        rimf_lbl = QLabel(
            f"<span style='font-size:18px'><b>rimf = {rimf:.4f}</b></span>"
            if np.isfinite(rimf) else "<span style='font-size:18px'><b>rimf = —</b></span>"
        )
        rimf_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rimf_lbl.setToolTip(
            "z-scaling factor = geomean(size_x, size_y) / size_z\n"
            "Apply to znm as: znm_corrected = znm * rimf"
        )
        root.addWidget(rimf_lbl)

        # Per-axis table (Gaussian-fit row highlighted)
        grp = QGroupBox("Per-axis size estimates (nm)  —  Gaussian fit used for rimf")
        form = QFormLayout(grp)

        row_labels = ["Mean", "Median", "Max-bin", "Gaussian fit ★"]
        headers = QHBoxLayout()
        for h in ("", "X", "Y", "Z"):
            lbl = QLabel(f"<b>{h}</b>")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            headers.addWidget(lbl)
        form.addRow("", _widget_from_layout(headers))

        for i, (row_lbl, vx, vy, vz) in enumerate(
            zip(row_labels, sizes_x, sizes_y, sizes_z)
        ):
            row = QHBoxLayout()
            bold = i == _DEFAULT_MODE
            for val in (vx, vy, vz):
                t = f"<b>{_nm(val)}</b>" if bold else _nm(val)
                cell = QLabel(t)
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                row.addWidget(cell)
            form.addRow(row_lbl + ":", _widget_from_layout(row))

        root.addWidget(grp)

        note = QLabel(
            "rimf = geomean(size_x, size_y) / size_z\n"
            "Use the Gaussian-fit (★) estimates.  Apply rimf to z coordinates\n"
            "before 3-D rendering or distance measurements."
        )
        note.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(note)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)


def _widget_from_layout(layout: QHBoxLayout) -> QWidget:
    w = QWidget()
    w.setLayout(layout)
    return w
