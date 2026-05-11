"""
minflux_viewer.analysis.localization_precision
================================================
Localization precision estimators.

Three independent measures are exposed in the *Analysis → Localization
Precision* menu:

* **FRC** — Fourier Ring Correlation (Banterle et al., *J. Struct. Biol.*
  183:363, 2013; Nieuwenhuizen et al., *Nat. Methods* 10:557, 2013).
  Renders two independent half-images, computes the radial Fourier ring
  correlation, finds the spatial frequency at which the correlation
  drops below the 1/7 threshold. **Stub for Phase 4** — the menu entry
  shows an info dialog citing the references; full implementation will
  follow in a dedicated session.

* **CRLB** — Cramér-Rao lower bound. Per-localization analytical bound
  on σ given photon counts, background, and PSF width
  (Mortensen et al., *Nat. Methods* 7:377, 2010). **Stub for Phase 4**
  for the same reason as FRC.

* **StdDev per trace** — Standard deviation of the localization
  coordinates within each trace (i.e. each individual molecule
  observed multiple times). This is the precision measure used in
  Schmidt et al., *Nat. Methods* 19:1246, 2022 (MINFLUX paper, see
  https://www.nature.com/articles/s41592-022-01577-1#Sec2 ). **Fully
  implemented here.**

The StdDev routine groups localizations by trace ID (``tid``), computes
σx / σy / σz per trace, then summarises across all traces with at
least :data:`MIN_LOCS_PER_TRACE` repeats. Results are reported in nm
(after multiplying the metres-stored coordinates by 1e9), shown in a
dialog and also written to ``dataset.attr`` so they're available for
follow-up work (export, plotting, journal entry).
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Constants & data class
# ---------------------------------------------------------------------------

MIN_LOCS_PER_TRACE: int = 3   # below this we skip the trace (σ undefined)


# ---------------------------------------------------------------------------
# StdDev per trace — full implementation
# ---------------------------------------------------------------------------

def stddev_per_trace(
    loc_xyz: np.ndarray,
    tid: np.ndarray,
    *,
    min_locs: int = MIN_LOCS_PER_TRACE,
) -> dict:
    """
    Compute per-trace standard deviation of the localization coordinates.

    Parameters
    ----------
    loc_xyz : (N, 3) float ndarray
        Localizations in *metres* (raw MINFLUX storage convention). Output
        is reported in nm.
    tid : (N,) int ndarray
        Trace ID per localization.
    min_locs : int
        Traces with fewer than this many localizations are dropped.

    Returns
    -------
    dict with keys:

        ``per_trace_sigma_xyz``  (M, 3) ndarray of σ in nm, M valid traces
        ``per_trace_n``           (M,) ndarray of N localizations per trace
        ``trace_ids``             (M,) ndarray of the corresponding tids
        ``median_sigma_xyz``      (3,) ndarray, median of column-wise σ in nm
        ``mean_sigma_xyz``        (3,) ndarray, mean of column-wise σ in nm
        ``n_traces_used``         int
        ``n_traces_total``        int
    """
    if loc_xyz.size == 0:
        empty3 = np.zeros(3, dtype=float)
        return dict(
            per_trace_sigma_xyz=np.empty((0, 3)),
            per_trace_n=np.empty(0, dtype=int),
            trace_ids=np.empty(0, dtype=int),
            median_sigma_xyz=empty3, mean_sigma_xyz=empty3,
            n_traces_used=0, n_traces_total=0,
        )

    # nm conversion
    loc_nm = np.asarray(loc_xyz, dtype=float) * 1.0e9

    tid = np.asarray(tid).ravel()
    uniq, inverse = np.unique(tid, return_inverse=True)
    n_total = uniq.size

    sigmas = []
    counts = []
    used_ids = []
    for i, t in enumerate(uniq):
        sel = inverse == i
        n = int(sel.sum())
        if n < min_locs:
            continue
        # ddof=1 → sample standard deviation
        s = loc_nm[sel].std(axis=0, ddof=1)
        sigmas.append(s)
        counts.append(n)
        used_ids.append(t)

    if not sigmas:
        empty3 = np.zeros(3, dtype=float)
        return dict(
            per_trace_sigma_xyz=np.empty((0, 3)),
            per_trace_n=np.empty(0, dtype=int),
            trace_ids=np.empty(0, dtype=int),
            median_sigma_xyz=empty3, mean_sigma_xyz=empty3,
            n_traces_used=0, n_traces_total=n_total,
        )

    per_trace = np.array(sigmas)
    return dict(
        per_trace_sigma_xyz=per_trace,
        per_trace_n=np.array(counts, dtype=int),
        trace_ids=np.array(used_ids),
        median_sigma_xyz=np.median(per_trace, axis=0),
        mean_sigma_xyz=per_trace.mean(axis=0),
        n_traces_used=per_trace.shape[0],
        n_traces_total=n_total,
    )


# ---------------------------------------------------------------------------
# UI launchers — these are what the menu items call
# ---------------------------------------------------------------------------

_FRC_INFO = (
    "<b>Fourier Ring Correlation (FRC)</b><br><br>"
    "Estimates resolution by splitting the localizations into two random halves,"
    " rendering each as an image, and computing the radial Fourier correlation."
    " Resolution is read off at the spatial frequency where correlation drops"
    " below the 1/7 threshold.<br><br>"
    "References:<br>"
    " • <a href='https://doi.org/10.1016/j.jsb.2013.05.004'>Banterle <i>et al.</i>, <i>J. Struct. Biol.</i> 183:363 (2013)</a><br>"
    " • <a href='https://doi.org/10.1038/nmeth.2448'>Nieuwenhuizen <i>et al.</i>, <i>Nat. Methods</i> 10:557 (2013)</a><br><br>"
    "<i>Implementation scheduled for the next analysis session — the menu"
    " entry is a placeholder that records this status in the processing"
    " journal.</i>"
)

_CRLB_INFO = (
    "<b>Cramér-Rao Lower Bound (CRLB)</b><br><br>"
    "Analytical lower bound on per-localization σ given photon count,"
    " background, and PSF width.<br><br>"
    "Reference:<br>"
    " • <a href='https://doi.org/10.1038/nmeth.1447'>Mortensen <i>et al.</i>, <i>Nat. Methods</i> 7:377 (2010)</a><br><br>"
    "<i>Implementation scheduled for the next analysis session — the menu"
    " entry is a placeholder that records this status in the processing"
    " journal.</i>"
)


def show_frc_info(parent, state) -> None:
    _show_info_dialog(parent, "FRC — Fourier Ring Correlation", _FRC_INFO)
    _try_journal(state, "Localization precision (FRC) — placeholder shown to user")


def show_crlb_info(parent, state) -> None:
    _show_info_dialog(parent, "CRLB — Cramér-Rao Lower Bound", _CRLB_INFO)
    _try_journal(state, "Localization precision (CRLB) — placeholder shown to user")


def run_stddev_per_trace(parent, state) -> None:
    """Compute σ per trace for the active dataset and show a results dialog."""
    ds = state.active_dataset
    if ds is None:
        return

    loc_x = np.asarray(ds.attr.get("loc_x"))
    loc_y = np.asarray(ds.attr.get("loc_y"))
    loc_z = np.asarray(ds.attr.get("loc_z", np.zeros_like(loc_x)))
    tid   = np.asarray(ds.attr.get("tid", np.arange(loc_x.size)))
    ftr   = np.asarray(ds.filter_mask, dtype=bool)

    loc_xyz = np.column_stack([loc_x[ftr], loc_y[ftr], loc_z[ftr]])
    tid_f   = tid[ftr]

    result = stddev_per_trace(loc_xyz, tid_f)

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
        f"median σ = ({result['median_sigma_xyz'][0]:.2f}, "
        f"{result['median_sigma_xyz'][1]:.2f}, "
        f"{result['median_sigma_xyz'][2]:.2f}) nm "
        f"over {result['n_traces_used']:,} of {result['n_traces_total']:,} traces "
        f"(≥{MIN_LOCS_PER_TRACE} loc/trace)."
    )

    dlg = StdDevResultDialog(result, parent=parent)
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    if parent is not None:
        parent._loc_precision_stddev_dialog = dlg


# ---------------------------------------------------------------------------
# Result dialog
# ---------------------------------------------------------------------------

class StdDevResultDialog(QDialog):
    """Show a one-page summary of σx / σy / σz across traces."""

    def __init__(self, result: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Localization Precision — StdDev per trace")
        self.setModal(False)
        self.resize(560, 420)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)

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
        else:
            root.addWidget(QLabel(
                f"<b>StdDev of localizations within each trace</b><br><br>"
                f"Reference: <a href='https://www.nature.com/articles/s41592-022-01577-1'>"
                f"Schmidt <i>et al.</i>, <i>Nat. Methods</i> 19:1246 (2022)</a>.<br><br>"
                f"Traces used: <b>{n_used:,}</b> of {n_total:,} "
                f"(≥{MIN_LOCS_PER_TRACE} loc/trace).<br>"
                f"Median σ: <b>{med[0]:.2f}, {med[1]:.2f}, {med[2]:.2f} nm</b> "
                f"(x, y, z)<br>"
                f"Mean σ:   {mean[0]:.2f}, {mean[1]:.2f}, {mean[2]:.2f} nm<br>"
            ))
            root.itemAt(root.count() - 1).widget().setOpenExternalLinks(True)
            root.itemAt(root.count() - 1).widget().setTextInteractionFlags(
                Qt.TextInteractionFlag.TextBrowserInteraction
            )

            # Histogram per axis
            try:
                import pyqtgraph as pg
                grid = QHBoxLayout()
                axes = ("σx", "σy", "σz")
                colours = ("#4cf", "#fa4", "#9f4")
                for i, (lab, col) in enumerate(zip(axes, colours)):
                    sigmas = result["per_trace_sigma_xyz"][:, i]
                    if sigmas.size == 0:
                        continue
                    pw = pg.PlotWidget(background="#222")
                    pw.setMinimumHeight(180)
                    pw.setLabel("bottom", lab, units="nm")
                    pw.setLabel("left", "traces")
                    h, edges = np.histogram(sigmas, bins=50)
                    xs = 0.5 * (edges[:-1] + edges[1:])
                    pw.plot(xs, h, stepMode=False,
                            fillLevel=0, brush=pg.mkBrush(col),
                            pen=pg.mkPen(col, width=1))
                    grid.addWidget(pw, stretch=1)
                root.addLayout(grid, stretch=1)
            except Exception:
                pass

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        bb.accepted.connect(self.accept)
        bb.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.close)
        root.addWidget(bb)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

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
