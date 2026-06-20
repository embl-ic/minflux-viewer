"""
"Show Beads Drift" dialog for the MSR reader.

One tab per dataset; each tab lists the MBM beads as rows of four plots
(Y–X, X–t, Y–t, Z–t). Points are coloured by **time** (blue→red) so the drift
trajectory's dynamics are visible; a horizontal colour bar shows the mapping.

Per-bead checkboxes mark beads for alignment, **linked across datasets by gri-ID**
(the same physical bead) with logical-AND semantics. *Common* beads (present in
every dataset — the ones an alignment can actually use) are checked by default
and shown in bold. *Apply* commits the selection and closes.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QLinearGradient, QPainter
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

# rainbow time colormap: blue → cyan → green → yellow → orange → red
_CMAP_POS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
_CMAP_RGB = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 165, 0), (255, 0, 0)]
_COL_TITLES = ("Y (nm) vs X (nm)", "X (nm) vs time (sec)",
               "Y (nm) vs time (sec)", "Z (nm) vs time (sec)")
#: (x source index, y index); x index None => time axis
_PLOT_KEYS = ((0, 1), (None, 0), (None, 1), (None, 2))
_MAX_PTS = 600                # display cap per bead (keeps the dialog snappy)


def _time_cmap():
    import pyqtgraph as pg
    return pg.ColorMap(_CMAP_POS, [(r, g, b, 255) for r, g, b in _CMAP_RGB])


def _subsample(n: int) -> np.ndarray | slice:
    if n <= _MAX_PTS:
        return slice(None)
    return np.linspace(0, n - 1, _MAX_PTS).astype(int)


class _TimeColorBar(QWidget):
    """Horizontal early→late colour legend for the time mapping."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(26)
        self.setMinimumWidth(240)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.drawText(0, h - 7, "time: early")
        bar_x0, bar_x1 = 78, w - 36
        grad = QLinearGradient(bar_x0, 0, bar_x1, 0)
        for pos, (r, g, b) in zip(_CMAP_POS, _CMAP_RGB):
            grad.setColorAt(pos, QColor(r, g, b))
        p.fillRect(bar_x0, 4, bar_x1 - bar_x0, h - 12, grad)
        p.drawText(bar_x1 + 3, h - 7, "late")


class BeadsDriftDialog(QDialog):
    """Inspect per-bead drift and select beads for alignment."""

    def __init__(self, datasets: list[dict], *, unchecked_gris=None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Beads drift — manual selection")
        self.setMinimumSize(1000, 640)

        self._datasets = datasets or []
        n_ds = len(self._datasets)

        # gri -> 1-based dataset numbers it appears in; "common" = in all datasets.
        self._gri_to_ds: dict[int, list[int]] = {}
        for i, d in enumerate(self._datasets, start=1):
            for b in d["beads"]:
                self._gri_to_ds.setdefault(b["gri"], []).append(i)
        all_gris = set(self._gri_to_ds)
        self._common_gris = {g for g, ds in self._gri_to_ds.items() if len(ds) == n_ds}

        # Default checked = common beads (those an alignment can use); a passed
        # selection (a prior Apply) overrides the default.
        if unchecked_gris is None:
            self._gri_checked = {g: (g in self._common_gris) for g in all_gris}
        else:
            unchecked = set(unchecked_gris)
            self._gri_checked = {g: (g not in unchecked) for g in all_gris}
        self._checkboxes: dict[int, list[QCheckBox]] = {g: [] for g in all_gris}
        self._built: set[int] = set()
        self._syncing = False
        # (plot, x_range, y_range) for resetting interactive zoom/pan to defaults.
        self._plots: list[tuple] = []

        root = QVBoxLayout(self)
        for line in (
            "Check bead drift and select beads for alignment; by default all common beads will be used.",
            "Uncheck a bead to exclude it — the selection is linked across datasets by bead ID.",
            "Be aware of the minimum number of beads (3 for 2D, 4 for 3D) required for alignment.",
            "When finished, click <b>Apply</b> to update the bead selection.",
        ):
            lbl = QLabel(line)
            lbl.setWordWrap(True)
            root.addWidget(lbl)

        self._selection_label = QLabel("")
        self._selection_label.setWordWrap(True)
        self._selection_label.setStyleSheet("color: #1a6; font-weight: bold;")
        root.addWidget(self._selection_label)

        cbar_row = QHBoxLayout()
        cbar_row.addStretch(1)
        cbar_row.addWidget(_TimeColorBar())
        root.addLayout(cbar_row)

        from PyQt6.QtWidgets import QTabWidget
        self._tabs = QTabWidget()
        for d in self._datasets:
            placeholder = QWidget()
            QVBoxLayout(placeholder)
            self._tabs.addTab(placeholder, d["name"])
        self._tabs.currentChanged.connect(self._build_tab)
        root.addWidget(self._tabs, 1)

        buttons = QDialogButtonBox()
        reset_btn = buttons.addButton("Reset", QDialogButtonBox.ButtonRole.ResetRole)
        reset_btn.setToolTip(
            "Reset to the default selection (common beads checked, single-dataset "
            "beads unchecked) and restore all plot views.")
        reset_btn.clicked.connect(self._reset)
        apply_btn = buttons.addButton("Apply", QDialogButtonBox.ButtonRole.AcceptRole)
        apply_btn.setToolTip("Use the checked beads for alignment and close.")
        cancel_btn = buttons.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        apply_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        root.addWidget(buttons)

        self._update_selection_line()
        if self._datasets:
            self._build_tab(0)

    # ------------------------------------------------------------------
    @staticmethod
    def _dataset_ranges(beads: list[dict]) -> dict:
        x = np.concatenate([b["xyz_nm"][:, 0] for b in beads])
        y = np.concatenate([b["xyz_nm"][:, 1] for b in beads])
        z = np.concatenate([b["xyz_nm"][:, 2] for b in beads])
        t = np.concatenate([b["tim_s"] for b in beads])

        def rng(a):
            lo, hi = float(np.min(a)), float(np.max(a))
            return (lo, hi if hi > lo else lo + 1.0)

        return {"x": rng(x), "y": rng(y), "z": rng(z), "t": rng(t)}

    def _build_tab(self, index: int) -> None:
        if index in self._built or not (0 <= index < len(self._datasets)):
            return
        self._built.add(index)
        import pyqtgraph as pg

        beads = self._datasets[index]["beads"]
        ranges = self._dataset_ranges(beads)
        cmap = _time_cmap()
        t_lo, t_hi = ranges["t"]
        y_range_for = {0: ranges["x"], 1: ranges["y"], 2: ranges["z"]}

        page = self._tabs.widget(index)
        old = page.layout()
        if old is not None:
            QWidget().setLayout(old)
        outer = QVBoxLayout(page)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(10)

        grid.addWidget(QLabel("<b>bead</b>"), 0, 0, 1, 2)
        for c, title in enumerate(_COL_TITLES):
            head = QLabel(f"<b>{title}</b>")
            head.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(head, 0, 2 + c)

        for r, bead in enumerate(beads, start=1):
            gri = bead["gri"]
            common = gri in self._common_gris

            cb = QCheckBox()
            cb.setChecked(self._gri_checked.get(gri, True))
            cb.setToolTip("uncheck to remove this bead from alignment computation")
            cb.toggled.connect(lambda checked, g=gri: self._on_toggle(g, checked))
            self._checkboxes[gri].append(cb)
            grid.addWidget(cb, r, 0, Qt.AlignmentFlag.AlignTop)

            gri_txt = f"<b>{gri}</b>" if common else str(gri)
            ds_txt = ", ".join(str(i) for i in self._gri_to_ds.get(gri, []))
            lbl = QLabel(f"{gri_txt}<br>({bead['rid']})<br>"
                         f"<span style='color:#888'>dataset: {ds_txt}</span>")
            lbl.setTextFormat(Qt.TextFormat.RichText)
            grid.addWidget(lbl, r, 1, Qt.AlignmentFlag.AlignTop)

            xyz = bead["xyz_nm"]
            tim = bead["tim_s"]
            sel = _subsample(len(tim))
            xs, ys, zs, ts = xyz[sel, 0], xyz[sel, 1], xyz[sel, 2], tim[sel]
            cols = (xs, ys, zs)
            span = (t_hi - t_lo) or 1.0
            rgba = cmap.map(np.clip((ts - t_lo) / span, 0, 1), mode="byte")
            brushes = [pg.mkBrush(int(a), int(b_), int(c_), int(d_)) for a, b_, c_, d_ in rgba]

            for c, (xi, yi) in enumerate(_PLOT_KEYS):
                xdata = cols[xi] if xi is not None else ts
                ydata = cols[yi]
                plot = pg.PlotWidget()
                plot.setFixedHeight(168)
                plot.setMinimumWidth(210)
                plot.setMenuEnabled(True)
                plot.showGrid(x=True, y=True, alpha=0.15)
                plot.plot(xdata, ydata, pen=pg.mkPen((170, 170, 170, 120), width=1))
                plot.addItem(pg.ScatterPlotItem(x=xdata, y=ydata, size=5, pen=None, brush=brushes))
                # shared axis limits per plot-type within the dataset
                x_rng = ranges["x"] if xi == 0 else ranges["t"]
                y_rng = y_range_for[yi]
                plot.setXRange(*x_rng, padding=0.06)
                plot.setYRange(*y_rng, padding=0.06)
                if xi == 0:                       # Y vs X — equal nm scale
                    plot.getPlotItem().setAspectLocked(True)
                self._plots.append((plot, x_rng, y_rng))
                grid.addWidget(plot, r, 2 + c)

        scroll.setWidget(inner)
        outer.addWidget(scroll)

    # ------------------------------------------------------------------
    def _on_toggle(self, gri: int, checked: bool) -> None:
        if self._syncing:
            return
        self._gri_checked[gri] = bool(checked)
        self._syncing = True
        for cb in self._checkboxes.get(gri, []):
            if cb.isChecked() != checked:
                cb.setChecked(checked)
        self._syncing = False
        self._update_selection_line()

    def _reset(self) -> None:
        """Restore the default bead selection (common checked, single-dataset
        unchecked) and reset every plot's view to its shared default range."""
        self._syncing = True
        for gri in self._gri_checked:
            default = gri in self._common_gris
            self._gri_checked[gri] = default
            for cb in self._checkboxes.get(gri, []):
                if cb.isChecked() != default:
                    cb.setChecked(default)
        self._syncing = False
        self._update_selection_line()
        for plot, x_rng, y_rng in self._plots:
            try:
                plot.setXRange(*x_rng, padding=0.06)
                plot.setYRange(*y_rng, padding=0.06)
            except Exception:
                pass

    def _update_selection_line(self) -> None:
        sel = sorted(self.selected_gris())
        ids = ", ".join(str(g) for g in sel) if sel else "(none)"
        self._selection_label.setText(f"Selected bead IDs: {ids}  —  {len(sel)} bead(s)")

    # ------------------------------------------------------------------
    def selected_gris(self) -> set[int]:
        """gri-IDs of the beads that remain checked (the alignment subset)."""
        return {g for g, on in self._gri_checked.items() if on}

    def unchecked_gris(self) -> set[int]:
        return {g for g, on in self._gri_checked.items() if not on}
