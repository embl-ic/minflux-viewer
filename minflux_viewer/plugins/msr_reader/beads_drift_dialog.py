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

#: axis key -> label; keys 0/1/2 = X/Y/Z columns, "t" = time
_AXIS_LABEL = {0: "X (nm)", 1: "Y (nm)", 2: "Z (nm)", "t": "time (sec)"}


def _fmt_axis_value(key, value: float) -> str:
    """`X (nm): 33.7` for nm axes, `time (sec): 1137` for the time axis."""
    return f"{_AXIS_LABEL[key]}: {value:.{0 if key == 't' else 1}f}"


def _point_tip(x, y, data) -> str:
    """ScatterPlotItem hover tip — the per-spot string we stashed in ``data``."""
    return str(data)


def _build_point_tips(idxs, axis_vals: dict, xi, yi) -> list[str]:
    """Per-point hover strings for a sub-plot: ``<idx>: <a>: v; <b>: v`` where
    *a*/*b* are the two axes NOT plotted here (the plot shows axes *xi*,*yi*;
    ``xi is None`` means the x-axis is time)."""
    shown = {xi, yi} if xi is not None else {"t", yi}
    k1, k2 = [k for k in (0, 1, 2, "t") if k not in shown]
    a1, a2 = axis_vals[k1], axis_vals[k2]
    return [f"{int(idxs[j])}: {_fmt_axis_value(k1, a1[j])}; {_fmt_axis_value(k2, a2[j])}"
            for j in range(len(idxs))]


def _time_cmap():
    import pyqtgraph as pg
    return pg.ColorMap(_CMAP_POS, [(r, g, b, 255) for r, g, b in _CMAP_RGB])


def _subsample(n: int) -> np.ndarray | slice:
    if n <= _MAX_PTS:
        return slice(None)
    return np.linspace(0, n - 1, _MAX_PTS).astype(int)


def _nice_time_ticks(lo: float, hi: float, target: int = 4) -> list[float]:
    """A handful of round tick values inside [lo, hi] (e.g. 0,1000,2000,3000)."""
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return [lo]
    raw = span / max(target, 1)
    mag = 10.0 ** np.floor(np.log10(raw))
    norm = raw / mag
    step = (1 if norm < 1.5 else 2 if norm < 3 else 5 if norm < 7 else 10) * mag
    first = np.ceil(lo / step) * step
    ticks: list[float] = []
    v = first
    while v <= hi + step * 1e-6:
        ticks.append(round(float(v), 6))
        v += step
    return ticks or [lo]


class _TimeColorBar(QWidget):
    """Horizontal begin→end colour legend for the time mapping, with a second
    ruler underneath showing the actual acquisition time the colours map to."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        from PyQt6.QtWidgets import QSizePolicy
        self.setFixedHeight(48)
        self.setMinimumWidth(420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._t_lo = 0.0
        self._t_hi = 0.0          # hi <= lo => ruler hidden (no time range yet)

    def set_time_range(self, t_lo: float, t_hi: float) -> None:
        self._t_lo, self._t_hi = float(t_lo), float(t_hi)
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        fm = p.fontMetrics()
        w = self.width()
        bar_y0, bar_h = 4, 16
        bar_x0 = max(78, fm.horizontalAdvance("time begin") + 10)
        bar_x1 = w - 50                                   # right margin for "end" / "(sec)"
        base = bar_y0 + bar_h - 3                         # text baseline aligned to the bar

        # flanking labels: "time begin  [====gradient====]  end"
        p.setPen(QColor(40, 40, 40))
        p.drawText(bar_x0 - 4 - fm.horizontalAdvance("time begin"), base, "time begin")
        grad = QLinearGradient(bar_x0, 0, bar_x1, 0)
        for pos, (r, g, b) in zip(_CMAP_POS, _CMAP_RGB):
            grad.setColorAt(pos, QColor(r, g, b))
        p.fillRect(bar_x0, bar_y0, bar_x1 - bar_x0, bar_h, grad)
        p.setPen(QColor(120, 120, 120))
        p.drawRect(bar_x0, bar_y0, bar_x1 - bar_x0, bar_h)
        p.setPen(QColor(40, 40, 40))
        p.drawText(bar_x1 + 5, base, "end")

        if self._t_hi <= self._t_lo:
            return
        # second ruler: axis line + a few round ticks under the gradient
        line_y = bar_y0 + bar_h + 6
        lbl_base = line_y + 4 + fm.ascent() + 1
        span = self._t_hi - self._t_lo
        p.setPen(QColor(90, 90, 90))
        p.drawLine(bar_x0, line_y, bar_x1, line_y)
        for v in _nice_time_ticks(self._t_lo, self._t_hi):
            x = int(round(bar_x0 + (v - self._t_lo) / span * (bar_x1 - bar_x0)))
            p.drawLine(x, line_y, x, line_y + 4)
            txt = f"{v:g}"
            p.drawText(x - fm.horizontalAdvance(txt) // 2, lbl_base, txt)
        p.drawText(bar_x1 + 5, lbl_base, "(sec)")


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
            "Minimum matched beads: 2 (translational XYZ), 3 (also rigid XY + translational Z), 4 (full rigid XYZ).",
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
        self._cbar = _TimeColorBar()
        cbar_row.addWidget(self._cbar, 1)
        root.addLayout(cbar_row)

        from PyQt6.QtWidgets import QTabWidget
        self._tabs = QTabWidget()
        for d in self._datasets:
            placeholder = QWidget()
            QVBoxLayout(placeholder)
            self._tabs.addTab(placeholder, d["name"])
        self._tabs.currentChanged.connect(self._on_tab_changed)
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
            self._update_colorbar_range(0)

    # ------------------------------------------------------------------
    def _on_tab_changed(self, index: int) -> None:
        self._build_tab(index)
        self._update_colorbar_range(index)

    def _update_colorbar_range(self, index: int) -> None:
        """Point the colour-bar's second ruler at the shown dataset's time span."""
        if not (0 <= index < len(self._datasets)):
            return
        beads = self._datasets[index]["beads"]
        times = [b["tim_s"] for b in beads if len(b["tim_s"])]
        if not times:
            self._cbar.set_time_range(0.0, 0.0)
            return
        t = np.concatenate(times)
        lo, hi = float(np.min(t)), float(np.max(t))
        self._cbar.set_time_range(lo, hi if hi > lo else lo + 1.0)

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
            idxs = np.arange(len(tim))[sel]      # original sequence index per shown point
            xs, ys, zs, ts = xyz[sel, 0], xyz[sel, 1], xyz[sel, 2], tim[sel]
            cols = (xs, ys, zs)
            axis_vals = {0: xs, 1: ys, 2: zs, "t": ts}   # value array per axis key
            span = (t_hi - t_lo) or 1.0
            rgba = cmap.map(np.clip((ts - t_lo) / span, 0, 1), mode="byte")
            brushes = [pg.mkBrush(int(a), int(b_), int(c_), int(d_)) for a, b_, c_, d_ in rgba]

            for c, (xi, yi) in enumerate(_PLOT_KEYS):
                xdata = cols[xi] if xi is not None else ts
                ydata = cols[yi]
                # hover datatip: show the index + the two axes NOT plotted here
                tips = _build_point_tips(idxs, axis_vals, xi, yi)
                plot = pg.PlotWidget()
                plot.setFixedHeight(168)
                plot.setMinimumWidth(210)
                plot.setMenuEnabled(True)
                plot.showGrid(x=True, y=True, alpha=0.15)
                plot.plot(xdata, ydata, pen=pg.mkPen((170, 170, 170, 120), width=1))
                plot.addItem(pg.ScatterPlotItem(
                    x=xdata, y=ydata, size=5, pen=None, brush=brushes,
                    data=tips, hoverable=True, hoverSize=9,
                    hoverPen=pg.mkPen(255, 255, 255, 220), tip=_point_tip))
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
