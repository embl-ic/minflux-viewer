"""
minflux_viewer.plugins.trace_viewer.trace_viewer_window
=======================================================
Trace Viewer window — a compact, app-consistent reproduction of the MATLAB
``MINFLUX_data_plot_trace_viewer``.

Three resizable regions (top = trace scatter | time-series, bottom = trace
table):

* **Scatter** of the selected trace (centred), with an orientation selector and
  an animated *head* marker (yellow star, averaged over ±head) and *tail* (magenta
  path over the last *tail* points).
* **Four linked time-series** — dt (ms), RMS velocity (µm/s), eco + eco-RMS,
  efo + efo-RMS — with a **play** button and a **time slider** that scrub a
  magenta window highlight across all four.
* **Sortable table** of every trace (ID, N loc, efo, [ch1/ch2], dcr, dt, velocity,
  size) with a **tid** picker, **Tile View**, **Overlay**, **remove tail** and
  **Export**.

All per-trace maths live in :mod:`trace_data` (Qt-free, unit-tested).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...core.loader import attr_values_1d
from ...ui.plot_format import plot_widget
from . import trace_data as td

_PLANE = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}
_HEAD_BRUSH = pg.mkBrush(255, 230, 40, 255)
_TAIL_PEN = pg.mkPen(230, 40, 200, width=2)
_REGION_BRUSH = (230, 40, 200, 60)
_OVERLAY_COLORS = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (23, 190, 207),
]


def _alive(widget) -> bool:
    """True if the underlying C++ widget still exists (not yet deleted)."""
    try:
        widget.objectName()
        return True
    except RuntimeError:
        return False


class TraceViewerWindow(QDialog):
    """Per-trace inspection window (Plugins › Trace Viewer)."""

    def __init__(self, state, dataset_idx: int, owner=None) -> None:
        super().__init__(None)
        self._state = state
        self._idx = dataset_idx
        self._owner = owner
        self.setWindowTitle("Trace Viewer")
        self.resize(1040, 820)

        self._tid = None                 # working tid array (may have tail removed)
        self._rows: list[dict] = []
        self._headers: list[str] = []
        self._series: dict = {}
        self._selected_tid: int | None = None
        self._child_windows: list = []   # keep Tile/Overlay windows alive (else GC'd)
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(40)
        self._play_timer.timeout.connect(self._advance_frame)

        self._build_ui()
        self._reload_tid(reset=True)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        ds = self._dataset()
        title = QLabel(f"<b>{ds.name if ds else '(no dataset)'}</b>")
        title.setWordWrap(True)
        root.addWidget(title)

        outer = QSplitter(Qt.Orientation.Vertical)
        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self._build_scatter_panel())
        top.addWidget(self._build_timeseries_panel())
        top.setSizes([440, 600])
        outer.addWidget(top)
        outer.addWidget(self._build_table_panel())
        outer.setSizes([520, 300])
        root.addWidget(outer, 1)

    def _build_scatter_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(3)

        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("View:"))
        self._orient_combo = QComboBox()
        self._orient_combo.addItems(["XY", "XZ", "YZ"])
        self._orient_combo.currentTextChanged.connect(lambda *_: self._redraw_scatter())
        ctl.addWidget(self._orient_combo)
        ctl.addSpacing(8)
        ctl.addWidget(QLabel("head ±"))
        self._head_spin = QSpinBox()
        self._head_spin.setRange(0, 9999)
        self._head_spin.setValue(2)
        self._head_spin.valueChanged.connect(lambda *_: self._frame_update())
        ctl.addWidget(self._head_spin)
        ctl.addWidget(QLabel("tail"))
        self._tail_spin = QSpinBox()
        self._tail_spin.setRange(0, 99999)
        self._tail_spin.setValue(10)
        self._tail_spin.valueChanged.connect(lambda *_: self._frame_update())
        ctl.addWidget(self._tail_spin)
        ctl.addStretch(1)
        lay.addLayout(ctl)

        self._scatter_plot = plot_widget()
        self._scatter_plot.setAspectLocked(True)
        self._scatter_plot.showGrid(x=True, y=True, alpha=0.15)
        self._scatter_item = pg.ScatterPlotItem(size=5, pen=None, brush=pg.mkBrush(70, 130, 200, 180))
        self._tail_item = pg.PlotDataItem(pen=_TAIL_PEN)
        self._head_item = pg.ScatterPlotItem(size=16, symbol="star", brush=_HEAD_BRUSH,
                                              pen=pg.mkPen(120, 90, 0, width=1))
        for it in (self._scatter_item, self._tail_item, self._head_item):
            self._scatter_plot.addItem(it)
        lay.addWidget(self._scatter_plot, 1)
        return w

    def _build_timeseries_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)

        self._ts_plots: dict[str, pg.PlotWidget] = {}
        self._ts_curves: dict[str, pg.PlotDataItem] = {}
        self._ts_rms: dict[str, pg.PlotDataItem] = {}
        self._ts_regions: dict[str, pg.LinearRegionItem] = {}
        specs = [("dt", "dt (ms)", False), ("v", "V (µm/s)", False),
                 ("eco", "eco", True), ("efo", "efo", True)]
        prev = None
        for key, ylabel, has_rms in specs:
            p = plot_widget()
            p.setLabel("left", ylabel)
            p.showGrid(x=True, y=True, alpha=0.12)
            p.setMinimumHeight(90)
            if key == "efo":
                p.setLabel("bottom", "t (s)")
            else:
                p.getAxis("bottom").setStyle(showValues=False)
            if prev is not None:
                p.setXLink(prev)
            prev = p
            self._ts_curves[key] = p.plot([], [], pen=pg.mkPen(70, 130, 200, 220)
                                          if not has_rms else pg.mkPen(120, 120, 120, 200))
            if key in ("dt", "v"):
                self._ts_curves[key].setSymbol("o")
                self._ts_curves[key].setSymbolSize(3)
                self._ts_curves[key].setSymbolBrush(pg.mkBrush(70, 130, 200, 200))
                self._ts_curves[key].setPen(None)
            if has_rms:
                self._ts_rms[key] = p.plot([], [], pen=pg.mkPen(220, 40, 40, width=2))
            region = pg.LinearRegionItem(values=(0, 0), movable=False,
                                         brush=pg.mkBrush(*_REGION_BRUSH))
            region.setZValue(-10)
            region.setVisible(False)
            p.addItem(region)
            self._ts_regions[key] = region
            self._ts_plots[key] = p
            lay.addWidget(p, 1)

        # play + time slider row
        row = QHBoxLayout()
        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedWidth(32)
        self._play_btn.setCheckable(True)
        self._play_btn.toggled.connect(self._on_play_toggled)
        row.addWidget(self._play_btn)
        self._time_slider = QSlider(Qt.Orientation.Horizontal)
        self._time_slider.setMinimum(0)
        self._time_slider.setMaximum(0)
        self._time_slider.valueChanged.connect(lambda *_: self._frame_update())
        row.addWidget(self._time_slider, 1)
        lay.addLayout(row)
        return w

    def _build_table_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(3)

        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("tid:"))
        self._tid_combo = QComboBox()
        self._tid_combo.setMinimumWidth(90)
        self._tid_combo.currentTextChanged.connect(self._on_tid_combo)
        ctl.addWidget(self._tid_combo)
        ctl.addSpacing(8)
        for text, slot in (("Tile View", self._on_tile_view), ("Overlay", self._on_overlay)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            ctl.addWidget(b)
        self._remove_tail_chk = QCheckBox("remove tail")
        self._remove_tail_chk.setToolTip(
            "Zero each trace's leading approach segment (velocity + distance heuristic).")
        self._remove_tail_chk.toggled.connect(self._on_remove_tail)
        ctl.addWidget(self._remove_tail_chk)
        exp = QPushButton("Export…")
        exp.clicked.connect(self._on_export)
        ctl.addWidget(exp)
        ctl.addStretch(1)
        self._count_label = QLabel("")
        self._count_label.setStyleSheet("color: gray;")
        ctl.addWidget(self._count_label)
        lay.addLayout(ctl)

        self._table = QTableWidget()
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSortingEnabled(True)
        self._table.verticalHeader().setVisible(False)
        self._table.itemSelectionChanged.connect(self._on_table_selection)
        lay.addWidget(self._table, 1)
        return w

    # ------------------------------------------------------------ data path
    def _dataset(self):
        if 0 <= self._idx < len(self._state.datasets):
            return self._state.datasets[self._idx]
        return None

    def _reload_tid(self, *, reset: bool) -> None:
        ds = self._dataset()
        if ds is None:
            return
        if reset or self._tid is None:
            v = attr_values_1d(ds, "tid")
            self._tid = np.zeros(0) if v is None else np.asarray(v).ravel().copy()
        self._rebuild_table()

    def _rebuild_table(self) -> None:
        ds = self._dataset()
        if ds is None:
            return
        self._rows, self._headers = td.build_trace_table(ds, self._tid)
        self._table.setSortingEnabled(False)
        self._table.clear()
        self._table.setColumnCount(len(self._headers))
        self._table.setHorizontalHeaderLabels(self._headers)
        self._table.setRowCount(len(self._rows))
        for r, row in enumerate(self._rows):
            for c, key in enumerate(self._headers):
                val = row.get(key)
                item = QTableWidgetItem()
                if isinstance(val, (int, np.integer)) or key in ("trace ID", "N loc"):
                    item.setData(Qt.ItemDataRole.DisplayRole, int(val))
                else:
                    fval = float(val)
                    item.setData(Qt.ItemDataRole.EditRole, fval)
                    item.setText("" if not np.isfinite(fval) else f"{fval:.3g}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(r, c, item)
        self._table.setSortingEnabled(True)
        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        ids = [str(row["trace ID"]) for row in self._rows]
        self._tid_combo.blockSignals(True)
        self._tid_combo.clear()
        self._tid_combo.addItems(ids)
        self._tid_combo.blockSignals(False)
        self._count_label.setText(f"{len(self._rows)} traces")

        if self._rows:
            keep = self._selected_tid if self._selected_tid in [row["trace ID"] for row in self._rows] else self._rows[0]["trace ID"]
            self._select_trace(keep)

    # ------------------------------------------------------- trace selection
    def _select_trace(self, trace_id: int) -> None:
        self._selected_tid = int(trace_id)
        ds = self._dataset()
        if ds is None:
            return
        self._series = td.trace_time_series(ds, self._tid, self._selected_tid)
        self._tid_combo.blockSignals(True)
        i = self._tid_combo.findText(str(self._selected_tid))
        if i >= 0:
            self._tid_combo.setCurrentIndex(i)
        self._tid_combo.blockSignals(False)
        # slider spans the full trace
        n = self._series["n"]
        self._time_slider.blockSignals(True)
        self._time_slider.setMaximum(max(0, n - 1))
        self._time_slider.setValue(0)
        self._time_slider.blockSignals(False)
        self._redraw_scatter()
        self._redraw_timeseries()
        self._frame_update()

    def _on_table_selection(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        item = self._table.item(rows[0].row(), 0)
        if item is not None:
            self._select_trace(int(item.data(Qt.ItemDataRole.DisplayRole)))

    def _on_tid_combo(self, text: str) -> None:
        if text:
            try:
                self._select_trace(int(text))
            except ValueError:
                pass

    # ------------------------------------------------------------- drawing
    def _redraw_scatter(self) -> None:
        xyz = self._series.get("xyz", np.empty((0, 3)))
        i, j = _PLANE[self._orient_combo.currentText()]
        if xyz.shape[0]:
            self._scatter_item.setData(x=xyz[:, i], y=xyz[:, j])
        else:
            self._scatter_item.setData(x=[], y=[])
        self._scatter_plot.setTitle(f"tid: {self._selected_tid}")
        a = "XYZ"
        self._scatter_plot.setLabel("bottom", f"{a[i]} (nm)")
        self._scatter_plot.setLabel("left", f"{a[j]} (nm)")

    def _redraw_timeseries(self) -> None:
        s = self._series
        ts = s.get("ts", np.zeros(0))
        self._ts_curves["dt"].setData(ts, s.get("dt_ms", np.zeros(0)))
        self._ts_curves["v"].setData(ts, s.get("v_rms", np.zeros(0)))
        self._ts_curves["eco"].setData(ts, s.get("eco", np.zeros(0)))
        self._ts_rms["eco"].setData(ts, s.get("eco_rms", np.zeros(0)))
        self._ts_curves["efo"].setData(ts, s.get("efo", np.zeros(0)))
        self._ts_rms["efo"].setData(ts, s.get("efo_rms", np.zeros(0)))
        for p in self._ts_plots.values():
            p.enableAutoRange()

    def _frame_update(self) -> None:
        s = self._series
        xyz = s.get("xyz", np.empty((0, 3)))
        n = xyz.shape[0]
        rng = self._tail_spin.value()
        if n <= 1 or rng == 0:
            self._head_item.setData(x=[], y=[])
            self._tail_item.setData([], [])
            for region in self._ts_regions.values():
                region.setVisible(False)
            return
        pos = int(np.clip(self._time_slider.value(), 0, n - 1))
        head = self._head_spin.value()
        i, j = _PLANE[self._orient_combo.currentText()]

        hb, he = max(0, pos - head), min(n, pos + head + 1)
        hc = xyz[hb:he].mean(axis=0)
        self._head_item.setData(x=[hc[i]], y=[hc[j]])

        tail = xyz[max(0, pos - rng):pos + 1]
        self._tail_item.setData(tail[:, i], tail[:, j])

        t = s.get("t", np.zeros(0))
        if t.size:
            left = float(t[max(0, pos - rng)])
            right = float(t[pos])
            for region in self._ts_regions.values():
                region.setRegion((left, right))
                region.setVisible(True)

    # ------------------------------------------------------------- play
    def _on_play_toggled(self, on: bool) -> None:
        self._play_btn.setText("❚❚" if on else "▶")
        if on:
            if self._time_slider.value() >= self._time_slider.maximum():
                self._time_slider.setValue(0)
            self._play_timer.start()
        else:
            self._play_timer.stop()

    def _advance_frame(self) -> None:
        v = self._time_slider.value()
        if v >= self._time_slider.maximum():
            self._play_btn.setChecked(False)
            return
        self._time_slider.setValue(v + 1)

    # ------------------------------------------------------ tile / overlay
    def _selected_trace_ids(self) -> list[int]:
        out = []
        for r in self._table.selectionModel().selectedRows():
            item = self._table.item(r.row(), 0)
            if item is not None:
                out.append(int(item.data(Qt.ItemDataRole.DisplayRole)))
        return out

    def _ordered_trace_ids(self) -> list[int]:
        """Trace IDs in the table's current (possibly sorted) display order."""
        return [int(self._table.item(r, 0).data(Qt.ItemDataRole.DisplayRole))
                for r in range(self._table.rowCount())]

    def _on_tile_view(self) -> None:
        dlg = _TileDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        rng, layout = dlg.values()
        ids = self._ordered_trace_ids()
        begin = max(1, rng[0]) - 1
        end = min(rng[1], len(ids))
        chosen = ids[begin:end]
        if not chosen:
            return
        nrow, ncol = layout
        if nrow * ncol < len(chosen):
            nrow = int(np.ceil(len(chosen) / ncol))
        self._keep_child(_TileWindow(self, chosen, nrow, ncol))

    def _on_overlay(self) -> None:
        ids = self._selected_trace_ids()
        if len(ids) < 2:
            QMessageBox.information(self, "Overlay", "Select two or more traces in the table to overlay.")
            return
        self._keep_child(_OverlayWindow(self, ids))

    def _keep_child(self, win) -> None:
        """Retain a Tile/Overlay window so Python doesn't GC the top-level widget
        (which would make it never appear); prune already-closed ones."""
        self._child_windows = [w for w in self._child_windows if _alive(w)]
        self._child_windows.append(win)
        win.show()
        win.raise_()
        win.activateWindow()

    def _trace_xyz(self, trace_id: int) -> np.ndarray:
        ds = self._dataset()
        loc = td._loc_nm(ds)
        m = np.asarray(self._tid).ravel() == int(trace_id)
        xyz = loc[m]
        return xyz - xyz.mean(axis=0) if xyz.shape[0] else xyz

    # --------------------------------------------------- remove tail / export
    def _on_remove_tail(self, checked: bool) -> None:
        ds = self._dataset()
        if ds is None:
            return
        if checked:
            self._tid = td.remove_trace_tail(ds)
        else:
            v = attr_values_1d(ds, "tid")
            self._tid = np.zeros(0) if v is None else np.asarray(v).ravel().copy()
        self._rebuild_table()

    def _on_export(self) -> None:
        ids = self._selected_trace_ids()
        if len(ids) <= 1:
            ids = self._ordered_trace_ids()
        if not ids:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export trace data", f"{self._dataset().name}_traces.npz",
            "NumPy archive (*.npz)")
        if not path:
            return
        payload = td.export_payload(self._dataset(), self._tid, ids)
        flat = {}
        for rec in payload:
            t = rec["tid"]
            for k, v in rec.items():
                if k == "tid":
                    continue
                flat[f"t{t}_{k}"] = np.asarray(v)
        try:
            np.savez_compressed(path, tids=np.array(ids), **flat)
            self._state.log(f"Exported {len(ids)} trace(s) to {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export", f"Could not save: {exc}")

    def closeEvent(self, event) -> None:
        self._play_timer.stop()
        for w in list(self._child_windows):
            if _alive(w):
                w.close()
        self._child_windows = []
        super().closeEvent(event)


# --------------------------------------------------------------- helpers
class _TileDialog(QDialog):
    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tile Layout")
        lay = QVBoxLayout(self)
        form = QHBoxLayout()
        form.addWidget(QLabel("Show traces:"))
        self._range_edit = QLineEdit("1-20")
        form.addWidget(self._range_edit)
        form.addWidget(QLabel("Layout (R x C):"))
        self._layout_edit = QLineEdit("4x5")
        form.addWidget(self._layout_edit)
        lay.addLayout(form)
        btns = QHBoxLayout()
        btns.addStretch(1)
        ok = QPushButton("OK"); ok.clicked.connect(self.accept)
        cancel = QPushButton("Cancel"); cancel.clicked.connect(self.reject)
        btns.addWidget(ok); btns.addWidget(cancel)
        lay.addLayout(btns)

    def values(self):
        try:
            a, b = self._range_edit.text().split("-")
            rng = (int(a), int(b))
        except Exception:
            rng = (1, 20)
        try:
            r, c = self._layout_edit.text().lower().split("x")
            layout = (int(r), int(c))
        except Exception:
            layout = (4, 5)
        return rng, layout


class _TileWindow(QWidget):
    def __init__(self, viewer: TraceViewerWindow, ids: list[int], nrow: int, ncol: int) -> None:
        super().__init__(None)
        self.setWindowTitle("Trace Tile View")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(min(220 * ncol, 1400), min(200 * nrow, 1000))
        self._viewer = viewer
        grid = QGridLayout(self)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(3)
        for k, tid in enumerate(ids):
            p = plot_widget()
            p.setAspectLocked(True)
            p.setTitle(f"tid: {tid}")
            p.getAxis("bottom").setStyle(showValues=False)
            p.getAxis("left").setStyle(showValues=False)
            xyz = viewer._trace_xyz(tid)
            if xyz.shape[0]:
                p.plot(xyz[:, 0], xyz[:, 1], pen=None, symbol="o", symbolSize=3,
                       symbolBrush=pg.mkBrush(70, 130, 200, 200), symbolPen=None)
            p.scene().sigMouseClicked.connect(lambda _ev, t=tid: viewer._select_trace(t))
            grid.addWidget(p, k // ncol, k % ncol)


class _OverlayWindow(QWidget):
    def __init__(self, viewer: TraceViewerWindow, ids: list[int]) -> None:
        super().__init__(None)
        self.setWindowTitle("Trace Overlay")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(640, 640)
        lay = QVBoxLayout(self)
        p = plot_widget()
        p.setAspectLocked(True)
        p.showGrid(x=True, y=True, alpha=0.15)
        p.addLegend()
        for k, tid in enumerate(ids):
            xyz = viewer._trace_xyz(tid)
            if xyz.shape[0]:
                col = _OVERLAY_COLORS[k % len(_OVERLAY_COLORS)]
                p.plot(xyz[:, 0], xyz[:, 1], pen=None, symbol="o", symbolSize=4,
                       symbolBrush=pg.mkBrush(*col, 200), symbolPen=None, name=f"tid {tid}")
        p.setLabel("bottom", "X (nm)")
        p.setLabel("left", "Y (nm)")
        lay.addWidget(p)
