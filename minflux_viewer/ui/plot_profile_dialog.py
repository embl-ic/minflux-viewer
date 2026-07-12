"""
minflux_viewer.ui.plot_profile_dialog
======================================
ImageJ/Fiji-style **Plot Profile** for a line / polyline / freehand-line ROI.

Samples the **localizations directly** (not the rendered raster) along the active
open-line ROI and plots density vs. distance-from-start (nm), integrating across a
tunable perpendicular width (drawn as a translucent band on the view). Because it
reads the data — not the display — the profile is independent of the render
viewport / zoom, and the same dialog works on a **render or a scatter** view. The
curve and the band track the line **live** as it is drawn or edited; hovering the
curve reads off any point; closing the dialog removes the band.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtGui import QPainterPath, QPolygonF
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGraphicsPathItem,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..analysis.plot_profile import band_polygon, line_profile_from_locs, path_cumlen

_CURVE_COLOR = "#4a90ff"
_POLL_MS = 80          # ~12 fps — smooth enough to track a line drag live
_DEFAULT_WIDTH_NM = 20.0
_MISSING = object()


class PlotProfileDialog(QWidget):
    """Live localization line-profile of the active open-line ROI in a 2-D view.

    Non-owned modeless window (shown via :func:`ui.modeless.show_modeless`). Binds
    to one render/scatter window; polls its active line ROI and re-bins the
    localizations when the line / width / bin / data change (never on zoom/pan).
    The width band it draws on the view is removed on close."""

    def __init__(self, view_win, owner=None) -> None:
        super().__init__(None)
        self._view = view_win
        self._owner = owner
        self._band_item: QGraphicsPathItem | None = None
        self._last_sig = None
        self._locs: np.ndarray | None = None
        self._locs_version = _MISSING
        self.setWindowTitle("Plot Profile")
        self.resize(680, 460)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self._info = QLabel("Draw a line, polyline or freehand line on the view.")
        self._info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info.setStyleSheet("color: #bbb;")
        root.addWidget(self._info)

        self._plot = pg.PlotWidget(background="#222")
        self._plot.showGrid(x=True, y=True, alpha=0.22)
        self._plot.setLabel("bottom", "distance (nm)")
        self._plot.setLabel("left", "density (loc/µm²)")
        self._curve = self._plot.plot([], [], pen=pg.mkPen(_CURVE_COLOR, width=2.0))
        # Hover read-out: a snap-to-nearest crosshair + dot + a (distance, density)
        # label, so any point of the profile (e.g. a peak) can be read by hovering.
        self._hover_vline = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen("#888", width=1, style=Qt.PenStyle.DashLine))
        self._hover_dot = pg.ScatterPlotItem(
            size=10, brush=pg.mkBrush(_CURVE_COLOR), pen=pg.mkPen("w", width=1.2))
        self._hover_text = pg.TextItem(color="#eee", anchor=(0, 1),
                                       fill=pg.mkBrush(20, 20, 20, 210))
        for it in (self._hover_vline, self._hover_dot, self._hover_text):
            it.setZValue(60)
            it.setVisible(False)
            self._plot.addItem(it, ignoreBounds=True)
        self._plot.scene().sigMouseMoved.connect(self._on_hover)
        root.addWidget(self._plot, stretch=1)

        row = QHBoxLayout()
        row.addWidget(QLabel("Line width (nm):"))
        self._width = QDoubleSpinBox()
        self._width.setRange(0.1, 1_000_000.0)
        self._width.setDecimals(1)
        self._width.setSingleStep(5.0)
        self._width.setValue(_DEFAULT_WIDTH_NM)
        self._width.setToolTip(
            "Perpendicular band across which localizations are integrated. Shown as "
            "a translucent band on the view; a wider band captures more, smoother.")
        self._width.valueChanged.connect(self._recompute_now)
        row.addWidget(self._width)
        row.addSpacing(12)
        row.addWidget(QLabel("bin (nm):"))
        self._bin = QDoubleSpinBox()
        self._bin.setRange(0.0, 1_000_000.0)
        self._bin.setDecimals(1)
        self._bin.setSingleStep(1.0)
        self._bin.setValue(0.0)
        self._bin.setSpecialValueText("auto")
        self._bin.setToolTip("Along-line bin size. 'auto' = ~256 bins over the line.")
        self._bin.valueChanged.connect(self._recompute_now)
        row.addWidget(self._bin)
        row.addStretch()
        save = QPushButton("Save…")
        save.setToolTip("Save the profile (distance, density) as CSV.")
        save.clicked.connect(self._save_csv)
        row.addWidget(save)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        row.addWidget(close)
        root.addLayout(row)

        self._dist = np.zeros(0)
        self._vals = np.zeros(0)

        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._tick()

    # ------------------------------------------------------------------ poll

    def _alive(self) -> bool:
        try:
            self._view.coordinate_view_box()
            return True
        except RuntimeError:
            return False

    def _recompute_now(self) -> None:
        self._last_sig = None
        self._tick()

    def _tick(self) -> None:
        if not self._alive():
            self._timer.stop()
            return
        overlay = getattr(self._view, "_roi_overlay", None)
        record = overlay.active_open_line_record() if overlay is not None else None
        if record is None:
            self._clear("Draw or select a line, polyline or freehand line on the view.")
            self._hide_band()
            self._last_sig = None
            return
        # Read the **live** item vertices (track a drag in progress, not just the
        # committed-on-release record geometry) — Fiji-style live update.
        live_pts = overlay.active_open_line_points()
        pts = np.asarray(live_pts if live_pts is not None else [], dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            self._clear("The line needs at least two vertices.")
            self._hide_band()
            return

        version = self._loc_version()
        width = float(self._width.value())
        bin_nm = float(self._bin.value()) or None
        sig = (round(width, 3), round(self._bin.value(), 3), np.round(pts, 1).tobytes(),
               version, record.stroke_color)
        if sig == self._last_sig:
            return
        self._last_sig = sig

        total = float(path_cumlen(pts)[-1])
        self._update_band(pts, width, max(total / 200.0, 1.0), record.stroke_color)

        # Fetch the localizations only when the data (not the view) changed.
        if version is not _MISSING and version != self._locs_version:
            self._locs = self._fetch_locs()
            self._locs_version = version
        if self._locs is None:
            self._clear("Plot Profile needs a 2-D localization view (render or scatter).")
            return

        centres, counts, bin_w = line_profile_from_locs(self._locs, pts, width, bin_nm=bin_nm)
        if centres.size == 0:
            self._clear("Line/width too small to profile.")
            return
        # counts per bin → areal density (loc/µm²): / (bin_length · width) · 1e6.
        if width > 0 and bin_w > 0:
            density = counts / (bin_w * width) * 1.0e6
        else:
            density = counts
        self._dist, self._vals = centres, density
        self._draw_curve(centres, density, length=total, width=width,
                         n_loc=int(counts.sum()))

    def _loc_version(self):
        getter = getattr(self._view, "profile_locs_version", None)
        try:
            return getter() if callable(getter) else None
        except Exception:
            return None

    def _fetch_locs(self):
        getter = getattr(self._view, "profile_localizations", None)
        try:
            return getter() if callable(getter) else None
        except Exception:
            return None

    # ---------------------------------------------------------------- hover

    def _nearest_profile_point(self, view_x: float):
        """The profile sample nearest a given distance (view x) →
        ``(distance_nm, density)`` or ``None`` when there's no profile."""
        if self._dist.size == 0:
            return None
        finite = np.isfinite(self._vals)
        if not finite.any():
            return None
        d = self._dist[finite]
        v = self._vals[finite]
        i = int(np.argmin(np.abs(d - float(view_x))))
        return float(d[i]), float(v[i])

    def _on_hover(self, pos) -> None:
        """Snap a crosshair + read-out to the profile point nearest the cursor's
        distance, so any point (peaks included) can be read off directly."""
        if self._dist.size == 0 or not self._plot.sceneBoundingRect().contains(pos):
            self._set_hover_visible(False)
            return
        vb = self._plot.getPlotItem().vb
        nearest = self._nearest_profile_point(vb.mapSceneToView(pos).x())
        if nearest is None:
            self._set_hover_visible(False)
            return
        dx, dy = nearest
        self._hover_vline.setPos(dx)
        self._hover_dot.setData([dx], [dy])
        self._hover_text.setText(f"d = {dx:.1f} nm\nρ = {dy:.3g}")
        self._hover_text.setAnchor(self._label_anchor(dx, dy, *vb.viewRange()))
        self._hover_text.setPos(dx, dy)
        self._set_hover_visible(True)

    @staticmethod
    def _label_anchor(dx, dy, x_range, y_range):
        """Text anchor ``(ax, ay)`` that keeps the read-out label inside the view:
        put it to the **left** near the right edge, and **below** the point when it
        is in the upper half — so a peak's label isn't clipped off the top.
        ``ay=0`` → label below the point; ``ay=1`` → above; ``ax=1`` → left."""
        x0, x1 = x_range
        y0, y1 = y_range
        ax = 1 if dx > 0.5 * (x0 + x1) else 0
        ay = 0 if dy > 0.5 * (y0 + y1) else 1
        return (ax, ay)

    def _set_hover_visible(self, visible: bool) -> None:
        for it in (self._hover_vline, self._hover_dot, self._hover_text):
            it.setVisible(visible)

    # --------------------------------------------------------------- drawing

    def _clear(self, message: str) -> None:
        self._curve.setData([], [])
        self._dist = np.zeros(0)
        self._vals = np.zeros(0)
        self._set_hover_visible(False)
        self._info.setText(message)

    def _draw_curve(self, dist, vals, *, length: float, width: float, n_loc: int) -> None:
        self._curve.setData(dist, vals)
        self._info.setText(
            f"length {length:.0f} nm · width {width:.0f} nm · "
            f"{dist.size} bins · {n_loc:,} localizations in band")

    def _update_band(self, pts, width: float, step: float, color) -> None:
        vb = self._view.coordinate_view_box()
        if vb is None:
            self._hide_band()
            return
        poly = band_polygon(pts, width, step)
        if poly.shape[0] < 3:
            self._hide_band()
            return
        if self._band_item is None:
            self._band_item = QGraphicsPathItem()
            # Above the opaque render image (ImageItem z=0); below the ROI edit
            # handles (z=11) so they stay grabbable.
            self._band_item.setZValue(5)
            vb.addItem(self._band_item, ignoreBounds=True)
        col = pg.mkColor(color or "#4a90ff")
        fill = pg.mkColor(col)
        fill.setAlpha(45)
        edge = pg.mkColor(col)
        edge.setAlpha(170)
        pen = pg.mkPen(edge, width=1.4)
        pen.setCosmetic(True)          # constant-pixel edge so the two sides stay visible
        self._band_item.setPen(pen)
        self._band_item.setBrush(pg.mkBrush(fill))
        path = QPainterPath()
        path.addPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in poly]))
        path.closeSubpath()
        self._band_item.setPath(path)
        self._band_item.setVisible(True)

    def _hide_band(self) -> None:
        if self._band_item is not None:
            self._band_item.setVisible(False)

    def _remove_band(self) -> None:
        if self._band_item is None:
            return
        try:
            vb = self._view.coordinate_view_box()
            if vb is not None:
                vb.removeItem(self._band_item)
        except Exception:
            pass
        self._band_item = None

    # ----------------------------------------------------------------- misc

    def _save_csv(self) -> None:
        if self._dist.size == 0:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save profile", "line_profile.csv", "CSV (*.csv)")
        if not path:
            return
        data = np.column_stack([self._dist, self._vals])
        try:
            np.savetxt(path, data, delimiter=",",
                       header="distance_nm,density_loc_per_um2", comments="", fmt="%.4f")
        except Exception as exc:
            state = getattr(self._owner, "_state", None)
            if state is not None:
                state.log(f"Plot Profile: could not save CSV: {exc}", "WARN")

    def closeEvent(self, event) -> None:
        self._timer.stop()
        self._remove_band()
        super().closeEvent(event)
