"""
minflux_viewer.ui.particle_inspector
=====================================
Per-particle inspector for the Particle Average tool.

Opened from a row of the particle fit table, it shows **one** particle's
localization point cloud (2-D or 3-D scatter) with the **fitted geometry model**
overlaid on top — the two-ring model (NPC Wanlu), the geometry template
(template-provided) or the pooled reference (template-free). Four checkboxes
toggle, independently: the **point cloud**, the **template/model**, the **axis**
and the **bounding box**. The axis + bounding box mirror the 3-D scatter view's
reference items (coloured X/Y/Z axes with numeric ticks + a data-extent
wireframe box in 3-D; the plot axes + an extent rectangle in 2-D).

Modeless / non-owned per the project window convention (shown via
``ui.modeless.show_modeless``).
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

_CLOUD_RGBA = (0.35, 0.72, 1.0, 0.9)      # localizations — light blue
_MODEL_RGBA = (1.0, 0.55, 0.15, 1.0)      # fitted model — orange
# Reference-item colours (match the 3-D scatter view on a black background).
_AXIS_COLORS = ((0.9, 0.15, 0.15, 0.95), (0.15, 0.7, 0.25, 0.95), (0.15, 0.35, 0.95, 0.95))
_REF_RGBA = (0.72, 0.72, 0.72, 0.72)
_TEXT_RGBA = (230, 230, 230, 230)


class ParticleInspectorWindow(QWidget):
    """Point cloud of one particle + its fitted model overlay (2-D or 3-D)."""

    def __init__(self, *, title: str, cloud, model_points=None, model_curves=None,
                 is_3d: bool = False, owner=None) -> None:
        super().__init__(None)
        self._owner = owner
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.resize(600, 640)

        self._cloud = _as_xyz(cloud)
        self._model_points = _as_xyz(model_points) if model_points is not None else None
        self._model_curves = [_as_xyz(c) for c in (model_curves or []) if c is not None]
        self._is_3d = bool(is_3d)
        self._plot = None          # 2-D PlotWidget (or None in 3-D)
        self._gl_view = None       # 3-D GLViewWidget (or None in 2-D)
        self._cloud_items: list = []
        self._model_items: list = []
        self._axis_items: list = []
        self._box_items: list = []

        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self._cloud_chk = self._toggle(top, "Point cloud", True, True)
        self._model_chk = self._toggle(top, "Template / model", True, self._has_model())
        self._axis_chk = self._toggle(top, "Axis", True, True)
        self._box_chk = self._toggle(top, "Bounding box", True, True)
        top.addStretch(1)
        top.addWidget(QLabel(f"{self._cloud.shape[0]:,} locs · {'3-D' if self._is_3d else '2-D'}"))
        root.addLayout(top)

        root.addWidget(self._build_view(), 1)
        self._apply_visibility()

    def _toggle(self, layout, text: str, checked: bool, enabled: bool) -> QCheckBox:
        chk = QCheckBox(text)
        chk.setChecked(checked)
        chk.setEnabled(enabled)
        chk.toggled.connect(self._apply_visibility)
        layout.addWidget(chk)
        return chk

    def _has_model(self) -> bool:
        return bool((self._model_points is not None and self._model_points.shape[0])
                    or self._model_curves)

    def _scene_points(self) -> np.ndarray:
        """All drawn points (cloud + model) — for the bounding box / axis extent."""
        parts = [self._cloud[:, :3]]
        if self._model_points is not None and self._model_points.shape[0]:
            parts.append(self._model_points[:, :3])
        parts.extend(c[:, :3] for c in self._model_curves if c.size)
        parts = [p for p in parts if p.size]
        return np.vstack(parts) if parts else np.empty((0, 3))

    # ------------------------------------------------------------------ views
    def _build_view(self) -> QWidget:
        if self._is_3d:
            w = self._build_3d()
            if w is not None:
                return w
            self._is_3d = False                 # OpenGL unavailable → 2-D fallback
        return self._build_2d()

    def _build_2d(self) -> QWidget:
        import pyqtgraph as pg

        plot = pg.PlotWidget()
        plot.setBackground("k")
        plot.setAspectLocked(True)
        plot.addLegend(offset=(10, 10))
        plot.setLabel("bottom", "X (nm)")
        plot.setLabel("left", "Y (nm)")
        self._plot = plot
        c = self._cloud
        sc = pg.ScatterPlotItem(x=c[:, 0], y=c[:, 1], size=4,
                                brush=pg.mkBrush(90, 185, 255, 200), pen=None, name="locs")
        plot.addItem(sc)
        self._cloud_items = [sc]
        model_pen = pg.mkPen(color=(255, 140, 40), width=2)
        for curve in self._model_curves:
            item = pg.PlotDataItem(curve[:, 0], curve[:, 1], pen=model_pen)
            plot.addItem(item)
            self._model_items.append(item)
        if self._model_points is not None and self._model_points.shape[0]:
            mp = pg.ScatterPlotItem(x=self._model_points[:, 0], y=self._model_points[:, 1],
                                    size=9, symbol="o", brush=pg.mkBrush(255, 140, 40, 220),
                                    pen=pg.mkPen("w", width=1))
            plot.addItem(mp)
            self._model_items.append(mp)
        # 2-D bounding box: a rectangle at the data extent (analogue of the 3-D box).
        b = _xy_bounds(self._scene_points())
        if b is not None:
            x0, y0, x1, y1 = b
            rect = pg.PlotDataItem([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                                   pen=pg.mkPen(_ref_rgba255(), width=1.2))
            plot.addItem(rect)
            self._box_items = [rect]
        return plot

    def _build_3d(self):
        try:
            import pyqtgraph as pg
            import pyqtgraph.opengl as gl
        except Exception:
            return None
        # Recentre on the cloud centroid for camera precision (absolute nm can be
        # large). Everything drawn is offset by the same anchor.
        anchor = self._cloud[:, :3].mean(axis=0) if self._cloud.shape[0] else np.zeros(3)
        view = gl.GLViewWidget()
        view.setBackgroundColor("k")
        self._gl_view = view
        cpts = (self._cloud[:, :3] - anchor).astype(np.float32)
        scatter = gl.GLScatterPlotItem(pos=cpts, color=_CLOUD_RGBA, size=4.0, pxMode=True)
        view.addItem(scatter)
        self._cloud_items = [scatter]
        for curve in self._model_curves:
            line = gl.GLLinePlotItem(pos=(curve[:, :3] - anchor).astype(np.float32),
                                     color=_MODEL_RGBA, width=2.0, antialias=True)
            view.addItem(line)
            self._model_items.append(line)
        if self._model_points is not None and self._model_points.shape[0]:
            mp = gl.GLScatterPlotItem(pos=(self._model_points[:, :3] - anchor).astype(np.float32),
                                      color=_MODEL_RGBA, size=10.0, pxMode=True)
            view.addItem(mp)
            self._model_items.append(mp)
        # Reference items over the recentred scene bounds.
        bounds = _expanded_bounds(self._scene_points() - anchor)
        if bounds is not None:
            mins, maxs, spans = bounds
            self._make_grid(view, mins, maxs, spans)
            self._make_axis(view, mins, maxs, spans)
            self._make_box(view, mins, maxs)
        span = float(np.ptp(cpts)) if cpts.size else 100.0
        view.setCameraPosition(distance=(span or 100.0) * 1.6)
        return view

    # ------------------------------------------------- 3-D reference builders
    def _add_line(self, view, items, a, b, color, width) -> None:
        import pyqtgraph.opengl as gl
        try:
            item = gl.GLLinePlotItem(pos=np.vstack([a, b]).astype(np.float32),
                                     color=color, width=width, antialias=True)
            view.addItem(item)
            items.append(item)
        except Exception:
            pass

    def _add_text(self, view, items, pos, text, *, label=False) -> None:
        import pyqtgraph.opengl as gl
        try:
            item = gl.GLTextItem(pos=np.asarray(pos, dtype=np.float64), text=str(text),
                                 color=_TEXT_RGBA, font=QFont("Helvetica", 11 if label else 8),
                                 glOptions="translucent")
            view.addItem(item)
            items.append(item)
        except Exception:
            pass

    def _make_grid(self, view, mins, maxs, spans) -> None:
        import pyqtgraph.opengl as gl
        try:
            grid = gl.GLGridItem()
            centre = (mins + maxs) / 2.0
            xy = max(float(np.max(spans[:2])) * 1.15, 1.0)
            sp = _nice_step(xy, target=8)
            grid.setSize(xy, xy)
            grid.setSpacing(sp, sp)
            grid.translate(float(centre[0]), float(centre[1]), float(min(mins[2], 0.0)))
            view.addItem(grid)
        except Exception:
            pass

    def _make_axis(self, view, mins, maxs, spans) -> None:
        origin = mins.copy()
        pad = max(float(np.max(spans)) * 0.04, 1.0)
        defs = (
            (0, np.array([maxs[0], origin[1], origin[2]]), _AXIS_COLORS[0], "X (nm)"),
            (1, np.array([origin[0], maxs[1], origin[2]]), _AXIS_COLORS[1], "Y (nm)"),
            (2, np.array([origin[0], origin[1], maxs[2]]), _AXIS_COLORS[2], "Z (nm)"),
        )
        for dim, end, color, label in defs:
            self._add_line(view, self._axis_items, origin, end, color, 2.0)
            lp = end.copy(); lp[dim] += pad
            self._add_text(view, self._axis_items, lp, label, label=True)
            for tick in _tick_values(float(mins[dim]), float(maxs[dim])):
                tp = origin.copy(); tp[dim] = tick
                side = 1 if dim != 1 else 0
                te = tp.copy(); te[side] += pad * 0.35
                self._add_line(view, self._axis_items, tp, te, color, 1.0)
                txp = te.copy(); txp[side] += pad * 0.12
                self._add_text(view, self._axis_items, txp, _fmt_tick(tick))

    def _make_box(self, view, mins, maxs) -> None:
        x0, y0, z0 = mins
        x1, y1, z1 = maxs
        corners = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ], dtype=np.float64)
        edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7))
        for a, b in edges:
            self._add_line(view, self._box_items, corners[a], corners[b], _REF_RGBA, 1.4)

    # ------------------------------------------------------------ visibility
    def _apply_visibility(self, *_a) -> None:
        for it in self._cloud_items:
            it.setVisible(self._cloud_chk.isChecked())
        model_on = self._model_chk.isChecked() and self._model_chk.isEnabled()
        for it in self._model_items:
            it.setVisible(model_on)
        axis_on = self._axis_chk.isChecked()
        for it in self._axis_items:
            it.setVisible(axis_on)
        box_on = self._box_chk.isChecked()
        for it in self._box_items:
            it.setVisible(box_on)
        if not self._is_3d and self._plot is not None:
            # In 2-D the plot's own axes stand in for the 3-D axis toggle.
            self._plot.showAxis("left", axis_on)
            self._plot.showAxis("bottom", axis_on)


# --------------------------------------------------------------------------- #
# Pure geometry helpers (mirror the 3-D scatter view's reference-item maths)
# --------------------------------------------------------------------------- #
def _nice_step(span: float, target: int = 6) -> float:
    raw = float(span) / max(int(target), 1)
    if not np.isfinite(raw) or raw <= 0:
        return 1.0
    mag = 10.0 ** np.floor(np.log10(raw))
    norm = raw / mag
    step = (1 if norm < 1.5 else 2 if norm < 3 else 5 if norm < 7 else 10) * mag
    return float(max(step, 1e-9))


def _tick_values(lo: float, hi: float, *, max_ticks: int = 5) -> list:
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return []
    step = _nice_step(hi - lo, target=max_ticks)
    start = np.ceil(lo / step) * step
    vals: list = []
    value = start
    while value <= hi + step * 1e-9 and len(vals) < max_ticks + 2:
        vals.append(float(value))
        value += step
    return vals[:max_ticks]


def _fmt_tick(value: float) -> str:
    if abs(value) >= 1000 or float(value).is_integer():
        return f"{value:.0f}"
    return f"{value:.3g}"


def _expanded_bounds(xyz):
    """(mins, maxs, spans) over finite XYZ, padding any degenerate axis."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[0] == 0 or xyz.shape[1] < 3:
        return None
    xyz = xyz[np.all(np.isfinite(xyz[:, :3]), axis=1), :3]
    if xyz.size == 0:
        return None
    mins, maxs = xyz.min(axis=0), xyz.max(axis=0)
    spans = maxs - mins
    max_span = max(float(np.max(spans)), 1.0)
    for i in range(3):
        if spans[i] <= 0:
            mins[i] -= max_span * 0.05
            maxs[i] += max_span * 0.05
    return mins, maxs, maxs - mins


def _xy_bounds(points):
    """(x0, y0, x1, y1) data extent (small pad on a degenerate axis), or None."""
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[0] == 0:
        return None
    p = p[np.all(np.isfinite(p[:, :2]), axis=1)]
    if p.size == 0:
        return None
    x0, y0 = p[:, 0].min(), p[:, 1].min()
    x1, y1 = p[:, 0].max(), p[:, 1].max()
    pad = max((x1 - x0), (y1 - y0), 1.0) * 0.02
    if x1 <= x0:
        x0 -= pad; x1 += pad
    if y1 <= y0:
        y0 -= pad; y1 += pad
    return float(x0), float(y0), float(x1), float(y1)


def _ref_rgba255():
    return tuple(int(round(c * 255)) for c in _REF_RGBA[:3]) + (int(_REF_RGBA[3] * 255),)


def _as_xyz(points) -> np.ndarray:
    """Coerce to an ``(N, 3)`` float array (pad Z with 0 for 2-D input)."""
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[0] == 0:
        return np.empty((0, 3))
    if p.shape[1] >= 3:
        return p[:, :3]
    return np.column_stack([p[:, 0], p[:, 1], np.zeros(p.shape[0])])
