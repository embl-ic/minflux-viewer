"""Interactive 3-D ROI drawing (Process › ROI › 3D ROI).

Reuses the conv-3D orthoviewer layout — a 2×2 grid of XY / YZ / XZ panes plus a
3-D OpenGL pane — but for *drawing* rather than convolution. The user draws a
2-D shape (rectangle / ellipse / polygon) in one or more ortho panes; each shape
is extruded along the axis perpendicular to its pane, and the 3-D ROI is the
**intersection** of those extrusions (``core/roi_3d.py``). A localization is
selected iff its projection lies inside every drawn shape, so:

    * an XY rectangle alone  -> an infinite-Z prism (bounds X, Y),
    * add an XZ shape        -> also bounds Z (a box / sculpted region),
    * a depth slab           -> just a rectangle drawn in XZ or YZ.

The live selection is highlighted across all panes + the 3-D view; "Crop to new
dataset" writes the selected localizations out as a standalone dataset.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.roi_3d import Roi3DConstraint, plane_axes, roi3d_mask

# (plane key, horizontal-axis label, vertical-axis label, grid row, grid col)
_PANE_LAYOUT = [
    ("XY", "X (nm)", "Y (nm)", 0, 0),
    ("YZ", "Z (nm)", "Y (nm)", 0, 1),
    ("XZ", "X (nm)", "Z (nm)", 1, 0),
]

_BASE_BRUSH = pg.mkBrush(120, 140, 200, 110)
_SEL_BRUSH = pg.mkBrush(255, 140, 20, 220)
_SEL_PEN = pg.mkPen(255, 255, 255, 200, width=0.6)
_ROI_PEN = pg.mkPen(60, 220, 120, width=2)

_DISPLAY_CAP = 150_000     # decimate the *displayed* scatter above this many locs


class Roi3DWindow(QDialog):
    """Modeless orthoviewer for drawing an intersection-of-extrusions 3-D ROI."""

    def __init__(self, state, dataset_idx: int, owner=None):
        super().__init__(None)  # non-owned top-level per the window convention
        self._state = state
        self._idx = int(dataset_idx)
        self._owner = owner

        ds = self._dataset()
        name = getattr(getattr(ds, "file", None), "name", "dataset") if ds else "dataset"
        self.setWindowTitle(f"3D ROI — {name}")
        self.resize(1120, 860)

        self._panes: dict[str, dict] = {}
        self._constraints: dict[str, dict] = {}   # plane -> {"item", "type"}
        self._sel: np.ndarray | None = None

        self._load_points(ds)
        self._build_ui()
        self._populate_panes()
        self._build_3d_scatters()
        self._recompute()

    # ------------------------------------------------------------------ data
    def _dataset(self):
        if 0 <= self._idx < len(self._state.datasets):
            return self._state.datasets[self._idx]
        return None

    def _load_points(self, ds) -> None:
        """Displayed points = filtered, finite localizations in **display nm**.

        ``_idx_full`` maps each displayed row back to its full-dataset row so the
        crop can build a full-length keep mask.
        """
        from ..core.roi_crop import display_coords

        loc = display_coords(ds) if ds is not None else np.empty((0, 3))
        if loc.ndim != 2 or loc.shape[1] < 3:
            loc = np.empty((0, 3), dtype=float)
        n = loc.shape[0]
        keep = np.ones(n, dtype=bool)
        fmask = getattr(ds, "filter_mask", None) if ds is not None else None
        if fmask is not None:
            fmask = np.asarray(fmask, dtype=bool).ravel()
            if fmask.size == n:
                keep &= fmask
        if n:
            keep &= np.all(np.isfinite(loc), axis=1)
        self._idx_full = np.nonzero(keep)[0]
        self._pts = loc[keep]

        m = self._pts.shape[0]
        if m > _DISPLAY_CAP:
            rng = np.random.default_rng(0)
            self._disp_idx = np.sort(rng.choice(m, _DISPLAY_CAP, replace=False))
        else:
            self._disp_idx = np.arange(m)

        # precompute per-pane projected coordinates for the displayed subset
        disp = self._pts[self._disp_idx] if m else np.empty((0, 3))
        self._proj: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for plane in ("XY", "YZ", "XZ"):
            h, v = plane_axes(plane)
            if disp.shape[0]:
                self._proj[plane] = (disp[:, h], disp[:, v])
            else:
                self._proj[plane] = (np.empty(0), np.empty(0))

    # -------------------------------------------------------------------- ui
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Shape:"))
        self._shape_combo = QComboBox()
        self._shape_combo.addItems(["Rectangle", "Ellipse", "Polygon"])
        self._shape_combo.setToolTip(
            "Shape to draw in the next ortho pane you add to. Drag its handles to "
            "resize; right-click it to remove.")
        bar.addWidget(self._shape_combo)

        bar.addSpacing(8)
        bar.addWidget(QLabel("Add to:"))
        for plane in ("XY", "YZ", "XZ"):
            btn = QPushButton(plane)
            btn.setFixedWidth(46)
            btn.setToolTip(f"Draw the selected shape in the {plane} view "
                           f"(replaces the existing {plane} shape).")
            btn.clicked.connect(lambda _c=False, p=plane: self._add_shape(p))
            bar.addWidget(btn)

        bar.addSpacing(12)
        clear = QPushButton("Clear all")
        clear.setToolTip("Remove every drawn shape.")
        clear.clicked.connect(self._clear_all)
        bar.addWidget(clear)

        bar.addStretch(1)
        self._count_label = QLabel("Selected: 0 / 0")
        self._count_label.setStyleSheet("font-weight:bold;")
        bar.addWidget(self._count_label)

        crop = QPushButton("Crop to new dataset")
        crop.setToolTip("Create a new dataset from the localizations inside the 3-D ROI.")
        crop.clicked.connect(self._crop_to_dataset)
        bar.addWidget(crop)
        root.addLayout(bar)

        hint = QLabel(
            "Draw a shape in one pane to bound two axes (extruded along the third); "
            "add a shape in another pane to bound the remaining axis. The 3-D ROI is "
            "the intersection of the extrusions.")
        hint.setStyleSheet("color:#888;")
        hint.setWordWrap(True)
        root.addWidget(hint)

        grid = QGridLayout()
        grid.setSpacing(4)
        for key, hlabel, vlabel, row, col in _PANE_LAYOUT:
            self._make_pane(grid, key, hlabel, vlabel, row, col)
        # share axes so the panes line up (Y across XY/YZ, X across XY/XZ)
        self._panes["YZ"]["plot"].setYLink(self._panes["XY"]["plot"])
        self._panes["XZ"]["plot"].setXLink(self._panes["XY"]["plot"])
        self._build_3d_pane(grid, 1, 1)
        root.addLayout(grid, 1)

    def _make_pane(self, grid, key, hlabel, vlabel, row, col) -> None:
        plot = pg.PlotWidget()
        plot.setLabel("bottom", hlabel)
        plot.setLabel("left", vlabel)
        plot.setTitle(key)
        base = pg.ScatterPlotItem(size=3, pxMode=True, pen=None, brush=_BASE_BRUSH)
        base.setZValue(0)
        plot.addItem(base)
        sel = pg.ScatterPlotItem(size=5, pxMode=True, pen=_SEL_PEN, brush=_SEL_BRUSH)
        sel.setZValue(10)
        plot.addItem(sel)
        grid.addWidget(plot, row, col)
        self._panes[key] = {"plot": plot, "base": base, "sel": sel}

    def _build_3d_pane(self, grid, row, col) -> None:
        self._gl_view = None
        self._gl_base = None
        self._gl_sel = None
        self._gl_center = np.zeros(3)
        try:
            import pyqtgraph.opengl as gl

            self._gl = gl
            self._gl_view = gl.GLViewWidget()
            self._gl_view.setBackgroundColor(pg.mkColor(10, 10, 14))
            grid.addWidget(self._gl_view, row, col)
        except Exception as exc:  # OpenGL unavailable → graceful placeholder
            ph = QLabel(f"3-D view unavailable\n({exc})")
            ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ph.setStyleSheet("color:#888; border:1px solid #333;")
            ph.setWordWrap(True)
            grid.addWidget(ph, row, col)

    def _populate_panes(self) -> None:
        for plane, pane in self._panes.items():
            hp, vp = self._proj[plane]
            pane["base"].setData(hp, vp)
            pane["plot"].getViewBox().autoRange()

    def _build_3d_scatters(self) -> None:
        if self._gl_view is None:
            return
        gl = self._gl
        disp = self._pts[self._disp_idx] if self._pts.shape[0] else np.empty((0, 3))
        if disp.shape[0]:
            self._gl_center = 0.5 * (disp.min(axis=0) + disp.max(axis=0))
            centered = (disp - self._gl_center).astype(np.float32)
            span = float(np.max(disp.max(axis=0) - disp.min(axis=0))) or 1.0
        else:
            centered = np.empty((0, 3), dtype=np.float32)
            span = 1.0
        grid = gl.GLGridItem()
        grid.setSize(span, span)
        grid.setSpacing(span / 10.0, span / 10.0)
        self._gl_view.addItem(grid)
        self._gl_base = gl.GLScatterPlotItem(
            pos=centered, size=2.0, color=(0.55, 0.62, 0.8, 0.5), pxMode=True)
        self._gl_view.addItem(self._gl_base)
        self._gl_sel = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32), size=5.0,
            color=(1.0, 0.55, 0.1, 1.0), pxMode=True)
        self._gl_view.addItem(self._gl_sel)
        self._gl_view.setCameraPosition(distance=span * 1.6)

    # ------------------------------------------------------------- drawing
    def _add_shape(self, plane: str) -> None:
        shape = self._shape_combo.currentText().lower()
        self._remove_constraint(plane)
        pane = self._panes[plane]
        vb = pane["plot"].getViewBox()
        (x0, x1), (y0, y1) = vb.viewRange()
        w = 0.35 * (x1 - x0) or 1.0
        h = 0.35 * (y1 - y0) or 1.0
        px, py = 0.5 * (x0 + x1) - w / 2.0, 0.5 * (y0 + y1) - h / 2.0
        if shape == "rectangle":
            item = pg.RectROI([px, py], [w, h], pen=_ROI_PEN, removable=True)
        elif shape == "ellipse":
            item = pg.EllipseROI([px, py], [w, h], pen=_ROI_PEN, removable=True)
        else:  # polygon — seed a quad the user can reshape / add vertices to
            corners = [[px, py], [px + w, py], [px + w, py + h], [px, py + h]]
            item = pg.PolyLineROI(corners, closed=True, pen=_ROI_PEN, removable=True)
        item.setZValue(50)
        pane["plot"].addItem(item)
        self._constraints[plane] = {"item": item, "type": shape}
        item.sigRegionChangeFinished.connect(lambda *_a: self._recompute())
        item.sigRemoveRequested.connect(
            lambda *_a, p=plane: self._remove_constraint(p, recompute=True))
        self._recompute()

    def _remove_constraint(self, plane: str, *, recompute: bool = False) -> None:
        c = self._constraints.pop(plane, None)
        if c is not None:
            try:
                self._panes[plane]["plot"].removeItem(c["item"])
            except Exception:
                pass
        if recompute:
            self._recompute()

    def _clear_all(self) -> None:
        for plane in list(self._constraints):
            self._remove_constraint(plane)
        self._recompute()

    def _item_to_constraint(self, plane: str, c: dict) -> Roi3DConstraint | None:
        item, kind = c["item"], c["type"]
        try:
            if kind in ("rectangle", "ellipse"):
                pos, size = item.pos(), item.size()
                geom = {"bounds": [float(pos.x()), float(pos.y()),
                                   float(size[0]), float(size[1])]}
                roi_type = "rectangle" if kind == "rectangle" else "oval"
                return Roi3DConstraint(plane, roi_type, geom)
            # polygon — map local handle vertices to data coords
            pts = [item.mapToParent(p) for p in item.getState()["points"]]
            poly = [[float(p.x()), float(p.y())] for p in pts]
            return Roi3DConstraint(plane, "polygon", {"points": poly})
        except Exception:
            return None

    # --------------------------------------------------------- recompute
    def _recompute(self) -> None:
        constraints = [
            con for plane, c in self._constraints.items()
            if (con := self._item_to_constraint(plane, c)) is not None
        ]
        m = self._pts.shape[0]
        sel = roi3d_mask(self._pts, constraints) if constraints else np.zeros(m, bool)
        self._sel = sel

        disp = self._disp_idx
        sel_disp = sel[disp] if sel.size else np.zeros(disp.shape[0], bool)
        for plane, pane in self._panes.items():
            hp, vp = self._proj[plane]
            pane["sel"].setData(hp[sel_disp], vp[sel_disp])
        if self._gl_sel is not None:
            disp_pts = self._pts[disp]
            centered = (disp_pts[sel_disp] - self._gl_center).astype(np.float32)
            self._gl_sel.setData(pos=centered)

        self._count_label.setText(f"Selected: {int(sel.sum()):,} / {m:,}")

    # -------------------------------------------------------------- output
    def _crop_to_dataset(self) -> None:
        from ..core.roi_crop import subset_dataset

        ds = self._dataset()
        if ds is None:
            QMessageBox.information(self, "3D ROI", "The source dataset is no longer open.")
            return
        sel = self._sel
        if sel is None or not sel.any():
            QMessageBox.information(
                self, "3D ROI",
                "Draw a 3-D ROI first — no localizations are currently selected.")
            return
        n = int(getattr(getattr(ds, "prop", None), "num_loc", 0) or self._pts.shape[0])
        keep = np.zeros(n, dtype=bool)
        rows = self._idx_full[sel]
        rows = rows[(rows >= 0) & (rows < n)]
        keep[rows] = True
        base = getattr(getattr(ds, "file", None), "name", "dataset")
        name = f"{base} [3D ROI]"
        try:
            new = subset_dataset(ds, keep, name=name, prefs=getattr(self._state, "prefs", None))
            self._state.add_dataset(new)
        except Exception as exc:
            QMessageBox.warning(self, "3D ROI", f"Crop failed:\n{exc}")
            return
        self._log(f"3D ROI crop: {int(keep.sum()):,} localizations → '{name}'")

    def _log(self, msg: str) -> None:
        try:
            self._state.log(msg)
        except Exception:
            pass
