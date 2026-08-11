"""The standalone image viewer's **single active ROI**.

ImageJ gives an image one active ROI and stores it in the file (see
:mod:`minflux_viewer.core.tiff_roi`); this is the on-screen half of that, for
:class:`~minflux_viewer.ui.tiff_viewer_window.TiffViewerWindow`.

Deliberately *not* the dataset ROI system (``ui/roi_overlay.py`` +
``ui/roi_manager.py``): that one is a multi-ROI store bound to a dataset, its
localizations, selection masks and depth planes — none of which an image window
has.  Here there is one ROI, no store and no manager, which is what the image
format can hold.  The pyqtgraph item classes are shared with the dataset overlay
so the two look and behave the same.

Coordinates: the viewer draws images in **nm** (the ``ImageItem`` is placed at
the origin and scaled by the pixel size), while ImageJ ROIs are in **pixels**.
This class owns that conversion — :meth:`set_roi` scales in, :meth:`current_roi`
scales out — so nothing above it deals in two unit systems.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt

from ..core.tiff_roi import TiffRoi, roi_from_shape
from .roi_overlay import FilledEllipseROI, FilledPolyLineROI, FilledRectROI

#: Tools the viewer can draw with (drag out a bounding box).
DRAW_TOOLS = ("rectangle", "oval")

#: Types drawn as a vertex chain rather than a box.
_POLY_TYPES = ("polygon", "freehand", "polyline", "freehand_line", "line", "angle")

_ROI_COLOR = "#ffff00"          # ImageJ's ROI yellow
_MIN_DRAG_NM = 1e-9             # below this a "drag" is really a click


class ImageRoiOverlay:
    """At most one ROI on an image ``ViewBox``, drawn in nm, stored in pixels."""

    def __init__(self, view_box, *, pixel_size=(1.0, 1.0), on_changed=None) -> None:
        self._vb = view_box
        self._sx, self._sy = (float(pixel_size[0]) or 1.0, float(pixel_size[1]) or 1.0)
        self._on_changed = on_changed
        self._item = None
        self._roi_type = ""
        self._name = ""
        self._tool: str | None = None
        self._drag_item = None
        # A ROI whose shape we cannot draw (a composite, say): kept verbatim so
        # saving does not silently discard it.
        self._preserved: TiffRoi | None = None
        # Drawing has to pre-empt the ViewBox's own drag (which pans), so the
        # handler is wrapped rather than connected to a signal. Whether the
        # ViewBox already had its *own* handler decides how detach undoes this:
        # restoring a bound method it never had would shadow the class method.
        self._had_own_drag = "mouseDragEvent" in vars(view_box)
        self._original_drag = view_box.mouseDragEvent
        view_box.mouseDragEvent = self._on_drag

    # -- lifecycle ------------------------------------------------------------
    def detach(self) -> None:
        """Restore the ViewBox and drop the item (call before the window dies)."""
        self.clear()
        try:
            if self._had_own_drag:
                self._vb.mouseDragEvent = self._original_drag
            else:
                vars(self._vb).pop("mouseDragEvent", None)
        except Exception:
            pass

    def set_pixel_size(self, sx: float, sy: float) -> None:
        """Re-scale the displayed ROI when the series (hence calibration) changes."""
        roi = self.current_roi()
        self._sx = float(sx) or 1.0
        self._sy = float(sy) or 1.0
        if roi is not None:
            self.set_roi(roi, notify=False)

    # -- tool ----------------------------------------------------------------
    @property
    def tool(self) -> str | None:
        return self._tool

    def set_tool(self, tool: str | None) -> None:
        self._tool = tool if tool in DRAW_TOOLS else None

    # -- content -------------------------------------------------------------
    def has_roi(self) -> bool:
        return self._item is not None

    def clear(self, *, notify: bool = False) -> None:
        if self._item is not None:
            try:
                self._vb.removeItem(self._item)
            except Exception:
                pass
        self._item = None
        self._roi_type = ""
        self._name = ""
        self._preserved = None
        if notify:
            self._notify()

    def set_roi(self, roi: TiffRoi | None, *, notify: bool = True) -> None:
        """Show *roi* (pixel coordinates) as the active ROI, replacing any other."""
        self.clear()
        if roi is None:
            if notify:
                self._notify()
            return
        self._roi_type = roi.roi_type
        self._name = roi.name
        if roi.roi_type in {"rectangle", "oval"}:
            x, y, w, h = roi.bounds
            self._item = self._make_box(roi.roi_type, x * self._sx, y * self._sy,
                                        w * self._sx, h * self._sy)
        elif roi.points is not None and len(roi.points) >= 2:
            pts = [[float(p[0]) * self._sx, float(p[1]) * self._sy] for p in roi.points]
            closed = roi.roi_type in {"polygon", "freehand"}
            self._item = FilledPolyLineROI(pts, closed=closed, pen=self._pen(),
                                           fill_color=_ROI_COLOR, movable=True)
            if not closed:
                self._item.setFillColor(_ROI_COLOR, alpha=0)
        else:
            self._preserved = roi
            if notify:
                self._notify()
            return
        self._vb.addItem(self._item)
        self._connect(self._item)
        if notify:
            self._notify()

    def current_roi(self) -> TiffRoi | None:
        """The active ROI in pixel coordinates, or ``None`` when there is none."""
        if self._item is None:
            return self._preserved
        if self._roi_type in {"rectangle", "oval"}:
            pos = self._item.pos()
            size = self._item.size()
            bounds = (float(pos.x()) / self._sx, float(pos.y()) / self._sy,
                      float(size.x()) / self._sx, float(size.y()) / self._sy)
            return roi_from_shape(self._roi_type, bounds=_normalize(bounds), name=self._name)
        points = [[float(p.x()) / self._sx, float(p.y()) / self._sy]
                  for p in self._handle_positions(self._item)]
        return roi_from_shape(self._roi_type, points=np.asarray(points), name=self._name)

    # -- drawing --------------------------------------------------------------
    def _on_drag(self, ev, axis=None):
        if self._tool is None or ev.button() != Qt.MouseButton.LeftButton:
            return self._original_drag(ev, axis=axis)
        ev.accept()
        start = self._vb.mapSceneToView(ev.buttonDownScenePos())
        now = self._vb.mapSceneToView(ev.scenePos())
        x0, x1 = sorted((float(start.x()), float(now.x())))
        y0, y1 = sorted((float(start.y()), float(now.y())))
        if self._drag_item is not None:
            self._vb.removeItem(self._drag_item)
            self._drag_item = None
        if (x1 - x0) <= _MIN_DRAG_NM or (y1 - y0) <= _MIN_DRAG_NM:
            return
        if ev.isFinish():
            self.clear()
            self._roi_type = self._tool
            self._name = ""
            self._item = self._make_box(self._tool, x0, y0, x1 - x0, y1 - y0)
            self._vb.addItem(self._item)
            self._connect(self._item)
            self._notify()
            return
        # Live preview: a plain, handle-less shape so the drag stays cheap.
        self._drag_item = self._make_box(self._tool, x0, y0, x1 - x0, y1 - y0,
                                         handles=False)
        self._vb.addItem(self._drag_item)

    def _make_box(self, roi_type: str, x: float, y: float, w: float, h: float,
                  *, handles: bool = True):
        cls = FilledEllipseROI if roi_type == "oval" else FilledRectROI
        item = cls([x, y], [w, h], pen=self._pen(), fill_color=_ROI_COLOR,
                   movable=handles, y_axis_inverted=True)
        if not handles:
            for handle in list(item.handles):
                item.removeHandle(handle["item"])
        return item

    @staticmethod
    def _pen():
        return pg.mkPen(_ROI_COLOR, width=2)

    @staticmethod
    def _handle_positions(item):
        """Vertex positions of a poly item in **parent** (view) coordinates."""
        return [item.mapToParent(h["item"].pos()) for h in item.handles]

    def _connect(self, item) -> None:
        for signal in ("sigRegionChangeFinished", "sigRegionChanged"):
            try:
                getattr(item, signal).connect(lambda *_a: self._notify())
                break
            except Exception:
                continue

    def _notify(self) -> None:
        if self._on_changed is not None:
            try:
                self._on_changed()
            except Exception:
                pass


def _normalize(bounds) -> tuple[float, float, float, float]:
    """``(x, y, w, h)`` with positive extents — a ROI dragged up/left has
    negative size, which ImageJ bounds cannot express."""
    x, y, w, h = (float(v) for v in bounds)
    if w < 0:
        x, w = x + w, -w
    if h < 0:
        y, h = y + h, -h
    return x, y, w, h
