"""pyqtgraph ROI drawing and overlay adapter."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pyqtgraph as pg
from pyqtgraph.graphicsItems.ROI import Handle
from PyQt6.QtCore import QEvent, QObject, QRectF, Qt, QTimer
from PyQt6.QtGui import QPainter, QPainterPath
from PyQt6.QtWidgets import QApplication

from ..core.roi import RoiRecord, RoiStore
from ..core.roi_selection import ROI_MASKS_STATE_KEY, store_roi_mask

# Polyline-family shapes whose vertices can be added/deleted via the right-click
# menu (rectangle/oval/point/line/angle have fixed or handle-defined geometry).
_VERTEX_EDIT_TYPES = {"polygon", "freehand", "polyline", "freehand_line", "magnetic_lasso"}

# For a 2-D view plane, which two XYZ columns are plotted (i, j) and which is the
# out-of-plane "depth" column (k). Used to give a drawn ROI a full 3-D position:
# the in-plane coords come from the click, the depth coord from the centre of the
# current viewing range of that out-of-plane dimension.
_PLANE_PLOT_AXES = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}
_PLANE_DEPTH_AXIS = {"XY": 2, "XZ": 1, "YZ": 0}
_PLANE_DEPTH_NAME = {"XY": "Z", "XZ": "Y", "YZ": "X"}


def point_to_3d(pos, plane: str | None, depth: float | None) -> list[float]:
    """Full XYZ for an in-plane click at *pos*; the out-of-plane axis = *depth*."""
    if plane not in _PLANE_PLOT_AXES:
        return [float(pos[0]), float(pos[1])]
    i, j = _PLANE_PLOT_AXES[plane]
    k = _PLANE_DEPTH_AXIS[plane]
    out = [0.0, 0.0, 0.0]
    out[i] = float(pos[0])
    out[j] = float(pos[1])
    out[k] = float(depth) if depth is not None else 0.0
    return out


def project_point(point, plane: str | None) -> tuple[float, float]:
    """Project a (possibly 3-D) point into *plane*'s two on-screen axes."""
    if plane in _PLANE_PLOT_AXES and len(point) >= 3:
        i, j = _PLANE_PLOT_AXES[plane]
        return float(point[i]), float(point[j])
    return float(point[0]), float(point[1])


def update_point_in_plane(new_xy, old_point, plane: str | None) -> list[float]:
    """Apply an in-plane drag to *old_point*, preserving its out-of-plane axis."""
    if plane not in _PLANE_PLOT_AXES or len(old_point) < 3:
        return [float(new_xy[0]), float(new_xy[1])]
    i, j = _PLANE_PLOT_AXES[plane]
    out = [float(v) for v in old_point]
    out[i] = float(new_xy[0])
    out[j] = float(new_xy[1])
    return out


# --- vertex-list variants (line / polyline / freehand line) -----------------
# A line/curve is just "a list of vertices projected per view": each vertex is
# the n=1 point case. Out-of-plane (depth) axis = centre of the current viewing
# range, set once when drawn; preserved per-vertex when edited in another plane.

def points_to_3d(xy_list, plane: str | None, depth: float | None) -> list[list[float]]:
    """Build full-XYZ vertices from a 2-D draw (all at the same *depth*)."""
    return [point_to_3d(p, plane, depth) for p in xy_list]


def project_points(points, plane: str | None) -> list[list[float]]:
    """Project a list of (possibly 3-D) vertices into *plane*'s two on-screen axes."""
    return [list(project_point(p, plane)) for p in points]


def update_points_in_plane(new_xy_list, old_points, plane: str | None) -> list[list[float]]:
    """Apply in-plane handle moves to each vertex, preserving its out-of-plane axis.

    Pairs ``new_xy_list[k]`` with ``old_points[k]``; if the counts differ (a
    vertex was added/removed) the unmatched new vertices are taken as 2-D.
    """
    out: list[list[float]] = []
    for k, xy in enumerate(new_xy_list):
        old = old_points[k] if k < len(old_points) else None
        if old is not None and len(old) >= 3:
            out.append(update_point_in_plane(xy, old, plane))
        else:
            out.append([float(xy[0]), float(xy[1])])
    return out


class FilledRotateHandle(Handle):
    """Rotate handle with a filled circular hit area."""

    def buildPath(self) -> None:
        radius = float(self.radius)
        self.path = QPainterPath()
        self.path.addEllipse(QRectF(-radius, -radius, radius * 2.0, radius * 2.0))

    def paint(self, p, opt, widget) -> None:
        color = self.currentPen.color()
        fill = pg.mkColor(color)
        fill.setAlpha(90)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, self._antialias)
        p.setPen(self.currentPen)
        p.setBrush(pg.mkBrush(fill))
        p.drawPath(self.shape())


class SquareHandle(Handle):
    """Scale handle drawn as an axis-aligned **square** (ImageJ-style) instead of
    pyqtgraph's default diamond. ``buildPath`` is regenerated on every device
    transform, so overriding it (rather than poking ``.path``) keeps it square."""

    def buildPath(self) -> None:
        r = float(self.radius)
        self.path = QPainterPath()
        self.path.addRect(QRectF(-r, -r, r * 2.0, r * 2.0))


class SolidSquareHandle(SquareHandle):
    """Square handle with a **filled** face in the ROI colour — marks the drawing
    origin / rotation-anchor corner of a rotated rectangle / ellipse."""

    def paint(self, p, opt, widget) -> None:
        fill = pg.mkColor(self.currentPen.color())
        fill.setAlpha(200)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, self._antialias)
        p.setPen(self.currentPen)
        p.setBrush(pg.mkBrush(fill))
        p.drawPath(self.shape())


def _mark_line_end_handle(item) -> None:
    """Restyle an open line ROI's **last** vertex handle as a round (**oval**) marker
    so the line's direction (begin → end, which the Plot Profile follows) is visible;
    the start / intermediate vertices keep their square edit handles. The circle is
    kept the **same size** as the square start handle (only the shape differs), not
    enlarged.

    Done **in place** by swapping the handle's local ``path`` to an ellipse — the
    handle's underlying sip/C++ type is left untouched and it stays an ordinary
    draggable free handle (performs no rotation). A previous ``handle.__class__ = …``
    swap **crashed** the app under the rapid create/destroy churn of a live draft
    drag, so never reassign ``__class__`` on a pyqtgraph handle. No-op for a closed
    loop, a single vertex, or an already-marked handle."""
    handles = getattr(item, "handles", None)
    if not handles or len(handles) < 2 or getattr(item, "closed", False):
        return
    entry = handles[-1]
    h = entry.get("item") if isinstance(entry, dict) else None
    if not isinstance(h, Handle) or getattr(h, "_is_line_end_marker", False):
        return
    try:
        r = float(h.radius)                   # same size as the square start handle
        path = QPainterPath()
        path.addEllipse(QRectF(-r, -r, r * 2.0, r * 2.0))
        h.path = path
        h._shape = None                       # force shape() to regenerate from the ellipse
        h._is_line_end_marker = True
        h.update()
    except Exception:
        pass


#: Scale factor on pyqtgraph's default ROI handle size for the rectangle / oval
#: bounding-box handles — smaller (≈ half) square + circle corners.
_BOX_HANDLE_SIZE_SCALE = 0.5
#: Bounding-box fraction of the ellipse's 45° point (≈0.854): the oval's diagonal
#: handles sit **on the curve** here rather than at the bbox corner (1.0).
_CURVE_FRAC = 0.5 + 0.5 / np.sqrt(2.0)


def box_scale_fracs(*, on_curve: bool) -> tuple[tuple[float, float], ...]:
    """The 8 scale-handle bbox fractions (4 edge midpoints + 4 diagonals). With
    ``on_curve`` the diagonals sit on the inscribed ellipse (≈0.854) — the oval's
    N/S/E/W/NE/NW/SE/SW handles; otherwise at the bbox corners — the rectangle's."""
    d = _CURVE_FRAC if on_curve else 1.0
    lo = 1.0 - d
    return (
        (0.0, 0.5), (1.0, 0.5), (0.5, 0.0), (0.5, 1.0),   # W, E, S, N (edge mids)
        (lo, lo), (d, lo), (d, d), (lo, d),               # SW, SE, NE, NW (corners / curve)
    )


#: minor:major axis ratio for the rotated-rectangle / ellipse default perpendicular
#: dimension — the golden ratio (√5−1)/2, so a long-axis drag gives a 1 : 0.618 box.
GOLDEN_MINOR = 0.6180339887498949


def normalize_angle_deg(angle: float) -> float:
    """Wrap an angle in degrees to ``(-180, 180]``. Rotation is periodic mod 360, so
    this only affects display/storage — the geometry is unchanged. Guards the ROI
    read-out against a pyqtgraph rotate handle that accumulates past ±360 when the
    ROI is spun several turns (e.g. an ellipse reporting -7140°)."""
    a = (float(angle) + 180.0) % 360.0 - 180.0
    return 180.0 if a <= -180.0 + 1e-9 else a


def center_axis_box(x0: float, y0: float, x1: float, y1: float,
                    *, minor_ratio: float = GOLDEN_MINOR):
    """Rotated box from a long-axis drag P0→P1: the drag **is** the long *centre*
    axis (its length + direction set the length + rotation); the perpendicular side
    = ``minor_ratio × length``, centred on the axis. Returns
    ``(bounds=[x, y, w, h], angle_deg)`` where *bounds* is the **unrotated**,
    centre-consistent axis-aligned box (its centre = the drag midpoint = the
    rotation centre) and *w* is the long (drag) dimension."""
    dx, dy = float(x1 - x0), float(y1 - y0)
    length = float(np.hypot(dx, dy)) or 1e-9
    minor = length * float(minor_ratio)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    angle = float(np.degrees(np.arctan2(dy, dx)))
    return [cx - length / 2.0, cy - minor / 2.0, length, minor], angle


#: Handle fractions for the rotated-variant layouts (used by both `_make_item`'s
#: handle builders and the arrow-key role detection so they never drift). Each is
#: ``(rotate_frac, (scale_fracs…))`` — the solid pivot rotates about the centre, the
#: rest resize. **Both** variants (Fiji-style) put their 4 handles at the **edge
#: midpoints / axis endpoints**: the solid pivot at ``(0, 0.5)`` = the drawing-origin
#: edge middle (the click point, `center_axis_box`'s P0), the opposite edge middle
#: ``(1, 0.5)`` + the two perpendicular edge middles ``(0.5, 0)``/``(0.5, 1)``.
_ROTATED_RECT_HANDLES = ((0.0, 0.5), ((1.0, 0.5), (0.5, 0.0), (0.5, 1.0)))
_ROTATED_OVAL_HANDLES = ((0.0, 0.5), ((1.0, 0.5), (0.5, 0.0), (0.5, 1.0)))


def rotated_variant_handles(roi_type: str):
    """``(rotate_frac, scale_fracs)`` for a rotated *rectangle* / *oval* variant."""
    return _ROTATED_OVAL_HANDLES if roi_type == "oval" else _ROTATED_RECT_HANDLES


def _add_rotated_variant_handles(roi, roi_type: str) -> None:
    """Build the 4-handle rotated-variant layout: one **solid** pivot handle that
    rotates the ROI about its centre when dragged, plus 3 square scale handles. Both
    the rotated rectangle and the ellipse put all four at the edge midpoints / axis
    endpoints — the pivot at the drawing-origin edge middle (the click point), the
    other three at the remaining edge middles (Fiji-style)."""
    roi.handleSize = float(roi.handleSize) * _BOX_HANDLE_SIZE_SCALE
    center = [0.5, 0.5]
    rotate_frac, scale_fracs = rotated_variant_handles(roi_type)
    pivot = SolidSquareHandle(
        float(roi.handleSize), typ="r", pen=roi.handlePen,
        hoverPen=roi.handleHoverPen, parent=roi, antialias=roi._antialias)
    roi.addRotateHandle(list(rotate_frac), center, item=pivot, name="rotate")
    for fx, fy in scale_fracs:
        handle = SquareHandle(
            float(roi.handleSize), typ="s", pen=roi.handlePen,
            hoverPen=roi.handleHoverPen, parent=roi, antialias=roi._antialias)
        roi.addScaleHandle([fx, fy], [1.0 - fx, 1.0 - fy], item=handle)


def _add_box_handles(roi, y_axis_inverted: bool, *, on_curve: bool = False) -> None:
    """Add the shared bounding-box edit handles — 8 **square** scale handles (4 edge
    midpoints + 4 corners) — used by **both** the rectangle and the oval. With
    ``on_curve`` (oval) the 4 diagonal handles sit on the ellipse curve (ImageJ-style)
    instead of the bbox corners. Handles are ≈ half the pyqtgraph default size.

    There is **no** rotate handle: an axis-aligned rectangle / oval is not rotatable
    — rotation is provided by the **Rotated Rectangle / Ellipse** tool variants (their
    solid pivot handle). ``y_axis_inverted`` is retained for signature compatibility
    (it only ever positioned the removed rotate handle)."""
    del y_axis_inverted                    # no longer used (rotate handle removed)
    # Shrink every scale handle this ROI creates (addHandle builds each at handleSize).
    roi.handleSize = float(roi.handleSize) * _BOX_HANDLE_SIZE_SCALE
    for fx, fy in box_scale_fracs(on_curve=on_curve):
        handle = SquareHandle(
            float(roi.handleSize), typ="s", pen=roi.handlePen,
            hoverPen=roi.handleHoverPen, parent=roi, antialias=roi._antialias)
        roi.addScaleHandle([fx, fy], [1.0 - fx, 1.0 - fy], item=handle)


class FilledRectROI(pg.ROI):
    """Rectangle ROI with a visible translucent face and edit handles."""

    def __init__(
        self,
        pos,
        size,
        *,
        angle: float = 0.0,
        fill_color="#ffff00",
        y_axis_inverted: bool = False,
        variant: str = "",
        **kwargs,
    ) -> None:
        super().__init__(pos, size, angle=angle, **kwargs)
        self._fill_color = pg.mkColor(fill_color)
        self._fill_color.setAlpha(32)
        if variant == "rotated":
            _add_rotated_variant_handles(self, "rectangle")
        else:
            _add_box_handles(self, y_axis_inverted)

    def setFillColor(self, color, alpha: int = 32) -> None:
        self._fill_color = pg.mkColor(color)
        self._fill_color.setAlpha(alpha)
        self.update()

    def paint(self, p, opt, widget) -> None:
        rect = QRectF(0, 0, self.state["size"][0], self.state["size"][1]).normalized()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, self._antialias)
        p.setPen(self.currentPen)
        p.setBrush(pg.mkBrush(self._fill_color))
        p.drawRect(rect)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addRect(QRectF(0, 0, self.state["size"][0], self.state["size"][1]).normalized())
        return path


class FilledEllipseROI(pg.EllipseROI):
    """Ellipse ROI with a visible translucent face.

    Uses the **rectangle's bounding-box handles** (8 scale + a top rotate handle)
    rather than ``EllipseROI``'s default single scale + rotate handle, so an oval
    is resized and rotated via its bounding box like a rectangle."""

    def __init__(self, pos, size, *, fill_color="#ffff00", y_axis_inverted: bool = False,
                 variant: str = "", **kwargs) -> None:
        # Set before super().__init__ — EllipseROI.__init__ calls _addHandles().
        self._y_axis_inverted = bool(y_axis_inverted)
        self._variant = str(variant or "")
        super().__init__(pos, size, **kwargs)
        self._fill_color = pg.mkColor(fill_color)
        self._fill_color.setAlpha(32)

    def _addHandles(self) -> None:
        if getattr(self, "_variant", "") == "rotated":
            _add_rotated_variant_handles(self, "oval")     # 4 axis handles, solid pivot
        else:
            _add_box_handles(self, getattr(self, "_y_axis_inverted", False), on_curve=True)

    def setFillColor(self, color, alpha: int = 32) -> None:
        self._fill_color = pg.mkColor(color)
        self._fill_color.setAlpha(alpha)
        self.update()

    def paint(self, p, opt, widget) -> None:
        rect = self.boundingRect()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, getattr(self, "_antialias", True))
        p.setPen(self.currentPen)
        p.setBrush(pg.mkBrush(self._fill_color))
        p.drawEllipse(rect)


class FilledPolyLineROI(pg.PolyLineROI):
    """Closed polyline (polygon / freehand) ROI with a translucent face.

    The face uses the **even-odd** fill rule, so a self-intersecting outline
    (e.g. a figure-8) fills both enclosed lobes."""

    def __init__(self, *args, fill_color="#ffff00", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._fill_color = pg.mkColor(fill_color)
        self._fill_color.setAlpha(32)

    def setFillColor(self, color, alpha: int = 32) -> None:
        self._fill_color = pg.mkColor(color)
        self._fill_color.setAlpha(alpha)
        self.update()

    def paint(self, p, opt, widget) -> None:
        # The outline is drawn by the segment child items; here we only add the
        # translucent face (no super().paint(), which would draw a bounding box).
        try:
            if getattr(self, "closed", False):
                positions = [h["item"].pos() for h in self.handles]
                if len(positions) >= 3:
                    path = QPainterPath()
                    path.moveTo(positions[0])
                    for pt in positions[1:]:
                        path.lineTo(pt)
                    path.closeSubpath()
                    path.setFillRule(Qt.FillRule.OddEvenFill)
                    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                    p.setPen(pg.mkPen(None))
                    p.setBrush(pg.mkBrush(self._fill_color))
                    p.drawPath(path)
        except Exception:
            pass


class PointMarkerItem(pg.TargetItem):
    """Fixed-size, draggable point marker (ilastik density-count style).

    Drawn as soft concentric circles — a red centre fading outward through a
    jet-like palette — all translucent so the data underneath stays visible. The
    on-screen size is constant regardless of zoom (it reuses ``TargetItem``'s
    pixel-based shape)."""

    #: (radius fraction, RGBA) outer → inner; low alpha keeps it see-through.
    _RINGS = (
        (1.00, (0, 70, 210, 30)),     # outer blue, very faint
        (0.78, (0, 195, 205, 45)),    # cyan
        (0.56, (0, 200, 70, 65)),     # green
        (0.36, (240, 220, 0, 95)),    # yellow
        (0.18, (235, 45, 30, 150)),   # red centre
    )

    def __init__(self, pos, *, size: int = 20, **kwargs) -> None:
        super().__init__(
            pos=pos, size=size, symbol="o", movable=True,
            pen=pg.mkPen(None), brush=pg.mkBrush(0, 0, 0, 0),
            hoverPen=pg.mkPen(255, 255, 255, 130),
            **kwargs,
        )

    def setFillColor(self, *args, **kwargs) -> None:  # overlay generic styling — keep our look
        pass

    def paint(self, p, *_args) -> None:
        rect = self.shape().boundingRect()
        center = rect.center()
        radius = rect.width() / 2.0
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(pg.mkPen(None))
        for frac, (r, g, b, a) in self._RINGS:
            p.setBrush(pg.mkBrush(r, g, b, a))
            p.drawEllipse(center, radius * frac, radius * frac)


def _angle_abc(a, b, c) -> float:
    """Angle ABC in degrees (0–180): the angle at vertex *b* between BA and BC."""
    ba = np.asarray(a, dtype=float)[:2] - np.asarray(b, dtype=float)[:2]
    bc = np.asarray(c, dtype=float)[:2] - np.asarray(b, dtype=float)[:2]
    nba = float(np.hypot(ba[0], ba[1]))
    nbc = float(np.hypot(bc[0], bc[1]))
    if nba < 1e-12 or nbc < 1e-12:
        return 0.0
    cos_ang = float(np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_ang)))


class RoiOverlayController(QObject):
    """Attach shared ROI state to one pyqtgraph plot/image view."""

    def __init__(
        self,
        store: RoiStore,
        owner,
        view_widget,
        plot_item,
        *,
        coordinate_space: str = "plot",
    ) -> None:
        super().__init__(owner)
        self.store = store
        self.owner = owner
        self.view_widget = view_widget
        self.plot_item = plot_item
        self.view_box = plot_item.vb
        self.coordinate_space = coordinate_space
        self.items: dict[str, Any] = {}
        self.labels: dict[str, pg.TextItem] = {}
        self.draft: RoiRecord | None = None
        self.draft_item = None
        # Pending point markers drawn this session but not yet added to the
        # Manager (the point tool stays pressed for multi-point drawing). They
        # live here, NOT in the store, until the user explicitly adds them.
        self._session_points: list[RoiRecord] = []
        self._session_items: dict[str, Any] = {}
        self._drag_start: tuple[float, float] | None = None
        self._freehand_points: list[list[float]] = []
        self._polygon_points: list[list[float]] = []
        self._polyline_points: list[list[float]] = []
        self._lasso_points: list[list[float]] = []          # committed snapped vertices
        self._angle_points: list[list[float]] = []
        self._pending_selection_record: RoiRecord | None = None
        # One-shot callback invoked with the next finished rectangle draft (used
        # by the Scale Bar "At Selection" draw-then-place flow).
        self._rect_request = None
        # Which sub-element the arrow keys act on for the active ROI: "move"
        # (translate whole ROI), "rotate", ("scale", fx, fy) or ("vertex", i). Set
        # by clicking that handle in edit mode; default whole-ROI move.
        self._kbd_handle = "move"
        # Identity of the ROI the current _kbd_handle refers to (reset to "move"
        # when the active target changes) + the active session point for arrows.
        self._kbd_target_id: str | None = None
        self._active_session_point_id: str | None = None
        self._selection_timer = QTimer(self)
        self._selection_timer.setSingleShot(True)
        self._selection_timer.setInterval(120)
        self._selection_timer.timeout.connect(self._compute_pending_selection)

        target = self._event_target()
        target.installEventFilter(self)
        # Key presses (arrow nudge / rotate, 't' → add) are delivered to whichever
        # widget holds keyboard focus, which is often NOT the mouse-event viewport.
        # Filter the view widget too, and allow the owner to register extra key
        # sources (e.g. the render window's pg.ImageView, which otherwise eats the
        # arrow keys for its unused timeline before they reach the ROI).
        self._extra_key_sources: set = set()
        if self.view_widget is not target:
            self.view_widget.installEventFilter(self)
        store.changed.connect(self.refresh)
        store.selection_changed.connect(self.refresh)
        self.refresh()

    def activate(self) -> None:
        self.store.set_active_adapter(self)

    def add_key_event_source(self, widget) -> None:
        """Also catch keyboard events (arrow nudge, ``t``) when *widget* holds
        focus. Used by the render window to register its ``pg.ImageView`` — whose
        own ``keyPressEvent`` otherwise consumes the arrow keys (for an unused
        timeline) before they reach the ROI controller."""
        if widget is None or widget in self._extra_key_sources:
            return
        self._extra_key_sources.add(widget)
        widget.installEventFilter(self)

    def current_record(self) -> RoiRecord | None:
        return self.draft

    def active_open_line_record(self) -> RoiRecord | None:
        """The open-line ROI (``line`` / ``polyline`` / ``freehand_line``) the Plot
        Profile acts on: the current draft if it is one, else the single selected
        stored one; else ``None``."""
        line_types = {"line", "polyline", "freehand_line"}
        if self.draft is not None and self.draft.type in line_types:
            return self.draft
        selected = list(self.store.selected_ids)
        if len(selected) == 1:
            for rec in self.store.records:
                if rec.id == selected[0] and rec.type in line_types:
                    return rec
        return None

    def active_open_line_points(self):
        """Current projected 2-D vertices ``(N, 2)`` of the active open-line ROI,
        read from its **live** pyqtgraph item so they track a drag *in progress*
        (the record geometry only commits on release, via
        ``sigRegionChangeFinished``). Falls back to the record geometry projected
        into the view plane. ``None`` when no open-line ROI is active."""
        record = self.active_open_line_record()
        if record is None:
            return None
        item = self.draft_item if (self.draft is not None and self.draft is record) \
            else self.items.get(record.id)
        if item is not None:
            try:
                pts = item_to_geometry(record.type, item).get("points", [])
                if len(pts) >= 2:
                    return [[float(p[0]), float(p[1])] for p in pts]
            except Exception:
                pass
        return project_points(record.geometry.get("points", []), self._view_plane())

    def consume_draft(self) -> RoiRecord | None:
        record = self.draft
        # Stamp the out-of-plane (depth) value / view plane when the draft is
        # persisted (line / angle / shapes reach the store via the ROI Manager,
        # which bypasses the live finalize path).
        if record is not None:
            record = self._normalize_record(record)
        self._clear_draft()
        return record

    def replace_selected_from_draft(self) -> RoiRecord | None:
        return self.consume_draft()

    def record_for_update(self, selected_record: RoiRecord) -> RoiRecord | None:
        """Return draft or live edited overlay geometry for Manager Update."""
        if self.draft is not None:
            return self.consume_draft()
        item = self.items.get(selected_record.id)
        if item is None:
            return None
        record = copy.deepcopy(selected_record)
        if record.type == "point":
            record.geometry = self._point_geometry_from_item(item, record.geometry)
        elif record.type in {"line", "polyline", "freehand_line"}:
            record.geometry = self._open_line_geometry_from_item(item, record.geometry, record.type)
        else:
            record.geometry = item_to_geometry(record.type, item, prev=record.geometry)
        record.selection_dirty = True
        record = self._normalize_record(record)
        self._queue_selection_update(record)
        return record

    def refresh(self) -> None:
        wanted = set()
        selected = set(self.store.selected_ids)
        plane = self._view_plane()
        plane_changed = plane != getattr(self, "_last_plane", plane)
        self._last_plane = plane
        y_inverted = self._view_y_inverted()
        y_orientation_changed = y_inverted != getattr(self, "_last_y_inverted", y_inverted)
        self._last_y_inverted = y_inverted
        for record in self.store.records:
            if not record.visible:
                continue
            # Stored ROIs (incl. points) follow the Manager's Show-all / selection
            # gate: with Show-all off, only the selected ROI(s) stay on screen, so
            # unchecking it faithfully hides everything. (Points being *drawn* live
            # in _session_points and stay visible via _refresh_session_points.)
            if not self.store.show_all and record.id not in selected:
                continue
            wanted.add(record.id)
            # On a view-plane flip, rebuild 3-D line/polyline/freehand-line items so
            # their vertices re-project onto the new plane (points use cheap setPos).
            if (
                record.id in self.items
                and (
                    (plane_changed and record.type in {"line", "polyline", "freehand_line"})
                    or (y_orientation_changed and record.type in {"rectangle", "oval"})
                )
            ):
                self.plot_item.removeItem(self.items.pop(record.id))
            if record.id not in self.items:
                item = self._make_item(record)
                self.items[record.id] = item
                self.plot_item.addItem(item)
                self._connect_edit(item, record)
            elif record.type == "point":
                # Re-project the 3-D marker onto the current view plane (so a
                # point drawn in XY shows at its true Z when the view flips to
                # XZ/YZ, instead of reusing its in-plane coordinate).
                px, py = self._project_point(record)
                self.items[record.id].setPos(pg.Point(px, py))
            self._style_item(self.items[record.id], record, record.id in selected,
                             manager_highlight=record.id in selected)
            self._sync_label(record)
        for roi_id in list(self.items):
            if roi_id not in wanted:
                self.plot_item.removeItem(self.items.pop(roi_id))
        for roi_id in list(self.labels):
            if roi_id not in wanted:
                self.plot_item.removeItem(self.labels.pop(roi_id))
        self._refresh_session_points()
        # An in-progress draft (line/polyline/freehand line) must also re-project
        # when the view plane flips, so its 3-D vertices show at the right depth.
        if (
            self.draft is not None
            and self.draft_item is not None
            and (
                (plane_changed and self.draft.type in {"line", "polyline", "freehand_line"})
                or (y_orientation_changed and self.draft.type in {"rectangle", "oval"})
            )
        ):
            try:
                self.plot_item.removeItem(self.draft_item)
            except Exception:
                pass
            self.draft_item = self._make_item(self.draft)
            self._style_item(self.draft_item, self.draft, True)
            self._connect_draft_edit(self.draft_item)
            self.plot_item.addItem(self.draft_item)

    def eventFilter(self, obj, event) -> bool:
        try:
            target = self._event_target()
        except RuntimeError:
            return False
        if obj is not target and obj is not self.view_widget and obj not in self._extra_key_sources:
            return False
        # Keyboard editing of the active ROI (arrow nudge / resize / rotate, and the
        # Fiji-style 't' → add to the ROI Manager), delivered to whichever of the
        # view widget / its viewport / registered key source currently holds focus.
        if event.type() == QEvent.Type.KeyPress:
            if self._handle_arrow_key(event):
                return True
            if self._handle_add_to_manager_key(event):
                return True
        if obj is not target:
            return False
        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.RightButton:
            # Fiji-style: a right-click finishes an in-progress polygon. The click
            # location is only a signal (not a vertex); the loop closes from the
            # last left-click point back to the first.
            if self.store.active_tool == "polygon" and self._polygon_points:
                self._finish_polygon()
                return True
            if self.store.active_tool == "polyline" and self._polyline_points:
                self._finish_polyline()
                return True
            if self.store.active_tool == "magnetic_lasso" and self._lasso_points:
                self._finish_lasso()
                return True
            pos = self._event_to_view(event)
            hit = self._record_at(pos) if pos is not None else None
            if hit is not None:
                self.activate()
                self._show_roi_context_menu(hit, event.globalPosition().toPoint(), pos)
                return True
            return False
        if event.type() == QEvent.Type.FocusIn:
            self.activate()
            return False
        tool = self.store.active_tool
        if not tool:
            # Edit mode: note which sub-element a left click selects on the active
            # ROI (body → move, corner/side → resize, rotation handle → rotate,
            # vertex → move that vertex) so a following arrow key behaves
            # accordingly. The event is not consumed — pyqtgraph still drives the
            # interactive edit.
            if (event.type() == QEvent.Type.MouseButtonPress
                    and event.button() == Qt.MouseButton.LeftButton):
                self._track_kbd_handle_selection(event)
            return False
        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self.activate()
            pos = self._event_to_view(event)
            if pos is None:
                return False
            # Accumulating multi-vertex tools register EVERY left click as a new
            # vertex in draw order — so an existing ROI (or the in-progress draft)
            # under the cursor must NOT swallow the click via _record_at below.
            if tool == "polygon":
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    self._finish_polygon()
                else:
                    self._polygon_points.append(list(pos))
                    self._update_polygon_draft()
                return True
            if tool == "polyline":
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    self._finish_polyline()
                else:
                    self._polyline_points.append(list(pos))
                    self._update_polyline_draft()
                return True
            if tool == "magnetic_lasso":
                snapped = self._snap_pos(pos)
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    self._finish_lasso()
                else:
                    self._lasso_points.append(list(snapped))
                    self._update_lasso_draft()
                return True
            if tool == "angle":
                # Click A, then B (vertex), then C — three points finish it.
                self._angle_points.append(list(pos))
                if len(self._angle_points) >= 3:
                    self._finish_angle()
                else:
                    self._update_angle_draft()
                return True
            if self._record_at(pos) is not None:
                return False
            if tool == "point":
                # Each click drops a persistent point; the tool stays pressed so
                # the user can keep marking points until the toolbar button is
                # clicked again (no auto-release).
                self._commit_point(pos)
                return True
            self._drag_start = pos
            self._freehand_points = [list(pos)]
            self._update_drag_draft(pos)
            return True
        # Live rubber-band preview for click-based open lines (no drag in progress).
        if event.type() == QEvent.Type.MouseMove and self._drag_start is None:
            if tool == "polyline" and self._polyline_points:
                pos = self._event_to_view(event)
                if pos is not None:
                    self._update_polyline_draft(preview=pos)
            elif tool == "magnetic_lasso" and self._lasso_points:
                pos = self._event_to_view(event)
                if pos is not None:
                    self._update_lasso_draft(preview=self._snap_pos(pos))
            return False
        if event.type() == QEvent.Type.MouseMove and self._drag_start is not None:
            pos = self._event_to_view(event)
            if pos is None:
                return False
            if tool in {"freehand", "freehand_line"}:
                self._freehand_points.append(list(pos))
            self._update_drag_draft(pos)
            return True
        if event.type() == QEvent.Type.MouseButtonRelease and self._drag_start is not None:
            pos = self._event_to_view(event)
            if pos is not None:
                self._update_drag_draft(pos)
            self._drag_start = None
            if tool in {"rectangle", "oval", "freehand", "rotated_rectangle", "ellipse"}:
                self._finalize_draft_selection(update_item=True)
            elif tool == "line":
                self._finish_line()
            elif tool == "freehand_line":
                self._finish_freehand_line()
            self._maybe_release_tool(event.modifiers())
            return True
        if event.type() == QEvent.Type.MouseButtonDblClick:
            if tool == "polygon":
                self._finish_polygon()
                return True
            if tool == "polyline":
                self._finish_polyline()
                return True
            if tool == "magnetic_lasso":
                self._finish_lasso()
                return True
        if event.type() == QEvent.Type.KeyPress:
            finisher = {"polygon": self._finish_polygon, "polyline": self._finish_polyline,
                        "magnetic_lasso": self._finish_lasso}.get(tool)
            if finisher is not None:
                if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                    finisher()
                    return True
                if event.key() == Qt.Key.Key_Escape:
                    self._polygon_points = []
                    self._polyline_points = []
                    self._lasso_points = []
                    self._clear_draft()
                    return True
        return False

    # ------------------------------------------------------------------
    # Keyboard fine-adjust for the active ROI (arrow keys)
    # ------------------------------------------------------------------
    # Arrow keys mirror what the mouse would do on the selected handle of the
    # active ROI (the current draft, else the single selected stored ROI, else the
    # active session point):
    #   • body / no handle → translate the whole ROI 1 nm;
    #   • rectangle/oval corner or side handle → nudge that handle 1 nm (resize,
    #     opposite corner anchored); rotation handle → rotate 1° about the centre;
    #   • line/polyline/polygon/freehand/freehand_line/angle vertex → move that
    #     vertex 1 nm;  point → translate the marker 1 nm.
    _KBD_STEP_NM = 1.0
    _KBD_ROTATE_DEG = 1.0
    _KBD_MIN_SIZE_NM = 1.0
    # Scale-handle fractions come from module-level ``box_scale_fracs(on_curve=…)``
    # so the rectangle (corners) and oval (on-curve) share one source of truth with
    # ``_add_box_handles``.

    def _kbd_target(self):
        """``(item, record, kind)`` the arrow keys act on: the current draft, else
        the single selected stored ROI, else the active session point; else
        ``None``."""
        if self.draft is not None and self.draft_item is not None:
            return self.draft_item, self.draft, "draft"
        selected = list(self.store.selected_ids)
        if len(selected) == 1:
            rid = selected[0]
            for rec in self.store.records:
                if rec.id == rid:
                    if rid in self.items:
                        return self.items[rid], rec, "stored"
                    break
        sid = self._active_session_point_id
        if sid is not None and sid in self._session_items:
            for rec in self._session_points:
                if rec.id == sid:
                    return self._session_items[sid], rec, "session"
        return None

    def _kbd_role_at(self, record: RoiRecord, pos):
        """Which sub-element a click at *pos* selects on the active ROI:
        rect/oval → ``"rotate"`` / ``("scale", fx, fy)`` / ``"move"``; point →
        ``"move"``; vertex ROIs → ``("vertex", i)`` / ``"move"``; else ``None``."""
        if record is None:
            return None
        if record.type in {"rectangle", "oval"}:
            return self._rect_handle_role_at(record, pos)
        if record.type == "point":
            px, py = self._project_point(record)
            tol = self._hit_tolerance() * 3.0
            if abs(pos[0] - px) <= tol and abs(pos[1] - py) <= tol:
                return "move"
            return None
        # vertex-based: line / polyline / freehand_line / polygon / freehand / angle
        from ..core.roi_vertex_edit import nearest_vertex

        projected = self._project_points(record)
        if projected:
            idx, dist = nearest_vertex(projected, pos)
            if idx >= 0 and dist <= self._hit_tolerance() * 3.0:
                return ("vertex", int(idx))
        if self._point_hits_record(pos, record, self._hit_tolerance()):
            return "move"
        return None

    def _rect_handle_role_at(self, record: RoiRecord, pos):
        """Which handle a click at *pos* selects on rectangle / oval *record*:
        ``"rotate"`` near the rotation handle, ``("scale", fx, fy)`` near one of
        the corner / side scale handles, ``"move"`` elsewhere on the body, else
        ``None`` (the click missed the ROI).

        A **rotated variant** (``geometry["variant"] == "rotated"``) has a solid
        pivot handle (rotate about centre) at a corner / axis endpoint and 3 scale
        handles; the classic layout has 8 scale handles + a rotate handle above."""
        if record is None or record.type not in {"rectangle", "oval"}:
            return None
        x, y, w, h = _bounds(record.geometry)
        cx, cy = x + w / 2.0, y + h / 2.0
        half_w = max(w / 2.0, 0.0)
        half_h = max(h / 2.0, 0.0)
        angle = float(record.geometry.get("angle", 0.0) or 0.0)
        theta = np.deg2rad(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        tol = self._hit_tolerance()

        def _handle_xy(fx, fy):
            ox, oy = (fx - 0.5) * w, (fy - 0.5) * h
            return cx + cos_t * ox - sin_t * oy, cy + sin_t * ox + cos_t * oy

        variant = str(record.geometry.get("variant") or "")
        if variant == "rotated":
            rotate_frac, scale_fracs = rotated_variant_handles(record.type)
            rot_radius = max(tol * 3.0, min(half_w, half_h) * 0.2)
            hx, hy = _handle_xy(*rotate_frac)
            if (float(pos[0]) - hx) ** 2 + (float(pos[1]) - hy) ** 2 <= rot_radius ** 2:
                return "rotate"
        else:
            # No rotate handle on a plain (axis-aligned) rectangle / oval — rotation
            # is only via the Rotated Rectangle / Ellipse variants (pivot above).
            scale_fracs = box_scale_fracs(on_curve=record.type == "oval")
        # Nearest corner / side scale handle on the (rotated) box boundary. Cap the
        # pick radius to half the smaller half-extent so a click in the body centre
        # of a thin box is never mistaken for an edge handle (stays "move").
        pick_radius = max(tol * 3.0, min(half_w, half_h) * 0.2)
        pick_radius = min(pick_radius, min(half_w, half_h) * 0.5)
        best = None
        best_d2 = pick_radius ** 2
        for fx, fy in scale_fracs:
            hx, hy = _handle_xy(fx, fy)
            d2 = (float(pos[0]) - hx) ** 2 + (float(pos[1]) - hy) ** 2
            if d2 <= best_d2:
                best_d2 = d2
                best = (fx, fy)
        if best is not None:
            return ("scale", best[0], best[1])
        dx = float(pos[0]) - cx
        dy = float(pos[1]) - cy
        local_x = cos_t * dx + sin_t * dy
        local_y = -sin_t * dx + cos_t * dy
        if abs(local_x) <= half_w + tol and abs(local_y) <= half_h + tol:
            return "move"
        return None

    def _track_kbd_handle_selection(self, event) -> None:
        """On a left click in edit mode, record which sub-element of the active ROI
        the user selected (and, for a clicked session point, make it the active
        keyboard target), so a following arrow key behaves accordingly."""
        pos = self._event_to_view(event)
        if pos is None:
            return
        tol = self._hit_tolerance() * 3.0
        # A click on a session point makes it the active (movable) keyboard target.
        for rec in reversed(self._session_points):
            px, py = self._project_point(rec)
            if abs(pos[0] - px) <= tol and abs(pos[1] - py) <= tol:
                self._active_session_point_id = rec.id
                self._kbd_target_id = rec.id
                self._kbd_handle = "move"
                return
        target = self._kbd_target()
        if target is None:
            return
        _item, record, _kind = target
        role = self._kbd_role_at(record, pos)
        if role is not None:
            self._kbd_handle = role
            self._kbd_target_id = record.id

    def _handle_arrow_key(self, event) -> bool:
        """Arrow-key move / resize / rotate for the active ROI. Returns ``True``
        when the key was consumed."""
        key = event.key()
        if key not in (Qt.Key.Key_Left, Qt.Key.Key_Right,
                       Qt.Key.Key_Up, Qt.Key.Key_Down):
            return False
        if event.modifiers() & (Qt.KeyboardModifier.ControlModifier
                                | Qt.KeyboardModifier.AltModifier
                                | Qt.KeyboardModifier.MetaModifier):
            return False
        target = self._kbd_target()
        if target is None:
            return False
        item, record, kind = target
        # Switching the active ROI resets the selected sub-element to whole-ROI move.
        if record.id != getattr(self, "_kbd_target_id", None):
            self._kbd_target_id = record.id
            self._kbd_handle = "move"
        role = getattr(self, "_kbd_handle", "move")
        if record.type in {"rectangle", "oval"}:
            if role == "rotate":
                self._rotate_active_rect(item, kind, record, key)
            elif isinstance(role, tuple) and len(role) == 3 and role[0] == "scale":
                self._resize_active_rect(item, kind, record, key, role[1], role[2])
            else:
                self._move_whole_roi(item, kind, record, key)
        elif record.type == "point":
            self._move_point(item, kind, record, key)
        elif isinstance(role, tuple) and len(role) == 2 and role[0] == "vertex":
            self._move_vertex(item, kind, record, key, role[1])
        else:
            self._move_whole_roi(item, kind, record, key)
        return True

    def _handle_add_to_manager_key(self, event) -> bool:
        """Fiji-style ``t``: add the active ROI to the ROI Manager. Returns ``True``
        when the key was consumed."""
        if event.key() != Qt.Key.Key_T:
            return False
        if event.modifiers() & (Qt.KeyboardModifier.ControlModifier
                                | Qt.KeyboardModifier.AltModifier
                                | Qt.KeyboardModifier.MetaModifier):
            return False
        return self._add_active_roi_to_manager()

    def _add_active_roi_to_manager(self) -> bool:
        """Add the active ROI — the current draft, else the active session point —
        to the ROI Manager, opening one if none is visible (``_add_hit_to_manager``
        → ``_show_manager_if_needed`` + ``store.add``). Returns ``True`` if a ROI
        was added."""
        if self.draft is not None:
            self.activate()
            self._add_hit_to_manager("draft", self.draft)
            return True
        sid = self._active_session_point_id
        if sid is not None:
            for rec in list(self._session_points):
                if rec.id == sid:
                    self.activate()
                    self._add_hit_to_manager("session", rec)
                    return True
        return False

    def _arrow_data_delta(self, key) -> tuple[float, float]:
        """Data-space (dx, dy) for a 1 nm step in the arrow's *screen* direction —
        Up is up on screen for either y-axis orientation."""
        step = self._KBD_STEP_NM
        up_sign = -1.0 if self._view_y_inverted() else 1.0
        if key == Qt.Key.Key_Left:
            return -step, 0.0
        if key == Qt.Key.Key_Right:
            return step, 0.0
        if key == Qt.Key.Key_Up:
            return 0.0, step * up_sign
        if key == Qt.Key.Key_Down:
            return 0.0, -step * up_sign
        return 0.0, 0.0

    def _move_whole_roi(self, item, kind: str, record, key) -> None:
        """Translate the whole active ROI one 1 nm step in the arrow direction
        (body / no specific handle selected)."""
        dx, dy = self._arrow_data_delta(key)
        try:
            item.translate((dx, dy), finish=True)
        except Exception:
            return
        self._after_roi_key_edit(item, kind, record)

    def _move_point(self, item, kind: str, record, key) -> None:
        """Translate a point marker one 1 nm step in the arrow direction, keeping
        its out-of-plane (depth) coordinate."""
        dx, dy = self._arrow_data_delta(key)
        try:
            pos = item.pos()
            new_xy = (float(pos.x()) + dx, float(pos.y()) + dy)
            item.setPos(pg.Point(*new_xy))
        except Exception:
            return
        # Draft / session records are ours to update in place; a stored point is a
        # live-edit copy (committed only on Manager Update).
        if kind in {"draft", "session"}:
            old = record.geometry.get("point", [0.0, 0.0])
            record.geometry = {"point": update_point_in_plane(new_xy, old, self._view_plane())}
        self._emit_status(self._roi_status_text("point", {"point": [new_xy[0], new_xy[1]]}))

    def _move_vertex(self, item, kind: str, record, key, idx: int) -> None:
        """Nudge the selected vertex of a line / polyline / polygon / freehand /
        angle ROI 1 nm in the arrow direction (mirrors dragging that handle)."""
        dx, dy = self._arrow_data_delta(key)
        handles = getattr(item, "handles", None) or []
        if idx < 0 or idx >= len(handles):
            return
        try:
            _name, scene = item.getSceneHandlePositions(idx)
            parent = item.mapSceneToParent(scene)
            item.movePoint(handles[idx]["item"],
                           pg.Point(float(parent.x()) + dx, float(parent.y()) + dy),
                           coords="parent", finish=True)
        except Exception:
            return
        self._after_roi_key_edit(item, kind, record)

    def _resize_active_rect(self, item, kind: str, record, key, fx: float, fy: float) -> None:
        """Nudge the selected corner / side handle 1 nm in the arrow direction,
        resizing the rectangle / oval with the opposite corner anchored — the
        keyboard equivalent of dragging that scale handle. Handles rotation via
        the item's own transform-aware :meth:`setSize`."""
        dx, dy = self._arrow_data_delta(key)
        try:
            size = item.size()
            w, h = float(size.x()), float(size.y())
            theta = np.deg2rad(float(item.angle()))
        except Exception:
            return
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Component of the move along the box's own (rotated) width / height axes.
        du = cos_t * dx + sin_t * dy
        dv = -sin_t * dx + cos_t * dy
        # Which side each fraction is on: >0.5 grows the +axis edge, <0.5 the −axis,
        # ≈0.5 (an edge-midpoint handle) leaves that axis unchanged. Works for both
        # the rectangle's corners (fx∈{0,1}) and the oval's on-curve diagonals (≈0.854).
        s_x = 1.0 if fx > 0.5 + 1e-3 else (-1.0 if fx < 0.5 - 1e-3 else 0.0)
        s_y = 1.0 if fy > 0.5 + 1e-3 else (-1.0 if fy < 0.5 - 1e-3 else 0.0)
        w2 = max(w + du * s_x, self._KBD_MIN_SIZE_NM)
        h2 = max(h + dv * s_y, self._KBD_MIN_SIZE_NM)
        try:
            # centerLocal = the opposite handle in old-size local coords → anchored.
            item.setSize([w2, h2], centerLocal=[(1.0 - fx) * w, (1.0 - fy) * h],
                         finish=True)
        except Exception:
            return
        self._after_roi_key_edit(item, kind, record)

    def _rotate_active_rect(self, item, kind: str, record, key) -> None:
        """Rotate the active rectangle / oval 1° per press about its centre. Right /
        Up turn clockwise on screen, Left / Down counter-clockwise (either y-axis
        orientation)."""
        # +delta reads as clockwise-on-screen in a y-inverted view (the render
        # default), counter-clockwise otherwise.
        cw_delta = self._KBD_ROTATE_DEG if self._view_y_inverted() else -self._KBD_ROTATE_DEG
        clockwise = key in (Qt.Key.Key_Right, Qt.Key.Key_Up)
        delta = cw_delta if clockwise else -cw_delta
        try:
            item.rotate(delta, center=[0.5, 0.5], finish=True)
        except Exception:
            return
        self._after_roi_key_edit(item, kind, record)

    def _after_roi_key_edit(self, item, kind: str, record) -> None:
        """Finish a keyboard ROI edit. For a draft the ``sigRegionChangeFinished``
        handler has already re-read the geometry, rebuilt the item and re-queued
        the selection; a stored ROI is a live-edit copy (committed only on Manager
        Update), so just refresh the read-out."""
        if kind != "stored":
            return
        try:
            if record.type == "point":
                p = item.pos()
                geom = {"point": [float(p.x()), float(p.y())]}
            else:
                geom = item_to_geometry(record.type, item)
            self._emit_status(self._roi_status_text(record.type, geom))
        except Exception:
            pass

    def _commit_point(self, pos) -> None:
        """Drop a point marker at *pos*.  Points are a multi-shot tool: the point
        tool stays active after each click (the toolbar button stays pressed) so
        the user can drop as many points as they like; clicking the toolbar
        button again releases the tool.

        The marker is held as a **session point** (drawn but not yet in the ROI
        Manager) — it only enters the store when the user right-clicks and
        chooses *Add to Manager* / *Add all session points to Manager*, the same
        explicit-commit rule as every other ROI shape.

        Each marker is a full 3-D coordinate: the in-plane axes come from the
        click and the out-of-plane (depth) axis is set to the centre of the
        current viewing range of that dimension (so a point drawn in XY gets
        z = mid of the depth slider, not an undefined/garbage value)."""
        record = RoiRecord.create("point", {"point": self._point_to_3d(pos)}, **self._record_kwargs())
        record = self._normalize_record(record)
        self._add_session_point(record)
        # Read-out: this point's X/Y + the running session-point count.
        p = record.geometry.get("point") or [0.0, 0.0, 0.0]
        self._emit_status(
            f"Point X={float(p[0]):.0f}, Y={float(p[1]):.0f}   "
            f"(session point {len(self._session_points)})")

    # ------------------------------------------------------------------
    # Session points (pending markers, not yet in the Manager/store)
    # ------------------------------------------------------------------

    def _add_session_point(self, record: RoiRecord) -> None:
        self._session_points.append(record)
        # The just-placed point becomes the active arrow-key target so it can be
        # nudged 1 nm at a time without leaving the point tool.
        self._active_session_point_id = record.id
        self._kbd_target_id = record.id
        self._kbd_handle = "move"
        item = self._make_item(record)
        self._session_items[record.id] = item
        self.plot_item.addItem(item)
        if hasattr(item, "sigPositionChangeFinished"):
            item.sigPositionChangeFinished.connect(
                lambda _item=item, rid=record.id: self._update_session_point(rid, _item)
            )

    def _update_session_point(self, roi_id: str, item) -> None:
        for rec in self._session_points:
            if rec.id == roi_id:
                rec.geometry = self._point_geometry_from_item(item, rec.geometry)
                return

    def _refresh_session_points(self) -> None:
        """Re-project pending point markers onto the current view plane."""
        for rec in self._session_points:
            item = self._session_items.get(rec.id)
            if item is not None:
                px, py = self._project_point(rec)
                item.setPos(pg.Point(px, py))

    def _remove_session_point(self, roi_id: str) -> None:
        self._session_points = [r for r in self._session_points if r.id != roi_id]
        if self._active_session_point_id == roi_id:
            self._active_session_point_id = None
        item = self._session_items.pop(roi_id, None)
        if item is not None:
            try:
                self.plot_item.removeItem(item)
            except Exception:
                pass

    def _add_all_session_points_to_manager(self) -> None:
        records = list(self._session_points)
        if not records:
            return
        for rec in records:
            item = self._session_items.pop(rec.id, None)
            if item is not None:
                try:
                    self.plot_item.removeItem(item)
                except Exception:
                    pass
        self._session_points = []
        self._active_session_point_id = None
        self._show_manager_if_needed()
        for rec in records:
            self.store.add(rec)
        self.store.deselect()            # all points filed into the Manager, off-screen

    # ------------------------------------------------------------------
    # 2-D view <-> 3-D coordinate helpers (out-of-plane = depth)
    # ------------------------------------------------------------------

    def _view_plane(self) -> str | None:
        """Current view plane (``"XY"``/``"XZ"``/``"YZ"``) from the owner, or
        ``None`` for owners without an orientation (e.g. histogram)."""
        getter = getattr(self.owner, "roi_view_plane", None)
        if callable(getter):
            try:
                plane = getter()
            except Exception:
                return None
            return plane if plane in _PLANE_PLOT_AXES else None
        return None

    def _depth_center(self) -> float | None:
        """Centre of the current viewing range of the out-of-plane dimension."""
        getter = getattr(self.owner, "roi_depth_center", None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                return None
        return None

    def _depths_at(self, pts_2d) -> list:
        """Per-vertex out-of-plane (depth) value for in-plane points: the
        data-aware weighted median of the structure under each vertex (so a point
        drawn in XY lands on the actual object along Z), falling back to the
        static view-range centre wherever the column is empty."""
        center = self._depth_center()
        getter = getattr(self.owner, "roi_depths_at", None)
        depths = None
        if callable(getter):
            try:
                depths = getter([(float(p[0]), float(p[1])) for p in pts_2d])
            except Exception:
                depths = None
        if depths is None:
            return [center] * len(pts_2d)
        return [center if d is None else float(d) for d in depths]

    def _point_to_3d(self, pos) -> list[float]:
        """Build a full XYZ coordinate for a point clicked at in-plane *pos*."""
        return self._points_to_3d([pos])[0]

    def _project_point(self, record: RoiRecord) -> tuple[float, float]:
        """Project a (possibly 3-D) point geometry into the current view plane."""
        return project_point(record.geometry.get("point", [0.0, 0.0]), self._view_plane())

    def _point_geometry_from_item(self, item, old_geometry: dict[str, Any]) -> dict[str, Any]:
        """Rebuild a point's 3-D geometry after a drag in the current plane,
        preserving the out-of-plane (depth) coordinate untouched — so dragging a
        point in XZ/YZ edits its Z (and the other in-plane axis) without
        disturbing the value set when it was drawn in another view."""
        pos = item.pos()
        old = old_geometry.get("point", [0.0, 0.0])
        return {"point": update_point_in_plane((pos.x(), pos.y()), old, self._view_plane())}

    def _points_to_3d(self, pts_2d) -> list[list[float]]:
        """Lift a 2-D draw (point / line / polyline / freehand line) to full XYZ,
        giving each vertex a data-aware out-of-plane (depth) value."""
        plane = self._view_plane()
        depths = self._depths_at(pts_2d)
        return [point_to_3d(p, plane, d) for p, d in zip(pts_2d, depths)]

    def _project_points(self, record: RoiRecord) -> list[list[float]]:
        """Project a (possibly 3-D) vertex-list geometry into the current plane."""
        return project_points(record.geometry.get("points", []), self._view_plane())

    def _open_line_geometry_from_item(self, item, old_geometry: dict[str, Any], roi_type: str) -> dict[str, Any]:
        """Rebuild a line / polyline / freehand-line geometry after handle moves,
        preserving each vertex's out-of-plane (depth) axis."""
        new_xy = item_to_geometry(roi_type, item).get("points", [])
        old = old_geometry.get("points", [])
        if len(new_xy) == len(old):
            # A pure move / reshape: pair 1:1 and keep each vertex's depth.
            pts = update_points_in_plane(new_xy, old, self._view_plane())
        else:
            # A vertex was added or removed (e.g. pyqtgraph's segment-click adds a
            # vertex when the line is clicked): the index pairing is invalid, so
            # re-derive a **consistent** 3-D geometry (data-aware depth) from the
            # item's 2-D vertices — never leave a mix of 2-D/3-D points (which
            # crashed np.asarray in _bounds).
            pts = self._points_to_3d([[float(p[0]), float(p[1])] for p in new_xy])
        return {"points": pts, "closed": bool(old_geometry.get("closed", False))}

    def _maybe_release_tool(self, modifiers) -> None:
        """After a completed ROI draw, deactivate the shape tool so it is no
        longer in a pressed state — unless **Ctrl** is held, which keeps it
        active for drawing several ROIs in a row."""
        if not (modifiers & Qt.KeyboardModifier.ControlModifier):
            self.store.set_tool(None)

    def _record_kwargs(self) -> dict[str, Any]:
        color = self.owner._state.prefs.get("plot", {}).get("roi_color", "Yellow")
        return {
            "coordinate_space": self.coordinate_space,
            "target_hint": self.owner.windowTitle(),
            "stroke_color": color,
        }

    def _event_target(self):
        return self.view_widget.viewport() if hasattr(self.view_widget, "viewport") else self.view_widget

    def _view_y_inverted(self) -> bool:
        try:
            return bool(getattr(self.view_box, "state", {}).get("yInverted", False))
        except Exception:
            return False

    def _event_to_view(self, event) -> tuple[float, float] | None:
        point = event.position().toPoint() if hasattr(event, "position") else event.pos()
        scene_pos = self.view_widget.mapToScene(point)
        view_pos = self.view_box.mapSceneToView(scene_pos)
        return float(view_pos.x()), float(view_pos.y())

    def _update_drag_draft(self, pos: tuple[float, float]) -> None:
        if self._drag_start is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = pos
        tool = self.store.active_tool
        if tool in {"rectangle", "oval"}:
            x, y = min(x0, x1), min(y0, y1)
            w, h = abs(x1 - x0), abs(y1 - y0)
            self._set_draft(RoiRecord.create(tool, {"bounds": [x, y, w, h]}, **self._record_kwargs()))
        elif tool in {"rotated_rectangle", "ellipse"}:
            # The drag is the long *centre* axis (length + rotation); the short side
            # is the golden ratio of it. The record type stays rectangle/oval (masks
            # & I/O unchanged) — geometry["variant"] only drives the handle layout.
            bounds, angle = center_axis_box(x0, y0, x1, y1)
            rtype = "rectangle" if tool == "rotated_rectangle" else "oval"
            self._set_draft(RoiRecord.create(
                rtype, {"bounds": bounds, "angle": angle, "variant": "rotated"},
                **self._record_kwargs()))
        elif tool == "line":
            self._set_draft(RoiRecord.create("line", {"points": [[x0, y0], [x1, y1]], "closed": False}, **self._record_kwargs()))
        elif tool == "freehand":
            self._set_draft(RoiRecord.create("freehand", {"points": self._freehand_points, "closed": True}, **self._record_kwargs()))
        elif tool == "freehand_line":
            self._set_draft(RoiRecord.create("freehand_line", {"points": self._freehand_points, "closed": False}, **self._record_kwargs()))

    def _update_polygon_draft(self) -> None:
        if len(self._polygon_points) >= 1:
            self._set_draft(RoiRecord.create("polygon", {"points": self._polygon_points, "closed": False}, **self._record_kwargs()))

    def _draw_points(self, committed, preview):
        """Helper: committed vertices + optional rubber-band preview, ≥2 long."""
        pts = list(committed)
        if preview is not None:
            pts = pts + [list(preview)]
        return pts if len(pts) >= 2 else (pts + pts if pts else [[0, 0], [0, 0]])

    def _update_polyline_draft(self, preview=None) -> None:
        if self._polyline_points:
            self._set_draft(RoiRecord.create(
                "polyline", {"points": self._draw_points(self._polyline_points, preview), "closed": False},
                **self._record_kwargs()))

    def _finish_polyline(self) -> None:
        if len(self._polyline_points) >= 2:
            self._set_draft(RoiRecord.create(
                "polyline", {"points": self._points_to_3d(self._polyline_points), "closed": False},
                **self._record_kwargs()))
            self._finalize_draft_selection(update_item=False)
        self._polyline_points = []
        # A right-click finish releases the tool (button unpresses); Ctrl keeps it
        # active for drawing several polylines in a row.
        self._maybe_release_tool(QApplication.keyboardModifiers())

    def _finish_line(self) -> None:
        if self.draft is None or self.draft.type != "line":
            return
        pts2d = [p[:2] for p in self.draft.geometry.get("points", [])[:2]]
        self.draft.geometry = {"points": self._points_to_3d(pts2d), "closed": False}
        self._finalize_draft_selection(update_item=True)

    def _finish_freehand_line(self) -> None:
        if len(self._freehand_points) >= 2:
            self._set_draft(RoiRecord.create(
                "freehand_line", {"points": self._points_to_3d(self._freehand_points), "closed": False},
                **self._record_kwargs()))
            self._finalize_draft_selection(update_item=True)
        self._freehand_points = []

    # --- magnetic lasso (snaps each vertex to the local density centre) -------
    def _snap_pos(self, pos):
        """Snap an in-plane cursor position to the high-density centre via the
        owner's render raster; falls back to the raw position if unavailable."""
        snapper = getattr(self.owner, "snap_to_density", None)
        if callable(snapper):
            try:
                snapped = snapper(float(pos[0]), float(pos[1]))
            except Exception:
                snapped = None
            if snapped is not None:
                return (float(snapped[0]), float(snapped[1]))
        return (float(pos[0]), float(pos[1]))

    def _update_lasso_draft(self, preview=None) -> None:
        if self._lasso_points:
            self._set_draft(RoiRecord.create(
                "polyline", {"points": self._draw_points(self._lasso_points, preview), "closed": False},
                **self._record_kwargs()))

    def _finish_lasso(self) -> None:
        if len(self._lasso_points) >= 2:
            self._set_draft(RoiRecord.create(
                "polyline", {"points": self._points_to_3d(self._lasso_points), "closed": False},
                **self._record_kwargs()))
            self._finalize_draft_selection(update_item=False)
        self._lasso_points = []
        self._maybe_release_tool(QApplication.keyboardModifiers())

    def _finish_polygon(self) -> None:
        completed = len(self._polygon_points) >= 3
        if completed:
            self._set_draft(RoiRecord.create("polygon", {"points": self._polygon_points, "closed": True}, **self._record_kwargs()))
            self._finalize_draft_selection(update_item=False)
        self._polygon_points = []
        if completed:
            # Ctrl-click finishes a polygon AND keeps the tool; double-click /
            # Enter (no Ctrl) finishes and releases it.
            self._maybe_release_tool(QApplication.keyboardModifiers())

    def _update_angle_draft(self) -> None:
        if self._angle_points:
            self._set_draft(RoiRecord.create(
                "angle", {"points": list(self._angle_points), "closed": False},
                **self._record_kwargs()))

    def _finish_angle(self) -> None:
        pts = self._angle_points[:3]
        self._angle_points = []
        if len(pts) >= 3:
            self._set_draft(RoiRecord.create(
                "angle", {"points": pts, "closed": False}, **self._record_kwargs()))
            self._report_angle(pts, log=True)
        self._maybe_release_tool(QApplication.keyboardModifiers())

    def _report_angle(self, pts, *, log: bool = False) -> None:
        """Report the measured angle to the main-window status bar (and the Log)."""
        try:
            deg = _angle_abc(pts[0], pts[1], pts[2])
        except Exception:
            return
        state = getattr(self.owner, "_state", None)
        if state is None:
            return
        signal = getattr(state, "status_message", None)
        if signal is not None:
            try:
                signal.emit(f"Angle = {deg:.2f}°")
            except Exception:
                pass
        if log:
            try:
                state.log("drawing angle")
            except Exception:
                pass

    def _report_angle_from_item(self, item) -> None:
        try:
            pts = item_to_geometry("angle", item).get("points", [])
        except Exception:
            return
        if len(pts) >= 3:
            self._report_angle(pts[:3])

    def _emit_status(self, text: str) -> None:
        """Push a live read-out to the main-window status bar (best effort)."""
        if not text:
            return
        state = getattr(self.owner, "_state", None)
        sig = getattr(state, "status_message", None) if state is not None else None
        if sig is not None:
            try:
                sig.emit(text)
            except Exception:
                pass

    def _clear_status(self) -> None:
        """Blank the status bar (ROI drawing/editing finished, deleted or committed)."""
        state = getattr(self.owner, "_state", None)
        sig = getattr(state, "status_message", None) if state is not None else None
        if sig is not None:
            try:
                sig.emit("")
            except Exception:
                pass

    @staticmethod
    def _roi_status_text(roi_type: str, geometry: dict) -> str:
        """Concise live geometry read-out (integer nm) for a draft ROI: origin +
        size for rectangle/oval, start point + length + angle for a straight line,
        type + vertex count + bounding box for other shapes, X/Y for a point.
        Coordinates may carry a depth (3-D) — always index ``[0]``/``[1]``, never
        ``x, y = p``."""
        g = geometry or {}
        if roi_type == "angle":
            return ""                                   # angle has its own read-out
        if roi_type in {"rectangle", "oval"}:
            b = g.get("bounds") or [0, 0, 0, 0]
            x, y, w, h = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            ang = normalize_angle_deg(float(g.get("angle", 0.0) or 0.0))  # (-180, 180]
            label = "Rectangle" if roi_type == "rectangle" else "Oval"
            text = f"{label} X={x:.0f}, Y={y:.0f}, W={w:.0f}, H={h:.0f}"
            return text + (f", angle={ang:.0f}°" if abs(ang) > 1e-6 else "")
        if roi_type == "point":
            p = g.get("point") or [0.0, 0.0, 0.0]
            return f"Point X={float(p[0]):.0f}, Y={float(p[1]):.0f}"
        pts = g.get("points") or []                     # line/polyline/polygon/freehand…
        if roi_type == "line" and len(pts) >= 2:
            # Straight line: start point, length, and the vector's angle from the
            # +X axis in (-180, 180] (start treated as the polar origin).
            x0, y0 = float(pts[0][0]), float(pts[0][1])
            x1, y1 = float(pts[-1][0]), float(pts[-1][1])
            length = float(np.hypot(x1 - x0, y1 - y0))
            angle = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
            return (f"Line {len(pts)} vertices, X={x0:.0f}, Y={y0:.0f}, "
                    f"length={length:.0f}, angle={angle:.1f}°")
        if len(pts) >= 1:
            arr = np.asarray([[float(p[0]), float(p[1])] for p in pts], dtype=float)
            x0, y0 = float(arr[:, 0].min()), float(arr[:, 1].min())
            w, h = float(np.ptp(arr[:, 0])), float(np.ptp(arr[:, 1]))
            label = {"line": "Line", "polyline": "Polyline", "freehand_line": "Freehand line",
                     "polygon": "Polygon", "freehand": "Freehand"}.get(roi_type, roi_type.capitalize())
            return (f"{label}  {len(pts)} vertices  "
                    f"Bounding Box X={x0:.0f}, Y={y0:.0f}, W={w:.0f}, H={h:.0f}")
        return ""

    def _report_roi_status(self, record) -> None:
        """Live status for a draft *record* (angle uses its dedicated read-out)."""
        rtype = getattr(record, "type", None)
        if rtype in (None, "angle"):
            return
        try:
            self._emit_status(self._roi_status_text(rtype, getattr(record, "geometry", {})))
        except Exception:
            pass

    def _report_status_from_item(self, item) -> None:
        """Live status while a draft ROI is edited (moved / reshaped)."""
        if self.draft is None or self.draft.type == "angle":
            return
        try:
            geom = item_to_geometry(self.draft.type, item)
        except Exception:
            return
        self._emit_status(self._roi_status_text(self.draft.type, geom))

    def _set_draft(self, record: RoiRecord) -> None:
        # A previous *region* draft's data highlight must not linger when the draft
        # is replaced by a **non-region** ROI (e.g. drawing a line after a
        # rectangle/oval selection): a non-region draft runs no selection pass, so
        # it never clears the old active-draft mask — drop the old highlight here.
        # (A region→region replacement is handled by _replace_active_draft_mask in
        # the selection recompute, so it is skipped to avoid redundant work while a
        # new region is being drag-drawn.)
        old = self.draft
        if (old is not None and old is not record and old.id != record.id
                and old.type in self._REGION_TYPES
                and record.type not in self._REGION_TYPES
                and old.id not in {r.id for r in self.store.records}):
            self._delete_mask_for_record(old)
        self.draft = record
        if self.draft_item is not None:
            self.plot_item.removeItem(self.draft_item)
        self.draft_item = self._make_item(record)
        self._style_item(self.draft_item, record, True)
        self._connect_draft_edit(self.draft_item)
        self.plot_item.addItem(self.draft_item)
        self._report_roi_status(record)          # live geometry read-out → status bar

    def _clear_draft(self) -> None:
        self._clear_status()                         # drawing done / cancelled / deleted
        self.draft = None
        self._polygon_points = []
        self._polyline_points = []
        self._lasso_points = []
        self._freehand_points = []
        self._angle_points = []
        if self.draft_item is not None:
            try:
                self.plot_item.removeItem(self.draft_item)
            except Exception:
                pass
        self.draft_item = None

    def _show_manager_if_needed(self) -> None:
        app = QApplication.instance()
        if app is None:
            return
        for widget in app.topLevelWidgets():
            if widget.__class__.__name__ == "RoiManagerWindow" and widget.isVisible():
                return
        for widget in app.topLevelWidgets():
            show_manager = getattr(widget, "_show_roi_manager", None)
            if callable(show_manager):
                show_manager()
                return

    def _make_item(self, record: RoiRecord):
        g = record.geometry
        if record.type in {"rectangle", "oval"}:
            x, y, w, h = _bounds(g)
            angle = float(g.get("angle", 0.0) or 0.0)
            # geometry["bounds"] is the *unrotated* axis-aligned box (its centre is
            # the rotation centre). pyqtgraph rotates a ROI about its origin, so
            # offset the origin by −R(angle)·(w/2, h/2) to keep the centre fixed.
            cx, cy = x + w / 2.0, y + h / 2.0
            if abs(angle) > 1e-9:
                t = np.deg2rad(angle)
                cos_t, sin_t = np.cos(t), np.sin(t)
                hx = cos_t * (w / 2.0) - sin_t * (h / 2.0)
                hy = sin_t * (w / 2.0) + cos_t * (h / 2.0)
                px, py = cx - hx, cy - hy
            else:
                px, py = x, y
            size = [max(w, 1e-9), max(h, 1e-9)]
            variant = str(g.get("variant") or "")
            if record.type == "rectangle":
                return FilledRectROI([px, py], size, angle=angle,
                                     fill_color=record.stroke_color,
                                     y_axis_inverted=self._view_y_inverted(),
                                     variant=variant, movable=True)
            return FilledEllipseROI([px, py], size, angle=angle,
                                    fill_color=record.stroke_color,
                                    y_axis_inverted=self._view_y_inverted(),
                                    variant=variant, movable=True)
        if record.type == "line":
            pts = self._project_points(record) or [[0, 0], [1, 1]]
            item = pg.LineSegmentROI(pts[:2], movable=True)
            _mark_line_end_handle(item)           # oval marker on the end vertex
            return item
        if record.type == "point":
            x, y = self._project_point(record)
            return PointMarkerItem((x, y))
        if record.type == "angle":
            pts = g.get("points", [[0, 0], [1, 0], [2, 1]])
            item = pg.PolyLineROI([p[:2] for p in pts[:3]], closed=False, movable=True)
            _mark_line_end_handle(item)           # round the end (C) vertex
            return item
        if record.type in {"polyline", "freehand_line"}:
            # open curves (no fill / no enclosed region) — but a region→Line
            # conversion yields a *closed* outline (still a line type, no mask).
            pts = self._project_points(record)
            if len(pts) < 2:
                pts = [[0, 0], [1, 1]]
            closed = bool(g.get("closed", False))
            item = pg.PolyLineROI(pts, closed=closed, movable=True)
            if not closed:                        # open line → mark the end vertex
                _mark_line_end_handle(item)
            return item
        pts = g.get("points", [])
        if len(pts) < 2:
            pts = [[0, 0], [1, 1]]
        return FilledPolyLineROI(
            [p[:2] for p in pts], closed=bool(g.get("closed", record.type != "line")),
            fill_color=record.stroke_color, movable=True,
        )

    #: Distinct outline/fill colour for a ROI **selected in the ROI Manager**, so it
    #: stands out from the other displayed (Show-all) ROIs drawn in their own colour.
    MANAGER_SELECT_COLOR = "#00e5ff"          # bright cyan — reads on black & white bg

    def _style_item(self, item, record: RoiRecord, selected: bool,
                    *, manager_highlight: bool = False) -> None:
        color = self.MANAGER_SELECT_COLOR if manager_highlight else (record.stroke_color or "#ffff00")
        width = record.line_width + (1.5 if selected else 0.0)
        pen = pg.mkPen(color, width=width)
        try:
            item.setPen(pen)
        except Exception:
            pass
        if record.type in {"rectangle", "oval", "polygon", "freehand"}:
            try:
                fill = pg.mkColor(color)
                fill.setAlpha(32 if selected else 22)
                if hasattr(item, "setFillColor"):
                    item.setFillColor(fill, alpha=fill.alpha())
                elif hasattr(item, "setBrush"):
                    item.setBrush(pg.mkBrush(fill))
            except Exception:
                pass

    def _connect_edit(self, item, record: RoiRecord) -> None:
        # A stored ROI displayed in the view is a live *editable copy*: moving or
        # reshaping it must NOT change the Manager's record until the user clicks
        # Update (which calls record_for_update). So we deliberately do not commit
        # edits here — only the angle ROI's live status read-out is wired up.
        if record.type == "angle" and hasattr(item, "sigRegionChanged"):
            item.sigRegionChanged.connect(lambda _item=item: self._report_angle_from_item(_item))

    def _connect_draft_edit(self, item) -> None:
        if self.draft is not None and self.draft.type == "point" and hasattr(item, "sigPositionChangeFinished"):
            item.sigPositionChangeFinished.connect(lambda _item=item: self._update_draft_from_item(_item))
            if hasattr(item, "sigPositionChanged"):      # live point drag → status
                item.sigPositionChanged.connect(lambda _item=item: self._report_status_from_item(_item))
            return
        if hasattr(item, "sigRegionChangeFinished"):
            item.sigRegionChangeFinished.connect(lambda _item=item: self._update_draft_from_item(_item))
        if self.draft is not None and self.draft.type == "angle" and hasattr(item, "sigRegionChanged"):
            item.sigRegionChanged.connect(lambda _item=item: self._report_angle_from_item(_item))
        elif hasattr(item, "sigRegionChanged"):          # live shape edit → status
            item.sigRegionChanged.connect(lambda _item=item: self._report_status_from_item(_item))

    def _update_draft_from_item(self, item) -> None:
        if self.draft is None:
            return
        if self.draft.type == "point":
            self.draft.geometry = self._point_geometry_from_item(item, self.draft.geometry)
            return
        if self.draft.type in {"line", "polyline", "freehand_line"}:
            self.draft.geometry = self._open_line_geometry_from_item(item, self.draft.geometry, self.draft.type)
            return
        self.draft.geometry = item_to_geometry(self.draft.type, item, prev=self.draft.geometry)
        if self.draft.type in {"rectangle", "oval", "polygon", "freehand"}:
            self._finalize_draft_selection(update_item=True)

    def _finalize_draft_selection(self, *, update_item: bool) -> None:
        if self.draft is None:
            return
        record = self._normalize_record(self.draft)
        self.draft = record
        if update_item and self.draft_item is not None:
            try:
                self.plot_item.removeItem(self.draft_item)
            except Exception:
                pass
            self.draft_item = self._make_item(record)
            self._style_item(self.draft_item, record, True)
            self._connect_draft_edit(self.draft_item)
            self.plot_item.addItem(self.draft_item)
        self._queue_selection_update(record)
        # One-shot: hand a freshly drawn rectangle to a pending requester.
        cb = self._rect_request
        if cb is not None and record.type == "rectangle":
            self._rect_request = None
            try:
                cb(record)
            except Exception:
                pass

    def request_rectangle(self, callback) -> None:
        """Activate the rectangle tool and call ``callback(record)`` once, with the
        next finished rectangle draft (Scale Bar 'At Selection' draw-then-place)."""
        self._rect_request = callback
        self.activate()
        self.store.set_tool("rectangle")

    def _finalize_record_selection(self, record: RoiRecord, *, update_item: bool) -> None:
        normalized = self._normalize_record(record)
        if normalized is not record:
            record.geometry = normalized.geometry
            record.context = normalized.context
        if update_item and record.id in self.items:
            self.refresh()
        self._queue_selection_update(record)

    def _normalize_record(self, record: RoiRecord) -> RoiRecord:
        normalizer = getattr(self.owner, "normalize_roi_record", None)
        if not callable(normalizer):
            return record
        try:
            normalized = normalizer(record)
        except Exception:
            return record
        return normalized if isinstance(normalized, RoiRecord) else record

    def _queue_selection_update(self, record: RoiRecord) -> None:
        if record.type not in {"rectangle", "oval", "polygon", "freehand"}:
            return
        record.selection_dirty = True
        self._pending_selection_record = record
        self._selection_timer.start()

    def _compute_pending_selection(self) -> None:
        record = self._pending_selection_record
        self._pending_selection_record = None
        if record is None:
            return
        compute = getattr(self.owner, "compute_roi_selection", None)
        if not callable(compute):
            return
        try:
            result = compute(record)
        except Exception as exc:
            log = getattr(getattr(self.owner, "_state", None), "log", None)
            if callable(log):
                log(f"ROI selection failed in {self.owner.windowTitle()}: {exc}", "WARN")
            return
        if result is None:
            return
        dataset, mask, context = result
        persisted_ids = {r.id for r in self.store.records}
        if record.id not in persisted_ids:
            self._replace_active_draft_mask(dataset, record.id, persisted_ids)
        store_roi_mask(dataset, record, mask, context=context)
        idx = context.get("dataset_idx") if isinstance(context, dict) else None
        notify = getattr(getattr(self.owner, "_state", None), "notify_roi_selection_changed", None)
        if callable(notify):
            notify(idx if isinstance(idx, int) else None)
        if record.id in persisted_ids:
            self.store.changed.emit()

    def _replace_active_draft_mask(self, dataset, roi_id: str, persisted_ids: set[str]) -> None:
        state_key = "active_roi_draft_id"
        old_id = dataset.state.get(state_key)
        if old_id and old_id != roi_id and old_id not in persisted_ids:
            masks = dataset.state.get(ROI_MASKS_STATE_KEY, {})
            meta = masks.pop(old_id, None)
            if meta is not None:
                key = meta.get("key")
                if key:
                    dataset.derived.pop(key, None)
        dataset.state[state_key] = roi_id

    def _record_at(self, pos: tuple[float, float] | None) -> tuple[str, RoiRecord] | None:
        if pos is None:
            return None
        tolerance = self._hit_tolerance()
        if self.draft is not None and self._point_hits_record(pos, self.draft, tolerance):
            return "draft", self.draft
        for record in reversed(self._session_points):
            if self._point_hits_record(pos, record, tolerance):
                return "session", record
        for record in reversed(self.store.records):
            if not record.visible:
                continue
            # Only hit-test ROIs that are actually on screen (same gate as refresh).
            if not self.store.show_all and record.id not in self.store.selected_ids:
                continue
            if self._point_hits_record(pos, record, tolerance):
                return "stored", record
        return None

    def _hit_tolerance(self) -> float:
        try:
            (x0, x1), (y0, y1) = self.view_box.viewRange()
            target = self._event_target()
            px_w = max(float(target.width()), 1.0)
            px_h = max(float(target.height()), 1.0)
            return max(abs(x1 - x0) / px_w, abs(y1 - y0) / px_h) * 10.0
        except Exception:
            return 1.0

    def _point_hits_record(self, pos: tuple[float, float], record: RoiRecord, tolerance: float) -> bool:
        if record.type == "point":
            px, py = self._project_point(record)
            return abs(pos[0] - px) <= tolerance and abs(pos[1] - py) <= tolerance
        if record.type not in {"rectangle", "oval"}:
            x, y, w, h = _bounds(record.geometry)
            return x - tolerance <= pos[0] <= x + w + tolerance and y - tolerance <= pos[1] <= y + h + tolerance
        # rectangle / oval: bounds is the unrotated box centred at the true centre,
        # so the rotated hit-test about that centre is correct for both.
        x, y, w, h = _bounds(record.geometry)
        cx = x + w / 2.0
        cy = y + h / 2.0
        half_w = max(w / 2.0, 0.0)
        half_h = max(h / 2.0, 0.0)
        angle = float(record.geometry.get("angle", 0.0) or 0.0)
        theta = np.deg2rad(angle)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        dx = float(pos[0]) - cx
        dy = float(pos[1]) - cy
        local_x = cos_t * dx + sin_t * dy
        local_y = -sin_t * dx + cos_t * dy
        if abs(local_x) <= half_w + tolerance and abs(local_y) <= half_h + tolerance:
            return True
        handle_offset = max(half_h * 0.22, tolerance * 2.0)
        hit_radius = max(tolerance * 3.0, min(half_w, half_h) * 0.18)
        for rotation_y in (-half_h - handle_offset, half_h + handle_offset):
            rot_x = cx + (-sin_t * rotation_y)
            rot_y = cy + (cos_t * rotation_y)
            if (float(pos[0]) - rot_x) ** 2 + (float(pos[1]) - rot_y) ** 2 <= hit_radius ** 2:
                return True
        return False

    def _show_roi_context_menu(self, hit: tuple[str, RoiRecord], global_pos,
                               view_pos: tuple[float, float] | None = None) -> None:
        from PyQt6.QtWidgets import QMenu

        from ..core.roi_convert import (
            REGION_TYPES,
            TARGET_LABELS,
            available_conversions,
            can_resize,
        )

        kind, record = hit
        menu = QMenu(self.view_widget)
        add_action = menu.addAction("Add to Manager")
        add_action.setEnabled(kind in {"draft", "session"})
        add_all_action = None
        if kind == "session":
            add_all_action = menu.addAction("Add all session points to Manager")
            add_all_action.setEnabled(len(self._session_points) > 0)
        # Vertex editing for polyline-family shapes: add a point on the nearest
        # edge, or delete the vertex nearest the right-click.
        add_point_action = delete_point_action = None
        if kind in {"draft", "stored"} and record.type in _VERTEX_EDIT_TYPES and view_pos is not None:
            menu.addSeparator()
            add_point_action = menu.addAction("Add point")
            delete_point_action = menu.addAction("Delete point")
            can_del, _vi = self._vertex_delete_candidate(record, view_pos)
            delete_point_action.setEnabled(can_del)
        menu.addSeparator()
        convert_actions = {}
        targets = available_conversions(record)
        if targets:
            convert_menu = menu.addMenu("Convert to")
            for tok in targets:
                act = convert_menu.addAction(TARGET_LABELS.get(tok, tok))
                convert_actions[act] = tok
        resize_action = menu.addAction("Enlarge / Shrink…")
        resize_action.setEnabled(can_resize(record))
        skeletonize_action = menu.addAction("Skeletonize")
        skeletonize_action.setEnabled(record.type in REGION_TYPES)
        menu.addSeparator()
        delete_action = menu.addAction("Delete")
        properties_action = menu.addAction("Properties...")
        chosen = menu.exec(global_pos)
        if chosen is None:
            return
        if chosen is add_action:
            self._add_hit_to_manager(kind, record)
        elif add_all_action is not None and chosen is add_all_action:
            self._add_all_session_points_to_manager()
        elif add_point_action is not None and chosen is add_point_action:
            self._add_vertex_hit(kind, record, view_pos)
        elif delete_point_action is not None and chosen is delete_point_action:
            self._delete_vertex_hit(kind, record, view_pos)
        elif chosen in convert_actions:
            self._convert_hit(kind, record, convert_actions[chosen])
        elif chosen is resize_action:
            self._resize_hit(kind, record)
        elif chosen is skeletonize_action:
            self._skeletonize_hit(kind, record)
        elif chosen is delete_action:
            self._delete_hit_roi(kind, record)
        elif chosen is properties_action:
            self._show_roi_properties(record)

    def replace_draft(self, new_record: RoiRecord) -> None:
        """Show *new_record* as the current editable draft. Convert / Enlarge /
        Skeletonize use this so a drawn-but-unsaved ROI changes type **in place**
        in the view — it is not filed into the Manager (nor is the Manager
        opened) until the user adds it."""
        self._set_draft(new_record)
        self._finalize_draft_selection(update_item=False)

    def _replace_hit(self, kind: str, record: RoiRecord, new_record: RoiRecord) -> None:
        """Apply a converted/resized record. A *stored* ROI is updated in place;
        a *draft* / *session* ROI stays in the view as the new type's draft — the
        Manager is neither opened nor written to until the user adds it."""
        store = self.store
        if kind == "stored":
            self._delete_mask_for_record(record)  # geometry changed → old mask is stale
            new_record.name = (
                f"{new_record.type}-{store.next_type_index(new_record.type)}"
                if store._is_auto_name(record) else record.name)
            store.update(record.id, new_record)
            store.select([record.id])
            return
        if kind == "session":
            self._remove_session_point(record.id)
        self.replace_draft(new_record)

    # ------------------------------------------------------- vertex editing
    def _is_closed(self, record: RoiRecord) -> bool:
        """Whether a polyline-family record forms a closed loop (mirrors _make_item)."""
        g = record.geometry
        if record.type in {"polyline", "freehand_line"}:
            return bool(g.get("closed", False))
        return bool(g.get("closed", record.type != "line"))

    def _vertex_delete_candidate(self, record: RoiRecord,
                                 view_pos: tuple[float, float]) -> tuple[bool, int]:
        """(can_delete, vertex_index): a vertex may be deleted when the right-click
        is within hit tolerance of it and the shape stays above its minimum count."""
        from ..core.roi_vertex_edit import min_vertices, nearest_vertex

        projected = self._project_points(record)
        pts = record.geometry.get("points", [])
        if len(projected) != len(pts) or len(projected) < 2:
            return False, -1
        idx, dist = nearest_vertex(projected, view_pos)
        if idx < 0:
            return False, -1
        if len(pts) <= min_vertices(self._is_closed(record)):
            return False, idx
        return dist <= self._hit_tolerance(), idx

    def _add_vertex_hit(self, kind: str, record: RoiRecord,
                        view_pos: tuple[float, float]) -> None:
        from ..core.roi_vertex_edit import insert_vertex, nearest_edge

        projected = self._project_points(record)
        pts = record.geometry.get("points", [])
        if len(projected) != len(pts) or len(pts) < 2:
            return
        seg, t = nearest_edge(projected, view_pos, closed=self._is_closed(record))
        if seg < 0:
            return
        new_pts = insert_vertex(pts, seg, t)
        self._commit_vertex_edit(kind, record, new_pts)

    def _delete_vertex_hit(self, kind: str, record: RoiRecord,
                           view_pos: tuple[float, float]) -> None:
        from ..core.roi_vertex_edit import delete_vertex

        can_del, idx = self._vertex_delete_candidate(record, view_pos)
        if not can_del:
            return
        new_pts = delete_vertex(record.geometry.get("points", []), idx)
        self._commit_vertex_edit(kind, record, new_pts)

    def _commit_vertex_edit(self, kind: str, record: RoiRecord,
                            new_points: list[list[float]]) -> None:
        """Apply an edited vertex list. Same type/name/colour as the source — a
        *stored* ROI is updated in place (like Convert/Resize); a *draft* stays an
        editable draft in the view."""
        from dataclasses import replace

        new_geom = {**record.geometry, "points": new_points}
        new_record = replace(record, geometry=new_geom)
        if kind == "stored":
            self._delete_mask_for_record(record)  # geometry changed → old mask is stale
            new_record.name = record.name          # vertex edit keeps type → keep name
            self.store.update(record.id, new_record)
            self.store.select([record.id])
            return
        self.replace_draft(new_record)

    def _roi_size_for_conversion(self, record: RoiRecord, target: str):
        """Prompt for the width/height a conversion needs; ``None`` = cancelled,
        ``()`` = no size required."""
        from .roi_convert_dialog import RoiSizeDialog

        if record.type == "point" and target in {"rectangle", "oval"}:
            dlg = RoiSizeDialog(self.view_widget, title=f"Point → {target}", need_height=True)
            if dlg.exec() != dlg.DialogCode.Accepted:
                return None
            w, h = dlg.values()
            return {"width": w, "height": h}
        if target == "region":
            dlg = RoiSizeDialog(self.view_widget, title="Line → region", need_height=False,
                                width_label="Width")
            if dlg.exec() != dlg.DialogCode.Accepted:
                return None
            w, _ = dlg.values()
            return {"width": w}
        return {}

    def _convert_hit(self, kind: str, record: RoiRecord, target: str) -> None:
        from PyQt6.QtWidgets import QMessageBox

        from ..core.roi_convert import convert_roi

        size = self._roi_size_for_conversion(record, target)
        if size is None:
            return
        try:
            new_record = convert_roi(record, target, **size)
        except Exception as exc:
            QMessageBox.warning(self.view_widget, "Convert ROI", str(exc))
            return
        self._replace_hit(kind, record, new_record)

    def _resize_hit(self, kind: str, record: RoiRecord) -> None:
        from PyQt6.QtWidgets import QMessageBox

        from ..core.roi_convert import LINE_TYPES, can_resize, enlarge_shrink_roi
        from .roi_convert_dialog import RoiResizeDialog

        if not can_resize(record):
            return
        allow_shrink = record.type not in ({"point"} | LINE_TYPES)
        dlg = RoiResizeDialog(self.view_widget, allow_shrink=allow_shrink)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        mode, value = dlg.values()
        try:
            new_record = enlarge_shrink_roi(record, value, mode=mode)
        except Exception as exc:
            QMessageBox.warning(self.view_widget, "Enlarge / Shrink ROI", str(exc))
            return
        self._replace_hit(kind, record, new_record)

    def _skeletonize_hit(self, kind: str, record: RoiRecord) -> None:
        from PyQt6.QtWidgets import QMessageBox

        from ..core.roi_convert import skeletonize_roi

        try:
            new_record = skeletonize_roi(record)
        except Exception as exc:
            QMessageBox.warning(self.view_widget, "Skeletonize ROI", str(exc))
            return
        self._replace_hit(kind, record, new_record)

    def _add_hit_to_manager(self, kind: str, record: RoiRecord) -> None:
        if kind == "session":
            self._remove_session_point(record.id)
            self._show_manager_if_needed()
            self.store.add(record)
            self.store.deselect()        # leave the viewer; live only in the Manager
            return
        if kind != "draft" or self.draft is None:
            return
        draft_item = self.draft_item
        self.draft = None
        self.draft_item = None
        if draft_item is not None:
            try:
                self.plot_item.removeItem(draft_item)
            except Exception:
                pass
        self._show_manager_if_needed()
        self.store.add(record)
        self.store.deselect()            # the drawn ROI disappears from the view
        # The draft is now a stored, deselected ROI: drop the dataset's "active
        # draft" marker so its persistent in-ROI highlight clears with it (the
        # stored mask is kept, so re-selecting the ROI re-highlights).
        self._demote_active_draft_marker(record)

    def _demote_active_draft_marker(self, record: RoiRecord) -> None:
        """Clear the ``active_roi_draft_id`` marker for *record* (a draft that has
        become a stored ROI) on every dataset that carries it, keeping the stored
        selection mask, and notify so render/scatter re-draw their highlights —
        otherwise the active-draft highlight lingers after the ROI is filed away
        (`_roi_masks_for_dataset` highlights the active draft regardless of
        selection)."""
        state = getattr(self.owner, "_state", None)
        datasets = getattr(state, "datasets", []) if state is not None else []
        notify = getattr(state, "notify_roi_selection_changed", None)
        for idx, ds in enumerate(datasets):
            if ds.state.get("active_roi_draft_id") == record.id:
                ds.state.pop("active_roi_draft_id", None)
                if callable(notify):
                    notify(idx)

    def _delete_hit_roi(self, kind: str, record: RoiRecord) -> None:
        self._clear_status()                         # deleted ROI → clear its read-out
        self._delete_mask_for_record(record)
        if kind == "draft":
            self._clear_draft()
            return
        if kind == "session":
            self._remove_session_point(record.id)
            return
        self.store.selected_ids = [record.id]
        self.store.delete_selected()

    def _delete_mask_for_record(self, record: RoiRecord) -> None:
        dataset = self._dataset_for_record(record)
        if dataset is None:
            return
        masks = dataset.state.get(ROI_MASKS_STATE_KEY, {})
        meta = masks.pop(record.id, None)
        key = record.mask_key or (meta or {}).get("key")
        if key:
            dataset.derived.pop(key, None)
        if dataset.state.get("active_roi_draft_id") == record.id:
            dataset.state.pop("active_roi_draft_id", None)
        idx = record.context.get("dataset_idx") if isinstance(record.context, dict) else None
        notify = getattr(getattr(self.owner, "_state", None), "notify_roi_selection_changed", None)
        if callable(notify):
            notify(idx if isinstance(idx, int) else None)

    def _dataset_for_record(self, record: RoiRecord):
        idx = record.context.get("dataset_idx") if isinstance(record.context, dict) else None
        datasets = getattr(getattr(self.owner, "_state", None), "datasets", [])
        if isinstance(idx, int) and 0 <= idx < len(datasets):
            return datasets[idx]
        return getattr(getattr(self.owner, "_state", None), "active_dataset", None)

    _REGION_TYPES = {"rectangle", "oval", "polygon", "freehand"}

    def _show_roi_properties(self, record: RoiRecord) -> None:
        import html

        from PyQt6.QtWidgets import QMessageBox

        if record.type in self._REGION_TYPES and record.selection_dirty:
            self._pending_selection_record = record
            self._compute_pending_selection()
        name = record.name or f"{record.type}-{self.store.next_type_index(record.type)}"
        lines = [
            f"Name: {name}",
            f"Type: {record.type}",
            f"Geometry: {self._geometry_text(record)}",
        ]
        if record.type in self._REGION_TYPES:
            count = "pending" if record.selected_count is None else f"{record.selected_count:,}"
            lines.append(f"Localizations within: {count}")
        # Monospace so the polygon vertex columns line up.
        box = QMessageBox(self.view_widget)
        box.setWindowTitle("ROI Properties")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText("<pre style='font-family:monospace; margin:0'>"
                    + html.escape("\n".join(lines)) + "</pre>")
        box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse
                                    | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        box.exec()

    @staticmethod
    def _fmt(value) -> str:
        """Integer nanometre precision (e.g. 12000) — easiest to estimate ROI
        sizes at the nm scale."""
        v = float(value)
        return f"{int(round(v))}" if np.isfinite(v) else ""

    def _geometry_text(self, record: RoiRecord) -> str:
        f = self._fmt
        g = record.geometry
        t = record.type
        if t in {"rectangle", "oval"}:
            x, y, w, h = _bounds(g)
            text = f"X={f(x)}, Y={f(y)}, W={f(w)}, H={f(h)}"
            angle = normalize_angle_deg(float(g.get("angle", 0.0) or 0.0))  # (-180, 180]
            if abs(angle) > 1e-9:
                text += f", angle={angle:.1f}°"
            return text
        if t == "point":
            pt = g.get("point", [0.0, 0.0])
            return "(" + ", ".join(f(c) for c in pt[:3]) + ")"
        pts = g.get("points", [])
        if t in {"freehand", "freehand_line"} and len(pts) > 24:
            x, y, w, h = _bounds(g)
            return f"{len(pts)} vertices (freehand)\nbbox X={f(x)}, Y={f(y)}, W={f(w)}, H={f(h)}"
        noun = "vertices" if t in {"polygon", "freehand", "polyline", "freehand_line"} else "points"
        out = [f"{len(pts)} {noun}:"]
        out += [f"  {f(p[0]):>9}  {f(p[1]):>9}" for p in pts]
        if t == "angle" and len(pts) >= 3:
            try:
                out.append(f"angle = {_angle_abc(pts[0], pts[1], pts[2]):.1f}°")
            except Exception:
                pass
        return "\n".join(out)

    def _sync_label(self, record: RoiRecord) -> None:
        if not self.store.show_labels or not record.label_visible:
            label = self.labels.pop(record.id, None)
            if label is not None:
                self.plot_item.removeItem(label)
            return
        text = self._label_text(record)
        label = self.labels.get(record.id)
        if label is None:
            label = pg.TextItem(text, color=record.stroke_color, anchor=(0, 1))
            self.labels[record.id] = label
            self.plot_item.addItem(label)
        label.setText(text)
        label.setColor(record.stroke_color)
        if record.type == "point":
            x, y = self._project_point(record)   # label at the projected marker
            label.setPos(x, y)
        elif record.type in {"line", "polyline", "freehand_line", "magnetic_lasso", "angle"}:
            # Line-family ROIs: label at the vertex centroid, not a bbox corner.
            pts = np.asarray(self._project_points(record), dtype=float)
            if pts.size:
                label.setPos(float(pts[:, 0].mean()), float(pts[:, 1].mean()))
            else:
                x, y, w, h = _bounds(record.geometry)
                label.setPos(x + w, y)
        else:
            x, y, w, h = _bounds(record.geometry)
            label.setPos(x + w, y)

    def _label_text(self, record: RoiRecord) -> str:
        total = max(len(self.store.records), 1)
        width = max(2, len(str(total)))
        for idx, candidate in enumerate(self.store.records, start=1):
            if candidate.id == record.id:
                return f"{idx:0{width}d}"
        return record.name


def item_to_geometry(roi_type: str, item, prev: dict[str, Any] | None = None) -> dict[str, Any]:
    if roi_type == "point":
        pos = item.pos()                     # PointMarkerItem (TargetItem): centre
        return {"point": [float(pos.x()), float(pos.y())]}
    if roi_type in {"rectangle", "oval"}:
        size = item.size()
        w, h = float(size.x()), float(size.y())
        angle = float(item.angle()) if hasattr(item, "angle") else 0.0
        # Store the *unrotated* box whose centre is the ROI's true (rotated)
        # centre, so masks/hit-tests (which rotate about that centre) are correct.
        centre = item.mapToParent(pg.Point(w / 2.0, h / 2.0))
        cx, cy = float(centre.x()), float(centre.y())
        geometry = {"bounds": [cx - w / 2.0, cy - h / 2.0, w, h]}
        # Normalize to (-180, 180]: a pyqtgraph rotate handle accumulates the raw
        # angle across full turns (an ellipse spun several times reported -7140°).
        norm = normalize_angle_deg(angle)
        if abs(norm) > 1e-9:
            geometry["angle"] = norm
        # Preserve the rotated-variant marker across a re-read (it drives the
        # 4-handle layout; item geometry alone can't distinguish it from a plain
        # rotated rectangle/oval).
        if prev and prev.get("variant"):
            geometry["variant"] = prev["variant"]
        return geometry
    # Multi-vertex shapes (line / polyline / freehand line / polygon / freehand /
    # angle): read each handle in *scene* coords and map to parent, so a body
    # move (which leaves the local handle coords unchanged) is also captured.
    pts: list[list[float]] = []
    if hasattr(item, "getSceneHandlePositions"):
        for entry in item.getSceneHandlePositions():
            scene_pt = entry[1] if isinstance(entry, (tuple, list)) else entry
            view = item.mapSceneToParent(scene_pt)
            pts.append([float(view.x()), float(view.y())])
    elif hasattr(item, "getState"):
        pts = [[float(p[0]), float(p[1])] for p in item.getState().get("points", [])]
    if roi_type == "line":
        return {"points": pts[:2], "closed": False}
    # polygon/freehand are always closed; a polyline keeps its current closed
    # state (a region→Line outline is a closed polyline).
    closed = roi_type in {"polygon", "freehand"} or bool(getattr(item, "closed", False))
    return {"points": pts, "closed": closed}


def _bounds(geometry: dict[str, Any]) -> tuple[float, float, float, float]:
    if "bounds" in geometry:
        x, y, w, h = geometry["bounds"]
        return float(x), float(y), float(w), float(h)
    if "point" in geometry:
        pt = geometry["point"]                       # may carry a 3rd (depth) coord
        return float(pt[0]), float(pt[1]), 0.0, 0.0
    raw = geometry.get("points", [])
    if not len(raw):
        return 0.0, 0.0, 0.0, 0.0
    # Take (x, y) from each vertex explicitly — robust to a mix of 2-D / 3-D
    # points (a raw ``np.asarray`` on an inhomogeneous list raises ValueError).
    pts = np.array([[float(p[0]), float(p[1])] for p in raw], dtype=float)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)
