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
from ..core.roi_selection import ROI_MASKS_STATE_KEY, rectangle_bounds, store_roi_mask

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


class FilledRectROI(pg.ROI):
    """Rectangle ROI with a visible translucent face and edit handles."""

    def __init__(self, pos, size, *, angle: float = 0.0, fill_color="#ffff00", **kwargs) -> None:
        super().__init__(pos, size, angle=angle, **kwargs)
        self._fill_color = pg.mkColor(fill_color)
        self._fill_color.setAlpha(32)
        center = [0.5, 0.5]
        for pos, opposite in (
            ([0.0, 0.5], [1.0, 0.5]),
            ([1.0, 0.5], [0.0, 0.5]),
            ([0.5, 0.0], [0.5, 1.0]),
            ([0.5, 1.0], [0.5, 0.0]),
            ([0.0, 0.0], [1.0, 1.0]),
            ([1.0, 0.0], [0.0, 1.0]),
            ([1.0, 1.0], [0.0, 0.0]),
            ([0.0, 1.0], [1.0, 0.0]),
        ):
            self.addScaleHandle(pos, opposite)
        rotate_handle = FilledRotateHandle(
            float(self.handleSize),
            typ="r",
            pen=self.handlePen,
            hoverPen=self.handleHoverPen,
            parent=self,
            antialias=self._antialias,
        )
        # In the displayed plot coordinates this places the handle above the ROI.
        self.addRotateHandle([0.5, 1.22], center, item=rotate_handle, name="rotate")

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
    """Ellipse ROI with a visible translucent face."""

    def __init__(self, pos, size, *, fill_color="#ffff00", **kwargs) -> None:
        super().__init__(pos, size, **kwargs)
        self._fill_color = pg.mkColor(fill_color)
        self._fill_color.setAlpha(32)

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
        self._angle_points: list[list[float]] = []
        self._pending_selection_record: RoiRecord | None = None
        self._selection_timer = QTimer(self)
        self._selection_timer.setSingleShot(True)
        self._selection_timer.setInterval(120)
        self._selection_timer.timeout.connect(self._compute_pending_selection)

        target = self._event_target()
        target.installEventFilter(self)
        store.changed.connect(self.refresh)
        store.selection_changed.connect(self.refresh)
        self.refresh()

    def activate(self) -> None:
        self.store.set_active_adapter(self)

    def current_record(self) -> RoiRecord | None:
        return self.draft

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
        else:
            record.geometry = item_to_geometry(record.type, item)
        record.selection_dirty = True
        record = self._normalize_record(record)
        self._queue_selection_update(record)
        return record

    def refresh(self) -> None:
        wanted = set()
        selected = set(self.store.selected_ids)
        for record in self.store.records:
            if not record.visible:
                continue
            # Point ROIs are persistent spatial markers — keep them all visible
            # regardless of the show-all / selection gate (multi-point drawing).
            if not self.store.show_all and record.id not in selected and record.type != "point":
                continue
            wanted.add(record.id)
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
            self._style_item(self.items[record.id], record, record.id in selected)
            self._sync_label(record)
        for roi_id in list(self.items):
            if roi_id not in wanted:
                self.plot_item.removeItem(self.items.pop(roi_id))
        for roi_id in list(self.labels):
            if roi_id not in wanted:
                self.plot_item.removeItem(self.labels.pop(roi_id))
        self._refresh_session_points()

    def eventFilter(self, obj, event) -> bool:
        try:
            target = self._event_target()
        except RuntimeError:
            return False
        if obj is not target:
            return False
        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.RightButton:
            # Fiji-style: a right-click finishes an in-progress polygon. The click
            # location is only a signal (not a vertex); the loop closes from the
            # last left-click point back to the first.
            if self.store.active_tool == "polygon" and self._polygon_points:
                self._finish_polygon()
                return True
            pos = self._event_to_view(event)
            hit = self._record_at(pos) if pos is not None else None
            if hit is not None:
                self.activate()
                self._show_roi_context_menu(hit, event.globalPosition().toPoint())
                return True
            return False
        if event.type() == QEvent.Type.FocusIn:
            self.activate()
            return False
        tool = self.store.active_tool
        if not tool:
            return False
        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self.activate()
            pos = self._event_to_view(event)
            if pos is None:
                return False
            if self._record_at(pos) is not None:
                return False
            if tool == "point":
                # Each click drops a persistent point; the tool stays pressed so
                # the user can keep marking points until the toolbar button is
                # clicked again (no auto-release).
                self._commit_point(pos)
                return True
            if tool == "polygon":
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    self._finish_polygon()
                else:
                    self._polygon_points.append(list(pos))
                    self._update_polygon_draft()
                return True
            if tool == "angle":
                # Click A, then B (vertex), then C — three points finish it.
                self._angle_points.append(list(pos))
                if len(self._angle_points) >= 3:
                    self._finish_angle()
                else:
                    self._update_angle_draft()
                return True
            self._drag_start = pos
            self._freehand_points = [list(pos)]
            self._update_drag_draft(pos)
            return True
        if event.type() == QEvent.Type.MouseMove and self._drag_start is not None:
            pos = self._event_to_view(event)
            if pos is None:
                return False
            if tool == "freehand":
                self._freehand_points.append(list(pos))
            self._update_drag_draft(pos)
            return True
        if event.type() == QEvent.Type.MouseButtonRelease and self._drag_start is not None:
            pos = self._event_to_view(event)
            if pos is not None:
                self._update_drag_draft(pos)
            self._drag_start = None
            if tool in {"rectangle", "oval", "freehand"}:
                self._finalize_draft_selection(update_item=True)
            self._maybe_release_tool(event.modifiers())
            return True
        if event.type() == QEvent.Type.MouseButtonDblClick and tool == "polygon":
            self._finish_polygon()
            return True
        if event.type() == QEvent.Type.KeyPress and tool == "polygon":
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._finish_polygon()
                return True
            if event.key() == Qt.Key.Key_Escape:
                self._polygon_points = []
                self._clear_draft()
                return True
        return False

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

    # ------------------------------------------------------------------
    # Session points (pending markers, not yet in the Manager/store)
    # ------------------------------------------------------------------

    def _add_session_point(self, record: RoiRecord) -> None:
        self._session_points.append(record)
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
        self._show_manager_if_needed()
        for rec in records:
            self.store.add(rec)

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

    def _point_to_3d(self, pos) -> list[float]:
        """Build a full XYZ coordinate for a point clicked at in-plane *pos*."""
        return point_to_3d(pos, self._view_plane(), self._depth_center())

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
        elif tool == "line":
            self._set_draft(RoiRecord.create("line", {"points": [[x0, y0], [x1, y1]], "closed": False}, **self._record_kwargs()))
        elif tool == "freehand":
            self._set_draft(RoiRecord.create("freehand", {"points": self._freehand_points, "closed": True}, **self._record_kwargs()))

    def _update_polygon_draft(self) -> None:
        if len(self._polygon_points) >= 1:
            self._set_draft(RoiRecord.create("polygon", {"points": self._polygon_points, "closed": False}, **self._record_kwargs()))

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

    def _set_draft(self, record: RoiRecord) -> None:
        self.draft = record
        if self.draft_item is not None:
            self.plot_item.removeItem(self.draft_item)
        self.draft_item = self._make_item(record)
        self._style_item(self.draft_item, record, True)
        self._connect_draft_edit(self.draft_item)
        self.plot_item.addItem(self.draft_item)

    def _clear_draft(self) -> None:
        self.draft = None
        self._polygon_points = []
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
            if record.type == "rectangle":
                return FilledRectROI(
                    [x, y],
                    [max(w, 1e-9), max(h, 1e-9)],
                    angle=float(g.get("angle", 0.0) or 0.0),
                    fill_color=record.stroke_color,
                    movable=True,
                )
            return FilledEllipseROI(
                [x, y], [max(w, 1e-9), max(h, 1e-9)],
                fill_color=record.stroke_color, movable=True,
            )
        if record.type == "line":
            pts = g.get("points", [[0, 0], [1, 1]])
            return pg.LineSegmentROI(pts[:2], movable=True)
        if record.type == "point":
            x, y = self._project_point(record)
            return PointMarkerItem((x, y))
        if record.type == "angle":
            pts = g.get("points", [[0, 0], [1, 0], [2, 1]])
            return pg.PolyLineROI(pts[:3], closed=False, movable=True)
        pts = g.get("points", [])
        if len(pts) < 2:
            pts = [[0, 0], [1, 1]]
        return FilledPolyLineROI(
            pts, closed=bool(g.get("closed", record.type != "line")),
            fill_color=record.stroke_color, movable=True,
        )

    def _style_item(self, item, record: RoiRecord, selected: bool) -> None:
        color = record.stroke_color or "#ffff00"
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
        if record.type == "point" and hasattr(item, "sigPositionChangeFinished"):
            item.sigPositionChangeFinished.connect(lambda _item=item, roi_id=record.id: self._update_record_from_item(roi_id, _item))
            return
        if hasattr(item, "sigRegionChangeFinished"):
            item.sigRegionChangeFinished.connect(lambda _item=item, roi_id=record.id: self._update_record_from_item(roi_id, _item))
        if record.type == "angle" and hasattr(item, "sigRegionChanged"):
            item.sigRegionChanged.connect(lambda _item=item: self._report_angle_from_item(_item))

    def _connect_draft_edit(self, item) -> None:
        if self.draft is not None and self.draft.type == "point" and hasattr(item, "sigPositionChangeFinished"):
            item.sigPositionChangeFinished.connect(lambda _item=item: self._update_draft_from_item(_item))
            return
        if hasattr(item, "sigRegionChangeFinished"):
            item.sigRegionChangeFinished.connect(lambda _item=item: self._update_draft_from_item(_item))
        if self.draft is not None and self.draft.type == "angle" and hasattr(item, "sigRegionChanged"):
            item.sigRegionChanged.connect(lambda _item=item: self._report_angle_from_item(_item))

    def _update_record_from_item(self, roi_id: str, item) -> None:
        for record in self.store.records:
            if record.id != roi_id:
                continue
            if record.type == "point":
                record.geometry = self._point_geometry_from_item(item, record.geometry)
            else:
                record.geometry = item_to_geometry(record.type, item)
            self._finalize_record_selection(record, update_item=True)
            self.store.changed.emit()
            return

    def _update_draft_from_item(self, item) -> None:
        if self.draft is None:
            return
        if self.draft.type == "point":
            self.draft.geometry = self._point_geometry_from_item(item, self.draft.geometry)
            return
        self.draft.geometry = item_to_geometry(self.draft.type, item)
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
            # Points are always on-screen, so they are always hit-testable
            # (lets a click on an existing point drag it instead of stacking).
            if not self.store.show_all and record.id not in self.store.selected_ids and record.type != "point":
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
        if record.type != "rectangle":
            x, y, w, h = _bounds(record.geometry)
            return x - tolerance <= pos[0] <= x + w + tolerance and y - tolerance <= pos[1] <= y + h + tolerance
        bounds = rectangle_bounds(record)
        if bounds is None:
            return False
        x0, x1, y0, y1 = bounds
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        half_w = max(0.5 * (x1 - x0), 0.0)
        half_h = max(0.5 * (y1 - y0), 0.0)
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

    def _show_roi_context_menu(self, hit: tuple[str, RoiRecord], global_pos) -> None:
        from PyQt6.QtWidgets import QMenu

        kind, record = hit
        menu = QMenu(self.view_widget)
        add_action = menu.addAction("Add to Manager")
        add_action.setEnabled(kind in {"draft", "session"})
        add_all_action = None
        if kind == "session":
            add_all_action = menu.addAction("Add all session points to Manager")
            add_all_action.setEnabled(len(self._session_points) > 0)
        delete_action = menu.addAction("Delete")
        properties_action = menu.addAction("Properties...")
        chosen = menu.exec(global_pos)
        if chosen is add_action:
            self._add_hit_to_manager(kind, record)
        elif add_all_action is not None and chosen is add_all_action:
            self._add_all_session_points_to_manager()
        elif chosen is delete_action:
            self._delete_hit_roi(kind, record)
        elif chosen is properties_action:
            self._show_roi_properties(record)

    def _add_hit_to_manager(self, kind: str, record: RoiRecord) -> None:
        if kind == "session":
            self._remove_session_point(record.id)
            self._show_manager_if_needed()
            self.store.add(record)
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

    def _delete_hit_roi(self, kind: str, record: RoiRecord) -> None:
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

    def _show_roi_properties(self, record: RoiRecord) -> None:
        from PyQt6.QtWidgets import QMessageBox

        if record.selection_dirty:
            self._pending_selection_record = record
            self._compute_pending_selection()
        geometry = self._geometry_text(record)
        count = "pending" if record.selected_count is None else f"{record.selected_count:,}"
        text = (
            f"Name: {record.name}\n"
            f"Type: {record.type}\n"
            f"Geometry: {geometry}\n"
            f"Localizations within: {count}"
        )
        QMessageBox.information(self.view_widget, "ROI Properties", text)

    def _geometry_text(self, record: RoiRecord) -> str:
        if record.type == "rectangle":
            x, y, w, h = _bounds(record.geometry)
            angle = float(record.geometry.get("angle", 0.0) or 0.0)
            return f"box ({x:.3g}, {y:.3g}, {x + w:.3g}, {y + h:.3g}), angle {angle:.3g} deg"
        return str(record.geometry)

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
            w = h = 0.0
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


def item_to_geometry(roi_type: str, item) -> dict[str, Any]:
    if roi_type == "point":
        pos = item.pos()                     # PointMarkerItem (TargetItem): centre
        return {"point": [float(pos.x()), float(pos.y())]}
    if roi_type in {"rectangle", "oval"}:
        pos = item.pos()
        size = item.size()
        x, y, w, h = float(pos.x()), float(pos.y()), float(size.x()), float(size.y())
        geometry = {"bounds": [x, y, w, h]}
        if roi_type == "rectangle" and hasattr(item, "angle"):
            geometry["angle"] = float(item.angle())
        return geometry
    if roi_type == "line":
        pts = []
        for _handle, point in item.getSceneHandlePositions():
            view = item.mapSceneToParent(point)
            pts.append([float(view.x()), float(view.y())])
        return {"points": pts[:2], "closed": False}
    if hasattr(item, "getState"):
        state = item.getState()
        points = [[float(p[0]), float(p[1])] for p in state.get("points", [])]
        return {"points": points, "closed": bool(state.get("closed", roi_type != "line"))}
    return {"points": [], "closed": roi_type in {"polygon", "freehand"}}


def _bounds(geometry: dict[str, Any]) -> tuple[float, float, float, float]:
    if "bounds" in geometry:
        x, y, w, h = geometry["bounds"]
        return float(x), float(y), float(w), float(h)
    if "point" in geometry:
        pt = geometry["point"]                       # may carry a 3rd (depth) coord
        return float(pt[0]), float(pt[1]), 0.0, 0.0
    pts = np.asarray(geometry.get("points", []), dtype=float)
    if pts.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)
