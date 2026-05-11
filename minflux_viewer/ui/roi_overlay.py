"""pyqtgraph ROI drawing and overlay adapter."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, QEvent, Qt

from ..core.roi import RoiRecord, RoiStore


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
        self._drag_start: tuple[float, float] | None = None
        self._freehand_points: list[list[float]] = []
        self._polygon_points: list[list[float]] = []

        target = self._event_target()
        target.installEventFilter(self)
        store.changed.connect(self.refresh)
        store.selection_changed.connect(self.refresh)
        store.tool_changed.connect(lambda _tool: self._clear_draft())
        self.refresh()

    def activate(self) -> None:
        self.store.set_active_adapter(self)

    def current_record(self) -> RoiRecord | None:
        return self.draft

    def consume_draft(self) -> RoiRecord | None:
        record = self.draft
        self._clear_draft()
        return record

    def replace_selected_from_draft(self) -> RoiRecord | None:
        return self.consume_draft()

    def refresh(self) -> None:
        wanted = set()
        selected = set(self.store.selected_ids)
        for record in self.store.records:
            if not record.visible:
                continue
            if not self.store.show_all and record.id not in selected:
                continue
            wanted.add(record.id)
            if record.id not in self.items:
                item = self._make_item(record)
                self.items[record.id] = item
                self.plot_item.addItem(item)
                self._connect_edit(item, record)
            self._style_item(self.items[record.id], record, record.id in selected)
            self._sync_label(record)
        for roi_id in list(self.items):
            if roi_id not in wanted:
                self.plot_item.removeItem(self.items.pop(roi_id))
        for roi_id in list(self.labels):
            if roi_id not in wanted:
                self.plot_item.removeItem(self.labels.pop(roi_id))

    def eventFilter(self, obj, event) -> bool:
        try:
            target = self._event_target()
        except RuntimeError:
            return False
        if obj is not target:
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
            if tool == "point":
                self._set_draft(RoiRecord.create("point", {"point": list(pos)}, **self._record_kwargs()))
                return True
            if tool == "polygon":
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    self._finish_polygon()
                else:
                    self._polygon_points.append(list(pos))
                    self._update_polygon_draft()
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
        if len(self._polygon_points) >= 3:
            self._set_draft(RoiRecord.create("polygon", {"points": self._polygon_points, "closed": True}, **self._record_kwargs()))
        self._polygon_points = []

    def _set_draft(self, record: RoiRecord) -> None:
        self.draft = record
        if self.draft_item is not None:
            self.plot_item.removeItem(self.draft_item)
        self.draft_item = self._make_item(record)
        self._style_item(self.draft_item, record, True)
        self.plot_item.addItem(self.draft_item)

    def _clear_draft(self) -> None:
        self.draft = None
        self._polygon_points = []
        self._freehand_points = []
        if self.draft_item is not None:
            try:
                self.plot_item.removeItem(self.draft_item)
            except Exception:
                pass
        self.draft_item = None

    def _make_item(self, record: RoiRecord):
        g = record.geometry
        if record.type in {"rectangle", "oval"}:
            x, y, w, h = _bounds(g)
            cls = pg.EllipseROI if record.type == "oval" else pg.RectROI
            return cls([x, y], [max(w, 1e-9), max(h, 1e-9)], movable=True)
        if record.type == "line":
            pts = g.get("points", [[0, 0], [1, 1]])
            return pg.LineSegmentROI(pts[:2], movable=True)
        if record.type == "point":
            x, y = g.get("point", [0.0, 0.0])
            return pg.CircleROI([x - 2.0, y - 2.0], [4.0, 4.0], movable=True)
        pts = g.get("points", [])
        if len(pts) < 2:
            pts = [[0, 0], [1, 1]]
        return pg.PolyLineROI(pts, closed=bool(g.get("closed", record.type != "line")), movable=True)

    def _style_item(self, item, record: RoiRecord, selected: bool) -> None:
        color = record.stroke_color or "#ffff00"
        width = record.line_width + (1.5 if selected else 0.0)
        pen = pg.mkPen(color, width=width)
        try:
            item.setPen(pen)
        except Exception:
            pass

    def _connect_edit(self, item, record: RoiRecord) -> None:
        if hasattr(item, "sigRegionChangeFinished"):
            item.sigRegionChangeFinished.connect(lambda _item=item, roi_id=record.id: self._update_record_from_item(roi_id, _item))

    def _update_record_from_item(self, roi_id: str, item) -> None:
        for record in self.store.records:
            if record.id != roi_id:
                continue
            record.geometry = item_to_geometry(record.type, item)
            self.store.changed.emit()
            return

    def _sync_label(self, record: RoiRecord) -> None:
        if not self.store.show_labels or not record.label_visible:
            label = self.labels.pop(record.id, None)
            if label is not None:
                self.plot_item.removeItem(label)
            return
        label = self.labels.get(record.id)
        if label is None:
            label = pg.TextItem(record.name, color=record.stroke_color, anchor=(0, 1))
            self.labels[record.id] = label
            self.plot_item.addItem(label)
        label.setText(record.name)
        label.setColor(record.stroke_color)
        x, y, w, h = _bounds(record.geometry)
        label.setPos(x + w, y)


def item_to_geometry(roi_type: str, item) -> dict[str, Any]:
    if roi_type in {"rectangle", "oval", "point"}:
        pos = item.pos()
        size = item.size()
        x, y, w, h = float(pos.x()), float(pos.y()), float(size.x()), float(size.y())
        if roi_type == "point":
            return {"point": [x + w / 2.0, y + h / 2.0]}
        return {"bounds": [x, y, w, h]}
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
        x, y = geometry["point"]
        return float(x), float(y), 0.0, 0.0
    pts = np.asarray(geometry.get("points", []), dtype=float)
    if pts.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)
