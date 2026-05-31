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
        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.RightButton:
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
            if tool == "rectangle":
                self._finalize_draft_selection(update_item=True)
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
        self._connect_draft_edit(self.draft_item)
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
            return pg.EllipseROI([x, y], [max(w, 1e-9), max(h, 1e-9)], movable=True)
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
        if record.type in {"rectangle", "oval", "polygon", "freehand"}:
            try:
                fill = pg.mkColor(color)
                fill.setAlpha(32 if selected else 22)
                if isinstance(item, FilledRectROI):
                    item.setFillColor(fill, alpha=fill.alpha())
                elif hasattr(item, "setBrush"):
                    item.setBrush(pg.mkBrush(fill))
            except Exception:
                pass

    def _connect_edit(self, item, record: RoiRecord) -> None:
        if hasattr(item, "sigRegionChangeFinished"):
            item.sigRegionChangeFinished.connect(lambda _item=item, roi_id=record.id: self._update_record_from_item(roi_id, _item))

    def _connect_draft_edit(self, item) -> None:
        if hasattr(item, "sigRegionChangeFinished"):
            item.sigRegionChangeFinished.connect(lambda _item=item: self._update_draft_from_item(_item))

    def _update_record_from_item(self, roi_id: str, item) -> None:
        for record in self.store.records:
            if record.id != roi_id:
                continue
            record.geometry = item_to_geometry(record.type, item)
            self._finalize_record_selection(record, update_item=True)
            self.store.changed.emit()
            return

    def _update_draft_from_item(self, item) -> None:
        if self.draft is None:
            return
        self.draft.geometry = item_to_geometry(self.draft.type, item)
        if self.draft.type == "rectangle":
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
        if record.type != "rectangle":
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
        for record in reversed(self.store.records):
            if not record.visible:
                continue
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
        add_action.setEnabled(kind == "draft")
        delete_action = menu.addAction("Delete")
        properties_action = menu.addAction("Properties...")
        chosen = menu.exec(global_pos)
        if chosen is add_action:
            self._add_hit_to_manager(kind, record)
        elif chosen is delete_action:
            self._delete_hit_roi(kind, record)
        elif chosen is properties_action:
            self._show_roi_properties(record)

    def _add_hit_to_manager(self, kind: str, record: RoiRecord) -> None:
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
    if roi_type in {"rectangle", "oval", "point"}:
        pos = item.pos()
        size = item.size()
        x, y, w, h = float(pos.x()), float(pos.y()), float(size.x()), float(size.y())
        if roi_type == "point":
            return {"point": [x + w / 2.0, y + h / 2.0]}
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
        x, y = geometry["point"]
        return float(x), float(y), 0.0, 0.0
    pts = np.asarray(geometry.get("points", []), dtype=float)
    if pts.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)
