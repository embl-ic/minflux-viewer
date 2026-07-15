"""
Shared ROI model and store.

This module keeps ROI state independent from any one viewer window.  Windows
attach lightweight canvas adapters that render and edit the records.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, QPointF, pyqtSignal
from PyQt6.QtGui import QPainterPath, QPolygonF

ROI_TYPES = {
    "rectangle", "oval", "polygon", "freehand", "line", "point", "angle",
    "polyline", "freehand_line", "magnetic_lasso",
}

#: Drawing-tool names accepted by ``set_tool``. Superset of ``ROI_TYPES`` with the
#: variant tools that produce an existing record type (a rotated rectangle is a
#: ``rectangle`` record + ``geometry["variant"]``; an ellipse is a rotated ``oval``),
#: so they are valid *tools* but never record *types*.
ROI_TOOLS = ROI_TYPES | {"rotated_rectangle", "ellipse"}

#: Open multi-vertex curve types (no enclosed area; drawn as open polylines).
OPEN_LINE_TYPES = {"polyline", "freehand_line"}


@dataclass
class RoiRecord:
    id: str
    name: str
    type: str
    geometry: dict[str, Any]
    coordinate_space: str = "plot"
    target_hint: str = ""
    stroke_color: str = "#ffff00"
    line_width: float = 1.5
    visible: bool = True
    label_visible: bool = True
    source_view: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    mask_key: str = ""
    selected_count: int | None = None
    selection_dirty: bool = True

    @classmethod
    def create(
        cls,
        roi_type: str,
        geometry: dict[str, Any],
        *,
        name: str | None = None,
        coordinate_space: str = "plot",
        target_hint: str = "",
        stroke_color: str = "#ffff00",
        line_width: float = 1.5,
    ) -> "RoiRecord":
        roi_id = uuid.uuid4().hex
        # Leave the name empty for auto-named ROIs; RoiStore.add assigns a
        # human-friendly 1-based name (e.g. "rectangle-1") when the ROI is stored.
        return cls(
            id=roi_id,
            name=name or "",
            type=roi_type,
            geometry=geometry,
            coordinate_space=coordinate_space,
            target_hint=target_hint,
            stroke_color=stroke_color,
            line_width=line_width,
        )


class RoiStore(QObject):
    changed = pyqtSignal()
    selection_changed = pyqtSignal()
    tool_changed = pyqtSignal(str)
    active_adapter_changed = pyqtSignal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.records: list[RoiRecord] = []
        self.selected_ids: list[str] = []
        self.active_tool: str | None = None
        self.show_all: bool = False
        self.show_labels: bool = False
        self.active_adapter = None

    def set_active_adapter(self, adapter) -> None:
        if adapter is self.active_adapter:
            return
        self.active_adapter = adapter
        self.active_adapter_changed.emit(adapter)
        self.changed.emit()

    def set_tool(self, tool: str | None) -> None:
        if tool is not None and tool not in ROI_TOOLS:
            return
        self.active_tool = tool
        self.tool_changed.emit(tool or "")

    def next_type_index(self, roi_type: str) -> int:
        """Next 1-based index for naming a ROI of *roi_type* (``rectangle-1`` …),
        one past the highest existing ``<type>-<n>`` so names don't collide even
        after deletions."""
        prefix = f"{roi_type}-"
        max_n = 0
        for r in self.records:
            if r.type == roi_type and (r.name or "").startswith(prefix):
                suffix = r.name[len(prefix):]
                if suffix.isdigit():
                    max_n = max(max_n, int(suffix))
        return max_n + 1

    def _is_auto_name(self, record: RoiRecord) -> bool:
        name = record.name or ""
        if not name:
            return True
        # legacy auto name was "<type>-<4 hex chars>"
        prefix = f"{record.type}-"
        suffix = name[len(prefix):] if name.startswith(prefix) else ""
        return len(suffix) == 4 and all(c in "0123456789abcdef" for c in suffix)

    def add(self, record: RoiRecord) -> None:
        if self._is_auto_name(record):
            record.name = f"{record.type}-{self.next_type_index(record.type)}"
        self.records.append(record)
        self.selected_ids = [record.id]
        self.changed.emit()
        self.selection_changed.emit()

    def update(self, roi_id: str, record: RoiRecord) -> None:
        for idx, old in enumerate(self.records):
            if old.id == roi_id:
                record.id = roi_id
                if not record.name:
                    record.name = old.name
                self.records[idx] = record
                self.changed.emit()
                return

    def delete_selected(self) -> None:
        ids = set(self.selected_ids)
        self.records = [r for r in self.records if r.id not in ids]
        self.selected_ids = []
        self.changed.emit()
        self.selection_changed.emit()

    def deselect(self) -> None:
        # Selection-only change → emit *only* selection_changed. Emitting `changed`
        # too would make the ROI Manager rebuild its whole list, which resets the
        # current item and breaks arrow-key navigation through the ROIs.
        self.selected_ids = []
        self.selection_changed.emit()

    def select(self, ids: list[str]) -> None:
        valid = {r.id for r in self.records}
        self.selected_ids = [i for i in ids if i in valid]
        self.selection_changed.emit()      # not `changed` — see deselect()

    def rename(self, roi_id: str, name: str) -> None:
        for record in self.records:
            if record.id == roi_id:
                record.name = name
                self.changed.emit()
                return

    def move_selected(self, direction: int) -> None:
        if not self.selected_ids:
            return
        selected = set(self.selected_ids)
        rng = range(1, len(self.records)) if direction < 0 else range(len(self.records) - 2, -1, -1)
        for i in rng:
            j = i + direction
            if self.records[i].id in selected and self.records[j].id not in selected:
                self.records[i], self.records[j] = self.records[j], self.records[i]
        self.changed.emit()

    def set_show_all(self, checked: bool) -> None:
        self.show_all = checked
        self.changed.emit()

    def set_show_labels(self, checked: bool) -> None:
        self.show_labels = checked
        for record in self.records:
            record.label_visible = checked
        self.changed.emit()

    def selected_records(self) -> list[RoiRecord]:
        ids = set(self.selected_ids)
        return [r for r in self.records if r.id in ids]

    def combine_selected(self) -> None:
        selected = self.selected_records()
        if len(selected) < 2:
            return
        combined = QPainterPath()
        for record in selected:
            combined = combined.united(record_to_path(record))
        polygon = combined.toFillPolygon()
        points = [[float(p.x()), float(p.y())] for p in polygon]
        if len(points) < 3:
            return
        base = selected[0]
        record = RoiRecord.create(
            "polygon",
            {"points": points, "closed": True},
            name="Combined",
            coordinate_space=base.coordinate_space,
            target_hint=base.target_hint,
            stroke_color=base.stroke_color,
            line_width=base.line_width,
        )
        self.add(record)

    def save(self, path: str | Path, records: list[RoiRecord] | None = None) -> None:
        path = Path(path)
        records = records or self.records
        if path.suffix.lower() == ".json":
            payload = {"version": 1, "rois": [asdict(r) for r in records]}
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return
        from roifile import roiwrite
        if path.suffix.lower() == ".roi" and len(records) != 1:
            raise ValueError("ImageJ .roi export requires exactly one ROI; use .zip for ROI sets.")
        rois = [record_to_imagej(r) for r in records]
        roiwrite(str(path), rois[0] if path.suffix.lower() == ".roi" and len(rois) == 1 else rois, mode="w")

    def load(self, path: str | Path) -> list[RoiRecord]:
        path = Path(path)
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = [RoiRecord(**item) for item in payload.get("rois", [])]
        else:
            from roifile import roiread
            loaded = roiread(str(path))
            if not isinstance(loaded, list):
                loaded = [loaded]
            records = [record_from_imagej(roi) for roi in loaded]
        for record in records:
            self.add(record)
        return records


def record_to_path(record: RoiRecord) -> QPainterPath:
    path = QPainterPath()
    g = record.geometry
    if record.type in {"rectangle", "oval"}:
        x, y, w, h = _bounds(g)
        if record.type == "oval":
            path.addEllipse(float(x), float(y), float(w), float(h))
        else:
            path.addRect(float(x), float(y), float(w), float(h))
    elif record.type == "line":
        pts = g.get("points", [])
        if len(pts) >= 2:
            path.moveTo(float(pts[0][0]), float(pts[0][1]))
            path.lineTo(float(pts[1][0]), float(pts[1][1]))
    elif record.type == "point":
        pt = g.get("point", [0.0, 0.0])
        x, y = float(pt[0]), float(pt[1])
        path.addEllipse(x - 2.0, y - 2.0, 4.0, 4.0)
    else:
        pts = g.get("points", [])          # vertices may carry a 3rd (depth) coord
        if pts:
            path.moveTo(float(pts[0][0]), float(pts[0][1]))
            for p in pts[1:]:
                path.lineTo(float(p[0]), float(p[1]))
            if g.get("closed", record.type in {"polygon", "freehand"}):
                path.closeSubpath()
    return path


def record_to_points(record: RoiRecord) -> np.ndarray:
    g = record.geometry
    if record.type in {"rectangle", "oval"}:
        x, y, w, h = _bounds(g)
        if record.type == "oval":
            theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
            return np.column_stack([x + w / 2.0 + np.cos(theta) * w / 2.0, y + h / 2.0 + np.sin(theta) * h / 2.0])
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=float)
    if record.type == "point":
        # A point may carry a third (depth) coordinate; ImageJ/2-D paths use x,y.
        pt = g.get("point", [0.0, 0.0])
        return np.asarray([[float(pt[0]), float(pt[1])]], dtype=float)
    raw = g.get("points", [])
    if not len(raw):
        return np.empty((0, 2), dtype=float)
    # Take (x, y) per vertex — robust to a mix of 2-D / 3-D points (drops depth).
    return np.array([[float(p[0]), float(p[1])] for p in raw], dtype=float)


def _rotated_outline_points(record: RoiRecord) -> np.ndarray:
    """Rotated outline vertices of a rectangle/oval (so a rotated box/ellipse can
    be exported to ImageJ as a polygon rather than an axis-aligned bbox)."""
    x, y, w, h = _bounds(record.geometry)
    angle = float((record.geometry or {}).get("angle", 0.0) or 0.0)
    cx, cy = x + w / 2.0, y + h / 2.0
    th = np.deg2rad(angle)
    cos_t, sin_t = np.cos(th), np.sin(th)
    if record.type == "oval":
        a = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        local = np.column_stack([(w / 2.0) * np.cos(a), (h / 2.0) * np.sin(a)])
    else:
        local = np.array([[-w / 2.0, -h / 2.0], [w / 2.0, -h / 2.0],
                          [w / 2.0, h / 2.0], [-w / 2.0, h / 2.0]])
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    return (local @ rot.T) + np.array([cx, cy])


def record_to_imagej(record: RoiRecord):
    """Convert a record to an ImageJ ROI.

    Plot/attribute-coordinate ROIs are exported by writing their (nm) geometry
    **directly as ImageJ pixel coordinates** — the spatial calibration is not
    preserved, but the shape/region is (matching a 1 nm/pixel viewer TIFF export).
    A rotated rectangle/oval is written as a polygon so the rotation survives.
    """
    from roifile import ROI_TYPE, ImagejRoi

    g = record.geometry or {}
    rotated = abs(float(g.get("angle", 0.0) or 0.0)) > 1e-9
    if record.type == "rectangle" and not rotated:
        x, y, w, h = _bounds(g)
        return ImagejRoi(roitype=ROI_TYPE.RECT, left=int(round(x)), top=int(round(y)), right=int(round(x + w)), bottom=int(round(y + h)), name=record.name)
    if record.type == "oval" and not rotated:
        x, y, w, h = _bounds(g)
        return ImagejRoi(roitype=ROI_TYPE.OVAL, left=int(round(x)), top=int(round(y)), right=int(round(x + w)), bottom=int(round(y + h)), name=record.name)
    if record.type == "line":
        points = record_to_points(record)
        if points.shape[0] >= 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            return ImagejRoi(
                roitype=ROI_TYPE.LINE,
                left=int(round(min(x1, x2))), top=int(round(min(y1, y2))),
                right=int(round(max(x1, x2))), bottom=int(round(max(y1, y2))),
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                name=record.name,
            )
    if rotated and record.type in {"rectangle", "oval"}:
        points = _rotated_outline_points(record)
        export_type = "polygon"
    else:
        points = record_to_points(record)
        export_type = record.type
    roi = ImagejRoi.frompoints(np.asarray(points, dtype=float), name=record.name)
    mapping = {
        "polygon": ROI_TYPE.POLYGON,
        "freehand": ROI_TYPE.FREEHAND,
        "point": ROI_TYPE.POINT,
        "polyline": ROI_TYPE.POLYLINE,
        "freehand_line": ROI_TYPE.FREELINE,
        "magnetic_lasso": ROI_TYPE.POLYLINE,
        "angle": ROI_TYPE.POLYLINE,
    }
    roi.roitype = mapping.get(export_type, ROI_TYPE.POLYGON)
    return roi


def is_roi_json_file(path: str | Path) -> bool:
    """True when *path* is a native ROI-set JSON (a dict with a ``"rois"`` list)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(data, dict) and isinstance(data.get("rois"), list)


def record_from_imagej(roi) -> RoiRecord:
    coords = np.asarray(roi.coordinates(), dtype=float)
    roi_name = getattr(roi, "name", None) or "ROI"
    roi_type_name = getattr(getattr(roi, "roitype", None), "name", "").lower()
    if roi_type_name == "point":
        roi_type = "point"
        geometry = {"point": coords[0].tolist() if coords.size else [float(roi.left), float(roi.top)]}
    elif roi_type_name == "polyline":
        roi_type = "polyline"
        geometry = {"points": coords.tolist(), "closed": False}
    elif roi_type_name == "freeline":
        roi_type = "freehand_line"
        geometry = {"points": coords.tolist(), "closed": False}
    elif roi_type_name == "line":
        roi_type = "line"
        geometry = {"points": coords.tolist(), "closed": False}
    elif roi_type_name == "oval":
        roi_type = "oval"
        geometry = {"bounds": [float(roi.left), float(roi.top), float(roi.right - roi.left), float(roi.bottom - roi.top)]}
    elif roi_type_name == "rect":
        roi_type = "rectangle"
        geometry = {"bounds": [float(roi.left), float(roi.top), float(roi.right - roi.left), float(roi.bottom - roi.top)]}
    else:
        roi_type = "polygon"
        geometry = {"points": coords.tolist(), "closed": True}
    return RoiRecord.create(roi_type, geometry, name=roi_name, coordinate_space="pixel")


def _bounds(geometry: dict[str, Any]) -> tuple[float, float, float, float]:
    if "bounds" in geometry:
        x, y, w, h = geometry["bounds"]
        return float(x), float(y), float(w), float(h)
    raw = geometry.get("points", [])
    if not len(raw):
        return 0.0, 0.0, 0.0, 0.0
    # (x, y) per vertex — robust to a mix of 2-D / 3-D points (a raw np.asarray on
    # an inhomogeneous list raises ValueError).
    pts = np.array([[float(p[0]), float(p[1])] for p in raw], dtype=float)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)
