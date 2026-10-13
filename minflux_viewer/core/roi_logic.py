"""
minflux_viewer.core.roi_logic
=============================
Shape (composite) ROI operations, as ImageJ performs them on the active
selection: drawing a new shape while holding **Shift** adds it to the current
selection, holding **Alt** removes it.

The rules follow ImageJ's source (``Roi.modifyRoi``, ``ShapeRoi``,
``ImageCanvas.setRoiModState``):

* The modifier counts only when the shape being drawn is an **area** (rectangle,
  oval, polygon, freehand and their rotated variants) or comes from the magic
  wand; drawing a line ignores it.
* The current selection may be an area, a **composite**, or a **line**. A line
  takes part as the area it would cover at its stroke width — ImageJ's
  ``Roi.convertLineToArea`` — here :data:`LINE_WIDTH_NM` (1 nm) wide, with ends
  extended by half that width.
* Points never mix with areas. (In ImageJ, Shift-clicking with the point tool
  extends a point selection, which this application's point tools do on every
  click anyway.)
* The result is simplified like ImageJ's ``ShapeRoi.trySimplify``: one
  axis-aligned box becomes a ``rectangle``; one outline whose every vertex is a
  genuine corner becomes a ``polygon``, with a handle at each corner; anything
  else is a ``composite`` — a set of closed rings read with the even-odd rule, so
  a ring inside another is a hole — drawn as an outline without vertex handles.

* ⚠ **Curves are flattened, and their vertices are not corners.** Qt's boolean
  operations, like ImageJ's (``ShapeRoi`` converts an ``OvalRoi`` with
  ``getPolygon`` before combining), replace every curve by short straight
  segments — an oval of 1.2 µm becomes ~200 of them, ≤ 0.1 nm from the true
  ellipse. ImageJ then gives handles only to an outline with few or long edges
  (``guessType``: ``POLYGON`` for ≤ 10 segments or an average edge ≥ 2√n px,
  else ``FREEROI``/``TRACED_ROI``, both without handles). The scale-free rule
  here: an outline is a ``polygon`` only when every input was polygonal
  (:func:`record_is_polygonal` — rectangles, polygons, stroked lines, point
  squares) and it has at most :data:`POLYGON_MAX_VERTICES` vertices. So
  rectangle ∪ oval and oval ∪ rectangle are the same ``composite``, whether the
  two overlap (one ring) or not (two), instead of a polygon whose flattened arc
  carries a handle on every segment.

The geometry of a ``composite`` is ``{"rings": [[[x, y], ...], ...],
"polygonal": bool}``. It is a 2-D shape like a polygon, scoped to the plane it
was drawn in; ``polygonal`` records whether its vertices are genuine corners, so
a later operation that leaves one ring can become a ``polygon`` again.

Qt's :class:`QPainterPath` does the polygon clipping (the counterpart of the
``java.awt.geom.Area`` ImageJ uses). QtGui only — no widgets.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QPainterPath, QPainterPathStroker, QPolygonF, QTransform

#: The width a line takes when it becomes part of an area (ImageJ: the line's
#: stroke width, at least one pixel; here the data unit, nm).
LINE_WIDTH_NM = 1.0
#: Side of the square a point would become.
POINT_SIZE_NM = 1.0

AREA_TYPES = frozenset({"rectangle", "oval", "polygon", "freehand", "composite"})
LINE_TYPES = frozenset({"line", "polyline", "freehand_line", "magnetic_lasso", "angle"})
POINT_TYPES = frozenset({"point", "points"})
#: Drawing tools whose shape is an area, so Shift/Alt apply (ImageJ: tools up
#: to FREEROI, plus the wand).
MODIFIER_TOOLS = frozenset({"rectangle", "oval", "polygon", "freehand",
                            "rotated_rectangle", "ellipse", "magic_wand"})

OPS = ("or", "not", "and", "xor")
#: More vertices than this and a single outline is a composite, not a polygon
#: whose every vertex is a handle.
POLYGON_MAX_VERTICES = 64
#: Shapes whose outline consists of genuine corners joined by straight edges —
#: a line takes part stroked with square caps and bevel joins, a point as a
#: square. An oval's outline is a flattened curve, a freehand one a traced path.
POLYGONAL_TYPES = frozenset({"rectangle", "polygon", "line", "polyline", "angle",
                             "point", "points"})
_MIN_RING_AREA = 1e-9


def _xy(p) -> tuple[float, float]:
    return float(p[0]), float(p[1])


def _closed_polygon_path(points) -> QPainterPath:
    path = QPainterPath()
    pts = [_xy(p) for p in points]
    if len(pts) < 3:
        return path
    path.addPolygon(QPolygonF([QPointF(x, y) for x, y in pts]))
    path.closeSubpath()
    return path


def _box_path(geometry, ellipse: bool) -> QPainterPath:
    x, y, w, h = (float(v) for v in geometry["bounds"])
    path = QPainterPath()
    if ellipse:
        path.addEllipse(x, y, w, h)
    else:
        path.addRect(x, y, w, h)
    angle = float(geometry.get("angle", 0.0) or 0.0)
    if abs(angle) > 1e-12:
        cx, cy = x + w / 2.0, y + h / 2.0
        t = QTransform()
        t.translate(cx, cy)
        t.rotate(angle)                    # degrees, counter-clockwise in data axes
        t.translate(-cx, -cy)
        path = t.map(path)
    return path


def area_path(record) -> QPainterPath:
    """The enclosed area of an area ROI (even-odd fill)."""
    kind = record.type
    g = record.geometry or {}
    if kind == "rectangle":
        path = _box_path(g, ellipse=False)
    elif kind == "oval":
        path = _box_path(g, ellipse=True)
    elif kind in ("polygon", "freehand"):
        path = _closed_polygon_path(g.get("points", []))
    elif kind == "composite":
        path = QPainterPath()
        for ring in g.get("rings", []):
            path.addPath(_closed_polygon_path(ring))
    else:
        raise ValueError(f"{kind!r} is not an area ROI")
    path.setFillRule(Qt.FillRule.OddEvenFill)
    return path


def line_path(record, width_nm: float = LINE_WIDTH_NM) -> QPainterPath:
    """The area a line ROI covers at ``width_nm`` (ImageJ ``convertLineToArea``)."""
    pts = [_xy(p) for p in (record.geometry or {}).get("points", [])]
    path = QPainterPath()
    if len(pts) < 2:
        return path
    path.moveTo(*pts[0])
    for p in pts[1:]:
        path.lineTo(*p)
    if (record.geometry or {}).get("closed"):
        path.closeSubpath()
    stroker = QPainterPathStroker()
    stroker.setWidth(float(width_nm))
    stroker.setCapStyle(Qt.PenCapStyle.SquareCap)    # ends extended by width/2, as ImageJ
    stroker.setJoinStyle(Qt.PenJoinStyle.BevelJoin)
    out = stroker.createStroke(path).simplified()
    out.setFillRule(Qt.FillRule.OddEvenFill)
    return out


def point_path(record, size_nm: float = POINT_SIZE_NM) -> QPainterPath:
    """Each point as a ``size_nm`` square centred on it."""
    g = record.geometry or {}
    pts = [g["point"]] if "point" in g else list(g.get("points", []))
    path = QPainterPath()
    h = float(size_nm) / 2.0
    for p in pts:
        x, y = _xy(p)
        path.addRect(x - h, y - h, 2 * h, 2 * h)
    path.setFillRule(Qt.FillRule.OddEvenFill)
    return path


def record_path(record, *, line_width_nm: float = LINE_WIDTH_NM,
                point_size_nm: float = POINT_SIZE_NM) -> QPainterPath:
    """Any 2-D ROI as an area: areas as they are, lines widened, points squared."""
    if record.type in AREA_TYPES:
        return area_path(record)
    if record.type in LINE_TYPES:
        return line_path(record, line_width_nm)
    if record.type in POINT_TYPES:
        return point_path(record, point_size_nm)
    raise ValueError(f"{record.type!r} ROIs take no part in shape operations")


def can_combine(previous, tool: str) -> bool:
    """Whether drawing with *tool* over *previous* is a shape operation."""
    return (previous is not None and tool in MODIFIER_TOOLS
            and getattr(previous, "type", None) in (AREA_TYPES | LINE_TYPES))


def path_rings(path: QPainterPath) -> list[list[list[float]]]:
    """Closed rings of a path (closing vertex dropped, degenerate rings skipped)."""
    rings = []
    for poly in path.toSubpathPolygons():
        pts = [[float(p.x()), float(p.y())] for p in poly]
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]
        if len(pts) < 3:
            continue
        a = np.asarray(pts)
        area = 0.5 * abs(np.dot(a[:, 0], np.roll(a[:, 1], -1))
                         - np.dot(a[:, 1], np.roll(a[:, 0], -1)))
        if area > _MIN_RING_AREA:
            rings.append(pts)
    return rings


def _axis_aligned_box(ring) -> list[float] | None:
    if len(ring) != 4:
        return None
    xs = sorted({round(p[0], 9) for p in ring})
    ys = sorted({round(p[1], 9) for p in ring})
    if len(xs) != 2 or len(ys) != 2:
        return None
    return [xs[0], ys[0], xs[1] - xs[0], ys[1] - ys[0]]


def record_is_polygonal(record) -> bool:
    """Whether every vertex *record* contributes to a combined outline is a corner.

    A composite answers from its own ``polygonal`` flag; one without the flag (an
    older record, or one read from an ImageJ file) is taken as not polygonal.
    """
    if record is None:
        return False
    if record.type == "composite":
        return bool((record.geometry or {}).get("polygonal", False))
    return record.type in POLYGONAL_TYPES


def geometry_from_path(path: QPainterPath, *, polygonal: bool = True) -> tuple[str, dict] | None:
    """``(type, geometry)`` of a path, simplified the way ImageJ does, or
    ``None`` when nothing is left.

    *polygonal*: whether the vertices of *path* are genuine corners (no flattened
    curve, no traced outline) — see the module notes.
    """
    rings = path_rings(path)
    if not rings:
        return None
    if len(rings) == 1:
        box = _axis_aligned_box(rings[0])
        if box is not None:
            return "rectangle", {"bounds": box}
        if polygonal and len(rings[0]) <= POLYGON_MAX_VERTICES:
            return "polygon", {"points": rings[0], "closed": True}
    return "composite", {"rings": rings, "polygonal": bool(polygonal)}


def combine_paths(a: QPainterPath, b: QPainterPath, op: str) -> QPainterPath:
    if op == "or":
        out = a.united(b)
    elif op == "not":
        out = a.subtracted(b)
    elif op == "and":
        out = a.intersected(b)
    elif op == "xor":
        out = a.united(b).subtracted(a.intersected(b))
    else:
        raise ValueError(f"unknown shape operation {op!r}")
    out = out.simplified()
    out.setFillRule(Qt.FillRule.OddEvenFill)
    return out


def combine(previous, new, op: str, *, line_width_nm: float = LINE_WIDTH_NM):
    """The record ImageJ would leave after drawing *new* over *previous*.

    Returns a new :class:`~minflux_viewer.core.roi.RoiRecord` carrying
    *previous*'s name, colour, scope and context, or ``None`` when the operation
    leaves nothing (everything removed).
    """
    from .roi import RoiRecord

    path = combine_paths(record_path(previous, line_width_nm=line_width_nm),
                         record_path(new, line_width_nm=line_width_nm), op)
    result = geometry_from_path(
        path, polygonal=record_is_polygonal(previous) and record_is_polygonal(new))
    if result is None:
        return None
    kind, geometry = result
    return RoiRecord.create(
        kind, geometry, name=getattr(previous, "name", "") or "",
        coordinate_space=getattr(previous, "coordinate_space", "plot"),
        target_hint=getattr(previous, "target_hint", "") or "",
        stroke_color=getattr(previous, "stroke_color", "#ffff00"),
        line_width=getattr(previous, "line_width", 1.0),
        context=dict(getattr(previous, "context", {}) or {}),
    )


def composite_rings(record) -> list[np.ndarray]:
    """The rings of a composite as ``(n, 2)`` arrays."""
    return [np.asarray([[float(p[0]), float(p[1])] for p in ring], dtype=float)
            for ring in (record.geometry or {}).get("rings", []) if len(ring) >= 3]


def translate_rings(rings: Iterable, dx: float, dy: float) -> list[list[list[float]]]:
    return [[[float(p[0]) + dx, float(p[1]) + dy] for p in ring] for ring in rings]


def imagej_path(record) -> np.ndarray:
    """A composite as ImageJ's shape path: MOVETO (0) x y, LINETO (1) x y, …,
    CLOSE (4) per ring, absolute coordinates (``RoiDecoder.getShapeRoi``)."""
    out: list[float] = []
    for ring in composite_rings(record):
        out += [0.0, float(ring[0, 0]), float(ring[0, 1])]
        for x, y in ring[1:]:
            out += [1.0, float(x), float(y)]
        out.append(4.0)
    return np.asarray(out, dtype=np.float32)


def rings_from_imagej_path(path_array) -> list[list[list[float]]]:
    """Inverse of :func:`imagej_path` (curves are not produced by ImageJ for
    composites read from file and are refused)."""
    vals = [float(v) for v in np.asarray(path_array, dtype=float).ravel()]
    rings, cur, i = [], [], 0
    while i < len(vals):
        op = int(round(vals[i]))
        if op == 0:
            if len(cur) >= 3:
                rings.append(cur)
            cur = [[vals[i + 1], vals[i + 2]]]
            i += 3
        elif op == 1:
            cur.append([vals[i + 1], vals[i + 2]])
            i += 3
        elif op == 4:
            if len(cur) >= 3:
                rings.append(cur)
            cur = []
            i += 1
        else:
            raise ValueError(f"unsupported path segment {op} in a composite ROI")
    if len(cur) >= 3:
        rings.append(cur)
    return rings
