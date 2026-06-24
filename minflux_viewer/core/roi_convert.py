"""
minflux_viewer.core.roi_convert
================================
ROI type conversion and enlarge/shrink/skeletonize geometry — pure, Qt-widget-free
helpers that take a :class:`~minflux_viewer.core.roi.RoiRecord` and return a new
record. Used by the render/scatter right-click *Convert to…* submenus, the ROI
Manager *Convert* / *Property* dialogs and the **Process › ROI** menu.

Conversions preserve ``context`` (dataset / view-plane / depth), colour and
coordinate space; geometry is computed in the source ROI's own 2-D view plane.
Only ``QtGui`` geometry primitives are used (``QPainterPathStroker`` for the
line→region buffer); SciPy supplies the convex hull.
"""

from __future__ import annotations

import numpy as np

from .roi import (
    OPEN_LINE_TYPES,
    RoiRecord,
    _bounds,
    record_to_points,
)

#: Closed, area-bearing ROI types.
REGION_TYPES = {"rectangle", "oval", "polygon", "freehand"}
#: Open curve ROI types (a poly-line skeleton, no area).
LINE_TYPES = {"line", "polyline", "freehand_line", "magnetic_lasso"}

#: Friendly label for each conversion target token (right-click submenu).
TARGET_LABELS = {
    "point": "to Point",
    "rectangle": "to Rectangle",
    "oval": "to Oval",
    "polygon": "to Polygon",
    "convex_hull": "to Convex Hull",
    "region": "to Region",
    "line": "to Line",
}


def _points_2d(record: RoiRecord) -> np.ndarray:
    """In-plane (N, 2) boundary/vertex points for any ROI."""
    pts = np.asarray(record_to_points(record), dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return np.empty((0, 2), dtype=float)
    return pts[:, :2]


def _centroid(record: RoiRecord) -> tuple[float, float]:
    if record.type in {"rectangle", "oval"}:
        x, y, w, h = _bounds(record.geometry)
        return x + w / 2.0, y + h / 2.0
    if record.type == "point":
        pt = record.geometry.get("point", [0.0, 0.0])
        return float(pt[0]), float(pt[1])
    if record.type == "angle":
        pts = np.asarray(record.geometry.get("points", []), dtype=float)
        if pts.shape[0] >= 2:
            return float(pts[1][0]), float(pts[1][1])  # the angle vertex
    pts = _points_2d(record)
    if pts.shape[0] == 0:
        return 0.0, 0.0
    c = pts.mean(axis=0)
    return float(c[0]), float(c[1])


def _depth_of(record: RoiRecord) -> float:
    ctx = record.context if isinstance(record.context, dict) else {}
    try:
        return float(ctx.get("depth_value", 0.0) or 0.0)
    except Exception:
        return 0.0


def available_conversions(record: RoiRecord) -> list[str]:
    """Conversion target tokens valid for *record* (in menu order)."""
    t = record.type
    out: list[str] = []
    if t == "angle":
        return ["point"]
    if t != "point":          # a point → point conversion is meaningless
        out.append("point")
    if t in REGION_TYPES:
        if t != "rectangle":
            out.append("rectangle")
        if t != "oval":
            out.append("oval")
        if t in {"polygon", "freehand"}:
            out.append("convex_hull")
        out.append("line")  # region → skeleton line
    elif t in LINE_TYPES:
        # line types can only grow into a region (not a box/oval — removed).
        out.append("region")  # line → buffered region
        if t in OPEN_LINE_TYPES or t == "magnetic_lasso":
            out.append("polygon")
    elif t == "point":
        out.append("rectangle")
        out.append("oval")
    return out


def _derive(source: RoiRecord, target_type: str, geometry: dict) -> RoiRecord:
    rec = RoiRecord.create(
        target_type,
        geometry,
        name=source.name or "",
        coordinate_space=source.coordinate_space,
        target_hint=source.target_hint,
        stroke_color=source.stroke_color,
        line_width=source.line_width,
    )
    rec.context = dict(source.context) if isinstance(source.context, dict) else {}
    rec.source_view = source.source_view
    rec.visible = source.visible
    rec.label_visible = source.label_visible
    rec.selection_dirty = True
    return rec


def _convex_hull(points: np.ndarray) -> list[list[float]]:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return [[float(p[0]), float(p[1])] for p in pts]
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(pts)
        return [[float(pts[i, 0]), float(pts[i, 1])] for i in hull.vertices]
    except Exception:
        return [[float(p[0]), float(p[1])] for p in pts]


def _oriented_bbox(points: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Minimal-PCA oriented bounding box of *points* → (center, width, height,
    angle°). ``angle`` is the rotation of the box's local X axis (the principal
    axis) about its centre, matching the rectangle/oval geometry convention."""
    pts = np.asarray(points, dtype=float)[:, :2]
    if pts.shape[0] < 2:
        c = pts.mean(axis=0) if pts.shape[0] else np.zeros(2)
        return c, 0.0, 0.0, 0.0
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    axes = evecs[:, np.argsort(evals)[::-1]]   # columns: major, minor
    proj = centered @ axes
    mins, maxs = proj.min(axis=0), proj.max(axis=0)
    w, h = float(maxs[0] - mins[0]), float(maxs[1] - mins[1])
    center = mean + axes @ ((mins + maxs) / 2.0)
    angle = float(np.degrees(np.arctan2(axes[1, 0], axes[0, 0])))
    return center, w, h, angle


def buffer_line_to_polygon(points2d, radius: float, *, closed: bool = False) -> list[list[float]]:
    """Round-cap/round-join buffer of a poly-line into a filled polygon outline,
    using ``QPainterPathStroker`` (radius nm on each side).

    An **open** line → one solid band (the largest stroke contour). A **closed**
    line (an enclosed outline) → a **ring/band** following the outline: both the
    outer and inner stroke contours are concatenated so the even-odd polygon fill
    (and `polygon_mask`) yields the annulus."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QPainterPath, QPainterPathStroker

    pts = np.asarray(points2d, dtype=float)
    if pts.shape[0] < 2 or radius <= 0:
        return []
    path = QPainterPath()
    path.moveTo(float(pts[0, 0]), float(pts[0, 1]))
    for p in pts[1:]:
        path.lineTo(float(p[0]), float(p[1]))
    if closed:
        path.closeSubpath()
    stroker = QPainterPathStroker()
    stroker.setWidth(2.0 * float(radius))
    stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
    stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    # simplified() resolves the stroke's self-overlaps (round caps/joins,
    # doubled-back segments) into clean contours — otherwise the even-odd polygon
    # fill turns those overlaps into voids / hollow ends.
    stroke = stroker.createStroke(path).simplified()
    polys = stroke.toSubpathPolygons()
    if not polys:
        polys = [stroke.toFillPolygon()]
    if closed and len(polys) > 1:
        out: list[list[float]] = []      # outer + inner contours → even-odd ring
        for poly in polys:
            out.extend([float(pt.x()), float(pt.y())] for pt in poly)
        return out
    poly = max(polys, key=lambda p: p.size())
    out = [[float(pt.x()), float(pt.y())] for pt in poly]
    if len(out) >= 2 and out[0] == out[-1]:
        out = out[:-1]
    return out


def region_principal_axis_line(record: RoiRecord) -> list[list[float]]:
    """Straight skeleton line along a region's principal (PCA) axis, clipped to
    the region's extent. A simple, robust 'region → line' / skeleton."""
    pts = _points_2d(record)
    if pts.shape[0] < 2:
        return []
    center = pts.mean(axis=0)
    centered = pts - center
    if pts.shape[0] == 2:
        direction = centered[1] - centered[0]
    else:
        cov = np.cov(centered.T)
        evals, evecs = np.linalg.eigh(cov)
        direction = evecs[:, int(np.argmax(evals))]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return []
    direction = direction / norm
    t = centered @ direction
    p0 = center + direction * float(t.min())
    p1 = center + direction * float(t.max())
    return [[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]]


def region_outline_points(record: RoiRecord) -> list[list[float]]:
    """Perimeter/outline vertices of a region as a vertex list (nm), honouring
    rotation. Rectangle → 4 corners; oval → 48-point ellipse; polygon/freehand →
    its own vertices. Used by region→Line (an enclosed, area-less outline)."""
    t = record.type
    if t in {"rectangle", "oval"}:
        x, y, w, h = _bounds(record.geometry)
        angle = float(record.geometry.get("angle", 0.0) or 0.0)
        cx, cy = x + w / 2.0, y + h / 2.0
        th = np.deg2rad(angle)
        cos_t, sin_t = np.cos(th), np.sin(th)
        if t == "rectangle":
            local = [(-w / 2.0, -h / 2.0), (w / 2.0, -h / 2.0),
                     (w / 2.0, h / 2.0), (-w / 2.0, h / 2.0)]
        else:
            a = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
            local = list(zip((w / 2.0) * np.cos(a), (h / 2.0) * np.sin(a)))
        return [[cx + cos_t * lx - sin_t * ly, cy + sin_t * lx + cos_t * ly]
                for lx, ly in local]
    return [[float(p[0]), float(p[1])] for p in _points_2d(record)]


def _rect_oval_major_axis_line(record: RoiRecord) -> list[list[float]]:
    """Centreline of a rectangle/oval along its **major axis** (the longer side),
    through the centre, honouring rotation. A square/circle → its (rotated)
    horizontal axis; otherwise the longer axis, splitting the shape into halves."""
    x, y, w, h = _bounds(record.geometry)
    angle = float(record.geometry.get("angle", 0.0) or 0.0)
    cx, cy = x + w / 2.0, y + h / 2.0
    th = np.deg2rad(angle)
    cos_t, sin_t = np.cos(th), np.sin(th)
    if w >= h:                       # local-X axis (square uses this → horizontal)
        dx, dy, half = cos_t, sin_t, w / 2.0
    else:                            # local-Y axis (major axis is the height)
        dx, dy, half = -sin_t, cos_t, h / 2.0
    return [[cx - dx * half, cy - dy * half], [cx + dx * half, cy + dy * half]]


def _zhang_suen(image: np.ndarray) -> np.ndarray:
    """Zhang–Suen morphological thinning → a 1-px-wide skeleton (bool array)."""
    img = (np.asarray(image) > 0).astype(np.uint8)
    img = np.pad(img, 1)

    def nb(a):
        return (a[:-2, 1:-1], a[:-2, 2:], a[1:-1, 2:], a[2:, 2:],
                a[2:, 1:-1], a[2:, :-2], a[1:-1, :-2], a[:-2, :-2])

    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p2, p3, p4, p5, p6, p7, p8, p9 = nb(img)
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            a_trans = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            b_count = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            cond = (img[1:-1, 1:-1] == 1) & (b_count >= 2) & (b_count <= 6) & (a_trans == 1)
            if step == 0:
                cond &= (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond &= (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            if cond.any():
                img[1:-1, 1:-1][cond] = 0
                changed = True
    return img[1:-1, 1:-1].astype(bool)


def _trace_skeleton(skel: np.ndarray) -> list[tuple[int, int]]:
    """Longest path (graph diameter) through 8-connected skeleton pixels →
    ordered (row, col) list."""
    from collections import deque

    coords = np.argwhere(skel)
    if coords.shape[0] == 0:
        return []
    index = {(int(r), int(c)): i for i, (r, c) in enumerate(coords)}
    adj: dict[int, list[int]] = {i: [] for i in range(len(coords))}
    for (r, c), i in index.items():
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    j = index.get((r + dr, c + dc))
                    if j is not None:
                        adj[i].append(j)

    def bfs(src: int):
        prev = {src: -1}
        q = deque([src])
        last = src
        while q:
            u = q.popleft()
            last = u
            for v in adj[u]:
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        return last, prev

    far, _ = bfs(0)
    end, prev = bfs(far)
    path = []
    u = end
    while u != -1:
        path.append(u)
        u = prev[u]
    path.reverse()
    return [(int(coords[i][0]), int(coords[i][1])) for i in path]


def region_skeleton_polyline(record: RoiRecord, *, max_pixels_long_axis: int = 200,
                             max_vertices: int = 64) -> list[list[float]]:
    """Shape-aware skeleton of a region ROI as an ordered vertex list (nm).

    Rasterizes the region, thins it (Zhang–Suen), traces the longest skeleton
    path and resamples it to at most ``max_vertices`` vertices. Rectangles/ovals
    (whose medial axis is their centreline) and any degenerate result fall back
    to the principal-axis line.
    """
    if record.type in {"rectangle", "oval"}:
        return region_principal_axis_line(record)
    pts = _points_2d(record)
    if pts.shape[0] < 3:
        return region_principal_axis_line(record)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    w, h = float(x1 - x0), float(y1 - y0)
    span = max(w, h)
    if span <= 0:
        return region_principal_axis_line(record)
    pixel = span / float(max(max_pixels_long_axis, 8))
    nx = int(np.ceil(w / pixel)) + 3
    ny = int(np.ceil(h / pixel)) + 3
    xs = x0 + (np.arange(nx) - 1.0 + 0.5) * pixel
    ys = y0 + (np.arange(ny) - 1.0 + 0.5) * pixel
    gx, gy = np.meshgrid(xs, ys)

    from .roi_selection import polygon_mask

    mask = polygon_mask(gx.ravel(), gy.ravel(), record).reshape(gy.shape)
    if not mask.any():
        return region_principal_axis_line(record)
    path = _trace_skeleton(_zhang_suen(mask))
    if len(path) < 2:
        return region_principal_axis_line(record)
    if len(path) > max_vertices:  # resample (keep endpoints)
        keep = np.linspace(0, len(path) - 1, max_vertices).round().astype(int)
        path = [path[i] for i in dict.fromkeys(keep)]
    return [[float(xs[c]), float(ys[r])] for (r, c) in path]


def convert_roi(
    record: RoiRecord,
    target: str,
    *,
    width: float | None = None,
    height: float | None = None,
) -> RoiRecord:
    """Convert *record* to a *target* type token. Raises ``ValueError`` when the
    conversion is not possible. ``width``/``height`` (nm) are required for
    point→rectangle/oval and used as the band width for line→region."""
    if target not in available_conversions(record):
        raise ValueError(f"cannot convert {record.type} → {target}")
    t = record.type

    if target == "point":
        cx, cy = _centroid(record)
        return _derive(record, "point", {"point": [cx, cy, _depth_of(record)]})

    if target in {"rectangle", "oval"}:
        if t == "point":
            if not width or not height or width <= 0 or height <= 0:
                raise ValueError("point → rectangle/oval needs a positive width and height")
            cx, cy = _centroid(record)
            geom = {"bounds": [cx - width / 2.0, cy - height / 2.0, float(width), float(height)],
                    "angle": 0.0}
        elif t in {"rectangle", "oval"}:
            # rectangle ↔ oval: keep the same box + rotation.
            x, y, w, h = _bounds(record.geometry)
            geom = {"bounds": [x, y, w, h], "angle": float(record.geometry.get("angle", 0.0) or 0.0)}
        else:
            # polygon / freehand → best-fit *oriented* (rotated) box.
            center, w, h, angle = _oriented_bbox(_points_2d(record))
            geom = {"bounds": [center[0] - w / 2.0, center[1] - h / 2.0, w, h], "angle": angle}
        return _derive(record, target, geom)

    if target == "polygon":
        pts = [[float(p[0]), float(p[1])] for p in _points_2d(record)]
        if len(pts) < 3:
            raise ValueError("need at least 3 vertices to form a polygon")
        return _derive(record, "polygon", {"points": pts, "closed": True})

    if target == "convex_hull":
        hull = _convex_hull(_points_2d(record))
        if len(hull) < 3:
            raise ValueError("convex hull needs at least 3 points")
        return _derive(record, "polygon", {"points": hull, "closed": True})

    if target == "region":
        if width is None or width <= 0:
            raise ValueError("line → region needs a positive width")
        closed = bool(record.geometry.get("closed", False))
        poly = buffer_line_to_polygon(_points_2d(record), float(width) / 2.0, closed=closed)
        if len(poly) < 3:
            raise ValueError("could not build a region from this line")
        return _derive(record, "polygon", {"points": poly, "closed": True})

    if target == "line":
        # region → an *enclosed* outline (a closed line, no area within); it can
        # be enlarged into a band. (Skeletonize is the separate centreline op.)
        outline = region_outline_points(record)
        if len(outline) < 3:
            raise ValueError("could not derive an outline from this region")
        return _derive(record, "polyline", {"points": outline, "closed": True})

    raise ValueError(f"unknown conversion target: {target}")


# --------------------------------------------------------------------------
# Enlarge / shrink / skeletonize
# --------------------------------------------------------------------------

def can_resize(record: RoiRecord) -> bool:
    """True when enlarge/shrink is defined for this ROI type (all but angle)."""
    return record.type != "angle"


def enlarge_shrink_roi(record: RoiRecord, value_nm: float, *, mode: str = "enlarge") -> RoiRecord:
    """Grow (``mode='enlarge'``) or shrink (``mode='shrink'``) *record* by
    ``value_nm`` and return a new record.

    Regions grow/shrink their bounds by ``value`` on every side (polygons/freehand
    rescale their vertices about the bbox centre). A point can only be enlarged →
    an oval of diameter ``value`` centred on it. A line is enlarged into a region
    by buffering its skeleton by ``value`` on each side.
    """
    if not can_resize(record):
        raise ValueError("angle ROIs cannot be enlarged or shrunk")
    value = float(value_nm)
    if value <= 0:
        raise ValueError("enter a positive amount")
    delta = value if mode == "enlarge" else -value
    t = record.type

    if t == "point":
        if mode != "enlarge":
            raise ValueError("a point ROI can only be enlarged")
        cx, cy = _centroid(record)
        return _derive(record, "oval", {"bounds": [cx - value / 2.0, cy - value / 2.0, value, value]})

    if t in LINE_TYPES:
        if mode != "enlarge":
            raise ValueError("a line ROI can only be enlarged (into a region)")
        # an enclosed (closed) line → a ring/band; an open line → a solid band.
        closed = bool(record.geometry.get("closed", False))
        poly = buffer_line_to_polygon(_points_2d(record), value, closed=closed)
        if len(poly) < 3:
            raise ValueError("could not build a region from this line")
        return _derive(record, "polygon", {"points": poly, "closed": True})

    if t in {"rectangle", "oval"}:
        x, y, w, h = _bounds(record.geometry)
        nw = w + 2.0 * delta
        nh = h + 2.0 * delta
        if nw <= 0 or nh <= 0:
            raise ValueError("shrink amount is too large for this ROI")
        geom = {"bounds": [x - delta, y - delta, nw, nh]}
        if record.geometry.get("angle"):     # preserve rotation for rect *and* oval
            geom["angle"] = float(record.geometry["angle"])
        return _derive(record, t, geom)

    if t in {"polygon", "freehand"}:
        pts = _points_2d(record)
        if pts.shape[0] < 3:
            raise ValueError("polygon needs at least 3 vertices")
        x, y, w, h = _bounds(record.geometry)
        cx, cy = x + w / 2.0, y + h / 2.0
        sx = (w + 2.0 * delta) / w if w > 0 else 1.0
        sy = (h + 2.0 * delta) / h if h > 0 else 1.0
        if sx <= 0 or sy <= 0:
            raise ValueError("shrink amount is too large for this ROI")
        scaled = np.column_stack([cx + (pts[:, 0] - cx) * sx, cy + (pts[:, 1] - cy) * sy])
        return _derive(record, t, {"points": [[float(p[0]), float(p[1])] for p in scaled],
                                   "closed": True})

    raise ValueError(f"cannot resize ROI of type {t}")


def skeletonize_roi(record: RoiRecord) -> RoiRecord:
    """Reduce a region ROI to a **shape-aware** skeleton (distinct from region→Line,
    which is the outline). Rectangle/oval → their **major-axis centreline** (a
    square/circle uses the rotated horizontal axis; otherwise the longer axis,
    splitting the shape into halves). Polygon/freehand → Zhang–Suen thinning of
    the rasterized region (a polyline following the shape)."""
    if record.type not in REGION_TYPES:
        raise ValueError("skeletonize is defined for region ROIs (rectangle/oval/polygon/freehand)")
    if record.type in {"rectangle", "oval"}:
        return _derive(record, "line", {"points": _rect_oval_major_axis_line(record), "closed": False})
    line = region_skeleton_polyline(record)
    if len(line) < 2:
        raise ValueError("could not skeletonize this ROI")
    roi_type = "polyline" if len(line) > 2 else "line"
    return _derive(record, roi_type, {"points": line, "closed": False})
