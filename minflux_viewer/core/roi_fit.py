"""
minflux_viewer.core.roi_fit
===========================
Fit ROI shapes to the **localizations a region ROI highlights** (Process › ROI ›
Fit), plus ImageJ-style **spline fit** / **interpolate** on a ROI outline. Pure
NumPy/SciPy, Qt-free, unit-tested.

The data-point fits (rectangle / circle / ellipse / convex hull / polygon) survey
the points *inside* the source ROI and fit an enclosing shape to them — distinct
from ``roi_convert``'s shape-fits, which fit the ROI's own *outline*. They reuse
:mod:`core.min_enclosing` for the enclosing rectangle / circle / ellipse.

References the ImageJ ``ij.plugin.Selection`` fit-ellipse / fitSpline / interpolate
behaviour.
"""

from __future__ import annotations

import numpy as np

from .min_enclosing import min_area_rectangle, min_enclosing_circle
from .roi import RoiRecord
from .roi_convert import _convex_hull, _derive, _points_2d, region_outline_points

# Data-point fit targets (Fit Rectangle / Circle / Ellipse / Polygon / Convex Hull).
DATA_FIT_TARGETS = ("rectangle", "circle", "ellipse", "polygon", "convex_hull")


# --------------------------------------------------------------------------
# best-fit ellipse (moments) + concave hull
# --------------------------------------------------------------------------

def fit_ellipse_moments(points) -> tuple[float, float, float, float, float]:
    """Best-fit (second-moment) ellipse of 2-D *points* →
    ``(cx, cy, width, height, angle_deg)``.

    Oriented by the principal axes of the point covariance; the full axes are
    ``2·(2·√eigenvalue)`` (a 2-σ ellipse), the point-cloud analogue of ImageJ's
    normalized-second-central-moment 'fit ellipse'. ``width`` is the major-axis
    extent, ``angle`` its CCW rotation (deg), matching the oval geometry."""
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] < 2 or P.shape[0] == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    P = P[:, :2]
    c = P.mean(axis=0)
    if P.shape[0] < 3:
        return float(c[0]), float(c[1]), 0.0, 0.0, 0.0
    cov = np.cov((P - c).T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]          # major first
    lam = np.maximum(evals[order], 0.0)
    major_vec = evecs[:, order[0]]
    semi_major = 2.0 * float(np.sqrt(lam[0]))
    semi_minor = 2.0 * float(np.sqrt(lam[1]))
    angle = float(np.degrees(np.arctan2(major_vec[1], major_vec[0])))
    return float(c[0]), float(c[1]), 2.0 * semi_major, 2.0 * semi_minor, angle


def _order_boundary(edges) -> list[int] | None:
    """Order alpha-shape boundary edges into a single closed vertex loop, or
    ``None`` if it isn't a simple loop (→ caller falls back to the convex hull)."""
    from collections import defaultdict

    adj: dict[int, list[int]] = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    if not adj or any(len(n) != 2 for n in adj.values()):
        return None
    start = next(iter(adj))
    loop = [start]
    prev, cur = None, start
    while True:
        nbrs = [n for n in adj[cur] if n != prev]
        if not nbrs:
            return None
        nxt = nbrs[0]
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
        if len(loop) > len(adj):
            return None
    return loop if len(loop) == len(adj) else None


def concave_hull(points, *, alpha_factor: float = 2.5) -> list[list[float]]:
    """Concave-hull (alpha-shape boundary) polygon of 2-D *points*.

    Delaunay-triangulates the points, keeps triangles whose circumradius is below
    ``alpha_factor × median circumradius``, and traces the single boundary loop of
    that kept region — a non-convex outline that hugs the point cloud. Falls back
    to the **convex hull** (which strictly encloses every point) when a simple
    boundary loop cannot be formed."""
    P = np.unique(np.asarray(points, dtype=float)[:, :2], axis=0)
    if P.shape[0] < 4:
        return _convex_hull(P)
    try:
        from scipy.spatial import Delaunay

        tri = Delaunay(P)
    except Exception:
        return _convex_hull(P)
    s = tri.simplices
    a, b, c = P[s[:, 0]], P[s[:, 1]], P[s[:, 2]]
    la = np.linalg.norm(b - c, axis=1)
    lb = np.linalg.norm(a - c, axis=1)
    lc = np.linalg.norm(a - b, axis=1)
    sp = (la + lb + lc) / 2.0
    area = np.sqrt(np.maximum(sp * (sp - la) * (sp - lb) * (sp - lc), 1e-12))
    circ_r = (la * lb * lc) / (4.0 * area)
    keep = circ_r <= alpha_factor * float(np.median(circ_r))
    from collections import defaultdict

    edge_count: dict[tuple[int, int], int] = defaultdict(int)
    for ti in np.flatnonzero(keep):
        v = s[ti]
        for u, w in ((v[0], v[1]), (v[1], v[2]), (v[2], v[0])):
            edge_count[(min(int(u), int(w)), max(int(u), int(w)))] += 1
    boundary = [e for e, n in edge_count.items() if n == 1]
    loop = _order_boundary(boundary)
    if loop is None or len(loop) < 3:
        return _convex_hull(P)
    return [[float(P[i, 0]), float(P[i, 1])] for i in loop]


# --------------------------------------------------------------------------
# data-point shape fits → RoiRecord
# --------------------------------------------------------------------------

def data_fit(source: RoiRecord, points, target: str) -> RoiRecord:
    """Fit a *target* shape to *points* (the localizations the source ROI
    highlights) and return a new RoiRecord deriving the source's context/color."""
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] < 2 or P.shape[0] < 3:
        raise ValueError("need at least 3 highlighted localizations to fit")
    P = P[:, :2]

    if target == "rectangle":                      # minimum-area oriented rectangle
        center, w, h, angle = min_area_rectangle(P)
        geom = {"bounds": [center[0] - w / 2.0, center[1] - h / 2.0, w, h],
                "angle": angle, "variant": "rotated"}
        return _derive(source, "rectangle", geom)

    if target == "circle":                         # minimum enclosing circle → oval
        cx, cy, r = min_enclosing_circle(P)
        d = 2.0 * r
        geom = {"bounds": [cx - r, cy - r, d, d], "angle": 0.0}
        return _derive(source, "oval", geom)

    if target == "ellipse":                        # best-fit (moment) ellipse → oval
        cx, cy, w, h, angle = fit_ellipse_moments(P)
        geom = {"bounds": [cx - w / 2.0, cy - h / 2.0, w, h],
                "angle": angle, "variant": "rotated"}
        return _derive(source, "oval", geom)

    if target == "convex_hull":                    # convex hull of the points
        hull = _convex_hull(P)
        if len(hull) < 3:
            raise ValueError("convex hull needs at least 3 points")
        return _derive(source, "polygon", {"points": hull, "closed": True})

    if target == "polygon":                        # concave (alpha-shape) outline
        poly = concave_hull(P)
        if len(poly) < 3:
            raise ValueError("could not form a polygon from the points")
        return _derive(source, "polygon", {"points": poly, "closed": True})

    raise ValueError(f"unknown data fit target: {target}")


# --------------------------------------------------------------------------
# outline operations: spline fit + interpolate (ImageJ-style)
# --------------------------------------------------------------------------

_REGION_TYPES = {"rectangle", "oval", "polygon", "freehand"}


def _outline(record: RoiRecord) -> tuple[np.ndarray, bool]:
    """(vertices ``(N, 2)``, closed) for the ROI's outline."""
    if record.type in _REGION_TYPES:
        pts = np.asarray(region_outline_points(record), dtype=float)
        return (pts[:, :2] if pts.size else np.empty((0, 2))), True
    pts = np.asarray(_points_2d(record), dtype=float)
    closed = bool(record.geometry.get("closed", False))
    return (pts[:, :2] if pts.size else np.empty((0, 2))), closed


def spline_fit(record: RoiRecord, *, n_points: int = 200) -> RoiRecord:
    """ImageJ ``fitSpline``: a smooth cubic spline through the ROI outline
    vertices → a many-vertex freehand/polyline (closed if the source was)."""
    pts, closed = _outline(record)
    if pts.shape[0] < 4:
        raise ValueError("spline fit needs at least 4 outline vertices")
    from scipy.interpolate import splev, splprep

    x, y = pts[:, 0], pts[:, 1]
    if closed:
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]
    tck, _u = splprep([x, y], s=0, per=1 if closed else 0, k=3)
    unew = np.linspace(0.0, 1.0, int(max(n_points, 8)))
    xs, ys = splev(unew, tck)
    poly = [[float(a), float(b)] for a, b in zip(xs, ys)]
    return _derive(record, "freehand" if closed else "freehand_line",
                   {"points": poly, "closed": closed})


def interpolate_outline(record: RoiRecord, *, interval_nm: float) -> RoiRecord:
    """ImageJ ``interpolate``: resample the ROI outline at ~``interval_nm``
    equal arc-length spacing → an evenly-spaced polygon/polyline."""
    pts, closed = _outline(record)
    if pts.shape[0] < 2:
        raise ValueError("interpolate needs an outline with at least 2 vertices")
    ring = np.vstack([pts, pts[0]]) if closed else pts
    seg = np.linalg.norm(np.diff(ring, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 0.0:
        raise ValueError("degenerate outline (zero length)")
    n = int(max(round(total / max(float(interval_nm), 1e-6)), 3))
    targets = np.linspace(0.0, total, n, endpoint=not closed)
    xs = np.interp(targets, cum, ring[:, 0])
    ys = np.interp(targets, cum, ring[:, 1])
    poly = [[float(a), float(b)] for a, b in zip(xs, ys)]
    return _derive(record, "polygon" if closed else "polyline",
                   {"points": poly, "closed": closed})
