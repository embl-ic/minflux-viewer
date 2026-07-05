"""Add / delete a vertex on a polyline-family ROI (pure geometry).

Hit-testing is done in the *projected* (view-plane) 2-D coordinates the user
sees; the returned edge index + parameter ``t`` let the caller edit the stored
(possibly 3-D) vertex list. Projection is linear (it selects two axes), so
interpolating the stored points at the same ``t`` puts the new vertex exactly on
the projected edge and interpolates any out-of-plane depth coordinate correctly.

These helpers are shape-agnostic: they take a plain list of vertices and never
touch Qt or the ROI store, so they are unit-tested in isolation.
"""

from __future__ import annotations

import numpy as np


def min_vertices(closed: bool) -> int:
    """Fewest vertices a shape may keep: 3 for a closed region, 2 for an open line."""
    return 3 if closed else 2


def nearest_vertex(projected, mouse) -> tuple[int, float]:
    """Return (index, distance) of the projected vertex closest to ``mouse``."""
    pts = np.asarray(projected, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
        return -1, float("inf")
    d = np.hypot(pts[:, 0] - float(mouse[0]), pts[:, 1] - float(mouse[1]))
    i = int(np.argmin(d))
    return i, float(d[i])


def _project_on_segment(a, b, p) -> tuple[float, float]:
    """Clamped projection of ``p`` onto segment a→b → (t in [0,1], squared distance)."""
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    px, py = float(p[0]), float(p[1])
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-30:
        t = 0.0
    else:
        t = ((px - ax) * dx + (py - ay) * dy) / denom
        t = min(1.0, max(0.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return t, (cx - px) ** 2 + (cy - py) ** 2


def nearest_edge(projected, mouse, *, closed: bool) -> tuple[int, float]:
    """Return (seg_start_index, t) of the edge whose nearest point to ``mouse`` is
    closest, with the clamped parameter ``t`` along it. Edges are (i → i+1); a
    closed shape also considers the closing edge (n-1 → 0), for which
    ``seg_start_index == n-1`` (i.e. "insert at the end")."""
    pts = np.asarray(projected, dtype=float)
    n = pts.shape[0]
    if pts.ndim != 2 or n < 2:
        return -1, 0.0
    edges = list(range(n - 1))
    if closed:
        edges.append(n - 1)  # closing edge n-1 -> 0
    best_i, best_t, best_d = edges[0], 0.0, float("inf")
    for i in edges:
        j = (i + 1) % n
        t, d = _project_on_segment(pts[i], pts[j], mouse)
        if d < best_d:
            best_i, best_t, best_d = i, t, d
    return best_i, best_t


def insert_vertex(points, seg_start_index: int, t: float) -> list[list[float]]:
    """Insert an interpolated vertex on the edge after ``seg_start_index`` (wrapping
    for the closing edge). Vertices may be 2-D or 3-D; depth is interpolated too."""
    pts = [list(map(float, p)) for p in points]
    n = len(pts)
    if n < 2 or not (0 <= seg_start_index < n):
        return pts
    a = pts[seg_start_index]
    b = pts[(seg_start_index + 1) % n]
    m = min(len(a), len(b))
    new = [a[k] + t * (b[k] - a[k]) for k in range(m)]
    pts.insert(seg_start_index + 1, new)
    return pts


def delete_vertex(points, index: int) -> list[list[float]]:
    """Return a copy of ``points`` with vertex ``index`` removed."""
    pts = [list(map(float, p)) for p in points]
    if 0 <= index < len(pts):
        del pts[index]
    return pts
