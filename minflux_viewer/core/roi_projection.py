"""
minflux_viewer.core.roi_projection
==================================
How a ROI drawn in one plane appears in another.

Pure NumPy, Qt-free.

In the orthogonal view a ROI has to be visible in all three panes, not only the
one it was drawn in — otherwise a ROI is invisible in two of the three views
that are meant to show the same region, and a user cannot tell "no ROI here"
from "the ROI is somewhere off this pane".

A **2-D** ROI has no extent on the axis it was drawn against, so in a pane that
shows that axis it is genuinely degenerate: a line (when the pane shares one of
its in-plane axes) or a point (when it shares neither). That is not a limitation
to be papered over — it is the honest picture, and it is the visual difference
between a flat ROI and a volume one:

    a rectangle drawn in XY, seen in XZ  ->  a horizontal segment at z = depth
    the same rectangle,       seen in YZ  ->  a horizontal segment at z = depth
    a cuboid,                 seen in XZ  ->  a filled box with real Z extent

⚠ The degenerate projection must be drawn **differently** (dashed, say) from a
real outline. A flat ROI at its depth and a volume ROI of zero thickness would
otherwise look identical, and a user would reasonably believe the flat one
constrains Z — which it does not. :data:`DEGENERATE` says which case a caller
got back so it can style accordingly.

Everything here is expressed in **data-axis columns** (0=X, 1=Y, 2=Z), never
plane names: a standalone YZ projection draws Y horizontally while an ortho YZ
pane draws Z horizontally, so a plane name does not say which convention a
geometry is in. See :mod:`minflux_viewer.core.roi_volume`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "FULL",
    "DEGENERATE_LINE",
    "DEGENERATE_POINT",
    "PLANE_COLUMNS",
    "flat_extent",
    "project_flat_record",
]

#: What :func:`project_flat_record` returned, so the caller can style it.
FULL = "full"                       #: the ROI's own outline, drawn as usual
DEGENERATE_LINE = "line"            #: edge-on: extent on one pane axis only
DEGENERATE_POINT = "point"          #: neither pane axis is in the ROI's plane

#: The two data-axis columns a **standalone** 2-D projection shows. Matches
#: ``roi_overlay._PLANE_PLOT_AXES`` — the convention a drawn record's geometry
#: is stored in.
PLANE_COLUMNS: dict[str, tuple[int, int]] = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}


def _points_of(record) -> np.ndarray | None:
    """In-plane ``(N, 2)`` vertices of a 2-D record, or ``None``."""
    g = getattr(record, "geometry", None) or {}
    if "bounds" in g:
        x, y, w, h = (float(v) for v in g["bounds"])
        # The corners are enough: every consumer here wants an extent, and a
        # rotated rectangle's own corners are what its extent is measured from.
        angle = float(g.get("angle", 0.0) or 0.0)
        corners = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=float)
        if abs(angle) > 1e-9:
            t = np.deg2rad(angle)
            rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            centre = np.array([w / 2.0, h / 2.0])
            corners = (corners - centre) @ rot.T + centre
        return corners + np.array([x, y], dtype=float)
    if "point" in g:
        pt = g["point"]
        return np.array([[float(pt[0]), float(pt[1])]], dtype=float)
    raw = g.get("points") or []
    if not len(raw):
        return None
    return np.array([[float(p[0]), float(p[1])] for p in raw], dtype=float)


def flat_extent(record, origin_plane: str) -> dict[int, tuple[float, float]] | None:
    """``{data-axis column: (lo, hi)}`` for a 2-D record's two in-plane axes.

    The record's geometry is in the coordinates of *origin_plane*, which is the
    projection it was drawn in (``context["view_plane"]``).
    """
    columns = PLANE_COLUMNS.get(str(origin_plane).upper())
    if columns is None:
        return None
    pts = _points_of(record)
    if pts is None or pts.size == 0:
        return None
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    return {columns[0]: (float(lo[0]), float(hi[0])),
            columns[1]: (float(lo[1]), float(hi[1]))}


def project_flat_record(record, h_axis: int, v_axis: int, *,
                        origin_plane: str, depth_value: float | None = None):
    """``(kind, outline)`` for a 2-D *record* as seen on axes ``(h_axis, v_axis)``.

    ``outline`` is ``[[h, v], ...]``. Three cases, and the *kind* is the point of
    the return value — a caller that ignores it will draw a flat ROI exactly like
    a volume one:

    * **FULL** — the pane shows the plane the ROI was drawn in, so its own
      outline is returned unchanged.
    * **DEGENERATE_LINE** — the pane shares one in-plane axis; the ROI has extent
      along it and sits at ``depth_value`` on the other.
    * **DEGENERATE_POINT** — the pane shares neither (only possible for a
      malformed request, since any two of X/Y/Z share an axis with any other).

    ``None`` when the record carries no usable geometry, or when the depth is
    needed and was never recorded — an invented depth would put the ROI at a
    position nothing chose.
    """
    extent = flat_extent(record, origin_plane)
    if extent is None:
        return None

    in_plane = set(extent)
    if h_axis in in_plane and v_axis in in_plane:
        pts = _points_of(record)
        columns = PLANE_COLUMNS[str(origin_plane).upper()]
        if pts is None:
            return None
        # The record's own vertex order, mapped onto however this pane orders
        # the two axes (an ortho YZ pane is the transpose of a standalone one).
        take_h = 0 if columns[0] == h_axis else 1
        take_v = 0 if columns[0] == v_axis else 1
        return FULL, [[float(p[take_h]), float(p[take_v])] for p in pts]

    if depth_value is None:
        return None

    def span(axis: int):
        return extent.get(axis)

    h_span, v_span = span(h_axis), span(v_axis)
    if h_span is None and v_span is None:
        return DEGENERATE_POINT, [[float(depth_value), float(depth_value)]]
    if h_span is None:
        h_span = (float(depth_value), float(depth_value))
    if v_span is None:
        v_span = (float(depth_value), float(depth_value))
    return DEGENERATE_LINE, [[h_span[0], v_span[0]], [h_span[1], v_span[1]]]
