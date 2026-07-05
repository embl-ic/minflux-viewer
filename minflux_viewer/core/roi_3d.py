"""3-D ROI as an intersection of per-view 2-D extrusions.

A 3-D ROI is defined by one or more 2-D shapes, each drawn in an orthogonal
view (XY / XZ / YZ) and extruded along the axis perpendicular to that view.
A localization is inside the 3-D ROI iff its projection lies inside *every*
constraint shape — the region is the intersection of the extruded prisms
("sculpt from orthogonal views"):

    * draw an XY outline  -> bounds X and Y  (extruded along Z),
    * add an XZ outline   -> bounds X and Z  (extruded along Y),
    * add a  YZ outline   -> bounds Y and Z  (extruded along X).

A depth slab is just a rectangle drawn in XZ or YZ, so no separate depth model
is needed. The 2-D membership per plane reuses ``roi_selection.roi_region_mask``
(rectangle / oval / polygon / freehand), so this stays pure and Qt-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .roi_selection import roi_region_mask

# Each ortho view spans two data axes: (horizontal, vertical). This matches the
# conv-3D orthoviewer panes (the YZ pane draws Z horizontally, Y vertically), so
# a shape's stored (x, y) geometry maps back to the right data axes.
PLANE_AXES: dict[str, tuple[int, int]] = {
    "XY": (0, 1),
    "XZ": (0, 2),
    "YZ": (2, 1),
}


def plane_axes(plane: str) -> tuple[int, int]:
    """Return the (horizontal, vertical) data-axis indices for an ortho plane."""
    try:
        return PLANE_AXES[str(plane).upper()]
    except (KeyError, AttributeError):
        raise ValueError(
            f"unknown ortho plane {plane!r}; expected one of {sorted(PLANE_AXES)}"
        ) from None


@dataclass
class Roi3DConstraint:
    """One 2-D shape drawn in an ortho view, extruded along its normal axis.

    Duck-types a ``RoiRecord`` for ``roi_region_mask`` (it reads ``.type`` and
    ``.geometry``), so it can be passed straight through as the mask's record.
    """

    plane: str                      # "XY" | "XZ" | "YZ"
    type: str                       # "rectangle" | "oval" | "polygon" | "freehand"
    geometry: dict[str, Any] = field(default_factory=dict)


def roi3d_mask(points: np.ndarray, constraints) -> np.ndarray:
    """Boolean mask over ``points`` (N, 3) selecting rows inside the 3-D ROI.

    The result is the logical AND of each constraint's 2-D region mask, taken on
    that constraint's projected axes (intersection of extrusions). With **no**
    constraints nothing is selected (all-False) — an empty ROI must not silently
    select the whole dataset.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return np.zeros(0, dtype=bool)
    if pts.shape[1] == 2:  # promote to XYZ so XZ/YZ constraints are well-defined
        pts = np.column_stack([pts, np.zeros(pts.shape[0], dtype=float)])
    n = pts.shape[0]
    if not constraints:
        return np.zeros(n, dtype=bool)
    out = np.ones(n, dtype=bool)
    for c in constraints:
        h, v = plane_axes(c.plane)
        out &= roi_region_mask(pts[:, h], pts[:, v], c)
    return out
