"""
minflux_viewer.ui.ortho_rotation
================================
The orthogonal grid's fourth cell: a rotating projection.

The 2×2 pane grid leaves its bottom-right cell empty (XY / YZ / XZ occupy the
other three). A rotating projection is the cheap thing to put there: it is one
matrix multiply on the localizations already in hand, then the same 2-D scatter
the other panes use -- no OpenGL, no volume, no second rendering path.

It answers the question the three fixed projections cannot: *is that shape
actually where it looks like it is?* Three orthogonal silhouettes are ambiguous
about depth ordering (the visual-hull problem in miniature); a few degrees of
rotation resolves it immediately.

⚠ Deliberately a **projection**, not a 3-D view. It spins the data about the
vertical screen axis and re-projects, so it stays in the same coordinate frame
and the same nm/px as its neighbours. A real 3-D camera would need its own
navigation, its own lighting and its own idea of scale, none of which the
orthogonal view has or wants.
"""

from __future__ import annotations

import numpy as np

__all__ = ["rotated_projection", "rotation_axis_labels", "ROTATION_AXES"]

#: Which data axes the rotation mixes, per named mode. The third axis is the
#: one held vertical on screen, so the view stays comparable to its neighbours.
ROTATION_AXES: dict[str, tuple[int, int, int]] = {
    # name        mixed-a  mixed-b  held-vertical
    "about Y": (0, 2, 1),      # X and Z turn, Y stays up  (XY -> ZY)
    "about X": (1, 2, 0),      # Y and Z turn, X stays up
    "about Z": (0, 1, 2),      # X and Y turn, Z stays up
}


def rotation_axis_labels(mode: str) -> tuple[str, str]:
    """Bottom/left labels that explain a rotating projection's coordinates."""
    try:
        a, b, up = ROTATION_AXES[mode]
    except KeyError:
        raise ValueError(
            f"unknown rotation mode {mode!r}; expected one of {sorted(ROTATION_AXES)}"
        ) from None
    names = "XYZ"
    return (
        f"{names[a]} cos θ + {names[b]} sin θ (nm)",
        f"{names[up]} — rotation axis (nm)",
    )


def rotated_projection(xyz, angle_deg: float, mode: str = "about Y", *, centre=None):
    """``(h, v)`` screen coordinates for *xyz* spun by *angle_deg*.

    At 0° this reproduces the pane's neighbour exactly (``about Y`` gives X
    horizontal, Y vertical -- the XY view), so the rotation reads as a
    continuous departure from a known picture rather than a new one.

    *centre* is the 3-D point the spin turns about, in the same coordinates as
    *xyz*. ⚠ Without it the rotation is about the coordinate ORIGIN, and MINFLUX
    coordinates sit tens of microns from it -- so a few degrees swung the whole
    cloud clean off the pane. Passing the crosshair keeps the marked feature
    fixed while everything around it turns, and the 0° identity is unaffected
    (the centre maps to itself at every angle).
    """
    pts = np.asarray(xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return None
    try:
        a, b, up = ROTATION_AXES[mode]
    except KeyError:
        raise ValueError(
            f"unknown rotation mode {mode!r}; expected one of {sorted(ROTATION_AXES)}"
        ) from None
    theta = np.deg2rad(float(angle_deg))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # Only the horizontal axis is a mix; the vertical one is held so the view
    # stays comparable with the fixed panes beside it.
    origin_a = origin_b = 0.0
    if centre is not None:
        pivot = np.asarray(centre, dtype=float).ravel()
        if pivot.size >= 3 and np.all(np.isfinite(pivot[:3])):
            origin_a, origin_b = float(pivot[a]), float(pivot[b])
    horizontal = (origin_a
                  + (pts[:, a] - origin_a) * cos_t
                  + (pts[:, b] - origin_b) * sin_t)
    return horizontal, pts[:, up]
