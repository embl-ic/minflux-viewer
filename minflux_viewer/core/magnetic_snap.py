"""
minflux_viewer.core.magnetic_snap
==================================
Pixel-domain "magnetic" snapping for the magnetic-lasso ROI tool.

Given a density raster (the render view's scalar tile) and a cursor pixel,
:func:`snap_density_centroid` mean-shifts the cursor toward the local
**intensity-weighted centroid** — i.e. the high-density centre of the structure
under the cursor. A few iterations climb onto the ridge / centreline of a
curvilinear feature. Optionally the move can be constrained to the direction
*perpendicular* to travel so the path hugs a filament's centreline instead of
drifting to a brighter blob.

Pure NumPy and Qt-free, so it stays fast (a few tiny array slices per hover) and
unit-testable. The render window converts cursor nm ↔ pixel and supplies the
raster; this module never touches data coordinates.
"""

from __future__ import annotations

import numpy as np


def snap_density_centroid(
    image,
    col: float,
    row: float,
    *,
    window: int = 7,
    iterations: int = 3,
    lateral_dir=None,
):
    """Mean-shift pixel ``(col, row)`` to the local intensity centroid of *image*.

    Parameters
    ----------
    image : 2-D array (row, col) of densities (row = y, col = x).
    col, row : starting pixel (floats allowed).
    window : odd-ish full side length of the square survey window (uses radius
        ``window // 2``).
    iterations : number of mean-shift steps.
    lateral_dir : optional ``(dx, dy)`` travel direction (pixel space); when
        given, only the component of the shift **perpendicular** to it is applied
        (centreline-hugging).

    Returns ``(col, row)`` snapped (floats). If the window is empty/flat the
    input is returned unchanged.
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2 or img.size == 0:
        return float(col), float(row)
    h, w = img.shape
    radius = max(1, int(window) // 2)
    c, r = float(col), float(row)

    lat = None
    if lateral_dir is not None:
        dx, dy = float(lateral_dir[0]), float(lateral_dir[1])
        norm = (dx * dx + dy * dy) ** 0.5
        if norm > 1e-9:
            lat = (dx / norm, dy / norm)

    for _ in range(max(1, int(iterations))):
        ci, ri = int(round(c)), int(round(r))
        c0, c1 = max(0, ci - radius), min(w, ci + radius + 1)
        r0, r1 = max(0, ri - radius), min(h, ri + radius + 1)
        if c1 <= c0 or r1 <= r0:
            break
        sub = img[r0:r1, c0:c1]
        total = float(sub.sum())
        if total <= 0.0:
            break
        ys, xs = np.mgrid[r0:r1, c0:c1]
        new_c = float((sub * xs).sum() / total)
        new_r = float((sub * ys).sum() / total)
        if lat is not None:
            off_c, off_r = new_c - c, new_r - r
            along = off_c * lat[0] + off_r * lat[1]      # component along travel
            new_c = c + (off_c - along * lat[0])         # keep only perpendicular
            new_r = r + (off_r - along * lat[1])
        moved = abs(new_c - c) + abs(new_r - r)
        c, r = new_c, new_r
        if moved < 1e-3:
            break
    return float(c), float(r)
