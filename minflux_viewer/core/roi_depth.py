"""
minflux_viewer.core.roi_depth
=============================
Data-aware out-of-plane (depth) placement for ROI points / vertices.

When a point or polyline/polygon vertex is drawn in a 2-D view of a 3-D dataset,
its out-of-plane (depth) coordinate should land on the actual structure at that
in-plane location rather than the static midpoint of the view's depth range. The
2-D view is a max-projection, so it carries no depth information; we recover it
from the localizations: for the clicked in-plane (a, b), gather nearby
localizations and take the **XY-proximity-weighted median** of their depth
coordinate. A dense Z slice therefore "drags" the vertex onto it.

If no localizations fall within the radius (empty XY column) the function returns
``None`` so the caller can fall back to the static range centre.

Pure NumPy/SciPy and Qt-free → unit-testable. The depth is computed once per
drawn vertex (not per hover), so a one-off KD-tree build for many vertices is
cheap.
"""

from __future__ import annotations

import numpy as np

#: default search radius (nm) around the clicked in-plane location
DEFAULT_RADIUS_NM = 60.0


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v = np.asarray(values, dtype=float)[order]
    w = np.asarray(weights, dtype=float)[order]
    total = float(w.sum())
    if total <= 0.0:
        return float(np.median(v))
    cw = np.cumsum(w)
    idx = int(np.searchsorted(cw, 0.5 * total))
    return float(v[min(idx, v.size - 1)])


def weighted_depths(
    points,
    in_a,
    in_b,
    depth,
    *,
    radius_nm: float = DEFAULT_RADIUS_NM,
    sigma_nm: float | None = None,
) -> list:
    """Per-point out-of-plane value from the local depth distribution.

    For each ``(a, b)`` in *points* (the in-plane coordinates of the clicked
    vertex), return the XY-proximity-weighted median of *depth* over the
    localizations whose ``(in_a, in_b)`` lie within ``radius_nm`` — or ``None``
    if none do. Weights are a Gaussian of in-plane distance (``sigma_nm`` defaults
    to ``radius_nm / 2``).
    """
    in_a = np.asarray(in_a, dtype=float).ravel()
    in_b = np.asarray(in_b, dtype=float).ravel()
    depth = np.asarray(depth, dtype=float).ravel()
    finite = np.isfinite(in_a) & np.isfinite(in_b) & np.isfinite(depth)
    in_a, in_b, depth = in_a[finite], in_b[finite], depth[finite]
    pts = [(float(a), float(b)) for a, b in points]
    if in_a.size == 0 or not pts:
        return [None] * len(pts)

    radius_nm = float(radius_nm)
    sigma = float(sigma_nm) if sigma_nm else radius_nm * 0.5
    two_sigma2 = 2.0 * sigma * sigma

    # KD-tree only pays off for many query points; brute-force a few.
    tree = None
    if len(pts) > 8:
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(np.column_stack([in_a, in_b]))
        except Exception:
            tree = None

    out: list = []
    r2 = radius_nm * radius_nm
    for q, (a, b) in enumerate(pts):
        if tree is not None:
            idx = np.asarray(tree.query_ball_point((a, b), r=radius_nm), dtype=int)
        else:
            idx = np.flatnonzero((in_a - a) ** 2 + (in_b - b) ** 2 <= r2)
        if idx.size == 0:
            out.append(None)
            continue
        d2 = (in_a[idx] - a) ** 2 + (in_b[idx] - b) ** 2
        w = np.exp(-d2 / two_sigma2)
        out.append(_weighted_median(depth[idx], w))
    return out
