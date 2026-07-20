"""
minflux_viewer.core.min_enclosing
==================================
Minimum-enclosing shape fits for point sets — the geometry behind the ROI
conversions **to Rectangle** (minimum-area *oriented* rectangle) and **to Ellipse**
(minimum-area enclosing ellipse). Pure NumPy/SciPy, Qt-free, unit-tested.

Deliberately kept **dimension-agnostic where the algorithm allows**, so new
enclosing shapes / 3-D variants slot in here:

* :func:`min_area_rectangle` — 2-D, **rotating calipers** on the convex hull. The
  optimal minimum-area rectangle always has one side flush with a hull edge, so we
  test each hull edge (O(h·n), h = hull size). This is the standard fast/exact
  method. (A 3-D minimum-*volume* oriented box would live alongside it — that's the
  O(n³)-ish hard case; PCA is the cheap approximation.)
* :func:`min_enclosing_ellipsoid` — **N-D** (Khachiyan's algorithm). The 2-D ellipse
  fit therefore already generalises unchanged to a 3-D min-volume enclosing
  ellipsoid — call it with 3-D points and read the 3×3 shape matrix.
"""

from __future__ import annotations

import numpy as np


def convex_hull_vertices(points) -> np.ndarray:
    """Ordered convex-hull vertices ``(M, 2)`` of 2-D *points*; returns the input
    (deduplicated order) for < 3 points or a degenerate/collinear set."""
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] < 2:
        return np.empty((0, 2))
    P = P[:, :2]
    if P.shape[0] < 3:
        return P.copy()
    try:
        from scipy.spatial import ConvexHull

        return P[ConvexHull(P).vertices]
    except Exception:
        return P.copy()


def min_area_rectangle(points):
    """Minimum-area **oriented** enclosing rectangle of 2-D *points*.

    Returns ``(center(2,), width, height, angle_deg)`` — *width* is the box's local-x
    extent, *angle* the CCW rotation of that axis (deg), matching the rectangle/oval
    geometry convention (``bounds`` centre = rotation centre)."""
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[0] == 0:
        return np.zeros(2), 0.0, 0.0, 0.0
    P = P[:, :2]
    hull = convex_hull_vertices(P)
    if hull.shape[0] < 2:
        c = hull.mean(axis=0) if hull.shape[0] else np.zeros(2)
        return np.asarray(c, float), 0.0, 0.0, 0.0
    best = None                                   # (area, cx, cy, w, h, angle)
    n = hull.shape[0]
    for i in range(n):
        edge = hull[(i + 1) % n] - hull[i]
        length = float(np.hypot(edge[0], edge[1]))
        if length < 1e-12:
            continue
        ex = edge / length                        # box local x (along the hull edge)
        ey = np.array([-ex[1], ex[0]])            # box local y (normal)
        pe = hull @ ex
        pn = hull @ ey
        emin, emax = float(pe.min()), float(pe.max())
        nmin, nmax = float(pn.min()), float(pn.max())
        w, h = emax - emin, nmax - nmin
        area = w * h
        if best is None or area < best[0]:
            center = ex * ((emin + emax) / 2.0) + ey * ((nmin + nmax) / 2.0)
            angle = float(np.degrees(np.arctan2(ex[1], ex[0])))
            best = (area, float(center[0]), float(center[1]), float(w), float(h), angle)
    _area, cx, cy, w, h, angle = best
    return np.array([cx, cy]), w, h, angle


def min_enclosing_ellipsoid(points, *, tol: float = 1e-3, max_iter: int = 200):
    """Minimum-volume enclosing ellipsoid (**Khachiyan's algorithm**) of *points*
    ``(N, d)``.

    Returns ``(center(d,), A(d, d))`` where the ellipsoid is
    ``{x : (x - center)ᵀ A (x - center) ≤ 1}``. Works in any dimension — 2-D yields
    the minimum-area enclosing ellipse, 3-D the minimum-volume enclosing ellipsoid.
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[0] == 0:
        return np.zeros(max(P.shape[1], 1) if P.ndim == 2 else 1), np.eye(1)
    N, d = P.shape
    if N <= d:                                    # too few points → enclosing sphere
        c = P.mean(axis=0)
        r = max(float(np.max(np.linalg.norm(P - c, axis=1))) if N else 1.0, 1e-9)
        return c, np.eye(d) / (r * r)
    Q = np.vstack([P.T, np.ones(N)])              # (d+1, N)
    u = np.full(N, 1.0 / N)
    for _ in range(max_iter):
        X = Q @ (u[:, None] * Q.T)                # (d+1, d+1)
        try:
            M = np.einsum("ij,ij->i", Q.T @ np.linalg.inv(X), Q.T)  # diag(Qᵀ X⁻¹ Q)
        except np.linalg.LinAlgError:
            break
        j = int(np.argmax(M))
        max_m = float(M[j])
        step = (max_m - d - 1.0) / ((d + 1.0) * (max_m - 1.0))
        if not np.isfinite(step) or step <= 0.0:
            break
        new_u = (1.0 - step) * u
        new_u[j] += step
        err = float(np.linalg.norm(new_u - u))
        u = new_u
        if err < tol:
            break
    c = P.T @ u
    cov = P.T @ (u[:, None] * P) - np.outer(c, c)
    try:
        A = np.linalg.inv(cov) / float(d)
    except np.linalg.LinAlgError:
        A = np.eye(d)
    # Guarantee enclosure: Khachiyan converges only to a tolerance, so scale A so the
    # outermost point lands exactly on the boundary ((x-c)ᵀA(x-c) ≤ 1 for all points).
    diff = P - c
    qform = np.einsum("ij,jk,ik->i", diff, A, diff)
    m = float(qform.max()) if qform.size else 0.0
    if m > 1.0:
        A = A / m
    return c, A


def ellipse_from_matrix(center, A):
    """2-D ellipse ``(cx, cy, width, height, angle_deg)`` from a Khachiyan shape
    matrix *A* (ellipse local x = major axis; width/height are full axis lengths)."""
    center = np.asarray(center, dtype=float).ravel()
    evals, evecs = np.linalg.eigh(np.asarray(A, dtype=float))
    evals = np.clip(evals, 1e-18, None)
    radii = 1.0 / np.sqrt(evals)                  # semi-axis lengths (= 1/√eigenvalue)
    order = np.argsort(radii)[::-1]               # major axis first
    major = evecs[:, order[0]]
    angle = float(np.degrees(np.arctan2(major[1], major[0])))
    return (float(center[0]), float(center[1]),
            2.0 * float(radii[order[0]]), 2.0 * float(radii[order[1]]), angle)


def min_enclosing_ellipse(points, *, tol: float = 1e-3, max_iter: int = 200):
    """2-D convenience: minimum-area enclosing ellipse as
    ``(cx, cy, width, height, angle_deg)``."""
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[0] == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    c, A = min_enclosing_ellipsoid(P[:, :2], tol=tol, max_iter=max_iter)
    return ellipse_from_matrix(c, A)


def axis_aligned_enclosing_ellipse(points):
    """Axis-aligned ellipse enclosing 2-D *points* → ``(cx, cy, width, height)``
    (angle is always 0).

    Centred on the axis-aligned bounding box, at the box's aspect ratio, then scaled
    by the largest radial factor so every point lies on/inside the ellipse. This is
    the **exact minimum-area** axis-aligned enclosing ellipse for box- or
    ellipse-shaped point sets (the common case — e.g. an E-coli outline); for
    irregular sets it is a tight enclosing ellipse at the bounding-box aspect. The
    axis-aligned counterpart of :func:`min_enclosing_ellipse` (which is oriented)."""
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[0] == 0:
        return 0.0, 0.0, 0.0, 0.0
    P = P[:, :2]
    lo = P.min(axis=0)
    hi = P.max(axis=0)
    cx, cy = (lo + hi) / 2.0
    sx = max((hi[0] - lo[0]) / 2.0, 1e-9)
    sy = max((hi[1] - lo[1]) / 2.0, 1e-9)
    q = ((P[:, 0] - cx) / sx) ** 2 + ((P[:, 1] - cy) / sy) ** 2
    scale = max(float(np.sqrt(q.max())), 1e-9)
    return float(cx), float(cy), 2.0 * sx * scale, 2.0 * sy * scale
