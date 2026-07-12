"""
minflux_viewer.analysis.plot_profile
=====================================
Line-profile sampling for the ImageJ/Fiji-style **Plot Profile** tool.

`line_profile_from_locs` samples **the localizations directly** (not the rendered
raster): each localization is assigned to the nearest segment of a straight /
poly / freehand line, kept if it lies within ±width/2 perpendicular of the line,
and binned by its arc-length position from the line's start (nm). The profile is
therefore decided **purely by the data** — independent of the render viewport /
zoom / LOD / LUT — and works the same on a render or a scatter view. `band_polygon`
draws the perpendicular width band; `resample_path` / `path_cumlen` are the shared
polyline geometry.

Pure NumPy, Qt-free, unit-tested. Broad-phase bbox culling keeps it fast; for a
straight line the per-segment loop is a single vectorised pass.
"""

from __future__ import annotations

import numpy as np


def path_cumlen(path) -> np.ndarray:
    """Cumulative arc length (nm) at each vertex of an ``(K, 2)`` polyline; the
    last element is the total length."""
    p = np.asarray(path, dtype=float)
    if p.ndim != 2 or p.shape[0] < 2:
        return np.zeros(max(p.shape[0], 1))
    seg = np.sqrt(((p[1:] - p[:-1]) ** 2).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample_path(path, step_nm: float):
    """Resample an ``(K, 2)`` polyline at ~``step_nm`` arc-length.

    Returns ``(points (N, 2), tangents (N, 2) unit, dist (N,))`` where ``dist`` is
    the cumulative distance (nm) from the path start."""
    p = np.asarray(path, dtype=float)
    if p.ndim != 2 or p.shape[0] < 2:
        pts = p.reshape(-1, 2)[:1] if p.size else np.zeros((0, 2))
        return pts, np.zeros_like(pts), np.zeros(pts.shape[0])
    cum = path_cumlen(p)
    total = float(cum[-1])
    if total <= 0:
        return p[:1].copy(), np.array([[1.0, 0.0]]), np.zeros(1)
    step = max(float(step_nm), 1e-9)
    n = max(int(np.ceil(total / step)) + 1, 2)
    dist = np.linspace(0.0, total, n)
    seg_idx = np.clip(np.searchsorted(cum, dist, side="right") - 1, 0, p.shape[0] - 2)
    pts = np.empty((n, 2))
    tan = np.empty((n, 2))
    for i, (d, si) in enumerate(zip(dist, seg_idx)):
        a, b = p[si], p[si + 1]
        seg_len = cum[si + 1] - cum[si]
        t = 0.0 if seg_len <= 0 else (d - cum[si]) / seg_len
        pts[i] = a + t * (b - a)
        v = b - a
        nv = float(np.hypot(v[0], v[1]))
        tan[i] = v / nv if nv > 0 else np.array([1.0, 0.0])
    return pts, tan, dist


def _normals(tan: np.ndarray) -> np.ndarray:
    """Unit normals (rotate each unit tangent by +90°)."""
    return np.column_stack([-tan[:, 1], tan[:, 0]])


def _n_bins(total: float, bin_nm: float | None, n_bins: int) -> int:
    if bin_nm and bin_nm > 0:
        return max(1, int(round(total / float(bin_nm))))
    return max(1, int(n_bins))


def line_profile_from_locs(
    locs_xy,
    path,
    width_nm: float,
    *,
    bin_nm: float | None = None,
    n_bins: int = 256,
):
    """Line-intensity profile sampled **directly from the localizations**.

    Each localization is assigned to the nearest segment of *path*, kept when it
    is within ``±width_nm/2`` perpendicular of the line, and binned by its
    arc-length position from the line's start. The result depends only on the
    data (not on any render viewport / zoom).

    Parameters
    ----------
    locs_xy : (N, 2) float
        Localization coordinates in the same (plane) nm frame as *path*.
    path : (K, 2) array
        Line / polyline / freehand vertices (nm).
    width_nm : float
        Full width of the perpendicular integration band (must be > 0 — a
        localization profile needs a finite band to catch points).
    bin_nm : float, optional
        Along-line bin size (nm). If omitted, ``n_bins`` uniform bins are used.
    n_bins : int
        Number of along-line bins when ``bin_nm`` is not given.

    Returns
    -------
    ``(centres_nm (B,), counts (B,), bin_width_nm)`` — bin-centre distance from the
    start, localization counts per bin, and the uniform bin width. Empty arrays
    when the line is degenerate or the width is not positive.
    """
    locs = np.asarray(locs_xy, dtype=float)
    p = np.asarray(path, dtype=float)
    if p.ndim != 2 or p.shape[0] < 2:
        return np.zeros(0), np.zeros(0), 0.0
    p = p[:, :2]
    seg_vec = p[1:] - p[:-1]                                   # (K, 2)
    seg_len = np.hypot(seg_vec[:, 0], seg_vec[:, 1])           # (K,)
    good = seg_len > 1e-9
    half_w = max(float(width_nm), 0.0) / 2.0
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if not good.any() or half_w <= 0.0 or total <= 0.0:
        return np.zeros(0), np.zeros(0), 0.0

    n = _n_bins(total, bin_nm, n_bins)
    edges = np.linspace(0.0, total, n + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    bin_w = total / n

    def _empty_counts():
        return centres, np.zeros(n, dtype=float), bin_w

    if locs.ndim != 2 or locs.shape[0] == 0 or locs.shape[1] < 2:
        return _empty_counts()
    locs = locs[:, :2]

    # Broad-phase: keep only localizations inside the path bbox expanded by the
    # half-width (a straight/short line then touches only a thin strip of data).
    lo = p.min(axis=0) - half_w
    hi = p.max(axis=0) + half_w
    m = ((locs[:, 0] >= lo[0]) & (locs[:, 0] <= hi[0])
         & (locs[:, 1] >= lo[1]) & (locs[:, 1] <= hi[1]))
    cand = locs[m]
    if cand.shape[0] == 0:
        return _empty_counts()

    # Nearest-segment assignment: track each candidate's minimum (clamped) distance
    # to the polyline and the arc position of the foot of that perpendicular. A
    # per-segment bbox cull keeps the O(N·K) work near O(N) — the expensive
    # projection runs only on the locals within ±half_w of each segment (so a long
    # wavy freehand over the whole field doesn't project every point per segment).
    best_d = np.full(cand.shape[0], np.inf)
    best_arc = np.zeros(cand.shape[0])
    cx, cy = cand[:, 0], cand[:, 1]
    for k in np.flatnonzero(good):
        a = p[k]
        b = p[k + 1]
        v = seg_vec[k]
        xlo, xhi = min(a[0], b[0]) - half_w, max(a[0], b[0]) + half_w
        ylo, yhi = min(a[1], b[1]) - half_w, max(a[1], b[1]) + half_w
        idx = np.flatnonzero((cx >= xlo) & (cx <= xhi) & (cy >= ylo) & (cy <= yhi))
        if idx.size == 0:
            continue
        sub = cand[idx]
        t = np.clip(((sub - a) @ v) / (seg_len[k] ** 2), 0.0, 1.0)
        foot = a + t[:, None] * v
        d = np.hypot(sub[:, 0] - foot[:, 0], sub[:, 1] - foot[:, 1])
        better = d < best_d[idx]
        gi = idx[better]
        best_d[gi] = d[better]
        best_arc[gi] = cum[k] + t[better] * seg_len[k]

    keep = best_d <= half_w
    counts, _ = np.histogram(best_arc[keep], bins=edges)
    return centres, counts.astype(float), bin_w


def band_polygon(path, width_nm: float, step_nm: float) -> np.ndarray:
    """Closed polygon ``(M, 2)`` outlining the perpendicular width band around a
    (poly)line — used to draw the translucent width indicator. Empty when the
    width is not positive."""
    if width_nm is None or width_nm <= 0:
        return np.zeros((0, 2))
    pts, tan, _ = resample_path(path, max(float(step_nm), 1e-9))
    if pts.shape[0] < 2:
        return np.zeros((0, 2))
    normals = _normals(tan)
    half = float(width_nm) / 2.0
    left = pts + half * normals
    right = pts - half * normals
    return np.vstack([left, right[::-1]])
