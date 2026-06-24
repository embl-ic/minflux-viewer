"""
minflux_viewer.analysis.hlyb_clustering
========================================
HlyB sub-unit detection and pair-distance analysis — a Python port of
``minflux_hlyb_analysis.m`` (Cigdem / Ddorf MINFLUX data).

Pipeline
--------
1. **Sub-unit detection** (:func:`locate_subunit_centers`) — group localization
   *traces* into protein sub-units:
     * trace centroids (median per ``tid``), with a per-trace localization count
       and a local density (neighbours within ``Dunit/2`` of the centroid);
     * **pass 1** — keep traces with enough localizations *and* local density;
     * **pass 2** — keep traces whose centroid sits on a Laplacian-of-Gaussian
       (LoG) blob of the rendered XY histogram (``> std`` of the LoG map);
     * **pass 3** — DBSCAN (eps ``Dunit/2``, minpts 1) merges nearby surviving
       centroids into one sub-unit; the sub-unit centre is the median of all
       member localizations.
   ``Dunit`` (basic unit diameter) is taken from the config, or auto-estimated
   as ``2 × median(trace size)`` via :func:`estimate_trace_size`.
2. **HlyB structures** (:func:`analyze_hlyb`) — DBSCAN the sub-unit centres
   (eps ``2·d_1b2b/√3``, minpts ``min_unit_count_per_HlyB``) into HlyB structures,
   then measure every within-structure sub-unit **pair distance**.

Pure NumPy/SciPy — Qt-free and unit-testable. Coordinates: ``loc`` is in metres
(raw, z **not** RIMF-baked); the analysis applies ``z_scaling_factor`` to z and
works in nm throughout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HlyBConfig:
    """HlyB analysis parameters (defaults match the MATLAB scripts)."""

    min_loc_per_trace: int = 1
    z_scaling_factor: float = 0.67
    unit_render_pixel_size: float = 2.5
    basic_unit_size_nm: float = 0.0  # 0 → auto from trace size
    min_unit_count_per_HlyB: int = 2
    diameter_distance_1a1b_nm: float = 14.0
    diameter_distance_1a2a_nm: float = 15.0
    diameter_distance_1b2b_nm: float = 23.0
    # 2-D-only: per-E.coli mask + border erosion (Ecoli_dimer_analysis_2D.m).
    border_size_nm: float = 200.0       # shrink each cell mask in by this much
    mask_pixel_size_nm: float = 20.0    # render pixel for the mask histogram
    mask_sigma_px: float = 2.5          # Gaussian smooth (pixels) before binarize
    mask_median_size_px: int = 25       # median filter window (pixels)


def dbscan(points: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    """DBSCAN labels for *points* (MATLAB ``dbscan(X, eps, minpts)`` semantics).

    A point is a *core* point when its eps-neighbourhood (Euclidean, **including
    itself**) holds at least ``min_pts`` points. Returns an ``int`` label per row:
    ``0..K-1`` for clusters, ``-1`` for noise.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return labels
    tree = cKDTree(pts)
    neighbors = tree.query_ball_point(pts, r=float(eps))
    visited = np.zeros(n, dtype=bool)
    cluster = -1
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nbrs = neighbors[i]
        if len(nbrs) < min_pts:
            continue  # leave as noise (may be claimed as a border point below)
        cluster += 1
        labels[i] = cluster
        seeds = list(nbrs)
        k = 0
        while k < len(seeds):
            j = seeds[k]
            k += 1
            if labels[j] == -1:
                labels[j] = cluster  # border point
            if not visited[j]:
                visited[j] = True
                jn = neighbors[j]
                if len(jn) >= min_pts:
                    seeds.extend(jn)
    return labels


def estimate_trace_size(loc_m: np.ndarray, tid: np.ndarray, z_scale: float = 0.67) -> np.ndarray:
    """Three candidate trace "sizes" (nm): geometric-mean, mode-bin and Gaussian
    fit of the per-localization distance-to-trace-centre distribution.

    Mirrors the MATLAB ``estimate_trace_size`` helper; ``2 × median`` of the
    returned candidates is used as the default sub-unit diameter ``Dunit``.
    """
    loc_nm = np.asarray(loc_m, dtype=np.float64) * 1e9
    loc_nm = loc_nm.copy()
    loc_nm[:, 2] *= float(z_scale)
    tid = np.asarray(tid).ravel()
    uid = np.unique(tid)
    dists = []
    for u in uid:
        pts = loc_nm[tid == u]
        if pts.shape[0] == 0:
            continue
        center = np.median(pts, axis=0)
        dists.append(np.linalg.norm(pts - center, axis=1))
    if not dists:
        return np.array([np.nan, np.nan, np.nan])
    dist = np.concatenate(dists)
    dist = dist[dist > 0]
    if dist.size < 3:
        return np.array([np.nan, np.nan, np.nan])
    logdist = np.log(dist)
    s1 = float(np.exp(np.mean(logdist)))
    counts, edges = np.histogram(logdist, bins="auto")
    if counts.size == 0 or counts.max() == 0:
        return np.array([s1, s1, s1])
    maxbin = int(np.argmax(counts))
    s2 = float(np.exp(0.5 * (edges[maxbin] + edges[maxbin + 1])))
    s3 = s1
    try:
        from scipy.optimize import curve_fit

        centers = 0.5 * (edges[:-1] + edges[1:])

        def _gauss(x, a, b, c):
            return a * np.exp(-(((x - b) / c) ** 2))

        p0 = [float(counts.max()), float(centers[maxbin]), max((edges[-1] - edges[0]) / 4.0, 1e-3)]
        popt, _ = curve_fit(_gauss, centers, counts.astype(float), p0=p0, maxfev=5000)
        s3 = float(np.exp(popt[1]))
    except Exception:
        s3 = s1
    return np.array([s1, s2, s3])


def _fspecial_log(hsize: int, sigma: float) -> np.ndarray:
    """Laplacian-of-Gaussian kernel matching MATLAB ``fspecial('log', n, sigma)``."""
    siz = (hsize - 1) / 2.0
    std2 = sigma ** 2
    ax = np.arange(-siz, siz + 1)
    x, y = np.meshgrid(ax, ax)
    arg = -(x * x + y * y) / (2.0 * std2)
    h = np.exp(arg)
    eps = np.finfo(float).eps
    h[h < eps * h.max()] = 0.0
    sumh = h.sum()
    if sumh != 0:
        h = h / sumh
    h1 = h * (x * x + y * y - 2.0 * std2) / (std2 ** 2)
    return h1 - h1.sum() / (hsize * hsize)


def locate_subunit_centers(
    loc_m: np.ndarray,
    tid: np.ndarray,
    *,
    z_scale: float = 0.67,
    pixel_size: float = 2.5,
    min_loc_per_trace: int = 1,
    basic_unit_size_nm: float = 0.0,
) -> dict:
    """Detect sub-unit centres (nm) from MINFLUX traces. See module docstring.

    Returns a dict with ``unit_centers`` (K×3 nm, z-scaled), ``unit_traces``
    (list of trace-ID arrays per unit), ``points_nm`` (all localizations, z
    scaled, for plotting), ``dunit_nm`` and pass counts.
    """
    from scipy.spatial import cKDTree

    loc_m = np.asarray(loc_m, dtype=np.float64)
    tid = np.asarray(tid).ravel()

    dunit = float(basic_unit_size_nm or 0.0)
    if dunit <= 0.0:
        dunit = 2.0 * float(np.median(estimate_trace_size(loc_m, tid, z_scale)))
    if not np.isfinite(dunit) or dunit <= 0.0:
        dunit = 10.0  # safe fallback when trace-size estimation degenerates

    xnm = loc_m[:, 0] * 1e9
    ynm = loc_m[:, 1] * 1e9
    znm = loc_m[:, 2] * 1e9 * float(z_scale)
    points = np.column_stack([xnm, ynm, znm])

    empty = {
        "unit_centers": np.empty((0, 3)),
        "unit_traces": [],
        "points_nm": points,
        "dunit_nm": dunit,
        "n_traces": 0,
        "n_pass1": 0,
        "n_pass2": 0,
    }
    if points.shape[0] == 0:
        return empty

    uid, inv = np.unique(tid, return_inverse=True)
    num_loc_per_trace = np.bincount(inv, minlength=uid.size)
    center_trace = np.array([np.median(points[tid == u], axis=0) for u in uid])

    # pass 1: localization count + local density around the trace centroid
    tree = cKDTree(points)
    nbr = tree.query_ball_point(center_trace, dunit / 2.0)
    density = np.array([len(x) - 1 for x in nbr])
    keep1 = (num_loc_per_trace >= min_loc_per_trace) & (density >= min_loc_per_trace)
    uid1 = uid[keep1]
    center1 = center_trace[keep1]
    if center1.shape[0] == 0:
        return {**empty, "n_traces": int(uid.size)}

    # pass 2: LoG of the rendered XY histogram; keep centroids on a blob
    keep_log = np.ones(center1.shape[0], dtype=bool)
    xedge = np.arange(xnm.min(), xnm.max(), pixel_size)
    yedge = np.arange(ynm.min(), ynm.max(), pixel_size)
    if xedge.size >= 2 and yedge.size >= 2:
        from scipy.signal import convolve2d

        H, _, _ = np.histogram2d(xnm, ynm, bins=[xedge, yedge])
        sigma = dunit / 2.0 / np.sqrt(2.0) / pixel_size
        if sigma > 0:
            hsize = 2 * int(np.ceil(3 * sigma)) + 1
            kernel = _fspecial_log(hsize, sigma)
            log_map = -convolve2d(H, kernel, mode="same")
            mx = log_map.max()
            if mx > 0:
                log_map = log_map / mx
            nx, ny = log_map.shape
            xbin = np.clip(np.floor((center1[:, 0] - xnm.min()) / pixel_size).astype(int), 0, nx - 1)
            ybin = np.clip(np.floor((center1[:, 1] - ynm.min()) / pixel_size).astype(int), 0, ny - 1)
            log_value = log_map[xbin, ybin]
            keep_log = log_value > np.std(log_map)
    uid2 = uid1[keep_log]
    center2 = center1[keep_log]
    if center2.shape[0] == 0:
        return {**empty, "n_traces": int(uid.size), "n_pass1": int(uid1.size)}

    # pass 3: merge nearby surviving centroids into one sub-unit
    cid = dbscan(center2, dunit / 2.0, 1)
    n_clusters = int(cid.max()) + 1 if cid.size else 0
    unit_centers = []
    unit_traces = []
    for c in range(n_clusters):
        members = np.where(cid == c)[0]
        trace_ids = uid2[members]
        loc_unit = loc_m[np.isin(tid, trace_ids)]
        center = np.median(loc_unit, axis=0) * 1e9
        center[2] *= float(z_scale)  # z scaled once (fixes a MATLAB double-scale on singletons)
        unit_centers.append(center)
        unit_traces.append(trace_ids)

    return {
        "unit_centers": np.array(unit_centers) if unit_centers else np.empty((0, 3)),
        "unit_traces": unit_traces,
        "points_nm": points,
        "dunit_nm": dunit,
        "n_traces": int(uid.size),
        "n_pass1": int(uid1.size),
        "n_pass2": int(uid2.size),
    }


def analyze_hlyb(loc_m: np.ndarray, tid: np.ndarray, cfg: HlyBConfig | None = None) -> dict:
    """Full HlyB sub-unit + pair-distance analysis. See module docstring."""
    from scipy.spatial.distance import pdist

    cfg = cfg or HlyBConfig()
    sub = locate_subunit_centers(
        loc_m,
        tid,
        z_scale=cfg.z_scaling_factor,
        pixel_size=cfg.unit_render_pixel_size,
        min_loc_per_trace=cfg.min_loc_per_trace,
        basic_unit_size_nm=cfg.basic_unit_size_nm,
    )
    centers = sub["unit_centers"]
    hlyb_diameter = 2.0 * cfg.diameter_distance_1b2b_nm / np.sqrt(3.0)

    if centers.shape[0] >= 1:
        cluster_id = dbscan(centers, hlyb_diameter, cfg.min_unit_count_per_HlyB)
    else:
        cluster_id = np.empty(0, dtype=np.int64)

    structures = []
    all_pair_dist: list[float] = []
    for k, hid in enumerate(sorted(set(int(c) for c in cluster_id if c != -1))):
        members = np.where(cluster_id == hid)[0]
        uc = centers[members]
        pd = pdist(uc) if uc.shape[0] >= 2 else np.empty(0)
        all_pair_dist.extend(pd.tolist())
        structures.append({
            "id": k + 1,
            "unit_indices": members,
            "unit_centers": uc,
            "centroid": uc.mean(axis=0) if uc.shape[0] else np.zeros(3),
            "pair_distances": pd,
            "traces": [sub["unit_traces"][i] for i in members],
        })

    return {
        "points_nm": sub["points_nm"],
        "subunit_centers": centers,
        "subunit_cluster": cluster_id,
        "structures": structures,
        "all_pair_distances": np.array(all_pair_dist, dtype=float),
        "dunit_nm": float(sub["dunit_nm"]),
        "hlyb_diameter_nm": float(hlyb_diameter),
        "n_traces": int(sub["n_traces"]),
        "n_subunits": int(centers.shape[0]),
        "n_structures": len(structures),
    }


# --------------------------------------------------------------------------
# 2-D variant: E.coli per-cell mask + border erosion preprocessing
# --------------------------------------------------------------------------

def _otsu_threshold(img: np.ndarray) -> float:
    """Otsu threshold of a 2-D image (MATLAB ``imbinarize`` default)."""
    vals = np.asarray(img, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0 or vals.max() <= vals.min():
        return float(vals.min()) if vals.size else 0.0
    hist, edges = np.histogram(vals, bins=256)
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = float(hist.sum())
    wb = np.cumsum(hist).astype(np.float64)
    wf = total - wb
    csum = np.cumsum(hist * centers)
    mb = csum / np.maximum(wb, 1.0)
    mf = (csum[-1] - csum) / np.maximum(wf, 1.0)
    between = wb * wf * (mb - mf) ** 2
    return float(centers[int(np.argmax(between))])


def _disk_struct(radius: int) -> np.ndarray:
    r = max(int(radius), 1)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def compute_border_mask(
    loc_nm_xy: np.ndarray,
    *,
    border_size_nm: float = 200.0,
    pixel_size_nm: float = 20.0,
    sigma_px: float = 2.5,
    median_size_px: int = 25,
) -> np.ndarray:
    """Boolean per-localization border flag (``True`` = in the eroded border
    margin → **excluded**), porting ``get_border_loc`` from the 2-D script.

    A coarse XY histogram is Gaussian-smoothed, Otsu-binarized, hole-filled and
    median-filtered into a per-cell mask; the mask is eroded inward by
    ``border_size_nm`` (a disk of ``round(border/pixel)`` px). A localization is
    flagged when it falls outside the eroded ROI.
    """
    from scipy.ndimage import (
        binary_erosion,
        binary_fill_holes,
        gaussian_filter,
        median_filter,
    )

    xy = np.asarray(loc_nm_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2 or xy.shape[0] == 0:
        return np.zeros(xy.shape[0] if xy.ndim == 2 else 0, dtype=bool)
    ps = float(pixel_size_nm)
    x, y = xy[:, 0], xy[:, 1]
    pad = 10.0 * ps
    x_edges = np.arange(x.min() - pad, x.max() + pad + ps, ps)
    y_edges = np.arange(y.min() - pad, y.max() + pad + ps, ps)
    nx, ny = x_edges.size - 1, y_edges.size - 1
    if nx < 1 or ny < 1:
        return np.zeros(xy.shape[0], dtype=bool)
    counts = np.histogram2d(x, y, bins=[x_edges, y_edges])[0]

    img = gaussian_filter(counts, float(sigma_px))
    binary = img > _otsu_threshold(img)
    filled = binary_fill_holes(binary)
    md = max(int(median_size_px), 1)
    mask = median_filter(filled.astype(np.uint8), size=md).astype(bool)
    erode_px = max(int(round(border_size_nm / ps)), 1)
    roi = binary_erosion(mask, structure=_disk_struct(erode_px))

    xb = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, nx - 1)
    yb = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, ny - 1)
    return ~roi[xb, yb]


def analyze_hlyb_2d(loc_m: np.ndarray, tid: np.ndarray, cfg: HlyBConfig | None = None) -> dict:
    """2-D HlyB sub-unit pair analysis with the E.coli border-removal preprocessing.

    Builds a per-cell mask, drops every trace that touches the eroded border
    margin (where 2-D sub-unit distances are unreliable), then runs the standard
    :func:`analyze_hlyb` pipeline on the interior (forced 2-D: z = 0). The result
    carries the extra keys ``border_points_nm`` / ``n_border_traces`` /
    ``n_total_traces`` / ``is_2d`` for plotting and reporting.
    """
    cfg = cfg or HlyBConfig()
    loc_m = np.asarray(loc_m, dtype=np.float64)
    tid = np.asarray(tid).ravel()
    loc_nm_xy = loc_m[:, :2] * 1e9

    border_loc = compute_border_mask(
        loc_nm_xy,
        border_size_nm=cfg.border_size_nm,
        pixel_size_nm=cfg.mask_pixel_size_nm,
        sigma_px=cfg.mask_sigma_px,
        median_size_px=cfg.mask_median_size_px,
    )

    if tid.size:
        uid, inv = np.unique(tid, return_inverse=True)
        border_trace = np.zeros(uid.size, dtype=bool)
        np.logical_or.at(border_trace, inv, border_loc)  # a trace is border if any loc is
        remove = border_trace[inv]
        n_total = int(uid.size)
        n_border = int(border_trace.sum())
    else:
        remove = np.zeros(0, dtype=bool)
        n_total = n_border = 0
    keep = ~remove

    loc_2d = loc_m.copy()
    loc_2d[:, 2] = 0.0  # 2-D analysis: ignore z entirely
    result = analyze_hlyb(loc_2d[keep], tid[keep], cfg)

    border_xy = loc_nm_xy[border_loc]
    result["border_points_nm"] = np.column_stack([border_xy, np.zeros(border_xy.shape[0])])
    result["n_border_traces"] = n_border
    result["n_total_traces"] = n_total
    result["is_2d"] = True
    return result
