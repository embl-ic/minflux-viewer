"""
minflux_viewer.analysis.npc_segmentation
=========================================
Nuclear Pore Complex (NPC) segmentation by **ring-kernel convolution** — a port
of Wanlu's MATLAB ``segment_NPC_with_units`` 2-D center-detection step.

Method (2-D):
    1. histogram the XY localizations into ``pixel_size`` nm bins;
    2. convolve with a donut kernel ``exp(-|x²+y²-(d/2)²| / (4·rim²))`` (peaks
       where the radius equals the NPC ring radius);
    3. find 2-D local-maxima peaks (min height + min separation = NPC diameter);
    4. score each candidate by ring **support** (annulus angular coverage,
       radial spread and radial bias) and keep those above ``min_support``.

Pure NumPy/SciPy — Qt-free and unit-testable. Distances are in nm.
"""

from __future__ import annotations

import numpy as np

DEFAULT_PIXEL_NM = 4.0
DEFAULT_MIN_PEAK_HEIGHT = 0.15
DEFAULT_MIN_SUPPORT = 0.25
_ANGULAR_BINS = 36
_MIN_POINTS_ANNULUS = 6


def ring_kernel(diameter_nm: float, rim_nm: float, pixel_size_nm: float) -> np.ndarray:
    """Normalised donut kernel; max along the circle of radius ``diameter/2``."""
    n = np.arange(-diameter_nm, diameter_nm + pixel_size_nm / 2.0, pixel_size_nm)
    xk, yk = np.meshgrid(n, n)
    r0 = diameter_nm / 2.0
    h = np.exp(-np.abs(xk ** 2 + yk ** 2 - r0 ** 2) / (4.0 * rim_nm ** 2))
    s = h.sum()
    return h / s if s > 0 else h


def _histogram2d(x: np.ndarray, y: np.ndarray, pixel_size: float):
    xedge = np.arange(x.min(), x.max() + pixel_size, pixel_size)
    yedge = np.arange(y.min(), y.max() + pixel_size, pixel_size)
    if xedge.size < 2:
        xedge = np.array([x.min(), x.min() + pixel_size])
    if yedge.size < 2:
        yedge = np.array([y.min(), y.min() + pixel_size])
    h, _, _ = np.histogram2d(x, y, bins=[xedge, yedge])
    return h, xedge, yedge


def _find_peaks_2d(prob: np.ndarray, min_height: float, min_distance_px: float) -> np.ndarray:
    """Local maxima ≥ ``min_height``, greedily thinned so none are closer than
    ``min_distance_px`` (strongest kept first). Returns (K, 2) (row, col)."""
    from scipy.ndimage import maximum_filter

    local_max = maximum_filter(prob, size=3, mode="constant", cval=-np.inf)
    rows, cols = np.nonzero((prob == local_max) & (prob >= min_height))
    if rows.size == 0:
        return np.empty((0, 2), dtype=int)
    order = np.argsort(prob[rows, cols])[::-1]
    rows, cols = rows[order], cols[order]
    coords = np.column_stack([rows, cols]).astype(float)
    taken = np.zeros(rows.size, dtype=bool)
    keep: list[int] = []
    for i in range(rows.size):
        if taken[i]:
            continue
        keep.append(i)
        d = np.hypot(coords[:, 0] - coords[i, 0], coords[:, 1] - coords[i, 1])
        taken |= (d <= min_distance_px) & (d > 0)
    return np.column_stack([rows[keep], cols[keep]])


def ring_support_scores(xy: np.ndarray, centers: np.ndarray, diameter_nm: float,
                        rim_nm: float, *, angular_bins: int = _ANGULAR_BINS,
                        min_points_annulus: int = _MIN_POINTS_ANNULUS) -> np.ndarray:
    """Per-center ring support: ``coverage · exp(-(spread/half)²) · exp(-(|bias|/half)²)``
    over localizations in the annulus ``[d/2 - rim/2, d/2 + rim/2]``.  NaN where
    the annulus has fewer than ``min_points_annulus`` points."""
    xy = np.asarray(xy, dtype=float)
    centers = np.asarray(centers, dtype=float).reshape(-1, 2)
    x, y = xy[:, 0], xy[:, 1]
    r0 = diameter_nm / 2.0
    half = rim_nm / 2.0
    inner_r = max(0.0, r0 - half)
    outer_r = r0 + half
    inner2, outer2 = inner_r ** 2, outer_r ** 2
    ang_edges = np.linspace(-np.pi, np.pi, angular_bins + 1)
    scores = np.full(centers.shape[0], np.nan)
    half_scale = max(half, np.finfo(float).eps)
    for k, (cx, cy) in enumerate(centers):
        dx, dy = x - cx, y - cy
        box = (np.abs(dx) <= outer_r) & (np.abs(dy) <= outer_r)
        if not box.any():
            continue
        dxb, dyb = dx[box], dy[box]
        r2 = dxb ** 2 + dyb ** 2
        ann = (r2 <= outer2) & (r2 >= inner2)
        if int(ann.sum()) < min_points_annulus:
            continue
        r_use = np.sqrt(r2[ann])
        spread = float(np.std(r_use))                 # population std (ddof=0)
        bias = float(np.median(r_use) - r0)
        theta = np.arctan2(dyb[ann], dxb[ann])
        counts, _ = np.histogram(theta, bins=ang_edges)
        coverage = float(np.mean(counts > 0))
        scores[k] = (coverage
                     * np.exp(-(spread / half_scale) ** 2)
                     * np.exp(-(abs(bias) / half_scale) ** 2))
    return scores


def segment_npc_2d(xy_nm, *, diameter_nm: float, rim_nm: float,
                   pixel_size_nm: float = DEFAULT_PIXEL_NM,
                   min_peak_height: float = DEFAULT_MIN_PEAK_HEIGHT,
                   min_support: float = DEFAULT_MIN_SUPPORT) -> dict:
    """Detect NPC centers in XY. Returns a dict with the probability map, the
    edges, all candidate centers + support scores, and the accepted ``centers``
    (nm, (K, 2)) / ``support`` (above ``min_support``)."""
    from scipy.signal import fftconvolve

    xy = np.asarray(xy_nm, dtype=float)
    xy = xy[np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])]
    out = {"prob_map": np.zeros((0, 0)), "xedge": np.zeros(0), "yedge": np.zeros(0),
           "all_centers": np.empty((0, 2)), "all_support": np.empty(0),
           "centers": np.empty((0, 2)), "support": np.empty(0)}
    if xy.shape[0] < 3:
        return out
    x, y = xy[:, 0], xy[:, 1]
    H, xedge, yedge = _histogram2d(x, y, pixel_size_nm)
    h = ring_kernel(diameter_nm, rim_nm, pixel_size_nm)
    prob = fftconvolve(H, h, mode="same")
    xcenter = 0.5 * (xedge[:-1] + xedge[1:])
    ycenter = 0.5 * (yedge[:-1] + yedge[1:])
    min_dist = diameter_nm / pixel_size_nm
    peaks_rc = _find_peaks_2d(prob, min_peak_height, min_dist)

    out.update(prob_map=prob, xedge=xedge, yedge=yedge)
    if peaks_rc.shape[0] == 0:
        return out
    centers = np.column_stack([xcenter[peaks_rc[:, 0]], ycenter[peaks_rc[:, 1]]])
    support = ring_support_scores(xy, centers, diameter_nm, rim_nm)
    take = np.isfinite(support) & (support > min_support)
    # accepted, strongest first
    order = np.argsort(support[take])[::-1]
    out.update(all_centers=centers, all_support=support,
               centers=centers[take][order], support=support[take][order])
    return out
