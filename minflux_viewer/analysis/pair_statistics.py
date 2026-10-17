"""Shared numerical geometry used by live analysis workflows."""
from __future__ import annotations

import numpy as np


def trace_centroids(
    loc_m: np.ndarray,
    tid: np.ndarray,
    tim: np.ndarray | None = None,
    *,
    z_scale: float = 0.67,
    min_loc_per_trace: int = 10,
) -> dict:
    """Per-trace centroid, uncertainty and time span.

    The centroid is the plain mean, whose standard error is ``sd/√n`` — unlike
    a median it has a closed-form uncertainty, which the model needs.  Traces
    with fewer than ``min_loc_per_trace`` localizations are dropped.
    """
    loc_m = np.asarray(loc_m, dtype=np.float64)
    tid = np.asarray(tid).ravel()
    if loc_m.ndim != 2 or loc_m.shape[1] < 3 or loc_m.shape[0] == 0:
        empty = np.empty((0, 3))
        return {"centroids_nm": empty, "sem_nm": empty, "n_locs": np.empty(0, dtype=int),
                "t_start": np.empty(0), "t_end": np.empty(0),
                "trace_ids": np.empty(0), "n_traces_total": 0}

    points = loc_m * 1e9
    points[:, 2] *= float(z_scale)

    uid, inv = np.unique(tid, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    inv_sorted = inv[order]
    starts = np.searchsorted(inv_sorted, np.arange(uid.size))
    ends = np.searchsorted(inv_sorted, np.arange(uid.size), side="right")
    n_locs = ends - starts

    cent = np.empty((uid.size, 3))
    sem = np.full((uid.size, 3), np.nan)
    for k, (s, e) in enumerate(zip(starts, ends)):
        block = points[order[s:e]]
        cent[k] = block.mean(axis=0)
        if block.shape[0] > 1:
            sem[k] = block.std(axis=0, ddof=1) / np.sqrt(block.shape[0])

    if tim is not None and np.asarray(tim).size == points.shape[0]:
        t = np.asarray(tim, dtype=float).ravel()
        t_start: np.ndarray | None = np.array([t[order[s:e]].min() for s, e in zip(starts, ends)])
        t_end: np.ndarray | None = np.array([t[order[s:e]].max() for s, e in zip(starts, ends)])
    else:
        t_start = t_end = None

    keep = n_locs >= int(min_loc_per_trace)
    return {
        "centroids_nm": cent[keep],
        "sem_nm": sem[keep],
        "n_locs": n_locs[keep],
        "t_start": None if t_start is None else t_start[keep],
        "t_end": None if t_end is None else t_end[keep],
        "trace_ids": uid[keep],
        "n_traces_total": int(uid.size),
        "points_nm": points,
    }

def pair_distance_profile(points_nm: np.ndarray, r_max_nm: float,
                          bin_nm: float) -> tuple[np.ndarray, np.ndarray]:
    """Histogram of all pair distances up to ``r_max_nm`` (exact, KD-tree)."""
    from scipy.spatial import cKDTree

    pts = np.asarray(points_nm, dtype=np.float64)
    edges = np.arange(0.0, float(r_max_nm) + float(bin_nm), float(bin_nm))
    if pts.ndim != 2 or pts.shape[0] < 2:
        return np.zeros(edges.size - 1, dtype=np.int64), edges
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=float(r_max_nm), output_type="ndarray")
    if pairs.shape[0] == 0:
        return np.zeros(edges.size - 1, dtype=np.int64), edges
    dist = np.linalg.norm(pts[pairs[:, 1]] - pts[pairs[:, 0]], axis=1)
    return np.histogram(dist, bins=edges)[0].astype(np.int64), edges
