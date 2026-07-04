"""
minflux_viewer.analysis.npc_wanlu
=================================
MINFLUX **sub-unit (trace-cluster) detection** — port of Wanlu/Z. Huang's
``estimate_trace_size.m`` + ``locate_unit_cluster.m``.

Localizations sharing a ``tid`` are one emitter's on-event (a trace); a *sub-unit*
is a spatial cluster of such traces — one labelled site / structural corner. The
detection: trace centroids → density gate → **Laplacian-of-Gaussian blob gate** at
the unit scale → single-linkage clustering.

The former Wanlu two-ring **detection** and **averaging** pipeline was removed;
the unbiased :mod:`minflux_viewer.analysis.npc_geomfit` model fitter supersedes
it and reuses :func:`estimate_trace_size` / :func:`locate_unit_clusters` here to
seed its fits from LoG sub-units. Pure NumPy/SciPy — Qt-free and unit-tested.
"""

from __future__ import annotations

import numpy as np

from .npc_segmentation import _histogram2d

_MIN_LOC_PER_TRACE = 10


def _trace_groups(tid: np.ndarray):
    """Return (unique_ids, inverse_index) for trace ids."""
    return np.unique(np.asarray(tid), return_inverse=True)


def estimate_trace_size(loc_nm: np.ndarray, tid: np.ndarray) -> float:
    """Typical trace radius (nm): the mode of the log loc-to-trace-centroid
    distance distribution (``estimate_trace_size.m``)."""
    loc = np.asarray(loc_nm, dtype=float)
    uid, inv = _trace_groups(tid)
    if uid.size == 0:
        return 10.0
    dists = []
    for k in range(uid.size):
        sel = loc[inv == k]
        if sel.shape[0] < 2:
            continue
        c = np.median(sel, axis=0)
        d = np.linalg.norm(sel - c, axis=1)
        dists.append(d[d > 0])
    if not dists:
        return 10.0
    d = np.concatenate(dists)
    logd = np.log(d)
    counts, edges = np.histogram(logd, bins="auto")
    if counts.size == 0:
        return float(np.exp(np.mean(logd)))
    kmax = int(np.argmax(counts))
    mode_size = float(np.exp(0.5 * (edges[kmax] + edges[kmax + 1])))
    mean_size = float(np.exp(np.mean(logd)))
    return float(np.median([mean_size, mode_size]))


def _single_linkage_labels(points: np.ndarray, eps: float) -> np.ndarray:
    """DBSCAN with minPts=1 ≡ connected components of the ``eps`` radius graph."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n = points.shape[0]
    if n == 0:
        return np.empty(0, dtype=int)
    tree = cKDTree(points)
    pairs = tree.query_pairs(eps, output_type="ndarray")
    if pairs.size == 0:
        return np.arange(n)
    g = csr_matrix((np.ones(pairs.shape[0]), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _ncomp, labels = connected_components(g, directed=False)
    return labels


def locate_unit_clusters(loc_nm: np.ndarray, tid: np.ndarray, d_unit: float, *,
                         pixel_size: float = 2.5,
                         min_loc_per_trace: int = _MIN_LOC_PER_TRACE):
    """Cluster MINFLUX traces into sub-units (``locate_unit_cluster.m``).

    Returns ``(unit_centers (M, 3) nm, loc_unit_id)`` where ``loc_unit_id`` is the
    per-localization 1-based unit id (0 = unassigned)."""
    from scipy.ndimage import gaussian_laplace

    loc = np.asarray(loc_nm, dtype=float)
    tid = np.asarray(tid)
    n = loc.shape[0]
    loc_unit_id = np.zeros(n, dtype=int)
    uid, inv = _trace_groups(tid)
    if uid.size == 0:
        return np.empty((0, 3)), loc_unit_id

    n_per_trace = np.bincount(inv, minlength=uid.size)
    centroids = np.array([np.median(loc[inv == k], axis=0) for k in range(uid.size)])

    # pass 1: density gate around each trace centroid
    from scipy.spatial import cKDTree
    tree = cKDTree(loc[:, :3])
    density = np.array([len(tree.query_ball_point(c, d_unit / 2.0)) - 1 for c in centroids])
    keep1 = (n_per_trace >= min_loc_per_trace) & (density >= min_loc_per_trace)

    # pass 2: Laplacian-of-Gaussian gate
    x, y = loc[:, 0], loc[:, 1]
    H, xedge, yedge = _histogram2d(x, y, pixel_size)
    sigma = max(d_unit / 2.0 / np.sqrt(2.0) / pixel_size, 0.5)
    log_map = -gaussian_laplace(H, sigma)
    m = float(log_map.max()) or 1.0
    log_map = log_map / m
    thr = float(log_map.std())
    keep2 = keep1.copy()
    cand = np.where(keep1)[0]
    for k in cand:
        cx, cy = centroids[k, 0], centroids[k, 1]
        ix = int(np.clip((cx - xedge[0]) / pixel_size, 0, H.shape[0] - 1))
        iy = int(np.clip((cy - yedge[0]) / pixel_size, 0, H.shape[1] - 1))
        if log_map[ix, iy] <= thr:
            keep2[k] = False

    sel = np.where(keep2)[0]
    if sel.size == 0:
        return np.empty((0, 3)), loc_unit_id

    # pass 3: single-linkage cluster the surviving trace centroids
    labels = _single_linkage_labels(centroids[sel, :3], d_unit / 2.0)
    unit_centers = []
    for c in np.unique(labels):
        trace_ks = sel[labels == c]
        loc_mask = np.isin(inv, trace_ks)
        unit_idx = len(unit_centers) + 1
        loc_unit_id[loc_mask] = unit_idx
        unit_centers.append(np.median(loc[loc_mask, :3], axis=0))
    return np.array(unit_centers).reshape(-1, 3), loc_unit_id
