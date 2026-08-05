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
from itertools import combinations, permutations
from math import comb

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
    neighboring_domain_distance_nm: float = 12.8
    cross_domain_distance_nm: float = 21.3
    # 3-D template matching.  The five distances above are retained for the
    # legacy clustering modes.  The template itself is instead generated from
    # one Euclidean C3-symmetric core, so its pair distances cannot contradict
    # one another.  The PDF estimates include a 2 nm sdAb at each endpoint;
    # that offset is represented as uncertainty rather than baked into the core.
    template_core_a_ring_side_nm: float = 11.0
    template_core_b_ring_side_nm: float = 19.0
    template_core_twist_deg: float = 65.452835488
    template_core_axial_offset_nm: float = 0.0
    template_label_offset_nm: float = 2.0
    min_observed_subunits_per_HlyB: int = 3
    max_observed_subunits_per_HlyB: int = 6
    model_pair_tolerance_nm: float = 5.0
    model_rms_threshold_nm: float = 4.0
    model_max_residual_nm: float = 8.0
    min_pair_match_fraction: float = 0.7
    candidate_edge_radius_nm: float = 0.0
    max_candidate_subsets_per_component: int = 20_000
    # 2-D-only: per-E.coli mask + border shrink (Ecoli_dimer_analysis_2D.m).
    border_size_nm: float = 200.0       # shrink each cell mask in by this much
    border_mode: str = "absolute"       # "absolute" (nm) or "relative" (fraction)
    border_fraction: float = 0.35       # used when border_mode == "relative"
    mask_pixel_size_nm: float = 20.0    # render pixel for the mask histogram
    mask_smooth_nm: float = 60.0        # Gaussian smoothing radius, in NANOMETRES
    mask_close_nm: float = 120.0        # morphological closing radius, nanometres
    mask_open_nm: float = 0.0           # optional opening radius, nanometres
    min_cell_area_nm2: float = 50_000.0  # drop mask components below this area
    # Deprecated pixel-indexed knobs, kept so old configs keep loading.
    mask_sigma_px: float = 0.0          # >0 overrides mask_smooth_nm
    mask_median_size_px: int = 0        # unused; the median filter was removed


def all_pair_distance_histogram(
    points_nm: np.ndarray,
    bin_size_nm: float,
    *,
    max_block_pairs: int = 2_000_000,
) -> dict:
    """Exact all-to-all distance histogram without materializing all pairs.

    ``points_nm`` are detected subunit centres before template gating. Each
    unordered pair contributes exactly once. Small inputs use ``pdist`` in one
    pass; larger inputs accumulate bounded ``cdist`` blocks so peak memory does
    not grow quadratically with the number of centres.
    """
    from scipy.spatial.distance import cdist, pdist

    points = np.asarray(points_nm, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("points_nm must be a 2-D coordinate array")
    points = points[np.all(np.isfinite(points), axis=1)]
    width = float(bin_size_nm)
    if not np.isfinite(width) or width <= 0:
        raise ValueError("bin_size_nm must be positive")

    n_points = int(points.shape[0])
    total_pairs = n_points * (n_points - 1) // 2
    counts = np.empty(0, dtype=np.int64)
    max_distance = 0.0

    def accumulate(distances: np.ndarray) -> None:
        nonlocal counts, max_distance
        values = np.asarray(distances, dtype=np.float64).ravel()
        values = values[np.isfinite(values) & (values >= 0)]
        if not values.size:
            return
        max_distance = max(max_distance, float(np.max(values)))
        bin_index = np.floor(values / width).astype(np.int64)
        local = np.bincount(bin_index)
        if local.size > counts.size:
            counts = np.pad(counts, (0, local.size - counts.size))
        counts[:local.size] += local.astype(np.int64, copy=False)

    if total_pairs:
        if total_pairs <= int(max_block_pairs):
            accumulate(pdist(points))
        else:
            outer_rows = 512
            for start in range(0, n_points, outer_rows):
                end = min(start + outer_rows, n_points)
                block = points[start:end]
                if block.shape[0] > 1:
                    accumulate(pdist(block))
                tail = points[end:]
                if not tail.shape[0]:
                    continue
                rows_per_chunk = max(1, int(max_block_pairs) // int(tail.shape[0]))
                for local_start in range(0, block.shape[0], rows_per_chunk):
                    local_end = min(local_start + rows_per_chunk, block.shape[0])
                    accumulate(cdist(block[local_start:local_end], tail))

    edges = width * np.arange(counts.size + 1, dtype=np.float64)
    return {
        "counts": counts,
        "edges_nm": edges,
        "max_distance_nm": float(max_distance),
        "n_points": n_points,
        "n_pairs": int(total_pairs),
    }


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
    point_source_indices = np.arange(points.shape[0], dtype=np.int64)

    empty = {
        "unit_centers": np.empty((0, 3)),
        "unit_traces": [],
        "points_nm": points,
        "point_source_indices": point_source_indices,
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
        "point_source_indices": point_source_indices,
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
        "point_source_indices": sub["point_source_indices"],
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
# 3-D template-matching variant: partial six-site HlyB model
# --------------------------------------------------------------------------

def hlyb_template_model(cfg: HlyBConfig) -> dict:
    """Return a physically embeddable six-site HlyB core model.

    The source diagrams describe three HlyB dimers arranged with C3 symmetry.
    Their tabulated distances include an estimated 2 nm sdAb contribution at
    both endpoints.  Subtracting that 4 nm allowance yields a core geometry
    that is represented to about 0.14 nm by two twisted equilateral rings.  We
    therefore build coordinates first and derive every matching distance from
    those coordinates; the label offset is retained separately as uncertainty.
    """
    labels = np.array(["1a", "1b", "2a", "2b", "3a", "3b"], dtype=object)
    n = labels.size
    a_radius = float(cfg.template_core_a_ring_side_nm) / np.sqrt(3.0)
    b_radius = float(cfg.template_core_b_ring_side_nm) / np.sqrt(3.0)
    a_angles = np.deg2rad(np.array([0.0, 120.0, 240.0]))
    b_angles = a_angles + np.deg2rad(float(cfg.template_core_twist_deg))
    z_offset = float(cfg.template_core_axial_offset_nm)
    a_coords = np.column_stack([
        a_radius * np.cos(a_angles),
        a_radius * np.sin(a_angles),
        np.full(3, -0.5 * z_offset),
    ])
    b_coords = np.column_stack([
        b_radius * np.cos(b_angles),
        b_radius * np.sin(b_angles),
        np.full(3, 0.5 * z_offset),
    ])
    coords = np.empty((n, 3), dtype=float)
    coords[0::2] = a_coords
    coords[1::2] = b_coords
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    classes = np.empty((n, n), dtype=object)
    classes[:, :] = ""

    def set_class(a: str, b: str, cls: str) -> None:
        ia = int(np.where(labels == a)[0][0])
        ib = int(np.where(labels == b)[0][0])
        classes[ia, ib] = classes[ib, ia] = cls

    for a, b in (("1a", "1b"), ("2a", "2b"), ("3a", "3b")):
        set_class(a, b, "dimer")
    for a, b in (("1b", "2a"), ("2b", "3a"), ("3b", "1a")):
        set_class(a, b, "neighboring domains")
    for a, b in (("1a", "2a"), ("2a", "3a"), ("3a", "1a")):
        set_class(a, b, "every second A-domain")
    for a, b in (("1b", "2b"), ("2b", "3b"), ("3b", "1b")):
        set_class(a, b, "every second B-domain")
    for a, b in (("1a", "2b"), ("2a", "3b"), ("3a", "1b")):
        set_class(a, b, "cross-domain")

    tri = np.triu_indices(n, 1)
    class_order = np.array([
        "neighboring domains",
        "dimer",
        "every second A-domain",
        "cross-domain",
        "every second B-domain",
    ], dtype=object)
    class_distances = {
        str(name): float(np.mean(dist[classes == name]))
        for name in class_order
    }
    return {
        "labels": labels,
        "template_coords_nm": coords,
        "distance_matrix_nm": dist,
        "class_matrix": classes,
        "prior_labels": class_order,
        "prior_distances_nm": np.array([class_distances[str(name)] for name in class_order]),
        "class_distances_nm": class_distances,
        "label_offset_nm": float(cfg.template_label_offset_nm),
        "source_estimated_labeled_distances_nm": {
            "neighboring domains": 12.8,
            "dimer": 14.0,
            "every second A-domain": 15.0,
            "cross-domain": 21.3,
            "every second B-domain": 23.0,
        },
        "max_pair_distance_nm": float(np.max(dist[tri])),
    }


def _pair_indices(n: int) -> np.ndarray:
    return np.array([(i, j) for i in range(n - 1) for j in range(i + 1, n)], dtype=int)


def _assignment_cache(model: dict, max_n: int) -> dict[int, dict]:
    labels = model["labels"]
    D = model["distance_matrix_nm"]
    out: dict[int, dict] = {}
    for n in range(2, max_n + 1):
        pair_idx = _pair_indices(n)
        assignments = np.array(list(permutations(range(labels.size), n)), dtype=int)
        expected = np.empty((assignments.shape[0], pair_idx.shape[0]), dtype=float)
        for r, assign in enumerate(assignments):
            expected[r] = [D[assign[i], assign[j]] for i, j in pair_idx]
        out[n] = {"pair_idx": pair_idx, "assignments": assignments, "expected": expected}
    return out


def _connected_components(adj: np.ndarray) -> list[np.ndarray]:
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps: list[np.ndarray] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            i = stack.pop()
            comp.append(i)
            for j in np.where(adj[i])[0]:
                if not seen[j]:
                    seen[j] = True
                    stack.append(int(j))
        comps.append(np.array(comp, dtype=int))
    return comps


def _score_template_candidate(
    unit_indices: np.ndarray,
    centers: np.ndarray,
    sub: dict,
    model: dict,
    cfg: HlyBConfig,
    cache: dict,
) -> dict:
    n = centers.shape[0]
    pair_idx = cache["pair_idx"]
    obs = np.array([np.linalg.norm(centers[j] - centers[i]) for i, j in pair_idx], dtype=float)
    expected_all = cache["expected"]
    residual_all = obs[None, :] - expected_all
    rms_all = np.sqrt(np.mean(residual_all * residual_all, axis=1))
    best = int(np.argmin(rms_all))
    assignment = cache["assignments"][best]
    expected = expected_all[best]
    residual = obs - expected
    max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
    match_fraction = float(np.mean(np.abs(residual) <= cfg.model_pair_tolerance_nm))
    labels = model["labels"][assignment]
    class_matrix = model["class_matrix"]
    pair_classes = np.array([class_matrix[assignment[i], assignment[j]] for i, j in pair_idx], dtype=object)

    if n == 2:
        accepted = max_abs <= cfg.model_pair_tolerance_nm
    else:
        accepted = (
            float(rms_all[best]) <= cfg.model_rms_threshold_nm
            and max_abs <= cfg.model_max_residual_nm
            and match_fraction >= cfg.min_pair_match_fraction
        )

    traces = [sub["unit_traces"][int(i)] for i in unit_indices]
    support = float(np.mean([len(t) for t in traces])) if traces else 0.0
    if n >= 4 and float(rms_all[best]) <= 0.5 * cfg.model_rms_threshold_nm:
        quality = "strong"
    elif n >= 3 and float(rms_all[best]) <= cfg.model_rms_threshold_nm:
        quality = "moderate"
    else:
        quality = "weak"

    return {
        "id": -1,
        "unit_indices": unit_indices.astype(int),
        "unit_centers": centers,
        "centroid": centers.mean(axis=0) if centers.shape[0] else np.zeros(3),
        "pair_index": pair_idx,
        "pair_distances": obs,
        "expected_pair_distances": expected,
        "pair_residuals": residual,
        "pair_classes": pair_classes,
        "template_labels": labels.astype(object),
        "rms_residual_nm": float(rms_all[best]),
        "max_abs_residual_nm": max_abs,
        "match_fraction": match_fraction,
        "mean_supporting_traces": support,
        "quality": quality,
        "traces": traces,
        "accepted": bool(accepted),
        "reject_reason": "",
    }


def _match_partial_templates(centers: np.ndarray, sub: dict, model: dict, cfg: HlyBConfig) -> tuple[list[dict], list[dict], dict]:
    from scipy.spatial.distance import pdist, squareform

    n_units = centers.shape[0]
    min_n = max(2, int(cfg.min_observed_subunits_per_HlyB or cfg.min_unit_count_per_HlyB))
    max_n = min(int(cfg.max_observed_subunits_per_HlyB), model["labels"].size)
    if n_units < min_n:
        return [], [], {
            "n_components": 0,
            "component_sizes": [],
            "n_candidates_tested": 0,
            "n_candidates_passed_thresholds": 0,
            "n_overlap_rejected": 0,
            "n_skipped_large_subsets": 0,
        }

    D = squareform(pdist(centers)) if n_units > 1 else np.empty((n_units, n_units))
    adj = (D <= cfg.candidate_edge_radius_nm) & (D > 0)
    comps = _connected_components(adj)
    cache = _assignment_cache(model, max_n)
    candidates: list[dict] = []
    n_skipped = 0

    for comp in comps:
        if comp.size < min_n:
            continue
        local_max = min(max_n, comp.size)
        for n in range(local_max, min_n - 1, -1):
            subset_count = int(comb(int(comp.size), n))
            if subset_count > cfg.max_candidate_subsets_per_component:
                n_skipped += subset_count
                continue
            for subset_local in combinations(range(comp.size), n):
                idx = comp[np.array(subset_local, dtype=int)]
                cand = _score_template_candidate(idx, centers[idx], sub, model, cfg, cache[n])
                candidates.append(cand)

    passed = [c for c in candidates if c["accepted"]]
    rejected = [c for c in candidates if not c["accepted"]]
    passed.sort(key=lambda c: (-c["unit_centers"].shape[0], c["rms_residual_nm"], -c["mean_supporting_traces"]))

    used = np.zeros(n_units, dtype=bool)
    structures: list[dict] = []
    n_overlap = 0
    for cand in passed:
        if np.any(used[cand["unit_indices"]]):
            cand = dict(cand)
            cand["reject_reason"] = "overlaps better-ranked accepted candidate"
            rejected.append(cand)
            n_overlap += 1
            continue
        used[cand["unit_indices"]] = True
        cand = dict(cand)
        cand["id"] = len(structures) + 1
        structures.append(cand)

    qc = {
        "n_components": len(comps),
        "component_sizes": [int(c.size) for c in comps],
        "n_candidates_tested": len(candidates),
        "n_candidates_passed_thresholds": len(passed),
        "n_overlap_rejected": n_overlap,
        "n_skipped_large_subsets": n_skipped,
    }
    return structures, rejected, qc


def analyze_hlyb_template3d(loc_m: np.ndarray, tid: np.ndarray, cfg: HlyBConfig | None = None) -> dict:
    """3-D HlyB analysis using partial six-site template matching.

    The MINFLUX data may observe only a subset of a complex. This mode first
    detects candidate sub-unit centres with the standard pipeline, then accepts
    non-overlapping groups of 2-6 centres whose 3-D pair-distance matrix can be
    assigned to a partial HlyB distance template.
    """
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
    model = hlyb_template_model(cfg)

    cfg = HlyBConfig(**vars(cfg))
    if cfg.model_pair_tolerance_nm <= 0:
        cfg.model_pair_tolerance_nm = 2.0 * float(model["label_offset_nm"]) + 1.0
    if cfg.model_rms_threshold_nm <= 0:
        cfg.model_rms_threshold_nm = 0.8 * cfg.model_pair_tolerance_nm
    if cfg.model_max_residual_nm <= 0:
        cfg.model_max_residual_nm = 1.6 * cfg.model_pair_tolerance_nm
    if cfg.candidate_edge_radius_nm <= 0:
        cfg.candidate_edge_radius_nm = float(model["max_pair_distance_nm"] + cfg.model_pair_tolerance_nm)

    structures, rejected, qc = _match_partial_templates(centers, sub, model, cfg)
    all_pair_dist: list[float] = []
    for st in structures:
        all_pair_dist.extend(np.asarray(st["pair_distances"], dtype=float).tolist())

    return {
        "points_nm": sub["points_nm"],
        "point_source_indices": sub["point_source_indices"],
        "subunit_centers": centers,
        "subunit_cluster": np.full(centers.shape[0], -1, dtype=np.int64),
        "structures": structures,
        "rejected_candidates": rejected,
        "all_pair_distances": np.array(all_pair_dist, dtype=float),
        "dunit_nm": float(sub["dunit_nm"]),
        "hlyb_diameter_nm": float(cfg.candidate_edge_radius_nm),
        "candidate_edge_radius_nm": float(cfg.candidate_edge_radius_nm),
        "model_pair_tolerance_nm": float(cfg.model_pair_tolerance_nm),
        "model_rms_threshold_nm": float(cfg.model_rms_threshold_nm),
        "model_max_residual_nm": float(cfg.model_max_residual_nm),
        "n_traces": int(sub["n_traces"]),
        "n_pass1": int(sub["n_pass1"]),
        "n_pass2": int(sub["n_pass2"]),
        "n_subunits": int(centers.shape[0]),
        "n_structures": len(structures),
        "model": model,
        "match_qc": qc,
        "template_matching": True,
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


@dataclass
class CellMaskResult:
    """Per-E.coli delineation plus the inward-shrunk interior.

    ``border_loc`` is the per-localization exclusion flag used by the 2-D
    pipeline; everything else is diagnostic/reusable — ``edge_distance_nm``
    (exact distance from each localization to its cell boundary) and
    ``cell_id`` make the delineation useful beyond border removal.
    """

    border_loc: np.ndarray          # bool per loc; True = excluded margin
    cell_id: np.ndarray             # int per loc; 0 = outside every cell
    edge_distance_nm: np.ndarray    # float per loc; 0 outside every cell
    mask: np.ndarray                # bool image, the cell mask
    interior: np.ndarray            # bool image, the retained interior
    labels: np.ndarray              # int image, per-cell labels (0 = background)
    x_edges: np.ndarray
    y_edges: np.ndarray
    pixel_size_nm: float
    cell_areas_nm2: np.ndarray
    cell_half_width_nm: np.ndarray  # max distance-to-edge per cell
    stats: dict


def compute_cell_mask(
    loc_nm_xy: np.ndarray,
    *,
    border_size_nm: float = 200.0,
    border_mode: str = "absolute",
    border_fraction: float = 0.35,
    pixel_size_nm: float = 20.0,
    smooth_nm: float = 60.0,
    close_nm: float = 120.0,
    open_nm: float = 0.0,
    min_cell_area_nm2: float = 50_000.0,
) -> CellMaskResult:
    """Delineate each E.coli from the localization density and shrink inward.

    Replaces the original ``get_border_loc`` port. The old pipeline binarized a
    lightly smoothed density and then applied a 25 px (500 nm) **median filter**
    to despeckle; on punctate MINFLUX data that filter deletes exactly the thin
    parts of a cell it is meant to clean, so the mask ended up far smaller than
    the cell and a subsequent 200 nm erosion annihilated it. Here instead:

    1. render an XY density histogram and smooth it with a radius given in
       **nanometres** (``smooth_nm``), so the bridging scale is physical rather
       than tied to the pixel size;
    2. Otsu-binarize;
    3. **morphologically close** by ``close_nm`` — dilation followed by erosion
       bridges the gaps between puncta *without* shrinking the object, which a
       median filter cannot do — then fill enclosed holes;
    4. optionally open by ``open_nm`` to drop spurs, and drop connected
       components below ``min_cell_area_nm2`` (despeckling by area, not by
       erosion);
    5. shrink inward using the exact Euclidean **distance transform** rather
       than a discretized disk erosion, which also yields each localization's
       distance to its cell boundary.

    ``border_mode="absolute"`` keeps localizations at least ``border_size_nm``
    from the boundary. ``border_mode="relative"`` keeps those at least
    ``border_fraction`` × that cell's own half-width from the boundary, which
    adapts to cells of different width and bounds the membrane tilt that the
    shrink is meant to exclude (a point kept at relative depth *f* of a
    cylinder of radius *R* sits within ``arcsin(1 - f)`` of face-on).
    """
    from scipy.ndimage import (
        binary_closing,
        binary_fill_holes,
        binary_opening,
        distance_transform_edt,
        gaussian_filter,
        label,
    )

    xy = np.asarray(loc_nm_xy, dtype=np.float64)
    n_loc = int(xy.shape[0]) if xy.ndim == 2 else 0
    ps = float(pixel_size_nm)
    if ps <= 0:
        raise ValueError("pixel_size_nm must be positive")
    mode = str(border_mode).lower()
    if mode not in ("absolute", "relative"):
        raise ValueError("border_mode must be 'absolute' or 'relative'")

    def _empty(reason: str) -> CellMaskResult:
        return CellMaskResult(
            border_loc=np.zeros(n_loc, dtype=bool),
            cell_id=np.zeros(n_loc, dtype=np.int64),
            edge_distance_nm=np.zeros(n_loc, dtype=float),
            mask=np.zeros((0, 0), dtype=bool),
            interior=np.zeros((0, 0), dtype=bool),
            labels=np.zeros((0, 0), dtype=np.int64),
            x_edges=np.empty(0), y_edges=np.empty(0), pixel_size_nm=ps,
            cell_areas_nm2=np.empty(0), cell_half_width_nm=np.empty(0),
            stats={"n_cells": 0, "reason": reason, "retained_fraction": 1.0,
                   "in_mask_fraction": 1.0},
        )

    if xy.ndim != 2 or xy.shape[1] < 2 or n_loc == 0:
        return _empty("no localizations")

    x, y = xy[:, 0], xy[:, 1]
    pad = max(10.0 * ps, 2.0 * float(close_nm), 2.0 * float(smooth_nm))
    x_edges = np.arange(x.min() - pad, x.max() + pad + ps, ps)
    y_edges = np.arange(y.min() - pad, y.max() + pad + ps, ps)
    nx, ny = x_edges.size - 1, y_edges.size - 1
    if nx < 1 or ny < 1:
        return _empty("field smaller than one pixel")

    counts = np.histogram2d(x, y, bins=[x_edges, y_edges])[0]
    sigma_px = max(float(smooth_nm) / ps, 0.0)
    img = gaussian_filter(counts, sigma_px) if sigma_px > 0 else counts
    mask = img > _otsu_threshold(img)

    close_px = int(round(float(close_nm) / ps))
    if close_px >= 1:
        mask = binary_closing(mask, structure=_disk_struct(close_px))
    mask = binary_fill_holes(mask)
    open_px = int(round(float(open_nm) / ps))
    if open_px >= 1:
        mask = binary_opening(mask, structure=_disk_struct(open_px))

    labels, n_lab = label(mask)
    px_area = ps * ps
    if n_lab:
        comp_px = np.bincount(labels.ravel(), minlength=n_lab + 1)
        too_small = np.flatnonzero(comp_px * px_area < float(min_cell_area_nm2))
        too_small = too_small[too_small > 0]
        if too_small.size:
            drop = np.isin(labels, too_small)
            mask = mask & ~drop
            labels, n_lab = label(mask)

    xb = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, nx - 1)
    yb = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, ny - 1)

    if not mask.any():
        res = _empty("no cell mask found")
        res.stats["in_mask_fraction"] = 0.0
        return res

    # Exact Euclidean distance to the cell boundary, in nm.
    edt_nm = distance_transform_edt(mask) * ps
    cell_ids = np.arange(1, n_lab + 1)
    cell_px = np.bincount(labels.ravel(), minlength=n_lab + 1)[1:]
    cell_area = cell_px * px_area
    half_width = np.zeros(n_lab, dtype=float)
    for k in cell_ids:
        sel = labels == k
        half_width[k - 1] = float(edt_nm[sel].max()) if sel.any() else 0.0

    if mode == "absolute":
        threshold_img = np.full(mask.shape, float(border_size_nm), dtype=float)
    else:
        f = float(border_fraction)
        threshold_img = np.zeros(mask.shape, dtype=float)
        for k in cell_ids:
            threshold_img[labels == k] = f * half_width[k - 1]
    interior = mask & (edt_nm >= threshold_img)

    border_loc = ~interior[xb, yb]
    in_mask = mask[xb, yb]
    stats = {
        "n_cells": int(n_lab),
        "mask_area_nm2": float(mask.sum() * px_area),
        "interior_area_nm2": float(interior.sum() * px_area),
        "in_mask_fraction": float(in_mask.mean()),
        "retained_fraction": float((~border_loc).mean()),
        "max_half_width_nm": float(half_width.max()) if n_lab else 0.0,
        "median_half_width_nm": float(np.median(half_width)) if n_lab else 0.0,
        "border_mode": mode,
        "border_size_nm": float(border_size_nm),
        "border_fraction": float(border_fraction),
        "smooth_nm": float(smooth_nm),
        "close_nm": float(close_nm),
        "pixel_size_nm": ps,
    }
    if mode == "relative":
        f = min(max(float(border_fraction), 0.0), 1.0)
        stats["implied_max_tilt_deg"] = float(np.degrees(np.arcsin(min(1.0 - f, 1.0))))
    return CellMaskResult(
        border_loc=border_loc,
        cell_id=np.where(in_mask, labels[xb, yb], 0).astype(np.int64),
        edge_distance_nm=np.where(in_mask, edt_nm[xb, yb], 0.0),
        mask=mask, interior=interior, labels=labels,
        x_edges=x_edges, y_edges=y_edges, pixel_size_nm=ps,
        cell_areas_nm2=cell_area, cell_half_width_nm=half_width, stats=stats,
    )


def compute_border_mask(
    loc_nm_xy: np.ndarray,
    *,
    border_size_nm: float = 200.0,
    pixel_size_nm: float = 20.0,
    sigma_px: float = 0.0,
    median_size_px: int = 0,
    border_mode: str = "absolute",
    border_fraction: float = 0.35,
    smooth_nm: float = 60.0,
    close_nm: float = 120.0,
    open_nm: float = 0.0,
    min_cell_area_nm2: float = 50_000.0,
) -> np.ndarray:
    """Boolean per-localization border flag (``True`` = margin → **excluded**).

    Thin wrapper over :func:`compute_cell_mask` kept for callers that only need
    the flag. ``sigma_px`` (if > 0) overrides ``smooth_nm`` for backward
    compatibility; ``median_size_px`` is accepted and ignored — the median
    filter it configured was the cause of the over-shrunk masks and has been
    replaced by morphological closing plus area-based despeckling.
    """
    smooth = float(sigma_px) * float(pixel_size_nm) if float(sigma_px) > 0 else float(smooth_nm)
    return compute_cell_mask(
        loc_nm_xy,
        border_size_nm=border_size_nm,
        border_mode=border_mode,
        border_fraction=border_fraction,
        pixel_size_nm=pixel_size_nm,
        smooth_nm=smooth,
        close_nm=close_nm,
        open_nm=open_nm,
        min_cell_area_nm2=min_cell_area_nm2,
    ).border_loc


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

    smooth_nm = (
        float(cfg.mask_sigma_px) * float(cfg.mask_pixel_size_nm)
        if float(cfg.mask_sigma_px) > 0 else float(cfg.mask_smooth_nm)
    )
    cells = compute_cell_mask(
        loc_nm_xy,
        border_size_nm=cfg.border_size_nm,
        border_mode=cfg.border_mode,
        border_fraction=cfg.border_fraction,
        pixel_size_nm=cfg.mask_pixel_size_nm,
        smooth_nm=smooth_nm,
        close_nm=cfg.mask_close_nm,
        open_nm=cfg.mask_open_nm,
        min_cell_area_nm2=cfg.min_cell_area_nm2,
    )
    border_loc = cells.border_loc

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
    interior_source_indices = np.flatnonzero(keep)
    relative_indices = np.asarray(
        result.get("point_source_indices", np.arange(interior_source_indices.size)),
        dtype=np.int64,
    )
    result["point_source_indices"] = interior_source_indices[relative_indices]

    border_xy = loc_nm_xy[border_loc]
    result["border_points_nm"] = np.column_stack([border_xy, np.zeros(border_xy.shape[0])])
    result["border_source_indices"] = np.flatnonzero(border_loc)
    result["n_border_traces"] = n_border
    result["n_total_traces"] = n_total
    result["is_2d"] = True
    result["cell_mask"] = cells
    result["cell_mask_stats"] = cells.stats
    result["n_cells"] = int(cells.stats.get("n_cells", 0))
    return result
