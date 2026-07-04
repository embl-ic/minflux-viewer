"""
minflux_viewer.analysis.npc_wanlu
=================================
Port of Wanlu/Z. Huang's MATLAB NPC **detection + two-ring averaging** pipeline
(``segment_NPC_with_units`` + ``fit_NPC_rings_from_raw6``).

Two stages, kept separate so the viewer exposes them as two commands:

**Detection** (:func:`detect_npc_wanlu`)
    1. ring-kernel convolution of the XY histogram → candidate NPC centres
       (peaks + ring support score, as in :mod:`npc_segmentation`);
    2. **sub-unit (corner) detection** — cluster MINFLUX *traces* into sub-units
       (trace centroids → density + Laplacian-of-Gaussian gate → single-linkage
       clustering); a sub-unit ≈ one labelled NPC corner;
    3. assign sub-units to the nearest NPC centre (within ``1.5·radius``);
    4. keep NPCs with ≥ ``min_units`` assigned sub-units.
    Output: a ``raw6`` table ``[x, y, z, npc_id, unit_id, trace_id]`` (nm) of the
    localizations belonging to accepted NPCs — the bridge to averaging.

**Averaging** (:func:`average_npc_wanlu`)
    1. per-NPC Z re-zero + outlier trim, pool Z, **detect two Z ring peaks**
       (KDE), with an expected-inter-ring-distance fallback;
    2. assign each sub-unit to the lower/upper ring by its mean Z;
    3. **constrained two-ring fit** per NPC (shared radius, inter-ring distance,
       small centre shift + tilt bounds) by Nelder-Mead on a robust ring loss;
    4. align accepted NPCs — tilt the fitted normal to +Z, then **8-fold phase
       align** (``angle(mean(e^{i·8θ}))/8``) — and pool into the averaged particle.

Pure NumPy/SciPy — Qt-free and unit-testable. Distances are in nm.
"""

from __future__ import annotations

import numpy as np

from .npc_segmentation import _find_peaks_2d, _histogram2d, ring_kernel, ring_support_scores

DEFAULT_PIXEL_NM = 4.0
DEFAULT_DIAMETER_NM = 75.0
DEFAULT_RIM_NM = 20.0
DEFAULT_MIN_UNITS = 3
DEFAULT_MIN_SUPPORT = 0.25
DEFAULT_MIN_PEAK_HEIGHT = 0.15
_MIN_LOC_PER_TRACE = 10


# --------------------------------------------------------------------------- #
# Sub-unit (trace cluster) detection
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def detect_npc_wanlu(loc_nm, tid, *, pixel_size=DEFAULT_PIXEL_NM,
                     diameter=DEFAULT_DIAMETER_NM, rim=DEFAULT_RIM_NM,
                     min_units=DEFAULT_MIN_UNITS, min_support=DEFAULT_MIN_SUPPORT,
                     min_peak_height=DEFAULT_MIN_PEAK_HEIGHT, search_factor=1.5,
                     z_scale=1.0) -> dict:
    """Detect NPC centres + assign sub-units. Returns a dict with ``centers``
    (Kacc, 2), ``raw6`` (``[x, y, z, npc_id, unit_id, trace_id]`` nm for accepted
    NPCs), ``unit_centers`` and ``prob_map``.

    ``z_scale`` multiplies Z before sub-unit clustering / averaging (the MATLAB
    ``rimf`` Z-flattening; use e.g. 0.67 for un-z-corrected "noScaling" data). It
    is baked into the stored ``raw6`` so averaging stays consistent."""
    from scipy.signal import fftconvolve

    loc = np.asarray(loc_nm, dtype=float)
    tid = np.asarray(tid).ravel()
    out = {"centers": np.empty((0, 2)), "raw6": np.empty((0, 6)),
           "unit_centers": np.empty((0, 3)), "prob_map": np.zeros((0, 0)),
           "n_npc": 0, "n_units": 0}
    if loc.ndim != 2 or loc.shape[0] < 10 or loc.shape[1] < 2:
        return out
    xy = loc[:, :2]
    finite = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    loc, xy, tid = loc[finite], xy[finite], tid[finite]

    # 1) ring-convolution candidate centres
    H, xedge, yedge = _histogram2d(xy[:, 0], xy[:, 1], pixel_size)
    h = ring_kernel(diameter, rim, pixel_size)
    prob = fftconvolve(H, h, mode="same")
    out["prob_map"] = prob
    peaks_rc = _find_peaks_2d(prob, min_peak_height, diameter / pixel_size)
    if peaks_rc.shape[0] == 0:
        return out
    xc = 0.5 * (xedge[:-1] + xedge[1:])
    yc = 0.5 * (yedge[:-1] + yedge[1:])
    centers = np.column_stack([xc[peaks_rc[:, 0]], yc[peaks_rc[:, 1]]])

    # 2) ring support filter
    support = ring_support_scores(xy, centers, diameter, rim)
    centers = centers[np.isfinite(support) & (support > min_support)]
    if centers.shape[0] == 0:
        return out

    # 3) sub-unit (trace cluster) detection (Z optionally flattened by z_scale)
    loc3 = loc[:, :3] if loc.shape[1] >= 3 else np.column_stack([loc, np.zeros(loc.shape[0])])
    if z_scale != 1.0:
        loc3 = loc3.copy()
        loc3[:, 2] = loc3[:, 2] * float(z_scale)
    d_unit = 2.0 * estimate_trace_size(loc3, tid)
    unit_centers, loc_unit_id = locate_unit_clusters(loc3, tid, d_unit)
    out["unit_centers"] = unit_centers
    if unit_centers.shape[0] == 0:
        return out

    # 4) assign sub-units to nearest NPC centre within search radius
    radius = diameter / 2.0
    search = search_factor * radius
    d = np.hypot(unit_centers[:, None, 0] - centers[None, :, 0],
                 unit_centers[:, None, 1] - centers[None, :, 1])
    nearest = np.argmin(d, axis=1)
    assigned = d[np.arange(d.shape[0]), nearest] <= search
    unit_npc = np.where(assigned, nearest, -1)        # NPC index per unit (0-based)

    # 5) keep NPCs with >= min_units; build raw6 from their localizations
    raw_rows = []
    accepted_centers = []
    npc_out_id = 0
    for k in range(centers.shape[0]):
        unit_ids_here = np.where(unit_npc == k)[0]    # 0-based unit indices
        if unit_ids_here.size < min_units:
            continue
        npc_out_id += 1
        accepted_centers.append(centers[k])
        for ui in unit_ids_here:
            loc_mask = loc_unit_id == (ui + 1)        # loc_unit_id is 1-based
            if not loc_mask.any():
                continue
            sub = loc3[loc_mask]
            t = tid[loc_mask].astype(float)
            rows = np.column_stack([
                sub[:, 0], sub[:, 1], sub[:, 2],
                np.full(sub.shape[0], npc_out_id, dtype=float),
                np.full(sub.shape[0], ui + 1, dtype=float), t])
            raw_rows.append(rows)
    if raw_rows:
        out["raw6"] = np.vstack(raw_rows)
    out["centers"] = np.array(accepted_centers).reshape(-1, 2)
    out["n_npc"] = len(accepted_centers)
    out["n_units"] = int(unit_centers.shape[0])
    return out


# --------------------------------------------------------------------------- #
# Z ring peaks
# --------------------------------------------------------------------------- #
def detect_z_peaks(z, *, bandwidth=1.5, grid_step=0.25, min_sep_nm=5.0,
                   min_frac=0.03):
    """Top-two KDE peaks of a Z distribution (``detect_Zpeaks.m``). Returns the
    sorted peak positions (≤2) in nm."""
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < 10:
        return np.empty(0)
    lo, hi = np.percentile(z, [0.5, 99.5])
    if hi <= lo:
        return np.array([float(np.median(z))])
    grid = np.arange(lo, hi + grid_step, grid_step)
    # Gaussian KDE on the grid
    diff = (grid[:, None] - z[None, :]) / bandwidth
    dens = np.exp(-0.5 * diff ** 2).sum(axis=1) / (z.size * bandwidth * np.sqrt(2 * np.pi))
    min_dist = max(1, int(round(min_sep_nm / grid_step)))
    rc = _find_peaks_2d(dens[:, None], min_frac * float(dens.max()), min_dist)
    if rc.shape[0] == 0:
        return np.array([grid[int(np.argmax(dens))]])
    idx = rc[:, 0]
    order = np.argsort(dens[idx])[::-1][:2]
    return np.sort(grid[idx[order]])


def _resolve_z_peaks(z, inter_ring_bounds, expected_inter):
    peaks = detect_z_peaks(z)
    exp_d = expected_inter if expected_inter else float(np.mean(inter_ring_bounds))
    if peaks.size >= 2:
        sep = peaks[-1] - peaks[0]
        if inter_ring_bounds[0] <= sep <= inter_ring_bounds[1]:
            return float(peaks[0]), float(peaks[-1]), False
    # fallback: symmetric about 0 at the expected distance
    return -exp_d / 2.0, exp_d / 2.0, True


# --------------------------------------------------------------------------- #
# Constrained two-ring fit (port of fit_two_ring_model)
# --------------------------------------------------------------------------- #
def _basis_from_normal(normal):
    normal = normal / (np.linalg.norm(normal) or 1.0)
    ref = np.array([1.0, 0, 0]) if abs(normal @ [1, 0, 0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(normal, ref)
    u = u / (np.linalg.norm(u) or 1.0)
    v = np.cross(normal, u)
    v = v / (np.linalg.norm(v) or 1.0)
    return u, v


def _ring_residuals(P, center, normal, u, v, radius):
    D = P - center
    plane = D @ normal
    xp, yp = D @ u, D @ v
    rad = np.hypot(xp, yp) - radius
    return rad, plane


def _robust(x, k):
    a = np.abs(x)
    return np.where(a <= k, 0.5 * x ** 2, k * (a - 0.5 * k))


def _decode(q, centroid, opt):
    def bounded_vec(qv, maxn):
        r = np.linalg.norm(qv)
        if maxn <= 0 or r < 1e-12:
            return np.zeros(2)
        return (maxn * np.tanh(r)) * (qv / r)

    def sig(x, lo, hi):
        # numerically stable logistic (Nelder-Mead can drive x large)
        if x >= 0:
            s = 1.0 / (1.0 + np.exp(-x))
        else:
            ex = np.exp(x)
            s = ex / (1.0 + ex)
        return lo + (hi - lo) * s

    xy = bounded_vec(q[0:2], opt["max_xy"])
    zsh = opt["max_z"] * np.tanh(q[2])
    center = centroid + np.array([xy[0], xy[1], zsh])
    if opt["tilt_max"] <= 0:
        normal = np.array([0.0, 0, 1])
        tilt = 0.0
    else:
        tmax = np.deg2rad(opt["tilt_max"])
        tv = bounded_vec(q[3:5], tmax)
        tilt = np.linalg.norm(tv)
        if tilt < 1e-12:
            normal = np.array([0.0, 0, 1])
        else:
            az = np.arctan2(tv[1], tv[0])
            normal = np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), np.cos(tilt)])
    normal = normal / (np.linalg.norm(normal) or 1.0)
    u, v = _basis_from_normal(normal)
    radius = sig(q[5], opt["r_bounds"][0], opt["r_bounds"][1])
    inter = sig(q[6], opt["d_bounds"][0], opt["d_bounds"][1])
    lower = center - 0.5 * inter * normal
    upper = center + 0.5 * inter * normal
    return dict(center=center, normal=normal, u=u, v=v, radius=radius, inter=inter,
                lower=lower, upper=upper, tilt_deg=np.rad2deg(tilt),
                shift_xy=float(np.hypot(xy[0], xy[1])), shift_z=abs(zsh))


def _objective(q, Plo, Pup, centroid, opt):
    m = _decode(q, centroid, opt)
    rlo, plo = _ring_residuals(Plo, m["lower"], m["normal"], m["u"], m["v"], m["radius"])
    rup, pup = _ring_residuals(Pup, m["upper"], m["normal"], m["u"], m["v"], m["radius"])
    s_rad = max(opt["rim"] / 2, 1.0)
    s_pl = max(opt["rim"] / 3, 1.0)
    rad = 0.5 * _robust(rlo / s_rad, opt["k"]).mean() + 0.5 * _robust(rup / s_rad, opt["k"]).mean()
    pln = 0.5 * _robust(plo / s_pl, opt["k"]).mean() + 0.5 * _robust(pup / s_pl, opt["k"]).mean()
    cen = 0.01 * (m["shift_xy"] / opt["max_xy"]) ** 2 + 0.01 * (m["shift_z"] / opt["max_z"]) ** 2
    tlt = 0.01 * (m["tilt_deg"] / opt["tilt_max"]) ** 2 if opt["tilt_max"] > 0 else 0.0
    val = rad + pln + cen + tlt
    return val if np.isfinite(val) else 1e18


def fit_two_ring(Plo, Pup, centroid, opt) -> dict:
    """Constrained two-ring fit by Nelder-Mead. ``opt`` carries the bounds."""
    from scipy.optimize import minimize

    Plo = np.asarray(Plo, float)
    Pup = np.asarray(Pup, float)
    if Plo.shape[0] < opt["min_fit_pts"] or Pup.shape[0] < opt["min_fit_pts"]:
        return {"ok": False}
    mlo, mup = Plo.mean(0), Pup.mean(0)
    axis = mup - mlo
    nrm = np.linalg.norm(axis)
    q0 = np.zeros(7)                         # start at centroid, no tilt, mid bounds
    res = minimize(_objective, q0, args=(Plo, Pup, centroid, opt),
                   method="Nelder-Mead",
                   options={"maxiter": opt["max_iter"], "maxfev": opt["max_fev"],
                            "xatol": 1e-6, "fatol": 1e-8})
    m = _decode(res.x, centroid, opt)
    m["ok"] = bool(np.all(np.isfinite(m["center"])) and np.isfinite(m["radius"]))
    m["objective"] = float(res.fun)
    return m


# --------------------------------------------------------------------------- #
# Alignment (tilt → +Z, then 8-fold phase) and averaging
# --------------------------------------------------------------------------- #
def _rotation_a_to_b(a, b):
    a = a / (np.linalg.norm(a) or 1.0)
    b = b / (np.linalg.norm(b) or 1.0)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(a @ b)
    if s < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))


def align_one_npc(P, unit_centers, fit, *, symmetry=8):
    """Tilt fitted normal to +Z, recentre, then phase-align the n-fold symmetry."""
    R = _rotation_a_to_b(np.asarray(fit["normal"], float), np.array([0.0, 0, 1]))
    P0 = (R @ (P - fit["center"]).T).T
    U0 = (R @ (unit_centers - fit["center"]).T).T if unit_centers.size else P0
    phase_pts = U0 if U0.shape[0] >= 2 else P0
    theta = np.arctan2(phase_pts[:, 1], phase_pts[:, 0])
    z8 = np.mean(np.exp(1j * symmetry * theta))
    phase = 0.0 if abs(z8) < 1e-6 else np.angle(z8) / symmetry
    cz, sz = np.cos(-phase), np.sin(-phase)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1.0]])
    locA = (Rz @ P0.T).T
    unitA = (Rz @ U0.T).T if unit_centers.size else U0
    return locA, unitA, float(np.rad2deg(phase))


def _fit_metrics(fit, Plo, Pup) -> dict:
    """Recompute per-NPC quality on the best fit.

    Residuals are the point-to-ring distances split into a **radial** part
    (``hypot(xp, yp) − radius`` in the ring plane) and an **out-of-plane** part
    (signed distance to the ring's plane). From them:

    * ``rim``   — ring radial width (1σ of the radial residuals), nm;
    * ``rmse``  — RMS 3-D point-to-ring distance ``sqrt(mean(rad² + plane²))``, nm;
    * ``nrmse`` — ``rmse / radius`` (dimensionless normalized error);
    * ``gof``   — R²-like ``1 − SS_res/SS_tot`` (fraction of the point spread the
      two-ring model explains; ≤ 1, < 0 means worse than the bare centroid).
    """
    rlo, plo = _ring_residuals(Plo, fit["lower"], fit["normal"], fit["u"], fit["v"], fit["radius"])
    rup, pup = _ring_residuals(Pup, fit["upper"], fit["normal"], fit["u"], fit["v"], fit["radius"])
    rad = np.concatenate([rlo, rup]) if (rlo.size or rup.size) else np.empty(0)
    pln = np.concatenate([plo, pup]) if (plo.size or pup.size) else np.empty(0)
    n = int(rad.size)
    if n == 0:
        return _nan_metrics()
    d2 = rad ** 2 + pln ** 2
    rmse = float(np.sqrt(np.mean(d2)))
    rim = float(np.std(rad))
    P = np.vstack([Plo, Pup])
    ss_res = float(np.sum(d2))
    ss_tot = float(np.sum((P[:, :3] - P[:, :3].mean(axis=0)) ** 2))
    gof = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    radius = float(fit["radius"])
    nrmse = float(rmse / radius) if radius > 0 else float("nan")
    return dict(radius=radius, rim=rim, inter=float(fit["inter"]),
                tilt=float(fit["tilt_deg"]), gof=gof, rmse=rmse, nrmse=nrmse, n=n)


def _nan_metrics() -> dict:
    nan = float("nan")
    return dict(radius=nan, rim=nan, inter=nan, tilt=nan, gof=nan, rmse=nan, nrmse=nan, n=0)


def average_npc_wanlu(raw6, *, diameter_bounds=(70.0, 90.0),
                      inter_ring_bounds=(20.0, 40.0), expected_inter_ring=None,
                      z_mad_factor=3.0, min_locs_per_npc=20, min_fit_points=3,
                      tilt_max_deg=15.0, max_center_shift_xy=20.0,
                      max_center_shift_z=5.0, rim=20.0, symmetry=8, min_gof=0.0,
                      robust_k=3.0, max_iter=2500, max_fev=8000, progress=None) -> dict:
    """Two-ring NPC averaging from a ``raw6`` ``[x,y,z,npc_id,unit_id,trace_id]``
    table (nm). Returns the pooled ``average_loc`` (N, 4: x,y,z,ring) + fit stats.

    Only NPCs whose fit goodness-of-fit (R²) is ``>= min_gof`` are pooled; a
    per-NPC ``table`` of recomputed metrics (radius, rim, inter-ring, tilt, GoF,
    RMSE, normalized RMSE, accepted) is always returned for every fitted NPC.

    ``progress(done, total)`` (optional) is called once per fitted NPC so a caller
    can report progress; the per-NPC fit is the dominant cost."""
    raw6 = np.asarray(raw6, dtype=float)
    out = {"average_loc": np.empty((0, 4)), "n_accepted": 0, "n_npc": 0,
           "z_peaks": (np.nan, np.nan), "z_fallback": True, "fits": []}
    if raw6.ndim != 2 or raw6.shape[0] == 0 or raw6.shape[1] < 6:
        return out
    raw6 = raw6[np.all(np.isfinite(raw6[:, :6]), axis=1)]
    npc_ids = np.unique(raw6[:, 3])
    out["n_npc"] = int(npc_ids.size)

    # 1) per-NPC cache + pooled re-zeroed Z
    cache = {}
    z_pool = []
    for nid in npc_ids:
        P = raw6[raw6[:, 3] == nid]
        if P.shape[0] < min_locs_per_npc:
            continue
        loc = P[:, :3]
        units = P[:, 4]
        z = loc[:, 2]
        zmean = float(np.mean(z))
        z0 = z - zmean
        ad = float(np.mean(np.abs(z0)))
        keep = np.isfinite(z0) if ad <= 0 else (np.abs(z0) <= z_mad_factor * ad)
        z_pool.append(z0[keep])
        # unit centres
        uids = np.unique(units)
        uc = np.array([loc[units == u].mean(axis=0) for u in uids])
        centroid = np.array([loc[:, 0].mean(), loc[:, 1].mean(), np.median(loc[:, 2])])
        cache[nid] = dict(loc=loc, units=units, uids=uids, uc=uc, zmean=zmean, centroid=centroid)
    if not z_pool:
        return out
    z_all = np.concatenate(z_pool)

    # 2) two Z ring peaks
    p1, p2, fallback = _resolve_z_peaks(z_all, inter_ring_bounds, expected_inter_ring)
    out["z_peaks"] = (p1, p2)
    out["z_fallback"] = fallback

    opt = dict(max_xy=max_center_shift_xy, max_z=max_center_shift_z,
               tilt_max=tilt_max_deg, r_bounds=tuple(np.array(diameter_bounds) / 2.0),
               d_bounds=tuple(inter_ring_bounds), rim=rim, k=robust_k,
               min_fit_pts=min_fit_points, max_iter=max_iter, max_fev=max_fev)

    pooled = []
    fits = []
    table = []
    total = len(cache)
    for i, (nid, C) in enumerate(cache.items(), start=1):
        if progress is not None:
            progress(i, total)
        unit_z0 = C["uc"][:, 2] - C["zmean"]
        ring_unit = np.where(np.abs(unit_z0 - p1) <= np.abs(unit_z0 - p2), -1, 1)
        lower_uids = C["uids"][ring_unit == -1]
        upper_uids = C["uids"][ring_unit == 1]
        ring_loc = np.zeros(C["loc"].shape[0])
        ring_loc[np.isin(C["units"], lower_uids)] = -1
        ring_loc[np.isin(C["units"], upper_uids)] = 1
        Plo = C["loc"][ring_loc == -1]
        Pup = C["loc"][ring_loc == 1]
        fit = fit_two_ring(Plo, Pup, C["centroid"], opt)
        fits.append(fit)
        metrics = _fit_metrics(fit, Plo, Pup) if fit.get("ok") else _nan_metrics()
        # goodness-of-fit gate: only well-fit NPCs are pooled into the average.
        accepted = bool(fit.get("ok")) and np.isfinite(metrics["gof"]) and metrics["gof"] >= min_gof
        table.append({"npc_id": int(nid), "n_locs": int(C["loc"].shape[0]),
                      **metrics, "accepted": bool(accepted)})
        if not accepted:
            continue
        locA, _unitA, _ph = align_one_npc(C["loc"], C["uc"], fit, symmetry=symmetry)
        pooled.append(np.column_stack([locA, ring_loc]))
    out["fits"] = fits
    out["table"] = table
    out["min_gof"] = float(min_gof)
    out["n_accepted"] = len(pooled)
    if pooled:
        out["average_loc"] = np.vstack(pooled)
    return out
