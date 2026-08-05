"""
minflux_viewer.analysis.hlyb_pairwise
=====================================
Ensemble pair-distance analysis of HlyB sub-unit labelling — the measurement
that replaces per-complex template matching when the labelled distances are
closer together than the localization precision.

Why this exists
---------------
The template-matching pipeline first condenses traces into "sub-unit centres"
by merging every trace pair closer than ``Dunit/2``.  That merge radius is set
by the *within-trace* scatter, and on real data it lands on top of the shortest
modelled distance (8.94 nm), so it deletes the signal it is meant to measure.
Everything downstream inherits the damage: distances below the merge radius are
absent by construction, and truncating an otherwise decreasing distribution
creates a maximum just above the cut that is easily mistaken for a structural
distance — it moves when the merge radius moves.

This module never merges.  It works directly on **trace centroids**, models the
same-emitter repeat population instead of deleting it, and fits the whole
observed pair-distance distribution with a forward model whose class weights
are fixed by the structure.  Three components:

``N_rep · K_rep(r)``
    Same molecule re-acquired as several traces.  ``K_rep`` is calibrated
    **empirically and non-circularly** from consecutive-in-time trace pairs
    (see :func:`calibrate_repeat_kernel`): selection is on the time gap alone,
    never on distance, so the resulting distance distribution is an unbiased
    sample of same-emitter separations.

``N_cx · Σ_c (1/5) · P(r; d_c + δ, σ)``
    Distinct sub-units of one complex.  ``d_c`` are the five HlyB distances,
    held fixed.  **The class weights are fixed at 1/5 and are not free**: each
    class has exactly three pairs, and every pair needs two labels, so all
    classes scale as p² with labelling efficiency and their *ratio* is
    independent of it.  That is what makes the fit a test of the model rather
    than a flexible curve fit.  ``δ`` — the fluorophore displacement caused by
    the single-domain antibody — is **fitted**, which settles from the data
    whether the label sits isotropically (δ ≈ +1 nm) or radially outward
    (δ ≈ +4 nm, reproducing the tabulated distances).

``A · null(r)``
    Pairs from different complexes.  Its shape is taken from an
    envelope-preserving surrogate (:func:`envelope_null`) rather than assumed.

Pure NumPy/SciPy — Qt-free and unit-testable.  Coordinates: ``loc`` in metres
(raw, z **not** RIMF-baked); ``z_scaling_factor`` is applied here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: The five HlyB sub-unit distance classes, in nanometres, as inter-domain
#: distances of the reference structure.  These are the *protein* distances:
#: they match the values annotated on the source structural diagram to better
#: than 0.25 nm, and are ~4 nm shorter than the tabulated ones, which include
#: an allowance of 2 nm per single-domain antibody at each endpoint.  The
#: antibody displacement is carried by the fitted ``label_offset_nm`` instead.
HLYB_CLASS_NAMES = (
    "neighboring domains",
    "dimer",
    "every second A-domain",
    "cross-domain",
    "every second B-domain",
)
HLYB_CLASS_DISTANCES_NM = (8.936, 10.138, 11.000, 17.302, 19.000)

#: Each class contributes exactly three of the fifteen pairs, so with a
#: labelling efficiency p every class scales as p² and the weights are equal.
HLYB_CLASS_WEIGHTS = tuple(1.0 / len(HLYB_CLASS_DISTANCES_NM)
                           for _ in HLYB_CLASS_DISTANCES_NM)


@dataclass
class PairFitConfig:
    """Parameters of the ensemble pair-distance analysis."""

    min_loc_per_trace: int = 10
    z_scaling_factor: float = 0.67

    # Observable
    r_max_nm: float = 60.0
    bin_nm: float = 0.5

    # Empirical repeat-kernel calibration (selection on TIME only).
    # The short-gap sample measures the *shape* of the same-site short-range
    # population, but it samples its tightest end: a molecule re-acquired
    # immediately has barely drifted, whereas one re-acquired minutes later has.
    # The population also contains genuinely distinct emitters that belong to
    # the same site — a FluoTag-X2 carries two fluorophores — which are not
    # time-correlated at all.  The kernel is therefore allowed a fitted radial
    # stretch within ``repeat_scale_bounds`` rather than being pinned.
    repeat_gap_s: float = 0.2
    repeat_max_nm: float = 40.0
    repeat_min_pairs: int = 30
    repeat_sigma_nm: float = 2.0        # fallback when time is unavailable
    repeat_scale_bounds: tuple = (1.0, 2.5)
    #: Primary comparison pins the kernel at its measured width.  The analysis
    #: additionally repeats the whole comparison with the stretch released and
    #: reports both, because the strength of the structural evidence depends on
    #: that choice and hiding it would overstate the result.
    fit_repeat_scale: bool = False

    # Envelope-preserving null
    null_cell_nm: float = 50.0
    null_replicates: int = 8
    rng_seed: int = 0

    # Forward model.
    # δ is the MEAN lengthening of an observed distance caused by the two
    # antibody displacements.  Adding two offsets to a fixed vector can only
    # increase the expected magnitude, so δ >= 0 is a physical constraint, not
    # a convenience; leaving it unbounded lets a one-distance hypothesis slide
    # its peak far from the modelled distance and impersonate a broad
    # background.  The tabulated allowance is 2 nm per endpoint, hence 6 nm of
    # headroom.
    label_offset_nm: float = 2.0        # starting value for the fitted δ
    label_offset_bounds_nm: tuple = (0.0, 6.0)
    fit_label_offset: bool = True
    # σ is a property of the MEASUREMENT, not of the structural hypothesis, so
    # it is bounded by the measured centroid precision and shared across
    # hypotheses.  The floor is set from the data (√2 × centroid error); the
    # headroom covers the antibody-offset and conformational spread.
    extra_sigma_nm: float = 3.0         # starting value for the fitted σ
    sigma_headroom_nm: float = 3.5
    sigma_floor_nm: float = 2.0         # used when the data cannot supply one
    fit_extra_sigma: bool = True
    fit_r_min_nm: float = 1.0
    fit_r_max_nm: float = 45.0

    #: Hypotheses compared against the data.  ``None`` uses the default set.
    hypotheses: tuple = field(default=("six_site", "dimer_only", "no_structure"))


# --------------------------------------------------------------------------
# Trace centroids
# --------------------------------------------------------------------------

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
        t_start = np.array([t[order[s:e]].min() for s, e in zip(starts, ends)])
        t_end = np.array([t[order[s:e]].max() for s, e in zip(starts, ends)])
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


def repeat_pair_index(
    centroids_nm: np.ndarray,
    t_start: np.ndarray | None,
    t_end: np.ndarray | None,
    *,
    gap_s: float = 0.2,
    max_nm: float = 40.0,
) -> np.ndarray:
    """Index pairs of the consecutive-in-time traces used to calibrate the kernel.

    Returned as an ``(M, 2)`` array of indices into ``centroids_nm`` so the
    calibration set can be drawn, and therefore judged, rather than taken on
    trust.
    """
    pts = np.asarray(centroids_nm, dtype=np.float64)
    if t_start is None or t_end is None or pts.shape[0] < 2:
        return np.empty((0, 2), dtype=np.int64)
    t0 = np.asarray(t_start, dtype=float).ravel()
    t1 = np.asarray(t_end, dtype=float).ravel()
    if t0.size != pts.shape[0]:
        return np.empty((0, 2), dtype=np.int64)
    order = np.argsort(t0)
    gaps = t0[order][1:] - t1[order][:-1]
    dist = np.linalg.norm(pts[order][1:] - pts[order][:-1], axis=1)
    sel = (gaps < float(gap_s)) & (dist <= float(max_nm))
    if not sel.any():
        return np.empty((0, 2), dtype=np.int64)
    idx = np.flatnonzero(sel)
    return np.column_stack([order[idx], order[idx + 1]]).astype(np.int64)


def pairs_in_band(
    centroids_nm: np.ndarray,
    lo_nm: float,
    hi_nm: float,
) -> np.ndarray:
    """Index pairs of centroids whose separation lies in ``[lo_nm, hi_nm]``.

    This is the spatial counterpart of selecting a range on the pair-distance
    profile.  It asserts **no** assignment of a pair to a complex or to a
    distance class — the measurement does not make one, and at this precision
    it could not.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(centroids_nm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2 or hi_nm <= 0:
        return np.empty((0, 2), dtype=np.int64)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=float(hi_nm), output_type="ndarray")
    if pairs.shape[0] == 0:
        return pairs.astype(np.int64)
    dist = np.linalg.norm(pts[pairs[:, 1]] - pts[pairs[:, 0]], axis=1)
    return pairs[dist >= float(lo_nm)].astype(np.int64)


# --------------------------------------------------------------------------
# Observable and null
# --------------------------------------------------------------------------

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


def envelope_null(
    points_nm: np.ndarray,
    *,
    r_max_nm: float = 60.0,
    bin_nm: float = 0.5,
    cell_nm: float = 50.0,
    replicates: int = 8,
    rng_seed: int = 0,
) -> dict:
    """Pair-distance profile of an envelope-preserving surrogate.

    Points are re-drawn from their own coarse (``cell_nm``) occupancy
    histogram, so the large-scale density — which cells are present, and how
    bright each is — is reproduced while all structure below ``cell_nm`` is
    destroyed.  This is the reference against which an excess is judged.
    """
    pts = np.asarray(points_nm, dtype=np.float64)
    n_bins = int(np.floor(float(r_max_nm) / float(bin_nm) + 1e-9))
    if pts.ndim != 2 or pts.shape[0] < 2:
        z = np.zeros(n_bins)
        return {"mean": z, "sd": z.copy(), "replicates": 0}

    rng = np.random.default_rng(rng_seed)
    lo = pts.min(axis=0) - cell_nm
    hi = pts.max(axis=0) + cell_nm
    n_cells = np.maximum(np.ceil((hi - lo) / cell_nm).astype(int), 1)
    edges = [np.linspace(lo[k], lo[k] + n_cells[k] * cell_nm, n_cells[k] + 1)
             for k in range(3)]
    counts, _ = np.histogramdd(pts, bins=edges)
    flat = counts.ravel()
    total = flat.sum()
    if total <= 0:
        z = np.zeros(n_bins)
        return {"mean": z, "sd": z.copy(), "replicates": 0}
    prob = flat / total

    acc = []
    for _ in range(max(int(replicates), 1)):
        pick = rng.choice(prob.size, size=pts.shape[0], p=prob)
        cell = np.array(np.unravel_index(pick, counts.shape)).T
        sample = np.empty_like(pts)
        for k in range(3):
            sample[:, k] = edges[k][cell[:, k]] + rng.random(pts.shape[0]) * cell_nm
        acc.append(pair_distance_profile(sample, r_max_nm, bin_nm)[0].astype(float))
    stack = np.vstack(acc)
    return {"mean": stack.mean(axis=0), "sd": stack.std(axis=0), "replicates": stack.shape[0]}


# --------------------------------------------------------------------------
# Same-emitter repeat kernel
# --------------------------------------------------------------------------

def calibrate_repeat_kernel(
    centroids_nm: np.ndarray,
    t_start: np.ndarray | None,
    t_end: np.ndarray | None,
    *,
    gap_s: float = 0.2,
    max_nm: float = 40.0,
    min_pairs: int = 30,
    fallback_sigma_nm: float = 2.0,
    r_max_nm: float = 60.0,
    bin_nm: float = 0.5,
) -> dict:
    """Empirical distance distribution of same-emitter trace repeats.

    A molecule that is lost and immediately re-acquired appears as two
    consecutive traces separated by a very short time gap.  Candidate pairs are
    selected **on the time gap alone** — never on distance — so the resulting
    distance distribution is an unbiased sample of same-emitter separations
    rather than a restatement of a distance threshold.  The ``max_nm`` cut only
    discards the fraction of consecutive pairs where the instrument genuinely
    moved to a new molecule; it is applied after the time selection and is
    reported as ``rejected_far_fraction`` so its effect is visible.

    Falls back to a Maxwell kernel of scale ``fallback_sigma_nm`` when no time
    column is available or too few pairs survive.
    """
    pts = np.asarray(centroids_nm, dtype=np.float64)
    n_bins = int(np.floor(float(r_max_nm) / float(bin_nm) + 1e-9))
    centres = (np.arange(n_bins) + 0.5) * bin_nm

    def _maxwell(sigma: float) -> dict:
        pdf = maxwell_pdf(centres, sigma)
        area = pdf.sum() * bin_nm
        return {
            "shape": pdf / area if area > 0 else pdf,
            "centres_nm": centres, "bin_nm": float(bin_nm),
            "sigma_nm": float(sigma), "n_pairs": 0, "source": "assumed",
            "median_nm": float(1.5382 * sigma), "rejected_far_fraction": float("nan"),
        }

    if t_start is None or t_end is None or pts.shape[0] < 3:
        return _maxwell(fallback_sigma_nm)

    t0 = np.asarray(t_start, dtype=float).ravel()
    t1 = np.asarray(t_end, dtype=float).ravel()
    if t0.size != pts.shape[0] or t1.size != pts.shape[0]:
        return _maxwell(fallback_sigma_nm)

    order = np.argsort(t0)
    p = pts[order]
    gaps = t0[order][1:] - t1[order][:-1]
    dist = np.linalg.norm(p[1:] - p[:-1], axis=1)
    in_time = gaps < float(gap_s)
    n_time = int(in_time.sum())
    if n_time == 0:
        return _maxwell(fallback_sigma_nm)
    near = in_time & (dist <= float(max_nm))
    rejected = 1.0 - (near.sum() / n_time)
    if int(near.sum()) < int(min_pairs):
        out = _maxwell(fallback_sigma_nm)
        out["n_pairs"] = int(near.sum())
        out["rejected_far_fraction"] = float(rejected)
        return out

    sample = dist[near]
    median = float(np.median(sample))
    # A raw histogram of a few hundred samples over ~100 bins is far too noisy
    # to serve as a fixed model component, and its spikes would be absorbed by
    # the fitted amplitudes.  Smooth it with a KDE, which keeps the heavy tail
    # that drift gives the repeat population -- precisely the part that
    # overlaps the shortest sub-unit distance -- instead of imposing a Maxwell
    # form that would understate it.
    shape = None
    if sample.size >= 8:
        try:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(sample)
            dens = np.clip(kde(centres), 0.0, None)
            area = dens.sum() * bin_nm
            if area > 0:
                shape = dens / area
        except Exception:
            shape = None
    if shape is None:
        hist = np.histogram(sample, bins=np.arange(0.0, r_max_nm + bin_nm, bin_nm))[0]
        area = hist.sum() * bin_nm
        shape = hist / area if area > 0 else hist.astype(float)
    return {
        "shape": shape,
        "centres_nm": centres,
        "bin_nm": float(bin_nm),
        # median of a 3-D zero-mean Gaussian separation is 1.5382 sigma
        "sigma_nm": float(median / 1.5382),
        "n_pairs": int(near.sum()),
        "source": "empirical (consecutive traces, time-gap selected)",
        "median_nm": median,
        "rejected_far_fraction": float(rejected),
        "samples_nm": sample,
    }


def maxwell_pdf(r: np.ndarray, sigma: float) -> np.ndarray:
    """Distance distribution between two points whose offset is 3-D Gaussian."""
    r = np.asarray(r, dtype=float)
    s = max(float(sigma), 1e-9)
    return np.sqrt(2.0 / np.pi) * r ** 2 / s ** 3 * np.exp(-r ** 2 / (2.0 * s ** 2))


def offset_gaussian_pdf(r: np.ndarray, d: float, sigma: float) -> np.ndarray:
    """Distance distribution for a true separation ``d`` blurred in 3-D.

    Exact non-central chi (3 dof) density, written as a difference of Gaussians
    so it stays stable when ``d/sigma`` is large.  Reduces to
    :func:`maxwell_pdf` as ``d → 0``.
    """
    r = np.asarray(r, dtype=float)
    s = max(float(sigma), 1e-9)
    d = float(d)
    if d <= 1e-9:
        return maxwell_pdf(r, s)
    pre = r / (d * s * np.sqrt(2.0 * np.pi))
    return pre * (np.exp(-((r - d) ** 2) / (2 * s ** 2))
                  - np.exp(-((r + d) ** 2) / (2 * s ** 2)))


def complex_profile(
    centres_nm: np.ndarray,
    *,
    label_offset_nm: float,
    sigma_nm: float,
    distances_nm=HLYB_CLASS_DISTANCES_NM,
    weights=None,
    bin_nm: float = 0.5,
) -> np.ndarray:
    """Normalised intra-complex pair-distance profile of the HlyB model."""
    centres = np.asarray(centres_nm, dtype=float)
    dists = np.asarray(distances_nm, dtype=float)
    if weights is None:
        w = np.full(dists.size, 1.0 / dists.size)
    else:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum() if w.sum() > 0 else w
    out = np.zeros_like(centres)
    for d, wi in zip(dists, w):
        out += wi * offset_gaussian_pdf(centres, d + float(label_offset_nm), sigma_nm)
    area = out.sum() * bin_nm
    return out / area if area > 0 else out


# --------------------------------------------------------------------------
# Fit
# --------------------------------------------------------------------------

def _poisson_nll(observed: np.ndarray, expected: np.ndarray) -> float:
    exp = np.clip(expected, 1e-12, None)
    return float(np.sum(exp - observed * np.log(exp)))


def fit_pair_model(
    counts: np.ndarray,
    edges: np.ndarray,
    repeat_shape: np.ndarray,
    null_mean: np.ndarray,
    cfg: PairFitConfig,
    *,
    distances_nm=HLYB_CLASS_DISTANCES_NM,
    weights=None,
    fit_complex: bool = True,
    sigma_floor_nm: float | None = None,
    fixed_sigma_nm: float | None = None,
) -> dict:
    """Maximum-likelihood fit of the three-component model to the profile.

    Free parameters: repeat amplitude, complex amplitude, background scale, and
    — when enabled — the label offset δ and the blur σ.  The class *weights*
    are never free, and δ and σ are bounded by physics rather than left open
    (see :class:`PairFitConfig`), so a hypothesis cannot win by translating or
    inflating its peak into a generic background.  ``fixed_sigma_nm`` pins σ,
    which is how competing hypotheses are held to the same measurement blur.
    """
    from scipy.optimize import minimize

    obs = np.asarray(counts, dtype=float)
    centres = 0.5 * (np.asarray(edges[:-1]) + np.asarray(edges[1:]))
    bin_nm = float(edges[1] - edges[0])
    rep = np.asarray(repeat_shape, dtype=float)
    bkg = np.asarray(null_mean, dtype=float)
    n = min(obs.size, centres.size, rep.size, bkg.size)
    obs, centres, rep, bkg = obs[:n], centres[:n], rep[:n], bkg[:n]

    window = (centres >= cfg.fit_r_min_nm) & (centres <= cfg.fit_r_max_nm)
    if not window.any():
        window = np.ones(n, dtype=bool)

    def stretched_repeat(scale: float) -> np.ndarray:
        """The measured repeat kernel dilated radially by ``scale``.

        Dilation preserves the calibrated shape and normalisation while letting
        the fit account for the drift- and two-dye-broadened part of the
        population that a short-gap sample under-represents.
        """
        s = max(float(scale), 1e-6)
        if abs(s - 1.0) < 1e-9:
            return rep
        dens = np.interp(centres / s, centres, rep, left=0.0, right=0.0) / s
        area = dens.sum() * bin_nm
        return dens / area if area > 0 else dens

    s_floor = float(sigma_floor_nm if sigma_floor_nm is not None else cfg.sigma_floor_nm)
    s_ceiling = s_floor + float(cfg.sigma_headroom_nm)
    d_lo, d_hi = (float(cfg.label_offset_bounds_nm[0]),
                  float(cfg.label_offset_bounds_nm[1]))
    fit_sigma = cfg.fit_extra_sigma and fixed_sigma_nm is None
    if fixed_sigma_nm is not None:
        s_start = float(np.clip(fixed_sigma_nm, s_floor, s_ceiling))
    else:
        s_start = float(np.clip(cfg.extra_sigma_nm, s_floor, s_ceiling))

    r_lo, r_hi = (float(cfg.repeat_scale_bounds[0]), float(cfg.repeat_scale_bounds[1]))
    fit_scale = cfg.fit_repeat_scale and r_hi > r_lo

    def expected(params):
        n_rep, n_cx, a_bkg, delta, sig, scale = params
        model = (max(n_rep, 0.0) * stretched_repeat(scale) * bin_nm
                 + max(a_bkg, 0.0) * bkg)
        if fit_complex:
            prof = complex_profile(centres, label_offset_nm=delta,
                                   sigma_nm=max(sig, 0.2),
                                   distances_nm=distances_nm, weights=weights,
                                   bin_nm=bin_nm)
            model = model + max(n_cx, 0.0) * prof * bin_nm
        return model

    def unpack(free):
        p = list(free)
        delta = p[3] if cfg.fit_label_offset else float(cfg.label_offset_nm)
        sig = p[4] if fit_sigma else s_start
        scale = p[5] if fit_scale else r_lo
        return [p[0], p[1], p[2], delta, sig, scale]

    def nll(free):
        params = unpack(free)
        if params[0] < 0 or params[1] < 0 or params[2] < 0:
            return 1e18
        return _poisson_nll(obs[window], expected(params)[window])

    total = max(obs.sum(), 1.0)
    x0 = [0.25 * total, 0.25 * total if fit_complex else 0.0, 1.0,
          float(np.clip(cfg.label_offset_nm, d_lo, d_hi)), s_start,
          float(np.clip(1.3, r_lo, r_hi))]
    bounds = [
        (0.0, 10 * total), (0.0, 10 * total), (0.0, 100.0),
        (d_lo, d_hi) if cfg.fit_label_offset else (x0[3], x0[3]),
        (s_floor, s_ceiling) if fit_sigma else (s_start, s_start),
        (r_lo, r_hi) if fit_scale else (x0[5], x0[5]),
    ]
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    p = unpack(res.x)
    delta, sig, scale = p[3], p[4], p[5]
    model = expected(p)

    k = 3 + int(cfg.fit_label_offset) + int(fit_sigma) + int(fit_scale)
    if not fit_complex:
        k -= 1
    nll_value = _poisson_nll(obs[window], model[window])

    tol = 1e-3
    at_bounds = []
    if cfg.fit_label_offset and (abs(delta - d_lo) < tol or abs(delta - d_hi) < tol):
        at_bounds.append("label_offset_nm")
    if fit_sigma and (abs(sig - s_floor) < tol or abs(sig - s_ceiling) < tol):
        at_bounds.append("sigma_nm")
    if fit_scale and (abs(scale - r_lo) < tol or abs(scale - r_hi) < tol):
        at_bounds.append("repeat_scale")
    return {
        "n_repeat_pairs": float(p[0]),
        "n_complex_pairs": float(p[1]) if fit_complex else 0.0,
        "background_scale": float(p[2]),
        "label_offset_nm": float(delta),
        "sigma_nm": float(sig),
        "repeat_scale": float(scale),
        "model": model,
        "repeat_component": max(p[0], 0.0) * stretched_repeat(scale) * bin_nm,
        "background_component": max(p[2], 0.0) * bkg,
        "complex_component": (
            max(p[1], 0.0) * complex_profile(
                centres, label_offset_nm=delta, sigma_nm=max(sig, 0.2),
                distances_nm=distances_nm, weights=weights, bin_nm=bin_nm) * bin_nm
            if fit_complex else np.zeros_like(centres)
        ),
        "centres_nm": centres,
        "fit_window": window,
        "nll": nll_value,
        "n_parameters": int(k),
        "aic": float(2 * k + 2 * nll_value),
        "success": bool(res.success),
        "parameters_at_bounds": at_bounds,
    }


def compare_hypotheses(
    counts: np.ndarray,
    edges: np.ndarray,
    repeat_shape: np.ndarray,
    null_mean: np.ndarray,
    cfg: PairFitConfig,
    *,
    sigma_floor_nm: float | None = None,
) -> dict:
    """Fit the competing structural hypotheses and rank them by AIC.

    σ describes the measurement, not the hypothesis, so it is fitted **once**
    on the full six-site model and then held fixed for the alternatives.
    Letting each hypothesis choose its own blur would let a one-distance model
    inflate into a featureless bump and win for the wrong reason.
    """
    reference = fit_pair_model(counts, edges, repeat_shape, null_mean, cfg,
                               sigma_floor_nm=sigma_floor_nm)
    shared_sigma = float(reference["sigma_nm"])

    out: dict[str, dict] = {}
    for name in cfg.hypotheses:
        if name == "six_site":
            fit = reference
        elif name == "dimer_only":
            fit = fit_pair_model(counts, edges, repeat_shape, null_mean, cfg,
                                 distances_nm=(HLYB_CLASS_DISTANCES_NM[1],),
                                 weights=(1.0,), sigma_floor_nm=sigma_floor_nm,
                                 fixed_sigma_nm=shared_sigma)
        elif name == "no_structure":
            fit = fit_pair_model(counts, edges, repeat_shape, null_mean, cfg,
                                 fit_complex=False, sigma_floor_nm=sigma_floor_nm,
                                 fixed_sigma_nm=shared_sigma)
        else:
            continue
        fit["shared_sigma_nm"] = shared_sigma
        out[name] = fit
    if out:
        best = min(out.values(), key=lambda f: f["aic"])["aic"]
        for fit in out.values():
            fit["delta_aic"] = float(fit["aic"] - best)
    return out


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------

def analyze_hlyb_pairwise(
    loc_m: np.ndarray,
    tid: np.ndarray,
    tim: np.ndarray | None = None,
    cfg: PairFitConfig | None = None,
) -> dict:
    """Ensemble pair-distance analysis. See the module docstring."""
    cfg = cfg or PairFitConfig()
    traces = trace_centroids(loc_m, tid, tim, z_scale=cfg.z_scaling_factor,
                             min_loc_per_trace=cfg.min_loc_per_trace)
    pts = traces["centroids_nm"]

    counts, edges = pair_distance_profile(pts, cfg.r_max_nm, cfg.bin_nm)
    null = envelope_null(pts, r_max_nm=cfg.r_max_nm, bin_nm=cfg.bin_nm,
                         cell_nm=cfg.null_cell_nm, replicates=cfg.null_replicates,
                         rng_seed=cfg.rng_seed)
    repeat = calibrate_repeat_kernel(
        pts, traces["t_start"], traces["t_end"],
        gap_s=cfg.repeat_gap_s, max_nm=cfg.repeat_max_nm,
        min_pairs=cfg.repeat_min_pairs, fallback_sigma_nm=cfg.repeat_sigma_nm,
        r_max_nm=cfg.r_max_nm, bin_nm=cfg.bin_nm)

    sem = traces["sem_nm"]
    median_sem = (np.nanmedian(sem, axis=0) if sem.size else np.full(3, np.nan))
    # The blur of a pair distance cannot be smaller than the combined centroid
    # error of its two traces; measure that floor rather than assuming one.
    # ``sigma`` in the kernels is the PER-AXIS spread of the separation vector,
    # so the per-axis centroid error is needed here, not its 3-D magnitude, and
    # two independent centroids contribute in quadrature.
    if np.isfinite(median_sem).any():
        per_axis = float(np.sqrt(np.nanmean(median_sem ** 2)))
    else:
        per_axis = 0.0
    sigma_floor = float(max(np.sqrt(2.0) * per_axis, 0.5))

    import dataclasses

    fits = compare_hypotheses(counts, edges, repeat["shape"], null["mean"], cfg,
                              sigma_floor_nm=sigma_floor)
    best_name = min(fits, key=lambda k: fits[k]["aic"]) if fits else ""
    best = fits.get(best_name, {})

    # Sensitivity pass: repeat the entire comparison with the short-range
    # kernel allowed to broaden.  Releasing it improves the description of the
    # 5-9 nm region but weakens the structural discrimination, and the reader
    # is entitled to see by how much rather than being shown only the more
    # favourable of the two.
    relaxed_cfg = dataclasses.replace(cfg, fit_repeat_scale=not cfg.fit_repeat_scale)
    relaxed = compare_hypotheses(counts, edges, repeat["shape"], null["mean"],
                                 relaxed_cfg, sigma_floor_nm=sigma_floor)
    relaxed_best = min(relaxed, key=lambda k: relaxed[k]["aic"]) if relaxed else ""

    centres = 0.5 * (edges[:-1] + edges[1:])
    excess = counts.astype(float) - null["mean"]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(null["sd"] > 0, excess / null["sd"], np.nan)
    finite = np.isfinite(z) & (centres <= cfg.fit_r_max_nm)
    outer = 0.0
    if finite.any():
        significant = finite & (z > 3.0)
        outer = float(centres[significant].max()) if significant.any() else 0.0

    return {
        "centres_nm": centres,
        "edges_nm": edges,
        "counts": counts,
        "null_mean": null["mean"],
        "null_sd": null["sd"],
        "null_replicates": int(null["replicates"]),
        "excess": excess,
        "excess_z": z,
        "excess_outer_nm": outer,
        "repeat_kernel": repeat,
        "fits": fits,
        "best_hypothesis": best_name,
        "best_fit": best,
        "fits_relaxed_kernel": relaxed,
        "best_hypothesis_relaxed": relaxed_best,
        "n_traces_total": int(traces["n_traces_total"]),
        "n_traces_used": int(pts.shape[0]),
        "centroid_sem_nm": median_sem,
        "sigma_floor_nm": sigma_floor,
        # Kept so the spatial view can show the actual observable.  These are
        # TRACE centroids, not sub-unit centres: several of them may belong to
        # one labelled site, which is exactly the population the repeat kernel
        # describes and which this method never merges away.
        "centroids_nm": pts,
        "centroid_n_locs": traces["n_locs"],
        "points_nm": traces["points_nm"],
        "repeat_pairs": repeat_pair_index(pts, traces["t_start"], traces["t_end"],
                                          gap_s=cfg.repeat_gap_s,
                                          max_nm=cfg.repeat_max_nm),
        "class_names": list(HLYB_CLASS_NAMES),
        "class_distances_nm": list(HLYB_CLASS_DISTANCES_NM),
        "config": cfg,
        "is_pairwise": True,
    }
