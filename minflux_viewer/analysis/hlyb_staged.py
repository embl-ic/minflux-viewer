"""Model-independent staged 3-D analysis of short-range HlyB populations.

This module is the conservative successor to the HlyB template and parametric
pair-fit analyses.  It does not assign sites to a six-subunit template and it
does not report a fitted molecular distance.  Instead it asks the narrower
question supported by the data: is there an excess of distinct inferred
labelling-site pairs at short range, relative to a conditional randomization
that preserves each cell's observed rod/membrane geometry?

The stages are:

1. localization records -> trace centroids with measured standard errors;
2. conservative, uncertainty-aware consolidation of repeated traces into
   labelling sites (no LoG image and no DBSCAN density threshold);
3. coarse separation into spatial/cell components;
4. exact within-component pair-distance profile;
5. rod-surface conditional randomization, preserving the observed axial
   coordinates and locally permuting transverse membrane coordinates;
6. model-free excess statistics and cell/component-level bootstrap intervals;
7. mandatory sensitivity checks for the site-consolidation radius and null
   stratification scale.

Coordinates supplied to :func:`analyze_hlyb_staged_3d` are raw metres.  The
configured RIMF is applied exactly once here.  Pure NumPy/SciPy; no Qt.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .hlyb_pairwise import pair_distance_profile, trace_centroids


@dataclass
class Staged3DConfig:
    """Parameters for the model-independent staged 3-D workflow."""

    min_loc_per_trace: int = 10
    z_scaling_factor: float = 0.67

    # Same-site consolidation.  ``site_merge_nm`` is a hard maximum cluster
    # diameter, deliberately well below the expected HlyB dimer scale.
    site_merge_nm: float = 4.0
    site_sigma_factor: float = 3.0
    site_precision_floor_nm: float = 0.5

    # Spatial components approximate individual cells/FOV objects.  The
    # analysis never forms pairs across components.
    cell_link_nm: float = 180.0
    min_sites_per_component: int = 20

    # Pair observable and pre-declared short-range band.
    r_max_nm: float = 60.0
    bin_nm: float = 0.5
    # Eight nanometres is twice the default same-site consolidation diameter.
    # The intervening 4--8 nm transition is displayed but not used for the
    # primary population test, limiting contamination from split label sites.
    short_range_lo_nm: float = 8.0
    short_range_hi_nm: float = 25.0

    # Conditional rod-surface null.  Within each axial rank stratum, complete
    # transverse (v,w) coordinates are permuted while u is held fixed.
    null_stratum_sites: int = 64
    null_replicates: int = 99
    rng_seed: int = 0

    # Component bootstrap.  It is intentionally unavailable when too few
    # independent components are present; no row/pair pseudo-replication.
    bootstrap_replicates: int = 399

    # Robustness audit run with fewer randomizations per variant.
    run_sensitivity: bool = True
    sensitivity_replicates: int = 31
    sensitivity_site_merge_nm: tuple[float, ...] = (3.0, 4.0, 5.0)
    sensitivity_stratum_sites: tuple[int, ...] = (32, 64, 128)
    sensitivity_cell_link_factors: tuple[float, ...] = (0.75, 1.0, 1.25)

    # Stratum profile.  ``null_stratum_sites`` sets the axial window inside
    # which the randomization destroys structure, so it also sets how much
    # structure the null absorbs: ``band_ratio`` is conditional on it and is
    # not an absolute effect size.  This scan varies the null alone, holding
    # the inferred sites and components fixed, so the dependence is visible
    # rather than buried in the sensitivity audit (which varies them too).
    run_stratum_profile: bool = True
    stratum_profile_sites: tuple[int, ...] = (16, 32, 64, 128, 256)
    # Above this max/min spread the ratio is reported as stratum-conditional.
    stratum_profile_ratio_tolerance: float = 1.5

    # Calibrated per-variant robustness criterion.  ``band_p`` is bounded below
    # by 1/(replicates + 1) and was measured to be anti-conservative, so the
    # calibrated flag requires a ratio this many null-ratio standard deviations
    # above the null instead of a nominal significance level.
    calibrated_ratio_z: float = 3.0


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, value: int) -> int:
        root = int(value)
        while self.parent[root] != root:
            self.parent[root] = self.parent[self.parent[root]]
            root = int(self.parent[root])
        while self.parent[value] != value:
            nxt = int(self.parent[value])
            self.parent[value] = root
            value = nxt
        return root

    def union(self, left: int, right: int) -> int:
        a, b = self.find(left), self.find(right)
        if a == b:
            return a
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]
        return a

    def labels(self) -> np.ndarray:
        roots = np.asarray([self.find(i) for i in range(self.parent.size)])
        _, labels = np.unique(roots, return_inverse=True)
        return labels.astype(np.int64)


def _validate_config(cfg: Staged3DConfig) -> None:
    if cfg.min_loc_per_trace < 1:
        raise ValueError("min_loc_per_trace must be at least 1")
    if not 0 < cfg.z_scaling_factor <= 2:
        raise ValueError("z_scaling_factor must be in (0, 2]")
    if cfg.site_merge_nm <= 0:
        raise ValueError("site_merge_nm must be positive")
    if cfg.cell_link_nm <= cfg.site_merge_nm:
        raise ValueError("cell_link_nm must exceed site_merge_nm")
    if cfg.min_sites_per_component < 3:
        raise ValueError("min_sites_per_component must be at least 3")
    if cfg.bin_nm <= 0 or cfg.r_max_nm <= cfg.bin_nm:
        raise ValueError("r_max_nm must exceed a positive bin_nm")
    if not 0 <= cfg.short_range_lo_nm < cfg.short_range_hi_nm <= cfg.r_max_nm:
        raise ValueError("short-range bounds must lie inside [0, r_max_nm]")
    if cfg.null_replicates < 9:
        raise ValueError("at least 9 null replicates are required")
    if cfg.null_stratum_sites < 4:
        raise ValueError("null_stratum_sites must be at least 4")
    if cfg.run_stratum_profile and any(int(s) < 4 for s in cfg.stratum_profile_sites):
        raise ValueError("every stratum_profile_sites entry must be at least 4")
    if cfg.stratum_profile_ratio_tolerance <= 1.0:
        raise ValueError("stratum_profile_ratio_tolerance must exceed 1")


def infer_label_sites(
    centroids_nm: np.ndarray,
    sem_nm: np.ndarray,
    n_locs: np.ndarray,
    t_start: np.ndarray | None,
    t_end: np.ndarray | None,
    *,
    merge_nm: float = 4.0,
    sigma_factor: float = 3.0,
    precision_floor_nm: float = 0.5,
) -> dict:
    """Conservatively consolidate repeated trace centroids into label sites.

    Candidate edges are uncertainty-aware, but a merge is accepted only when
    *every* trace-centroid pair in the combined cluster remains within the hard
    ``merge_nm`` diameter and the weighted cluster RMS remains below half that
    diameter.  The complete-link condition prevents transitive chains from
    joining two distinct nearby sites.  Acquisition time is not thresholded:
    repeat visits minutes apart remain eligible, and their full time span is
    reported as a diagnostic.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(centroids_nm, dtype=float)
    sem = np.asarray(sem_nm, dtype=float)
    nloc = np.asarray(n_locs, dtype=int).ravel()
    if pts.ndim != 2 or pts.shape[1] != 3 or sem.shape != pts.shape:
        raise ValueError("centroids_nm and sem_nm must both have shape (N, 3)")
    if nloc.size != pts.shape[0]:
        raise ValueError("n_locs must align with centroids_nm")
    finite = np.isfinite(pts).all(axis=1)
    if not finite.all():
        pts = pts[finite]
        sem = sem[finite]
        nloc = nloc[finite]
        if t_start is not None:
            t_start = np.asarray(t_start, dtype=float).ravel()[finite]
        if t_end is not None:
            t_end = np.asarray(t_end, dtype=float).ravel()[finite]
    n = pts.shape[0]
    if n == 0:
        empty3 = np.empty((0, 3), dtype=float)
        return {
            "centers_nm": empty3, "sem_nm": empty3.copy(),
            "trace_to_site": np.empty(0, dtype=np.int64),
            "n_traces": np.empty(0, dtype=int), "n_locs": np.empty(0, dtype=int),
            "t_start": np.empty(0), "t_end": np.empty(0),
            "within_site_rms_nm": np.empty(0), "n_candidate_edges": 0,
            "n_accepted_merges": 0,
        }

    axis_sem = sem.copy()
    finite_sem = axis_sem[np.isfinite(axis_sem) & (axis_sem > 0)]
    fallback = float(np.median(finite_sem)) if finite_sem.size else 1.0
    axis_sem[~np.isfinite(axis_sem) | (axis_sem <= 0)] = fallback
    scalar_sem = np.sqrt(np.mean(axis_sem ** 2, axis=1))

    tree = cKDTree(pts)
    edges = tree.query_pairs(float(merge_nm), output_type="ndarray")
    if edges.size:
        dist = np.linalg.norm(pts[edges[:, 1]] - pts[edges[:, 0]], axis=1)
        uncertainty = np.sqrt(
            scalar_sem[edges[:, 0]] ** 2
            + scalar_sem[edges[:, 1]] ** 2
            + float(precision_floor_nm) ** 2
        )
        allowed = np.minimum(float(merge_nm), float(sigma_factor) * uncertainty)
        keep = dist <= allowed
        edges, dist, uncertainty = edges[keep], dist[keep], uncertainty[keep]
        order = np.argsort(dist / np.maximum(uncertainty, 1e-9), kind="stable")
        edges = edges[order]

    uf = _UnionFind(n)
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    accepted = 0
    scalar_weight = 1.0 / (scalar_sem ** 2 + float(precision_floor_nm) ** 2)
    for left, right in edges:
        a, b = uf.find(int(left)), uf.find(int(right))
        if a == b:
            continue
        ma, mb = members[a], members[b]
        cross = pts[np.asarray(ma)][:, None, :] - pts[np.asarray(mb)][None, :, :]
        if float(np.max(np.linalg.norm(cross, axis=2))) > float(merge_nm):
            continue
        joined = np.asarray(ma + mb, dtype=np.int64)
        w = scalar_weight[joined]
        centre = np.average(pts[joined], axis=0, weights=w)
        rms = np.sqrt(np.average(np.sum((pts[joined] - centre) ** 2, axis=1), weights=w))
        if float(rms) > 0.5 * float(merge_nm):
            continue
        root = uf.union(a, b)
        other = b if root == a else a
        members[root] = ma + mb
        members.pop(other, None)
        accepted += 1

    labels = uf.labels()
    n_sites = int(labels.max()) + 1
    centers = np.empty((n_sites, 3), dtype=float)
    site_sem = np.empty((n_sites, 3), dtype=float)
    trace_counts = np.empty(n_sites, dtype=int)
    loc_counts = np.empty(n_sites, dtype=int)
    within_rms = np.zeros(n_sites, dtype=float)
    site_t0 = np.full(n_sites, np.nan)
    site_t1 = np.full(n_sites, np.nan)
    ts = None if t_start is None else np.asarray(t_start, dtype=float).ravel()
    te = None if t_end is None else np.asarray(t_end, dtype=float).ravel()

    for site in range(n_sites):
        idx = np.flatnonzero(labels == site)
        weights = 1.0 / (axis_sem[idx] ** 2 + float(precision_floor_nm) ** 2)
        denom = weights.sum(axis=0)
        centers[site] = (weights * pts[idx]).sum(axis=0) / denom
        residual = pts[idx] - centers[site]
        between = np.average(residual ** 2, axis=0, weights=weights.mean(axis=1))
        site_sem[site] = np.sqrt(1.0 / denom + between / max(idx.size, 1))
        trace_counts[site] = idx.size
        loc_counts[site] = int(nloc[idx].sum())
        within_rms[site] = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
        if ts is not None and te is not None and ts.size == n:
            site_t0[site] = float(np.nanmin(ts[idx]))
            site_t1[site] = float(np.nanmax(te[idx]))

    return {
        "centers_nm": centers,
        "sem_nm": site_sem,
        "trace_to_site": labels,
        "n_traces": trace_counts,
        "n_locs": loc_counts,
        "t_start": site_t0,
        "t_end": site_t1,
        "within_site_rms_nm": within_rms,
        "n_candidate_edges": int(edges.shape[0]),
        "n_accepted_merges": int(accepted),
    }


def segment_spatial_components(
    sites_nm: np.ndarray,
    *,
    link_nm: float = 180.0,
    min_sites: int = 20,
) -> dict:
    """Separate coarse cell/FOV components without forming cross-cell pairs."""
    from scipy.spatial import cKDTree

    pts = np.asarray(sites_nm, dtype=float)
    n = pts.shape[0]
    if n == 0:
        return {"labels": np.empty(0, dtype=np.int64), "components": [],
                "n_excluded_sites": 0, "n_all_components": 0}
    uf = _UnionFind(n)
    edges = cKDTree(pts).query_pairs(float(link_nm), output_type="ndarray")
    for i, j in edges:
        uf.union(int(i), int(j))
    raw = uf.labels()
    ids, counts = np.unique(raw, return_counts=True)
    keep_ids = ids[counts >= int(min_sites)]
    labels = np.full(n, -1, dtype=np.int64)
    components = []
    for new_id, old_id in enumerate(keep_ids):
        idx = np.flatnonzero(raw == old_id)
        labels[idx] = new_id
        block = pts[idx]
        centre = block.mean(axis=0)
        centered = block - centre
        cov = centered.T @ centered / max(block.shape[0] - 1, 1)
        values, vectors = np.linalg.eigh(cov)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        q = centered @ vectors
        aspect = float(np.sqrt(max(values[0], 0) / max(values[1], 1e-12)))
        components.append({
            "id": int(new_id), "indices": idx, "center_nm": centre,
            "axes": vectors, "eigenvalues_nm2": values,
            "aspect_ratio": aspect,
            "axial_span_nm": float(np.ptp(q[:, 0])),
            "n_sites": int(idx.size), "rod_like": bool(aspect >= 1.25),
        })
    return {
        "labels": labels,
        "components": components,
        "n_excluded_sites": int(np.sum(labels < 0)),
        "n_all_components": int(ids.size),
        "n_graph_edges": int(edges.shape[0]),
    }


def _profile_components(
    points_nm: np.ndarray,
    labels: np.ndarray,
    *,
    r_max_nm: float,
    bin_nm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points_nm, dtype=float)
    lab = np.asarray(labels, dtype=np.int64)
    component_ids = np.unique(lab[lab >= 0])
    rows = []
    edges = np.arange(0.0, float(r_max_nm) + float(bin_nm), float(bin_nm))
    for component in component_ids:
        hist, edges = pair_distance_profile(pts[lab == component], r_max_nm, bin_nm)
        rows.append(hist.astype(float))
    matrix = np.vstack(rows) if rows else np.zeros((0, edges.size - 1), dtype=float)
    return matrix.sum(axis=0), matrix, edges


def _surface_models(points_nm: np.ndarray, components: list[dict]) -> list[dict]:
    pts = np.asarray(points_nm, dtype=float)
    models = []
    for component in components:
        idx = np.asarray(component["indices"], dtype=np.int64)
        q = (pts[idx] - component["center_nm"]) @ component["axes"]
        models.append({**component, "local_coordinates_nm": q})
    return models


def surface_conditioned_null(
    sites_nm: np.ndarray,
    components: list[dict],
    *,
    r_max_nm: float = 60.0,
    bin_nm: float = 0.5,
    stratum_sites: int = 64,
    replicates: int = 99,
    rng_seed: int = 0,
) -> dict:
    """Conditional randomization on the observed rod/membrane support.

    PCA supplies only the longitudinal coordinate system; it does not impose a
    radius or fill a volume.  For each component, observed axial coordinates
    are held exactly fixed.  Complete observed transverse vectors are permuted
    within adjacent axial-rank strata, so each surrogate preserves exact site
    count, axial density, local radial distribution, one-sided visibility and
    field boundaries while breaking nanoscale axial/transverse association.
    """
    pts = np.asarray(sites_nm, dtype=float)
    models = _surface_models(pts, components)
    rng = np.random.default_rng(int(rng_seed))
    n_bins = int(np.ceil(float(r_max_nm) / float(bin_nm)))
    stack = np.zeros((int(replicates), n_bins), dtype=float)
    component_stack = np.zeros((int(replicates), len(models), n_bins), dtype=float)
    preview = np.empty((0, 3), dtype=float)

    for rep in range(int(replicates)):
        total = np.zeros(n_bins, dtype=float)
        preview_parts = []
        for ci, model in enumerate(models):
            q = np.asarray(model["local_coordinates_nm"], dtype=float).copy()
            order = np.argsort(q[:, 0], kind="stable")
            for start in range(0, order.size, int(stratum_sites)):
                idx = order[start:start + int(stratum_sites)]
                if idx.size > 1:
                    q[idx, 1:] = q[rng.permutation(idx), 1:]
            sample = q @ np.asarray(model["axes"]).T + np.asarray(model["center_nm"])
            hist, _ = pair_distance_profile(sample, r_max_nm, bin_nm)
            component_stack[rep, ci] = hist
            total += hist
            if rep == 0:
                preview_parts.append(sample)
        stack[rep] = total
        if rep == 0 and preview_parts:
            preview = np.vstack(preview_parts)

    return {
        "mean": stack.mean(axis=0),
        "sd": stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros(n_bins),
        "lo": np.quantile(stack, 0.025, axis=0),
        "hi": np.quantile(stack, 0.975, axis=0),
        "profiles": stack,
        "component_mean": component_stack.mean(axis=0),
        "preview_sites_nm": preview,
        "replicates": int(stack.shape[0]),
        "stratum_sites": int(stratum_sites),
    }


def _band_descriptors(
    observed: np.ndarray,
    null_mean: np.ndarray,
    centers_nm: np.ndarray,
    band: np.ndarray,
) -> dict:
    """Deterministic band descriptors of one profile against one null mean.

    Shared by :func:`_excess_summary` and the component bootstrap so the two
    can never drift apart.  Takes a single expected profile, so it needs no
    replicate stack and fabricates none.
    """
    obs = np.asarray(observed, dtype=float)
    mean = np.asarray(null_mean, dtype=float)
    excess = obs - mean
    band_centers = np.asarray(centers_nm, dtype=float)[band]
    positive = np.clip(excess[band], 0.0, None)
    obs_count = float(obs[band].sum())
    expected = float(mean[band].sum())
    ratio = obs_count / expected if expected > 0 else float("nan")
    if positive.sum() > 0:
        peak = float(band_centers[np.argmax(positive)])
        centroid = float(np.sum(band_centers * positive) / positive.sum())
        cumulative = np.cumsum(positive)
        median = float(band_centers[np.searchsorted(cumulative, 0.5 * cumulative[-1])])
    else:
        peak = centroid = median = float("nan")
    return {
        "band_observed_pairs": obs_count,
        "band_expected_pairs": expected,
        "band_ratio": ratio,
        "peak_nm": peak,
        "positive_excess_centroid_nm": centroid,
        "positive_excess_median_nm": median,
        "excess_counts": excess,
    }


def _null_ratio_distribution(null_band_counts: np.ndarray) -> np.ndarray:
    """Leave-one-out band ratios of the null replicates against each other.

    Each replicate is divided by the mean of the *remaining* replicates, which
    is the same construction as the observed ratio (a band count over a null
    mean).  It therefore gives a reference distribution on the ratio scale, so
    ``band_ratio`` can be reported as "2.04 against a null of 1.00 +/- 0.02"
    and scored by an unbounded z -- unlike the rank-based p-value, which
    saturates at ``1/(replicates + 1)`` and cannot separate a strong effect
    from an overwhelming one.

    Being a ratio-scale rescaling of the same replicate counts, the resulting
    z is close to ``band_z`` by construction; it is not independent evidence.
    Its value is that it is expressed in the units actually quoted and that it
    exposes the null spread of the ratio explicitly.  On the reference dataset
    it reads 48.3, against 45.9 measured externally from fresh surrogates.
    """
    counts = np.asarray(null_band_counts, dtype=float)
    n = counts.size
    if n < 3:
        return np.empty(0, dtype=float)
    loo_mean = (counts.sum() - counts) / (n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(loo_mean > 0, counts / loo_mean, np.nan)
    return ratios[np.isfinite(ratios)]


def _excess_summary(
    observed: np.ndarray,
    null_profiles: np.ndarray,
    edges_nm: np.ndarray,
    *,
    lo_nm: float,
    hi_nm: float,
) -> dict:
    obs = np.asarray(observed, dtype=float)
    null = np.asarray(null_profiles, dtype=float)
    centers = 0.5 * (edges_nm[:-1] + edges_nm[1:])
    band = (centers >= float(lo_nm)) & (centers < float(hi_nm))
    mean = null.mean(axis=0)
    sd = null.std(axis=0, ddof=1) if null.shape[0] > 1 else np.zeros_like(mean)
    descriptors = _band_descriptors(obs, mean, centers, band)
    excess = descriptors["excess_counts"]
    obs_count = descriptors["band_observed_pairs"]
    null_count = null[:, band].sum(axis=1)
    expected = float(null_count.mean())
    count_sd = float(null_count.std(ddof=1)) if null_count.size > 1 else 0.0
    p = float((1 + np.sum(null_count >= obs_count)) / (null_count.size + 1))
    ratio = descriptors["band_ratio"]
    z = (obs_count - expected) / count_sd if count_sd > 0 else float("nan")

    # Calibrated effect size.  ``band_p`` is bounded below by 1/(R+1) and was
    # measured to be anti-conservative (it re-segments and re-stratifies), so
    # the ratio and its distance from the null ratio distribution are the
    # statistics to report.
    null_ratios = _null_ratio_distribution(null_count)
    if null_ratios.size >= 3:
        nr_mean = float(null_ratios.mean())
        nr_sd = float(null_ratios.std(ddof=1))
        ratio_z = ((ratio - nr_mean) / nr_sd
                   if nr_sd > 0 and np.isfinite(ratio) else float("nan"))
    else:
        nr_mean = nr_sd = ratio_z = float("nan")

    safe_sd = np.where(sd > 0, sd, np.inf)
    z_curve = excess / safe_sd
    obs_max = float(np.nanmax(z_curve[band])) if band.any() else float("nan")
    null_z = (null - mean) / safe_sd
    null_max = np.nanmax(null_z[:, band], axis=1) if band.any() else np.empty(0)
    max_p = (float((1 + np.sum(null_max >= obs_max)) / (null_max.size + 1))
             if null_max.size and np.isfinite(obs_max) else float("nan"))
    return {
        "band_observed_pairs": int(obs_count),
        "band_null_mean_pairs": expected,
        "band_null_sd_pairs": count_sd,
        "band_ratio": ratio,
        "band_z": z,
        "band_p": p,
        # Smallest p this many randomizations can express.  A ``band_p`` equal
        # to it is censored, not measured.
        "band_p_resolution": float(1.0 / (null_count.size + 1)),
        "band_ratio_z": ratio_z,
        "null_band_ratio_mean": nr_mean,
        "null_band_ratio_sd": nr_sd,
        "peak_nm": descriptors["peak_nm"],
        "positive_excess_centroid_nm": descriptors["positive_excess_centroid_nm"],
        "positive_excess_median_nm": descriptors["positive_excess_median_nm"],
        "pointwise_z": z_curve,
        "max_pointwise_z": obs_max,
        "max_pointwise_p": max_p,
        "excess_counts": excess,
    }


def _component_bootstrap(
    observed_by_component: np.ndarray,
    null_by_component: np.ndarray,
    edges_nm: np.ndarray,
    cfg: Staged3DConfig,
) -> dict:
    obs = np.asarray(observed_by_component, dtype=float)
    null = np.asarray(null_by_component, dtype=float)
    n = obs.shape[0]
    if n < 3 or cfg.bootstrap_replicates < 10:
        return {"available": False, "n_components": int(n), "reason":
                "fewer than three independent spatial components"}
    rng = np.random.default_rng(int(cfg.rng_seed) + 7919)
    centers = 0.5 * (edges_nm[:-1] + edges_nm[1:])
    band = ((centers >= float(cfg.short_range_lo_nm))
            & (centers < float(cfg.short_range_hi_nm)))
    values = []
    ratios = []
    for _ in range(int(cfg.bootstrap_replicates)):
        pick = rng.integers(0, n, size=n)
        # Deterministic descriptors against the resampled component null means.
        # The resample supplies one expected profile, so the descriptors are
        # computed directly rather than through the replicate-stack summary.
        d = _band_descriptors(obs[pick].sum(axis=0), null[pick].sum(axis=0),
                              centers, band)
        values.append(d["positive_excess_centroid_nm"])
        ratios.append(d["band_ratio"])
    values = np.asarray(values, dtype=float)
    ratios = np.asarray(ratios, dtype=float)
    values = values[np.isfinite(values)]
    ratios = ratios[np.isfinite(ratios)]
    return {
        "available": bool(values.size and ratios.size),
        "n_components": int(n),
        "replicates": int(cfg.bootstrap_replicates),
        "centroid_ci95_nm": (np.quantile(values, [0.025, 0.975]).tolist()
                             if values.size else [float("nan"), float("nan")]),
        "band_ratio_ci95": (np.quantile(ratios, [0.025, 0.975]).tolist()
                            if ratios.size else [float("nan"), float("nan")]),
        # Resampling only a handful of components cannot express the
        # between-component variance faithfully; the sensitivity spread is the
        # honest uncertainty to quote at these component counts.
        "narrow_ci_warning": bool(n < 5),
    }


def _stratum_profile(base: dict, cfg: Staged3DConfig) -> dict:
    """Vary the null stratification alone, holding sites and components fixed.

    Isolates how much of ``band_ratio`` is set by the randomization scale
    rather than by the data.  The excess *location* is expected to be stable
    across the scan; the ratio's magnitude is not, and reporting it without the
    stratum it was computed at would overstate a conditional number.
    """
    sites = base["sites"]["centers_nm"]
    components = base["components"]["components"]
    rows: list[dict] = []
    for i, stratum in enumerate(sorted({int(s) for s in cfg.stratum_profile_sites})):
        null = surface_conditioned_null(
            sites, components, r_max_nm=cfg.r_max_nm, bin_nm=cfg.bin_nm,
            stratum_sites=stratum, replicates=cfg.sensitivity_replicates,
            rng_seed=cfg.rng_seed + 3571 * (i + 1),
        )
        s = _excess_summary(
            base["observed"], null["profiles"], base["edges_nm"],
            lo_nm=cfg.short_range_lo_nm, hi_nm=cfg.short_range_hi_nm,
        )
        rows.append({
            "null_stratum_sites": stratum,
            "band_ratio": float(s["band_ratio"]),
            "band_ratio_z": float(s["band_ratio_z"]),
            "band_p": float(s["band_p"]),
            "positive_excess_centroid_nm": float(s["positive_excess_centroid_nm"]),
            "peak_nm": float(s["peak_nm"]),
        })
    ratios = np.asarray([r["band_ratio"] for r in rows], dtype=float)
    centroids = np.asarray([r["positive_excess_centroid_nm"] for r in rows],
                           dtype=float)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    centroids = centroids[np.isfinite(centroids)]
    spread = (float(ratios.max() / ratios.min()) if ratios.size else float("nan"))
    return {
        "rows": rows,
        "band_ratio_range": ([float(ratios.min()), float(ratios.max())]
                             if ratios.size else [float("nan")] * 2),
        "band_ratio_spread": spread,
        "centroid_range_nm": ([float(centroids.min()), float(centroids.max())]
                              if centroids.size else [float("nan")] * 2),
        # True when the ratio moves more than the configured tolerance across
        # the scan, i.e. when it must be quoted together with its stratum.
        "band_ratio_is_stratum_conditional": bool(
            np.isfinite(spread)
            and spread > float(cfg.stratum_profile_ratio_tolerance)),
    }


def _analyze_sites(
    traces: dict,
    cfg: Staged3DConfig,
    *,
    merge_nm: float,
    cell_link_nm: float,
    stratum_sites: int,
    null_replicates: int,
    rng_seed: int,
) -> dict:
    sites = infer_label_sites(
        traces["centroids_nm"], traces["sem_nm"], traces["n_locs"],
        traces["t_start"], traces["t_end"], merge_nm=merge_nm,
        sigma_factor=cfg.site_sigma_factor,
        precision_floor_nm=cfg.site_precision_floor_nm,
    )
    components = segment_spatial_components(
        sites["centers_nm"], link_nm=cell_link_nm,
        min_sites=cfg.min_sites_per_component,
    )
    observed, observed_by_component, edges = _profile_components(
        sites["centers_nm"], components["labels"],
        r_max_nm=cfg.r_max_nm, bin_nm=cfg.bin_nm,
    )
    null = surface_conditioned_null(
        sites["centers_nm"], components["components"],
        r_max_nm=cfg.r_max_nm, bin_nm=cfg.bin_nm,
        stratum_sites=stratum_sites, replicates=null_replicates,
        rng_seed=rng_seed,
    )
    summary = _excess_summary(
        observed, null["profiles"], edges,
        lo_nm=cfg.short_range_lo_nm, hi_nm=cfg.short_range_hi_nm,
    )
    return {
        "sites": sites, "components": components,
        "observed": observed, "observed_by_component": observed_by_component,
        "edges_nm": edges, "null": null, "summary": summary,
    }


def analyze_hlyb_staged_3d(
    loc_m: np.ndarray,
    tid: np.ndarray,
    tim: np.ndarray | None = None,
    cfg: Staged3DConfig | None = None,
) -> dict:
    """Run the staged model-independent HlyB short-range analysis."""
    cfg = cfg or Staged3DConfig()
    _validate_config(cfg)
    loc = np.asarray(loc_m, dtype=float)
    if loc.ndim != 2 or loc.shape[1] < 3:
        raise ValueError("loc_m must have shape (N, 3)")
    finite_z = loc[np.isfinite(loc).all(axis=1), 2]
    if finite_z.size < 3 or float(np.ptp(finite_z)) * 1e9 * cfg.z_scaling_factor < 5.0:
        raise ValueError("The staged workflow requires a genuinely 3-D dataset")

    traces = trace_centroids(
        loc, tid, tim, z_scale=cfg.z_scaling_factor,
        min_loc_per_trace=cfg.min_loc_per_trace,
    )
    if traces["centroids_nm"].shape[0] < cfg.min_sites_per_component:
        raise ValueError("Too few qualifying traces for a 3-D spatial component")

    base = _analyze_sites(
        traces, cfg, merge_nm=cfg.site_merge_nm,
        cell_link_nm=cfg.cell_link_nm,
        stratum_sites=cfg.null_stratum_sites,
        null_replicates=cfg.null_replicates, rng_seed=cfg.rng_seed,
    )
    if not base["components"]["components"]:
        raise ValueError(
            "No spatial component contains the configured minimum number of inferred sites")

    sites = base["sites"]
    components = base["components"]
    summary = base["summary"]
    bootstrap = _component_bootstrap(
        base["observed_by_component"], base["null"]["component_mean"],
        base["edges_nm"], cfg,
    )

    sensitivity = []
    if cfg.run_sensitivity:
        seen: set[tuple[float, float, int]] = set()
        variants = [(float(v), float(cfg.cell_link_nm),
                     int(cfg.null_stratum_sites), "site radius")
                    for v in cfg.sensitivity_site_merge_nm]
        variants += [(float(cfg.site_merge_nm), float(cfg.cell_link_nm),
                      int(v), "null stratum")
                     for v in cfg.sensitivity_stratum_sites]
        variants += [(float(cfg.site_merge_nm),
                      float(cfg.cell_link_nm) * float(factor),
                      int(cfg.null_stratum_sites), "component link")
                     for factor in cfg.sensitivity_cell_link_factors]
        for vi, (merge_nm, cell_link_nm, stratum, source) in enumerate(variants):
            key = (round(merge_nm, 6), round(cell_link_nm, 6), stratum)
            if key in seen:
                continue
            seen.add(key)
            if (abs(merge_nm - cfg.site_merge_nm) < 1e-9
                    and abs(cell_link_nm - cfg.cell_link_nm) < 1e-9
                    and stratum == cfg.null_stratum_sites):
                variant = base
            else:
                variant = _analyze_sites(
                    traces, cfg, merge_nm=merge_nm, cell_link_nm=cell_link_nm,
                    stratum_sites=stratum,
                    null_replicates=cfg.sensitivity_replicates,
                    rng_seed=cfg.rng_seed + 1009 * (vi + 1),
                )
            s = variant["summary"]
            sensitivity.append({
                "source": source, "site_merge_nm": merge_nm,
                "cell_link_nm": cell_link_nm,
                "null_stratum_sites": stratum,
                "n_sites": int(variant["sites"]["centers_nm"].shape[0]),
                "n_sites_used": int(np.sum(variant["components"]["labels"] >= 0)),
                "n_components": int(len(variant["components"]["components"])),
                "band_ratio": float(s["band_ratio"]),
                "band_p": float(s["band_p"]),
                "band_ratio_z": float(s["band_ratio_z"]),
                "peak_nm": float(s["peak_nm"]),
                "positive_excess_centroid_nm": float(
                    s["positive_excess_centroid_nm"]),
            })

    trace_counts = np.asarray(sites["n_traces"], dtype=int)
    repeated = trace_counts > 1
    time_span = np.asarray(sites["t_end"]) - np.asarray(sites["t_start"])
    sem = np.asarray(traces["sem_nm"], dtype=float)
    centroid_sem = np.nanmedian(sem, axis=0) if sem.size else np.full(3, np.nan)
    rod_components = sum(bool(c["rod_like"]) for c in components["components"])
    finite_sensitivity = [
        row for row in sensitivity
        if np.isfinite(row["band_ratio"]) and np.isfinite(row["band_p"])
    ]
    sensitivity_passes = sum(
        row["band_ratio"] > 1.0 and row["band_p"] <= 0.05
        for row in finite_sensitivity
    )
    robust_excess = (bool(finite_sensitivity)
                     and sensitivity_passes == len(finite_sensitivity))
    # Calibrated companion to the flag above.  ``band_p`` cannot fall below
    # 1/(replicates + 1) and was measured to call roughly 15% of draws from the
    # method's own null significant at a nominal 5%, so the calibrated variant
    # asks the ratio to clear the null ratio distribution by a stated margin
    # instead.  Both are reported; they answer the same question at different
    # strictness, and only the calibrated one is safe to quote as evidence.
    calibrated_rows = [row for row in finite_sensitivity
                       if np.isfinite(row.get("band_ratio_z", np.nan))]
    calibrated_passes = sum(
        row["band_ratio"] > 1.0 and row["band_ratio_z"] >= cfg.calibrated_ratio_z
        for row in calibrated_rows
    )
    robust_calibrated = (bool(calibrated_rows)
                         and calibrated_passes == len(calibrated_rows))

    centroid_spread = np.asarray(
        [row["positive_excess_centroid_nm"] for row in finite_sensitivity],
        dtype=float)
    centroid_spread = centroid_spread[np.isfinite(centroid_spread)]
    ratio_spread = np.asarray([row["band_ratio"] for row in finite_sensitivity],
                              dtype=float)
    ratio_spread = ratio_spread[np.isfinite(ratio_spread)]

    stratum_profile = (_stratum_profile(base, cfg)
                       if cfg.run_stratum_profile else None)

    return {
        "schema": "hlyb_staged_short_range_3d/v1",
        "config": cfg,
        "n_localizations": int(loc.shape[0]),
        "n_traces_total": int(traces["n_traces_total"]),
        "n_traces_used": int(traces["centroids_nm"].shape[0]),
        "n_sites": int(sites["centers_nm"].shape[0]),
        "n_sites_used": int(np.sum(components["labels"] >= 0)),
        "n_components": int(len(components["components"])),
        "n_components_all": int(components["n_all_components"]),
        "n_rod_like_components": int(rod_components),
        "n_excluded_sites": int(components["n_excluded_sites"]),
        "n_repeated_sites": int(np.sum(repeated)),
        "n_traces_consolidated": int(np.sum(np.clip(trace_counts - 1, 0, None))),
        "median_within_site_rms_nm": (float(np.median(sites["within_site_rms_nm"][repeated]))
                                      if repeated.any() else float("nan")),
        "median_repeat_time_span_s": (float(np.nanmedian(time_span[repeated]))
                                      if repeated.any() and np.isfinite(time_span[repeated]).any()
                                      else float("nan")),
        "centroid_sem_nm": centroid_sem,
        "points_nm": traces.get("points_nm", np.empty((0, 3))),
        "trace_centroids_nm": traces["centroids_nm"],
        "site_centers_nm": sites["centers_nm"],
        "site_sem_nm": sites["sem_nm"],
        "trace_to_site": sites["trace_to_site"],
        "component_labels": components["labels"],
        "components": components["components"],
        "edges_nm": base["edges_nm"],
        "centers_nm": 0.5 * (base["edges_nm"][:-1] + base["edges_nm"][1:]),
        "observed": base["observed"],
        "observed_by_component": base["observed_by_component"],
        "null_mean": base["null"]["mean"],
        "null_sd": base["null"]["sd"],
        "null_lo": base["null"]["lo"],
        "null_hi": base["null"]["hi"],
        "null_profiles": base["null"]["profiles"],
        "null_preview_sites_nm": base["null"]["preview_sites_nm"],
        "summary": summary,
        "bootstrap": bootstrap,
        "sensitivity": sensitivity,
        "stratum_profile": stratum_profile,
        "robust_short_range_excess": (robust_excess if cfg.run_sensitivity else None),
        "robust_short_range_excess_calibrated": (
            robust_calibrated if cfg.run_sensitivity else None),
        "sensitivity_passes": int(sensitivity_passes),
        "sensitivity_calibrated_passes": int(calibrated_passes),
        "sensitivity_valid_variants": int(len(finite_sensitivity)),
        # Preferred uncertainty on the excess location: the spread actually
        # produced by the audited parameter choices.  Wider and more honest
        # than the component bootstrap at these component counts.
        "centroid_sensitivity_range_nm": (
            [float(centroid_spread.min()), float(centroid_spread.max())]
            if centroid_spread.size else [float("nan"), float("nan")]),
        "band_ratio_sensitivity_range": (
            [float(ratio_spread.min()), float(ratio_spread.max())]
            if ratio_spread.size else [float("nan"), float("nan")]),
        "limitations": [
            "Inferred sites are label-site estimates, not identified HlyB protomers.",
            "Time spans are retained across all visits, but this implementation does not yet fit a full DDC/BaGoL temporal emitter model.",
            "The conditional null relies on coarse spatial components and a local rod axis.",
            "A short-range excess is not by itself a molecular dimer-distance estimate.",
            "Biological replication must be assessed across acquisitions, not pair counts.",
            "The randomization p-value is bounded below by 1/(replicates + 1) and is anti-conservative; quote the band ratio and its calibrated z instead.",
            "The band ratio is conditional on the null stratification scale and must be quoted together with it; the excess location is the stable descriptor.",
        ],
    }


def config_with(cfg: Staged3DConfig, **changes) -> Staged3DConfig:
    """Small public helper used by batch/sensitivity scripts."""
    return replace(cfg, **changes)
