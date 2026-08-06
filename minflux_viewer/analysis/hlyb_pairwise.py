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
observed pair-distance distribution.  Three components:

``N_rep · K_rep(r)``
    Same molecule re-acquired as several traces.  ``K_rep`` is calibrated
    **empirically and non-circularly** from consecutive-in-time trace pairs
    (see :func:`calibrate_repeat_kernel`): selection is on the time gap alone,
    never on distance, so the resulting distance distribution is an unbiased
    sample of same-emitter separations.

``N_str · S(r)``
    Distinct labelled sites of one assembly.  ``S`` is the *observed* distance
    distribution implied by a distribution of true inter-site distances,
    blurred by the positional error of a pair of centres.

``A · null(r)``
    Pairs from different assemblies.  Its shape is taken from an
    envelope-preserving surrogate (:func:`envelope_null`) rather than assumed.

The structural term is deliberately not tied to one architecture
-----------------------------------------------------------------
The published HlyB diagram describes three dimers in a C3-symmetric trimer, but
that is a *reference architecture*, not a certainty for a given preparation: the
trimer may not survive sample handling, leaving dimers whose inter-subunit
distance can differ from the tabulated one and need not be sharp — a flexible
linkage produces a broad, even flat, band of distances.  Fixing the five-class
trimer geometry would impose exactly the answer the experiment is meant to test.

:data:`STRUCTURE_MODELS` therefore holds a family of candidate structural terms,
and the default analysis is **dimer-centred**:

``dimer_gaussian``
    One inter-subunit distance with a fitted centre and a fitted conformational
    spread.  The primary measurement: *what is the dimer distance?*
``dimer_uniform``
    The distance lies anywhere in a fitted band, with no preferred value — the
    fully elastic case, which a flat region of the histogram should select.
``dimer_lognormal``
    A skewed distance distribution, the natural shape for a tether that
    stretches more easily than it compresses.
``trimer_six_site``
    The published C3 geometry, retained as **one competing hypothesis among
    several** rather than as the model.

Every shape supplies a distribution ``p(d)`` over true distances; the observed
profile is ``∫ p(d)·P(r | d, σ) dd`` with ``P`` the exact 3-D blurred-distance
density.  The shapes are ranked by AIC, so the data — not the diagram — decides
whether the distance is sharp, broad, or trimeric.

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

#: The tabulated dimer distance of the reference diagram, as an inter-domain
#: (protein) distance.  Used only as a STARTING VALUE for the fitted dimer
#: distance and as a reference to quote the fitted value against — never as a
#: constraint.
HLYB_REFERENCE_DIMER_NM = HLYB_CLASS_DISTANCES_NM[1]


@dataclass(frozen=True)
class StructureModel:
    """A candidate distribution of true inter-site distances.

    ``pdf(d_grid, params)`` returns the (unnormalised) density over true
    distances; the caller convolves it with the positional blur.  Keeping the
    true-distance distribution separate from the blur is what allows a fitted
    width to be reported as conformational spread rather than as instrument
    response.
    """

    key: str
    label: str
    param_names: tuple
    pdf: object
    start: object          # (cfg, centres) -> list[float]
    bounds: object         # (cfg, centres) -> list[tuple]
    describe: object       # (params) -> str


def _delta_pdf(d_grid: np.ndarray, centre: float) -> np.ndarray:
    """Unit mass at ``centre``, split linearly between its two neighbours.

    Snapping to the nearest grid point quantises a rigid distance by up to half
    a grid step.  That is harmless when the blur is several times the step, but
    on sharp data it displaces a modelled peak by a sixth of the blur and costs
    a fixed-geometry hypothesis a great deal of likelihood against a flexible
    one — an artificial handicap that has nothing to do with the structure.
    Linear splitting preserves the exact mean distance for any grid.
    """
    out = np.zeros_like(d_grid)
    n = d_grid.size
    if n == 0:
        return out
    value = float(centre)
    if n == 1 or value <= d_grid[0]:
        out[0] = 1.0
        return out
    if value >= d_grid[-1]:
        out[-1] = 1.0
        return out
    upper = int(np.searchsorted(d_grid, value))
    lower = upper - 1
    span = d_grid[upper] - d_grid[lower]
    if span <= 0:
        out[lower] = 1.0
        return out
    weight = (value - d_grid[lower]) / span
    out[lower] = 1.0 - weight
    out[upper] = weight
    return out


def _dimer_gaussian_pdf(d_grid, params):
    centre, spread = float(params[0]), float(params[1])
    if spread <= 1e-6:
        return _delta_pdf(d_grid, centre)
    return np.exp(-0.5 * ((d_grid - centre) / spread) ** 2)


#: Edge softness of the uniform band, in nm.  A hard top-hat has zero gradient
#: with respect to its edges almost everywhere -- moving an edge by a
#: finite-difference step does not change which grid points are inside -- so the
#: optimiser cannot move it and the band stays at its starting values.  A soft
#: edge of well under one bin makes the shape differentiable while remaining a
#: flat band for every practical purpose.
_BAND_EDGE_NM = 0.35


def _dimer_uniform_pdf(d_grid, params):
    lo, hi = float(params[0]), float(params[1])
    if hi < lo:
        lo, hi = hi, lo
    if hi - lo < 1e-6:
        return _delta_pdf(d_grid, 0.5 * (lo + hi))
    from scipy.special import expit

    return expit((d_grid - lo) / _BAND_EDGE_NM) * expit((hi - d_grid) / _BAND_EDGE_NM)


def _dimer_lognormal_pdf(d_grid, params):
    median, shape = float(params[0]), max(float(params[1]), 1e-3)
    safe = np.clip(d_grid, 1e-6, None)
    out = np.exp(-0.5 * ((np.log(safe) - np.log(max(median, 1e-6))) / shape) ** 2) / safe
    return np.where(d_grid > 0, out, 0.0)


def _trimer_pdf(d_grid, params):
    """Five equally weighted classes, shifted by a common label offset."""
    offset = float(params[0])
    out = np.zeros_like(d_grid)
    for dist in HLYB_CLASS_DISTANCES_NM:
        out += _delta_pdf(d_grid, dist + offset)
    return out


def _span(centres) -> float:
    arr = np.asarray(centres, dtype=float)
    return float(arr.max()) if arr.size else 60.0


STRUCTURE_MODELS: dict[str, StructureModel] = {
    "dimer_gaussian": StructureModel(
        key="dimer_gaussian",
        label="single dimer distance (Gaussian spread)",
        param_names=("distance_nm", "spread_nm"),
        pdf=_dimer_gaussian_pdf,
        start=lambda cfg, c: [float(cfg.dimer_start_nm), 1.5],
        bounds=lambda cfg, c: [tuple(cfg.dimer_distance_bounds_nm), (0.0, 12.0)],
        describe=lambda p: (f"distance {p[0]:.2f} nm, conformational spread "
                            f"{p[1]:.2f} nm"),
    ),
    "dimer_uniform": StructureModel(
        key="dimer_uniform",
        label="dimer distance uniform in a band (fully elastic)",
        param_names=("lo_nm", "hi_nm"),
        pdf=_dimer_uniform_pdf,
        start=lambda cfg, c: [max(cfg.dimer_distance_bounds_nm[0], 6.0), 16.0],
        bounds=lambda cfg, c: [tuple(cfg.dimer_distance_bounds_nm),
                               tuple(cfg.dimer_distance_bounds_nm)],
        describe=lambda p: (f"band {min(p[0], p[1]):.2f} to {max(p[0], p[1]):.2f} nm "
                            f"(width {abs(p[1] - p[0]):.2f} nm)"),
    ),
    "dimer_lognormal": StructureModel(
        key="dimer_lognormal",
        label="single dimer distance (log-normal spread)",
        param_names=("median_nm", "log_sigma"),
        pdf=_dimer_lognormal_pdf,
        start=lambda cfg, c: [float(cfg.dimer_start_nm), 0.2],
        bounds=lambda cfg, c: [tuple(cfg.dimer_distance_bounds_nm), (0.02, 1.2)],
        describe=lambda p: (f"median {p[0]:.2f} nm, log-sigma {p[1]:.3f}"),
    ),
    "trimer_six_site": StructureModel(
        key="trimer_six_site",
        label="published six-site C3 trimer",
        param_names=("label_offset_nm",),
        pdf=_trimer_pdf,
        start=lambda cfg, c: [float(np.clip(cfg.label_offset_nm,
                                            *cfg.label_offset_bounds_nm))],
        bounds=lambda cfg, c: [tuple(cfg.label_offset_bounds_nm)],
        describe=lambda p: f"label offset {p[0]:.2f} nm on the tabulated classes",
    ),
}

#: Default ranking.  Dimer shapes first: the trimer is a hypothesis here, not a
#: constraint, because it may not survive sample preparation.
DEFAULT_HYPOTHESES = ("dimer_gaussian", "dimer_uniform", "dimer_lognormal",
                      "trimer_six_site", "no_structure")


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
    label_offset_nm: float = 2.0        # starting value for the trimer's δ
    label_offset_bounds_nm: tuple = (0.0, 6.0)
    fit_label_offset: bool = True
    # Dimer-centred defaults.  The distance is FITTED over a wide range: the
    # tabulated value is a reference architecture, and the real separation in a
    # given preparation can differ.
    dimer_start_nm: float = 12.0
    dimer_distance_bounds_nm: tuple = (4.0, 40.0)
    # ---- 2-D variant -----------------------------------------------------
    #: 2 projects onto XY and models the foreshortening; 3 uses the full
    #: three-dimensional separation.
    dimensions: int = 3
    #: Per-E.coli delineation and inward shrink, shared with the 2-D clustering
    #: workflow (analysis/hlyb_clustering.py::compute_cell_mask).  In relative
    #: mode the retained fraction of each cell bounds the membrane tilt, which
    #: is what makes the projected distances interpretable.
    border_mode: str = "relative"
    border_fraction: float = 0.35
    border_size_nm: float = 200.0
    mask_pixel_size_nm: float = 20.0
    mask_smooth_nm: float = 60.0
    mask_close_nm: float = 120.0
    min_cell_area_nm2: float = 50_000.0
    #: Azimuth samples per retained tilt when building the projection kernel.
    projection_azimuth_samples: int = 48
    #: Physical floor on a TRUE inter-site distance.  Two labelled N-terminal
    #: domains, each carrying an antibody, cannot occupy the same place, so
    #: p(d) is zero below this.
    #:
    #: Without the floor a broad shape leaks its tail through zero and takes
    #: over the sub-5 nm region, which belongs to the same-site population:
    #: on the reference dataset the structural term claimed 6944 pairs against
    #: 548 for the same-site term, although 1828 centroid pairs lie below 6 nm
    #: and the calibrated kernel puts most of its mass there.  The leak drags
    #: the reported median down and inflates the apparent width.  Bounding the
    #: distribution *centre* does not help -- the spread leaks regardless — so
    #: the truncation is applied to p(d) itself.
    structure_min_nm: float = 5.0
    #: Grid over true inter-site distances used to convolve p(d) with the blur.
    distance_grid_nm: float = 0.25
    # σ is the positional blur of a pair distance.  It is a property of the
    # MEASUREMENT and is therefore **not fitted** by default: it is computed
    # from the measured centroid precision combined in quadrature with the
    # labelling allowance below.
    #
    # Fitting it was actively harmful.  A free σ absorbs the very width the
    # analysis is meant to measure, and because σ is shared across hypotheses
    # the shape that best absorbs structure into blur then dictates a σ that
    # cripples the others: on trimer ground-truth data the trimer hypothesis
    # lost to a flat band by ~2100 AIC units purely through an inherited σ.
    # Deriving σ from the measurement removes both problems and makes the
    # fitted width of p(d) interpretable as a property of the sample.
    extra_sigma_nm: float = 3.0         # starting value if σ is fitted anyway
    sigma_headroom_nm: float = 3.5
    sigma_floor_nm: float = 2.0         # used when the data cannot supply one
    #: Per-pair spread contributed by the two antibody displacements.  Two
    #: independent ~2 nm offsets in arbitrary directions perturb a separation by
    #: roughly this much along the pair axis.
    label_spread_nm: float = 2.3
    fit_extra_sigma: bool = False
    fit_r_min_nm: float = 1.0
    fit_r_max_nm: float = 45.0

    #: Structural shapes compared against the data, ranked by AIC.
    hypotheses: tuple = field(default=DEFAULT_HYPOTHESES)


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


def offset_gaussian_pdf_2d(r: np.ndarray, d: float, sigma: float) -> np.ndarray:
    """Distance distribution for a true separation ``d`` blurred in the PLANE.

    The two-dimensional counterpart of :func:`offset_gaussian_pdf`: a Rice
    density, written through the exponentially scaled Bessel function so it
    stays finite when ``r·d/σ²`` is large.
    """
    from scipy.special import i0e

    r = np.asarray(r, dtype=float)
    s = max(float(sigma), 1e-9)
    d = float(d)
    z = r * d / s ** 2
    return (r / s ** 2) * np.exp(-((r - d) ** 2) / (2 * s ** 2)) * i0e(z)


def tilt_projection_factors(tilt_deg, n_azimuth: int = 48,
                            n_bins: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Distribution of the projected/true length ratio, as (values, weights).

    A pair on a membrane whose tangent plane is tilted by ``θ`` from the image
    plane has, at in-plane azimuth ``φ``, an out-of-plane component
    ``sin θ · sin φ``, so its projected length is
    ``d·√(1 − sin²θ·sin²φ)``.  Marginalising over azimuth for each observed tilt
    gives the ratio distribution.

    This is what turns the border shrink from a filter into a model: the
    retained localizations' own depth within their cell fixes ``θ``, so the
    foreshortening is measured from the delineation rather than assumed away.

    Returned **binned**.  One entry per (localization, azimuth) would be tens of
    thousands of values, and the blurred-distance matrix costs one pass per
    value; a few dozen weighted bins carry the same distribution and keep the
    build to well under a second.
    """
    tilts = np.atleast_1d(np.asarray(tilt_deg, dtype=float))
    tilts = tilts[np.isfinite(tilts)]
    if tilts.size == 0:
        return np.ones(1), np.ones(1)
    n_az = int(max(n_azimuth, 1))
    phi = (np.arange(n_az) + 0.5) * (2.0 * np.pi / n_az)
    sin_t = np.sin(np.deg2rad(np.clip(tilts, 0.0, 90.0)))[:, None]
    factors = np.sqrt(np.clip(1.0 - (sin_t * np.sin(phi)[None, :]) ** 2, 0.0, 1.0))
    factors = factors.ravel()
    bins = int(max(n_bins, 1))
    lo = float(min(factors.min(), 1.0))
    if 1.0 - lo < 1e-6:
        return np.ones(1), np.ones(1)
    edges = np.linspace(lo, 1.0, bins + 1)
    weights, _ = np.histogram(factors, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    keep = weights > 0
    w = weights[keep].astype(float)
    return centres[keep], w / w.sum()


def blurred_distance_matrix(
    centres_nm: np.ndarray,
    d_grid: np.ndarray,
    sigma_nm: float,
    *,
    dimensions: int = 3,
    factors: np.ndarray | None = None,
) -> np.ndarray:
    """``K[i, j]`` = observed density at ``centres[i]`` for true distance ``d[j]``.

    Precomputing this once is what keeps the fit fast: the blur is derived from
    the measurement rather than fitted, so ``K`` does not change between
    likelihood evaluations and each one costs a single matrix-vector product.
    In two dimensions the columns are additionally averaged over the projection
    factors, so ``K`` carries the foreshortening.
    """
    centres = np.asarray(centres_nm, dtype=float)
    grid = np.asarray(d_grid, dtype=float)
    out = np.empty((centres.size, grid.size), dtype=float)
    if int(dimensions) == 2:
        if factors is None:
            f, w = np.ones(1), np.ones(1)
        else:
            f, w = factors
            f = np.asarray(f, dtype=float)
            w = np.asarray(w, dtype=float)
        good = (f > 0) & (w > 0)
        f, w = (f[good], w[good]) if np.any(good) else (np.ones(1), np.ones(1))
        w = w / w.sum()
        for j, d in enumerate(grid):
            acc = np.zeros_like(centres)
            for factor, weight in zip(f, w):
                acc += weight * offset_gaussian_pdf_2d(centres, d * factor, sigma_nm)
            out[:, j] = acc
    else:
        for j, d in enumerate(grid):
            out[:, j] = offset_gaussian_pdf(centres, float(d), sigma_nm)
    return out


def summarise_tilts(edge_distance_nm, half_width_nm) -> np.ndarray:
    """Local membrane tilt (degrees) implied by each point's depth in its cell.

    For a rod of projected half-width ``R`` viewed end-on, a surface point whose
    distance to the projected boundary is ``e`` sits at in-plane offset
    ``ρ = R − e`` from the axis, and its tangent plane is tilted by
    ``arcsin(ρ/R)``.  Points at the projected centre are face-on; points at the
    rim are edge-on, which is exactly the population the border shrink removes.
    """
    edge = np.asarray(edge_distance_nm, dtype=float)
    width = np.asarray(half_width_nm, dtype=float)
    good = np.isfinite(edge) & np.isfinite(width) & (width > 0)
    if not np.any(good):
        return np.zeros(0)
    ratio = np.clip(1.0 - edge[good] / width[good], 0.0, 1.0)
    return np.degrees(np.arcsin(ratio))


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


def distance_grid(cfg_or_step, r_max_nm: float = 60.0,
                  min_nm: float | None = None) -> np.ndarray:
    """Grid of candidate TRUE inter-site distances.

    The grid starts at the physical floor (:attr:`PairFitConfig.structure_min_nm`),
    so every shape is truncated there and none can leak its tail into the
    sub-floor region that belongs to the same-site population.
    """
    step = float(getattr(cfg_or_step, "distance_grid_nm", cfg_or_step))
    step = step if step > 0 else 0.25
    if min_nm is None:
        min_nm = float(getattr(cfg_or_step, "structure_min_nm", step))
    start = max(float(min_nm), step)
    return np.arange(start, float(r_max_nm) + step, step)


def structure_profile(
    centres_nm: np.ndarray,
    model_key: str,
    params,
    *,
    sigma_nm: float,
    bin_nm: float = 0.5,
    d_grid: np.ndarray | None = None,
    kernel: np.ndarray | None = None,
) -> np.ndarray:
    """Observed pair-distance profile for a distribution of true distances.

    ``S(r) = ∫ p(d) · P(r | d, σ) dd`` where ``p`` comes from the named
    structure model and ``P`` is the blurred-distance density — three- or
    two-dimensional, the latter also carrying the projection foreshortening
    when ``kernel`` was built by :func:`blurred_distance_matrix`.  The result is
    normalised to unit area, so its amplitude is carried by the fit and the
    shape alone distinguishes the hypotheses.
    """
    centres = np.asarray(centres_nm, dtype=float)
    model = STRUCTURE_MODELS[model_key]
    grid = distance_grid(0.25, float(centres[-1]) if centres.size else 60.0,
                         min_nm=0.25) \
        if d_grid is None else np.asarray(d_grid, dtype=float)
    weights = np.asarray(model.pdf(grid, params), dtype=float)
    weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    if total <= 0:
        return np.zeros_like(centres)
    weights = weights / total
    if kernel is not None:
        out = np.asarray(kernel, dtype=float) @ weights
    else:
        keep = weights > 1e-9
        if not keep.any():
            return np.zeros_like(centres)
        out = np.zeros_like(centres)
        for d, w in zip(grid[keep], weights[keep]):
            out += w * offset_gaussian_pdf(centres, float(d), sigma_nm)
    area = out.sum() * bin_nm
    return out / area if area > 0 else out


def structure_distance_summary(model_key: str, params, d_grid: np.ndarray) -> dict:
    """Summarise the fitted distribution of TRUE inter-site distances.

    Raw shape parameters are not comparable between shapes, and for a broad
    Gaussian truncated at zero they are not even interpretable — a centre of
    7 nm with a spread of 9 nm is not "a 7 nm distance".  Reporting the
    percentiles of ``p(d)`` instead gives one honest description that applies to
    every shape and shows immediately whether the distance is localized or the
    population is simply broad.
    """
    grid = np.asarray(d_grid, dtype=float)
    weights = np.clip(np.asarray(STRUCTURE_MODELS[model_key].pdf(grid, params),
                                 dtype=float), 0.0, None)
    total = weights.sum()
    if total <= 0 or grid.size == 0:
        nan = float("nan")
        return {"median_nm": nan, "mean_nm": nan, "mode_nm": nan,
                "p16_nm": nan, "p84_nm": nan, "iqr_nm": nan, "spread_nm": nan}
    p = weights / total
    cdf = np.cumsum(p)

    def q(fraction):
        return float(grid[int(np.searchsorted(cdf, fraction, side="left"))
                          if np.searchsorted(cdf, fraction, side="left") < grid.size
                          else grid.size - 1])

    p16, p84 = q(0.16), q(0.84)
    return {
        "median_nm": q(0.5),
        "mean_nm": float(np.sum(p * grid)),
        "mode_nm": float(grid[int(np.argmax(p))]),
        "p16_nm": p16,
        "p84_nm": p84,
        # half the central 68 % width: the shape-independent "spread"
        "spread_nm": float(0.5 * (p84 - p16)),
        "iqr_nm": float(q(0.75) - q(0.25)),
    }


def complex_profile(
    centres_nm: np.ndarray,
    *,
    label_offset_nm: float,
    sigma_nm: float,
    distances_nm=HLYB_CLASS_DISTANCES_NM,
    weights=None,
    bin_nm: float = 0.5,
) -> np.ndarray:
    """Normalised profile for a fixed set of equally weighted distances.

    Retained for the trimer hypothesis and for callers that want an explicit
    distance list; the general path is :func:`structure_profile`.
    """
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
    structure: str = "dimer_gaussian",
    sigma_floor_nm: float | None = None,
    fixed_sigma_nm: float | None = None,
    dimensions: int = 3,
    projection_factors: np.ndarray | None = None,
) -> dict:
    """Maximum-likelihood fit of the three-component model to the profile.

    ``structure`` names an entry of :data:`STRUCTURE_MODELS`, or
    ``"no_structure"`` to drop the structural term entirely.  Free parameters:
    the two amplitudes, the background scale, the blur σ, the structure model's
    own shape parameters, and — when enabled — the repeat-kernel stretch.

    σ describes the measurement, so it is floored by the measured centroid
    precision and can be pinned with ``fixed_sigma_nm``; holding it common
    across hypotheses stops a broad shape from winning by absorbing blur that a
    narrow one is denied.
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
    fit_sigma = cfg.fit_extra_sigma and fixed_sigma_nm is None
    if fixed_sigma_nm is not None:
        s_start = float(np.clip(fixed_sigma_nm, s_floor, s_ceiling))
    elif not cfg.fit_extra_sigma:
        # measured centroid precision ⊕ labelling allowance
        s_start = float(np.hypot(s_floor, float(cfg.label_spread_nm)))
    else:
        s_start = float(np.clip(cfg.extra_sigma_nm, s_floor, s_ceiling))

    r_lo, r_hi = (float(cfg.repeat_scale_bounds[0]), float(cfg.repeat_scale_bounds[1]))
    fit_scale = cfg.fit_repeat_scale and r_hi > r_lo

    has_structure = structure != "no_structure"
    model_spec = STRUCTURE_MODELS[structure] if has_structure else None
    shape_x0 = list(model_spec.start(cfg, centres)) if has_structure else []
    shape_bounds = list(model_spec.bounds(cfg, centres)) if has_structure else []
    n_shape = len(shape_x0)
    grid = distance_grid(cfg, float(centres[-1]) if centres.size else cfg.r_max_nm)
    # The blur does not change between likelihood evaluations, so the whole
    # true-distance -> observed-distance mapping is built once and each
    # evaluation is a single matrix-vector product.  In two dimensions this
    # matrix also carries the projection foreshortening.
    kernel = (blurred_distance_matrix(centres, grid, max(s_start, 0.2),
                                      dimensions=dimensions,
                                      factors=projection_factors)
              if (has_structure and not fit_sigma) else None)

    total = max(obs.sum(), 1.0)

    def expected(params):
        n_rep, n_str, a_bkg, sig, scale, shape = params
        model = (max(n_rep, 0.0) * stretched_repeat(scale) * bin_nm
                 + max(a_bkg, 0.0) * bkg)
        if has_structure:
            use = kernel
            if use is None:
                use = blurred_distance_matrix(centres, grid, max(sig, 0.2),
                                              dimensions=dimensions,
                                              factors=projection_factors)
            prof = structure_profile(centres, structure, shape,
                                     sigma_nm=max(sig, 0.2), bin_nm=bin_nm,
                                     d_grid=grid, kernel=use)
            model = model + max(n_str, 0.0) * prof * bin_nm
        return model

    # The two amplitudes are counts (order 1e3-1e4) while the blur, the kernel
    # stretch and the shape parameters are order 1.  L-BFGS-B takes a single
    # absolute finite-difference step, so on the raw scale the amplitude
    # derivatives are evaluated at a relative step of ~1e-12 and the optimiser
    # stalls with the amplitudes still at their starting values.  Fitting them
    # as FRACTIONS of the observed total puts every parameter at order 1 and
    # converges (measured: 193 log-likelihood units better on the reference
    # dataset).
    def unpack(free):
        q = list(free)
        sig = q[3] if fit_sigma else s_start
        scale = q[4] if fit_scale else r_lo
        shape = q[5:5 + n_shape]
        return [q[0] * total, q[1] * total, q[2], sig, scale, shape]

    def nll(free):
        params = unpack(free)
        if params[0] < 0 or params[1] < 0 or params[2] < 0:
            return 1e18
        return _poisson_nll(obs[window], expected(params)[window])

    x0 = [0.25, 0.25 if has_structure else 0.0, 1.0, s_start,
          float(np.clip(1.3, r_lo, r_hi))] + shape_x0
    bounds = [
        (0.0, 10.0), (0.0, 10.0), (0.0, 100.0),
        (s_floor, s_ceiling) if fit_sigma else (s_start, s_start),
        (r_lo, r_hi) if fit_scale else (x0[4], x0[4]),
    ] + shape_bounds
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    p = unpack(res.x)
    sig, scale, shape = p[3], p[4], list(p[5])
    model = expected(p)

    k = 3 + int(fit_sigma) + int(fit_scale) + n_shape
    if not has_structure:
        k -= 1
    nll_value = _poisson_nll(obs[window], model[window])

    tol = 1e-3
    at_bounds = []
    if fit_sigma and (abs(sig - s_floor) < tol or abs(sig - s_ceiling) < tol):
        at_bounds.append("sigma_nm")
    if fit_scale and (abs(scale - r_lo) < tol or abs(scale - r_hi) < tol):
        at_bounds.append("repeat_scale")
    if has_structure:
        for name, value, (lo_b, hi_b) in zip(model_spec.param_names, shape, shape_bounds):
            if abs(value - lo_b) < tol or abs(value - hi_b) < tol:
                at_bounds.append(name)

    shape_params = dict(zip(model_spec.param_names, shape)) if has_structure else {}
    summary = (structure_distance_summary(structure, shape, grid) if has_structure
               else {})
    final_kernel = kernel if kernel is not None else (
        blurred_distance_matrix(centres, grid, max(sig, 0.2),
                                dimensions=dimensions, factors=projection_factors)
        if has_structure else None)
    structure_component = (
        max(p[1], 0.0) * structure_profile(centres, structure, shape,
                                           sigma_nm=max(sig, 0.2), bin_nm=bin_nm,
                                           d_grid=grid, kernel=final_kernel) * bin_nm
        if has_structure else np.zeros_like(centres)
    )
    return {
        "structure": structure,
        "structure_label": model_spec.label if has_structure else "no structural term",
        "structure_params": shape_params,
        "structure_description": (model_spec.describe(shape) if has_structure
                                  else "no structural term"),
        "distance_summary": summary,
        "n_repeat_pairs": float(p[0]),
        "n_structure_pairs": float(p[1]) if has_structure else 0.0,
        # legacy alias, so existing callers keep working
        "n_complex_pairs": float(p[1]) if has_structure else 0.0,
        "background_scale": float(p[2]),
        "sigma_nm": float(sig),
        "repeat_scale": float(scale),
        "model": model,
        "repeat_component": max(p[0], 0.0) * stretched_repeat(scale) * bin_nm,
        "background_component": max(p[2], 0.0) * bkg,
        "structure_component": structure_component,
        "complex_component": structure_component,
        "centres_nm": centres,
        "fit_window": window,
        "nll": nll_value,
        "n_parameters": int(k),
        "aic": float(2 * k + 2 * nll_value),
        "success": bool(res.success),
        "parameters_at_bounds": at_bounds,
    }


def profile_likelihood_distance(
    counts: np.ndarray,
    edges: np.ndarray,
    repeat_shape: np.ndarray,
    null_mean: np.ndarray,
    cfg: PairFitConfig,
    *,
    structure: str = "dimer_gaussian",
    sigma_floor_nm: float | None = None,
    fixed_sigma_nm: float | None = None,
    dimensions: int = 3,
    projection_factors: np.ndarray | None = None,
    n_points: int = 41,
) -> dict:
    """Scan the dimer distance, refitting everything else at each value.

    Gives a confidence interval on the distance rather than a bare point
    estimate: the 68 % and 95 % intervals are where the negative log-likelihood
    rises by 0.5 and 1.92 above its minimum.  A scan that stays flat means the
    data do not localize the distance, and saying so is the point.
    """
    import dataclasses

    spec = STRUCTURE_MODELS.get(structure)
    if spec is None or "distance_nm" not in spec.param_names:
        if spec is None or "median_nm" not in spec.param_names:
            return {"available": False}
    key = "distance_nm" if "distance_nm" in spec.param_names else "median_nm"
    lo, hi = (float(cfg.dimer_distance_bounds_nm[0]),
              float(cfg.dimer_distance_bounds_nm[1]))
    scan = np.linspace(lo, hi, int(max(n_points, 5)))
    nll_values = []
    for value in scan:
        pinned = dataclasses.replace(cfg, dimer_start_nm=float(value),
                                     dimer_distance_bounds_nm=(float(value), float(value)))
        fit = fit_pair_model(counts, edges, repeat_shape, null_mean, pinned,
                             structure=structure, sigma_floor_nm=sigma_floor_nm,
                             fixed_sigma_nm=fixed_sigma_nm, dimensions=dimensions,
                             projection_factors=projection_factors)
        nll_values.append(fit["nll"])
    nll_values = np.asarray(nll_values, dtype=float)
    best = int(np.argmin(nll_values))
    delta = nll_values - nll_values[best]

    def _interval(threshold):
        inside = np.flatnonzero(delta <= threshold)
        if inside.size == 0:
            return (float("nan"), float("nan"))
        return float(scan[inside.min()]), float(scan[inside.max()])

    step = float(scan[1] - scan[0]) if scan.size > 1 else float("nan")
    ci68 = _interval(0.5)
    return {
        "available": True,
        "parameter": key,
        "distance_nm": scan,
        "nll": nll_values,
        "delta_nll": delta,
        "best_nm": float(scan[best]),
        "ci68_nm": ci68,
        "ci95_nm": _interval(1.92),
        "step_nm": step,
        # an interval no wider than one scan step is unresolved, not tight
        "ci68_below_scan_step": bool(np.isfinite(step) and (ci68[1] - ci68[0]) <= step),
        # a scan that never rises by 1.92 does not constrain the distance
        "constrained": bool(delta.max() > 1.92),
    }


def compare_hypotheses(
    counts: np.ndarray,
    edges: np.ndarray,
    repeat_shape: np.ndarray,
    null_mean: np.ndarray,
    cfg: PairFitConfig,
    *,
    sigma_floor_nm: float | None = None,
    dimensions: int = 3,
    projection_factors: np.ndarray | None = None,
) -> dict:
    """Fit the competing structural hypotheses and rank them by AIC.

    σ describes the measurement, not the hypothesis, so it is fitted **once**
    on the full six-site model and then held fixed for the alternatives.
    Letting each hypothesis choose its own blur would let a one-distance model
    inflate into a featureless bump and win for the wrong reason.
    """
    names = [n for n in cfg.hypotheses
             if n == "no_structure" or n in STRUCTURE_MODELS]
    if not names:
        return {}
    # Every shape gets the same blur, derived from the measurement rather than
    # fitted (see PairFitConfig.fit_extra_sigma).  Fitting it on one "anchor"
    # shape and imposing the result on the rest was worse than either
    # alternative: the anchor absorbed structure into blur and then handed the
    # others a σ that made them fail on their own ground truth.
    out: dict[str, dict] = {}
    shared_sigma = None
    for name in names:
        fit = fit_pair_model(counts, edges, repeat_shape, null_mean, cfg,
                             structure=name, sigma_floor_nm=sigma_floor_nm,
                             fixed_sigma_nm=shared_sigma, dimensions=dimensions,
                             projection_factors=projection_factors)
        if shared_sigma is None:
            shared_sigma = float(fit["sigma_nm"])
        fit["shared_sigma_nm"] = float(shared_sigma)
        out[name] = fit
    best = min(out.values(), key=lambda f: f["aic"])["aic"]
    for fit in out.values():
        fit["delta_aic"] = float(fit["aic"] - best)
    return out


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------

def analyze_hlyb_pairwise_2d(
    loc_m: np.ndarray,
    tid: np.ndarray,
    tim: np.ndarray | None = None,
    cfg: PairFitConfig | None = None,
) -> dict:
    """Two-dimensional ensemble pair-distance analysis.

    Not simply "the 3-D analysis with z discarded".  Discarding z foreshortens
    every distance by an amount that depends on the pair's orientation, and for
    a membrane protein on a rod-shaped cell that orientation is a strong
    function of *where in the projected cell* the pair sits: face-on at the
    projected centre, edge-on at the rim.  Two things follow, and this function
    does both.

    First, each E.coli is delineated from the localization density and shrunk
    inward (:func:`~minflux_viewer.analysis.hlyb_clustering.compute_cell_mask`),
    dropping the rim where an in-plane distance is systematically too short.

    Second — and this is what makes the shrink a model rather than a filter —
    the retained localizations' own depth within their cell gives their local
    membrane tilt, and the projection kernel is built from that measured tilt
    distribution.  So the foreshortening that survives the shrink is corrected
    for rather than ignored.

    The 2-D projection still superimposes the upper and lower membrane; that is
    not corrected here and is reported as a limitation.
    """
    cfg = cfg or PairFitConfig()
    if int(cfg.dimensions) != 2:
        cfg = _replace_cfg(cfg, dimensions=2)
    return analyze_hlyb_pairwise(loc_m, tid, tim, cfg)


def _replace_cfg(cfg: PairFitConfig, **changes) -> PairFitConfig:
    import dataclasses

    return dataclasses.replace(cfg, **changes)


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

    two_d = int(cfg.dimensions) == 2
    cells = None
    tilts = np.zeros(0)
    projection = None
    if two_d and pts.shape[0]:
        from .hlyb_clustering import compute_cell_mask

        cells = compute_cell_mask(
            pts[:, :2],
            border_size_nm=cfg.border_size_nm,
            border_mode=cfg.border_mode,
            border_fraction=cfg.border_fraction,
            pixel_size_nm=cfg.mask_pixel_size_nm,
            smooth_nm=cfg.mask_smooth_nm,
            close_nm=cfg.mask_close_nm,
            min_cell_area_nm2=cfg.min_cell_area_nm2,
        )
        keep = ~cells.border_loc
        widths = np.asarray(cells.cell_half_width_nm, dtype=float)
        if widths.size:
            half = np.where(cells.cell_id > 0,
                            widths[np.clip(cells.cell_id - 1, 0, widths.size - 1)],
                            np.nan)
            tilts = summarise_tilts(cells.edge_distance_nm[keep], half[keep])
        else:
            # No cell could be delineated -- too few localizations, or the field
            # is not cell-shaped at all.  Then the membrane tilt is unmeasured,
            # so NO foreshortening correction is applied rather than one being
            # invented; distances are consequently biased short and the result
            # is flagged so the report can say so.
            tilts = np.zeros(0)
        projection = (tilt_projection_factors(tilts, cfg.projection_azimuth_samples)
                      if tilts.size else None)
        # keep the interior only, and drop z: the analysis is in the image plane
        pts = pts[keep]
        for key in ("sem_nm", "n_locs", "trace_ids"):
            if traces.get(key) is not None and len(traces[key]) == keep.size:
                traces[key] = np.asarray(traces[key])[keep]
        for key in ("t_start", "t_end"):
            if traces.get(key) is not None:
                traces[key] = np.asarray(traces[key])[keep]
        pts = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(pts.shape[0])])
        traces["centroids_nm"] = pts

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
    axes = slice(0, 2) if two_d else slice(0, 3)
    if np.isfinite(median_sem[axes]).any():
        per_axis = float(np.sqrt(np.nanmean(median_sem[axes] ** 2)))
    else:
        per_axis = 0.0
    sigma_floor = float(max(np.sqrt(2.0) * per_axis, 0.5))

    import dataclasses

    dims = 2 if two_d else 3
    fits = compare_hypotheses(counts, edges, repeat["shape"], null["mean"], cfg,
                              sigma_floor_nm=sigma_floor, dimensions=dims,
                              projection_factors=projection)
    best_name = min(fits, key=lambda k: fits[k]["aic"]) if fits else ""
    best = fits.get(best_name, {})

    # Sensitivity pass: repeat the entire comparison with the short-range
    # kernel allowed to broaden.  Releasing it improves the description of the
    # 5-9 nm region but weakens the structural discrimination, and the reader
    # is entitled to see by how much rather than being shown only the more
    # favourable of the two.
    relaxed_cfg = dataclasses.replace(cfg, fit_repeat_scale=not cfg.fit_repeat_scale)
    relaxed = compare_hypotheses(counts, edges, repeat["shape"], null["mean"],
                                 relaxed_cfg, sigma_floor_nm=sigma_floor,
                                 dimensions=dims, projection_factors=projection)
    relaxed_best = min(relaxed, key=lambda k: relaxed[k]["aic"]) if relaxed else ""

    # Confidence interval on the dimer distance, from a likelihood scan.  A
    # point estimate alone would not say whether the data localize the distance
    # at all -- which is exactly the question when the population may be broad.
    scan_structure = best_name if best_name in STRUCTURE_MODELS else "dimer_gaussian"
    scan = profile_likelihood_distance(
        counts, edges, repeat["shape"], null["mean"], cfg,
        structure=scan_structure, sigma_floor_nm=sigma_floor,
        fixed_sigma_nm=float(best.get("sigma_nm")) if best else None,
        dimensions=dims, projection_factors=projection)

    centres = 0.5 * (edges[:-1] + edges[1:])
    excess = counts.astype(float) - null["mean"]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(null["sd"] > 0, excess / null["sd"], np.nan)
    # Outer extent of the excess, taken as a CONTIGUOUS run outward from the
    # strongest bin rather than the largest radius at which any bin happens to
    # clear the threshold.  With a finite number of surrogate replicates the
    # null's spread is itself noisy, so isolated far bins cross it by chance;
    # reporting the maximum such radius claimed structure out to 32 nm on data
    # whose profile sits *below* the null over most of that range.
    finite = np.isfinite(z) & (centres <= cfg.fit_r_max_nm)
    outer = 0.0
    if finite.any() and np.any(finite & (z > 3.0)):
        idx = np.flatnonzero(finite)
        peak = idx[int(np.argmax(np.where(finite, z, -np.inf)[idx]))]
        misses = 0
        outer = float(centres[peak])
        for i in range(peak, int(idx.max()) + 1):
            if finite[i] and z[i] > 3.0:
                outer = float(centres[i])
                misses = 0
            else:
                misses += 1
                if misses >= 3:      # three consecutive bins back inside the null
                    break

    # Is the structural term supported by a real excess, or is it patching a
    # mismatch between a flat observation and a rising surrogate?  A large AIC
    # gap alone does not answer that: a broad component can earn likelihood by
    # absorbing a background misfit while the observed profile sits AT or BELOW
    # the null exactly where that component lives.  So the excess is integrated
    # over the fitted distribution's own central range and expressed in units of
    # the null's spread there.
    support_z = float("nan")
    support_range = (float("nan"), float("nan"))
    summary = (best or {}).get("distance_summary") or {}
    if summary and np.isfinite(summary.get("p16_nm", np.nan)):
        lo_r = float(summary["p16_nm"])
        hi_r = float(summary["p84_nm"])
        band = (centres >= lo_r) & (centres <= hi_r)
        if band.any():
            var = float(np.sum(np.asarray(null["sd"])[band] ** 2))
            support_z = (float(np.sum(excess[band]) / np.sqrt(var))
                         if var > 0 else float("nan"))
            support_range = (lo_r, hi_r)

    return {
        "structure_support_z": support_z,
        "structure_support_range_nm": support_range,
        # A structural distance is only claimed when the data genuinely exceed
        # the randomized reference across the range that distance occupies.
        "structure_detected": bool(np.isfinite(support_z) and support_z > 3.0),
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
        "distance_scan": scan,
        "reference_dimer_nm": float(HLYB_REFERENCE_DIMER_NM),
        "structure_labels": {k: v.label for k, v in STRUCTURE_MODELS.items()},
        "dimensions": dims,
        "is_2d": bool(two_d),
        "cell_mask": cells,
        "cell_mask_stats": (cells.stats if cells is not None else {}),
        "membrane_tilt_deg": tilts,
        "median_tilt_deg": (float(np.median(tilts)) if tilts.size else float("nan")),
        "median_foreshortening": (
            float(np.sum(projection[0] * projection[1]))
            if projection is not None and np.size(projection[0]) else float("nan")),
        # True when the 2-D variant could not delineate a cell, so no
        # foreshortening correction was applied and distances are biased short.
        "delineation_failed": bool(two_d and projection is None),
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
