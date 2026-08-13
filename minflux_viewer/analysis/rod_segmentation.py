"""Targeted detection of rod-shaped (capsule) cells in a projected XY view.

Built for E. coli, whose footprint is a *stadium* — a rectangle with rounded
short ends — of a well-constrained **width** (typically 800-1100 nm) and a much
less constrained length (1.5-5 um, and dividing cells are dumbbells).  That
asymmetry drives the method: the width prior is encoded directly through the
Euclidean **distance transform** of the cell mask, whose ridge value is the
local half-width.  This is rotation- and length-invariant by construction, so
no oriented-template bank over angle x length is needed.

Stages:

1. project the localizations into an XY density image and delineate the cells
   exactly as the 2-D E. coli pipeline already does -- reusing
   :func:`minflux_viewer.analysis.hlyb_clustering.compute_cell_mask` (smooth ->
   Otsu -> morphological close -> fill -> open -> area despeckle) rather than
   re-implementing it;
2. distance-transform the mask, in nm;
3. optionally **cut thin bridges**: threshold the distance transform near the
   expected half-width to seed one marker per cell body, then grow the markers
   back over the mask by nearest-seed (Voronoi) assignment, constrained so a
   marker can only claim pixels of its own connected component;
4. fit each region with the minimum-area **oriented** rectangle
   (:func:`minflux_viewer.core.min_enclosing.min_area_rectangle`) and accept it
   as a rod only if it passes the size and shape gates.

What step 3 does and does not do, precisely.  It separates cell bodies joined
by a **constriction** — cells whose caps nearly touch and are bridged by the
morphological closing, a dividing cell with a septum, a spurious filament of
scattered localizations.  That is the merge mode that would otherwise pass
every gate silently: two 3 um cells end to end read as one 6 um rod of exactly
the right width.  It cannot separate cells that **overlap in projection while
running parallel**, because their distance-transform ridges are connected along
their whole length and the projected density carries no evidence of two cells.
Those are caught instead by the width gate and by ``width_ratio``, and are
*rejected* rather than analysed — under-segmentation that survives into a pair
analysis biases it, so exclusion is the safe outcome.

The two width measures are kept separate on purpose.  ``width_nm`` is twice the
distance-transform ridge (the true membrane-to-membrane width) while
``box_width_nm`` is the oriented box's short side.  They agree for one clean
rod; for two cells merged **side by side** the box widens while the ridge does
not, so their ratio is the diagnostic that catches that specific failure.

Pure NumPy/SciPy plus the Qt-free geometry helper; no Qt, unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.min_enclosing import min_area_rectangle
from .hlyb_clustering import compute_cell_mask

# Guard against a large field of view combined with a fine pixel silently
# allocating a huge grid.  Raised as an error with the remedy, never worked
# around silently.
MAX_GRID_PIXELS = 40_000_000

REJECT_REASONS = (
    "too narrow",
    "too wide",
    "too short",
    "too long",
    "not elongated",
    "outline not rod-like",
    "width mismatch (merged or bent)",
)


@dataclass
class RodConfig:
    """Parameters of the rod/cell detector."""

    # --- density image -------------------------------------------------
    # ``smooth_nm`` is the bridging length: how far apart two labelled spots
    # may be and still end up in one cell body.  In MINFLUX data the right
    # value is set by the *labelling* sparsity, not by the optics — a membrane
    # protein at low density leaves gaps of tens of nm between distinct
    # positions, and a bridging length below those gaps shatters one cell into
    # fragments.  Negative means auto: derived from the measured spacing of
    # distinct positions by :func:`resolve_smoothing`.  ``close_nm`` defaults
    # to twice the bridging length.
    pixel_size_nm: float = 20.0
    smooth_nm: float = -1.0
    close_nm: float = -1.0
    open_nm: float = 0.0
    # Neighbour order used to estimate the local spacing of labelled positions,
    # and the bounds the resulting bridging length is held between.  The upper
    # bound is a fraction of the minimum cell width, because smoothing inflates
    # the measured width (see ``width_tolerance_nm``).
    smooth_neighbours: int = 8
    smooth_min_pixels: float = 2.0
    smooth_max_width_fraction: float = 6.0
    # 0 = derive from the size window; components below this area are dropped
    # as speckle before any shape is fitted.
    min_area_nm2: float = 0.0

    # --- target size window --------------------------------------------
    # These are the **structure's** width and length.  The measured
    # ``Rod.width_nm`` is the width of the smoothed, Otsu-thresholded density
    # mask, which necessarily runs wider than the structure it envelopes: the
    # projected density of a membrane-labelled rod peaks at its two edges, and
    # smoothing spreads those peaks outward before the threshold is applied.
    # Measured on synthetic membrane-labelled 1000 nm cells at the default
    # 60 nm smoothing, the overshoot is ~50 nm per side — about 0.8 smoothing
    # lengths each side, so ~1.7 of them across the full width.
    # ``width_tolerance_nm`` therefore widens the gate by **twice** the
    # smoothing length at each end, covering both sides with a little headroom,
    # so the window above can be stated as the real cell width rather than as a
    # mask width.
    min_width_nm: float = 800.0
    max_width_nm: float = 1100.0
    min_length_nm: float = 1000.0
    max_length_nm: float = 8000.0
    width_tolerance_nm: float = -1.0        # < 0 = auto: 2 x smooth_nm

    # --- splitting cells that touch ------------------------------------
    split_touching: bool = True
    # 0 = auto: ``seed_level_fraction`` x the minimum half-width.  A neck
    # between two touching cells carries a smaller distance-transform value
    # than either cell's ridge, so thresholding near the half-width yields one
    # seed per cell.
    seed_level_nm: float = 0.0
    seed_level_fraction: float = 0.85

    # --- shape gates ---------------------------------------------------
    # Region area / oriented-box area.  A perfect stadium of aspect 3 fills
    # 0.93 of its box; the gate mainly rejects branched or L-shaped blobs.
    min_fill_fraction: float = 0.55
    # box short side / distance-transform width.  1.0 for a clean straight rod,
    # ~2 for two cells merged side by side, and mildly above 1 for a bent cell.
    max_width_ratio: float = 1.6


@dataclass
class Rod:
    """One detected region, accepted or not, with the numbers it was judged on."""

    id: int
    center_nm: np.ndarray            # (2,) centre of the oriented box
    axis_nm: np.ndarray              # (2,) unit vector along the long axis
    angle_deg: float                 # axis direction, in (-90, 90]
    length_nm: float                 # oriented box long side
    width_nm: float                  # 2 x distance-transform ridge
    box_width_nm: float              # oriented box short side
    area_nm2: float
    fill_fraction: float
    width_ratio: float
    n_pixels: int
    endpoints_nm: np.ndarray         # (2, 2) capsule centreline ends
    accepted: bool
    reject_reason: str = ""
    # Points of the image the detection ran on that fell in this region.
    n_points: int = 0
    # Points of a *second* set assigned afterwards through :func:`assign_points`
    # — the inferred label sites, when the detection ran on the raw cloud.
    n_sites: int = 0

    def as_dict(self) -> dict:
        """Plain-Python view for logs, method text and saved payloads."""
        return {
            "id": int(self.id),
            "center_nm": [float(v) for v in self.center_nm],
            "axis_nm": [float(v) for v in self.axis_nm],
            "angle_deg": float(self.angle_deg),
            "length_nm": float(self.length_nm),
            "width_nm": float(self.width_nm),
            "box_width_nm": float(self.box_width_nm),
            "area_nm2": float(self.area_nm2),
            "fill_fraction": float(self.fill_fraction),
            "width_ratio": float(self.width_ratio),
            "n_pixels": int(self.n_pixels),
            "endpoints_nm": [[float(v) for v in p] for p in self.endpoints_nm],
            "accepted": bool(self.accepted),
            "reject_reason": str(self.reject_reason),
            "n_points": int(self.n_points),
            "n_sites": int(self.n_sites),
        }


@dataclass
class RodSegmentationResult:
    """Detection outcome plus everything needed to draw and audit it."""

    rods: list[Rod] = field(default_factory=list)
    labels: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.int64))
    mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), bool))
    edt_nm: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), float))
    # Per input point.  ``region_of_point`` is the 1-based region label (0 =
    # outside every region); ``component_of_point`` is the 0-based index into
    # :attr:`accepted` (-1 = not in an accepted rod).
    region_of_point: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    component_of_point: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    x_edges: np.ndarray = field(default_factory=lambda: np.empty(0))
    y_edges: np.ndarray = field(default_factory=lambda: np.empty(0))
    pixel_size_nm: float = 0.0
    stats: dict = field(default_factory=dict)

    @property
    def accepted(self) -> list[Rod]:
        return [rod for rod in self.rods if rod.accepted]

    @property
    def rejected(self) -> list[Rod]:
        return [rod for rod in self.rods if not rod.accepted]


def _validate(cfg: RodConfig) -> None:
    if cfg.pixel_size_nm <= 0:
        raise ValueError("pixel_size_nm must be positive")
    if cfg.min_width_nm <= 0 or cfg.max_width_nm < cfg.min_width_nm:
        raise ValueError("rod width window must be positive with max >= min")
    if cfg.min_length_nm <= 0 or cfg.max_length_nm < cfg.min_length_nm:
        raise ValueError("rod length window must be positive with max >= min")
    if not 0.0 <= cfg.min_fill_fraction <= 1.0:
        raise ValueError("min_fill_fraction must lie in [0, 1]")
    if cfg.max_width_ratio < 1.0:
        raise ValueError("max_width_ratio must be at least 1")
    if not 0.0 < cfg.seed_level_fraction <= 1.0:
        raise ValueError("seed_level_fraction must lie in (0, 1]")


def _axis_angle_deg(deg: float) -> float:
    """Reduce a box angle to an *axis* direction in ``(-90, 90]``."""
    a = float(deg) % 180.0
    return a - 180.0 if a > 90.0 else a


def label_spacing_nm(xy_nm, *, neighbours: int = 8) -> float:
    """Typical spacing of labelled positions, in nm.

    Estimated from the *k*-th nearest-neighbour distance: for a locally uniform
    process of areal density ``lam``, ``E[d_k] ~ sqrt(k / (pi * lam))``, so
    ``d_k * sqrt(pi / k)`` estimates the mean spacing ``1 / sqrt(lam)``.  The
    **k**-th neighbour rather than the first, because the first is dominated by
    clustering; the **median** over points, because that is insensitive to the
    margins of the field, where the neighbourhood runs into empty space.

    Feed this *distinct molecular positions* — trace centroids or inferred
    sites.  Raw localizations pile up tens deep at one molecule, which measures
    the pile-up rather than how completely the cell is covered.
    """
    from scipy.spatial import cKDTree

    xy = np.asarray(xy_nm, dtype=float)[:, :2]
    xy = xy[np.isfinite(xy).all(axis=1)]
    k = max(int(neighbours), 1)
    if xy.shape[0] <= k:
        return float("nan")
    distances = cKDTree(xy).query(xy, k=k + 1)[0][:, k]
    return float(np.median(distances) * np.sqrt(np.pi / k))


def resolve_smoothing(xy_nm, cfg: RodConfig,
                      spacing_points_nm=None) -> tuple[float, float]:
    """Bridging and closing lengths for this point cloud, in nm.

    Returns the configured values when they are set.  Otherwise the bridging
    length is the typical spacing of labelled positions
    (:func:`label_spacing_nm`), held between ``smooth_min_pixels`` pixels and a
    fraction of the minimum cell width, so it always does something and never
    smears the cell so far that the width measurement stops meaning anything.
    The closing length is twice the bridging length.

    *spacing_points_nm* supplies the distinct molecular positions when they are
    known — the caller usually has them, and they give the right answer.
    Without them the input is collapsed onto the detection grid as a fallback,
    which **underestimates** the spacing for data with heavy trace pile-up.
    """
    smooth = float(cfg.smooth_nm)
    if smooth < 0:
        ps = float(cfg.pixel_size_nm)
        lo = float(cfg.smooth_min_pixels) * ps
        hi = max(float(cfg.min_width_nm) / float(cfg.smooth_max_width_fraction), lo)
        if spacing_points_nm is None:
            xy = np.asarray(xy_nm, dtype=float)[:, :2]
            xy = xy[np.isfinite(xy).all(axis=1)]
            spacing_points_nm = np.unique(
                np.round(xy / ps).astype(np.int64), axis=0) * ps
        smooth = label_spacing_nm(spacing_points_nm,
                                  neighbours=cfg.smooth_neighbours)
        if not np.isfinite(smooth):
            smooth = lo
        smooth = float(min(max(smooth, lo), hi))
    close = float(cfg.close_nm)
    if close < 0:
        close = 2.0 * smooth
    return smooth, close


def width_tolerance_for(cfg: RodConfig, smooth_nm: float | None = None) -> float:
    """Slack added at each end of the width window, in nm.

    Absorbs the difference between the width of the density **mask** and the
    width of the structure it envelopes — see :class:`RodConfig`.
    """
    if cfg.width_tolerance_nm >= 0:
        return float(cfg.width_tolerance_nm)
    if smooth_nm is None:
        smooth_nm = cfg.smooth_nm if cfg.smooth_nm >= 0 else float(cfg.pixel_size_nm)
    return 2.0 * float(smooth_nm)


def seed_level_for(cfg: RodConfig) -> float:
    """Distance-transform level that seeds one marker per cell, in nm."""
    if cfg.seed_level_nm > 0:
        return float(cfg.seed_level_nm)
    return float(cfg.seed_level_fraction) * 0.5 * float(cfg.min_width_nm)


def capsule_outline(rod: Rod, *, n_cap: int = 16) -> np.ndarray:
    """Closed stadium outline ``(M, 2)`` of *rod* — the drawable cell shape."""
    p0, p1 = np.asarray(rod.endpoints_nm, dtype=float)
    axis = np.asarray(rod.axis_nm, dtype=float)
    normal = np.array([-axis[1], axis[0]])
    r = 0.5 * float(rod.width_nm)
    if r <= 0:
        return np.vstack([p0, p1])
    # Half-circle caps around each centreline end, joined by the two flanks.
    base = np.linspace(-np.pi / 2.0, np.pi / 2.0, max(int(n_cap), 2))
    end_cap = p1 + r * (np.cos(base)[:, None] * axis + np.sin(base)[:, None] * normal)
    start_cap = p0 + r * (np.cos(base)[:, None] * -axis + np.sin(base)[:, None] * -normal)
    return np.vstack([end_cap, start_cap, end_cap[:1]])


def _region_at(labels, x_edges, y_edges, xy: np.ndarray) -> np.ndarray:
    """Region label under each point of *xy*; 0 where the point is outside."""
    n = int(xy.shape[0])
    out = np.zeros(n, dtype=np.int64)
    if labels.size == 0 or n == 0:
        return out
    finite = np.isfinite(xy).all(axis=1)
    nx, ny = labels.shape
    xb = np.clip(np.searchsorted(x_edges, xy[finite, 0], side="right") - 1, 0, nx - 1)
    yb = np.clip(np.searchsorted(y_edges, xy[finite, 1], side="right") - 1, 0, ny - 1)
    out[finite] = labels[xb, yb]
    # A point outside the padded grid clamps onto an edge pixel, which is
    # background by construction, so no spurious membership is created.
    inside = np.zeros(n, dtype=bool)
    inside[finite] = ((xy[finite, 0] >= x_edges[0]) & (xy[finite, 0] <= x_edges[-1])
                      & (xy[finite, 1] >= y_edges[0]) & (xy[finite, 1] <= y_edges[-1]))
    return np.where(inside, out, 0)


def assign_points(result: "RodSegmentationResult", xy_nm) -> tuple[np.ndarray, np.ndarray]:
    """Assign an arbitrary point set to an existing detection.

    Returns ``(region_of_point, component_of_point)`` with the same conventions
    as the corresponding :class:`RodSegmentationResult` fields.  This is how a
    detection made on the dense localization cloud — the best available image
    of the cell footprint — is transferred onto a sparser derived point set
    such as inferred label sites.
    """
    xy = np.asarray(xy_nm, dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError("xy_nm must have shape (N, 2+)")
    region = _region_at(result.labels, result.x_edges, result.y_edges, xy[:, :2])
    component = np.full(region.shape[0], -1, dtype=np.int64)
    for index, rod in enumerate(result.accepted):
        component[region == rod.id] = index
    return region, component


def _split_regions(
    mask: np.ndarray,
    labels: np.ndarray,
    edt_nm: np.ndarray,
    seed_level_nm: float,
) -> tuple[np.ndarray, int]:
    """Split each connected component into one region per distance-transform seed.

    Returns ``(regions, n_split)``.  A component holding fewer than two seeds is
    passed through untouched, so a thin or small object is never annexed by a
    neighbouring cell across background.
    """
    from scipy.ndimage import distance_transform_edt, label

    orig = np.asarray(labels, dtype=np.int64)
    n_lab = int(orig.max())
    if n_lab == 0:
        return orig, 0

    seeds = mask & (edt_nm >= float(seed_level_nm))
    seed_lab, n_seed = label(seeds)
    if n_seed < 2:
        return orig, 0

    # Each seed is connected and lies inside the mask, so it belongs to exactly
    # one original component.
    picked = seed_lab > 0
    orig_of_seed = np.zeros(n_seed + 1, dtype=np.int64)
    orig_of_seed[seed_lab[picked]] = orig[picked]
    seeds_per_component = np.bincount(orig_of_seed[1:], minlength=n_lab + 1)

    indices = distance_transform_edt(
        seed_lab == 0, return_distances=False, return_indices=True)
    claim = seed_lab[indices[0], indices[1]]

    # A claim is honoured only inside the claiming seed's own component, and
    # only where that component actually holds two or more seeds.
    honoured = (mask
                & (seeds_per_component[orig] >= 2)
                & (orig_of_seed[claim] == orig))
    sub = np.where(honoured, claim, 0)

    key = np.where(mask, orig * (n_seed + 1) + sub, 0)
    uniq, inverse = np.unique(key.ravel(), return_inverse=True)
    regions = inverse.reshape(key.shape).astype(np.int64)
    if uniq[0] != 0:                       # no background at all (padding lost)
        regions += 1
    return regions, int(regions.max()) - n_lab


def _fit_region(
    region_id: int,
    selected: np.ndarray,
    edt_nm: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    cfg: RodConfig,
    tolerance: float,
) -> Rod:
    ps = float(cfg.pixel_size_nm)
    ix, iy = np.nonzero(selected)
    pixels = np.column_stack([x_centers[ix], y_centers[iy]])
    area = float(ix.size) * ps * ps
    width = 2.0 * float(edt_nm[selected].max())

    center, box_w, box_h, angle = min_area_rectangle(pixels)
    # The box encloses pixel *centres*, so it understates the footprint by half
    # a pixel on each side.
    box_w += ps
    box_h += ps
    if box_w >= box_h:
        length, box_width, axis_deg = box_w, box_h, angle
    else:
        length, box_width, axis_deg = box_h, box_w, angle + 90.0
    axis_deg = _axis_angle_deg(axis_deg)
    theta = np.radians(axis_deg)
    axis = np.array([np.cos(theta), np.sin(theta)])

    box_area = length * box_width
    fill = float(area / box_area) if box_area > 0 else 0.0
    ratio = float(box_width / width) if width > 0 else float("inf")

    reason = ""
    if width < cfg.min_width_nm - tolerance:
        reason = "too narrow"
    elif width > cfg.max_width_nm + tolerance:
        reason = "too wide"
    elif length < cfg.min_length_nm:
        reason = "too short"
    elif length > cfg.max_length_nm:
        reason = "too long"
    elif length <= width:
        reason = "not elongated"
    elif fill < cfg.min_fill_fraction:
        reason = "outline not rod-like"
    elif ratio > cfg.max_width_ratio:
        reason = "width mismatch (merged or bent)"

    half_line = max(0.5 * (length - width), 0.0)
    endpoints = np.vstack([center - half_line * axis, center + half_line * axis])
    return Rod(
        id=int(region_id),
        center_nm=np.asarray(center, dtype=float),
        axis_nm=axis,
        angle_deg=float(axis_deg),
        length_nm=float(length),
        width_nm=float(width),
        box_width_nm=float(box_width),
        area_nm2=area,
        fill_fraction=fill,
        width_ratio=ratio,
        n_pixels=int(ix.size),
        endpoints_nm=endpoints,
        accepted=not reason,
        reject_reason=reason,
    )


def detect_rods(xy_nm, cfg: RodConfig | None = None,
                spacing_points_nm=None) -> RodSegmentationResult:
    """Detect rod/capsule cells of the configured size in a projected XY view.

    *xy_nm* is an ``(N, 2+)`` array of localization coordinates in nm; only the
    first two columns are used.  Non-finite rows take no part in the image and
    are reported as belonging to no region.

    *spacing_points_nm* are the distinct molecular positions (trace centroids
    or inferred sites), used **only** to set the automatic bridging length —
    see :func:`resolve_smoothing`.  The image itself is always built from
    *xy_nm*, which covers the cell better.
    """
    from scipy.ndimage import distance_transform_edt

    cfg = cfg or RodConfig()
    _validate(cfg)
    xy = np.asarray(xy_nm, dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError("xy_nm must have shape (N, 2+)")
    xy = xy[:, :2]
    n_points = int(xy.shape[0])
    finite = np.isfinite(xy).all(axis=1)

    empty_points = (np.zeros(n_points, dtype=np.int64),
                    np.full(n_points, -1, dtype=np.int64))

    def _empty(reason: str) -> RodSegmentationResult:
        return RodSegmentationResult(
            region_of_point=empty_points[0], component_of_point=empty_points[1],
            pixel_size_nm=float(cfg.pixel_size_nm),
            stats={"n_regions": 0, "n_accepted": 0, "n_rejected": 0,
                   "n_split": 0, "reason": reason,
                   "pixel_size_nm": float(cfg.pixel_size_nm),
                   "seed_level_nm": seed_level_for(cfg), "rejections": {}},
        )

    points = xy[finite]
    if points.shape[0] < 3:
        return _empty("fewer than three finite localizations")

    ps = float(cfg.pixel_size_nm)
    smooth_nm, close_nm = resolve_smoothing(points, cfg, spacing_points_nm)
    tolerance = width_tolerance_for(cfg, smooth_nm)
    pad = max(10.0 * ps, 2.0 * close_nm, 2.0 * smooth_nm)
    span = points.max(axis=0) - points.min(axis=0) + 2.0 * pad
    if float(np.prod(np.ceil(span / ps))) > MAX_GRID_PIXELS:
        raise ValueError(
            f"the field of view needs more than {MAX_GRID_PIXELS:,} pixels at "
            f"{ps:g} nm/pixel — increase the detection pixel size")

    min_area = (float(cfg.min_area_nm2) if cfg.min_area_nm2 > 0
                else 0.25 * float(cfg.min_width_nm) * float(cfg.min_length_nm))
    cell = compute_cell_mask(
        points, border_size_nm=0.0, pixel_size_nm=ps,
        smooth_nm=smooth_nm, close_nm=close_nm,
        open_nm=float(cfg.open_nm), min_cell_area_nm2=min_area,
    )
    mask = cell.mask
    if mask.size == 0 or not mask.any():
        return _empty(cell.stats.get("reason", "no cell mask found"))

    edt_nm = distance_transform_edt(mask) * ps
    seed_level = seed_level_for(cfg)
    if cfg.split_touching:
        regions, n_split = _split_regions(mask, cell.labels, edt_nm, seed_level)
    else:
        regions, n_split = cell.labels.astype(np.int64), 0

    x_centers = 0.5 * (cell.x_edges[:-1] + cell.x_edges[1:])
    y_centers = 0.5 * (cell.y_edges[:-1] + cell.y_edges[1:])
    rods = [
        _fit_region(rid, regions == rid, edt_nm, x_centers, y_centers, cfg, tolerance)
        for rid in range(1, int(regions.max()) + 1)
    ]

    region_of_point = _region_at(regions, cell.x_edges, cell.y_edges, xy)
    accepted_index = {rod.id: i for i, rod in enumerate(rods) if rod.accepted}
    component_of_point = np.full(n_points, -1, dtype=np.int64)
    for region_id, index in accepted_index.items():
        component_of_point[region_of_point == region_id] = index
    for rod in rods:
        rod.n_points = int(np.count_nonzero(region_of_point == rod.id))

    rejections: dict[str, int] = {}
    for rod in rods:
        if not rod.accepted:
            rejections[rod.reject_reason] = rejections.get(rod.reject_reason, 0) + 1

    accepted = [rod for rod in rods if rod.accepted]
    stats = {
        "n_regions": len(rods),
        "n_accepted": len(accepted),
        "n_rejected": len(rods) - len(accepted),
        "n_split": int(n_split),
        "n_points_in_rods": int(np.count_nonzero(component_of_point >= 0)),
        "rejections": rejections,
        "pixel_size_nm": ps,
        "seed_level_nm": seed_level,
        "smooth_nm": smooth_nm,
        "close_nm": close_nm,
        "smooth_is_auto": bool(cfg.smooth_nm < 0),
        "width_tolerance_nm": tolerance,
        "width_window_nm": [float(cfg.min_width_nm), float(cfg.max_width_nm)],
        "min_area_nm2": min_area,
        # Every region's measured width, rejected ones included, so a window
        # that is merely slightly off is visible in one pass instead of
        # presenting as "nothing detected".
        "region_widths_nm": [float(rod.width_nm) for rod in rods],
        "median_width_nm": (float(np.median([r.width_nm for r in accepted]))
                            if accepted else float("nan")),
        "median_length_nm": (float(np.median([r.length_nm for r in accepted]))
                             if accepted else float("nan")),
        "mask_area_nm2": float(mask.sum()) * ps * ps,
    }
    return RodSegmentationResult(
        rods=rods, labels=regions, mask=mask, edt_nm=edt_nm,
        region_of_point=region_of_point, component_of_point=component_of_point,
        x_edges=cell.x_edges, y_edges=cell.y_edges, pixel_size_nm=ps, stats=stats,
    )
