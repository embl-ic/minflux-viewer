"""Magic-wand selection for localization data: grow a region, trace its outline.

ImageJ's wand needs three things -- a scalar field, a tolerance and a
connectivity rule -- and localization data supplies none of them natively. Here
the field is any **per-localization attribute** (the viewer's *Color by* value in
the scatter plot, local density in the render view), the tolerance is a band
around the clicked point's own value, and connectivity is plain Cartesian
distance between coordinates. There is deliberately no 4-/8-connectivity option:
that question only exists for rastered pixels.

Pure NumPy/SciPy, Qt-free, so the rule can be tested without a display.

⚠ The value band is measured against the **seed**, not against each neighbour.
Comparing against the neighbour lets the value drift arbitrarily far from the
clicked point along a chain -- you walk a gradient and end up selecting the whole
field -- which is both surprising and unbounded. Fiji derives its range from the
seed pixel, and so does this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "WandResult",
    "DEFAULT_DISTANCE_NM",
    "DEFAULT_VALUE_TOLERANCE_PCT",
    "MAX_GRID_CELLS",
    "nearest_index",
    "value_tolerance_from_percent",
    "grow_selection",
    "trace_selection_outline",
    "wand_polygon",
    "wand_from_rows",
    "wand_parameters",
    "describe_wand",
]

#: Connectivity distance a fresh install starts from, in nm. A MINFLUX
#: structure's localizations sit tens of nm apart, so this is the scale that
#: joins one structure without bridging to its neighbour.
DEFAULT_DISTANCE_NM = 20.0

#: Value band as a percentage of the attribute's finite range. A percentage
#: rather than absolute units because the attributes differ by orders of
#: magnitude -- ``cfr`` spans 0..1 while ``tid`` spans thousands -- so one
#: absolute slider could not serve them.
DEFAULT_VALUE_TOLERANCE_PCT = 10.0

#: Upper bound on the outline grid, guarding a fine tolerance over a wide field
#: the same way the local-density histogram caps its voxel count.
MAX_GRID_CELLS = 4_000_000

#: Smallest hole kept in the outline, in grid cells. A hole one cell across is
#: at the resolution the outline has and is speckle rather than structure --
#: uniform noise produced 3479 such rings out of 4599, which made the polygon
#: 33,556 vertices long and told the user nothing. Dropped holes are counted and
#: reported, never silently discarded: the polygon is then slightly *generous*
#: (it covers points the grow rejected), which is the opposite error from the
#: seam bug and must be visible.
MIN_HOLE_CELLS = 4.0

#: Hard cap on rings in one outline, so a pathological click cannot produce an
#: unusable ROI however large its holes are.
MAX_OUTLINE_RINGS = 64


@dataclass
class WandResult:
    """What one wand click produced.

    ``mask`` is the grown set over the *candidate* rows handed in; ``polygon`` is
    its traced outline, with any holes appended as further rings (even-odd, the
    rule ``roi_selection.polygon_mask`` applies). ``rings`` records how many
    closed loops the outline is made of, so a caller can say "with 2 hole(s)"
    rather than leaving the user to wonder.
    """

    mask: np.ndarray
    polygon: list[list[float]]
    rings: int
    rings_dropped: int
    seed_value: float
    value_tolerance: float
    distance_nm: float
    n_selected: int
    n_candidates: int


def describe_wand(result: "WandResult", field: str) -> str:
    """One status line both views share, so the wand reads the same everywhere.

    It names the field, the counts and both tolerances in the units they were
    applied in, and it reports dropped speckle holes -- the one place the outline
    is deliberately more generous than the selection it came from.
    """
    text = (f"Magic wand on '{field}': {result.n_selected} of "
            f"{result.n_candidates} localization(s) — {result.distance_nm:g} nm reach, "
            f"value within ±{result.value_tolerance:.4g} of {result.seed_value:.4g}")
    holes = max(0, result.rings - 1)
    if holes:
        text += f", {holes} hole(s)"
    if result.rings_dropped:
        text += f" ({result.rings_dropped} speckle hole(s) dropped)"
    return text


def wand_parameters(prefs: dict | None) -> dict:
    """The saved wand tolerances, as keyword arguments for a view's hook."""
    saved = ((prefs or {}).get("wand") or {})
    try:
        distance = float(saved.get("distance_nm", DEFAULT_DISTANCE_NM))
    except (TypeError, ValueError):
        distance = DEFAULT_DISTANCE_NM
    try:
        percent = float(saved.get("value_percent", DEFAULT_VALUE_TOLERANCE_PCT))
    except (TypeError, ValueError):
        percent = DEFAULT_VALUE_TOLERANCE_PCT
    return {"distance_nm": max(1e-9, distance), "value_percent": max(0.0, percent)}


def wand_from_rows(
    locs,
    values,
    base_mask,
    columns,
    position,
    *,
    distance_nm: float = DEFAULT_DISTANCE_NM,
    value_percent: float = DEFAULT_VALUE_TOLERANCE_PCT,
    grow_3d: bool = False,
) -> WandResult | None:
    """A wand click over row-aligned coordinates and values.

    *locs* is the view's ``(N, 3)`` display-nm array, *values* the matching
    per-localization attribute, *base_mask* the rows that are actually visible
    (the active filter, finite coordinates) and *columns* the two data-axis
    columns the view plots. The returned ``mask`` is full length, so a caller can
    use it against its own dataset rows directly.

    ⚠ Invisible rows are excluded **before** the grow, not after. A wand that
    grew through filtered-out localizations would bridge across a gap the user
    can see is empty, and its outline would enclose points that are not on
    screen -- the same discipline the detection tools follow by reading
    ``display_xyz_filtered``.
    """
    pts = np.asarray(locs, dtype=float)
    vals = np.asarray(values, dtype=float).ravel()
    if pts.ndim != 2 or pts.shape[0] == 0:
        return None
    n = pts.shape[0]
    if vals.size != n:
        return None
    try:
        ci, cj = int(columns[0]), int(columns[1])
    except (TypeError, ValueError, IndexError):
        return None
    if max(ci, cj) >= pts.shape[1]:
        return None

    base = (np.ones(n, dtype=bool) if base_mask is None
            else np.asarray(base_mask, dtype=bool).ravel())
    if base.size != n:
        base = np.ones(n, dtype=bool)
    plane = np.column_stack([pts[:, ci], pts[:, cj]])
    usable = base & np.all(np.isfinite(plane), axis=1) & np.isfinite(vals)
    idx = np.flatnonzero(usable)
    if idx.size == 0:
        return None

    # ⚠ With grow_3d the connectivity is measured on all three coordinates while
    # the click is still resolved in the pane's two. Growing in the projection
    # instead treats points that merely overlap on screen as neighbours, however
    # far apart they are on the axis the pane collapses -- the same class of
    # error as ignoring the depth slice.
    grow_points = None
    if grow_3d and pts.shape[1] >= 3:
        grow_points = pts[idx][:, :3]
    result = wand_polygon(plane[idx], vals[idx], position,
                          distance_nm=distance_nm, value_percent=value_percent,
                          grow_points=grow_points)
    if result is None:
        return None
    full = np.zeros(n, dtype=bool)
    full[idx[result.mask]] = True
    result.mask = full
    result.n_candidates = int(idx.size)
    return result


def nearest_index(points: np.ndarray, position,
                  *, max_distance: float | None = None) -> int | None:
    """Index of the coordinate-wise nearest row to *position*, or ``None``.

    The wand is seeded by a click, and a click lands between points, so the
    nearest one is the only defensible reading of "the point you clicked" --
    but only within *max_distance*.

    ⚠ Without that bound a click on empty space still seeds from whatever point
    happened to be closest, however far away, so the wand appeared to select a
    region out of nowhere. One tolerance governs the whole gesture: the same
    distance that decides whether two points are connected decides whether the
    click landed on anything at all.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
        return None
    pos = np.asarray(position, dtype=float).ravel()[:2]
    if pos.size < 2 or not np.all(np.isfinite(pos)):
        return None
    finite = np.all(np.isfinite(pts[:, :2]), axis=1)
    if not finite.any():
        return None
    d2 = np.sum((pts[finite, :2] - pos) ** 2, axis=1)
    best = int(np.argmin(d2))
    if max_distance is not None:
        limit = float(max_distance)
        if not np.isfinite(limit) or limit < 0.0 or d2[best] > limit * limit:
            return None
    return int(np.flatnonzero(finite)[best])


def value_tolerance_from_percent(values, percent: float) -> float:
    """Absolute value band for *percent* of an attribute's finite range.

    Returns 0.0 for a constant or empty attribute -- the band is then exact
    equality, which is the honest reading of "10% of no range at all".
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    spread = float(v.max() - v.min())
    if not np.isfinite(spread) or spread <= 0.0:
        return 0.0
    return spread * max(0.0, float(percent)) / 100.0


def grow_selection(
    points: np.ndarray,
    values: np.ndarray,
    seed: int,
    *,
    distance_nm: float,
    value_tolerance: float,
    max_points: int | None = None,
) -> np.ndarray:
    """Grow from *seed* through neighbours within the distance and value bands.

    *points* is ``(N, 2)`` or ``(N, 3)`` display-nm coordinates and *values* the
    matching per-row attribute. Returns a boolean mask over those N rows.

    The admissible set (inside the value band) is resolved **first**, and the
    region grow is then plain connectivity inside it. That ordering is what keeps
    the cost proportional to the selection rather than to the dataset: a tight
    band leaves few candidates for the tree to index.
    """
    pts = np.asarray(points, dtype=float)
    vals = np.asarray(values, dtype=float).ravel()
    n = pts.shape[0] if pts.ndim == 2 else 0
    out = np.zeros(max(n, 0), dtype=bool)
    if n == 0 or vals.size != n or not (0 <= int(seed) < n):
        return out
    seed = int(seed)
    radius = float(distance_nm)
    if not np.isfinite(radius) or radius <= 0.0:
        return out

    dims = min(3, pts.shape[1])
    coords = pts[:, :dims]
    seed_value = vals[seed]
    if not np.isfinite(seed_value):
        return out

    band = float(max(0.0, value_tolerance))
    admissible = (np.abs(vals - seed_value) <= band) & np.all(np.isfinite(coords), axis=1)
    if not admissible[seed]:
        admissible[seed] = True          # the clicked point is always in its own region
    idx = np.flatnonzero(admissible)
    if idx.size == 0:
        return out

    local_seed = int(np.searchsorted(idx, seed))
    sub = coords[idx]

    try:
        from scipy.spatial import cKDTree
    except ImportError:                  # pragma: no cover - SciPy is a dependency
        cKDTree = None

    reached = np.zeros(idx.size, dtype=bool)
    reached[local_seed] = True
    frontier = np.array([local_seed], dtype=np.intp)
    budget = idx.size if max_points is None else min(idx.size, int(max_points))

    if cKDTree is not None:
        tree = cKDTree(sub)
        while frontier.size:
            neighbours = tree.query_ball_point(sub[frontier], radius, workers=-1)
            nxt: list[int] = []
            for group in neighbours:
                for j in group:
                    if not reached[j]:
                        reached[j] = True
                        nxt.append(j)
            if not nxt or int(reached.sum()) >= budget:
                break
            frontier = np.asarray(nxt, dtype=np.intp)
    else:                                # pragma: no cover - brute-force fallback
        r2 = radius * radius
        while frontier.size:
            nxt = []
            for f in frontier:
                d2 = np.sum((sub - sub[f]) ** 2, axis=1)
                for j in np.flatnonzero((d2 <= r2) & ~reached):
                    reached[j] = True
                    nxt.append(int(j))
            if not nxt:
                break
            frontier = np.asarray(nxt, dtype=np.intp)

    out[idx[reached]] = True
    return out


def _occupancy(points: np.ndarray, cell: float):
    """Binary grid of the cells the selected points fall in, plus its origin."""
    xs, ys = points[:, 0], points[:, 1]
    x0 = float(np.floor(xs.min() / cell) - 1.0) * cell
    y0 = float(np.floor(ys.min() / cell) - 1.0) * cell
    cols = int(np.ceil((xs.max() - x0) / cell)) + 2
    rows = int(np.ceil((ys.max() - y0) / cell)) + 2
    if rows <= 0 or cols <= 0 or rows * cols > MAX_GRID_CELLS:
        return None, 0.0, 0.0
    ci = np.clip(((xs - x0) / cell).astype(np.intp), 0, cols - 1)
    ri = np.clip(((ys - y0) / cell).astype(np.intp), 0, rows - 1)
    grid = np.zeros((rows, cols), dtype=bool)
    grid[ri, ci] = True
    return grid, x0, y0


def _boundary_rings(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    """Closed rings of lattice points bounding the occupied cells.

    Every occupied cell contributes each of its four edges that faces an
    unoccupied cell, directed so the traversal of a single cell is consistent.
    Linking those directed unit edges end-to-start closes every loop -- the outer
    boundary and one ring per hole -- with no marching-squares table and no
    dependency on scikit-image, which this project does not ship.
    """
    rows, cols = grid.shape
    padded = np.zeros((rows + 2, cols + 2), dtype=bool)
    padded[1:-1, 1:-1] = grid
    occ = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]

    starts: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def add_edges(mask, d_start, d_end):
        rr, cc = np.nonzero(mask)
        for r, c in zip(rr.tolist(), cc.tolist()):
            s = (c + d_start[0], r + d_start[1])
            e = (c + d_end[0], r + d_end[1])
            starts.setdefault(s, []).append(e)

    # Clockwise around each cell in (x=col, y=row): top edge goes +x, right +y,
    # bottom -x, left -y.
    add_edges(occ & ~up, (0, 0), (1, 0))
    add_edges(occ & ~right, (1, 0), (1, 1))
    add_edges(occ & ~down, (1, 1), (0, 1))
    add_edges(occ & ~left, (0, 1), (0, 0))

    rings: list[list[tuple[int, int]]] = []
    while starts:
        first = next(iter(starts))
        ring = [first]
        node = first
        while True:
            outgoing = starts.get(node)
            if not outgoing:
                break
            nxt = outgoing.pop()
            if not outgoing:
                del starts[node]
            if nxt == first:
                break
            ring.append(nxt)
            node = nxt
            if len(ring) > 4 * grid.size + 8:      # cannot happen; refuse to spin
                break
        if len(ring) >= 4:
            rings.append(ring)
    return rings


def _drop_collinear(ring: list[tuple[float, float]]) -> list[list[float]]:
    """Keep only the corners of a rectilinear ring (a straight run needs two).

    ⚠ A zero cross product alone does NOT mean "straight". Where the boundary
    doubles back along a one-cell-wide spur the incoming and outgoing directions
    are opposite, which is also collinear -- dropping that vertex would collapse
    the spur, so the polygon would lose a thin protrusion while its points stayed
    selected. The dot product separates a straight continuation from a reversal.

    (Guarded by ``test_a_one_cell_wide_arm_survives_the_outline``. This is a
    defensive fix: the annulus points that went missing turned out to be the
    seam bug in ``_splice_rings``, not this.)
    """
    n = len(ring)
    if n < 3:
        return [[float(x), float(y)] for x, y in ring]
    out: list[list[float]] = []
    for i in range(n):
        px, py = ring[i - 1]
        cx, cy = ring[i]
        nx, ny = ring[(i + 1) % n]
        ix, iy = cx - px, cy - py
        ox, oy = nx - cx, ny - cy
        cross = ix * oy - iy * ox
        dot = ix * ox + iy * oy
        if abs(cross) > 1e-9 or dot <= 0.0:
            out.append([float(cx), float(cy)])
    return out or [[float(x), float(y)] for x, y in ring]


def _splice_rings(rings: list[list[list[float]]]) -> list[list[float]]:
    """One even-odd vertex list from several closed rings (the "keyhole" bridge).

    ⚠ Concatenating the rings is NOT enough, and the failure is silent. A single
    vertex list is one closed polygon to ``_point_in_polygon``, so joining an
    outer ring to a hole ring introduces an implicit seam edge straight across
    the region -- and that edge flips the parity of every point beside it.
    Measured on a 4000-point annulus: 16 selected points fell outside their own
    outline, all of them lying along the seam.

    Walking the bridge **out and back** fixes it: the two coincident, opposite
    segments are each counted by the ray test, so their parity contributions
    cancel and only the real ring edges decide membership. The same trick merges
    a genuinely disconnected component, because a cancelling bridge carries no
    parity either way -- so no ring has to be classified as hole or component.
    """
    if not rings:
        return []
    outer = [list(map(float, pt)) for pt in rings[0]]
    if len(rings) == 1:
        return outer

    # ⚠ Anchor every bridge to the OUTER ring, not to the polygon built so far.
    # Bridging to the growing list rebuilt a search tree per ring over an
    # ever-longer vertex list -- quadratic, and measured at 46.1 s of a 46.9 s
    # click on 2 M rows. The outer ring is an equally valid anchor (a cancelling
    # bridge carries no parity), and one tree serves every hole.
    inserts: dict[int, list[list[float]]] = {}
    outer_xy = np.asarray(outer, dtype=float)[:, :2]
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(outer_xy)
    except ImportError:                       # pragma: no cover - SciPy is a dependency
        tree = None

    for ring in rings[1:]:
        if len(ring) < 3:
            continue
        hole = [list(map(float, pt)) for pt in ring]
        hole_xy = np.asarray(hole, dtype=float)[:, :2]
        if tree is not None:
            dist, idx = tree.query(hole_xy, k=1, workers=-1)
            h = int(np.argmin(dist))
            i = int(idx[h])
        else:                                 # pragma: no cover
            d2 = ((hole_xy[:, None, :] - outer_xy[None, :, :]) ** 2).sum(-1)
            h, i = (int(v) for v in np.unravel_index(int(np.argmin(d2)), d2.shape))
        bridge = [hole[(h + k) % len(hole)] for k in range(len(hole))]
        bridge.append(list(hole[h]))          # close the ring just walked
        bridge.append(list(outer[i]))         # and return along the same seam
        inserts.setdefault(i, []).extend(bridge)

    polygon: list[list[float]] = []
    for i, pt in enumerate(outer):
        polygon.append(pt)
        polygon.extend(inserts.get(i, ()))
    return polygon


def _ring_cell_area(ring) -> float:
    """Enclosed area of a lattice ring, in grid cells (shoelace, sign dropped)."""
    a = np.asarray(ring, dtype=float)
    if a.shape[0] < 3:
        return 0.0
    x, y = a[:, 0], a[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def trace_selection_outline(points: np.ndarray,
                            cell: float) -> tuple[list[list[float]], int, int]:
    """Outline of a selected point set as ``(polygon, rings_kept, rings_dropped)``.

    Holes are appended as further rings, which the even-odd point-in-polygon rule
    reads as holes -- the same representation ``roi_convert.buffer_line_to_polygon``
    already uses for an annulus.

    The outline is rectilinear at the *connectivity* scale, deliberately: the
    selection has no finer resolution than the distance that joined it, so a
    smoother curve would imply precision the grow never had.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
        return [], 0, 0
    cell = float(cell)
    if not np.isfinite(cell) or cell <= 0.0:
        return [], 0, 0
    grid, x0, y0 = _occupancy(pts[:, :2], cell)
    if grid is None or not grid.any():
        return [], 0, 0
    rings = _boundary_rings(grid)
    if not rings:
        return [], 0, 0
    areas = [_ring_cell_area(ring) for ring in rings]
    order = sorted(range(len(rings)), key=lambda i: areas[i], reverse=True)
    kept = [rings[order[0]]]                  # the largest ring is the boundary
    dropped = 0
    for i in order[1:]:
        if areas[i] < MIN_HOLE_CELLS or len(kept) >= MAX_OUTLINE_RINGS:
            dropped += 1
            continue
        kept.append(rings[i])
    simplified = [_drop_collinear([(x0 + gx * cell, y0 + gy * cell) for gx, gy in ring])
                  for ring in kept]
    return _splice_rings(simplified), len(kept), dropped


def wand_polygon(
    points: np.ndarray,
    values: np.ndarray,
    position,
    *,
    distance_nm: float = DEFAULT_DISTANCE_NM,
    value_percent: float = DEFAULT_VALUE_TOLERANCE_PCT,
    value_tolerance: float | None = None,
    cell_nm: float | None = None,
    grow_points=None,
) -> WandResult | None:
    """One wand click: seed, grow, trace. ``None`` when there is nothing to seed.

    *values* must be row-aligned with *points*. ``value_tolerance`` overrides the
    percentage form; ``cell_nm`` overrides the outline resolution, which defaults
    to the connectivity distance because that is the scale the selection has.

    ``grow_points`` supplies the coordinates CONNECTIVITY is measured on, which
    need not be the ones the click is measured on: the orthogonal view grows in
    3-D while still being clicked in a flat pane. Seeding stays 2-D on purpose --
    the user clicks what they can see, and the depth of a click in a projection
    is not knowable.
    """
    pts = np.asarray(points, dtype=float)
    vals = np.asarray(values, dtype=float).ravel()
    if pts.ndim != 2 or pts.shape[0] == 0 or vals.size != pts.shape[0]:
        return None
    # The click must land on something: no point within the connectivity
    # distance means the user clicked empty space, and empty space selects
    # nothing rather than the nearest structure somewhere else.
    seed = nearest_index(pts, position, max_distance=distance_nm)
    if seed is None or not np.isfinite(vals[seed]):
        return None
    band = (value_tolerance_from_percent(vals, value_percent)
            if value_tolerance is None else float(value_tolerance))
    growth = pts if grow_points is None else np.asarray(grow_points, dtype=float)
    if growth.ndim != 2 or growth.shape[0] != pts.shape[0]:
        growth = pts
    mask = grow_selection(growth, vals, seed,
                          distance_nm=distance_nm, value_tolerance=band)
    if not mask.any():
        return None
    cell = float(cell_nm) if cell_nm else float(distance_nm)
    polygon, rings, dropped = trace_selection_outline(pts[mask], cell)
    return WandResult(
        mask=mask,
        polygon=polygon,
        rings=rings,
        rings_dropped=dropped,
        seed_value=float(vals[seed]),
        value_tolerance=band,
        distance_nm=float(distance_nm),
        n_selected=int(mask.sum()),
        n_candidates=int(pts.shape[0]),
    )
