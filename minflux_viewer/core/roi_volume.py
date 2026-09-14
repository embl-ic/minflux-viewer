"""
minflux_viewer.core.roi_volume
==============================
Volume (3-D) ROIs: geometry, membership, seeding and per-view silhouettes.

Pure NumPy, Qt-free, so the whole contract is testable headless.

Geometry is stored in **named data axes**, never as "a 2-D shape plus a plane
name". That is not a style preference — this codebase carries two mutually
incompatible readings of the string ``"YZ"``: an ortho pane draws Z horizontally
(``ortho_view.ORTHO_AXIS_COLUMNS["YZ"] == (2, 1)``) while a standalone YZ
projection draws Y horizontally (``AXIS_COLUMNS["YZ"] == (1, 2)``). Both are
right for their own view, and four numbers plus a plane name do not say which
one they are in. Naming the axes removes the question::

    cuboid      {"x": [lo, hi], "y": [lo, hi], "z": [lo, hi]}
    sphere      {"center": [cx, cy, cz], "radii": [rx, ry, rz]}
    polyhedron  {"axis": "Z",
                 "levels": [{"at": z0, "polygon": [[u, v], ...]}, ...]}

``sphere`` is the user-facing name for an **axis-aligned ellipsoid** — a drag
almost never produces equal radii, and the name is kept because it is what a
user looks for.

For a ``polyhedron`` the ``axis`` is the stacking axis and each level's polygon
is given in the **remaining two axes, in canonical (ascending) order**:

    axis "Z" -> polygon vertices are (X, Y)
    axis "Y" -> polygon vertices are (X, Z)
    axis "X" -> polygon vertices are (Y, Z)

A **prism is the degenerate case with one level** plus ``thickness``; the same
record type, mask and editor serve both, so a prism upgrades to a multi-level
polyhedron with no migration.

Interpolation between levels is **angular resampling** (star-shaped
cross-sections): each polygon is described by its centroid plus a radius at
fixed angles, and intermediate cross-sections interpolate those radii. Unlike
pairing vertices it tolerates different vertex counts and never self-intersects,
and unlike rasterised signed-distance interpolation it introduces no pixel size
— which matters here because a localization dataset has no native resolution, so
a rasterised ROI would make "is this point inside?" depend on a grid chosen for
smoothing. The trade-off is that a cross-section must be star-shaped about its
centroid; ``radial_profile`` says so rather than guessing.
"""

from __future__ import annotations

import numpy as np

from .roi_selection import VOLUME_ROI_TYPES, _point_in_polygon

__all__ = [
    "VOLUME_ROI_TYPES",
    "AXIS_INDEX",
    "AXIS_NAMES",
    "DEFAULT_ANGLES",
    "MIN_SEED_LOCS",
    "MIN_SEED_THICKNESS_NM",
    "EMPTY_SEED_FRACTION",
    "cross_axes",
    "seed_interval",
    "radial_profile",
    "cross_section_at",
    "roi_volume_mask",
    "volume_bounds",
    "volume_silhouette",
    "NotStarShaped",
    "PLANE_NORMAL_AXIS",
    "FLAT_TO_VOLUME",
    "plane_in_plane_axes",
    "volume_from_flat",
    "scale_z",
    "translate_volume",
]


#: The data axis a standalone 2-D projection does NOT show.
PLANE_NORMAL_AXIS: dict[str, str] = {"XY": "Z", "XZ": "Y", "YZ": "X"}

#: Which volume type a 2-D drawing tool's shape becomes.
FLAT_TO_VOLUME: dict[str, str] = {
    "rectangle": "cuboid",
    "oval": "sphere",
    "polygon": "polyhedron",
    "freehand": "polyhedron",
}


#: Data-axis name -> column of an ``(N, 3)`` display-nm array.
AXIS_INDEX: dict[str, int] = {"X": 0, "Y": 1, "Z": 2}
AXIS_NAMES: tuple[str, str, str] = ("X", "Y", "Z")

#: Angles used to describe a cross-section radially. 64 keeps a 100 nm pore's
#: boundary within ~0.1 nm of the drawn polygon, far below the localization
#: precision, while staying cheap enough to recompute per mask call.
DEFAULT_ANGLES = 64

#: Below this many localizations, "bounding of the contained data" is noise
#: rather than a measurement, so the empty-region rule applies instead.
MIN_SEED_LOCS = 10

#: A zero-thickness region is not a region: its mask selects nothing.
MIN_SEED_THICKNESS_NM = 20.0

#: Fraction of the visible range a seeded interval spans when there is no data
#: to measure. One constant for cuboid, sphere and polyhedron alike — the third
#: dimension of all three is an interval, so one rule governs them.
EMPTY_SEED_FRACTION = 0.5


class NotStarShaped(ValueError):
    """A cross-section is not star-shaped about its centroid.

    Raised rather than silently approximated: the radial description would
    quietly drop a lobe, and a ROI that selects the wrong localizations without
    saying so is worse than one that refuses to be built.
    """


def cross_axes(axis: str) -> tuple[int, int]:
    """Column indices of the two axes a ``polyhedron``'s polygons live in.

    The stacking *axis* is excluded and the remaining two are returned in
    ascending order, so the mapping is fixed by the data axes alone and cannot
    depend on which view happened to draw the shape.
    """
    key = str(axis).upper()
    if key not in AXIS_INDEX:
        raise ValueError(f"unknown axis {axis!r}; expected one of {AXIS_NAMES}")
    keep = [i for i in range(3) if i != AXIS_INDEX[key]]
    return keep[0], keep[1]


# --------------------------------------------------------------------- seeding
def seed_interval(
    depth_of_contained,
    visible_lo: float,
    visible_hi: float,
    *,
    crosshair: float | None = None,
) -> tuple[float, float]:
    """``(lo, hi)`` for the axis **normal to the drawing plane** of a new ROI.

    One rule for every volume shape, because the third dimension of all of them
    is an interval — a cuboid's extent, an ellipsoid's diameter, a prism's
    extrusion length. Only the in-plane shape differs.

    With enough contained localizations the interval is their full extent, so
    nothing the user drew over is left out. With too few (or none — drawing in a
    void beside a structure is a normal thing to do) it falls back to a fraction
    of the **visible** range centred on the crosshair, which is a visible
    placeholder rather than a measurement of noise. The result never exceeds the
    visible range, and never collapses to zero thickness.
    """
    z = np.asarray(depth_of_contained, dtype=float).ravel()
    z = z[np.isfinite(z)]
    visible_lo, visible_hi = float(visible_lo), float(visible_hi)
    if visible_hi < visible_lo:
        visible_lo, visible_hi = visible_hi, visible_lo

    if z.size >= MIN_SEED_LOCS:
        lo, hi = float(z.min()), float(z.max())
    else:
        centre = float(crosshair) if crosshair is not None else 0.5 * (visible_lo + visible_hi)
        half = 0.5 * EMPTY_SEED_FRACTION * (visible_hi - visible_lo)
        lo, hi = centre - half, centre + half

    if hi - lo < MIN_SEED_THICKNESS_NM:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.5 * MIN_SEED_THICKNESS_NM, mid + 0.5 * MIN_SEED_THICKNESS_NM

    # Clamp into the visible range, but never below the minimum thickness: on a
    # view zoomed tighter than MIN_SEED_THICKNESS_NM the clamp would otherwise
    # hand back the degenerate interval this function exists to avoid.
    if visible_hi - visible_lo >= MIN_SEED_THICKNESS_NM:
        lo, hi = max(lo, visible_lo), min(hi, visible_hi)
    return float(lo), float(hi)


# ------------------------------------------------------- radial cross-sections
def _polygon_centroid(poly: np.ndarray) -> np.ndarray:
    """Area-weighted centroid, falling back to the vertex mean for a degenerate
    (zero-area) outline so a collapsed polygon still yields a usable centre."""
    x, y = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = 0.5 * float(cross.sum())
    if abs(area) < 1e-12:
        return poly.mean(axis=0)
    cx = float(((x + x1) * cross).sum()) / (6.0 * area)
    cy = float(((y + y1) * cross).sum()) / (6.0 * area)
    return np.array([cx, cy], dtype=float)


def radial_profile(
    polygon,
    *,
    n_angles: int = DEFAULT_ANGLES,
    strict: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Describe a polygon as ``(centroid, radii)`` at ``n_angles`` fixed angles.

    Rays are cast from the centroid at angles ``2*pi*j/n_angles``; the radius is
    the distance to the outline along each. This is the representation that
    makes interpolation between levels well defined for any two polygons,
    whatever their vertex counts or starting vertices.

    ⚠ Valid only for a **star-shaped** outline (every ray crosses the boundary
    once). With ``strict`` a ray that crosses more than once raises
    :class:`NotStarShaped`; without it the outermost crossing is taken, which
    fills concavities the user drew. A ray that misses entirely (possible only
    for a degenerate outline) contributes radius 0.
    """
    poly = np.asarray(polygon, dtype=float)
    if poly.ndim != 2 or poly.shape[0] < 3:
        raise ValueError("a cross-section needs at least three vertices")
    poly = poly[:, :2]
    centre = _polygon_centroid(poly)

    angles = np.arange(n_angles, dtype=float) * (2.0 * np.pi / n_angles)
    dirs = np.column_stack([np.cos(angles), np.sin(angles)])          # (A, 2)

    p = poly                                                           # (M, 2)
    q = np.roll(poly, -1, axis=0)                                      # (M, 2)
    edge = q - p                                                       # (M, 2)
    rel = p - centre                                                   # (M, 2)

    # Ray c + t*d meets segment p + s*edge:  t = rel x edge / (d x edge)
    denom = dirs[:, None, 0] * edge[None, :, 1] - dirs[:, None, 1] * edge[None, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (rel[None, :, 0] * edge[None, :, 1] - rel[None, :, 1] * edge[None, :, 0]) / denom
        s = (rel[None, :, 0] * dirs[:, None, 1] - rel[None, :, 1] * dirs[:, None, 0]) / denom
    hit = np.isfinite(t) & np.isfinite(s) & (t > 1e-9) & (s >= -1e-9) & (s <= 1.0 + 1e-9)

    radii = np.zeros(n_angles, dtype=float)
    for j in range(n_angles):
        ts = t[j][hit[j]]
        if ts.size == 0:
            continue
        if strict and ts.size > 1:
            # A ray grazing a shared vertex hits both edges at the same t; that
            # is one crossing, not two, so dedupe before judging the shape.
            if np.ptp(ts) > 1e-6:
                raise NotStarShaped(
                    "cross-section is not star-shaped about its centroid "
                    f"(ray {j} of {n_angles} crosses the outline {ts.size} times)"
                )
        radii[j] = float(ts.max())
    return centre, radii


def _levels_sorted(record_geometry: dict) -> tuple[np.ndarray, list]:
    levels = list(record_geometry.get("levels") or [])
    if not levels:
        raise ValueError("a polyhedron needs at least one level")
    at = np.array([float(lv["at"]) for lv in levels], dtype=float)
    order = np.argsort(at)
    return at[order], [levels[i] for i in order]


def cross_section_at(
    geometry: dict,
    at: float,
    *,
    n_angles: int = DEFAULT_ANGLES,
    strict: bool = True,
) -> np.ndarray | None:
    """The interpolated ``(n_angles, 2)`` outline at stacking coordinate *at*.

    ``None`` outside the level range — a polyhedron is not extrapolated past the
    outermost cross-section the user drew. A single-level record (a prism) uses
    its ``thickness`` to define that range and returns the same outline
    throughout.
    """
    at_values, levels = _levels_sorted(geometry)
    at = float(at)

    if at_values.size == 1:
        half = 0.5 * float(geometry.get("thickness", MIN_SEED_THICKNESS_NM))
        if not (at_values[0] - half <= at <= at_values[0] + half):
            return None
        centre, radii = radial_profile(
            levels[0]["polygon"], n_angles=n_angles, strict=strict)
        return _outline(centre, radii, n_angles)

    if at < at_values[0] or at > at_values[-1]:
        return None
    k = int(np.clip(np.searchsorted(at_values, at, side="right") - 1,
                    0, at_values.size - 2))
    span = at_values[k + 1] - at_values[k]
    w = 0.0 if span <= 0 else (at - at_values[k]) / span
    c0, r0 = radial_profile(levels[k]["polygon"], n_angles=n_angles, strict=strict)
    c1, r1 = radial_profile(levels[k + 1]["polygon"], n_angles=n_angles, strict=strict)
    return _outline((1.0 - w) * c0 + w * c1, (1.0 - w) * r0 + w * r1, n_angles)


def _outline(centre: np.ndarray, radii: np.ndarray, n_angles: int) -> np.ndarray:
    angles = np.arange(n_angles, dtype=float) * (2.0 * np.pi / n_angles)
    return np.column_stack([centre[0] + radii * np.cos(angles),
                            centre[1] + radii * np.sin(angles)])


# ------------------------------------------------------------------ membership
def _as_interval(value) -> tuple[float, float]:
    lo, hi = float(value[0]), float(value[1])
    return (lo, hi) if lo <= hi else (hi, lo)


def _cuboid_mask(xyz: np.ndarray, g: dict) -> np.ndarray:
    out = np.ones(xyz.shape[0], dtype=bool)
    for name, col in AXIS_INDEX.items():
        span = g.get(name.lower())
        if span is None:
            continue
        lo, hi = _as_interval(span)
        out &= (xyz[:, col] >= lo) & (xyz[:, col] <= hi)
    return out


def _sphere_mask(xyz: np.ndarray, g: dict) -> np.ndarray:
    centre = np.asarray(g.get("center", (0.0, 0.0, 0.0)), dtype=float).ravel()[:3]
    radii = np.asarray(g.get("radii", (0.0, 0.0, 0.0)), dtype=float).ravel()[:3]
    if centre.size < 3 or radii.size < 3 or not np.all(radii > 0):
        return np.zeros(xyz.shape[0], dtype=bool)
    d = (xyz - centre) / radii
    return np.einsum("ij,ij->i", d, d) <= 1.0


def _polyhedron_mask(xyz: np.ndarray, g: dict, *, n_angles: int, strict: bool) -> np.ndarray:
    axis = str(g.get("axis", "Z")).upper()
    stack_col = AXIS_INDEX[axis]
    ui, vi = cross_axes(axis)
    at_values, levels = _levels_sorted(g)
    n = xyz.shape[0]
    out = np.zeros(n, dtype=bool)

    if at_values.size == 1:
        half = 0.5 * float(g.get("thickness", MIN_SEED_THICKNESS_NM))
        inside_slab = ((xyz[:, stack_col] >= at_values[0] - half)
                       & (xyz[:, stack_col] <= at_values[0] + half))
        if not inside_slab.any():
            return out
        poly = np.asarray(levels[0]["polygon"], dtype=float)[:, :2]
        out[inside_slab] = _point_in_polygon(
            xyz[inside_slab, ui], xyz[inside_slab, vi], poly)
        return out

    # Radial form once per level, then one vectorised test for every point:
    # interpolate the centroid and the radius profile at the point's own
    # stacking coordinate, and compare its own radius against the boundary.
    profiles = [radial_profile(lv["polygon"], n_angles=n_angles, strict=strict)
                for lv in levels]
    centres = np.array([c for c, _ in profiles], dtype=float)          # (L, 2)
    radii = np.array([r for _, r in profiles], dtype=float)            # (L, A)

    t = xyz[:, stack_col]
    inside_range = (t >= at_values[0]) & (t <= at_values[-1]) & np.isfinite(t)
    if not inside_range.any():
        return out

    idx = np.flatnonzero(inside_range)
    k = np.clip(np.searchsorted(at_values, t[idx], side="right") - 1,
                0, at_values.size - 2)
    span = at_values[k + 1] - at_values[k]
    w = np.where(span > 0, (t[idx] - at_values[k]) / np.where(span > 0, span, 1.0), 0.0)

    c = centres[k] * (1.0 - w)[:, None] + centres[k + 1] * w[:, None]   # (P, 2)
    r = radii[k] * (1.0 - w)[:, None] + radii[k + 1] * w[:, None]       # (P, A)

    du = xyz[idx, ui] - c[:, 0]
    dv = xyz[idx, vi] - c[:, 1]
    point_r = np.hypot(du, dv)
    ang = np.mod(np.arctan2(dv, du), 2.0 * np.pi)

    n_angles_actual = r.shape[1]
    step = 2.0 * np.pi / n_angles_actual
    j0 = np.floor(ang / step).astype(int) % n_angles_actual
    j1 = (j0 + 1) % n_angles_actual
    frac = ang / step - np.floor(ang / step)
    rows = np.arange(idx.size)
    boundary = r[rows, j0] * (1.0 - frac) + r[rows, j1] * frac

    out[idx] = point_r <= boundary
    return out


def roi_volume_mask(
    x, y, z, record, *, base_mask=None,
    n_angles: int = DEFAULT_ANGLES, strict: bool = True,
) -> np.ndarray:
    """Boolean mask over ``(x, y, z)`` selecting the rows inside a volume ROI.

    ⚠ Raises for a 2-D type. The 2-D counterpart ``roi_region_mask`` returns
    all-False for a shape it does not know, which is right there (a line
    encloses nothing) and would be a trap here — a volume ROI silently selecting
    nothing looks exactly like a correct empty selection.
    """
    kind = getattr(record, "type", None)
    if kind not in VOLUME_ROI_TYPES:
        raise ValueError(
            f"{kind!r} is not a volume ROI; use roi_region_mask for 2-D shapes")

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    n = min(x.size, y.size, z.size)
    if n == 0:
        return np.zeros(0, dtype=bool)
    xyz = np.column_stack([x[:n], y[:n], z[:n]])
    finite = np.all(np.isfinite(xyz), axis=1)

    g = getattr(record, "geometry", None) or {}
    out = np.zeros(n, dtype=bool)
    if finite.any():
        sub = xyz[finite]
        if kind == "cuboid":
            inside = _cuboid_mask(sub, g)
        elif kind == "sphere":
            inside = _sphere_mask(sub, g)
        else:
            inside = _polyhedron_mask(sub, g, n_angles=n_angles, strict=strict)
        out[finite] = inside

    if base_mask is not None:
        base = np.asarray(base_mask, dtype=bool).ravel()
        if base.size >= n:
            out &= base[:n]
    return out


# --------------------------------------------------------- bounds & silhouette
def volume_bounds(record) -> tuple[tuple[float, float], ...] | None:
    """``((x0, x1), (y0, y1), (z0, z1))`` for a volume ROI, or ``None``."""
    kind = getattr(record, "type", None)
    g = getattr(record, "geometry", None) or {}
    if kind == "cuboid":
        try:
            return tuple(_as_interval(g[name]) for name in ("x", "y", "z"))
        except (KeyError, TypeError, IndexError):
            return None
    if kind == "sphere":
        centre = np.asarray(g.get("center", ()), dtype=float).ravel()
        radii = np.asarray(g.get("radii", ()), dtype=float).ravel()
        if centre.size < 3 or radii.size < 3:
            return None
        return tuple((float(centre[i] - radii[i]), float(centre[i] + radii[i]))
                     for i in range(3))
    if kind == "polyhedron":
        try:
            at_values, levels = _levels_sorted(g)
        except ValueError:
            return None
        axis = str(g.get("axis", "Z")).upper()
        stack_col = AXIS_INDEX[axis]
        ui, vi = cross_axes(axis)
        polys = [np.asarray(lv["polygon"], dtype=float)[:, :2] for lv in levels]
        allp = np.vstack(polys)
        spans = [None, None, None]
        spans[ui] = (float(allp[:, 0].min()), float(allp[:, 0].max()))
        spans[vi] = (float(allp[:, 1].min()), float(allp[:, 1].max()))
        if at_values.size == 1:
            half = 0.5 * float(g.get("thickness", MIN_SEED_THICKNESS_NM))
            spans[stack_col] = (float(at_values[0] - half), float(at_values[0] + half))
        else:
            spans[stack_col] = (float(at_values[0]), float(at_values[-1]))
        return tuple(spans)
    return None


def volume_silhouette(record, h_axis: int, v_axis: int,
                      *, n_angles: int = DEFAULT_ANGLES) -> np.ndarray | None:
    """Outline of a volume ROI as seen in a view spanning two data axes.

    ``h_axis`` / ``v_axis`` are **data-axis columns** (0=X, 1=Y, 2=Z), so a view
    states the axes it shows and never a plane name — the caller cannot pick the
    wrong convention because there is no convention to pick.

    Returns an ``(M, 2)`` closed outline in ``(h, v)`` order, or ``None``.
    """
    kind = getattr(record, "type", None)
    g = getattr(record, "geometry", None) or {}

    if kind in ("cuboid", "sphere"):
        bounds = volume_bounds(record)
        if bounds is None:
            return None
        (h0, h1), (v0, v1) = bounds[h_axis], bounds[v_axis]
        if kind == "cuboid":
            return np.array([[h0, v0], [h1, v0], [h1, v1], [h0, v1]], dtype=float)
        # An axis-aligned ellipsoid's silhouette is exactly the ellipse of that
        # plane's two radii — no projection maths needed.
        ang = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
        return np.column_stack([
            0.5 * (h0 + h1) + 0.5 * (h1 - h0) * np.cos(ang),
            0.5 * (v0 + v1) + 0.5 * (v1 - v0) * np.sin(ang),
        ])

    if kind != "polyhedron":
        return None

    axis = str(g.get("axis", "Z")).upper()
    stack_col = AXIS_INDEX[axis]
    ui, vi = cross_axes(axis)
    at_values, levels = _levels_sorted(g)
    polys = [np.asarray(lv["polygon"], dtype=float)[:, :2] for lv in levels]

    if stack_col not in (h_axis, v_axis):
        # Looking down the stack: the silhouette is the union of the
        # cross-sections, which for star-shaped outlines is the per-angle max
        # radius about the mean centroid.
        profiles = [radial_profile(p, n_angles=n_angles, strict=False) for p in polys]
        centre = np.mean([c for c, _ in profiles], axis=0)
        radii = np.max([r + np.hypot(*(c - centre)) for c, r in profiles], axis=0)
        outline = _outline(centre, radii, n_angles)
        return outline if (ui, vi) == (h_axis, v_axis) else outline[:, ::-1]

    # Looking across the stack: each level contributes its extent on the
    # in-plane axis that is visible, giving a band from one side and back.
    other = vi if stack_col == h_axis else ui
    col = 0 if other == ui else 1
    lo = np.array([p[:, col].min() for p in polys], dtype=float)
    hi = np.array([p[:, col].max() for p in polys], dtype=float)
    if at_values.size == 1:
        half = 0.5 * float(g.get("thickness", MIN_SEED_THICKNESS_NM))
        at_values = np.array([at_values[0] - half, at_values[0] + half])
        lo, hi = np.repeat(lo, 2), np.repeat(hi, 2)
    side_a = np.column_stack([hi, at_values])
    side_b = np.column_stack([lo[::-1], at_values[::-1]])
    outline = np.vstack([side_a, side_b])                 # (other, stack) order
    return outline if other == h_axis else outline[:, ::-1]


def plane_in_plane_axes(plane: str) -> tuple[int, int]:
    """The two data-axis columns a standalone projection shows.

    ⚠ Identical to ``cross_axes(PLANE_NORMAL_AXIS[plane])`` by construction, and
    the test asserts it: a projection shows everything except its normal axis,
    in ascending order. That is what lets a 2-D drawing be lifted into a volume
    without anyone choosing a convention -- the axes fall out of the plane.
    """
    return cross_axes(PLANE_NORMAL_AXIS[str(plane).upper()])


def volume_from_flat(flat_type: str, geometry: dict, plane: str,
                     interval: tuple[float, float]) -> tuple[str, dict]:
    """Lift a 2-D shape drawn in *plane* into a volume record.

    *interval* is the extent on the axis normal to *plane* -- what
    :func:`seed_interval` derived from the data under the drawing. Returns
    ``(volume_type, geometry)`` in named data axes, so nothing downstream has to
    know which view drew it.

    The in-plane shape is carried across unchanged: a rectangle becomes the two
    in-plane sides of a cuboid, an oval the two in-plane radii of an ellipsoid,
    a polygon the cross-section of a prism. Only the third axis is new.
    """
    volume_type = FLAT_TO_VOLUME.get(flat_type)
    if volume_type is None:
        raise ValueError(f"{flat_type!r} has no volume counterpart")
    plane = str(plane).upper()
    if plane not in PLANE_NORMAL_AXIS:
        raise ValueError(f"unknown plane {plane!r}; expected one of {sorted(PLANE_NORMAL_AXIS)}")
    normal = PLANE_NORMAL_AXIS[plane]
    ui, vi = plane_in_plane_axes(plane)
    lo, hi = (float(interval[0]), float(interval[1]))
    if hi < lo:
        lo, hi = hi, lo
    g = geometry or {}

    if volume_type == "cuboid":
        x, y, w, h = (float(v) for v in (g.get("bounds") or [0.0, 0.0, 0.0, 0.0]))
        spans = [None, None, None]
        spans[ui] = [x, x + w]
        spans[vi] = [y, y + h]
        spans[AXIS_INDEX[normal]] = [lo, hi]
        return volume_type, {name.lower(): spans[AXIS_INDEX[name]] for name in AXIS_NAMES}

    if volume_type == "sphere":
        x, y, w, h = (float(v) for v in (g.get("bounds") or [0.0, 0.0, 0.0, 0.0]))
        centre = [0.0, 0.0, 0.0]
        radii = [0.0, 0.0, 0.0]
        centre[ui], radii[ui] = x + 0.5 * w, 0.5 * abs(w)
        centre[vi], radii[vi] = y + 0.5 * h, 0.5 * abs(h)
        k = AXIS_INDEX[normal]
        centre[k], radii[k] = 0.5 * (lo + hi), 0.5 * abs(hi - lo)
        return volume_type, {"center": centre, "radii": radii}

    # polyhedron: a prism -- one cross-section plus a thickness. The polygon is
    # already in (ui, vi) order, which cross_axes(normal) reproduces exactly.
    points = [[float(pt[0]), float(pt[1])] for pt in (g.get("points") or [])]
    return volume_type, {
        "axis": normal,
        "thickness": abs(hi - lo),
        "levels": [{"at": 0.5 * (lo + hi), "polygon": points}],
    }


def scale_z(record, factor: float):
    """Rescale a ROI's Z about **z = 0**, returning new geometry (or ``None``).

    ⚠ About the origin, never about the ROI's own centre. The data rescales as
    ``z_calibrated = loc_z * 1e9 * cali.z_scaling_factor`` -- about zero -- so
    centre-anchored scaling would keep the ROI's thickness plausible and leave
    every off-centre ROI in the wrong place. It reads as the natural
    implementation and is correct only for a dataset centred near z = 0, which
    is exactly the case a test on synthetic data would have.

    A volume ROI's Z is captured when it is drawn and deliberately does **not**
    follow the Z scaling factor on its own; this is the explicit command that
    brings one back into step.
    """
    factor = float(factor)
    kind = getattr(record, "type", None)
    g = dict(getattr(record, "geometry", None) or {})
    if not np.isfinite(factor) or factor == 0.0:
        return None

    if kind == "cuboid":
        if "z" not in g:
            return None
        lo, hi = _as_interval(g["z"])
        g["z"] = [lo * factor, hi * factor]
        return g
    if kind == "sphere":
        centre = list(g.get("center") or [])
        radii = list(g.get("radii") or [])
        if len(centre) < 3 or len(radii) < 3:
            return None
        centre[2] = float(centre[2]) * factor
        radii[2] = abs(float(radii[2]) * factor)
        g["center"], g["radii"] = centre, radii
        return g
    if kind == "polyhedron":
        axis = str(g.get("axis", "Z")).upper()
        levels = [dict(lv) for lv in (g.get("levels") or [])]
        if not levels:
            return None
        if axis == "Z":
            for lv in levels:
                lv["at"] = float(lv["at"]) * factor
            if "thickness" in g:
                g["thickness"] = abs(float(g["thickness"]) * factor)
        else:
            # Z is one of the cross-section's own axes here, so it is a polygon
            # column rather than the stacking coordinate.
            column = 0 if cross_axes(axis)[0] == AXIS_INDEX["Z"] else 1
            for lv in levels:
                poly = [[float(v) for v in pt[:2]] for pt in lv.get("polygon") or []]
                for pt in poly:
                    pt[column] *= factor
                lv["polygon"] = poly
        g["levels"] = levels
        return g

    # 2-D types: only a point or a vertex list carries a Z to scale.
    if kind == "point":
        pt = list(g.get("point") or [])
        if len(pt) < 3:
            return None
        pt[2] = float(pt[2]) * factor
        g["point"] = pt
        return g
    pts = g.get("points")
    if pts and any(len(p) >= 3 for p in pts):
        g["points"] = [
            ([float(p[0]), float(p[1]), float(p[2]) * factor] if len(p) >= 3
             else [float(p[0]), float(p[1])])
            for p in pts
        ]
        return g
    return None


def translate_volume(record, deltas: dict) -> dict | None:
    """Shift a volume ROI along one or more axes; new geometry, or ``None``.

    *deltas* is keyed by **axis column** (0=X, 1=Y, 2=Z), so a view states the
    axes it moved the shape along and never a plane name -- the same rule the
    rest of this module follows.

    This is what makes a volume ROI draggable: a view can only move it on the
    two axes it shows, and the third must come through untouched rather than be
    recomputed from a projection that never saw it.
    """
    kind = getattr(record, "type", None)
    g = dict(getattr(record, "geometry", None) or {})
    moves = {int(k): float(v) for k, v in (deltas or {}).items()
             if np.isfinite(float(v))}
    if not moves or not any(abs(v) > 1e-12 for v in moves.values()):
        return None

    if kind == "cuboid":
        for column, delta in moves.items():
            name = AXIS_NAMES[column].lower()
            if name in g:
                lo, hi = _as_interval(g[name])
                g[name] = [lo + delta, hi + delta]
        return g

    if kind == "sphere":
        centre = list(g.get("center") or [])
        if len(centre) < 3:
            return None
        for column, delta in moves.items():
            centre[column] = float(centre[column]) + delta
        g["center"] = centre
        return g

    if kind == "polyhedron":
        axis = str(g.get("axis", "Z")).upper()
        stack = AXIS_INDEX[axis]
        ui, vi = cross_axes(axis)
        levels = [dict(lv) for lv in (g.get("levels") or [])]
        if not levels:
            return None
        for lv in levels:
            if stack in moves:
                lv["at"] = float(lv["at"]) + moves[stack]
            poly = [[float(pt[0]), float(pt[1])] for pt in lv.get("polygon") or []]
            if ui in moves or vi in moves:
                for pt in poly:
                    pt[0] += moves.get(ui, 0.0)
                    pt[1] += moves.get(vi, 0.0)
            lv["polygon"] = poly
        g["levels"] = levels
        return g
    return None
