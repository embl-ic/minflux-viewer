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
    cylinder    {"axis": "Z", "center": [cx, cy, cz],
                 "radii": [ru, rv], "height": h}
    polyhedron  {"representation": "projection_hull",
                 "projections": {"XY": {"points": ...}, "XZ": ..., "YZ": ...}}

``sphere`` is the user-facing name for an **axis-aligned ellipsoid** — a drag
almost never produces equal radii, and the name is kept because it is what a
user looks for.

Newly drawn polyhedra are the intersection of their three editable projection
polygons, stored in canonical XY=(X,Y), XZ=(X,Z), YZ=(Y,Z) order.  Legacy
contour-stack records remain supported. For those, ``axis`` is the stacking
axis and every level polygon uses the **remaining two axes in ascending order**:

    axis "Z" -> polygon vertices are (X, Y)
    axis "Y" -> polygon vertices are (X, Z)
    axis "X" -> polygon vertices are (Y, Z)

A legacy **prism is the degenerate case with one level** plus ``thickness``; it
can still upgrade to a multi-level contour stack with no migration.

Legacy interpolation between levels is **angular resampling** (star-shaped
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
    "PROJECTION_AXES",
    "cross_axes",
    "seed_interval",
    "radial_profile",
    "cross_section_at",
    "roi_volume_mask",
    "volume_bounds",
    "volume_mesh",
    "volume_silhouette",
    "NotStarShaped",
    "PLANE_NORMAL_AXIS",
    "FLAT_TO_VOLUME",
    "plane_in_plane_axes",
    "volume_from_flat",
    "scale_z",
    "translate_volume",
    "set_volume_extent",
    "volume_geometry_text",
    "projection_hull_from_flat",
    "projection_polygon",
    "set_projection_polygon",
    "add_cross_section",
    "convex_hull_polyhedron",
    "points_to_polyhedron",
    "MAX_POINT_LEVELS",
]


#: The data axis a standalone 2-D projection does NOT show.
PLANE_NORMAL_AXIS: dict[str, str] = {"XY": "Z", "XZ": "Y", "YZ": "X"}

#: Which volume type a 2-D drawing tool's shape becomes **by default**.
#: ⚠ An oval lifts to either a ``sphere`` or a ``cylinder`` -- the same drawn
#: ellipse, extruded rather than revolved -- so that choice cannot come from the
#: flat shape alone. ``volume_from_flat(..., volume_type=)`` names it explicitly;
#: this table is only the fallback for callers that have no tool in hand.
FLAT_TO_VOLUME: dict[str, str] = {
    "rectangle": "cuboid",
    "oval": "sphere",
    "polygon": "polyhedron",
    "freehand": "polyhedron",
}


#: Data-axis name -> column of an ``(N, 3)`` display-nm array.
AXIS_INDEX: dict[str, int] = {"X": 0, "Y": 1, "Z": 2}
AXIS_NAMES: tuple[str, str, str] = ("X", "Y", "Z")

#: Canonical coordinate order of the three ordinary projections.  A stored
#: projection-hull polygon always follows this table; a view may reverse it
#: (the orthogonal YZ pane shows Z horizontally and Y vertically), so the
#: public accessors below take actual axis columns and perform that reversal.
PROJECTION_AXES: dict[str, tuple[int, int]] = {
    "XY": (0, 1),
    "XZ": (0, 2),
    "YZ": (1, 2),
}

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
#: to measure. One constant for cuboid, sphere, cylinder and a projection-hull
#: fallback alike -- the derived dimension begins as an interval in each case.
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


def _projection_key(columns) -> tuple[str, bool] | None:
    """``(plane, reversed)`` for an actual pair of displayed axis columns.

    ``reversed`` says the view order is opposite to :data:`PROJECTION_AXES`.
    This is what keeps the orthogonal YZ pane's (Z, Y) screen convention out of
    the stored geometry, where YZ is always canonical (Y, Z).
    """
    try:
        columns = int(columns[0]), int(columns[1])
    except (TypeError, ValueError, IndexError):
        return None
    for plane, canonical in PROJECTION_AXES.items():
        if columns == canonical:
            return plane, False
        if columns == canonical[::-1]:
            return plane, True
    return None


def _projection_points(geometry: dict, plane: str) -> np.ndarray | None:
    entry = (geometry.get("projections") or {}).get(str(plane).upper())
    raw = entry.get("points") if isinstance(entry, dict) else entry
    try:
        points = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return None
    points = points[:, :2]
    return points if np.all(np.isfinite(points)) else None


def projection_polygon(geometry: dict, h_axis: int, v_axis: int) -> np.ndarray | None:
    """Editable projection constraint in a view's actual ``(h, v)`` order.

    ``None`` means *geometry* is not a projection-hull polyhedron or lacks that
    view.  The returned polygon is the user's constraint, deliberately not
    advertised as the exact silhouette of the intersection (which may be a
    subset after the other two constraints are applied).
    """
    if str(geometry.get("representation") or "") != "projection_hull":
        return None
    key = _projection_key((h_axis, v_axis))
    if key is None:
        return None
    plane, reverse = key
    points = _projection_points(geometry, plane)
    if points is None:
        return None
    return points[:, ::-1].copy() if reverse else points.copy()


def set_projection_polygon(record, columns, polygon) -> dict | None:
    """Replace one editable projection constraint of a projection-hull ROI.

    *polygon* is supplied in the calling view's screen-axis order.  Existing
    provenance is retained but changed to ``manual``: once the user edits an
    auto hull it must not silently refit itself.
    """
    if getattr(record, "type", None) != "polyhedron":
        return None
    geometry = dict(getattr(record, "geometry", None) or {})
    if str(geometry.get("representation") or "") != "projection_hull":
        return None
    key = _projection_key(columns)
    if key is None:
        return None
    plane, reverse = key
    try:
        points = np.asarray(polygon, dtype=float)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return None
    points = points[:, :2]
    if not np.all(np.isfinite(points)):
        return None
    if reverse:
        points = points[:, ::-1]
    projections = {
        str(name): (dict(value) if isinstance(value, dict)
                    else {"points": value})
        for name, value in (geometry.get("projections") or {}).items()
    }
    old = projections.get(plane, {})
    projections[plane] = {
        **old,
        "points": [[float(a), float(b)] for a, b in points],
        "source": "manual",
    }
    geometry["projections"] = projections
    return geometry


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


def _cylinder_parts(g: dict):
    """``(axis, stack_col, u, v, centre, radii, height)`` for a cylinder, or ``None``.

    The schema is named data axes, like every other volume type:
    ``{"axis": "Z", "center": [cx, cy, cz], "radii": [r_u, r_v], "height": h}``.
    ``radii`` is in ``cross_axes(axis)`` **ascending column order** -- the same
    rule a polyhedron's polygon columns follow -- and ``height`` is the full
    length along ``axis``, centred on the axis component of ``center``. The
    extent is therefore stored once (one fact, one place): there is deliberately
    no separate ``extent`` key that could disagree with the centre.
    """
    axis = str(g.get("axis", "Z")).upper()
    if axis not in AXIS_INDEX:
        return None
    centre = np.asarray(g.get("center", ()), dtype=float).ravel()
    radii = np.asarray(g.get("radii", ()), dtype=float).ravel()
    try:
        height = float(g.get("height"))
    except (TypeError, ValueError):
        return None
    if centre.size < 3 or radii.size < 2 or not np.all(np.isfinite(centre[:3])):
        return None
    if not np.all(np.isfinite(radii[:2])) or not np.all(radii[:2] > 0):
        return None
    if not np.isfinite(height) or height <= 0:
        return None
    u, v = cross_axes(axis)
    return axis, AXIS_INDEX[axis], u, v, centre[:3], radii[:2], height


def _cylinder_mask(xyz: np.ndarray, g: dict) -> np.ndarray:
    """Inside an axis-aligned elliptic cylinder: in the ellipse AND in the slab.

    Closed-form and fully vectorised, like the cuboid and the ellipsoid -- the
    cross-section is analytic, so unlike a polyhedron there is no radial profile
    to interpolate.
    """
    parts = _cylinder_parts(g)
    if parts is None:
        return np.zeros(xyz.shape[0], dtype=bool)
    _axis, stack_col, u, v, centre, radii, height = parts
    du = (xyz[:, u] - centre[u]) / radii[0]
    dv = (xyz[:, v] - centre[v]) / radii[1]
    in_ellipse = (du * du + dv * dv) <= 1.0
    in_slab = np.abs(xyz[:, stack_col] - centre[stack_col]) <= 0.5 * height
    return in_ellipse & in_slab


def _polyhedron_mask(xyz: np.ndarray, g: dict, *, n_angles: int, strict: bool) -> np.ndarray:
    if str(g.get("representation") or "") == "projection_hull":
        # A projection hull is the intersection of three generalized prisms:
        # every point must lie in the polygon made by its XY, XZ and YZ
        # projection.  This remains exact for concave polygons and needs no
        # voxel grid or reconstructed surface.
        out = np.ones(xyz.shape[0], dtype=bool)
        for plane, (u, v) in PROJECTION_AXES.items():
            polygon = _projection_points(g, plane)
            if polygon is None:
                return np.zeros(xyz.shape[0], dtype=bool)
            out &= _point_in_polygon(xyz[:, u], xyz[:, v], polygon)
            if not out.any():
                break
        return out

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
        elif kind == "cylinder":
            inside = _cylinder_mask(sub, g)
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
    if kind == "cylinder":
        parts = _cylinder_parts(g)
        if parts is None:
            return None
        _axis, stack_col, u, v, centre, radii, height = parts
        spans = [None, None, None]
        spans[u] = (float(centre[u] - radii[0]), float(centre[u] + radii[0]))
        spans[v] = (float(centre[v] - radii[1]), float(centre[v] + radii[1]))
        spans[stack_col] = (float(centre[stack_col] - 0.5 * height),
                            float(centre[stack_col] + 0.5 * height))
        return tuple(spans)
    if kind == "polyhedron":
        if str(g.get("representation") or "") == "projection_hull":
            # Each world axis occurs in two projection constraints.  Their
            # interval intersection is a conservative axis-aligned bound of
            # the visual hull and, unlike a union, cannot claim space excluded
            # by one of the user's other views.
            candidates: list[list[tuple[float, float]]] = [[], [], []]
            for plane, axes in PROJECTION_AXES.items():
                polygon = _projection_points(g, plane)
                if polygon is None:
                    return None
                for local, column in enumerate(axes):
                    candidates[column].append(
                        (float(polygon[:, local].min()),
                         float(polygon[:, local].max())))
            spans = []
            for values in candidates:
                lo = max(value[0] for value in values)
                hi = min(value[1] for value in values)
                if hi < lo:
                    return None
                spans.append((lo, hi))
            return tuple(spans)
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


def _clean_polygon(polygon: np.ndarray) -> np.ndarray:
    """Drop consecutive duplicate vertices and a repeated closing vertex."""
    points = np.asarray(polygon, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        return np.empty((0, 2), dtype=float)
    points = points[:, :2]
    if points.shape[0] > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    if points.shape[0] > 1:
        keep = np.ones(points.shape[0], dtype=bool)
        keep[1:] = np.any(np.abs(np.diff(points, axis=0)) > 1e-12, axis=1)
        points = points[keep]
    return points


def _convex_halfspaces(geometry: dict) -> np.ndarray | None:
    """Half-spaces of a convex projection hull, or ``None`` if one constraint
    is missing, degenerate or concave.

    Each 2-D edge becomes a 3-D plane whose coefficient on the projection's
    hidden axis is zero.  The intersection of all those generalized prisms is
    therefore the exact convex volume described by the three polygons.
    """
    rows = []
    for plane, axes in PROJECTION_AXES.items():
        polygon = _projection_points(geometry, plane)
        if polygon is None:
            return None
        polygon = _clean_polygon(polygon)
        if polygon.shape[0] < 3:
            return None
        x, y = polygon[:, 0], polygon[:, 1]
        area2 = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        if abs(area2) <= 1e-12:
            return None
        edge = np.roll(polygon, -1, axis=0) - polygon
        turns = edge[:, 0] * np.roll(edge[:, 1], -1) \
            - edge[:, 1] * np.roll(edge[:, 0], -1)
        significant = turns[np.abs(turns) > 1e-10]
        if significant.size and np.any(significant * significant[0] < 0.0):
            return None
        # For a CCW polygon, interior points lie left of every directed edge:
        # cross(edge, point-a) >= 0.  scipy uses A*x + b <= 0, hence the
        # coefficients below.  Reverse them for clockwise input.
        winding = 1.0 if area2 > 0.0 else -1.0
        u, v = axes
        for a, delta in zip(polygon, edge):
            if float(np.hypot(delta[0], delta[1])) <= 1e-12:
                continue
            row = np.zeros(4, dtype=float)
            row[u] = winding * delta[1]
            row[v] = winding * -delta[0]
            row[3] = winding * (delta[0] * a[1] - delta[1] * a[0])
            rows.append(row)
    return np.asarray(rows, dtype=float) if len(rows) >= 4 else None


def _projection_hull_mesh(geometry: dict):
    """Exact mesh of a convex projection hull.

    Arbitrary concave projection polygons still have exact point membership,
    but tessellating their Boolean prism intersection needs a separate CSG or
    voxel-resolution contract.  Returning ``None`` for that case keeps the 3-D
    display honest rather than silently convexifying a user's drawing.
    """
    halfspaces = _convex_halfspaces(geometry)
    if halfspaces is None:
        return None
    try:
        from scipy.optimize import linprog
        from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError

        normals = np.linalg.norm(halfspaces[:, :3], axis=1)
        # Chebyshev centre: maximize distance t from every plane.  A positive
        # t supplies the strictly interior point HalfspaceIntersection needs.
        result = linprog(
            np.array([0.0, 0.0, 0.0, -1.0]),
            A_ub=np.column_stack([halfspaces[:, :3], normals]),
            b_ub=-halfspaces[:, 3],
            bounds=[(None, None), (None, None), (None, None), (0.0, None)],
            method="highs",
        )
        if not result.success or result.x[3] <= 1e-8:
            return None
        vertices = np.asarray(
            HalfspaceIntersection(halfspaces, result.x[:3]).intersections,
            dtype=np.float64,
        )
        hull = ConvexHull(vertices)
    except (QhullError, ValueError, RuntimeError):
        return None

    faces = np.asarray(hull.simplices, dtype=np.uint32).copy()
    # scipy associates equations and simplices facet-for-facet.  Orient every
    # triangle toward its facet's outward normal for consistent GL shading.
    for index, face in enumerate(faces):
        p = vertices[face]
        if np.dot(np.cross(p[1] - p[0], p[2] - p[0]),
                  hull.equations[index, :3]) < 0.0:
            faces[index, 1], faces[index, 2] = faces[index, 2], faces[index, 1]
    # ConvexHull triangulates every planar facet.  Do not expose those internal
    # diagonals as wire-frame edges: they are a tessellation detail, not an edge
    # of the user's solid.  A real edge either joins non-coplanar triangles or
    # occurs at a boundary (the latter is defensive; a valid hull is closed).
    adjacency: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for a, b in ((face[0], face[1]),
                     (face[1], face[2]),
                     (face[2], face[0])):
            adjacency.setdefault(tuple(sorted((int(a), int(b)))), []).append(face_index)
    facet_normals = np.asarray(hull.equations[:, :3], dtype=float)
    edges = []
    for edge, neighbours in adjacency.items():
        if len(neighbours) != 2:
            edges.append(edge)
            continue
        first, second = facet_normals[neighbours]
        cosine = abs(float(np.dot(first, second)) / max(
            float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-15))
        if cosine < 1.0 - 1e-9:
            edges.append(edge)
    return vertices, faces, np.asarray(edges, dtype=np.uint32)


def volume_mesh(
    record,
    *,
    latitude_segments: int = 12,
    longitude_segments: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Surface geometry for a cuboid, axis-aligned ellipsoid or cylinder ROI.

    Returns ``(vertices, triangle_faces, wire_edges)`` in XYZ display-nm
    coordinates. Faces and edges contain integer indices into ``vertices``.
    Keeping this pure geometry here lets the OpenGL 3-D view and the rotating
    2-D projection display exactly the same object without either UI inventing
    its own interpretation of a volume record.

    A convex projection-hull ``polyhedron`` is returned exactly as the
    half-space intersection of its three projected polygons.  The legacy
    contour-stack polyhedron and a concave projection hull deliberately return
    ``None`` until they have a separately tested tessellation contract.
    """
    kind = getattr(record, "type", None)
    geometry = getattr(record, "geometry", None) or {}
    if (kind == "polyhedron"
            and str(geometry.get("representation") or "") == "projection_hull"):
        return _projection_hull_mesh(geometry)
    if kind == "cuboid":
        bounds = volume_bounds(record)
        if bounds is None:
            return None
        (x0, x1), (y0, y1), (z0, z1) = bounds
        vertices = np.array([
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ], dtype=np.float64)
        faces = np.array([
            [0, 2, 1], [0, 3, 2],       # -Z
            [4, 5, 6], [4, 6, 7],       # +Z
            [0, 1, 5], [0, 5, 4],       # -Y
            [3, 7, 6], [3, 6, 2],       # +Y
            [0, 4, 7], [0, 7, 3],       # -X
            [1, 2, 6], [1, 6, 5],       # +X
        ], dtype=np.uint32)
        edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ], dtype=np.uint32)
        return vertices, faces, edges

    if kind == "cylinder":
        parts = _cylinder_parts(geometry)
        if parts is None:
            return None
        _axis, stack_col, u, v, centre, radii, height = parts
        n_lon = max(3, int(longitude_segments))
        lo = float(centre[stack_col] - 0.5 * height)
        hi = float(centre[stack_col] + 0.5 * height)
        ang = np.arange(n_lon, dtype=float) * (2.0 * np.pi / n_lon)
        # Built in the ROI's own (u, v, axis) frame and scattered into XYZ
        # columns, so one code path serves a cylinder about X, Y or Z with no
        # per-axis branch -- the same reasoning that keeps the geometry in named
        # data axes rather than as a 2-D shape plus a plane name.
        rim = np.zeros((n_lon, 3), dtype=np.float64)
        rim[:, u] = centre[u] + radii[0] * np.cos(ang)
        rim[:, v] = centre[v] + radii[1] * np.sin(ang)
        bottom = rim.copy()
        bottom[:, stack_col] = lo
        top = rim.copy()
        top[:, stack_col] = hi
        cap_lo = np.zeros(3, dtype=np.float64)
        cap_hi = np.zeros(3, dtype=np.float64)
        cap_lo[u] = cap_hi[u] = centre[u]
        cap_lo[v] = cap_hi[v] = centre[v]
        cap_lo[stack_col], cap_hi[stack_col] = lo, hi
        vertices = np.vstack([cap_lo[None, :], cap_hi[None, :], bottom, top])
        base, apex = 2, 2 + n_lon              # first vertex index of each rim
        faces: list[list[int]] = []
        edges: list[list[int]] = []
        # A rim edge per segment reads as a circle; a vertical at every segment
        # reads as a solid band, so the wireframe keeps about eight of them.
        step = max(1, n_lon // 8)
        # Faces are wound so every normal points OUTWARD -- a translucent
        # GLMeshItem shades an inward normal visibly wrong.
        #
        # The (u, v, axis) frame is not always right-handed: cross_axes returns
        # the two cross axes in ascending order, so an axis of Y gives the frame
        # (X, Z, Y) whose u x v is -axis. Emitting one index order for every axis
        # would turn a Y-cylinder's surface inside out, so the handedness is
        # measured and the winding flipped with it.
        basis = np.eye(3)
        handed = float(np.dot(np.cross(basis[u], basis[v]), basis[stack_col]))

        def tri(a, b, c):
            return [a, c, b] if handed < 0.0 else [a, b, c]

        for i in range(n_lon):
            j = (i + 1) % n_lon
            faces.append(tri(0, base + j, base + i))          # lower cap, -axis
            faces.append(tri(1, apex + i, apex + j))          # upper cap, +axis
            faces.append(tri(base + i, base + j, apex + i))   # side, first half
            faces.append(tri(base + j, apex + j, apex + i))   # side, second half
            edges.append([base + i, base + j])                # lower rim
            edges.append([apex + i, apex + j])                # upper rim
            if i % step == 0:
                edges.append([base + i, apex + i])            # a few verticals
        return (vertices,
                np.asarray(faces, dtype=np.uint32),
                np.asarray(edges, dtype=np.uint32))

    if kind != "sphere":
        return None
    centre = np.asarray(geometry.get("center", ()), dtype=np.float64).ravel()
    radii = np.asarray(geometry.get("radii", ()), dtype=np.float64).ravel()
    if (
        centre.size < 3
        or radii.size < 3
        or not np.all(np.isfinite(centre[:3]))
        or not np.all(np.isfinite(radii[:3]))
        or not np.all(radii[:3] > 0)
    ):
        return None

    n_lat = max(3, int(latitude_segments))
    n_lon = max(3, int(longitude_segments))
    cx, cy, cz = centre[:3]
    rx, ry, rz = radii[:3]
    vertices_list = [[cx, cy, cz + rz]]
    for latitude in range(1, n_lat):
        theta = np.pi * latitude / n_lat
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        for longitude in range(n_lon):
            phi = 2.0 * np.pi * longitude / n_lon
            vertices_list.append([
                cx + rx * sin_theta * np.cos(phi),
                cy + ry * sin_theta * np.sin(phi),
                cz + rz * cos_theta,
            ])
    south = len(vertices_list)
    vertices_list.append([cx, cy, cz - rz])
    vertices = np.asarray(vertices_list, dtype=np.float64)

    def ring_index(ring: int, longitude: int) -> int:
        return 1 + ring * n_lon + (longitude % n_lon)

    faces_list: list[list[int]] = []
    edges_list: list[list[int]] = []
    for longitude in range(n_lon):
        nxt = (longitude + 1) % n_lon
        faces_list.append([0, ring_index(0, longitude), ring_index(0, nxt)])
        edges_list.append([0, ring_index(0, longitude)])
    for ring in range(n_lat - 1):
        for longitude in range(n_lon):
            edges_list.append([
                ring_index(ring, longitude),
                ring_index(ring, longitude + 1),
            ])
    for ring in range(n_lat - 2):
        for longitude in range(n_lon):
            a = ring_index(ring, longitude)
            b = ring_index(ring, longitude + 1)
            c = ring_index(ring + 1, longitude)
            d = ring_index(ring + 1, longitude + 1)
            faces_list.extend(([a, c, b], [b, c, d]))
            edges_list.append([a, c])
    last_ring = n_lat - 2
    for longitude in range(n_lon):
        nxt = (longitude + 1) % n_lon
        faces_list.append([
            south,
            ring_index(last_ring, nxt),
            ring_index(last_ring, longitude),
        ])
        edges_list.append([ring_index(last_ring, longitude), south])

    return (
        vertices,
        np.asarray(faces_list, dtype=np.uint32),
        np.asarray(edges_list, dtype=np.uint32),
    )


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

    if kind in ("cuboid", "sphere", "cylinder"):
        bounds = volume_bounds(record)
        if bounds is None:
            return None
        (h0, h1), (v0, v1) = bounds[h_axis], bounds[v_axis]
        rectangular = kind == "cuboid"
        if kind == "cylinder":
            # A cylinder is the one volume type whose silhouette KIND depends on
            # the view: the drawn ellipse looking down its axis, a rectangle from
            # either side. Both fall straight out of the bounds, so neither needs
            # projection maths -- and because the axis is named in the geometry,
            # no view has to know which plane drew it.
            rectangular = AXIS_INDEX.get(
                str(g.get("axis", "Z")).upper()) in (h_axis, v_axis)
        if rectangular:
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

    constraint = projection_polygon(g, h_axis, v_axis)
    if constraint is not None:
        # This is the editable constraint in that projection.  The actual
        # silhouette of the three-prism intersection may be smaller, but
        # replacing the user's primary polygon with it would violate the
        # projection-hull editing contract.
        return constraint

    axis = str(g.get("axis", "Z")).upper()
    stack_col = AXIS_INDEX[axis]
    ui, vi = cross_axes(axis)
    at_values, levels = _levels_sorted(g)
    polys = [np.asarray(lv["polygon"], dtype=float)[:, :2] for lv in levels]

    if stack_col not in (h_axis, v_axis):
        # Looking down the stack. ⚠ With ONE level there is nothing to union, and
        # the polygon the user drew is the answer -- resampling it radially
        # rounds off every corner of their own shape.
        if len(polys) == 1:
            outline = polys[0]
            return outline if (ui, vi) == (h_axis, v_axis) else outline[:, ::-1]
        # Several levels: the union, which for star-shaped outlines is the
        # per-angle max radius about the mean centroid.
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


def _projected_convex_hull(points: np.ndarray, margin: float) -> np.ndarray | None:
    """Convex hull of 2-D *points*, with a cheap approximate round buffer.

    Only the already-small hull vertex set is expanded, so the margin work does
    not scale with a large localization cloud.  Sixteen directions are ample
    here: the margin is a visual drawing allowance, not a measurement result.
    """
    try:
        from scipy.spatial import ConvexHull, QhullError

        points = np.asarray(points, dtype=float)
        points = points[np.all(np.isfinite(points[:, :2]), axis=1), :2]
        points = np.unique(points, axis=0)
        if points.shape[0] < 3 or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
            return None
        hull = points[ConvexHull(points).vertices]
        margin = max(0.0, float(margin))
        if margin > 0.0:
            angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
            offsets = margin * np.column_stack([np.cos(angles), np.sin(angles)])
            expanded = (hull[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
            hull = expanded[ConvexHull(expanded).vertices]
        return hull
    except (QhullError, ValueError, IndexError):
        return None


def _projection_fallback(axis_bounds: list[tuple[float, float]], plane: str) -> np.ndarray:
    u, v = PROJECTION_AXES[plane]
    u0, u1 = axis_bounds[u]
    v0, v1 = axis_bounds[v]
    return np.asarray([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], dtype=float)


def projection_hull_from_flat(
    geometry: dict,
    columns,
    interval: tuple[float, float],
    contained_points,
    *,
    min_points: int = MIN_SEED_LOCS,
) -> dict | None:
    """Build three editable projection constraints from one drawn polygon.

    The primary polygon is preserved exactly.  With enough contained 3-D data,
    each secondary projection is a buffered convex hull of those same rows.  No
    data and too few data deliberately share one path: both secondary views get
    the existing four-corner fallback, using the primary extent on their shared
    axis and the crosshair/visible-range *interval* on the hidden axis.

    The small inherited margin is the mean axis-aligned clearance between the
    primary polygon's bounding box and the contained data's bounding box.  It
    is intentionally cheap and merely keeps a generated boundary from visually
    sitting on top of the outermost localization.
    """
    key = _projection_key(columns)
    if key is None:
        return None
    primary_plane, reverse = key
    try:
        primary = np.asarray((geometry or {}).get("points"), dtype=float)
    except (TypeError, ValueError):
        return None
    if primary.ndim != 2 or primary.shape[0] < 3 or primary.shape[1] < 2:
        return None
    primary = primary[:, :2]
    if not np.all(np.isfinite(primary)):
        return None
    if reverse:
        primary = primary[:, ::-1]

    try:
        points = np.asarray(contained_points, dtype=float)
    except (TypeError, ValueError):
        points = np.empty((0, 3), dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        points = np.empty((0, 3), dtype=float)
    else:
        points = points[np.all(np.isfinite(points[:, :3]), axis=1), :3]

    primary_axes = PROJECTION_AXES[primary_plane]
    normal = ({0, 1, 2} - set(primary_axes)).pop()
    lo, hi = sorted((float(interval[0]), float(interval[1])))
    axis_bounds: list[tuple[float, float]] = [(0.0, 0.0)] * 3
    axis_bounds[primary_axes[0]] = (
        float(primary[:, 0].min()), float(primary[:, 0].max()))
    axis_bounds[primary_axes[1]] = (
        float(primary[:, 1].min()), float(primary[:, 1].max()))
    axis_bounds[normal] = (lo, hi)

    enough = points.shape[0] >= max(3, int(min_points))
    margin = 0.0
    if enough:
        projected = points[:, primary_axes]
        data_lo, data_hi = projected.min(axis=0), projected.max(axis=0)
        roi_lo, roi_hi = primary.min(axis=0), primary.max(axis=0)
        gaps = np.concatenate([data_lo - roi_lo, roi_hi - data_hi])
        margin = max(1.0, float(np.mean(np.maximum(gaps, 0.0))))

    projections: dict[str, dict] = {
        primary_plane: {
            "points": [[float(a), float(b)] for a, b in primary],
            "source": "manual",
        }
    }
    for plane, axes in PROJECTION_AXES.items():
        if plane == primary_plane:
            continue
        polygon = (_projected_convex_hull(points[:, axes], margin)
                   if enough else None)
        source = "auto_hull"
        if polygon is None:
            polygon = _projection_fallback(axis_bounds, plane)
            source = "fallback"
        projections[plane] = {
            "points": [[float(a), float(b)] for a, b in polygon],
            "source": source,
        }
    return {
        "representation": "projection_hull",
        "primary_plane": primary_plane,
        "margin_nm": margin,
        "projections": projections,
    }


def volume_from_flat(flat_type: str, geometry: dict, plane: str,
                     interval: tuple[float, float],
                     *, volume_type: str | None = None) -> tuple[str, dict]:
    """Lift a 2-D shape drawn in *plane* into a volume record.

    *interval* is the extent on the axis normal to *plane* -- what
    :func:`seed_interval` derived from the data under the drawing. Returns
    ``(volume_type, geometry)`` in named data axes, so nothing downstream has to
    know which view drew it.

    The in-plane shape is carried across unchanged: a rectangle becomes the two
    in-plane sides of a cuboid, an oval the two in-plane radii of an ellipsoid
    or cylinder, and the legacy polygon branch becomes a prism cross-section.
    New UI polyhedra use :func:`projection_hull_from_flat` instead.
    """
    volume_type = volume_type or FLAT_TO_VOLUME.get(flat_type)
    if volume_type is None:
        raise ValueError(f"{flat_type!r} has no volume counterpart")
    if volume_type not in VOLUME_ROI_TYPES:
        raise ValueError(f"{volume_type!r} is not a volume ROI type")
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

    if volume_type == "cylinder":
        # The same drawn ellipse as a sphere, extruded along the plane normal
        # instead of revolved: the two in-plane radii are kept and the third axis
        # becomes a length rather than a third radius.
        x, y, w, h = (float(v) for v in (g.get("bounds") or [0.0, 0.0, 0.0, 0.0]))
        centre = [0.0, 0.0, 0.0]
        centre[ui] = x + 0.5 * w
        centre[vi] = y + 0.5 * h
        centre[AXIS_INDEX[normal]] = 0.5 * (lo + hi)
        # ``radii`` is in cross_axes(normal) ascending order, and
        # plane_in_plane_axes returns exactly that pair, so (ui, vi) IS the order.
        return volume_type, {
            "axis": normal,
            "center": centre,
            "radii": [0.5 * abs(w), 0.5 * abs(h)],
            "height": abs(hi - lo),
        }

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
    if kind == "cylinder":
        parts = _cylinder_parts(g)
        if parts is None:
            return None
        _axis, stack_col, u, _v, centre, radii, height = parts
        centre = [float(c) for c in centre]
        radii = [float(r) for r in radii]
        zcol = AXIS_INDEX["Z"]
        centre[zcol] = centre[zcol] * factor
        if stack_col == zcol:
            height = abs(height * factor)
        else:
            # Z is one of the cross-section's own axes here, so it is a radius
            # rather than the extruded length.
            which = 0 if u == zcol else 1
            radii[which] = abs(radii[which] * factor)
        g["center"], g["radii"], g["height"] = centre, radii, height
        return g
    if kind == "polyhedron":
        if str(g.get("representation") or "") == "projection_hull":
            projections = {
                str(name): (dict(value) if isinstance(value, dict)
                            else {"points": value})
                for name, value in (g.get("projections") or {}).items()
            }
            for plane, axes in PROJECTION_AXES.items():
                entry = projections.get(plane)
                if not isinstance(entry, dict):
                    return None
                polygon = [[float(p[0]), float(p[1])]
                           for p in (entry.get("points") or [])]
                if len(polygon) < 3:
                    return None
                if AXIS_INDEX["Z"] in axes:
                    column = axes.index(AXIS_INDEX["Z"])
                    for point in polygon:
                        point[column] *= factor
                entry["points"] = polygon
            g["projections"] = projections
            if "margin_nm" in g:
                # A single inherited margin is only drawing flavour.  Retain a
                # sensible scalar under Z scaling rather than introducing a
                # false per-axis precision contract.
                g["margin_nm"] = abs(float(g["margin_nm"]) * factor)
            return g
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

    if kind in ("sphere", "cylinder"):
        # Both store a full 3-D ``center``, and a cylinder's extent is centred on
        # it, so moving the centre moves the whole shape.
        centre = list(g.get("center") or [])
        if len(centre) < 3:
            return None
        for column, delta in moves.items():
            centre[column] = float(centre[column]) + delta
        g["center"] = centre
        return g

    if kind == "polyhedron":
        if str(g.get("representation") or "") == "projection_hull":
            projections = {
                str(name): (dict(value) if isinstance(value, dict)
                            else {"points": value})
                for name, value in (g.get("projections") or {}).items()
            }
            for plane, axes in PROJECTION_AXES.items():
                entry = projections.get(plane)
                if not isinstance(entry, dict):
                    return None
                polygon = [[float(p[0]), float(p[1])]
                           for p in (entry.get("points") or [])]
                if len(polygon) < 3:
                    return None
                for local, column in enumerate(axes):
                    delta = moves.get(column, 0.0)
                    if delta:
                        for point in polygon:
                            point[local] += delta
                entry["points"] = polygon
            g["projections"] = projections
            return g
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


def set_volume_extent(record, columns, bounds) -> dict | None:
    """Resize a volume ROI on the two axes a view shows; new geometry or ``None``.

    *columns* are the data-axis columns the view is showing and *bounds* is the
    ``(x, y, w, h)`` its item now occupies. The axis the view does **not** show
    is left exactly as it was -- a projection has nothing to say about it, and
    recomputing it would let an XY resize silently change a Z extent.

    Only ``cuboid``, ``sphere`` and ``cylinder`` have an extent expressible this
    way; a polyhedron's cross-section is a polygon and is edited vertex-wise.
    """
    kind = getattr(record, "type", None)
    g = dict(getattr(record, "geometry", None) or {})
    try:
        h_axis, v_axis = int(columns[0]), int(columns[1])
        x, y, w, h = (float(v) for v in bounds)
    except Exception:
        return None
    if not all(np.isfinite(v) for v in (x, y, w, h)):
        return None
    spans = {h_axis: (x, x + w), v_axis: (y, y + h)}

    if kind == "cuboid":
        for column, (lo, hi) in spans.items():
            g[AXIS_NAMES[column].lower()] = [min(lo, hi), max(lo, hi)]
        return g
    if kind == "sphere":
        centre = list(g.get("center") or [])
        radii = list(g.get("radii") or [])
        if len(centre) < 3 or len(radii) < 3:
            return None
        for column, (lo, hi) in spans.items():
            centre[column] = 0.5 * (lo + hi)
            radii[column] = 0.5 * abs(hi - lo)
        g["center"], g["radii"] = centre, radii
        return g
    if kind == "cylinder":
        parts = _cylinder_parts(g)
        if parts is None:
            return None
        _axis, stack_col, u, v, centre, radii, height = parts
        centre = [float(c) for c in centre]
        radii = [float(r) for r in radii]
        for column, (lo, hi) in spans.items():
            centre[column] = 0.5 * (lo + hi)
            half = 0.5 * abs(hi - lo)
            if column == stack_col:
                height = 2.0 * half        # resized along the extrusion axis
            elif column == u:
                radii[0] = half
            elif column == v:
                radii[1] = half
        g["center"], g["radii"], g["height"] = centre, radii, height
        return g
    return None


def volume_geometry_text(record, fmt) -> str:
    """The geometry read-out for a volume ROI, in the caller's number format.

    Shared by both ROI property dialogs so they cannot drift. ⚠ It exists at
    all because the 2-D ``_bounds`` returns ``(0, 0, 0, 0)`` for a named-axis
    geometry, so both dialogs reported a volume ROI as *0 vertices, bbox
    X=0, Y=0, W=0, H=0* -- a wrong answer rather than a missing one.
    """
    spans = volume_bounds(record)
    if spans is None:
        return f"{getattr(record, 'type', '?')} (geometry not readable)"
    (x0, x1), (y0, y1), (z0, z1) = spans
    lines = [f"X={fmt(x0)}, Y={fmt(y0)}, Z={fmt(z0)}, "
             f"W={fmt(x1 - x0)}, H={fmt(y1 - y0)}, D={fmt(z1 - z0)}"]

    g = getattr(record, "geometry", None) or {}
    kind = getattr(record, "type", None)
    if kind == "sphere":
        centre = list(g.get("center") or [])
        radii = list(g.get("radii") or [])
        if len(centre) >= 3 and len(radii) >= 3:
            lines.append("centre=(" + ", ".join(fmt(c) for c in centre[:3]) + "), "
                         "radii=(" + ", ".join(fmt(r) for r in radii[:3]) + ")")
    elif kind == "cylinder":
        parts = _cylinder_parts(g)
        if parts is not None:
            axis, _sc, _u, _v, centre, radii, height = parts
            lines.append(
                f"axis={axis}, centre=(" + ", ".join(fmt(c) for c in centre) + "), "
                "radii=(" + ", ".join(fmt(r) for r in radii) + "), "
                f"height={fmt(height)}")
    elif kind == "polyhedron":
        if str(g.get("representation") or "") == "projection_hull":
            projections = g.get("projections") or {}
            sources = []
            for plane in PROJECTION_AXES:
                entry = projections.get(plane) or {}
                source = entry.get("source", "unknown") if isinstance(entry, dict) else "unknown"
                sources.append(f"{plane}={source}")
            lines.append("projection hull (" + ", ".join(sources) + ")")
            if g.get("margin_nm") is not None:
                lines.append(f"generated margin={fmt(g['margin_nm'])}")
            return "\n".join(lines)
        axis = str(g.get("axis", "Z")).upper()
        levels = list(g.get("levels") or [])
        at = ", ".join(fmt(lv.get("at", 0.0)) for lv in levels)
        noun = "cross-section" if len(levels) == 1 else "cross-sections"
        line = f"{len(levels)} {noun} along {axis} at {at}"
        if len(levels) == 1 and g.get("thickness") is not None:
            line += f", thickness={fmt(g['thickness'])}"
        lines.append(line)
    return "\n".join(lines)


def add_cross_section(record, at: float, polygon) -> dict | None:
    """Add (or replace) a polyhedron's cross-section at stacking coordinate *at*.

    This is what turns a prism into a genuine multi-slice polyhedron: draw the
    outline again at another Z and the shape between the two is interpolated by
    :func:`cross_section_at`, rather than being an extrusion of one outline.

    A second level supersedes ``thickness``: with one level the thickness *is*
    the extent, with several the outermost levels are. Re-adding at an existing
    level replaces it, so nudging a slice is an edit rather than an accumulation
    of near-identical cross-sections.
    """
    if getattr(record, "type", None) != "polyhedron":
        return None
    g = dict(getattr(record, "geometry", None) or {})
    # The new projection-hull representation is edited directly in XY/XZ/YZ.
    # Adding a legacy stack level to it would create geometry the mask ignores,
    # making the command appear to succeed while changing nothing.
    if str(g.get("representation") or "") == "projection_hull":
        return None
    points = [[float(p[0]), float(p[1])] for p in (polygon or [])]
    if len(points) < 3:
        return None
    at = float(at)
    levels = [dict(lv) for lv in (g.get("levels") or [])
              if abs(float(lv.get("at", 0.0)) - at) > 1e-9]
    levels.append({"at": at, "polygon": points})
    levels.sort(key=lambda lv: float(lv["at"]))
    g["levels"] = levels
    if len(levels) > 1:
        g.pop("thickness", None)
    return g


#: Levels a point cloud is bounded with at most. Each level costs a polygon in
#: the record and an interpolation in every mask call, and the shape between
#: levels is interpolated anyway, so more levels buy accuracy that the radial
#: description cannot express.
MAX_POINT_LEVELS = 24


def _radial_level(plane_pts: np.ndarray, n_angles: int, pad: float) -> list | None:
    """One star-shaped cross-section enclosing *plane_pts*.

    The per-angle furthest point about the centroid, so a concave outline stays
    concave in the radial direction -- tighter than a convex hull, which fills
    every notch.

    ⚠ Rays from the point-cloud mean do NOT make the result star-shaped about
    its own centroid, which is what ``radial_profile`` measures -- an early
    version claimed they did and raised ``NotStarShaped`` (ray 32 of 64 crossing
    three times) on the first concave cloud. Since ``roi_volume_mask`` defaults
    to ``strict=True``, such a level makes the finished ROI throw inside every
    consumer that asks it for a mask, so :func:`_star_shaped_level` validates
    this outline and degrades instead of trusting it.
    """
    if plane_pts.shape[0] == 0:
        return None
    centre = plane_pts.mean(axis=0)
    delta = plane_pts - centre
    radius = np.hypot(delta[:, 0], delta[:, 1])
    step = 2.0 * np.pi / int(n_angles)
    bins = (np.arctan2(delta[:, 1], delta[:, 0]) % (2.0 * np.pi) / step).astype(np.intp)
    bins = np.clip(bins, 0, int(n_angles) - 1)
    radii = np.zeros(int(n_angles), dtype=float)
    np.maximum.at(radii, bins, radius)

    filled = np.flatnonzero(radii > 0.0)
    if filled.size == 0:
        radii[:] = max(pad, 1e-6)                  # a single point still has extent
    elif filled.size < radii.size:
        # Empty angular bins are gaps in the sampling, not a boundary at the
        # centre; interpolate around the circle rather than collapsing inward.
        index = np.arange(radii.size, dtype=float)
        wrapped_i = np.concatenate([filled - radii.size, filled, filled + radii.size])
        wrapped_r = np.tile(radii[filled], 3)
        radii = np.interp(index, wrapped_i.astype(float), wrapped_r)
    radii = radii + max(0.0, float(pad))
    return [[float(x), float(y)] for x, y in _outline(centre, radii, int(n_angles))]


def _convex_level(plane_pts: np.ndarray, pad: float) -> list | None:
    """The level's convex hull, pushed out by *pad*.

    A convex polygon is star-shaped about every interior point, its own centroid
    included, so this always validates -- at the cost of filling this level's
    concavities.
    """
    if plane_pts.shape[0] < 3:
        return None
    try:
        from scipy.spatial import ConvexHull, QhullError
        hull = ConvexHull(plane_pts)
    except Exception:
        return None
    verts = plane_pts[hull.vertices]
    centre = verts.mean(axis=0)
    out = []
    for v in verts:
        delta = v - centre
        length = float(np.hypot(delta[0], delta[1]))
        if length > 0.0 and pad > 0.0:
            v = centre + delta * (1.0 + pad / length)
        out.append([float(v[0]), float(v[1])])
    return out


def _star_shaped_level(plane_pts: np.ndarray, n_angles: int, pad: float) -> list | None:
    """A cross-section for *plane_pts* that ``radial_profile`` will accept.

    Tried in order of tightness: the radial profile, then the convex hull, then a
    circle. Each candidate is VALIDATED rather than assumed -- the guarantee has
    to hold for the ROI to be usable at all, and only a convex outline or a
    circle holds it unconditionally.
    """
    outline = _radial_level(plane_pts, n_angles, pad)
    if outline is None:
        return None
    for candidate in (outline, _convex_level(plane_pts, pad)):
        if candidate is None or len(candidate) < 3:
            continue
        try:
            radial_profile(candidate, n_angles=n_angles, strict=True)
        except NotStarShaped:
            continue
        except ValueError:
            continue
        return candidate
    # A circle about the cloud's mean, which is star-shaped about its own centre.
    centre = plane_pts.mean(axis=0)
    delta = plane_pts - centre
    radius = float(np.hypot(delta[:, 0], delta[:, 1]).max()) + max(0.0, float(pad))
    radii = np.full(int(n_angles), max(radius, 1e-6), dtype=float)
    return [[float(x), float(y)] for x, y in _outline(centre, radii, int(n_angles))]


def points_to_polyhedron(points, *, axis: str = "Z", level_thickness: float | None = None,
                         max_levels: int = MAX_POINT_LEVELS,
                         n_angles: int = DEFAULT_ANGLES,
                         pad: float = 0.0) -> tuple[dict, float] | None:
    """A ``polyhedron`` bounding *points*, as one radial cross-section per level.

    Returns ``(geometry, recovered)``, where *recovered* is the fraction of
    *points* the reconstruction actually contains -- reported rather than
    assumed, because the radial description cannot express a hole or a spiral
    and the caller should be able to say how close the shape is.

    This is the sibling of :func:`convex_hull_polyhedron`: same stack-of-levels
    representation, but each level follows the cloud radially instead of taking
    its convex hull, so a C-shaped selection does not come back filled in.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return None
    pts = pts[np.all(np.isfinite(pts[:, :3]), axis=1)]
    if pts.shape[0] == 0:
        return None
    key = str(axis).upper()
    if key not in AXIS_INDEX:
        return None
    stack = AXIS_INDEX[key]
    ui, vi = cross_axes(key)

    lo, hi = float(pts[:, stack].min()), float(pts[:, stack].max())
    span = hi - lo
    thickness = (float(level_thickness) if level_thickness
                 else max(span / 8.0, MIN_SEED_THICKNESS_NM))
    if not np.isfinite(thickness) or thickness <= 0.0:
        thickness = MIN_SEED_THICKNESS_NM
    n_levels = int(np.clip(int(np.ceil(span / thickness)) + 1, 1, int(max_levels)))

    levels: list[dict] = []
    if n_levels <= 1 or span <= 0.0:
        # A single cross-section plus a thickness -- the prism the polyhedron
        # model already understands, and the one case whose polygon is used
        # verbatim rather than radially re-sampled.
        polygon = _star_shaped_level(np.column_stack([pts[:, ui], pts[:, vi]]),
                                     n_angles, pad)
        if polygon is None:
            return None
        geometry = {
            "axis": key,
            "thickness": max(span, MIN_SEED_THICKNESS_NM),
            "levels": [{"at": 0.5 * (lo + hi), "polygon": polygon}],
        }
    else:
        positions = np.linspace(lo, hi, n_levels)
        half = 0.5 * (positions[1] - positions[0])
        for at in positions:
            near = pts[np.abs(pts[:, stack] - at) <= half * 1.5]
            polygon = _star_shaped_level(np.column_stack([near[:, ui], near[:, vi]]),
                                         n_angles, pad)
            if polygon is None:
                continue
            levels.append({"at": float(at), "polygon": polygon})
        if not levels:
            return None
        geometry = {"axis": key, "levels": levels}

    class _Rec:                                    # what the mask reader needs
        type = "polyhedron"

    probe = _Rec()
    probe.geometry = geometry
    inside = roi_volume_mask(pts[:, 0], pts[:, 1], pts[:, 2], probe,
                             n_angles=n_angles, strict=False)
    recovered = float(inside.mean()) if inside.size else 0.0
    return geometry, recovered


def convex_hull_polyhedron(points, *, axis: str = "Z", levels: int = 9) -> dict | None:
    """A ``polyhedron`` enclosing *points*, as cross-sections of their 3-D hull.

    ``scipy.spatial.ConvexHull`` gives the exact enclosing solid, but this
    application's volume ROI is a stack of cross-sections rather than a face
    list -- so the hull is *sampled*: at each of ``levels`` evenly spaced
    positions along *axis*, the convex hull of the points near that level
    becomes one cross-section, and the interpolation between them reproduces the
    solid.

    ⚠ Sampling, not an exact face list, and it says so: the reconstruction is a
    little tighter than the true hull between levels. That is the price of one
    representation serving both a drawn prism and a fitted hull, and it keeps
    every consumer -- mask, silhouette, Scale Z, the editor -- working unchanged.
    """
    from scipy.spatial import ConvexHull, QhullError

    pts = np.asarray(points, dtype=float)
    pts = pts[np.all(np.isfinite(pts), axis=1)] if pts.ndim == 2 else pts
    if pts.ndim != 2 or pts.shape[1] < 3 or pts.shape[0] < 4:
        return None
    key = str(axis).upper()
    if key not in AXIS_INDEX:
        return None
    stack = AXIS_INDEX[key]
    ui, vi = cross_axes(key)

    lo, hi = float(pts[:, stack].min()), float(pts[:, stack].max())
    if hi - lo < MIN_SEED_THICKNESS_NM:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.5 * MIN_SEED_THICKNESS_NM, mid + 0.5 * MIN_SEED_THICKNESS_NM
    positions = np.linspace(lo, hi, max(2, int(levels)))
    half = 0.5 * (positions[1] - positions[0]) if len(positions) > 1 else 1.0

    out = []
    for at in positions:
        near = pts[np.abs(pts[:, stack] - at) <= half * 1.5]
        if near.shape[0] < 3:
            continue
        plane_pts = np.column_stack([near[:, ui], near[:, vi]])
        try:
            hull = ConvexHull(plane_pts)
        except (QhullError, ValueError):
            continue
        polygon = [[float(plane_pts[i, 0]), float(plane_pts[i, 1])]
                   for i in hull.vertices]
        if len(polygon) >= 3:
            out.append({"at": float(at), "polygon": polygon})
    if not out:
        return None
    if len(out) == 1:
        return {"axis": key, "thickness": float(hi - lo), "levels": out}
    return {"axis": key, "levels": out}
