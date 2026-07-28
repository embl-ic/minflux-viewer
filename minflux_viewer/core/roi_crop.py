"""
minflux_viewer.core.roi_crop
============================
ROI duplicate / crop engine (Phase 1: rectangle ROIs).

Two crop models share one per-localization selection mask:

- **Model A** (ROI as spatial filter) — the duplicate keeps *all* data; the crop
  is applied as the dataset's ``filter_mask`` and (when expressible) recorded as
  re-evaluable coordinate ``filter_specs`` so it shows in the Filter dialog and
  round-trips through Save. Wiring lives in the UI; this module supplies the mask
  and the specs.
- **Model B** (real subset) — :func:`subset_dataset` builds a new dataset from
  only the in-ROI localizations (native coordinates, no baked-in RIMF/transform).

The selection mask itself ( :func:`compute_crop_mask` ) is computed in **display
nm** (``loc_nm`` + overlay transform), so it matches what the user drew, and
supports per-localization clipping or **trace-complete-by-centroid** selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .overlay import apply_display_transform_nm
from .roi_selection import rectangle_bounds, roi_region_mask


@dataclass
class CropOptions:
    """User choices for an ROI duplicate/crop (see BACKLOG ROI-crop design).

    A region ROI (rectangle/oval/polygon/freehand) always crops; ``exact_shape``
    chooses the ROI's exact outline vs its (axis-aligned) bounding box.
    """

    only_roi: bool = True               # retained for compat; region ROI ⇒ always crops
    exact_shape: bool = True            # True = exact ROI shape; False = bounding box
    clip: bool = False                  # True = per-loc clip; False = trace-complete
    channels: list[int] = field(default_factory=list)  # 1-based; empty = active only
    z_all: bool = True
    z_range: tuple[float, float] | None = None
    spatial_filter: bool = False        # False = subset (Model B); True = Model A (gate)
    stop_asking: bool = False


def _region_bbox(record) -> tuple[float, float, float, float] | None:
    """Axis-aligned bounding box (x0, x1, y0, y1) of any region ROI in display nm,
    honouring rotation (rect/oval rotated outline, polygon/freehand vertices)."""
    t = getattr(record, "type", None)
    if t in {"rectangle", "oval"}:
        from .roi_convert import region_outline_points
        pts = np.asarray(region_outline_points(record), dtype=float)
    else:
        from .roi import record_to_points
        pts = np.asarray(record_to_points(record), dtype=float)
        if pts.ndim == 2 and pts.shape[1] > 2:
            pts = pts[:, :2]
    if pts.size == 0:
        return None
    return (float(pts[:, 0].min()), float(pts[:, 0].max()),
            float(pts[:, 1].min()), float(pts[:, 1].max()))


def _bbox_mask(px: np.ndarray, py: np.ndarray, bbox) -> np.ndarray:
    if bbox is None:
        return np.zeros(px.shape[0], dtype=bool)
    x0, x1, y0, y1 = bbox
    return np.isfinite(px) & np.isfinite(py) & (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)


def parse_channel_spec(text: str, n_channels: int) -> list[int]:
    """Parse a channel field like ``1-3`` / ``1,3`` / ``1,2`` into sorted unique
    **1-based** channel indices clamped to ``[1, n_channels]``."""
    out: set[int] = set()
    for part in str(text or "").replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            try:
                lo, hi = int(a), int(b)
            except ValueError:
                continue
            for i in range(min(lo, hi), max(lo, hi) + 1):
                if 1 <= i <= n_channels:
                    out.add(i)
        else:
            try:
                i = int(part)
            except ValueError:
                continue
            if 1 <= i <= n_channels:
                out.add(i)
    return sorted(out)


def display_coords(ds) -> np.ndarray:
    """(N, 3) localization coordinates in **display nm** (loc_nm + overlay transform)."""
    try:
        loc = np.asarray(ds.loc_nm, dtype=float)
    except Exception:
        return np.empty((0, 3), dtype=float)
    if loc.ndim != 2 or loc.shape[1] < 2:
        return np.empty((0, 3), dtype=float)
    if loc.shape[1] == 2:
        loc = np.column_stack([loc, np.zeros(loc.shape[0], dtype=float)])
    transform = ds.state.get("overlay_transform") or ds.state.get("render_transform_2d")
    return apply_display_transform_nm(loc[:, :3], transform)


def display_xy_filtered(ds) -> np.ndarray:
    """(M, 2) finite **display-nm** XY with the filter mask applied.

    This is the coordinate frame ROIs live in (loc_nm + overlay transform), so
    detection tools that place ROIs on a dataset (NPC / convolution segmentation)
    must run here — using raw ``loc_nm`` puts ROIs at the wrong place for an
    overlay channel whose transform ≠ identity."""
    loc = display_coords(ds)
    if loc.ndim != 2 or loc.shape[0] == 0 or loc.shape[1] < 2:
        return np.empty((0, 2), dtype=float)
    mask = np.asarray(getattr(ds, "filter_mask", np.ones(loc.shape[0], bool)), dtype=bool)
    if mask.shape[0] == loc.shape[0]:
        loc = loc[mask]
    xy = loc[:, :2]
    return xy[np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])]


def display_xyz_filtered(ds) -> np.ndarray:
    """(M, 3) finite **display-nm** XYZ with the filter mask applied (3-D segmentation)."""
    loc = display_coords(ds)
    if loc.ndim != 2 or loc.shape[0] == 0 or loc.shape[1] < 3:
        return np.empty((0, 3), dtype=float)
    mask = np.asarray(getattr(ds, "filter_mask", np.ones(loc.shape[0], bool)), dtype=bool)
    if mask.shape[0] == loc.shape[0]:
        loc = loc[mask]
    xyz = loc[:, :3]
    return xyz[np.all(np.isfinite(xyz), axis=1)]


_PLANE_AXES = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}


def plane_localizations(state, channels, plane) -> np.ndarray:
    """``(M, 2)`` filtered **display-nm** localizations of the visible localization
    *channels* projected into *plane* (``XY``/``XZ``/``YZ``) — the same frame ROIs
    live in. Concatenates every visible non-image channel; used by the Plot Profile
    so the profile is decided by the data (not any render viewport)."""
    axes = _PLANE_AXES.get(plane)
    if axes is None:
        return np.zeros((0, 2))
    i, j = axes
    parts = []
    for ch in channels or []:
        if not ch.get("visible", True) or ch.get("kind") == "image":
            continue
        di = ch.get("dataset_idx")
        if di is None or not (0 <= di < len(state.datasets)):
            continue
        xyz = display_xyz_filtered(state.datasets[di])
        if xyz.shape[0]:
            parts.append(xyz[:, [i, j]])
    return np.concatenate(parts) if parts else np.zeros((0, 2))


def plane_localizations_version(state, channels, plane):
    """A cheap token that changes only when :func:`plane_localizations` would
    (dataset / filter mask / RIMF / channel visibility / plane) — **not** on
    zoom/pan — so a poller can skip re-fetching while idle. ``None`` off-plane."""
    if plane not in _PLANE_AXES:
        return None
    toks: list = [plane]
    for ch in channels or []:
        if ch.get("kind") == "image":
            continue
        di = ch.get("dataset_idx")
        if di is None or not (0 <= di < len(state.datasets)):
            continue
        ds = state.datasets[di]
        mask = getattr(ds, "filter_mask", None)
        mtok = None
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            mtok = (int(mask.sum()), int(mask.shape[0]))
        rimf = round(float(getattr(getattr(ds, "cali", None), "RIMF", 1.0) or 1.0), 6)
        toks.append((di, bool(ch.get("visible", True)), mtok, rimf))
    return tuple(toks)


def _trace_ids(ds, n: int) -> np.ndarray:
    from .loader import attr_values_1d
    tid = attr_values_1d(ds, "tid")
    if tid is None:
        return np.arange(n)
    tid = np.asarray(tid).ravel()
    return tid if tid.size == n else np.arange(n)


def compute_crop_mask(
    ds,
    record,
    *,
    z_range: tuple[float, float] | None = None,
    trace_complete: bool = False,
    exact_shape: bool = False,
) -> np.ndarray:
    """Per-localization boolean mask (length ``num_loc``) for a region ROI.

    ``exact_shape`` selects with the ROI's exact outline (rectangle/oval/polygon/
    freehand); otherwise (default) with its **axis-aligned bounding box** — so an
    oval/polygon/rotated-rect crops "as if the bounding box were a rectangle ROI".
    ``z_range=(lo, hi)`` additionally bounds the depth (Z) axis in display nm;
    ``None`` means *all Z*. With ``trace_complete`` a whole trace is included iff
    its centroid (mean position) is inside.
    """
    coords = display_coords(ds)
    n = coords.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    bbox = None if exact_shape else _region_bbox(record)

    def region_in(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        return roi_region_mask(px, py, record) if exact_shape else _bbox_mask(px, py, bbox)

    if trace_complete:
        tid = _trace_ids(ds, n)
        _uniq, inv = np.unique(tid, return_inverse=True)
        cnt = np.bincount(inv).astype(float)
        cx = np.bincount(inv, weights=x) / cnt
        cy = np.bincount(inv, weights=y) / cnt
        trace_in = region_in(cx, cy)
        if z_range is not None:
            cz = np.bincount(inv, weights=z) / cnt
            lo, hi = sorted(float(v) for v in z_range)
            trace_in &= (cz >= lo) & (cz <= hi)
        return trace_in[inv]

    inside = region_in(x, y)
    if z_range is not None:
        lo, hi = sorted(float(v) for v in z_range)
        inside &= np.isfinite(z) & (z >= lo) & (z <= hi)
    return inside


def crop_is_axis_aligned(ds, record, *, exact_shape: bool = False) -> bool:
    """True when the crop maps to plain xnm/ynm bounds (re-evaluable filter specs).

    A **bounding-box** crop is always axis-aligned (absent an overlay transform);
    an **exact-shape** crop only is for an unrotated rectangle. Otherwise the
    caller stores the computed mask directly (Model A ``crop_mask_only``).
    """
    transform = ds.state.get("overlay_transform") or ds.state.get("render_transform_2d")
    if transform:
        return False
    if not exact_shape:
        return True
    if getattr(record, "type", None) != "rectangle":
        return False
    angle = float((record.geometry or {}).get("angle", 0.0) or 0.0)
    return abs(angle) <= 1e-9


def crop_filter_specs(
    record,
    *,
    trace_complete: bool = False,
    z_range: tuple[float, float] | None = None,
    exact_shape: bool = False,
) -> list[dict]:
    """Re-evaluable coordinate filter specs for an axis-aligned crop.

    Caller must first check :func:`crop_is_axis_aligned`. Uses the ROI's exact
    rectangle bounds when ``exact_shape`` (unrotated rect), else its bounding box.
    """
    if exact_shape:
        bounds = rectangle_bounds(record)
    else:
        bb = _region_bbox(record)
        bounds = (bb[0], bb[1], bb[2], bb[3]) if bb else None
    if bounds is None:
        return []
    x0, x1, y0, y1 = bounds
    mode = "trace mean" if trace_complete else "per loc"
    specs = [
        {"attribute": "xnm", "mode": mode, "lo": float(x0), "hi": float(x1),
         "lo_inc": True, "hi_inc": True},
        {"attribute": "ynm", "mode": mode, "lo": float(y0), "hi": float(y1),
         "lo_inc": True, "hi_inc": True},
    ]
    if z_range is not None:
        lo, hi = sorted(float(v) for v in z_range)
        specs.append({"attribute": "znm", "mode": mode, "lo": lo, "hi": hi,
                      "lo_inc": True, "hi_inc": True})
    return specs


def subset_dataset(src, keep_mask, *, name: str, prefs: dict | None = None):
    """Build a new dataset from only the kept (in-ROI) localizations (Model B).

    Uses **native** coordinates (loc_x/y/z × 1e9, no RIMF/transform baked in), so
    the subset is a clean standalone dataset whose RIMF/dimensionality are
    re-derived like any fresh load.
    """
    from .dataset import build_localization_dataset
    from .loader import attr_values_1d

    keep = np.asarray(keep_mask, dtype=bool).ravel()
    n = int(getattr(src.prop, "num_loc", keep.size))
    if keep.size != n:
        fixed = np.zeros(n, dtype=bool)
        m = min(n, keep.size)
        fixed[:m] = keep[:m]
        keep = fixed

    def native_nm(attr: str):
        v = attr_values_1d(src, attr)
        return None if v is None else np.asarray(v, dtype=float).ravel()[keep] * 1.0e9

    def col(attr: str):
        v = attr_values_1d(src, attr)
        return None if v is None else np.asarray(v).ravel()[keep]

    x_nm = native_nm("loc_x")
    y_nm = native_nm("loc_y")
    z_nm = native_nm("loc_z")

    # Carry the other per-localization 1-D attributes through the crop.
    skip = {"loc_x", "loc_y", "loc_z", "xnm", "ynm", "znm", "tid", "tim", "ftr",
            "den", "local_density"}
    attrs: dict[str, np.ndarray] = {}
    for key in src.attr.keys():
        if key in skip:
            continue
        v = src.attr.get(key)
        if v is None:
            continue
        v = np.asarray(v)
        if v.ndim == 1 and v.size == n:
            attrs[key] = v[keep]

    new = build_localization_dataset(
        name=name,
        x_nm=x_nm if x_nm is not None else np.zeros(int(keep.sum())),
        y_nm=y_nm if y_nm is not None else np.zeros(int(keep.sum())),
        z_nm=z_nm,
        folder=getattr(src.file, "folder", ""),
        attrs=attrs or None,
        tid=col("tid"),
        tim=col("tim"),
        source_version=src.metadata.get("source_version", "imported"),
        prefs=prefs,
    )
    new.metadata["cropped_from_dataset"] = getattr(src.file, "name", "")
    return new
