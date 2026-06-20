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
from .roi_selection import rectangle_bounds, rectangle_mask


@dataclass
class CropOptions:
    """User choices for an ROI duplicate/crop (see the backlog ROI-crop design)."""

    only_roi: bool = True               # master gate; off ⇒ plain duplicate
    clip: bool = False                  # True = per-loc clip; False = trace-complete
    channels: list[int] = field(default_factory=list)  # 1-based; empty = active only
    z_all: bool = True
    z_range: tuple[float, float] | None = None
    spatial_filter: bool = True         # Model A (gate); False = subset (Model B)
    stop_asking: bool = False


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
) -> np.ndarray:
    """Per-localization boolean mask (length ``num_loc``) for a rectangle ROI.

    ``z_range=(lo, hi)`` additionally bounds the depth (Z) axis in display nm;
    ``None`` means *all Z*. With ``trace_complete`` a whole trace is included iff
    its centroid (mean position) is inside — same O(N) cost as a per-loc test
    (both dominated by the tid group-by), and the preferred semantics.
    """
    coords = display_coords(ds)
    n = coords.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    if trace_complete:
        tid = _trace_ids(ds, n)
        _uniq, inv = np.unique(tid, return_inverse=True)
        cnt = np.bincount(inv).astype(float)
        cx = np.bincount(inv, weights=x) / cnt
        cy = np.bincount(inv, weights=y) / cnt
        trace_in = rectangle_mask(cx, cy, record)
        if z_range is not None:
            cz = np.bincount(inv, weights=z) / cnt
            lo, hi = sorted(float(v) for v in z_range)
            trace_in &= (cz >= lo) & (cz <= hi)
        return trace_in[inv]

    inside = rectangle_mask(x, y, record)
    if z_range is not None:
        lo, hi = sorted(float(v) for v in z_range)
        inside &= np.isfinite(z) & (z >= lo) & (z <= hi)
    return inside


def crop_is_axis_aligned(ds, record) -> bool:
    """True when the rect maps to plain xnm/ynm bounds (no rotation/transform).

    Only then can the crop be recorded as re-evaluable coordinate ``filter_specs``
    (Model A round-trip); otherwise the caller stores the computed mask directly.
    """
    angle = float((record.geometry or {}).get("angle", 0.0) or 0.0)
    if abs(angle) > 1e-9:
        return False
    transform = ds.state.get("overlay_transform") or ds.state.get("render_transform_2d")
    return not transform


def crop_filter_specs(
    record,
    *,
    trace_complete: bool = False,
    z_range: tuple[float, float] | None = None,
) -> list[dict]:
    """Re-evaluable coordinate filter specs for an axis-aligned rectangle crop.

    Caller must first check :func:`crop_is_axis_aligned`; for rotated/transformed
    rectangles return value is unusable and the mask should be stored instead.
    """
    bounds = rectangle_bounds(record)
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
