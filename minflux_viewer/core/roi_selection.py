"""Derived ROI selection masks.

ROI geometry is viewer-facing state.  The helpers here turn geometry into
dataset-length masks and cache them as derived data without modifying raw
loaded attributes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .roi import RoiRecord

ROI_MASKS_STATE_KEY = "roi_masks"


def roi_mask_key(record: RoiRecord) -> str:
    """Stable short attribute-like key for one ROI mask."""
    return record.mask_key or f"roi_{record.id[:8]}"


def rectangle_bounds(record: RoiRecord) -> tuple[float, float, float, float] | None:
    """Return normalized (x0, x1, y0, y1) bounds for a rectangle ROI."""
    if record.type != "rectangle":
        return None
    try:
        x, y, w, h = record.geometry["bounds"]
    except Exception:
        return None
    x0, x1 = sorted((float(x), float(x) + float(w)))
    y0, y1 = sorted((float(y), float(y) + float(h)))
    return x0, x1, y0, y1


def rectangle_mask(
    x: np.ndarray,
    y: np.ndarray,
    record: RoiRecord,
    *,
    base_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return a boolean mask for points inside a rectangle ROI."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(x.size, y.size)
    out = np.zeros(n, dtype=bool)
    bounds = rectangle_bounds(record)
    if bounds is None or n == 0:
        return out
    x0, x1, y0, y1 = bounds
    out = np.isfinite(x[:n]) & np.isfinite(y[:n])
    angle = float(record.geometry.get("angle", 0.0) or 0.0)
    if abs(angle) > 1e-9:
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        half_w = max(0.5 * (x1 - x0), 0.0)
        half_h = max(0.5 * (y1 - y0), 0.0)
        theta = np.deg2rad(angle)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        dx = x[:n] - cx
        dy = y[:n] - cy
        local_x = cos_t * dx + sin_t * dy
        local_y = -sin_t * dx + cos_t * dy
        out &= (np.abs(local_x) <= half_w) & (np.abs(local_y) <= half_h)
    else:
        out &= (x[:n] >= x0) & (x[:n] <= x1) & (y[:n] >= y0) & (y[:n] <= y1)
    if base_mask is not None:
        base = np.asarray(base_mask, dtype=bool).ravel()
        if base.size == n:
            out &= base
    return out


def value_range_mask(
    values: np.ndarray,
    lo: float,
    hi: float,
    *,
    base_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return a boolean mask for values inside an inclusive numeric range."""
    values = np.asarray(values, dtype=float).ravel()
    lo, hi = sorted((float(lo), float(hi)))
    out = np.isfinite(values) & (values >= lo) & (values <= hi)
    if base_mask is not None:
        base = np.asarray(base_mask, dtype=bool).ravel()
        if base.size == out.size:
            out &= base
    return out


def store_roi_mask(
    dataset,
    record: RoiRecord,
    mask: np.ndarray,
    *,
    context: dict[str, Any] | None = None,
) -> str:
    """Cache an ROI mask on a dataset and mirror it in derived attributes."""
    arr = np.asarray(mask, dtype=bool).ravel()
    key = roi_mask_key(record)
    record.mask_key = key
    record.selected_count = int(np.count_nonzero(arr))
    record.selection_dirty = False
    record.context = dict(context or {})

    dataset.derived[key] = arr
    dataset.state.setdefault(ROI_MASKS_STATE_KEY, {})[record.id] = {
        "key": key,
        "name": record.name,
        "type": record.type,
        "stroke_color": record.stroke_color,
        "selected_count": record.selected_count,
        "context": record.context,
    }
    return key


def active_roi_mask(
    dataset,
    *,
    selected_ids: list[str] | tuple[str, ...] | set[str] | None = None,
    include_active_draft: bool = True,
) -> np.ndarray | None:
    """Return the union of currently active ROI masks for one dataset.

    Draft ROIs are stored under ``active_roi_draft_id`` while persistent ROIs
    are highlighted only when selected in the ROI Manager. The returned mask is
    aligned to the dataset's localization rows.
    """
    wanted = set(selected_ids or [])
    if include_active_draft:
        draft_id = dataset.state.get("active_roi_draft_id")
        if draft_id:
            wanted.add(str(draft_id))
    if not wanted:
        return None
    meta_by_id = dataset.state.get(ROI_MASKS_STATE_KEY, {})
    masks: list[np.ndarray] = []
    target_len = int(getattr(getattr(dataset, "prop", None), "num_loc", 0) or 0)
    for roi_id in wanted:
        meta = meta_by_id.get(roi_id)
        key = meta.get("key") if isinstance(meta, dict) else None
        if not key:
            continue
        arr = dataset.derived.get(key)
        if arr is None:
            continue
        mask = np.asarray(arr, dtype=bool).ravel()
        if target_len and mask.size != target_len:
            fixed = np.zeros(target_len, dtype=bool)
            fixed[: min(target_len, mask.size)] = mask[: min(target_len, mask.size)]
            mask = fixed
        masks.append(mask)
    if not masks:
        return None
    out = masks[0].copy()
    for mask in masks[1:]:
        if mask.size == out.size:
            out |= mask
    return out
