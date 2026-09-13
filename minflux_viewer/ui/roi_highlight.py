"""The ROI data highlight — one rule for every view of a dataset.

A ROI's stored artefact is **one boolean per localization row** (``ds.derived``,
keyed through ``ds.state["roi_masks"]``), not a shape. Geometry only matters
inside the view that *computes* the mask; every view that *paints* it asks the
same question of each row: is this localization in the ROI? So "data in the
active ROI" means the same rows everywhere, and a view only has to decide where
to put them on its own axes.

That is why this module deliberately ignores :mod:`minflux_viewer.core.roi_scope`.
Scope decides where a ROI's *outline* may be drawn — a spatial rectangle has no
meaning on a histogram's attribute axis — while the highlight follows the rows.
A rectangle drawn in the Attribute Plot therefore lights up its localizations in
render and scatter, and a region drawn in render lights up its rows in the
Attribute Plot and the Histogram.

Every consumer here sees a 1-D boolean array, so a future 3-D ROI changes only
the mask *producer* (``roi_region_mask`` → a volume test) and nothing below.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from ..colors import rgba_hex, viewer_color
from ..core.roi_selection import (
    ROI_MASKS_STATE_KEY,
    active_roi_mask,
    roi_highlight_enabled,
)

__all__ = [
    "MAX_HIGHLIGHT_POINTS",
    "decimate",
    "highlight_brushes",
    "highlight_color",
    "highlight_masks",
    "highlight_rgba",
    "owns_active_draft",
    "roi_highlight_enabled",
    "union_mask",
]

#: Default ceiling on painted highlight points per view (a paint budget, not a
#: selection limit — the mask itself is never thinned).
MAX_HIGHLIGHT_POINTS = 100_000


def highlight_masks(state, ds, *, apply_filter: bool = True) -> list[tuple[object, np.ndarray]]:
    """``(record, mask)`` for every ROI that should highlight data on *ds*.

    The ROI-Manager selection plus the dataset's own active draft, each mask
    aligned to the dataset's localization rows.

    ``apply_filter`` intersects with the dataset's filter mask: right for a view
    that only ever draws filtered localizations (render, scatter), wrong for one
    that can show unfiltered rows (the Attribute Plot's *Filtered only* box),
    where it would hide highlights on points that are plainly on screen.
    """
    selected = set(getattr(state.rois, "selected_ids", []) or [])
    records = [record for record in state.rois.records if record.id in selected]
    draft_id = ds.state.get("active_roi_draft_id")
    if draft_id:
        draft_meta = (ds.state.get(ROI_MASKS_STATE_KEY) or {}).get(draft_id, {})
        draft_record = next((record for record in records if record.id == draft_id), None)
        if draft_record is None and isinstance(draft_meta, dict):
            draft_record = type("_RoiHighlight", (), {
                "id": draft_id,
                "stroke_color": draft_meta.get("stroke_color", "#ffff00"),
            })()
        if draft_record is not None and all(record.id != draft_id for record in records):
            records.append(draft_record)

    ftr = None
    if apply_filter:
        raw_filter = getattr(ds, "filter_mask", None)
        if raw_filter is not None:
            ftr = np.asarray(raw_filter, dtype=bool).ravel()

    out: list[tuple[object, np.ndarray]] = []
    for record in records:
        mask = active_roi_mask(ds, selected_ids=[record.id], include_active_draft=False)
        if mask is None and record.id == draft_id:
            mask = active_roi_mask(ds, selected_ids=[], include_active_draft=True)
        if mask is None:
            continue
        mask = np.asarray(mask, dtype=bool).ravel()
        if ftr is not None and ftr.size == mask.size:
            mask = mask & ftr
        out.append((record, mask))
    return out


def union_mask(pairs, total: int) -> np.ndarray | None:
    """OR of the masks in ``(record, mask)`` *pairs*, or ``None`` when there are none.

    For a view that cannot paint one ROI per mark (the Histogram's bars), the
    highlight is the union: every row inside any active ROI.
    """
    out: np.ndarray | None = None
    for _record, mask in pairs:
        arr = np.asarray(mask, dtype=bool).ravel()
        if arr.size != total:
            fixed = np.zeros(total, dtype=bool)
            keep = min(total, arr.size)
            fixed[:keep] = arr[:keep]
            arr = fixed
        out = arr.copy() if out is None else (out | arr)
    return out


def highlight_color(prefs):
    """COLOR ▸ ROI ▸ "highlight data in ROI" — one colour for every ROI.

    Trade-off kept from render/scatter: with several ROIs shown the highlighted
    points are no longer attributable to a particular one.
    """
    return pg.mkColor(rgba_hex(viewer_color(prefs, "roi_highlight")))


def highlight_brushes(prefs, count: int, *, alpha: int = 75) -> list:
    """One shared brush per point — shared, so pyqtgraph's symbol atlas stays small."""
    fill = highlight_color(prefs)
    fill.setAlpha(int(alpha))
    return [pg.mkBrush(fill)] * int(count)


def highlight_rgba(prefs, count: int, alpha: float = 0.95) -> np.ndarray:
    """``(count, 4)`` float RGBA for an OpenGL scatter."""
    color = highlight_color(prefs)
    rgba = np.array(
        [[color.redF(), color.greenF(), color.blueF(), float(alpha)]],
        dtype=np.float32,
    )
    return np.tile(rgba, (int(count), 1))


def decimate(mask, total: int, max_points: int) -> np.ndarray:
    """Row indices of *mask*, thinned by a deterministic stride to *max_points*."""
    arr = np.asarray(mask, dtype=bool).ravel()
    if arr.size != total:
        arr = np.ones(int(total), dtype=bool)
    indices = np.flatnonzero(arr)
    if max_points > 0 and indices.size > max_points:
        step = int(np.ceil(indices.size / max_points))
        indices = indices[::step]
    return indices


def owns_active_draft(controller) -> bool:
    """True when a ROI is being drawn in the view owning *controller*.

    Picks which preference gates the highlight: ``plot.roi_highlight_in_roi`` for
    the drawing view, ``plot.roi_sync_highlight`` for every other view.
    """
    if controller is None:
        return False
    try:
        return controller.current_record() is not None
    except Exception:
        return getattr(controller, "draft", None) is not None
