"""Reset a dataset to its as-loaded state (data *and* view).

Reached from the Dataset Manager's right-click **Reset**.  The rule that makes
this well defined is the existing separation between the two dicts a dataset
carries:

* ``metadata`` records what the **import** decided — the transform the MSR
  channel alignment produced, the LUT the overlay import assigned, the RIMF
  history.  It is provenance and is never edited by the viewer.
* ``state`` holds **live, user-edited** view/filter preferences layered on top.

So "back to how it was opened" is: drop the live ``state`` layer and fall back
to what ``metadata`` recorded.  Overlay *membership* is deliberately kept — a
per-dataset reset must not silently dissolve a group of channels (that is what
*Close* and *Combine* are for); only the member's own transform/LUT revert.

Qt-free and side-effect-free apart from the dataset itself, so the UI just calls
:func:`reset_dataset` and then notifies.
"""

from __future__ import annotations

import numpy as np

from .roi_selection import ROI_MASKS_STATE_KEY

#: Live view-state keys that revert to the imported value in ``metadata`` when
#: one was recorded, and are otherwise dropped so the preference default applies.
_VIEW_KEYS_FROM_METADATA = (
    "overlay_transform",
    "render_transform_2d",
    "overlay_lut",
)

#: Live view-state keys with no import counterpart — always dropped.
_VIEW_KEYS_DROPPED = (
    "render_channel_lut",
    "overlay_default_hidden",
    "channel_transform",
    "invert",
    "size",
    "itr",
    "attribute",
    "aggregation",
    "channels",
)

#: Overlay grouping keys.  Reset keeps these: it is a per-dataset command and
#: must not break the other channels' view.
_MEMBERSHIP_KEYS = (
    "overlay_id", "render_group_id", "overlay_index", "overlay_order",
)


def loaded_rimf(ds) -> float | None:
    """The RIMF the dataset had when it was opened, or ``None`` if unknown.

    ``set_rimf`` appends to ``metadata["rimf_provenance"]["history"]``, so the
    **first** entry is the value the post-load chain established (auto estimate,
    fixed preference, or 1.0 for 2-D).  Everything after it is a later edit.
    """
    prov = (getattr(ds, "metadata", {}) or {}).get("rimf_provenance") or {}
    history = prov.get("history") or []
    if not history:
        return None
    try:
        return float(history[0]["value"])
    except (KeyError, TypeError, ValueError):
        return None


def clear_filters(ds) -> bool:
    """Drop every filter row and un-gate every localization."""
    had = bool(ds.state.get("filter_specs")) or not _mask_is_all(ds)
    ds.state["filter_specs"] = []
    try:
        ds.filter_mask = np.ones(int(ds.prop.num_loc), dtype=bool)
    except Exception:
        pass
    return had


def _mask_is_all(ds) -> bool:
    try:
        mask = np.asarray(ds.filter_mask, dtype=bool)
    except Exception:
        return True
    return bool(mask.size == 0 or mask.all())


def clear_roi_masks(ds) -> int:
    """Drop this dataset's cached ROI selection masks (the highlight overlays).

    The ROI *records* live in the shared ``RoiStore`` and are not touched — only
    the per-dataset masks they were evaluated into, so a reset dataset stops
    showing an in-ROI highlight it no longer has a selection for.
    """
    meta_by_id = ds.state.get(ROI_MASKS_STATE_KEY) or {}
    for meta in meta_by_id.values():
        key = meta.get("key") if isinstance(meta, dict) else None
        if key:
            ds.derived.pop(key, None)
    ds.state.pop(ROI_MASKS_STATE_KEY, None)
    ds.state.pop("active_roi_draft_id", None)
    return len(meta_by_id)


def reset_view_state(ds) -> bool:
    """Revert the live view layer to what the import recorded."""
    changed = False
    for key in _VIEW_KEYS_FROM_METADATA:
        imported = ds.metadata.get(key)
        current = ds.state.get(key)
        if imported is not None:
            if current is not imported:
                ds.state[key] = imported
                changed = True
        elif key in ds.state:
            ds.state.pop(key, None)
            changed = True
    for key in _VIEW_KEYS_DROPPED:
        if key in ds.state:
            ds.state.pop(key, None)
            changed = True
    return changed


def reset_dataset(ds) -> list[str]:
    """Restore *ds* to its as-loaded state; return what actually changed.

    The returned strings are user-facing phrases for the Log — an empty list
    means the dataset was already in its as-loaded state.
    """
    changes: list[str] = []

    n_specs = len(ds.state.get("filter_specs") or [])
    if clear_filters(ds):
        changes.append(f"{n_specs} filter(s) cleared" if n_specs else "filter mask cleared")

    n_rois = clear_roi_masks(ds)
    if n_rois:
        changes.append(f"{n_rois} ROI selection mask(s) dropped")

    rimf = loaded_rimf(ds)
    current = float(getattr(ds.cali, "RIMF", 1.0) or 1.0)
    if rimf is not None and abs(rimf - current) > 1e-12:
        ds.set_rimf(rimf, source="reset (as loaded)")
        changes.append(f"RIMF restored to {rimf:.4g}")

    if reset_view_state(ds):
        changes.append("view state (LUT / transform) restored")

    return changes
