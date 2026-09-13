"""Per-localization channel labels from a selection.

``MainWindow.apply_channel_separation`` builds one dataset per channel from an
**exclusive label array** — ``0 .. N-1`` per channel, ``-1`` for unassigned — and
never looks at where those labels came from (it reads only each carrier's name
and LUT). This module is the other end of that contract: it turns selections the
viewer already holds into such an array, so *channel from ROI* and *channel from
filter* need no new machinery in the separation path.

Three sources, one output:

* :func:`labels_from_rois`          — one channel per ROI,
* :func:`labels_from_filter`        — in / out of the dataset's live filter,
* :func:`labels_from_filter_specs`  — one channel per saved filter preset.

Pure, and Qt-free at import time (the ROI helpers it reaches for are imported
lazily), so a label set can be built headlessly (scripting, tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: How a trace is assigned when only some of its localizations are selected.
#: ``"per loc"`` leaves localizations independent (a trace may split between
#: channels); the others keep every trace whole.
TRACE_RULES = ("per loc", "any", "majority", "all")


@dataclass
class LabelPlan:
    """An exclusive label array plus what it says about the selection."""

    labels: np.ndarray
    names: list[str]
    #: Localizations assigned to each channel, in channel order.
    counts: list[int]
    #: Rows claimed by more than one selection; they went to the earlier channel.
    overlaps: int
    #: Rows in no selection at all — ``apply_channel_separation`` puts these in
    #: its hidden "unassigned" channel.
    unassigned: int
    #: Selections that could not be evaluated, with the reason.
    skipped: list[str] = field(default_factory=list)

    @property
    def n_channels(self) -> int:
        return len(self.names)


def _as_mask(mask, n: int) -> np.ndarray:
    """*mask* as a length-*n* boolean array (padded or truncated)."""
    arr = np.asarray(mask, dtype=bool).ravel()
    if arr.size == n:
        return arr
    fixed = np.zeros(int(n), dtype=bool)
    keep = min(int(n), arr.size)
    fixed[:keep] = arr[:keep]
    return fixed


def expand_to_traces(mask, trace_idx, rule: str = "any") -> np.ndarray:
    """Grow a per-localization *mask* to whole traces under *rule*.

    A ROI selects localizations, but a trace can straddle its boundary. ``"any"``
    keeps a trace when at least one of its localizations is selected,
    ``"majority"`` when more than half are, ``"all"`` only when every one is.
    ``"per loc"`` returns the mask unchanged.
    """
    arr = np.asarray(mask, dtype=bool).ravel()
    if rule == "per loc":
        return arr
    if rule not in TRACE_RULES:
        raise ValueError(f"unsupported trace rule {rule!r}")
    ti = np.asarray(trace_idx, dtype=int)
    if ti.ndim != 2 or ti.shape[0] == 0 or ti.shape[1] < 2:
        return arr
    # Counting through one cumulative sum, so this does not scale with trace count.
    cum = np.concatenate([[0], np.cumsum(arr.astype(np.int64))])
    starts = np.clip(ti[:, 0], 0, arr.size)
    stops = np.clip(ti[:, 1] + 1, 0, arr.size)
    inside = cum[stops] - cum[starts]
    length = np.maximum(stops - starts, 0)
    if rule == "all":
        keep = (length > 0) & (inside == length)
    elif rule == "majority":
        keep = inside * 2 > length
    else:
        keep = inside > 0
    out = np.zeros(arr.size, dtype=bool)
    for index in np.flatnonzero(keep):
        out[starts[index]:stops[index]] = True
    return out


def labels_from_masks(
    masks,
    n: int,
    *,
    names=None,
    trace_idx=None,
    trace_rule: str = "per loc",
    skipped=None,
) -> LabelPlan:
    """Exclusive labels from per-localization *masks*, earlier channel winning.

    Labels have to be exclusive (one channel per localization), so overlapping
    selections need a precedence rule: the first mask that claims a row keeps it,
    which makes the channel order the user sees the order that decides. The
    number of contested rows is reported rather than hidden.
    """
    n = int(n)
    masks = [_as_mask(mask, n) for mask in masks]
    if trace_rule != "per loc" and trace_idx is not None:
        masks = [expand_to_traces(mask, trace_idx, trace_rule) for mask in masks]
    labels = np.full(n, -1, dtype=int)
    taken = np.zeros(n, dtype=bool)
    overlaps = 0
    counts: list[int] = []
    for index, mask in enumerate(masks):
        overlaps += int(np.count_nonzero(mask & taken))
        fresh = mask & ~taken
        labels[fresh] = index
        taken |= mask
        counts.append(int(np.count_nonzero(fresh)))
    if names is None:
        names = [f"channel {i + 1}" for i in range(len(masks))]
    return LabelPlan(
        labels=labels,
        names=list(names),
        counts=counts,
        overlaps=overlaps,
        unassigned=int(np.count_nonzero(labels < 0)),
        skipped=list(skipped or []),
    )


# ---------------------------------------------------------------------------
# From ROIs
# ---------------------------------------------------------------------------
def roi_mask_for_record(ds, record, *, exact_shape: bool = True) -> np.ndarray | None:
    """One ROI's per-localization mask, recomputing it when necessary.

    Prefers the mask the drawing view already stored. A ROI loaded from a set, or
    one whose geometry was edited since, carries ``selection_dirty`` and no usable
    mask, so it is recomputed from the dataset's display coordinates — the same
    frame ROIs live in — without needing an open window.

    Returns ``None`` for a ROI that encloses no area (line, point, angle).
    """
    from .roi_crop import compute_crop_mask
    from .roi_selection import REGION_ROI_TYPES, ROI_MASKS_STATE_KEY

    if str(getattr(record, "type", "")) not in REGION_ROI_TYPES:
        return None
    n = int(getattr(getattr(ds, "prop", None), "num_loc", 0) or 0)
    if n == 0:
        return None
    meta = (ds.state.get(ROI_MASKS_STATE_KEY) or {}).get(getattr(record, "id", ""), None)
    key = meta.get("key") if isinstance(meta, dict) else None
    stored = ds.derived.get(key) if key else None
    if stored is not None and not bool(getattr(record, "selection_dirty", False)):
        return _as_mask(stored, n)
    try:
        return _as_mask(compute_crop_mask(ds, record, exact_shape=exact_shape), n)
    except Exception:
        return None


def labels_from_rois(
    ds,
    records,
    *,
    exact_shape: bool = True,
    trace_rule: str = "per loc",
    names=None,
) -> LabelPlan:
    """One channel per region ROI, in the order given."""
    n = int(getattr(getattr(ds, "prop", None), "num_loc", 0) or 0)
    masks: list[np.ndarray] = []
    used_names: list[str] = []
    skipped: list[str] = []
    for index, record in enumerate(records or []):
        mask = roi_mask_for_record(ds, record, exact_shape=exact_shape)
        label = str(getattr(record, "name", "") or f"roi {index + 1}")
        if mask is None:
            skipped.append(f"{label}: encloses no area")
            continue
        if not mask.any():
            skipped.append(f"{label}: no localizations inside")
            continue
        masks.append(mask)
        used_names.append(label)
    return labels_from_masks(
        masks, n,
        names=list(names) if names is not None else used_names,
        trace_idx=getattr(getattr(ds, "prop", None), "trace_idx", None),
        trace_rule=trace_rule,
        skipped=skipped,
    )


# ---------------------------------------------------------------------------
# From filters
# ---------------------------------------------------------------------------
def labels_from_filter(ds, *, names=("in filter", "out of filter")) -> LabelPlan:
    """Two channels: the localizations passing the dataset's live filter, and the rest."""
    n = int(getattr(getattr(ds, "prop", None), "num_loc", 0) or 0)
    mask = _as_mask(getattr(ds, "filter_mask", np.ones(n, dtype=bool)), n)
    return labels_from_masks([mask, ~mask], n, names=list(names))


def mask_from_filter_specs(ds, specs) -> tuple[np.ndarray, list[str]]:
    """Evaluate a filter spec list against *ds* without touching its live filter.

    Each spec is evaluated at its own iteration, exactly as the Filter dialog
    applies it, so a preset means the same thing here as it does there.
    """
    from ..utils.filters import compute_filter_mask
    from .loader import attr_values_for_selection, resolve_spec_iteration

    n = int(getattr(getattr(ds, "prop", None), "num_loc", 0) or 0)
    mask = np.ones(n, dtype=bool)
    skipped: list[str] = []
    for spec in specs or []:
        attr = str(spec.get("attribute", "") or "")
        if not attr or attr not in ds.attr:
            skipped.append(f"{attr or '<unnamed>'}: not in this dataset")
            continue
        values = attr_values_for_selection(ds, attr, itr=resolve_spec_iteration(ds, spec))
        if values is None:
            skipped.append(f"{attr}: no per-localization values")
            continue
        values = np.asarray(values, dtype=float).ravel()
        if values.size != n:
            skipped.append(f"{attr}: {values.size} values for {n} localizations")
            continue
        mask &= compute_filter_mask(
            values,
            str(spec.get("mode", "per loc")),
            float(spec.get("lo", -np.inf)),
            float(spec.get("hi", np.inf)),
            ds.prop.trace_idx,
            ds.prop.num_loc_per_trace,
            ds.prop.num_traces,
            bool(spec.get("lo_inc", True)),
            bool(spec.get("hi_inc", True)),
        )
    return mask, skipped


def labels_from_filter_specs(
    ds,
    spec_sets,
    *,
    names=None,
    trace_rule: str = "per loc",
) -> LabelPlan:
    """One channel per filter spec list (e.g. one per saved preset)."""
    n = int(getattr(getattr(ds, "prop", None), "num_loc", 0) or 0)
    masks: list[np.ndarray] = []
    used_names: list[str] = []
    skipped: list[str] = []
    for index, specs in enumerate(spec_sets or []):
        mask, spec_skipped = mask_from_filter_specs(ds, specs)
        label = (list(names)[index] if names is not None and index < len(list(names))
                 else f"filter {index + 1}")
        skipped.extend(f"{label}: {reason}" for reason in spec_skipped)
        if not mask.any():
            skipped.append(f"{label}: no localizations pass")
            continue
        masks.append(mask)
        used_names.append(label)
    return labels_from_masks(
        masks, n,
        names=used_names,
        trace_idx=getattr(getattr(ds, "prop", None), "trace_idx", None),
        trace_rule=trace_rule,
        skipped=skipped,
    )
