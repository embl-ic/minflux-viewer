"""Create separate dataset channels from acquisition-time windows."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
from math import isfinite

import numpy as np

from .dataset import (
    AttrStore,
    AttributeComponent,
    DatasetComponents,
    FileInfo,
    MinfluxDataset,
)
from .loader import _build_channel, _compute_properties


_OVERLAY_KEYS = (
    "overlay_id",
    "overlay_index",
    "overlay_order",
    "overlay_transform",
    "render_group_id",
    "render_transform_2d",
)


@dataclass(frozen=True)
class TimeWindow:
    """One named output channel bounded in canonical acquisition seconds."""

    name: str
    start_s: float
    end_s: float
    lut: str


@dataclass(frozen=True)
class TimeChannelSelection:
    """A validated time window and its materialized per-localization mask."""

    window: TimeWindow
    mask: np.ndarray
    upper_inclusive: bool


def validate_time_windows(
    windows: list[TimeWindow] | tuple[TimeWindow, ...],
    *,
    minimum: int = 2,
) -> list[TimeWindow]:
    """Validate and return windows in chronological order.

    Gaps are allowed so washing or transition periods can be omitted. Overlaps
    are rejected because separate channels must not silently share rows.
    """
    ordered = sorted(list(windows), key=lambda item: (item.start_s, item.end_s))
    if len(ordered) < minimum:
        raise ValueError(f"Define at least {minimum} time windows.")

    names: set[str] = set()
    for item in ordered:
        name = item.name.strip()
        if not name:
            raise ValueError("Every time window needs a channel name.")
        folded = name.casefold()
        if folded in names:
            raise ValueError(f"Channel names must be unique: {name!r}.")
        names.add(folded)
        if not (isfinite(item.start_s) and isfinite(item.end_s)):
            raise ValueError(f"Time bounds for {name!r} must be finite.")
        if item.end_s <= item.start_s:
            raise ValueError(f"The end time for {name!r} must be after its start time.")

    for previous, current in zip(ordered, ordered[1:]):
        scale = max(1.0, abs(previous.end_s), abs(current.start_s))
        tolerance = np.finfo(float).eps * scale * 16.0
        if current.start_s < previous.end_s - tolerance:
            raise ValueError(
                f"Time windows {previous.name!r} and {current.name!r} overlap."
            )
    return ordered


def time_channel_selections(
    tim_s,
    windows: list[TimeWindow] | tuple[TimeWindow, ...],
    *,
    base_mask=None,
) -> list[TimeChannelSelection]:
    """Build one per-localization mask for every validated time window.

    Adjacent windows use ``[start, end)`` for the earlier window and
    ``[start, end]`` for the final one, assigning a shared boundary exactly
    once. A window followed by a gap includes its upper endpoint.
    """
    tim = np.asarray(tim_s, dtype=float).ravel()
    if base_mask is None:
        base = np.ones(tim.size, dtype=bool)
    else:
        base = np.asarray(base_mask, dtype=bool).ravel()
        if base.size != tim.size:
            raise ValueError("The active filter mask does not align with the time attribute.")

    ordered = validate_time_windows(windows)
    finite = np.isfinite(tim)
    selections: list[TimeChannelSelection] = []
    for index, window in enumerate(ordered):
        upper_inclusive = True
        if index + 1 < len(ordered):
            next_start = ordered[index + 1].start_s
            scale = max(1.0, abs(window.end_s), abs(next_start))
            tolerance = np.finfo(float).eps * scale * 16.0
            upper_inclusive = bool(next_start > window.end_s + tolerance)

        upper = tim <= window.end_s if upper_inclusive else tim < window.end_s
        mask = base & finite & (tim >= window.start_s) & upper
        selections.append(
            TimeChannelSelection(
                window=window,
                mask=mask,
                upper_inclusive=upper_inclusive,
            )
        )
    return selections


def _subset_value(value, mask: np.ndarray, source_len: int):
    """Copy *value*, slicing row-aligned arrays to the selected channel rows."""
    arr = np.asarray(value)
    if arr.ndim >= 1 and arr.shape[0] == source_len:
        return arr[mask].copy()
    return copy.deepcopy(value)


def _subset_attr_store(store: AttrStore, mask: np.ndarray, source_len: int) -> AttrStore:
    """Return an AttrStore whose per-localization arrays are row-subset copies."""
    return AttrStore({
        key: _subset_value(value, mask, source_len)
        for key, value in store.items()
    })


def clone_time_channel_dataset(
    source: MinfluxDataset,
    selection: TimeChannelSelection,
    *,
    name: str | None = None,
    timestamp: str | None = None,
) -> MinfluxDataset:
    """Return a standalone dataset containing only one selected time window."""
    channel_name = (name or selection.window.name).strip()
    if not channel_name:
        raise ValueError("A time channel needs a dataset name.")
    stamp = timestamp or datetime.now().strftime("%Y-%b-%d, %H:%M:%S")

    mask = np.asarray(selection.mask, dtype=bool).ravel()
    source_len = int(source.prop.num_loc)
    if mask.size != source_len:
        raise ValueError("The time-window mask does not align with the source dataset.")
    selected_count = int(mask.sum())
    if selected_count <= 0:
        raise ValueError(f"Time window {selection.window.name!r} is empty.")

    new_attrs = _subset_attr_store(source.attr, mask, source_len)
    prop = _compute_properties(new_attrs, selected_count, 1)
    prop.attr_names = new_attrs.keys()

    mfx = AttributeComponent(
        new_attrs,
        roles=copy.deepcopy(source.mfx.roles),
        meta=copy.deepcopy(source.mfx.meta),
    )
    new_state = copy.deepcopy(source.state)
    new_state.pop("filter_mask", None)
    new_components = DatasetComponents(
        mfx=mfx,
        mbm=copy.deepcopy(source.components.mbm),
        metadata=copy.deepcopy(source.metadata),
        derived=_subset_attr_store(source.derived, mask, source_len),
        state=new_state,
        mfx_raw=AttrStore(),
        derived_last=AttrStore(),
    )

    duplicate = MinfluxDataset(
        file=FileInfo(
            name=channel_name,
            folder=source.file.folder,
            datetime=stamp,
            raw_data=None,
            recent_path=source.file.recent_path,
        ),
        prop=prop,
        attr=new_attrs,
        cali=copy.deepcopy(source.cali),
        channel=_build_channel(new_attrs, prop),
        components=new_components,
    )

    for key in _OVERLAY_KEYS:
        duplicate.state.pop(key, None)
        duplicate.metadata.pop(key, None)

    time_filter = {
        "attribute": "tim",
        "mode": "per loc",
        "lo": float(selection.window.start_s),
        "hi": float(selection.window.end_s),
        "lo_inc": True,
        "hi_inc": bool(selection.upper_inclusive),
    }
    duplicate.state["filter_specs"] = (
        copy.deepcopy(list(source.state.get("filter_specs") or [])) + [time_filter]
    )
    duplicate.filter_mask = np.ones(selected_count, dtype=bool)
    duplicate.metadata["time_channels_source_dataset"] = source.name
    duplicate.metadata["time_channels_source_num_loc"] = source_len
    duplicate.metadata["time_channels_selected_num_loc"] = selected_count
    duplicate.metadata["time_channels_source_num_itr"] = int(source.prop.num_itr or 1)
    duplicate.metadata["raw_num_itr"] = 1
    duplicate.metadata["iteration_load_mode"] = "last"
    vld = new_attrs.get("vld")
    if vld is None:
        duplicate.metadata["valid_num_loc"] = selected_count
        duplicate.metadata["includes_invalid"] = False
    else:
        valid = np.asarray(vld, dtype=bool).ravel()
        duplicate.metadata["valid_num_loc"] = int(valid.sum()) if valid.size else 0
        duplicate.metadata["includes_invalid"] = bool(valid.size and np.any(~valid))
    duplicate.metadata["time_channels_raw_store"] = "omitted to avoid duplicating the source all-iteration cache"
    duplicate.metadata["created_note"] = (
        f"{stamp}, derived from {source.name} by time-window channel separation."
    )
    return duplicate
