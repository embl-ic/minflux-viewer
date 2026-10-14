"""What a Log event records about its dataset when it is written.

A Log event used to remember its dataset by index, and an index names a
position that shifts whenever an earlier dataset closes -- so Generate Method
Text could describe one dataset with another's metadata. An event now records
the dataset's :class:`~.dataset_key.DatasetKey` (to find it while it is open)
and a snapshot of what the method-text rules read, taken at that moment. The
snapshot is what the event describes: a dataset closed, renamed or re-stamped
later is still described as it was.

A snapshot holds plain values only, never the dataset: the Log keeps thousands
of events, and a reference would keep every closed dataset in memory.

Qt-free. Keys stay in memory; nothing here is written to a file.
"""

from __future__ import annotations

import copy
from typing import Any


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def dataset_summary(ds: Any) -> dict[str, Any]:
    """The facts about *ds* the method-text rules describe, as plain values."""
    md = getattr(ds, "metadata", None) or {}
    prop = getattr(ds, "prop", None)
    num_loc = _int(getattr(prop, "num_loc", 0))
    file = getattr(ds, "file", None)
    return {
        "name": getattr(ds, "name", None),
        "source_path": str(getattr(file, "path", "") or "") or None,
        "did": md.get("msr_dataset_did"),
        "acquisition_date": md.get("acquisition_date"),
        "msr_dataset_name": md.get("msr_dataset_name"),
        "source_format": md.get("source_format"),
        "source_version": md.get("source_version"),
        "raw_num_itr": md.get("raw_num_itr", 1),
        "num_loc": num_loc,
        "valid_num_loc": md.get("valid_num_loc", num_loc),
        "num_dim": _int(getattr(prop, "num_dim", 2), 2) or 2,
        "num_traces": _int(getattr(prop, "num_traces", 0)),
        "iteration_load_mode": md.get("iteration_load_mode", "last"),
        "includes_invalid": bool(md.get("includes_invalid")),
        "overlay_reference": md.get("overlay_reference"),
        "overlay_alignment_mode": md.get("overlay_alignment_mode", "none"),
        "overlay_bead_excluded": list(md.get("overlay_bead_excluded") or []),
        "overlay_transform": copy.deepcopy(md.get("overlay_transform")),
    }


def attribution_snapshot(state: Any, idx: int) -> dict[str, Any] | None:
    """The snapshot a Log event about the dataset at *idx* records.

    ``label`` is the key as shown (``ds-7``); ``overlay_members`` lists every
    channel of a multi-channel overlay, in channel order, each with its own
    label, or is None for a dataset that is not part of one.
    """
    key = state.key_at(idx)
    if key is None:
        return None
    snapshot = dataset_summary(state.datasets[idx])
    snapshot["label"] = str(key)
    snapshot["overlay_members"] = None
    from .overlay import is_multichannel_overlay, overlay_members
    if is_multichannel_overlay(state, idx):
        members = []
        for member_idx, member in overlay_members(state, idx):
            entry = dataset_summary(member)
            member_key = state.key_at(member_idx)
            entry["label"] = str(member_key) if member_key is not None else None
            members.append(entry)
        snapshot["overlay_members"] = members
    return snapshot
