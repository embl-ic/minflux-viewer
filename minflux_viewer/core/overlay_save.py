"""Saving a multi-channel overlay to the one-dataset-per-file formats.

``.mat``/``.npy``/``.json``/``.csv`` hold **one** dataset each -- that is not
changing. An overlay is therefore several files, and what makes them a set again
is the folder they share plus an ``overlay`` block in each one's metadata
sidecar naming its siblings **by filename**.

Why filenames and not paths: the set has to survive being moved or renamed as a
whole, which is exactly how a folder of results travels between machines. Why in
every sidecar and not one manifest: each file already carries its own recipe, so
a channel opened on its own still knows what it belongs to, and there is no
extra file to lose.

The ``overlay`` block is an **addition** to the sidecar, not a change to it. A
reader that does not know the key ignores it, and identity still comes from the
same three signals (`core/metadata_match.py`); nothing about pairing one recipe
with one dataset moves.

Pure and Qt-free.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = [
    "OVERLAY_KEY",
    "safe_name",
    "unique_names",
    "default_group_name",
    "overlay_block",
    "save_overlay_group",
    "DATASET_OVERLAY_KEY",
    "overlay_from_metadata",
    "dataset_overlay_block",
    "group_saved_datasets",
    "missing_members",
    "resolve_siblings",
]

#: Sidecar key holding the group. Namespaced by the application so it cannot
#: collide with anything a future Abberior or third-party writer adds.
OVERLAY_KEY = "overlay"

#: The same block once it is on a dataset. Namespaced, because dataset metadata
#: is a flat bag shared with the vendor's own keys -- unlike the sidecar, which
#: is entirely ours. ⚠ Two names for one thing: read a *sidecar* with
#: :func:`overlay_from_metadata` and a *dataset* with
#: :func:`dataset_overlay_block`, never the sidecar key off a dataset.
DATASET_OVERLAY_KEY = "minflux_viewer_overlay"

_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def safe_name(text: str, *, fallback: str = "dataset") -> str:
    """A filesystem-safe stem that still reads like the dataset's name.

    Only characters no filesystem accepts are replaced, unlike
    ``msr_reader_dialog.safe_export_stem`` which also rewrites spaces and
    braces: these names are what the user typed into the save dialog, and
    changing them further would make the file hard to recognise.
    """
    cleaned = _ILLEGAL.sub("_", str(text or "")).strip(" .")
    return cleaned or fallback


def unique_names(names) -> list[str]:
    """De-duplicate stems, preserving order.

    ⚠ Distinct dataset names can sanitise to the same stem (``ch a`` and
    ``ch/a`` both become ``ch_a``), and two channels writing the same filename
    would silently leave one of them missing.
    """
    out, seen = [], {}
    for name in names:
        stem = safe_name(name)
        key = stem.casefold()
        if key in seen:
            seen[key] += 1
            stem = f"{stem}_{seen[key]}"
        else:
            seen[key] = 1
        out.append(stem)
    return out


def default_group_name(datasets, fallback: str = "overlay") -> str:
    """What to call the folder a group is saved into.

    There is no user-facing overlay *group* number to borrow -- what the Dataset
    Manager shows as "Overlay 1/2" is each channel's index within one group --
    so this walks a chain of things that actually identify the set: the ``.msr``
    they came from, then the first channel's name, then *fallback*. It is only a
    default; the save dialog lets the user type over it.
    """
    sources = {
        str((getattr(ds, "metadata", {}) or {}).get("msr_source_path") or "")
        for ds in datasets
    }
    sources.discard("")
    if len(sources) == 1:
        return safe_name(Path(sources.pop()).stem, fallback=fallback)
    for ds in datasets:
        name = str(getattr(ds, "name", "") or "")
        if name:
            # "run.msr | channel" -> "run"
            return safe_name(name.split("|")[0].strip(), fallback=fallback)
    return fallback


def overlay_block(datasets, data_files, *, index: int, group_id: str = "") -> dict:
    """The ``overlay`` block for member *index* of a saved group."""
    members = []
    for position, (ds, filename) in enumerate(zip(datasets, data_files)):
        state = getattr(ds, "state", {}) or {}
        meta = getattr(ds, "metadata", {}) or {}
        members.append({
            "name": str(getattr(ds, "name", "") or ""),
            "data_file": str(filename),
            "did": str(meta.get("msr_dataset_did") or ""),
            "index": int(state.get("overlay_index", position) or 0),
            "lut": state.get("overlay_lut"),
        })
    anchor = datasets[index] if 0 <= index < len(datasets) else None
    state = (getattr(anchor, "state", {}) or {}) if anchor is not None else {}
    if not group_id:
        group_id = str(state.get("overlay_id") or state.get("render_group_id") or "")
    return {
        "group_id": group_id,
        "member_index": index,
        "member_count": len(datasets),
        "members": members,
    }


def save_overlay_group(datasets, folder, fmt: str, *, names=None,
                       content: str = "raw", include=None,
                       filter_mode: str = "flag", roi_records_for=None,
                       save=None) -> list[Path]:
    """Write every channel of an overlay into *folder*, each with its sidecar.

    ``names`` are the file stems (the dialog's per-channel Name fields);
    ``roi_records_for(index)`` supplies that channel's ROIs. ``save`` is the
    writer, defaulting to :func:`minflux_viewer.core.save.save_processed` --
    injected so this stays testable without touching the filesystem writers.

    The filenames are resolved **before** anything is written, because each
    sidecar's overlay block names its siblings and cannot do that until every
    name is known.
    """
    from .save import save_processed

    save = save or save_processed
    datasets = list(datasets)
    if not datasets:
        raise ValueError("save_overlay_group: no datasets to save")

    from . import formats as _formats

    ext = _formats.extension_for(fmt)
    stems = unique_names(names or [getattr(ds, "name", "") for ds in datasets])
    data_files = [f"{stem}{ext}" for stem in stems]

    target = Path(folder)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, ds in enumerate(datasets):
        rois = roi_records_for(index) if roi_records_for else None
        written.extend(save(
            ds,
            data_path=target / data_files[index],
            fmt=fmt,
            content=content,
            include=include or {"attrs": True, "derived": False, "recipe": True},
            filter_mode=filter_mode,
            roi_records=rois,
            overlay=overlay_block(datasets, data_files, index=index),
        ))
    return written


# ---------------------------------------------------------------------------
# Recognising a saved group again
# ---------------------------------------------------------------------------
def overlay_from_metadata(meta) -> dict | None:
    """The ``overlay`` block of a sidecar payload, if it carries one."""
    if not isinstance(meta, dict):
        return None
    block = meta.get(OVERLAY_KEY)
    if not isinstance(block, dict):
        return None
    members = block.get("members")
    if not isinstance(members, list) or len(members) < 2:
        # A "group" of one is not a group; nothing to regroup or go looking for.
        return None
    return block


def dataset_overlay_block(ds) -> dict | None:
    """The overlay block carried by a *dataset*, under its namespaced key."""
    meta = getattr(ds, "metadata", {}) or {}
    block = meta.get(DATASET_OVERLAY_KEY)
    return block if isinstance(block, dict) and block.get("members") else None


def group_saved_datasets(datasets) -> dict:
    """``{group_id: [(index, dataset, block)]}`` for datasets carrying a block.

    Only datasets that are **not already** in a multi-channel overlay are
    returned: re-grouping something the user has already arranged would undo
    their arrangement.
    """
    groups: dict = {}
    for index, ds in enumerate(datasets):
        block = dataset_overlay_block(ds)
        if not block:
            continue
        state = getattr(ds, "state", {}) or {}
        if state.get("overlay_id") or state.get("render_group_id"):
            continue
        groups.setdefault(str(block.get("group_id") or ""), []).append(
            (index, ds, block))
    return groups


def missing_members(block: dict, loaded_files) -> list[dict]:
    """Members of *block* whose data file is not among *loaded_files*.

    Compared by filename, case-folded: the set is identified by the names its
    sidecars recorded, so it survives the folder being moved or renamed.
    """
    have = {str(name).casefold() for name in loaded_files if name}
    return [member for member in (block.get("members") or [])
            if str(member.get("data_file") or "").casefold() not in have]


def resolve_siblings(block: dict, folder, loaded_files) -> tuple[list, list]:
    """``(found, absent)`` sibling data files, looked for beside the saved set.

    Only the folder is searched: a sidecar names siblings by filename precisely
    so the set can be moved as a whole, and hunting for them elsewhere would
    risk pairing with a same-named file from a different acquisition.
    """
    folder = Path(folder)
    found, absent = [], []
    for member in missing_members(block, loaded_files):
        candidate = folder / str(member.get("data_file") or "")
        if candidate.is_file():
            found.append((member, candidate))
        else:
            absent.append(member)
    return found, absent
