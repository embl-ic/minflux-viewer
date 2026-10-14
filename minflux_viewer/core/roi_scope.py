"""Which views a ROI belongs in — the scoping rule for ROI display.

Until now ``RoiOverlayController.refresh`` drew **every record in the store in
every view of every dataset**. Two consequences the user hit:

* a spatial ROI drawn on a render (nanometre coordinates) was also drawn on an
  Attribute Histogram, whose x axis is an attribute value and whose y axis is a
  count — the same numbers meaning something entirely different;
* opening a render for a second dataset immediately covered it in the first
  dataset's ROIs, because nothing tied a ROI to the data it was drawn on.

Two independent facts decide it, and both already travel on the record:

**View family** — what the axes mean. ``render`` and ``scatter`` are one family
(both are nm coordinates, which is why *Process ▸ ROI ▸ Restore ROI* copies a
draft between them on purpose); the Attribute Plot and the Histogram are each
their own. A ROI never crosses families.

**Dataset** — the dataset it was drawn on. A view shows a ROI when it displays
that dataset, which for an overlay render means any of its channels, not just
the anchor.

**In memory the dataset is a key; on disk it is a position.** A ROI in the store
(or drawn in a view) holds ``context["dataset_key"]`` and, when shared,
``context["shared_dataset_keys"]`` -- the datasets' keys, which never change
while they are open. Nothing has to be remapped when an earlier dataset closes,
and a ROI whose dataset closed resolves to :data:`ORPHANED_DATASET` by itself.
Files keep the positions they always held (``dataset_idx`` /
``shared_datasets``): :func:`bind_scope` turns them into keys as a record enters
the store, :func:`portable_scope` turns keys back into positions as it is
written. A record never bound (built by hand, read from a file and not yet
stored) keeps the positional form, and every reader understands both.

**Sharing is explicit and manual.** Comparing one region across datasets from the
same sample, or carrying a ROI found on a cropped copy back to the original, is
a real workflow — so ``context["shared_datasets"]`` lists the *extra* datasets
the user has asked to see this ROI on. Nothing writes it automatically: the ROI
Manager's *Show on active dataset* is the only way in, which is exactly the
"not automatic, but available when I set it up" behaviour that was asked for.

**A record with no scope is shown everywhere**, deliberately: ROIs saved before
this existed carry no ``source_view``, and silently hiding a user's saved work is
worse than showing it in one view too many.

Qt-free and pure, so the rule is testable without a window.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .dataset_key import DatasetKey

__all__ = [
    "VIEW_FAMILIES",
    "COORDINATE_VIEWS",
    "SHARED_DATASETS_KEY",
    "DATASET_KEY_FIELD",
    "SHARED_DATASET_KEYS_FIELD",
    "view_family",
    "roi_family",
    "roi_owner_dataset",
    "roi_shared_datasets",
    "roi_dataset_indices",
    "roi_visible_in",
    "share_with_dataset",
    "unshare_dataset",
    "scope_context",
    "bind_scope",
    "portable_scope",
    "without_scope_keys",
    "with_owner",
    "references_dataset",
    "ORPHANED_DATASET",
]

#: Owner index of a ROI whose dataset has been closed. It matches no view, so
#: the ROI stays listed in the Manager (and re-attachable through *Show on
#: active dataset*) instead of being deleted or -- worse -- falling back to
#: "unscoped" and reappearing everywhere.
ORPHANED_DATASET = -1

#: ``source_view`` -> the family of axes it draws in.
VIEW_FAMILIES: dict[str, str] = {
    "render": "coordinate",
    "scatter": "coordinate",
    "attribute": "attribute",
    "histogram": "histogram",
}
#: The views that share the nanometre coordinate space.
COORDINATE_VIEWS: frozenset[str] = frozenset(
    name for name, family in VIEW_FAMILIES.items() if family == "coordinate"
)
#: Context key holding the extra datasets the user asked to display a ROI on
#: (positions: the form written to files).
SHARED_DATASETS_KEY = "shared_datasets"
#: In memory: the key of the dataset a ROI was drawn on ...
DATASET_KEY_FIELD = "dataset_key"
#: ... and the keys of the extra datasets it is shown on.
SHARED_DATASET_KEYS_FIELD = "shared_dataset_keys"


def _position(value) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _resolve(key: DatasetKey, registry) -> int | None:
    if registry is None:
        raise ValueError(
            "This ROI's dataset is held by key; resolving it needs the dataset "
            "registry (the AppState).")
    return registry.index_of(key)


def view_family(source_view) -> str | None:
    """The family of a ``source_view`` name, or ``None`` when unknown."""
    return VIEW_FAMILIES.get(str(source_view or "").strip().lower())


def _context(record) -> dict:
    context = getattr(record, "context", None)
    return context if isinstance(context, dict) else {}


def roi_family(record) -> str | None:
    """The view family a ROI belongs to, or ``None`` when it was never stamped."""
    return view_family(_context(record).get("source_view"))


def roi_owner_dataset(record, registry=None) -> int | None:
    """The current position of the dataset the ROI was drawn on.

    :data:`ORPHANED_DATASET` when that dataset has closed, ``None`` when the ROI
    records no dataset. A keyed scope is resolved through *registry*.
    """
    context = _context(record)
    key = context.get(DATASET_KEY_FIELD)
    if isinstance(key, DatasetKey):
        index = _resolve(key, registry)
        return ORPHANED_DATASET if index is None else index
    return _position(context.get("dataset_idx"))


def roi_shared_datasets(record, registry=None) -> tuple[int, ...]:
    """Extra open datasets the user asked to display this ROI on, in the order set."""
    context = _context(record)
    keys = context.get(SHARED_DATASET_KEYS_FIELD)
    raw = context.get(SHARED_DATASETS_KEY)
    out: list[int] = []
    if isinstance(keys, (list, tuple)):
        for key in keys:
            if isinstance(key, DatasetKey):
                index = _resolve(key, registry)
                if index is not None and index not in out:
                    out.append(index)
        return tuple(out)
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return ()
    for value in raw:
        index = _position(value)
        if index is not None and index not in out:
            out.append(index)
    return tuple(out)


def roi_dataset_indices(record, registry=None) -> frozenset[int] | None:
    """Every dataset this ROI may appear on, or ``None`` when it is unscoped."""
    owner = roi_owner_dataset(record, registry)
    shared = roi_shared_datasets(record, registry)
    if owner is None and not shared:
        return None
    indices = set(shared)
    if owner is not None:
        indices.add(owner)
    return frozenset(indices)


def roi_visible_in(record, *, family: str | None,
                   dataset_indices: Iterable[int] | None, registry=None) -> bool:
    """Whether *record* should be drawn in a view of *family* showing those datasets.

    ``family=None`` or ``dataset_indices=None`` means the view declines to
    constrain on that axis, so the corresponding half of the rule is skipped.
    """
    record_family = roi_family(record)
    if family is not None and record_family is not None and record_family != family:
        return False
    allowed = roi_dataset_indices(record, registry)
    if dataset_indices is None or allowed is None:
        return True
    return bool(allowed & {int(index) for index in dataset_indices})


def _set_shared(record, indices, registry) -> None:
    context = bind_scope(_context(record), registry) if registry is not None \
        else dict(_context(record))
    if registry is not None:
        keys = [registry.key_at(i) for i in indices]
        keys = [key for key in keys if key is not None]
        context.pop(SHARED_DATASETS_KEY, None)
        if keys:
            context[SHARED_DATASET_KEYS_FIELD] = keys
        else:
            context.pop(SHARED_DATASET_KEYS_FIELD, None)
    elif indices:
        context[SHARED_DATASETS_KEY] = list(indices)
    else:
        context.pop(SHARED_DATASETS_KEY, None)
    record.context = context


def share_with_dataset(record, index: int, registry=None) -> bool:
    """Also display *record* on dataset *index*. ``False`` when already there.

    With *registry* the sharing is recorded by key (and the rest of the scope
    bound too), so it follows the dataset whatever closes later.
    """
    owner = roi_owner_dataset(record, registry)
    index = int(index)
    shared = roi_shared_datasets(record, registry)
    if owner == index or index in shared:
        return False
    _set_shared(record, [*shared, index], registry)
    return True


def unshare_dataset(record, index: int | None = None, registry=None) -> bool:
    """Stop displaying *record* on dataset *index* (or on every shared dataset).

    The owning dataset is never removed: a ROI always belongs somewhere, and a
    record that could be displayed nowhere would simply look deleted.
    """
    shared = roi_shared_datasets(record, registry)
    if not shared:
        return False
    remaining = () if index is None else tuple(i for i in shared if i != int(index))
    if remaining == shared:
        return False
    _set_shared(record, list(remaining), registry)
    return True


#: Facts stamped on a ROI when it is DRAWN (the views' ``normalize_roi_record``)
#: that a later selection pass does not compute. A selection result that lacks
#: one keeps the drawn value; a selection detail it does compute always wins.
DRAWN_CONTEXT_KEYS = ("view_plane", "depth_axis", "depth_value", "z_scaling_factor")


def scope_context(previous, new) -> dict:
    """Merge a freshly computed selection context over the record's own scope.

    ``roi_selection.store_roi_mask`` replaces ``record.context`` wholesale with
    whatever the view returned. That is right for the selection details, but it
    would drop the user's manual sharing list — and, when a selection is computed
    against a *shared* dataset, would silently retarget the ROI's owner to it.
    Both are preserved here. So are the facts recorded when the ROI was drawn
    (``DRAWN_CONTEXT_KEYS``): without them a flat ROI loses the depth the
    side panes draw it at, and Scale Z the factor it pre-fills from.
    """
    merged = dict(new or {})
    old = previous if isinstance(previous, dict) else {}
    for key in DRAWN_CONTEXT_KEYS:
        if key in old and key not in merged:
            merged[key] = old[key]
    if isinstance(old.get(DATASET_KEY_FIELD), DatasetKey):
        # A keyed scope is authoritative: a selection result names the dataset
        # it was computed on by position, which must not replace the key.
        merged.pop("dataset_idx", None)
        merged.pop(SHARED_DATASETS_KEY, None)
        merged[DATASET_KEY_FIELD] = old[DATASET_KEY_FIELD]
        if old.get(SHARED_DATASET_KEYS_FIELD):
            merged[SHARED_DATASET_KEYS_FIELD] = list(old[SHARED_DATASET_KEYS_FIELD])
        return merged
    shared = old.get(SHARED_DATASETS_KEY)
    if shared:
        merged[SHARED_DATASETS_KEY] = list(shared)
    owner = old.get("dataset_idx")
    if isinstance(owner, int) and not isinstance(owner, bool):
        merged["dataset_idx"] = owner
    return merged


def bind_scope(context, registry) -> dict[str, Any]:
    """*context* with its dataset positions replaced by the datasets' keys.

    A position that names no open dataset (an orphan written by an earlier
    session, a position past the end) is kept as it is: there is no dataset
    to hold a key for, and the reader treats it exactly as before.
    """
    out = dict(context if isinstance(context, dict) else {})
    if registry is None:
        return out
    owner = _position(out.get("dataset_idx"))
    if owner is not None and not isinstance(out.get(DATASET_KEY_FIELD), DatasetKey):
        key = registry.key_at(owner) if owner >= 0 else None
        if key is not None:
            out.pop("dataset_idx", None)
            out[DATASET_KEY_FIELD] = key
    shared = out.get(SHARED_DATASETS_KEY)
    if isinstance(shared, (list, tuple)) and shared:
        keys = []
        for value in shared:
            position = _position(value)
            key = registry.key_at(position) if position is not None and position >= 0 else None
            if key is not None and key not in keys:
                keys.append(key)
        out.pop(SHARED_DATASETS_KEY, None)
        if keys:
            out[SHARED_DATASET_KEYS_FIELD] = keys
    return out


def portable_scope(context, registry) -> dict[str, Any]:
    """*context* as written to a file: keys replaced by current positions.

    A ROI whose dataset has closed is written as :data:`ORPHANED_DATASET`, and
    a shared dataset that has closed is left out -- what the remapping on every
    close used to produce, now produced only when something is saved.
    """
    out = dict(context if isinstance(context, dict) else {})
    key = out.pop(DATASET_KEY_FIELD, None)
    if isinstance(key, DatasetKey):
        index = _resolve(key, registry)
        out["dataset_idx"] = ORPHANED_DATASET if index is None else index
    keys = out.pop(SHARED_DATASET_KEYS_FIELD, None)
    if isinstance(keys, (list, tuple)):
        positions = []
        for shared_key in keys:
            if isinstance(shared_key, DatasetKey):
                index = _resolve(shared_key, registry)
                if index is not None and index not in positions:
                    positions.append(index)
        if positions:
            out[SHARED_DATASETS_KEY] = positions
    return out


def with_owner(context, index: int, registry=None) -> dict[str, Any]:
    """*context* with its owner set to the dataset at *index*.

    By key when *registry* is given (any previous owner, keyed or positional,
    is replaced), otherwise by position.
    """
    out = dict(context if isinstance(context, dict) else {})
    out.pop("dataset_idx", None)
    out.pop(DATASET_KEY_FIELD, None)
    key = registry.key_at(int(index)) if registry is not None else None
    if key is not None:
        out[DATASET_KEY_FIELD] = key
    else:
        out["dataset_idx"] = int(index)
    return out


def references_dataset(record, key: DatasetKey) -> bool:
    """Whether *record*'s scope names the dataset with *key* (owner or shared)."""
    context = _context(record)
    if context.get(DATASET_KEY_FIELD) == key:
        return True
    shared = context.get(SHARED_DATASET_KEYS_FIELD)
    return isinstance(shared, (list, tuple)) and key in shared


def without_scope_keys(context) -> dict[str, Any]:
    """*context* without its in-memory dataset keys (for a copy kept elsewhere)."""
    out = dict(context if isinstance(context, dict) else {})
    out.pop(DATASET_KEY_FIELD, None)
    out.pop(SHARED_DATASET_KEYS_FIELD, None)
    return out
