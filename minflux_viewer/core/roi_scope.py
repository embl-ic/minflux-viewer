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

**Dataset** — ``context["dataset_idx"]`` is the dataset it was drawn on. A view
shows a ROI when it displays that dataset, which for an overlay render means any
of its channels, not just the anchor.

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

__all__ = [
    "VIEW_FAMILIES",
    "COORDINATE_VIEWS",
    "SHARED_DATASETS_KEY",
    "view_family",
    "roi_family",
    "roi_owner_dataset",
    "roi_shared_datasets",
    "roi_dataset_indices",
    "roi_visible_in",
    "share_with_dataset",
    "unshare_dataset",
    "scope_context",
    "ORPHANED_DATASET",
    "remap_dataset_indices",
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
#: Context key holding the extra datasets the user asked to display a ROI on.
SHARED_DATASETS_KEY = "shared_datasets"


def view_family(source_view) -> str | None:
    """The family of a ``source_view`` name, or ``None`` when unknown."""
    return VIEW_FAMILIES.get(str(source_view or "").strip().lower())


def _context(record) -> dict:
    context = getattr(record, "context", None)
    return context if isinstance(context, dict) else {}


def roi_family(record) -> str | None:
    """The view family a ROI belongs to, or ``None`` when it was never stamped."""
    return view_family(_context(record).get("source_view"))


def roi_owner_dataset(record) -> int | None:
    """The dataset the ROI was drawn on, or ``None`` when unknown."""
    value = _context(record).get("dataset_idx")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def roi_shared_datasets(record) -> tuple[int, ...]:
    """Extra datasets the user asked to display this ROI on, in the order set."""
    raw = _context(record).get(SHARED_DATASETS_KEY)
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return ()
    out: list[int] = []
    for value in raw:
        if isinstance(value, int) and not isinstance(value, bool) and value not in out:
            out.append(int(value))
    return tuple(out)


def roi_dataset_indices(record) -> frozenset[int] | None:
    """Every dataset this ROI may appear on, or ``None`` when it is unscoped."""
    owner = roi_owner_dataset(record)
    shared = roi_shared_datasets(record)
    if owner is None and not shared:
        return None
    indices = set(shared)
    if owner is not None:
        indices.add(owner)
    return frozenset(indices)


def roi_visible_in(record, *, family: str | None,
                   dataset_indices: Iterable[int] | None) -> bool:
    """Whether *record* should be drawn in a view of *family* showing those datasets.

    ``family=None`` or ``dataset_indices=None`` means the view declines to
    constrain on that axis, so the corresponding half of the rule is skipped.
    """
    record_family = roi_family(record)
    if family is not None and record_family is not None and record_family != family:
        return False
    allowed = roi_dataset_indices(record)
    if dataset_indices is None or allowed is None:
        return True
    return bool(allowed & {int(index) for index in dataset_indices})


def share_with_dataset(record, index: int) -> bool:
    """Also display *record* on dataset *index*. ``False`` when already there."""
    owner = roi_owner_dataset(record)
    index = int(index)
    if owner == index or index in roi_shared_datasets(record):
        return False
    context = dict(_context(record))
    context[SHARED_DATASETS_KEY] = [*roi_shared_datasets(record), index]
    record.context = context
    return True


def unshare_dataset(record, index: int | None = None) -> bool:
    """Stop displaying *record* on dataset *index* (or on every shared dataset).

    The owning dataset is never removed: a ROI always belongs somewhere, and a
    record that could be displayed nowhere would simply look deleted.
    """
    shared = roi_shared_datasets(record)
    if not shared:
        return False
    remaining = () if index is None else tuple(i for i in shared if i != int(index))
    if remaining == shared:
        return False
    context = dict(_context(record))
    if remaining:
        context[SHARED_DATASETS_KEY] = list(remaining)
    else:
        context.pop(SHARED_DATASETS_KEY, None)
    record.context = context
    return True


def scope_context(previous, new) -> dict:
    """Merge a freshly computed selection context over the record's own scope.

    ``roi_selection.store_roi_mask`` replaces ``record.context`` wholesale with
    whatever the view returned. That is right for the selection details, but it
    would drop the user's manual sharing list — and, when a selection is computed
    against a *shared* dataset, would silently retarget the ROI's owner to it.
    Both are preserved here.
    """
    merged = dict(new or {})
    old = previous if isinstance(previous, dict) else {}
    shared = old.get(SHARED_DATASETS_KEY)
    if shared:
        merged[SHARED_DATASETS_KEY] = list(shared)
    owner = old.get("dataset_idx")
    if isinstance(owner, int) and not isinstance(owner, bool):
        merged["dataset_idx"] = owner
    return merged


def remap_dataset_indices(records, removed: int) -> int:
    """Fix ROI scopes after the dataset at *removed* was closed.

    ``AppState.remove_dataset`` pops from a list, so every later dataset's index
    shifts down by one while ROI contexts hold absolute indices. This was
    harmless while every ROI was drawn in every view; once display is scoped by
    dataset it silently re-attributes a ROI to whatever now sits at that index.

    ROIs of the closed dataset become :data:`ORPHANED_DATASET` rather than being
    deleted — losing a user's regions to a mis-click would be worse than leaving
    them listed and re-attachable. Returns the number of records changed.
    """
    removed = int(removed)

    def shift(index: int) -> int:
        if index == removed:
            return ORPHANED_DATASET
        return index - 1 if index > removed else index

    changed = 0
    for record in records or ():
        context = _context(record)
        owner = roi_owner_dataset(record)
        shared = roi_shared_datasets(record)
        if owner is None and not shared:
            continue
        new_context = dict(context)
        touched = False
        if owner is not None:
            new_owner = shift(owner)
            if new_owner != owner:
                new_context["dataset_idx"] = new_owner
                touched = True
        if shared:
            # A shared entry pointing at the closed dataset is simply dropped:
            # unlike the owner it carries no identity worth preserving.
            new_shared = [shift(i) for i in shared]
            new_shared = [i for i in new_shared if i != ORPHANED_DATASET]
            if new_shared != list(shared):
                if new_shared:
                    new_context[SHARED_DATASETS_KEY] = new_shared
                else:
                    new_context.pop(SHARED_DATASETS_KEY, None)
                touched = True
        if touched:
            record.context = new_context
            changed += 1
    return changed
