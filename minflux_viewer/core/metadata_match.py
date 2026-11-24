"""Matching a processing-metadata sidecar to a loaded dataset.

A ``<stem>_viewer_metadata.json`` file is a **processing recipe**, not a property of
one particular dataset: applying a saved processing to another copy of the same
run, or to a different dataset entirely, is a legitimate thing to want. So
identity here only ever picks the **default** in a dropdown — it never gates
what may be applied, and there is deliberately **no application-minted dataset
uid**. Weak identity is sufficient precisely because a wrong guess costs the
user one dropdown change, not a refused operation.

Three signals, strongest first, all of which already exist:

``msr_dataset_did``
    The UUID Imspector mints per acquisition. Strongest, but present only for
    data that came from a ``.msr`` (or a ``.zarr`` written from one) — a
    ``.mat``/``.npy``/``.csv``/``.json`` data file cannot carry it, so the
    *sidecar* is the only way it reaches such a dataset.

``data_file``
    The filename the sidecar was written beside. Exactly what "the sidecar next
    to this data" means, and it is in every sidecar ever written, including
    those predating the did. Compared by **name**, not path, because that is
    all the sidecar records — so it still matches after the pair is moved.

``acquisition.date``
    The instrument timestamp. Survives renaming and copying, and unlike the
    other two it matches a dataset re-exported to a different format.

Restoring Imspector's acquisition record is the one thing a recipe carries
that is *not* processing: it describes how the data were acquired, and CRLB and
aggregation read it. So it is gated, by :func:`acquisition_record_match`, and
not merely defaulted -- see there.

Pure and Qt-free, so the rules are testable without a window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "MATCH_DID",
    "MATCH_DATA_FILE",
    "MATCH_ACQUISITION",
    "MATCH_ORDER",
    "RECIPE_IDENTITY_KEY",
    "IDENTITY_BEFORE_RECIPE_KEY",
    "RecordMatch",
    "sidecar_identity",
    "dataset_identity",
    "intrinsic_identity",
    "mark_recipe_identity",
    "preserve_intrinsic_identity",
    "match_kind",
    "best_match",
    "acquisition_record_match",
    "describe_match",
    "is_snapshot_recipe",
    "recipe_summary",
]

MATCH_DID = "did"
MATCH_DATA_FILE = "data_file"
MATCH_ACQUISITION = "acquisition"

#: Strongest first. ``best_match`` takes the first kind that matches anything,
#: so a weaker signal can never outrank a stronger one.
MATCH_ORDER: tuple[str, ...] = (MATCH_DID, MATCH_DATA_FILE, MATCH_ACQUISITION)

_MATCH_TEXT = {
    MATCH_DID: "same acquisition (MSR dataset ID)",
    MATCH_DATA_FILE: "written beside this data file",
    MATCH_ACQUISITION: "same acquisition time",
}


def describe_match(kind: str | None) -> str:
    """Plain-language reason a dataset was matched, for a dialog or the Log."""
    return _MATCH_TEXT.get(str(kind or ""), "no match")


def _text(value: Any) -> str:
    return str(value or "").strip()


def sidecar_identity(meta: dict | None) -> dict[str, str]:
    """The three identity signals carried by a sidecar payload (any may be "")."""
    meta = meta if isinstance(meta, dict) else {}
    acquisition = meta.get("acquisition")
    acquisition = acquisition if isinstance(acquisition, dict) else {}
    return {
        MATCH_DID: _text(meta.get("msr_dataset_did")),
        MATCH_DATA_FILE: _text(meta.get("data_file")).casefold(),
        MATCH_ACQUISITION: _text(acquisition.get("date")),
    }


def dataset_identity(ds: Any) -> dict[str, Any]:
    """The same three signals for a loaded dataset.

    ``data_file`` is a *set* of candidate names: a dataset knows the file it was
    read from in more than one place, and which one is populated depends on the
    loader.
    """
    metadata = getattr(ds, "metadata", None) or {}
    names: set[str] = set()
    file_info = getattr(ds, "file", None)
    for candidate in (getattr(file_info, "name", None),
                      getattr(file_info, "recent_path", None),
                      getattr(ds, "name", None)):
        text = _text(candidate)
        if text:
            names.add(Path(text).name.casefold())
    return {
        MATCH_DID: _text(metadata.get("msr_dataset_did")),
        MATCH_DATA_FILE: names,
        MATCH_ACQUISITION: _text(metadata.get("acquisition_date")),
    }


#: ``ds.metadata`` key listing the identity keys (``msr_dataset_did``,
#: ``acquisition_date``) that a processing recipe supplied rather than the data.
#: It travels with the metadata (Viewer Zarr, our ``.msr``), so an identity a
#: recipe once copied never becomes evidence that the same recipe fits.
RECIPE_IDENTITY_KEY = "identity_from_recipe"

_IDENTITY_METADATA = {MATCH_DID: "msr_dataset_did", MATCH_ACQUISITION: "acquisition_date"}


def mark_recipe_identity(metadata: dict, key: str) -> None:
    """Record that ``metadata[key]`` came from a recipe, not from the data."""
    marked = metadata.get(RECIPE_IDENTITY_KEY)
    keys = {str(k) for k in marked} if isinstance(marked, (list, tuple, set)) else set()
    keys.add(str(key))
    metadata[RECIPE_IDENTITY_KEY] = sorted(keys)


#: ``ds.metadata`` key holding the dataset's own identity values that a recipe
#: replaced (``{"acquisition_date": …}``), written once and never overwritten.
#: A recipe that overwrites the date therefore cannot erase the evidence that
#: refused its record: applied again, it meets the same conflict.
IDENTITY_BEFORE_RECIPE_KEY = "identity_before_recipe"


def preserve_intrinsic_identity(metadata: dict, key: str, previous: Any) -> None:
    """Keep *previous*, the value of ``metadata[key]`` a recipe just replaced,
    as intrinsic evidence.

    Kept only when it was intrinsic (present, not itself recipe-supplied --
    call this before :func:`mark_recipe_identity`), and only the first time:
    later recipes never overwrite it.
    """
    value = _text(previous)
    marked = metadata.get(RECIPE_IDENTITY_KEY)
    copied = {str(k) for k in marked} if isinstance(marked, (list, tuple, set)) else set()
    if not value or key in copied:
        return
    kept = metadata.get(IDENTITY_BEFORE_RECIPE_KEY)
    kept = dict(kept) if isinstance(kept, dict) else {}
    kept.setdefault(str(key), value)
    metadata[IDENTITY_BEFORE_RECIPE_KEY] = kept


def intrinsic_identity(ds: Any) -> dict[str, str]:
    """The dataset's own DID and acquisition time -- not ones a recipe copied.

    A value a recipe replaced is answered from :data:`IDENTITY_BEFORE_RECIPE_KEY`.
    """
    metadata = getattr(ds, "metadata", None) or {}
    marked = metadata.get(RECIPE_IDENTITY_KEY)
    copied = {str(k) for k in marked} if isinstance(marked, (list, tuple, set)) else set()
    kept = metadata.get(IDENTITY_BEFORE_RECIPE_KEY)
    kept = kept if isinstance(kept, dict) else {}
    out: dict[str, str] = {}
    for kind, key in _IDENTITY_METADATA.items():
        if _text(kept.get(key)):
            out[kind] = _text(kept.get(key))
        else:
            out[kind] = "" if key in copied else _text(metadata.get(key))
    return out


@dataclass(frozen=True)
class RecordMatch:
    """Whether a sidecar's acquisition record may be restored onto a dataset.

    ``basis`` is the signal it was accepted on (``None``: refused), ``reason``
    the plain-language account for the Log.
    """

    basis: str | None
    reason: str

    @property
    def accepted(self) -> bool:
        return self.basis is not None

    @property
    def verified(self) -> bool:
        """Accepted on the acquisition's own identity, not only its file name."""
        return self.basis in (MATCH_DID, MATCH_ACQUISITION)


def _same_time(a: str, b: str) -> bool:
    """Whether two acquisition timestamps name the same instant.

    Both with a zone: compared as instants. One without: it agrees when it
    equals the other's time as written (its wall clock) or that time in UTC
    -- a stamp that dropped its zone is the same acquisition, not a conflict.
    Unparseable text is compared as text.

    Not transitive: ``12:00`` agrees with ``12:00+02:00`` and with
    ``12:00+00:00``, which disagree. A caller deciding on several stamps
    compares every pair.
    """
    from datetime import timezone

    from minflux_viewer.core.acquisition_time import parse_acquisition_date

    first, second = parse_acquisition_date(a), parse_acquisition_date(b)
    if first is None or second is None:
        return a == b
    if (first.tzinfo is None) == (second.tzinfo is None):
        return bool(first == second)
    naive, aware = (first, second) if first.tzinfo is None else (second, first)
    return naive in (aware.replace(tzinfo=None),
                     aware.astimezone(timezone.utc).replace(tzinfo=None))


def acquisition_record_match(meta: dict | None, ds: Any,
                             record: dict | None = None) -> RecordMatch:
    """Decide whether *meta*'s acquisition *record* belongs to *ds*.

    Unlike :func:`match_kind` (which only picks a dropdown default) this gates,
    and every rule is about *evidence*:

    * **Any conflict vetoes.** Two different acquisition IDs -- the sidecar's,
      its record's, the dataset's own -- or two different acquisition times
      refuse the record, whatever else agrees and whatever the file names
      say. Agreement on the ID does not excuse a time conflict: the record is
      restored only when no evidence speaks against it. Times are compared
      pairwise -- the sidecar's, its record's and the dataset's -- because a
      time without a zone agrees with two zoned times that disagree with each
      other. A shared basename is common (``run.npy`` in every folder); a
      shared ID across acquisitions is not.
    * **Only intrinsic identity counts.** An ID or time a recipe copied onto
      the dataset (:data:`RECIPE_IDENTITY_KEY`) is not evidence; otherwise
      applying an unrelated recipe once would make it "match" the second time.
      A dataset's own date that a recipe replaced still is
      (:data:`IDENTITY_BEFORE_RECIPE_KEY`), so repetition cannot turn a
      refusal into acceptance.
    * **Agreement on the ID, else the time, is verified acceptance.**
    * **File name alone is accepted, and says so.** A ``.mat``/``.npy``/
      ``.json`` file has no ID or time of its own, so "written beside this
      data file" is the only link it can have to its record. The caller
      records ``matched_by="data_file"``, ``verified=False`` as provenance,
      which travels with the record and is reported in the Log.
    """
    meta = meta if isinstance(meta, dict) else {}
    record = record if isinstance(record, dict) else {}
    want = sidecar_identity(meta)
    have = intrinsic_identity(ds)

    dids = {d for d in (want[MATCH_DID], _text(record.get("did"))) if d}
    if len(dids) > 1:
        return RecordMatch(None, "the sidecar names two different acquisition IDs "
                                 f"({', '.join(sorted(dids))})")
    times = [t for t in (want[MATCH_ACQUISITION], _text(record.get("acquisition_date"))) if t]
    if len(times) == 2 and not _same_time(*times):
        return RecordMatch(None, "the sidecar names two different acquisition times "
                                 f"({times[0]}, {times[1]})")

    # Every comparison first; acceptance only when none conflicts.
    did_agrees = time_agrees = False
    if dids and have[MATCH_DID]:
        theirs = next(iter(dids))
        if theirs != have[MATCH_DID]:
            return RecordMatch(None, f"a different acquisition: the sidecar's ID {theirs} "
                                     f"is not this dataset's {have[MATCH_DID]}")
        did_agrees = True
    # Each recorded time against the dataset's, not only the first: a stamp
    # without a zone agrees with both of two zoned stamps that differ from each
    # other, so agreement is not transitive and every pair is checked.
    if times and have[MATCH_ACQUISITION]:
        for recorded in times:
            if not _same_time(recorded, have[MATCH_ACQUISITION]):
                return RecordMatch(None, f"a different acquisition: recorded {recorded}, "
                                         f"this dataset {have[MATCH_ACQUISITION]}")
        time_agrees = True
    if did_agrees:
        return RecordMatch(MATCH_DID, describe_match(MATCH_DID))
    if time_agrees:
        return RecordMatch(MATCH_ACQUISITION, describe_match(MATCH_ACQUISITION))

    name = want[MATCH_DATA_FILE]
    if name and name in dataset_identity(ds)[MATCH_DATA_FILE]:
        return RecordMatch(MATCH_DATA_FILE,
                           "matched by file name only - the data carry no acquisition "
                           "ID or time to check it against")
    return RecordMatch(None, "nothing shows this sidecar was written for this acquisition")


def match_kind(meta: dict | None, ds: Any) -> str | None:
    """The strongest signal on which *meta* and *ds* agree, or ``None``.

    An empty signal never matches — two datasets that both lack a did are not
    thereby the same dataset.
    """
    want = sidecar_identity(meta)
    have = dataset_identity(ds)
    for kind in MATCH_ORDER:
        wanted = want.get(kind)
        if not wanted:
            continue
        held = have.get(kind)
        if kind == MATCH_DATA_FILE:
            if wanted in (held or set()):
                return kind
        elif held and held == wanted:
            return kind
    return None


def best_match(meta: dict | None, datasets) -> tuple[int | None, str | None]:
    """``(index, kind)`` of the best-matching dataset, or ``(None, None)``.

    Resolved by **signal strength first, position second**: every dataset is
    tested for the strongest signal before any is tested for a weaker one, so
    an early dataset agreeing only on acquisition time cannot outrank a later
    one carrying the same did.
    """
    datasets = list(datasets or [])
    per_dataset = [match_kind(meta, ds) for ds in datasets]
    for kind in MATCH_ORDER:
        for index, found in enumerate(per_dataset):
            if found == kind:
                return index, kind
    return None, None


def is_snapshot_recipe(meta: dict | None) -> bool:
    """True for a recipe whose data file already has the processing baked in.

    Worth asking about before applying one to a *different* dataset: by the
    baked-XOR-recipe rule a snapshot pins ``z_scaling_factor`` to 1.0 and carries
    no transform or filters, so applying it elsewhere silently flattens that
    dataset's Z scaling and contributes nothing else.
    """
    meta = meta if isinstance(meta, dict) else {}
    return _text(meta.get("content")).casefold() == "snapshot"


def recipe_summary(meta: dict | None) -> str:
    """One line naming what a recipe would apply, for the confirmation dialog."""
    meta = meta if isinstance(meta, dict) else {}
    parts: list[str] = []
    calibration = meta.get("calibration")
    calibration = calibration if isinstance(calibration, dict) else {}
    z_scaling = calibration.get("z_scaling_factor")
    if z_scaling is not None:
        try:
            parts.append(f"Z scaling factor {float(z_scaling):.4g}")
        except (TypeError, ValueError):
            pass
    if meta.get("transform") is not None:
        parts.append("overlay transform")
    filters = meta.get("filters") or []
    if filters:
        parts.append(f"{len(filters)} filter(s)")
    rois = meta.get("rois") or []
    if rois:
        parts.append(f"{len(rois)} ROI(s)")
    acquisition = meta.get("acquisition")
    if isinstance(acquisition, dict) and acquisition.get("date"):
        parts.append("acquisition date")
    return ", ".join(parts) if parts else "nothing (empty recipe)"
