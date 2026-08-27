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

Pure and Qt-free, so the rules are testable without a window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "MATCH_DID",
    "MATCH_DATA_FILE",
    "MATCH_ACQUISITION",
    "MATCH_ORDER",
    "sidecar_identity",
    "dataset_identity",
    "match_kind",
    "best_match",
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
