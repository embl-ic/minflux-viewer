"""The single registry of file formats this application reads and writes.

Every format table used to be written out by hand in several places -- the save
writers, the save dialog, the Preferences checkboxes, the MSR reader's export
checkboxes, and the drag-and-drop router -- so adding or retiring one format
meant editing six files and hoping none was missed. This module is the one
source of truth; the others derive from it.

Two things are declared per format:

* **How it is written** -- extension, label, whether it can hold only canonical
  raw data, and whether it is offered by default.
* **What opening it should DO** -- which is not always "load a dataset". A
  ``.msr`` opens the MSR reader, a filter preset opens the Filter dialog, a ROI
  set goes to the ROI Manager, a metadata sidecar is applied as a recipe to an
  existing dataset. :class:`OpenAction` names those routes so the router can
  dispatch without a chain of special cases.

**Ambiguous extensions are resolved by content, in a declared order.** ``.json``
is the reason: it may be a ROI set, a filter preset, a metadata sidecar or
actual localization data. Each of the first three carries a positive marker, so
they are probed first and *data is the fallback*. The router previously tried to
load data first and only looked for a filter preset if that raised, which made a
malformed data file indistinguishable from a filter file.

Qt-free and pure, so the routing table is testable without a GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

__all__ = [
    "OpenAction",
    "FormatSpec",
    "FORMATS",
    "normalize_path",
    "save_formats",
    "default_save_formats",
    "raw_only_formats",
    "extension_for",
    "label_for",
    "supported_extensions",
    "roi_extensions",
    "drop_on_dataset_extensions",
    "resolve_open",
]


class OpenAction(str, Enum):
    """What opening a file should do -- not always "load a dataset"."""

    DATASET = "dataset"                  # normal localization dataset load
    MSR_READER = "msr_reader"            # the MSR reader dialog (parse + choose)
    IMAGE_VIEWER = "image_viewer"        # standalone TIFF/OBF image viewer
    SPREADSHEET_DIALOG = "spreadsheet"   # column-mapping confirmation dialog
    ROI_MANAGER = "roi_manager"          # ROI set -> ROI Manager
    FILTER_DIALOG = "filter_dialog"      # filter preset -> Filter dialog rows
    METADATA_RECIPE = "metadata_recipe"  # sidecar -> applied to a dataset


@dataclass(frozen=True)
class FormatSpec:
    key: str
    label: str
    extensions: tuple[str, ...] = ()
    action: OpenAction = OpenAction.DATASET

    # --- write side -------------------------------------------------------
    #: Offered in File > Save / Save As at all.
    writable: bool = False
    #: Can carry only canonical raw data, never a baked snapshot.
    raw_only: bool = False
    #: Ticked in a fresh installation's Preferences.
    default_offered: bool = False

    # --- read side --------------------------------------------------------
    readable: bool = True
    #: ``core.format_sniff`` key this format is reached by, when it has one.
    sniff_key: str | None = None
    #: Dotted ``module:function`` content predicate for an ambiguous extension.
    detect: str | None = None
    #: Lower is probed first when several formats share an extension.
    detect_order: int = 100
    #: May be dropped **onto a dataset row** to act on that dataset.
    drop_on_dataset: bool = False
    notes: str = ""


FORMATS: tuple[FormatSpec, ...] = (
    # --- this application's own format ------------------------------------
    FormatSpec(
        "zarr", "MINFLUX Viewer Zarr v2 (.zarr)", (".zarr",), OpenAction.DATASET,
        writable=True, raw_only=True, default_offered=True, sniff_key="zarr",
        notes="Self-contained: raw canonical data plus processing state, no sidecar.",
    ),
    FormatSpec(
        "zarr_zip", "MINFLUX Viewer Zarr v2, single file (.zarr.zip)",
        (".zarr.zip",), OpenAction.DATASET,
        writable=True, raw_only=True, default_offered=True, sniff_key="zarr",
        notes="The same store sealed into one file: raw data plus processing "
              "state, opens without unpacking. It cannot take a "
              "processing-only update -- a zip appends rather than replaces a "
              "member -- so saving over one always rewrites it. Use the .zarr "
              "directory when you want in-place processing updates.",
    ),
    # --- MINFLUX defaults --------------------------------------------------
    FormatSpec(
        "msr", "MINFLUX (.msr)", (".msr",), OpenAction.MSR_READER,
        writable=True, raw_only=True, default_offered=False, sniff_key="msr",
        notes="Opening always goes through the MSR reader. The writer is "
              "reverse-engineered, so it is available but off by default.",
    ),
    FormatSpec(
        "mat", "MATLAB (.mat)", (".mat",), OpenAction.DATASET,
        writable=True, default_offered=True, sniff_key="mat",
    ),
    FormatSpec(
        "npy", "NumPy (.npy)", (".npy",), OpenAction.DATASET,
        writable=True, default_offered=True, sniff_key="npy",
    ),
    FormatSpec(
        "json", "JSON (.json)", (".json",), OpenAction.DATASET,
        writable=True, default_offered=True, sniff_key="json",
        # Probed last: the other .json kinds carry positive markers, plain
        # localization data does not.
        detect_order=900,
    ),
    # --- generic table interchange ----------------------------------------
    FormatSpec(
        "csv", "Canonical table (.csv)", (".csv",), OpenAction.SPREADSHEET_DIALOG,
        writable=True, default_offered=True, sniff_key="spreadsheet",
        notes="Not a MINFLUX format: the interchange path for arbitrary "
              "localization tables (ThunderSTORM/SMAP/Picasso column conventions).",
    ),
    FormatSpec(
        "spreadsheet", "Spreadsheet table",
        (".tsv", ".txt", ".xlsx", ".xlsm"), OpenAction.SPREADSHEET_DIALOG,
        sniff_key="spreadsheet",
    ),
    # --- read-only ---------------------------------------------------------
    FormatSpec(
        "npz", "NumPy zip (.npz)", (".npz",), OpenAction.DATASET,
        writable=False, sniff_key="npz",
        notes="Retired as a save format; the reader stays so existing files open.",
    ),
    FormatSpec(
        "tiff", "TIFF image", (".tif", ".tiff"), OpenAction.IMAGE_VIEWER,
        sniff_key="tiff", drop_on_dataset=True,
        notes="Images are not datasets. Dropped on a dataset row it becomes a "
              "confocal channel instead.",
    ),
    # --- things that act on an existing dataset ---------------------------
    FormatSpec(
        "roi_set", "ROI set", (".json", ".roi", ".zip"), OpenAction.ROI_MANAGER,
        detect="minflux_viewer.core.roi:is_roi_json_file",
        detect_order=10, drop_on_dataset=True,
    ),
    FormatSpec(
        "filter_preset", "Filter preset", (".json",), OpenAction.FILTER_DIALOG,
        detect="minflux_viewer.core.filter_io:is_filter_json_file",
        detect_order=20, drop_on_dataset=True,
    ),
    FormatSpec(
        "metadata_sidecar", "Processing metadata sidecar", (".json",),
        OpenAction.METADATA_RECIPE,
        detect="minflux_viewer.core.save:is_metadata_json_file",
        detect_order=30, drop_on_dataset=True,
    ),
)

_BY_KEY = {spec.key: spec for spec in FORMATS}


def normalize_path(key: str, path) -> Path:
    """Give *path* the extension of format *key*.

    ``Path.with_suffix`` only replaces the LAST suffix, so it turns
    ``run.zarr`` into ``run.zarr.zip`` correctly but also turns an already
    correct ``run.zarr.zip`` into ``run.zarr.zarr.zip``. Compound extensions
    therefore need explicit handling.
    """
    target = Path(path)
    ext = _BY_KEY[key].extensions[0]
    name = target.name
    lower = name.lower()
    if lower.endswith(ext):
        return target
    # Strip any extension this application owns, then append the wanted one.
    for known in sorted(
            {e for spec in FORMATS for e in spec.extensions},
            key=len, reverse=True):
        if lower.endswith(known):
            name = name[: -len(known)]
            break
    else:
        name = target.stem if target.suffix else name
    return target.with_name(name + ext)


def save_formats() -> tuple[FormatSpec, ...]:
    """Formats File > Save can write, in menu order."""
    return tuple(spec for spec in FORMATS if spec.writable)


def default_save_formats() -> list[str]:
    """Format keys ticked in a fresh installation."""
    return [spec.key for spec in FORMATS if spec.writable and spec.default_offered]


def raw_only_formats() -> set[str]:
    return {spec.key for spec in FORMATS if spec.raw_only}


def extension_for(key: str) -> str:
    return _BY_KEY[key].extensions[0]


def label_for(key: str) -> str:
    return _BY_KEY[key].label


def supported_extensions() -> tuple[str, ...]:
    """Every extension the application will attempt to open."""
    seen: list[str] = []
    for spec in FORMATS:
        if not spec.readable:
            continue
        for ext in spec.extensions:
            if ext not in seen:
                seen.append(ext)
    return tuple(seen)


def roi_extensions() -> frozenset[str]:
    """Extensions that always mean a ROI set, whatever their content."""
    roi = _BY_KEY["roi_set"]
    shared = {e for spec in FORMATS if spec is not roi for e in spec.extensions}
    return frozenset(set(roi.extensions) - shared)


def drop_on_dataset_extensions() -> tuple[str, ...]:
    seen: list[str] = []
    for spec in FORMATS:
        if spec.drop_on_dataset:
            for ext in spec.extensions:
                if ext not in seen:
                    seen.append(ext)
    return tuple(seen)


def _load_predicate(dotted: str):
    module_name, _, attr = dotted.partition(":")
    from importlib import import_module

    return getattr(import_module(module_name), attr)


def resolve_open(path) -> FormatSpec | None:
    """The format a path should be opened as, or ``None`` if unsupported.

    Candidates sharing the extension are probed in ``detect_order``; a spec with
    no ``detect`` matches unconditionally, which is why plain data sits last.
    A predicate that raises is treated as "not this kind" -- an unreadable file
    should fall through to the next candidate, not abort the routing.
    """
    p = Path(path)
    # Compound suffixes matter: ".zarr.zip" is a sealed store, NOT the ".zip"
    # RoiSet it would otherwise resolve to. Longest match wins.
    suffixes = [s.lower() for s in p.suffixes]
    tails = ["".join(suffixes[-n:]) for n in range(len(suffixes), 0, -1)]
    candidates: list[FormatSpec] = []
    for ext in tails:
        candidates = sorted(
            (spec for spec in FORMATS if spec.readable and ext in spec.extensions),
            key=lambda spec: spec.detect_order,
        )
        if candidates:
            break
    if not candidates:
        return None
    # Content detection exists only to split a SHARED extension. When one format
    # owns the extension outright the answer is already known, and probing would
    # wrongly reject it -- a ``.roi`` is a ROI set whatever its bytes say, and
    # ``is_roi_json_file`` would refuse it for not being JSON.
    if len(candidates) == 1:
        return candidates[0]
    for spec in candidates:
        if spec.detect is None:
            return spec
        try:
            if _load_predicate(spec.detect)(p):
                return spec
        except Exception:
            continue
    return None
