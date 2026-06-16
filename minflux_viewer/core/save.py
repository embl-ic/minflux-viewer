"""
minflux_viewer.core.save
========================
Export a processed dataset to ``.mat`` / ``.npy`` / ``.json`` / ``.csv``.

The exported *table* is a dict of 1-D columns assembled through the canonical
retrieval layer (:func:`mfx_get` / :func:`mfx_filter_mask`), so iteration and
validity selection are handled consistently:

- **only valid** → ``vld_only`` of the selection.
- **all iterations** → ``itr="all"`` instead of the last-valid materialization.
- **filter state (`ftr`)** → the current combined filter mask as a column (the
  data is never deleted; the column records the gated/visible state so it can be
  re-applied later).
- **derived attributes** → the computed attributes (``den``, ``siz``, …),
  broadcast onto the selected rows.
- **metadata** → a distinguishable sidecar JSON (RIMF, transform, filter specs,
  provenance). It is a dict tagged with :data:`METADATA_JSON_MARKER`, so it is
  never confused with a filter-preset JSON or an exported-data JSON (both lists).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .attributes import DERIVED_ATTRIBUTE_NAMES, LEGACY_DERIVED_ATTRIBUTE_ALIASES
from .loader import mfx_filter_mask, mfx_get

#: Tag identifying the metadata sidecar JSON. A metadata file is a *dict* with
#: this key; filter-preset JSON and exported-data JSON are both *lists*.
METADATA_JSON_MARKER = "minflux_viewer_metadata"

EXPORT_FORMATS = ("mat", "npy", "json", "csv")
_EXT = {"mat": ".mat", "npy": ".npy", "json": ".json", "csv": ".csv"}


def is_metadata_json_payload(payload: Any) -> bool:
    """True for a metadata sidecar payload (a dict tagged with the marker)."""
    return isinstance(payload, dict) and METADATA_JSON_MARKER in payload


def is_metadata_json_file(path) -> bool:
    """True if *path* is a metadata sidecar written by :func:`save_dataset`."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return is_metadata_json_payload(json.load(fh))
    except Exception:
        return False

_COORDS = ("xnm", "ynm", "znm")
_COORD_RAW = {"loc_x", "loc_y", "loc_z"}
_DERIVED = set(DERIVED_ATTRIBUTE_NAMES) | set(LEGACY_DERIVED_ATTRIBUTE_ALIASES)


# ---------------------------------------------------------------------------
# Column assembly
# ---------------------------------------------------------------------------
def _column(ds, name: str, itr, vld_only: bool):
    """One 1-D column for *name* under the selection, or None."""
    try:
        v = mfx_get(ds, name, itr=itr, vld_only=vld_only)
    except Exception:
        return None
    if v is None:
        return None
    arr = np.asarray(v)
    return arr.ravel() if arr.ndim == 1 else None


def build_export_table(
    ds,
    *,
    only_valid: bool = True,
    all_iterations: bool = False,
    include_ftr: bool = True,
    include_derived: bool = True,
) -> dict[str, np.ndarray]:
    """Assemble the export columns for *ds* under the chosen options."""
    itr = "all" if all_iterations else "last"
    vld_only = bool(only_valid)
    table: dict[str, np.ndarray] = {}

    for name in _COORDS:                                  # coordinates first
        col = _column(ds, name, itr, vld_only)
        if col is not None:
            table[name] = col
    n = len(next(iter(table.values()))) if table else None

    exclude = _COORD_RAW | _DERIVED | {"ftr"}
    for name in ds.attr.keys():                          # raw / quality attrs
        if name in exclude or name in table:
            continue
        col = _column(ds, name, itr, vld_only)
        if col is not None and (n is None or len(col) == n):
            table[name] = col

    if include_derived:
        for name in DERIVED_ATTRIBUTE_NAMES:
            if name in table:
                continue
            col = _column(ds, name, itr, vld_only)
            if col is not None and (n is None or len(col) == n):
                table[name] = col

    if include_ftr:
        ftr = _filter_column(ds, itr, vld_only, n)
        if ftr is not None:
            table["ftr"] = ftr

    return table


def _filter_column(ds, itr, vld_only: bool, n: int | None):
    """The current combined filter mask aligned to the selection (uint8)."""
    try:
        res = mfx_filter_mask(ds, itr=itr, vld_only=vld_only)
    except Exception:
        res = None
    if res is not None:
        mask = np.asarray(res[0]).ravel()
        if n is None or len(mask) == n:
            return mask.astype(np.uint8)
    try:                                                  # fallback: live mask
        fm = np.asarray(ds.filter_mask).ravel()
        if n is not None and len(fm) == n:
            return fm.astype(np.uint8)
    except Exception:
        pass
    return np.ones(n, dtype=np.uint8) if n is not None else None


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def _sanitize(obj: Any) -> Any:
    """Recursively convert numpy types/arrays to JSON-native values."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def build_metadata(ds, options: dict | None = None) -> dict:
    """Build the distinguishable metadata sidecar payload."""
    md = ds.metadata
    transform = ds.state.get("overlay_transform") or ds.state.get("render_transform_2d")
    meta = {
        METADATA_JSON_MARKER: 1,
        "name": getattr(ds, "name", None),
        "source_version": md.get("source_version"),
        "source_format": md.get("source_format"),
        "num_loc": int(ds.prop.num_loc),
        "num_dim": int(ds.prop.num_dim),
        "num_traces": int(ds.prop.num_traces),
        "num_iterations": int(md.get("raw_num_itr", 1) or 1),
        "iteration_load_mode": md.get("iteration_load_mode"),
        "includes_invalid": bool(md.get("includes_invalid", False)),
        "valid_num_loc": md.get("valid_num_loc"),
        "rimf": ds.rimf_provenance,
        "rimf_value": float(getattr(ds.cali, "RIMF", 1.0) or 1.0),
        "transform": transform,
        "filter_specs": ds.state.get("filter_specs"),
    }
    if options is not None:
        meta["export_options"] = dict(options)
    return _sanitize(meta)


# ---------------------------------------------------------------------------
# Format writers
# ---------------------------------------------------------------------------
def _scalar(v):
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    return v


def _write_mat(path: Path, table: dict[str, np.ndarray]) -> None:
    from scipy.io import savemat

    payload = {k: np.asarray(v).reshape(-1, 1) for k, v in table.items()}
    savemat(str(path), payload, do_compression=True, oned_as="column")


def _write_npy(path: Path, table: dict[str, np.ndarray]) -> None:
    n = len(next(iter(table.values())))
    dtype = [(k, np.asarray(v).dtype) for k, v in table.items()]
    arr = np.empty(n, dtype=dtype)
    for k, v in table.items():
        arr[k] = np.asarray(v)
    np.save(str(path), arr, allow_pickle=False)


def _write_json(path: Path, table: dict[str, np.ndarray]) -> None:
    names = list(table)
    cols = {k: np.asarray(v) for k, v in table.items()}
    n = len(next(iter(cols.values())))
    records = [{k: _scalar(cols[k][i]) for k in names} for i in range(n)]
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, table: dict[str, np.ndarray]) -> None:
    names = list(table)
    cols = {k: np.asarray(v) for k, v in table.items()}
    n = len(next(iter(cols.values())))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(names)
        for i in range(n):
            writer.writerow([_scalar(cols[k][i]) for k in names])


_WRITERS = {"mat": _write_mat, "npy": _write_npy, "json": _write_json, "csv": _write_csv}


def save_dataset(
    ds,
    path,
    *,
    fmt: str,
    only_valid: bool = True,
    all_iterations: bool = False,
    include_ftr: bool = True,
    include_derived: bool = True,
    include_metadata: bool = True,
) -> list[Path]:
    """Write *ds* to *path* in *fmt*; returns the list of written file paths.

    When *include_metadata* is set, a ``<stem>_metadata.json`` sidecar is also
    written (tagged so it is distinguishable from filter / data JSON).
    """
    if fmt not in _WRITERS:
        raise ValueError(f"Unsupported export format: {fmt!r}")
    path = Path(path).with_suffix(_EXT[fmt])
    table = build_export_table(
        ds,
        only_valid=only_valid,
        all_iterations=all_iterations,
        include_ftr=include_ftr,
        include_derived=include_derived,
    )
    if not table:
        raise ValueError("No exportable columns for this dataset.")

    _WRITERS[fmt](path, table)
    written = [path]

    if include_metadata:
        options = {
            "format": fmt,
            "only_valid": only_valid,
            "all_iterations": all_iterations,
            "include_ftr": include_ftr,
            "include_derived": include_derived,
            "columns": list(table),
        }
        meta_path = path.with_name(path.stem + "_metadata.json")
        meta_path.write_text(
            json.dumps(build_metadata(ds, options), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written.append(meta_path)

    return written
