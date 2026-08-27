"""
minflux_viewer.core.save
========================
Save a *processed* dataset.

Design (per project decision):

- The **data** is written as **canonical raw** values — the dataset's
  all-iteration ``mfx_raw`` reconstructed into an **m2410-style structured
  array** (Abberior layout: ``loc`` (N×3), ``dcr`` (N×2), flat ``itr``, …),
  *regardless of the original source/version*. It therefore loads back through
  the normal parser, which re-applies the same Z-dimensionality check — so
  dimensionality and calibration are reproduced, not baked in.
- For the ordinary structured/flat exports, **gating, calibration and filtering** live in a sidecar
  ``<stem>_metadata.json`` (tagged with :data:`METADATA_JSON_MARKER`), never in
  the data file. No ``ftr`` column is written; the saved filter specs rebuild
  the filter state on load.
- A dataset that already has a physical data file only needs the metadata
  sidecar (the raw data is already on disk). A dataset with no file (``.msr``
  extract, duplicate) also writes the data. The marked MINFLUX Viewer ``.zarr``
  format is the exception: it stores canonical raw data and processing state in
  separate groups inside one self-contained directory and writes no sidecar.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .acquisition_time import acquisition_metadata

from . import formats as _formats

#: Tag identifying the metadata sidecar JSON (a dict; filter/data JSON are lists).
METADATA_JSON_MARKER = "minflux_viewer_metadata"

# Derived from the one registry (:mod:`minflux_viewer.core.formats`) so the
# writers, the save dialog, Preferences and the drag-and-drop router cannot
# disagree about which formats exist.
DATA_FORMATS = tuple(spec.key for spec in _formats.save_formats())
_EXT = {spec.key: spec.extensions[0] for spec in _formats.save_formats()}
#: Formats that only carry the **canonical raw** data (no processed snapshot).
_RAW_ONLY_FORMATS = _formats.raw_only_formats()
#: Formats that take a flat columns dict (rest take the structured mfx array).
#: ``.zarr`` is deliberately absent: it is the marked, self-contained
#: application store written by :mod:`minflux_viewer.core.minflux_zarr`, not a
#: bag of flat columns.
_COLUMN_FORMATS = {"csv"}
#: Derived attributes offered in a processed snapshot when "freeze derived" is on.
_DERIVED_FOR_SNAPSHOT = ("den", "siz", "dst", "spd", "dt", "dur", "len", "tim_trace")

#: Spatial vector bases that the parser split into _x/_y/_z components.
_VEC_BASES = {"loc": ("x", "y", "z"), "lnc": ("x", "y", "z"), "ext": ("x", "y", "z")}

#: Keys never written into the canonical raw data (filter mask + derived attrs).
_EXCLUDE_KEYS = {
    "ftr", "den", "siz", "dst", "dur", "len", "spd", "dt", "tim_trace", "idx",
    # `loc_id` is an internal cache: `loader._raw_loc_id` lazily writes it into
    # `mfx_raw` the first time anything groups raw rows by localization (applying
    # a filter is enough). Without this exclusion it was exported as though it
    # were acquisition data, so the columns in a saved file depended on what the
    # user had clicked beforehand and two saves of the same dataset differed.
    "loc_id",
}


# ---------------------------------------------------------------------------
# Sidecar discrimination
# ---------------------------------------------------------------------------
def is_metadata_json_payload(payload: Any) -> bool:
    """True for a metadata sidecar payload (a dict tagged with the marker)."""
    return isinstance(payload, dict) and METADATA_JSON_MARKER in payload


def is_metadata_json_file(path) -> bool:
    """True if *path* is a metadata sidecar written by :func:`save_processed`."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return is_metadata_json_payload(json.load(fh))
    except Exception:
        return False


def metadata_sidecar_path(data_path) -> Path:
    """The ``<stem>_metadata.json`` path that accompanies *data_path*."""
    p = Path(data_path)
    return p.with_name(p.stem + "_metadata.json")


# ---------------------------------------------------------------------------
# Canonical raw mfx reconstruction (Abberior / m2410 layout)
# ---------------------------------------------------------------------------
def dataset_to_mfx_array(ds) -> np.ndarray:
    """Reconstruct the canonical raw ``mfx`` (all iterations) as an m2410-style
    structured array, recombining the split components back into vectors.

    ``loc_x/y/z → loc (N×3)``, ``lnc/ext`` likewise, ``dcr_0/dcr_1 → dcr (N×2)``;
    every other 1-D field is kept as a scalar column. ``itr`` stays flat, so the
    result loads through the normal m2410 path.
    """
    raw = getattr(ds, "mfx_raw", None)
    if raw is None or len(raw) == 0:
        # Datasets built directly from coordinates (build_localization_dataset,
        # flat-table re-import) have no raw store — fall back to the materialized
        # attr store, which carries the same split components.
        raw = ds.attr
    if raw is None or len(raw) == 0:
        raise ValueError("Dataset has no data to save.")
    # Never write the filter mask or recomputed-on-load derived attributes into
    # the canonical raw data — gating/derived state lives in the metadata recipe.
    keys = set(raw.keys()) - _EXCLUDE_KEYS
    n = 0
    for probe in ("loc_x", "itr", "vld", "tid"):
        v = raw.get(probe)
        if v is not None:
            n = len(np.asarray(v).ravel())
            break
    if n == 0:
        n = len(np.asarray(next(iter(raw.values()))).ravel())

    fields: list[tuple] = []
    fill: dict[str, np.ndarray] = {}
    used: set[str] = set()

    for base, sfx in _VEC_BASES.items():
        comps = [f"{base}_{s}" for s in sfx]
        if all(c in keys for c in comps):
            cols = [np.asarray(raw[c]).ravel() for c in comps]
            if all(c.size == n for c in cols):
                fields.append((base, cols[0].dtype, (3,)))
                fill[base] = np.column_stack(cols)
                used.update(comps)

    if "dcr_0" in keys and "dcr_1" in keys:
        a = np.column_stack([np.asarray(raw["dcr_0"]).ravel(),
                             np.asarray(raw["dcr_1"]).ravel()])
        if a.shape[0] == n:
            fields.append(("dcr", a.dtype, (2,)))
            fill["dcr"] = a
            used.update({"dcr_0", "dcr_1", "dcr"})

    for key in sorted(keys - used):
        v = np.asarray(raw[key]).ravel()
        if v.ndim == 1 and v.size == n:
            fields.append((key, v.dtype))
            fill[key] = v

    # The m2410 parser requires a flat 'itr'; coordinate-built datasets (attr
    # fallback) may lack itr/vld — synthesize a single-iteration, all-valid store.
    if "itr" not in fill:
        fields.append(("itr", np.int32))
        fill["itr"] = np.zeros(n, dtype=np.int32)
    if "vld" not in fill:
        fields.append(("vld", np.uint8))
        fill["vld"] = np.ones(n, dtype=np.uint8)

    arr = np.zeros(n, dtype=np.dtype(fields))
    for key, v in fill.items():
        arr[key] = v
    return arr


# ---------------------------------------------------------------------------
# Flat columns (CSV) and the processed snapshot
# ---------------------------------------------------------------------------
def flatten_mfx_array(mfx: np.ndarray) -> dict[str, np.ndarray]:
    """Structured ``mfx`` → ``{column: 1-D array}``, splitting vector fields with
    the loader-recognized suffixes (``loc → loc_x/loc_y/loc_z``,
    ``dcr → dcr_0/dcr_1``) so a flat CSV/columns export reloads with no new parser.
    The split/join convention here is the inverse of :func:`dataset_to_mfx_array`.
    """
    cols: dict[str, np.ndarray] = {}
    for name in mfx.dtype.names:
        v = np.asarray(mfx[name])
        if v.ndim == 1:
            cols[name] = v
            continue
        k = v.shape[1]
        if name in _VEC_BASES and k == 3:
            sfx = _VEC_BASES[name]
        elif name == "dcr" and k == 2:
            sfx = ("0", "1")
        else:
            sfx = tuple(str(i) for i in range(k))
        for i, s in enumerate(sfx):
            cols[f"{name}_{s}"] = v[:, i]
    return cols


def columns_to_structured(columns: dict[str, np.ndarray]) -> np.ndarray:
    """A ``{column: 1-D array}`` dict → a flat structured array (one field each).
    Used so the .mat/.npy/.json writers can serialize a snapshot table."""
    if not columns:
        return np.zeros(0, dtype=np.dtype([("xnm", np.float64)]))
    n = len(np.asarray(next(iter(columns.values()))).ravel())
    dt = [(str(k), np.asarray(v).dtype) for k, v in columns.items()]
    arr = np.zeros(n, dtype=np.dtype(dt))
    for k, v in columns.items():
        arr[str(k)] = np.asarray(v).ravel()
    return arr


def columns_to_mfx_array(columns: dict[str, np.ndarray]) -> np.ndarray:
    """Recompose flat canonical columns into a structured m2410 ``mfx`` array.

    ``flatten_mfx_array`` is used by CSV/NPZ/Zarr exports.  The ordinary
    structured-array readers can consume ``loc``/``dcr`` vectors, but cannot
    infer those vectors from component columns on their own.  Keep this inverse
    next to the writer so Zarr (and any future flat-column reader) has one
    canonical recomposition rule.
    """
    if not columns:
        raise ValueError("The flat export contains no columns.")

    ordered = {str(k): np.asarray(v).ravel() for k, v in columns.items()}
    n = len(next(iter(ordered.values())))
    if any(values.size != n for values in ordered.values()):
        raise ValueError("Flat export columns do not have a common row count.")

    fields: list[tuple] = []
    fill: dict[str, np.ndarray] = {}
    used: set[str] = set()

    for base, suffixes in _VEC_BASES.items():
        names = [f"{base}_{suffix}" for suffix in suffixes]
        if all(name in ordered for name in names):
            values = [ordered[name] for name in names]
            fields.append((base, np.result_type(*[value.dtype for value in values]), (3,)))
            fill[base] = np.column_stack(values)
            used.update(names)

    dcr_names = ("dcr_0", "dcr_1")
    if all(name in ordered for name in dcr_names):
        values = [ordered[name] for name in dcr_names]
        fields.append(("dcr", np.result_type(*[value.dtype for value in values]), (2,)))
        fill["dcr"] = np.column_stack(values)
        used.update(dcr_names)

    for name, values in ordered.items():
        if name in used:
            continue
        fields.append((name, values.dtype))
        fill[name] = values

    arr = np.zeros(n, dtype=np.dtype(fields))
    for name, values in fill.items():
        arr[name] = values
    return arr


def build_snapshot_table(
    ds, *, include_attrs: bool = True, include_derived: bool = False,
    filter_mode: str = "flag",
) -> tuple[dict[str, np.ndarray], int]:
    """Build the **processed snapshot** of the current view: the last-valid
    selection with Z scaling and transform baked into ``xnm/ynm/znm`` (canonical
    "baked XOR recipe" rule), the standard attributes, optionally frozen derived
    attributes, and the filter applied (rows dropped) or flagged (``ftr`` column).

    Returns ``(columns, rows_dropped)``. Coordinates are in nm.
    """
    from . import overlay as _ov
    from .loader import mfx_filter_mask, mfx_get

    xnm = mfx_get(ds, "xnm", itr="last", vld_only=True)
    ynm = mfx_get(ds, "ynm", itr="last", vld_only=True)
    if xnm is None or ynm is None:
        raise ValueError("Dataset has no coordinates to snapshot.")
    xnm = np.asarray(xnm, dtype=float)
    ynm = np.asarray(ynm, dtype=float)
    znm = mfx_get(ds, "znm", itr="last", vld_only=True)
    znm = np.asarray(znm, dtype=float) if znm is not None else np.zeros_like(xnm)
    n = xnm.size

    # Bake the per-dataset display transform into the (already Z-scaled) coords.
    transform = ds.state.get("overlay_transform") or ds.state.get("render_transform_2d")
    pts = _ov.apply_display_transform_nm(np.column_stack([xnm, ynm, znm]), transform)
    cols: dict[str, np.ndarray] = {"xnm": pts[:, 0], "ynm": pts[:, 1], "znm": pts[:, 2]}

    raw = getattr(ds, "mfx_raw", None)
    source_keys = set(raw.keys()) if (raw is not None and len(raw)) else set(ds.attr.keys())
    if include_attrs:
        skip = _EXCLUDE_KEYS | {"loc_x", "loc_y", "loc_z", "itr", "vld"}
        for name in sorted(source_keys - skip):
            v = mfx_get(ds, name, itr="last", vld_only=True)
            if v is None:
                continue
            v = np.asarray(v)
            if v.ndim == 1 and v.size == n:
                cols[name] = v
    if include_derived:
        for name in _DERIVED_FOR_SNAPSHOT:
            v = mfx_get(ds, name, itr="last", vld_only=True)
            if v is None:
                # Local-density and plugin results can live only in ``derived``
                # while still aligning exactly to the materialized last-valid
                # rows. Include them in a frozen snapshot when that contract is
                # explicit from their length.
                candidate = ds.derived.get(name)
                if candidate is not None:
                    candidate = np.asarray(candidate)
                    if candidate.ndim == 1 and candidate.size == n:
                        v = candidate
            if v is None:
                continue
            v = np.asarray(v)
            if v.ndim == 1 and v.size == n:
                cols[name] = v

    rows_dropped = 0
    fm = mfx_filter_mask(ds, itr="last", vld_only=True)
    mask = np.asarray(fm[0], dtype=bool) if fm and fm[0] is not None else None
    if mask is not None and mask.size == n and not mask.all():
        if filter_mode == "apply":
            rows_dropped = int((~mask).sum())
            cols = {k: v[mask] for k, v in cols.items()}
        elif filter_mode == "flag":
            cols["ftr"] = mask
    elif filter_mode == "flag" and mask is not None and mask.size == n:
        cols["ftr"] = mask          # all-True ftr still records "nothing filtered"
    return cols, rows_dropped


# ---------------------------------------------------------------------------
# Metadata recipe
# ---------------------------------------------------------------------------
def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def roi_records_to_json(records) -> list[dict]:
    """ROI records as plain JSON dicts.

    One representation shared by the Zarr v2 store, the OME-Zarr v3 package and
    this sidecar, so a ROI means the same thing wherever it is written.
    """
    import dataclasses as _dc

    out = []
    for record in records or ():
        if _dc.is_dataclass(record) and not isinstance(record, type):
            out.append(_sanitize(_dc.asdict(record)))
        elif isinstance(record, dict):
            out.append(_sanitize(record))
        else:
            raise TypeError(f"Unsupported ROI record type: {type(record).__name__}")
    return out


def build_metadata(ds, *, data_filename: str | None = None,
                   content: str = "raw", snapshot: dict | None = None,
                   roi_records=None) -> dict:
    """Build the processing-recipe sidecar: source, gating, calibration, filters.

    For ``content="raw"`` (default) nothing is baked — ``z_scaling_factor``/``transform``/
    ``filters`` are recorded so reload reproduces the processed state. For
    ``content="snapshot"`` the data file already has Z scaling factor + transform + filter
    baked in, so the **applied** ``z_scaling_factor`` is pinned to ``1.0`` and ``transform``/
    ``filters`` are cleared (the baked-XOR-recipe rule); the original values are
    kept under ``*_baked`` keys for provenance only.
    """
    md = ds.metadata
    transform = ds.state.get("overlay_transform") or ds.state.get("render_transform_2d")
    z_scaling_factor = float(getattr(ds.cali, "z_scaling_factor", 1.0) or 1.0)
    filters = ds.state.get("filter_specs") or []
    meta = {
        METADATA_JSON_MARKER: 1,
        "content": content,
        "name": getattr(ds, "name", None),
        "data_file": data_filename,
        "original_source_version": md.get("source_version"),
        "source_format": md.get("source_format"),
        # Identity, so a sidecar handed over on its own can find the dataset it
        # came from (core/metadata_match.py). The DID is Imspector's per-run
        # UUID, and a .mat/.npy/.csv/.json data file cannot carry it -- this
        # sidecar is the only route by which it reaches such a dataset. Written
        # only when known, so it never claims an identity the source lacked.
        **({"msr_dataset_did": str(md["msr_dataset_did"])}
           if md.get("msr_dataset_did") else {}),
        # When the instrument recorded the run. A provenance fact, so it is
        # carried on BOTH raw and snapshot content — unlike z_scaling_factor/transform/
        # filters, there is nothing here to bake, hence nothing to un-bake.
        "acquisition": acquisition_metadata(ds),
        "gating": {
            "iteration_load_mode": md.get("iteration_load_mode"),
            "includes_invalid": bool(md.get("includes_invalid", False)),
        },
        "calibration": {
            "z_scaling_factor": z_scaling_factor,
            "z_scaling_factor_provenance": ds.z_scaling_factor_provenance,
        },
        "transform": transform,
        "filters": filters,
        # ROIs are selections over the data, not a transformation of it, so they
        # are carried on snapshot content too — a baked export still has the
        # same regions marked on it.
        "rois": roi_records_to_json(
            roi_records if roi_records is not None
            else ds.metadata.get("minflux_viewer_roi_records")),
    }
    if content == "snapshot":
        # Everything is baked into the data; reload must NOT re-apply it.
        meta["calibration"]["z_scaling_factor"] = 1.0
        meta["calibration"]["baked_z_scaling_factor"] = z_scaling_factor
        meta["transform"] = None
        meta["transform_baked"] = transform
        meta["filters"] = []
        meta["filter_specs_baked"] = filters
        if snapshot:
            meta["snapshot"] = snapshot
    return _sanitize(meta)


# ---------------------------------------------------------------------------
# Data writers (Abberior-style structured mfx)
# ---------------------------------------------------------------------------
def _write_mat(path: Path, mfx: np.ndarray, *, root: str = "mfx") -> None:
    from scipy.io import savemat

    # A single MATLAB struct with length-N array fields (Abberior 'data_new'
    # layout) — NOT a length-N struct array (which a structured ndarray becomes).
    fields = {name: mfx[name] for name in mfx.dtype.names}
    savemat(str(path), {str(root): fields}, do_compression=True, long_field_names=True)


def _write_npy(path: Path, mfx: np.ndarray) -> None:
    np.save(str(path), mfx, allow_pickle=False)


def _write_json(path: Path, mfx: np.ndarray) -> None:
    names = mfx.dtype.names
    # Keep the File > Save JSON shape (a JSON array of records), but stream one
    # record at a time.  Building a Python dict/list for every row made a large
    # MSR export consume many times the raw data size before it reached disk.
    with path.open("w", encoding="utf-8") as handle:
        handle.write("[")
        first_chunk = True
        chunk: list[dict] = []
        for row in mfx:
            record = {}
            for key in names:
                value = row[key]
                record[key] = value.tolist() if isinstance(value, np.ndarray) else (
                    value.item() if isinstance(value, np.generic) else value)
            chunk.append(record)
            if len(chunk) < 4096:
                continue
            if not first_chunk:
                handle.write(",")
            encoded = json.dumps(chunk, ensure_ascii=False, separators=(",", ":"))
            handle.write(encoded[1:-1])
            first_chunk = False
            chunk.clear()
        if chunk:
            if not first_chunk:
                handle.write(",")
            encoded = json.dumps(chunk, ensure_ascii=False, separators=(",", ":"))
            handle.write(encoded[1:-1])
        handle.write("]")


_WRITERS = {"mat": _write_mat, "npy": _write_npy, "json": _write_json}


# --- flat-columns writers (CSV) --------------------------------------------
def _write_csv(path: Path, columns: dict[str, np.ndarray]) -> None:
    names = list(columns)
    if not names:
        path.write_text("", encoding="utf-8")
        return
    arrays = {name: np.asarray(columns[name]).ravel() for name in names}
    if all(
        np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.bool_)
        for values in arrays.values()
    ):
        # Numeric canonical exports are common and can be written much faster
        # in bounded chunks than one csv.writer call per row.  Per-column
        # formats keep integer/bool fields textual-exact while floats retain
        # enough precision for a lossless reload.
        formats = [
            "%d" if np.issubdtype(arrays[name].dtype, np.integer) or
            np.issubdtype(arrays[name].dtype, np.bool_) else "%.17g"
            for name in names
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(",".join(names) + "\n")
            n = len(arrays[names[0]])
            for start in range(0, n, 100_000):
                stop = min(start + 100_000, n)
                matrix = np.column_stack([arrays[name][start:stop] for name in names])
                np.savetxt(handle, matrix, delimiter=",", fmt=formats)
        return
    # Keep integer/bool columns textual-exact.  The previous float conversion
    # made CSV a lossy format for large integer identifiers and timestamps.
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(names)
        n = len(arrays[names[0]])
        values = arrays
        for row_idx in range(n):
            row = []
            for name in names:
                value = values[name][row_idx]
                row.append(value.item() if isinstance(value, np.generic) else value)
            writer.writerow(row)


def spreadsheet_export_columns(ds) -> dict[str, np.ndarray]:
    """Return selectable columns for a processed spreadsheet export.

    The table uses the current filtered, last-valid view. Coordinates are
    Z scaling factor/overlay-transform aware and expressed in nanometres.
    """
    from .loader import mfx_filter_mask, mfx_get

    columns, _dropped = build_snapshot_table(
        ds,
        include_attrs=True,
        include_derived=True,
        filter_mode="apply",
    )
    full_n = np.asarray(mfx_get(ds, "xnm", itr="last", vld_only=True)).size
    fm = mfx_filter_mask(ds, itr="last", vld_only=True)
    mask = np.asarray(fm[0], dtype=bool) if fm and fm[0] is not None else None
    for name in ("itr", "vld", "idx"):
        values = mfx_get(ds, name, itr="last", vld_only=True)
        if values is None:
            continue
        values = np.asarray(values)
        if values.ndim != 1 or values.size != full_n:
            continue
        if mask is not None and mask.size == full_n:
            values = values[mask]
        expected_n = len(next(iter(columns.values()))) if columns else values.size
        if values.size == expected_n:
            columns[name] = values
    return columns


def decode_csv_separator(value: str) -> str:
    """Decode a user-entered CSV delimiter (including the literal ``\\t``)."""
    text = str(value)
    separator = "\t" if text == r"\t" else text
    if len(separator) != 1 or separator in {"\r", "\n"}:
        raise ValueError("The separator must be one character (or \\t for a tab).")
    return separator


def write_spreadsheet_csv(
    ds,
    path: str | Path,
    *,
    column_headers: list[tuple[str, str]],
    separator: str = ",",
) -> Path:
    """Write selected processed attributes with editable column headers."""
    delimiter = decode_csv_separator(separator)
    if not column_headers:
        raise ValueError("Select at least one attribute to export.")

    columns = spreadsheet_export_columns(ds)
    selected: list[tuple[str, str, np.ndarray]] = []
    n_rows: int | None = None
    for attribute, header in column_headers:
        name = str(attribute)
        label = str(header).strip()
        if name not in columns:
            raise ValueError(f"Attribute {name!r} is not available for export.")
        if not label:
            raise ValueError(f"The header for {name!r} cannot be empty.")
        values = np.asarray(columns[name])
        if values.ndim != 1:
            raise ValueError(f"Attribute {name!r} is not a one-dimensional column.")
        if n_rows is None:
            n_rows = values.size
        elif values.size != n_rows:
            raise ValueError(f"Attribute {name!r} does not align with the export rows.")
        selected.append((name, label, values))

    output = Path(path)
    if output.suffix.lower() != ".csv":
        output = output.with_suffix(".csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter, lineterminator="\n")
        writer.writerow([header for _name, header, _values in selected])
        for row_idx in range(n_rows or 0):
            row = []
            for _name, _header, values in selected:
                value = values[row_idx]
                row.append(value.item() if isinstance(value, np.generic) else value)
            writer.writerow(row)
    return output


def _write_columns(fmt: str, path: Path, columns: dict[str, np.ndarray]) -> None:
    """Write a flat columns dict in *fmt*. Structured formats (.mat/.npy/.json)
    serialize a structured array built from the columns; .csv takes the
    columns directly. ``.zarr`` never reaches here — see ``_COLUMN_FORMATS``."""
    if fmt == "csv":
        _write_csv(path, columns)
    elif fmt in _WRITERS:
        _WRITERS[fmt](path, columns_to_structured(columns))
    else:
        raise ValueError(f"Unsupported data format: {fmt!r}")


def write_raw_array(
    path: str | Path,
    fmt: str,
    array: np.ndarray,
    *,
    root: str = "mfx",
) -> Path:
    """Write one structured raw component using the canonical save writers.

    ``mfx`` arrays passed to this helper must already be canonical flat m2410
    arrays.  The MSR reader uses this for the optional MBM companion, which is
    a separate structured component and therefore is not normalized as mfx.
    Keeping this small public adapter prevents the plugin from maintaining a
    second set of MATLAB/NumPy/JSON/CSV/Zarr serializers.

    ``root`` affects MATLAB only (for example ``"mbm"`` for an MBM companion);
    column formats remain flat by design.
    """
    fmt = str(fmt).lower().lstrip(".")
    if fmt not in _EXT:
        raise ValueError(f"Unsupported data format: {fmt!r}")
    if fmt in {"msr", "zarr"}:
        raise ValueError(
            f"The .{fmt} format must be written from a dataset so acquisition, "
            "MBM/search metadata and processing state are preserved."
        )

    arr = np.asarray(array)
    if arr.dtype.names is None:
        raise ValueError("Raw export components must be one-dimensional structured arrays.")
    if arr.ndim != 1:
        raise ValueError(f"Raw export components must be one-dimensional; got shape {arr.shape}.")

    output = Path(path).with_suffix(_EXT[fmt])
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt in _COLUMN_FORMATS:
        _write_columns(fmt, output, flatten_mfx_array(arr))
    elif fmt == "mat":
        _write_mat(output, arr, root=root)
    elif fmt == "npy":
        _write_npy(output, arr)
    elif fmt == "json":
        _write_json(output, arr)
    else:  # pragma: no cover - guarded by _EXT above
        raise ValueError(f"Unsupported raw export format: {fmt!r}")
    return output


# --- Picasso localization HDF5 ---------------------------------------------
def _scale_precision_to_nm(values: np.ndarray) -> np.ndarray:
    """Conservatively infer metres vs nanometres for precision-like columns."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    finite = np.abs(arr[np.isfinite(arr)])
    median = float(np.median(finite)) if finite.size else 1.0
    return arr * (1.0e9 if median < 1.0e-3 else 1.0)


def _first_column(columns: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        value = columns.get(name)
        if value is not None:
            return np.asarray(value).ravel()
    return None


def _picasso_precision_nm(columns: dict[str, np.ndarray], axis: str, n: int) -> np.ndarray:
    if axis == "x":
        names = ("lpx_nm", "precision_x_nm", "sigma_x_nm", "sx_nm", "loc_precision_x")
    elif axis == "y":
        names = ("lpy_nm", "precision_y_nm", "sigma_y_nm", "sy_nm", "loc_precision_y")
    else:
        names = ("lpz_nm", "precision_z_nm", "sigma_z_nm", "sz_nm", "loc_precision_z")
    values = _first_column(columns, names)
    if values is None and axis in {"x", "y"}:
        values = _first_column(columns, ("loc_precision_xy", "precision_xy_nm", "sigma_xy_nm"))
    if values is None:
        return np.ones(n, dtype=np.float64)
    arr = _scale_precision_to_nm(values)
    if arr.size != n:
        return np.ones(n, dtype=np.float64)
    arr[~np.isfinite(arr) | (arr <= 0.0)] = 1.0
    return arr


def _picasso_frame(columns: dict[str, np.ndarray], n: int) -> np.ndarray:
    tim = columns.get("tim")
    if tim is not None:
        t = np.asarray(tim, dtype=np.float64).ravel()
        if t.size == n and np.any(np.isfinite(t)):
            order = np.argsort(np.where(np.isfinite(t), t, np.inf), kind="stable")
            frame = np.empty(n, dtype=np.uint64)
            frame[order] = np.arange(n, dtype=np.uint64)
            return frame
    return np.arange(n, dtype=np.uint64)


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _write_simple_yaml_documents(path: Path, docs: list[dict[str, Any]]) -> None:
    chunks = []
    for doc in docs:
        lines = ["---"]
        for key, value in doc.items():
            lines.append(f"{key}: {_yaml_scalar(_sanitize(value))}")
        chunks.append("\n".join(lines))
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def write_picasso_hdf5(
    ds,
    path: str | Path,
    *,
    pixel_size_nm: float = 1.0,
) -> list[Path]:
    """Write the active, filtered dataset as a Picasso Render localization file.

    Picasso expects an HDF5 table at ``/locs`` plus a same-stem ``.yaml`` sidecar.
    The x/y columns are in camera pixels, while z is in nm. For MINFLUX data we
    define a virtual camera pixel size (default 1 nm), shift x/y to a positive
    field-of-view origin, and record the original nm origin in the YAML sidecar.
    """
    import h5py

    pixel_size_nm = float(pixel_size_nm)
    if not np.isfinite(pixel_size_nm) or pixel_size_nm <= 0.0:
        raise ValueError("pixel_size_nm must be a positive finite value")

    columns, _dropped = build_snapshot_table(
        ds, include_attrs=True, include_derived=False, filter_mode="apply"
    )
    xnm = np.asarray(columns.get("xnm"), dtype=np.float64).ravel()
    ynm = np.asarray(columns.get("ynm"), dtype=np.float64).ravel()
    znm = np.asarray(columns.get("znm", np.zeros_like(xnm)), dtype=np.float64).ravel()
    finite = np.isfinite(xnm) & np.isfinite(ynm) & np.isfinite(znm)
    if not np.any(finite):
        raise ValueError("No finite localizations to export.")
    xnm, ynm, znm = xnm[finite], ynm[finite], znm[finite]
    n = xnm.size

    x0 = float(np.min(xnm))
    y0 = float(np.min(ynm))
    x_px = (xnm - x0) / pixel_size_nm
    y_px = (ynm - y0) / pixel_size_nm
    width = int(np.ceil(float(np.max(x_px))) + 1)
    height = int(np.ceil(float(np.max(y_px))) + 1)
    width = max(width, 1)
    height = max(height, 1)
    from .dataset_kind import is_3d as dataset_is_3d
    is_3d = bool(dataset_is_3d(ds))

    lpx = _picasso_precision_nm(columns, "x", len(finite))[finite] / pixel_size_nm
    lpy = _picasso_precision_nm(columns, "y", len(finite))[finite] / pixel_size_nm
    lpz = _picasso_precision_nm(columns, "z", len(finite))[finite]
    photons = _first_column(columns, ("photons", "eco", "efo"))
    if photons is None or photons.size != len(finite):
        photons_out = np.ones(n, dtype=np.float32)
    else:
        photons_out = np.asarray(photons, dtype=np.float64)[finite]
        photons_out[~np.isfinite(photons_out) | (photons_out < 0.0)] = 0.0
    bg = _first_column(columns, ("bg", "fbg"))
    bg_out = np.zeros(n, dtype=np.float32) if bg is None or bg.size != len(finite) else np.asarray(bg, dtype=np.float64)[finite]
    bg_out[~np.isfinite(bg_out) | (bg_out < 0.0)] = 0.0
    tid = _first_column(columns, ("tid",))
    group = None if tid is None or tid.size != len(finite) else np.asarray(tid)[finite]

    fields = [
        ("frame", np.uint64),
        ("x", np.float32),
        ("y", np.float32),
        ("photons", np.float32),
        ("lpx", np.float32),
        ("lpy", np.float32),
        ("bg", np.float32),
    ]
    if is_3d:
        fields.extend([("z", np.float32), ("lpz", np.float32)])
    if group is not None:
        fields.append(("group", np.int64))
    locs = np.zeros(n, dtype=np.dtype(fields))
    locs["frame"] = _picasso_frame(columns, len(finite))[finite]
    locs["x"] = x_px.astype(np.float32)
    locs["y"] = y_px.astype(np.float32)
    locs["photons"] = photons_out.astype(np.float32)
    locs["lpx"] = lpx.astype(np.float32)
    locs["lpy"] = lpy.astype(np.float32)
    locs["bg"] = bg_out.astype(np.float32)
    if is_3d:
        locs["z"] = znm.astype(np.float32)
        locs["lpz"] = lpz.astype(np.float32)
    if group is not None:
        locs["group"] = group.astype(np.int64, copy=False)

    path = Path(path).with_suffix(".hdf5")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as h5:
        h5.create_dataset("locs", data=locs)

    yaml_path = path.with_suffix(".yaml")
    _write_simple_yaml_documents(
        yaml_path,
        [{
            "Width": width,
            "Height": height,
            "Frames": int(np.max(locs["frame"])) + 1 if n else 1,
            "Pixelsize": pixel_size_nm,
            "Generated by": "MINFLUX Viewer Picasso HDF5 export",
            "Source dataset": getattr(ds, "name", ""),
            "MINFLUX Viewer x_origin_nm": x0,
            "MINFLUX Viewer y_origin_nm": y0,
            "MINFLUX Viewer z_unit": "nm",
        }],
    )
    return [path, yaml_path]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def save_processed(
    ds,
    *,
    data_path: str | Path | None = None,
    fmt: str | None = None,
    metadata_dir: str | Path | None = None,
    content: str = "raw",
    include: dict | None = None,
    filter_mode: str = "flag",
    related_datasets=None,
    roi_records=None,
    image_specs=None,
    project_name: str | None = None,
    zarr_overwrite: str = "replace",
) -> list[Path]:
    """Save *ds* to file; returns the written paths.

    Parameters
    ----------
    data_path, fmt :
        Write the data file. ``fmt`` is one of :data:`DATA_FORMATS`. Omit
        ``data_path`` for a metadata-only save (file-backed raw already on disk).
    content :
        ``"raw"`` — canonical all-iteration ``mfx`` (m2410 layout); reloads and
        re-processes via the recipe. ``"snapshot"`` — the current view with Z scaling factor/
        transform/filter baked in (self-contained).
    include :
        ``{"attrs": bool, "derived": bool, "recipe": bool}`` (defaults
        True/False/True). ``recipe`` controls the ``_metadata.json`` sidecar.
    filter_mode :
        Snapshot only — ``"apply"`` drops filtered rows; ``"flag"`` keeps all and
        adds an ``ftr`` column.
    zarr_overwrite :
        Existing MINFLUX Viewer Zarr stores use ``"viewer"`` to update only
        processing/viewer groups after raw-data verification, ``"replace"`` to
        replace the complete store, or ``"error"`` to refuse overwriting.
    """
    include = include or {}
    inc_attrs = bool(include.get("attrs", True))
    inc_derived = bool(include.get("derived", False))
    inc_recipe = bool(include.get("recipe", True))
    written: list[Path] = []
    snapshot_meta: dict | None = None
    data_filename: str | None

    if data_path is not None:
        if fmt not in _EXT:
            raise ValueError(f"Unsupported data format: {fmt!r}")
        data_path = _formats.normalize_path(fmt, data_path)
        if fmt in _RAW_ONLY_FORMATS and content == "snapshot":
            raise ValueError(
                f"{_EXT[fmt]} stores canonical raw data plus separate processing state "
                "and cannot replace the raw acquisition with a baked snapshot — "
                "choose 'raw' content, or save the snapshot to another format.")
        if fmt == "zarr":
            # The new Zarr v2 format is one marked, self-contained dataset. It
            # intentionally does not use the old unmarked flat-column layout or
            # a sibling metadata JSON; the processing state lives inside it.
            from .minflux_zarr import write_minflux_zarr, write_minflux_zarr_project

            if zarr_overwrite not in {"viewer", "replace", "error"}:
                raise ValueError(f"Unsupported Zarr overwrite mode: {zarr_overwrite!r}")
            members = list(related_datasets or [ds])
            if len(members) > 1:
                data_path = write_minflux_zarr_project(
                    members,
                    data_path,
                    overwrite=zarr_overwrite != "error",
                    viewer_only=zarr_overwrite == "viewer",
                    roi_records=roi_records,
                    image_specs=image_specs,
                    name=project_name,
                )
            else:
                data_path = write_minflux_zarr(
                    members[0], data_path,
                    overwrite=zarr_overwrite != "error",
                    viewer_only=zarr_overwrite == "viewer",
                    roi_records=roi_records,
                    image_specs=image_specs,
                )
        elif fmt == "zarr_zip":
            # The same content as .zarr, sealed into one file. There is no
            # processing-only mode: a zip cannot replace a member, so saving
            # over a package always rewrites it (see write_minflux_zarr_package).
            from .minflux_zarr import write_minflux_zarr_package

            members = list(related_datasets or [ds])
            data_path = write_minflux_zarr_package(
                members if len(members) > 1 else members[0],
                data_path,
                overwrite=zarr_overwrite != "error",
                roi_records=roi_records,
                image_specs=image_specs,
                name=project_name,
            )
        elif fmt == "msr":
            # Our custom OBF/MFXDTA writer: a round-trippable .msr that reopens in
            # this viewer via the MSR reader (raw canonical mfx + any MBM beads).
            from ..msr.writer import write_datasets_msr
            write_datasets_msr(data_path, [ds])
        elif content == "snapshot":
            columns, dropped = build_snapshot_table(
                ds, include_attrs=inc_attrs, include_derived=inc_derived,
                filter_mode=filter_mode)
            _write_columns(fmt, data_path, columns)
            snapshot_meta = {"filter_mode": filter_mode, "rows_dropped": dropped,
                             "n_rows": len(next(iter(columns.values()))) if columns else 0}
        else:
            mfx = dataset_to_mfx_array(ds)
            write_raw_array(data_path, fmt, mfx)
        written.append(data_path)
        if fmt in {"zarr", "zarr_zip"}:
            return written
        sidecar = metadata_sidecar_path(data_path)
        data_filename = data_path.name
    else:
        # Metadata-only (file-backed): sidecar next to the existing data file.
        stem = Path(getattr(ds.file, "name", "dataset") or "dataset").stem or "dataset"
        folder = Path(metadata_dir or getattr(ds.file, "folder", "") or ".")
        sidecar = folder / f"{stem}_metadata.json"
        data_filename = getattr(ds.file, "name", None)
        content = "raw"          # nothing baked when only the recipe is written

    if inc_recipe:
        sidecar.write_text(
            json.dumps(build_metadata(ds, data_filename=data_filename,
                                      content=content, snapshot=snapshot_meta,
                                      roi_records=roi_records),
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written.append(sidecar)
    return written
