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
- All **gating, calibration and filtering** live in a sidecar
  ``<stem>_metadata.json`` (tagged with :data:`METADATA_JSON_MARKER`), never in
  the data file. No ``ftr`` column is written; the saved filter specs rebuild
  the filter state on load.
- A dataset that already has a physical data file only needs the metadata
  sidecar (the raw data is already on disk). A dataset with no file (``.msr``
  extract, duplicate) also writes the data, as ``.mat`` / ``.npy`` / ``.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

#: Tag identifying the metadata sidecar JSON (a dict; filter/data JSON are lists).
METADATA_JSON_MARKER = "minflux_viewer_metadata"

DATA_FORMATS = ("mat", "npy", "npz", "json", "csv", "zarr")
_EXT = {"mat": ".mat", "npy": ".npy", "npz": ".npz", "json": ".json",
        "csv": ".csv", "zarr": ".zarr"}
#: Formats that take a flat columns dict (rest take the structured mfx array).
_COLUMN_FORMATS = {"csv", "npz", "zarr"}
#: Derived attributes offered in a processed snapshot when "freeze derived" is on.
_DERIVED_FOR_SNAPSHOT = ("den", "siz", "dst", "spd", "dt", "dur", "len", "tim_trace")

#: Spatial vector bases that the parser split into _x/_y/_z components.
_VEC_BASES = {"loc": ("x", "y", "z"), "lnc": ("x", "y", "z"), "ext": ("x", "y", "z")}

#: Keys never written into the canonical raw data (filter mask + derived attrs).
_EXCLUDE_KEYS = {
    "ftr", "den", "siz", "dst", "dur", "len", "spd", "dt", "tim_trace", "idx",
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
# Flat columns (CSV / npz / zarr) and the processed snapshot
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


def build_snapshot_table(
    ds, *, include_attrs: bool = True, include_derived: bool = False,
    filter_mode: str = "flag",
) -> tuple[dict[str, np.ndarray], int]:
    """Build the **processed snapshot** of the current view: the last-valid
    selection with RIMF- and transform-baked ``xnm/ynm/znm`` (canonical
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

    # Bake the per-dataset display transform into the (already RIMF-scaled) coords.
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


def build_metadata(ds, *, data_filename: str | None = None,
                   content: str = "raw", snapshot: dict | None = None) -> dict:
    """Build the processing-recipe sidecar: source, gating, calibration, filters.

    For ``content="raw"`` (default) nothing is baked — ``rimf``/``transform``/
    ``filters`` are recorded so reload reproduces the processed state. For
    ``content="snapshot"`` the data file already has RIMF + transform + filter
    baked in, so the **applied** ``rimf`` is pinned to ``1.0`` and ``transform``/
    ``filters`` are cleared (the baked-XOR-recipe rule); the original values are
    kept under ``*_baked`` keys for provenance only.
    """
    md = ds.metadata
    transform = ds.state.get("overlay_transform") or ds.state.get("render_transform_2d")
    rimf = float(getattr(ds.cali, "RIMF", 1.0) or 1.0)
    filters = ds.state.get("filter_specs") or []
    meta = {
        METADATA_JSON_MARKER: 1,
        "content": content,
        "name": getattr(ds, "name", None),
        "data_file": data_filename,
        "original_source_version": md.get("source_version"),
        "source_format": md.get("source_format"),
        "gating": {
            "iteration_load_mode": md.get("iteration_load_mode"),
            "includes_invalid": bool(md.get("includes_invalid", False)),
        },
        "calibration": {
            "rimf": rimf,
            "rimf_provenance": ds.rimf_provenance,
        },
        "transform": transform,
        "filters": filters,
    }
    if content == "snapshot":
        # Everything is baked into the data; reload must NOT re-apply it.
        meta["calibration"]["rimf"] = 1.0
        meta["calibration"]["baked_rimf"] = rimf
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
def _write_mat(path: Path, mfx: np.ndarray) -> None:
    from scipy.io import savemat

    # A single MATLAB struct with length-N array fields (Abberior 'data_new'
    # layout) — NOT a length-N struct array (which a structured ndarray becomes).
    fields = {name: mfx[name] for name in mfx.dtype.names}
    savemat(str(path), {"mfx": fields}, do_compression=True, long_field_names=True)


def _write_npy(path: Path, mfx: np.ndarray) -> None:
    np.save(str(path), mfx, allow_pickle=False)


def _write_json(path: Path, mfx: np.ndarray) -> None:
    names = mfx.dtype.names
    records = []
    for row in mfx:
        rec = {}
        for k in names:
            val = row[k]
            rec[k] = val.tolist() if isinstance(val, np.ndarray) else (
                val.item() if isinstance(val, np.generic) else val)
        records.append(rec)
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


_WRITERS = {"mat": _write_mat, "npy": _write_npy, "json": _write_json}


# --- flat-columns writers (CSV / npz / zarr) -------------------------------
def _write_csv(path: Path, columns: dict[str, np.ndarray]) -> None:
    names = list(columns)
    if not names:
        path.write_text("", encoding="utf-8")
        return
    data = np.column_stack([np.asarray(columns[k], dtype=float).ravel() for k in names])
    # %.10g keeps coordinate precision while staying compact; ints/bools round-trip
    # as their numeric value (the flat-table loader re-casts on read).
    np.savetxt(str(path), data, delimiter=",", header=",".join(names),
               comments="", fmt="%.10g")


def _write_npz(path: Path, columns: dict[str, np.ndarray]) -> None:
    np.savez(str(path), **{str(k): np.asarray(v) for k, v in columns.items()})


def _write_zarr(path: Path, columns: dict[str, np.ndarray]) -> None:
    import zarr

    store = zarr.DirectoryStore(str(path))
    root = zarr.group(store=store, overwrite=True)
    for k, v in columns.items():
        root.create_dataset(str(k), data=np.asarray(v))


def _write_columns(fmt: str, path: Path, columns: dict[str, np.ndarray]) -> None:
    """Write a flat columns dict in *fmt*. Structured formats (.mat/.npy/.json)
    serialize a structured array built from the columns; .csv/.npz/.zarr take the
    columns directly."""
    if fmt == "csv":
        _write_csv(path, columns)
    elif fmt == "npz":
        _write_npz(path, columns)
    elif fmt == "zarr":
        _write_zarr(path, columns)
    elif fmt in _WRITERS:
        _WRITERS[fmt](path, columns_to_structured(columns))
    else:
        raise ValueError(f"Unsupported data format: {fmt!r}")


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
) -> list[Path]:
    """Save *ds* to file; returns the written paths.

    Parameters
    ----------
    data_path, fmt :
        Write the data file. ``fmt`` is one of :data:`DATA_FORMATS`. Omit
        ``data_path`` for a metadata-only save (file-backed raw already on disk).
    content :
        ``"raw"`` — canonical all-iteration ``mfx`` (m2410 layout); reloads and
        re-processes via the recipe. ``"snapshot"`` — the current view with RIMF/
        transform/filter baked in (self-contained).
    include :
        ``{"attrs": bool, "derived": bool, "recipe": bool}`` (defaults
        True/False/True). ``recipe`` controls the ``_metadata.json`` sidecar.
    filter_mode :
        Snapshot only — ``"apply"`` drops filtered rows; ``"flag"`` keeps all and
        adds an ``ftr`` column.
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
        data_path = Path(data_path).with_suffix(_EXT[fmt])
        if content == "snapshot":
            columns, dropped = build_snapshot_table(
                ds, include_attrs=inc_attrs, include_derived=inc_derived,
                filter_mode=filter_mode)
            _write_columns(fmt, data_path, columns)
            snapshot_meta = {"filter_mode": filter_mode, "rows_dropped": dropped,
                             "n_rows": len(next(iter(columns.values()))) if columns else 0}
        else:
            mfx = dataset_to_mfx_array(ds)
            if fmt in _COLUMN_FORMATS:
                _write_columns(fmt, data_path, flatten_mfx_array(mfx))
            else:
                _WRITERS[fmt](data_path, mfx)
        written.append(data_path)
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
                                      content=content, snapshot=snapshot_meta),
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written.append(sidecar)
    return written
