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

DATA_FORMATS = ("mat", "npy", "json")
_EXT = {"mat": ".mat", "npy": ".npy", "json": ".json"}

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


def build_metadata(ds, *, data_filename: str | None = None) -> dict:
    """Build the processing-recipe sidecar: source, gating, calibration, filters.

    Deliberately omits derived/dimensionality state (``num_dim`` is re-derived
    from the raw z by the same Z-check on load) and the ``ftr`` mask (the filter
    specs reproduce it).
    """
    md = ds.metadata
    transform = ds.state.get("overlay_transform") or ds.state.get("render_transform_2d")
    meta = {
        METADATA_JSON_MARKER: 1,
        "name": getattr(ds, "name", None),
        "data_file": data_filename,
        "original_source_version": md.get("source_version"),
        "source_format": md.get("source_format"),
        "gating": {
            "iteration_load_mode": md.get("iteration_load_mode"),
            "includes_invalid": bool(md.get("includes_invalid", False)),
        },
        "calibration": {
            "rimf": float(getattr(ds.cali, "RIMF", 1.0) or 1.0),
            "rimf_provenance": ds.rimf_provenance,
        },
        "transform": transform,
        "filters": ds.state.get("filter_specs") or [],
    }
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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def save_processed(
    ds,
    *,
    data_path: str | Path | None = None,
    fmt: str | None = None,
    metadata_dir: str | Path | None = None,
) -> list[Path]:
    """Save *ds* as processed data; returns the written file paths.

    - ``data_path`` + ``fmt`` set → write the canonical raw mfx (no-file case).
    - ``data_path`` omitted → metadata only (the raw data already exists on
      disk); the sidecar is written into ``metadata_dir`` (the source folder)
      as ``<source-stem>_metadata.json``.
    """
    written: list[Path] = []

    if data_path is not None:
        if fmt not in _WRITERS:
            raise ValueError(f"Unsupported data format: {fmt!r}")
        data_path = Path(data_path).with_suffix(_EXT[fmt])
        _WRITERS[fmt](data_path, dataset_to_mfx_array(ds))
        written.append(data_path)
        sidecar = metadata_sidecar_path(data_path)
        data_filename = data_path.name
    else:
        # Metadata-only (file-backed): sidecar next to the existing data file.
        stem = Path(getattr(ds.file, "name", "dataset") or "dataset").stem or "dataset"
        folder = Path(metadata_dir or getattr(ds.file, "folder", "") or ".")
        sidecar = folder / f"{stem}_metadata.json"
        data_filename = getattr(ds.file, "name", None)

    sidecar.write_text(
        json.dumps(build_metadata(ds, data_filename=data_filename), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    written.append(sidecar)
    return written
