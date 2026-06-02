"""
minflux_viewer.core.loader
==========================
Load and parse MINFLUX ``.mat`` files into a :class:`MinfluxDataset`.

Ported from MATLAB:

* ``arrange_MINFLUX_data_structure.m``
* ``data_parser_m2410.m``
* ``validate_raw_data.m``
* ``compute_extra_property.m``  (derived attributes section)

Supported formats
-----------------
``m2205``
    Abberior Imspector export prior to 2024.
    ``itr`` is a nested struct; ``loc`` lives inside it.

``m2410``
    Abberior Imspector export from 2024 onwards.
    ``itr`` is a flat integer array (per-localisation iteration index).

``m2205-unwrapped``
    Already-flattened variants produced by SMAP / Ries-lab export scripts.

Both MATLAB v5/v6 (``scipy.io``) and v7.3 HDF5 (``h5py``) files are
supported.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio

from .dataset import (
    AttrStore,
    Calibration,
    Channel,
    DataProp,
    FileInfo,
    MinfluxDataset,
    build_image_dataset,
    build_localization_dataset,
)

SPATIAL_COORD_FIELDS = {"loc", "lnc", "ext"}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(path: str | Path, prefs: dict | None = None) -> MinfluxDataset:
    """
    Load a MINFLUX ``.mat`` file and return a fully populated
    :class:`MinfluxDataset`.

    Parameters
    ----------
    path:
        Full path to the ``.mat`` file.
    prefs:
        Optional application-preferences dict
        (mirrors ``app.Prefs`` in MATLAB). Falls back to safe defaults
        when not provided.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be recognised as a MINFLUX dataset.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    prefs = prefs or {}
    data_prefs      = prefs.get("data", {})
    load_all_itr    = data_prefs.get("iter_load", "last") == "all"
    load_efc_cfr    = data_prefs.get("load_efc_cfr", True)
    load_all_dcr    = data_prefs.get("load_all_dcr", True)

    # 1. Read the raw .mat file
    raw = _load_mat(path)

    # 2. Detect format and parse into an AttrStore
    attrs, num_loc, num_itr = _parse_raw(
        raw,
        load_all_itr=load_all_itr,
        load_efc_cfr=load_efc_cfr,
    )
    if attrs is None:
        raise ValueError(
            f"'{path.name}' could not be parsed as a MINFLUX dataset. "
            "Check that it contains an 'itr' field."
        )

    # 3. File metadata
    mtime = path.stat().st_mtime
    file_info = FileInfo(
        name=path.name,
        folder=str(path.parent),
        datetime=datetime.fromtimestamp(mtime).strftime("%Y-%b-%d, %H:%M:%S"),
        raw_data=raw,
    )

    # 4. Derived scalar properties (trace structure, dimensionality)
    prop = _compute_properties(attrs, num_loc, num_itr, prefs=prefs)

    # 5. Derived per-localisation attributes
    attrs = _add_derived_attributes(attrs, prop, prefs=prefs)
    prop.attr_names = attrs.keys()

    # 6. Calibration defaults (populated further by the analysis functions)
    cali = Calibration()

    # 7. Channel
    channel = _build_channel(attrs, prop)

    dataset = MinfluxDataset(
        file=file_info,
        prop=prop,
        attr=attrs,
        cali=cali,
        channel=channel,
    )
    dataset.metadata.update(_source_metadata_from_raw(raw, load_all_itr=load_all_itr))
    dataset.metadata["valid_num_loc"] = int(prop.num_loc)
    dataset.metadata["includes_invalid"] = False
    return dataset


# ---------------------------------------------------------------------------
# .mat file loading
# ---------------------------------------------------------------------------

def _load_mat(path: Path) -> dict:
    """
    Load a ``.mat`` file, handling MATLAB v5/v6 and v7.3 (HDF5) formats.
    Returns a plain Python dict mapping field names → numpy arrays.
    """
    try:
        raw = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
        # Remove scipy bookkeeping keys
        raw = {k: v for k, v in raw.items() if not k.startswith("__")}
        # Some exporters wrap everything inside a single top-level struct
        if len(raw) == 1:
            inner = next(iter(raw.values()))
            if hasattr(inner, "_fieldnames"):
                raw = _mat_struct_to_dict(inner)
        return raw
    except NotImplementedError:
        # v7.3 HDF5 — scipy can't read these
        pass

    try:
        import h5py
    except ImportError:
        raise ImportError(
            "This file is in MATLAB v7.3 (HDF5) format. "
            "Install h5py to read it:  pip install h5py"
        ) from None

    with h5py.File(str(path), "r") as f:
        return _h5_to_dict(f)


def _mat_struct_to_dict(s) -> dict:
    """Recursively convert a scipy ``mat_struct`` to a plain dict."""
    d: dict = {}
    for fname in s._fieldnames:
        val = getattr(s, fname)
        if hasattr(val, "_fieldnames"):
            d[fname] = _mat_struct_to_dict(val)
        else:
            d[fname] = val
    return d


def _h5_to_dict(group) -> dict:
    """Recursively convert an h5py group to a dict of numpy arrays."""
    import h5py
    d: dict = {}
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            d[key] = _h5_to_dict(item)
        else:
            arr = item[()]
            if arr.ndim > 1:
                arr = arr.T   # HDF5 uses Fortran order; transpose to C
            d[key] = arr
    return d


# ---------------------------------------------------------------------------
# Format detection and dispatching
# ---------------------------------------------------------------------------

def _parse_raw(
    raw: dict,
    load_all_itr: bool = False,
    load_efc_cfr: bool = True,
) -> tuple[AttrStore | None, int, int]:
    """
    Detect which MINFLUX format ``raw`` uses and call the appropriate parser.

    Returns
    -------
    (AttrStore, num_loc, num_itr)  or  (None, 0, 0) on failure.
    """
    if "itr" not in raw:
        return None, 0, 0

    itr_val = raw["itr"]

    # m2410: flat integer array of per-localisation iteration indices
    if isinstance(itr_val, np.ndarray) and np.issubdtype(itr_val.dtype, np.integer):
        return _parse_m2410(raw)

    # m2205: nested struct inside itr
    if hasattr(itr_val, "_fieldnames") or isinstance(itr_val, dict):
        return _parse_m2205(raw, load_all_itr=load_all_itr, load_efc_cfr=load_efc_cfr)

    # m2205-unwrapped: loc already at top level, itr is numeric
    if "loc" in raw:
        return _parse_m2205_unwrapped(
            raw, load_all_itr=load_all_itr, load_efc_cfr=load_efc_cfr
        )

    return None, 0, 0


def _source_metadata_from_raw(raw: dict, *, load_all_itr: bool = False) -> dict[str, Any]:
    """Best-effort metadata used by the dataset info dialog."""
    version = _detect_raw_version(raw)
    detail_map = {
        "m2410": "m2410: all iteration value in N by 1 array",
        "m2205": "m2205: iteration in N by M array, N: number of locs, M: number of iterations",
        "legacy": "legacy: legacy version derived or older than m2205",
        "unidentified": "unidentified version: unidentified minflux data version",
    }
    meta: dict[str, Any] = {
        "source_version": version,
        "source_version_detail": detail_map.get(version, detail_map["unidentified"]),
        "raw_num_loc": _raw_localization_count(raw),
        "raw_num_itr": _raw_iteration_count(raw),
        "iteration_load_mode": "all" if load_all_itr else "last",
    }
    cfr_iter = _effective_iteration_column(raw, "cfr")
    efc_iter = _effective_iteration_column(raw, "efc") or _effective_iteration_column(raw, "efo")
    if cfr_iter is not None:
        meta["cfr_iteration"] = cfr_iter
    if efc_iter is not None:
        meta["efc_iteration"] = efc_iter
    return meta


def _detect_raw_version(raw: dict) -> str:
    itr_val = raw.get("itr")
    if itr_val is None:
        return "unidentified"
    if isinstance(itr_val, np.ndarray) and np.issubdtype(itr_val.dtype, np.integer):
        return "m2410"
    if hasattr(itr_val, "_fieldnames") or isinstance(itr_val, dict):
        return "m2205"
    if "loc" in raw:
        return "m2205"
    return "legacy"


def _raw_localization_count(raw: dict) -> int:
    itr_val = raw.get("itr")
    try:
        if hasattr(itr_val, "_fieldnames"):
            itr_val = _mat_struct_to_dict(itr_val)
        if isinstance(itr_val, dict):
            for key in ("itr", "loc", "vld"):
                if key in itr_val:
                    arr = np.asarray(itr_val[key])
                    break
            else:
                return 0
        else:
            arr = np.asarray(itr_val)
        if arr.ndim == 0:
            return 0
        if arr.ndim == 1:
            return int(arr.shape[0])
        if arr.ndim == 2:
            if arr.shape[0] == 1:
                return int(arr.shape[1])
            return int(arr.shape[0])
        return int(arr.shape[0])
    except Exception:
        return 0


def _raw_iteration_count(raw: dict) -> int:
    try:
        itr_val = raw.get("itr")
        if hasattr(itr_val, "_fieldnames"):
            itr_val = _mat_struct_to_dict(itr_val)
        if isinstance(itr_val, dict):
            for key in ("itr", "loc"):
                if key in itr_val:
                    arr = np.asarray(itr_val[key])
                    break
            else:
                return 1
        else:
            arr = np.asarray(itr_val)
        if arr.ndim == 1:
            if np.issubdtype(arr.dtype, np.integer):
                valid = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.number) else arr
                return int(np.nanmax(valid)) + 1 if valid.size else 1
            return 1
        if arr.ndim == 2:
            if arr.shape[0] == 1 or arr.shape[1] == 1:
                if np.issubdtype(arr.dtype, np.integer):
                    flat = arr.ravel()
                    return int(np.nanmax(flat)) + 1 if flat.size else 1
                return 1
            return int(arr.shape[1])
        if arr.ndim == 3:
            return int(arr.shape[1])
    except Exception:
        pass
    return 1


def _effective_iteration_column(raw: dict, field: str) -> int | None:
    """Return the 1-based iteration column selected for CFR/EFC-like arrays."""
    try:
        itr_val = raw.get("itr")
        merged = raw
        if hasattr(itr_val, "_fieldnames"):
            merged = {k: v for k, v in raw.items() if k != "itr"}
            merged.update(_mat_struct_to_dict(itr_val))
        elif isinstance(itr_val, dict):
            merged = {k: v for k, v in raw.items() if k != "itr"}
            merged.update(itr_val)
        arr = np.asarray(merged.get(field))
        itr_arr = np.asarray(merged.get("itr"))
        if arr.ndim != 2 or itr_arr.ndim < 2:
            return None
        num_loc, num_itr = itr_arr.shape if itr_arr.shape[0] != 1 else (itr_arr.shape[1], 1)
        if arr.shape[0] != num_loc and arr.shape[1] == num_loc:
            arr = arr.T
        if arr.shape[0] != num_loc or arr.shape[1] != num_itr:
            return None
        vld = np.asarray(merged.get("vld", np.ones(num_loc, bool)), dtype=bool).ravel()
        col = int(np.argmax(np.nansum(np.abs(arr[vld, :]), axis=0)))
        return col + 1
    except Exception:
        return None


# ---------------------------------------------------------------------------
# m2410 parser
# ---------------------------------------------------------------------------

def _parse_m2410(raw: dict) -> tuple[AttrStore, int, int]:
    """
    Flat Abberior m2410 format.

    Only the last (maximum) iteration index is kept; localizations at
    earlier iterations are discarded.

    Field layout in this format:
    * ``itr`` is a flat integer array of per-localisation iteration indices.
    * Most fields are scalar per localisation — shape (N,) or (1, N).
    * ``loc`` (and sometimes ``lnc``/``ext``) are xyz arrays — shape (N, 3)
      or (N, 2). Only known spatial coordinate fields are split into
      ``_x/_y/_z`` columns.
    * ``dcr`` may be stored as (N, 1) or (N, 2); for (N, 2), the first
      column is the channel-ratio value used by the viewer.
    """
    # "itr" is always present and always scalar-per-loc in m2410
    itr_raw = np.asarray(raw["itr"])
    itr     = itr_raw.ravel()   # handles (N,), (1, N), and (N, 1)
    n_raw   = itr.size

    vld     = np.asarray(
        raw.get("vld", np.ones(n_raw, dtype=bool)), dtype=bool,
    ).ravel()
    max_itr = int(itr[vld].max()) if vld.any() else 0
    num_itr = max_itr + 1

    mask    = vld & (itr == max_itr)
    num_loc = int(mask.sum())

    attrs   = AttrStore()

    for key, val in raw.items():
        arr = np.asarray(val)

        # Skip scalars, empty containers, and the HDF5 "#refs#" placeholder
        if arr.ndim == 0 or arr.size == 0 or key == "#refs#":
            continue

        # Flatten to canonical (N, …) shape where N == n_raw
        if arr.ndim == 1:
            if arr.shape[0] != n_raw:
                continue            # unrelated array
            arr2 = arr.reshape(n_raw, 1)
        elif arr.ndim == 2:
            if arr.shape[0] == n_raw:
                arr2 = arr
            elif arr.shape[1] == n_raw:
                arr2 = arr.T        # (C, N) → (N, C)
            else:
                continue
        elif arr.ndim == 3:
            # Rare, but handle (N, K, 3) defensively by picking the last K
            if arr.shape[0] == n_raw:
                arr2 = arr[:, -1, :]
            else:
                continue
        else:
            continue

        # Classify by semantic field name first. Only spatial coordinates are
        # split into x/y/z; non-spatial N x 2 fields such as dcr stay scalar.
        n_cols = arr2.shape[1] if arr2.ndim == 2 else 0

        sub = arr2[mask]

        if key in SPATIAL_COORD_FIELDS and n_cols in (2, 3):
            attrs[f"{key}_x"] = sub[:, 0].ravel()
            attrs[f"{key}_y"] = sub[:, 1].ravel()
            attrs[f"{key}_z"] = sub[:, 2].ravel() if n_cols == 3 else np.zeros(num_loc)
        elif key == "dcr" and n_cols > 1:
            attrs[key] = sub[:, 0].ravel()
        else:
            attrs[key] = sub.ravel() if n_cols == 1 else sub

    return attrs, num_loc, num_itr


# ---------------------------------------------------------------------------
# m2205 parser (nested itr struct)
# ---------------------------------------------------------------------------

def _parse_m2205(
    raw: dict,
    load_all_itr: bool = False,
    load_efc_cfr: bool = True,
) -> tuple[AttrStore, int, int]:
    """
    Nested Abberior m2205 format.  Unwraps the ``.itr`` sub-struct then
    delegates to :func:`_parse_m2205_unwrapped`.
    """
    itr_struct = raw["itr"]
    if hasattr(itr_struct, "_fieldnames"):
        itr_dict = _mat_struct_to_dict(itr_struct)
    elif isinstance(itr_struct, dict):
        itr_dict = itr_struct
    else:
        return None, 0, 0

    merged = {k: v for k, v in raw.items() if k != "itr"}
    merged.update(itr_dict)
    merged.pop("mbm", None)     # remove reference-point field

    return _parse_m2205_unwrapped(
        merged, load_all_itr=load_all_itr, load_efc_cfr=load_efc_cfr
    )


# ---------------------------------------------------------------------------
# m2205 unwrapped parser
# ---------------------------------------------------------------------------

def _parse_m2205_unwrapped(
    raw: dict,
    load_all_itr: bool = False,
    load_efc_cfr: bool = True,
) -> tuple[AttrStore, int, int]:
    """
    Already-unwrapped m2205 variant.

    ``itr`` is a numeric (num_loc × num_itr) array; ``loc`` is at the
    top level.
    """
    if "loc" not in raw:
        return None, 0, 0

    itr_arr = np.asarray(raw["itr"])

    # Determine num_loc and num_itr from itr shape
    if itr_arr.ndim == 1:
        num_loc, num_itr = itr_arr.shape[0], 1
    elif itr_arr.shape[0] == 1:
        num_loc, num_itr = itr_arr.shape[1], 1
        itr_arr = itr_arr.T
    else:
        num_loc, num_itr = itr_arr.shape

    vld          = np.asarray(raw.get("vld", np.ones(num_loc, bool)), dtype=bool).ravel()
    num_loc_vld  = int(vld.sum())
    iter_last    = num_itr - 1

    attrs = AttrStore()

    for key, val in raw.items():
        arr = np.asarray(val)

        # Skip scalars, empty containers, and the HDF5 "#refs#" placeholder
        if arr.ndim == 0 or arr.size == 0 or key == "#refs#":
            continue

        # Cast numeric fields to float (keep integer fields as-is)
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(float, copy=False)

        # Normalise to canonical shape with leading axis == num_loc.
        # Top-level scalar-per-loc fields can arrive as:
        #   (num_loc,)            – already correct
        #   (1, num_loc)          – needs transpose then squeeze
        #   (num_loc, 1)          – already correct, 1 column
        # Per-iteration fields:
        #   (num_loc, num_itr)    – already correct
        #   (num_itr, num_loc)    – needs transpose
        # xyz fields:
        #   (num_loc, num_itr, 3) – already correct
        #   (3, num_itr, num_loc) – needs axis reorder (HDF5 legacy)
        if arr.ndim == 1:
            if arr.shape[0] != num_loc:
                continue
            # keep 1-D; treated as scalar-per-loc below
        elif arr.ndim == 2:
            if arr.shape[0] == num_loc:
                pass            # OK
            elif arr.shape[1] == num_loc:
                arr = arr.T     # (C, N) → (N, C); handles (1, N) → (N, 1) too
            else:
                continue
        elif arr.ndim == 3:
            # xyz-per-iteration; accept (N, K, 3) or (3, K, N)
            if arr.shape[0] == num_loc and arr.shape[-1] in (2, 3):
                pass
            elif arr.shape[-1] == num_loc and arr.shape[0] in (2, 3):
                arr = np.moveaxis(arr, (0, -1), (-1, 0))  # (3,K,N) → (N,K,3)
            else:
                continue
        else:
            continue

        # ── spatial coordinate arrays ────────────────────────────────
        if key in SPATIAL_COORD_FIELDS and arr.ndim == 3 and arr.shape[-1] in (2, 3):
            if arr.shape[1] == num_itr:
                sub = arr[vld, iter_last, :]
            else:
                # single-iteration xyz stored as (N, K, 3) with K != num_itr
                sub = arr[vld, -1, :]
            n_cols = sub.shape[1]
            attrs[f"{key}_x"] = sub[:, 0].ravel()
            attrs[f"{key}_y"] = sub[:, 1].ravel()
            attrs[f"{key}_z"] = (
                sub[:, 2].ravel() if n_cols == 3 else np.zeros(num_loc_vld)
            )
            continue

        # ── spatial coordinate arrays, single iteration: (N, 2/3) ───
        if key in SPATIAL_COORD_FIELDS and arr.ndim == 2 and arr.shape[1] in (2, 3) and arr.shape[1] != num_itr:
            sub = arr[vld, :]
            n_cols = sub.shape[1]
            attrs[f"{key}_x"] = sub[:, 0].ravel()
            attrs[f"{key}_y"] = sub[:, 1].ravel()
            attrs[f"{key}_z"] = (
                sub[:, 2].ravel() if n_cols == 3 else np.zeros(num_loc_vld)
            )
            continue

        # ── Detection channel ratio: use first column if stored as N x 2.
        if key == "dcr" and arr.ndim == 2 and arr.shape[0] == num_loc and arr.shape[1] > 1:
            attrs[key] = arr[vld, 0].ravel()
            continue

        # ── Per-iteration attributes: (num_loc, num_itr) ─────────────
        if arr.ndim == 2 and arr.shape[1] == num_itr:
            if load_efc_cfr and key in ("cfr", "efc"):
                col = int(np.argmax(np.nansum(np.abs(arr[vld, :]), axis=0)))
                attrs[key] = arr[vld, col].ravel()
            elif load_all_itr:
                attrs[key] = arr[vld, :]
            else:
                attrs[key] = arr[vld, iter_last].ravel()
            continue

        # ── Scalar per loc ───────────────────────────────────────────
        if arr.ndim == 1:
            attrs[key] = arr[vld]
        else:
            sub = arr[vld, :]
            attrs[key] = sub.squeeze() if sub.ndim > 1 and 1 in sub.shape else sub

    return attrs, num_loc_vld, num_itr


# ---------------------------------------------------------------------------
# Derived properties
# ---------------------------------------------------------------------------

def _compute_properties(
    attrs: AttrStore, num_loc: int, num_itr: int,
    prefs: dict | None = None,
) -> DataProp:
    """
    Compute trace structure, dimensionality, etc.

    If ``prefs["data"]["enforce_min_z_range"]`` is True and the Z range of
    ``attrs["loc_z"]`` is smaller than ``prefs["data"]["min_z_range_nm"]``
    (value in nm — raw data is stored in metres), the Z axis is forced to
    zero and the dataset is marked as 2-D. This filters out tiny residual
    Z variations caused by drift correction in genuinely 2-D acquisitions.
    """
    prop = DataProp(num_loc=num_loc, num_itr=num_itr)

    # Dimensionality
    z = np.asarray(attrs.get("loc_z", np.zeros(num_loc)))

    # Optional 2D/3D threshold (user preference)
    if prefs is not None and z.size > 0:
        data_prefs = prefs.get("data", {}) or {}
        if data_prefs.get("enforce_min_z_range", False):
            min_z_nm   = float(data_prefs.get("min_z_range_nm", 5.0))
            # Raw z values are in metres; convert the threshold from nm.
            z_range_m  = float(z.max() - z.min())
            z_range_nm = z_range_m * 1e9
            if z_range_nm < min_z_nm:
                z = np.zeros_like(z)
                attrs["loc_z"] = z

    prop.num_dim = 3 if np.any(z != 0) else 2

    # Trace structure
    tid     = np.asarray(attrs.get("tid", np.arange(num_loc))).ravel()
    _, first_idx, inverse = np.unique(tid, return_index=True, return_inverse=True)
    counts  = np.bincount(inverse)

    prop.trace_idx         = np.column_stack([first_idx, first_idx + counts - 1])
    prop.num_loc_per_trace = counts
    prop.num_traces        = len(first_idx)

    return prop


def _add_derived_attributes(attrs: AttrStore, prop: DataProp, prefs: dict | None = None) -> AttrStore:
    """
    Port of ``compute_extra_property.m`` — appends:

    ``itr`` (1-based), optional computed attributes, and always ``ftr``.
    """
    n = prop.num_loc
    attr_prefs = (prefs or {}).get("attributes", {}) if prefs else {}
    default_compute = ["idx", "siz", "dst", "dur", "len", "spd", "dt", "tim_trace", "den"]
    compute = {"siz" if name == "nLoc" else name for name in (attr_prefs.get("computed", default_compute) or [])}

    # Make iteration 1-based (MATLAB convention kept for user display)
    if "itr" in attrs:
        attrs["itr"] = (np.asarray(attrs["itr"]) + 1).astype(np.int16)

    tim = np.asarray(attrs.get("tim", np.zeros(n)), dtype=float).ravel()
    tid = np.asarray(attrs.get("tid", np.arange(n))).ravel()

    trace_change = np.diff(tid) != 0   # indices where trace ID changes

    dt_values = None
    dst_values = None

    if {"dt", "spd"} & compute:
        dt = np.diff(tim) if n > 1 else np.empty(0, dtype=float)
        if dt.size:
            dt[trace_change] = 0.0
        dt_values = np.concatenate([[0.0], dt]) if n else np.empty(0, dtype=float)
        if "dt" in compute:
            attrs["dt"] = dt_values

    if {"dst", "spd", "len"} & compute:
        lx = np.asarray(attrs.get("loc_x", np.zeros(n))).ravel()
        ly = np.asarray(attrs.get("loc_y", np.zeros(n))).ravel()
        lz = np.asarray(attrs.get("loc_z", np.zeros(n))).ravel()
        dloc = np.diff(np.column_stack([lx, ly, lz]), axis=0) if n > 1 else np.empty((0, 3))
        dst = np.linalg.norm(dloc, axis=1)
        if dst.size:
            dst[trace_change] = 0.0
        dst_values = np.concatenate([[0.0], dst]) if n else np.empty(0, dtype=float)
        if "dst" in compute:
            attrs["dst"] = dst_values

    if "spd" in compute:
        if dt_values is None:
            dt = np.diff(tim) if n > 1 else np.empty(0, dtype=float)
            if dt.size:
                dt[trace_change] = 0.0
            dt_values = np.concatenate([[0.0], dt]) if n else np.empty(0, dtype=float)
        if dst_values is None:
            lx = np.asarray(attrs.get("loc_x", np.zeros(n))).ravel()
            ly = np.asarray(attrs.get("loc_y", np.zeros(n))).ravel()
            lz = np.asarray(attrs.get("loc_z", np.zeros(n))).ravel()
            dloc = np.diff(np.column_stack([lx, ly, lz]), axis=0) if n > 1 else np.empty((0, 3))
            dst = np.linalg.norm(dloc, axis=1)
            if dst.size:
                dst[trace_change] = 0.0
            dst_values = np.concatenate([[0.0], dst]) if n else np.empty(0, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            attrs["spd"] = np.where(dt_values > 0, dst_values / dt_values, 0.0)

    if "siz" in compute:
        attrs["siz"] = np.repeat(prop.num_loc_per_trace, prop.num_loc_per_trace)

    if "dur" in compute:
        durations = np.zeros(prop.num_traces, dtype=float)
        for i, (start, stop) in enumerate(np.asarray(prop.trace_idx, dtype=int)):
            if 0 <= start <= stop < tim.size:
                durations[i] = float(tim[stop] - tim[start])
        attrs["dur"] = np.repeat(durations, prop.num_loc_per_trace)

    if "len" in compute:
        if dst_values is None:
            lx = np.asarray(attrs.get("loc_x", np.zeros(n))).ravel()
            ly = np.asarray(attrs.get("loc_y", np.zeros(n))).ravel()
            lz = np.asarray(attrs.get("loc_z", np.zeros(n))).ravel()
            dloc = np.diff(np.column_stack([lx, ly, lz]), axis=0) if n > 1 else np.empty((0, 3))
            dst = np.linalg.norm(dloc, axis=1)
            if dst.size:
                dst[trace_change] = 0.0
            dst_values = np.concatenate([[0.0], dst]) if n else np.empty(0, dtype=float)
        lengths = np.zeros(prop.num_traces, dtype=float)
        for i, (start, stop) in enumerate(np.asarray(prop.trace_idx, dtype=int)):
            if 0 <= start <= stop < dst_values.size:
                lengths[i] = float(np.nansum(dst_values[start : stop + 1]))
        attrs["len"] = np.repeat(lengths, prop.num_loc_per_trace)

    # Relative time within each trace
    if "tim_trace" in compute:
        t_start = tim[prop.trace_idx[:, 0]]
        t_min = np.repeat(t_start, prop.num_loc_per_trace)
        attrs["tim_trace"] = tim - t_min

    # Filter mask (all pass by default)
    attrs["ftr"]  = np.ones(n, dtype=bool)

    # 1-based index
    if "idx" in compute:
        attrs["idx"] = np.arange(1, n + 1, dtype=np.uint32)

    return attrs


def _build_channel(attrs: AttrStore, prop: DataProp) -> Channel:
    """Build a :class:`Channel` object from the parsed attributes."""
    dcr = np.asarray(attrs.get("dcr", np.zeros(prop.num_loc))).ravel()
    ti  = prop.trace_idx
    dcr_trace = np.array([
        float(np.mean(trace_vals)) if (trace_vals := dcr[ti[i, 0] : ti[i, 1] + 1][np.isfinite(dcr[ti[i, 0] : ti[i, 1] + 1])]).size else np.nan
        for i in range(prop.num_traces)
    ])
    return Channel(dcr=dcr, dcr_trace=dcr_trace)


# ---------------------------------------------------------------------------
# MSR path: build MinfluxDataset directly from a structured mfx numpy array
# ---------------------------------------------------------------------------

def load_from_mfx_array(
    mfx: "np.ndarray",
    name: str,
    folder: str = "",
    datetime_str: str = "",
    recent_path: str | None = None,
    prefs: dict | None = None,
) -> "MinfluxDataset":
    """
    Build a :class:`MinfluxDataset` directly from a structured ``mfx``
    numpy array produced by ``minflux_msr``.

    This avoids writing a temporary ``.mat`` file — the array is already
    in memory after parsing the ``.msr``.

    Parameters
    ----------
    mfx:
        Structured 1-D numpy array with named fields
        (vld, tim, tid, itr, loc, lnc, efo, cfr, dcr, …).
    name:
        Display name for the dataset (e.g. the dataset label or .msr filename).
    folder:
        Source file folder (for display purposes only).
    datetime_str:
        Load/file datetime string.
    prefs:
        Application preferences dict.
    """
    prefs = prefs or {}
    data_prefs = prefs.get("data", {}) or {}
    load_all_itr = data_prefs.get("iter_load", "last") == "all"

    file_info = FileInfo(
        name=name,
        folder=folder,
        datetime=datetime_str or datetime.now().strftime("%Y-%b-%d, %H:%M:%S"),
        raw_data=None,
        recent_path=recent_path,
    )

    # Convert structured array fields into the AttrStore that our pipeline expects.
    # The mfx array is in m2410-style flat format (already flattened per-loc).
    attrs = AttrStore()
    field_names = list(getattr(mfx.dtype, "names", []) or [])
    num_raw = int(mfx.shape[0])

    for field_name in field_names:
        val = mfx[field_name]
        if field_name in SPATIAL_COORD_FIELDS:
            # (N, 3) xyz arrays — split into _x/_y/_z
            arr = np.asarray(val)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            attrs[f"{field_name}_x"] = arr[:, 0].ravel()
            attrs[f"{field_name}_y"] = arr[:, 1].ravel() if arr.shape[1] > 1 else np.zeros(len(arr))
            attrs[f"{field_name}_z"] = arr[:, 2].ravel() if arr.shape[1] > 2 else np.zeros(len(arr))
        elif field_name == "dcr":
            arr = np.asarray(val)
            if arr.ndim > 1 and arr.shape[0] != num_raw and arr.shape[1] == num_raw:
                arr = arr.T
            attrs[field_name] = arr[:, 0].ravel() if arr.ndim > 1 else arr.ravel()
        else:
            arr = np.asarray(val)
            attrs[field_name] = arr.ravel() if arr.ndim == 1 else arr

    valid_mask = np.ones(num_raw, dtype=bool)
    if "vld" in attrs:
        try:
            valid_mask &= np.asarray(attrs["vld"], dtype=bool).ravel()
        except Exception:
            pass

    loc_x = np.asarray(attrs.get("loc_x", np.full(num_raw, np.nan)), dtype=float).ravel()
    loc_y = np.asarray(attrs.get("loc_y", np.full(num_raw, np.nan)), dtype=float).ravel()
    valid_mask &= np.isfinite(loc_x) & np.isfinite(loc_y)

    if "loc_z" in attrs:
        loc_z = np.asarray(attrs.get("loc_z", np.zeros(num_raw)), dtype=float).ravel()
        valid_mask &= np.isfinite(loc_z)

    itr_arr = attrs.get("itr", None)
    num_itr = 1
    if itr_arr is not None:
        try:
            itr_raw = np.asarray(itr_arr).ravel()
            valid_itr = itr_raw[valid_mask]
            if valid_itr.size:
                last_itr = int(np.nanmax(valid_itr))
                num_itr = last_itr + 1
                if not load_all_itr:
                    valid_mask &= itr_raw == last_itr
        except Exception:
            pass

    for key, val in list(attrs.items()):
        arr = np.asarray(val)
        if arr.ndim >= 1 and arr.shape[0] == num_raw:
            attrs[key] = arr[valid_mask]

    num_loc = int(valid_mask.sum())

    prop = _compute_properties(attrs, num_loc, num_itr, prefs=prefs)
    attrs = _add_derived_attributes(attrs, prop, prefs=prefs)
    prop.attr_names = attrs.keys()

    cali = Calibration()
    channel = _build_channel(attrs, prop)

    dataset = MinfluxDataset(
        file=file_info,
        prop=prop,
        attr=attrs,
        cali=cali,
        channel=channel,
    )
    dataset.metadata.update(
        {
            "source_version": "m2410",
            "source_version_detail": "m2410: all iteration value in N by 1 array",
            "raw_num_loc": int(num_raw),
            "valid_num_loc": int(prop.num_loc),
            "raw_num_itr": int(num_itr),
            "iteration_load_mode": "all" if load_all_itr else "last",
            "includes_invalid": False,
        }
    )
    return dataset


# ---------------------------------------------------------------------------
# .npy loader  — structured or plain numpy arrays
# ---------------------------------------------------------------------------

def load_npy(path: str | Path, prefs: dict | None = None) -> "MinfluxDataset":
    """
    Load a ``.npy`` file exported from the MSR reader or the viewer itself.

    Accepts:
    * A structured numpy array with named fields (mfx layout — same as
      what ``load_from_mfx_array`` consumes).
    * A plain 2-D float array with shape (N, 3) or (N, 2) — interpreted
      as x, y[, z] coordinates in metres.

    Parameters
    ----------
    path:
        Full path to the ``.npy`` file.
    prefs:
        Optional application preferences dict.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    arr = np.load(str(path), allow_pickle=False)

    file_info = FileInfo(
        name=path.name,
        folder=str(path.parent),
        datetime=datetime.fromtimestamp(path.stat().st_mtime).strftime(
            "%Y-%b-%d, %H:%M:%S"
        ),
    )

    # ── Structured array (named fields) ──────────────────────────────
    if arr.dtype.names:
        return load_from_mfx_array(
            mfx=arr,
            name=path.name,
            folder=str(path.parent),
            datetime_str=file_info.datetime,
            prefs=prefs,
        )

    # ── Plain numeric array — treat as xyz coordinates ───────────────
    arr = np.atleast_2d(arr.squeeze())
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(
            f"Cannot interpret '{path.name}' as MINFLUX data. "
            f"Expected a structured array or a (N, 2/3) coordinate array, "
            f"got shape {arr.shape}."
        )

    n   = arr.shape[0]
    ndim = arr.shape[1]

    attrs = AttrStore({
        "loc_x": arr[:, 0].ravel(),
        "loc_y": arr[:, 1].ravel(),
        "loc_z": arr[:, 2].ravel() if ndim == 3 else np.zeros(n),
        "tid":   np.arange(n, dtype=np.int32),
        "tim":   np.zeros(n),
    })

    prop = _compute_properties(attrs, n, 1, prefs=prefs)
    attrs = _add_derived_attributes(attrs, prop, prefs=prefs)
    prop.attr_names = attrs.keys()

    return MinfluxDataset(
        file=file_info,
        prop=prop,
        attr=attrs,
        cali=Calibration(),
        channel=Channel(),
    )


# ---------------------------------------------------------------------------
# .csv loader  — tabular MINFLUX data
# ---------------------------------------------------------------------------

# Expected column names (case-insensitive).  At minimum loc_x / loc_y must
# be present (or x / y as aliases).
_CSV_COL_ALIASES: dict[str, str] = {
    "x":     "loc_x",
    "y":     "loc_y",
    "z":     "loc_z",
    "xnm":   "loc_x_nm",
    "ynm":   "loc_y_nm",
    "znm":   "loc_z_nm",
    "x_nm":  "loc_x_nm",
    "y_nm":  "loc_y_nm",
    "z_nm":  "loc_z_nm",
    "pos_x": "loc_x",
    "pos_y": "loc_y",
    "pos_z": "loc_z",
}


def load_csv(path: str | Path, prefs: dict | None = None) -> "MinfluxDataset":
    """
    Load a ``.csv`` file exported from the MSR reader or Imspector.

    The file must have a header row.  Columns are matched case-insensitively
    to MINFLUX attribute names.  At minimum ``loc_x`` / ``loc_y`` (or their
    aliases ``x`` / ``y``) must be present.

    Parameters
    ----------
    path:
        Full path to the ``.csv`` file.
    prefs:
        Optional application preferences dict.
    """
    import csv as _csv

    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Sniff delimiter (comma or tab)
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
    dialect = _csv.Sniffer().sniff(sample, delimiters=",\t;")

    # Read into dict of lists
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = _csv.DictReader(f, dialect=dialect)
        if reader.fieldnames is None:
            raise ValueError(f"'{path.name}' has no header row.")
        rows: list[dict] = list(reader)

    if not rows:
        raise ValueError(f"'{path.name}' is empty.")

    # Normalise column names (lowercase + alias substitution)
    raw_cols = list(reader.fieldnames)
    col_map: dict[str, str] = {}
    for raw in raw_cols:
        norm = raw.strip().lower().replace(" ", "_")
        col_map[raw] = _CSV_COL_ALIASES.get(norm, norm)

    # Build numpy arrays per column
    data: dict[str, np.ndarray] = {}
    for raw_col, attr_name in col_map.items():
        try:
            values = np.array([float(r[raw_col]) for r in rows], dtype=np.float64)
        except (ValueError, KeyError):
            continue   # skip non-numeric or missing columns
        data[attr_name] = values

    n = len(rows)

    has_meter_xy = "loc_x" in data and "loc_y" in data
    has_nm_xy = "loc_x_nm" in data and "loc_y_nm" in data
    if not (has_meter_xy or has_nm_xy):
        raise ValueError(
            f"'{path.name}' must contain either metre columns ('loc_x'/'x', 'loc_y'/'y') "
            f"or nanometre columns ('xnm'/'x_nm', 'ynm'/'y_nm'). "
            f"Found columns: {list(col_map.values())}"
        )

    file_info = FileInfo(
        name=path.name,
        folder=str(path.parent),
        datetime=datetime.fromtimestamp(path.stat().st_mtime).strftime(
            "%Y-%b-%d, %H:%M:%S"
        ),
    )

    if has_nm_xy:
        if "loc_z_nm" not in data:
            data["loc_z_nm"] = np.zeros(n)
        if "tid" not in data:
            data["tid"] = np.arange(n, dtype=np.int32)
        if "tim" not in data:
            data["tim"] = np.zeros(n)
        extra_attrs = {
            k: v for k, v in data.items()
            if k not in {"loc_x_nm", "loc_y_nm", "loc_z_nm"}
        }
        return build_localization_dataset(
            name=path.name,
            folder=str(path.parent),
            datetime_str=file_info.datetime,
            x_nm=data["loc_x_nm"],
            y_nm=data["loc_y_nm"],
            z_nm=data.get("loc_z_nm"),
            attrs=extra_attrs,
        )

    if "loc_z" not in data:
        data["loc_z"] = np.zeros(n)
    if "tid" not in data:
        data["tid"] = np.arange(n, dtype=np.int32)
    if "tim" not in data:
        data["tim"] = np.zeros(n)

    attrs = AttrStore(data)

    prop = _compute_properties(attrs, n, 1, prefs=prefs)
    attrs = _add_derived_attributes(attrs, prop, prefs=prefs)
    prop.attr_names = attrs.keys()

    return MinfluxDataset(
        file=file_info,
        prop=prop,
        attr=attrs,
        cali=Calibration(),
        channel=Channel(),
    )


def load_json(path: str | Path, prefs: dict | None = None) -> "MinfluxDataset":
    """
    Load Abberior-style MINFLUX JSON localization records.

    The expected shape is a list of record dictionaries, or a dictionary with
    an ``mfx``/``records``/``data`` list. Coordinates are expected either in a
    vector field such as ``loc``/``lnc``/``ext`` using metres, or in explicit
    nanometre columns such as ``xnm``/``ynm``. The dataset is normalized into
    the existing internal ``loc_x``/``loc_y``/``loc_z`` metre convention so all
    current render, filter, plot, and analysis paths keep working.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    records = _extract_json_records(payload)
    if records is None or not _looks_like_minflux_json_records(records):
        raise ValueError(
            f"'{path.name}' is not recognised as MINFLUX localization JSON."
        )

    loc_m = _json_coordinate_array_m(records)
    valid = np.isfinite(loc_m).all(axis=1)
    vld = _json_values(records, "vld")
    if vld is not None:
        valid &= np.asarray([bool(v) for v in vld], dtype=bool)

    if not np.any(valid):
        raise ValueError(f"'{path.name}' contains no valid finite localizations.")

    loc_m = loc_m[valid]
    kept_records = [record for record, keep in zip(records, valid) if keep]
    n = int(loc_m.shape[0])

    attrs = AttrStore({
        "loc_x": loc_m[:, 0],
        "loc_y": loc_m[:, 1],
        "loc_z": loc_m[:, 2],
    })

    keys = sorted({key for record in kept_records for key in record.keys()})
    vector_xyz_keys = {"loc", "lnc", "ext"}
    coordinate_aliases = {
        "x", "y", "z", "xnm", "ynm", "znm",
        "x_nm", "y_nm", "z_nm",
        "loc_x", "loc_y", "loc_z",
        "loc_x_nm", "loc_y_nm", "loc_z_nm",
    }

    for key in keys:
        if key in coordinate_aliases:
            continue
        values = _json_values(kept_records, key)
        if values is None:
            continue

        if key in vector_xyz_keys:
            arr_m = _json_vector_array(values, width=3)
            if arr_m is None:
                continue
            attrs[f"{key}_x"] = arr_m[:, 0]
            attrs[f"{key}_y"] = arr_m[:, 1]
            attrs[f"{key}_z"] = arr_m[:, 2] if arr_m.shape[1] > 2 else np.zeros(n)
            continue

        vector = _json_vector_array(values)
        if vector is not None:
            if key == "dcr":
                attrs["dcr"] = vector[:, 0]
                for col in range(vector.shape[1]):
                    attrs[f"dcr_{col}"] = vector[:, col]
            elif vector.shape[1] in (2, 3):
                attrs[f"{key}_x"] = vector[:, 0]
                attrs[f"{key}_y"] = vector[:, 1]
                attrs[f"{key}_z"] = vector[:, 2] if vector.shape[1] > 2 else np.zeros(n)
            continue

        scalar = _json_scalar_array(values)
        if scalar is not None and scalar.shape[0] == n:
            attrs[key] = scalar

    if "tid" not in attrs:
        attrs["tid"] = np.arange(n, dtype=np.int32)
    if "tim" not in attrs:
        attrs["tim"] = np.zeros(n, dtype=float)
    if "itr" not in attrs:
        attrs["itr"] = np.zeros(n, dtype=np.int16)

    num_itr_arr = attrs.get("itr", None)
    num_itr = (
        int(np.nanmax(num_itr_arr) + 1)
        if num_itr_arr is not None and np.asarray(num_itr_arr).size
        else 1
    )

    prop = _compute_properties(attrs, n, num_itr, prefs=prefs)
    attrs = _add_derived_attributes(attrs, prop, prefs=prefs)
    prop.attr_names = attrs.keys()

    return MinfluxDataset(
        file=FileInfo(
            name=path.name,
            folder=str(path.parent),
            datetime=datetime.fromtimestamp(path.stat().st_mtime).strftime(
                "%Y-%b-%d, %H:%M:%S"
            ),
        ),
        prop=prop,
        attr=attrs,
        cali=Calibration(),
        channel=_build_channel(attrs, prop),
    )


def _extract_json_records(payload: Any) -> list[dict] | None:
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload
    if isinstance(payload, dict):
        for key in ("mfx", "records", "localizations", "localisations", "data"):
            value = payload.get(key)
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return value
    return None


def _looks_like_minflux_json_records(records: list[dict]) -> bool:
    if not records:
        return False
    keys = set(records[0].keys())
    return bool(
        {"loc", "lnc", "ext"} & keys
        or {"x", "y"} <= keys
        or {"xnm", "ynm"} <= keys
        or {"x_nm", "y_nm"} <= keys
        or {"loc_x", "loc_y"} <= keys
        or {"loc_x_nm", "loc_y_nm"} <= keys
    )


def _json_values(records: list[dict], key: str) -> list[Any] | None:
    if not any(key in record for record in records):
        return None
    return [record.get(key) for record in records]


def _json_coordinate_array_m(records: list[dict]) -> np.ndarray:
    loc_values = _json_values(records, "loc")
    if loc_values is not None:
        arr = _json_vector_array(loc_values, width=3)
        if arr is None:
            raise ValueError("JSON 'loc' field must contain numeric xy or xyz values.")
        return arr[:, :3]

    lnc_values = _json_values(records, "lnc")
    if lnc_values is not None:
        arr = _json_vector_array(lnc_values, width=3)
        if arr is not None:
            return arr[:, :3]

    for x_key, y_key, z_key, scale in (
        ("loc_x", "loc_y", "loc_z", 1.0),
        ("x", "y", "z", 1.0),
        ("loc_x_nm", "loc_y_nm", "loc_z_nm", 1e-9),
        ("xnm", "ynm", "znm", 1e-9),
        ("x_nm", "y_nm", "z_nm", 1e-9),
    ):
        x = _json_scalar_array(_json_values(records, x_key) or [])
        y = _json_scalar_array(_json_values(records, y_key) or [])
        if x is None or y is None:
            continue
        z_values = _json_values(records, z_key)
        z = _json_scalar_array(z_values) if z_values is not None else np.zeros_like(x)
        if z is None:
            z = np.zeros_like(x)
        return np.column_stack([x, y, z]) * scale

    raise ValueError("JSON data must contain loc/lnc or x/y localization coordinates.")


def _json_vector_array(values: list[Any], width: int | None = None) -> np.ndarray | None:
    rows: list[list[float]] = []
    observed_width = width
    for value in values:
        if not isinstance(value, (list, tuple)):
            return None
        if observed_width is None:
            observed_width = len(value)
        if len(value) < 2:
            return None
        padded = list(value[:observed_width])
        while len(padded) < int(observed_width):
            padded.append(0.0)
        try:
            rows.append([float(item) for item in padded])
        except (TypeError, ValueError):
            return None
    if observed_width is None:
        return None
    return np.asarray(rows, dtype=float)


def _json_scalar_array(values: list[Any]) -> np.ndarray | None:
    if not values:
        return None
    non_null = [value for value in values if value is not None]
    if not non_null:
        return None
    if all(isinstance(value, bool) for value in non_null):
        return np.asarray([False if value is None else bool(value) for value in values], dtype=bool)
    if any(isinstance(value, (list, tuple, dict)) for value in non_null):
        return None
    try:
        arr = np.asarray(
            [np.nan if value is None else float(value) for value in values],
            dtype=float,
        )
    except (TypeError, ValueError):
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size == arr.size and np.all(np.equal(np.mod(arr, 1), 0)):
        return arr.astype(np.int64)
    return arr


def load_tiff(path: str | Path, prefs: dict | None = None) -> "MinfluxDataset":
    """Load a TIFF image as an image-backed dataset for the render window."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        import tifffile
    except ImportError as exc:
        raise ImportError("tifffile is required to load TIFF images.") from exc

    image = tifffile.imread(str(path))
    return build_image_dataset(
        name=path.name,
        folder=str(path.parent),
        datetime_str=datetime.fromtimestamp(path.stat().st_mtime).strftime(
            "%Y-%b-%d, %H:%M:%S"
        ),
        image=image,
        pixel_size_nm=1.0,
    )
