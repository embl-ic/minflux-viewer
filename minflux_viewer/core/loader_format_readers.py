'Loader format readers'

from __future__ import annotations

import itertools
import json
from array import array
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio

from minflux_viewer.core.dataset import (
    AttrStore,
    Calibration,
    Channel,
    FileInfo,
    MinfluxDataset,
    build_image_dataset,
    build_localization_dataset,
)
from minflux_viewer.core.final_localization import final_rows, records_final_status
from minflux_viewer.core.loader_metadata_recipe import _restore_filter_mask, apply_metadata_sidecar
from minflux_viewer.core.loader_normalization import (
    _add_derived_attributes,
    _add_mfx_attr,
    _build_channel,
    _build_mfx_raw_from_raw,
    _compute_properties,
    _detect_raw_version,
    _effective_iteration_metadata,
    _is_legacy_nested_itr_mfx,
    _mat_struct_to_dict,
    _mfx_columns,
    _normalize_mfx_attrs,
    _parse_raw,
    _source_metadata_from_raw,
)
from minflux_viewer.core.loader_raw_access import (
    _attrs_include_invalid,
    _attrs_valid_count,
    _derived_last_from_raw,
    attr_matches_last_valid,
)

# Processed coordinate columns in nm, first match wins. The ``*_nm`` spellings
# are the same family the CSV reader aliases ``xnm`` to; bare ``x`` comes last
# (nm in a .mat/.npy table, but metres in JSON -- see _is_processed_flat_json).
_FLAT_X = ("xnm", "x_nm", "loc_x_nm", "x")


_FLAT_Y = ("ynm", "y_nm", "loc_y_nm", "y")


_FLAT_Z = ("znm", "z_nm", "loc_z_nm", "z")


def _flat_table_from_dict(
    raw: dict, name: str, folder: str, prefs: dict | None
) -> MinfluxDataset | None:
    """Build a dataset from a flat localization table, or return None.

    A flat table is a dict of 1-D columns with ``xnm``/``ynm`` (or ``x``/``y``)
    coordinates and **no** structured mfx ``loc`` field — i.e. the layout written
    by *Save Processed Data*. Coordinates are treated as final/processed (already
    Z-scaled), so Z scaling factor is pinned to 1.0 and the post-load auto-estimation is
    suppressed; derived attributes are recomputed; the saved ``ftr`` column, if
    present, restores the filter state.
    """
    from minflux_viewer.core.attributes import (
        DERIVED_ATTRIBUTE_NAMES,
        LEGACY_DERIVED_ATTRIBUTE_ALIASES,
    )

    if not isinstance(raw, dict):
        return None
    lookup = {str(k).lower(): k for k in raw.keys()}

    def find(aliases):
        return next((lookup[a] for a in aliases if a in lookup), None)

    xk, yk = find(_FLAT_X), find(_FLAT_Y)
    if xk is None or yk is None:
        return None
    loc = raw.get("loc")
    if loc is not None and getattr(np.asarray(loc).dtype, "names", None):
        return None                                  # structured mfx, not flat
    x = np.asarray(raw[xk]).ravel().astype(float)
    y = np.asarray(raw[yk]).ravel().astype(float)
    # One row is a table too: whether coordinates are final must not depend on
    # how many there are (a one-localization export was otherwise not pinned).
    if x.ndim != 1 or x.size < 1 or x.size != y.size:
        return None
    n = x.size
    zk = find(_FLAT_Z)
    z = None
    if zk is not None:
        zc = np.asarray(raw[zk]).ravel().astype(float)
        if zc.ndim == 1 and zc.size == n:
            z = zc

    coord_keys = {xk, yk} | ({zk} if zk else set())
    skip = set(DERIVED_ATTRIBUTE_NAMES) | set(LEGACY_DERIVED_ATTRIBUTE_ALIASES)
    attrs: dict[str, np.ndarray] = {}
    for key, val in raw.items():
        if key in coord_keys or key in skip:
            continue
        col = np.asarray(val).ravel()
        if col.ndim == 1 and col.size == n:
            attrs[key] = col
    tid = attrs.pop("tid", None)
    tim = attrs.pop("tim", None)
    ftr = attrs.pop("ftr", None)

    ds = build_localization_dataset(
        name=name, x_nm=x, y_nm=y, z_nm=z, folder=folder,
        tid=tid, tim=tim, attrs=attrs or None, prefs=prefs,
    )
    # Re-imported processed coordinates are final: pin Z scaling factor=1 and pre-empt the
    # post-load automatic Z scaling factor estimation (which would otherwise double-correct z).
    ds.set_z_scaling_factor(1.0, source="processed (re-imported, coordinates final)")
    ds.derived["z_scaling_factor"] = np.asarray([1.0], dtype=float)
    if ftr is not None and np.asarray(ftr).ravel().size == n:
        ds.filter_mask = np.asarray(ftr).ravel().astype(bool)
    return ds


def load_dataset(path: str | Path, prefs: dict | None = None, *,
                 apply_sidecar: bool = True) -> MinfluxDataset:
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
    # Localizations always materialize at the last valid iteration; every other
    # iteration stays reachable through mfx_raw (see core/iteration.py).
    load_all_itr    = False
    only_valid      = data_prefs.get("only_valid_locs", True)

    # 1. Read the raw .mat file
    raw = _load_mat(path)

    # 1b. A flat localization table (xnm/ynm columns, no structured mfx) — e.g.
    # one written by Save Processed Data — loads generically, not via the mfx
    # parser (which would lose the coordinates).
    flat = _flat_table_from_dict(raw, path.name, str(path.parent), prefs)
    if flat is not None:
        if apply_sidecar:
            apply_metadata_sidecar(flat, path)
        return flat

    # 1c. A flat (m2410) table: the same record table Imspector writes to .npy
    # and .json, so it is built by the same builder -- columns cut out of the
    # MATLAB arrays, no record buffer. m2205/legacy layouts keep their parser.
    if _detect_raw_version(raw) == "m2410":
        ds = load_from_mfx_columns(
            _bool_flag_columns(_mat_m2410_columns(raw)), path.name,
            folder=str(path.parent),
            datetime_str=datetime.fromtimestamp(path.stat().st_mtime).strftime(
                "%Y-%b-%d, %H:%M:%S"),
            prefs=prefs)
        ds.metadata["source_format"] = "mat"
        if apply_sidecar:
            apply_metadata_sidecar(ds, path)
        return ds

    # 2. Build mfx_raw — all iterations, all measurements, no filtering
    mfx_raw_store = _build_mfx_raw_from_raw(raw)

    # 3. Detect format and parse into an AttrStore (filtered working set)
    attrs, num_loc, num_itr = _parse_raw(
        raw,
        load_all_itr=load_all_itr,
        only_valid=only_valid,
    )
    if attrs is None:
        raise ValueError(
            f"'{path.name}' could not be parsed as a MINFLUX dataset. "
            "Check that it contains an 'itr' field."
        )

    # 4. File metadata (raw_data cleared — mfx_raw supersedes it)
    mtime = path.stat().st_mtime
    file_info = FileInfo(
        name=path.name,
        folder=str(path.parent),
        datetime=datetime.fromtimestamp(mtime).strftime("%Y-%b-%d, %H:%M:%S"),
        raw_data=None,
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
    dataset.metadata["source_format"] = "mat"
    dataset.metadata["valid_num_loc"] = _attrs_valid_count(attrs, prop.num_loc)
    dataset.metadata["includes_invalid"] = _attrs_include_invalid(attrs)
    dataset.components.mfx_raw = mfx_raw_store
    # ds.attr is not the last-valid materialization (all-iteration and/or
    # invalid-inclusive load): precompute the derived attributes at last-valid
    # so mfx_get can broadcast them to any selection.
    if not attr_matches_last_valid(dataset):
        dataset.components.derived_last = _derived_last_from_raw(mfx_raw_store, num_itr, prefs)
    if apply_sidecar:
        apply_metadata_sidecar(dataset, path)
    return dataset


def _mat_m2410_columns(raw: dict) -> dict[str, np.ndarray]:
    """The m2410 fields of a loaded ``.mat`` dict as builder columns.

    Same field handling the m2410 parser always had: a length-N column as
    ``(N,)``, ``(1, N)`` or ``(N, 1)``; an ``(N, k)`` or ``(k, N)`` vector as
    ``(N, k)``; an ``(N, K, k)`` array by its last plane; ``tic`` (the raw FPGA
    clock, superseded by ``tim``) and the HDF5 ``#refs#`` placeholder skipped,
    as is anything not numeric or not N long (a struct, the MBM companion).
    """
    n = np.asarray(raw["itr"]).ravel().size
    columns: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        arr = np.asarray(value)
        if key in ("#refs#", "tic") or arr.ndim == 0 or arr.size == 0:
            continue
        if arr.dtype.kind not in "biuf":
            continue
        if arr.ndim == 1:
            if arr.shape[0] != n:
                continue
            column = arr
        elif arr.ndim == 2:
            if arr.shape[0] == n:
                column = arr
            elif arr.shape[1] == n:
                column = arr.T
            else:
                continue
            if column.shape[1] == 1:
                column = column[:, 0]
        elif arr.ndim == 3 and arr.shape[0] == n:
            column = arr[:, -1, :]
        else:
            continue
        columns[str(key)] = column
    return columns


#: m2410 flags: Imspector writes them as booleans; MATLAB's logical loads as
#: uint8, and a JSON 0/1 as int. A flag column holding only 0/1 becomes bool.
_FLAG_FIELDS = ("vld", "fnl", "bot", "eot")


def _bool_flag_columns(columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    for key in _FLAG_FIELDS:
        values = columns.get(key)
        if values is not None and values.dtype != np.bool_ and values.ndim == 1:
            if np.isin(values, (0, 1)).all():
                columns[key] = values.astype(bool)
    return columns


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
            elif getattr(inner, "dtype", None) is not None and inner.dtype.names:
                # A single structured ndarray (e.g. a saved canonical mfx) —
                # explode its fields to the top level so the parser finds 'itr'.
                raw = {nm: inner[nm] for nm in inner.dtype.names}
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


def load_from_mfx_array(
    mfx: np.ndarray,
    name: str,
    folder: str = "",
    datetime_str: str = "",
    recent_path: str | None = None,
    prefs: dict | None = None,
    progress=None,
) -> MinfluxDataset:
    """
    Build a :class:`MinfluxDataset` directly from a structured ``mfx``
    numpy array produced by ``minflux_msr``.

    Qt-free, so it may run on a worker thread. ``progress(fraction)`` (0..1),
    when given, is called as the build advances; it is a plain callback and is
    also a natural cancellation checkpoint for the caller (it may raise).

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
    report = _progress_reporter(progress)
    report(0.0)
    if _is_legacy_nested_itr_mfx(mfx):
        normalized = _normalize_mfx_attrs(
            mfx, load_all_itr=True, progress=lambda f: report(0.7 * f))
        selected = _normalize_mfx_attrs(mfx, load_all_itr=False)
        report(0.7)
        return _assemble_dataset(
            normalized, selected, raw_num_loc=int(mfx.shape[0]),
            file_info=_file_info(name, folder, datetime_str, recent_path),
            prefs=prefs, report=report)
    # A flat (m2410) table: copy the record buffer into columns ONCE (the copy
    # dominates the build, ~70 % of it), then build from the columns.
    names = list(getattr(mfx.dtype, "names", None) or ())
    if mfx.ndim == 1 and names:
        columns = _mfx_columns(mfx, progress=lambda f: report(0.7 * f))
    else:
        columns = {field: mfx[field] for field in names}
    return load_from_mfx_columns(
        columns, name, folder=folder, datetime_str=datetime_str,
        recent_path=recent_path, prefs=prefs, progress=progress, _started=True)


#: ``source_version_detail`` of a flat table.
_M2410_DETAIL = "m2410: all iteration value in N by 1 array"


def load_from_mfx_columns(
    columns: dict,
    name: str,
    folder: str = "",
    datetime_str: str = "",
    recent_path: str | None = None,
    prefs: dict | None = None,
    progress=None,
    *,
    _started: bool = False,
) -> MinfluxDataset:
    """Build a :class:`MinfluxDataset` from flat (m2410) columns.

    ``columns`` maps each field to a ``(N,)`` array, or ``(N, k)`` for a vector
    field (``loc``/``lnc``/``ext`` in metres, ``dcr``) -- exactly what
    ``_mfx_columns`` cuts out of a structured array. This is the ONE builder
    every flat MINFLUX reader ends in: a reader decodes its container into
    columns and stamps ``source_format``; version, effective iterations,
    validity, last-iteration materialization and ``mfx_raw`` happen here.
    Handing columns straight in avoids interleaving them into a record buffer
    only for it to be copied back out (measured 3.7x slower on a 59 MB .mat).
    """
    report = _progress_reporter(progress)
    if not _started:
        report(0.0)
    n = _common_row_count(columns)
    attrs_all = AttrStore()
    for field in list(columns):
        _add_mfx_attr(attrs_all, str(field), columns[field], n)
    report(0.7)
    return _assemble_dataset(
        (attrs_all, n, 1, "m2410", _M2410_DETAIL), None, raw_num_loc=n,
        file_info=_file_info(name, folder, datetime_str, recent_path),
        prefs=prefs, report=report)


def _progress_reporter(progress):
    def report(fraction: float) -> None:
        if progress is not None:
            progress(float(fraction))
    return report


def _file_info(name, folder, datetime_str, recent_path) -> FileInfo:
    return FileInfo(
        name=name,
        folder=folder,
        datetime=datetime_str or datetime.now().strftime("%Y-%b-%d, %H:%M:%S"),
        raw_data=None,
        recent_path=recent_path,
    )


def _common_row_count(columns: dict) -> int:
    """The row count every column shares; a disagreement is a format error."""
    if not columns:
        raise ValueError("No MINFLUX fields to build a dataset from.")
    counts = {str(key): int(np.asarray(values).shape[0]) if np.ndim(values) else -1
              for key, values in columns.items()}
    distinct = set(counts.values())
    if len(distinct) != 1 or -1 in distinct:
        raise ValueError(f"MINFLUX fields do not share one row count: {counts}")
    return distinct.pop()


def _assemble_dataset(normalized, selected, *, raw_num_loc: int, file_info: FileInfo,
                      prefs: dict | None, report) -> MinfluxDataset:
    """Everything after the fields are columns: raw store, validity and
    last-iteration selection, derived attributes, metadata.

    ``normalized`` is ``_normalize_mfx_attrs``' all-iteration result;
    ``selected`` its last-iteration result for a nested (m2205) table, or
    ``None`` for a flat one, whose all-iteration columns are reused.
    """
    prefs = prefs or {}
    data_prefs = prefs.get("data", {}) or {}
    load_all_itr = False          # always last-valid; browse via mfx_raw
    only_valid   = data_prefs.get("only_valid_locs", True)
    (attrs_all, num_raw_all, all_num_itr,
     all_version, all_version_detail) = normalized
    mfx_raw_store = AttrStore()
    for _k, _v in attrs_all.items():
        mfx_raw_store[_k] = np.asarray(_v)
    if "vld" not in mfx_raw_store:
        mfx_raw_store["vld"] = np.ones(num_raw_all, dtype=bool)
    if "itr" not in mfx_raw_store:
        mfx_raw_store["itr"] = np.zeros(num_raw_all, dtype=np.int16)

    if selected is not None:
        attrs, num_raw, detected_num_itr, source_version, source_version_detail = selected
    else:
        # A flat (m2410) table ignores ``load_all_itr``, so a second pass would
        # copy every field out of the record buffer again for an identical
        # result. Reuse the first: the row selection below replaces each
        # per-row column with a new masked array, so mfx_raw keeps its own
        # unfiltered columns.
        attrs = AttrStore()
        for _k, _v in attrs_all.items():
            attrs[_k] = _v
        num_raw, detected_num_itr = num_raw_all, all_num_itr
        source_version, source_version_detail = all_version, all_version_detail

    valid_mask = np.ones(num_raw, dtype=bool)
    if only_valid and "vld" in attrs:
        try:
            valid_mask &= np.asarray(attrs["vld"], dtype=bool).ravel()
        except (TypeError, ValueError):
            pass

    loc_x = np.asarray(attrs.get("loc_x", np.full(num_raw, np.nan)), dtype=float).ravel()
    loc_y = np.asarray(attrs.get("loc_y", np.full(num_raw, np.nan)), dtype=float).ravel()
    valid_mask &= np.isfinite(loc_x) & np.isfinite(loc_y)

    if "loc_z" in attrs:
        loc_z = np.asarray(attrs.get("loc_z", np.zeros(num_raw)), dtype=float).ravel()
        valid_mask &= np.isfinite(loc_z)

    itr_arr = attrs.get("itr", None)
    num_itr = max(int(detected_num_itr), 1)
    # The acquisition's iteration count comes from EVERY raw row, not from the
    # valid ones: when no row of the highest iteration is valid, counting only
    # valid rows under-reported it (2 where the .mat/.json readers said 3), and
    # iteration browsing and the stacked-iteration colours take this count as
    # the sequence length. ``num_itr`` below stays the materialized iteration.
    raw_num_itr = _flat_iteration_count(mfx_raw_store.get("itr"))
    sta_arr = attrs.get("sta")
    if selected is None and itr_arr is not None and records_final_status(sta_arr, num_raw):
        # The file's own per-row status: ``sta == 1`` is a valid final
        # localization, in whichever thread it ran -- the rule raw browsing
        # uses too (core.final_localization), so the rows stay aligned.
        valid_mask &= final_rows(itr_arr, attrs.get("vld"), sta_arr, attrs.get("fnl"),
                                 vld_only=only_valid)
        chosen = np.asarray(itr_arr).ravel()[valid_mask]
        if chosen.size:
            num_itr = int(np.max(chosen)) + 1
    elif itr_arr is not None:
        try:
            itr_raw = np.asarray(itr_arr).ravel()
            valid_itr = itr_raw[valid_mask]
            if valid_itr.size:
                last_itr = int(np.nanmax(valid_itr))
                if source_version != "legacy":
                    num_itr = last_itr + 1
                if not load_all_itr:
                    valid_mask &= itr_raw == last_itr
        except (IndexError, TypeError, ValueError, OverflowError):
            pass

    for key, val in list(attrs.items()):
        arr = np.asarray(val)
        if arr.ndim >= 1 and arr.shape[0] == num_raw:
            attrs[key] = arr[valid_mask]

    num_loc = int(valid_mask.sum())
    report(0.8)

    prop = _compute_properties(attrs, num_loc, num_itr, prefs=prefs)
    attrs = _add_derived_attributes(attrs, prop, prefs=prefs)
    prop.attr_names = attrs.keys()
    report(0.9)

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
            "source_version": source_version,
            "source_version_detail": source_version_detail,
            "raw_num_loc": int(raw_num_loc),
            "valid_num_loc": _attrs_valid_count(attrs, prop.num_loc),
            "raw_num_itr": int(
                num_itr if source_version != "m2410" or raw_num_itr is None
                else max(raw_num_itr, num_itr)
            ),
            "iteration_load_mode": "all" if load_all_itr else "last",
            "includes_invalid": _attrs_include_invalid(attrs),
        }
    )
    # Effective CFR/EFC iteration — computed from the all-iteration raw store,
    # since the materialized last-valid view (itr == global max) may not be the
    # iteration where CFR/EFC were measured.
    dataset.metadata.update(_effective_iteration_metadata(mfx_raw_store))
    dataset.components.mfx_raw = mfx_raw_store
    if not attr_matches_last_valid(dataset):
        dataset.components.derived_last = _derived_last_from_raw(mfx_raw_store, num_itr, prefs)
    report(1.0)
    return dataset


def _flat_iteration_count(itr_values) -> int | None:
    """``max(itr) + 1`` over every row of a flat (m2410) ``itr`` column.

    ``None`` when there is no usable integer column, so the caller keeps its
    own count rather than inventing one.
    """
    if itr_values is None:
        return None
    arr = np.asarray(itr_values).ravel()
    if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
        return None
    finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
    if finite.size == 0:
        return None
    return int(np.max(finite)) + 1


def load_npy(path: str | Path, prefs: dict | None = None, *,
             apply_sidecar: bool = True) -> MinfluxDataset:
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
        # A flat localization table (xnm/ynm fields) from Save Processed Data
        # loads generically; a real mfx structured array goes to the mfx parser.
        flat = _flat_table_from_dict(
            {nm: np.asarray(arr[nm]) for nm in arr.dtype.names},
            path.name, str(path.parent), prefs,
        )
        if flat is not None:
            flat.metadata["source_format"] = "npy"
            if apply_sidecar:
                apply_metadata_sidecar(flat, path)
            return flat
        ds = load_from_mfx_array(
            mfx=arr,
            name=path.name,
            folder=str(path.parent),
            datetime_str=file_info.datetime,
            prefs=prefs,
        )
        ds.metadata["source_format"] = "npy"
        if apply_sidecar:
            apply_metadata_sidecar(ds, path)
        return ds

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

    mfx_raw_store = AttrStore({
        "loc_x": arr[:, 0].ravel(),
        "loc_y": arr[:, 1].ravel(),
        "loc_z": arr[:, 2].ravel() if ndim == 3 else np.zeros(n),
        "tid":   np.arange(n, dtype=np.int32),
        "tim":   np.zeros(n),
        "itr":   np.zeros(n, dtype=np.int16),
        "vld":   np.ones(n, dtype=bool),
    })

    ds = MinfluxDataset(
        file=file_info,
        prop=prop,
        attr=attrs,
        cali=Calibration(),
        channel=Channel(),
    )
    ds.metadata.update({
        "source_version": "plain_array",
        "source_format": "npy",
        "raw_num_loc": n,
        "raw_num_itr": 1,
        "iteration_load_mode": "last",
    })
    ds.components.mfx_raw = mfx_raw_store
    if apply_sidecar:
        apply_metadata_sidecar(ds, path)
    return ds


def load_npz(path: str | Path, prefs: dict | None = None, *,
             apply_sidecar: bool = True) -> MinfluxDataset:
    """Load a ``.npz`` flat-column export.

    The retired writer is not offered by the save dialog; this reader preserves
    existing files. It accepts flat columns in two shapes: canonical raw
    (``loc_x``/``loc_y``/``itr``/``vld``) or a processed table (``xnm``/``ynm``).
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with np.load(str(path), allow_pickle=False) as bundle:
        columns = {str(key): np.asarray(bundle[key]) for key in bundle.files}
    if not columns:
        raise ValueError(f"'{path.name}' contains no arrays.")

    required = {"loc_x", "loc_y", "itr", "vld"}
    if required <= set(columns):
        from minflux_viewer.core.save import columns_to_mfx_array

        ds = load_from_mfx_array(
            mfx=columns_to_mfx_array(columns),
            name=path.name,
            folder=str(path.parent),
            datetime_str=datetime.fromtimestamp(path.stat().st_mtime).strftime(
                "%Y-%b-%d, %H:%M:%S"
            ),
            prefs=prefs,
        )
    else:
        ds = _flat_table_from_dict(columns, path.name, str(path.parent), prefs)
        if ds is None:
            raise ValueError(
                f"'{path.name}' is not a MINFLUX .npz export; it has neither the "
                f"canonical raw columns ({', '.join(sorted(required))}) nor a "
                f"processed table (xnm/ynm)."
            )
    ds.metadata["source_format"] = "npz"
    if apply_sidecar:
        apply_metadata_sidecar(ds, path)
    return ds


def load_zarr(path: str | Path, prefs: dict | None = None) -> MinfluxDataset:
    """Load the first supported MINFLUX Viewer Zarr dataset schema.

    No legacy user data exists for the former experimental flat-column layout,
    so unmarked stores are rejected instead of being guessed or silently
    reinterpreted. Vendor/Imspector Zarr stores remain inspectable through the
    MSR reader and are not application save files by themselves.
    """
    from minflux_viewer.core.minflux_zarr import (
        PROJECT_FORMAT_ID,
        MinfluxZarrError,
        load_minflux_zarr,
        load_minflux_zarr_project,
    )

    try:
        return load_minflux_zarr(path, prefs=prefs)
    except MinfluxZarrError:
        # A multi-dataset acquisition is marked as a *project*, and the
        # single-dataset loader rejects it as "format marker missing" -- which
        # is untrue and unhelpful. The GUI never hit this because it calls the
        # project loader directly; scripts calling the documented entry point
        # did. Return the first member, matching what this function promises.
        project = load_minflux_zarr_project(path, prefs=prefs)
        if not project.datasets:
            raise
        if len(project.datasets) > 1:
            import warnings

            warnings.warn(
                f"'{Path(path).name}' is a {PROJECT_FORMAT_ID} store with "
                f"{len(project.datasets)} datasets; load_zarr returns the first. "
                "Use load_minflux_zarr_project to get them all.",
                stacklevel=2,
            )
        return project.datasets[0]


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
    # A flat export that split the ``loc`` vector by component index rather than
    # by axis letter (``flatten_mfx_array``'s fallback naming for a vector field
    # it does not know) still means x/y/z.
    "loc_0": "loc_x",
    "loc_1": "loc_y",
    "loc_2": "loc_z",
}


_CANONICAL_CSV_BOOL_FIELDS = frozenset({"bot", "eot", "fnl", "vld"})


_CANONICAL_CSV_U8_FIELDS = frozenset({"sqi", "sta", "thi"})


_CANONICAL_CSV_I32_FIELDS = frozenset({"ecc", "eco", "itr"})


_CANONICAL_CSV_I64_FIELDS = frozenset({"gri", "tid"})


def _canonical_csv_dtype(headers: list[str]) -> np.dtype:
    """Structured dtype matching the flattened canonical CSV column order.

    CSV carries values but no binary dtype metadata. Integer/boolean fields use
    the same conservative types as the existing loader; other values are read
    as float64. Vector fields are recomposed directly while ``np.loadtxt``
    parses, avoiding a multi-gigabyte temporary all-float matrix.
    """
    names = [str(value).strip().lower() for value in headers]
    vectors = {
        "loc_x": ("loc", ("loc_x", "loc_y", "loc_z"), 3),
        "lnc_x": ("lnc", ("lnc_x", "lnc_y", "lnc_z"), 3),
        "ext_x": ("ext", ("ext_x", "ext_y", "ext_z"), 3),
        "dcr_0": ("dcr", ("dcr_0", "dcr_1"), 2),
    }
    fields: list[tuple] = []
    index = 0
    while index < len(names):
        name = names[index]
        vector = vectors.get(name)
        if vector is not None:
            output_name, expected, width = vector
            if tuple(names[index:index + width]) != expected:
                raise ValueError(
                    f"Canonical CSV vector columns for {output_name!r} are not contiguous."
                )
            fields.append((output_name, np.float64, (width,)))
            index += width
            continue
        if name in _CANONICAL_CSV_BOOL_FIELDS:
            dtype = np.bool_
        elif name in _CANONICAL_CSV_U8_FIELDS:
            dtype = np.uint8
        elif name in _CANONICAL_CSV_I32_FIELDS:
            dtype = np.int32
        elif name in _CANONICAL_CSV_I64_FIELDS:
            dtype = np.int64
        else:
            dtype = np.float64
        fields.append((name, dtype))
        index += 1
    return np.dtype(fields)


def load_csv(path: str | Path, prefs: dict | None = None, *,
             apply_sidecar: bool = True) -> MinfluxDataset:
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

    # Read the header once.  Values are collected in compact C double arrays,
    # rather than a Python list of one dict per row; the latter made reopening
    # a large canonical CSV needlessly consume multiple gigabytes.
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = _csv.reader(f, dialect)
        try:
            raw_cols = next(reader)
        except StopIteration:
            raise ValueError(f"'{path.name}' has no header row.")
    col_map: dict[str, str] = {}
    for raw in raw_cols:
        norm = raw.strip().lower().replace(" ", "_")
        col_map[raw] = _CSV_COL_ALIASES.get(norm, norm)

    # Two name sets, deliberately: the FAST np.loadtxt path below builds its
    # dtype from the header text itself and hands the result straight to the mfx
    # parser, so it needs the exact canonical spelling. The general reassembly
    # further down works from the aliased columns, where ``loc_0/loc_1/loc_2``
    # has already become ``loc_x/loc_y/loc_z``. (A processed snapshot also
    # carries itr/vld but spells its coordinates ``xnm/ynm/znm``, which alias to
    # ``loc_x_nm`` -- a different key -- so the two stay distinguishable.)
    header_names = {str(raw).strip().lower() for raw in raw_cols}
    canonical_names = set(col_map.values())
    if (
        dialect.delimiter == ","
        and {"loc_x", "loc_y", "loc_z", "itr", "vld"} <= header_names
        and all(str(raw).strip().lower() == str(raw).strip() for raw in raw_cols)
    ):
        try:
            mfx = np.loadtxt(
                path,
                delimiter=",",
                skiprows=1,
                dtype=_canonical_csv_dtype(raw_cols),
                ndmin=1,
            )
            if mfx.size == 0:
                raise ValueError(f"'{path.name}' is empty.")
        except (ValueError, OSError):
            # Fall through to the bounded csv.reader path for unusual numeric
            # spellings or a noncanonical table.
            mfx = None
        if mfx is not None:
            ds = load_from_mfx_array(
                mfx=mfx,
                name=path.name,
                folder=str(path.parent),
                datetime_str=datetime.fromtimestamp(path.stat().st_mtime).strftime(
                    "%Y-%b-%d, %H:%M:%S"
                ),
                prefs=prefs,
            )
            ds.metadata["source_format"] = "csv"
            if apply_sidecar:
                apply_metadata_sidecar(ds, path)
            return ds
    else:
        mfx = None

    data = None

    if data is None:
        buffers = {raw: array("d") for raw in raw_cols}
        bad_columns: set[str] = set()
        n = 0
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = _csv.reader(f, dialect)
            next(reader, None)
            for row in reader:
                n += 1
                for index, raw_col in enumerate(raw_cols):
                    if raw_col in bad_columns:
                        continue
                    try:
                        buffers[raw_col].append(float(row[index]))
                    except (IndexError, TypeError, ValueError):
                        bad_columns.add(raw_col)
                        buffers.pop(raw_col, None)

        if n == 0:
            raise ValueError(f"'{path.name}' is empty.")

        data = {
            col_map[raw_col]: np.frombuffer(buffer, dtype=np.float64)
            for raw_col, buffer in buffers.items()
        }

    # Nanometre coordinate columns mean a **processed snapshot** (Save Processed
    # Data bakes the Z scaling factor and the transform into xnm/ynm/znm), which
    # the .mat/.npy flat-table reader already recognises. Remember that before
    # the columns are renamed, so the same file pins Z scaling the same way in
    # every format -- otherwise a 3-D snapshot reopened as CSV would be
    # z-corrected a second time.
    is_processed_snapshot = any(f"loc_{axis}_nm" in data for axis in ("x", "y"))

    # Normalise any nm coordinate columns to the canonical metres store, so a
    # dataset only ever carries one coordinate convention (loc_x/loc_y/loc_z).
    for axis in ("x", "y", "z"):
        nm_key, m_key = f"loc_{axis}_nm", f"loc_{axis}"
        if nm_key in data and m_key not in data:
            data[m_key] = data.pop(nm_key) * 1.0e-9

    if "loc_x" not in data or "loc_y" not in data:
        raise ValueError(
            f"'{path.name}' must contain x/y localization columns "
            f"('loc_x'/'x'/'xnm', 'loc_y'/'y'/'ynm'). "
            f"Found columns: {list(col_map.values())}"
        )

    # --- canonical raw CSV: reassemble, do not treat rows as localizations ---
    # A raw canonical export is one row per (localization x iteration). Building
    # an AttrStore straight from those rows makes every iteration a separate
    # localization, drops the iteration axis entirely and loses Z scaling factor/filters.
    # `.zarr` already reassembles through `columns_to_mfx_array`; CSV must too.
    #
    # The discriminator is the *original* header, not the aliased names: a
    # processed snapshot also carries `itr`/`vld`, but spells its coordinates
    # `xnm/ynm/znm` (nanometres), whereas a raw export writes `loc_x/loc_y/loc_z`
    # (metres). Aliasing maps `xnm -> loc_x`, so testing `data` would confuse the
    # two.
    if {"loc_x", "loc_y", "itr", "vld"} <= canonical_names:
        from minflux_viewer.core.save import columns_to_mfx_array

        # np.loadtxt/csv give float64 for every column; restore the dtypes the
        # canonical layout expects so `vld` masks and `itr` compares correctly.
        typed = dict(data)
        for name, dtype in (("itr", np.int32), ("tid", np.int64), ("eco", np.int32),
                            ("ecc", np.int32), ("gri", np.int64), ("sta", np.uint8),
                            ("thi", np.uint8), ("sqi", np.uint8)):
            if name in typed:
                typed[name] = np.asarray(typed[name]).astype(dtype, copy=False)
        for name in ("vld", "fnl", "bot", "eot"):
            if name in typed:
                typed[name] = np.asarray(typed[name]).astype(bool, copy=False)
        typed.pop("loc_id", None)          # internal cache column, never data

        ds = load_from_mfx_array(
            mfx=columns_to_mfx_array(typed),
            name=path.name,
            folder=str(path.parent),
            datetime_str=datetime.fromtimestamp(path.stat().st_mtime).strftime(
                "%Y-%b-%d, %H:%M:%S"
            ),
            prefs=prefs,
        )
        ds.metadata["source_format"] = "csv"
        if apply_sidecar:
            apply_metadata_sidecar(ds, path)
        return ds

    file_info = FileInfo(
        name=path.name,
        folder=str(path.parent),
        datetime=datetime.fromtimestamp(path.stat().st_mtime).strftime(
            "%Y-%b-%d, %H:%M:%S"
        ),
    )

    if "loc_z" not in data:
        data["loc_z"] = np.zeros(n)
    has_real_tid = "tid" in data
    if not has_real_tid:
        data["tid"] = np.arange(1, n + 1, dtype=np.int64)
    if "tim" not in data:
        data["tim"] = np.zeros(n)

    saved_ftr = data.pop("ftr", None)   # captured before derived attrs reset it
    attrs = AttrStore(data)

    prop = _compute_properties(attrs, n, 1, prefs=prefs)
    attrs = _add_derived_attributes(attrs, prop, prefs=prefs)
    prop.attr_names = attrs.keys()

    ds = MinfluxDataset(
        file=file_info,
        prop=prop,
        attr=attrs,
        cali=Calibration(),
        channel=Channel(),
    )
    ds.metadata["source_version"] = "csv"
    ds.metadata["is_minflux"] = bool(has_real_tid)
    ds.metadata["has_real_tid"] = bool(has_real_tid)
    if is_processed_snapshot:
        # Same contract as _flat_table_from_dict: coordinates are final, so pin
        # the factor and pre-empt the post-load automatic estimation.
        ds.set_z_scaling_factor(
            1.0, source="processed (re-imported, coordinates final)")
        ds.derived["z_scaling_factor"] = np.asarray([1.0], dtype=float)
    _restore_filter_mask(ds, saved_ftr)
    if apply_sidecar:
        apply_metadata_sidecar(ds, path)
    return ds


def load_json(path: str | Path, prefs: dict | None = None, *,
              apply_sidecar: bool = True) -> MinfluxDataset:
    """
    Load Abberior-style MINFLUX JSON localization records.

    The expected shape is a list of record dictionaries, or a dictionary with
    an ``mfx``/``records``/``localizations``/``data`` list. The records are
    decoded into columns and built by :func:`load_from_mfx_columns`, the same
    builder ``.npy``, MSR, Zarr and CSV data go through, so a JSON file yields
    the same dataset -- structural version, effective CFR/EFC iteration,
    validity -- as the identical table in any other container.

    Coordinates are a ``loc`` (or ``lnc``) vector in metres, or explicit
    columns: ``loc_x``/``x`` in metres, ``loc_x_nm``/``x_nm``/``xnm`` in nm. A
    table of ``xnm``/``ynm`` columns without a ``loc`` vector is a processed
    flat table (what a snapshot export writes) and loads exactly like the same
    table in ``.npy``/``.mat``: coordinates final, Z scaling factor pinned.

    Files below ``_JSON_STREAM_BYTES`` are decoded with ``json.load`` (the
    fastest decoder measured); larger ones are streamed record by record. Both
    accept the same shapes -- a bare record array, or an object wrapping exactly
    one -- so the size picks only the decoder.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.stat().st_size >= _JSON_STREAM_BYTES:
        return _load_json_streaming_canonical(path, prefs, apply_sidecar=apply_sidecar)

    with open(path, encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    records = _extract_json_records(payload)
    if records is None or not _looks_like_minflux_json_records(records):
        raise ValueError(
            f"'{path.name}' is not recognised as MINFLUX localization JSON."
        )
    ds = _dataset_from_json_columns(_json_columns(records), path, prefs)
    if apply_sidecar:
        apply_metadata_sidecar(ds, path)
    return ds


#: At or above this size a JSON file is streamed instead of ``json.load``-ed.
_JSON_STREAM_BYTES = 128 * 1024 * 1024
#: A gap in these is a format error, not missing data: they say which row is a
#: valid localization of which trace at which iteration.
_JSON_GAP_INTOLERANT = frozenset({"vld", "fnl", "bot", "eot", "itr", "tid"})
#: Explicit coordinate columns, first match wins: (x, y, z, metres per unit).
_JSON_COORDINATE_COLUMNS = (
    ("loc_x", "loc_y", "loc_z", 1.0),
    ("x", "y", "z", 1.0),
    ("loc_x_nm", "loc_y_nm", "loc_z_nm", 1e-9),
    ("xnm", "ynm", "znm", 1e-9),
    ("x_nm", "y_nm", "z_nm", 1e-9),
)
_JSON_COORDINATE_ALIASES = frozenset(
    name for triple in _JSON_COORDINATE_COLUMNS for name in triple[:3])
_MISSING = object()


def _json_columns(records: list[dict]) -> dict[str, np.ndarray]:
    """Records -> one column per key, in first-seen key order.

    The dtype follows the JSON literals: all booleans -> ``bool``, all integers
    -> ``int64``, any float -> ``float64``, equal-length lists -> ``(N, k)``
    ``float64``. A key absent from a record is NaN there (a gap in a flag,
    ``itr`` or ``tid`` is an error naming the record). Text and objects are not
    data and are skipped, as before.
    """
    keys = list(dict.fromkeys(key for record in records for key in record))
    columns: dict[str, np.ndarray] = {}
    for key in keys:
        values = [record.get(key, _MISSING) for record in records]
        column = _json_column(str(key), values)
        if column is not None:
            columns[str(key)] = column
    return columns


def _json_column(key: str, values: list) -> np.ndarray | None:
    try:
        arr = np.asarray(values)
    except ValueError:                    # ragged lists
        arr = None
    if arr is not None and arr.ndim in (1, 2) and arr.dtype.kind in "bif":
        if arr.ndim == 2 or arr.dtype.kind == "f":
            return arr.astype(np.float64, copy=False)
        return arr                        # bool or int64
    if arr is not None and arr.dtype.kind in "US":
        return None
    return _json_column_slow(key, values)


def _json_column_slow(key: str, values: list) -> np.ndarray | None:
    """Gaps, mixed literals or ragged vectors -- the per-value path."""
    present = [v for v in values if v is not _MISSING and v is not None]
    if not present:
        return None
    gaps = len(present) != len(values)
    if gaps and key in _JSON_GAP_INTOLERANT:
        first = next(i for i, v in enumerate(values) if v is _MISSING or v is None)
        raise ValueError(f"JSON record {first} has no '{key}', which other records carry.")
    if all(isinstance(v, (list, tuple)) for v in present):
        width = max(len(v) for v in present)
        out = np.full((len(values), width), np.nan)
        try:
            for row, value in enumerate(values):
                if isinstance(value, (list, tuple)):
                    # A short vector is zero-padded (a 2-D ``loc`` gets z = 0),
                    # as the reader always did; an absent one stays NaN.
                    out[row, :] = 0.0
                    out[row, :len(value)] = [_json_number(item) for item in value]
        except (TypeError, ValueError, OverflowError):
            return None
        return out
    if any(isinstance(v, (list, tuple, dict)) for v in present):
        return None
    try:
        return np.asarray(
            [np.nan if (v is _MISSING or v is None) else _json_number(v) for v in values],
            dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None


def _json_number(value) -> float:
    """A JSON number (or boolean) as float; text is not a number here, as the
    whole-column path (where NumPy makes such a column text) already treats it."""
    kind = type(value)
    if kind is not float and kind is not int and kind is not bool:
        raise TypeError(value)
    return float(value)


#: Coordinate families in nm that mark a processed table (what the snapshot
#: export writes): final coordinates, so the Z scaling factor is pinned.
_PROCESSED_JSON_FAMILIES = (("xnm", "ynm"), ("x_nm", "y_nm"), ("loc_x_nm", "loc_y_nm"))


def _is_processed_flat_json(columns: dict) -> bool:
    """Exactly one nm coordinate family and nothing in metres: a processed table.

    ``xnm``, ``x_nm`` and ``loc_x_nm`` qualify alike, at any row count. Bare
    ``x``/``y`` never do -- they are metres in JSON, while the flat-table
    builder reads them as nanometres -- and neither does a table that also
    carries a ``loc``/``lnc`` vector or metre columns (raw data, built as such).
    """
    names = set(columns)
    families = [family for family in _PROCESSED_JSON_FAMILIES if set(family) <= names]
    raw = {"loc", "lnc", "loc_x", "loc_y", "loc_z", "x", "y", "z"} & names
    return len(families) == 1 and not raw


def _with_loc_column(columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """``columns`` with one ``loc (N, 3)`` column in metres, aliases removed.

    First match wins: a ``loc`` vector, an ``lnc`` vector (copied to ``loc``),
    then the explicit columns of ``_JSON_COORDINATE_COLUMNS``. ``loc`` takes
    the place of the first column it was made from, so the attribute order
    follows the file.
    """
    def padded(vector: np.ndarray) -> np.ndarray:
        if vector.ndim != 2 or vector.shape[1] < 2:
            raise ValueError("JSON 'loc' field must contain numeric xy or xyz values.")
        if vector.shape[1] == 2:
            vector = np.column_stack([vector, np.zeros(len(vector))])
        return vector[:, :3]

    def rebuilt(anchor: str, loc: np.ndarray, drop: set[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for key, values in columns.items():
            if key == anchor:
                out["loc"] = loc
            if key in drop or key == "loc":
                continue
            out[key] = values
        return out

    aliases = set(_JSON_COORDINATE_ALIASES) & set(columns)
    if "loc" in columns:
        return rebuilt("loc", padded(np.asarray(columns["loc"], dtype=float)), aliases)
    lnc = columns.get("lnc")
    if lnc is not None and np.ndim(lnc) == 2:
        return rebuilt("lnc", padded(np.asarray(lnc, dtype=float)).copy(), aliases)
    for x_key, y_key, z_key, scale in _JSON_COORDINATE_COLUMNS:
        x, y = columns.get(x_key), columns.get(y_key)
        if x is None or y is None or np.ndim(x) != 1 or np.ndim(y) != 1:
            continue
        z = columns.get(z_key)
        z = np.zeros(len(x)) if z is None or np.ndim(z) != 1 else z
        loc = np.column_stack([x, y, z]).astype(float) * scale
        return rebuilt(x_key, loc, aliases)
    raise ValueError("JSON data must contain loc/lnc or x/y localization coordinates.")


def _dataset_from_json_columns(columns: dict[str, np.ndarray], path: Path,
                               prefs: dict | None) -> MinfluxDataset:
    """Decoded JSON columns -> a dataset, through the shared builders."""
    datetime_str = datetime.fromtimestamp(path.stat().st_mtime).strftime(
        "%Y-%b-%d, %H:%M:%S")
    if _is_processed_flat_json(columns):
        flat = _flat_table_from_dict(dict(columns), path.name, str(path.parent), prefs)
        if flat is not None:
            flat.metadata["source_format"] = "json"
            return flat
    columns = _with_loc_column(columns)
    n = len(columns["loc"])
    has_itr = "itr" in columns
    defaults = (("tid", np.arange(n, dtype=np.int32)), ("tim", np.zeros(n)),
                ("itr", np.zeros(n, dtype=np.int32)))
    for key, value in defaults:
        columns.setdefault(key, value)
    columns = _bool_flag_columns(columns)
    ds = load_from_mfx_columns(columns, path.name, folder=str(path.parent),
                               datetime_str=datetime_str, prefs=prefs)
    if ds.prop.num_loc == 0:
        raise ValueError(f"'{path.name}' contains no valid finite localizations.")
    ds.metadata["source_format"] = "json"
    if not has_itr:
        # Records without an iteration index are not an m2410 table, whatever
        # else they carry; say "imported" rather than claim the structure.
        ds.metadata["source_version"] = "imported"
        ds.metadata.pop("source_version_detail", None)
    return ds


#: Characters that can continue a JSON number (digits, sign, point, exponent).
_NUMBER_CHARS = frozenset("0123456789+-.eE")


class _JsonTextStream:
    """A JSON text file read in chunks, one complete value at a time."""

    CHUNK = 1024 * 1024

    def __init__(self, handle) -> None:
        from json import JSONDecoder

        self._handle = handle
        self._decoder = JSONDecoder()
        self.buffer = ""
        self.position = 0
        self.consumed = 0               # characters dropped from the buffer's front
        self.eof = False

    @property
    def offset(self) -> int:
        return self.consumed + self.position

    def fill(self) -> bool:
        if self.eof:
            return False
        chunk = self._handle.read(self.CHUNK)
        if chunk:
            self.buffer += chunk
            return True
        self.eof = True
        return False

    def peek(self) -> str:
        """The next non-whitespace character ("" at the end)."""
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position] in " \t\r\n":
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if not self.fill():
                return ""

    def value(self):
        """Decode the next complete value (the caller has peeked past whitespace)."""
        from json import JSONDecodeError

        while True:
            try:
                value, end = self._decoder.raw_decode(self.buffer, self.position)
            except JSONDecodeError:
                if self.fill():
                    continue
                raise ValueError(f"Malformed JSON at character {self.offset}.") from None
            # A number cut by the end of what was read may go on in the next
            # chunk -- "12" of "1234", but also "1" of "1e-09": the decoder
            # stops before an "e" with no digits yet, so test whether the
            # characters after its end could still belong to the number, up to
            # the end of the buffer. Then read on and decode again.
            tail = end
            while tail < len(self.buffer) and self.buffer[tail] in _NUMBER_CHARS:
                tail += 1
            if tail >= len(self.buffer) and self.fill():
                continue
            self.position = end
            # Do not retain already-decoded text while a large file streams.
            if self.position >= self.CHUNK * 2:
                self.buffer = self.buffer[self.position:]
                self.consumed += self.position
                self.position = 0
            return value


def _stream_record_array(stream: _JsonTextStream):
    """Yield the objects of the array at the stream's position.

    As strict as ``json.load``: exactly one comma between records, none after
    the last. Each error names the character offset.
    """
    stream.position += 1                                  # the "["
    if stream.peek() == "]":
        stream.position += 1
        return
    while True:
        record = stream.value()
        if not isinstance(record, dict):
            raise ValueError("MINFLUX JSON records must be objects.")
        yield record
        char = stream.peek()
        if char == ",":
            stream.position += 1
            char = stream.peek()
            if char in ("]", ","):
                raise ValueError(
                    f"Extra comma in the JSON record array at character {stream.offset}.")
            if char == "":
                raise ValueError("Unexpected end of the JSON record array.")
            continue
        if char == "]":
            stream.position += 1
            return
        if char == "":
            raise ValueError("Unexpected end of the JSON record array.")
        raise ValueError(f"Missing ',' between JSON records at character {stream.offset}.")


def _several_wrappers(keys) -> ValueError:
    return ValueError(f"The JSON object holds record arrays under {', '.join(keys)}; "
                      "a MINFLUX file holds exactly one.")


def _iter_json_records(path: Path):
    """Yield the localization records of a JSON file without loading it whole.

    The same two shapes :func:`_extract_json_records` accepts: a top-level
    record array, or an object holding exactly one record array under a
    :data:`~minflux_viewer.core.json_probe.RECORD_WRAPPER_KEYS` key (its other
    members are decoded and dropped). Strict like ``json.load`` about the
    surrounding syntax, and nothing but whitespace may follow the document.
    """
    from minflux_viewer.core.json_probe import RECORD_WRAPPER_KEYS

    with path.open("r", encoding="utf-8-sig") as handle:
        stream = _JsonTextStream(handle)
        top = stream.peek()
        if top == "[":
            yield from _stream_record_array(stream)
        elif top == "{":
            stream.position += 1
            wrapper = None
            if stream.peek() == "}":
                stream.position += 1
            else:
                while True:
                    if stream.peek() != '"':
                        raise ValueError(f"Expected a key at character {stream.offset}.")
                    key = stream.value()
                    if stream.peek() != ":":
                        raise ValueError(f"Expected ':' at character {stream.offset}.")
                    stream.position += 1
                    if key in RECORD_WRAPPER_KEYS and stream.peek() == "[":
                        if wrapper is not None:
                            raise _several_wrappers((wrapper, key))
                        wrapper = key
                        yield from _stream_record_array(stream)
                    else:
                        if stream.peek() == "":
                            raise ValueError("Unexpected end of the JSON object.")
                        stream.value()               # another member: not data
                    char = stream.peek()
                    if char == ",":
                        stream.position += 1
                        continue
                    if char == "}":
                        stream.position += 1
                        break
                    raise ValueError(f"Expected ',' or '}}' at character {stream.offset}.")
            if wrapper is None:
                raise ValueError("The JSON object holds no record array under "
                                 f"{'/'.join(RECORD_WRAPPER_KEYS)}.")
        else:
            raise ValueError("MINFLUX JSON is a record array or an object wrapping one.")
        if stream.peek() != "":
            raise ValueError(
                f"Unexpected content after the JSON document at character {stream.offset}.")


def _load_json_streaming_canonical(path: Path, prefs: dict | None, *,
                                   apply_sidecar: bool = True) -> MinfluxDataset:
    """Load a large JSON record array (bare or wrapped) without a list of it.

    Values go straight into compact per-key buffers while decoding
    (:class:`_StreamedColumn`), by exactly the rules :func:`_json_column`
    applies to a whole column -- so the size threshold, which picks the
    decoder, never changes a value, a dtype or which columns are kept. The
    columns are then built by the same function as a small file's.
    """
    ds = _dataset_from_json_columns(_stream_json_columns(path), path, prefs)
    if apply_sidecar:
        apply_metadata_sidecar(ds, path)
    return ds


def _stream_json_columns(path: Path) -> dict[str, np.ndarray]:
    """The columns :func:`_json_columns` would build from the file's records,
    decoded record by record. Raises for a file that is not MINFLUX JSON."""
    iterator = _iter_json_records(path)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError(f"'{path.name}' is empty.") from None
    if not _looks_like_minflux_json_records([first]):
        raise ValueError(f"'{path.name}' is not recognised as MINFLUX localization JSON.")

    streamed: dict[str, _StreamedColumn] = {}
    rows = 0
    for record in itertools.chain((first,), iterator):
        if not record.keys() <= streamed.keys():               # a key first seen here
            for key in record:
                if key not in streamed:
                    streamed[key] = _StreamedColumn(rows)
        for key, column in streamed.items():
            column.add(record.get(key, _MISSING))
        rows += 1

    columns: dict[str, np.ndarray] = {}
    for key, column in streamed.items():
        values = column.result(str(key))
        if values is not None:
            columns[str(key)] = values
    return columns


_NAN = float("nan")


class _StreamedColumn:
    """One key's values while a large JSON file streams.

    Follows :func:`_json_column` / :func:`_json_column_slow` exactly, without
    holding the values as Python objects:

    * integers (and booleans) stay exact in an ``int64`` buffer until a float,
      a gap or an out-of-range integer turns the column into ``float64`` (as
      NumPy does for the whole column);
    * a vector column may grow: a longer vector later adds components, earlier
      vectors padded with 0 and absent rows NaN;
    * the kind (scalar or vector) is fixed by the first non-null value, so a
      column that starts with ``null`` is not lost;
    * text, objects, a scalar among vectors (or the reverse) and a
      non-numeric vector item make the column "not data", as in the small path.
    """

    __slots__ = ("rows", "first_gap", "mode", "kinds", "ints", "floats", "comps", "present")

    def __init__(self, rows: int) -> None:
        self.rows = rows                     # rows accounted for
        self.first_gap: int | None = 0 if rows else None
        self.mode: str | None = None         # None (no value yet), "scalar", "vector", "skip"
        self.kinds = 0                       # 1 bool, 2 int, 4 float
        self.ints: array | None = None       # exact integers, while no float or gap
        self.floats: array | None = None
        self.comps: list[array] = []         # a vector's components
        self.present = bytearray()           # a vector row was present (1) or absent (0)

    def _to_floats(self) -> None:
        assert self.ints is not None
        self.floats = array("d", (float(v) for v in self.ints))
        self.ints = None

    def _skip(self) -> None:
        self.mode = "skip"
        self.ints = self.floats = None
        self.comps = []
        self.present = bytearray()

    def _start(self, value) -> None:
        kind = type(value)
        if kind is list:
            self.mode = "vector"
            self.present = bytearray(self.rows)      # earlier rows: absent
        elif kind is int or kind is bool or kind is float:
            self.mode = "scalar"
            if self.rows or kind is float:
                self.floats = array("d", [_NAN]) * self.rows
            else:
                self.ints = array("q")
        else:
            self.mode = "skip"

    def add(self, value) -> None:
        # Fast paths for what nearly every row is: a number in a scalar column,
        # a full-width numeric vector in a vector column. Anything else falls
        # through to the general rules below.
        mode = self.mode
        kind = type(value)
        if mode == "scalar":
            if kind is float and self.ints is None:
                assert self.floats is not None
                self.floats.append(value)
                self.kinds |= 4
                self.rows += 1
                return
            if (kind is int or kind is bool) and self.ints is not None:
                try:
                    self.ints.append(value)
                except OverflowError:
                    pass                          # general path: to float
                else:
                    self.kinds |= 1 if kind is bool else 2
                    self.rows += 1
                    return
        elif mode == "vector" and kind is list and len(value) == len(self.comps):
            try:
                for comp, item in zip(self.comps, value):
                    item_kind = type(item)
                    if item_kind is not float and item_kind is not int and item_kind is not bool:
                        raise TypeError(item)
                    comp.append(item)
            except (TypeError, OverflowError):
                self._skip()
            else:
                self.present.append(1)
            self.rows += 1
            return
        if value is _MISSING or value is None:
            if self.first_gap is None:
                self.first_gap = self.rows
            if self.mode == "scalar":
                if self.ints is not None:
                    self._to_floats()
                assert self.floats is not None
                self.floats.append(_NAN)
            elif self.mode == "vector":
                for comp in self.comps:
                    comp.append(_NAN)
                self.present.append(0)
            self.rows += 1
            return
        if self.mode is None:
            self._start(value)
        mode = self.mode
        if mode == "scalar":
            self._add_scalar(value)
        elif mode == "vector":
            self._add_vector(value)
        self.rows += 1

    def _add_scalar(self, value) -> None:
        kind = type(value)
        if kind is int or kind is bool:
            self.kinds |= 1 if kind is bool else 2
            if self.ints is not None:
                try:
                    self.ints.append(value)
                    return
                except OverflowError:           # beyond int64: NumPy goes to float
                    self._to_floats()
            try:
                assert self.floats is not None
                self.floats.append(float(value))
            except OverflowError:                # beyond float64: not data
                self._skip()
        elif kind is float:
            self.kinds |= 4
            if self.ints is not None:
                self._to_floats()
            assert self.floats is not None
            self.floats.append(value)
        else:
            self._skip()

    def _add_vector(self, value) -> None:
        if type(value) is not list:
            self._skip()
            return
        comps = self.comps
        if len(value) > len(comps):
            pad = [0.0 if row else _NAN for row in self.present]
            while len(comps) < len(value):
                comps.append(array("d", pad))
        try:
            for j, comp in enumerate(comps):
                if j < len(value):
                    item = value[j]
                    kind = type(item)
                    if kind is not float and kind is not int and kind is not bool:
                        raise TypeError(item)
                    comp.append(float(item))
                else:
                    comp.append(0.0)              # a short vector: zero-padded
        except (TypeError, OverflowError):
            self._skip()
            return
        self.present.append(1)

    def result(self, key: str) -> np.ndarray | None:
        """The column, or None when it is not data (or never had a value)."""
        if self.mode is None:
            return None
        if self.first_gap is not None and key in _JSON_GAP_INTOLERANT:
            raise ValueError(
                f"JSON record {self.first_gap} has no '{key}', which other records carry.")
        if self.mode == "skip":
            return None
        if self.mode == "vector":
            if not self.comps:
                return np.zeros((self.rows, 0))
            return np.column_stack([np.frombuffer(c, dtype=np.float64) for c in self.comps])
        if self.floats is not None:
            return np.frombuffer(self.floats, dtype=np.float64)
        assert self.ints is not None
        values = np.frombuffer(self.ints, dtype=np.int64)
        return values.astype(bool) if self.kinds == 1 else values


def _extract_json_records(payload: Any) -> list[dict] | None:
    """The record list of a decoded document, by the streaming decoder's rule:
    a record array, or an object with exactly one record array under a wrapper
    key (several are an error, not a choice)."""
    from minflux_viewer.core.json_probe import RECORD_WRAPPER_KEYS

    records = None
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        wrapped = [key for key in payload if key in RECORD_WRAPPER_KEYS
                   and isinstance(payload[key], list)]
        if len(wrapped) > 1:
            raise _several_wrappers(wrapped)
        records = payload[wrapped[0]] if wrapped else None
    if records is None or not all(isinstance(item, dict) for item in records):
        return None
    return records


def _looks_like_minflux_json_records(records: list[dict]) -> bool:
    # The same test the classifier applies to a wrapped array's first record.
    from minflux_viewer.core.json_probe import looks_like_localization

    return bool(records) and looks_like_localization(records[0].keys())


def load_tiff(path: str | Path, prefs: dict | None = None) -> MinfluxDataset:
    """Load a TIFF image as an image-backed dataset for the render window."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        import tifffile
    except ImportError as exc:
        raise ImportError("tifffile is required to load TIFF images.") from exc

    image = tifffile.imread(str(path))
    arr = np.asarray(image)
    is_color_2d = arr.ndim == 3 and arr.shape[-1] in (3, 4)
    is_stack = (arr.ndim == 3 and not is_color_2d) or (arr.ndim == 4 and arr.shape[-1] in (3, 4))
    return build_image_dataset(
        name=path.name,
        folder=str(path.parent),
        datetime_str=datetime.fromtimestamp(path.stat().st_mtime).strftime(
            "%Y-%b-%d, %H:%M:%S"
        ),
        image=image,
        pixel_size_nm=1.0,
        attrs={
            "image_source_format": np.asarray("TIFF"),
            "image_depth_count": np.asarray(arr.shape[0] if is_stack else 1, dtype=np.int32),
            "image_shape": np.asarray(arr.shape, dtype=np.int64),
        },
    )
