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

#: Canonical nm coordinate view name → underlying metres attribute. xnm/ynm/znm
#: are exposed in the UI; loc_x/loc_y/loc_z (metres) are the hidden raw store.
_COORD_NM_BASE = {"xnm": "loc_x", "ynm": "loc_y", "znm": "loc_z"}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

#: Coordinate column aliases for a flat (exported / generic) localization table.
_FLAT_X = ("xnm", "x")
_FLAT_Y = ("ynm", "y")
_FLAT_Z = ("znm", "z")


def _flat_table_from_dict(
    raw: dict, name: str, folder: str, prefs: dict | None
) -> "MinfluxDataset | None":
    """Build a dataset from a flat localization table, or return None.

    A flat table is a dict of 1-D columns with ``xnm``/``ynm`` (or ``x``/``y``)
    coordinates and **no** structured mfx ``loc`` field — i.e. the layout written
    by *Save Processed Data*. Coordinates are treated as final/processed (already
    RIMF-applied), so RIMF is pinned to 1.0 and the post-load auto-estimation is
    suppressed; derived attributes are recomputed; the saved ``ftr`` column, if
    present, restores the filter state.
    """
    from .attributes import DERIVED_ATTRIBUTE_NAMES, LEGACY_DERIVED_ATTRIBUTE_ALIASES

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
    if x.ndim != 1 or x.size < 2 or x.size != y.size:
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
    # Re-imported processed coordinates are final: pin RIMF=1 and pre-empt the
    # post-load auto-RIMF estimation (which would otherwise double-correct z).
    ds.set_rimf(1.0, source="processed (re-imported, coordinates final)")
    ds.derived["rimf"] = np.asarray([1.0], dtype=float)
    if ftr is not None and np.asarray(ftr).ravel().size == n:
        ds.filter_mask = np.asarray(ftr).ravel().astype(bool)
    return ds


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
    only_valid      = data_prefs.get("only_valid_locs", True)

    # 1. Read the raw .mat file
    raw = _load_mat(path)

    # 1b. A flat localization table (xnm/ynm columns, no structured mfx) — e.g.
    # one written by Save Processed Data — loads generically, not via the mfx
    # parser (which would lose the coordinates).
    flat = _flat_table_from_dict(raw, path.name, str(path.parent), prefs)
    if flat is not None:
        return flat

    # 2. Build mfx_raw — all iterations, all measurements, no filtering
    mfx_raw_store = _build_mfx_raw_from_raw(raw)

    # 3. Detect format and parse into an AttrStore (filtered working set)
    attrs, num_loc, num_itr = _parse_raw(
        raw,
        load_all_itr=load_all_itr,
        load_efc_cfr=load_efc_cfr,
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
    dataset.metadata["valid_num_loc"] = _attrs_valid_count(attrs, prop.num_loc)
    dataset.metadata["includes_invalid"] = _attrs_include_invalid(attrs)
    dataset.components.mfx_raw = mfx_raw_store
    # ds.attr is not the last-valid materialization (all-iteration and/or
    # invalid-inclusive load): precompute the derived attributes at last-valid
    # so mfx_get can broadcast them to any selection.
    if not attr_matches_last_valid(dataset):
        dataset.components.derived_last = _derived_last_from_raw(mfx_raw_store, num_itr, prefs)
    apply_metadata_sidecar(dataset, path)
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
    only_valid: bool = True,
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

    # m2205: nested struct inside itr (MATLAB struct or already-dict)
    if hasattr(itr_val, "_fieldnames") or isinstance(itr_val, dict):
        return _parse_m2205(raw, load_all_itr=load_all_itr, load_efc_cfr=load_efc_cfr,
                            only_valid=only_valid)

    if isinstance(itr_val, np.ndarray) and np.issubdtype(itr_val.dtype, np.integer):
        # m2410: flat 1-D integer array — one iteration index per localization event.
        # Legacy .mat (m2205 flat): itr is 2-D (nLoc × nItr).  Check dimensionality
        # after squeezing out any size-1 axes; (N,1) and (1,N) are still 1-D.
        if np.squeeze(itr_val).ndim <= 1:
            return _parse_m2410(raw, only_valid=only_valid, load_all_itr=load_all_itr)
        # 2-D integer itr with loc at top level → m2205 flat format
        if "loc" in raw:
            return _parse_m2205_unwrapped(raw, load_all_itr=load_all_itr,
                                          load_efc_cfr=load_efc_cfr, only_valid=only_valid)

    # m2205-unwrapped: loc already at top level, itr is numeric (float or other)
    if "loc" in raw:
        return _parse_m2205_unwrapped(
            raw, load_all_itr=load_all_itr, load_efc_cfr=load_efc_cfr, only_valid=only_valid,
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
    meta.update(_effective_iteration_metadata(raw))
    return meta


def _effective_iteration_metadata(source: Any) -> dict[str, int]:
    """CFR/EFC effective-iteration metadata for the dataset info dialog.

    ``source`` is any dict-like (``.get``) exposing the per-localization /
    per-iteration arrays — the raw ``.mat`` dict (``_source_metadata_from_raw``)
    or the all-iteration ``mfx_raw`` store (``load_from_mfx_array``).
    """
    meta: dict[str, int] = {}
    cfr_iter = _effective_iteration_column(source, "cfr")
    efc_iter = _effective_iteration_column(source, "efc") or _effective_iteration_column(source, "efo")
    if cfr_iter is not None:
        meta["cfr_iteration"] = cfr_iter
    if efc_iter is not None:
        meta["efc_iteration"] = efc_iter
    return meta


# Attributes only measured at one (possibly non-last) loop iteration. Their
# materialized last-iteration value is zero/NaN, so the meaningful per-loc value
# is the value at the iteration recorded in ``metadata[<…>_iteration]``.
_EFFECTIVE_ITER_META = {"cfr": "cfr_iteration", "efc": "efc_iteration"}


def effective_iteration_for_attr(ds: "MinfluxDataset", attr: str) -> "int | None":
    """0-based iteration index where *attr* (cfr/efc) is actually measured.

    Returns ``None`` when *attr* has no special effective iteration, the
    metadata is missing, or the effective iteration is already the last one
    (so the default last-iteration materialization is correct and no special
    handling is needed).
    """
    key = _EFFECTIVE_ITER_META.get(attr)
    if key is None or ds is None:
        return None
    val = ds.metadata.get(key)
    if val is None:
        return None
    try:
        idx0 = int(val) - 1                       # 1-based metadata → 0-based
    except (TypeError, ValueError):
        return None
    n_itr = int(ds.metadata.get("raw_num_itr", getattr(ds.prop, "num_itr", 1)) or 1)
    if idx0 < 0 or idx0 >= n_itr - 1:
        return None                               # effective == last → no change
    return idx0


def _detect_raw_version(raw: dict) -> str:
    itr_val = raw.get("itr")
    if itr_val is None:
        return "unidentified"
    if isinstance(itr_val, np.ndarray) and np.issubdtype(itr_val.dtype, np.integer):
        # 1-D (or squeezable to 1-D) integer itr → m2410 flat format.
        # 2-D integer itr (nLoc × nItr) → legacy .mat m2205 flat format.
        return "m2410" if np.squeeze(itr_val).ndim <= 1 else "m2205"
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
    """Return the 1-based iteration that carries a CFR/EFC-like array.

    Handles both source layouts:

    * **m2205** 2-D ``(num_loc, num_itr)``: pick the iteration column whose
      values carry the most signal (``sum |value|``) across valid localizations.
    * **m2410** flat 1-D (one row per localization-iteration, alongside a 1-D
      ``itr`` index column): group rows by iteration index and pick the
      iteration that carries the most signal across valid rows.

    In m2410 acquisitions CFR/EFC are only measured at one (or a few) of the
    loop iterations and are zero/NaN elsewhere — the analog of the m2205
    "one non-zero column", just flattened — so the same dominant-signal rule
    applies to both.
    """
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
        if arr.dtype == object or arr.ndim == 0 or itr_arr.ndim == 0:
            return None

        # m2410 flat layout: 1-D itr index + a per-row field of equal length
        # ((N,) or a squeezable (N, 1)).
        if itr_arr.ndim == 1 and arr.size == itr_arr.size:
            return _effective_iteration_flat(itr_arr, arr, merged.get("vld"))

        # m2205 2-D layout: (num_loc, num_itr).
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


def _effective_iteration_flat(itr_arr, field_arr, vld_val) -> int | None:
    """Effective (1-based) iteration for a flat m2410 per-row field.

    Groups rows by iteration index and returns the iteration carrying the most
    total signal (``sum |value|``, NaN treated as 0) across valid rows.
    ``None`` when the field/itr are non-numeric or no iteration carries signal.
    """
    itr = np.asarray(itr_arr).ravel()
    field = np.asarray(field_arr).ravel()
    if itr.size == 0 or field.size != itr.size:
        return None
    if not np.issubdtype(itr.dtype, np.number) or not np.issubdtype(field.dtype, np.number):
        return None
    if vld_val is not None:
        vld = np.asarray(vld_val, dtype=bool).ravel()
        if vld.size == itr.size:
            itr = itr[vld]
            field = field[vld]
    if itr.size == 0 or itr.min() < 0:
        return None
    weight = np.abs(field.astype(np.float64, copy=False))
    weight[~np.isfinite(weight)] = 0.0
    sums = np.bincount(itr.astype(np.int64, copy=False), weights=weight)
    if sums.size == 0 or not np.any(sums > 0):
        return None
    return int(np.argmax(sums)) + 1


def _dcr_is_two_channel(arr: np.ndarray, sample: int = 500) -> bool:
    """
    Heuristic: return True when a (N, 2) dcr array holds m2410-style
    two-channel ratios ch1/(ch1+ch2) and ch2/(ch1+ch2).

    In that format each row sums to 1.0 (within floating-point tolerance).
    In legacy m2205 format with exactly 2 iterations the same (N, 2) shape
    arises, but the two columns are independent per-iteration ratios that
    generally do NOT sum to 1.
    """
    n = min(sample, arr.shape[0])
    row_sums = arr[:n, 0] + arr[:n, 1]
    finite = np.isfinite(row_sums)
    if not finite.any():
        return False
    return bool(np.all(np.abs(row_sums[finite] - 1.0) < 1e-6))


# ---------------------------------------------------------------------------
# m2410 parser
# ---------------------------------------------------------------------------

def _parse_m2410(
    raw: dict,
    *,
    only_valid: bool = True,
    load_all_itr: bool = False,
) -> tuple[AttrStore, int, int]:
    """
    Flat Abberior m2410 format.

    When ``load_all_itr`` is False (default), only the last (maximum)
    iteration index is kept. When True, all vld-passing rows are kept.

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

    if only_valid:
        vld = np.asarray(
            raw.get("vld", np.ones(n_raw, dtype=bool)), dtype=bool,
        ).ravel()
    else:
        vld = np.ones(n_raw, dtype=bool)
    max_itr = int(itr[vld].max()) if vld.any() else 0
    num_itr = max_itr + 1

    mask = vld if load_all_itr else vld & (itr == max_itr)
    num_loc = int(mask.sum())

    attrs   = AttrStore()

    for key, val in raw.items():
        arr = np.asarray(val)

        # Skip scalars, empty containers, the HDF5 "#refs#" placeholder,
        # and tic (obsolete raw FPGA-tick timestamp — tim already provides
        # calibrated seconds, so tic is redundant and often large).
        if arr.ndim == 0 or arr.size == 0 or key in ("#refs#", "tic"):
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
    only_valid: bool = True,
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
        merged, load_all_itr=load_all_itr, load_efc_cfr=load_efc_cfr, only_valid=only_valid,
    )


# ---------------------------------------------------------------------------
# m2205 unwrapped parser
# ---------------------------------------------------------------------------

def _parse_m2205_unwrapped(
    raw: dict,
    load_all_itr: bool = False,
    load_efc_cfr: bool = True,
    only_valid: bool = True,
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

    if only_valid:
        vld = np.asarray(raw.get("vld", np.ones(num_loc, bool)), dtype=bool).ravel()
    else:
        vld = np.ones(num_loc, dtype=bool)
    num_loc_vld  = int(vld.sum())
    iter_last    = num_itr - 1

    attrs = AttrStore()

    for key, val in raw.items():
        arr = np.asarray(val)

        # Skip scalars, empty containers, the HDF5 "#refs#" placeholder,
        # and tic (obsolete raw FPGA-tick timestamp — redundant with tim).
        if arr.ndim == 0 or arr.size == 0 or key in ("#refs#", "tic"):
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

        # ── Detection channel ratio ─────────────────────────────────────
        # m2410-style: (N, 2) with ch1/(ch1+ch2) and ch2/(ch1+ch2) — rows
        # sum to 1.0; take the first column.
        # m2205-style: (N, num_itr) single-channel ratio per iteration —
        # rows do NOT sum to 1.0 in general; fall through to per-iteration.
        # When num_itr != 2 shape alone is enough.  When num_itr == 2 the
        # shapes are identical, so we sample up to 500 rows and check whether
        # the two columns sum to 1.0 to decide.
        if key == "dcr" and arr.ndim == 2 and arr.shape[0] == num_loc:
            if arr.shape[1] == 2 and (num_itr != 2 or _dcr_is_two_channel(arr)):
                attrs[key] = arr[vld, 0].ravel()
                continue
            # else: fall through to per-iteration handling below

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
# mfx_raw builders — all-iteration, all-vld, no filtering
# ---------------------------------------------------------------------------

def _build_mfx_raw_m2410(raw: dict) -> "AttrStore":
    """
    Build flat all-itr, all-vld mfx_raw from m2410 raw dict.

    Mirrors _parse_m2410 structure but applies no vld/itr mask.
    Spatial coordinates are split into _x/_y/_z. Two-column dcr keeps
    both channels as dcr_0 and dcr_1 plus dcr (primary).
    """
    itr_raw = np.asarray(raw["itr"])
    n_raw   = itr_raw.ravel().size
    mfx     = AttrStore()

    for key, val in raw.items():
        arr = np.asarray(val)
        if arr.ndim == 0 or arr.size == 0 or key in ("#refs#", "tic"):
            continue

        if arr.ndim == 1:
            if arr.shape[0] != n_raw:
                continue
            arr2 = arr.reshape(n_raw, 1)
        elif arr.ndim == 2:
            if arr.shape[0] == n_raw:
                arr2 = arr
            elif arr.shape[1] == n_raw:
                arr2 = arr.T
            else:
                continue
        elif arr.ndim == 3:
            if arr.shape[0] == n_raw:
                arr2 = arr[:, -1, :]
            else:
                continue
        else:
            continue

        n_cols = arr2.shape[1]

        if key in SPATIAL_COORD_FIELDS and n_cols in (2, 3):
            mfx[f"{key}_x"] = arr2[:, 0].ravel()
            mfx[f"{key}_y"] = arr2[:, 1].ravel()
            mfx[f"{key}_z"] = arr2[:, 2].ravel() if n_cols == 3 else np.zeros(n_raw)
        elif key == "dcr" and n_cols > 1:
            mfx["dcr"] = arr2[:, 0].ravel()
            for c in range(n_cols):
                mfx[f"dcr_{c}"] = arr2[:, c].ravel()
        else:
            mfx[key] = arr2.ravel() if n_cols == 1 else arr2

    return mfx


def _build_mfx_raw_m2205_unwrapped(merged: dict, n_loc: int, n_itr: int) -> "AttrStore":
    """
    Build flat all-itr, all-vld mfx_raw from m2205 (unwrapped) format.

    Row ordering matches C-contiguous (num_loc × num_itr) reshape:
    [loc_0_itr_0, loc_0_itr_1, ..., loc_1_itr_0, ...].

    Scalar-per-loc fields are np.repeat'd n_itr times.
    Per-iteration (n_loc, n_itr) fields are .ravel()'d.
    Spatial (n_loc, n_itr, 3) fields are reshaped to (n_total, 3).
    The synthetic ``itr`` column is np.tile(0…n_itr-1, n_loc).
    """
    n_total = n_loc * n_itr
    mfx     = AttrStore()

    for key, val in merged.items():
        arr = np.asarray(val)
        if arr.ndim == 0 or arr.size == 0 or key in ("#refs#", "tic", "itr"):
            continue
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(float, copy=False)

        # Normalize leading axis to n_loc
        if arr.ndim == 1:
            if arr.shape[0] != n_loc:
                continue
        elif arr.ndim == 2:
            if arr.shape[0] == n_loc:
                pass
            elif arr.shape[1] == n_loc:
                arr = arr.T
            else:
                continue
        elif arr.ndim == 3:
            if arr.shape[0] == n_loc and arr.shape[-1] in (2, 3):
                pass
            elif arr.shape[-1] == n_loc and arr.shape[0] in (2, 3):
                arr = np.moveaxis(arr, (0, -1), (-1, 0))
            else:
                continue
        else:
            continue

        # Spatial (n_loc, n_itr, 2/3)
        if key in SPATIAL_COORD_FIELDS and arr.ndim == 3 and arr.shape[-1] in (2, 3):
            if arr.shape[1] == n_itr:
                flat = arr.reshape(n_total, arr.shape[2])
            else:
                # single-iteration xyz, interleave-repeat for all itr
                sub = arr[:, -1, :]
                flat = sub[np.repeat(np.arange(n_loc), n_itr), :]
            mfx[f"{key}_x"] = flat[:, 0]
            mfx[f"{key}_y"] = flat[:, 1]
            mfx[f"{key}_z"] = flat[:, 2] if arr.shape[-1] == 3 else np.zeros(n_total)
            continue

        # Spatial (n_loc, 2/3), not per-iteration
        if key in SPATIAL_COORD_FIELDS and arr.ndim == 2 and arr.shape[1] in (2, 3) and arr.shape[1] != n_itr:
            sub = arr[np.repeat(np.arange(n_loc), n_itr), :]
            mfx[f"{key}_x"] = sub[:, 0]
            mfx[f"{key}_y"] = sub[:, 1]
            mfx[f"{key}_z"] = sub[:, 2] if arr.shape[1] == 3 else np.zeros(n_total)
            continue

        # DCR: two-channel vs per-iteration
        if key == "dcr" and arr.ndim == 2 and arr.shape[0] == n_loc:
            if arr.shape[1] == 2 and (n_itr != 2 or _dcr_is_two_channel(arr)):
                mfx["dcr"]   = np.repeat(arr[:, 0], n_itr)
                mfx["dcr_0"] = np.repeat(arr[:, 0], n_itr)
                mfx["dcr_1"] = np.repeat(arr[:, 1], n_itr)
                continue
            if arr.shape[1] == n_itr:
                mfx["dcr"] = arr.ravel()
                continue

        # Per-iteration (n_loc, n_itr)
        if arr.ndim == 2 and arr.shape[1] == n_itr:
            mfx[key] = arr.ravel()
            continue

        # Scalar per loc (MATLAB column vectors arrive as (n_loc, 1) — keep
        # scalar columns 1-D in the raw store)
        if arr.ndim == 1:
            mfx[key] = np.repeat(arr, n_itr)
        else:
            sub = arr[np.repeat(np.arange(n_loc), n_itr), :]
            mfx[key] = sub.ravel() if sub.ndim == 2 and sub.shape[1] == 1 else sub

    # Synthetic 0-indexed itr column for consistency across formats
    mfx["itr"] = np.tile(np.arange(n_itr, dtype=np.int16), n_loc)
    return mfx


def _build_mfx_raw_from_raw(raw: dict) -> "AttrStore":
    """Format-dispatch: build mfx_raw from a loaded .mat raw dict."""
    if "itr" not in raw:
        return AttrStore()

    itr_val = raw["itr"]

    # m2205 nested struct
    if hasattr(itr_val, "_fieldnames") or isinstance(itr_val, dict):
        itr_dict = _mat_struct_to_dict(itr_val) if hasattr(itr_val, "_fieldnames") else itr_val
        merged = {k: v for k, v in raw.items() if k != "itr"}
        merged.update(itr_dict)
        merged.pop("mbm", None)
        itr_arr = np.asarray(merged.get("itr", np.array([])))
        if itr_arr.ndim == 2:
            if itr_arr.shape[0] == 1:
                n_loc, n_itr = int(itr_arr.shape[1]), 1
            else:
                n_loc, n_itr = int(itr_arr.shape[0]), int(itr_arr.shape[1])
        elif itr_arr.ndim == 1:
            n_loc, n_itr = int(itr_arr.shape[0]), 1
        else:
            return AttrStore()
        return _build_mfx_raw_m2205_unwrapped(merged, n_loc, n_itr)

    if isinstance(itr_val, np.ndarray) and np.issubdtype(itr_val.dtype, np.integer):
        if np.squeeze(itr_val).ndim <= 1:
            return _build_mfx_raw_m2410(raw)
        # 2-D integer itr → m2205-flat with loc at top level
        if "loc" in raw:
            itr_arr = np.asarray(itr_val)
            if itr_arr.shape[0] == 1:
                n_loc, n_itr = int(itr_arr.shape[1]), 1
            else:
                n_loc, n_itr = int(itr_arr.shape[0]), int(itr_arr.shape[1])
            return _build_mfx_raw_m2205_unwrapped(raw, n_loc, n_itr)

    # m2205-unwrapped with float/other itr at top level
    if "loc" in raw:
        itr_arr = np.asarray(itr_val)
        if itr_arr.ndim == 1:
            n_loc, n_itr = int(itr_arr.shape[0]), 1
        elif itr_arr.shape[0] == 1:
            n_loc, n_itr = int(itr_arr.shape[1]), 1
        else:
            n_loc, n_itr = int(itr_arr.shape[0]), int(itr_arr.shape[1])
        return _build_mfx_raw_m2205_unwrapped(raw, n_loc, n_itr)

    return AttrStore()


def _build_mfx_raw_from_json_records(records: list) -> "AttrStore":
    """Build mfx_raw from ALL JSON records (no vld/itr filtering)."""
    n = len(records)
    if n == 0:
        return AttrStore()
    mfx = AttrStore()

    try:
        loc_m = _json_coordinate_array_m(records)
    except Exception:
        return AttrStore()
    mfx["loc_x"] = loc_m[:, 0]
    mfx["loc_y"] = loc_m[:, 1]
    mfx["loc_z"] = loc_m[:, 2]

    keys = sorted({key for record in records for key in record.keys()})
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
        values = _json_values(records, key)
        if values is None:
            continue

        if key in vector_xyz_keys:
            arr_m = _json_vector_array(values, width=3)
            if arr_m is None:
                continue
            mfx[f"{key}_x"] = arr_m[:, 0]
            mfx[f"{key}_y"] = arr_m[:, 1]
            mfx[f"{key}_z"] = arr_m[:, 2] if arr_m.shape[1] > 2 else np.zeros(n)
            continue

        vector = _json_vector_array(values)
        if vector is not None:
            if key == "dcr":
                mfx["dcr"] = vector[:, 0]
                for col in range(vector.shape[1]):
                    mfx[f"dcr_{col}"] = vector[:, col]
            elif vector.shape[1] in (2, 3):
                mfx[f"{key}_x"] = vector[:, 0]
                mfx[f"{key}_y"] = vector[:, 1]
                mfx[f"{key}_z"] = vector[:, 2] if vector.shape[1] > 2 else np.zeros(n)
            continue

        scalar = _json_scalar_array(values)
        if scalar is not None and scalar.shape[0] == n:
            mfx[key] = scalar

    if "tid" not in mfx:
        mfx["tid"] = np.arange(n, dtype=np.int32)
    if "tim" not in mfx:
        mfx["tim"] = np.zeros(n, dtype=float)
    if "itr" not in mfx:
        mfx["itr"] = np.zeros(n, dtype=np.int16)
    if "vld" not in mfx:
        mfx["vld"] = np.ones(n, dtype=bool)
    return mfx


# Derived attributes computed from the (last-iteration) localization position,
# sequence, or trace. They have no genuine per-iteration value; in non-default
# iteration views each localization's last-valid value is duplicated across all
# of its raw rows (see :func:`mfx_get`). ``idx`` is handled separately as a
# synthetic positional index.
_DERIVED_BROADCAST_ATTRS = frozenset(
    {"den", "dst", "spd", "dt", "tim_trace", "siz", "dur", "len"}
)


def _mfx_raw_len(raw: "AttrStore") -> int:
    """Number of rows in the flattened raw store (0 if empty)."""
    arr = raw.get("itr")
    if arr is None:
        for key in raw.keys():
            arr = raw.get(key)
            break
    return int(np.asarray(arr).shape[0]) if arr is not None else 0


def _raw_loc_id(raw: "AttrStore") -> "np.ndarray | None":
    """Per-row localization-group id over the flattened raw store (cached).

    A group is the set of raw rows belonging to one localization: its
    (possibly partial) iteration ladder up to the final row. A new group
    starts whenever the trace id changes or the iteration index fails to
    increase within a trace. This single rule covers m2205-style fixed
    ``n_loc × n_itr`` grids, m2410 flat event streams (variable iterations
    per localization, traces interleaved in time, repeated final-iteration
    rows in tracking sequences), and single-iteration stores. Computed once
    and cached in the store as ``loc_id``.
    """
    cached = raw.get("loc_id")
    if cached is not None:
        return np.asarray(cached).ravel()
    n = _mfx_raw_len(raw)
    if n == 0:
        return None
    itr = np.asarray(raw.get("itr", np.zeros(n, dtype=np.int64))).ravel().astype(np.int64)
    tid_arr = raw.get("tid")
    if tid_arr is None:
        # No trace ids: groups are contiguous strictly-increasing itr runs.
        loc_id = np.cumsum(np.r_[True, np.diff(itr) <= 0]) - 1
    else:
        tid = np.asarray(tid_arr).ravel()
        order = np.argsort(tid, kind="stable")  # stable: keeps row order per trace
        itr_s, tid_s = itr[order], tid[order]
        new_s = np.r_[True, (tid_s[1:] != tid_s[:-1]) | (np.diff(itr_s) <= 0)]
        loc_id = np.empty(n, dtype=np.int64)
        loc_id[order] = np.cumsum(new_s) - 1
    raw["loc_id"] = loc_id
    return loc_id


def _derived_last_from_raw(
    raw: "AttrStore", num_itr: int, prefs: dict | None,
) -> "AttrStore":
    """Compute broadcastable derived attributes at the last-valid selection.

    Re-materializes ``loc``/``tid``/``tim`` from the raw store at
    ``itr="last", vld_only=True`` and runs the same trace-structure +
    derived-attribute pipeline as an ``iter_load="last"`` load, so the values
    are identical regardless of the iteration load mode. The result is aligned
    to ``mfx_row_mask(raw, itr="last", vld_only=True)`` rows. ``den`` is not
    computed here — it is appended at dataset-add time where the density
    preferences apply.
    """
    out = AttrStore()
    mask = mfx_row_mask(raw, itr="last", vld_only=True)
    if mask is None or not mask.any():
        return out
    n = int(mask.sum())
    temp = AttrStore()
    for key in ("loc_x", "loc_y", "loc_z", "tid", "tim"):
        val = raw.get(key)
        if val is not None:
            arr = np.asarray(val)
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr.ravel()   # tolerate (n, 1) columns in older stores
            if arr.ndim == 1 and arr.shape[0] == mask.size:
                temp[key] = arr[mask]
    if "loc_x" not in temp:
        # No coordinates in the raw store — computing position-based derived
        # attributes from zero-filled fallbacks would fabricate values.
        return out
    prop = _compute_properties(temp, n, num_itr, prefs=prefs)
    temp = _add_derived_attributes(temp, prop, prefs=prefs)
    for key in sorted(_DERIVED_BROADCAST_ATTRS):
        val = temp.get(key)
        if val is not None:
            out[key] = np.asarray(val)
    return out


def mfx_row_mask(
    raw: "AttrStore",
    *,
    itr: "str | int" = "last",
    vld_only: bool = True,
) -> "np.ndarray | None":
    """Boolean row selection over the flattened raw store for one (itr, vld_only).

    Rows are the m2410-style flat ``(localization, iteration)`` events. ``itr``
    is ``"last"`` (final valid iteration), ``"all"`` (every iteration), or a
    0-based iteration index. Returns ``None`` when the store is empty.
    """
    n = _mfx_raw_len(raw)
    if n == 0:
        return None
    itr_col = np.asarray(raw.get("itr", np.zeros(n, int))).ravel()
    vld_col = np.asarray(raw.get("vld", np.ones(n, bool)), dtype=bool).ravel()
    base = vld_col if vld_only else np.ones(n, bool)
    if itr == "last":
        max_i = int(itr_col[base].max()) if base.any() else 0
        return base & (itr_col == max_i)
    if itr == "all":
        return base
    if isinstance(itr, (int, np.integer)):
        return base & (itr_col == int(itr))
    return base


def _broadcast_derived(
    ds: "MinfluxDataset",
    attr: str,
    raw: "AttrStore",
    mask: np.ndarray,
) -> "np.ndarray | None":
    """Duplicate a derived attribute's per-localization (last-valid) values
    onto the raw rows selected by ``mask``.

    The source values are aligned to ``mfx_row_mask(itr="last",
    vld_only=True)`` rows: ``ds.attr`` when that IS the materialization,
    otherwise ``ds.components.derived_last``. Rows whose localization group has
    no last-valid row (failed probes, traces that never reach the final
    iteration) get NaN.
    """
    if attr_matches_last_valid(ds):
        src = ds.attr.get(attr)
    else:
        src = ds.components.derived_last.get(attr)
    if src is None:
        return None
    src = np.asarray(src).ravel()
    last_mask = mfx_row_mask(raw, itr="last", vld_only=True)
    if last_mask is None or src.shape[0] != int(last_mask.sum()):
        return None
    loc_id = _raw_loc_id(raw)
    if loc_id is None:
        return None
    per_loc = np.full(int(loc_id.max()) + 1, np.nan)
    per_loc[loc_id[last_mask]] = src.astype(float, copy=False)
    return per_loc[loc_id[mask]]


def _gather_iteration_values(
    raw: "AttrStore",
    attr: str,
    source_mask: np.ndarray,
    target_mask: "np.ndarray | None",
) -> "np.ndarray | None":
    """Per-localization value of *attr* taken from ``source_mask`` rows, indexed
    onto ``target_mask`` rows.

    The inverse of :func:`_broadcast_derived`: instead of spreading a last-valid
    per-loc value across all rows, it gathers each localization's value from the
    rows selected by ``source_mask`` (e.g. its effective-iteration row) and lays
    it onto the rows selected by ``target_mask``. NaN for groups absent from
    ``source_mask``. Both masks index the same flattened raw store.
    """
    arr = raw.get(attr)
    if arr is None or target_mask is None or source_mask is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim != 1:
        return None
    loc_id = _raw_loc_id(raw)
    if loc_id is None:
        return None
    per_loc = np.full(int(loc_id.max()) + 1, np.nan)
    per_loc[loc_id[source_mask]] = arr[source_mask].astype(float, copy=False)
    return per_loc[loc_id[target_mask]]


def effective_attr_values_1d(ds: "MinfluxDataset", attr: str) -> "np.ndarray | None":
    """``ds.attr``-aligned values of *attr* (cfr/efc) at its effective iteration.

    cfr/efc are only measured at one loop iteration; their materialized
    last-iteration value is zero/NaN. This gathers each localization's value
    from its effective-iteration row (:func:`effective_iteration_for_attr`) and
    aligns it to ``ds.attr`` rows, so filters/histograms see the real per-loc
    value. Returns ``None`` when *attr* has no effective iteration, the raw
    store is empty, or the materialized rows have no aligned raw selection.
    """
    eff = effective_iteration_for_attr(ds, attr)
    if eff is None:
        return None
    raw = getattr(ds, "mfx_raw", None)
    if raw is None or len(raw) == 0:
        return None
    rows = _attr_row_selection(ds)
    if rows is None:
        return None
    target_mask = mfx_row_mask(raw, itr=rows[0], vld_only=rows[1])
    source_mask = mfx_row_mask(raw, itr=eff, vld_only=True)
    vals = _gather_iteration_values(raw, attr, source_mask, target_mask)
    if vals is None:
        return None
    arr = np.asarray(vals)
    if arr.ndim == 1 and arr.shape[0] == int(getattr(ds.prop, "num_loc", 0)):
        return arr
    return None


def mfx_get(
    ds: "MinfluxDataset",
    attr: str,
    *,
    itr: "str | int" = "last",
    vld_only: bool = True,
) -> "np.ndarray | None":
    """
    Retrieve an attribute from a dataset's raw mfx store.

    Parameters
    ----------
    ds :
        Dataset to query.
    attr :
        Attribute name, e.g. ``"efo"``, ``"cfr"``, ``"loc_x"``, ``"idx"``.
    itr :
        ``"last"`` — final iteration only (default).
        ``"all"``  — all iterations pooled.
        ``int``    — specific 0-based iteration index.
    vld_only :
        When True, return only rows where ``vld`` is True.

    Notes
    -----
    ``idx`` is synthetic: the **1-based flattened row position** of each selected
    row (``all`` → ``1,2,3,…``; ``last`` → ``n_itr, 2·n_itr,…``). Derived
    attributes (``den``, ``dst``, ``spd``, …) have no per-iteration values:
    each localization's last-valid value is duplicated across all of its raw
    rows (NaN for rows whose localization has no last-valid materialization,
    e.g. invalid probes). The per-localization source is ``ds.attr`` for
    last-valid loads, else ``ds.components.derived_last``; ``None`` is returned
    when neither holds the attribute.
    """
    # xnm/ynm/znm are the canonical nm coordinate views of the metres store.
    if attr in _COORD_NM_BASE:
        base = _COORD_NM_BASE[attr]
        v = mfx_get(ds, base, itr=itr, vld_only=vld_only)
        if v is None:
            return None
        v = np.asarray(v, dtype=float) * 1.0e9
        if attr == "znm":
            v = v * float(getattr(ds.cali, "RIMF", 1.0) or 1.0)
        return v

    raw = getattr(ds, "mfx_raw", None)
    if raw is None or len(raw) == 0:
        val = ds.attr.get(attr)
        return np.asarray(val) if val is not None else None

    mask = mfx_row_mask(raw, itr=itr, vld_only=vld_only)
    if mask is None:
        return None

    # Synthetic 1-based flattened index of the selected rows.
    if attr == "idx":
        return np.flatnonzero(mask).astype(np.int64) + 1

    # Derived attributes: duplicate each localization's last-valid value
    # across its iteration rows.
    if attr in _DERIVED_BROADCAST_ATTRS:
        if itr == "last" and vld_only and attr_matches_last_valid(ds):
            val = ds.attr.get(attr)
            return np.asarray(val) if val is not None else None
        return _broadcast_derived(ds, attr, raw, mask)

    arr = raw.get(attr)
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 0 or arr.size == 0:
        return arr

    return arr[mask] if arr.ndim == 1 else arr[mask, :]


def mfx_filter_mask(
    ds: "MinfluxDataset",
    *,
    itr: "str | int" = "last",
    vld_only: bool = True,
) -> "tuple[np.ndarray, list[str]] | None":
    """Re-evaluate the dataset's persisted filter specs over a raw-store selection.

    The dataset's active filters are stored as re-evaluable specs in
    ``ds.state["filter_specs"]`` (attribute, mode, bounds). This re-applies
    them against :func:`mfx_get` rows for the given ``itr``/``vld_only``
    selection, so the same logical filter extends to any iteration and to
    invalid localizations.

    Returns
    -------
    (mask, unevaluable) or None
        ``mask`` is a boolean array aligned with ``mfx_get(ds, <attr>, itr=itr,
        vld_only=vld_only)``. ``unevaluable`` lists attribute names of specs that
        could not be evaluated on the raw store (e.g. derived attributes such as
        ``den`` that have no per-iteration values). Returns ``None`` when the
        dataset has no raw store.
    """
    from ..utils.filters import raw_spec_mask

    tid_sel = mfx_get(ds, "tid", itr=itr, vld_only=vld_only)
    if tid_sel is None:
        return None
    tid_sel = np.asarray(tid_sel).ravel()
    n = tid_sel.shape[0]

    raw = getattr(ds, "mfx_raw", None)
    browse_mask = (
        mfx_row_mask(raw, itr=itr, vld_only=vld_only)
        if raw is not None and len(raw) else None
    )

    specs = ds.state.get("filter_specs") or []
    mask = np.ones(n, dtype=bool)
    unevaluable: list[str] = []
    for spec in specs:
        attr = spec.get("attribute", "")
        # cfr/efc are per-localization properties measured at one iteration; the
        # filter must use that effective value broadcast onto the browse rows,
        # not the (zero/NaN) value at whatever iteration is being viewed.
        eff = effective_iteration_for_attr(ds, attr)
        if eff is not None and browse_mask is not None:
            src_mask = mfx_row_mask(raw, itr=eff, vld_only=True)
            vals = _gather_iteration_values(raw, attr, src_mask, browse_mask)
        else:
            vals = mfx_get(ds, attr, itr=itr, vld_only=vld_only)
        if vals is None or np.asarray(vals).ravel().shape[0] != n:
            if attr and attr not in unevaluable:
                unevaluable.append(attr)
            continue
        mask &= raw_spec_mask(
            np.asarray(vals).ravel(),
            tid_sel,
            spec.get("mode", "per loc"),
            float(spec.get("lo", -np.inf)),
            float(spec.get("hi", np.inf)),
            bool(spec.get("lo_inc", True)),
            bool(spec.get("hi_inc", True)),
        )
    return mask, unevaluable


def apply_saved_filters(ds: "MinfluxDataset") -> bool:
    """Apply ``ds.state['filter_specs']`` over the materialized attributes and
    set ``ds.filter_mask``; returns True if at least one spec was applied.

    Resolves each attribute through :func:`attr_values_1d`, so coordinate
    (``xnm``/``ynm``/``znm``) and derived (``den`` …) filters map to per-loc
    values — mirroring the live FilterDialog. A spec whose attribute is not yet
    materialized (e.g. ``den`` before post-load density) is skipped; callers
    re-run this once those attributes exist.
    """
    from ..utils.filters import raw_spec_mask

    specs = ds.state.get("filter_specs") or []
    if not specs:
        return False
    n = int(ds.prop.num_loc)
    tid_v = attr_values_1d(ds, "tid")
    tid = np.arange(n) if tid_v is None else np.asarray(tid_v).ravel()
    mask = np.ones(n, dtype=bool)
    applied = False
    for spec in specs:
        vals = attr_values_1d(ds, str(spec.get("attribute", "")))
        if vals is None:
            continue
        vals = np.asarray(vals).ravel()
        if vals.shape[0] != n:
            continue
        mask &= raw_spec_mask(
            vals.astype(float), tid, spec.get("mode", "per loc"),
            float(spec.get("lo", -np.inf)), float(spec.get("hi", np.inf)),
            bool(spec.get("lo_inc", True)), bool(spec.get("hi_inc", True)),
        )
        applied = True
    if applied:
        ds.filter_mask = mask
    return applied


def apply_metadata_sidecar(ds: "MinfluxDataset", data_path: "str | Path") -> bool:
    """Restore the processing recipe from a ``<stem>_metadata.json`` sidecar.

    If a metadata sidecar sits next to *data_path*, restore RIMF (pinned via
    ``ds.derived['rimf']`` so post-load auto-estimation is suppressed), the
    overlay transform, and the filter specs (and apply what is evaluable now).
    Returns True if a sidecar was consumed.
    """
    from .save import is_metadata_json_payload, metadata_sidecar_path

    side = metadata_sidecar_path(data_path)
    try:
        if not side.is_file():
            return False
        meta = json.loads(side.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not is_metadata_json_payload(meta):
        return False

    rimf = (meta.get("calibration") or {}).get("rimf")
    if rimf is not None:
        try:
            ds.set_rimf(float(rimf), source="processed (metadata sidecar)")
            ds.derived["rimf"] = np.asarray([float(rimf)], dtype=float)
        except Exception:
            pass

    tf = meta.get("transform")
    if tf is not None:
        try:
            ds.state["overlay_transform"] = np.asarray(tf, dtype=float)
        except Exception:
            pass

    filters = meta.get("filters") or []
    if filters:
        ds.state["filter_specs"] = list(filters)
        apply_saved_filters(ds)
    return True


def _attrs_include_invalid(attrs: "AttrStore") -> bool:
    """True when the materialized store retains rows with ``vld == False``."""
    vld = attrs.get("vld")
    if vld is None:
        return False
    try:
        return bool(np.any(~np.asarray(vld).astype(bool).ravel()))
    except Exception:
        return False


def _attrs_valid_count(attrs: "AttrStore", fallback: int) -> int:
    """Number of vld rows in the materialized store (= loaded rows when the
    load was valid-only)."""
    vld = attrs.get("vld")
    if vld is None:
        return int(fallback)
    try:
        return int(np.asarray(vld).astype(bool).ravel().sum())
    except Exception:
        return int(fallback)


def attr_materialized_selection(ds: "MinfluxDataset") -> tuple[str, bool]:
    """The raw-store selection ``(itr, vld_only)`` that ``ds.attr`` materializes.

    Derived from the load metadata: ``iteration_load_mode`` ("last"/"all")
    and ``includes_invalid`` (set when ``only_valid_locs=False`` actually
    retained invalid rows).
    """
    mode = str(ds.metadata.get("iteration_load_mode", "last") or "last").lower()
    itr_sel = "all" if mode == "all" else "last"
    vld_only = not bool(ds.metadata.get("includes_invalid", False))
    return itr_sel, vld_only


def _selection_rows_match_attr(
    ds: "MinfluxDataset", itr: "str | int", vld_only: bool,
) -> bool:
    """True when the raw-store selection's row count equals ``ds.attr`` rows."""
    raw = getattr(ds, "mfx_raw", None)
    if raw is None or len(raw) == 0:
        return True
    mask = mfx_row_mask(raw, itr=itr, vld_only=vld_only)
    if mask is None:
        return True
    return int(mask.sum()) == int(getattr(ds.prop, "num_loc", 0))


def _attr_row_selection(ds: "MinfluxDataset") -> "tuple[str, bool] | None":
    """The raw-store selection whose rows align 1:1 with ``ds.attr`` rows.

    Usually the materialized selection from the load metadata; m2205-style
    ``iter_load="all"`` stores keep one row per localization (2-D
    per-iteration columns), so their rows align with the ``last`` selection
    instead. ``None`` when no candidate matches (no aligned raw view exists).
    """
    sel_itr, sel_vld = attr_materialized_selection(ds)
    for cand_itr in dict.fromkeys((sel_itr, "last")):
        if _selection_rows_match_attr(ds, cand_itr, sel_vld):
            return cand_itr, sel_vld
    return None


def attr_matches_selection(
    ds: "MinfluxDataset",
    *,
    itr: "str | int" = "last",
    vld_only: bool = True,
) -> bool:
    """True when ``ds.attr`` rows ARE the given raw-store selection.

    Generalizes :func:`attr_matches_last_valid` to any (iteration, validity)
    selection, so the windows can use the materialized fast path (filter
    edit, ROI selection) for whatever selection the dataset was loaded with —
    e.g. ``("last", False)`` for an ``only_valid_locs=False`` load.
    """
    return _attr_row_selection(ds) == (itr, bool(vld_only))


def attr_values_1d(ds: "MinfluxDataset", attr: str) -> "np.ndarray | None":
    """1-D values of *attr* aligned to ``ds.attr`` rows.

    Returns ``ds.attr[attr]`` when it is already 1-D — except for derived
    attributes on non-last-valid materializations, which are read through the
    :func:`mfx_get` broadcast so they show the same (last-valid) values in
    every view. m2205-style ``iter_load="all"`` materializations keep
    per-iteration columns 2-D (``n_loc × n_itr``); for those (and for attrs
    absent from ``ds.attr``), fall back to the raw-store selection whose rows
    align with ``ds.attr`` (:func:`_attr_row_selection`). Returns ``None``
    when no aligned 1-D representation exists.
    """
    n_loc = int(getattr(ds.prop, "num_loc", 0))

    # xnm/ynm/znm: nm views of the metres loc_x/loc_y/loc_z store.
    if attr in _COORD_NM_BASE:
        v = attr_values_1d(ds, _COORD_NM_BASE[attr])
        if v is None:
            return None
        v = np.asarray(v, dtype=float) * 1.0e9
        if attr == "znm":
            v = v * float(getattr(ds.cali, "RIMF", 1.0) or 1.0)
        return v

    # cfr/efc measured at a non-last iteration: the meaningful per-loc value is
    # the value at that effective iteration, not the zero/NaN last-iteration one.
    if attr in _EFFECTIVE_ITER_META:
        eff_vals = effective_attr_values_1d(ds, attr)
        if eff_vals is not None:
            return eff_vals

    # Derived attrs: keep the Stage B contract (same last-valid values in
    # every view) on all-iteration / invalid-inclusive materializations.
    if attr in _DERIVED_BROADCAST_ATTRS and not attr_matches_last_valid(ds):
        rows = _attr_row_selection(ds)
        if rows is None:
            return None
        vals = mfx_get(ds, attr, itr=rows[0], vld_only=rows[1])
        if vals is not None:
            arr = np.asarray(vals)
            if arr.ndim == 1 and arr.shape[0] == n_loc:
                return arr
        return None

    val = ds.attr.get(attr)
    if val is not None:
        arr = np.asarray(val)
        if arr.ndim == 1:
            return arr
    rows = _attr_row_selection(ds)
    if rows is not None:
        vals = mfx_get(ds, attr, itr=rows[0], vld_only=rows[1])
        if vals is not None and np.asarray(vals).ndim == 1:
            return np.asarray(vals)
    return None


def attr_matches_last_valid(ds: "MinfluxDataset") -> bool:
    """True when ``ds.attr`` is the last-iteration, valid-only materialization.

    This holds for the common ``iter_load="last"`` case, where the default plot
    path (``ds.attr`` + ``ds.filter_mask``) equals the ``last`` iteration
    selection. It is False when the dataset was loaded with all iterations
    (``iter_load="all"``) or invalid localizations, so a ``last`` view must be
    taken from the raw store instead of ``ds.attr``.
    """
    raw = getattr(ds, "mfx_raw", None)
    if raw is None or len(raw) == 0:
        return True
    mask = mfx_row_mask(raw, itr="last", vld_only=True)
    if mask is None:
        return True
    return int(mask.sum()) == int(getattr(ds.prop, "num_loc", 0))


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

def _dtype_base_and_shape(dt: np.dtype) -> tuple[np.dtype, tuple]:
    return getattr(dt, "base", dt), tuple(getattr(dt, "shape", ()) or ())


def _add_mfx_attr(attrs: AttrStore, field_name: str, val: np.ndarray, num_raw: int) -> None:
    if field_name in SPATIAL_COORD_FIELDS:
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


def _is_legacy_nested_itr_mfx(mfx: np.ndarray) -> bool:
    fields = getattr(getattr(mfx, "dtype", None), "fields", None) or {}
    if "itr" not in fields:
        return False
    itr_dt = fields["itr"][0]
    base, subshape = _dtype_base_and_shape(itr_dt)
    return bool(subshape and getattr(base, "names", None))


def _iteration_count(values: np.ndarray, fallback: int) -> int:
    try:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            return max(int(np.unique(finite).size), 1)
    except Exception:
        pass
    return max(int(fallback), 1)


def _normalize_mfx_attrs(
    mfx: np.ndarray,
    *,
    load_all_itr: bool,
) -> tuple[AttrStore, int, int, str, str]:
    """Normalize modern flat or legacy nested-iteration mfx tables."""
    attrs = AttrStore()
    field_names = list(getattr(mfx.dtype, "names", []) or [])
    num_measurements = int(mfx.shape[0])

    if _is_legacy_nested_itr_mfx(mfx):
        itr_dt = mfx.dtype.fields["itr"][0]
        itr_base, itr_shape = _dtype_base_and_shape(itr_dt)
        num_itr_slots = int(itr_shape[0]) if itr_shape else 1
        nested = mfx["itr"]
        nested_names = list(getattr(itr_base, "names", []) or [])
        try:
            itr_values = np.asarray(nested["itr"])
        except Exception:
            itr_values = np.tile(np.arange(num_itr_slots), (num_measurements, 1))
        num_itr = _iteration_count(itr_values, num_itr_slots)

        if load_all_itr:
            select_mask = np.ones((num_measurements, num_itr_slots), dtype=bool)
        else:
            try:
                valid_itr_values = np.asarray(itr_values, dtype=float)
                last_itr = np.nanmax(valid_itr_values)
                select_mask = valid_itr_values == last_itr
                if not np.any(select_mask):
                    select_mask = np.zeros((num_measurements, num_itr_slots), dtype=bool)
                    select_mask[:, -1] = True
            except Exception:
                select_mask = np.zeros((num_measurements, num_itr_slots), dtype=bool)
                select_mask[:, -1] = True
        flat_select = select_mask.ravel()

        for nested_name in nested_names:
            val = np.asarray(nested[nested_name])
            if val.shape[:2] != select_mask.shape:
                continue
            selected = val[select_mask]
            _add_mfx_attr(attrs, nested_name, selected, int(selected.shape[0]))

        for field_name in field_names:
            if field_name == "itr" or field_name in nested_names:
                continue
            val = np.asarray(mfx[field_name])
            if val.ndim >= 1 and val.shape[0] == num_measurements:
                repeated = np.repeat(val, num_itr_slots, axis=0)
                selected = repeated[flat_select]
                _add_mfx_attr(attrs, field_name, selected, int(selected.shape[0]))

        num_raw = int(np.count_nonzero(select_mask))
        detail = "legacy nested iteration table: per-iteration values stored in mfx.itr"
        return attrs, num_raw, num_itr, "legacy", detail

    # The mfx array is in m2410-style flat format (already flattened per-loc).
    for field_name in field_names:
        val = mfx[field_name]
        _add_mfx_attr(attrs, field_name, val, num_measurements)
    return (
        attrs,
        num_measurements,
        1,
        "m2410",
        "m2410: all iteration value in N by 1 array",
    )

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
    only_valid   = data_prefs.get("only_valid_locs", True)

    file_info = FileInfo(
        name=name,
        folder=folder,
        datetime=datetime_str or datetime.now().strftime("%Y-%b-%d, %H:%M:%S"),
        raw_data=None,
        recent_path=recent_path,
    )

    # Build mfx_raw: all iterations, no vld filtering
    attrs_all, num_raw_all, _, _, _ = _normalize_mfx_attrs(mfx, load_all_itr=True)
    mfx_raw_store = AttrStore()
    for _k, _v in attrs_all.items():
        mfx_raw_store[_k] = np.asarray(_v)
    if "vld" not in mfx_raw_store:
        mfx_raw_store["vld"] = np.ones(num_raw_all, dtype=bool)
    if "itr" not in mfx_raw_store:
        mfx_raw_store["itr"] = np.zeros(num_raw_all, dtype=np.int16)

    attrs, num_raw, detected_num_itr, source_version, source_version_detail = _normalize_mfx_attrs(
        mfx,
        load_all_itr=load_all_itr,
    )

    valid_mask = np.ones(num_raw, dtype=bool)
    if only_valid and "vld" in attrs:
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
    num_itr = max(int(detected_num_itr), 1)
    if itr_arr is not None:
        try:
            itr_raw = np.asarray(itr_arr).ravel()
            valid_itr = itr_raw[valid_mask]
            if valid_itr.size:
                last_itr = int(np.nanmax(valid_itr))
                if source_version != "legacy":
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
            "source_version": source_version,
            "source_version_detail": source_version_detail,
            "raw_num_loc": int(mfx.shape[0]),
            "valid_num_loc": _attrs_valid_count(attrs, prop.num_loc),
            "raw_num_itr": int(num_itr),
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
        # A flat localization table (xnm/ynm fields) from Save Processed Data
        # loads generically; a real mfx structured array goes to the mfx parser.
        flat = _flat_table_from_dict(
            {nm: np.asarray(arr[nm]) for nm in arr.dtype.names},
            path.name, str(path.parent), prefs,
        )
        if flat is not None:
            return flat
        ds = load_from_mfx_array(
            mfx=arr,
            name=path.name,
            folder=str(path.parent),
            datetime_str=file_info.datetime,
            prefs=prefs,
        )
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
        "raw_num_loc": n,
        "raw_num_itr": 1,
        "iteration_load_mode": "last",
    })
    ds.components.mfx_raw = mfx_raw_store
    apply_metadata_sidecar(ds, path)
    return ds


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
    return ds


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

    prefs = prefs or {}
    data_prefs   = prefs.get("data", {}) or {}
    only_valid   = data_prefs.get("only_valid_locs", True)
    load_all_itr = data_prefs.get("iter_load", "last") == "all"

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    records = _extract_json_records(payload)
    if records is None or not _looks_like_minflux_json_records(records):
        raise ValueError(
            f"'{path.name}' is not recognised as MINFLUX localization JSON."
        )

    # Build mfx_raw from ALL records (no vld/itr filtering)
    mfx_raw_store = _build_mfx_raw_from_json_records(records)

    # Compute raw iteration count before filtering
    _itr_all_vals = _json_values(records, "itr")
    if _itr_all_vals is not None:
        _itr_all_arr = np.asarray([0 if v is None else int(v) for v in _itr_all_vals], dtype=np.int32)
        _raw_num_itr = int(_itr_all_arr.max()) + 1 if _itr_all_arr.size else 1
    else:
        _raw_num_itr = 1

    loc_m = _json_coordinate_array_m(records)
    valid = np.isfinite(loc_m).all(axis=1)

    # vld filter — mirrors _parse_m2410 behaviour
    if only_valid:
        vld = _json_values(records, "vld")
        if vld is not None:
            valid &= np.asarray([bool(v) for v in vld], dtype=bool)

    # Iteration filter — keep only the last (highest) iteration, same as _parse_m2410.
    # The JSON file is flat m2410-style: one record per localization event, so records
    # with itr < max_itr are intermediate measurements that the other loaders discard.
    if not load_all_itr:
        itr_values = _json_values(records, "itr")
        if itr_values is not None:
            itr_arr = np.asarray(
                [0 if v is None else int(v) for v in itr_values], dtype=np.int32
            )
            if valid.any():
                max_itr = int(itr_arr[valid].max())
                valid &= (itr_arr == max_itr)

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

    ds = MinfluxDataset(
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
    ds.metadata.update({
        "source_version": "json",
        "raw_num_loc": len(records),
        "valid_num_loc": _attrs_valid_count(attrs, prop.num_loc),
        "raw_num_itr": _raw_num_itr,
        "iteration_load_mode": "all" if load_all_itr else "last",
        "includes_invalid": _attrs_include_invalid(attrs),
    })
    ds.components.mfx_raw = mfx_raw_store
    if not attr_matches_last_valid(ds):
        ds.components.derived_last = _derived_last_from_raw(mfx_raw_store, _raw_num_itr, prefs)
    apply_metadata_sidecar(ds, path)
    return ds


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
