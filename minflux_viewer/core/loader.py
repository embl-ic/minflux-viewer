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

from datetime import datetime
from pathlib import Path

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
    attrs = _add_derived_attributes(attrs, prop)
    prop.attr_names = attrs.keys()

    # 6. Calibration defaults (populated further by the analysis functions)
    cali = Calibration()

    # 7. Channel
    channel = _build_channel(attrs, prop)

    return MinfluxDataset(
        file=file_info,
        prop=prop,
        attr=attrs,
        cali=cali,
        channel=channel,
    )


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
    * ``loc`` (and sometimes ``lnc``) are xyz arrays — shape (N, 3) or (N, 2).
    * Some exports include ``ext`` as an xyz array; others as a scalar.
      We detect structurally from the shape, not by field name.
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

        # Classify by trailing-dimension size, not by name
        #   (N, 1)     → scalar per loc
        #   (N, 3)     → xyz
        #   (N, 2)     → xy only (rare)
        n_cols = arr2.shape[1] if arr2.ndim == 2 else 0

        sub = arr2[mask]

        if n_cols == 3:
            attrs[f"{key}_x"] = sub[:, 0].ravel()
            attrs[f"{key}_y"] = sub[:, 1].ravel()
            attrs[f"{key}_z"] = sub[:, 2].ravel()
        elif n_cols == 2:
            attrs[f"{key}_x"] = sub[:, 0].ravel()
            attrs[f"{key}_y"] = sub[:, 1].ravel()
            attrs[f"{key}_z"] = np.zeros(num_loc)
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

        # ── xyz (3-D arrays) ─────────────────────────────────────────
        if arr.ndim == 3 and arr.shape[-1] in (2, 3):
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

        # ── xyz (2-D arrays, single iteration): (N, 3) ───────────────
        if arr.ndim == 2 and arr.shape[1] in (2, 3) and arr.shape[1] != num_itr:
            sub = arr[vld, :]
            n_cols = sub.shape[1]
            attrs[f"{key}_x"] = sub[:, 0].ravel()
            attrs[f"{key}_y"] = sub[:, 1].ravel()
            attrs[f"{key}_z"] = (
                sub[:, 2].ravel() if n_cols == 3 else np.zeros(num_loc_vld)
            )
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


def _add_derived_attributes(attrs: AttrStore, prop: DataProp) -> AttrStore:
    """
    Port of ``compute_extra_property.m`` — appends:

    ``itr`` (1-based), ``dt``, ``dst``, ``spd``, ``nLoc``, ``tim_trace``,
    ``ftr``, ``idx``.
    """
    n = prop.num_loc

    # Make iteration 1-based (MATLAB convention kept for user display)
    if "itr" in attrs:
        attrs["itr"] = (np.asarray(attrs["itr"]) + 1).astype(np.int16)

    tim = np.asarray(attrs.get("tim", np.zeros(n)), dtype=float).ravel()
    tid = np.asarray(attrs.get("tid", np.arange(n))).ravel()

    trace_change = np.diff(tid) != 0   # indices where trace ID changes

    # Time interval (dt)
    dt            = np.diff(tim)
    dt[trace_change] = 0.0
    attrs["dt"]   = np.concatenate([[0.0], dt])

    # Travel distance (dst) and speed (spd)
    lx = np.asarray(attrs.get("loc_x", np.zeros(n))).ravel()
    ly = np.asarray(attrs.get("loc_y", np.zeros(n))).ravel()
    lz = np.asarray(attrs.get("loc_z", np.zeros(n))).ravel()
    dloc = np.diff(np.column_stack([lx, ly, lz]), axis=0)
    dst  = np.linalg.norm(dloc, axis=1)
    dst[trace_change] = 0.0
    attrs["dst"]  = np.concatenate([[0.0], dst])

    with np.errstate(divide="ignore", invalid="ignore"):
        attrs["spd"] = np.where(attrs["dt"] > 0, attrs["dst"] / attrs["dt"], 0.0)

    # Localization count per trace (per-loc expanded)
    attrs["nLoc"] = np.repeat(prop.num_loc_per_trace, prop.num_loc_per_trace)

    # Relative time within each trace
    t_start      = tim[prop.trace_idx[:, 0]]
    t_min        = np.repeat(t_start, prop.num_loc_per_trace)
    attrs["tim_trace"] = tim - t_min

    # Filter mask (all pass by default)
    attrs["ftr"]  = np.ones(n, dtype=bool)

    # 1-based index
    attrs["idx"]  = np.arange(1, n + 1, dtype=np.uint32)

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

    _XYZ = {"loc", "lnc", "ext"}

    for field_name in field_names:
        val = mfx[field_name]
        if field_name in _XYZ:
            # (N, 3) xyz arrays — split into _x/_y/_z
            arr = np.asarray(val)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            attrs[f"{field_name}_x"] = arr[:, 0].ravel()
            attrs[f"{field_name}_y"] = arr[:, 1].ravel() if arr.shape[1] > 1 else np.zeros(len(arr))
            attrs[f"{field_name}_z"] = arr[:, 2].ravel() if arr.shape[1] > 2 else np.zeros(len(arr))
        elif field_name == "dcr":
            arr = np.asarray(val)
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

    for key, val in list(attrs.items()):
        arr = np.asarray(val)
        if arr.ndim >= 1 and arr.shape[0] == num_raw:
            attrs[key] = arr[valid_mask]

    num_loc = int(valid_mask.sum())
    num_itr_arr = attrs.get("itr", None)
    num_itr = int(np.nanmax(num_itr_arr) + 1) if (num_itr_arr is not None and np.asarray(num_itr_arr).size) else 1

    prop = _compute_properties(attrs, num_loc, num_itr, prefs=prefs)
    attrs = _add_derived_attributes(attrs, prop)
    prop.attr_names = attrs.keys()

    cali = Calibration()
    channel = _build_channel(attrs, prop)

    return MinfluxDataset(
        file=file_info,
        prop=prop,
        attr=attrs,
        cali=cali,
        channel=channel,
    )


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
    attrs = _add_derived_attributes(attrs, prop)
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
    attrs = _add_derived_attributes(attrs, prop)
    prop.attr_names = attrs.keys()

    return MinfluxDataset(
        file=file_info,
        prop=prop,
        attr=attrs,
        cali=Calibration(),
        channel=Channel(),
    )


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
