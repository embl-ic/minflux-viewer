'Loader format readers'

from __future__ import annotations

import json
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
from minflux_viewer.core.loader_metadata_recipe import _restore_filter_mask, apply_metadata_sidecar
from minflux_viewer.core.loader_normalization import (
    SPATIAL_COORD_FIELDS,
    _add_derived_attributes,
    _build_channel,
    _build_mfx_raw_from_json_records,
    _build_mfx_raw_from_raw,
    _compute_properties,
    _effective_iteration_metadata,
    _is_legacy_nested_itr_mfx,
    _json_coordinate_array_m,
    _json_scalar_array,
    _json_values,
    _json_vector_array,
    _mat_struct_to_dict,
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

_FLAT_X = ("xnm", "x")


_FLAT_Y = ("ynm", "y")


_FLAT_Z = ("znm", "z")


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
        return flat

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
    prefs = prefs or {}
    data_prefs = prefs.get("data", {}) or {}
    load_all_itr = False          # always last-valid; browse via mfx_raw
    only_valid   = data_prefs.get("only_valid_locs", True)

    file_info = FileInfo(
        name=name,
        folder=folder,
        datetime=datetime_str or datetime.now().strftime("%Y-%b-%d, %H:%M:%S"),
        raw_data=None,
        recent_path=recent_path,
    )

    # Build mfx_raw: all iterations, no vld filtering
    def _report(fraction: float) -> None:
        if progress is not None:
            progress(float(fraction))

    _report(0.0)
    # The field copy dominates the build (~70 % of it on a flat table).
    (attrs_all, num_raw_all, all_num_itr,
     all_version, all_version_detail) = _normalize_mfx_attrs(
        mfx, load_all_itr=True, progress=lambda f: _report(0.7 * f))
    _report(0.7)
    mfx_raw_store = AttrStore()
    for _k, _v in attrs_all.items():
        mfx_raw_store[_k] = np.asarray(_v)
    if "vld" not in mfx_raw_store:
        mfx_raw_store["vld"] = np.ones(num_raw_all, dtype=bool)
    if "itr" not in mfx_raw_store:
        mfx_raw_store["itr"] = np.zeros(num_raw_all, dtype=np.int16)

    if _is_legacy_nested_itr_mfx(mfx):
        attrs, num_raw, detected_num_itr, source_version, source_version_detail = _normalize_mfx_attrs(
            mfx,
            load_all_itr=load_all_itr,
        )
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
    _report(0.8)

    prop = _compute_properties(attrs, num_loc, num_itr, prefs=prefs)
    attrs = _add_derived_attributes(attrs, prop, prefs=prefs)
    prop.attr_names = attrs.keys()
    _report(0.9)

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
    _report(1.0)
    return dataset


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
            return flat
        ds = load_from_mfx_array(
            mfx=arr,
            name=path.name,
            folder=str(path.parent),
            datetime_str=file_info.datetime,
            prefs=prefs,
        )
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

    ``.npz`` is offered by the save dialog and enabled by default in
    Preferences, but had no reader at all - every ``.npz`` save was a file the
    application could never reopen. It carries the same flat columns as the
    ``.zarr`` export, so it follows the same two shapes: canonical raw
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
        from array import array

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
    an ``mfx``/``records``/``data`` list. Coordinates are expected either in a
    vector field such as ``loc``/``lnc``/``ext`` using metres, or in explicit
    nanometre columns such as ``xnm``/``ynm``. The dataset is normalized into
    the existing internal ``loc_x``/``loc_y``/``loc_z`` metre convention so all
    current render, filter, plot, and analysis paths keep working.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # File > Save writes a flat JSON array.  Use the streaming canonical path
    # for large arrays so reopening a large MSR export does not first create a
    # Python list of millions of record dictionaries.
    if path.stat().st_size >= 128 * 1024 * 1024:
        return _load_json_streaming_canonical(
            path, prefs, apply_sidecar=apply_sidecar)

    prefs = prefs or {}
    data_prefs   = prefs.get("data", {}) or {}
    only_valid   = data_prefs.get("only_valid_locs", True)
    load_all_itr = False          # always last-valid; browse via mfx_raw

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

    saved_ftr = attrs.pop("ftr", None)   # before derived attrs reset it
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
    _restore_filter_mask(ds, saved_ftr)
    if apply_sidecar:
        apply_metadata_sidecar(ds, path)
    return ds


def _iter_json_array_records(path: Path):
    """Yield records from a top-level JSON array without loading the array."""
    from json import JSONDecodeError, JSONDecoder

    decoder = JSONDecoder()
    chunk_size = 1024 * 1024
    with path.open("r", encoding="utf-8-sig") as handle:
        buffer = ""
        position = 0
        eof = False

        def fill() -> bool:
            nonlocal buffer, eof
            if eof:
                return False
            chunk = handle.read(chunk_size)
            if chunk:
                buffer += chunk
                return True
            eof = True
            return False

        while position >= len(buffer) and fill():
            pass
        while position < len(buffer) and buffer[position].isspace():
            position += 1
        if position >= len(buffer) or buffer[position] != "[":
            raise ValueError("Large JSON MINFLUX exports must be a top-level record array.")
        position += 1

        while True:
            while True:
                while position >= len(buffer):
                    if not fill():
                        raise ValueError("Unexpected end of large JSON record array.")
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                    if position >= len(buffer) and not fill():
                        raise ValueError("Unexpected end of large JSON record array.")
                if position < len(buffer) and buffer[position] == ",":
                    position += 1
                    continue
                break
            if buffer[position] == "]":
                return

            while True:
                try:
                    record, end = decoder.raw_decode(buffer, position)
                    break
                except JSONDecodeError:
                    if not fill():
                        raise ValueError("Malformed large JSON record array.") from None
            if not isinstance(record, dict):
                raise ValueError("MINFLUX JSON records must be objects.")
            yield record
            position = end
            # Avoid retaining already-decoded record text indefinitely while a
            # large array is streamed.
            if position >= chunk_size * 2:
                buffer = buffer[position:]
                position = 0


def _load_json_streaming_canonical(path: Path, prefs: dict | None, *,
                                   apply_sidecar: bool = True) -> MinfluxDataset:
    """Load the large flat JSON representation emitted by the canonical saver."""
    from array import array

    iterator = _iter_json_array_records(path)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError(f"'{path.name}' is empty.") from None
    if not _looks_like_minflux_json_records([first]) or "loc" not in first:
        raise ValueError(
            f"'{path.name}' is not a supported large MINFLUX JSON export."
        )

    # The canonical writer emits numeric scalars and fixed-width loc/dcr
    # vectors.  Store them in compact C arrays while decoding, then recompose
    # the normal structured m2410 array once at the end.
    specs: dict[str, tuple[str, int | None]] = {}
    buffers: dict[str, array] = {}
    row_count = 0

    def add_spec(key: str, value: Any) -> None:
        if key in specs:
            return
        def component_name(col: int) -> str:
            if key == "dcr":
                return f"{key}_{col}"
            return f"{key}_{('x', 'y', 'z')[col]}"

        if key in SPATIAL_COORD_FIELDS:
            specs[key] = ("vector", 3)
            for suffix in ("x", "y", "z"):
                buffers[f"{key}_{suffix}"] = array("d")
        elif key == "dcr" or isinstance(value, (list, tuple)):
            width = len(value) if isinstance(value, (list, tuple)) else 2
            if width not in (2, 3):
                return
            specs[key] = ("vector", width)
            for col in range(width):
                buffers[component_name(col)] = array("d")
        else:
            specs[key] = ("scalar", None)
            buffers[key] = array("d")

    for key, value in first.items():
        add_spec(str(key), value)

    def append_record(record: dict) -> None:
        nonlocal row_count
        for key, value in record.items():
            add_spec(str(key), value)
        for key, (kind, width) in specs.items():
            value = record.get(key)
            if kind == "vector":
                values = value if isinstance(value, (list, tuple)) else ()
                for col in range(int(width or 0)):
                    item = values[col] if col < len(values) else np.nan
                    buffer_name = (
                        f"{key}_{col}" if key == "dcr"
                        else f"{key}_{('x', 'y', 'z')[col]}"
                    )
                    try:
                        buffers[buffer_name].append(float(item) if item is not None else np.nan)
                    except (TypeError, ValueError):
                        buffers[buffer_name].append(np.nan)
            else:
                try:
                    buffers[key].append(float(value) if value is not None else np.nan)
                except (TypeError, ValueError):
                    buffers[key].append(np.nan)
        row_count += 1

    append_record(first)
    for record in iterator:
        append_record(record)

    columns: dict[str, np.ndarray] = {}
    for key, buffer in buffers.items():
        values = np.frombuffer(buffer, dtype=np.float64)
        if key == "vld":
            columns[key] = values.astype(bool)
        elif key == "itr":
            columns[key] = np.nan_to_num(values, nan=0.0).astype(np.int32)
        elif key == "tid":
            columns[key] = np.nan_to_num(values, nan=0.0).astype(np.int64)
        else:
            columns[key] = values

    from minflux_viewer.core.save import columns_to_mfx_array

    mfx = columns_to_mfx_array(columns)
    ds = load_from_mfx_array(
        mfx=mfx,
        name=path.name,
        folder=str(path.parent),
        datetime_str=datetime.fromtimestamp(path.stat().st_mtime).strftime(
            "%Y-%b-%d, %H:%M:%S"
        ),
        prefs=prefs,
    )
    ds.metadata["source_version"] = "json"
    ds.metadata["source_format"] = "json"
    ds.metadata["raw_num_loc"] = row_count
    if apply_sidecar:
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
