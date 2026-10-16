'Loader normalization'

from __future__ import annotations

from typing import Any

import numpy as np

from minflux_viewer.core.dataset import (
    AttrStore,
    Channel,
    DataProp,
)

SPATIAL_COORD_FIELDS = {"loc", "lnc", "ext"}


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


def _parse_raw(
    raw: dict,
    load_all_itr: bool = False,
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
        return _parse_m2205(raw, load_all_itr=load_all_itr, only_valid=only_valid)

    if isinstance(itr_val, np.ndarray) and np.issubdtype(itr_val.dtype, np.integer):
        # m2410: flat 1-D integer array — one iteration index per localization event.
        # Legacy .mat (m2205 flat): itr is 2-D (nLoc × nItr).  Check dimensionality
        # after squeezing out any size-1 axes; (N,1) and (1,N) are still 1-D.
        if np.squeeze(itr_val).ndim <= 1:
            return _parse_m2410(raw, only_valid=only_valid, load_all_itr=load_all_itr)
        # 2-D integer itr with loc at top level → m2205 flat format
        if "loc" in raw:
            return _parse_m2205_unwrapped(raw, load_all_itr=load_all_itr,
                                          only_valid=only_valid)

    # m2205-unwrapped: loc already at top level, itr is numeric (float or other)
    if "loc" in raw:
        return _parse_m2205_unwrapped(
            raw, load_all_itr=load_all_itr, only_valid=only_valid,
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
    except (AttributeError, TypeError, ValueError):
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
    except (AttributeError, TypeError, ValueError, OverflowError):
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
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
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


def _dcr_is_two_channel(arr: np.ndarray, sample: int = 500, tol: float = 1e-3) -> bool:
    """
    Heuristic: return True when a (N, 2) dcr array holds m2410-style
    two-channel ratios ch1/(ch1+ch2) and ch2/(ch1+ch2).

    In that format the two columns are **complementary** and sum to 1.0 —
    but only *approximately*: the instrument quantises each channel ratio
    independently (≈12-bit, step ≈ 1/4096 ≈ 2.4e-4), so a genuine two-channel
    sum lands on ``(4096 ± 1)/4096`` — i.e. within ~2.5e-4 of 1.0, not exactly 1.
    Hence the tolerance is ``1e-3`` (not float-epsilon): tight enough to reject
    legacy m2205 2-iteration data, whose two columns are *independent*
    per-iteration ratios that do not sum to 1.

    All-zero (invalid-localization) placeholder rows are skipped so they don't
    fail the test.
    """
    col0 = np.asarray(arr[:, 0], dtype=float)
    col1 = np.asarray(arr[:, 1], dtype=float)
    s = col0 + col1
    good = np.isfinite(s) & ((col0 != 0.0) | (col1 != 0.0))
    idx = np.flatnonzero(good)[:sample]
    if idx.size == 0:
        return False
    return bool(np.all(np.abs(s[idx] - 1.0) < tol))


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


def _parse_m2205(
    raw: dict,
    load_all_itr: bool = False,
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
        merged, load_all_itr=load_all_itr, only_valid=only_valid,
    )


def _parse_m2205_unwrapped(
    raw: dict,
    load_all_itr: bool = False,
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
            # cfr/efc are measured at ONE loop iteration and are zero/NaN
            # elsewhere, so materialize the iteration carrying the signal.
            # Always on: metadata["cfr_iteration"]/["efc_iteration"] and
            # effective_attr_values_1d assume this per-loc value everywhere.
            if key in ("cfr", "efc"):
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


def _build_mfx_raw_m2410(raw: dict) -> AttrStore:
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


def _build_mfx_raw_m2205_unwrapped(merged: dict, n_loc: int, n_itr: int) -> AttrStore:
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


def _build_mfx_raw_from_raw(raw: dict) -> AttrStore:
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


def _build_mfx_raw_from_json_records(records: list) -> AttrStore:
    """Build mfx_raw from ALL JSON records (no vld/itr filtering)."""
    n = len(records)
    if n == 0:
        return AttrStore()
    mfx = AttrStore()

    try:
        loc_m = _json_coordinate_array_m(records)
    except (IndexError, KeyError, TypeError, ValueError):
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


def _compute_properties(
    attrs: AttrStore, num_loc: int, num_itr: int,
    prefs: dict | None = None,
) -> DataProp:
    """
    Compute trace structure, dimensionality, etc.

    If ``prefs["data"]["enforce_min_z_range"]`` is True and the finite Z range
    of ``attrs["loc_z"]`` is smaller than ``prefs["data"]["min_z_range_nm"]``
    (value in nm — raw data is stored in metres), the Z axis is forced to
    zero and the dataset is marked as 2-D. Non-finite Z values are ignored for
    dimensionality and a dataset with no finite non-zero Z is normalized to a
    zero-filled 2-D axis. This filters out tiny residual Z variations caused by
    drift correction in genuinely 2-D acquisitions.
    """
    prop = DataProp(num_loc=num_loc, num_itr=num_itr)

    # Dimensionality
    z = np.asarray(attrs.get("loc_z", np.zeros(num_loc)), dtype=float).ravel()
    finite_z = z[np.isfinite(z)]

    # Optional 2D/3D threshold (user preference)
    if prefs is not None and finite_z.size > 0:
        data_prefs = prefs.get("data", {}) or {}
        if data_prefs.get("enforce_min_z_range", False):
            min_z_nm   = float(data_prefs.get("min_z_range_nm", 5.0))
            # Raw z values are in metres; convert the threshold from nm.
            z_range_m  = float(np.ptp(finite_z))
            z_range_nm = z_range_m * 1e9
            if z_range_nm < min_z_nm:
                z = np.zeros_like(z)

    # NaNs/Infs are missing Z measurements, not evidence of a third axis.  If
    # no finite non-zero value remains, keep the canonical 2-D contract that
    # loc_z is present and zero-filled so renderers and analyses agree.
    has_nonzero_z = bool(np.any(np.isfinite(z) & (z != 0.0)))
    if not has_nonzero_z:
        z = np.zeros_like(z)
    attrs["loc_z"] = z

    prop.num_dim = 3 if has_nonzero_z else 2

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
        if arr.ndim > 1:
            # Keep every native DCR channel. The viewer's historical ``dcr``
            # alias remains the primary/first column, but the second channel is
            # an independent measured value and is required for a lossless Zarr
            # round trip (real files do not always sum exactly to one).
            attrs[field_name] = arr[:, 0].ravel()
            for column in range(arr.shape[1]):
                attrs[f"dcr_{column}"] = arr[:, column].ravel()
        else:
            attrs[field_name] = arr.ravel()
    else:
        arr = np.asarray(val)
        attrs[field_name] = arr.ravel() if arr.ndim == 1 else arr


_FIELD_COPY_BLOCK_ROWS = 4096


def _mfx_columns(mfx: np.ndarray, progress=None) -> dict[str, np.ndarray]:
    """Every field of a flat structured *mfx* as its own contiguous array.

    Copying field by field (``mfx[name]`` then ``ravel``) walks the whole
    record buffer once **per field**, because each field is strided by the full
    row size -- on a 14 M-row acquisition that was ~2 s per pass. Copying all
    fields of a cache-sized block of rows at a time reads the buffer once
    (~0.5 s on the same data) and returns identical values.

    ``progress(fraction)``, when given, is called every few hundred blocks.
    """
    fields = mfx.dtype.fields or {}
    n = int(mfx.shape[0])
    out = {
        name: np.empty((n,) + tuple(fields[name][0].shape), dtype=fields[name][0].base)
        for name in (mfx.dtype.names or ())
    }
    step = _FIELD_COPY_BLOCK_ROWS
    report_every = step * 256
    for start in range(0, n, step):
        block = mfx[start:start + step]
        for name, dst in out.items():
            dst[start:start + step] = block[name]
        if progress is not None and start % report_every == 0:
            progress(start / max(n, 1))
    return out


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
    except (TypeError, ValueError, OverflowError):
        pass
    return max(int(fallback), 1)


def _normalize_mfx_attrs(
    mfx: np.ndarray,
    *,
    load_all_itr: bool,
    progress=None,
) -> tuple[AttrStore, int, int, str, str]:
    """Normalize modern flat or legacy nested-iteration mfx tables.

    ``progress(fraction)`` is reported only for the flat layout, whose field
    copy is the bulk of the work.
    """
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
        except (KeyError, TypeError, ValueError):
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
            except (TypeError, ValueError):
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
        # This is the native m2205 layout used by Imspector: every outer
        # localization owns a fixed-width structured ``itr`` subarray.  The
        # MAT loader has always called the equivalent unwrapped layout m2205;
        # calling the direct MSR path "legacy" made the same acquisition change
        # version merely because it arrived through a different container.
        detail = "m2205: per-iteration values stored in the structured mfx.itr table"
        return attrs, num_raw, num_itr, "m2205", detail

    # The mfx array is in m2410-style flat format (already flattened per-loc).
    if mfx.ndim == 1 and field_names:
        columns = _mfx_columns(mfx, progress)
    else:
        columns = {name: mfx[name] for name in field_names}
    for field_name in field_names:
        _add_mfx_attr(attrs, field_name, columns.pop(field_name), num_measurements)
    return (
        attrs,
        num_measurements,
        1,
        "m2410",
        "m2410: all iteration value in N by 1 array",
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
