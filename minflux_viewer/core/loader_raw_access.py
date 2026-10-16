'Loader raw access'

from __future__ import annotations

import threading

import numpy as np

from minflux_viewer.core.dataset import (
    AttrStore,
    MinfluxDataset,
)
from minflux_viewer.core.loader_normalization import _add_derived_attributes, _compute_properties

_COORD_NM_BASE = {"xnm": "loc_x", "ynm": "loc_y", "znm": "loc_z"}


_EFFECTIVE_ITER_META = {"cfr": "cfr_iteration", "efc": "efc_iteration"}


def effective_iteration_for_attr(ds: MinfluxDataset, attr: str) -> int | None:
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


def effective_iterations_for_attr(ds: MinfluxDataset, attr: str) -> np.ndarray:
    """Boolean array (length ``n_itr``, 0-based) marking iterations that hold
    **real** (finite, non-zero) values for *attr*.

    Attributes recorded at every loop iteration (``dcr``, ``efo``, coordinates …)
    are effective everywhere → all ``True``; single-iteration measurements
    (``cfr``/``efc``) are effective only at the iteration they are measured at.
    Used to **bold the useful iterations** in the iteration selectors so the user
    can browse straight to them. Returns all-``True`` when it cannot be
    determined (no raw store, or a 2-D/misaligned column).
    """
    n_itr = max(1, int(ds.metadata.get("raw_num_itr", getattr(ds.prop, "num_itr", 1) or 1)))
    out = np.zeros(n_itr, dtype=bool)
    raw = getattr(ds, "mfx_raw", None)
    if raw is None or _mfx_raw_len(raw) == 0:
        out[:] = True
        return out
    itr_all = np.asarray(raw.get("itr", np.zeros(_mfx_raw_len(raw), int))).ravel()
    vals = mfx_get(ds, attr, itr="all", vld_only=False)
    if vals is None:
        out[:] = True
        return out
    vals = np.asarray(vals).ravel()
    if vals.shape[0] != itr_all.shape[0]:
        out[:] = True                             # per-iteration 2-D / can't align
        return out
    with np.errstate(invalid="ignore"):
        good = np.isfinite(vals) & (vals != 0.0)
    idx = itr_all[good]
    idx = idx[(idx >= 0) & (idx < n_itr)]
    if idx.size:
        out[np.unique(idx.astype(int))] = True
    return out


_DERIVED_BROADCAST_ATTRS = frozenset(
    {"den", "dst", "spd", "dt", "tim_trace", "siz", "dur", "len"}
)


def _mfx_raw_len(raw: AttrStore) -> int:
    """Number of rows in the flattened raw store (0 if empty)."""
    arr = raw.get("itr")
    if arr is None:
        for key in raw.keys():
            arr = raw.get(key)
            break
    return int(np.asarray(arr).shape[0]) if arr is not None else 0


_LOC_ID_PUBLISH_LOCK = threading.Lock()


def _raw_loc_id(raw: AttrStore) -> np.ndarray | None:
    """Per-row localization-group id over the flattened raw store (cached).

    A group is the set of raw rows belonging to one localization: its
    (possibly partial) iteration ladder up to the final row. A new group
    starts whenever the trace id changes or the iteration index fails to
    increase within a trace. This single rule covers m2205-style fixed
    ``n_loc × n_itr`` grids, m2410 flat event streams (variable iterations
    per localization, traces interleaved in time, repeated final-iteration
    rows in tracking sequences), and single-iteration stores.

    ⚠ Computed once and **cached into the store** as ``loc_id`` — so this
    reader writes to ``mfx_raw``. It is the one such write, it is
    idempotent, and it is thread-safe (see the publish below), which is
    what lets a background worker call it while the GUI thread reads.
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
    # Published under a lock because a background worker may compute this at the
    # same time as the GUI thread (see core/post_load.py). The value is a pure
    # function of the store, so a concurrent computation is duplicated work, not
    # a disagreement; the lock is held only for the publish so neither thread
    # ever waits out the other's argsort, and every reader ends up with the same
    # array. Do not widen it to cover the computation above.
    with _LOC_ID_PUBLISH_LOCK:
        cached = raw.get("loc_id")
        if cached is not None:
            return np.asarray(cached).ravel()
        raw["loc_id"] = loc_id
    return loc_id


def _derived_last_from_raw(
    raw: AttrStore, num_itr: int, prefs: dict | None,
) -> AttrStore:
    """Compute broadcastable derived attributes at the last-valid selection.

    Re-materializes ``loc``/``tid``/``tim`` from the raw store at
    ``itr="last", vld_only=True`` and runs the same trace-structure +
    derived-attribute pipeline as the last-valid materialization, so the values
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
    raw: AttrStore,
    *,
    itr: str | int = "last",
    vld_only: bool = True,
) -> np.ndarray | None:
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
    ds: MinfluxDataset,
    attr: str,
    raw: AttrStore,
    mask: np.ndarray,
) -> np.ndarray | None:
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
    raw: AttrStore,
    attr: str,
    source_mask: np.ndarray,
    target_mask: np.ndarray | None,
) -> np.ndarray | None:
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


def effective_attr_values_1d(ds: MinfluxDataset, attr: str) -> np.ndarray | None:
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


def iteration_attr_values_1d(
    ds: MinfluxDataset,
    attr: str,
    iter_index: int,
    *,
    itr: str | int = "last",
    vld_only: bool = True,
) -> np.ndarray | None:
    """Per-localization value of *attr* taken from a **specific iteration**.

    Gathers each localization's value of *attr* from its ``itr == iter_index``
    raw row and lays it onto the ``itr``/``vld_only`` materialized selection
    (default the last-valid rows, i.e. the order of
    ``mfx_get(ds, …, itr="last", vld_only=True)`` / ``ds.attr`` when materialized
    last-valid). Used e.g. to read the photon count (``eco``) at the final
    *lateral* iteration for the lateral CRLB bound, since the materialized value
    is the final *axial* iteration. ``None`` when the raw store is empty or the
    attribute/iteration is unavailable. NaN for localizations lacking that
    iteration.
    """
    raw = getattr(ds, "mfx_raw", None)
    if raw is None or len(raw) == 0:
        return None
    target_mask = mfx_row_mask(raw, itr=itr, vld_only=vld_only)
    source_mask = mfx_row_mask(raw, itr=int(iter_index), vld_only=False)
    if target_mask is None or source_mask is None:
        return None
    return _gather_iteration_values(raw, attr, source_mask, target_mask)


def last_valid_attr_values(ds: MinfluxDataset, attr: str) -> np.ndarray | None:
    """Per-localization *attr* on the last-valid rows, cfr/efc at their measured
    iteration.

    The rows are those of ``mfx_get(ds, …, itr="last", vld_only=True)`` -- what
    an export or a derived dataset built from that selection must align to.
    This is the counterpart of :func:`attr_values_1d`, which aligns to
    ``ds.attr`` instead; the two coincide whenever
    :func:`attr_matches_last_valid` holds.

    ``cfr``/``efc`` are measured at ONE loop iteration, and at the last one they
    read 0 / NaN (a 3-D m2410 sequence measures them at iteration 7 of 10), so
    reading them with ``itr="last"`` exports a column of zeros. Here they are
    gathered from each localization's valid row at the effective iteration
    (:func:`effective_iteration_for_attr`) -- the same values the Filter dialog
    and the Histogram show. Every other attribute is ``mfx_get(itr="last")``.
    Returns ``None`` when *attr* is absent, or when cfr/efc have an effective
    iteration that cannot be aligned to these rows -- never the last-iteration
    placeholder in its place.
    """
    eff = effective_iteration_for_attr(ds, attr)
    raw = getattr(ds, "mfx_raw", None)
    if eff is None or raw is None or len(raw) == 0:
        return mfx_get(ds, attr, itr="last", vld_only=True)
    target_mask = mfx_row_mask(raw, itr="last", vld_only=True)
    source_mask = mfx_row_mask(raw, itr=eff, vld_only=True)
    return _gather_iteration_values(raw, attr, source_mask, target_mask)


VALUE_POOL_SELECTORS = ("sum", "average")


def is_value_pool_selector(itr: str | int | None) -> bool:
    """True for the ``"sum"`` / ``"average"`` per-localization pooling selectors."""
    return isinstance(itr, str) and itr.strip().lower() in VALUE_POOL_SELECTORS


def _pooled_loc_values(
    ds: MinfluxDataset,
    attr: str,
    *,
    how: str,
    vld_only: bool,
) -> np.ndarray | None:
    """Per-localization pooling of *attr* over **all** of its iterations::

        sum      ->  Σ a_i
        average  ->  mean(a_i)

    The result is aligned to the ``last`` selection rows — one entry per
    localization — so it drops into any per-localization consumer unchanged.
    Non-finite values are skipped; a localization with no usable iteration gets
    ``NaN`` rather than a misleading ``0``.
    """
    raw = getattr(ds, "mfx_raw", None)
    if raw is None or _mfx_raw_len(raw) == 0:
        return None
    all_mask = mfx_row_mask(raw, itr="all", vld_only=vld_only)
    target_mask = mfx_row_mask(raw, itr="last", vld_only=vld_only)
    loc_id = _raw_loc_id(raw)
    if all_mask is None or target_mask is None or loc_id is None:
        return None

    vals = _mfx_get_rows(ds, attr, itr="all", vld_only=vld_only)
    if vals is None:
        return None
    vals = np.asarray(vals, dtype=float).ravel()
    if vals.shape[0] != int(all_mask.sum()):
        return None

    ids = loc_id[all_mask]
    good = np.isfinite(vals)
    n_groups = int(loc_id.max()) + 1
    num = np.bincount(ids[good], weights=vals[good], minlength=n_groups)
    cnt = np.bincount(ids[good], minlength=n_groups)

    per_loc = np.full(n_groups, np.nan)
    if str(how).strip().lower() == "sum":
        np.copyto(per_loc, num, where=cnt > 0)
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = num / cnt
        np.copyto(per_loc, avg, where=cnt > 0)
    return per_loc[loc_id[target_mask]]


def mfx_get(
    ds: MinfluxDataset,
    attr: str,
    *,
    itr: str | int = "last",
    vld_only: bool = True,
) -> np.ndarray | None:
    """
    Retrieve an attribute from a dataset's raw mfx store.

    Parameters
    ----------
    ds :
        Dataset to query.
    attr :
        Attribute name, e.g. ``"efo"``, ``"cfr"``, ``"loc_x"``, ``"idx"``.
    itr :
        ``"last"``    — final iteration only (default).
        ``"all"``     — all iterations pooled, one row each.
        ``int``       — specific 0-based iteration index.
        ``"sum"`` / ``"average"`` — collapse each localization's iterations to
        one value (see :func:`_pooled_loc_values`); the result is aligned to the
        ``last`` selection rows, i.e. one value per localization.
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
    if is_value_pool_selector(itr):
        return _pooled_loc_values(
            ds, attr, how=str(itr).strip().lower(), vld_only=vld_only,
        )
    return _mfx_get_rows(ds, attr, itr=itr, vld_only=vld_only)


def _mfx_get_rows(
    ds: MinfluxDataset,
    attr: str,
    *,
    itr: str | int = "last",
    vld_only: bool = True,
) -> np.ndarray | None:
    """Row-wise retrieval for a non-pooled selection.

    The body of :func:`mfx_get`; kept separate so the pooling layer can reuse it
    without recursing through its own dispatch.
    """
    # xnm/ynm/znm are the canonical nm coordinate views of the metres store.
    if attr in _COORD_NM_BASE:
        base = _COORD_NM_BASE[attr]
        v = _mfx_get_rows(ds, base, itr=itr, vld_only=vld_only)
        if v is None:
            return None
        v = np.asarray(v, dtype=float) * 1.0e9
        if attr == "znm":
            v = v * float(getattr(ds.cali, "z_scaling_factor", 1.0) or 1.0)
        return v

    raw = getattr(ds, "mfx_raw", None)
    if raw is None or len(raw) == 0:
        if attr == "idx":
            return np.arange(1, int(getattr(ds.prop, "num_loc", 0)) + 1, dtype=np.uint32)
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


def resolve_spec_iteration(ds: MinfluxDataset, spec: dict) -> str | int:
    """Resolve a filter spec's iteration selector to a concrete evaluation token.

    A filter spec may carry an optional ``"itr"`` key recording *which*
    iteration its bound was authored against (so the same logical filter is
    reproducible across iteration browsing and file round-trips):

    ``"last"``       — the localization's final valid iteration (broadcast).
    ``"effective"``  — the attribute's measured iteration (cfr/efc); falls back
                       to ``"last"`` when the attribute has none.
    ``"sum"`` /
    ``"average"``    — the localization's value pooled over all iterations
                       (broadcast); see :func:`_pooled_loc_values`.
    ``"all"``        — evaluate the bound at each browse row's own value
                       (per-iteration; a localization's rows may split).
    ``int``          — a specific 0-based iteration index (broadcast).

    A missing/``None`` ``"itr"`` is the **legacy default for filter files saved
    before the selector existed**: ``"effective"`` for cfr/efc (their measured
    iteration), otherwise ``"last"``. Both read the attribute's raw value at that
    iteration. Returns ``"browse"``, ``"last"``, ``"all"``, ``"effective"``,
    ``"sum"``, ``"average"`` or an ``int``.
    """
    attr = spec.get("attribute", "")
    raw_sel = spec.get("itr", None)
    if raw_sel is None:
        return "effective" if effective_iteration_for_attr(ds, attr) is not None else "last"
    if isinstance(raw_sel, (int, np.integer)) and not isinstance(raw_sel, bool):
        return int(raw_sel)
    text = str(raw_sel).strip().lower()
    if text in ("browse", "last", "all", "effective") or text in VALUE_POOL_SELECTORS:
        return text
    if text.isdigit():
        return int(text)
    return "last"


def _broadcast_iteration_to_rows(
    ds: MinfluxDataset,
    attr: str,
    *,
    src_itr: str | int,
    src_vld: bool,
    browse_mask: np.ndarray,
    raw: AttrStore,
) -> np.ndarray | None:
    """Per-localization value of *attr* at ``(src_itr, src_vld)`` duplicated onto
    the ``browse_mask`` rows by localization id.

    Sourcing the per-localization values through :func:`mfx_get` makes this work
    uniformly for raw, coordinate (``xnm``…), derived (``den``…), ``idx`` and
    value-pooled (``"sum"``/``"average"``) attributes — the pooled selectors
    already return one value per localization, aligned to the ``last`` rows,
    which is why the source *mask* uses ``"last"`` for them. NaN for
    localizations absent from the source selection. Returns ``None`` when the
    value cannot be aligned (so the caller marks the spec unevaluable rather
    than mis-filtering).
    """
    mask_itr = "last" if is_value_pool_selector(src_itr) else src_itr
    src_mask = mfx_row_mask(raw, itr=mask_itr, vld_only=src_vld)
    if src_mask is None:
        return None
    src_vals = mfx_get(ds, attr, itr=src_itr, vld_only=src_vld)
    if src_vals is None:
        return None
    src_vals = np.asarray(src_vals).ravel()
    if src_vals.shape[0] != int(src_mask.sum()):
        return None
    loc_id = _raw_loc_id(raw)
    if loc_id is None:
        return None
    per_loc = np.full(int(loc_id.max()) + 1, np.nan)
    per_loc[loc_id[src_mask]] = src_vals.astype(float, copy=False)
    return per_loc[loc_id[browse_mask]]


def _spec_filter_values(
    ds: MinfluxDataset,
    spec: dict,
    *,
    itr: str | int,
    vld_only: bool,
    raw: AttrStore | None,
    browse_mask: np.ndarray | None,
) -> np.ndarray | None:
    """Values to test a single filter spec's bound against, aligned to the
    ``(itr, vld_only)`` browse selection and honouring the spec's own ``"itr"``.

    See :func:`resolve_spec_iteration` for the iteration semantics.
    """
    attr = spec.get("attribute", "")
    sel = resolve_spec_iteration(ds, spec)

    # Per-row: evaluate at each browse row's own value (localization rows may
    # individually pass/fail).
    if sel in ("browse", "all") or browse_mask is None or raw is None or len(raw) == 0:
        return mfx_get(ds, attr, itr=itr, vld_only=vld_only)

    # Per-localization broadcast at the spec's own iteration / pooling.
    if sel == "effective":
        eff = effective_iteration_for_attr(ds, attr)
        src_itr, src_vld = (eff, True) if eff is not None else ("last", True)
    elif sel == "last" or is_value_pool_selector(sel):
        src_itr, src_vld = sel, True
    else:  # concrete 0-based int iteration
        src_itr, src_vld = int(sel), False
    return _broadcast_iteration_to_rows(
        ds, attr, src_itr=src_itr, src_vld=src_vld, browse_mask=browse_mask,
        raw=raw,
    )


def mfx_filter_mask(
    ds: MinfluxDataset,
    *,
    itr: str | int = "last",
    vld_only: bool = True,
) -> tuple[np.ndarray, list[str]] | None:
    """Re-evaluate the dataset's persisted filter specs over a raw-store selection.

    The dataset's active filters are stored as re-evaluable specs in
    ``ds.state["filter_specs"]`` (attribute, mode, bounds, and an optional
    per-spec ``"itr"`` iteration selector). This re-applies them against
    :func:`mfx_get` rows for the given ``itr``/``vld_only`` browse selection, so
    the same logical filter extends to any iteration and to invalid
    localizations. Each spec is evaluated at **its own** iteration
    (:func:`resolve_spec_iteration`), independent of the browse selection —
    e.g. an ``itr="last"`` spec keeps/drops a localization's iteration rows
    together by its last-valid value, matching what render/scatter show.

    Returns
    -------
    (mask, unevaluable) or None
        ``mask`` is a boolean array aligned with ``mfx_get(ds, <attr>, itr=itr,
        vld_only=vld_only)``. ``unevaluable`` lists attribute names of specs that
        could not be evaluated on the raw store (e.g. derived attributes such as
        ``den`` that have no per-iteration values). Returns ``None`` when the
        dataset has no raw store.
    """
    from minflux_viewer.utils.filters import raw_spec_mask

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
        vals = _spec_filter_values(
            ds, spec, itr=itr, vld_only=vld_only, raw=raw, browse_mask=browse_mask,
        )
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


def evaluate_saved_filters(ds: MinfluxDataset) -> np.ndarray | None:
    """Return the persisted-filter mask without mutating *ds*.

    When a raw store aligned with ``ds.attr`` is available, this delegates to
    :func:`mfx_filter_mask` over the materialized selection so each spec is
    evaluated at **its own** iteration (:func:`resolve_spec_iteration`) — the
    same result the plot windows show for the materialized view. Otherwise
    (no raw store, e.g. a re-imported flat table) it resolves each attribute
    through :func:`attr_values_for_selection`, so coordinate
    (``xnm``/``ynm``/``znm``) and derived (``den`` …) filters map to per-loc
    values — mirroring the live FilterDialog. A spec whose attribute is not yet materialized (e.g. ``den``
    before post-load density) is skipped; callers re-run this once those
    attributes exist.
    """
    from minflux_viewer.utils.filters import raw_spec_mask

    specs = ds.state.get("filter_specs") or []
    if not specs:
        return None
    n = int(ds.prop.num_loc)

    # Preferred path: re-evaluate over the raw store's materialized selection so
    # per-spec iteration selectors are honoured. Fall through to the materialized
    # attr_values_1d path when no aligned raw view exists or nothing was
    # evaluable (so a deferred den filter re-runs once density is computed).
    raw = getattr(ds, "mfx_raw", None)
    rows = _attr_row_selection(ds) if raw is not None and len(raw) else None
    if rows is not None:
        res = mfx_filter_mask(ds, itr=rows[0], vld_only=rows[1])
        if res is not None:
            rmask, uneval = res
            if rmask.shape[0] == n and len(uneval) < len(specs):
                return np.asarray(rmask, dtype=bool)

    tid_v = attr_values_1d(ds, "tid")
    tid = np.arange(n) if tid_v is None else np.asarray(tid_v).ravel()
    mask = np.ones(n, dtype=bool)
    applied = False
    for spec in specs:
        vals = attr_values_for_selection(
            ds,
            str(spec.get("attribute", "")),
            itr=resolve_spec_iteration(ds, spec),
        )
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
    return mask if applied else None


def apply_saved_filters(ds: MinfluxDataset) -> bool:
    """Apply ``ds.state['filter_specs']`` and report whether any were usable."""
    mask = evaluate_saved_filters(ds)
    if mask is None:
        return False
    ds.filter_mask = mask
    return True


def _attrs_include_invalid(attrs: AttrStore) -> bool:
    """True when the materialized store retains rows with ``vld == False``."""
    vld = attrs.get("vld")
    if vld is None:
        return False
    try:
        return bool(np.any(~np.asarray(vld).astype(bool).ravel()))
    except (TypeError, ValueError):
        return False


def _attrs_valid_count(attrs: AttrStore, fallback: int) -> int:
    """Number of vld rows in the materialized store (= loaded rows when the
    load was valid-only)."""
    vld = attrs.get("vld")
    if vld is None:
        return int(fallback)
    try:
        return int(np.asarray(vld).astype(bool).ravel().sum())
    except (TypeError, ValueError):
        return int(fallback)


def attr_materialized_selection(ds: MinfluxDataset) -> tuple[str, bool]:
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
    ds: MinfluxDataset, itr: str | int, vld_only: bool,
) -> bool:
    """True when the raw-store selection's row count equals ``ds.attr`` rows."""
    raw = getattr(ds, "mfx_raw", None)
    if raw is None or len(raw) == 0:
        return True
    mask = mfx_row_mask(raw, itr=itr, vld_only=vld_only)
    if mask is None:
        return True
    return int(mask.sum()) == int(getattr(ds.prop, "num_loc", 0))


def _attr_row_selection(ds: MinfluxDataset) -> tuple[str, bool] | None:
    """The raw-store selection whose rows align 1:1 with ``ds.attr`` rows.

    Usually the materialized selection from the load metadata; m2205-style
    m2205-style all-iteration stores keep one row per localization (2-D
    per-iteration columns), so their rows align with the ``last`` selection
    instead. ``None`` when no candidate matches (no aligned raw view exists).
    """
    sel_itr, sel_vld = attr_materialized_selection(ds)
    for cand_itr in dict.fromkeys((sel_itr, "last")):
        if _selection_rows_match_attr(ds, cand_itr, sel_vld):
            return cand_itr, sel_vld
    return None


def attr_matches_selection(
    ds: MinfluxDataset,
    *,
    itr: str | int = "last",
    vld_only: bool = True,
) -> bool:
    """True when ``ds.attr`` rows ARE the given raw-store selection.

    Generalizes :func:`attr_matches_last_valid` to any (iteration, validity)
    selection, so the windows can use the materialized fast path (filter
    edit, ROI selection) for whatever selection the dataset was loaded with —
    e.g. ``("last", False)`` for an ``only_valid_locs=False`` load.
    """
    return _attr_row_selection(ds) == (itr, bool(vld_only))


def attr_values_1d(ds: MinfluxDataset, attr: str) -> np.ndarray | None:
    """1-D values of *attr* aligned to ``ds.attr`` rows.

    Returns ``ds.attr[attr]`` when it is already 1-D — except for derived
    attributes on non-last-valid materializations, which are read through the
    :func:`mfx_get` broadcast so they show the same (last-valid) values in
    every view. m2205-style all-iteration materializations keep
    per-iteration columns 2-D (``n_loc × n_itr``); for those (and for attrs
    absent from ``ds.attr``), fall back to the raw-store selection whose rows
    align with ``ds.attr`` (:func:`_attr_row_selection`). Returns ``None``
    when no aligned 1-D representation exists.
    """
    n_loc = int(getattr(ds.prop, "num_loc", 0))

    # ``idx`` is application-owned and always available, including for generic
    # imported datasets that do not persist a physical index column.
    if attr == "idx":
        return np.arange(1, n_loc + 1, dtype=np.uint32)

    # xnm/ynm/znm: nm views of the metres loc_x/loc_y/loc_z store.
    if attr in _COORD_NM_BASE:
        v = attr_values_1d(ds, _COORD_NM_BASE[attr])
        if v is None:
            return None
        v = np.asarray(v, dtype=float) * 1.0e9
        if attr == "znm":
            v = v * float(getattr(ds.cali, "z_scaling_factor", 1.0) or 1.0)
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


def _values_on_attr_rows(
    ds: MinfluxDataset,
    attr: str,
    *,
    src_itr: str | int,
    src_vld: bool,
) -> np.ndarray | None:
    """*attr* at ``(src_itr, src_vld)`` gathered onto ``ds.attr`` rows.

    The materialized-view counterpart of :func:`_broadcast_iteration_to_rows`:
    the target rows are whatever raw selection ``ds.attr`` materializes
    (:func:`_attr_row_selection`) instead of an arbitrary browse selection.
    ``None`` when there is no aligned raw view, so the caller can fall back to
    the plain materialized values.
    """
    raw = getattr(ds, "mfx_raw", None)
    if raw is None or _mfx_raw_len(raw) == 0:
        return None
    rows = _attr_row_selection(ds)
    if rows is None:
        return None
    target_mask = mfx_row_mask(raw, itr=rows[0], vld_only=rows[1])
    mask_itr = "last" if is_value_pool_selector(src_itr) else src_itr
    src_mask = mfx_row_mask(raw, itr=mask_itr, vld_only=src_vld)
    loc_id = _raw_loc_id(raw)
    if target_mask is None or src_mask is None or loc_id is None:
        return None
    src_vals = mfx_get(ds, attr, itr=src_itr, vld_only=src_vld)
    if src_vals is None:
        return None
    src_vals = np.asarray(src_vals, dtype=float).ravel()
    if src_vals.shape[0] != int(src_mask.sum()):
        return None
    per_loc = np.full(int(loc_id.max()) + 1, np.nan)
    per_loc[loc_id[src_mask]] = src_vals
    out = per_loc[loc_id[target_mask]]
    return out if out.shape[0] == int(getattr(ds.prop, "num_loc", 0)) else None


def attr_values_for_selection(
    ds: MinfluxDataset,
    attr: str,
    *,
    itr: str | int | None = "auto",
) -> np.ndarray | None:
    """1-D values of *attr* aligned to ``ds.attr`` rows for one iteration selection.

    This is the per-localization accessor the Filter dialog and the Histogram's
    materialized path share, so a filter row's ``Iter`` setting and the histogram
    it is authored on always show the same numbers.

    ``itr`` accepts:

    ``"auto"`` / ``None``   the materialized default — cfr/efc at their effective
                            iteration, everything else the last-valid value
                            (i.e. exactly :func:`attr_values_1d`).
    ``"last"``              the value at the global-max iteration. For cfr/efc
                            this is deliberately the raw (≈0) last value, not the
                            effective one — pick ``"effective"`` for that.
    ``"effective"``         cfr/efc's measured iteration; ``"last"`` otherwise.
    ``int``                 a specific 0-based iteration (NaN where a
                            localization has no such iteration).
    ``"sum"`` / ``"average"``  pooled over all of the localization's iterations.

    Returns ``None`` when no aligned 1-D representation exists.
    """
    if ds is None:
        return None
    sel: str | int = itr if itr is not None else "auto"
    if isinstance(sel, str):
        sel = sel.strip().lower() or "auto"
    if sel in ("auto", "browse"):
        sel = "effective" if effective_iteration_for_attr(ds, attr) is not None else "last"
    if sel == "effective":
        eff = effective_iteration_for_attr(ds, attr)
        sel = "last" if eff is None else int(eff)

    # Fast path: the plain last-valid materialization already holds the value.
    if sel == "last" and attr not in _EFFECTIVE_ITER_META and attr_matches_last_valid(ds):
        base = attr_values_1d(ds, attr)
        if base is not None:
            return np.asarray(base)

    if isinstance(sel, (int, np.integer)) and not isinstance(sel, bool):
        src_itr, src_vld = int(sel), False
    else:                                    # "last" / "sum" / "average" / "all"
        src_itr, src_vld = ("all" if sel == "all" else sel), True
    vals = _values_on_attr_rows(ds, attr, src_itr=src_itr, src_vld=src_vld)
    if vals is not None:
        return vals

    # No aligned raw view (e.g. a re-imported flat table): the materialized
    # values are all there is.
    base = attr_values_1d(ds, attr)
    return None if base is None else np.asarray(base)


def attr_matches_last_valid(ds: MinfluxDataset) -> bool:
    """True when ``ds.attr`` is the last-iteration, valid-only materialization.

    This holds for the normal last-valid materialization, where the default plot
    path (``ds.attr`` + ``ds.filter_mask``) equals the ``last`` iteration
    selection. It is False when the dataset was loaded with all iterations
    (an all-iteration store) or invalid localizations, so a ``last`` view must be
    taken from the raw store instead of ``ds.attr``.
    """
    raw = getattr(ds, "mfx_raw", None)
    if raw is None or len(raw) == 0:
        return True
    mask = mfx_row_mask(raw, itr="last", vld_only=True)
    if mask is None:
        return True
    return int(mask.sum()) == int(getattr(ds.prop, "num_loc", 0))
