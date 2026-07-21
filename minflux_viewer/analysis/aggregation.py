"""
minflux_viewer.analysis.aggregation
=====================================
Reproduce Abberior **Imspector "Aggregation"** from raw MINFLUX data.

Imspector's *Aggregation* post-processing bins each trace's localizations by a
**photon threshold**: within a trace, valid final localizations are accumulated
in time order until their combined photon count reaches ``N`` (the ``aggN`` in
the exported filename, e.g. ``agg3000``), then emitted as one **averaged**
localization. This was reverse-engineered from reference raw/aggregated file
pairs and reproduces Imspector's output to ≈99 % (exact photon match, sub-nm
positions, exact begin/end flags) directly from the ``.msr`` raw stream.

Algorithm
---------
1. Keep **valid final** localizations only (``fnl == 1 & vld == 1``); position is
   ``loc`` at that row.
2. **Photon count** per localization = ``eco`` summed over the **final-scale
   iterations** (the localization iterations; see
   :mod:`minflux_viewer.core.mfx_sequence`) — for the standard 3-D sequence the
   final lateral (xy) + final axial (z) iterations.
3. Group by trace id (``tid``); walk each trace in time order, accumulating the
   photon count; when it reaches the threshold, emit one aggregated point
   (mean position, summed photons, mean time) and reset. The trailing remainder
   is emitted as a final point.
4. Flag ``bot`` on the first aggregated point of each trace and ``eot`` on the
   last.

Pure NumPy (no Qt) — unit-tested against the reference files.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Per-localization scalar attributes carried onto the aggregated points (mean
# over each group). Vector attributes (loc 3-vector, dcr 2-vector) are handled
# separately. ``eco`` is summed, not averaged.
_MEAN_SCALAR_ATTRS = ("ecc", "efo", "efc", "fbg", "cfr")
_MEAN_VECTOR_ATTRS = {"dcr": 2, "lnc": 3}


def event_ids(fnl: np.ndarray) -> np.ndarray:
    """0-based event index per raw row (events end at ``fnl == 1``)."""
    f = np.asarray(fnl).astype(np.int64).ravel()
    return (np.cumsum(f) - f).astype(np.int64)


def localization_photons(
    itr: np.ndarray,
    fnl: np.ndarray,
    eco: np.ndarray,
    photon_iters: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-event photon count = eco summed over the ``photon_iters`` rows.

    Returns ``(event_photon, fnl_event_id)`` where ``event_photon[e]`` is the
    summed photon count of event ``e`` and ``fnl_event_id`` maps each ``fnl``
    row to its event id (both indexable by the same event ordering).
    """
    itr = np.asarray(itr).ravel()
    eco = np.asarray(eco, dtype=float).ravel()
    eid = event_ids(fnl)
    n_ev = int(eid.max()) + 1 if eid.size else 0
    photon_mask = np.isin(itr, np.asarray(list(photon_iters), dtype=itr.dtype))
    ev_photon = np.zeros(n_ev, dtype=float)
    if photon_mask.any():
        np.add.at(ev_photon, eid[photon_mask], eco[photon_mask])
    return ev_photon, eid


def aggregate_localizations(
    raw: dict,
    *,
    photon_threshold: float,
    photon_iters: Sequence[int],
    carry_attrs: bool = True,
) -> dict:
    """Aggregate valid final localizations by the Imspector photon-threshold rule.

    Parameters
    ----------
    raw :
        Mapping with 1-D ``itr``, ``fnl``, ``vld``, ``eco``, ``tid``, ``tim`` and
        ``loc`` (``(N, 3)``) raw-iteration arrays (as produced by the MSR reader
        / stored in ``mfx_raw``). Optional extra per-row attributes are carried.
    photon_threshold :
        Photon count per aggregated point (the ``aggN`` parameter).
    photon_iters :
        0-based iteration indices whose ``eco`` sums to the localization photon
        count (from :mod:`minflux_viewer.core.mfx_sequence`).
    carry_attrs :
        Also aggregate the standard per-localization attributes
        (``ecc/efo/efc/fbg/cfr/dcr``) as group means.

    Returns
    -------
    dict with aggregated 1-D arrays ``tid``, ``eco`` (summed photons), ``tim``
    (mean), ``n`` (localizations per point), ``bot``/``eot`` flags, and ``loc``
    (``(M, 3)`` mean position), plus carried mean attributes. Empty arrays when
    there are no valid localizations.
    """
    itr = np.asarray(raw["itr"]).ravel()
    fnl = np.asarray(raw["fnl"]).astype(np.int64).ravel()
    vld = np.asarray(raw["vld"]).astype(np.int64).ravel()
    eco = np.asarray(raw["eco"], dtype=float).ravel()
    tid = np.asarray(raw["tid"]).ravel()
    tim = np.asarray(raw["tim"], dtype=float).ravel()
    loc = np.asarray(raw["loc"], dtype=float).reshape(-1, 3)

    thr = float(photon_threshold)
    ev_photon, eid = localization_photons(itr, fnl, eco, photon_iters)

    # Valid final localizations (one row per localization event).
    fv = np.where((fnl == 1) & (vld == 1))[0]
    if fv.size == 0:
        return _empty_result(loc.shape[1])
    ev_tid = tid[fv]
    ev_tim = tim[fv]
    ev_loc = loc[fv]
    ev_ph = ev_photon[eid[fv]]

    # Extra per-event attributes to carry (values at the fnl row).
    extras_scalar = {}
    extras_vector = {}
    if carry_attrs:
        for name in _MEAN_SCALAR_ATTRS:
            arr = raw.get(name)
            if arr is not None:
                a = np.asarray(arr, dtype=float).ravel()
                if a.size == itr.size:
                    extras_scalar[name] = a[fv]
        for name, width in _MEAN_VECTOR_ATTRS.items():
            arr = raw.get(name)
            if arr is None:
                continue
            a = np.asarray(arr, dtype=float)
            # Accept only a genuine per-row vector: 2-D (rows, width), or a flat
            # (rows*width,) form. A 1-D per-row array (e.g. the primary `dcr`
            # ratio, one value per row) is NOT a (rows, width) vector — skip it
            # rather than reshaping (which corrupts data or raises on odd sizes).
            if a.ndim == 1 and a.size == itr.size * width and width > 1:
                a = a.reshape(-1, width)
            if a.ndim == 2 and a.shape[0] == itr.size and a.shape[1] == width:
                extras_vector[name] = a[fv]

    # Sort valid events by (tid, tim) for a single accumulation pass.
    order = np.lexsort((ev_tim, ev_tid))
    st = ev_tid[order]
    sph = ev_ph[order]
    stm = ev_tim[order]
    slc = ev_loc[order]
    s_scalar = {k: v[order] for k, v in extras_scalar.items()}
    s_vector = {k: v[order] for k, v in extras_vector.items()}

    out_tid, out_eco, out_tim, out_n = [], [], [], []
    out_bot, out_eot = [], []
    out_loc = []
    out_scalar = {k: [] for k in s_scalar}
    out_vector = {k: [] for k in s_vector}

    n = st.size
    i = 0
    while i < n:
        j = i
        while j < n and st[j] == st[i]:
            j += 1
        # trace block [i, j): accumulate to threshold in time order.
        chunks = []
        acc = 0.0
        start = i
        for k in range(i, j):
            acc += sph[k]
            if acc >= thr:
                chunks.append((start, k + 1))
                start = k + 1
                acc = 0.0
        if start < j:
            chunks.append((start, j))
        for ci, (a, b) in enumerate(chunks):
            sl = slice(a, b)
            out_tid.append(st[i])
            out_eco.append(float(sph[sl].sum()))
            out_tim.append(float(stm[sl].mean()))
            out_n.append(b - a)
            out_loc.append(slc[sl].mean(axis=0))
            out_bot.append(1 if ci == 0 else 0)
            out_eot.append(1 if ci == len(chunks) - 1 else 0)
            for key, vv in s_scalar.items():
                seg = vv[sl]
                finite = seg[np.isfinite(seg)]
                out_scalar[key].append(float(finite.mean()) if finite.size else float("nan"))
            for key, vv in s_vector.items():
                out_vector[key].append(vv[sl].mean(axis=0))
        i = j

    result = {
        "tid": np.asarray(out_tid),
        "eco": np.asarray(out_eco, dtype=float),
        "tim": np.asarray(out_tim, dtype=float),
        "n": np.asarray(out_n, dtype=np.int64),
        "bot": np.asarray(out_bot, dtype=np.int8),
        "eot": np.asarray(out_eot, dtype=np.int8),
        "loc": np.asarray(out_loc, dtype=float).reshape(-1, 3),
    }
    for key, vals in out_scalar.items():
        result[key] = np.asarray(vals, dtype=float)
    for key, vals in out_vector.items():
        result[key] = np.asarray(vals, dtype=float)
    return result


def raw_dict_from_dataset(ds) -> Optional[dict]:
    """Assemble the raw-iteration arrays aggregation needs from ``ds.mfx_raw``.

    Returns ``None`` when the dataset lacks the all-iteration raw store or the
    final-iteration flag (``fnl``) required to identify localization events.
    """
    raw = getattr(ds, "mfx_raw", None)
    if raw is None or len(raw) == 0:
        return None
    need = ("itr", "fnl", "vld", "eco", "tid", "tim", "loc_x", "loc_y")
    if any(raw.get(k) is None for k in need):
        return None
    loc_z = raw.get("loc_z")
    n = np.asarray(raw.get("loc_x")).ravel().shape[0]
    loc = np.column_stack([
        np.asarray(raw.get("loc_x"), dtype=float).ravel(),
        np.asarray(raw.get("loc_y"), dtype=float).ravel(),
        (np.asarray(loc_z, dtype=float).ravel() if loc_z is not None else np.zeros(n)),
    ])
    out = {"itr": raw.get("itr"), "fnl": raw.get("fnl"), "vld": raw.get("vld"),
           "eco": raw.get("eco"), "tid": raw.get("tid"), "tim": raw.get("tim"),
           "loc": loc}
    for name in _MEAN_SCALAR_ATTRS:
        if raw.get(name) is not None:
            out[name] = raw.get(name)
    # ``mfx_raw`` stores DCR split into per-channel columns dcr_0/dcr_1 (plus a
    # 1-D primary ``dcr`` ratio). Rebuild the (n, 2) two-channel form so it
    # aggregates as a vector (matches the original mfx / Imspector output). The
    # 1-D primary ``dcr`` is NOT a 2-vector, so never pass it as one.
    d0, d1 = raw.get("dcr_0"), raw.get("dcr_1")
    if d0 is not None and d1 is not None:
        d0 = np.asarray(d0, dtype=float).ravel()
        d1 = np.asarray(d1, dtype=float).ravel()
        if d0.shape == d1.shape:
            out["dcr"] = np.column_stack([d0, d1])
    return out


def aggregate_dataset(ds, *, photon_threshold: float,
                      photon_iters: Optional[Sequence[int]] = None) -> Optional[dict]:
    """Aggregate a loaded dataset (metres coordinates in the result's ``loc``).

    ``photon_iters`` defaults to the dataset's detected photon iterations
    (:func:`minflux_viewer.core.mfx_sequence.photon_iterations_for_dataset`).
    Returns ``None`` when the raw store is unusable.
    """
    raw = raw_dict_from_dataset(ds)
    if raw is None:
        return None
    if photon_iters is None:
        from ..core.mfx_sequence import photon_iterations_for_dataset
        photon_iters = photon_iterations_for_dataset(ds).photon_iters
    return aggregate_localizations(raw, photon_threshold=photon_threshold,
                                   photon_iters=photon_iters)


def _empty_result(width: int = 3) -> dict:
    return {
        "tid": np.empty(0, dtype=np.int64),
        "eco": np.empty(0, dtype=float),
        "tim": np.empty(0, dtype=float),
        "n": np.empty(0, dtype=np.int64),
        "bot": np.empty(0, dtype=np.int8),
        "eot": np.empty(0, dtype=np.int8),
        "loc": np.empty((0, width), dtype=float),
    }
