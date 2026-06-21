"""
minflux_viewer.plugins.trace_viewer.trace_data
==============================================
Pure per-trace computations for the Trace Viewer plugin — a faithful port of the
MATLAB ``MINFLUX_data_plot_trace_viewer`` data layer.

Everything here is NumPy/SciPy only (no Qt), so the trace table, time-series, the
"remove tail" heuristic and the export payload are unit-testable. Units mirror
the MATLAB app: dt in ms, velocity in µm/s, sizes in nm.
"""

from __future__ import annotations

import numpy as np

from ...core.loader import attr_values_1d

# Per-trace table columns (always present); ch1/ch2 are inserted after efo when
# the dataset carries conf_ch1 / conf_ch2 (matches the MATLAB conditional table).
BASE_HEADERS = ["trace ID", "N loc", "efo", "dcr", "dt (ms)", "velocity (µm/s)", "size (nm)"]


def _col(ds, name):
    v = attr_values_1d(ds, name)
    return None if v is None else np.asarray(v, dtype=float).ravel()


def _loc_nm(ds) -> np.ndarray:
    loc = np.asarray(ds.loc_nm, dtype=float)
    if loc.ndim != 2:
        return np.empty((0, 3))
    if loc.shape[1] == 2:
        loc = np.column_stack([loc, np.zeros(loc.shape[0])])
    return loc[:, :3]


def moving_rms(x, window: int) -> np.ndarray:
    """``sqrt(movmean(x**2, window))`` — RMS over a centred moving window."""
    return np.sqrt(_movmean(np.asarray(x, dtype=float) ** 2, window))


def _movmean(x: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average matching MATLAB ``movmean`` (shrinking edges)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    w = max(1, int(window))
    if n == 0 or w <= 1:
        return x.copy()
    before, after = (w - 1) // 2, w // 2
    csum = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(n)
    lo = np.maximum(0, idx - before)
    hi = np.minimum(n, idx + after + 1)
    return (csum[hi] - csum[lo]) / (hi - lo)


def bbox_diagonal_nm(xyz: np.ndarray) -> float:
    """Diagonal length of the axis-aligned bounding box (``vecnorm(range(xyz))``)."""
    xyz = np.asarray(xyz, dtype=float)
    if xyz.shape[0] == 0:
        return 0.0
    rng = np.nanmax(xyz, axis=0) - np.nanmin(xyz, axis=0)
    return float(np.linalg.norm(rng))


def trace_ids(tid: np.ndarray) -> np.ndarray:
    """Unique non-zero trace IDs."""
    u = np.unique(np.asarray(tid).ravel())
    return u[u != 0]


def build_trace_table(ds, tid: np.ndarray | None = None):
    """Per-trace summary rows + headers, sorted by N loc (descending).

    Returns ``(rows, headers)`` where each row is a dict keyed by header.
    """
    tid = _col_tid(ds) if tid is None else np.asarray(tid).ravel()
    loc = _loc_nm(ds)
    efo, dcr = _col(ds, "efo"), _col(ds, "dcr")
    dt, spd = _col(ds, "dt"), _col(ds, "spd")
    ch1, ch2 = _col(ds, "conf_ch1"), _col(ds, "conf_ch2")

    headers = ["trace ID", "N loc", "efo"]
    if ch1 is not None:
        headers.append("ch1")
    if ch2 is not None:
        headers.append("ch2")
    headers += ["dcr", "dt (ms)", "velocity (µm/s)", "size (nm)"]

    def mean_of(arr, m):
        return float(np.nanmean(arr[m])) if arr is not None and m.any() else np.nan

    rows = []
    for t in trace_ids(tid):
        m = tid == t
        n = int(m.sum())
        row = {"trace ID": int(t), "N loc": n, "efo": mean_of(efo, m)}
        if ch1 is not None:
            row["ch1"] = mean_of(ch1, m)
        if ch2 is not None:
            row["ch2"] = mean_of(ch2, m)
        row["dcr"] = mean_of(dcr, m)
        row["dt (ms)"] = 1e3 * mean_of(dt, m) if (dt is not None and n > 1) else np.nan
        row["velocity (µm/s)"] = 1e6 * mean_of(spd, m) if spd is not None else np.nan
        row["size (nm)"] = bbox_diagonal_nm(loc[m]) if n > 1 else 0.0
        rows.append(row)

    rows.sort(key=lambda r: r["N loc"], reverse=True)
    return rows, headers


def _col_tid(ds) -> np.ndarray:
    v = attr_values_1d(ds, "tid")
    return np.zeros(0) if v is None else np.asarray(v).ravel()


def trace_time_series(ds, tid: np.ndarray, trace_id: int) -> dict:
    """Centred XYZ + the four time-series for one trace (first point dropped for
    the time plots, exactly as the MATLAB app does)."""
    tid = np.asarray(tid).ravel()
    idx = np.flatnonzero(tid == trace_id)
    loc = _loc_nm(ds)
    tim = _col(ds, "tim")
    xyz = loc[idx] - loc[idx].mean(axis=0) if idx.size else np.empty((0, 3))
    t = (tim[idx] - tim[idx][0]) if (tim is not None and idx.size) else np.zeros(idx.size)

    out = {"xyz": xyz, "t": t, "n": int(idx.size)}
    if idx.size <= 1:
        out.update(ts=np.zeros(0), dt_ms=np.zeros(0), v_rms=np.zeros(0),
                   eco=np.zeros(0), eco_rms=np.zeros(0), efo=np.zeros(0), efo_rms=np.zeros(0))
        return out

    s = idx[1:]
    ts = t[1:]
    dt, spd = _col(ds, "dt"), _col(ds, "spd")
    eco, efo = _col(ds, "eco"), _col(ds, "efo")
    out["ts"] = ts
    out["dt_ms"] = 1e3 * dt[s] if dt is not None else np.zeros(ts.size)
    out["v_rms"] = 1e6 * moving_rms(spd[s], 10) if spd is not None else np.zeros(ts.size)
    if eco is not None:
        eco_v = eco[s]
        eco_rms = moving_rms(eco_v, 50)
        out["eco"], out["eco_rms"] = eco_v, eco_rms
    else:
        eco_v = eco_rms = None
        out["eco"], out["eco_rms"] = np.zeros(ts.size), np.zeros(ts.size)
    if efo is not None:
        efo_v = efo[s]
        out["efo"] = efo_v
        # rescale the eco RMS into efo units (MATLAB: match means/stds)
        if eco_v is not None:
            eco_mean, eco_std = float(eco_v.mean()), float(eco_v.std())
            efo_mean, efo_std = float(efo_v.mean()), float(efo_v.std())
            scale = (efo_std / eco_std) if eco_std else 0.0
            out["efo_rms"] = (eco_rms - eco_mean) * scale + efo_mean
        else:
            out["efo_rms"] = moving_rms(efo_v, 50)
    else:
        out["efo"], out["efo_rms"] = np.zeros(ts.size), np.zeros(ts.size)
    return out


def remove_trace_tail(ds, tid: np.ndarray | None = None, *,
                      vel_window: int = 10, dist_thresh_nm: float = 20.0) -> np.ndarray:
    """Return a copy of ``tid`` with each trace's leading 'approach' segment
    zeroed (the MATLAB *remove tail* heuristic): cut where the RMS velocity first
    rises, then extend to where the point's mean distance to the trace drops below
    ``dist_thresh_nm``."""
    tid = (_col_tid(ds) if tid is None else np.asarray(tid)).ravel().copy()
    loc = _loc_nm(ds)
    spd = _col(ds, "spd")
    if spd is None or tid.size == 0:
        return tid
    try:
        from scipy.spatial.distance import cdist
    except Exception:
        return tid
    for t in trace_ids(tid):
        idx = np.flatnonzero(tid == t)
        if idx.size < 2:
            continue
        vrms = 1e6 * moving_rms(spd[idx], vel_window)
        rising = np.flatnonzero(np.diff(vrms) > 0)
        first_cut = int(rising[0]) + 1 if rising.size else 0
        d_mean = cdist(loc[idx], loc[idx]).mean(axis=0)
        tail = np.flatnonzero(d_mean[first_cut:] < dist_thresh_nm)
        second_cut = int(tail[0]) if tail.size else 0
        final_cut = first_cut + second_cut
        if final_cut > 0:
            tid[idx[:final_cut]] = 0
    return tid


def export_payload(ds, tid: np.ndarray, indices) -> list[dict]:
    """Per-trace raw data for export (tid, loc[m], tim, efo, eco, dcr, ch1/ch2)."""
    tid = np.asarray(tid).ravel()
    loc = _loc_nm(ds)
    cols = {k: _col(ds, k) for k in ("tim", "efo", "eco", "dcr", "conf_ch1", "conf_ch2")}
    out = []
    for t in indices:
        m = tid == int(t)
        rec = {"tid": int(t), "loc_nm": loc[m]}
        for k, v in cols.items():
            if v is not None:
                rec[k.replace("conf_", "")] = v[m]
        out.append(rec)
    return out
