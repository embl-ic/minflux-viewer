"""
minflux_viewer.utils.filters
=============================
Pure-numpy filter and aggregation helpers.

Extracted from the UI layer so they can be unit-tested without a
QApplication and reused across histogram, filter dialog, and future
analysis modules.
"""

from __future__ import annotations

import numpy as np


def _trace_first(a: np.ndarray) -> float:
    """First value of a trace, positionally (rows are in store/time order)."""
    return float(a[0]) if a.size else float("nan")


def _trace_last(a: np.ndarray) -> float:
    return float(a[-1]) if a.size else float("nan")


def _nan_stat(a: np.ndarray, fn) -> float:
    """Apply a NaN-skipping statistic, defining empty/all-NaN as missing.

    NumPy returns NaN for these inputs but also emits ``RuntimeWarning``. Fully
    unmapped traces are expected for fluorescent attributes outside the image
    footprint, so handle that state explicitly instead of treating it as an
    exceptional numerical condition.
    """
    values = np.asarray(a, dtype=float).ravel()
    if values.size == 0 or np.isnan(values).all():
        return float("nan")
    with np.errstate(all="ignore"):
        return float(fn(values))


def _trace_mean(a: np.ndarray) -> float:
    return _nan_stat(a, np.nanmean)


def _trace_median(a: np.ndarray) -> float:
    return _nan_stat(a, np.nanmedian)


def _trace_min(a: np.ndarray) -> float:
    return _nan_stat(a, np.nanmin)


def _trace_max(a: np.ndarray) -> float:
    return _nan_stat(a, np.nanmax)


def _trace_stdev(a: np.ndarray) -> float:
    return _nan_stat(a, np.nanstd)


def _trace_range(a: np.ndarray) -> float:
    values = np.asarray(a, dtype=float).ravel()
    if values.size == 0 or np.isnan(values).all():
        return float("nan")
    with np.errstate(all="ignore"):
        return float(np.nanmax(values)) - float(np.nanmin(values))


#: The one registry of trace read-outs: label -> function over a trace's values.
#:
#: Every trace aggregation in the app dispatches through this — the histogram's
#: materialized and raw paths, the filter's bounds/spinners/mask. Adding a
#: read-out here makes it available everywhere at once; the previous
#: table-per-call-site arrangement silently degraded to per-localization values
#: wherever a table was missed.
#:
#: ``mean``/``median``/``stdev``/``range`` are NaN-skipping summaries;
#: ``min``/``max`` are NaN-skipping order statistics; ``1st``/``last`` are
#: **positional** — the trace's first and last row in store (time) order, taken
#: literally rather than skipping NaN.
TRACE_AGG_FUNCS = {
    "trace mean":   _trace_mean,
    "trace median": _trace_median,
    "trace min":    _trace_min,
    "trace max":    _trace_max,
    "trace 1st":    _trace_first,
    "trace last":   _trace_last,
    "trace stdev":  _trace_stdev,
    "trace range":  _trace_range,
}

#: Trace read-outs whose result is a derived statistic rather than a data value,
#: so a bound on them is float even when the source attribute is integral.
FLOAT_RESULT_MODES = frozenset({
    "trace mean", "trace median", "trace stdev", "trace range",
})

# Aggregation mode labels — shared by histogram and filter dialog
AGG_MODES: list[str] = ["per loc"] + list(TRACE_AGG_FUNCS)


def trace_agg_func(mode: str):
    """Function for a trace read-out, or ``None`` for ``"per loc"``.

    Raises ``ValueError`` for an unknown mode: returning the input unchanged
    would silently produce one value per row where the caller promised one per
    trace.
    """
    if mode == "per loc":
        return None
    try:
        return TRACE_AGG_FUNCS[mode]
    except KeyError:
        raise ValueError(f"unsupported aggregation mode {mode!r}") from None


def aggregate(
    raw: np.ndarray,
    ftr: np.ndarray,
    mode: str,
    trace_idx: np.ndarray,
    num_traces: int,
) -> np.ndarray:
    """
    Aggregate a per-localisation attribute array.

    Parameters
    ----------
    raw:
        1-D float array, length = num_loc.
    ftr:
        Boolean filter mask, length = num_loc.
    mode:
        One of :data:`AGG_MODES`.
    trace_idx:
        (num_traces, 2) array of [start, end] indices (inclusive).
    num_traces:
        Number of traces.

    Returns
    -------
    np.ndarray
        For "per loc": filtered per-loc values (length = ftr.sum()).
        For trace modes: one value per trace (length = num_traces).
    """
    raw = np.asarray(raw, dtype=float).ravel()

    if mode == "per loc":
        return raw[ftr]

    fn = trace_agg_func(mode)
    with np.errstate(all="ignore"):
        return np.array([
            fn(raw[trace_idx[i, 0] : trace_idx[i, 1] + 1])
            for i in range(num_traces)
        ])


def compute_filter_mask(
    raw: np.ndarray,
    mode: str,
    lo: float,
    hi: float,
    trace_idx: np.ndarray,
    num_loc_per_trace: np.ndarray,
    num_traces: int,
    lo_inclusive: bool = True,
    hi_inclusive: bool = True,
) -> np.ndarray:
    """
    Return a per-localisation boolean mask for a single filter row.

    Parameters
    ----------
    raw:
        1-D float array, length = num_loc.
    mode:
        One of :data:`AGG_MODES`.
    lo, hi:
        Filter range (inclusive).
    trace_idx:
        (num_traces, 2) start/end index array.
    num_loc_per_trace:
        Length of each trace (used to expand trace mask back to per-loc).
    num_traces:
        Number of traces.

    Returns
    -------
    np.ndarray
        Boolean mask, length = num_loc.
    """
    raw = np.asarray(raw, dtype=float).ravel()

    def _in_bounds(values: np.ndarray) -> np.ndarray:
        lo_mask = values >= lo if lo_inclusive else values > lo
        hi_mask = values <= hi if hi_inclusive else values < hi
        return lo_mask & hi_mask

    if mode == "per loc":
        return _in_bounds(raw)

    fn = trace_agg_func(mode)
    with np.errstate(all="ignore"):
        agg = np.array([
            fn(raw[trace_idx[i, 0] : trace_idx[i, 1] + 1])
            for i in range(num_traces)
        ])
    trace_pass = _in_bounds(agg)
    return np.repeat(trace_pass, num_loc_per_trace)


def raw_spec_mask(
    vals: np.ndarray,
    tid: np.ndarray,
    mode: str,
    lo: float,
    hi: float,
    lo_inclusive: bool = True,
    hi_inclusive: bool = True,
) -> np.ndarray:
    """Boolean mask for a single filter spec over an arbitrary row selection.

    Unlike :func:`compute_filter_mask`, this does not assume contiguous
    trace blocks — trace grouping is derived from *tid* on the fly. This lets
    a persisted filter spec be re-evaluated against the raw all-iteration
    store (where each trace has ``n_itr`` rows per localisation).

    Parameters
    ----------
    vals:
        1-D float array of the spec's attribute, length = N (the selection).
    tid:
        1-D trace-id array, same length as *vals*.
    mode:
        One of :data:`AGG_MODES`.
    lo, hi, lo_inclusive, hi_inclusive:
        Filter bounds.

    Returns
    -------
    np.ndarray
        Boolean mask, length N. NaN values fail the bounds test (numpy
        comparisons with NaN yield False), matching the "empty -> excluded"
        convention.
    """
    vals = np.asarray(vals, dtype=float).ravel()

    def _in_bounds(values: np.ndarray) -> np.ndarray:
        lo_mask = values >= lo if lo_inclusive else values > lo
        hi_mask = values <= hi if hi_inclusive else values < hi
        return lo_mask & hi_mask

    if mode == "per loc":
        return _in_bounds(vals)

    tid = np.asarray(tid).ravel()
    if tid.shape[0] != vals.shape[0] or vals.size == 0:
        return _in_bounds(vals)

    # Group by trace id without assuming contiguity.
    order = np.argsort(tid, kind="stable")
    sorted_tid = tid[order]
    sorted_vals = vals[order]
    boundaries = np.flatnonzero(np.diff(sorted_tid)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [sorted_tid.size]])
    counts = ends - starts

    fn = trace_agg_func(mode)
    with np.errstate(all="ignore"):
        agg = np.array([fn(sorted_vals[s:e]) for s, e in zip(starts, ends)])
    trace_pass = _in_bounds(agg)
    sorted_mask = np.repeat(trace_pass, counts)
    out = np.empty(sorted_tid.size, dtype=bool)
    out[order] = sorted_mask
    return out


def raw_trace_aggregate(vals: np.ndarray, tid: np.ndarray, mode: str) -> np.ndarray:
    """One value per trace for an arbitrary (non-contiguous) row selection.

    Mirrors the histogram's trace aggregation but derives trace groups from
    *tid* instead of assuming contiguous trace blocks, so it works on the raw
    all-iteration store. Returns *vals* unchanged for ``"per loc"``.
    """
    vals = np.asarray(vals, dtype=float).ravel()
    if mode == "per loc":
        return vals
    tid = np.asarray(tid).ravel()
    if tid.shape[0] != vals.shape[0] or vals.size == 0:
        return vals

    order = np.argsort(tid, kind="stable")
    sorted_vals = vals[order]
    sorted_tid = tid[order]
    boundaries = np.flatnonzero(np.diff(sorted_tid)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [sorted_tid.size]])

    # The stable sort above keeps each trace's rows in store order, which is
    # localization-major / iteration-minor — so the positional read-outs are the
    # trace's first and last localization under a single-iteration selection,
    # and its first and last raw row under `all [flatten]`.
    fn = trace_agg_func(mode)
    with np.errstate(all="ignore"):
        return np.array([fn(sorted_vals[s:e]) for s, e in zip(starts, ends)])


def fast_density_2d(x: np.ndarray, y: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Fast 2-D histogram-based density estimate.

    Assigns each point the bin count of the histogram cell it falls in.
    Much faster than KD-tree range search; good enough for colour-coding
    scatter plots.

    Parameters
    ----------
    x, y:
        1-D coordinate arrays in any units.
    bins:
        Number of histogram bins per axis.

    Returns
    -------
    np.ndarray
        Per-point density values (non-negative floats), same length as *x*.
    """
    if x.size == 0:
        return np.empty(0, dtype=float)

    h, xedge, yedge = np.histogram2d(x, y, bins=bins)
    xi = np.clip(np.searchsorted(xedge, x) - 1, 0, bins - 1)
    yi = np.clip(np.searchsorted(yedge, y) - 1, 0, bins - 1)
    return h[xi, yi].astype(float)


def fast_density_3d(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, bins: int = 64
) -> np.ndarray:
    """
    Fast 3-D histogram-based density estimate.

    Assigns each point the bin count of the 3-D histogram cell it falls in.
    Same idea as :func:`fast_density_2d` but with an extra dimension.

    Parameters
    ----------
    x, y, z:
        1-D coordinate arrays in any units; all must be the same length.
    bins:
        Number of histogram bins per axis. ``64**3 = 262 144`` cells —
        enough resolution for colour coding without using too much memory.

    Returns
    -------
    np.ndarray
        Per-point density values (non-negative floats), same length as *x*.
    """
    if x.size == 0:
        return np.empty(0, dtype=float)

    h, edges = np.histogramdd(np.column_stack([x, y, z]), bins=bins)
    xe, ye, ze = edges
    xi = np.clip(np.searchsorted(xe, x) - 1, 0, bins - 1)
    yi = np.clip(np.searchsorted(ye, y) - 1, 0, bins - 1)
    zi = np.clip(np.searchsorted(ze, z) - 1, 0, bins - 1)
    return h[xi, yi, zi].astype(float)
