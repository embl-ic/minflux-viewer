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


# Aggregation mode labels — shared by histogram and filter dialog
AGG_MODES: list[str] = [
    "per loc",
    "trace mean",
    "trace stdev",
    "trace max",
    "trace min",
    "trace range",
]


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

    def _trace_agg(fn) -> np.ndarray:
        return np.array([
            fn(raw[trace_idx[i, 0] : trace_idx[i, 1] + 1])
            for i in range(num_traces)
        ])

    dispatch = {
        "trace mean":  lambda: _trace_agg(np.nanmean),
        "trace stdev": lambda: _trace_agg(np.nanstd),
        "trace max":   lambda: _trace_agg(np.nanmax),
        "trace min":   lambda: _trace_agg(np.nanmin),
        "trace range": lambda: _trace_agg(
            lambda a: float(np.nanmax(a)) - float(np.nanmin(a))
        ),
    }
    if mode in dispatch:
        return dispatch[mode]()

    # fallback
    return raw[ftr]


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

    def _agg(fn) -> np.ndarray:
        return np.array([
            fn(raw[trace_idx[i, 0] : trace_idx[i, 1] + 1])
            for i in range(num_traces)
        ])

    if mode == "trace mean":
        agg = _agg(np.nanmean)
    elif mode == "trace stdev":
        agg = _agg(np.nanstd)
    elif mode == "trace max":
        agg = _agg(np.nanmax)
    elif mode == "trace min":
        agg = _agg(np.nanmin)
    elif mode == "trace range":
        agg = _agg(lambda a: float(np.nanmax(a)) - float(np.nanmin(a)))
    else:
        return _in_bounds(raw)

    trace_pass = _in_bounds(agg)
    return np.repeat(trace_pass, num_loc_per_trace)


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
