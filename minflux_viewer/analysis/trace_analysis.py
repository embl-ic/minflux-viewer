"""
minflux_viewer.analysis.trace_analysis
=======================================
Trace-level size and anisotropy estimation.

Python port of the MATLAB functions ``estimate_size_ND`` and
``estimate_anisotropy`` (Wanlu Liu / Abberior group).

Public API
----------
estimate_size_nd(data, ids, ...)
    Fit a log-normal distribution to intra-cluster distances and return
    four size estimates (mean, median, max-bin, Gaussian-fit peak).

estimate_anisotropy(loc_nm, tid, ...)
    Estimate the Z scaling factor from per-axis size ratios.

show_trace_size_dialog(parent, ds, state)
show_anisotropy_dialog(parent, ds)
    Convenience wrappers that pull data from a MinfluxDataset and display
    a result dialog.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from ..utils.import_compat import resolve_moved_symbol

# ---------------------------------------------------------------------------
# Core computation — estimate_size_nd
# ---------------------------------------------------------------------------

@dataclass
class SizeFitResult:
    sizes: np.ndarray
    logdist: np.ndarray
    counts: np.ndarray
    edges: np.ndarray
    bin_centers: np.ndarray
    fit_params: tuple[float, float, float] | None


def _gauss1(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """MATLAB ``fit(..., 'gauss1')`` model: a*exp(-((x-b)/c)^2)."""
    return float(a) * np.exp(-((x - float(b)) / max(abs(float(c)), 1e-9)) ** 2)


def estimate_size_nd_details(
    data: np.ndarray,
    ids: np.ndarray,
    *,
    bin_width: float = 0.1,
    exclude_zero_id: bool = True,
) -> SizeFitResult:
    """Detailed version of :func:`estimate_size_nd` with histogram/fit data."""
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    ids = np.asarray(ids, dtype=np.int64).ravel()

    uid = np.unique(ids)
    if exclude_zero_id:
        uid = uid[uid != 0]

    empty = SizeFitResult(
        sizes=np.full(4, np.nan),
        logdist=np.array([]),
        counts=np.array([]),
        edges=np.array([]),
        bin_centers=np.array([]),
        fit_params=None,
    )
    if uid.size == 0:
        return empty

    # Group rows by id with a single stable sort, then walk contiguous blocks.
    # The previous ``mask = ids == cid`` inside the id loop was O(N × n_ids) —
    # with tens of thousands of traces that dominated dataset open time.
    order = np.argsort(ids, kind="stable")
    ids_sorted = ids[order]
    data_sorted = data[order]
    starts = np.flatnonzero(np.r_[True, ids_sorted[1:] != ids_sorted[:-1]])
    ends = np.r_[starts[1:], ids_sorted.size]

    parts: list[np.ndarray] = []
    for s, e in zip(starts, ends):
        if exclude_zero_id and ids_sorted[s] == 0:
            continue
        if e - s < 2:
            continue
        pts = data_sorted[s:e]
        centroid = np.median(pts, axis=0)
        parts.append(np.linalg.norm(pts - centroid, axis=1))

    if not parts:
        return empty

    dist = np.concatenate(parts)
    dist = dist[np.isfinite(dist) & (dist > 0)]
    if dist.size < 3:
        return empty

    logdist = np.log(dist)
    sizes = np.full(4, np.nan)
    sizes[0] = np.exp(np.mean(logdist))
    sizes[1] = np.exp(np.median(logdist))

    lo = float(np.nanmin(logdist))
    hi = float(np.nanmax(logdist))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return SizeFitResult(sizes, logdist, np.array([]), np.array([]), np.array([]), None)

    # MATLAB ``histcounts(logdist, 'BinWidth', bw)`` aligns bin edges to integer
    # multiples of the bin width; reproduce that exactly so the max-bin and
    # Gaussian-fit estimates match ``estimate_size_ND.m``.
    e_lo = np.floor(lo / bin_width) * bin_width
    e_hi = np.ceil(hi / bin_width) * bin_width
    edges = np.arange(e_lo, e_hi + bin_width / 2.0, bin_width, dtype=float)
    if edges.size < 3:
        edges = np.linspace(e_lo, e_hi + bin_width, 10)
    counts, edges = np.histogram(logdist, bins=edges)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    if counts.size == 0:
        return SizeFitResult(sizes, logdist, counts, edges, bin_centers, None)

    max_idx = int(np.argmax(counts))
    sizes[2] = np.exp((edges[max_idx] + edges[max_idx + 1]) / 2.0)

    fit_params = None
    try:
        # MATLAB ``fit(bincenter, counts, 'gauss1')`` fits ALL bins, including
        # the zero-count tails — do the same.
        fit_x = bin_centers
        fit_y = counts.astype(float)
        p0 = [
            float(counts[max_idx]),
            float(bin_centers[max_idx]),
            max(float(np.nanstd(logdist)), 0.1),
        ]
        popt, _ = curve_fit(_gauss1, fit_x, fit_y, p0=p0, maxfev=10000)
        fit_params = (float(popt[0]), float(popt[1]), float(popt[2]))
        sizes[3] = np.exp(fit_params[1])
    except Exception:
        sizes[3] = sizes[2]

    return SizeFitResult(sizes, logdist, counts, edges, bin_centers, fit_params)


def estimate_size_nd(
    data: np.ndarray,
    ids: np.ndarray,
    *,
    bin_width: float = 0.1,
    exclude_zero_id: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate cluster / trace size from intra-cluster distance distribution.

    Port of MATLAB ``estimate_size_ND``.

    Parameters
    ----------
    data : array_like, shape (N,) or (N, M)
        Coordinates in nm. Pass the full N×3 array for 3-D size, or a
        single column for per-axis size.
    ids : array_like, shape (N,)
        Cluster / trace ID per row.  IDs equal to 0 are excluded by default.
    bin_width : float
        Histogram bin width for the log-distance distribution.
    exclude_zero_id : bool
        If True (default), rows whose ID is 0 are ignored.

    Returns
    -------
    sizes : ndarray of shape (4,)
        ``[mean, median, max_bin, gaussian_fit]`` in nm.
    logdist : ndarray
        Log-transformed distances used for the fit (useful for plotting).
    """
    result = estimate_size_nd_details(
        data,
        ids,
        bin_width=bin_width,
        exclude_zero_id=exclude_zero_id,
    )
    return result.sizes, result.logdist


# ---------------------------------------------------------------------------
# Core computation — estimate_anisotropy
# ---------------------------------------------------------------------------

def estimate_anisotropy(
    loc_nm: np.ndarray,
    tid: np.ndarray,
    *,
    mode: int = 3,
    bin_width: float = 0.1,
    exclude_zero_id: bool = False,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the Z scaling factor from per-axis trace sizes.

    Port of MATLAB ``estimate_anisotropy``.

    Parameters
    ----------
    loc_nm : (N, 3) ndarray
        Localizations in **nanometres** (x, y, z columns).
    tid : (N,) ndarray
        Trace / cluster IDs.
    mode : int
        Which size estimate to use: 0=mean, 1=median, 2=max-bin, 3=Gauss-fit.
    bin_width, exclude_zero_id : passed through to estimate_size_nd.

    Returns
    -------
    z_scaling_factor : float
        Z scaling factor = geomean(size_x, size_y) / size_z.
    sizes_x, sizes_y, sizes_z : ndarray (4,)
        The four size estimates for each axis.
    """
    loc_nm = np.asarray(loc_nm, dtype=float)
    tid = np.asarray(tid).ravel()

    result = estimate_anisotropy_details(
        loc_nm,
        tid,
        mode=mode,
        bin_width=bin_width,
        exclude_zero_id=exclude_zero_id,
    )
    return (
        result["z_scaling_factor"],
        result["x"].sizes,
        result["y"].sizes,
        result["z"].sizes,
    )


def estimate_anisotropy_details(
    loc_nm: np.ndarray,
    tid: np.ndarray,
    *,
    mode: int = 3,
    bin_width: float = 0.1,
    exclude_zero_id: bool = False,
) -> dict:
    """Detailed anisotropy result including per-axis log-histogram fits.

    ``exclude_zero_id`` defaults to ``False`` to match ``estimate_anisotropy.m``
    / ``estimate_size_ND.m``, which take ``unique(ID)`` without dropping id 0.
    """
    loc_nm = np.asarray(loc_nm, dtype=float)
    tid = np.asarray(tid).ravel()

    kw = dict(bin_width=bin_width, exclude_zero_id=exclude_zero_id)
    fit_x = estimate_size_nd_details(loc_nm[:, 0], tid, **kw)
    fit_y = estimate_size_nd_details(loc_nm[:, 1], tid, **kw)
    fit_z = estimate_size_nd_details(loc_nm[:, 2], tid, **kw)

    # Z scaling factor for every size-estimate mode (mean, median, max-bin,
    # Gaussian fit); mode 3 is the default/final estimate.
    z_scaling_factor_by_mode = [
        _z_scaling_factor_from_sizes(
            float(fit_x.sizes[m]), float(fit_y.sizes[m]), float(fit_z.sizes[m])
        )
        for m in range(4)
    ]
    z_scaling_factor = z_scaling_factor_by_mode[mode]

    z_corrected = estimate_size_nd_details(loc_nm[:, 2] * z_scaling_factor, tid, **kw) if np.isfinite(z_scaling_factor) else None
    return {
        "z_scaling_factor": z_scaling_factor,
        "z_scaling_factor_by_mode": z_scaling_factor_by_mode,
        "mode": int(mode),
        "bin_width": float(bin_width),
        "x": fit_x,
        "y": fit_y,
        "z": fit_z,
        "z_corrected": z_corrected,
    }


def _z_scaling_factor_from_sizes(sx: float, sy: float, sz: float) -> float:
    """Z scaling factor = geomean(size_x, size_y) / size_z (NaN if undefined)."""
    if (not np.isfinite(sx) or not np.isfinite(sy) or not np.isfinite(sz)
            or sx <= 0 or sy <= 0 or sz <= 0):
        return float("nan")
    return float(np.exp((np.log(sx) + np.log(sy)) / 2.0) / sz)


# ---------------------------------------------------------------------------
# Result dialogs
# ---------------------------------------------------------------------------

_MODE_LABELS = ["Mean", "Median", "Max-bin", "Gaussian fit"]
_DEFAULT_MODE = 3  # Gaussian fit (same default as MATLAB)

# Per-axis plot colors (vivid on the dark plot background).
_AXIS_COLORS = {"X": "#35d07f", "Y": "#4a90ff", "Z": "#ff5a5a"}  # green / blue / red
_TRACE_SIZE_COLORS = {"XY": "#35d07f", "Z": "#ff5a5a", "XYZ": "#4a90ff"}  # green / red / blue




def raw_last_valid_loc_nm(ds) -> tuple[np.ndarray, np.ndarray] | None:
    """Raw last-valid-iteration ``loc`` in **nanometres** plus trace ids.

    Reads the raw ``loc_x``/``loc_y``/``loc_z`` (metres) from the last valid
    iteration via :func:`mfx_get` and scales to nm. The raw coordinates are
    used deliberately and unconditionally: Z scaling factor estimation from
    trace anisotropy must not see the Z-scaled ``ds.loc_nm`` (that would make
    the estimate circular and dependent on the current calibration).
    Returns ``None`` if coordinates are unavailable.
    """
    from minflux_viewer.core.loader_raw_access import mfx_get

    def _col(name: str) -> np.ndarray | None:
        v = mfx_get(ds, name, itr="last", vld_only=True)
        if v is None:
            v = ds.attr.get(name)
        return None if v is None else np.asarray(v, dtype=float).ravel()

    x, y, z = _col("loc_x"), _col("loc_y"), _col("loc_z")
    if x is None or y is None or z is None:
        return None
    n = min(x.size, y.size, z.size)
    if n == 0:
        return None
    loc_nm = np.column_stack([x[:n], y[:n], z[:n]]) * 1e9

    tid = mfx_get(ds, "tid", itr="last", vld_only=True)
    if tid is None:
        tid = ds.attr.get("tid")
    if tid is None:
        return None
    tid = np.asarray(tid, dtype=np.int64).ravel()[:n]
    return loc_nm, tid


def estimate_anisotropy_for_dataset(ds, *, mode: int = 3) -> dict | None:
    """Anisotropy estimate from a dataset's RAW last-valid coordinates.

    Always reads raw ``loc`` (never the Z-scaled ``ds.loc_nm``), so the result
    is independent of the current Z scaling factor: running it twice gives the
    same value. Returns the detailed result dict, or ``None``
    when 3-D coordinates / traces are unavailable.
    """
    from ..core.dataset_kind import is_3d

    if not is_3d(ds):
        return None
    data = raw_last_valid_loc_nm(ds)
    if data is None:
        return None
    loc, tid = data
    if loc.shape[1] < 3 or int(np.unique(tid[tid != 0]).size) == 0:
        return None
    return estimate_anisotropy_details(loc, tid, mode=mode)




# ---------------------------------------------------------------------------
# Private dialog classes
# ---------------------------------------------------------------------------
























_MOVED_SYMBOLS = {'show_trace_size_dialog': 'minflux_viewer.ui.trace_analysis_dialogs:show_trace_size_dialog', 'show_anisotropy_dialog': 'minflux_viewer.ui.trace_analysis_dialogs:show_anisotropy_dialog', '_nm': 'minflux_viewer.ui.trace_analysis_dialogs:_nm', '_TraceSizeDialog': 'minflux_viewer.ui.trace_analysis_dialogs:_TraceSizeDialog', '_AnisotropyDialog': 'minflux_viewer.ui.trace_analysis_dialogs:_AnisotropyDialog', '_plot_color': 'minflux_viewer.ui.trace_analysis_dialogs:_plot_color', '_fill_brush': 'minflux_viewer.ui.trace_analysis_dialogs:_fill_brush', '_anisotropy_hist_plot': 'minflux_viewer.ui.trace_analysis_dialogs:_anisotropy_hist_plot', '_anisotropy_overlay_plot': 'minflux_viewer.ui.trace_analysis_dialogs:_anisotropy_overlay_plot', '_plot_gaussian_fit': 'minflux_viewer.ui.trace_analysis_dialogs:_plot_gaussian_fit', '_plot_size_marker': 'minflux_viewer.ui.trace_analysis_dialogs:_plot_size_marker', '_value_hist_plot': 'minflux_viewer.ui.trace_analysis_dialogs:_value_hist_plot', '_widget_from_layout': 'minflux_viewer.ui.trace_analysis_dialogs:_widget_from_layout'}

def __getattr__(name):
    return resolve_moved_symbol(__name__, name, _MOVED_SYMBOLS)
