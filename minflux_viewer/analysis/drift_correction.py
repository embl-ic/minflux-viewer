"""
minflux_viewer.analysis.drift_correction
==========================================
Time-window auto-correlation **drift correction** for MINFLUX localizations.

This is a faithful, Qt-free reimplementation of the 2D/3D drift-correction
functions exposed by **pyMINFLUX** (``pyminflux.correct``), which are themselves
reimplemented (with modifications) from the DNA-PAINT MINFLUX nanoscopy method:

* [paper] Ostersehlt, L.M., Jans, D.C., Wittek, A. et al. *DNA-PAINT MINFLUX
  nanoscopy.* Nat Methods 19, 1072-1075 (2022).
  https://doi.org/10.1038/s41592-022-01577-1
* [code]  https://zenodo.org/record/6563100  (Apache-2.0)
* [pyMINFLUX] https://github.com/bsse-scf/pyMINFLUX  (Apache-2.0)

Algorithm
---------
1. Split the acquisition time span into ``Ns`` equal windows of length ``T``.
2. Render each window as a Gaussian-smoothed localization image / volume.
3. FFT-cross-correlate every window pair (up to ``D`` apart) and refine the
   correlation peak with an iterative weighted centroid → a set of pairwise
   shift measurements.
4. Solve, per spatial axis, a regularised (roughness-penalised) least-squares
   problem for the per-window drift trajectory.
5. Linearly interpolate the trajectory onto every localization's time point.

The rendering here uses a fast histogram + Gaussian blur (equivalent for the
cross-correlation peak) instead of pyMINFLUX's per-localization Gaussian splat,
but the image **orientation, cross-correlation, centroid refinement and
optimisation are identical**, so the returned drift has the same sign
convention and magnitude.

Units
-----
``x``/``y``/``z`` are in **nanometres**, ``t`` in **seconds**, ``tid`` are
integer trace ids, ``resolution_nm`` is the render pixel/voxel size in nm and
``time_window_s`` the window length in seconds. The returned drift (``dx``,
``dy``, ``dz``) is in **nanometres** and is meant to be **subtracted** from the
coordinates to correct them (``x_corrected = x - dx``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.fft import ifftshift
from scipy.fft import fftn, ifftn
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize

# FWHM (nm) -> Gaussian sigma in pixels conversion constant (2*sqrt(2*ln2)).
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

# Minimum floor on the auto time window (seconds): at least 10 minutes, matching
# pyMINFLUX. Kept as a module constant so the UI can show/adjust the heuristic.
_MIN_AUTO_WINDOW_S = 600.0
_MAX_AUTO_WINDOW_S = 3600.0

# Guard against a huge render grid (FOV / resolution too fine) exhausting memory.
_MAX_RENDER_CELLS = 64_000_000


@dataclass
class DriftResult:
    """Result of a drift estimation.

    ``dx``/``dy``/``dz`` are per-localization drift in nm (subtract to correct).
    ``dxt``/``dyt``/``dzt`` are the drift at the ``ti`` window-center times, i.e.
    the estimated trajectory before interpolation. ``dz*`` are ``None`` in 2D.
    """

    dims: int
    dx: np.ndarray
    dy: np.ndarray
    dz: Optional[np.ndarray]
    ti: np.ndarray
    dxt: np.ndarray
    dyt: np.ndarray
    dzt: Optional[np.ndarray]
    time_window_s: float
    n_windows: int
    resolution_nm: float
    rx: tuple
    ry: tuple
    rz: Optional[tuple] = None
    rms: tuple = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Time-window heuristic
# ---------------------------------------------------------------------------

def default_time_window(
    t: np.ndarray,
    tid: np.ndarray,
    rx: tuple,
    ry: tuple,
) -> float:
    """pyMINFLUX heuristic for the analysis time window ``T`` (seconds).

    ``T = n_traces * width * height / 3e6`` clipped so there are at least two
    windows (``<= span/2``, ``<= 3600``) and each is at least 10 minutes long.
    """
    t = np.asarray(t)
    tid = np.asarray(tid)
    rt = (t[0], t[-1])
    span = float(rt[1] - rt[0])
    n_traces = len(np.unique(tid))
    T = n_traces * (rx[1] - rx[0]) * (ry[1] - ry[0]) / 3e6
    T = float(np.min([T, span / 2.0, _MAX_AUTO_WINDOW_S]))
    T = float(max(T, _MIN_AUTO_WINDOW_S))
    return T


# ---------------------------------------------------------------------------
# Rendering (histogram + Gaussian blur, pyMINFLUX orientation)
# ---------------------------------------------------------------------------

def _round_edges(r0: float, n: int, s: float) -> np.ndarray:
    """Bin edges that reproduce round-to-nearest binning of ``(v - r0) / s``."""
    return r0 + (np.arange(n + 1) - 0.5) * s


def _render_2d(x, y, s, rx, ry, nx, ny, sigma_px) -> np.ndarray:
    """Gaussian-smoothed 2D render, oriented like pyMINFLUX (row 0 = max y)."""
    counts, _, _ = np.histogram2d(
        x, y, bins=[_round_edges(rx[0], nx, s), _round_edges(ry[0], ny, s)]
    )
    # counts is (nx, ny) indexed [ix, iy]; pyMINFLUX image is (ny, nx) with
    # row = ny - iy - 1 (image y increases downward).
    h = counts.T[::-1, :].astype(np.float64)
    if sigma_px > 0:
        h = gaussian_filter(h, sigma=sigma_px, mode="constant")
    return h


def _render_3d(x, y, z, s, rx, ry, rz, nx, ny, nz, sigma_px) -> np.ndarray:
    """Gaussian-smoothed 3D render, oriented like pyMINFLUX (y flipped, z direct)."""
    counts, _ = np.histogramdd(
        np.column_stack([x, y, z]),
        bins=[_round_edges(rx[0], nx, s),
              _round_edges(ry[0], ny, s),
              _round_edges(rz[0], nz, s)],
    )
    # counts is (nx, ny, nz) indexed [ix, iy, iz]; target is (nz, ny, nx) with
    # the y axis flipped (image convention), z direct.
    h = np.transpose(counts, (2, 1, 0))[:, ::-1, :].astype(np.float64)
    if sigma_px > 0:
        h = gaussian_filter(h, sigma=sigma_px, mode="constant")
    return h


# ---------------------------------------------------------------------------
# Regularised least-squares cost (verbatim port)
# ---------------------------------------------------------------------------

def lse_distance(x, a, b, s, w, l):
    """Least-squares error of predicted pairwise shifts with a roughness penalty.

    ``a``/``b`` index the window pairs, ``s`` the measured shifts, ``w`` the
    distance weights and ``l`` the roughness regularisation weight. Port of the
    pyMINFLUX ``lse_distance`` cost.
    """
    x = np.concatenate(([0], x))
    dx = x[b] - x[a]
    y = w[b - a] * (dx - s) ** 2
    y = np.mean(y)
    r = l * np.sum(np.diff(x) ** 2)
    return y + r


def _refine_centroid_nd(hj, c, grid_axes, CR):
    """Iterative weighted-centroid refinement of a cross-correlation peak.

    ``grid_axes`` is a list of the 1D integer sample ranges per axis (already
    clamped in-bounds); returns the refined centroid coordinates in the same
    axis order as ``c``.
    """
    mesh = np.meshgrid(*grid_axes, indexing="ij")
    d = hj[tuple(mesh)].astype(np.float64)
    flat = [m.flatten() for m in mesh]
    d = d.flatten()
    d = d - np.min(d)
    total = np.sum(d)
    if total <= 0:
        return np.array([float(v) for v in c])
    d = d / total
    ctr = [float(v) for v in c]           # start at geometric center
    for _ in range(20):
        sq = np.zeros_like(d)
        for axis_vals, cv in zip(flat, ctr):
            sq = sq + (cv - axis_vals) ** 2
        wc = np.exp(-4.0 * np.log(2.0) * sq / CR ** 2)
        n = np.sum(wc * d)
        if n <= 0:
            break
        ctr = [float(np.sum(av * d * wc) / n) for av in flat]
    return np.array(ctr)


def _clamped_range(center, half, n):
    """Integer sample range ``[center-half, center+half]`` clamped to ``[0, n)``."""
    lo = int(max(0, np.floor(center - half)))
    hi = int(min(n - 1, np.ceil(center + half)))
    if hi <= lo:
        lo = 0
        hi = n - 1
    return np.arange(lo, hi + 1)


# ---------------------------------------------------------------------------
# 2D / 3D drift correction
# ---------------------------------------------------------------------------

def drift_correction_time_windows_2d(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    resolution_nm: float,
    rx: Optional[tuple] = None,
    ry: Optional[tuple] = None,
    T: Optional[float] = None,
    tid: Optional[np.ndarray] = None,
):
    """Estimate 2D drift by time-window auto-correlation (pyMINFLUX port).

    Returns ``(dx, dy, dxt, dyt, ti, T)`` — see module docstring for units.
    """
    if T is None and tid is None:
        raise ValueError("If the time window T is not given, tid must be provided.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    s = float(resolution_nm)

    if rx is None:
        rx = (float(x.min()), float(x.max()))
    if ry is None:
        ry = (float(y.min()), float(y.max()))

    if T is None:
        T = default_time_window(t, tid, rx, ry)
    T = float(T)

    Rt = [float(t[0]), float(t[-1])]
    Ns = int(np.floor((Rt[1] - Rt[0]) / T))
    if Ns <= 1:
        raise ValueError(
            "At least two time windows are needed; reduce the time window T "
            f"(current T={T:.1f}s gives {Ns} window(s) over a "
            f"{Rt[1] - Rt[0]:.1f}s acquisition).")

    CR = 10                              # centroid sampling radius (px)
    D = Ns                               # max window separation in the cross-corr
    w = np.linspace(1, 0.5, D)           # distance weights
    l = 0.1                              # roughness regularisation

    nx = int(np.ceil((rx[1] - rx[0]) / s))
    ny = int(np.ceil((ry[1] - ry[0]) / s))
    if nx < 4 or ny < 4:
        raise ValueError(
            "Render resolution too coarse for the data extent; reduce the "
            "resolution value (nm).")
    if nx * ny > _MAX_RENDER_CELLS:
        raise ValueError(
            f"Render grid too large ({nx}x{ny} px); increase the resolution "
            "value (nm) so the grid fits in memory.")
    c = np.round(np.array([ny, nx]) / 2)
    sigma_px = (3.0 * s) * _FWHM_TO_SIGMA / s   # fwhm = 3*s -> sigma in px

    # Render + FFT every window.
    h = [None] * Ns
    ti = np.zeros(Ns)
    for j in range(Ns):
        t0 = Rt[0] + j * T
        idx = (t >= t0) & (t < t0 + T)
        ti[j] = np.mean(t[idx]) if np.any(idx) else (t0 + T / 2)
        h[j] = fftn(_render_2d(x[idx], y[idx], s, rx, ry, nx, ny, sigma_px))

    dx = np.zeros((Ns, Ns))
    dy = np.zeros((Ns, Ns))
    dm = np.zeros((Ns, Ns), dtype=bool)
    dx0 = np.zeros(Ns - 1)
    dy0 = np.zeros(Ns - 1)

    for i in range(Ns - 1):
        hi = np.conj(h[i])
        for j in range(i + 1, min(Ns, i + D)):
            cc = ifftshift(np.real(ifftn(hi * h[j])))
            gy = _clamped_range(c[0], 2 * CR, ny)
            gx = _clamped_range(c[1], 2 * CR, nx)
            yc, xc = _refine_centroid_nd(cc, c, [gy, gx], CR)
            sh = np.array([-1.0, 1.0]) * (np.array([yc, xc]) - c)
            dy[i, j] = sh[0]
            dx[i, j] = sh[1]
            dm[i, j] = True
            if j == i + 1:
                dy0[i] = sh[0]
                dx0[i] = sh[1]

    a, b = np.nonzero(dm)
    dx = dx[dm]
    dy = dy[dm]
    sx0 = np.cumsum(dx0)
    sy0 = np.cumsum(dy0)

    options = {"disp": False, "maxiter": int(1e5)}
    res = minimize(lambda v: lse_distance(v, a, b, dx, w, l), sx0,
                   options=options, method="BFGS")
    sx = np.concatenate(([0], res.x))
    res = minimize(lambda v: lse_distance(v, a, b, dy, w, l), sy0,
                   options=options, method="BFGS")
    sy = np.concatenate(([0], res.x))

    sx = (sx - np.mean(sx)) * s          # center + scale to nm
    sy = (sy - np.mean(sy)) * s

    fx = interp1d(ti, sx, kind="slinear", fill_value="extrapolate")
    fy = interp1d(ti, sy, kind="slinear", fill_value="extrapolate")

    return fx(t), fy(t), fx(ti), fy(ti), ti, T


def drift_correction_time_windows_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    t: np.ndarray,
    resolution_nm: float,
    rx: Optional[tuple] = None,
    ry: Optional[tuple] = None,
    rz: Optional[tuple] = None,
    T: Optional[float] = None,
    tid: Optional[np.ndarray] = None,
):
    """Estimate 3D drift by time-window auto-correlation (pyMINFLUX port).

    Returns ``(dx, dy, dz, dxt, dyt, dzt, ti, T)`` — see module docstring.
    """
    if T is None and tid is None:
        raise ValueError("If the time window T is not given, tid must be provided.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    s = float(resolution_nm)

    if rx is None:
        rx = (float(x.min()), float(x.max()))
    if ry is None:
        ry = (float(y.min()), float(y.max()))
    if rz is None:
        rz = (float(z.min()), float(z.max()))

    if T is None:
        T = default_time_window(t, tid, rx, ry)
    T = float(T)

    Rt = [float(t[0]), float(t[-1])]
    Ns = int(np.floor((Rt[1] - Rt[0]) / T))
    if Ns <= 1:
        raise ValueError(
            "At least two time windows are needed; reduce the time window T "
            f"(current T={T:.1f}s gives {Ns} window(s) over a "
            f"{Rt[1] - Rt[0]:.1f}s acquisition).")

    CR = 8                               # centroid sampling radius (px), 3D
    D = Ns
    w = np.linspace(1, 0.2, D)
    l = 0.01

    nx = int(np.ceil((rx[1] - rx[0]) / s))
    ny = int(np.ceil((ry[1] - ry[0]) / s))
    nz = int(np.ceil((rz[1] - rz[0]) / s))
    if nx < 4 or ny < 4 or nz < 4:
        raise ValueError(
            "Render resolution too coarse for the data extent; reduce the "
            "resolution value (nm).")
    if nx * ny * nz > _MAX_RENDER_CELLS:
        raise ValueError(
            f"Render grid too large ({nx}x{ny}x{nz} voxels); increase the "
            "resolution value (nm) so the grid fits in memory.")
    c = np.round(np.array([nz, ny, nx]) / 2)
    sigma_px = (3.0 * s) * _FWHM_TO_SIGMA / s

    h = [None] * Ns
    ti = np.zeros(Ns)
    for j in range(Ns):
        t0 = Rt[0] + j * T
        idx = (t >= t0) & (t < t0 + T)
        ti[j] = np.mean(t[idx]) if np.any(idx) else (t0 + T / 2)
        h[j] = fftn(_render_3d(x[idx], y[idx], z[idx], s,
                               rx, ry, rz, nx, ny, nz, sigma_px))

    dx = np.zeros((Ns, Ns))
    dy = np.zeros((Ns, Ns))
    dz = np.zeros((Ns, Ns))
    dm = np.zeros((Ns, Ns), dtype=bool)
    dx0 = np.zeros(Ns - 1)
    dy0 = np.zeros(Ns - 1)
    dz0 = np.zeros(Ns - 1)

    for i in range(Ns - 1):
        hi = np.conj(h[i])
        for j in range(i + 1, min(Ns, i + D)):
            cc = ifftshift(np.real(ifftn(hi * h[j])))
            gz = _clamped_range(c[0], 2 * CR, nz)
            gy = _clamped_range(c[1], 2 * CR, ny)
            gx = _clamped_range(c[2], 2 * CR, nx)
            zc, yc, xc = _refine_centroid_nd(cc, c, [gz, gy, gx], CR)
            sh = np.array([1.0, -1.0, 1.0]) * (np.array([zc, yc, xc]) - c)
            dz[i, j] = sh[0]
            dy[i, j] = sh[1]
            dx[i, j] = sh[2]
            dm[i, j] = True
            if j == i + 1:
                dz0[i] = sh[0]
                dy0[i] = sh[1]
                dx0[i] = sh[2]

    a, b = np.nonzero(dm)
    dx = dx[dm]
    dy = dy[dm]
    dz = dz[dm]
    sx0 = np.cumsum(dx0)
    sy0 = np.cumsum(dy0)
    sz0 = np.cumsum(dz0)

    options = {"disp": False, "maxiter": int(1e5)}
    res = minimize(lambda v: lse_distance(v, a, b, dx, w, l), sx0,
                   options=options, method="BFGS")
    sx = np.concatenate(([0], res.x))
    res = minimize(lambda v: lse_distance(v, a, b, dy, w, l), sy0,
                   options=options, method="BFGS")
    sy = np.concatenate(([0], res.x))
    res = minimize(lambda v: lse_distance(v, a, b, dz, w, l), sz0,
                   options=options, method="BFGS")
    sz = np.concatenate(([0], res.x))

    sx = (sx - np.mean(sx)) * s
    sy = (sy - np.mean(sy)) * s
    sz = (sz - np.mean(sz)) * s

    fx = interp1d(ti, sx, kind="slinear", fill_value="extrapolate")
    fy = interp1d(ti, sy, kind="slinear", fill_value="extrapolate")
    fz = interp1d(ti, sz, kind="slinear", fill_value="extrapolate")

    return fx(t), fy(t), fz(t), fx(ti), fy(ti), fz(ti), ti, T


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def estimate_drift(
    x: np.ndarray,
    y: np.ndarray,
    z: Optional[np.ndarray],
    t: np.ndarray,
    tid: np.ndarray,
    *,
    resolution_nm: float,
    time_window_s: Optional[float] = None,
    dims: Optional[int] = None,
    rx: Optional[tuple] = None,
    ry: Optional[tuple] = None,
    rz: Optional[tuple] = None,
) -> DriftResult:
    """Estimate drift for a set of localizations, dispatching to 2D or 3D.

    ``dims`` forces the analysis dimensionality; if ``None`` it is 3 when a
    non-degenerate ``z`` is supplied, else 2. ``time_window_s=None`` uses the
    pyMINFLUX heuristic (needs ``tid``).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    tid = np.asarray(tid).ravel()

    if dims is None:
        z_ok = z is not None and np.ptp(np.asarray(z, dtype=float)) > 0
        dims = 3 if z_ok else 2

    if dims == 3:
        if z is None:
            raise ValueError("3D drift correction requires z coordinates.")
        z = np.asarray(z, dtype=float).ravel()
        dx, dy, dz, dxt, dyt, dzt, ti, T = drift_correction_time_windows_3d(
            x, y, z, t, resolution_nm, rx=rx, ry=ry, rz=rz,
            T=time_window_s, tid=tid)
        rms = (float(np.std(dx)), float(np.std(dy)), float(np.std(dz)))
        return DriftResult(
            dims=3, dx=dx, dy=dy, dz=dz, ti=ti, dxt=dxt, dyt=dyt, dzt=dzt,
            time_window_s=float(T), n_windows=len(ti), resolution_nm=float(resolution_nm),
            rx=_range_of(rx, x), ry=_range_of(ry, y), rz=_range_of(rz, z), rms=rms)

    dx, dy, dxt, dyt, ti, T = drift_correction_time_windows_2d(
        x, y, t, resolution_nm, rx=rx, ry=ry, T=time_window_s, tid=tid)
    rms = (float(np.std(dx)), float(np.std(dy)))
    return DriftResult(
        dims=2, dx=dx, dy=dy, dz=None, ti=ti, dxt=dxt, dyt=dyt, dzt=None,
        time_window_s=float(T), n_windows=len(ti), resolution_nm=float(resolution_nm),
        rx=_range_of(rx, x), ry=_range_of(ry, y), rz=None, rms=rms)


def _range_of(r, v) -> tuple:
    if r is not None:
        return (float(r[0]), float(r[1]))
    v = np.asarray(v, dtype=float)
    return (float(v.min()), float(v.max()))
