"""
minflux_viewer.analysis.drift_correction
========================================
Fiducial-free drift correction by **time-window cross-correlation** (Ostersehlt
et al., *DNA-PAINT MINFLUX nanoscopy*, Nat. Methods 19:1072, 2022).

The acquisition is cut into ``Ns`` consecutive time windows of length ``T``; each
window's localizations are rendered as an image (a Gaussian of FWHM ``3·s`` per
localization on an ``s``-nm grid); every pair of windows is cross-correlated and
the displacement between them is read off the correlation peak by an iterative
Gaussian-weighted centroid; and the drift trajectory is the one whose pairwise
differences best match all those displacements, with a roughness penalty —
a least-squares problem solved per axis. The trajectory, known at the window
centres, is interpolated onto every localization time with a cubic spline
(linear beyond the first and last centre) and subtracted.

The method needs no fiducials and sees *any* drift the localizations share,
including what fiducial beads do not; its time resolution is the window length
(minutes, not seconds) and it assumes the imaged structure does not move.

Re-derived from the reference MATLAB implementation published with that paper
(``drift_correction_time_windows_2D.m`` / ``_3D.m``), keeping its parameters: in
2-D a centroid radius of 10 px, pair weights falling linearly from 1 (adjacent
windows) to 0.5 (the most distant pair) and a roughness weight of 0.1; in 3-D
8 px, 1 → 0.2 and 0.01. One implementation serves both dimensionalities. Two
deliberate differences, neither of which changes the estimate beyond pixel
noise: the per-localization Gaussian is rendered as a sub-pixel (multilinear)
histogram blurred with the same Gaussian, which is O(pixels) instead of
O(localizations × kernel), and localizations near the border are kept instead
of dropped.

The time window ``T`` is the operator's choice in the reference code. Our
automatic proposal (:func:`auto_time_window`) is the largest number of windows
— at most :data:`AUTO_MAX_WINDOWS` — for which 80 % of the windows hold at least
:data:`AUTO_MIN_LOCS_PER_WINDOW` localizations: the cross-correlation needs the
same structures to appear in both windows of a pair, so the window must collect
enough of them, while every additional window adds time resolution.

Reference implementation: https://github.com/jkfindeisen/dna_paint_minflux_nanoscopy_2021
(also Zenodo record 6563100), distributed under the MIT license:

    Copyright (c) 2022 Lynn M. Ostersehlt, Jan Keller-Findeisen

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Pure NumPy/SciPy, Qt-free.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

#: Render cell budget (pixels or voxels) — a guard against an accidental fine
#: grid over a large field of view.
_MAX_RENDER_CELLS = 64_000_000
#: Every window's spectrum is kept for the pairwise correlations.
_MAX_SPECTRA_BYTES = 3 * 2 ** 30


@dataclass(frozen=True)
class WindowCorrelationParameters:
    """The reference implementation's constants for one dimensionality."""
    centroid_radius_px: float   # FWHM of the centroid weight; search half-width 2·this
    weight_far: float           # pair weight of the most distant pair (adjacent = 1)
    roughness: float            # weight of Σ (Δ trajectory)² in the cost

    @property
    def search_half_px(self) -> int:
        return int(round(2 * self.centroid_radius_px))


PARAMETERS = {
    2: WindowCorrelationParameters(centroid_radius_px=10.0, weight_far=0.5, roughness=0.1),
    3: WindowCorrelationParameters(centroid_radius_px=8.0, weight_far=0.2, roughness=0.01),
}

#: Automatic time window: at most this many windows …
AUTO_MAX_WINDOWS = 20
#: … provided 80 % of them hold at least this many localizations.
AUTO_MIN_LOCS_PER_WINDOW = {2: 1000, 3: 3000}
_AUTO_COVERAGE_PERCENTILE = 20.0


@dataclass
class DriftResult:
    """A drift estimate. ``dx/dy/dz`` are per localization (nm) and are to be
    *subtracted* from the coordinates; ``ti`` are the window centres (s) and
    ``dxt/dyt/dzt`` the trajectory there; ``curve_t``/``curve`` a dense sampling
    of the interpolant for plotting."""
    dims: int
    dx: np.ndarray
    dy: np.ndarray
    dz: np.ndarray | None
    ti: np.ndarray
    dxt: np.ndarray
    dyt: np.ndarray
    dzt: np.ndarray | None
    time_window_s: float
    n_windows: int
    resolution_nm: float
    rx: tuple
    ry: tuple
    rz: tuple | None
    rms: tuple
    method: str = "time-window cross-correlation"
    auto_window: bool = False
    locs_per_window: np.ndarray = field(default_factory=lambda: np.empty(0, int))
    curve_t: np.ndarray = field(default_factory=lambda: np.empty(0))
    curve: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))


class DriftCancelled(Exception):
    """Raised at a progress checkpoint once the caller asked to stop."""


# ---------------------------------------------------------------------------
# Time windows
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AutoWindow:
    time_window_s: float
    n_windows: int
    locs_per_window: np.ndarray
    sparse: bool          # True: even two windows fall short of the target


def _window_counts(t: np.ndarray, t0: float, T: float, n: int) -> np.ndarray:
    k = np.floor((t - t0) / T).astype(np.int64)
    k = k[(k >= 0) & (k < n)]
    return np.bincount(k, minlength=n)


def auto_time_window(t: np.ndarray, dims: int = 2) -> AutoWindow:
    """Our proposal for ``T`` (see the module docstring)."""
    t = np.asarray(t, dtype=float)
    t = t[np.isfinite(t)]
    if t.size < 2 or np.ptp(t) <= 0:
        raise ValueError("The localizations span no time range.")
    t0, span = float(t.min()), float(np.ptp(t))
    need = AUTO_MIN_LOCS_PER_WINDOW.get(int(dims), AUTO_MIN_LOCS_PER_WINDOW[2])
    for n in range(AUTO_MAX_WINDOWS, 1, -1):
        # A hair under span/n so floor(span/T) is n, not n − 1, in floating point.
        T = span / n * (1.0 - 1e-9)
        counts = _window_counts(t, t0, T, n)
        if np.percentile(counts, _AUTO_COVERAGE_PERCENTILE) >= need:
            return AutoWindow(T, n, counts, sparse=False)
    T = span / 2 * (1.0 - 1e-9)
    return AutoWindow(T, 2, _window_counts(t, t0, T, 2), sparse=True)


def default_time_window(t, tid=None, rx=None, ry=None, *, dims: int = 2) -> float:
    """``T`` of :func:`auto_time_window` (the extra arguments are accepted for
    backward compatibility and unused)."""
    return auto_time_window(np.sort(np.asarray(t, dtype=float)), dims).time_window_s


# ---------------------------------------------------------------------------
# Rendering and correlation
# ---------------------------------------------------------------------------

def _render(points: np.ndarray, s: float, lows, shape, sigma_px: float) -> np.ndarray:
    """Sub-pixel histogram of ``points`` (n × d, nm) on an ``s``-nm grid starting
    at ``lows``, blurred by a Gaussian of ``sigma_px`` pixels. Axis k of the
    image is coordinate k (x first), as in the reference implementation."""
    from scipy.ndimage import gaussian_filter

    d = len(shape)
    size = int(np.prod(shape))
    if points.shape[0]:
        pos = (points - np.asarray(lows, dtype=float)) / s - 0.5   # pixel-centre coords
        base = np.floor(pos).astype(np.int64)
        frac = pos - base
        flats, weights = [], []
        for corner in range(1 << d):
            offs = np.array([(corner >> k) & 1 for k in range(d)])
            idx = base + offs
            w = np.prod(np.where(offs == 1, frac, 1.0 - frac), axis=1)
            ok = np.all((idx >= 0) & (idx < np.asarray(shape)), axis=1)
            if ok.any():
                flats.append(np.ravel_multi_index(tuple(idx[ok].T), shape))
                weights.append(w[ok])
        if flats:          # one accumulation, not one full-size array per corner
            img = np.bincount(np.concatenate(flats), weights=np.concatenate(weights),
                              minlength=size).astype(np.float32).reshape(shape)
        else:
            img = np.zeros(shape, dtype=np.float32)
    else:
        img = np.zeros(shape, dtype=np.float32)
    if sigma_px > 0:
        img = gaussian_filter(img, sigma_px, mode="constant").astype(np.float32)
    return img


def _render_2d(x, y, s, rx, ry, nx, ny, sigma_px) -> np.ndarray:
    """2-D convenience wrapper of :func:`_render` (image axis 0 is x)."""
    pts = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    return _render(pts, s, (rx[0], ry[0]), (nx, ny), sigma_px)


def _centroid_shift(corr: np.ndarray, params: WindowCorrelationParameters) -> np.ndarray:
    """Displacement (px) at the cross-correlation peak near zero shift.

    ``corr`` is fftshift-ed, so zero shift sits at ``n // 2`` on each axis. The
    search box is ±2·CR around it; inside, an iterative Gaussian-weighted
    centroid (FWHM = CR, 20 iterations, starting at zero shift) locates the peak
    — the reference implementation's estimator.
    """
    d = corr.ndim
    centre = np.array([n // 2 for n in corr.shape], dtype=float)
    half = params.search_half_px
    axes = [np.arange(max(0, int(c) - half), min(n, int(c) + half + 1))
            for c, n in zip(centre, corr.shape)]
    grids = np.meshgrid(*axes, indexing="ij")
    vals = corr[np.ix_(*axes)].astype(np.float64).ravel()
    coords = [g.ravel().astype(np.float64) for g in grids]
    vals = vals - vals.min()
    total = vals.sum()
    if total <= 0:
        return np.zeros(d)
    vals /= total
    c = centre.copy()
    cr2 = params.centroid_radius_px ** 2
    for _ in range(20):
        r2 = sum((c[k] - coords[k]) ** 2 for k in range(d))
        wc = np.exp(-4.0 * np.log(2.0) * r2 / cr2) * vals
        norm = wc.sum()
        if norm <= 0:
            break
        c = np.array([np.sum(coords[k] * wc) / norm for k in range(d)])
    return c - centre


def _trajectory(n_windows: int, a: np.ndarray, b: np.ndarray, shifts: np.ndarray,
                params: WindowCorrelationParameters) -> np.ndarray:
    """Per-window positions (px, first window = 0) best matching the measured
    pairwise ``shifts`` (window ``b`` relative to window ``a``)."""
    from scipy.optimize import minimize

    # Pair weight: 1 for adjacent windows, falling linearly to weight_far for
    # the most distant pair (MATLAB's w(b − a), 1-based).
    w = np.linspace(1.0, params.weight_far, n_windows)[(b - a) - 1]
    lam = params.roughness

    def cost(free):
        x = np.concatenate(([0.0], free))
        resid = x[b] - x[a] - shifts
        return np.mean(w * resid ** 2) + lam * np.sum(np.diff(x) ** 2)

    adjacent = (b - a) == 1
    order = np.argsort(a[adjacent])
    start = np.cumsum(shifts[adjacent][order])
    if start.size != n_windows - 1:
        start = np.zeros(n_windows - 1)
    res = minimize(cost, start, method="BFGS", options={"maxiter": 100_000})
    return np.concatenate(([0.0], res.x))


def _spline(ti: np.ndarray, values: np.ndarray):
    """Cubic (not-a-knot) spline through ``(ti, values)``, linear outside — the
    reference's ``griddedInterpolant(ti, v, 'spline', 'linear')``."""
    from scipy.interpolate import CubicSpline

    if ti.size == 2:
        slope = (values[1] - values[0]) / (ti[1] - ti[0])
        return lambda t: values[0] + slope * (np.asarray(t, float) - ti[0])
    cs = CubicSpline(ti, values, bc_type="not-a-knot", extrapolate=True)
    d0, d1 = float(cs(ti[0], 1)), float(cs(ti[-1], 1))

    def f(t):
        t = np.asarray(t, dtype=float)
        out = cs(t)
        lo, hi = t < ti[0], t > ti[-1]
        out = np.where(lo, values[0] + d0 * (t - ti[0]), out)
        return np.where(hi, values[-1] + d1 * (t - ti[-1]), out)
    return f


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def estimate_drift(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None,
    t: np.ndarray,
    tid: np.ndarray | None = None,
    *,
    resolution_nm: float,
    time_window_s: float | None = None,
    dims: int | None = None,
    rx: tuple | None = None,
    ry: tuple | None = None,
    rz: tuple | None = None,
    progress: Callable[[float], None] | None = None,
) -> DriftResult:
    """Estimate the drift of a set of localizations (nm, seconds).

    ``dims`` forces 2 or 3; ``None`` picks 3 when a non-degenerate ``z`` is
    given. ``time_window_s=None`` uses :func:`auto_time_window`. ``progress``
    receives a fraction in [0, 1]; raising :class:`DriftCancelled` from it
    stops the estimate. ``tid`` is accepted for backward compatibility.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    zarr = None if z is None else np.asarray(z, dtype=float).ravel()
    if dims is None:
        fz = zarr[np.isfinite(zarr)] if zarr is not None else np.empty(0)
        dims = 3 if (fz.size and np.ptp(fz) > 0) else 2
    dims = int(dims)
    if dims == 3 and zarr is None:
        raise ValueError("3-D drift correction requires z coordinates.")
    params = PARAMETERS[dims]
    s = float(resolution_nm)
    if not s > 0:
        raise ValueError("The render resolution must be positive.")

    cols = [x, y] + ([zarr] if dims == 3 else [])
    coords = np.column_stack(cols)
    ok = np.all(np.isfinite(coords), axis=1) & np.isfinite(t)
    if ok.sum() < 2:
        raise ValueError("Too few localizations with finite coordinates and time.")
    pts, tt = coords[ok], t[ok]

    auto = time_window_s is None
    if auto:
        aw = auto_time_window(tt, dims)
        T = aw.time_window_s
    else:
        T = float(time_window_s)
    t0, span = float(tt.min()), float(np.ptp(tt))
    n_win = int(np.floor(span / T)) if T > 0 else 0
    if n_win < 2:
        raise ValueError(
            f"A time window of {T:.0f} s leaves fewer than two windows in the "
            f"{span:.0f} s acquisition; shorten it.")

    ranges = [rx, ry] + ([rz] if dims == 3 else [])
    ranges = [(float(r[0]), float(r[1])) if r is not None
              else (float(pts[:, k].min()), float(pts[:, k].max()))
              for k, r in enumerate(ranges)]
    shape = tuple(max(1, int(np.ceil((hi - lo) / s))) for lo, hi in ranges)
    cells = int(np.prod(shape))
    if cells > _MAX_RENDER_CELLS:
        raise ValueError(
            f"The render grid would hold {cells:,} cells (over {_MAX_RENDER_CELLS:,}); "
            "use a coarser resolution.")
    spectra_bytes = n_win * cells * 4        # one half-spectrum (complex64) per window
    if spectra_bytes > _MAX_SPECTRA_BYTES:
        raise ValueError(
            f"{n_win} windows of a {cells:,}-cell grid need "
            f"{spectra_bytes / 2**30:.1f} GiB of spectra (limit "
            f"{_MAX_SPECTRA_BYTES / 2**30:.0f} GiB); use a coarser resolution or a "
            "longer time window.")
    lows = [lo for lo, _ in ranges]
    sigma_px = 3.0 * _FWHM_TO_SIGMA        # FWHM = 3 pixels, as in the reference

    from scipy import fft as sfft

    def _tick(frac):
        if progress is not None:
            progress(float(frac))

    win = np.floor((tt - t0) / T).astype(np.int64)
    ti = np.zeros(n_win)
    counts = np.zeros(n_win, dtype=np.int64)
    spectra = []
    for j in range(n_win):
        sel = win == j
        counts[j] = int(sel.sum())
        ti[j] = float(tt[sel].mean()) if counts[j] else t0 + (j + 0.5) * T
        img = _render(pts[sel], s, lows, shape, sigma_px)
        spectra.append(sfft.rfftn(img, workers=-1).astype(np.complex64))
        _tick(0.25 * (j + 1) / n_win)

    pairs = [(i, j) for i in range(n_win - 1) for j in range(i + 1, n_win)]
    shifts = np.zeros((len(pairs), dims))
    for p, (i, j) in enumerate(pairs):
        corr = sfft.irfftn(np.conj(spectra[i]) * spectra[j], s=shape, workers=-1)
        shifts[p] = _centroid_shift(np.fft.fftshift(corr), params)
        _tick(0.25 + 0.7 * (p + 1) / len(pairs))
    del spectra
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])

    traj = np.zeros((n_win, dims))
    for k in range(dims):
        traj[:, k] = _trajectory(n_win, a, b, shifts[:, k], params)
    traj = (traj - traj.mean(axis=0)) * s          # mean removed, in nm

    funcs = [_spline(ti, traj[:, k]) for k in range(dims)]
    per_loc = [f(t) for f in funcs]                 # every localization, even unused
    curve_t = np.arange(t0, t0 + span + T / 20.0, T / 10.0)
    curve = np.column_stack([f(curve_t) for f in funcs])
    _tick(1.0)

    rng = tuple(ranges)
    return DriftResult(
        dims=dims,
        dx=per_loc[0], dy=per_loc[1], dz=per_loc[2] if dims == 3 else None,
        ti=ti, dxt=traj[:, 0], dyt=traj[:, 1], dzt=traj[:, 2] if dims == 3 else None,
        time_window_s=T, n_windows=n_win, resolution_nm=s,
        rx=rng[0], ry=rng[1], rz=rng[2] if dims == 3 else None,
        rms=tuple(float(np.std(v[np.isfinite(v)])) for v in per_loc),
        auto_window=auto, locs_per_window=counts, curve_t=curve_t, curve=curve,
    )
