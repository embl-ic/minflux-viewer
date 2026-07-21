"""Tests for analysis.drift_correction (pure NumPy/SciPy core, no Qt)."""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.analysis import drift_correction as dc


def _synthetic_drifted_2d(seed=0, n_struct=400, span=2000.0, total_time=4000.0,
                          drift_amp=15.0):
    """Build a drifting 2D point cloud with a known smooth drift trajectory.

    A fixed set of underlying feature points is imaged over time; a smooth
    (linear + sinusoidal) drift is *added* to the true positions, and each
    localization gets a small measurement jitter. Returns the observed x/y, the
    time t, the trace ids, and the ground-truth drift sampled at each loc.
    """
    rng = np.random.default_rng(seed)
    # Underlying features (the "structure"), static in the sample frame.
    fx = rng.uniform(-span / 2, span / 2, n_struct)
    fy = rng.uniform(-span / 2, span / 2, n_struct)

    # Time span: revisit each feature many times across the acquisition.
    reps = 60
    t = np.linspace(0.0, total_time, n_struct * reps)
    order = rng.permutation(t.size)
    t = np.sort(t)
    feat = np.tile(np.arange(n_struct), reps)
    feat = feat[order]                       # random feature per time sample

    # Ground-truth drift trajectory d(t): smooth, mean removed later.
    dxt = drift_amp * (t / total_time) + 0.4 * drift_amp * np.sin(2 * np.pi * t / total_time)
    dyt = -0.7 * drift_amp * (t / total_time) + 0.3 * drift_amp * np.cos(2 * np.pi * t / total_time)

    jitter = 1.0
    x = fx[feat] + dxt + rng.normal(0, jitter, t.size)
    y = fy[feat] + dyt + rng.normal(0, jitter, t.size)
    tid = feat.astype(np.int64)             # each feature ~ a trace
    return x, y, t, tid, dxt, dyt


def test_default_time_window_bounds():
    t = np.linspace(0, 4000, 10000)
    tid = np.repeat(np.arange(500), 20)
    T = dc.default_time_window(t, tid, (-1000, 1000), (-1000, 1000))
    assert T >= dc._MIN_AUTO_WINDOW_S               # at least 10 minutes
    assert T <= (t[-1] - t[0]) / 2 + 1e-6           # at least two windows


def test_2d_recovers_known_drift_sign_and_shape():
    x, y, t, tid, dxt, dyt = _synthetic_drifted_2d()

    res = dc.estimate_drift(x, y, None, t, tid,
                            resolution_nm=4.0, time_window_s=400.0, dims=2)

    assert res.dims == 2
    assert res.dx.shape == x.shape and res.dy.shape == y.shape
    assert res.dz is None
    assert res.n_windows >= 2

    # The estimated drift is mean-subtracted; so is the ground truth for a fair
    # comparison. Correlation must be strongly positive (same sign & trend).
    gx = dxt - dxt.mean()
    gy = dyt - dyt.mean()
    ex = res.dx - res.dx.mean()
    ey = res.dy - res.dy.mean()
    cx = np.corrcoef(gx, ex)[0, 1]
    cy = np.corrcoef(gy, ey)[0, 1]
    assert cx > 0.9, f"x drift not recovered (corr={cx:.3f})"
    assert cy > 0.9, f"y drift not recovered (corr={cy:.3f})"


def test_2d_correction_reduces_spread():
    """Subtracting the estimated drift must tighten the localization cloud."""
    x, y, t, tid, dxt, dyt = _synthetic_drifted_2d()
    res = dc.estimate_drift(x, y, None, t, tid,
                            resolution_nm=4.0, time_window_s=400.0, dims=2)
    # Residual drift (ground truth minus estimate) should be much smaller than
    # the original drift amplitude.
    resid_x = (dxt - dxt.mean()) - (res.dx - res.dx.mean())
    assert np.std(resid_x) < 0.5 * np.std(dxt - dxt.mean())


def test_estimate_drift_requires_two_windows():
    x, y, t, tid, _, _ = _synthetic_drifted_2d(total_time=1000.0)
    with pytest.raises(ValueError):
        # A window longer than the whole acquisition -> < 2 windows.
        dc.estimate_drift(x, y, None, t, tid,
                          resolution_nm=4.0, time_window_s=5000.0, dims=2)


def test_render_2d_orientation_matches_shift_direction():
    """A +x shift of all points must move the render's centroid toward +x pixel."""
    rng = np.random.default_rng(1)
    x = rng.uniform(-500, 500, 5000)
    y = rng.uniform(-500, 500, 5000)
    rx, ry = (-600, 600), (-600, 600)
    s = 4.0
    nx = int(np.ceil((rx[1] - rx[0]) / s))
    ny = int(np.ceil((ry[1] - ry[0]) / s))
    sigma = (3 * s) * dc._FWHM_TO_SIGMA / s
    h0 = dc._render_2d(x, y, s, rx, ry, nx, ny, sigma)
    h1 = dc._render_2d(x + 40.0, y, s, rx, ry, nx, ny, sigma)  # +40 nm in x
    # Column centroid (x pixel) should increase.
    cols = np.arange(nx)
    cx0 = (h0.sum(0) * cols).sum() / h0.sum()
    cx1 = (h1.sum(0) * cols).sum() / h1.sum()
    assert cx1 > cx0
