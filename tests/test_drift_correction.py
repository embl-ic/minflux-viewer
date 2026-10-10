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


def test_auto_time_window_is_the_most_windows_with_enough_localizations():
    t = np.linspace(0, 4000, 10000)                 # 10 000 localizations, uniform
    aw = dc.auto_time_window(t, dims=2)
    # 1000 per window is the 2-D floor → 10 windows of 400 s
    assert aw.n_windows == 10 and not aw.sparse
    assert aw.time_window_s == pytest.approx(400.0, rel=1e-6)
    assert np.percentile(aw.locs_per_window, 20) >= dc.AUTO_MIN_LOCS_PER_WINDOW[2]
    # dense data stop at the window cap; 3-D asks for more per window
    assert dc.auto_time_window(np.linspace(0, 4000, 10 ** 6), 2).n_windows == dc.AUTO_MAX_WINDOWS
    assert dc.auto_time_window(t, dims=3).n_windows == 3
    # too sparse for even two windows of the floor: two windows, flagged
    sparse = dc.auto_time_window(np.linspace(0, 100, 500), 2)
    assert sparse.n_windows == 2 and sparse.sparse
    assert dc.default_time_window(t) == pytest.approx(aw.time_window_s)


def test_reference_parameters_are_kept():
    """Centroid radius, pair weights and roughness of the reference MATLAB code."""
    p2, p3 = dc.PARAMETERS[2], dc.PARAMETERS[3]
    assert (p2.centroid_radius_px, p2.weight_far, p2.roughness) == (10.0, 0.5, 0.1)
    assert (p3.centroid_radius_px, p3.weight_far, p3.roughness) == (8.0, 0.2, 0.01)


def test_trajectory_weights_adjacent_pairs_by_one():
    """MATLAB indexes w(b − a) 1-based: adjacent windows weigh 1, the most distant
    pair weight_far. A consistent shift set must come back exactly."""
    truth = np.array([0.0, 2.0, 5.0, 4.0])
    a, b = np.triu_indices(4, 1)
    shifts = truth[b] - truth[a]
    params = dc.WindowCorrelationParameters(10.0, 0.5, 0.0)       # no roughness
    est = dc._trajectory(4, a, b, shifts, params)
    assert np.allclose(est, truth, atol=1e-5)


def test_spline_extrapolates_linearly():
    ti = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    f = dc._spline(ti, ti ** 2)
    assert f(2.0) == pytest.approx(4.0)
    # slope at the last centre carried on beyond it
    slope = float(f(4.0) - f(3.999)) / 0.001
    assert f(6.0) == pytest.approx(16.0 + 2.0 * slope, rel=1e-3)


def test_3d_recovers_known_drift():
    rng = np.random.default_rng(3)
    n_struct, reps, T_tot = 400, 50, 4000.0
    f = np.column_stack([rng.uniform(-800, 800, n_struct), rng.uniform(-800, 800, n_struct),
                         rng.uniform(-150, 150, n_struct)])
    t = np.sort(rng.uniform(0, T_tot, n_struct * reps))
    feat = rng.integers(0, n_struct, t.size)
    d = np.column_stack([10 * t / T_tot, -8 * t / T_tot, 6 * np.sin(np.pi * t / T_tot)])
    p = f[feat] + d + rng.normal(0, 1.0, (t.size, 3))
    res = dc.estimate_drift(p[:, 0], p[:, 1], p[:, 2], t, resolution_nm=5.0,
                            time_window_s=500.0, dims=3)
    for k, est in enumerate((res.dx, res.dy, res.dz)):
        truth = d[:, k] - d[:, k].mean()
        assert np.sqrt(np.mean((est - est.mean() - truth) ** 2)) < 0.25 * truth.std()


def test_progress_reaches_one_and_cancel_stops():
    x, y, t, tid, _, _ = _synthetic_drifted_2d(n_struct=80)
    seen = []
    dc.estimate_drift(x, y, None, t, resolution_nm=4.0, time_window_s=400.0,
                      progress=seen.append)
    assert seen and seen[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(seen, seen[1:]))

    def stop(_frac):
        raise dc.DriftCancelled()
    with pytest.raises(dc.DriftCancelled):
        dc.estimate_drift(x, y, None, t, resolution_nm=4.0, time_window_s=400.0,
                          progress=stop)


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


def test_nan_z_values_are_ignored_when_inferring_2d_drift():
    x, y, t, tid, _, _ = _synthetic_drifted_2d(n_struct=40)
    z = np.zeros_like(x)
    z[::3] = np.nan

    res = dc.estimate_drift(
        x, y, z, t, tid, resolution_nm=4.0, time_window_s=400.0
    )

    assert res.dims == 2
    assert res.dz is None


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
    """Image axis 0 is x (the reference's convention): a +x shift of all points
    moves the render's centroid along axis 0."""
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
    rows = np.arange(nx)
    cx0 = (h0.sum(1) * rows).sum() / h0.sum()
    cx1 = (h1.sum(1) * rows).sum() / h1.sum()
    assert cx1 - cx0 == pytest.approx(40.0 / s, abs=0.2)
