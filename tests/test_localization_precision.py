"""Tests for the localization-precision estimators (StdDev per trace + FRC).

The CRLB has its own file, tests/test_crlb.py."""

import numpy as np
import pytest

from minflux_viewer.analysis.localization_precision import (
    FRC_DEFAULT_REPEATS,
    FRC_THRESHOLD,
    MIN_LOCS_PER_TRACE,
    excess_precision,
    frc_resolution,
    stddev_per_trace,
)

NM = 1.0e-9


# ---------------------------------------------------------------------------
# StdDev per trace — Schmidt et al. definition
# ---------------------------------------------------------------------------

def test_stddev_definition_hand_checked():
    """σ_r, combined values match a hand-computed two-trace example."""
    loc, tid = [], []
    # Trace 1: x = 0,2,4,6,8 nm (σ_x = sqrt(10)); y = z = 0
    for v in (0, 2, 4, 6, 8):
        loc.append([v * NM, 0.0, 0.0]); tid.append(1)
    # Trace 2: y = 0,3,6,9,12 nm (σ_y = sqrt(22.5)); x = z = 0
    for v in (0, 3, 6, 9, 12):
        loc.append([0.0, v * NM, 0.0]); tid.append(2)
    res = stddev_per_trace(np.array(loc), np.array(tid))

    assert res["n_traces_used"] == 2
    # σ_r,A = sqrt(10/2) = sqrt(5);  σ_r,B = sqrt(22.5/2) = sqrt(11.25)
    assert np.allclose(res["per_trace_sigma_r"], [np.sqrt(5.0), np.sqrt(11.25)], atol=1e-6)
    # combined_r = mean(σ_r / sqrt(n)) = mean(sqrt5/sqrt5, sqrt11.25/sqrt5) = (1 + 1.5)/2
    assert res["combined_sigma_r"] == pytest.approx(1.25, abs=1e-6)
    # z is flat → all z precision zero
    assert res["combined_sigma_xyz"][2] == pytest.approx(0.0, abs=1e-12)
    assert res["median_sigma_r"] == pytest.approx(np.median([np.sqrt(5.0), np.sqrt(11.25)]))


def test_stddev_min_locs_is_five_and_drops_short_traces():
    assert MIN_LOCS_PER_TRACE == 5
    loc, tid = [], []
    for v in (0, 1, 2, 3, 4):                       # 5 locs → kept
        loc.append([v * NM, 0.0, 0.0]); tid.append(1)
    for v in (0, 1, 2, 3):                          # 4 locs → dropped
        loc.append([v * NM, 0.0, 0.0]); tid.append(2)
    res = stddev_per_trace(np.array(loc), np.array(tid))
    assert res["n_traces_used"] == 1
    assert res["n_traces_total"] == 2


def test_stddev_empty_input():
    res = stddev_per_trace(np.empty((0, 3)), np.empty(0))
    assert res["n_traces_used"] == 0
    assert res["per_trace_sigma_r"].size == 0


# ---------------------------------------------------------------------------
# FRC — Fourier Ring Correlation resolution
# ---------------------------------------------------------------------------

def _molecules(rng, sites, k, sigma_nm):
    """k blurred localizations per site; returns x, y (nm) and trace ids."""
    xs, ys, tids = [], [], []
    for i, (sx, sy) in enumerate(sites):
        xs.append(rng.normal(sx, sigma_nm, k))
        ys.append(rng.normal(sy, sigma_nm, k))
        tids.append(np.full(k, i))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(tids)


def test_frc_curve_shape_and_resolution():
    rng = np.random.default_rng(0)
    sites = rng.uniform(0, 1000, size=(400, 2))
    x, y, _ = _molecules(rng, sites, k=6, sigma_nm=5.0)
    r = frc_resolution(x, y, seed=1)

    assert r.freq_inv_nm.size > 2
    assert np.isfinite(r.resolution_nm) and r.resolution_nm > 0
    assert r.threshold == pytest.approx(FRC_THRESHOLD)
    # FRC starts high near DC and decays.
    assert r.frc_mean[1] > 0.5
    assert r.frc_mean[-1] < r.frc_mean[1]
    assert r.n_repeats >= 1
    # tail cut at 0.8·q_max (reference behaviour)
    assert r.freq_inv_nm[-1] <= 0.8 / r.pixel_size_nm + 1e-9


def test_frc_resolution_worsens_with_blur():
    sites = np.random.default_rng(2).uniform(0, 1000, size=(400, 2))
    xs, ys, _ = _molecules(np.random.default_rng(10), sites, k=6, sigma_nm=3.0)
    xb, yb, _ = _molecules(np.random.default_rng(10), sites, k=6, sigma_nm=15.0)
    r_sharp = frc_resolution(xs, ys, seed=3)
    r_blur = frc_resolution(xb, yb, seed=3)
    assert r_sharp.resolution_nm < r_blur.resolution_nm


def test_frc_per_localization_split():
    rng = np.random.default_rng(4)
    sites = rng.uniform(0, 1000, size=(500, 2))
    x, y, _ = _molecules(rng, sites, k=5, sigma_nm=6.0)
    r = frc_resolution(x, y, seed=5)
    assert r.mode == "localization"
    assert r.n_repeats == FRC_DEFAULT_REPEATS      # default repeat count
    assert r.n_points == r.n_locs
    assert np.isfinite(r.resolution_nm) and 0 < r.resolution_nm < 1000


def test_frc_combined_mode():
    rng = np.random.default_rng(11)
    sites = rng.uniform(0, 1000, size=(300, 2))
    x, y, tid = _molecules(rng, sites, k=8, sigma_nm=5.0)
    r = frc_resolution(x, y, tid, mode="combined", seed=3)
    assert r.mode == "combined"
    assert r.n_points == 300                      # one mean position per trace
    assert r.n_locs == 300 * 8
    assert np.isfinite(r.resolution_nm) and r.resolution_nm > 0


def test_frc_combined_without_tid_falls_back():
    rng = np.random.default_rng(12)
    sites = rng.uniform(0, 800, size=(300, 2))
    x, y, _ = _molecules(rng, sites, k=6, sigma_nm=5.0)
    r = frc_resolution(x, y, None, mode="combined", seed=3)
    assert r.mode == "localization"               # no tid → fall back
    assert r.n_points == r.n_locs


def test_frc_pixel_override_covers_extent():
    rng = np.random.default_rng(6)
    sites = rng.uniform(0, 500, size=(300, 2))
    x, y, _ = _molecules(rng, sites, k=6, sigma_nm=4.0)
    r = frc_resolution(x, y, pixel_size_nm=2.0, seed=7)
    assert r.pixel_size_nm == pytest.approx(2.0)
    extent = max(x.max() - x.min(), y.max() - y.min())
    # the square grid always covers the data extent
    assert r.image_px * r.pixel_size_nm >= extent
    # freq axis step = 1 / (n_px * pixel)
    assert r.freq_inv_nm[1] == pytest.approx(1.0 / (r.image_px * r.pixel_size_nm), rel=1e-6)


def test_frc_default_grid_size_honors_image_px():
    rng = np.random.default_rng(9)
    sites = rng.uniform(0, 300, size=(300, 2))       # extent < image_px → floor applies
    x, y, _ = _molecules(rng, sites, k=6, sigma_nm=4.0)
    r = frc_resolution(x, y, image_px=512, seed=7)   # pixel auto → grid = image_px floor
    assert r.image_px == 512


def test_frc_determinism():
    rng = np.random.default_rng(8)
    sites = rng.uniform(0, 800, size=(350, 2))
    x, y, _ = _molecules(rng, sites, k=5, sigma_nm=5.0)
    r1 = frc_resolution(x, y, seed=42)
    r2 = frc_resolution(x, y, seed=42)
    assert r1.resolution_nm == pytest.approx(r2.resolution_nm)
    assert np.allclose(r1.frc_mean, r2.frc_mean)


def test_frc_too_few_localizations():
    r = frc_resolution(np.arange(5.0), np.arange(5.0))
    assert not np.isfinite(r.resolution_nm)
    assert "few" in r.status.lower()


# ---------------------------------------------------------------------------
# Legacy quadrature arithmetic — not a physical precision-budget inference
# ---------------------------------------------------------------------------

def test_excess_precision_decomposition():
    # σ_fl = sqrt(STD² − σ_CRB²): 10² = 8² + 6²
    assert excess_precision(10.0, 6.0) == pytest.approx(8.0)
    # round-trips: STD = sqrt(σ_CRB² + σ_fl²)
    crb, fl = 6.0, 8.0
    assert np.hypot(crb, excess_precision(10.0, crb)) == pytest.approx(10.0)


def test_excess_precision_below_bound_is_unresolved():
    assert np.isnan(excess_precision(6.0, 10.0))
    assert excess_precision(5.0, 5.0) == 0.0


def test_excess_precision_invalid_inputs_are_nan():
    for bad in (np.nan, np.inf):
        assert not np.isfinite(excess_precision(bad, 6.0))
        assert not np.isfinite(excess_precision(10.0, bad))
    # non-positive measured spread → NaN (nothing to decompose)
    assert not np.isfinite(excess_precision(0.0, 6.0))
    assert not np.isfinite(excess_precision(-1.0, 6.0))
