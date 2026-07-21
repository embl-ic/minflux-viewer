"""Tests for the localization-precision estimators (StdDev per trace + FRC)."""

import numpy as np
import pytest

from minflux_viewer.analysis.localization_precision import (
    FRC_DEFAULT_REPEATS,
    FRC_THRESHOLD,
    MIN_LOCS_PER_TRACE,
    crlb_precision,
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
# CRLB — MINFLUX Cramér-Rao bound (Marin & Ries 2024, 2-D donut, Eq. 44/45)
# ---------------------------------------------------------------------------

def test_crlb_ideal_formula():
    eco = np.array([100.0, 400.0, 900.0])
    L, sq = 50.0, 250.0
    r = crlb_precision(eco, L_nm=L, sigma_q_nm=sq)
    grad = 1.0 - L * L * np.log(2.0) / (sq * sq)
    expected = L / (2.0 * np.sqrt(2.0 * eco)) / grad     # Eq. 45
    assert np.allclose(r.per_loc_sigma_ideal, expected)
    assert r.median_sigma_ideal == pytest.approx(np.median(expected))
    assert not r.has_background
    assert np.allclose(r.per_loc_sigma_bg, r.per_loc_sigma_ideal)   # no bg data
    assert r.median_n == pytest.approx(400.0)
    assert r.sigma_q_nm == pytest.approx(sq)


def test_crlb_increases_with_L_and_psf_correction():
    eco = np.array([150.0, 200.0])
    r1 = crlb_precision(eco, L_nm=40.0, sigma_q_nm=250.0)
    r2 = crlb_precision(eco, L_nm=80.0, sigma_q_nm=250.0)
    assert r2.median_sigma_ideal > r1.median_sigma_ideal          # larger L → worse
    # a tighter σ_q strengthens the (1 − L²ln2/σ_q²)⁻¹ correction → worse precision
    r_tight = crlb_precision(eco, L_nm=80.0, sigma_q_nm=120.0)
    assert r_tight.median_sigma_ideal > r2.median_sigma_ideal


def test_crlb_background_degradation():
    eco = np.full(5, 100.0)
    efo = np.full(5, 100_000.0)
    fbg = np.full(5, 20_000.0)                # SBR = (efo − fbg)/fbg = 4
    r = crlb_precision(eco, L_nm=50.0, sigma_q_nm=250.0, efo=efo, fbg=fbg)
    assert r.has_background
    assert r.median_sbr == pytest.approx(4.0)
    factor = np.sqrt((1.0 + 1.0 / 4.0) * (1.0 + 3.0 / (4.0 * 4.0)))   # Eq. 44 factor
    assert np.allclose(r.per_loc_sigma_bg, r.per_loc_sigma_ideal * factor)
    assert r.median_sigma_bg > r.median_sigma_ideal


def test_crlb_high_sbr_approaches_ideal():
    eco = np.full(4, 150.0)
    efo = np.full(4, 1.0e7)
    fbg = np.full(4, 1.0)                      # SBR ~ 1e7 → factor → 1
    r = crlb_precision(eco, L_nm=50.0, sigma_q_nm=250.0, efo=efo, fbg=fbg)
    assert np.allclose(r.per_loc_sigma_bg, r.per_loc_sigma_ideal, rtol=1e-5)


def test_crlb_sigma_q_too_small_is_invalid():
    eco = np.full(4, 200.0)
    r = crlb_precision(eco, L_nm=100.0, sigma_q_nm=50.0)   # L > σ_q·√ln2
    assert not np.isfinite(r.median_sigma_ideal)
    assert "quadratic" in r.status.lower()


def test_crlb_axial_z_formula():
    eco = np.array([100.0, 400.0])
    Lz = 60.0
    r = crlb_precision(eco, L_nm=50.0, sigma_q_nm=250.0, L_z_nm=Lz)   # no bg
    assert r.has_z and r.L_z_nm == pytest.approx(Lz)
    # σ_z = L_z/(4√N), no σ_q, no (1−L²ln2/σq²) correction (1-D quadratic)
    assert np.allclose(r.per_loc_sigma_z_ideal, Lz / (4.0 * np.sqrt(eco)))
    assert np.allclose(r.per_loc_sigma_z_bg, r.per_loc_sigma_z_ideal)   # no bg


def test_crlb_axial_background_factor():
    eco = np.full(5, 100.0)
    efo = np.full(5, 100_000.0)
    fbg = np.full(5, 20_000.0)                  # SBR = 4
    r = crlb_precision(eco, L_nm=50.0, sigma_q_nm=250.0, efo=efo, fbg=fbg, L_z_nm=60.0)
    factor_z = np.sqrt((1.0 + 1.0 / 4.0) * (1.0 + 2.0 / (3.0 * 4.0)))   # Eq. 28 form
    assert np.allclose(r.per_loc_sigma_z_bg, r.per_loc_sigma_z_ideal * factor_z)
    assert r.median_sigma_z_bg > r.median_sigma_z_ideal


def test_crlb_per_dimension_photons():
    """N_xy (lateral) drives σ_xy; N_z (axial) drives σ_z — independently."""
    eco_xy = np.full(4, 200.0)          # final lateral iteration photons
    eco_z = np.full(4, 100.0)           # final axial iteration photons
    r = crlb_precision(eco_xy, L_nm=50.0, sigma_q_nm=250.0, L_z_nm=60.0, eco_z=eco_z)
    assert r.per_dimension_photons is True
    assert r.median_n == pytest.approx(200.0)          # N_xy
    assert r.median_n_z == pytest.approx(100.0)        # N_z
    # σ_xy uses N_xy=200, σ_z uses N_z=100.
    assert np.allclose(r.per_loc_sigma_ideal, 50.0 / (2.0 * np.sqrt(2.0 * 200.0)
                       * (1.0 - 50.0**2 * np.log(2.0) / 250.0**2)))
    assert np.allclose(r.per_loc_sigma_z_ideal, 60.0 / (4.0 * np.sqrt(100.0)))


def test_crlb_backward_compatible_single_eco():
    """Without eco_z, N_z equals N_xy (old behaviour) and per_dimension is False."""
    eco = np.full(4, 150.0)
    r = crlb_precision(eco, L_nm=50.0, sigma_q_nm=250.0, L_z_nm=60.0)
    assert r.per_dimension_photons is False
    assert np.allclose(r.per_loc_sigma_z_ideal, 60.0 / (4.0 * np.sqrt(150.0)))


def test_crlb_no_axial_when_lz_absent():
    r = crlb_precision(np.full(4, 150.0), L_nm=50.0, sigma_q_nm=250.0)
    assert not r.has_z
    assert r.per_loc_sigma_z_ideal.size == 0


def test_crlb_no_valid_counts():
    r = crlb_precision(np.array([0.0, -1.0, np.nan]), L_nm=50.0)
    assert r.n_locs == 0
    assert not np.isfinite(r.median_sigma_ideal)
    assert r.status


# ---------------------------------------------------------------------------
# Precision budget — STD² = σ_fl² + σ_CRB² (Marin & Ries 2026 decomposition)
# ---------------------------------------------------------------------------

def test_excess_precision_decomposition():
    # σ_fl = sqrt(STD² − σ_CRB²): 10² = 8² + 6²
    assert excess_precision(10.0, 6.0) == pytest.approx(8.0)
    # round-trips: STD = sqrt(σ_CRB² + σ_fl²)
    crb, fl = 6.0, 8.0
    assert np.hypot(crb, excess_precision(10.0, crb)) == pytest.approx(10.0)


def test_excess_precision_photon_limited_clamps_to_zero():
    # measured spread at/below the bound → no excess (photon-limited)
    assert excess_precision(6.0, 10.0) == 0.0
    assert excess_precision(5.0, 5.0) == 0.0


def test_excess_precision_invalid_inputs_are_nan():
    for bad in (np.nan, np.inf):
        assert not np.isfinite(excess_precision(bad, 6.0))
        assert not np.isfinite(excess_precision(10.0, bad))
    # non-positive measured spread → NaN (nothing to decompose)
    assert not np.isfinite(excess_precision(0.0, 6.0))
    assert not np.isfinite(excess_precision(-1.0, 6.0))
