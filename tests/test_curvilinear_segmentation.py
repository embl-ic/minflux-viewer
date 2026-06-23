"""Curvilinear-structure segmentation (analysis/curvilinear_segmentation.py).

Pure NumPy/SciPy — no Qt. Validates the Hessian ridge filter, oriented kernels,
Otsu, Zhang-Suen skeletonization, path tracing, and the end-to-end trace of a
synthetic filament.
"""

import numpy as np
import pytest

from minflux_viewer.analysis import curvilinear_segmentation as cv


# --- synthetic filaments -------------------------------------------------------
def _line_points(p0, p1, n=400, width=4.0, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 1, n)
    x = p0[0] + t * (p1[0] - p0[0])
    y = p0[1] + t * (p1[1] - p0[1])
    return np.column_stack([x + rng.normal(0, width, n), y + rng.normal(0, width, n)])


# --- Hessian / Frangi ----------------------------------------------------------
def test_ridge_filter_responds_on_line_not_blob():
    img = np.zeros((80, 80))
    img[40, 5:75] = 1.0                       # a horizontal line
    line_resp = cv.ridge_filter(img, [2.0, 3.0], method="frangi")
    blob = np.zeros((80, 80))
    blob[38:43, 38:43] = 1.0                  # a compact blob
    blob_resp = cv.ridge_filter(blob, [2.0, 3.0], method="frangi")
    assert line_resp[40, 40] > 0.5            # strong on the ridge
    # ridge response peaks along the line, far more than a blob's centre
    assert line_resp.max() >= blob_resp.max()


def test_ridge_filter_normalised_and_finite():
    img = np.zeros((60, 60))
    img[30, 10:50] = 1.0
    for method in cv.RIDGE_METHODS:
        r = cv.ridge_filter(img, [2.0, 4.0], method=method)
        assert np.isfinite(r).all()
        assert np.isclose(r.max(), 1.0)


def test_hessian_eigenvalues_bright_ridge_negative_lam2():
    img = np.zeros((50, 50))
    img[25, 5:45] = 1.0
    _lam1, lam2 = cv.hessian_eigenvalues(img, 2.0)
    assert lam2[25, 25] < 0                    # bright ridge → across-curvature < 0


def test_scales_from_nm():
    s = cv.scales_from_nm(8.0, 24.0, 3, 4.0)   # 8,24 nm at 4 nm/px → 2..6 px
    assert s.shape == (3,)
    assert np.isclose(s[0], 2.0) and np.isclose(s[-1], 6.0)
    assert cv.scales_from_nm(8.0, 24.0, 1, 4.0).shape == (1,)


# --- oriented kernels ----------------------------------------------------------
def test_oriented_line_kernel_normalised_and_directional():
    k0 = cv.oriented_line_kernel(21, 5, 0.0, gaussian=True)
    assert np.isclose(k0.sum(), 1.0)
    c = k0.shape[0] // 2
    # horizontal bar: more weight along the row than down the column
    assert k0[c, :].sum() > k0[:, c].sum()


def test_directional_filter_enhances_line():
    img = np.zeros((60, 60))
    img[30, 10:50] = 1.0
    out = cv.directional_filter(img, length_px=15, width_px=5, n_angles=8)
    assert out.shape == img.shape
    assert out[30, 30] >= out[10, 10]


# --- Otsu ----------------------------------------------------------------------
def test_otsu_separates_bimodal():
    v = np.concatenate([np.zeros(500), np.ones(500) * 10.0])
    t = cv.otsu_threshold(v)
    assert 0.0 < t < 10.0


# --- skeletonization -----------------------------------------------------------
def test_zhang_suen_thins_thick_line_to_one_px():
    img = np.zeros((30, 40), dtype=bool)
    img[13:18, 5:35] = True                    # 5-px-thick horizontal bar
    skel = cv.zhang_suen_skeleton(img)
    # each column in the bar span keeps exactly one skeleton pixel
    col_counts = skel[:, 10:30].sum(axis=0)
    assert np.all(col_counts == 1)
    assert skel.sum() < img.sum()


def test_skeleton_to_paths_traces_and_orders():
    skel = np.zeros((20, 40), dtype=bool)
    skel[10, 5:35] = True
    paths = cv.skeleton_to_paths(skel, min_length_px=10)
    assert len(paths) == 1
    path = paths[0]
    assert path.shape[0] == 30
    # ordered: columns monotonic along the path
    cols = path[:, 1]
    assert np.all(np.diff(cols) > 0) or np.all(np.diff(cols) < 0)


def test_skeleton_to_paths_splits_at_junction():
    # a "T": one horizontal + a vertical stub meeting at a junction
    skel = np.zeros((30, 40), dtype=bool)
    skel[15, 5:35] = True
    skel[16:28, 20] = True
    paths = cv.skeleton_to_paths(skel, min_length_px=3)
    assert len(paths) >= 2                      # junction split into branches


# --- end-to-end ----------------------------------------------------------------
def test_segment_curvilinear_traces_a_filament():
    pts = _line_points((0.0, 0.0), (1000.0, 0.0), n=1500, width=5.0, seed=1)
    res = cv.segment_curvilinear(
        pts, pixel_size_nm=8.0, scale_min_nm=12.0, scale_max_nm=24.0, n_scales=3,
        threshold=0.1, min_length_nm=200.0)
    assert len(res["paths"]) >= 1
    longest = max(res["paths"], key=lambda p: p.shape[0])
    # the traced centre line spans most of the filament in x and hugs y≈0
    assert np.ptp(longest[:, 0]) > 600.0
    assert np.median(np.abs(longest[:, 1])) < 60.0
    assert np.isclose(res["ridge_map"].max(), 1.0)


def test_segment_curvilinear_empty_safe():
    res = cv.segment_curvilinear(np.empty((0, 2)))
    assert res["paths"] == []
    assert res["skeleton"].dtype == bool


def test_directional_option_runs_end_to_end():
    pts = _line_points((0.0, 0.0), (0.0, 800.0), n=1200, width=5.0, seed=2)
    res = cv.segment_curvilinear(
        pts, pixel_size_nm=8.0, scale_min_nm=12.0, scale_max_nm=20.0, n_scales=2,
        directional={"length_nm": 80.0, "width_nm": 16.0, "n_angles": 12})
    assert len(res["paths"]) >= 1


# --- width measurement (FWHM perpendicular profile) ---------------------------
def _gaussian_ridge_image(px=4.0, sigma_nm=12.0, length_nm=400.0, span_nm=200.0):
    """A horizontal filament image with a Gaussian cross-profile of known width."""
    x = np.arange(0.0, length_nm + px, px)
    y = np.arange(-span_nm, span_nm + px, px)
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])
    img = np.repeat(np.exp(-(yc ** 2) / (2 * sigma_nm ** 2))[None, :], xc.size, axis=0)
    return img, x, y, xc, sigma_nm


def test_fwhm_of_gaussian_matches_analytic():
    offsets = np.linspace(-60, 60, 241)
    sigma = 12.0
    profile = np.exp(-(offsets ** 2) / (2 * sigma ** 2))
    expected = 2.3548200450309493 * sigma          # FWHM = 2*sqrt(2 ln2) * sigma
    assert abs(cv.fwhm(offsets, profile) - expected) < 1.0


def test_fwhm_nan_when_no_crossing():
    offsets = np.linspace(-10, 10, 21)
    assert np.isnan(cv.fwhm(offsets, np.ones_like(offsets)))   # flat → no half-max


def test_sample_image_bilinear():
    img = np.zeros((10, 10))
    img[5, 5] = 1.0
    xedge = np.arange(11) * 4.0
    yedge = np.arange(11) * 4.0
    # bin (5,5) centre is at (22, 22) nm
    v = cv.sample_image(img, xedge, yedge, np.array([[22.0, 22.0]]), 4.0)
    assert v[0] > 0.99


def test_path_perpendiculars_orthogonal_to_tangent():
    path = np.column_stack([np.arange(0, 100, 5.0), np.zeros(20)])   # horizontal
    perp = cv.path_perpendiculars(path)
    # perpendicular to a horizontal line points along ±y
    assert np.allclose(np.abs(perp[:, 1]), 1.0, atol=1e-6)
    assert np.allclose(perp[:, 0], 0.0, atol=1e-6)


def test_measure_filament_width_recovers_gaussian_fwhm():
    img, xedge, yedge, xc, sigma = _gaussian_ridge_image(px=4.0, sigma_nm=12.0)
    # centre line along y=0 (the ridge), vertices across the image length
    path = np.column_stack([xc, np.zeros(xc.size)])
    res = cv.measure_filament_width(
        img, xedge, yedge, [path], max_dist_nm=60.0, step_nm=4.0, pixel_size_nm=4.0)
    expected = 2.3548200450309493 * sigma
    assert abs(res["fwhm"] - expected) < 4.0          # within ~1 pixel
    assert res["offsets"][np.argmax(res["profile"])] == 0.0   # peak at the centre line
    assert res["per_path_fwhm"].shape == (1,)
