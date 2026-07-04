"""General convolution-based segmentation (analysis/conv_segmentation.py).

Pure NumPy/SciPy — no Qt. Validates the geometry-model registry, kernel building,
non-maximum suppression and the end-to-end detect-from-synthetic-data pipeline.
"""

import numpy as np
import pytest

from minflux_viewer.analysis import conv_segmentation as cs


# --- synthetic structures ------------------------------------------------------
def _ring(cx, cy, diameter, n=160, sigma=4.0, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    r = diameter / 2.0 + rng.normal(0, sigma, n)
    return np.column_stack([cx + r * np.cos(theta), cy + r * np.sin(theta)])


def _blob(cx, cy, sigma, n=200, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.normal(cx, sigma, n), rng.normal(cy, sigma, n)])


def _disk(cx, cy, radius, n=300, seed=0):
    """Uniformly-filled disk (a solid cluster — the ring kernel peaks at centre)."""
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)])


# --- registry ------------------------------------------------------------------
def test_registry_models_have_params_and_defaults():
    assert set(cs.GEOMETRY_MODELS) >= {"ring", "disk", "gaussian", "log"}
    for model in cs.GEOMETRY_MODELS.values():
        assert model.params, f"{model.key} has no params"
        d = model.defaults()
        assert set(d) == {ps.name for ps in model.params}
        assert model.size(d) > 0


def test_get_model_unknown_raises():
    with pytest.raises(ValueError):
        cs.get_model("does-not-exist")


# --- kernels -------------------------------------------------------------------
def test_ring_kernel_peaks_at_radius():
    # positive (non-zero-mean) ring: centre lower than the ring circle
    k = cs.build_kernel("ring", {"diameter_nm": 100.0, "rim_nm": 20.0}, 4.0,
                        zero_mean=False)
    c = k.shape[0] // 2
    assert k[c, c] < k.max()


def test_kernel_l2_normalised_and_zero_mean():
    k = cs.build_kernel("gaussian", {"fwhm_nm": 40.0}, 4.0, zero_mean=True)
    assert np.isclose(np.sqrt((k ** 2).sum()), 1.0)
    assert abs(k.mean()) < 1e-9          # DC removed
    assert np.isfinite(k).all()


def test_disk_kernel_is_finite_and_square():
    k = cs.build_kernel("disk", {"diameter_nm": 80.0, "edge_nm": 8.0}, 4.0)
    assert k.ndim == 2 and k.shape[0] == k.shape[1]
    assert np.isfinite(k).all()


def test_log_kernel_is_mexican_hat():
    # positive centre, negative surround, ~zero total (DC-free blob detector)
    k = cs.build_kernel("log", {"diameter_nm": 60.0}, 4.0, zero_mean=False)
    c = k.shape[0] // 2
    assert k[c, c] > 0                       # positive centre
    assert k.min() < 0                       # negative surround
    assert abs(k.sum()) < 0.05 * np.abs(k).sum()   # near zero-mean
    assert np.isfinite(k).all()


def test_segment_detects_blobs_with_log():
    truth = [(0.0, 0.0), (300.0, 200.0)]
    pts = np.vstack([_blob(cx, cy, 10.0, seed=i) for i, (cx, cy) in enumerate(truth)])
    res = cs.segment_by_convolution(
        pts, model_key="log", params={"diameter_nm": 50.0},
        pixel_size_nm=4.0, min_response=0.3, min_distance_nm=120.0)
    centers = res["centers"]
    assert centers.shape[0] == 2
    for cx, cy in truth:
        assert np.min(np.hypot(centers[:, 0] - cx, centers[:, 1] - cy)) < 25.0


# --- NMS -----------------------------------------------------------------------
def test_nms_picks_separated_maxima():
    img = np.zeros((50, 50))
    img[10, 10] = 5.0
    img[10, 13] = 4.0          # within min_distance of the first → suppressed
    img[40, 40] = 3.0
    rc, vals = cs.non_max_suppression(img, min_height=1.0, min_distance_px=5.0)
    assert rc.shape[0] == 2
    # strongest first
    assert vals[0] >= vals[1]
    assert {tuple(r) for r in rc} == {(10, 10), (40, 40)}


def test_nms_threshold_filters():
    img = np.zeros((20, 20))
    img[5, 5] = 0.2
    img[15, 15] = 0.9
    rc, _ = cs.non_max_suppression(img, min_height=0.5, min_distance_px=2.0)
    assert rc.tolist() == [[15, 15]]


def test_nms_empty_safe():
    rc, vals = cs.non_max_suppression(np.empty((0, 0)), 0.5, 3.0)
    assert rc.shape == (0, 2) and vals.shape == (0,)


# --- two-phase helpers (used by the interactive UI) ---------------------------
def test_convolve_response_normalised_to_unit_max():
    H = np.zeros((30, 30))
    H[15, 15] = 10.0
    k = cs.build_kernel("gaussian", {"fwhm_nm": 20.0}, 4.0, zero_mean=False)
    resp = cs.convolve_response(H, k)
    assert resp.shape == H.shape
    assert np.isclose(resp.max(), 1.0)


def test_convolve_response_empty_safe():
    assert cs.convolve_response(np.zeros((0, 0)), np.ones((3, 3))).size == 0


def test_detections_from_response_maps_to_nm():
    # a single bright pixel → one detection at that bin's nm centre
    response = np.zeros((20, 20))
    response[8, 12] = 1.0
    xedge = np.arange(0.0, 21.0) * 5.0       # 5 nm bins starting at 0
    yedge = np.arange(0.0, 21.0) * 5.0
    centers, vals = cs.detections_from_response(
        response, xedge, yedge, min_response=0.5, min_distance_px=2.0)
    assert centers.shape == (1, 2)
    # bin 8 centre = 5*8 + 2.5 = 42.5 ; bin 12 centre = 5*12 + 2.5 = 62.5
    assert np.allclose(centers[0], [42.5, 62.5])
    assert np.isclose(vals[0], 1.0)


def test_segment_matches_two_phase_helpers():
    """The all-in-one path agrees with render → convolve → detect."""
    pts = np.vstack([_ring(0, 0, 100.0, seed=1), _ring(400, 0, 100.0, seed=2)])
    params = {"diameter_nm": 100.0, "rim_nm": 22.0}
    res = cs.segment_by_convolution(pts, model_key="ring", params=params,
                                    pixel_size_nm=5.0, min_response=0.3)
    H, xe, ye = cs.render_histogram_2d(pts[:, 0], pts[:, 1], 5.0)
    k = cs.build_kernel("ring", params, 5.0)
    resp = cs.convolve_response(H, k)
    centers, _ = cs.detections_from_response(
        resp, xe, ye, min_response=0.3, min_distance_px=100.0 / 5.0)
    assert np.array_equal(np.sort(res["centers"], axis=0), np.sort(centers, axis=0))


# --- end-to-end ----------------------------------------------------------------
def test_detects_three_rings():
    truth = [(0.0, 0.0), (400.0, 0.0), (0.0, 350.0)]
    pts = np.vstack([_ring(cx, cy, 100.0, seed=i) for i, (cx, cy) in enumerate(truth)])
    res = cs.segment_by_convolution(
        pts, model_key="ring", params={"diameter_nm": 100.0, "rim_nm": 22.0},
        pixel_size_nm=5.0, min_response=0.3)
    centers = res["centers"]
    assert centers.shape[0] == 3
    for cx, cy in truth:
        assert np.min(np.hypot(centers[:, 0] - cx, centers[:, 1] - cy)) < 25.0


def test_detects_blobs_with_gaussian():
    truth = [(0.0, 0.0), (300.0, 200.0)]
    pts = np.vstack([_blob(cx, cy, 12.0, seed=i) for i, (cx, cy) in enumerate(truth)])
    res = cs.segment_by_convolution(
        pts, model_key="gaussian", params={"fwhm_nm": 40.0},
        pixel_size_nm=4.0, min_response=0.3, min_distance_nm=120.0)
    centers = res["centers"]
    assert centers.shape[0] == 2
    for cx, cy in truth:
        assert np.min(np.hypot(centers[:, 0] - cx, centers[:, 1] - cy)) < 25.0


def test_response_map_rescaled_to_unit_max():
    pts = _ring(0, 0, 100.0, seed=1)
    res = cs.segment_by_convolution(pts, model_key="ring", pixel_size_nm=5.0)
    rm = res["response_map"]
    assert rm.size > 0
    assert np.isclose(rm.max(), 1.0)


def test_max_detections_caps_output():
    truth = [(0.0, 0.0), (400.0, 0.0), (0.0, 350.0)]
    pts = np.vstack([_ring(cx, cy, 100.0, seed=i) for i, (cx, cy) in enumerate(truth)])
    res = cs.segment_by_convolution(
        pts, model_key="ring", params={"diameter_nm": 100.0, "rim_nm": 22.0},
        pixel_size_nm=5.0, min_response=0.1, max_detections=2)
    assert res["centers"].shape[0] == 2


def test_empty_and_sparse_safe():
    for pts in (np.empty((0, 2)), np.zeros((2, 2))):
        res = cs.segment_by_convolution(pts, model_key="ring")
        assert res["centers"].shape == (0, 2)
        assert res["kernel"].ndim == 2     # kernel still built


# --- sqrt stabilization -------------------------------------------------------
def test_sqrt_stabilize_compresses_counts():
    H = np.array([[0.0, 1.0], [4.0, 9.0]])
    out = cs.stabilize_histogram(H, sqrt_stabilize=True)
    assert np.allclose(out, [[0.0, 1.0], [2.0, 3.0]])
    assert np.array_equal(cs.stabilize_histogram(H, sqrt_stabilize=False), H)


def test_sqrt_stabilize_still_detects_rings():
    pts = np.vstack([_ring(0, 0, 100.0, seed=1), _ring(400, 0, 100.0, seed=2)])
    res = cs.segment_by_convolution(
        pts, model_key="ring", params={"diameter_nm": 100.0, "rim_nm": 22.0},
        pixel_size_nm=5.0, min_response=0.3, sqrt_stabilize=True)
    assert res["centers"].shape[0] == 2


# --- DoG band-pass ------------------------------------------------------------
def test_dog_bandpass_suppresses_flat_background():
    flat = np.ones((60, 60)) * 5.0
    out = cs.dog_bandpass(flat, radius_px=4.0)
    # a constant field has no band-pass content → response ~0 in the interior
    assert abs(out[30, 30]) < 1e-6


def test_dog_bandpass_responds_to_blob():
    img = np.zeros((80, 80))
    img[40, 40] = 100.0
    out = cs.dog_bandpass(img, radius_px=3.0)
    assert out[40, 40] > 0.0          # positive at the blob centre


def test_segment_with_dog_detects_rings():
    pts = np.vstack([_ring(0, 0, 100.0, seed=1), _ring(400, 0, 100.0, seed=2)])
    res = cs.segment_by_convolution(
        pts, model_key="ring", params={"diameter_nm": 100.0, "rim_nm": 22.0},
        pixel_size_nm=5.0, min_response=0.3, dog=True)
    assert res["centers"].shape[0] == 2


# --- ring validation (reject solid blobs) -------------------------------------
def test_inside_ratio_high_for_blob_low_for_ring():
    ring = _ring(0, 0, 100.0, n=200, sigma=3.0, seed=1)
    disk = _disk(0, 0, 50.0, n=200, seed=2)         # solid filled cluster
    centers = np.array([[0.0, 0.0]])
    ins_ring, _, nr_ring = cs.ring_inside_outside_ratios(ring, centers, 50.0, 20.0)
    ins_disk, _, _ = cs.ring_inside_outside_ratios(disk, centers, 50.0, 20.0)
    assert ins_ring[0] < 0.3
    assert ins_disk[0] > 0.6           # filled centre → high inside ratio
    assert nr_ring[0] >= 6


def test_ring_validation_rejects_blob_keeps_rings():
    rings = [(0.0, 0.0), (400.0, 0.0)]
    pts = np.vstack([_ring(cx, cy, 100.0, n=180, seed=i)
                     for i, (cx, cy) in enumerate(rings)])
    pts = np.vstack([pts, _disk(0.0, 350.0, 48.0, n=260, seed=9)])  # decoy solid disk
    common = dict(model_key="ring", params={"diameter_nm": 100.0, "rim_nm": 22.0},
                  pixel_size_nm=5.0, min_response=0.12)
    no_val = cs.segment_by_convolution(pts, **common)
    with_val = cs.segment_by_convolution(
        pts, **common, ring_validation={"max_inside": 0.5, "max_outside": 2.0})

    def near(centers, pt, tol=35.0):
        if centers.shape[0] == 0:
            return False
        return np.min(np.hypot(centers[:, 0] - pt[0], centers[:, 1] - pt[1])) < tol

    # the disk is detectable without validation but rejected with it; rings survive
    assert near(no_val["centers"], (0.0, 350.0))
    assert not near(with_val["centers"], (0.0, 350.0))
    for cx, cy in rings:
        assert near(with_val["centers"], (cx, cy))
    assert with_val["inside_ratio"].shape[0] == with_val["centers"].shape[0]


def test_ring_validation_ignored_for_non_ring_model():
    pts = np.vstack([_blob(0, 0, 12.0, seed=1), _blob(300, 200, 12.0, seed=2)])
    res = cs.segment_by_convolution(
        pts, model_key="gaussian", params={"fwhm_nm": 40.0}, pixel_size_nm=4.0,
        min_response=0.3, min_distance_nm=120.0,
        ring_validation={"max_inside": 0.1})    # must be a no-op for gaussian
    assert res["centers"].shape[0] == 2
