"""3-D convolution-based segmentation (analysis/conv_segmentation_3d.py).

Pure NumPy/SciPy — no Qt. Validates the 3-D geometry-model registry, kernel
building, 3-D non-maximum suppression and the end-to-end detect-from-synthetic
pipeline. Mirrors tests/test_conv_segmentation.py stepped up to 3-D.
"""

import numpy as np
import pytest

from minflux_viewer.analysis import conv_segmentation_3d as cs3


# --- synthetic 3-D structures --------------------------------------------------
def _shell(cx, cy, cz, diameter, n=400, sigma=4.0, seed=0):
    """Points scattered on a spherical surface of the given diameter."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = diameter / 2.0 + rng.normal(0, sigma, n)
    return np.column_stack([cx, cy, cz]) + v * r[:, None]


def _blob3(cx, cy, cz, sigma, n=300, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.normal(cx, sigma, n),
                            rng.normal(cy, sigma, n),
                            rng.normal(cz, sigma, n)])


# --- registry ------------------------------------------------------------------
def test_registry_models_have_params_and_defaults():
    assert set(cs3.GEOMETRY_MODELS_3D) >= {"shell", "ball", "gaussian", "log"}
    for model in cs3.GEOMETRY_MODELS_3D.values():
        assert model.params, f"{model.key} has no params"
        d = model.defaults()
        assert set(d) == {ps.name for ps in model.params}
        assert model.size(d) > 0


def test_get_model_unknown_raises():
    with pytest.raises(ValueError):
        cs3.get_model("does-not-exist")


# --- kernels -------------------------------------------------------------------
def test_kernels_are_3d_and_finite():
    for key in cs3.GEOMETRY_MODELS_3D:
        k = cs3.build_kernel_3d(key, {}, 8.0, 16.0)
        assert k.ndim == 3, f"{key} kernel is not 3-D"
        assert np.isfinite(k).all()


def test_shell_kernel_peaks_off_centre():
    # positive (non-zero-mean) shell: centre lower than the spherical surface
    k = cs3.build_kernel_3d("shell", {"diameter_nm": 100.0, "rim_nm": 20.0},
                            8.0, 8.0, zero_mean=False)
    c = tuple(s // 2 for s in k.shape)
    assert k[c] < k.max()


def test_kernel_l2_normalised_and_zero_mean():
    k = cs3.build_kernel_3d("gaussian", {"fwhm_nm": 40.0}, 8.0, 16.0, zero_mean=True)
    assert np.isclose(np.sqrt((k ** 2).sum()), 1.0)
    assert abs(k.mean()) < 1e-9
    assert np.isfinite(k).all()


def test_log_kernel_is_mexican_hat():
    k = cs3.build_kernel_3d("log", {"diameter_nm": 60.0}, 8.0, 8.0, zero_mean=False)
    c = tuple(s // 2 for s in k.shape)
    assert k[c] > 0                              # positive centre
    assert k.min() < 0                           # negative surround
    assert abs(k.sum()) < 0.05 * np.abs(k).sum()  # near zero-mean
    assert np.isfinite(k).all()


def test_anisotropic_voxels_change_kernel_z_extent():
    fine = cs3.build_kernel_3d("ball", {"diameter_nm": 100.0}, 8.0, 8.0)
    coarse = cs3.build_kernel_3d("ball", {"diameter_nm": 100.0}, 8.0, 24.0)
    assert coarse.shape[2] < fine.shape[2]       # coarser z → fewer z planes
    assert coarse.shape[0] == fine.shape[0]      # xy sampling unchanged


# --- voxelisation + voxel-count guard -----------------------------------------
def test_render_histogram_3d_shapes():
    pts = _blob3(0, 0, 0, 20.0, n=500, seed=1)
    H, xe, ye, ze = cs3.render_histogram_3d(pts[:, 0], pts[:, 1], pts[:, 2], 8.0, 16.0)
    assert H.ndim == 3
    assert H.shape == (xe.size - 1, ye.size - 1, ze.size - 1)
    assert H.sum() == pts.shape[0]


def test_estimate_voxel_count_matches_histogram():
    pts = _blob3(0, 0, 0, 30.0, n=400, seed=2)
    est = cs3.estimate_voxel_count(pts, 8.0, 16.0)
    H, *_ = cs3.render_histogram_3d(pts[:, 0], pts[:, 1], pts[:, 2], 8.0, 16.0)
    # estimate is an upper-ish bound (pads by 2 per axis), never an under-count
    assert est >= H.size


def test_suggest_voxel_sizes_fits_cap_for_large_fov():
    # a multi-µm field of view that would blow past the cap at a fine voxel size
    rng = np.random.default_rng(3)
    pts = rng.uniform([-3000, 0, -2000], [2000, 6000, 600], size=(20000, 3))
    cap = 8_000_000
    px, pz = cs3.suggest_voxel_sizes(pts, max_voxels=cap, z_ratio=2.0)
    assert px > 0 and pz == pytest.approx(2.0 * px)
    assert cs3.estimate_voxel_count(pts, px, pz) <= cap
    # one round step finer must exceed the cap (i.e. it is the *smallest* that fits)
    assert cs3.estimate_voxel_count(pts, px - 1.0, 2.0 * (px - 1.0)) > cap


def test_suggest_voxel_sizes_degenerate_safe():
    px, pz = cs3.suggest_voxel_sizes(np.empty((0, 3)))
    assert px > 0 and pz > 0


# --- 3-D NMS -------------------------------------------------------------------
def test_nms_3d_picks_separated_maxima():
    vol = np.zeros((20, 20, 20))
    vol[5, 5, 5] = 5.0
    vol[6, 5, 5] = 4.0            # adjacent → within min distance → suppressed
    vol[15, 15, 15] = 3.0
    ijk, vals = cs3.non_max_suppression_3d(vol, 1.0, min_distance_nm=30.0,
                                           voxel_nm=(8.0, 8.0, 8.0))
    assert ijk.shape[0] == 2
    assert vals[0] >= vals[1]
    assert {tuple(r) for r in ijk} == {(5, 5, 5), (15, 15, 15)}


def test_nms_3d_threshold_filters():
    vol = np.zeros((10, 10, 10))
    vol[2, 2, 2] = 0.2
    vol[7, 7, 7] = 0.9
    ijk, _ = cs3.non_max_suppression_3d(vol, 0.5, 10.0, (8.0, 8.0, 8.0))
    assert ijk.tolist() == [[7, 7, 7]]


def test_nms_3d_empty_safe():
    ijk, vals = cs3.non_max_suppression_3d(np.empty((0, 0, 0)), 0.5, 10.0,
                                           (8.0, 8.0, 8.0))
    assert ijk.shape == (0, 3) and vals.shape == (0,)


def test_nms_3d_anisotropic_distance():
    # two peaks 2 z-voxels apart; coarse z makes them far enough to both survive
    vol = np.zeros((8, 8, 8))
    vol[4, 4, 2] = 2.0
    vol[4, 4, 4] = 1.0
    near, _ = cs3.non_max_suppression_3d(vol, 0.5, 40.0, (8.0, 8.0, 8.0))
    far, _ = cs3.non_max_suppression_3d(vol, 0.5, 40.0, (8.0, 8.0, 30.0))
    assert near.shape[0] == 1     # 16 nm apart < 40 nm → merged
    assert far.shape[0] == 2      # 60 nm apart > 40 nm → both kept


# --- two-phase helpers ---------------------------------------------------------
def test_convolve_response_3d_normalised_to_unit_max():
    H = np.zeros((20, 20, 20))
    H[10, 10, 10] = 10.0
    k = cs3.build_kernel_3d("gaussian", {"fwhm_nm": 24.0}, 8.0, 8.0, zero_mean=False)
    resp = cs3.convolve_response_3d(H, k)
    assert resp.shape == H.shape
    assert np.isclose(resp.max(), 1.0)


def test_convolve_response_3d_empty_safe():
    assert cs3.convolve_response_3d(np.zeros((0, 0, 0)), np.ones((3, 3, 3))).size == 0


def test_detections_from_response_3d_maps_to_nm():
    response = np.zeros((20, 20, 20))
    response[8, 12, 4] = 1.0
    xedge = np.arange(0.0, 21.0) * 5.0
    yedge = np.arange(0.0, 21.0) * 5.0
    zedge = np.arange(0.0, 21.0) * 10.0
    centers, vals = cs3.detections_from_response_3d(
        response, xedge, yedge, zedge, min_response=0.5, min_distance_nm=8.0)
    assert centers.shape == (1, 3)
    # bin centres: x 5*8+2.5=42.5, y 5*12+2.5=62.5, z 10*4+5=45
    assert np.allclose(centers[0], [42.5, 62.5, 45.0])
    assert np.isclose(vals[0], 1.0)


# --- end-to-end ----------------------------------------------------------------
def test_segment_matches_two_phase_helpers():
    pts = np.vstack([_blob3(0, 0, 0, 12.0, seed=1), _blob3(400, 0, 0, 12.0, seed=2)])
    params = {"fwhm_nm": 40.0}
    res = cs3.segment_by_convolution_3d(
        pts, model_key="gaussian", params=params, pixel_size_nm=8.0,
        z_pixel_size_nm=8.0, min_response=0.3, min_distance_nm=120.0)
    H, xe, ye, ze = cs3.render_histogram_3d(pts[:, 0], pts[:, 1], pts[:, 2], 8.0, 8.0)
    k = cs3.build_kernel_3d("gaussian", params, 8.0, 8.0)
    resp = cs3.convolve_response_3d(H, k)
    centers, _ = cs3.detections_from_response_3d(
        resp, xe, ye, ze, min_response=0.3, min_distance_nm=120.0)
    assert np.array_equal(np.sort(res["centers"], axis=0), np.sort(centers, axis=0))


def test_detects_two_blobs_with_log():
    truth = [(0.0, 0.0, 0.0), (300.0, 200.0, 100.0)]
    pts = np.vstack([_blob3(*c, 10.0, seed=i) for i, c in enumerate(truth)])
    res = cs3.segment_by_convolution_3d(
        pts, model_key="log", params={"diameter_nm": 50.0},
        pixel_size_nm=8.0, z_pixel_size_nm=8.0, min_response=0.3, min_distance_nm=120.0)
    centers = res["centers"]
    assert centers.shape[0] == 2
    for cx, cy, cz in truth:
        d = np.sqrt((centers[:, 0] - cx) ** 2 + (centers[:, 1] - cy) ** 2
                    + (centers[:, 2] - cz) ** 2)
        assert d.min() < 30.0


def test_detects_two_shells():
    truth = [(0.0, 0.0, 0.0), (400.0, 0.0, 0.0)]
    pts = np.vstack([_shell(*c, 100.0, n=500, seed=i) for i, c in enumerate(truth)])
    res = cs3.segment_by_convolution_3d(
        pts, model_key="shell", params={"diameter_nm": 100.0, "rim_nm": 22.0},
        pixel_size_nm=8.0, z_pixel_size_nm=8.0, min_response=0.3)
    centers = res["centers"]
    assert centers.shape[0] == 2
    for cx, cy, cz in truth:
        d = np.sqrt((centers[:, 0] - cx) ** 2 + (centers[:, 1] - cy) ** 2
                    + (centers[:, 2] - cz) ** 2)
        assert d.min() < 35.0


def test_response_map_rescaled_to_unit_max():
    pts = _shell(0, 0, 0, 100.0, n=500, seed=1)
    res = cs3.segment_by_convolution_3d(pts, model_key="shell", pixel_size_nm=8.0,
                                        z_pixel_size_nm=8.0)
    rm = res["response_map"]
    assert rm.ndim == 3 and rm.size > 0
    assert np.isclose(rm.max(), 1.0)


def test_max_detections_caps_output():
    truth = [(0.0, 0.0, 0.0), (400.0, 0.0, 0.0), (0.0, 350.0, 0.0)]
    pts = np.vstack([_blob3(*c, 12.0, seed=i) for i, c in enumerate(truth)])
    res = cs3.segment_by_convolution_3d(
        pts, model_key="gaussian", params={"fwhm_nm": 40.0},
        pixel_size_nm=8.0, z_pixel_size_nm=8.0, min_response=0.1,
        min_distance_nm=120.0, max_detections=2)
    assert res["centers"].shape[0] == 2


def test_empty_and_sparse_safe():
    for pts in (np.empty((0, 3)), np.zeros((2, 3))):
        res = cs3.segment_by_convolution_3d(pts, model_key="shell")
        assert res["centers"].shape == (0, 3)
        assert res["kernel"].ndim == 3


def test_sqrt_stabilize_still_detects():
    pts = np.vstack([_blob3(0, 0, 0, 12.0, seed=1), _blob3(400, 0, 0, 12.0, seed=2)])
    res = cs3.segment_by_convolution_3d(
        pts, model_key="gaussian", params={"fwhm_nm": 40.0},
        pixel_size_nm=8.0, z_pixel_size_nm=8.0, min_response=0.3,
        min_distance_nm=120.0, sqrt_stabilize=True)
    assert res["centers"].shape[0] == 2


def test_segment_with_dog_detects_blobs():
    pts = np.vstack([_blob3(0, 0, 0, 12.0, seed=1), _blob3(400, 0, 0, 12.0, seed=2)])
    res = cs3.segment_by_convolution_3d(
        pts, model_key="gaussian", params={"fwhm_nm": 40.0},
        pixel_size_nm=8.0, z_pixel_size_nm=8.0, min_response=0.3,
        min_distance_nm=120.0, dog=True)
    assert res["centers"].shape[0] == 2
