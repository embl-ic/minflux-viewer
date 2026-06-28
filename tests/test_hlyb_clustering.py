import numpy as np
import pytest

from minflux_viewer.analysis.hlyb_clustering import (
    HlyBConfig,
    analyze_hlyb,
    analyze_hlyb_2d,
    analyze_hlyb_template3d,
    compute_border_mask,
    dbscan,
    estimate_trace_size,
    hlyb_template_model,
    locate_subunit_centers,
)


def test_dbscan_two_clusters_and_noise():
    pts = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],      # cluster A
        [50, 50, 0], [51, 50, 0], [50, 51, 0],  # cluster B
        [200, 200, 0],                          # noise
    ], dtype=float)
    labels = dbscan(pts, eps=3.0, min_pts=2)
    assert labels[6] == -1                       # isolated point is noise
    assert len(set(labels[:3])) == 1 and labels[0] != -1
    assert len(set(labels[3:6])) == 1 and labels[3] != -1
    assert labels[0] != labels[3]                # two distinct clusters


def test_dbscan_empty():
    assert dbscan(np.empty((0, 3)), 1.0, 2).shape == (0,)


def _make_subunit(center_nm, n=40, sigma_nm=1.2, seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(center_nm) + rng.normal(0, sigma_nm, size=(n, 3)) * np.array([1, 1, 0])


def _synthetic_two_hlyb():
    """Two HlyB structures, each 3 sub-units in an ~18 nm triangle (z = 0)."""
    tri = np.array([[0, 0, 0], [18, 0, 0], [9, 15.6, 0]], dtype=float)
    structures = [np.array([0.0, 0.0, 0.0]), np.array([300.0, 0.0, 0.0])]
    locs = []
    tids = []
    tid = 1
    for s, base in enumerate(structures):
        for u, off in enumerate(tri):
            pts = _make_subunit(base + off, n=40, sigma_nm=1.0, seed=10 * s + u)
            locs.append(pts)
            tids.append(np.full(pts.shape[0], tid))
            tid += 1
    loc_nm = np.vstack(locs)
    tid_arr = np.concatenate(tids)
    return loc_nm / 1e9, tid_arr  # loc in metres


def test_estimate_trace_size_positive():
    loc_m, tid = _synthetic_two_hlyb()
    sizes = estimate_trace_size(loc_m, tid, z_scale=1.0)
    assert sizes.shape == (3,)
    assert np.all(np.isfinite(sizes)) and np.all(sizes > 0)


def test_locate_subunit_centers_finds_six():
    loc_m, tid = _synthetic_two_hlyb()
    out = locate_subunit_centers(
        loc_m, tid, z_scale=1.0, pixel_size=2.5, min_loc_per_trace=1, basic_unit_size_nm=8.0)
    assert out["unit_centers"].shape[0] == 6
    assert out["n_traces"] == 6


def test_analyze_hlyb_two_structures_pairs():
    loc_m, tid = _synthetic_two_hlyb()
    cfg = HlyBConfig(basic_unit_size_nm=8.0, min_unit_count_per_HlyB=2, z_scaling_factor=1.0)
    res = analyze_hlyb(loc_m, tid, cfg)
    assert res["n_subunits"] == 6
    assert res["n_structures"] == 2
    pd = res["all_pair_distances"]
    assert pd.size == 6  # 3 pairs per triangle × 2 structures
    # All pair distances are the triangle edges (~18 nm).
    assert np.all((pd > 12) & (pd < 24))


def test_analyze_hlyb_empty_input():
    res = analyze_hlyb(np.empty((0, 3)), np.empty(0), HlyBConfig())
    assert res["n_subunits"] == 0
    assert res["n_structures"] == 0
    assert res["all_pair_distances"].size == 0


# -- 3-D template matching --------------------------------------------------

def _partial_template_centers():
    """Two observed 1a/1b/2a partial HlyB complexes."""
    x = (15.0**2 + 14.0**2 - 12.8**2) / (2 * 14.0)
    y = np.sqrt(15.0**2 - x**2)
    tri = np.array([[0.0, 0.0, 0.0], [14.0, 0.0, 0.0], [x, y, 0.0]])
    return [tri, tri + np.array([300.0, 0.0, 0.0])]


def _template_partial_locs():
    locs, tids, tid = [], [], 1
    for base in _partial_template_centers():
        for center in base:
            pts = _make_subunit(center, n=40, sigma_nm=0.8, seed=tid)
            locs.append(pts)
            tids.append(np.full(pts.shape[0], tid))
            tid += 1
    loc_nm = np.vstack(locs)
    return loc_nm / 1e9, np.concatenate(tids)


def test_hlyb_template_model_contains_all_priors():
    model = hlyb_template_model(HlyBConfig())
    assert list(model["labels"]) == ["1a", "1b", "2a", "2b", "3a", "3b"]
    assert set(np.round(model["prior_distances_nm"], 1)) == {12.8, 14.0, 15.0, 21.3, 23.0}


def test_analyze_hlyb_template3d_finds_partial_matches():
    loc_m, tid = _template_partial_locs()
    cfg = HlyBConfig(
        basic_unit_size_nm=8.0,
        min_loc_per_trace=1,
        min_unit_count_per_HlyB=2,
        z_scaling_factor=1.0,
    )
    res = analyze_hlyb_template3d(loc_m, tid, cfg)
    assert res["template_matching"] is True
    assert res["n_subunits"] == 6
    assert res["n_structures"] == 2
    assert res["all_pair_distances"].size == 6
    assert all(st["unit_centers"].shape[0] == 3 for st in res["structures"])
    assert np.all(np.abs(np.sort(res["all_pair_distances"]) - np.array([12.8, 12.8, 14, 14, 15, 15])) < 3)


def test_analyze_hlyb_template3d_rejects_single_overlarge_structure():
    locs, tids, tid = [], [], 1
    centers = np.vstack(_partial_template_centers())
    centers = np.vstack([centers, centers[0] + np.array([0.0, 35.0, 0.0])])
    for center in centers:
        pts = _make_subunit(center, n=40, sigma_nm=0.8, seed=tid + 50)
        locs.append(pts)
        tids.append(np.full(pts.shape[0], tid))
        tid += 1
    loc_m = np.vstack(locs) / 1e9
    res = analyze_hlyb_template3d(
        loc_m,
        np.concatenate(tids),
        HlyBConfig(basic_unit_size_nm=8.0, z_scaling_factor=1.0),
    )
    assert res["n_subunits"] == 7
    assert res["n_structures"] >= 2
    assert all(st["unit_centers"].shape[0] <= 6 for st in res["structures"])


# -- 2-D variant: E.coli border preprocessing ------------------------------

def test_compute_border_mask_shrinks_rod_interior():
    rng = np.random.default_rng(0)
    rod = np.column_stack([rng.uniform(0, 2000, 40000), rng.uniform(0, 800, 40000)])
    border = compute_border_mask(rod, border_size_nm=200, pixel_size_nm=20)
    interior = rod[~border]
    assert interior.shape[0] > 0
    # interior shrinks ~200 nm in from every edge of the [0,2000]x[0,800] rod
    assert interior[:, 0].min() >= 180 and interior[:, 0].max() <= 1820
    assert interior[:, 1].min() >= 180 and interior[:, 1].max() <= 620


def test_compute_border_mask_empty():
    assert compute_border_mask(np.empty((0, 2))).shape == (0,)


def test_analyze_hlyb_2d_removes_border_and_finds_interior():
    rng = np.random.default_rng(1)
    locs, tids, tid = [], [], 1
    # sparse single-loc cell body forms the mask (excluded as sub-units by min_loc)
    for p in np.column_stack([rng.uniform(0, 2000, 6000), rng.uniform(0, 800, 6000)]):
        locs.append(p[None, :])
        tids.append([tid])
        tid += 1
    tri = np.array([[0, 0], [18, 0], [9, 15.6]])

    def add_tri(cx, cy):
        nonlocal tid
        for off in tri:
            locs.append(np.array([cx, cy]) + off + rng.normal(0, 1.0, size=(40, 2)))
            tids.append(np.full(40, tid))
            tid += 1

    add_tri(600, 400)
    add_tri(1400, 400)   # two interior HlyB
    add_tri(100, 400)    # near the x=0 border → removed
    loc_nm = np.vstack(locs)
    loc_m = np.column_stack([loc_nm / 1e9, np.zeros(loc_nm.shape[0])])
    res = analyze_hlyb_2d(np.asarray(loc_m), np.concatenate(tids),
                          HlyBConfig(min_loc_per_trace=5, basic_unit_size_nm=8.0))
    assert res["is_2d"] is True
    assert res["n_total_traces"] == 6009
    assert res["n_border_traces"] >= 3
    assert res["border_points_nm"].shape[0] > 0
    assert res["n_structures"] == 2
    # both structures are interior; none near the removed x≈100 border triangle
    xs = sorted(round(float(st["centroid"][0])) for st in res["structures"])
    assert all(x > 300 for x in xs)


def test_analyze_hlyb_2d_ignores_z():
    # z is dropped in 2-D mode → a large z must not change the result
    rng = np.random.default_rng(2)
    locs, tids, tid = [], [], 1
    for p in np.column_stack([rng.uniform(0, 2000, 4000), rng.uniform(0, 800, 4000)]):
        locs.append(p[None, :])
        tids.append([tid])
        tid += 1
    loc_nm = np.vstack(locs)
    flat = np.column_stack([loc_nm / 1e9, np.zeros(loc_nm.shape[0])])
    tall = np.column_stack([loc_nm / 1e9, np.full(loc_nm.shape[0], 500e-9)])  # 500 nm z
    cfg = HlyBConfig(min_loc_per_trace=1, basic_unit_size_nm=8.0)
    r1 = analyze_hlyb_2d(flat, np.concatenate(tids), cfg)
    r2 = analyze_hlyb_2d(tall, np.concatenate(tids), cfg)
    assert r1["n_border_traces"] == r2["n_border_traces"]
    assert r1["n_subunits"] == r2["n_subunits"]
