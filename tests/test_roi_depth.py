"""Data-aware out-of-plane (depth) placement for drawn ROI vertices."""

import numpy as np

from minflux_viewer.core.roi_depth import _weighted_median, weighted_depths


def test_weighted_median_basic():
    assert _weighted_median(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0])) == 2.0
    # weight pulls the median toward the heavy value
    assert _weighted_median(np.array([0.0, 10.0]), np.array([1.0, 100.0])) == 10.0


def test_depth_follows_dense_z_slice():
    # a column at (x=0,y=0): most localizations sit at z=500, a few elsewhere
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(0, 10, n)
    y = rng.normal(0, 10, n)
    z = np.concatenate([np.full(360, 500.0), rng.uniform(-1000, 1000, 40)])
    # far-away cluster at z=-800 that must NOT affect the (0,0) column
    x = np.concatenate([x, rng.normal(5000, 10, 200)])
    y = np.concatenate([y, rng.normal(5000, 10, 200)])
    z = np.concatenate([z, np.full(200, -800.0)])
    out = weighted_depths([(0.0, 0.0)], x, y, z, radius_nm=60.0)
    assert out[0] is not None
    assert abs(out[0] - 500.0) < 30.0          # dragged onto the dense z slice


def test_empty_column_returns_none():
    x = np.array([1000.0, 1010.0]); y = np.array([1000.0, 1010.0]); z = np.array([0.0, 5.0])
    # query far from any localization → no neighbours → None (caller uses centre)
    assert weighted_depths([(0.0, 0.0)], x, y, z, radius_nm=60.0) == [None]


def test_batch_many_points_uses_kdtree_path():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 100, 500); y = rng.uniform(0, 100, 500)
    z = np.full(500, 42.0)
    pts = [(float(a), float(b)) for a, b in rng.uniform(0, 100, (20, 2))]
    out = weighted_depths(pts, x, y, z, radius_nm=40.0)
    assert len(out) == 20
    for d in out:
        assert d is None or abs(d - 42.0) < 1e-6


def test_nonfinite_filtered():
    x = np.array([0.0, 0.0, np.nan]); y = np.array([0.0, 0.0, 0.0]); z = np.array([10.0, 20.0, np.nan])
    out = weighted_depths([(0.0, 0.0)], x, y, z, radius_nm=60.0)
    assert out[0] is not None and 10.0 <= out[0] <= 20.0
