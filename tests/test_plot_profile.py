"""Pure line-profile sampling for the Plot Profile tool."""

import numpy as np
import pytest

from minflux_viewer.analysis.plot_profile import (
    band_polygon,
    line_profile_from_locs,
    path_cumlen,
    resample_path,
)


# ------------------------------------------------------------ geometry

def test_path_cumlen_straight_and_polyline():
    assert path_cumlen([[0, 0], [3, 4]]).tolist() == [0.0, 5.0]
    assert path_cumlen([[0, 0], [3, 4], [3, 16]]).tolist() == [0.0, 5.0, 17.0]


def test_resample_straight_line():
    pts, tan, dist = resample_path([[0, 0], [10, 0]], step_nm=1.0)
    assert pts.shape[0] == 11
    assert np.allclose(pts[:, 1], 0.0)
    assert np.allclose(pts[:, 0], np.linspace(0, 10, 11))
    assert np.allclose(tan, [1.0, 0.0])
    assert dist[0] == 0.0 and dist[-1] == pytest.approx(10.0)


def test_resample_polyline_arc_length_monotonic():
    _pts, _tan, dist = resample_path([[0, 0], [10, 0], [10, 10]], step_nm=2.0)
    assert dist[0] == 0.0
    assert dist[-1] == pytest.approx(20.0)
    assert np.all(np.diff(dist) > 0)


# -------------------------------------------- localization-based sampling

def test_locs_binned_by_arc_length():
    # 3 localizations along a length-10 line at distances 1, 5, 9 (bin centres, n_bins=5)
    locs = np.array([[1.0, 0.0], [5.0, 0.0], [9.0, 0.0]])
    c, counts, bw = line_profile_from_locs(locs, [[0, 0], [10, 0]], 4.0, n_bins=5)
    assert counts.sum() == 3 and bw == pytest.approx(2.0)
    assert counts[np.argmin(np.abs(c - 1.0))] == 1
    assert counts[np.argmin(np.abs(c - 5.0))] == 1
    assert counts[np.argmin(np.abs(c - 9.0))] == 1


def test_width_band_selects_perpendicular_neighbours():
    locs = np.array([[5.0, 1.0], [5.0, 3.0], [5.0, 8.0]])   # perp offsets 1, 3, 8
    path = [[0, 0], [10, 0]]
    _c, narrow, _ = line_profile_from_locs(locs, path, width_nm=4.0, n_bins=10)   # ±2
    _c, wide, _ = line_profile_from_locs(locs, path, width_nm=8.0, n_bins=10)     # ±4
    assert narrow.sum() == 1                        # only the perp=1 loc (|y|<=2)
    assert wide.sum() == 2                          # perp=1 and perp=3 (|y|<=4)


def test_locs_outside_band_and_beyond_ends_excluded():
    locs = np.array([[5.0, 100.0], [-50.0, 0.0]])   # far off perpendicular / before start
    _c, counts, _ = line_profile_from_locs(locs, [[0, 0], [10, 0]], width_nm=4.0, n_bins=10)
    assert counts.sum() == 0


def test_polyline_arc_length_across_the_bend():
    # L-shape: (0,0)->(10,0)->(10,10); a loc near the second segment reads a
    # distance > 10 (past the first segment's length)
    locs = np.array([[10.0, 5.0]])                  # on the vertical segment, arc ~15
    c, counts, _ = line_profile_from_locs(locs, [[0, 0], [10, 0], [10, 10]], 4.0, n_bins=20)
    assert counts.sum() == 1
    hit = c[counts > 0][0]
    assert hit == pytest.approx(15.0, abs=1.0)


def test_result_is_viewport_independent():
    # the same locs + line give the same profile regardless of any external state
    rng = np.random.default_rng(0)
    locs = rng.uniform(-5, 15, size=(500, 2))
    a = line_profile_from_locs(locs, [[0, 0], [10, 0]], 6.0, n_bins=32)
    b = line_profile_from_locs(locs.copy(), [[0, 0], [10, 0]], 6.0, n_bins=32)
    assert np.array_equal(a[1], b[1])


def test_empty_or_degenerate_inputs():
    assert line_profile_from_locs(np.zeros((0, 2)), [[0, 0], [10, 0]], 4.0)[1].sum() == 0
    assert line_profile_from_locs(np.ones((5, 2)), [[1, 1]], 4.0)[0].size == 0       # 1 vertex
    assert line_profile_from_locs(np.ones((5, 2)), [[0, 0], [10, 0]], 0.0)[0].size == 0  # width 0


# ------------------------------------------------------------ width band

def test_band_polygon_spans_the_width():
    poly = band_polygon([[0, 0], [10, 0]], width_nm=4.0, step_nm=5.0)
    assert poly.shape[0] >= 4
    assert poly[:, 1].max() == pytest.approx(2.0)
    assert poly[:, 1].min() == pytest.approx(-2.0)


def test_band_polygon_empty_for_zero_width():
    assert band_polygon([[0, 0], [10, 0]], width_nm=0.0, step_nm=5.0).shape[0] == 0
