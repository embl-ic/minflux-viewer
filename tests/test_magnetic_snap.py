"""Magnetic-lasso pixel snapping (mean-shift to local density centre)."""

import numpy as np

from minflux_viewer.core.magnetic_snap import snap_density_centroid


def test_snaps_to_local_peak():
    ys, xs = np.mgrid[0:41, 0:41]
    img = np.exp(-(((xs - 25) ** 2 + (ys - 20) ** 2) / (2 * 3.0 ** 2)))
    c, r = snap_density_centroid(img, 22, 18, window=15, iterations=6)
    assert abs(c - 25) < 1.0 and abs(r - 20) < 1.0


def test_flat_region_returns_input():
    img = np.zeros((20, 20))
    assert snap_density_centroid(img, 5, 5, window=7) == (5.0, 5.0)


def test_lateral_only_moves_perpendicular_to_travel():
    # horizontal ridge at row=10; travelling along x → snap only laterally (in row)
    img = np.zeros((21, 41))
    img[10, :] = 1.0
    img[9, :] = img[11, :] = 0.5
    c, r = snap_density_centroid(img, 20, 7, window=9, iterations=5, lateral_dir=(1, 0))
    assert abs(r - 10) < 0.6        # pulled onto the ridge centreline
    assert abs(c - 20) < 0.5        # along-travel column basically unchanged


def test_empty_image_safe():
    assert snap_density_centroid(np.zeros((0, 0)), 1, 1) == (1.0, 1.0)
