"""Tests for the pure 3-D ROI intersection-of-extrusions core."""

import numpy as np
import pytest

from minflux_viewer.core.roi_3d import Roi3DConstraint, plane_axes, roi3d_mask


def _rect(plane, x, y, w, h):
    return Roi3DConstraint(plane, "rectangle", {"bounds": [x, y, w, h]})


def test_plane_axes_mapping():
    assert plane_axes("XY") == (0, 1)
    assert plane_axes("xz") == (0, 2)
    assert plane_axes("YZ") == (2, 1)
    with pytest.raises(ValueError):
        plane_axes("QQ")


def test_no_constraints_selects_nothing():
    pts = np.random.default_rng(0).uniform(-10, 10, size=(50, 3))
    mask = roi3d_mask(pts, [])
    assert mask.shape == (50,)
    assert not mask.any()


def test_single_xy_rectangle_is_an_infinite_z_prism():
    # points spread across z; XY rect should ignore z entirely.
    pts = np.array(
        [
            [1.0, 1.0, -500.0],   # inside XY, extreme z -> still in (prism)
            [1.0, 1.0, 900.0],
            [9.0, 9.0, 0.0],      # outside XY rect
        ]
    )
    mask = roi3d_mask(pts, [_rect("XY", 0, 0, 2, 2)])
    assert mask.tolist() == [True, True, False]


def test_intersection_of_xy_and_xz_makes_a_box():
    # XY rect bounds x,y in [0,2]; XZ rect bounds x in [0,2], z in [0,2].
    xy = _rect("XY", 0, 0, 2, 2)      # bounds (x, y, w, h)
    xz = _rect("XZ", 0, 0, 2, 2)      # bounds (x, z, w, h)  -> x in [0,2], z in [0,2]
    pts = np.array(
        [
            [1.0, 1.0, 1.0],   # in both -> selected
            [1.0, 1.0, 5.0],   # z outside XZ -> rejected
            [1.0, 5.0, 1.0],   # y outside XY -> rejected
            [5.0, 1.0, 1.0],   # x outside both -> rejected
        ]
    )
    mask = roi3d_mask(pts, [xy, xz])
    assert mask.tolist() == [True, False, False, False]


def test_yz_rectangle_uses_z_horizontal_y_vertical():
    # YZ pane draws Z on the horizontal axis, Y on the vertical (conv-3D convention).
    # bounds = (z, y, w=Δz, h=Δy): z in [10,20], y in [0,4].
    yz = _rect("YZ", 10, 0, 10, 4)
    pts = np.array(
        [
            [999.0, 2.0, 15.0],  # y in [0,4], z in [10,20] -> selected (x ignored)
            [0.0, 2.0, 25.0],    # z outside -> rejected
            [0.0, 9.0, 15.0],    # y outside -> rejected
        ]
    )
    mask = roi3d_mask(pts, [yz])
    assert mask.tolist() == [True, False, False]


def test_polygon_constraint_matches_region_mask():
    # a triangle in XY
    poly = Roi3DConstraint("XY", "polygon",
                           {"points": [[0, 0], [10, 0], [0, 10]]})
    pts = np.array([[1.0, 1.0, 3.0], [8.0, 8.0, 0.0]])
    mask = roi3d_mask(pts, [poly])
    assert mask.tolist() == [True, False]


def test_two_d_points_promoted_to_z_zero():
    pts = np.array([[1.0, 1.0], [9.0, 9.0]])
    mask = roi3d_mask(pts, [_rect("XY", 0, 0, 2, 2)])
    assert mask.tolist() == [True, False]
