"""Point/ROI third-dimension placement: out-of-plane axis = centre of view range."""

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from minflux_viewer.ui.roi_overlay import (
    point_to_3d,
    project_point,
    update_point_in_plane,
)


def test_point_to_3d_per_plane():
    # drawn at in-plane (100, 200), depth axis pinned to 50
    assert point_to_3d((100, 200), "XY", 50) == [100.0, 200.0, 50.0]   # depth = Z
    assert point_to_3d((100, 200), "XZ", 50) == [100.0, 50.0, 200.0]   # depth = Y
    assert point_to_3d((100, 200), "YZ", 50) == [50.0, 100.0, 200.0]   # depth = X


def test_point_to_3d_no_plane_or_depth():
    assert point_to_3d((1, 2), None, 50) == [1.0, 2.0]       # owner w/o orientation
    assert point_to_3d((1, 2), "XY", None) == [1.0, 2.0, 0.0]  # 2-D dataset (no depth)


def test_project_point_round_trips_drawing_plane():
    # A point drawn in XY at z=50 must show at (x,y) in XY, (x,z) in XZ, (y,z) in YZ
    p = point_to_3d((100, 200), "XY", 50)
    assert project_point(p, "XY") == (100.0, 200.0)
    assert project_point(p, "XZ") == (100.0, 50.0)    # was the strange ~2.2e4 bug
    assert project_point(p, "YZ") == (200.0, 50.0)


def test_project_legacy_2d_point():
    assert project_point([7.0, 8.0], "XZ") == (7.0, 8.0)


def test_drag_in_other_plane_preserves_depth():
    # drawn in XY (z=50); drag in XZ moves x and z, leaves y untouched
    p = point_to_3d((100, 200), "XY", 50)        # [100, 200, 50]
    moved = update_point_in_plane((130, 70), p, "XZ")
    assert moved == [130.0, 200.0, 70.0]          # x->130, z->70, y preserved


def test_drag_legacy_2d_point():
    assert update_point_in_plane((5, 6), [1.0, 2.0], "XY") == [5.0, 6.0]
