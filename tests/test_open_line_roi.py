"""Open-line ROIs (line / polyline / freehand line): 3-D vertices + ImageJ map."""

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from minflux_viewer.core.roi import (
    ROI_TYPES,
    RoiRecord,
    record_from_imagej,
    record_to_imagej,
    record_to_points,
)
from minflux_viewer.core.roi import _bounds as _core_bounds
from minflux_viewer.ui.roi_overlay import _bounds as _ov_bounds
from minflux_viewer.ui.roi_overlay import (
    points_to_3d,
    project_points,
    update_points_in_plane,
)


def test_new_types_registered():
    assert {"polyline", "freehand_line", "magnetic_lasso"} <= ROI_TYPES


def test_points_to_3d_all_share_depth():
    pts = points_to_3d([[10, 20], [30, 40]], "XY", 50)
    assert pts == [[10.0, 20.0, 50.0], [30.0, 40.0, 50.0]]


def test_project_points_per_plane():
    pts = [[10.0, 20.0, 50.0], [30.0, 40.0, 50.0]]
    assert project_points(pts, "XY") == [[10.0, 20.0], [30.0, 40.0]]
    assert project_points(pts, "XZ") == [[10.0, 50.0], [30.0, 50.0]]   # off-plane Z shown
    assert project_points(pts, "YZ") == [[20.0, 50.0], [40.0, 50.0]]


def test_update_points_preserves_depth_per_vertex():
    old = [[10.0, 20.0, 50.0], [30.0, 40.0, 50.0]]
    # drag vertex 0 in the XZ view (edits x and z; preserves y depth)
    moved = update_points_in_plane([[12.0, 70.0], [30.0, 50.0]], old, "XZ")
    assert moved[0] == [12.0, 20.0, 70.0]      # x->12, z->70, y(20) untouched
    assert moved[1] == [30.0, 40.0, 50.0]      # z unchanged → same


def test_polyline_imagej_roundtrip():
    roifile = pytest.importorskip("roifile")
    rec = RoiRecord.create("polyline", {"points": [[1, 2], [3, 4], [5, 6]], "closed": False},
                           coordinate_space="pixel")
    ij = record_to_imagej(rec)
    assert ij.roitype == roifile.ROI_TYPE.POLYLINE
    assert record_from_imagej(ij).type == "polyline"


def test_freehand_line_imagej_roundtrip():
    roifile = pytest.importorskip("roifile")
    rec = RoiRecord.create("freehand_line", {"points": [[1, 2], [3, 4], [5, 6]], "closed": False},
                           coordinate_space="pixel")
    ij = record_to_imagej(rec)
    assert ij.roitype == roifile.ROI_TYPE.FREELINE
    assert record_from_imagej(ij).type == "freehand_line"


def test_imagej_export_drops_depth():
    # a 3-D polyline still exports clean 2-D ImageJ coordinates
    rec = RoiRecord.create("polyline", {"points": [[1, 2, 9], [3, 4, 9]], "closed": False},
                           coordinate_space="pixel")
    ij = record_to_imagej(rec)
    coords = ij.coordinates()
    assert len(coords[0]) == 2


# --- robustness: a vertex-add (pyqtgraph segment click) can transiently leave a
#     mix of 2-D / 3-D vertices; geometry readers must not raise on that ---------
@pytest.mark.parametrize("fn", [_ov_bounds, _core_bounds])
def test_bounds_tolerates_mixed_2d_3d_points(fn):
    geom = {"points": [[0.0, 0.0, 5.0], [10.0, 4.0], [2.0, 8.0, 5.0]]}   # 3-D, 2-D, 3-D
    assert fn(geom) == (0.0, 0.0, 10.0, 8.0)                             # no ValueError


def test_record_to_points_tolerates_mixed_points():
    rec = RoiRecord.create(
        "freehand_line", {"points": [[0, 0, 5], [10, 4], [2, 8, 5]], "closed": False})
    pts = record_to_points(rec)
    assert pts.shape == (3, 2)
    assert pts[1].tolist() == [10.0, 4.0]


# Note: the polyline *drawing mechanism* (every left click registers a vertex even
# when the cursor is over the in-progress draft; right-click finishes and releases
# the tool so the button unpresses) is verified by a manual/runtime smoke rather
# than a unit test — driving RoiOverlayController.eventFilter with synthetic
# QMouseEvents segfaults under pytest-qt's QApplication. The logic lives in
# roi_overlay.py: the accumulating tools are handled before the _record_at gate,
# and _finish_polyline always calls _maybe_release_tool.
