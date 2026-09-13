"""Multi-point ROI: N markers filed, named, saved and deleted as ONE entry.

ImageJ's Multi-point. The grouping is the feature — a screenful of picks should
not become a screenful of Manager rows.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication

from minflux_viewer.core.roi import (
    MULTI_POINT_TYPES,
    ROI_TOOLS,
    ROI_TYPES,
    RoiRecord,
    RoiStore,
    record_to_points,
    roi_record_from_dict,
)


@pytest.fixture
def _qt_app():
    if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
        pytest.skip("No display available for Qt tests")
    return QApplication.instance() or QApplication(sys.argv)


def _marks(n=4):
    return [[100.0 * i, 50.0 * i, 10.0 * i] for i in range(1, n + 1)]


def _record(marks=None):
    return RoiRecord.create("points", {"points": marks or _marks()})


# ------------------------------------------------------------------- the model
def test_points_is_a_record_type_and_multi_point_is_only_a_tool():
    """Same discipline as the rotated-rectangle variants: a drawing tool need
    not be a record type. ``multi_point`` files a ``points`` record."""
    assert "points" in ROI_TYPES
    assert "points" in MULTI_POINT_TYPES
    assert "multi_point" in ROI_TOOLS
    assert "multi_point" not in ROI_TYPES


def test_every_marker_keeps_its_own_three_d_coordinate():
    rec = _record()
    pts = rec.geometry["points"]
    assert all(len(p) == 3 for p in pts)
    assert [p[2] for p in pts] == [10.0, 20.0, 30.0, 40.0]


def test_record_to_points_reads_a_multi_point():
    """The shared vertex-list accessor already serves it: same geometry key."""
    arr = record_to_points(_record())
    assert arr.shape[0] == 4


def test_the_store_names_it_as_one_entry():
    store = RoiStore()
    store.add(_record())
    store.add(_record())
    assert [r.name for r in store.records] == ["points-1", "points-2"]
    assert len(store.records) == 2       # not 8


def test_it_round_trips_through_the_native_json_shape():
    rec = _record()
    payload = {
        "id": rec.id, "name": "picks", "type": "points",
        "geometry": rec.geometry, "coordinate_space": "plot",
    }
    back = roi_record_from_dict(payload)
    assert back.type == "points"
    assert back.geometry["points"] == rec.geometry["points"]


# ------------------------------------------------------------- ImageJ export
def test_imagej_export_keeps_every_marker():
    """ImageJ's POINT type already holds N coordinates, so a multi-point
    round-trips exactly rather than being flattened to one marker."""
    roifile = pytest.importorskip("roifile")
    from minflux_viewer.core.roi import record_to_imagej

    rec = _record(_marks(5))
    rec.name = "picks"
    roi = record_to_imagej(rec)
    assert roi.roitype == roifile.ROI_TYPE.POINT
    assert len(roi.coordinates()) == 5


# ------------------------------------------------------------------- geometry
def test_bounds_cover_every_marker():
    from minflux_viewer.ui.roi_overlay import _bounds

    x, y, w, h = _bounds({"points": [[0.0, 0.0, 5.0], [300.0, 120.0, 9.0]]})
    assert (x, y, w, h) == (0.0, 0.0, 300.0, 120.0)


def test_status_read_out_leads_with_the_count():
    from minflux_viewer.ui.roi_overlay import RoiOverlayController

    text = RoiOverlayController._roi_status_text("points", {"points": _marks(3)})
    assert text.startswith("Points  3 points")
    assert "Bounding Box" in text


def test_an_empty_multi_point_says_so_rather_than_reporting_a_box():
    from minflux_viewer.ui.roi_overlay import RoiOverlayController

    assert "empty" in RoiOverlayController._roi_status_text("points", {"points": []})


# -------------------------------------------------------------- the drawn item
def test_the_item_is_one_plot_item_carrying_every_marker(_qt_app):
    """The controller keys exactly one item per record id, so a multi-point
    cannot be N marker items."""
    from minflux_viewer.ui.roi_overlay import MultiPointItem

    item = MultiPointItem([[0.0, 0.0], [10.0, 20.0], [30.0, 5.0]])
    xs, ys = item.getData()
    assert list(xs) == [0.0, 10.0, 30.0]
    assert list(ys) == [0.0, 20.0, 5.0]

    item.set_points([])
    xs, _ys = item.getData()
    assert xs is None or len(xs) == 0


def test_markers_stay_the_same_size_on_screen(_qt_app):
    """pxMode -- a marker is a marker at any zoom, not a shape that grows."""
    from minflux_viewer.ui.roi_overlay import MultiPointItem

    assert MultiPointItem([[0.0, 0.0]]).opts["pxMode"] is True


# ------------------------------------------------------------- the projection
def test_markers_project_per_view_plane():
    """A set drawn in XY must show its true Z when the view flips, exactly as a
    single point does."""
    from minflux_viewer.ui.roi_overlay import project_points

    marks = [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
    assert project_points(marks, "XY") == [[10.0, 20.0], [40.0, 50.0]]
    assert project_points(marks, "XZ") == [[10.0, 30.0], [40.0, 60.0]]
    assert project_points(marks, "YZ") == [[20.0, 30.0], [50.0, 60.0]]


# ---------------------------------------------------------------- the grouping
def test_grouping_files_one_record_and_ungrouped_files_many(_qt_app):
    """The two menu routes are kept apart deliberately: one files N independent
    markers, the other one multi-point. Silently changing what the existing
    entry produces would be worse than offering both."""
    from minflux_viewer.ui.roi_overlay import RoiOverlayController

    ctrl = RoiOverlayController.__new__(RoiOverlayController)
    ctrl.store = RoiStore()
    ctrl._session_points = [
        RoiRecord.create("point", {"point": [float(i), float(i * 2), float(i * 3)]})
        for i in range(1, 4)
    ]
    ctrl._session_items = {}
    ctrl._active_session_point_id = None
    ctrl.plot_item = None
    ctrl._record_kwargs = lambda: {}
    ctrl._normalize_record = lambda rec: rec
    ctrl._show_manager_if_needed = lambda: None
    ctrl._emit_status = lambda _text: None

    ctrl._add_session_points_as_multi_point()

    assert len(ctrl.store.records) == 1
    rec = ctrl.store.records[0]
    assert rec.type == "points"
    assert rec.geometry["points"] == [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]]
    assert ctrl._session_points == []          # the pending set is consumed


def test_grouping_nothing_is_a_no_op(_qt_app):
    from minflux_viewer.ui.roi_overlay import RoiOverlayController

    ctrl = RoiOverlayController.__new__(RoiOverlayController)
    ctrl.store = RoiStore()
    ctrl._session_points = []
    ctrl._session_items = {}
    ctrl._active_session_point_id = None
    ctrl._add_session_points_as_multi_point()
    assert ctrl.store.records == []


# ------------------------------------------------------------------- the tool
def test_the_point_toolbar_button_hosts_both_members():
    from minflux_viewer.ui.main_window import _POINT_FAMILY

    tools = [tool for _label, tool, _icon, _rot in _POINT_FAMILY]
    assert tools == ["point", "multi_point"]


def test_the_multi_point_icon_exists():
    from minflux_viewer import resource_path

    assert os.path.exists(resource_path("icons/multipoint.png"))
