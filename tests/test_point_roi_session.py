"""Point ROIs are pending 'session points' until explicitly added to the Manager."""

from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QWidget

from minflux_viewer.core.roi import RoiRecord, RoiStore
from minflux_viewer.ui.roi_overlay import RoiOverlayController, _bounds


@pytest.fixture(scope="module")
def _app():
    yield QApplication.instance() or QApplication([])


class _Owner(QWidget):
    """Minimal stand-in for a render/scatter window owning a ROI overlay."""

    def __init__(self):
        super().__init__()
        self._state = SimpleNamespace(prefs={"plot": {"roi_color": "Yellow"}}, datasets=[])

    def roi_view_plane(self):
        return "XY"

    def roi_depth_center(self):
        return 50.0

    def normalize_roi_record(self, record):
        return record


def _make_controller(_app):
    plot = pg.PlotWidget()
    store = RoiStore()
    owner = _Owner()
    ctrl = RoiOverlayController(store, owner, plot, plot.getPlotItem())
    return ctrl, store


def test_bounds_accepts_3d_point():
    # regression: the Labels checkbox crashed unpacking a 3-element point
    assert _bounds({"point": [10.0, 20.0, 30.0]}) == (10.0, 20.0, 0.0, 0.0)


def test_drawn_point_is_session_not_in_manager(_app):
    ctrl, store = _make_controller(_app)
    ctrl._commit_point((100.0, 200.0))
    assert len(ctrl._session_points) == 1
    assert store.records == []          # NOT silently added to the Manager


def test_add_single_session_point_to_manager(_app):
    ctrl, store = _make_controller(_app)
    ctrl._commit_point((1.0, 2.0))
    ctrl._commit_point((3.0, 4.0))
    first = ctrl._session_points[0]
    ctrl._add_hit_to_manager("session", first)
    assert len(store.records) == 1
    assert store.records[0].id == first.id
    assert len(ctrl._session_points) == 1   # the other one stays pending


def test_add_all_session_points_to_manager(_app):
    ctrl, store = _make_controller(_app)
    ctrl._commit_point((1.0, 2.0))
    ctrl._commit_point((3.0, 4.0))
    ctrl._commit_point((5.0, 6.0))
    ctrl._add_all_session_points_to_manager()
    assert len(store.records) == 3
    assert ctrl._session_points == []
    assert ctrl._session_items == {}


def test_delete_session_point(_app):
    ctrl, store = _make_controller(_app)
    ctrl._commit_point((1.0, 2.0))
    rec = ctrl._session_points[0]
    ctrl._delete_hit_roi("session", rec)
    assert ctrl._session_points == []
    assert store.records == []


def test_session_point_carries_depth_center(_app):
    ctrl, _store = _make_controller(_app)
    ctrl._commit_point((100.0, 200.0))
    pt = ctrl._session_points[0].geometry["point"]
    assert pt == [100.0, 200.0, 50.0]    # z = roi_depth_center()


# --------------------------------------------- ending a multi-point session
# A multi-point session ends either with the finishing right-click or by
# unchecking the toolbar button. Both must produce the same thing: ONE `points`
# ROI holding every marker placed. Before this, unchecking left the markers on
# screen unfiled, so a set the user had just placed never reached the Manager.
def test_unchecking_the_multi_point_tool_files_the_set_as_one_roi(_app):
    ctrl, store = _make_controller(_app)
    store.set_tool("multi_point")
    for pos in ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)):
        ctrl._commit_point(pos)
    assert len(ctrl._session_points) == 3
    assert store.records == []              # still pending, nothing filed yet

    store.set_tool(None)                    # the toolbar button is unchecked

    assert len(store.records) == 1          # ONE entry, not three
    rec = store.records[0]
    assert rec.type == "points"
    assert len(rec.geometry["points"]) == 3
    assert ctrl._session_points == []       # nothing left pending


def test_switching_from_multi_point_to_another_tool_also_files_the_set(_app):
    ctrl, store = _make_controller(_app)
    store.set_tool("multi_point")
    ctrl._commit_point((7.0, 8.0))
    store.set_tool("rectangle")
    assert len(store.records) == 1
    assert store.records[0].type == "points"


def test_unchecking_the_plain_point_tool_leaves_its_markers_pending(_app):
    """`point` and `multi_point` differ only in how the set is FILED: the plain
    tool leaves each pick its own pending marker, so ending it must NOT group."""
    ctrl, store = _make_controller(_app)
    store.set_tool("point")
    ctrl._commit_point((1.0, 2.0))
    ctrl._commit_point((3.0, 4.0))
    store.set_tool(None)
    assert store.records == []
    assert len(ctrl._session_points) == 2


def test_right_click_route_does_not_double_file_on_release(_app):
    """The right-click finish empties the session before releasing the tool, so
    the deactivate hook must find nothing left to file."""
    ctrl, store = _make_controller(_app)
    store.set_tool("multi_point")
    ctrl._commit_point((1.0, 2.0))
    ctrl._commit_point((3.0, 4.0))
    ctrl._add_session_points_as_multi_point()   # what the right-click does
    assert len(store.records) == 1
    store.set_tool(None)                        # then the tool is released
    assert len(store.records) == 1              # not two
