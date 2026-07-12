"""Line ROIs mark their END vertex with a larger oval handle (direction cue:
begin → end), while start / intermediate vertices keep the square edit handle.

The end handle is restyled **in place** (its local path swapped to an ellipse);
its sip/C++ type is never reassigned — a ``handle.__class__`` swap crashed the app
under the create/destroy churn of a live draft drag."""

from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg
from pyqtgraph.graphicsItems.ROI import Handle
from PyQt6.QtWidgets import QApplication, QWidget

from minflux_viewer.core.roi import RoiRecord, RoiStore
from minflux_viewer.ui.roi_overlay import RoiOverlayController, _mark_line_end_handle


@pytest.fixture(scope="module")
def _app():
    yield QApplication.instance() or QApplication([])


class _Owner(QWidget):
    def __init__(self):
        super().__init__()
        self._state = SimpleNamespace(prefs={"plot": {"roi_color": "Yellow"}}, datasets=[])

    def roi_view_plane(self):
        return "XY"

    def roi_depth_center(self):
        return 0.0

    def normalize_roi_record(self, record):
        return record


def _ctrl(_app):
    plot = pg.PlotWidget()
    return RoiOverlayController(RoiStore(), _Owner(), plot, plot.getPlotItem())


def _is_end(h) -> bool:
    return bool(getattr(h, "_is_line_end_marker", False))


@pytest.mark.parametrize("roi_type,pts", [
    ("line", [[0, 0], [10, 0]]),
    ("polyline", [[0, 0], [5, 5], [10, 0]]),
    ("freehand_line", [[0, 0], [5, 5], [10, 0]]),
])
def test_open_line_marks_end_vertex(_app, roi_type, pts):
    ctrl = _ctrl(_app)
    item = ctrl._make_item(RoiRecord.create(roi_type, {"points": pts, "closed": False}))
    handles = [h["item"] for h in item.handles]
    assert _is_end(handles[-1])                         # end vertex = oval marker
    assert not any(_is_end(h) for h in handles[:-1])    # begin + middle = square boxes


def test_end_handle_is_still_a_draggable_free_handle(_app):
    ctrl = _ctrl(_app)
    item = ctrl._make_item(RoiRecord.create("line", {"points": [[0, 0], [10, 0]], "closed": False}))
    h = item.handles[-1]["item"]
    # in-place restyle — never a __class__ swap: still an ordinary pyqtgraph Handle
    assert type(h) is Handle
    assert h.typ == "f"                                 # ordinary free handle — no rotation
    assert _is_end(h)


def test_closed_polyline_is_not_marked(_app):
    ctrl = _ctrl(_app)
    item = ctrl._make_item(RoiRecord.create(
        "polyline", {"points": [[0, 0], [5, 5], [10, 0]], "closed": True}))
    assert not any(_is_end(h["item"]) for h in item.handles)  # a loop has no direction


def test_angle_roi_end_corner_is_marked(_app):
    ctrl = _ctrl(_app)
    item = ctrl._make_item(RoiRecord.create(
        "angle", {"points": [[0, 0], [5, 5], [10, 0]], "closed": False}))
    handles = [h["item"] for h in item.handles]
    assert _is_end(handles[-1])                         # end corner (C) = oval
    assert not any(_is_end(h) for h in handles[:-1])    # A + apex B stay square


def test_mark_helper_is_idempotent_and_safe(_app):
    line = pg.PolyLineROI([[0, 0], [5, 5], [10, 0]], closed=False, movable=True)
    end = line.handles[-1]["item"]
    _mark_line_end_handle(line)
    r_after_first = float(end.radius)
    _mark_line_end_handle(line)                         # second call is a no-op
    assert _is_end(end) and float(end.radius) == r_after_first
    # a single-vertex / empty item is a no-op, not an error
    _mark_line_end_handle(pg.PolyLineROI([[0, 0]], closed=False))


def test_vertex_add_keeps_line_geometry_homogeneous(_app):
    """A pyqtgraph segment-click adds a vertex → the item gains a handle. Rebuilding
    the draft geometry must re-derive a **consistent** 3-D point list, never a mix
    of 2-D / 3-D vertices (which crashed np.asarray in _bounds when the user then
    clicked / arrow-moved the freehand line)."""
    class _DepthOwner(_Owner):
        def roi_depths_at(self, pts):
            return [5.0] * len(pts)                     # 3-D depths (a render view)

    plot = pg.PlotWidget()
    ctrl = RoiOverlayController(RoiStore(), _DepthOwner(), plot, plot.getPlotItem())
    ctrl._set_draft(RoiRecord.create(
        "freehand_line", {"points": [[float(i), float(i), 5.0] for i in range(6)],
                          "closed": False}))
    # simulate the item having gained a vertex (now 7 handles vs 6 in the geometry)
    item = pg.PolyLineROI([[float(i), float(i)] for i in range(7)], closed=False)
    geom = ctrl._open_line_geometry_from_item(item, ctrl.draft.geometry, "freehand_line")
    assert len(geom["points"]) == 7
    assert {len(p) for p in geom["points"]} == {3}      # all 3-D — homogeneous
    import numpy as np
    np.asarray(geom["points"], dtype=float)             # must not raise
