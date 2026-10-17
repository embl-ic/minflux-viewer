"""Shift/Alt shape operations through the overlay's real mouse-event path."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QKeyEvent, QMouseEvent
from PyQt6.QtWidgets import QWidget

from minflux_viewer.core.roi import RoiRecord, RoiStore
from minflux_viewer.ui.roi.roi_overlay import CompositeROI, RoiOverlayController, item_to_geometry

SHIFT = Qt.KeyboardModifier.ShiftModifier
ALT = Qt.KeyboardModifier.AltModifier
NONE = Qt.KeyboardModifier.NoModifier


@pytest.fixture(scope="module")
def _app(qapp):
    yield qapp


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
    plot.resize(400, 400)
    ctrl = RoiOverlayController(RoiStore(), _Owner(), plot, plot.getPlotItem())
    ctrl._queue_selection_update = lambda record: None      # no dataset to select in
    return ctrl


def _mouse(kind, mods, button=Qt.MouseButton.LeftButton):
    buttons = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseButtonRelease else button
    return QMouseEvent(kind, QPointF(1, 1), QPointF(1, 1), button, buttons, mods)


def _drag(ctrl, tool, start, end, mods=NONE):
    """Press–move–release through eventFilter; the data positions are fed in
    because the pixel→data mapping depends on axis layout, not on what is tested."""
    ctrl.store.set_tool(tool)
    feed = iter([start, end, end])
    ctrl._event_to_view = lambda _e: next(feed)
    target = ctrl._event_target()
    assert ctrl.eventFilter(target, _mouse(QEvent.Type.MouseButtonPress, mods)) is True
    ctrl.eventFilter(target, _mouse(QEvent.Type.MouseMove, mods))
    ctrl.eventFilter(target, _mouse(QEvent.Type.MouseButtonRelease, mods))


def _rect_draft(ctrl, x, y, w, h):
    ctrl._set_draft(RoiRecord.create("rectangle", {"bounds": [x, y, w, h]}))


def test_shift_drag_adds_to_the_selection(_app):
    ctrl = _ctrl(_app)
    _rect_draft(ctrl, 0, 0, 20, 20)
    _drag(ctrl, "rectangle", (10.0, 10.0), (30.0, 30.0), SHIFT)
    assert ctrl.draft.type == "polygon"                      # the union outline
    assert ctrl._shape_op is None and ctrl._shape_op_preview is None


def test_alt_drag_inside_cuts_a_hole_instead_of_editing(_app):
    ctrl = _ctrl(_app)
    _rect_draft(ctrl, 0, 0, 40, 40)
    # the press lands ON the selection: without Alt that would start an edit
    _drag(ctrl, "rectangle", (10.0, 10.0), (20.0, 20.0), ALT)
    assert ctrl.draft.type == "composite"
    assert isinstance(ctrl.draft_item, CompositeROI)
    assert not ctrl._point_hits_record((15.0, 15.0), ctrl.draft, 0.1)   # in the hole
    assert ctrl._point_hits_record((5.0, 5.0), ctrl.draft, 0.1)


def test_a_plain_drag_replaces_the_selection(_app):
    ctrl = _ctrl(_app)
    _rect_draft(ctrl, 0, 0, 20, 20)
    _drag(ctrl, "rectangle", (50.0, 50.0), (60.0, 60.0))
    assert ctrl.draft.type == "rectangle"
    assert ctrl.draft.geometry["bounds"] == pytest.approx([50, 50, 10, 10])


def test_a_line_tool_ignores_the_modifier(_app):
    ctrl = _ctrl(_app)
    _rect_draft(ctrl, 0, 0, 20, 20)
    _drag(ctrl, "line", (50.0, 50.0), (60.0, 60.0), SHIFT)
    assert ctrl.draft.type == "line" and ctrl._shape_op is None


def test_alt_removing_everything_clears_the_selection(_app):
    ctrl = _ctrl(_app)
    _rect_draft(ctrl, 10, 10, 5, 5)
    _drag(ctrl, "rectangle", (0.0, 0.0), (40.0, 40.0), ALT)
    assert ctrl.draft is None


def test_shift_polygon_combines_when_closed_and_escape_cancels(_app):
    ctrl = _ctrl(_app)
    _rect_draft(ctrl, 0, 0, 20, 20)
    ctrl.store.set_tool("polygon")
    target = ctrl._event_target()
    for i, pos in enumerate([(30.0, 0.0), (40.0, 0.0), (35.0, 10.0)]):
        ctrl._event_to_view = lambda _e, _p=pos: _p
        ctrl.eventFilter(target, _mouse(QEvent.Type.MouseButtonPress, SHIFT if i == 0 else NONE))
    assert ctrl._shape_op is not None
    ctrl._finish_polygon()
    assert ctrl.draft.type == "composite"                    # rectangle + separate triangle

    ctrl2 = _ctrl(_app)
    _rect_draft(ctrl2, 0, 0, 20, 20)
    ctrl2.store.set_tool("polygon")
    ctrl2._event_to_view = lambda _e: (30.0, 0.0)
    ctrl2.eventFilter(ctrl2._event_target(), _mouse(QEvent.Type.MouseButtonPress, SHIFT))
    assert ctrl2._shape_op is not None
    ctrl2.eventFilter(ctrl2._event_target(),
                      QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, NONE))
    assert ctrl2._shape_op is None and ctrl2._shape_op_preview is None


def test_a_moved_composite_moves_its_rings(_app):
    ctrl = _ctrl(_app)
    _rect_draft(ctrl, 0, 0, 40, 40)
    _drag(ctrl, "rectangle", (10.0, 10.0), (20.0, 20.0), ALT)
    rings = ctrl.draft.geometry["rings"]
    ctrl.draft_item.setPos(5.0, -2.0)
    moved = item_to_geometry("composite", ctrl.draft_item, prev=ctrl.draft.geometry)["rings"]
    for a, b in zip(rings, moved):
        assert np.allclose(np.asarray(b) - np.asarray(a), [5.0, -2.0])


@pytest.mark.parametrize("first_tool, first, second_tool, second", [
    ("rectangle", ((0.0, 0.0), (25.0, 20.0)), "oval", ((15.0, 5.0), (45.0, 35.0))),
    ("oval", ((15.0, 5.0), (45.0, 35.0)), "rectangle", ((0.0, 0.0), (25.0, 20.0))),
])
def test_shift_combining_rectangle_and_oval_is_drawn_the_same_in_either_order(
        _app, first_tool, first, second_tool, second):
    """The reported case: the result must not depend on which shape came first,
    and a flattened arc must not sprout a handle on every segment."""
    ctrl = _ctrl(_app)
    _drag(ctrl, first_tool, *first)
    _drag(ctrl, second_tool, *second, SHIFT)
    assert ctrl.draft.type == "composite"
    assert isinstance(ctrl.draft_item, CompositeROI)
    assert not getattr(ctrl.draft_item, "handles", [])
