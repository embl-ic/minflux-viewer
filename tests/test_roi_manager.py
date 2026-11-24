"""ROI Manager list behaviour:
1. arrow keys navigate the list and switch the active/selected ROI (previously a
   selection-triggered `changed` signal rebuilt the whole list mid-navigation);
2. a ROI selected in the Manager is drawn in a distinct color in the views so it
   stands out from the other (Show-all) ROIs;
3. the other direction, as in Fiji: pressing a ROI (or its label) in a view
   selects its Manager entry and brings it into view.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtWidgets import QWidget

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.roi import RoiRecord, RoiStore
from minflux_viewer.ui.roi.roi_overlay import RoiOverlayController


@pytest.fixture(scope="module")
def _app(qapp):
    return qapp


def _rect(name, x=0.0, stroke="#ffff00"):
    return RoiRecord.create("rectangle", {"bounds": [float(x), 0.0, 10.0, 10.0]},
                            name=name, coordinate_space="pixel", stroke_color=stroke)


# ------------------------------------------------------------- arrow navigation

def test_selecting_does_not_rebuild_manager_list(_app):
    from minflux_viewer.ui.roi.roi_manager import RoiManagerWindow

    st = AppState()
    for i in range(3):
        st.rois.add(_rect(f"r{i}", x=i * 20))
    st.rois.deselect()
    w = RoiManagerWindow(st)
    # Spy by connecting fresh slots to the same signals the manager listens to.
    changed, selection = [], []
    st.rois.changed.connect(lambda: changed.append(1))
    st.rois.selection_changed.connect(lambda: selection.append(1))

    st.rois.select([st.rois.records[1].id])
    assert selection and not changed          # selection-only → no list rebuild
    w.close()


def test_arrow_keys_navigate_and_switch_active_roi(_app):
    from PyQt6.QtTest import QTest

    from minflux_viewer.ui.roi.roi_manager import RoiManagerWindow

    st = AppState()
    for i in range(3):
        st.rois.add(_rect(f"r{i}", x=i * 20))
    st.rois.deselect()
    w = RoiManagerWindow(st)
    w.show()
    QTest.qWaitForWindowExposed(w)
    w._list.setFocus()
    w._list.setCurrentRow(0)
    _app.processEvents()
    assert st.rois.selected_ids == [st.rois.records[0].id]

    QTest.keyClick(w._list, Qt.Key.Key_Down)
    _app.processEvents()
    assert st.rois.selected_ids == [st.rois.records[1].id]   # moved to next ROI

    QTest.keyClick(w._list, Qt.Key.Key_Down)                 # a *second* step still works
    _app.processEvents()
    assert st.rois.selected_ids == [st.rois.records[2].id]

    QTest.keyClick(w._list, Qt.Key.Key_Up)
    _app.processEvents()
    assert st.rois.selected_ids == [st.rois.records[1].id]
    w.close()


# ------------------------------------------------------------- selection color

def _ctrl(_app, store=None):
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

    plot = pg.PlotWidget()
    return RoiOverlayController(store or RoiStore(), _Owner(), plot, plot.getPlotItem())


def test_selected_stored_roi_uses_manager_highlight_color(_app):
    ctrl = _ctrl(_app)
    store = ctrl.store
    rec = _rect("r", stroke="#ffff00")
    store.add(rec)
    store.set_show_all(True)

    store.select([rec.id])
    item = ctrl.items[rec.id]
    assert item.pen.color().name().lower() == ctrl.MANAGER_SELECT_COLOR.lower()

    store.deselect()                             # still shown (Show-all) but not selected
    item = ctrl.items[rec.id]
    assert item.pen.color().name().lower() == "#ffff00"   # back to its own color


# --------------------------------------------- convert rebuilds the displayed item

def test_converting_stored_roi_rebuilds_the_displayed_item(_app):
    """Converting a *displayed stored* ROI (right-click Convert to…) must rebuild the
    view item to the new type. Regression: on the cropped-view carried ROI, convert
    silently changed the record type but the old shape lingered because `refresh`
    reused the cached item by id (the record object was replaced by store.update)."""
    from minflux_viewer.core.roi_convert import convert_roi

    ctrl = _ctrl(_app)
    rec = RoiRecord.create("freehand", {"points": [[0, 0], [100, 0], [100, 100], [0, 100]],
                                        "closed": True}, coordinate_space="pixel")
    ctrl.store.add(rec)
    ctrl.store.set_show_all(True)
    assert type(ctrl.items[rec.id]).__name__ == "FilledPolyLineROI"

    ctrl._replace_hit("stored", rec, convert_roi(rec, "oval"))    # freehand → oval
    rid = ctrl.store.records[0].id
    assert ctrl.store.records[0].type == "oval"
    assert type(ctrl.items[rid]).__name__ == "FilledEllipseROI"   # rebuilt, not the stale poly

    rec2 = ctrl.store.records[0]
    ctrl._replace_hit("stored", rec2, convert_roi(rec2, "rectangle"))  # oval → rectangle
    assert type(ctrl.items[ctrl.store.records[0].id]).__name__ == "FilledRectROI"


def test_stored_roi_live_drag_survives_refresh(_app):
    """A live-edit drag of a stored ROI must NOT be reset by an unrelated refresh —
    the record object is unchanged, so the item is preserved (only a store.update
    replaces the record and triggers a rebuild)."""
    ctrl = _ctrl(_app)
    rec = _rect("r")
    ctrl.store.add(rec)
    ctrl.store.set_show_all(True)
    item = ctrl.items[rec.id]
    item.setPos(pg.Point(500.0, 500.0))          # user drags the displayed copy
    ctrl.store.selection_changed.emit()          # some unrelated refresh
    assert ctrl.items[rec.id] is item            # same item — drag preserved


# ------------------------------- restore: the stored record wins until Update
# ImageJ's RoiManager shows a *clone* of the stored ROI (`restore` →
# `imp.setRoi((Roi)roi.clone())`) and only `update` writes the edited clone back
# (`rois.set(index, (Roi)imp.getRoi().clone())`). So acting on the list — picking
# another entry, or picking the selected entry again — re-runs `restore` and the
# uncommitted move is discarded.

def _bounds_x(record):
    return float(record.geometry["bounds"][0])


def test_selecting_another_roi_discards_the_uncommitted_edit(_app):
    """Show-all keeps a dragged ROI on screen; selecting another entry must redraw
    it from its (untouched) record instead of leaving the uncommitted shape."""
    ctrl = _ctrl(_app)
    store = ctrl.store
    a, b = _rect("a"), _rect("b", x=100.0)
    store.add(a)
    store.add(b)
    store.set_show_all(True)
    store.select([a.id])

    item = ctrl.items[a.id]
    item.setPos(pg.Point(500.0, 500.0))          # user drags the displayed copy
    assert _bounds_x(store.records[0]) == pytest.approx(0.0)   # record untouched

    store.select([b.id])                          # act on the list → restore
    restored = ctrl.items[a.id]
    assert restored is not item                   # rebuilt from the record
    assert restored.pos().x() == pytest.approx(0.0)
    assert _bounds_x(store.records[0]) == pytest.approx(0.0)


def test_reclicking_the_selected_entry_restores_it(_app):
    """Clicking the already-selected row restores the stored geometry. Qt emits no
    selection signal for that click, so the Manager catches it on the viewport."""
    from PyQt6.QtTest import QTest

    from minflux_viewer.ui.roi.roi_manager import RoiManagerWindow

    st = AppState()
    ctrl = _ctrl(_app, store=st.rois)
    rec = _rect("r")
    st.rois.add(rec)
    st.rois.set_show_all(True)
    st.rois.select([rec.id])
    w = RoiManagerWindow(st)
    w.show()
    QTest.qWaitForWindowExposed(w)
    _app.processEvents()

    item = ctrl.items[rec.id]
    item.setPos(pg.Point(500.0, 500.0))
    assert st.rois.selected_ids == [rec.id]      # selection unchanged by the drag

    row = w._list.visualItemRect(w._list.item(0))
    QTest.mouseClick(w._list.viewport(), Qt.MouseButton.LeftButton, pos=row.center())
    _app.processEvents()

    restored = ctrl.items[rec.id]
    assert restored is not item
    assert restored.pos().x() == pytest.approx(0.0)
    w.close()


def test_update_commits_the_edit_and_it_then_survives_selection(_app):
    """Update writes the edited copy into the record; after that the geometry is
    the stored one, so a later selection no longer moves it back."""
    from minflux_viewer.ui.roi.roi_manager import RoiManagerWindow

    st = AppState()
    ctrl = _ctrl(_app, store=st.rois)
    ctrl.activate()                              # Update reads the active adapter
    rec = _rect("r")
    st.rois.add(rec)
    st.rois.set_show_all(True)
    st.rois.select([rec.id])
    w = RoiManagerWindow(st)

    ctrl.items[rec.id].setPos(pg.Point(500.0, 500.0))
    w._update()
    assert _bounds_x(st.rois.records[0]) == pytest.approx(500.0)

    st.rois.select([st.rois.records[0].id])      # restore now finds the new geometry
    assert ctrl.items[st.rois.records[0].id].pos().x() == pytest.approx(500.0)
    w.close()


# ------------------- a press in a view selects the Manager entry (Fiji)
# Fiji's canvas activates a Show-All ROI from the image and selects it in the
# ROI Manager (``ImageCanvas.activateOverlayRoi`` → ``roiManagerSelect`` →
# ``RoiManager.selectAndMakeVisible``), from the ROI or from its label.

def _box(name, x, w=30.0):
    return RoiRecord.create("rectangle", {"bounds": [float(x), 0.0, w, 30.0]},
                            name=name, coordinate_space="pixel")


def _shown_ctrl(qtbot, store):
    """A controller on a plot that is on screen, showing data x/y −10…110."""
    from PyQt6.QtTest import QTest

    ctrl = _ctrl(None, store=store)
    plot = ctrl.view_widget
    qtbot.addWidget(plot)
    plot.resize(420, 420)
    plot.show()
    QTest.qWaitForWindowExposed(plot)
    ctrl.view_box.setRange(xRange=(-10.0, 110.0), yRange=(-10.0, 110.0), padding=0)
    for _ in range(3):
        qtbot.wait(1)
    return ctrl


def _click_view(ctrl, x, y):
    """A real left click at data (x, y), through the viewport's event filter."""
    from PyQt6.QtTest import QTest

    scene = ctrl.view_box.mapViewToScene(QPointF(float(x), float(y)))
    pt = ctrl.view_widget.mapFromScene(scene)
    QTest.mouseClick(ctrl.view_widget.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, pt)


def _manager(qtbot, st):
    from minflux_viewer.ui.roi.roi_manager import RoiManagerWindow

    w = RoiManagerWindow(st)
    qtbot.addWidget(w)
    return w


def _manager_rows(w):
    return [w._list.row(item) for item in w._list.selectedItems()]


def test_pressing_a_roi_in_the_view_selects_its_manager_entry(qtbot):
    st = AppState()
    a, b = _box("a", 0.0), _box("b", 60.0)
    st.rois.add(a)
    st.rois.add(b)
    st.rois.deselect()
    st.rois.set_show_all(True)
    ctrl = _shown_ctrl(qtbot, st.rois)
    w = _manager(qtbot, st)

    _click_view(ctrl, 75.0, 15.0)                     # inside b, edit mode
    assert st.rois.selected_ids == [b.id]
    assert _manager_rows(w) == [1] and w._list.currentRow() == 1
    assert (ctrl.items[b.id].pen.color().name().lower()
            == ctrl.MANAGER_SELECT_COLOR.lower())   # highlighted in the view too

    _click_view(ctrl, 15.0, 15.0)                     # inside a
    assert st.rois.selected_ids == [a.id]
    assert _manager_rows(w) == [0] and w._list.currentRow() == 0

    _click_view(ctrl, 50.0, 80.0)                     # empty space: Fiji keeps it
    assert st.rois.selected_ids == [a.id]
    ctrl.dispose()


def test_pressing_a_roi_with_a_drawing_tool_armed_selects_it_and_draws_nothing(qtbot):
    st = AppState()
    a, b = _box("a", 0.0), _box("b", 60.0)
    st.rois.add(a)
    st.rois.add(b)
    st.rois.deselect()
    st.rois.set_show_all(True)
    ctrl = _shown_ctrl(qtbot, st.rois)
    st.rois.set_tool("rectangle")

    _click_view(ctrl, 75.0, 15.0)
    assert st.rois.selected_ids == [b.id]
    assert ctrl.draft is None                         # the press edits b; no new shape
    ctrl.dispose()


def test_pressing_the_selected_roi_keeps_its_uncommitted_edit(qtbot):
    """The press that starts (or continues) an edit must not re-select the ROI:
    `select` restores the stored geometry, which would throw the edit away."""
    st = AppState()
    a = _box("a", 0.0)
    st.rois.add(a)
    st.rois.set_show_all(True)
    st.rois.select([a.id])
    ctrl = _shown_ctrl(qtbot, st.rois)
    item = ctrl.items[a.id]
    ctrl._dirty_items.add(a.id)                       # as after a drag of the copy

    restores = []
    st.rois.restore_requested.connect(lambda: restores.append(1))
    _click_view(ctrl, 15.0, 15.0)
    assert st.rois.selected_ids == [a.id]
    assert not restores and ctrl.items[a.id] is item
    ctrl.dispose()


def test_pressing_a_roi_label_selects_it(qtbot):
    """The label of a V-shaped polyline sits at the vertex centroid, well off
    its line: only the label can take a press there."""
    st = AppState()
    vee = RoiRecord.create("polyline", {"points": [[0.0, 0.0], [50.0, 100.0], [100.0, 0.0]]},
                           name="vee", coordinate_space="pixel")
    other = _box("other", 70.0, w=20.0)
    st.rois.add(vee)
    st.rois.add(other)
    st.rois.select([other.id])
    st.rois.set_show_all(True)
    st.rois.set_show_labels(True)
    ctrl = _shown_ctrl(qtbot, st.rois)
    w = _manager(qtbot, st)

    label = ctrl.labels[vee.id]
    centre = ctrl.view_box.mapSceneToView(label.sceneBoundingRect().center())
    lx, ly = float(centre.x()), float(centre.y())
    assert not ctrl._point_hits_record((lx, ly), vee, ctrl._hit_tolerance())  # off the line

    st.rois.set_tool("rectangle")                     # a label press never draws
    _click_view(ctrl, lx, ly)
    assert st.rois.selected_ids == [vee.id]
    assert _manager_rows(w) == [0]
    assert ctrl.draft is None

    # Without labels the same spot is empty space: nothing is picked and the
    # armed tool draws there as usual.
    st.rois.select([other.id])
    st.rois.set_show_labels(False)
    _click_view(ctrl, lx, ly)
    assert st.rois.selected_ids == [other.id]
    ctrl.dispose()


def test_hit_test_follows_the_drawn_shape(_app):
    """A press selects what it lands on, so the hit-test is the shape, not its
    bounding box (nor the zone of the removed rotate handle beyond it)."""
    ctrl = _ctrl(_app)
    line = RoiRecord.create("line", {"points": [[0.0, 0.0], [100.0, 100.0]]})
    assert ctrl._point_hits_record((50.0, 50.0), line, 1.0)
    assert not ctrl._point_hits_record((90.0, 10.0), line, 1.0)   # in its box only

    tri = RoiRecord.create("polygon", {"points": [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]],
                                       "closed": True})
    assert ctrl._point_hits_record((20.0, 20.0), tri, 1.0)        # inside
    assert ctrl._point_hits_record((50.0, 50.5), tri, 1.0)        # on the edge
    assert not ctrl._point_hits_record((80.0, 80.0), tri, 1.0)    # in its box only

    oval = RoiRecord.create("oval", {"bounds": [0.0, 0.0, 100.0, 100.0]})
    assert ctrl._point_hits_record((50.0, 50.0), oval, 1.0)
    assert ctrl._point_hits_record((100.5, 50.0), oval, 1.0)      # on the curve
    assert not ctrl._point_hits_record((3.0, 3.0), oval, 1.0)     # box corner

    rect = RoiRecord.create("rectangle", {"bounds": [0.0, 0.0, 100.0, 40.0]})
    assert ctrl._point_hits_record((99.5, 39.5), rect, 1.0)
    assert not ctrl._point_hits_record((50.0, 44.0), rect, 1.0)   # old rotate zone


def test_a_selection_made_elsewhere_is_scrolled_into_view(qtbot):
    """Fiji's ``selectAndMakeVisible``: the entry becomes current and visible,
    so the list's arrow keys continue from it."""
    from PyQt6.QtTest import QTest

    st = AppState()
    for i in range(80):
        st.rois.add(_rect(f"r{i}", x=i * 20))
    st.rois.deselect()
    w = _manager(qtbot, st)
    w.show()
    QTest.qWaitForWindowExposed(w)
    w._list.scrollToTop()

    last = w._list.count() - 1
    st.rois.select([st.rois.records[last].id])        # e.g. a press in a view
    assert w._list.currentRow() == last
    assert w._list.viewport().rect().intersects(w._list.visualItemRect(w._list.item(last)))

    w._list.setFocus()
    QTest.keyClick(w._list, Qt.Key.Key_Up)
    assert st.rois.selected_ids == [st.rois.records[last - 1].id]


def test_a_click_in_the_list_does_not_move_its_current_row(qtbot):
    """Only selections made elsewhere move the current row: a Ctrl-click in the
    list keeps its own current row instead of jumping to the first selected."""
    from PyQt6.QtTest import QTest

    st = AppState()
    for i in range(5):
        st.rois.add(_rect(f"r{i}", x=i * 20))
    st.rois.deselect()
    w = _manager(qtbot, st)
    w.show()
    QTest.qWaitForWindowExposed(w)
    w._list.setCurrentRow(0)

    rect = w._list.visualItemRect(w._list.item(3))
    QTest.mouseClick(w._list.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.ControlModifier, rect.center())
    assert sorted(_manager_rows(w)) == [0, 3]
    assert w._list.currentRow() == 3
