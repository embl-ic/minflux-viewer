"""ROI Manager list behaviour:
1. arrow keys navigate the list and switch the active/selected ROI (previously a
   selection-triggered `changed` signal rebuilt the whole list mid-navigation);
2. a ROI selected in the Manager is drawn in a distinct color in the views so it
   stands out from the other (Show-all) ROIs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.roi import RoiRecord, RoiStore
from minflux_viewer.ui.roi_overlay import RoiOverlayController


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


def _rect(name, x=0.0, stroke="#ffff00"):
    return RoiRecord.create("rectangle", {"bounds": [float(x), 0.0, 10.0, 10.0]},
                            name=name, coordinate_space="pixel", stroke_color=stroke)


# ------------------------------------------------------------- arrow navigation

def test_selecting_does_not_rebuild_manager_list(_app):
    from minflux_viewer.ui.roi_manager import RoiManagerWindow

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

    from minflux_viewer.ui.roi_manager import RoiManagerWindow

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

    from minflux_viewer.ui.roi_manager import RoiManagerWindow

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
    from minflux_viewer.ui.roi_manager import RoiManagerWindow

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
