"""Right-click Convert/Resize on a *draft* ROI keeps it a draft in the view —
it must not open or write to the ROI Manager (regression)."""

from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QWidget

from minflux_viewer.core.roi import RoiRecord, RoiStore
from minflux_viewer.ui.roi_overlay import RoiOverlayController


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
        return 50.0

    def normalize_roi_record(self, record):
        return record


def _make_controller(_app):
    plot = pg.PlotWidget()
    store = RoiStore()
    ctrl = RoiOverlayController(store, _Owner(), plot, plot.getPlotItem())
    return ctrl, store


def _rect_draft(ctrl):
    ctrl._set_draft(RoiRecord.create("rectangle", {"bounds": [0.0, 0.0, 100.0, 200.0]}))
    return ctrl.draft


def test_convert_draft_stays_draft_and_skips_manager(_app):
    ctrl, store = _make_controller(_app)
    opened = []
    ctrl._show_manager_if_needed = lambda: opened.append(True)
    _rect_draft(ctrl)
    ctrl._convert_hit("draft", ctrl.draft, "point")
    assert store.records == []            # not filed into the Manager
    assert opened == []                   # Manager not opened
    assert ctrl.draft is not None and ctrl.draft.type == "point"


def test_resize_draft_stays_draft(_app):
    ctrl, store = _make_controller(_app)
    opened = []
    ctrl._show_manager_if_needed = lambda: opened.append(True)
    _rect_draft(ctrl)
    from minflux_viewer.core.roi_convert import enlarge_shrink_roi

    new = enlarge_shrink_roi(ctrl.draft, 10, mode="enlarge")
    ctrl._replace_hit("draft", ctrl.draft, new)
    assert store.records == []
    assert opened == []
    assert ctrl.draft.geometry["bounds"] == [-10.0, -10.0, 120.0, 220.0]


def test_convert_stored_updates_in_place(_app):
    ctrl, store = _make_controller(_app)
    rec = RoiRecord.create("rectangle", {"bounds": [0.0, 0.0, 100.0, 200.0]})
    store.add(rec)
    store.select([rec.id])
    ctrl._convert_hit("stored", rec, "oval")
    assert len(store.records) == 1            # updated in place, not appended
    assert store.records[0].id == rec.id
    assert store.records[0].type == "oval"
