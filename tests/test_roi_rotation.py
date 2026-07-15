"""Rotated rectangle/oval geometry round-trip, mask correctness, hit-test, and
that a stored ROI's view edits do NOT mutate the Manager record until Update."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QWidget

from minflux_viewer.core.roi import RoiRecord, RoiStore
from minflux_viewer.core.roi_selection import oval_mask, rectangle_mask
from minflux_viewer.ui.roi_overlay import (
    RoiOverlayController,
    item_to_geometry,
    normalize_angle_deg,
)


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


def test_rotated_rectangle_geometry_round_trip(_app):
    ctrl = _ctrl(_app)
    rec = RoiRecord.create("rectangle", {"bounds": [0.0, 0.0, 100.0, 50.0], "angle": 30.0})
    g = item_to_geometry("rectangle", ctrl._make_item(rec))
    assert g["bounds"] == pytest.approx([0.0, 0.0, 100.0, 50.0], abs=1e-6)
    assert g["angle"] == pytest.approx(30.0, abs=1e-6)
    # centre preserved (rotation is about the box centre)
    bx, by, bw, bh = g["bounds"]
    assert (bx + bw / 2, by + bh / 2) == pytest.approx((50.0, 25.0))


def test_rotated_oval_angle_persists(_app):
    ctrl = _ctrl(_app)
    rec = RoiRecord.create("oval", {"bounds": [0.0, 0.0, 100.0, 50.0], "angle": 40.0})
    g = item_to_geometry("oval", ctrl._make_item(rec))
    assert g.get("angle") == pytest.approx(40.0, abs=1e-6)  # was previously dropped → snapped back


def test_rotated_rectangle_mask_matches_display():
    # a 90°-rotated 100×20 rect centred at (0,0): long axis becomes vertical
    rec = RoiRecord.create("rectangle", {"bounds": [-50.0, -10.0, 100.0, 20.0], "angle": 90.0})
    x = np.array([0.0, 0.0, 40.0])
    y = np.array([40.0, 0.0, 0.0])
    m = rectangle_mask(x, y, rec)
    # after 90° rotation the box spans y∈[-50,50], x∈[-10,10]:
    assert m.tolist() == [True, True, False]


def test_rotated_oval_mask_matches_display():
    rec = RoiRecord.create("oval", {"bounds": [-50.0, -10.0, 100.0, 20.0], "angle": 90.0})
    m = oval_mask(np.array([0.0, 40.0]), np.array([40.0, 0.0]), rec)
    assert m.tolist() == [True, False]  # inside along the rotated long axis; outside across it


def test_rotated_rectangle_hit_test(_app):
    ctrl = _ctrl(_app)
    rec = RoiRecord.create("rectangle", {"bounds": [-50.0, -10.0, 100.0, 20.0], "angle": 90.0})
    ctrl.store.add(rec)
    ctrl.store.set_show_all(True)
    # a point on the rotated body (y far, x near centre) should hit; far point not
    assert ctrl._point_hits_record((0.0, 40.0), rec, 0.5) is True
    assert ctrl._point_hits_record((200.0, 200.0), rec, 0.5) is False


@pytest.mark.parametrize("roi_type", ["rectangle", "oval"])
def test_plain_rect_oval_have_no_rotate_handle(_app, roi_type):
    """A plain (axis-aligned) rectangle / oval is not rotatable — the rotate handle
    was removed; rotation is via the Rotated Rectangle / Ellipse variants."""
    ctrl = _ctrl(_app)
    item = ctrl._make_item(RoiRecord.create(roi_type, {"bounds": [0.0, 0.0, 100.0, 50.0]}))
    assert len(item.handles) == 8
    assert not any(h.get("type") == "r" for h in item.handles)


def test_normalize_angle_deg_wraps_to_bounded_range():
    """A pyqtgraph rotate handle accumulates the raw angle across full turns; the
    read-out must wrap to (-180, 180] (the ellipse reported -7140° before)."""
    assert normalize_angle_deg(30.0) == pytest.approx(30.0, abs=1e-6)
    assert normalize_angle_deg(360.0) == pytest.approx(0.0, abs=1e-6)
    assert normalize_angle_deg(190.0) == pytest.approx(-170.0, abs=1e-6)
    assert normalize_angle_deg(-180.0) == pytest.approx(180.0, abs=1e-6)   # (-180, 180]
    assert normalize_angle_deg(180.0) == pytest.approx(180.0, abs=1e-6)
    # Big accumulated angles: bounded AND congruent to the input mod 360.
    for raw in (-7140.0, -3680.0, 99999.0, 750.0):
        w = normalize_angle_deg(raw)
        assert -180.0 < w <= 180.0
        assert abs((w - raw) / 360.0 - round((w - raw) / 360.0)) < 1e-9


def test_item_to_geometry_normalizes_accumulated_angle(_app):
    """A rotated variant spun several turns stores a wrapped angle (not e.g. 750°)."""
    ctrl = _ctrl(_app)
    item = ctrl._make_item(RoiRecord.create(
        "oval", {"bounds": [0.0, 0.0, 100.0, 62.0], "angle": 30.0, "variant": "rotated"}))
    item.rotate(720.0)                                   # spin two full extra turns
    ang = item_to_geometry("oval", item, prev={"variant": "rotated"}).get("angle")
    assert -180.0 < ang <= 180.0 and ang == pytest.approx(30.0, abs=1e-6)


def test_stored_roi_edit_does_not_mutate_record_until_update(_app):
    ctrl = _ctrl(_app)
    rec = RoiRecord.create("rectangle", {"bounds": [0.0, 0.0, 100.0, 50.0]})
    ctrl.store.add(rec)
    ctrl.store.select([rec.id])
    ctrl.refresh()
    item = ctrl.items[rec.id]
    item.setPos(pg.Point(500.0, 500.0))            # move the displayed copy
    # the stored record is unchanged (no auto-commit)
    assert ctrl.store.records[0].geometry["bounds"] == [0.0, 0.0, 100.0, 50.0]
    # explicit Update reads the moved item
    updated = ctrl.record_for_update(ctrl.store.records[0])
    bx, by, bw, bh = updated.geometry["bounds"]
    assert (round(bx + bw / 2), round(by + bh / 2)) == (550, 525)
