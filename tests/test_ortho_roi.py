"""ROIs in the orthogonal view: visible in all three panes, and honestly so."""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

import numpy as np
from PyQt6.QtWidgets import QApplication

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.ui.ortho_roi import pane_outline
from minflux_viewer.ui.ortho_view import ORTHO_AXIS, ORTHO_AXIS_COLUMNS


@pytest.fixture
def _qt_app():
    if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
        pytest.skip("No display available for Qt tests")
    return QApplication.instance() or QApplication(sys.argv)


class Rec:
    def __init__(self, type, geometry, context=None, rid="r1"):
        self.type = type
        self.geometry = geometry
        self.context = context or {}
        self.id = rid
        self.stroke_color = "#ffff00"


# --------------------------------------------------------------- the geometry
def test_a_volume_roi_shows_a_real_outline_in_both_side_panes():
    rec = Rec("cuboid", {"x": [0.0, 100.0], "y": [0.0, 200.0], "z": [300.0, 400.0]})
    kind, xz = pane_outline(rec, "XZ")
    assert kind == "full"
    assert min(p[0] for p in xz) == 0.0 and max(p[0] for p in xz) == 100.0     # X
    assert min(p[1] for p in xz) == 300.0 and max(p[1] for p in xz) == 400.0   # Z

    kind, yz = pane_outline(rec, "YZ")
    assert kind == "full"
    # ⚠ The ortho YZ pane draws Z HORIZONTALLY -- the transposition that a plane
    # name cannot express and that asking by axis columns removes.
    assert ORTHO_AXIS_COLUMNS["YZ"] == (2, 1)
    assert min(p[0] for p in yz) == 300.0 and max(p[0] for p in yz) == 400.0   # Z
    assert min(p[1] for p in yz) == 0.0 and max(p[1] for p in yz) == 200.0     # Y


def test_a_flat_roi_is_edge_on_in_the_side_panes():
    """It has no extent on the axis it was drawn against, so a segment at its
    recorded depth is the honest picture."""
    rec = Rec("rectangle", {"bounds": [0.0, 0.0, 100.0, 200.0]},
              {"view_plane": "XY", "depth_value": 55.0})
    kind, xz = pane_outline(rec, "XZ")
    assert kind == "line"
    assert xz == [[0.0, 55.0], [100.0, 55.0]]        # X extent, flat in Z

    kind, yz = pane_outline(rec, "YZ")
    assert kind == "line"
    assert yz == [[55.0, 0.0], [55.0, 200.0]]        # transposed: Z, then Y


def test_a_flat_roi_without_a_recorded_depth_is_not_placed_by_guesswork():
    rec = Rec("rectangle", {"bounds": [0.0, 0.0, 10.0, 10.0]}, {"view_plane": "XY"})
    assert pane_outline(rec, "XZ") is None


def test_a_three_d_point_carries_its_own_depth():
    """A point's Z is in its geometry, so it needs no context to be placed."""
    rec = Rec("point", {"point": [10.0, 20.0, 30.0]}, {"view_plane": "XY"})
    kind, xz = pane_outline(rec, "XZ")
    assert kind in {"line", "point"}
    assert xz[0][1] == 30.0


def test_the_kind_distinguishes_flat_from_volume():
    """A caller that ignores it draws a flat ROI like a volume one, and the user
    reads it as constraining Z."""
    flat = Rec("rectangle", {"bounds": [0.0, 0.0, 10.0, 10.0]},
               {"view_plane": "XY", "depth_value": 0.0})
    solid = Rec("cuboid", {"x": [0, 10], "y": [0, 10], "z": [-5, 5]})
    assert pane_outline(flat, "XZ")[0] == "line"
    assert pane_outline(solid, "XZ")[0] == "full"


def test_an_unknown_plane_draws_nothing():
    rec = Rec("cuboid", {"x": [0, 1], "y": [0, 1], "z": [0, 1]})
    assert pane_outline(rec, "XY-ish") is None


# ------------------------------------------------------------------- the view
def _state(three_d: bool = True):
    rng = np.random.default_rng(3)
    n = 400
    ds = build_localization_dataset(
        name="ortho-roi",
        x_nm=rng.normal(0.0, 800.0, n),
        y_nm=rng.normal(0.0, 600.0, n),
        z_nm=rng.normal(0.0, 200.0, n) if three_d else np.zeros(n),
        tid=rng.integers(0, 40, n),
        source_version="simulation",
    )
    state = AppState()
    state.add_dataset(ds)
    state.set_active(0)
    return state


def _scatter(app, state):
    from minflux_viewer.ui.scatter_window import ScatterWindow

    win = ScatterWindow(state, dataset_idx=0)
    win.resize(900, 900)
    win.show()
    for _ in range(3):
        app.processEvents()
    return win


def test_a_volume_roi_appears_in_the_side_panes_of_a_live_view(_qt_app):
    from minflux_viewer.core.roi import RoiRecord

    state = _state()
    win = _scatter(_qt_app, state)
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        for _ in range(3):
            _qt_app.processEvents()

        rec = RoiRecord.create(
            "cuboid", {"x": [-100.0, 100.0], "y": [-100.0, 100.0], "z": [-50.0, 50.0]},
            context={"source_view": "scatter", "dataset_idx": 0})
        state.rois.add(rec)
        for _ in range(3):
            _qt_app.processEvents()

        drawn = win._ortho_roi_outlines._items
        assert ("XZ", rec.id) in drawn and ("YZ", rec.id) in drawn
    finally:
        win.close()


def test_deleting_a_roi_removes_its_side_pane_outlines(_qt_app):
    from minflux_viewer.core.roi import RoiRecord

    state = _state()
    win = _scatter(_qt_app, state)
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        for _ in range(3):
            _qt_app.processEvents()
        rec = RoiRecord.create(
            "cuboid", {"x": [-10.0, 10.0], "y": [-10.0, 10.0], "z": [-10.0, 10.0]},
            context={"source_view": "scatter", "dataset_idx": 0})
        state.rois.add(rec)
        for _ in range(3):
            _qt_app.processEvents()
        assert win._ortho_roi_outlines._items

        state.rois.select([rec.id])
        state.rois.delete_selected()
        for _ in range(3):
            _qt_app.processEvents()
        assert not win._ortho_roi_outlines._items
    finally:
        win.close()


def test_leaving_ortho_clears_the_side_pane_outlines(_qt_app):
    from minflux_viewer.core.roi import RoiRecord

    state = _state()
    win = _scatter(_qt_app, state)
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        for _ in range(3):
            _qt_app.processEvents()
        state.rois.add(RoiRecord.create(
            "cuboid", {"x": [-10.0, 10.0], "y": [-10.0, 10.0], "z": [-10.0, 10.0]},
            context={"source_view": "scatter", "dataset_idx": 0}))
        for _ in range(3):
            _qt_app.processEvents()

        win._axis_combo.setCurrentText("XY")
        for _ in range(3):
            _qt_app.processEvents()
        win._refresh_ortho_roi_outlines()
        assert not win._ortho_roi_outlines._items
    finally:
        win.close()


# ------------------------------------------------------- the tool -> mode link
def test_a_volume_tool_turns_the_orthogonal_view_on(_qt_app):
    """A volume shape is drawn in one plane and bounded in the other two, so a
    single projection cannot show what is being made."""
    state = _state()
    win = _scatter(_qt_app, state)
    try:
        assert not win._ortho_active()
        assert win.enter_ortho_mode() is True
        assert win._ortho_active()
    finally:
        win.close()


def test_the_mode_is_refused_on_two_d_data_rather_than_half_entered(_qt_app):
    state = _state(three_d=False)
    win = _scatter(_qt_app, state)
    try:
        assert win.enter_ortho_mode() is False
        assert not win._ortho_active()
    finally:
        win.close()


def test_entering_is_idempotent(_qt_app):
    state = _state()
    win = _scatter(_qt_app, state)
    try:
        assert win.enter_ortho_mode() is True
        assert win.enter_ortho_mode() is True      # already on, still True
        assert win._ortho_active()
    finally:
        win.close()
