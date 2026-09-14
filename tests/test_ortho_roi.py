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


# --------------------------------------------------- the two reported defects
def test_a_freshly_drawn_volume_roi_reaches_the_side_panes_at_once(_qt_app):
    """Reported: the silhouette never showed until another shape was drawn.

    ⚠ A draft is deliberately NOT in the store, so ``rois.changed`` never fires
    for it; the side panes were refreshed only by store signals and the view
    debounce, so a drawn shape appeared there only when something unrelated
    happened next.
    """
    state = _state()
    win = _scatter(_qt_app, state)
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        for _ in range(3):
            _qt_app.processEvents()

        ctrl = win._roi_overlay
        ctrl._set_draft(_volume_draft())
        ctrl._finalize_draft_selection(update_item=True)
        for _ in range(3):
            _qt_app.processEvents()

        drawn = win._ortho_roi_outlines._items
        assert any(plane == "XZ" for plane, _rid in drawn), "no XZ silhouette"
        assert any(plane == "YZ" for plane, _rid in drawn), "no YZ silhouette"
    finally:
        win.close()


def test_clearing_the_draft_takes_its_silhouettes_with_it(_qt_app):
    state = _state()
    win = _scatter(_qt_app, state)
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        for _ in range(3):
            _qt_app.processEvents()
        ctrl = win._roi_overlay
        ctrl._set_draft(_volume_draft())
        ctrl._finalize_draft_selection(update_item=True)
        for _ in range(3):
            _qt_app.processEvents()
        assert win._ortho_roi_outlines._items

        ctrl._clear_draft()
        for _ in range(3):
            _qt_app.processEvents()
        assert not win._ortho_roi_outlines._items
    finally:
        win.close()


def _volume_draft():
    from minflux_viewer.core.roi import RoiRecord

    return RoiRecord.create(
        "cuboid", {"x": [-100.0, 100.0], "y": [-80.0, 80.0], "z": [-40.0, 40.0]},
        context={"source_view": "scatter", "dataset_idx": 0})


def test_a_volume_roi_can_be_hit_so_it_can_be_selected_and_deleted(_qt_app):
    """Reported: the drawn shape could not be right-clicked or deleted.

    ⚠ ``_bounds`` knows only ``bounds`` / ``point`` / ``points``; a volume
    geometry has none of them, so it returned a degenerate box at the origin and
    every hit-test missed. The silhouette's own extent is the answer.
    """
    state = _state()
    win = _scatter(_qt_app, state)
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        for _ in range(3):
            _qt_app.processEvents()
        ctrl = win._roi_overlay
        rec = _volume_draft()
        ctrl._set_draft(rec)
        ctrl._finalize_draft_selection(update_item=True)

        assert ctrl._volume_view_bounds(rec) is not None
        assert ctrl._point_hits_record((0.0, 0.0), rec, 1.0)          # inside
        assert not ctrl._point_hits_record((5000.0, 5000.0), rec, 1.0)  # far away
    finally:
        win.close()


def test_the_hit_box_follows_the_pane_that_is_showing_it(_qt_app):
    """In XY the box is the X/Y footprint; in XZ it is X/Z. A single stored
    geometry, read through whichever axes the view shows."""
    state = _state()
    win = _scatter(_qt_app, state)
    try:
        ctrl = win._roi_overlay
        rec = _volume_draft()          # X ±100, Y ±80, Z ±40
        win._axis_combo.setCurrentText("XY")
        for _ in range(2):
            _qt_app.processEvents()
        x, y, w, h = ctrl._volume_view_bounds(rec)
        assert (w, h) == (200.0, 160.0)

        win._axis_combo.setCurrentText("XZ")
        for _ in range(2):
            _qt_app.processEvents()
        x, y, w, h = ctrl._volume_view_bounds(rec)
        assert (w, h) == (200.0, 80.0)
    finally:
        win.close()


def test_dragging_a_volume_roi_moves_only_the_axes_the_view_shows():
    """A view can move a shape on the two axes it shows; the third must come
    through untouched rather than be recomputed from a projection."""
    from minflux_viewer.core.roi_volume import translate_volume

    class R:
        type = "cuboid"
        geometry = {"x": [0.0, 10.0], "y": [0.0, 10.0], "z": [100.0, 200.0]}

    moved = translate_volume(R(), {0: 5.0, 1: -2.0})       # an XY drag
    assert moved["x"] == [5.0, 15.0]
    assert moved["y"] == [-2.0, 8.0]
    assert moved["z"] == [100.0, 200.0]                    # untouched


def test_a_zero_drag_changes_nothing():
    from minflux_viewer.core.roi_volume import translate_volume

    class R:
        type = "cuboid"
        geometry = {"x": [0.0, 10.0], "y": [0.0, 10.0], "z": [0.0, 10.0]}

    assert translate_volume(R(), {0: 0.0, 1: 0.0}) is None
