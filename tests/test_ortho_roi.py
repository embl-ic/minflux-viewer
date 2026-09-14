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


def _shown_in_pane(win, plane: str, roi_id: str) -> bool:
    """Is that ROI on that side pane, by whichever layer draws it?

    A volume ROI is drawn by the pane's own controller (grabbable) and a flat
    one by the display-only outline layer, so a test that names one layer is
    asserting the mechanism rather than the behaviour -- and broke when the
    controllers took over the volume shapes.
    """
    controller = (getattr(win, "_pane_roi_controllers", None) or {}).get(plane)
    if controller is not None and roi_id in controller.items:
        return True
    drawer = getattr(win, "_ortho_roi_outlines", None)
    return drawer is not None and (plane, roi_id) in drawer._items


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

        assert _shown_in_pane(win, "XZ", rec.id)
        assert _shown_in_pane(win, "YZ", rec.id)
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
        assert _shown_in_pane(win, "XZ", rec.id)
        assert _shown_in_pane(win, "YZ", rec.id)

        state.rois.select([rec.id])
        state.rois.delete_selected()
        for _ in range(3):
            _qt_app.processEvents()
        assert not _shown_in_pane(win, "XZ", rec.id)
        assert not _shown_in_pane(win, "YZ", rec.id)
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


# ------------------------------------------ shape preservation while editing
def test_a_cuboid_is_a_RECTANGLE_item_in_every_plane(_qt_app):
    """Reported: dragging a corner deformed the box into a polygon.

    ⚠ The item KIND has to match the shape, not just the outline. Drawn as a
    PolyLineROI a cuboid *is* a polygon on screen -- its handles move single
    vertices -- so a corner drag produced a shape the record could no longer
    represent. An axis-aligned box is a rectangle in every plane.
    """
    from minflux_viewer.ui.roi_overlay import FilledEllipseROI, FilledRectROI

    state = _state()
    win = _scatter(_qt_app, state)
    try:
        ctrl = win._roi_overlay
        cuboid = _volume_draft()
        sphere = type(cuboid)(**{**cuboid.__dict__})
        sphere.type = "sphere"
        sphere.geometry = {"center": [0.0, 0.0, 0.0], "radii": [100.0, 80.0, 40.0]}

        for plane in ("XY", "XZ", "YZ"):
            win._axis_combo.setCurrentText(plane)
            for _ in range(2):
                _qt_app.processEvents()
            assert isinstance(ctrl._make_item(cuboid), FilledRectROI), plane
            assert isinstance(ctrl._make_item(sphere), FilledEllipseROI), plane
    finally:
        win.close()


def test_resizing_in_one_plane_leaves_the_third_axis_alone():
    """A projection has nothing to say about the axis it does not show."""
    from minflux_viewer.core.roi_volume import set_volume_extent

    class R:
        type = "cuboid"
        geometry = {"x": [0.0, 10.0], "y": [0.0, 10.0], "z": [100.0, 200.0]}

    resized = set_volume_extent(R(), (0, 1), (5.0, 5.0, 20.0, 30.0))   # an XY resize
    assert resized["x"] == [5.0, 25.0] and resized["y"] == [5.0, 35.0]
    assert resized["z"] == [100.0, 200.0]                              # untouched


def test_resizing_a_sphere_keeps_it_an_ellipsoid():
    from minflux_viewer.core.roi_volume import set_volume_extent

    class R:
        type = "sphere"
        geometry = {"center": [0.0, 0.0, 7.0], "radii": [10.0, 10.0, 3.0]}

    resized = set_volume_extent(R(), (0, 2), (-50.0, -5.0, 100.0, 20.0))  # XZ
    assert resized["center"] == [0.0, 0.0, 5.0]
    assert resized["radii"] == [50.0, 10.0, 10.0]     # Y radius untouched


def test_a_polyhedron_keeps_the_corners_the_user_drew():
    """⚠ A single-level prism's own polygon IS its silhouette down the stack;
    resampling it radially rounded off every corner of their shape."""
    from minflux_viewer.core.roi_volume import volume_silhouette

    square = [[0.0, 0.0], [100.0, 0.0], [100.0, 80.0], [0.0, 80.0]]

    class R:
        type = "polyhedron"
        geometry = {"axis": "Z", "thickness": 40.0,
                    "levels": [{"at": 0.0, "polygon": square}]}

    outline = volume_silhouette(R(), 0, 1)
    assert len(outline) == 4
    assert [[float(a), float(b)] for a, b in outline] == square


def test_a_polyhedron_shows_a_band_across_the_stack():
    """It is not missing from the side panes -- it is a band there, because that
    is what an extruded cross-section looks like edge-on."""
    from minflux_viewer.core.roi_volume import volume_silhouette

    class R:
        type = "polyhedron"
        geometry = {"axis": "Z", "thickness": 40.0,
                    "levels": [{"at": 0.0,
                                "polygon": [[0.0, 0.0], [100.0, 0.0], [100.0, 80.0]]}]}

    xz = volume_silhouette(R(), 0, 2)
    assert xz is not None and len(xz) >= 4
    assert float(min(p[1] for p in xz)) == -20.0     # thickness/2 either side
    assert float(max(p[1] for p in xz)) == 20.0


# ------------------------------------------------- drawing in the side panes
def test_each_side_pane_gets_its_own_controller_reading_its_own_axes(_qt_app):
    """⚠ The columns are the load-bearing part: an ortho YZ pane plots (2, 1)
    -- Z horizontally -- where a standalone YZ projection plots (1, 2). A
    controller that used the plane name would write Z into a Y coordinate,
    silently, and past every range-based test."""
    state = _state()
    win = _scatter(_qt_app, state)
    try:
        win.enter_ortho_mode()
        for _ in range(4):
            _qt_app.processEvents()
        controllers = win._pane_roi_controllers
        assert sorted(controllers) == ["XZ", "YZ"]
        assert controllers["XZ"]._view_axes() == (0, 2)
        assert controllers["YZ"]._view_axes() == (2, 1)
    finally:
        win.close()


def test_a_cuboid_drawn_in_xz_takes_x_and_z_from_the_drag_and_seeds_y(_qt_app):
    """The rule is plane-agnostic: the derived axis is the one normal to the
    pane that was drawn in."""
    from minflux_viewer.core.roi import RoiRecord

    state = _state()
    win = _scatter(_qt_app, state)
    try:
        win.enter_ortho_mode()
        for _ in range(4):
            _qt_app.processEvents()
        xz = win._pane_roi_controllers["XZ"]
        state.rois.set_tool("cuboid")
        xz._set_draft(RoiRecord.create(
            "rectangle", {"bounds": [-500.0, -100.0, 1000.0, 200.0]},
            **xz._record_kwargs()))
        assert xz._promote_draft_to_volume("cuboid") is True

        g = xz.draft.geometry
        assert xz.draft.type == "cuboid"
        assert g["x"] == [-500.0, 500.0]          # the drag's horizontal axis
        assert g["z"] == [-100.0, 100.0]          # the drag's vertical axis
        assert g["y"][0] < g["y"][1]              # Y came from the data
    finally:
        win.close()


def test_the_pane_controllers_share_the_store(_qt_app):
    """A ROI drawn in any pane is the same record everywhere; the panes differ
    only in the axes they read it through."""
    state = _state()
    win = _scatter(_qt_app, state)
    try:
        win.enter_ortho_mode()
        for _ in range(4):
            _qt_app.processEvents()
        primary = win._roi_overlay
        for controller in win._pane_roi_controllers.values():
            assert controller.store is primary.store
    finally:
        win.close()


def test_a_point_drawn_in_an_ortho_yz_pane_lands_where_it_was_clicked():
    """The transposition, end to end: in an ortho YZ pane the horizontal
    coordinate is Z and the vertical is Y."""
    from minflux_viewer.ui.roi_overlay import point_to_3d, project_point

    xyz = point_to_3d((300.0, 20.0), (2, 1), depth=7.0)    # h=Z, v=Y, depth=X
    assert xyz == [7.0, 20.0, 300.0]
    assert project_point(xyz, (2, 1)) == (300.0, 20.0)     # and back again
    # Read with the STANDALONE convention it would be wrong -- which is the
    # whole reason the columns are passed rather than a plane name.
    assert project_point(xyz, "YZ") == (20.0, 300.0)


# --- a volume ROI selects rows, so every "selects rows" gate must include it ---
# ⚠ A volume type is NOT in `_REGION_TYPES`, and three gates spelled that set
# literally: the "Localizations within" line, its pending-selection recompute,
# and the rule that drops a stale highlight when a selecting draft is replaced
# by a line. `_SELECTING_TYPES` is the set those three mean.

def test_the_selecting_type_gates_include_the_volume_shapes():
    from minflux_viewer.core.roi_selection import VOLUME_ROI_TYPES
    from minflux_viewer.ui.roi_overlay import RoiOverlayController

    selecting = RoiOverlayController._SELECTING_TYPES
    assert VOLUME_ROI_TYPES <= selecting
    assert RoiOverlayController._REGION_TYPES <= selecting
    for line_type in ("line", "polyline", "point", "angle"):
        assert line_type not in selecting          # these enclose nothing


def test_a_volume_edit_queues_a_fresh_localization_selection(_qt_app):
    """The shared selecting-type rule must reach the debounce entry point too."""
    state = _state()
    win = _scatter(_qt_app, state)
    try:
        record = _volume_draft()
        win._roi_overlay._queue_selection_update(record)
        assert win._roi_overlay._pending_selection_record is record
        assert record.selection_dirty is True
    finally:
        win.close()


def test_the_properties_read_out_describes_a_volume_roi_in_three_dimensions():
    from minflux_viewer.core.roi import RoiRecord
    from minflux_viewer.ui.roi_overlay import RoiOverlayController

    class _Formatter:                    # _geometry_text reads only self._fmt
        # staticmethod(): a plain function assigned in a class body would bind,
        # and _fmt takes the value, not a self.
        _fmt = staticmethod(RoiOverlayController._fmt)

    rec = RoiRecord.create("cuboid", {"x": [0, 100], "y": [10, 30], "z": [-20, 60]})
    text = RoiOverlayController._geometry_text(_Formatter(), rec)
    assert text == "X=0, Y=10, Z=-20, W=100, H=20, D=80"
    assert "0 points" not in text                  # what it used to say


# --------------------------------------------- drawing in a side pane, for real
# ⚠ The tests above drive `_set_draft` / `_promote_draft_to_volume` directly, so
# they proved the geometry and not that a **drag** in a side pane does anything.
# It did not, in the render view, which had no pane controllers at all: the
# volume tools turned the mode on, said "draw in any pane", and XZ/YZ ignored
# the mouse. Drive the viewport, the way a user does.

def _render(app, state):
    from minflux_viewer.ui.render_window import RenderWindow

    win = RenderWindow(state, 0)
    win.resize(900, 900)
    win.show()
    for _ in range(6):
        app.processEvents()
    return win


def _drag(app, widget, frm=(0.35, 0.35), to=(0.65, 0.65)):
    """A real press / move / release on a pane's viewport."""
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest

    vp = widget.viewport()
    a = QPoint(int(vp.width() * frm[0]), int(vp.height() * frm[1]))
    b = QPoint(int(vp.width() * to[0]), int(vp.height() * to[1]))
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, a)
    for _ in range(3):
        app.processEvents()
    QTest.mouseMove(vp, b)
    for _ in range(3):
        app.processEvents()
    QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, b)
    for _ in range(8):
        app.processEvents()


def _click_view_point(app, widget, point=(0.0, 0.0)):
    """Click a PlotWidget at a data-space point, not an arbitrary pixel."""
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtTest import QTest

    scene = widget.getPlotItem().getViewBox().mapViewToScene(QPointF(*point))
    pixel = widget.mapFromScene(scene)
    QTest.mouseClick(
        widget.viewport(), Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier, pixel,
    )
    for _ in range(4):
        app.processEvents()


def _move_to_view_point(app, widget, point=(0.0, 0.0)):
    """Send an unpressed mouse move to a PlotWidget data-space point."""
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    scene = widget.getPlotItem().getViewBox().mapViewToScene(QPointF(*point))
    pixel = widget.mapFromScene(scene)
    viewport = widget.viewport()
    event = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(pixel),
        QPointF(viewport.mapToGlobal(pixel)),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    app.sendEvent(viewport, event)
    for _ in range(4):
        app.processEvents()


@pytest.mark.parametrize("view", ["scatter", "render"])
def test_dragging_in_a_side_pane_draws_a_volume_roi(_qt_app, view):
    state = _state()
    win = _scatter(_qt_app, state) if view == "scatter" else _render(_qt_app, state)
    try:
        win.enter_ortho_mode()
        for _ in range(8):
            _qt_app.processEvents()
        assert sorted(win._pane_roi_controllers) == ["XZ", "YZ"]

        state.rois.set_tool("cuboid")
        panes = getattr(win, "_pane_plots", None) or win._pane_widgets
        _drag(_qt_app, panes["XZ"])

        draft = win._pane_roi_controllers["XZ"].draft
        assert draft is not None and draft.type == "cuboid"
        # ⚠ and it is filed as an XZ shape: the owner's normalize_roi_record
        # reads self.roi_view_plane(), which through __getattr__ would bind to
        # the window and answer "XY" for the whole mode.
        assert (draft.context or {}).get("view_plane") == "XZ"
    finally:
        win.close()


def test_a_stored_volume_roi_is_drawn_once_per_pane(_qt_app):
    """Once the pane has its own controller the decorative outline must stand
    down, or every ROI is two overlapping shapes, one of them ungrabbable."""
    import pyqtgraph as pg

    from minflux_viewer.core.roi import RoiRecord

    state = _state()
    win = _scatter(_qt_app, state)
    try:
        win.enter_ortho_mode()
        for _ in range(8):
            _qt_app.processEvents()
        record = RoiRecord.create(
            "cuboid", {"x": [-200.0, 200.0], "y": [-150.0, 150.0], "z": [-80.0, 80.0]},
            **win._roi_overlay._record_kwargs())
        state.rois.add(record)
        state.rois.show_all = True
        for _ in range(8):
            _qt_app.processEvents()

        items = win._pane_plots["XZ"].getPlotItem().items
        curves = [i for i in items if isinstance(i, pg.PlotCurveItem)]
        assert record.id in win._pane_roi_controllers["XZ"].items    # grabbable
        assert curves == []                                          # and only once
    finally:
        win.close()


def test_a_flat_roi_from_another_plane_is_shown_but_not_edited_in_a_side_pane(_qt_app):
    """Its bounds are in ITS plane's axes. Drawn by a side pane's controller,
    a rectangle's Y coordinate would land on that pane's Z axis -- and be
    draggable there, writing the nonsense back. The dashed outline is the
    honest way to show it."""
    from minflux_viewer.core.roi import RoiRecord

    state = _state()
    win = _scatter(_qt_app, state)
    try:
        win.enter_ortho_mode()
        for _ in range(8):
            _qt_app.processEvents()
        flat = RoiRecord.create(
            "rectangle", {"bounds": [-200.0, 5000.0, 400.0, 300.0]},
            **win._roi_overlay._record_kwargs())
        flat.context = dict(flat.context or {})
        flat.context["view_plane"] = "XY"
        flat.context["depth_value"] = 10.0
        state.rois.add(flat)
        state.rois.show_all = True
        for _ in range(8):
            _qt_app.processEvents()

        xz = win._pane_roi_controllers["XZ"]
        assert flat.id not in xz.items                 # not drawn on XZ's axes
        assert xz._record_in_scope(flat) is False      # and not hit-tested there
        # still visible, as the edge-on dashed outline
        assert ("XZ", flat.id) in win._ortho_roi_outlines._items
    finally:
        win.close()


def test_an_in_progress_draft_still_reaches_the_side_panes(_qt_app):
    """A draft belongs to the controller drawing it, so no other pane's
    controller holds it -- the outline layer is the only thing that can show
    it, which is the whole point while a volume ROI is being seeded."""
    from minflux_viewer.core.roi import RoiRecord

    state = _state()
    win = _scatter(_qt_app, state)
    try:
        win.enter_ortho_mode()
        for _ in range(8):
            _qt_app.processEvents()
        draft = RoiRecord.create(
            "cuboid", {"x": [-100.0, 100.0], "y": [-90.0, 90.0], "z": [-40.0, 40.0]},
            **win._roi_overlay._record_kwargs())
        win._roi_overlay.replace_draft(draft)
        for _ in range(8):
            _qt_app.processEvents()

        panes = {key[0] for key in win._ortho_roi_outlines._items if key[1] == draft.id}
        assert panes == {"XZ", "YZ"}
    finally:
        win.close()


@pytest.mark.parametrize("view", ["scatter", "render"])
@pytest.mark.parametrize("plane,axes", [("XZ", ("x", "z")), ("YZ", ("z", "y"))])
def test_pointing_into_a_side_pane_makes_a_pending_volume_editable_there(
        _qt_app, view, plane, axes):
    """Regression: the pending ROI was editable only in the pane that drew it.

    Its other projections were decorative curves because a draft is not in the
    shared store.  A real edit-mode hover now hands that one draft to the pane's
    controller, where its normal edit signal updates the correct two axes.
    """
    from minflux_viewer.ui.roi_overlay import FilledRectROI

    state = _state()
    win = _scatter(_qt_app, state) if view == "scatter" else _render(_qt_app, state)
    try:
        win.enter_ortho_mode()
        for _ in range(8):
            _qt_app.processEvents()
        primary = win._roi_overlay
        side = win._pane_roi_controllers[plane]
        pane = (getattr(win, "_pane_plots", None) or win._pane_widgets)[plane]

        # Give the side viewport focus first, then explicitly make XY active.
        # The following move therefore exercises MouseMove rather than passing
        # only because a FocusIn event happened to activate the pane.
        pane.setFocus()
        _qt_app.processEvents()
        primary.replace_draft(_volume_draft())
        primary.activate()
        assert state.rois.active_adapter is primary
        assert side.draft is None

        before = dict(primary.draft.geometry)
        _move_to_view_point(_qt_app, pane)

        assert state.rois.active_adapter is side
        assert primary.draft is None
        assert side.draft is not None and side.draft.type == "cuboid"
        assert isinstance(side.draft_item, FilledRectROI)
        other_plane = "YZ" if plane == "XZ" else "XZ"
        assert (plane, side.draft.id) not in win._ortho_roi_outlines._items
        assert (other_plane, side.draft.id) in win._ortho_roi_outlines._items
        side.draft_item.translate((25.0, 10.0), finish=True)
        for _ in range(4):
            _qt_app.processEvents()
        expected_delta = {"x": 0.0, "y": 0.0, "z": 0.0}
        expected_delta[axes[0]] = 25.0
        expected_delta[axes[1]] = 10.0
        for axis, delta in expected_delta.items():
            assert side.draft.geometry[axis] == [v + delta for v in before[axis]]
    finally:
        win.close()


@pytest.mark.parametrize("view", ["scatter", "render"])
@pytest.mark.parametrize("plane,axes", [("XZ", ("x", "z")), ("YZ", ("z", "y"))])
def test_side_pane_edit_becomes_the_roi_manager_update_source(
        _qt_app, view, plane, axes):
    """A side edit must activate that adapter before Manager Update reads it."""
    state = _state()
    win = _scatter(_qt_app, state) if view == "scatter" else _render(_qt_app, state)
    try:
        win.enter_ortho_mode()
        for _ in range(8):
            _qt_app.processEvents()
        record = _volume_draft()
        state.rois.add(record)
        state.rois.show_all = True
        state.rois.changed.emit()
        for _ in range(6):
            _qt_app.processEvents()

        primary = win._roi_overlay
        side = win._pane_roi_controllers[plane]
        pane = (getattr(win, "_pane_plots", None) or win._pane_widgets)[plane]
        pane.setFocus()
        _qt_app.processEvents()
        primary.activate()
        assert state.rois.active_adapter is primary

        _click_view_point(_qt_app, pane)
        assert state.rois.active_adapter is side

        side.items[record.id].translate((25.0, 10.0), finish=True)
        for _ in range(3):
            _qt_app.processEvents()
        updated = state.rois.active_adapter.record_for_update(record)
        expected_delta = {"x": 0.0, "y": 0.0, "z": 0.0}
        expected_delta[axes[0]] = 25.0
        expected_delta[axes[1]] = 10.0
        for axis, delta in expected_delta.items():
            assert updated.geometry[axis] == [v + delta for v in record.geometry[axis]]
    finally:
        win.close()
