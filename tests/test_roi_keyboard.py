"""Keyboard fine-adjust for the active ROI (arrow keys): they mirror what the
mouse would do on the selected handle — translate the whole ROI, resize a
rectangle/oval corner, rotate via the rotation handle, or move a vertex/point —
1 nm (or 1°) per press."""

from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication, QWidget

from minflux_viewer.core.roi import RoiRecord, RoiStore
from minflux_viewer.ui.roi_overlay import RoiOverlayController, item_to_geometry


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
    plot.resize(400, 400)
    plot.setXRange(0, 400, padding=0)
    plot.setYRange(0, 400, padding=0)
    return RoiOverlayController(RoiStore(), _Owner(), plot, plot.getPlotItem())


def _key(k):
    return QKeyEvent(QEvent.Type.KeyPress, k, Qt.KeyboardModifier.NoModifier)


def _draft(ctrl, roi_type, geom):
    ctrl._set_draft(RoiRecord.create(roi_type, dict(geom)))
    return ctrl.draft


def _select_handle(ctrl, role):
    """Simulate having clicked a handle: set the selected sub-element together with
    its target id (as _track_kbd_handle_selection does), so the target-change guard
    in _handle_arrow_key doesn't reset it back to whole-ROI move."""
    ctrl._kbd_handle = role
    target = ctrl._kbd_target()
    if target is not None:
        ctrl._kbd_target_id = target[1].id


# =========================================================== rectangle / oval

@pytest.mark.parametrize("roi_type", ["rectangle", "oval"])
def test_arrow_keys_move_whole_shape_one_nm(_app, roi_type):
    ctrl = _ctrl(_app)
    _draft(ctrl, roi_type, {"bounds": [100.0, 200.0, 300.0, 400.0]})
    ctrl._handle_arrow_key(_key(Qt.Key.Key_Left))
    x, y, w, h = ctrl.draft.geometry["bounds"]
    assert (round(x), round(y), round(w), round(h)) == (99, 200, 300, 400)
    ctrl._handle_arrow_key(_key(Qt.Key.Key_Up))     # non-inverted: up == +Y
    assert round(ctrl.draft.geometry["bounds"][1]) == 201


@pytest.mark.parametrize("roi_type", ["rectangle", "oval"])
def test_arrow_keys_resize_selected_corner(_app, roi_type):
    ctrl = _ctrl(_app)
    _draft(ctrl, roi_type, {"bounds": [0.0, 0.0, 100.0, 50.0]})
    _select_handle(ctrl, ("scale", 1.0, 1.0))       # top-right corner
    ctrl._handle_arrow_key(_key(Qt.Key.Key_Right))
    x, y, w, h = ctrl.draft.geometry["bounds"]
    assert (round(x), round(y), round(w), round(h)) == (0, 0, 101, 50)   # (0,0) anchored


@pytest.mark.parametrize("roi_type", ["rectangle", "oval"])
def test_plain_rect_oval_have_no_rotate_handle(_app, roi_type):
    """A plain (axis-aligned) rectangle / oval is not rotatable — the rotate handle
    was removed (rotation is now via the Rotated Rectangle / Ellipse variants)."""
    ctrl = _ctrl(_app)
    item = ctrl._make_item(RoiRecord.create(roi_type, {"bounds": [0.0, 0.0, 100.0, 50.0]}))
    assert len(item.handles) == 8                                # 8 scale handles only
    assert not any(h.get("type") == "r" for h in item.handles)   # no rotate handle
    rec = RoiRecord.create(roi_type, {"bounds": [0.0, 0.0, 100.0, 50.0]})
    assert ctrl._kbd_role_at(rec, (50.0, 90.0)) != "rotate"      # no rotate role above box


def test_oval_has_bounding_box_handles(_app):
    ctrl = _ctrl(_app)
    item = ctrl._make_item(RoiRecord.create("oval", {"bounds": [0.0, 0.0, 100.0, 50.0]}))
    # 8 square scale handles, no rotate handle (like the rectangle)
    assert len(item.handles) == 8 and all(h.get("type") == "s" for h in item.handles)


def test_box_scale_fracs_rect_corners_vs_oval_on_curve():
    from minflux_viewer.ui.roi_overlay import box_scale_fracs

    rect = set(box_scale_fracs(on_curve=False))
    oval = set(box_scale_fracs(on_curve=True))
    edges = {(0.0, 0.5), (1.0, 0.5), (0.5, 0.0), (0.5, 1.0)}
    assert edges <= rect and edges <= oval                   # both keep the 4 edge mids
    assert (1.0, 1.0) in rect and (0.0, 0.0) in rect         # rect diagonals at bbox corners
    assert (1.0, 1.0) not in oval and (0.0, 0.0) not in oval # oval diagonals NOT at corners
    d = 0.5 + 0.5 / (2 ** 0.5)                                 # ≈0.854, the 45° curve point
    assert any(abs(fx - d) < 1e-9 and abs(fy - d) < 1e-9 for fx, fy in oval)


def test_rect_handles_square_and_oval_diagonals_on_curve(_app):
    from minflux_viewer.ui.roi_overlay import SquareHandle

    ctrl = _ctrl(_app)
    rect = ctrl._make_item(RoiRecord.create("rectangle", {"bounds": [0.0, 0.0, 100.0, 50.0]}))
    oval = ctrl._make_item(RoiRecord.create("oval", {"bounds": [0.0, 0.0, 100.0, 50.0]}))
    rscale = [h for h in rect.handles if h.get("type") == "s"]
    oscale = [h for h in oval.handles if h.get("type") == "s"]
    assert len(rscale) == 8 and all(isinstance(h["item"], SquareHandle) for h in rscale)
    assert len(oscale) == 8 and all(isinstance(h["item"], SquareHandle) for h in oscale)
    of = {(round(float(h["pos"].x()), 3), round(float(h["pos"].y()), 3)) for h in oscale}
    assert (0.854, 0.854) in of and (1.0, 1.0) not in of      # oval diagonal on the curve


def test_oval_on_curve_diagonal_handle_resizes(_app):
    """The oval's on-curve diagonal (fx≈0.854) must resize the width, not be treated
    as an edge-midpoint (fx=0.5, no-op) — verifies the generalised resize thresholds."""
    ctrl = _ctrl(_app)
    _draft(ctrl, "oval", {"bounds": [0.0, 0.0, 100.0, 50.0]})
    d = 0.5 + 0.5 / (2 ** 0.5)
    _select_handle(ctrl, ("scale", d, d))                    # NE on-curve handle
    ctrl._handle_arrow_key(_key(Qt.Key.Key_Right))
    assert round(ctrl.draft.geometry["bounds"][2]) == 101    # width grew (would be 100 before)


# =============================== rotated rectangle / ellipse variants (Phase 2/3)

def test_center_axis_box_is_long_axis_drag_with_golden_minor():
    """The drag is the long *centre* axis (length + rotation); the short side is the
    golden ratio of it, centred on the axis."""
    from minflux_viewer.ui.roi_overlay import GOLDEN_MINOR, center_axis_box

    bounds, angle = center_axis_box(0.0, 0.0, 100.0, 0.0)     # horizontal
    x, y, w, h = bounds
    assert round(w) == 100 and abs(h - 100 * GOLDEN_MINOR) < 1e-6 and abs(angle) < 1e-9
    assert round(x + w / 2) == 50 and round(y + h / 2) == 0    # centre == drag midpoint

    _b, a = center_axis_box(0.0, 0.0, 10.0, 10.0)            # 45° drag → angle 45
    assert abs(a - 45.0) < 1e-6


def test_rotated_variant_handle_fractions():
    from minflux_viewer.ui.roi_overlay import rotated_variant_handles

    # Fiji-style: BOTH variants put their 4 handles at the edge midpoints, with the
    # solid pivot at the drawing-origin edge middle (0, 0.5).
    edge_mids = {(1.0, 0.5), (0.5, 0.0), (0.5, 1.0)}
    rrot, rscale = rotated_variant_handles("rectangle")
    assert rrot == (0.0, 0.5) and set(rscale) == edge_mids
    orot, oscale = rotated_variant_handles("oval")
    assert orot == (0.0, 0.5) and set(oscale) == edge_mids


def test_roi_tools_superset_but_variant_never_a_record_type():
    from minflux_viewer.core.roi import ROI_TOOLS, ROI_TYPES

    assert {"rotated_rectangle", "ellipse"} <= ROI_TOOLS
    assert not ({"rotated_rectangle", "ellipse"} & ROI_TYPES)  # tools only, never types


def test_draw_rotated_rectangle_makes_rotated_rectangle_record(_app):
    ctrl = _ctrl(_app)
    ctrl.store.set_tool("rotated_rectangle")
    ctrl._drag_start = (0.0, 0.0)
    ctrl._update_drag_draft((100.0, 0.0))                     # horizontal long axis
    d = ctrl.draft
    assert d.type == "rectangle"                             # record type unchanged
    assert d.geometry.get("variant") == "rotated"
    x, y, w, h = d.geometry["bounds"]
    assert round(w) == 100 and round(h) == 62                # golden-ratio minor
    assert abs(d.geometry.get("angle", 0.0)) < 1e-6


def test_draw_ellipse_makes_rotated_oval_record(_app):
    ctrl = _ctrl(_app)
    ctrl.store.set_tool("ellipse")
    ctrl._drag_start = (0.0, 0.0)
    ctrl._update_drag_draft((0.0, 100.0))                    # vertical long axis
    d = ctrl.draft
    assert d.type == "oval"                                  # record type unchanged
    assert d.geometry.get("variant") == "rotated"
    x, y, w, h = d.geometry["bounds"]
    assert round(w) == 100 and round(h) == 62
    assert round(d.geometry.get("angle", 0.0)) == 90         # atan2(100, 0) = 90°


@pytest.mark.parametrize("roi_type", ["rectangle", "oval"])
def test_rotated_variant_item_has_four_handles_one_solid_pivot(_app, roi_type):
    from minflux_viewer.ui.roi_overlay import SolidSquareHandle

    ctrl = _ctrl(_app)
    item = ctrl._make_item(RoiRecord.create(
        roi_type, {"bounds": [0.0, 0.0, 100.0, 62.0], "angle": 30.0, "variant": "rotated"}))
    solids = [h for h in item.handles if isinstance(h["item"], SolidSquareHandle)]
    assert len(item.handles) == 4 and len(solids) == 1
    assert sum(1 for h in item.handles if h.get("type") == "r") == 1   # solid pivot rotates
    assert sum(1 for h in item.handles if h.get("type") == "s") == 3   # 3 resize handles


@pytest.mark.parametrize("roi_type", ["rectangle", "oval"])
def test_rotated_variant_pivot_is_rotate_role(_app, roi_type):
    ctrl = _ctrl(_app)
    rec = RoiRecord.create(roi_type, {"bounds": [0.0, 0.0, 100.0, 62.0], "variant": "rotated"})
    # Both variants: solid pivot at the (0, 0.5) edge-middle; opposite edge-middle
    # (1, 0.5) is a scale handle.
    assert ctrl._kbd_role_at(rec, (0.0, 31.0)) == "rotate"    # pivot = rotate
    role = ctrl._kbd_role_at(rec, (100.0, 31.0))              # opposite edge middle = scale
    assert isinstance(role, tuple) and role[0] == "scale"


def test_rotated_variant_arrow_rotates_about_centre(_app):
    ctrl = _ctrl(_app)
    _draft(ctrl, "rectangle", {"bounds": [0.0, 0.0, 100.0, 60.0], "variant": "rotated"})
    _select_handle(ctrl, "rotate")
    ctrl._handle_arrow_key(_key(Qt.Key.Key_Right))
    assert round(abs(ctrl.draft.geometry.get("angle", 0.0))) == 1    # rotated 1°
    assert ctrl.draft.geometry.get("variant") == "rotated"          # variant preserved
    x, y, w, h = ctrl.draft.geometry["bounds"]
    assert round(x + w / 2) == 50 and round(y + h / 2) == 30         # centre fixed


def test_item_to_geometry_preserves_variant_marker(_app):
    ctrl = _ctrl(_app)
    item = ctrl._make_item(RoiRecord.create(
        "rectangle", {"bounds": [0.0, 0.0, 100.0, 62.0], "angle": 20.0, "variant": "rotated"}))
    assert item_to_geometry("rectangle", item, prev={"variant": "rotated"}).get("variant") == "rotated"
    assert "variant" not in item_to_geometry("rectangle", item)      # no prev → dropped


def test_move_is_screen_intuitive_under_y_inversion(_app):
    ctrl = _ctrl(_app)
    ctrl.view_box.invertY(True)
    _draft(ctrl, "rectangle", {"bounds": [0.0, 100.0, 50.0, 60.0]})
    y0 = ctrl.draft.geometry["bounds"][1]
    ctrl._handle_arrow_key(_key(Qt.Key.Key_Up))     # y-inverted: up on screen == lower data-Y
    assert ctrl.draft.geometry["bounds"][1] == pytest.approx(y0 - 1.0, abs=1e-6)


# ==================================================================== point

def test_point_arrow_moves_marker_and_keeps_depth(_app):
    ctrl = _ctrl(_app)
    _draft(ctrl, "point", {"point": [100.0, 200.0, 50.0]})
    ctrl._handle_arrow_key(_key(Qt.Key.Key_Right))
    p = ctrl.draft.geometry["point"]
    assert (round(p[0]), round(p[1]), round(p[2])) == (101, 200, 50)  # depth preserved


def test_session_point_is_arrow_target_after_placing(_app):
    ctrl = _ctrl(_app)
    ctrl._commit_point((120.0, 140.0))              # drawn but not in the Manager
    assert ctrl._active_session_point_id is not None
    ctrl._handle_arrow_key(_key(Qt.Key.Key_Down))   # non-inverted: down == -Y
    rec = ctrl._session_points[-1]
    assert round(rec.geometry["point"][1]) == 139


# ==================================================================== line

def test_line_vertex_moves_only_that_endpoint(_app):
    ctrl = _ctrl(_app)
    _draft(ctrl, "line", {"points": [[0.0, 0.0], [100.0, 100.0]], "closed": False})
    _select_handle(ctrl, ("vertex", 0))
    ctrl._handle_arrow_key(_key(Qt.Key.Key_Right))
    pts = ctrl.draft.geometry["points"]
    assert round(pts[0][0]) == 1 and round(pts[1][0]) == 100   # only endpoint 0 moved


def test_line_whole_move_shifts_both_endpoints(_app):
    ctrl = _ctrl(_app)
    _draft(ctrl, "line", {"points": [[0.0, 0.0], [100.0, 100.0]], "closed": False})
    ctrl._handle_arrow_key(_key(Qt.Key.Key_Right))  # default handle == "move"
    pts = ctrl.draft.geometry["points"]
    assert round(pts[0][0]) == 1 and round(pts[1][0]) == 101


# ================================================= polygon / freehand / etc.

def test_polygon_vertex_moves_only_that_vertex(_app):
    ctrl = _ctrl(_app)
    _draft(ctrl, "polygon", {"points": [[0.0, 0.0], [100.0, 0.0], [50.0, 100.0]], "closed": True})
    _select_handle(ctrl, ("vertex", 2))
    ctrl._handle_arrow_key(_key(Qt.Key.Key_Up))     # non-inverted up == +Y
    pts = ctrl.draft.geometry["points"]
    assert round(pts[2][1]) == 101
    assert round(pts[0][1]) == 0 and round(pts[1][1]) == 0


def test_polygon_whole_move_shifts_all_vertices(_app):
    ctrl = _ctrl(_app)
    _draft(ctrl, "polygon", {"points": [[0.0, 0.0], [100.0, 0.0], [50.0, 100.0]], "closed": True})
    ctrl._handle_arrow_key(_key(Qt.Key.Key_Right))
    pts = ctrl.draft.geometry["points"]
    assert [round(p[0]) for p in pts] == [1, 101, 51]


# ========================================================== role classification

def test_role_classification_rectangle(_app):
    ctrl = _ctrl(_app)
    rec = RoiRecord.create("rectangle", {"bounds": [0.0, 0.0, 100.0, 50.0]})
    assert ctrl._kbd_role_at(rec, (50.0, 25.0)) == "move"               # body centre
    assert ctrl._kbd_role_at(rec, (5000.0, 5000.0)) is None             # far away
    assert ctrl._kbd_role_at(rec, (100.0, 50.0)) == ("scale", 1.0, 1.0)  # top-right corner
    # plain rectangle has no rotate handle → the old top-handle spot is not "rotate"
    assert ctrl._kbd_role_at(rec, (50.0, 90.0)) != "rotate"


def test_role_classification_vertex_and_point(_app):
    ctrl = _ctrl(_app)
    poly = RoiRecord.create("polygon", {"points": [[0.0, 0.0], [100.0, 0.0], [50.0, 100.0]], "closed": True})
    assert ctrl._kbd_role_at(poly, (100.0, 0.0)) == ("vertex", 1)       # click near vertex 1
    assert ctrl._kbd_role_at(poly, (5000.0, 5000.0)) is None
    pt = RoiRecord.create("point", {"point": [10.0, 20.0, 0.0]})
    assert ctrl._kbd_role_at(pt, (10.0, 20.0)) == "move"
    assert ctrl._kbd_role_at(pt, (5000.0, 5000.0)) is None


def test_switching_active_roi_resets_handle_to_move(_app):
    ctrl = _ctrl(_app)
    _draft(ctrl, "rectangle", {"bounds": [0.0, 0.0, 100.0, 50.0]})
    ctrl._kbd_handle = ("scale", 1.0, 1.0)
    # a fresh draft is a different ROI → the stale scale handle must not resize it
    _draft(ctrl, "polygon", {"points": [[0.0, 0.0], [100.0, 0.0], [50.0, 100.0]], "closed": True})
    ctrl._handle_arrow_key(_key(Qt.Key.Key_Right))
    assert [round(p[0]) for p in ctrl.draft.geometry["points"]] == [1, 101, 51]  # moved whole


# ==================================================== eventFilter / no target

def test_eventfilter_routes_arrow_from_view_widget(_app):
    ctrl = _ctrl(_app)
    _draft(ctrl, "rectangle", {"bounds": [10.0, 10.0, 100.0, 100.0]})
    assert ctrl.eventFilter(ctrl.view_widget, _key(Qt.Key.Key_Right)) is True
    assert round(ctrl.draft.geometry["bounds"][0]) == 11


def test_arrow_from_registered_key_source_moves_roi(_app):
    """A registered extra key source (e.g. the render window's pg.ImageView, which
    otherwise eats arrow keys) routes the arrow nudge to the active ROI."""
    ctrl = _ctrl(_app)
    _draft(ctrl, "line", {"points": [[0.0, 0.0], [100.0, 100.0]], "closed": False})
    other = QWidget()
    assert ctrl.eventFilter(other, _key(Qt.Key.Key_Right)) is False   # not a source yet
    ctrl.add_key_event_source(other)
    assert ctrl.eventFilter(other, _key(Qt.Key.Key_Right)) is True    # now handled
    assert round(ctrl.draft.geometry["points"][0][0]) == 1


def test_arrow_ignored_when_no_active_roi(_app):
    ctrl = _ctrl(_app)
    assert ctrl._kbd_target() is None
    assert ctrl._handle_arrow_key(_key(Qt.Key.Key_Left)) is False


# ================================================ 't' → add to ROI Manager

def test_t_key_adds_draft_to_manager(_app):
    ctrl = _ctrl(_app)
    _draft(ctrl, "rectangle", {"bounds": [0.0, 0.0, 100.0, 50.0]})
    assert len(ctrl.store.records) == 0
    assert ctrl.eventFilter(ctrl.view_widget, _key(Qt.Key.Key_T)) is True
    assert [r.type for r in ctrl.store.records] == ["rectangle"]
    assert ctrl.draft is None                     # the draft was consumed


def test_t_key_adds_session_point_to_manager(_app):
    ctrl = _ctrl(_app)
    ctrl._commit_point((50.0, 60.0))              # drawn but not in the Manager
    assert ctrl.eventFilter(ctrl.view_widget, _key(Qt.Key.Key_T)) is True
    assert [r.type for r in ctrl.store.records] == ["point"]
    assert ctrl._session_points == []             # moved out of the pending set


def test_t_key_ignored_without_active_roi(_app):
    ctrl = _ctrl(_app)
    assert ctrl.eventFilter(ctrl.view_widget, _key(Qt.Key.Key_T)) is False
    assert len(ctrl.store.records) == 0


def test_t_key_ignored_with_modifier(_app):
    ctrl = _ctrl(_app)
    _draft(ctrl, "rectangle", {"bounds": [0.0, 0.0, 100.0, 50.0]})
    ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_T, Qt.KeyboardModifier.ControlModifier)
    assert ctrl.eventFilter(ctrl.view_widget, ev) is False
    assert len(ctrl.store.records) == 0           # Ctrl+T not consumed as add-to-manager


def test_new_line_draft_clears_previous_region_highlight(_app):
    """Drawing a line after a rectangle/oval selection must clear the region's data
    highlight (its active-draft mask), not just the shape — a non-region draft runs
    no selection pass, so the old mask has to be dropped when the draft is replaced."""
    from types import SimpleNamespace

    ds = SimpleNamespace(state={}, derived={})
    notified = []
    state = SimpleNamespace(
        prefs={"plot": {"roi_color": "Yellow"}},
        datasets=[ds], active_dataset=ds,
        notify_roi_selection_changed=lambda idx: notified.append(idx),
    )
    owner = _Owner()
    owner._state = state
    plot = pg.PlotWidget()
    ctrl = RoiOverlayController(RoiStore(), owner, plot, plot.getPlotItem())

    _draft(ctrl, "rectangle", {"bounds": [0.0, 0.0, 100.0, 50.0]})
    rid = ctrl.draft.id
    # the highlight state a drawn region draft leaves behind
    ds.state["active_roi_draft_id"] = rid
    ds.state["roi_masks"] = {rid: {"key": "roi_mask_key"}}
    ds.derived["roi_mask_key"] = object()

    # switch to the line tool and draw a line → replaces the rectangle draft
    ctrl._set_draft(RoiRecord.create("line", {"points": [[0.0, 0.0], [50.0, 50.0]], "closed": False}))

    assert ctrl.draft.type == "line"
    assert ds.state.get("active_roi_draft_id") is None      # highlight marker cleared
    assert "roi_mask_key" not in ds.derived                 # mask removed → highlight gone
    assert ds.state["roi_masks"].get(rid) is None
    assert notified                                          # windows told to re-draw


def test_add_to_manager_clears_active_draft_highlight_marker(_app):
    """Filing a draft into the Manager must drop the dataset's active-draft marker
    (so the in-ROI highlight clears with it) while keeping the mask for re-select."""
    from types import SimpleNamespace

    ds = SimpleNamespace(state={}, derived={})
    notified = []
    state = SimpleNamespace(
        prefs={"plot": {"roi_color": "Yellow"}},
        datasets=[ds],
        notify_roi_selection_changed=lambda idx: notified.append(idx),
    )
    owner = _Owner()
    owner._state = state
    plot = pg.PlotWidget()
    ctrl = RoiOverlayController(RoiStore(), owner, plot, plot.getPlotItem())

    _draft(ctrl, "rectangle", {"bounds": [0.0, 0.0, 100.0, 50.0]})
    rid = ctrl.draft.id
    # simulate the highlight state a drawn region draft leaves behind
    ds.state["active_roi_draft_id"] = rid
    ds.state["roi_masks"] = {rid: {"key": "roi_mask_key"}}
    ds.derived["roi_mask_key"] = object()

    assert ctrl.eventFilter(ctrl.view_widget, _key(Qt.Key.Key_T)) is True
    assert ds.state.get("active_roi_draft_id") is None      # marker demoted → highlight clears
    assert "roi_mask_key" in ds.derived                     # mask kept for re-selection
    assert ds.state["roi_masks"].get(rid) is not None
    assert notified == [0]                                   # windows told to re-draw


# ==================================================================== stored

def test_stored_roi_arrow_moves_display_not_record_until_update(_app):
    ctrl = _ctrl(_app)
    rec = RoiRecord.create("rectangle", {"bounds": [0.0, 0.0, 100.0, 50.0]})
    ctrl.store.add(rec)
    ctrl.store.select([rec.id])
    ctrl.refresh()
    assert ctrl._handle_arrow_key(_key(Qt.Key.Key_Right)) is True
    assert ctrl.store.records[0].geometry["bounds"] == [0.0, 0.0, 100.0, 50.0]  # untouched
    moved = item_to_geometry("rectangle", ctrl.items[rec.id])
    assert round(moved["bounds"][0]) == 1
