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
def test_arrow_keys_rotate_via_rotation_handle(_app, roi_type):
    ctrl = _ctrl(_app)
    _draft(ctrl, roi_type, {"bounds": [0.0, 0.0, 100.0, 50.0]})
    _select_handle(ctrl, "rotate")
    ctrl._handle_arrow_key(_key(Qt.Key.Key_Right))  # non-inverted: CW == negative angle
    assert ctrl.draft.geometry.get("angle") == pytest.approx(-1.0, abs=1e-6)
    b = ctrl.draft.geometry["bounds"]
    assert (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0) == pytest.approx((50.0, 25.0), abs=1e-6)


def test_oval_has_bounding_box_handles(_app):
    ctrl = _ctrl(_app)
    item = ctrl._make_item(RoiRecord.create("oval", {"bounds": [0.0, 0.0, 100.0, 50.0]}))
    names = [h.get("name") for h in item.handles]
    assert len(item.handles) == 9 and "rotate" in names   # 8 scale + rotate, like the rect


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
    tol = ctrl._hit_tolerance()
    hy = 25.0 + 25.0 + max(25.0 * 0.22, tol * 2.0)
    assert ctrl._kbd_role_at(rec, (50.0, hy)) == "rotate"


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
