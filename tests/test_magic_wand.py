"""Magic wand: grow a region from a click, trace its outline, refuse honestly.

The wand has no pixels to flood-fill, so the contract is: connect points within
a Cartesian distance whose attribute value is within a band of the **clicked**
point's own value, and hand back a polygon -- holes included -- that every
existing ROI consumer can read.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from minflux_viewer.analysis import wand_select as ws
from minflux_viewer.core.dataset import AttrStore, DataProp, FileInfo, MinfluxDataset
from minflux_viewer.core.roi import ROI_TOOLS, ROI_TYPES, RoiRecord
from minflux_viewer.core.roi_selection import polygon_mask


@pytest.fixture
def _qt_app():
    from PyQt6.QtWidgets import QApplication
    if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
        pytest.skip("No display available for Qt tests")
    return QApplication.instance() or QApplication(sys.argv)


# --------------------------------------------------------------- the data
def _two_clusters(n_per: int = 60, gap_nm: float = 800.0, seed: int = 0):
    """Two spatially separate clusters with distinct ``efo``, in nm."""
    rng = np.random.default_rng(seed)
    a = rng.normal((0.0, 0.0), 15.0, size=(n_per, 2))
    b = rng.normal((gap_nm, gap_nm), 15.0, size=(n_per, 2))
    return np.vstack([a, b]), np.concatenate([np.full(n_per, 10.0),
                                              np.full(n_per, 100.0)])


def _make_ds(n_per: int = 60) -> MinfluxDataset:
    xy, efo = _two_clusters(n_per)
    n = xy.shape[0]
    tid = np.repeat(np.arange(2), n_per)
    trace_idx = np.column_stack([np.arange(2) * n_per,
                                 np.arange(2) * n_per + n_per - 1])
    prop = DataProp(
        num_loc=n, num_itr=1, num_dim=2, num_traces=2,
        trace_idx=trace_idx, num_loc_per_trace=np.full(2, n_per),
        attr_names=["loc_x", "loc_y", "loc_z", "tid", "efo", "den", "ftr", "idx"],
    )
    attrs = AttrStore({
        "loc_x": xy[:, 0] / 1e9,                 # nm -> m
        "loc_y": xy[:, 1] / 1e9,
        "loc_z": np.zeros(n),
        "tid": tid.astype(float),
        "efo": efo,
        "den": efo.copy(),                       # a stand-in density field
        "ftr": np.ones(n, dtype=bool),
        "idx": np.arange(1, n + 1, dtype=np.uint32),
    })
    return MinfluxDataset(file=FileInfo(name="wand.mat", folder="/tmp"),
                          prop=prop, attr=attrs)


# ----------------------------------------------------------- the model
def test_magic_wand_is_a_tool_and_files_an_ordinary_polygon():
    """Same discipline as the rotated variants: a drawing tool need not be a
    record type. Keeping the record a plain polygon is what lets every existing
    consumer -- masks, crop, channels, save, ImageJ export -- read it unchanged."""
    assert "magic_wand" in ROI_TOOLS
    assert "magic_wand" not in ROI_TYPES


def test_distance_alone_separates_two_clusters():
    xy, _efo = _two_clusters()
    r = ws.wand_polygon(xy, np.zeros(xy.shape[0]), (0.0, 0.0),
                        distance_nm=60.0, value_percent=100.0)
    assert r.n_selected == 60           # the near cluster only


def test_the_value_band_separates_populations_that_overlap_in_space():
    """The point of growing on an attribute: two interleaved populations are
    separable when their values differ, which distance alone cannot do."""
    a, _ = _two_clusters()
    near = a[:60]
    mixed = np.vstack([near, near + (3.0, 3.0)])
    values = np.concatenate([np.full(60, 10.0), np.full(60, 100.0)])
    tight = ws.wand_polygon(mixed, values, (0.0, 0.0),
                            distance_nm=60.0, value_percent=10.0)
    wide = ws.wand_polygon(mixed, values, (0.0, 0.0),
                           distance_nm=60.0, value_percent=100.0)
    assert tight.n_selected == 60        # the band did the work
    assert wide.n_selected == 120        # a wide band re-unites them


def test_the_band_is_measured_against_the_seed_not_the_neighbour():
    """⚠ A chain comparison lets the value drift arbitrarily far from the click
    -- you walk a gradient and select everything. A ramp of points 1 nm apart
    whose value rises by 1 per point must stop at the band, not run the ramp."""
    n = 200
    pts = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    values = np.arange(n, dtype=float)               # range 0..199
    r = ws.wand_polygon(pts, values, (0.0, 0.0), distance_nm=5.0,
                        value_tolerance=10.0)
    assert r.n_selected == 11, r.n_selected          # values 0..10 inclusive


def test_the_traced_outline_contains_every_point_it_was_traced_from():
    xy, efo = _two_clusters()
    r = ws.wand_polygon(xy, efo, (0.0, 0.0), distance_nm=60.0, value_percent=10.0)
    rec = RoiRecord.create("polygon", {"points": r.polygon, "closed": True})
    inside = polygon_mask(xy[:, 0], xy[:, 1], rec)
    assert inside[r.mask].all()
    assert not inside[~r.mask].any()                 # and nothing from the far cluster


def test_a_ring_selection_keeps_its_hole():
    """⚠ Concatenating an outer ring and a hole ring is NOT enough: the implicit
    seam edge flips the parity of every point beside it, and 16 of 4000 annulus
    points fell outside their own outline. The bridge is walked out and back so
    the two coincident segments cancel."""
    rng = np.random.default_rng(1)
    ang = rng.uniform(0.0, 2.0 * np.pi, 4000)
    rad = rng.uniform(150.0, 260.0, 4000)
    ring = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
    r = ws.wand_polygon(ring, np.ones(ring.shape[0]), (200.0, 0.0),
                        distance_nm=30.0, value_percent=50.0)
    assert r.rings == 2                              # a boundary and one hole
    rec = RoiRecord.create("polygon", {"points": r.polygon, "closed": True})
    assert polygon_mask(ring[:, 0], ring[:, 1], rec)[r.mask].all()
    probe = np.array([[0.0, 0.0], [205.0, 0.0], [0.0, 205.0],
                      [-205.0, 0.0], [0.0, -205.0], [400.0, 0.0]])
    inside = polygon_mask(probe[:, 0], probe[:, 1], rec).tolist()
    assert inside == [False, True, True, True, True, False]


def test_a_one_cell_wide_arm_survives_the_outline():
    """⚠ A zero cross product is not proof of a straight run: where the boundary
    doubles back along a thin arm the turn-around vertex is also collinear, and
    dropping it collapsed the arm out of the polygon."""
    rng = np.random.default_rng(2)
    blob = rng.normal((0.0, 0.0), 10.0, size=(300, 2))
    arm = np.column_stack([np.linspace(20.0, 200.0, 40), np.zeros(40)])
    pts = np.vstack([blob, arm])
    r = ws.wand_polygon(pts, np.ones(pts.shape[0]), (0.0, 0.0),
                        distance_nm=12.0, value_percent=50.0)
    rec = RoiRecord.create("polygon", {"points": r.polygon, "closed": True})
    assert polygon_mask(pts[:, 0], pts[:, 1], rec)[r.mask].all()


def test_speckle_holes_are_dropped_and_counted_never_silently():
    """Uniform noise produced 3479 one-cell holes of 4599 rings, a 33,556-vertex
    polygon and a 46 s click. They are dropped, which makes the outline slightly
    generous -- so the count is reported rather than hidden."""
    rng = np.random.default_rng(3)
    pts = rng.normal(0.0, 600.0, size=(120_000, 2))
    values = rng.normal(0.0, 1.0, pts.shape[0])
    r = ws.wand_polygon(pts, values, (0.0, 0.0), distance_nm=20.0, value_percent=20.0)
    assert r.rings <= ws.MAX_OUTLINE_RINGS
    assert r.rings_dropped >= 0
    assert len(r.polygon) < 20_000            # bounded, unlike the 33k it replaced


def test_a_constant_attribute_gives_an_exact_band_not_an_error():
    assert ws.value_tolerance_from_percent(np.full(10, 5.0), 10.0) == 0.0
    pts = np.column_stack([np.arange(10.0), np.zeros(10)])
    r = ws.wand_polygon(pts, np.full(10, 5.0), (0.0, 0.0),
                        distance_nm=2.0, value_percent=10.0)
    assert r.n_selected == 10                 # equal values are all admissible


# ------------------------------------------------- the row-aligned entry point
def test_wand_from_rows_excludes_invisible_rows_before_growing():
    """⚠ Before, not after: growing through filtered-out localizations bridges a
    gap the user can see is empty, and the outline then encloses points that are
    not on screen."""
    pts = np.column_stack([np.arange(0.0, 100.0, 10.0), np.zeros(10),
                           np.zeros(10)])          # 10 points, 10 nm apart
    values = np.ones(10)
    base = np.ones(10, dtype=bool)
    base[3:6] = False                              # a filtered-out gap
    r = ws.wand_from_rows(pts, values, base, (0, 1), (0.0, 0.0),
                          distance_nm=15.0, value_percent=50.0)
    assert r.mask.size == 10                       # full length, not the subset
    assert r.mask[:3].all() and not r.mask[3:].any()
    assert r.n_candidates == 7                     # only the visible rows


def test_wand_from_rows_reads_the_columns_the_view_plots():
    """A view states the axes it shows, so an XZ pane grows on X and Z."""
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 900.0, 5.0], [5.0, 0.0, 0.0]])
    values = np.ones(3)
    xy = ws.wand_from_rows(pts, values, None, (0, 1), (0.0, 0.0),
                           distance_nm=20.0, value_percent=50.0)
    xz = ws.wand_from_rows(pts, values, None, (0, 2), (0.0, 0.0),
                           distance_nm=20.0, value_percent=50.0)
    assert xy.n_selected == 2          # row 1 is 900 nm away in Y
    assert xz.n_selected == 3          # in XZ it is 5 nm away, so it joins


# ------------------------------------------------------- the scatter plot hook
def test_scatter_wand_grows_on_the_color_by_attribute(_qt_app):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.scatter_window import ScatterWindow

    state = AppState()
    state.add_dataset(_make_ds())
    win = ScatterWindow(state, dataset_idx=0)
    try:
        win._cbar_combo.setCurrentText("efo")
        kind, geometry, message = win.wand_select((0.0, 0.0), distance_nm=60.0,
                                                  value_percent=10.0)
        assert kind == "polygon", message
        assert "efo" in message
        rec = RoiRecord.create(kind, geometry)
        selection = win.compute_roi_selection(rec)
        assert selection is not None
        _ds, mask, _ctx = selection
        assert int(mask.sum()) == 60           # the clicked cluster, not both
    finally:
        win.close()


def test_scatter_wand_refuses_in_3d_with_the_reason(_qt_app):
    """A click in the OpenGL view carries no data coordinate, so there is nothing
    to seed from -- said out loud rather than silently doing nothing."""
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.scatter_window import ScatterWindow

    state = AppState()
    state.add_dataset(_make_ds())
    win = ScatterWindow(state, dataset_idx=0)
    try:
        win._axis_combo.setCurrentText("3D")
        kind, _geometry, message = win.wand_select((0.0, 0.0), distance_nm=60.0,
                                                   value_percent=10.0)
        assert kind is None
        assert "2-D" in message
    finally:
        win.close()


def test_scatter_wand_refuses_a_solid_colour_with_the_reason(_qt_app):
    """A channel coloured by a solid has no per-point value to grow on, so the
    wand refuses and names the fix rather than quietly substituting a field the
    user never chose (``den``, say)."""
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.scatter_window import ScatterWindow

    state = AppState()
    state.add_dataset(_make_ds())
    win = ScatterWindow(state, dataset_idx=0)
    try:
        ds = state.datasets[0]
        win._wand_source = lambda: (ds, "")        # what a solid colour reports
        kind, _geometry, message = win.wand_select((0.0, 0.0), distance_nm=60.0,
                                                   value_percent=10.0)
        assert kind is None
        assert "Color by" in message
    finally:
        win.close()


def test_scatter_wand_refuses_an_attribute_with_no_per_loc_values(_qt_app):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.scatter_window import ScatterWindow

    state = AppState()
    state.add_dataset(_make_ds())
    win = ScatterWindow(state, dataset_idx=0)
    try:
        ds = state.datasets[0]
        win._wand_source = lambda: (ds, "efo")
        win._color_cache_for_dataset = lambda _ds, _name: None
        kind, _geometry, message = win.wand_select((0.0, 0.0), distance_nm=60.0,
                                                   value_percent=10.0)
        assert kind is None
        assert "efo" in message
    finally:
        win.close()


# -------------------------------------------------------- the render view hook
def test_render_wand_grows_on_local_density(_qt_app):
    """The render view has no per-point colour, so the field is local density --
    which is what the picture is already showing. Exercised through the hook on
    a bare instance: no window is needed to test the rule."""
    from types import SimpleNamespace

    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.render_window import RenderWindow

    state = AppState()
    state.add_dataset(_make_ds())
    ds = state.datasets[0]
    xy, _efo = _two_clusters()

    win = RenderWindow.__new__(RenderWindow)        # no Qt construction
    win._state = state
    win._idx = 0
    locs = np.column_stack([xy, np.zeros(xy.shape[0])])
    win._raw_render_locs = lambda _ds: locs
    win._active_plane = lambda: "XY"
    win._has_depth = False                          # "All Z" — no slab gating
    win._all_depth_check = SimpleNamespace(isChecked=lambda: True)
    win._orientation = "XY"                         # not the orthogonal view
    win._wand_density = lambda _ds, _locs, _r: (
        np.asarray(ds.attr["den"], dtype=float), "den")

    kind, geometry, message = win.wand_select((0.0, 0.0), distance_nm=60.0,
                                              value_percent=10.0)
    assert kind == "polygon", message          # a 2-D dataset keeps a polygon
    assert "den" in message
    rec = RoiRecord.create(kind, geometry)
    inside = polygon_mask(xy[:, 0], xy[:, 1], rec)
    assert int(inside.sum()) == 60


def test_render_wand_refuses_when_the_view_shows_an_image(_qt_app):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.render_window import RenderWindow

    state = AppState()
    state.add_dataset(_make_ds())
    win = RenderWindow.__new__(RenderWindow)
    win._state = state
    win._idx = 0
    win._raw_render_locs = lambda _ds: np.zeros((0, 3))
    win._active_plane = lambda: "XY"
    win._has_depth = False
    win._orientation = "XY"
    kind, _geometry, message = win.wand_select((0.0, 0.0), distance_nm=60.0,
                                               value_percent=10.0)
    assert kind is None
    assert "image" in message


# ------------------------------------------------------------- the overlay
from PyQt6.QtWidgets import QWidget                          # noqa: E402


class _WandOwner(QWidget):
    """Stand-in view: serves one wand click and records what it was asked."""

    def __init__(self, polygon, message="ok"):
        from types import SimpleNamespace
        super().__init__()
        self._polygon = polygon
        self._message = message
        self.calls = []
        self._state = SimpleNamespace(
            prefs={"plot": {"roi_color": "Yellow"},
                   "wand": {"distance_nm": 33.0, "value_percent": 7.0}},
            datasets=[])

    def roi_view_plane(self):
        return "XY"

    def roi_depth_center(self):
        return 0.0

    def roi_depth_range(self):
        return None

    def compute_roi_selection(self, record, *, columns=None):
        return None

    def normalize_roi_record(self, record):
        return record

    def wand_select(self, position, *, distance_nm, value_percent):
        self.calls.append((position, distance_nm, value_percent))
        if self._polygon is None:
            return None, None, self._message
        return "polygon", {"points": self._polygon, "closed": True}, self._message


def _controller(owner):
    import pyqtgraph as pg

    from minflux_viewer.core.roi import RoiStore
    from minflux_viewer.ui.roi_overlay import RoiOverlayController

    plot = pg.PlotWidget()
    store = RoiStore()
    return RoiOverlayController(store, owner, plot, plot.getPlotItem()), store, plot


def test_a_wand_click_drafts_a_polygon_and_passes_the_saved_tolerances(_qt_app):
    square = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]
    owner = _WandOwner(square, "grew 12 of 20")
    ctrl, _store, _plot = _controller(owner)
    try:
        ctrl._wand_click((5.0, 5.0))
        assert ctrl.draft is not None
        assert ctrl.draft.type == "polygon"
        assert len(ctrl.draft.geometry["points"]) == 4
        assert owner.calls == [((5.0, 5.0), 33.0, 7.0)]   # straight from prefs
    finally:
        ctrl.dispose()


def test_a_refused_wand_click_leaves_no_draft(_qt_app):
    owner = _WandOwner(None, "pick an attribute instead of a solid colour")
    ctrl, _store, _plot = _controller(owner)
    try:
        ctrl._wand_click((5.0, 5.0))
        assert ctrl.draft is None
    finally:
        ctrl.dispose()


def test_redo_wand_reuses_the_last_seed(_qt_app):
    square = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    owner = _WandOwner(square)
    ctrl, _store, _plot = _controller(owner)
    try:
        assert ctrl.redo_wand() is False        # nothing clicked yet
        ctrl._wand_click((7.0, 8.0))
        assert ctrl.redo_wand() is True
        assert [c[0] for c in owner.calls] == [(7.0, 8.0), (7.0, 8.0)]
    finally:
        ctrl.dispose()


# -------------------------------------------------------------- the dialog
def test_the_dialog_writes_the_tolerances_and_announces_them(_qt_app):
    from minflux_viewer.ui.wand_dialog import MagicWandDialog

    prefs = {"wand": {"distance_nm": 20.0, "value_percent": 10.0}}
    dialog = MagicWandDialog(prefs)
    try:
        fired = []
        dialog.parameters_changed.connect(lambda: fired.append(True))
        dialog._distance.setValue(75)
        dialog._value.setValue(42)
        # written through at once, so a click before the timer uses what is shown
        assert prefs["wand"] == {"distance_nm": 75.0, "value_percent": 42.0}
        assert dialog.parameters() == {"distance_nm": 75.0, "value_percent": 42.0}
        dialog._apply_timer.timeout.emit()       # what the coalescing timer does
        assert fired
        dialog._reset()
        assert prefs["wand"]["distance_nm"] == float(ws.DEFAULT_DISTANCE_NM)
    finally:
        dialog.close()


def test_wand_parameters_falls_back_to_the_defaults():
    assert ws.wand_parameters(None) == {
        "distance_nm": ws.DEFAULT_DISTANCE_NM,
        "value_percent": ws.DEFAULT_VALUE_TOLERANCE_PCT,
    }
    assert ws.wand_parameters({"wand": {"distance_nm": "nonsense"}})["distance_nm"] == \
        ws.DEFAULT_DISTANCE_NM


# ------------------------------------------------------------- end to end
def test_a_click_in_a_real_scatter_window_files_a_usable_roi(_qt_app):
    """The whole path in one test: arm the tool, click, and the result is an
    ordinary polygon ROI whose recomputed selection is the clicked cluster.

    The tool stays armed afterwards, like the point tool, so several structures
    can be picked in a row.
    """
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.scatter_window import ScatterWindow

    state = AppState()
    state.add_dataset(_make_ds())
    win = ScatterWindow(state, dataset_idx=0)
    try:
        win._cbar_combo.setCurrentText("efo")
        state.rois.set_tool("magic_wand")
        controller = win._roi_overlay
        controller._wand_click((0.0, 0.0))

        assert controller.draft is not None
        assert controller.draft.type == "polygon"
        _ds, mask, _ctx = win.compute_roi_selection(controller.draft)
        assert int(mask.sum()) == 60                  # the clicked cluster only
        assert mask.size == 120

        # the seed is remembered, so a tolerance change re-grows it
        assert controller._wand_last_pos == (0.0, 0.0)
        assert state._wand_last_controller is controller
        assert controller.redo_wand() is True

        # and it files like any other ROI
        controller._add_hit_to_manager("draft", controller.draft)
        assert [r.type for r in state.rois.records] == ["polygon"]
        assert state.rois.active_tool == "magic_wand"
    finally:
        win.close()


# ------------------------------------- a click on empty space selects nothing
def test_a_click_on_empty_space_selects_nothing():
    """⚠ Without a bound on the seed the wand grew from whatever point happened
    to be nearest, however far away, so clicking a void appeared to select a
    region out of nowhere. One tolerance governs the whole gesture."""
    pts = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [1000.0, 1000.0]])
    values = np.ones(4)
    assert ws.wand_polygon(pts, values, (500.0, 500.0),
                           distance_nm=20.0, value_percent=50.0) is None
    near = ws.wand_polygon(pts, values, (0.0, 0.0),
                           distance_nm=20.0, value_percent=50.0)
    assert near.n_selected == 3                 # the reachable run, not the far point


def test_the_seed_bound_is_exactly_the_connectivity_distance():
    one = np.array([[5.0, 0.0]])
    assert ws.wand_polygon(one, np.ones(1), (0.0, 0.0),
                           distance_nm=4.9, value_percent=50.0) is None
    assert ws.wand_polygon(one, np.ones(1), (0.0, 0.0),
                           distance_nm=5.0, value_percent=50.0) is not None


def test_nearest_index_honours_its_bound():
    pts = np.array([[0.0, 0.0], [1000.0, 0.0]])
    assert ws.nearest_index(pts, (500.0, 0.0)) in (0, 1)
    assert ws.nearest_index(pts, (500.0, 0.0), max_distance=20.0) is None


def test_scatter_reports_an_empty_click_rather_than_selecting(_qt_app):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.scatter_window import ScatterWindow

    state = AppState()
    state.add_dataset(_make_ds())
    win = ScatterWindow(state, dataset_idx=0)
    try:
        win._cbar_combo.setCurrentText("efo")
        kind, _geometry, message = win.wand_select((400.0, 400.0), distance_nm=20.0,
                                                   value_percent=10.0)
        assert kind is None                     # the gap between the two clusters
        assert "20 nm" in message
    finally:
        win.close()


# ------------------------------------------- the render view keeps its Z slice
def _render_stub(locs, den, num_dim, *, all_z=True, depth=(-40.0, 40.0)):
    from types import SimpleNamespace

    from minflux_viewer.ui.render_window import RenderWindow

    ds = SimpleNamespace(attr={"den": den},
                         filter_mask=np.ones(locs.shape[0], dtype=bool),
                         prop=SimpleNamespace(num_dim=num_dim))
    win = RenderWindow.__new__(RenderWindow)
    win._state = SimpleNamespace(datasets=[ds])
    win._idx = 0
    win._raw_render_locs = lambda _d: locs
    win._active_plane = lambda: "XY"
    win._has_depth = True
    win._all_depth_check = SimpleNamespace(isChecked=lambda: all_z)
    win._depth_range = depth
    win._orientation = "XY"                        # a flat projection, not ortho
    return win, ds


def _three_sheets(n=400, seed=0):
    rng = np.random.default_rng(seed)
    sheet = rng.normal(0.0, 60.0, size=(n, 2))
    return np.vstack([np.column_stack([sheet, np.full(n, z)])
                      for z in (-200.0, 0.0, 200.0)])


def test_the_depth_slice_restricts_what_the_render_wand_can_reach():
    locs = _three_sheets()
    win, _ds = _render_stub(locs, np.ones(locs.shape[0]), 3, all_z=False)
    gate = win._wand_depth_mask(locs, (0, 1))
    assert int(gate.sum()) == 400               # only the middle sheet is visible
    win_all, _ = _render_stub(locs, np.ones(locs.shape[0]), 3, all_z=True)
    assert int(win_all._wand_depth_mask(locs, (0, 1)).sum()) == locs.shape[0]


def test_a_render_selection_in_3d_data_is_a_polyhedron_that_keeps_the_slab():
    """⚠ The whole point of the polyhedron: a 2-D polygon's mask spans the whole
    depth axis by design, so a selection grown inside a thin Z slab came back out
    selecting everything above and below it."""
    from minflux_viewer.core.roi_volume import roi_volume_mask

    locs = _three_sheets()
    win, _ds = _render_stub(locs, np.ones(locs.shape[0]), 3, all_z=False)
    kind, geometry, message = win.wand_select((0.0, 0.0), distance_nm=30.0,
                                              value_percent=50.0)
    assert kind == "polyhedron", message
    record = RoiRecord.create(kind, geometry)
    # strict=True is what compute_roi_selection uses: it must not raise
    mask = roi_volume_mask(locs[:, 0], locs[:, 1], locs[:, 2], record)
    assert sorted({float(v) for v in locs[mask, 2]}) == [0.0]
    assert "polyhedron" in message and "enclosed" in message


def test_a_render_selection_in_2d_data_is_still_a_polygon():
    locs = _three_sheets()
    locs[:, 2] = 0.0
    win, _ds = _render_stub(locs, np.ones(locs.shape[0]), 2, all_z=True)
    kind, _geometry, _message = win.wand_select((0.0, 0.0), distance_nm=30.0,
                                                value_percent=50.0)
    assert kind == "polygon"


# ------------------------------------------- 3-D growing in the orthogonal view
def _two_z_planes(n=300, gap=400.0, seed=0):
    """Two clouds that OVERLAP in XY and sit *gap* apart in Z."""
    rng = np.random.default_rng(seed)
    xy = rng.normal(0.0, 30.0, size=(n, 2))
    return np.vstack([np.column_stack([xy, np.zeros(n)]),
                      np.column_stack([xy, np.full(n, gap)])])


def test_growing_in_3d_does_not_bridge_a_z_gap_that_2d_growing_does():
    """⚠ Growing in the projection treats points that merely overlap on screen
    as neighbours however far apart they are on the collapsed axis -- the same
    class of error as ignoring the depth slice. The orthogonal view shows all
    three axes, so that is where 3-D connectivity belongs."""
    locs = _two_z_planes()
    values = np.ones(locs.shape[0])

    flat = ws.wand_from_rows(locs, values, None, (0, 1), (0.0, 0.0),
                             distance_nm=40.0, value_percent=50.0, grow_3d=False)
    solid = ws.wand_from_rows(locs, values, None, (0, 1), (0.0, 0.0),
                              distance_nm=40.0, value_percent=50.0, grow_3d=True)
    assert sorted({float(v) for v in locs[flat.mask, 2]}) == [0.0, 400.0]
    assert sorted({float(v) for v in locs[solid.mask, 2]}) == [0.0]


def test_seeding_stays_two_dimensional_even_when_growing_in_3d():
    """The user clicks what they can see; the depth of a click in a projection
    is not knowable, so only connectivity becomes 3-D."""
    locs = _two_z_planes()
    values = np.ones(locs.shape[0])
    for columns in ((0, 1), (0, 2), (2, 1)):
        result = ws.wand_from_rows(locs, values, None, columns, (0.0, 0.0),
                                   distance_nm=40.0, value_percent=50.0, grow_3d=True)
        assert result is not None and result.n_selected > 0


def test_an_ortho_side_pane_serves_the_click_on_its_own_axes():
    """⚠ Reaching the window's hook through __getattr__ binds it to the WINDOW,
    whose _active_plane() is XY in orthogonal mode -- so a click in the XZ pane
    would be read as an XY click."""
    from minflux_viewer.ui.ortho_roi import OrthoPaneOwner
    from minflux_viewer.ui.ortho_view import ORTHO_AXIS_COLUMNS

    from PyQt6.QtCore import QObject

    class _Window(QObject):
        def __init__(self):
            super().__init__()
            self.seen = None

        def wand_select(self, position, **kwargs):
            self.seen = (position, kwargs)
            return "polygon", {"points": [], "closed": True}, "ok"

    window = _Window()
    for plane in ("XZ", "YZ"):
        pane = OrthoPaneOwner(window, plane)
        pane.wand_select((1.0, 2.0), distance_nm=20.0, value_percent=10.0)
        assert window.seen[1]["columns"] == ORTHO_AXIS_COLUMNS[plane]


def test_arming_the_magic_wand_does_not_open_the_orthogonal_view():
    """Unlike a cuboid or sphere tool, which need all three panes to show what
    is being made: the wand draws nothing, so it must not rearrange the view."""
    import inspect

    from minflux_viewer.core.roi_selection import VOLUME_ROI_TYPES
    from minflux_viewer.ui.main_window import MainWindow

    assert "magic_wand" not in VOLUME_ROI_TYPES
    source = inspect.getsource(MainWindow._enter_ortho_for_volume_tool)
    assert "if tool not in VOLUME_ROI_TYPES:" in source


def test_the_lasso_button_activates_its_current_variant(_qt_app):
    """⚠ Wired to the literal "magnetic_lasso", a left-click on the shared button
    re-activated the lasso and silently discarded a chosen Magic Wand, so the
    variant would not stay put."""
    import inspect
    from types import SimpleNamespace

    from minflux_viewer.core.roi import RoiStore
    from minflux_viewer.ui.main_window import MainWindow

    source = inspect.getsource(MainWindow._connect_actions)
    assert "self._on_roi_tool(self._lasso_variant, checked)" in source
    assert 'self._on_roi_tool("magnetic_lasso", checked)' not in source

    class _Label:
        def __init__(self):
            self._text = ""

        def setText(self, text):
            self._text = text

        def text(self):
            return self._text

    win = MainWindow.__new__(MainWindow)
    store = RoiStore()
    win._status_label = _Label()
    win._roi_tool_status_text = ""
    win._ui = SimpleNamespace()
    win._roi_tool_actions = {}
    win._line_variant, win._rect_variant = "line", "rectangle"
    win._oval_variant, win._poly_variant, win._point_variant = "oval", "polygon", "point"
    win._lasso_variant = "magic_wand"
    win._state = SimpleNamespace(
        status_message=SimpleNamespace(emit=win._status_label.setText),
        rois=store, datasets=[], active_idx=None,
        log=lambda message, level="INFO": None)
    win._on_roi_tool(win._lasso_variant, True)
    assert store.active_tool == "magic_wand"


def _make_3d_ds():
    locs = _two_z_planes()
    n = locs.shape[0]
    prop = DataProp(
        num_loc=n, num_itr=1, num_dim=3, num_traces=2,
        trace_idx=np.array([[0, n // 2 - 1], [n // 2, n - 1]]),
        num_loc_per_trace=np.full(2, n // 2),
        attr_names=["loc_x", "loc_y", "loc_z", "tid", "efo", "ftr", "idx"],
    )
    attrs = AttrStore({
        "loc_x": locs[:, 0] / 1e9, "loc_y": locs[:, 1] / 1e9,
        "loc_z": locs[:, 2] / 1e9,
        "tid": np.repeat([0.0, 1.0], n // 2),
        "efo": np.ones(n),
        "ftr": np.ones(n, dtype=bool),
        "idx": np.arange(1, n + 1, dtype=np.uint32),
    })
    return MinfluxDataset(file=FileInfo(name="wand3d.mat", folder="/tmp"),
                          prop=prop, attr=attrs), locs


def test_scatter_in_ortho_highlights_the_3d_selection_and_files_no_shape(_qt_app):
    """⚠ In 3-D the selection is a SET OF LOCALIZATIONS: two points can be
    neighbours in space and still disagree on the attribute, so neither a
    polygon nor a polyhedron encloses exactly what was selected. The exact mask
    is stored as the active draft mask, which the existing highlight path draws
    in every pane and in the 3-D view."""
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.core.roi_selection import active_roi_mask
    from minflux_viewer.ui.ortho_view import ORTHO_AXIS
    from minflux_viewer.ui.scatter_window import ScatterWindow

    ds, locs = _make_3d_ds()
    state = AppState()
    state.add_dataset(ds)
    win = ScatterWindow(state, dataset_idx=0)
    try:
        win._cbar_combo.setCurrentText("efo")

        kind, geometry, _msg = win.wand_select((0.0, 0.0), distance_nm=40.0,
                                               value_percent=50.0)
        assert kind == "polygon" and geometry            # a flat projection

        win._axis_combo.setCurrentText(ORTHO_AXIS)
        assert win._ortho_active()
        kind, geometry, message = win.wand_select((0.0, 0.0), distance_nm=40.0,
                                                  value_percent=50.0)
        assert kind == "highlight"
        assert geometry is None
        assert "no ROI shape" in message
        mask = active_roi_mask(ds)
        assert mask is not None
        assert sorted({float(v) for v in locs[mask, 2]}) == [0.0]
        assert state.rois.records == []                  # nothing filed
    finally:
        win.close()
