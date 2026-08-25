"""Analyze › Segmentation › Shape Model… — dialog wiring and ROI hand-off."""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from minflux_viewer.analysis.shape_segmentation import (  # noqa: E402
    SHAPE_MODELS,
    ShapeInstance,
    instance_mask,
)
from minflux_viewer.ui.shape_segmentation_dialog import (  # noqa: E402
    ShapeSegmentationWindow,
)

PIXEL = 20.0
_KEEP_ALIVE: list = []


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(sys.argv[:1])


def _two_touching_cells(seed=7):
    """A blob holding two parallel rods that touch — the hard case."""
    rng = np.random.default_rng(seed)
    shape = (320, 340)
    xs, ys = [], []
    for centre in ((2400.0, 3200.0), (3250.0, 3200.0)):
        mask = instance_mask(
            ShapeInstance("capsule", centre, 90.0, (2600.0, 900.0),
                          1.0, 1.0, False, 1.0, 1), shape, PIXEL)
        rows, cols = np.nonzero(mask)
        pick = rng.choice(rows.size, 7000, replace=True)
        xs.append((cols[pick] + rng.random(pick.size)) * PIXEL)
        ys.append((rows[pick] + rng.random(pick.size)) * PIXEL)
    return np.column_stack([np.concatenate(xs), np.concatenate(ys)])


class _Dataset:
    def __init__(self, name="cells"):
        self.name = name
        self.metadata = {}


class _Rois:
    def __init__(self):
        self.records = []
        self.show_all = False

    def add(self, record):
        self.records.append(record)

    def set_show_all(self, value):
        self.show_all = bool(value)


class _State:
    def __init__(self):
        self.datasets = [_Dataset()]
        self.active_idx = 0
        self.rois = _Rois()
        self.prefs = {}
        self.messages = []

    def log(self, message, *args, **kwargs):
        self.messages.append(message)


class _Owner:
    """Stands in for MainWindow, exercising the real add_polygon_rois contract."""

    def __init__(self, state):
        self._state = state
        self.calls = []

    def add_polygon_rois(self, idx, polygons, *, name_prefix, source,
                         stroke_color=None, names=None, log_message=None):
        from minflux_viewer.core.roi import RoiRecord
        added = 0
        for i, polygon in enumerate(polygons, start=1):
            pts = np.asarray(polygon, dtype=float).reshape(-1, 2)
            if pts.shape[0] > 2 and np.allclose(pts[0], pts[-1]):
                pts = pts[:-1]
            if pts.shape[0] < 3:
                continue
            name = (names[i - 1] if names and i - 1 < len(names)
                    else f"{name_prefix} {i}")
            rec = RoiRecord.create("polygon", {"points": pts.tolist(), "closed": True},
                                   name=name, coordinate_space="plot",
                                   stroke_color=stroke_color or "#ffff00")
            rec.context = {"dataset_idx": idx, "source": source}
            self._state.rois.add(rec)
            added += 1
        self.calls.append({"idx": idx, "n": added, "source": source,
                           "log": log_message})
        if added:
            # Mirrors MainWindow.add_polygon_rois: reveal what was just filed.
            self._state.rois.set_show_all(True)
            if log_message:
                self._state.log(log_message)
        return added


def _window(app, monkeypatch, xy):
    state = _State()
    owner = _Owner(state)
    monkeypatch.setattr(
        "minflux_viewer.core.roi_crop.display_xy_filtered", lambda ds: xy)
    win = ShapeSegmentationWindow(state, 0, owner=owner)
    return win, state, owner


def test_every_registered_shape_is_offered_and_builds_its_own_controls(
        app, monkeypatch):
    """Registering a geometry must surface it here with no change to the dialog."""
    win, _state, _owner = _window(app, monkeypatch, _two_touching_cells())
    try:
        offered = {win._model_combo.itemData(i)
                   for i in range(win._model_combo.count())}
        assert offered == set(SHAPE_MODELS)
        for index in range(win._model_combo.count()):
            win._model_combo.setCurrentIndex(index)
            key = win._model_combo.itemData(index)
            model = SHAPE_MODELS[key]
            assert set(win._range_spins) == set(model.size_names)
            prior = win._prior()
            prior.validate()
            assert prior.model_key == key
            assert len(prior.size_lo) == model.n_size
    finally:
        win.close()


def test_detect_separates_touching_cells_and_lists_them(app, monkeypatch):
    win, _state, _owner = _window(app, monkeypatch, _two_touching_cells())
    try:
        win._detect()
        assert win._result is not None
        assert len(win._result.instances) == 2
        assert win._list.count() == 2
        assert len(win._contours) == 2
        assert win._add_btn.isEnabled()
        for contour in win._contours:
            assert contour.ndim == 2 and contour.shape[1] == 2
    finally:
        win.close()


def test_add_to_manager_files_editable_closed_polygons(app, monkeypatch):
    win, state, owner = _window(app, monkeypatch, _two_touching_cells())
    try:
        win._detect()
        win._add_to_manager()
        assert owner.calls and owner.calls[0]["n"] == 2
        assert owner.calls[0]["source"] == "shape_segmentation_2d"
        assert len(state.rois.records) == 2
        assert state.rois.show_all is True
        for record in state.rois.records:
            assert record.type == "polygon"
            assert record.geometry["closed"] is True
            pts = np.asarray(record.geometry["points"], dtype=float)
            # A closed ring is stored once, not with a duplicated first vertex.
            assert pts.shape[0] >= 3
            assert not np.allclose(pts[0], pts[-1])
            assert record.context["dataset_idx"] == 0
        assert any("Shape-model segmentation" in m for m in state.messages)
    finally:
        win.close()


def test_the_filed_polygons_are_regions_and_vertex_editable(app, monkeypatch):
    """The two properties that make manual correction work at all."""
    from minflux_viewer.core.roi import record_to_points
    from minflux_viewer.core.roi_selection import roi_region_mask
    from minflux_viewer.ui.roi_overlay import _VERTEX_EDIT_TYPES

    win, state, _owner = _window(app, monkeypatch, _two_touching_cells())
    try:
        win._detect()
        win._add_to_manager()
        record = state.rois.records[0]
        assert record.type in _VERTEX_EDIT_TYPES        # draggable vertices
        pts = record_to_points(record)
        assert pts.shape[0] >= 3
        # And it encloses area, so masks / crop / highlighting work.
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        inside = roi_region_mask(np.array([cx]), np.array([cy]), record)
        assert bool(inside[0])
    finally:
        win.close()


def test_only_the_selected_objects_are_filed(app, monkeypatch):
    win, state, owner = _window(app, monkeypatch, _two_touching_cells())
    try:
        win._detect()
        win._list.setCurrentRow(0)
        assert win._selected_indices().tolist() == [0]
        win._add_to_manager()
        assert owner.calls[0]["n"] == 1
        assert len(state.rois.records) == 1
    finally:
        win.close()


def test_contour_vertex_count_is_honoured(app, monkeypatch):
    win, _state, _owner = _window(app, monkeypatch, _two_touching_cells())
    try:
        win._vertices.setValue(16)
        win._detect()
        coarse = [c.shape[0] for c in win._contours]
        win._vertices.setValue(96)
        win._detect()
        fine = [c.shape[0] for c in win._contours]
        assert min(fine) > max(coarse)
    finally:
        win.close()


def test_acquisition_roi_control_is_disabled_without_a_source_msr(
        app, monkeypatch):
    win, _state, _owner = _window(app, monkeypatch, _two_touching_cells())
    try:
        assert win._acquisition_bounds() is None
        assert not win._use_roi.isEnabled()
        win._detect()
        assert not any(item.clipped for item in win._result.instances)
    finally:
        win.close()


def test_preferences_round_trip_per_model(app, monkeypatch):
    xy = _two_touching_cells()
    win, state, _owner = _window(app, monkeypatch, xy)
    try:
        win._model_combo.setCurrentIndex(win._model_combo.findData("capsule"))
        win._range_spins["length_nm"][1].setValue(3300.0)
        win._pixel_spin.setValue(25.0)
        win._vertices.setValue(24)
        win._save_prefs()
    finally:
        win.close()
    assert state.prefs["shape_segmentation"]["model"] == "capsule"

    monkeypatch.setattr(
        "minflux_viewer.core.roi_crop.display_xy_filtered", lambda ds: xy)
    again = ShapeSegmentationWindow(state, 0, owner=None)
    try:
        assert again._model_key() == "capsule"
        assert again._range_spins["length_nm"][1].value() == pytest.approx(3300.0)
        assert again._pixel_spin.value() == pytest.approx(25.0)
        assert again._vertices.value() == 24
    finally:
        again.close()


def test_a_swapped_size_range_is_normalised_rather_than_rejected(app, monkeypatch):
    """The spin boxes clamp to each SizeSpec's limits, so the only malformed
    input the UI can produce is min/max the wrong way round — take it as given."""
    win, _state, _owner = _window(app, monkeypatch, _two_touching_cells())
    try:
        win._range_spins["length_nm"][0].setValue(3000.0)
        win._range_spins["length_nm"][1].setValue(1500.0)
        prior = win._prior()
        prior.validate()
        assert prior.size_lo[0] == pytest.approx(1500.0)
        assert prior.size_hi[0] == pytest.approx(3000.0)
        win._detect()
        assert win._result is not None
    finally:
        win.close()


def test_a_failing_fit_is_reported_in_the_status_not_raised(app, monkeypatch):
    win, _state, _owner = _window(app, monkeypatch, _two_touching_cells())
    try:
        def _boom(*args, **kwargs):
            raise RuntimeError("synthetic failure")
        monkeypatch.setattr(
            "minflux_viewer.ui.shape_segmentation_dialog.ss.segment_shapes_in_points",
            _boom)
        win._detect()                      # must not propagate
        assert "synthetic failure" in win._status_label.text()
        assert win._detect_btn.isEnabled()
    finally:
        win.close()


def test_command_metadata_points_at_the_dialog():
    from minflux_viewer.ui.command_meta import COMMAND_META
    meta = COMMAND_META["actionSegShapeModel"]
    assert meta.source.endswith("shape_segmentation_dialog.py")
    assert "capsule" in meta.keywords
    assert "ecoli" in meta.keywords


# --------------------------------------------------------------------------- #
# End-to-end through the real MainWindow / RoiStore
# --------------------------------------------------------------------------- #
def test_menu_entry_and_roi_handoff_through_the_real_main_window(app, monkeypatch):
    """Analyze › Segmentation › Shape Model… reaches the real ROI store."""
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.main_window import MainWindow

    xy = _two_touching_cells()
    win = MainWindow(AppState())
    try:
        labels = [a.text() for a in win.menuAnalyzeSegmentation.actions()]
        assert any("Shape Model" in text for text in labels)

        dataset = _dataset_from_xy(xy, "cells")
        win._state.add_dataset(dataset)
        win._state.set_active(0)

        opened = {}
        monkeypatch.setattr(MainWindow, "_show_render",
                            lambda self, i: opened.setdefault("render", i))
        monkeypatch.setattr(MainWindow, "_show_roi_manager",
                            lambda self: opened.setdefault("manager", True))
        monkeypatch.setattr(
            "minflux_viewer.core.roi_crop.display_xy_filtered", lambda ds: xy)

        win.actionSegShapeModel.trigger()
        from minflux_viewer.ui.shape_segmentation_dialog import ShapeSegmentationWindow
        dialogs = [w for w in win._modeless_windows
                   if isinstance(w, ShapeSegmentationWindow)]
        assert len(dialogs) == 1
        dialog = dialogs[0]

        dialog._detect()
        assert len(dialog._result.instances) == 2
        dialog._add_to_manager()

        records = list(win._state.rois.records)
        assert len(records) == 2
        assert all(r.type == "polygon" for r in records)
        assert all(r.geometry["closed"] is True for r in records)
        assert opened.get("render") == 0 and opened.get("manager") is True
    finally:
        win.close()


def _dataset_from_xy(xy, name):
    from minflux_viewer.core.dataset import build_localization_dataset
    return build_localization_dataset(
        name=name, x_nm=xy[:, 0], y_nm=xy[:, 1],
        z_nm=np.zeros(xy.shape[0]), source_version="simulation")


def test_a_filed_contour_can_be_corrected_by_hand_and_committed(app, monkeypatch):
    """The manual-correction requirement, end to end through the real overlay.

    A fitted contour is filed as a polygon; the displayed item is a live-edit
    copy, so moving a vertex leaves the record alone until the ROI Manager's
    Update reads the edited geometry back (ImageJ RoiManager semantics).
    """
    import pyqtgraph as pg
    from minflux_viewer.core.roi import RoiStore
    from minflux_viewer.ui.roi_overlay import RoiOverlayController

    win, state, _owner = _window(app, monkeypatch, _two_touching_cells())
    try:
        win._detect()
        win._add_to_manager()
        record = state.rois.records[0]
    finally:
        win.close()

    store = RoiStore()
    store.add(record)
    plot = pg.PlotWidget()
    # Held for the process lifetime: tearing a ViewBox down mid-run makes
    # pyqtgraph print destructor tracebacks that drown the test output.
    _KEEP_ALIVE.append(plot)
    if True:
        controller = RoiOverlayController(
            store, None, plot, plot.getPlotItem())
        item = controller._make_item(record)
        assert item is not None
        handles = item.getHandles()
        assert len(handles) >= 3            # one draggable handle per vertex

        controller.items[record.id] = item
        before = np.asarray(record.geometry["points"], dtype=float)
        item.movePoint(handles[0], pg.Point(float(before[0][0]) + 400.0,
                                            float(before[0][1]) + 250.0),
                       finish=True)

        # The stored record is untouched until Update...
        assert np.allclose(np.asarray(store.records[0].geometry["points"],
                                      dtype=float), before)
        # ...which reads the edited geometry back off the item.
        updated = controller.record_for_update(store.records[0])
        assert updated is not None
        after = np.asarray(updated.geometry["points"], dtype=float)
        assert after.shape == before.shape
        assert not np.allclose(after, before)
        assert updated.type == "polygon"


# --------------------------------------------------------------------------- #
# Display orientation, layers and the point cloud
# --------------------------------------------------------------------------- #
def _painted_extent_nm(image_item):
    """Bounding box (nm) of the bright mass the ImageItem actually paints.

    Reads the array the way the ImageItem will, then maps through the item's own
    transform. The *extent* is what matters: ``setRect`` stretches whatever array
    it is given onto the same nm rectangle, so a transposed image keeps its
    centroid and betrays itself only by painting the object along the wrong axis.
    """
    import pyqtgraph as pg
    arr = np.asarray(image_item.image, dtype=float)
    mapped = image_item.mapRectToParent(image_item.boundingRect())
    rows, cols = np.nonzero(arr > arr.max() * 0.5)
    if rows.size == 0:
        return None
    row_major = pg.getConfigOption("imageAxisOrder") == "row-major"
    y_idx, x_idx = (rows, cols) if row_major else (cols, rows)
    n_y, n_x = ((arr.shape[0], arr.shape[1]) if row_major
                else (arr.shape[1], arr.shape[0]))
    to_x = lambda i: mapped.x() + (i + 0.5) / n_x * mapped.width()   # noqa: E731
    to_y = lambda j: mapped.y() + (j + 0.5) / n_y * mapped.height()  # noqa: E731
    return (to_x(x_idx.min()), to_x(x_idx.max()),
            to_y(y_idx.min()), to_y(y_idx.max()))


#: A lone horizontal rod far from the x==y diagonal, so a transposed image is
#: painted along the wrong axis by a wide margin rather than a tolerance.
_CELL_CENTRE = (3200.0, 800.0)


def _one_offset_cell(seed=3):
    rng = np.random.default_rng(seed)
    mask = instance_mask(
        ShapeInstance("capsule", _CELL_CENTRE, 0.0, (2600.0, 800.0),
                      1.0, 1.0, False, 1.0, 1), (120, 300), PIXEL)
    rows, cols = np.nonzero(mask)
    pick = rng.choice(rows.size, 9000, replace=True)
    return np.column_stack([(cols[pick] + rng.random(pick.size)) * PIXEL,
                            (rows[pick] + rng.random(pick.size)) * PIXEL])


@pytest.mark.parametrize("axis_order", ["row-major", "col-major"])
def test_the_density_image_lands_on_the_data_under_either_axis_order(
        app, monkeypatch, axis_order):
    """The reported flip: pyqtgraph's image axis order is a *runtime global*
    that opening a render window switches, so the image orientation has to be
    resolved per draw rather than assumed."""
    import pyqtgraph as pg
    previous = pg.getConfigOption("imageAxisOrder")
    pg.setConfigOption("imageAxisOrder", axis_order)
    try:
        win, _state, _owner = _window(app, monkeypatch, _one_offset_cell())
        try:
            win._detect()
            assert len(win._result.instances) == 1
            centre = win._result.instances[0].center_nm
            length, width = win._result.instances[0].size_nm

            # Structural: the array handed to the ImageItem must be indexed the
            # way the current axis order says.
            ny, nx = win._result.detection_field.shape
            expected = (ny, nx) if axis_order == "row-major" else (nx, ny)
            assert np.asarray(win._image.image).shape == expected

            # Behavioural: the rod is horizontal, so the painted mass must be
            # long in x and short in y, at the fitted position.
            painted = _painted_extent_nm(win._image)
            assert painted is not None
            x0, x1, y0, y1 = painted
            assert abs(0.5 * (x0 + x1) - centre[0]) < 250.0, painted
            assert abs(0.5 * (y0 + y1) - centre[1]) < 250.0, painted
            assert abs((x1 - x0) - length) < 0.35 * length, painted
            assert abs((y1 - y0) - width) < 0.5 * width, painted
            assert (x1 - x0) > 2.0 * (y1 - y0), painted
        finally:
            win.close()
    finally:
        pg.setConfigOption("imageAxisOrder", previous)


def test_contours_and_localizations_agree_with_the_fitted_geometry(
        app, monkeypatch):
    xy = _one_offset_cell()
    win, _state, _owner = _window(app, monkeypatch, xy)
    try:
        win._detect()
        centre = win._result.instances[0].center_nm
        contour = win._contours[0]
        assert abs(contour[:, 0].mean() - centre[0]) < 250.0
        assert abs(contour[:, 1].mean() - centre[1]) < 250.0
        shown_x, shown_y = win._scatter.getData()
        assert abs(float(np.mean(shown_x)) - centre[0]) < 250.0
        assert abs(float(np.mean(shown_y)) - centre[1]) < 250.0
    finally:
        win.close()


def test_y_axis_direction_follows_the_render_view_preference(app, monkeypatch):
    xy = _one_offset_cell()
    monkeypatch.setattr(
        "minflux_viewer.core.roi_crop.display_xy_filtered", lambda ds: xy)
    for origin, inverted in (("top_left", True), ("bottom_left", False)):
        state = _State()
        state.prefs = {"plot": {"render_xy_origin": origin}}
        win = ShapeSegmentationWindow(state, 0, owner=None)
        try:
            assert win._y_inverted() is inverted
            assert win._plot.getPlotItem().getViewBox().yInverted() is inverted
        finally:
            win.close()


def test_localizations_are_shown_before_any_fit(app, monkeypatch):
    xy = _one_offset_cell()
    win, _state, _owner = _window(app, monkeypatch, xy)
    try:
        assert win._result is None
        shown_x, _ = win._scatter.getData()
        assert shown_x is not None and len(shown_x) > 0
        assert "localizations" in win._loc_count_label.text()
    finally:
        win.close()


def test_the_point_cloud_is_thinned_for_display_only(app, monkeypatch):
    from minflux_viewer.ui import shape_segmentation_dialog as mod

    xy = _one_offset_cell()
    monkeypatch.setattr(mod, "_MAX_SCATTER_POINTS", 500)
    win, _state, _owner = _window(app, monkeypatch, xy)
    try:
        shown_x, _ = win._scatter.getData()
        assert 0 < len(shown_x) <= 500 < xy.shape[0]
        assert "of" in win._loc_count_label.text()
        # Thinning must not reach the fit: it still sees every localization.
        win._detect()
        assert win._result.stats["n_points"] == xy.shape[0]
    finally:
        win.close()


def test_layer_checkboxes_toggle_each_layer(app, monkeypatch):
    win, _state, _owner = _window(app, monkeypatch, _one_offset_cell())
    try:
        win._detect()
        assert win._curves
        for box, probe in (
                (win._show_density, lambda: win._image.isVisible()),
                (win._show_locs, lambda: win._scatter.isVisible()),
                (win._show_contours, lambda: win._curves[0].isVisible())):
            box.setChecked(True)
            assert probe() is True
            box.setChecked(False)
            assert probe() is False
            box.setChecked(True)
            assert probe() is True
    finally:
        win.close()


def test_layer_choices_are_remembered(app, monkeypatch):
    xy = _one_offset_cell()
    win, state, _owner = _window(app, monkeypatch, xy)
    try:
        win._show_density.setChecked(False)
        win._show_locs.setChecked(True)
        win._show_contours.setChecked(False)
        win._save_prefs()
    finally:
        win.close()
    monkeypatch.setattr(
        "minflux_viewer.core.roi_crop.display_xy_filtered", lambda ds: xy)
    again = ShapeSegmentationWindow(state, 0, owner=None)
    try:
        assert again._show_density.isChecked() is False
        assert again._show_locs.isChecked() is True
        assert again._show_contours.isChecked() is False
    finally:
        again.close()
