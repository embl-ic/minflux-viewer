"""Coordinate-service integration with registry, scripts, preferences and views."""

from types import MethodType

import numpy as np
import pytest

from minflux_viewer.core.app_state import AppState, default_prefs
from minflux_viewer.core.coordinates import coordinate_view, display_coordinate_view
from minflux_viewer.core.dataset import build_localization_dataset

# Capture the production method before the session's preference-isolation fixture
# substitutes it. Tests invoking this method replace QSettings with a fake below.
_save_prefs = AppState.save_prefs


def dataset(name="coordinates"):
    rng = np.random.default_rng(17)
    xyz = rng.normal(size=(400, 3)) * 200
    return build_localization_dataset(name=name, x_nm=xyz[:, 0], y_nm=xyz[:, 1], z_nm=xyz[:, 2])


@pytest.fixture
def state(qapp):
    state = AppState()
    state.prefs = default_prefs()
    state.prefs["data"].update(show_render=False, show_data_info=False)
    state.coordinates.set_budget_bytes(1_000_000)
    return state


def test_close_earlier_dataset_keeps_survivor_cache_and_reopen_gets_new_key(state):
    first, second = dataset("first"), dataset("second")
    state.add_dataset(first)
    state.add_dataset(second)
    a, b = coordinate_view(first), coordinate_view(second)
    key = state.key_of(first)
    state.remove_dataset(0)
    assert coordinate_view(second) is b
    assert state.coordinates.retained_bytes == b.nbytes
    state.add_dataset(first)
    assert state.key_of(first) != key
    assert coordinate_view(first) is not a
    state.datasets.clear()  # legacy direct list edits are reconciled too
    assert not state.entries
    assert state.coordinates.retained_bytes == 0


@pytest.mark.parametrize("transformed", [False, True])
@pytest.mark.parametrize("filtered", [False, True])
def test_script_coordinates_remain_independent_writable_copies(state, transformed, filtered):
    ds = dataset()
    ds.state["overlay_transform"] = {"matrix_3x3": [[1, 0, 20], [0, 1, 0], [0, 0, 1]]}
    state.add_dataset(ds)
    expected = display_coordinate_view(ds) if transformed else coordinate_view(ds)
    public = state.mfv.data.loc(dataset=0, transformed=transformed, filtered=filtered)
    np.testing.assert_array_equal(public, expected)
    public[:] = 999
    assert not np.any(expected == 999)


def test_preferences_apply_live_budget_and_cancel_preserves_it(state, monkeypatch):
    from minflux_viewer.core import app_state
    from minflux_viewer.ui.preferences_dialog import PreferencesDialog

    class Settings:
        def __init__(self, *args):
            pass

        def setValue(self, *args):
            pass

    monkeypatch.setattr(app_state, "QSettings", Settings)
    state.save_prefs = MethodType(_save_prefs, state)
    ds = dataset()
    state.add_dataset(ds)
    coordinate_view(ds)
    dialog = PreferencesDialog(state)
    try:
        assert dialog._coordinate_cache_percent.value() == 25
        dialog._coordinate_cache_percent.setValue(0)
        dialog._accept()
        assert state.prefs["data"]["coordinate_cache_percent"] == 0
        assert state.coordinates.budget_bytes == 0
        assert state.coordinates.retained_bytes == 0
    finally:
        dialog.deleteLater()
    dialog = PreferencesDialog(state)
    try:
        dialog._coordinate_cache_percent.setValue(50)
        dialog.reject()
        assert state.coordinates.budget_bytes == 0
    finally:
        dialog.deleteLater()


def test_render_scatter_roi_share_display_views_through_pan_zoom_and_edits(state, qapp, monkeypatch):
    from PyQt6.QtCore import QCoreApplication, QEvent

    from minflux_viewer.core.roi_crop import display_coords
    from minflux_viewer.ui.render_window import RenderWindow
    from minflux_viewer.ui.scatter_window import ScatterWindow

    ds = dataset()
    ds.state["overlay_transform"] = {"matrix_3x3": [[1, 0, 20], [0, 1, 0], [0, 0, 1]]}
    state.add_dataset(ds)
    render = RenderWindow(state, dataset_idx=0)
    scatter = ScatterWindow(state, dataset_idx=0)
    try:
        render.show()
        scatter.show()
        qapp.processEvents()
        shared = display_coordinate_view(ds)
        assert scatter._current_locs(ds) is shared
        assert np.shares_memory(render._raw_render_locs(ds), shared)
        assert np.shares_memory(display_coords(ds), shared)
        np.testing.assert_array_equal(render._dataset_locs(ds), shared)
        retained = state.coordinates.retained_bytes
        for step in range(3):
            render._view_box.setRange(xRange=(-100 + step, 100 + step), yRange=(-100, 100))
            scatter._redraw_current(save_state=False)
            qapp.processEvents()
            assert scatter._current_locs(ds) is shared
            assert state.coordinates.retained_bytes == retained
        ds.filter_mask = np.arange(400) % 2 == 0
        state.notify_filter_changed(0)
        assert scatter._current_locs(ds) is shared
        np.testing.assert_array_equal(render._dataset_locs(ds), shared[ds.filter_mask])
        ds.attr.replace_coordinates(loc_x=np.zeros(400))
        state.notify_attributes_changed(0)
        changed = display_coordinate_view(ds)
        assert changed is not shared
        assert scatter._current_locs(ds) is changed
        np.testing.assert_array_equal(render._dataset_locs(ds), changed[ds.filter_mask])
        redraws = []
        monkeypatch.setattr(render, "_on_filter_changed", lambda event: redraws.append("render"))
        monkeypatch.setattr(scatter, "_redraw_current", lambda **kwargs: redraws.append("scatter"))
        ds.state["overlay_transform"]["matrix_3x3"][0][2] += 1e-8
        state.notify_overlay_transform_changed(0)
        assert set(redraws) == {"render", "scatter"}
        assert scatter._current_locs(ds) is not changed
    finally:
        for window in (render, scatter):
            window.close()
            window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
