"""CPU bulk/aggregation attribute renderer and startup GPU-budget regressions."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.ui.attribute_cpu import (
    BulkScatterItem,
    aggregate_screen_points,
    spatial_representative_indices,
)
from minflux_viewer.ui.gpu_capabilities import (
    CPU_WORK_BYTES_PER_POINT,
    GPU_BYTES_PER_POINT,
    GpuCapabilities,
    point_limit_from_memory,
)


@pytest.fixture
def _qt_app():
    if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
        pytest.skip("No display available for Qt tests")
    return QApplication.instance() or QApplication(sys.argv)


def test_screen_aggregation_consumes_all_visible_rows_and_ignores_nan_pairs():
    x = np.array([0.1, 0.2, 0.8, np.nan, 2.0])
    y = np.array([0.1, 0.2, 0.8, 0.5, 2.0])
    result = aggregate_screen_points(
        x, y, bounds=(0.0, 1.0, 0.0, 1.0), width=2, height=2
    )
    assert result.input_count == 5
    assert result.drawable_count == 4
    assert result.visible_count == 3
    assert int(result.counts.sum()) == 3
    assert result.occupied_count == 2


def test_screen_aggregation_reports_mean_c_per_cell():
    result = aggregate_screen_points(
        [0.1, 0.2, 0.8, 0.8],
        [0.1, 0.2, 0.8, 0.8],
        values=[2.0, 4.0, 10.0, np.nan],
        bounds=(0.0, 1.0, 0.0, 1.0),
        width=2,
        height=2,
    )
    means = result.mean_values()
    assert result.visible_count == 3
    assert means is not None
    assert means[0, 0] == pytest.approx(3.0)
    assert means[1, 1] == pytest.approx(10.0)


def test_screen_aggregation_cannot_invert_valid_only_density():
    rng = np.random.default_rng(7)
    x = rng.uniform(0.0, 30_000.0, 100_000)
    y = rng.normal(5_000.0, 600.0, 100_000)
    valid = rng.random(100_000) < 0.35
    all_rows = aggregate_screen_points(
        x, y, bounds=(0.0, 30_000.0, 2_000.0, 8_000.0), width=300, height=200
    )
    valid_rows = aggregate_screen_points(
        x[valid], y[valid],
        bounds=(0.0, 30_000.0, 2_000.0, 8_000.0),
        width=300,
        height=200,
    )
    assert valid_rows.visible_count < all_rows.visible_count
    assert np.all(valid_rows.counts <= all_rows.counts)


def test_spatial_representative_sampling_keeps_an_isolated_feature():
    x = np.zeros(1_000)
    y = np.zeros(1_000)
    x[5], y[5] = 100.0, 100.0
    selected = spatial_representative_indices(x, y, 20)
    assert selected.size <= 20
    assert 5 in selected
    assert np.any(selected != 5)


def test_gpu_point_limit_is_derived_from_both_memory_sides():
    gib = 1024**3
    limit = point_limit_from_memory(
        available_system_memory_bytes=8 * gib,
        free_gpu_memory_bytes=2 * gib,
    )
    assert limit == min(
        (8 * gib // 8) // CPU_WORK_BYTES_PER_POINT,
        (2 * gib // 2) // GPU_BYTES_PER_POINT,
    )
    doubled = point_limit_from_memory(
        available_system_memory_bytes=16 * gib,
        free_gpu_memory_bytes=4 * gib,
    )
    assert doubled == 2 * limit


def _state(n: int) -> AppState:
    state = AppState()
    state.add_dataset(
        build_localization_dataset(
            name="cpu-attribute",
            x_nm=np.linspace(0.0, 1_000.0, n),
            y_nm=np.sin(np.linspace(0.0, 80.0, n)) * 100.0,
            z_nm=np.zeros(n),
        )
    )
    return state


def test_cpu_fix_window_uses_bulk_item_for_sparse_data(_qt_app):
    from minflux_viewer.ui.attribute_window import AttributeWindow

    window = AttributeWindow(_state(5_000), dataset_idx=0, cpu_fix=True)
    try:
        window.show()
        _qt_app.processEvents()
        assert window.windowTitle().startswith("Attribute Plot (CPU fix)")
        assert not window.gpu_2d
        assert isinstance(window._series_items[0][1], BulkScatterItem)
        assert "CPU bulk painting" in window._info.text()
        assert window._view_state_key == "attribute_plot_cpu_state"
    finally:
        window.close()
        _qt_app.processEvents()


def test_cpu_fix_window_switches_to_screen_aggregation_when_overplotted(
    _qt_app, monkeypatch,
):
    from pyqtgraph import ImageItem

    from minflux_viewer.ui.attribute_window import AttributeWindow

    window = AttributeWindow(_state(50_000), dataset_idx=0, cpu_fix=True)
    try:
        window.resize(180, 140)
        window.show()
        _qt_app.processEvents()
        original_geometry = window._cpu_view_geometry
        monkeypatch.setattr(
            window,
            "_cpu_view_geometry",
            lambda records, dimensions: (
                original_geometry(records, dimensions)[0], 100, 100, False
            ),
        )
        window._draw()
        _qt_app.processEvents()
        assert isinstance(window._series_items[0][1], ImageItem)
        assert window._cpu_aggregate_active
        assert "CPU screen aggregation (count)" in window._info.text()
    finally:
        window.close()
        _qt_app.processEvents()


def test_cpu_curve_lod_connects_skipped_vertices_but_preserves_nan_gaps(_qt_app):
    from minflux_viewer.ui.attribute_window import AttributeWindow

    window = AttributeWindow(_state(100), dataset_idx=0, cpu_fix=True)
    try:
        x = np.arange(1_000, dtype=float)
        y = np.sin(x / 30.0)
        line_x, _line_y = window._cpu_line_values(
            x, y, bounds=(0.0, 999.0, -2.0, 2.0), budget=50
        )
        assert line_x.size <= 50
        assert np.isfinite(line_x).all()

        y[500] = np.nan
        line_x, _line_y = window._cpu_line_values(
            x, y, bounds=(0.0, 999.0, -2.0, 2.0), budget=50
        )
        assert np.isnan(line_x).any()
    finally:
        window.close()


def test_view_menu_has_separate_cpu_attribute_plot(_qt_app):
    from minflux_viewer.ui.main_window import MainWindow

    state = _state(100)
    state.prefs["data"].update({
        "show_render": False,
        "show_data_info": False,
        "compute_loc_prec": False,
        "compute_local_density": False,
    })
    window = MainWindow(state)
    try:
        assert window.actionAttributeCpu.text() == "Attribute Plot (CPU fix)"
        assert window.actionAttributeCpu in window._ui.menuView.actions()
        cpu_window = window._show_attr_plot_cpu(0)
        assert cpu_window is window._attr_cpu_windows[0]
        assert cpu_window is not window._attr_windows.get(0)
        assert not cpu_window.gpu_2d
    finally:
        window.close()
        _qt_app.processEvents()


def test_startup_gpu_result_disables_gpu_action_but_not_cpu_fix(_qt_app):
    from minflux_viewer.ui.main_window import MainWindow

    state = _state(100)
    state.gpu_capabilities = GpuCapabilities(
        available=False, reason="test OpenGL context unavailable"
    )
    window = MainWindow(state)
    try:
        assert not window.actionAttributeGpu.isEnabled()
        assert "test OpenGL context unavailable" in window.actionAttributeGpu.toolTip()
        assert window.actionAttributeCpu.isEnabled()
        cpu_window = window._show_attr_plot_cpu(0)
        assert not cpu_window.gpu_2d
    finally:
        window.close()
        _qt_app.processEvents()
