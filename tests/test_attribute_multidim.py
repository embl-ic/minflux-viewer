"""Context-driven 2-D/3-D/4-D Attribute Plot regressions."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QContextMenuEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMenu, QWidget

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.core.iteration import FLATTEN_LABEL, STACKED_LABEL


@pytest.fixture
def _qt_app():
    if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
        pytest.skip("No display available for Qt tests")
    return QApplication.instance() or QApplication(sys.argv)


def _state() -> AppState:
    state = AppState()
    dataset = build_localization_dataset(
        name="attribute-dimensions",
        x_nm=np.arange(8, dtype=float),
        y_nm=np.arange(8, dtype=float) * 2.0,
        z_nm=np.arange(8, dtype=float) * 0.5,
        tid=np.repeat([1, 2], 4),
        tim=np.linspace(0.0, 0.7, 8),
        attrs={
            "efo": np.linspace(1_000.0, 8_000.0, 8),
            "cfr": np.linspace(0.2, 0.9, 8),
        },
    )
    # Expose the pooled iteration choices without needing a raw acquisition in
    # these UI-only tests.
    dataset.prop.num_itr = 3
    dataset.metadata["raw_num_itr"] = 3
    state.add_dataset(dataset)
    return state


def _capture_context_menu(monkeypatch, window) -> QMenu:
    captured: list[QMenu] = []
    monkeypatch.setattr(QMenu, "exec", lambda menu, *_args: captured.append(menu))
    window._show_context_menu(QPoint(0, 0))
    assert len(captured) == 1
    return captured[0]


def _texts(menu: QMenu) -> list[str]:
    return [action.text() for action in menu.actions()]


def _submenu(menu: QMenu, prefix: str) -> QMenu:
    action = next(action for action in menu.actions() if action.text().startswith(prefix))
    submenu = action.menu()
    assert submenu is not None
    return submenu


def test_dimensions_are_added_silently_and_projection_remaps_top_row(
    monkeypatch, _qt_app,
):
    from minflux_viewer.ui.attribute_window import AttributeWindow

    window = AttributeWindow(_state(), dataset_idx=0)
    try:
        assert window._dimension_count == 2
        assert window._view_mode == "XY"
        assert (window._axis_a_label.text(), window._axis_b_label.text()) == ("X:", "Y:")

        window._add_attribute_dimension()
        assert window._dimension_count == 3
        assert window._view_mode == "XY"
        assert (window._axis_a_label.text(), window._axis_b_label.text()) == ("X:", "Y:")

        menu = _capture_context_menu(monkeypatch, window)
        assert _texts(menu) == [
            "View",
            "",
            f"Z: {window._dimension_attrs['Z']}",
            "add new attribute as plot dimension",
            "reduce attribute dimension",
            "",
            "Reset View",
        ]
        assert _texts(_submenu(menu, "View")) == [
            "XY",
            "XZ",
            "YZ",
            "3D",
            "",
            "Black background",
            "Axis",
            "Grid lines",
            "Plot style",
        ]
        z_menu = _submenu(menu, "Z:")
        assert _texts(z_menu)[0] == "idx"

        window._set_view_mode("XZ")
        assert (window._axis_a_label.text(), window._axis_b_label.text()) == ("X:", "Z:")
        assert window._x_combo.currentText() == window._dimension_attrs["X"]
        assert window._y_combo.currentText() == window._dimension_attrs["Z"]

        menu = _capture_context_menu(monkeypatch, window)
        assert any(text.startswith("Y:") for text in _texts(menu))
    finally:
        window.close()


def test_default_context_menu_has_view_controls_and_reset(monkeypatch, _qt_app):
    from minflux_viewer.ui.attribute_window import AttributeWindow

    window = AttributeWindow(_state(), dataset_idx=0)
    try:
        menu = _capture_context_menu(monkeypatch, window)
        assert _texts(menu) == [
            "View",
            "",
            "add new attribute as plot dimension",
            "reduce attribute dimension",
            "",
            "Reset View",
        ]
        assert not menu.actions()[3].isEnabled()
        assert _texts(_submenu(menu, "View")) == [
            "XY",
            "",
            "Black background",
            "Axis",
            "Grid lines",
            "Plot style",
        ]
    finally:
        window.close()


def test_fourth_dimension_adds_linear_color_menu_and_disables_stacked(
    monkeypatch, _qt_app,
):
    from minflux_viewer.ui.attribute_window import _LINEAR_COLORMAPS, AttributeWindow

    window = AttributeWindow(_state(), dataset_idx=0)
    try:
        window._add_attribute_dimension()
        window._iter_combo.setCurrentText(STACKED_LABEL)
        assert window._iter_combo.currentText() == STACKED_LABEL

        window._add_attribute_dimension()
        assert window._dimension_count == 4
        assert window._view_mode == "XY"
        assert window._iter_combo.currentText() == FLATTEN_LABEL
        stacked_item = window._iter_combo.model().item(
            window._iter_combo.findText(STACKED_LABEL)
        )
        assert stacked_item is not None and not stacked_item.isEnabled()

        menu = _capture_context_menu(monkeypatch, window)
        assert _texts(menu) == [
            "View",
            "",
            f"Z: {window._dimension_attrs['Z']}",
            f"C: {window._dimension_attrs['C']}",
            "C: mapping",
            "add new attribute as plot dimension",
            "reduce attribute dimension",
            "",
            "Reset View",
        ]
        assert _texts(_submenu(menu, "C: mapping"))[:len(_LINEAR_COLORMAPS)] == (
            list(_LINEAR_COLORMAPS)
        )
        mapping_actions = _submenu(menu, "C: mapping").actions()
        assert [action.text() for action in mapping_actions][-2:] == ["", "Colorbar"]
        assert mapping_actions[-1].isCheckable()
        assert mapping_actions[-1].isChecked()
        assert not menu.actions()[-4].isEnabled()

        window._set_c_mapping("cividis")
        assert window._c_mapping == "cividis"
        assert "C=" in window._info.text() and "[cividis]" in window._info.text()

        window._reduce_attribute_dimension()
        assert window._dimension_count == 3
        assert stacked_item.isEnabled()
    finally:
        window.close()


def test_lines_keep_styled_markers_visible(_qt_app):
    from minflux_viewer.ui.attribute_window import AttributeWindow

    window = AttributeWindow(_state(), dataset_idx=0)
    try:
        window._lines_chk.setChecked(True)
        window._apply_plot_style(
            {
                "symbol": "d",
                "size": 9,
                "alpha": 140,
                "color": (12, 34, 56),
            },
            color_changed=True,
        )
        curve, scatter = window._series_items[0]
        assert len(curve.xData) == 8
        assert len(scatter.data) == 8
        assert scatter.opts["size"] == 9
        assert scatter.opts["symbol"] == "d"
        assert window._point_color == (12, 34, 56, 140)
    finally:
        window.close()


def test_custom_c_mapping_is_offered_and_drives_shared_lut(_qt_app):
    from minflux_viewer.colormaps import (
        delete_custom_colormap,
        store_custom_colormap,
    )
    from minflux_viewer.ui.attribute_window import AttributeWindow

    state = _state()
    custom_name = "Attribute test linear"
    store_custom_colormap(
        state.prefs,
        custom_name,
        [[0.0, [0, 10, 20, 255]], [1.0, [230, 240, 250, 255]]],
    )
    window = AttributeWindow(state, dataset_idx=0)
    try:
        window._add_attribute_dimension()
        window._add_attribute_dimension()
        assert custom_name in window._c_mapping_names()
        window._set_c_mapping(custom_name)
        window.open_lut_dialog()
        assert window._lut_dialog is not None
        assert window._lut_dialog.isVisible()
        assert window._lut_dialog._cmap_combo.currentText() == custom_name

        window._on_lut_levels_changed(0.3, 0.7)
        window._on_lut_invert_changed(True)
        assert window._manual_color_levels == (0.3, 0.7)
        assert window._lut_invert is True
    finally:
        if window._lut_dialog is not None:
            window._lut_dialog.close()
        window.close()
        delete_custom_colormap(state.prefs, custom_name)


def test_floating_colorbar_is_linked_editable_and_persistent(
    monkeypatch, _qt_app,
):
    from minflux_viewer.ui.attribute_window import AttributeWindow

    state = _state()
    window = AttributeWindow(state, dataset_idx=0)
    try:
        window._add_attribute_dimension()
        window._add_attribute_dimension()
        colorbar = window._colorbar
        assert not colorbar.isHidden()
        assert colorbar.orientation == "vertical"
        assert colorbar.show_values is True
        window.show()
        _qt_app.processEvents()
        assert not window.grab().isNull()  # exercises the painted gradient/ticks
        assert window.width() == 700 + colorbar.width()
        assert window._stack.contentsMargins().right() == colorbar.width()
        assert colorbar.x() + colorbar.width() == window._stack.width()
        assert colorbar.height() == window._stack.height()

        captured: list[QMenu] = []
        monkeypatch.setattr(QMenu, "exec", lambda menu, *_args: captured.append(menu))
        event = QContextMenuEvent(
            QContextMenuEvent.Reason.Mouse,
            QPoint(4, 4),
            colorbar.mapToGlobal(QPoint(4, 4)),
        )
        colorbar.contextMenuEvent(event)
        assert _texts(captured[0]) == [
            "Hide colorbar",
            "Attribute:",
            "Show values",
            "Placement",
            "",
            "Customize",
        ]
        attribute_menu = _submenu(captured[0], "Attribute:")
        assert _texts(attribute_menu)[0] == "idx"
        assert next(
            action for action in attribute_menu.actions()
            if action.text() == window._dimension_attrs["C"]
        ).isChecked()
        next(
            action for action in attribute_menu.actions()
            if action.text() == "efo"
        ).trigger()
        assert window._dimension_attrs["C"] == "efo"
        assert colorbar._label == "C: efo"
        assert state.datasets[0].state["attribute_plot_state"]["c"] == "efo"
        assert _texts(_submenu(captured[0], "Placement")) == [
            "Vertical",
            "Horizontal",
        ]

        colorbar.restore_geometry([20, 25, 140, 110])
        assert not colorbar.uses_default_placement
        assert colorbar.serialized_geometry() == [20, 25, 140, 110]
        colorbar.set_orientation("vertical")
        _qt_app.processEvents()
        assert colorbar.uses_default_placement
        assert colorbar.x() + colorbar.width() == window._stack.width()

        colorbar.restore_geometry([20, 25, 140, 110])
        colorbar.set_orientation("horizontal")
        _qt_app.processEvents()
        assert colorbar.uses_default_placement
        assert colorbar.serialized_geometry() is None
        assert window._stack.contentsMargins().right() == 0
        assert window._stack.contentsMargins().top() == colorbar.height()
        assert colorbar.y() == 0
        assert colorbar.width() == window._stack.width()
        assert (
            state.datasets[0].state["attribute_plot_state"]["colorbar_geometry"]
            is None
        )
        colorbar.set_show_values(False)

        customize_calls: list[bool] = []
        colorbar._on_customize = lambda: customize_calls.append(True)
        QTest.mouseDClick(
            colorbar,
            Qt.MouseButton.LeftButton,
            pos=colorbar.rect().center(),
        )
        assert customize_calls == [True]

        window._set_colorbar_visible(False)
        assert colorbar.isHidden()
        menu = _capture_context_menu(monkeypatch, window)
        mapping = _submenu(menu, "C: mapping")
        assert not mapping.actions()[-1].isChecked()
    finally:
        window.close()

    restored = AttributeWindow(state, dataset_idx=0)
    try:
        assert restored._show_colorbar is False
        assert restored._colorbar.orientation == "horizontal"
        assert restored._colorbar.show_values is False
        assert restored._colorbar.uses_default_placement
        assert restored._dimension_attrs["C"] == "efo"
    finally:
        restored.close()


def test_floating_colorbar_uses_regular_compact_ruler_values():
    from minflux_viewer.ui.floating_colorbar import FloatingColorBar

    small = FloatingColorBar._regular_tick_values(0.01, 0.10)
    np.testing.assert_allclose(small, np.arange(1, 11, dtype=float) / 100.0)

    large = FloatingColorBar._regular_tick_values(50_000.0, 250_000.0)
    np.testing.assert_allclose(
        large,
        np.arange(50_000.0, 250_001.0, 25_000.0),
    )
    exponent = FloatingColorBar._scale_exponent(large)
    assert exponent == 3
    assert [FloatingColorBar._format_tick(value, exponent) for value in large] == [
        "50",
        "75",
        "100",
        "125",
        "150",
        "175",
        "200",
        "225",
        "250",
    ]
    assert FloatingColorBar._format_tick(0.21, 0) == "0.21"
    assert FloatingColorBar._format_endpoint(0.0109, 0, 0.01) == "0.01"
    assert FloatingColorBar._format_endpoint(0.0991, 0, 0.01) == "0.10"
    assert FloatingColorBar._format_endpoint(52_100.0, 3, 25_000.0) == "52"

    major, step = FloatingColorBar._regular_tick_spec(50_000.0, 250_000.0)
    minor = FloatingColorBar._minor_tick_values(50_000.0, 250_000.0, major, step)
    assert step == 25_000.0
    assert len(minor) == 32
    assert not any(value in major for value in minor)


def test_three_plane_grid_geometry_contains_xy_xz_and_yz_faces():
    from minflux_viewer.ui.gl_3d_reference import three_plane_grid_positions

    positions = three_plane_grid_positions(
        np.array([-0.5, -0.5, -0.5]),
        np.array([0.5, 0.5, 0.5]),
        target=5,
    )
    segments = positions.reshape(-1, 2, 3)
    assert np.any(np.all(segments[:, :, 2] == -0.5, axis=1))  # XY
    assert np.any(np.all(segments[:, :, 1] == -0.5, axis=1))  # XZ
    assert np.any(np.all(segments[:, :, 0] == -0.5, axis=1))  # YZ


def test_attribute_3d_axis_has_attribute_labels_and_numeric_ticks(_qt_app):
    pytest.importorskip("pyqtgraph.opengl")
    from minflux_viewer.ui.attribute_window import AttributeWindow

    window = AttributeWindow(_state(), dataset_idx=0)
    try:
        window._add_attribute_dimension()
        window._set_dimension_attribute("X", "xnm")
        window._set_dimension_attribute("Y", "ynm")
        window._set_dimension_attribute("Z", "znm")
        window._set_view_mode("3D")
        texts = [
            str(getattr(item, "text", ""))
            for item in window._gl_axis_items
            if getattr(item, "text", "")
        ]
        assert "X: xnm" in texts
        assert "Y: ynm" in texts
        assert "Z: znm" in texts
        assert any(text.replace(".", "", 1).isdigit() for text in texts)

        window._set_current_axis_visible(False)
        assert not window._gl_axis.visible()
        assert all(not item.visible() for item in window._gl_axis_items)
    finally:
        window.close()


def test_attribute_3d_bounding_box_menu_is_linked_and_persistent(
    monkeypatch, _qt_app,
):
    pytest.importorskip("pyqtgraph.opengl")
    from minflux_viewer.ui.attribute_window import AttributeWindow

    state = _state()
    window = AttributeWindow(state, dataset_idx=0)
    try:
        window._add_attribute_dimension()
        window._set_view_mode("3D")
        menu = _capture_context_menu(monkeypatch, window)
        texts = _texts(menu)
        assert texts[-4:] == ["", "Bounding Box", "", "Reset View"]
        action = next(
            action for action in menu.actions()
            if action.text() == "Bounding Box"
        )
        assert action.isCheckable() and action.isChecked()
        assert window._gl_box is not None and window._gl_box.visible()

        action.trigger()
        assert window._show_3d_bounding_box is False
        assert not window._gl_box.visible()
        assert state.datasets[0].state["attribute_plot_state"][
            "show_3d_bounding_box"
        ] is False
    finally:
        window.close()

    restored = AttributeWindow(state, dataset_idx=0)
    try:
        assert restored._view_mode == "3D"
        assert restored._show_3d_bounding_box is False
        assert restored._gl_box is not None and not restored._gl_box.visible()
    finally:
        restored.close()


def test_linear_mapping_uses_even_full_range_steps(_qt_app):
    from minflux_viewer.ui.attribute_window import AttributeWindow

    bins, lo, hi = AttributeWindow._linear_color_bins(
        np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    )
    np.testing.assert_array_equal(bins, [0, 64, 128, 191, 255])
    assert (lo, hi) == (0.0, 4.0)

    window = AttributeWindow(_state(), dataset_idx=0)
    try:
        _brushes, rgba, color_lo, color_hi = window._mapped_colors(
            np.array([0.0, 1.0, 2.0])
        )
        assert rgba.shape == (3, 4)
        assert np.unique(rgba[:, :3], axis=0).shape[0] == 3
        assert (color_lo, color_hi) == (0.0, 2.0)
    finally:
        window.close()


def test_3d_view_keeps_top_controls_and_uses_unit_cube(monkeypatch, _qt_app):
    from minflux_viewer.ui.attribute_window import AttributeWindow

    window = AttributeWindow(_state(), dataset_idx=0)
    fake_view = QWidget()
    window._stack.addWidget(fake_view)
    try:
        window._add_attribute_dimension()
        window._3d_view = fake_view
        monkeypatch.setattr(window, "_ensure_3d_built", lambda: True)
        monkeypatch.setattr(window, "_draw", lambda: None)
        window._set_view_mode("3D")

        assert window._stack.currentWidget() is fake_view
        assert window._x_combo.isVisibleTo(window)
        assert window._y_combo.isVisibleTo(window)
        assert (window._axis_a_label.text(), window._axis_b_label.text()) == ("X:", "Y:")

        positions = AttributeWindow._normalize_3d_positions(np.array([
            [0.0, 1_000.0, -2.0],
            [5.0, 2_000.0, 0.0],
            [10.0, 3_000.0, 2.0],
        ]))
        np.testing.assert_allclose(positions[0], [-0.5, -0.5, -0.5])
        np.testing.assert_allclose(positions[-1], [0.5, 0.5, 0.5])
    finally:
        window.close()


def test_3d_blending_stays_visible_on_light_and_dark_backgrounds():
    from minflux_viewer.ui.attribute_window import AttributeWindow

    # Additive OpenGL blending washes every point out against white.  Light
    # backgrounds need alpha-over, while dark backgrounds retain the brighter
    # additive rendering used by the existing 3-D scatter plot.
    assert AttributeWindow._gl_blend_mode_for_background((255, 255, 255, 255)) == (
        "translucent"
    )
    assert AttributeWindow._gl_blend_mode_for_background((0, 0, 0, 255)) == (
        "additive"
    )


def test_multidimensional_attribute_state_restores(_qt_app):
    from minflux_viewer.ui.attribute_window import AttributeWindow

    state = _state()
    window = AttributeWindow(state, dataset_idx=0)
    window._add_attribute_dimension()
    window._set_dimension_attribute("Z", "cfr")
    window._set_view_mode("XZ")
    window._add_attribute_dimension()
    window._set_dimension_attribute("C", "tid")
    window._set_c_mapping("plasma")
    window._set_black_background(True)
    window._set_current_axis_visible(False)
    window._set_current_grid_visible(False)
    window._apply_plot_style(
        {"symbol": "s", "size": 6, "alpha": 111, "color": (20, 40, 60)},
        color_changed=True,
    )
    window.close()

    restored = AttributeWindow(state, dataset_idx=0)
    try:
        assert restored._dimension_count == 4
        assert restored._view_mode == "XZ"
        assert restored._dimension_attrs["Z"] == "cfr"
        assert restored._dimension_attrs["C"] == "tid"
        assert restored._c_mapping == "plasma"
        assert restored._black_background is True
        assert restored._show_2d_axis is False
        assert restored._show_2d_grid is False
        assert restored._point_symbol == "s"
        assert restored._point_size == 6
        assert restored._point_alpha == 111
        assert restored._point_color == (20, 40, 60, 111)
        assert (restored._axis_a_label.text(), restored._axis_b_label.text()) == (
            "X:", "Z:",
        )
    finally:
        restored.close()
