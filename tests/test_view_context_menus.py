"""Context-menu layout and scatter marker-style regressions."""

from __future__ import annotations

import copy
import os
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtCore import QPoint
from PyQt6.QtGui import QContextMenuEvent
from PyQt6.QtWidgets import QApplication, QMenu

from minflux_viewer.colors import configure_colors
from minflux_viewer.core.app_state import AppState, default_prefs
from minflux_viewer.core.dataset import build_localization_dataset


@pytest.fixture
def _qt_app():
    if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
        pytest.skip("No display available for Qt tests")
    return QApplication.instance() or QApplication(sys.argv)


def _state() -> AppState:
    state = AppState()
    state.add_dataset(
        build_localization_dataset(
            name="menu-test",
            x_nm=np.array([0.0, 10.0, 20.0]),
            y_nm=np.array([0.0, 20.0, 10.0]),
            z_nm=np.array([0.0, 5.0, 10.0]),
        )
    )
    return state


def _capture_context_menu(monkeypatch, callback) -> QMenu:
    captured: list[QMenu] = []
    monkeypatch.setattr(QMenu, "exec", lambda menu, *_args: captured.append(menu))
    callback(QPoint(0, 0))
    assert len(captured) == 1
    return captured[0]


def _menu_texts(menu: QMenu) -> list[str]:
    return ["---" if action.isSeparator() else action.text() for action in menu.actions()]


def _submenu(menu: QMenu, title: str) -> QMenu:
    action = next(action for action in menu.actions() if action.text() == title)
    submenu = action.menu()
    assert submenu is not None
    return submenu


def test_scatter_view_menu_order(monkeypatch, _qt_app):
    from minflux_viewer.ui.scatter_window import ScatterWindow

    win = ScatterWindow(_state(), dataset_idx=0)
    try:
        assert win._info_label.text().startswith("3D  |  ")
        menu = _capture_context_menu(monkeypatch, win._show_context_menu)
        assert _menu_texts(_submenu(menu, "View")) == [
            "XY",
            "XZ",
            "YZ",
            "3D",
            "---",
            "Black background",
            "Axis",
            "Grid lines",
            "Plot style",
        ]
        assert "Black background" not in _menu_texts(menu)
        assert "Axis" not in _menu_texts(menu)

        win._set_current_axis_visible(True)
        win._set_current_grid_visible(False)
        assert not win._plot_2d.getPlotItem().getAxis("bottom").grid
        win._set_current_grid_visible(True)
        assert win._plot_2d.getPlotItem().getAxis("bottom").grid
    finally:
        win.close()


def test_scatter_color_by_starts_with_idx(monkeypatch, _qt_app):
    from minflux_viewer.ui.scatter_window import ScatterWindow

    state = _state()
    win = ScatterWindow(state, dataset_idx=0)
    try:
        assert win._cbar_combo.itemText(0) == "idx"

        menu = _capture_context_menu(monkeypatch, win._show_context_menu)
        assert _menu_texts(_submenu(menu, "Color by"))[0] == "idx"

        win._cbar_combo.setCurrentText("idx")
        dataset = state.datasets[0]
        values, bins, label, _vmin, _vmax = win._color_bins_for_points(
            None, None, None, dataset, np.arange(dataset.prop.num_loc),
        )
        np.testing.assert_array_equal(values, [1.0, 2.0, 3.0])
        assert np.unique(bins).size > 1
        assert label == "idx"
    finally:
        win.close()


def test_scatter_colorbar_is_linked_floating_and_persistent(
    monkeypatch, _qt_app,
):
    from minflux_viewer.ui.scatter_window import ScatterWindow

    state = _state()
    win = ScatterWindow(state, dataset_idx=0)
    try:
        win.show()
        _qt_app.processEvents()
        colorbar = win._colorbar
        assert not colorbar.isHidden()
        assert colorbar.uses_default_placement
        assert win.width() == 720 + colorbar.width()
        assert win._stack.contentsMargins().right() == colorbar.width()
        assert colorbar.x() + colorbar.width() == win._stack.width()
        assert colorbar.height() == win._stack.height()

        root_menu = _capture_context_menu(monkeypatch, win._show_context_menu)
        root_texts = _menu_texts(root_menu)
        assert root_texts.index("Color by") < root_texts.index("Colormap")
        assert root_texts.index("Colormap") < root_texts.index("Colorbar")
        colorbar_action = next(
            action for action in root_menu.actions()
            if action.text() == "Colorbar"
        )
        assert colorbar_action.isCheckable() and colorbar_action.isChecked()

        captured: list[QMenu] = []
        monkeypatch.setattr(QMenu, "exec", lambda menu, *_args: captured.append(menu))
        event = QContextMenuEvent(
            QContextMenuEvent.Reason.Mouse,
            QPoint(4, 4),
            colorbar.mapToGlobal(QPoint(4, 4)),
        )
        colorbar.contextMenuEvent(event)
        assert _menu_texts(captured[0]) == [
            "Hide colorbar",
            "Attribute:",
            "Show values",
            "Placement",
            "---",
            "Customize",
        ]
        attribute_menu = _submenu(captured[0], "Attribute:")
        assert _menu_texts(attribute_menu)[0] == "idx"
        next(
            action for action in attribute_menu.actions()
            if action.text() == "idx"
        ).trigger()
        assert win._cbar_combo.currentText() == "idx"
        assert colorbar._label == "idx"

        win._axis_combo.setCurrentText("3D")
        _qt_app.processEvents()
        assert win._3d_view is not None
        assert win._stack.currentWidget() is win._3d_view
        assert not colorbar.isHidden()
        win._axis_combo.setCurrentText("XY")

        win._set_colorbar_visible(False)
        assert colorbar.isHidden()
        assert win._stack.contentsMargins().right() == 0
    finally:
        win.close()

    restored = ScatterWindow(state, dataset_idx=0)
    try:
        assert restored._show_colorbar is False
        assert restored._colorbar.isHidden()
        assert restored._cbar_combo.currentText() == "idx"
    finally:
        restored.close()


def test_render_view_menu_contains_axis_and_grid_in_order(monkeypatch, _qt_app):
    from minflux_viewer.ui.render_window import RenderWindow

    win = RenderWindow(_state(), dataset_idx=0)
    try:
        menu = _capture_context_menu(monkeypatch, win._show_context_menu)
        texts = _menu_texts(_submenu(menu, "View"))
        assert texts.index("White background") < texts.index("Axis")
        assert texts.index("Axis") < texts.index("Grid lines")
        assert texts.index("Grid lines") < texts.index("Render Method")
        assert "Axis" not in _menu_texts(menu)

        win._set_axes_visible(False)
        win._set_grid_visible(True)
        assert not win._image_view.view.getAxis("bottom").isVisible()
        assert win._grid_item.isVisible()
    finally:
        win.close()


def test_render_colormap_menu_matches_scatter(monkeypatch, _qt_app):
    from minflux_viewer.ui.render_window import RenderWindow
    from minflux_viewer.ui.scatter_window import ScatterWindow

    render = RenderWindow(_state(), dataset_idx=0)
    scatter = ScatterWindow(_state(), dataset_idx=0)
    try:
        render_menu = _capture_context_menu(monkeypatch, render._show_context_menu)
        scatter_menu = _capture_context_menu(monkeypatch, scatter._show_context_menu)
        render_cmap = _submenu(render_menu, "Colormap")
        scatter_cmap = _submenu(scatter_menu, "Colormap")

        assert _menu_texts(render_cmap) == _menu_texts(scatter_cmap)
        render_solid = _submenu(render_cmap, "Solid color")
        assert _menu_texts(render_solid) == _menu_texts(
            _submenu(scatter_cmap, "Solid color")
        )
        # The one-off picker is gone: the list is the global COLOR registry.
        assert "Custom..." not in _menu_texts(render_solid)
        assert not hasattr(render, "_pick_solid_color")
        assert not hasattr(scatter, "_pick_solid_color")
    finally:
        render.close()
        scatter.close()


def test_saved_custom_solid_lut_still_renders(_qt_app):
    """Removing the menu entry must not break views already saved with one."""
    from minflux_viewer.ui.render_window import RenderWindow

    render = RenderWindow(_state(), dataset_idx=0)
    try:
        rgb = render._map_norm_to_rgb(
            np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
            "solid:custom:#123456",
        )
        np.testing.assert_allclose(rgb[0], 0.0)
        np.testing.assert_allclose(
            rgb[1],
            np.asarray([0x12, 0x34, 0x56], dtype=np.float32) / 255.0,
        )
        np.testing.assert_allclose(rgb[2], 1.0)
    finally:
        render.close()


def test_custom_solid_registry_reaches_render_and_scatter_menus(monkeypatch, _qt_app):
    from minflux_viewer.ui.render_window import RenderWindow
    from minflux_viewer.ui.scatter_window import ScatterWindow

    state = _state()
    colors = copy.deepcopy(state.prefs["colors"])
    colors["solid"].pop("Orange")
    colors["solid"]["Ocean"] = [12, 34, 56, 78]
    state.apply_color_preferences(colors)
    expected = [*colors["solid"]]

    render = RenderWindow(state, dataset_idx=0)
    scatter = ScatterWindow(state, dataset_idx=0)
    try:
        render_menu = _capture_context_menu(monkeypatch, render._show_context_menu)
        scatter_menu = _capture_context_menu(monkeypatch, scatter._show_context_menu)
        assert _menu_texts(_submenu(_submenu(render_menu, "Colormap"), "Solid color")) == expected
        assert _menu_texts(_submenu(_submenu(scatter_menu, "Colormap"), "Solid color")) == expected
    finally:
        render.close()
        scatter.close()
        configure_colors(default_prefs())


def test_scatter_plot_style_applies_and_persists(_qt_app):
    from minflux_viewer.ui.scatter_window import ScatterWindow

    state = _state()
    win = ScatterWindow(state, dataset_idx=0)
    try:
        win._apply_plot_style(
            {
                "symbol": "s",
                "size": 7,
                "alpha": 128,
                "color": (12, 34, 56),
            },
            color_changed=True,
        )
        assert win._point_symbol == "s"
        assert win._point_size == 7
        assert win._point_alpha == 128
        assert win._cmap_combo.currentText() == "solid:custom:#0c2238"
        saved = state.datasets[0].state["scatter_plot_state"]
        assert saved["point_symbol"] == "s"
        assert saved["point_size"] == 7
        assert saved["point_alpha"] == 128
        assert saved["colormap"] == "solid:custom:#0c2238"
    finally:
        win.close()

    restored = ScatterWindow(state, dataset_idx=0)
    try:
        assert restored._point_symbol == "s"
        assert restored._point_size == 7
        assert restored._point_alpha == 128
    finally:
        restored.close()
