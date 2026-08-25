"""Context-driven 2-D/3-D/4-D Attribute Plot regressions."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QContextMenuEvent, QMouseEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMenu, QWidget

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.core.iteration import FLATTEN_LABEL, STACKED_LABEL
from minflux_viewer.core.roi import RoiRecord


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


def test_adding_z_offers_the_attribute_list_and_shows_the_top_row_control(
    monkeypatch, _qt_app,
):
    """Z and C are added from the menu, each choosing its attribute there."""
    from minflux_viewer.ui.attribute_window import AttributeWindow

    window = AttributeWindow(_state(), dataset_idx=0)
    try:
        assert (window._has_z, window._has_c) == (False, False)
        assert window._view_mode == "XY"
        assert window._z_label.isHidden() and window._c_label.isHidden()

        menu = _capture_context_menu(monkeypatch, window)
        assert _texts(menu) == [
            "View",
            "",
            "add new attribute as Z",
            "add new attribute as C",
            "",
            "Reset View",
        ]
        # The attribute is chosen in the same gesture.
        add_z = _submenu(menu, "add new attribute as Z")
        assert _texts(add_z) == window._numeric_attrs
        next(a for a in add_z.actions() if a.text() == "efo").trigger()

        assert (window._has_z, window._has_c) == (True, False)
        assert window._dimension_attrs["Z"] == "efo"
        assert not window._z_label.isHidden() and not window._z_combo.isHidden()
        assert window._z_combo.currentText() == "efo"
        assert window._c_label.isHidden()
        window._z_combo.setCurrentText("cfr")
        assert window._dimension_attrs["Z"] == "cfr"

        # With Z present the entry is replaced by its removal, and the
        # projections open up.
        menu = _capture_context_menu(monkeypatch, window)
        assert _texts(menu) == [
            "View",
            "",
            "add new attribute as C",
            "remove Z attribute",
            "swap Z / C",
            "",
            "Reset View",
        ]
        assert _texts(_submenu(menu, "View"))[:4] == ["XY", "XZ", "YZ", "3D"]
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
            "add new attribute as Z",
            "add new attribute as C",
            "",
            "Reset View",
        ]
        view_menu = _submenu(menu, "View")
        assert _texts(view_menu) == [
            "XY",
            "",
            "Black background",
            "Axis",
            "Grid lines",
            "Plot style",
            "Thinning",
            "Legend",
            "Colorbar",
        ]
        thinning = next(a for a in view_menu.actions() if a.text() == "Thinning")
        assert thinning.isCheckable() and thinning.isChecked() is window._thinning
        colorbar = next(a for a in view_menu.actions() if a.text() == "Colorbar")
        # Nothing to colour yet, so the toggle is offered but inert.
        assert colorbar.isCheckable() and not colorbar.isEnabled()
    finally:
        window.close()


def test_c_can_exist_without_z_and_each_is_removed_on_its_own(
    monkeypatch, _qt_app,
):
    """C is independent of Z: XY, XYZ, XYC and XYZC are all reachable."""
    from minflux_viewer.ui.attribute_window import AttributeWindow

    window = AttributeWindow(_state(), dataset_idx=0)
    try:
        menu = _capture_context_menu(monkeypatch, window)
        add_c = _submenu(menu, "add new attribute as C")
        next(a for a in add_c.actions() if a.text() == "tid").trigger()

        # XYC: colour without a third axis, so no 3-D projections.
        assert (window._has_z, window._has_c) == (False, True)
        assert window._dimension_attrs["C"] == "tid"
        assert window._z_label.isHidden()
        assert not window._c_label.isHidden()
        assert window._c_combo.currentText() == "tid"
        menu = _capture_context_menu(monkeypatch, window)
        assert _texts(_submenu(menu, "View"))[:2] == ["XY", ""]
        assert _texts(menu) == [
            "View",
            "",
            "add new attribute as Z",
            "remove C attribute",
            "swap Z / C",
            "",
            "Reset View",
        ]

        window._add_dimension("Z", "efo")
        window.show()
        _qt_app.processEvents()
        assert min(
            window._x_combo.width(),
            window._y_combo.width(),
            window._z_combo.width(),
            window._c_combo.width(),
        ) >= 56
        menu = _capture_context_menu(monkeypatch, window)
        assert _texts(menu) == [
            "View",
            "",
            "remove Z attribute",
            "remove C attribute",
            "swap Z / C",
            "",
            "Reset View",
        ]

        # Removing one leaves the other alone.
        window._remove_dimension("Z")
        assert (window._has_z, window._has_c) == (False, True)
        assert window._z_label.isHidden() and not window._c_label.isHidden()
        window._remove_dimension("C")
        assert (window._has_z, window._has_c) == (False, False)
        assert window._c_label.isHidden()
    finally:
        window.close()


def test_swap_moves_the_attribute_between_z_and_c(_qt_app):
    """With both present the two exchange; with one, the dimension moves."""
    from minflux_viewer.ui.attribute_window import AttributeWindow

    window = AttributeWindow(_state(), dataset_idx=0)
    try:
        window._add_dimension("Z", "efo")
        window._add_dimension("C", "cfr")
        window._swap_z_and_c()
        assert (window._has_z, window._has_c) == (True, True)
        assert window._dimension_attrs["Z"] == "cfr"
        assert window._dimension_attrs["C"] == "efo"

        # XYZ -> XYC: Z leaves, so a Z projection has to come back to XY.
        window._remove_dimension("C")
        window._set_view_mode("XZ")
        assert window._view_mode == "XZ"
        window._swap_z_and_c()
        assert (window._has_z, window._has_c) == (False, True)
        assert window._dimension_attrs["C"] == "cfr"
        assert window._view_mode == "XY"

        # ...and back the other way.
        window._swap_z_and_c()
        assert (window._has_z, window._has_c) == (True, False)
        assert window._dimension_attrs["Z"] == "cfr"
    finally:
        window.close()


def test_c_dimension_disables_stacked_iterations(monkeypatch, _qt_app):
    """C owns the colour, so the per-iteration colour series cannot coexist."""
    from minflux_viewer.ui.attribute_window import AttributeWindow

    window = AttributeWindow(_state(), dataset_idx=0)
    try:
        window._add_dimension("Z")
        window._iter_combo.setCurrentText(STACKED_LABEL)
        assert window._iter_combo.currentText() == STACKED_LABEL

        window._add_dimension("C")
        assert (window._has_z, window._has_c) == (True, True)
        assert window._view_mode == "XY"
        assert not window._c_label.isHidden()
        assert window._c_combo.currentText() == window._dimension_attrs["C"]
        assert window._iter_combo.currentText() == FLATTEN_LABEL
        stacked_item = window._iter_combo.model().item(
            window._iter_combo.findText(STACKED_LABEL)
        )
        assert stacked_item is not None and not stacked_item.isEnabled()

        # The colormap is no longer a context submenu; it is set through the
        # LUT editor and the API the colorbar drives.
        menu = _capture_context_menu(monkeypatch, window)
        assert "C: mapping" not in _texts(menu)
        window._set_c_mapping("cividis")
        assert window._c_mapping == "cividis"
        assert "C=" in window._info.text() and "[cividis]" in window._info.text()

        window._remove_dimension("C")
        assert (window._has_z, window._has_c) == (True, False)
        assert window._c_label.isHidden()
        assert window._c_combo.isHidden()
        assert stacked_item.isEnabled()
    finally:
        window.close()


def test_lines_keep_styled_markers_visible(_qt_app):
    """Lines adds a connecting curve without taking the markers away.

    On the GPU the markers are on the canvas and the curve is still the
    pyqtgraph one, so the option works in both renderers.
    """
    from minflux_viewer.ui.attribute_window import AttributeWindow

    window = AttributeWindow(_state(), dataset_idx=0)
    try:
        window.set_gpu_2d(False)               # the CPU marker path
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

        window.set_gpu_2d(True)
        curve, scatter = window._series_items[0]
        assert len(curve.xData) == 8           # the line is still drawn
        assert len(scatter.data) == 0          # ...and the points are not paid for twice
        if window._gl2d_view is not None:
            assert window._gl2d_items
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
        window._add_dimension("Z")
        window._add_dimension("C")
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
        window._add_dimension("Z")
        window._add_dimension("C")
        colorbar = window._colorbar
        assert not colorbar.isHidden()
        assert colorbar.orientation == "vertical"
        assert colorbar.show_values is True
        window.show()
        _qt_app.processEvents()
        assert not window.grab().isNull()  # exercises the painted gradient/ticks
        # Headless/platform size hints can impose a wider minimum; the
        # colorbar must add at least its own width to the requested plot width.
        assert window.width() >= 780 + colorbar.width()
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
            "Undock",
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
        # The way back is View ▸ Colorbar; the colorbar's own menu can only
        # hide it.
        menu = _capture_context_menu(monkeypatch, window)
        entry = next(
            item for item in _submenu(menu, "View").actions()
            if item.text() == "Colorbar"
        )
        assert entry.isEnabled() and not entry.isChecked()
        entry.trigger()
        assert window._show_colorbar is True
        window._set_colorbar_visible(False)
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


def test_docked_colorbar_paints_flush_and_undocked_paints_a_panel(_qt_app):
    """Docked reverts to the in-plot look; undocked keeps the floating panel."""
    from PyQt6.QtCore import QRect
    from PyQt6.QtGui import QColor

    from minflux_viewer.ui.floating_colorbar import FloatingColorBar

    parent = QWidget()
    parent.resize(400, 300)
    plot_area = QRect(10, 20, 300, 240)          # the ViewBox, in parent coords
    bar = FloatingColorBar(
        parent,
        on_visibility_changed=lambda _visible: None,
        on_customize=lambda: None,
        on_state_changed=lambda: None,
        attribute_names=lambda: ["efo"],
        current_attribute=lambda: "efo",
        on_attribute_changed=lambda _name: None,
        plot_area=lambda: plot_area,
        background_color=lambda: QColor(0, 0, 0),
    )
    try:
        # Docked: the plot's own background, light ink on it, and a gradient
        # spanning exactly the plot's data area.
        assert bar.docked and bar.uses_default_placement
        assert bar._panel_color() == QColor(0, 0, 0)
        ink, halo = bar._ink_colors()
        assert ink.lightness() > halo.lightness()
        assert bar._docked_plot_span() == (20, 259)  # QRect.bottom() is inclusive
        assert bar.serialized_geometry() is None

        # A docked bar is fixed: a drag must not move it.
        docked_geometry = bar.geometry()
        bar.mousePressEvent(
            QMouseEvent(
                QEvent.Type.MouseButtonPress,
                QPointF(20.0, 40.0),
                QPointF(20.0, 40.0),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        bar.mouseMoveEvent(
            QMouseEvent(
                QEvent.Type.MouseMove,
                QPointF(-60.0, 90.0),
                QPointF(-60.0, 90.0),
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        assert bar.geometry() == docked_geometry

        # Undocked: back to the movable translucent panel, and no longer tied
        # to the plot's geometry.
        bar.set_docked(False)
        assert not bar.docked
        assert bar._panel_color() == QColor(250, 250, 250, 225)
        assert bar._ink_colors()[0] == QColor(25, 25, 25)
        assert bar._docked_plot_span() is None
        assert bar.serialized_geometry() is not None

        # Endpoint labels stay inside the widget at both ends of the ruler.
        assert bar._value_label_rect(10, 0, 30, 14).top() == 0
        assert (
            bar._value_label_rect(10, bar.height(), 30, 14).bottom()
            == bar.height() - 1
        )
        assert bar._value_label_rect(10, 100, 30, 14).top() == 93
    finally:
        bar.deleteLater()
        parent.deleteLater()


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


def _thinning_window(_qt_app, n_points: int = 400):
    """An Attribute Plot over a plain idx/efo dataset, with a settled view."""
    from minflux_viewer.ui.attribute_window import AttributeWindow

    state = AppState()
    state.add_dataset(
        build_localization_dataset(
            name="thinning",
            x_nm=np.arange(n_points, dtype=float),
            y_nm=np.arange(n_points, dtype=float),
            z_nm=np.zeros(n_points),
            attrs={"efo": np.linspace(1_000.0, 9_000.0, n_points)},
        )
    )
    window = AttributeWindow(state, dataset_idx=0)
    window._dimension_attrs.update({"X": "idx", "Y": "efo"})
    window._view_mode = "XY"
    # Thinning is the CPU renderer's remedy; the GPU never applies it.
    window.set_gpu_2d(False)
    return state, window


def test_thinning_is_zoom_aware_and_ignores_the_unseen_rest(monkeypatch, _qt_app):
    """The drawn rows follow the view, not the size of the rest of the series.

    This is the defect that made 'Valid only' look inverted: a global stride of
    ceil(n / budget) sampled a fixed zoom window 44x more coarsely once
    unchecking the box grew the *unseen* part of the selection.
    """
    from minflux_viewer.ui import attribute_window as aw

    _state_obj, window = _thinning_window(_qt_app)
    try:
        monkeypatch.setattr(aw, "_MAX_DISPLAY_POINTS", 20)
        window._view_box.disableAutoRange()
        window._view_box.setRange(xRange=(0.0, 9.0), yRange=(-1.0, 1.0), padding=0.0)

        shared = np.arange(12.0)
        small = {"X": np.arange(1_000.0), "Y": np.zeros(1_000)}
        large = {
            "X": np.concatenate([np.arange(1_000.0), np.arange(1e5, 2e5)]),
            "Y": np.zeros(101_000),
        }

        drawn_small, n_small = window._thin_for_view(small, small["X"].size)
        assert window._view_restricted and window._thin_step == 1
        drawn_large, n_large = window._thin_for_view(large, large["X"].size)
        assert window._view_restricted and window._thin_step == 1

        # Same window, same points — regardless of how much lies outside it.
        np.testing.assert_array_equal(drawn_small["X"], shared)
        np.testing.assert_array_equal(drawn_large["X"], shared)
        # ...and the reported total still counts everything selected.
        assert (n_small, n_large) == (1_000, 101_000)

        # Zooming in further reveals nothing new here (all 12 were already
        # drawn); zooming out past the budget falls back to a stride.
        window._view_box.setRange(
            xRange=(0.0, 1e5), yRange=(-1.0, 1.0), padding=0.0
        )
        wide, _n = window._thin_for_view(large, large["X"].size)
        assert wide["X"].size <= 20
    finally:
        window.close()


def test_thinning_leaves_the_view_alone_while_it_still_auto_ranges(
    monkeypatch, _qt_app
):
    """Before the user's first zoom the range describes nothing yet.

    pyqtgraph reports a placeholder range until it refits to the new data, so
    restricting to it would draw almost nothing and then fit the view to that
    remnant.
    """
    from minflux_viewer.ui import attribute_window as aw

    _state_obj, window = _thinning_window(_qt_app)
    try:
        monkeypatch.setattr(aw, "_MAX_DISPLAY_POINTS", 20)
        window._view_box.enableAutoRange()
        values = {"X": np.arange(1_000.0), "Y": np.zeros(1_000)}
        drawn, n = window._thin_for_view(values, 1_000)
        assert not window._view_restricted
        assert window._thin_step == 50 and drawn["X"].size == 20 and n == 1_000
    finally:
        window.close()


def test_thinning_spends_the_budget_on_rows_that_can_be_drawn(
    monkeypatch, _qt_app
):
    """A NaN coordinate paints nothing, so it must not consume the budget.

    This is the 3-D form of the same defect: there is no view range to restrict
    to, so a global stride spread the budget over rows that could never appear.
    On a real m2410 file, unchecking 'Valid only' in 3-D drew 82,162 points
    where checking it drew 246,437 — 88.8 % of the selection being empty probes
    with NaN coordinates.
    """
    from minflux_viewer.ui import attribute_window as aw

    _state_obj, window = _thinning_window(_qt_app)
    try:
        window._view_box.enableAutoRange()          # no view restriction
        y = np.zeros(1_000)
        y[::2] = np.nan                             # half cannot be drawn
        values = {"X": np.arange(1_000.0), "Y": y}

        monkeypatch.setattr(aw, "_MAX_DISPLAY_POINTS", 50)
        drawn, n = window._thin_for_view(values, 1_000)
        assert n == 1_000
        assert np.all(np.isfinite(drawn["Y"]))
        assert drawn["X"].size == 50
        assert window._thin_step == 10 and window._thin_drawable == 500
        assert window._point_count_text([{"values": drawn}], n) == (
            "50 of 1,000 points (spatially representative cells of the 500 "
            "with finite coordinates)"
        )

        # Room for every drawable row: no stride at all, and the count no
        # longer depends on how many undrawable rows sit beside them.
        monkeypatch.setattr(aw, "_MAX_DISPLAY_POINTS", 600)
        drawn, n = window._thin_for_view(values, 1_000)
        assert drawn["X"].size == 500 and window._thin_step == 1
        assert window._point_count_text([{"values": drawn}], n) == (
            "500 of 1,000 points (all 500 with a finite value)"
        )
    finally:
        window.close()


def test_thinning_off_draws_every_point_and_says_so(monkeypatch, _qt_app):
    from minflux_viewer.ui import attribute_window as aw

    _state_obj, window = _thinning_window(_qt_app)
    try:
        monkeypatch.setattr(aw, "_MAX_DISPLAY_POINTS", 20)
        values = {"X": np.arange(1_000.0), "Y": np.zeros(1_000)}

        window._thinning = False
        drawn, n = window._thin_for_view(values, 1_000)
        assert drawn["X"].size == 1_000 and n == 1_000
        assert not window._view_restricted
        records = [{"values": drawn}]
        assert window._point_count_text(records, n) == "1,000 points"
        assert window._thinned is False

        # ...and the read-out distinguishes "off screen" from "sampled".
        window._thinning = True
        window._view_restricted = True
        window._thin_step = 1
        assert window._point_count_text([{"values": {"X": np.zeros(12)}}], 1_000) == (
            "12 of 1,000 points (visible range)"
        )
        window._thin_step = 4
        assert window._point_count_text([{"values": {"X": np.zeros(12)}}], 1_000) == (
            "12 of 1,000 points (1 in 4 of the visible range)"
        )
        window._view_restricted = False
        window._thin_step = 50
        assert window._point_count_text([{"values": {"X": np.zeros(20)}}], 1_000) == (
            "20 of 1,000 points (1 in 50)"
        )
    finally:
        window.close()


def test_c_colour_brushes_are_cached_per_lut_entry(_qt_app):
    """Indexing <=257 brushes must paint exactly what one-per-point did."""
    from minflux_viewer.colormaps import colormap_lut

    _state_obj, window = _thinning_window(_qt_app)
    try:
        values = np.concatenate([np.linspace(0.0, 1.0, 500), [np.nan]])
        brushes, rgba, lo, hi = window._mapped_colors(values)
        assert len(brushes) == values.size

        bins, _lo, _hi = window._linear_color_bins(values, levels=None)
        lut = colormap_lut(
            window._c_mapping,
            n=256,
            invert=window._lut_invert,
            gamma=window._lut_gamma,
            alpha=True,
        ).copy()
        lut[:, 3] = window._point_alpha
        expected = lut[bins]
        expected[~np.isfinite(values), 3] = 0
        for brush, want in zip(brushes, expected):
            colour = brush.color()
            if want[3] == 0:
                assert colour.alpha() == 0
                continue
            assert (colour.red(), colour.green(), colour.blue(), colour.alpha()) == (
                tuple(int(channel) for channel in want)
            )
        # Distinct QBrush objects are shared, not rebuilt per point.
        assert len({id(brush) for brush in brushes}) <= 257
        assert lo == 0.0 and hi == 1.0
        assert rgba.shape == (values.size, 4)
    finally:
        window.close()


def test_thinning_lives_in_preferences_and_the_view_menu(monkeypatch, _qt_app):
    """No top-row checkbox: the default is a preference, the switch is in View."""
    from minflux_viewer.core.app_state import DEFAULT_PREFS

    assert DEFAULT_PREFS["plot"]["attribute_thinning"] is True

    state, window = _thinning_window(_qt_app)
    try:
        assert not hasattr(window, "_thin_chk")
        assert window._thinning

        menu = _capture_context_menu(monkeypatch, window)
        action = next(
            item for item in _submenu(menu, "View").actions()
            if item.text() == "Thinning"
        )
        assert action.isCheckable() and action.isChecked()
        action.trigger()                       # a menu click toggles first
        assert window._thinning is False
        saved = state.datasets[0].state["attribute_plot_state"]
        assert saved["thinning"] is False

        # Preferences OK pushes the new default onto open windows.
        state.prefs["plot"]["attribute_thinning"] = True
        window.refresh_preferences()
        assert window._thinning
    finally:
        window.close()


def test_toggles_keep_the_view_range_so_states_can_be_compared(_qt_app):
    """Valid only / Lines / Filtered only must not move the view.

    Their whole use is A/B comparison at one zoom level, and the data extent
    can change under them — auto-range would refit and throw away what the
    user was looking at.
    """
    _state_obj, window = _thinning_window(_qt_app, n_points=300)
    try:
        window.show()
        _qt_app.processEvents()
        window._view_box.setRange(
            xRange=(50.0, 150.0), yRange=(2_000.0, 6_000.0), padding=0.0
        )
        before = [tuple(pair) for pair in window._view_box.viewRange()]
        for checkbox in (window._valid_chk, window._lines_chk, window._filter_chk):
            checkbox.setChecked(not checkbox.isChecked())
            _qt_app.processEvents()
            assert [
                tuple(pair) for pair in window._view_box.viewRange()
            ] == pytest.approx(before)
    finally:
        window.close()


def test_gpu_2d_is_the_default_and_draws_behind_the_plot(monkeypatch, _qt_app):
    """The GPU renderer replaces the markers, not the plot, and is the default.

    The GL canvas sits in the same layout cell *behind* a transparented
    pyqtgraph plot, so axes, grid, zoom, Reset View and — the point of it — ROI
    drawing and selection keep working, while the points go to the GPU. The
    colour budget exists only because of pyqtgraph's per-point Python cost, so
    it does not apply here.
    """
    from minflux_viewer.ui.attribute_window import AttributeWindow

    state = _state()
    window = AttributeWindow(state, dataset_idx=0)
    try:
        assert window._use_gl_2d is True            # tried without being asked
        window._add_dimension("Z")
        window._add_dimension("C")
        window.show()
        _qt_app.processEvents()

        # No renderer switch in the plot's own menu any more.
        menu = _capture_context_menu(monkeypatch, window)
        assert not any(
            item.text().startswith("GPU rendering")
            for item in _submenu(menu, "View").actions()
        )
        values = {"X": np.zeros(3), "Y": np.zeros(3), "C": np.zeros(3)}
        assert window._display_budget(values) == window._gpu_point_limit()
        assert window._display_budget(values) > 0

        if window._gl2d_view is None or not window._gl2d_view.isValid():
            pytest.skip("OpenGL unavailable")
        assert window._stack.currentWidget() is window._plot_page
        assert window._gl2d_view.parent() is window._plot_page
        assert window._gl2d_view.isVisible()
        assert window._gl2d_items                     # points on the GPU
        assert window._plot.testAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground
        )
        # The CPU markers are not drawn twice.
        assert all(not item[1].isVisible() for item in window._series_items)

        # ROI selection is the ordinary ViewBox one, so it still resolves.
        record = RoiRecord.create(
            "rectangle", {"bounds": [-1.0, -1.0, 1e6, 1e6]}
        )
        selection = window.compute_roi_selection(record)
        assert selection is not None
        _dataset, mask, context = selection
        assert mask.any() and context["source_view"] == "attribute"

        # Back to pyqtgraph: opaque plot, canvas hidden, markers drawn again.
        window.set_gpu_2d(False)
        _qt_app.processEvents()
        assert window._display_budget(values) == 50_000
        assert not window._plot.testAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground
        )
        assert not window._gl2d_view.isVisible()
        assert all(item[1].isVisible() for item in window._series_items)
    finally:
        window.close()


class _FakeDrag:
    """The parts of pyqtgraph's MouseDragEvent the zoom handler uses."""

    def __init__(self, scene_pos, down_pos, *, start=False, finish=False):
        self._scene_pos, self._down_pos = scene_pos, down_pos
        self._start, self._finish = start, finish

    def button(self):
        return Qt.MouseButton.LeftButton

    def isStart(self):
        return self._start

    def isFinish(self):
        return self._finish

    def accept(self):
        pass

    def buttonDownScenePos(self, _button=None):
        return self._down_pos

    def scenePos(self):
        return self._scene_pos


def test_gpu_zoom_and_reset_follow_the_data_not_the_rubber_band(_qt_app):
    """The ViewBox must know where the GPU-drawn data is.

    ``childrenBounds`` skips invisible items and the GPU path leaves the
    scatter empty and hidden, so without an explicit bounds item the view has
    nothing to fit: auto-range then locked onto the zoom rubber band as it was
    dragged, threw the data out of range, and neither Reset View nor the ``A``
    button could find it again — the plot had to be closed and reopened.
    """
    from PyQt6.QtCore import QPointF

    _state_obj, window = _thinning_window(_qt_app, n_points=400)
    try:
        window.show()
        _qt_app.processEvents()
        window._use_gl_2d = True
        window._draw()
        _qt_app.processEvents()
        if window._gl2d_view is None or not window._gl2d_view.isValid():
            pytest.skip("OpenGL unavailable")

        # The data rectangle is published to the ViewBox, not drawn.
        assert window._gl_bounds_item is not None
        rect = window._gl_bounds_item.boundingRect()
        assert rect.left() == pytest.approx(1.0)
        assert rect.right() == pytest.approx(400.0)
        assert rect.top() == pytest.approx(1_000.0)
        assert rect.bottom() == pytest.approx(9_000.0)

        view_box = window._view_box
        before = [tuple(pair) for pair in view_box.viewRange()]
        window._set_zoom_active(True)
        start = view_box.mapViewToScene(QPointF(100.0, 3_000.0))
        end = view_box.mapViewToScene(QPointF(200.0, 5_000.0))

        window._zoom_mouse_drag_event(_FakeDrag(start, start, start=True))
        _qt_app.processEvents()
        assert [tuple(pair) for pair in view_box.viewRange()] == before
        window._zoom_mouse_drag_event(_FakeDrag(end, start))
        _qt_app.processEvents()
        assert [tuple(pair) for pair in view_box.viewRange()] == before

        window._zoom_mouse_drag_event(_FakeDrag(end, start, finish=True))
        _qt_app.processEvents()
        (x0, x1), (y0, y1) = view_box.viewRange()
        assert (x0, x1) == pytest.approx((100.0, 200.0))
        assert (y0, y1) == pytest.approx((3_000.0, 5_000.0))

        # ...and the data comes back.
        window._reset_view()
        _qt_app.processEvents()
        (x0, x1), (y0, y1) = view_box.viewRange()
        assert x0 <= 1.0 and x1 >= 400.0
        assert y0 <= 1_000.0 and y1 >= 9_000.0

        # Leaving GPU mode takes the stand-in with it; pyqtgraph reports its own.
        window._use_gl_2d = False
        window._draw()
        assert window._gl_bounds_item is None
    finally:
        window.close()


def test_gpu_mode_falls_back_to_the_cpu_without_opengl(monkeypatch, _qt_app):
    """A machine with no usable OpenGL must still draw, and must say why.

    The bundled app ships PyOpenGL and Qt's software OpenGL, but a remote
    session or a locked-down driver can still leave the canvas without a
    context. Falling back silently would leave a ticked menu entry over an
    empty plot.
    """
    import sys

    # Poison the lazy import *before* the window is built: GPU rendering is
    # the default, so the canvas is attempted on the very first draw.
    monkeypatch.setitem(sys.modules, "pyqtgraph.opengl", None)
    _state_obj, window = _thinning_window(_qt_app, n_points=200)
    try:
        assert window._use_gl_2d is False       # tried, failed, reverted
        window._draw()

        assert window._use_gl_2d is False           # the toggle reverts
        assert window._gl2d_view is None
        assert window._gl_bounds_item is None
        assert "GPU rendering unavailable" in window._info.text()
        assert "drawing on the CPU" in window._info.text()
        assert any(
            "GPU rendering unavailable" in entry.get("message", "")
            for entry in _state_obj.log_history
        )

        # ...and the points are on screen, drawn by pyqtgraph.
        drawn = sum(
            len(item[1].getData()[0]) for item in window._series_items
        )
        assert drawn == 200
        assert all(item[1].isVisible() for item in window._series_items)
        assert not window._plot.testAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground
        )
    finally:
        window.close()


def test_gpu_mode_falls_back_when_the_canvas_draws_nothing(monkeypatch, _qt_app):
    """A valid context that paints nothing must not leave an empty plot.

    Forcing Qt onto its software rasterizer reproduces exactly that: the widget
    reports `isValid()`, the draw completes, and the canvas stays one flat
    colour while pyqtgraph swallows the GL errors. The pixels are the only
    evidence, so they are checked once per canvas.
    """
    from minflux_viewer.ui.attribute_window import AttributeWindow

    _state_obj, window = _thinning_window(_qt_app, n_points=200)
    try:
        window.show()
        _qt_app.processEvents()
        window._use_gl_2d = True
        window._draw()
        _qt_app.processEvents()
        if window._gl2d_view is None or not window._gl2d_view.isValid():
            pytest.skip("OpenGL unavailable")

        monkeypatch.setattr(
            AttributeWindow, "_canvas_rendered_nothing", staticmethod(lambda _v: True)
        )
        # Two blank frames in a row are what triggers the fallback.
        window._verify_gl_canvas()
        window._verify_gl_canvas()
        _qt_app.processEvents()

        assert window._use_gl_2d is False
        assert "rendered nothing" in window._info.text()
        assert not window._gl2d_view.isVisible()
        drawn = sum(len(item[1].getData()[0]) for item in window._series_items)
        assert drawn == 200
    finally:
        window.close()


def test_stacked_iterations_use_a_ramp_spread_over_the_series(_qt_app):
    """Iterations are ordered, so their colours run as an ordered ramp.

    And the ramp is spread over the iterations actually drawn: three of them
    should still run end to end, not sit together in the ramp's dark corner.
    """
    from minflux_viewer.core.app_state import default_prefs
    from minflux_viewer.ui.attribute_window import _iter_color
    from minflux_viewer.ui.histogram_window import _iter_color as _hist_iter_color

    prefs = default_prefs()
    ten = [_iter_color(prefs, k, 10)[:3] for k in range(10)]
    assert ten[0] == (68, 1, 84) and ten[-1] == (253, 231, 37)   # viridis ends
    # Monotone in luminance: neighbouring iterations are ordered, not random.
    luminance = [0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in ten]
    assert luminance == sorted(luminance)

    three = [_iter_color(prefs, k, 3)[:3] for k in range(3)]
    assert three[0] == ten[0] and three[-1] == ten[-1]
    assert three[1] not in (ten[0], ten[-1])

    # The histogram must colour the same series identically.
    assert [_hist_iter_color(k, prefs, 3)[:3] for k in range(3)] == three


def test_stacked_legend_is_drawn_by_us_on_every_renderer(_qt_app):
    """The iteration notation is our own widget, not pyqtgraph's legend.

    pyqtgraph samples the *item*, so on the GPU path — where the scatter items
    are empty and hidden — every row drew an "invisible eye" that turned into a
    dot when clicked; and its legend cannot appear over the 3-D view at all.
    """
    from minflux_viewer.core.iteration import STACKED_LABEL
    from minflux_viewer.ui.attribute_window import AttributeWindow

    state = _state()
    window = AttributeWindow(state, dataset_idx=0)
    try:
        window.show()
        _qt_app.processEvents()
        window._iter_combo.setCurrentText(STACKED_LABEL)
        _qt_app.processEvents()

        legend = window._legend
        assert legend.isVisible()
        labels = [label for label, _color in legend._entries]
        assert labels == ["1st", "2nd", "3rd"]
        assert len({color for _label, color in legend._entries}) == 3
        assert legend.docked

        # It follows the plot into 3-D, where a scene legend could not go.
        window._add_dimension("Z", "cfr")
        window._set_view_mode("3D")
        _qt_app.processEvents()
        assert legend.isVisible()

        window._set_view_mode("XY")
        _qt_app.processEvents()
        legend.set_docked(False)
        assert not legend.docked and legend.serialized_geometry() is not None
        legend.set_docked(True)
        assert legend.docked and legend.serialized_geometry() is None

        # Only the stacked view colours by iteration.
        window._iter_combo.setCurrentText(window._iter_combo.itemText(0))
        _qt_app.processEvents()
        assert not legend.isVisible()

        # ...and it can be switched off and back on from the View menu.
        window._iter_combo.setCurrentText(STACKED_LABEL)
        _qt_app.processEvents()
        window._set_legend_visible(False)
        assert not legend.isVisible()
        assert (
            state.datasets[0].state["attribute_plot_state"]["show_legend"] is False
        )
        window._set_legend_visible(True)
        assert legend.isVisible()
    finally:
        window.close()


def test_line_style_follows_the_matlab_specifiers(_qt_app):
    """Plot style carries a MATLAB line spec, honoured by both renderers."""
    from PyQt6.QtCore import Qt as _Qt

    from minflux_viewer.ui.attribute_window import AttributeWindow

    state = _state()
    window = AttributeWindow(state, dataset_idx=0)
    try:
        window._lines_chk.setChecked(True)
        expected = {
            "-": _Qt.PenStyle.SolidLine,
            "--": _Qt.PenStyle.DashLine,
            ":": _Qt.PenStyle.DotLine,
            "-.": _Qt.PenStyle.DashDotLine,
        }
        for spec, style in expected.items():
            window._apply_plot_style(
                {"symbol": "star", "size": 7, "alpha": 200,
                 "line_style": spec, "line_width": 2.5}
            )
            curve, _scatter = window._series_items[0]
            pen = curve.opts["pen"]
            assert pen.style() == style and pen.widthF() == pytest.approx(2.5)

        saved = state.datasets[0].state["attribute_plot_state"]
        assert saved["line_style"] == "-." and saved["line_width"] == 2.5
    finally:
        window.close()


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
        window._add_dimension("Z")
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
        window._add_dimension("Z")
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


def test_c_mapping_hides_missing_values_and_expands_constant_ranges(_qt_app):
    from minflux_viewer.ui.attribute_window import AttributeWindow

    # Constant C data always occupies a real, documented range rather than
    # being silently sent to the LUT midpoint.  A bool keeps its full domain
    # even when every displayed row has the same value.
    cases = (
        (np.array([True, True]), (0.0, 1.0), [255, 255]),
        (np.array([0.0, 0.0]), (0.0, 1.0), [0, 0]),
        (np.array([2.5, 2.5]), (0.0, 2.5), [255, 255]),
        (np.array([-2.5, -2.5]), (-2.5, 0.0), [0, 0]),
    )
    for values, expected_levels, expected_bins in cases:
        bins, lo, hi = AttributeWindow._linear_color_bins(values)
        np.testing.assert_array_equal(bins, expected_bins)
        assert (lo, hi) == expected_levels

    window = AttributeWindow(_state(), dataset_idx=0)
    try:
        _brushes, rgba, lo, hi = window._mapped_colors(
            np.array([np.nan, 2.5, 2.5])
        )
        assert (lo, hi) == (0.0, 2.5)
        assert rgba[0, 3] == 0.0
        assert np.all(rgba[1:, 3] > 0.0)

        # The same C visibility mask creates a gap in a connected 2-D trace.
        window._clear_series()
        window._add_series(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 2.0]),
            (0, 0, 0),
            use_lines=True,
            line_visible_mask=np.array([True, False, True]),
        )
        line_x, line_y = window._series_items[-1][0].getData()
        assert np.isnan(line_x[1]) and np.isnan(line_y[1])

        # The artificial range reaches the bar too, so it has a complete
        # ruler instead of an empty constant-value colorbar.
        window._add_dimension("Z")
        window._add_dimension("C")
        window._last_color_values = np.array([2.5, 2.5])
        window._update_colorbar()
        assert (window._colorbar._lo, window._colorbar._hi) == (0.0, 2.5)
        assert not window._colorbar.isHidden()

        window._last_color_values = np.array([np.nan, np.inf])
        window._update_colorbar()
        assert window._colorbar.isHidden()
    finally:
        window.close()


def test_3d_view_keeps_top_controls_and_uses_unit_cube(monkeypatch, _qt_app):
    from minflux_viewer.ui.attribute_window import AttributeWindow

    window = AttributeWindow(_state(), dataset_idx=0)
    fake_view = QWidget()
    window._stack.addWidget(fake_view)
    try:
        window._add_dimension("Z")
        window._3d_view = fake_view
        monkeypatch.setattr(window, "_ensure_3d_built", lambda: True)
        monkeypatch.setattr(window, "_draw", lambda: None)
        window._set_view_mode("3D")

        assert window._stack.currentWidget() is fake_view
        assert window._x_combo.isVisibleTo(window)
        assert window._y_combo.isVisibleTo(window)
        assert window._z_combo.isVisibleTo(window)
        assert (window._axis_a_label.text(), window._axis_b_label.text()) == ("X:", "Y:")

        raw = np.array([
            [0.0, 1_000.0, -2.0],
            [5.0, 2_000.0, 0.0],
            [10.0, 3_000.0, 2.0],
        ])
        positions, mins, maxs = AttributeWindow._normalize_3d_positions(raw)
        np.testing.assert_allclose(positions[0], [-0.5, -0.5, -0.5])
        np.testing.assert_allclose(positions[-1], [0.5, 0.5, 0.5])
        np.testing.assert_allclose(mins, [0.0, 1_000.0, -2.0])
        np.testing.assert_allclose(maxs, [10.0, 3_000.0, 2.0])

        # Pinned to that extent, a later, narrower set keeps its place in the
        # cube instead of being re-stretched across it.
        narrower = np.array([[5.0, 2_000.0, 0.0], [10.0, 3_000.0, 2.0]])
        pinned, _mins, _maxs = AttributeWindow._normalize_3d_positions(
            narrower, (mins, maxs)
        )
        np.testing.assert_allclose(pinned[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(pinned[-1], [0.5, 0.5, 0.5])
    finally:
        window.close()


def test_changing_an_axis_attribute_refits_only_that_axis(_qt_app):
    """A new attribute on an axis is a new value range, so that axis re-fits.

    `efo` lives in the tens of thousands where `cfr` is 0..1, so the old range
    shows nothing — but the *other* axis is still the window the user chose,
    and C only recolours the same points, so neither of those moves.
    """
    from minflux_viewer.ui.attribute_window import AttributeWindow

    window = AttributeWindow(_state(), dataset_idx=0)
    try:
        window.show()
        _qt_app.processEvents()
        window._set_dimension_attribute("X", "idx")
        window._set_dimension_attribute("Y", "efo")
        window._view_box.setRange(
            xRange=(2.0, 6.0), yRange=(2_000.0, 5_000.0), padding=0.0
        )
        _qt_app.processEvents()
        x_before = tuple(window._view_box.viewRange()[0])

        window._set_dimension_attribute("Y", "cfr")
        _qt_app.processEvents()
        (x0, x1), (y0, y1) = window._view_box.viewRange()
        assert (x0, x1) == pytest.approx(x_before)      # the X window is kept
        assert y0 <= 0.2 and 0.9 <= y1 <= 2.0           # ...and Y found cfr

        window._set_dimension_attribute("X", "efo")
        _qt_app.processEvents()
        (x0, x1), (y0, y1) = window._view_box.viewRange()
        assert x0 <= 1_000.0 and x1 >= 8_000.0          # X found efo
        assert y0 <= 0.2 and 0.9 <= y1 <= 2.0           # ...and Y stayed

        # C is colour, not geometry: nothing moves.
        window._add_dimension("C", "cfr")
        _qt_app.processEvents()
        before = [tuple(pair) for pair in window._view_box.viewRange()]
        window._set_dimension_attribute("C", "tid")
        _qt_app.processEvents()
        assert [
            tuple(pair) for pair in window._view_box.viewRange()
        ] == pytest.approx(before)
    finally:
        window.close()


def test_3d_toggles_keep_the_cube_so_points_stay_put(_qt_app):
    """In 3-D the toggles must not re-scale the cube.

    3-D coordinates are normalized per axis from the data, so a changed extent
    slides every point to a new place in the box — the 3-D form of the view
    jumping. The extent is pinned across Valid only / Lines / Filtered only,
    so points appear and disappear where they are.
    """
    from minflux_viewer.ui.attribute_window import AttributeWindow

    window = AttributeWindow(_state(), dataset_idx=0)
    try:
        window._add_dimension("Z", "cfr")
        window._set_view_mode("3D")
        _qt_app.processEvents()
        if window._3d_view is None:
            pytest.skip("OpenGL unavailable")
        before = window._3d_extent
        assert before is not None

        # A toggle that changes what is plotted keeps the cube...
        window._filter_chk.setChecked(not window._filter_chk.isChecked())
        _qt_app.processEvents()
        np.testing.assert_allclose(window._3d_extent[0], before[0])
        np.testing.assert_allclose(window._3d_extent[1], before[1])

        # ...while a change of attribute is a different plot and re-fits.
        window._set_dimension_attribute("Z", "efo")
        _qt_app.processEvents()
        assert not np.allclose(window._3d_extent[1], before[1])
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
    window._add_dimension("Z")
    window._set_dimension_attribute("Z", "cfr")
    window._set_view_mode("XZ")
    window._add_dimension("C")
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
        assert (restored._has_z, restored._has_c) == (True, True)
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
            "X:", "Y:",
        )
        assert restored._z_combo.currentText() == "cfr"
        assert restored._c_combo.currentText() == "tid"
    finally:
        restored.close()
