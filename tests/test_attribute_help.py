"""Every attribute picker explains its entries, from one shared source."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QContextMenuEvent
from PyQt6.QtWidgets import QApplication, QComboBox, QMenu

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.attributes import attribute_description
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.ui.attribute_help import (
    apply_attribute_menu_tooltips,
    apply_attribute_tooltips,
)


@pytest.fixture
def _qt_app():
    if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
        pytest.skip("No display available for Qt tests")
    return QApplication.instance() or QApplication(sys.argv)


def _state() -> AppState:
    rng = np.random.default_rng(0)
    n = 40
    state = AppState()
    state.add_dataset(
        build_localization_dataset(
            name="help",
            x_nm=rng.normal(0.0, 100.0, n),
            y_nm=rng.normal(0.0, 100.0, n),
            z_nm=np.zeros(n),
            tid=np.repeat(np.arange(1, n // 4 + 1), 4).astype(float),
            attrs={
                "efo": rng.normal(5e4, 9e3, n),
                "cfr": rng.uniform(0.0, 1.0, n),
                "dcr": rng.uniform(0.0, 1.0, n),
            },
        )
    )
    return state


def _tips(combo: QComboBox) -> list[str]:
    return [
        combo.itemData(i, Qt.ItemDataRole.ToolTipRole) or ""
        for i in range(combo.count())
    ]


def test_helper_describes_every_entry_and_follows_the_selection(_qt_app):
    combo = QComboBox()
    combo.addItems(["efo", "cfr"])
    apply_attribute_tooltips(combo)

    assert _tips(combo) == [attribute_description("efo"), attribute_description("cfr")]
    assert combo.toolTip() == attribute_description("efo")
    combo.setCurrentText("cfr")
    assert combo.toolTip() == attribute_description("cfr")

    # Repopulating must not stack a second connection on the same combo.
    apply_attribute_tooltips(combo)
    apply_attribute_tooltips(combo)
    combo.setCurrentText("efo")
    assert combo.toolTip() == attribute_description("efo")


def test_menu_helper_leaves_non_attribute_entries_alone(_qt_app):
    menu = QMenu()
    menu.addAction("efo")
    menu.addAction("Colormap")            # not an attribute
    apply_attribute_menu_tooltips(menu, ["efo"])

    assert menu.toolTipsVisible()
    assert menu.actions()[0].toolTip() == attribute_description("efo")
    # Untouched, so Qt keeps its default (the action's own text). Without the
    # guard it would read "Unknown parameter…", which is worse than nothing.
    assert menu.actions()[1].toolTip() == "Colormap"
    assert "Unknown parameter" not in menu.actions()[1].toolTip()


def test_every_attribute_picker_carries_the_help(_qt_app):
    """Attribute Plot, Histogram, Scatter, Filter and the menus that list names."""
    from minflux_viewer.ui.attribute_window import AttributeWindow
    from minflux_viewer.ui.filter_dialog import FilterDialog
    from minflux_viewer.ui.histogram_window import HistogramWindow
    from minflux_viewer.ui.scatter_window import ScatterWindow

    state = _state()
    windows = []
    try:
        plot = AttributeWindow(state, dataset_idx=0)
        windows.append(plot)
        plot._add_dimension("C", "cfr")
        for combo in (plot._x_combo, plot._y_combo, plot._c_combo):
            assert all(_tips(combo)) and combo.toolTip()

        histogram = HistogramWindow(state, dataset_idx=0)
        windows.append(histogram)
        assert all(_tips(histogram._attr_combo))

        scatter = ScatterWindow(state, dataset_idx=0)
        windows.append(scatter)
        assert all(_tips(scatter._cbar_combo))

        filters = FilterDialog(state, dataset_idx=0)
        windows.append(filters)
        filters._add_row()
        combo = filters._table.cellWidget(0, 1)
        assert isinstance(combo, QComboBox)
        assert all(_tips(combo))

        # ...and the menus that list the same names.
        captured: list[QMenu] = []
        original_exec = QMenu.exec
        QMenu.exec = lambda self, *args: captured.append(self)
        try:
            scatter._show_context_menu(QPoint(5, 5))
            color_menu = next(
                action.menu() for action in captured[0].actions()
                if action.text() == "Color by"
            )
            assert color_menu.toolTipsVisible()
            assert color_menu.actions()[0].toolTip()

            captured.clear()
            plot._show_context_menu(QPoint(5, 5))
            add_z = next(
                action.menu() for action in captured[0].actions()
                if action.text().startswith("add new attribute as Z")
            )
            assert add_z.toolTipsVisible() and add_z.actions()[0].toolTip()

            captured.clear()
            bar = scatter._colorbar
            bar.contextMenuEvent(
                QContextMenuEvent(
                    QContextMenuEvent.Reason.Mouse,
                    QPoint(4, 4),
                    bar.mapToGlobal(QPoint(4, 4)),
                )
            )
            attribute_menu = next(
                action.menu() for action in captured[0].actions()
                if action.text() == "Attribute:"
            )
            assert attribute_menu.toolTipsVisible()
            assert attribute_menu.actions()[0].toolTip()
        finally:
            QMenu.exec = original_exec
    finally:
        for window in windows:
            window.close()
