"""Custom-colormap stop editing and PyQtGraph menu integration."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtWidgets import QApplication, QDialog, QLabel


def _dialog(qtbot):
    from minflux_viewer.ui.custom_colormap_dialog import CustomColormapDialog

    dialog = CustomColormapDialog()
    qtbot.addWidget(dialog)
    return dialog


def _positions(dialog) -> list[float]:
    return [
        float(position)
        for _tick, position in dialog._gradient.item.listTicks()
    ]


def test_editor_has_only_the_double_click_add_affordance(qtbot):
    dialog = _dialog(qtbot)

    assert not hasattr(dialog, "_add_stop_button")
    assert "Add color stop" not in [
        action.text() for action in dialog._gradient.item.menu.actions()
    ]
    help_text = " ".join(label.text() for label in dialog.findChildren(QLabel))
    assert "Double-click in empty stop area" in help_text


def test_empty_stop_area_requires_double_click(qtbot):
    dialog = _dialog(qtbot)
    dialog.show()
    QApplication.processEvents()
    item = dialog._gradient.item
    scene_position = item.mapToScene(QPointF(0.3 * item.length, 5.0))
    viewport_position = dialog._gradient.mapFromScene(scene_position)

    before = _positions(dialog)
    qtbot.mouseClick(
        dialog._gradient.viewport(), Qt.MouseButton.LeftButton, pos=viewport_position
    )
    assert _positions(dialog) == pytest.approx(before)

    qtbot.mouseDClick(
        dialog._gradient.viewport(), Qt.MouseButton.LeftButton, pos=viewport_position
    )
    positions = _positions(dialog)
    assert len(positions) == len(before) + 1
    assert any(position == pytest.approx(0.3, abs=0.01) for position in positions)


@pytest.mark.parametrize(
    ("name", "expected_stops"),
    [
        ("viridis", 3),
        ("inferno", 4),
        ("turbo", 5),
        ("CET-L1", 3),
        ("CET-CBL1", 4),
        ("CET-C1", 5),
    ],
)
def test_dense_local_colormap_uses_adaptive_editable_stops(
    qtbot, name, expected_stops
):
    from pyqtgraph import colormap

    dialog = _dialog(qtbot)
    dialog._gradient.item.menu.sigColorMapTriggered.emit(colormap.get(name))

    state = dialog._gradient.saveState()
    positions = _positions(dialog)
    assert state["ticksVisible"] is True
    assert dialog._gradient.item.allowAdd is False
    assert len(positions) == expected_stops
    assert positions[0] == pytest.approx(0.0)
    assert positions[-1] == pytest.approx(1.0)


def test_tick_context_menu_says_remove(qtbot):
    from pyqtgraph import Point

    dialog = _dialog(qtbot)
    tick = dialog._gradient.item.listTicks()[1][0]

    class _ContextEvent:
        @staticmethod
        def screenPos():
            return Point(0.0, 0.0)

    dialog._gradient.item.raiseTickContextMenu(tick, _ContextEvent())
    assert dialog._gradient.item.tickMenu.removeAct.text() == "Remove"
    dialog._gradient.item.tickMenu.close()


def test_simplified_local_colormap_can_be_saved(qtbot):
    from pyqtgraph import colormap

    dialog = _dialog(qtbot)
    dialog._name_edit.setText("CET local editable regression")
    dialog._on_gradient_colormap_selected(colormap.get("CET-D1"))

    dialog._accept_if_valid()

    assert dialog.result() == QDialog.DialogCode.Accepted
    stops = dialog.result_stops()
    assert 3 <= len(stops) <= 5
    assert stops[0][0] == pytest.approx(0.0)
    assert stops[-1][0] == pytest.approx(1.0)
