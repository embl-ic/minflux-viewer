"""Preference-defined shortcuts must work from modeless child windows."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialog, QLineEdit, QVBoxLayout

from minflux_viewer.core.app_state import AppState
from minflux_viewer.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


def test_preference_shortcut_reaches_focused_modeless_dialog(_app):
    main = MainWindow(AppState())
    triggered: list[bool] = []

    # Replace the real render action so the test observes dispatch without
    # opening a data-dependent render dialog.
    render_action = QAction("Render", main)
    render_action.triggered.connect(lambda: triggered.append(True))
    main._shortcut_actions["render"] = render_action
    main._state.prefs["shortcuts"]["render"] = "Ctrl+R"

    dialog = QDialog(main)
    edit = QLineEdit(dialog)
    QVBoxLayout(dialog).addWidget(edit)
    dialog.show()
    edit.setFocus()
    _app.processEvents()

    QTest.keyClick(edit, Qt.Key.Key_R, Qt.KeyboardModifier.ControlModifier)
    _app.processEvents()

    assert triggered == [True]

    dialog.close()
    main.close()
    _app.processEvents()
