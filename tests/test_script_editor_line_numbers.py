"""Regression tests for the Script Editor's line-number gutter."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qt_app():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def editor_window(qt_app):
    from minflux_viewer import scripting
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.script_editor_window import ScriptEditorWindow

    state = AppState()
    state.save_prefs = lambda: None
    state.mfv = scripting.create_facade(state)
    window = ScriptEditorWindow(state)
    window.show()
    qt_app.processEvents()
    yield window
    window.force_close()


def test_line_number_gutter_grows_with_the_document(editor_window, qt_app):
    editor = editor_window.editor
    editor.setPlainText("pass")
    one_digit_width = editor.line_number_area_width()

    editor.setPlainText("\n".join("pass" for _ in range(120)))
    qt_app.processEvents()

    assert editor.blockCount() == 120
    assert editor.line_number_area_width() > one_digit_width
    assert editor._line_number_area.width() == editor.line_number_area_width()
    assert editor.viewport().geometry().left() >= editor.line_number_area_width()


def test_line_number_gutter_tracks_editor_resize(editor_window, qt_app):
    editor = editor_window.editor
    editor.resize(640, 360)
    qt_app.processEvents()

    gutter = editor._line_number_area
    assert gutter.geometry().top() == editor.contentsRect().top()
    assert gutter.height() == editor.contentsRect().height()
    assert gutter.width() == editor.line_number_area_width()


def test_open_dialog_defaults_to_project_scripts(editor_window, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    selected_directory = None

    def fake_get_open_file_name(_parent, _caption, directory, _file_filter):
        nonlocal selected_directory
        selected_directory = directory
        return "", ""

    monkeypatch.setattr(QFileDialog, "getOpenFileName", fake_get_open_file_name)
    editor_window._open_script()

    expected = Path(__file__).resolve().parents[1] / "scripts"
    assert Path(selected_directory) == expected
