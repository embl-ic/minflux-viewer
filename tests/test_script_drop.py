"""A Python script dropped on the main window opens in the Script Editor.

Opening is reading, never running: a script with a syntax error, or one that
would fail, opens exactly like a working one. A dropped folder still opens only
its data, and one Script Editor holds one script, so a multi-script drop opens
the first and names the rest.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

from minflux_viewer.core.app_state import AppState
from minflux_viewer.ui.main_window import MainWindow

# Not valid Python, and it would leave a marker file behind if it were run.
_BROKEN = "this is not ( valid python\nopen({marker!r}, 'w').write('ran')\n"


@pytest.fixture(scope="module")
def _app(qapp):
    return qapp


@pytest.fixture
def win(_app):
    window = MainWindow(AppState())
    window._state.prefs.setdefault("data", {}).update(show_render=False, show_data_info=False)
    try:
        yield window
    finally:
        editor = window._script_editor_win
        if editor is not None:
            editor.force_close()
        window.close()


def _mime(*paths):
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return mime


def _drop(window, *paths):
    # ⚠ Keep the QMimeData referenced while the event lives: the event does not
    # own it, and a temporary is freed at once (an access violation on read).
    mime = _mime(*paths)
    event = QDropEvent(QPointF(5.0, 5.0), Qt.DropAction.CopyAction, mime,
                       Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    window.dropEvent(event)
    return event


def _messages(window):
    return [entry.get("message", "") for entry in window._state.log_history]


def _lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def test_a_dropped_script_opens_in_the_editor_and_is_not_run(win, tmp_path):
    marker = tmp_path / "ran.txt"
    script = tmp_path / "broken_analysis.py"
    text = _BROKEN.format(marker=str(marker))
    script.write_text(text, encoding="utf-8")

    mime = _mime(script)
    enter = QDragEnterEvent(QPoint(5, 5), Qt.DropAction.CopyAction, mime,
                            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    win.dragEnterEvent(enter)
    assert enter.isAccepted()

    _drop(win, script)

    editor = win._script_editor_win
    assert editor is not None and editor.isVisible()
    assert editor.editor.toPlainText() == text
    assert editor._path == script
    assert "broken_analysis.py" in editor.windowTitle()
    assert not editor.editor.document().isModified()
    assert not marker.exists(), "opening a script must never run it"
    assert ">>> Running script" not in editor.output.toPlainText()
    assert any("Opened script 'broken_analysis.py'" in m and "not run" in m
               for m in _messages(win))
    assert len(win._state.datasets) == 0


def test_dropping_another_script_replaces_the_first_like_open(win, tmp_path):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("a = 1\n", encoding="utf-8")
    second.write_text("b = 2\n", encoding="utf-8")

    _drop(win, first)
    editor = win._script_editor_win
    _drop(win, second)

    assert win._script_editor_win is editor, "one editor, reused"
    assert editor.editor.toPlainText() == "b = 2\n"
    assert editor._path == second


def test_a_multi_script_drop_opens_the_first_and_names_the_rest(win, tmp_path):
    paths = []
    for name in ("one.py", "two.py", "three.py"):
        path = tmp_path / name
        path.write_text(f"# {name}\n", encoding="utf-8")
        paths.append(path)

    _drop(win, *paths)

    editor = win._script_editor_win
    assert editor.editor.toPlainText() == "# one.py\n"
    warnings = [m for m in _messages(win) if "one script at a time" in m]
    assert len(warnings) == 1
    assert "'one.py'" in warnings[0] and "'two.py'" in warnings[0] and "'three.py'" in warnings[0]
    opened = [m for m in _messages(win) if m.startswith("Opened script")]
    assert len(opened) == 1, "the Log must not claim the skipped scripts were opened"


def test_a_dropped_folder_opens_its_data_not_its_scripts(win, tmp_path):
    folder = tmp_path / "analysis"
    folder.mkdir()
    (folder / "helper.py").write_text("x = 1\n", encoding="utf-8")

    _drop(win, folder)

    assert win._script_editor_win is None
    assert any("contains no supported files" in m for m in _messages(win))


def test_a_script_dropped_on_a_dataset_row_opens_in_the_editor(win, tmp_path):
    from minflux_viewer.core.dataset import build_localization_dataset

    p = np.random.default_rng(0).normal(size=(16, 3)) * 100.0
    win._state.add_dataset(build_localization_dataset(
        name="ds0", x_nm=p[:, 0], y_nm=p[:, 1], z_nm=p[:, 2], source_version="simulation"))
    script = tmp_path / "row.py"
    script.write_text("print('row')\n", encoding="utf-8")

    assert win.drop_file_on_dataset(0, str(script)) is True

    assert win._script_editor_win.editor.toPlainText() == "print('row')\n"
    assert any("is a Python script" in m and "not used" in m for m in _messages(win))
    assert not any("is a data file" in m for m in _messages(win))


def test_an_unreadable_script_is_reported_and_the_editor_is_unchanged(
        win, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    shown = []
    monkeypatch.setattr(QMessageBox, "critical",
                        staticmethod(lambda *a, **k: shown.append(a[2])))
    good = tmp_path / "good.py"
    good.write_text("kept = True\n", encoding="utf-8")
    _drop(win, good)
    bad = tmp_path / "bad.py"
    bad.write_bytes(b"x = '\xff\xfe not utf-8'\n")       # no encoding declaration

    _drop(win, bad)

    editor = win._script_editor_win
    assert editor.editor.toPlainText() == "kept = True\n"
    assert editor._path == good
    assert shown, "a failed read is reported, not skipped"
    assert any("Open file failed" in m for m in _messages(win))


def test_an_encoding_declaration_is_honoured_and_kept_on_save(_app, tmp_path):
    from minflux_viewer import scripting
    from minflux_viewer.ui.tools.script_editor_window import ScriptEditorWindow

    state = AppState()
    state.save_prefs = lambda: None
    state.mfv = scripting.create_facade(state)
    editor = ScriptEditorWindow(state)
    try:
        latin = tmp_path / "latin.py"
        source = "# -*- coding: latin-1 -*-\nname = 'Müller'\n"
        latin.write_bytes(source.encode("latin-1"))
        editor.open_script_file(latin)
        assert editor.editor.toPlainText() == source
        editor._save_script()
        # Save writes the platform's line endings (unchanged behaviour); the
        # point here is the encoding, so compare the bytes line-ending-blind.
        assert _lf(latin.read_bytes()) == source.encode("latin-1")

        bom = tmp_path / "bom.py"
        bom.write_bytes("﻿value = 'é'\n".encode())
        editor.open_script_file(bom)
        assert editor.editor.toPlainText() == "value = 'é'\n"
        editor._save_script()
        assert _lf(bom.read_bytes()) == "﻿value = 'é'\n".encode()
    finally:
        editor.force_close()
