"""A script that stops itself is not a crash.

``sys.exit()`` / ``raise SystemExit`` is the documented way for a Python script
to stop early, and a cancelled parameter dialog is the usual reason. The Script
Editor caught it in a blanket ``except BaseException`` and printed a traceback,
so cancelling a dialog looked exactly like the script had failed.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qt_app():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def editor(qt_app):
    from minflux_viewer import scripting
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.script_editor_window import ScriptEditorWindow

    state = AppState()
    state.save_prefs = lambda: None
    state.mfv = scripting.create_facade(state)
    window = ScriptEditorWindow(state)
    yield window
    window.close()


def test_cancelling_a_script_reports_the_reason_without_a_traceback(editor):
    editor._execute('raise SystemExit("Cancelled.")')
    out = editor.output.toPlainText()

    assert ">>> Stopped: Cancelled." in out
    assert "Traceback" not in out
    assert "SystemExit" not in out


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("import sys; sys.exit()", ">>> Stopped"),
        ("import sys; sys.exit(0)", ">>> Stopped"),
        ("import sys; sys.exit(3)", ">>> Stopped with exit code 3"),
        ("raise SystemExit", ">>> Stopped"),
    ],
)
def test_every_self_stop_reads_as_a_stop(editor, code, expected):
    editor._clear_output()
    editor._execute(code)
    out = editor.output.toPlainText()

    assert expected in out
    assert "Traceback" not in out


def test_a_real_error_still_shows_its_traceback(editor):
    """The fix must not swallow genuine failures."""
    editor._clear_output()
    editor._execute("raise ValueError('boom')")
    out = editor.output.toPlainText()

    assert "Traceback" in out
    assert "ValueError" in out
    assert "boom" in out


def test_a_normal_script_still_reports_done(editor):
    editor._clear_output()
    editor._execute("x = 1 + 1")
    assert ">>> Done" in editor.output.toPlainText()


def test_only_a_non_zero_int_code_counts_as_failure():
    """Python's own rule: a string code is a message, not a status."""
    from minflux_viewer.ui.script_editor_window import (
        _is_failure_exit,
        _stopped_message,
    )

    assert not _is_failure_exit(SystemExit())
    assert not _is_failure_exit(SystemExit(0))
    assert not _is_failure_exit(SystemExit("Cancelled."))
    assert _is_failure_exit(SystemExit(1))
    assert _is_failure_exit(SystemExit(2))
    # A bool is an int subclass, but `sys.exit(True)` is not an exit code 1.
    assert not _is_failure_exit(SystemExit(True))

    assert _stopped_message(SystemExit()) == ">>> Stopped\n"
    assert _stopped_message(SystemExit("Cancelled.")) == ">>> Stopped: Cancelled.\n"
    assert _stopped_message(SystemExit(3)) == ">>> Stopped with exit code 3\n"


def test_the_process_folder_template_cancels_cleanly(editor):
    """The shipped template's cancel path is the case that reported this."""
    from pathlib import Path

    template = (
        Path(__file__).resolve().parents[1] / "scripts" / "process_folder_template.py"
    )
    if not template.is_file():                       # pragma: no cover
        pytest.skip("template not present in this checkout")

    editor._state.mfv.ui.ask = lambda fields, **kwargs: None   # user pressed Cancel
    editor._clear_output()
    editor._execute(template.read_text(encoding="utf-8"))
    out = editor.output.toPlainText()

    assert ">>> Stopped: Cancelled." in out
    assert "Traceback" not in out
