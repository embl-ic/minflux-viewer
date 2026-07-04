"""Log window replays history and is not auto-shown by logging."""

from __future__ import annotations

import sys

import pytest


@pytest.fixture
def _app():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


def test_log_window_replays_history_and_updates_live(_app):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.log_window import LogWindow

    state = AppState()
    state.log("loading dataset one")
    state.log("a warning here", "WARN")

    # Opening the window later must show everything logged while it was closed —
    # the window is no longer auto-created on the first message.
    win = LogWindow(state)
    try:
        txt = win._text.toPlainText()
        assert "loading dataset one" in txt
        assert "a warning here" in txt

        state.log("a live message")                 # live updates still work
        assert "a live message" in win._text.toPlainText()
    finally:
        win.close()
        _app.processEvents()
