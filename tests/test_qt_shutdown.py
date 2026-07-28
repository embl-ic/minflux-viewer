"""Regression tests for native Qt teardown failures."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_main_window_qt_teardown_exits_cleanly():
    """Destroying MainWindow must not double-delete QApplication's style."""
    code = textwrap.dedent(
        """
        import ctypes
        import os

        if os.name == "nt":
            ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)

        from PyQt6.QtCore import QCoreApplication, QEvent
        from PyQt6.QtWidgets import QApplication
        from minflux_viewer.core.app_state import AppState
        from minflux_viewer.ui.main_window import MainWindow

        app = QApplication([])
        state = AppState()
        state.prefs.setdefault("file", {})["check_updates_on_startup"] = False
        state.save_prefs = lambda: None
        window = MainWindow(state)
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
        """
    )
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, (
        f"Qt subprocess exited with {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
