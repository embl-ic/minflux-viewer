"""Close-window shortcut scheme:
- W              → close current window   (existing `close_window`)
- Shift+W        → close all datasets      (actionCloseAll)
- Ctrl+Shift+W   → close all windows       (actionCloseAllWindows) — keeps Log/Console
- Ctrl+W         → close active dataset     (actionClose)
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QWidget

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.ui import modeless
from minflux_viewer.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


def _closed(widget) -> bool:
    try:
        return not widget.isVisible()
    except RuntimeError:              # C++ object deleted (WA_DeleteOnClose) → closed
        return True


def test_close_action_shortcuts(_app):
    w = MainWindow(AppState())
    assert w.actionClose.shortcut().toString() == "Ctrl+W"
    assert w.actionCloseAll.shortcut().toString() == "Shift+W"
    assert w.actionCloseAllWindows.shortcut().toString() == "Ctrl+Shift+W"
    texts = [a.text() for a in w._ui.menuFile.actions()]
    assert "Close All Windows" in texts
    w.close()


def test_close_all_windows_keeps_log_and_console_closes_dialogs(_app):
    w = MainWindow(AppState())
    w._ensure_log_window(show=True)
    assert w._log_win.isVisible()

    # A modeless plugin/analysis dialog (e.g. Particle Average).
    dlg = QWidget()
    dlg.setWindowTitle("FakeParticleAverage")
    modeless.show_modeless(dlg, w)
    dlg.show()
    assert dlg.isVisible()

    for i in range(3):
        w._state.add_dataset(build_localization_dataset(
            name=f"d{i}", x_nm=np.arange(10.0), y_nm=np.arange(10.0)))
    assert len(w._state.datasets) == 3

    w._close_all_windows()
    _app.processEvents()

    assert len(w._state.datasets) == 0            # all datasets closed
    assert w._log_win.isVisible()                 # Log kept
    assert _closed(dlg)                            # plugin dialog closed
    w.close()


def test_close_all_datasets_leaves_dialogs(_app):
    """Shift+W closes datasets but NOT standalone dialogs (that's Ctrl+Shift+W)."""
    w = MainWindow(AppState())
    dlg = QWidget()
    dlg.setWindowTitle("FakeDialog")
    modeless.show_modeless(dlg, w)
    dlg.show()
    w._state.add_dataset(build_localization_dataset(
        name="d", x_nm=np.arange(10.0), y_nm=np.arange(10.0)))

    w._close_all_datasets()
    _app.processEvents()

    assert len(w._state.datasets) == 0
    assert not _closed(dlg)                        # dialog still open
    w.close()
