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


def test_file_menu_save_as_layout(_app):
    w = MainWindow(AppState())
    actions = [a for a in w._ui.menuFile.actions() if not a.isSeparator()]
    texts = [a.text() for a in actions]

    assert texts[:6] == [
        "Open...",
        "Open Sample Data",
        "Open Recent",
        "Close Dataset",
        "Close All Datasets",
        "Close All Windows",
    ]
    assert texts[6:8] == ["Save...", "Save As"]

    save_as = w.menuSaveAs
    # This application's own format plus the MINFLUX defaults. Picasso HDF5 is
    # deliberately absent: the writer is kept and callable, but application-
    # specific formats are not offered here (BACKLOG.md > Nice to have).
    assert [a.text() for a in save_as.actions()] == [
        "MINFLUX data formats (.mat; .npy; .json)",
        "MINFLUX .msr file (experimental)",
        "Custom table (.csv)...",
        "Zarr (.zarr v2) format",
        "Zarr (.zarr.zip v2) single file",
        "OME-TIFF...",
        "OME-NGFF 0.5 / Zarr v3...",
    ]
    assert hasattr(w, "actionSaveAsHdf5"), "the Picasso writer stays reachable"
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
