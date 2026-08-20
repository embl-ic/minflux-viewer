"""
The toolbar LUT button targets the plot window the user is working in — render,
scatter, or Attribute Plot. Clicking the toolbar activates the main window, so
the button relies on the last-*focused* plot window (tracked via
QApplication.focusChanged).

Kept to a single MainWindow to limit pyqtgraph teardown churn on Windows.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")


def _ds():
    from minflux_viewer.core.dataset import build_localization_dataset
    rng = np.random.default_rng(0)
    return build_localization_dataset(
        name="d",
        x_nm=rng.uniform(0, 1000, 500), y_nm=rng.uniform(0, 1000, 500),
        tid=(np.arange(500) // 2).astype(float),
        attrs={"efo": rng.normal(8e4, 1e3, 500)})


def test_lut_button_targets_focused_plot_window(qtbot):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.main_window import MainWindow

    state = AppState()
    state.add_dataset(_ds())
    mw = MainWindow(state)
    qtbot.addWidget(mw)
    for w in list(mw._render_windows.values()):              # clean slate
        try:
            w.close()
        except Exception:
            pass
    mw._render_windows.clear()

    sw = mw._show_scatter(0)
    assert sw is not None

    # (1) focusing the scatter records it as the last-active plot window
    mw._on_focus_changed(None, sw)
    assert mw._last_active_plot_window is sw

    # (2) the LUT button opens the scatter's LUT — and does NOT spawn a render
    #     (the old bug: the fallback only ever found/created a render window)
    calls: list[str] = []
    sw.open_lut_dialog = lambda: calls.append("scatter")
    mw._show_lut()
    assert calls == ["scatter"]
    assert 0 not in mw._render_windows

    # (3) with both a render and a scatter open, it follows the last-focused one
    rw = mw._show_render(0)
    rw.open_lut_dialog = lambda: calls.append("render")
    mw._on_focus_changed(None, rw)
    calls.clear()
    mw._show_lut()
    assert calls == ["render"]

    mw._on_focus_changed(None, sw)
    calls.clear()
    mw._show_lut()
    assert calls == ["scatter"]

    # (4) a focused Attribute Plot is also a first-class LUT owner (its real
    #     implementation applies the editor to the fourth/C dimension).
    aw = mw._show_attr_plot(0)
    assert aw is not None
    aw.open_lut_dialog = lambda: calls.append("attribute")
    mw._on_focus_changed(None, aw)
    calls.clear()
    mw._show_lut()
    assert calls == ["attribute"]

# NB: global-shortcut behaviour (Preferences shortcuts firing from any focused
# window, via the app-wide event filter) is covered by tests/test_global_shortcuts.py.
