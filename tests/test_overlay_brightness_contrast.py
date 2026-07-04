"""Shift+C on a non-primary overlay channel must reuse the overlay render window,
not spawn a duplicate one (the render window is keyed in `_render_windows` under
only its anchor dataset, but displays every channel)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest


@pytest.fixture
def _qt_app():
    pytest.importorskip("PyQt6")
    if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
        pytest.skip("No display available for Qt tests")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


def _ds(name):
    from minflux_viewer.core.dataset import build_localization_dataset
    p = np.random.default_rng(0).normal(0, 50, (30, 3))
    return build_localization_dataset(name=name, x_nm=p[:, 0], y_nm=p[:, 1], z_nm=p[:, 2])


def _stub_render(_qt_app):
    """Minimal QWidget stand-in for an overlay RenderWindow (channels 0 and 1)."""
    from PyQt6.QtWidgets import QWidget

    class _StubRender(QWidget):
        def __init__(self):
            super().__init__()
            self._channels = [{"dataset_idx": 0}, {"dataset_idx": 1}]
            self.bc_calls = 0
            self.refresh_calls = 0

        def _show_brightness_contrast(self):
            self.bc_calls += 1

        def _refresh_from_dataset(self):
            self.refresh_calls += 1

    return _StubRender()


def test_shift_c_on_overlay_channel_reuses_window(_qt_app):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.main_window import MainWindow

    state = AppState()
    state.prefs.setdefault("data", {}).update({"show_data_info": False, "show_render": False})
    win = MainWindow(state)
    try:
        state.add_dataset(_ds("ch0"))                 # idx 0 — overlay anchor
        state.add_dataset(_ds("ch1"))                 # idx 1 — green channel

        stub = _stub_render(_qt_app)
        win._render_windows[0] = stub                 # one overlay window, keyed by anchor

        # Found for the anchor AND for the non-primary channel (the fix).
        assert win._render_window_for_dataset(0) is stub
        assert win._render_window_for_dataset(1) is stub
        assert win._render_window_for_dataset(2) is None

        # Shift+C with channel 1 active reuses the overlay window, no duplicate.
        state.set_active(1)
        win._show_brightness_contrast()
        assert stub.bc_calls == 1
        assert 1 not in win._render_windows
        assert set(win._render_windows) == {0}
    finally:
        win.close()
        _qt_app.processEvents()


def test_show_render_on_overlay_channel_reuses_window(_qt_app):
    """`_show_render(non_anchor_channel)` (e.g. segmentation adding ROIs on the
    2nd channel) must reuse the overlay window, not spawn a duplicate."""
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.main_window import MainWindow

    state = AppState()
    state.prefs.setdefault("data", {}).update({"show_data_info": False, "show_render": False})
    win = MainWindow(state)
    try:
        state.add_dataset(_ds("ch0"))                 # idx 0 — overlay anchor
        state.add_dataset(_ds("ch1"))                 # idx 1 — green channel

        stub = _stub_render(_qt_app)
        win._render_windows[0] = stub                 # overlay window, keyed by anchor

        state.set_active(1)
        returned = win._show_render(1)                # segmentation-style call
        assert returned is stub                       # reused, not duplicated
        assert stub.refresh_calls == 1                # refreshed the existing window
        assert set(win._render_windows) == {0}        # no new key 1 created
    finally:
        win.close()
        _qt_app.processEvents()
