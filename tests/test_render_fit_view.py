"""Opening a render window must show the *full* data extent, not clip a wide
dataset's left/right.

The initial fit runs during construction, before the window is on screen, so the
ViewBox reports a placeholder pixel size; PyQtGraph's aspect-lock re-enforcement on
the first real resize could then clip the X extent of a wide dataset (full height
shown, left/right cut off). RenderWindow.showEvent re-fits once on first show to
guarantee full coverage.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest


@pytest.fixture
def _qt_app():
    pytest.importorskip("PyQt6")
    pytest.importorskip("pyqtgraph")
    if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
        pytest.skip("No display available for Qt tests")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


def _wide_dataset():
    """A dataset far wider than tall (16 µm × 6 µm) — the aspect that triggered the
    clipping on a square-ish default window."""
    from minflux_viewer.core.dataset import build_localization_dataset
    rng = np.random.default_rng(0)
    x = rng.uniform(-8000.0, 8000.0, 4000)      # 16000 nm wide
    y = rng.uniform(-3000.0, 3000.0, 4000)      # 6000 nm tall
    z = np.zeros_like(x)
    return build_localization_dataset(name="wide", x_nm=x, y_nm=y, z_nm=z)


def _pump(app, n=5):
    for _ in range(n):
        app.processEvents()


def test_wide_dataset_shows_full_x_on_open(_qt_app):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.render_window import RenderWindow

    ds = _wide_dataset()
    x0d, x1d = float(ds.loc_nm[:, 0].min()), float(ds.loc_nm[:, 0].max())

    state = AppState()
    state.prefs.setdefault("data", {}).update({"show_data_info": False, "show_render": False})
    state.add_dataset(ds)

    win = RenderWindow(state, dataset_idx=0)
    try:
        win.resize(720, 720)                    # square-ish → the failing aspect
        win.show()
        _pump(_qt_app)                           # let the deferred first-show fit run

        (vx0, vx1), _ = win._view_box.viewRange()
        assert win._did_initial_fit is True
        # The full data width must be visible (previously the view clipped to the
        # centre, showing only ~2/3 of the X extent).
        assert vx0 <= x0d + 1.0, f"left clipped: view x0={vx0} > data x0={x0d}"
        assert vx1 >= x1d - 1.0, f"right clipped: view x1={vx1} < data x1={x1d}"
    finally:
        win.close()


def test_reshow_does_not_reset_user_zoom(_qt_app):
    """The one-shot fit must not fire again when an existing window is re-raised, or
    it would throw away the user's zoom/pan."""
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.render_window import RenderWindow

    ds = _wide_dataset()
    state = AppState()
    state.prefs.setdefault("data", {}).update({"show_data_info": False, "show_render": False})
    state.add_dataset(ds)

    win = RenderWindow(state, dataset_idx=0)
    try:
        win.resize(720, 720)
        win.show()
        _pump(_qt_app)

        # user zooms into a small region
        win._suppress_zoom_limit = True
        win._view_box.setRange(xRange=(-500.0, 500.0), yRange=(-500.0, 500.0), padding=0)
        win._suppress_zoom_limit = False
        _pump(_qt_app, 2)
        (zx0, zx1), _ = win._view_box.viewRange()
        zoom_w = zx1 - zx0

        win.hide()
        win.show()                               # simulates _show_render raising it
        _pump(_qt_app)

        (x0, x1), _ = win._view_box.viewRange()
        assert abs((x1 - x0) - zoom_w) < 1.0, "re-show reset the user's zoom"
    finally:
        win.close()
