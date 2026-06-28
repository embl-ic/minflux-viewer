"""
Tests for the Brightness/Contrast dialog over overlay (multi-channel) renders:

* B/C reads and edits only the **active** channel's contrast levels — switching
  the active channel re-targets it, and editing one channel never disturbs
  another.
* The histogram is coloured with the active channel's viewer LUT.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from minflux_viewer.core.dataset import AttrStore, DataProp, FileInfo, MinfluxDataset


def _make_ds(name: str, seed: int, n: int = 30, n_traces: int = 3) -> MinfluxDataset:
    per = n // n_traces
    tid = np.repeat(np.arange(n_traces), per)
    ti = np.column_stack([
        np.arange(n_traces) * per,
        np.arange(n_traces) * per + per - 1,
    ])
    prop = DataProp(
        num_loc=n, num_itr=1, num_dim=2, num_traces=n_traces,
        trace_idx=ti, num_loc_per_trace=np.full(n_traces, per),
        attr_names=["loc_x", "loc_y", "loc_z", "tid", "ftr"],
    )
    rng = np.random.default_rng(seed)
    attrs = AttrStore({
        "loc_x": rng.uniform(0, 1e-6, n),
        "loc_y": rng.uniform(0, 1e-6, n),
        "loc_z": np.zeros(n),
        "tid": tid.astype(float),
        "ftr": np.ones(n, dtype=bool),
    })
    return MinfluxDataset(file=FileInfo(name=name, folder="/tmp"), prop=prop, attr=attrs)


@pytest.fixture
def _qt_app():
    pytest.importorskip("PyQt6")
    if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
        pytest.skip("No display available for Qt tests")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


def _seed_scalar_tile(win, n_channels: int) -> None:
    """Give the window a ready scalar tile so B/C reads real pixels and
    recomposes synchronously (no async render thread is scheduled)."""
    rng = np.random.default_rng(0)
    win._last_scalar_tile = rng.random((n_channels, 16, 16)).astype(np.float32)
    win._last_tile_geometry = (0.0, 100.0, 0.0, 100.0)
    win._last_px_nm = 100.0 / 16.0


def _teardown(win, app) -> None:
    from PyQt6.QtCore import QThreadPool
    try:
        win._scheduler.cancel()
    except Exception:
        pass
    win.close()
    app.processEvents()
    QThreadPool.globalInstance().waitForDone(3000)


def _overlay_state():
    from minflux_viewer.core.app_state import AppState
    state = AppState()
    a = _make_ds("chA.mat", 1)
    a.state["overlay_id"] = "grp"; a.state["overlay_index"] = 1
    b = _make_ds("chB.mat", 2)
    b.state["overlay_id"] = "grp"; b.state["overlay_index"] = 2
    state.add_dataset(a)
    state.add_dataset(b)
    return state


def test_bc_edits_only_active_channel(_qt_app):
    from minflux_viewer.ui.render_window import RenderWindow

    state = _overlay_state()
    win = RenderWindow(state, dataset_idx=0)
    try:
        _seed_scalar_tile(win, len(win._channels))
        assert len(win._channels) == 2          # both channels overlaid
        state.set_active(0)                      # deterministic active = chA
        assert win._active_channel_index() == 0

        win._show_brightness_contrast()
        win._on_levels_changed(5.0, 50.0)
        assert win._channels[0]["levels"] == (5.0, 50.0)
        assert win._channels[1]["levels"] is None   # other channel untouched

        # Switch active channel → B/C re-targets it (via active_changed wiring).
        state.set_active(1)
        assert win._active_channel_index() == 1
        win._on_levels_changed(7.0, 77.0)
        assert win._channels[0]["levels"] == (5.0, 50.0)   # chA still unchanged
        assert win._channels[1]["levels"] == (7.0, 77.0)

        # Representative histogram colour (same window, no extra RenderWindow):
        # solid channel → its pure colour; gradient colormap → one clamped mid
        # colour (never a per-bin gradient).
        win._channels[1]["lut"] = "Red"
        assert win._histogram_bar_color() == (1.0, 0.0, 0.0)
        win._channels[1]["lut"] = "hot"
        rgb = win._histogram_bar_color()
        assert _lum(rgb) <= 0.72 + 1e-6          # readable on white
        assert rgb[0] > rgb[2]                   # warm: red over blue
        assert win._bc_dialog._histogram._bar_rgb is not None  # single colour, not a LUT
    finally:
        _teardown(win, _qt_app)


def _lum(rgb) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def test_luminance_clamp_keeps_dark_darkens_bright():
    """Pure logic (no Qt): readable colours pass through, near-white ones darken."""
    from minflux_viewer.ui.render_window import _luminance_clamped

    assert _luminance_clamped((1.0, 0.0, 0.0)) == (1.0, 0.0, 0.0)   # red unchanged
    assert _luminance_clamped((0.0, 0.0, 1.0)) == (0.0, 0.0, 1.0)   # blue unchanged
    yellow = _luminance_clamped((1.0, 1.0, 0.0))                    # too bright
    assert _lum(yellow) <= 0.72 + 1e-6
    assert yellow[0] == pytest.approx(yellow[1]) and yellow[2] == 0.0   # hue kept
    gray = _luminance_clamped((1.0, 1.0, 1.0))                      # white → mid-gray
    assert _lum(gray) <= 0.72 + 1e-6
    assert gray[0] == pytest.approx(gray[1]) == pytest.approx(gray[2])


def test_bc_single_dataset_resolves_channel_zero(_qt_app):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.render_window import RenderWindow

    state = AppState()
    state.add_dataset(_make_ds("solo.mat", 3))
    win = RenderWindow(state, dataset_idx=0)
    try:
        _seed_scalar_tile(win, len(win._channels))
        assert win._active_channel_index() == 0
        win._show_brightness_contrast()
        win._on_levels_changed(2.0, 20.0)
        assert win._channels[0]["levels"] == (2.0, 20.0)
    finally:
        _teardown(win, _qt_app)
