"""Verify whether B/C (per-channel `levels`) in an overlay render is independent:
changing one channel's contrast must not change another channel's contribution to
the composited RGBA."""

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


def _ds(name):
    from minflux_viewer.core.dataset import build_localization_dataset
    p = np.random.default_rng(0).normal(0, 50, (30, 3))
    d = build_localization_dataset(name=name, x_nm=p[:, 0], y_nm=p[:, 1], z_nm=p[:, 2])
    d.state["overlay_id"] = "grp"                 # same overlay group → one render window
    return d


def test_overlay_compose_is_per_channel(_qt_app):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.render_window import RenderWindow

    state = AppState()
    state.prefs.setdefault("data", {}).update({"show_data_info": False, "show_render": False})
    state.add_dataset(_ds("red"))                 # idx 0
    state.add_dataset(_ds("green"))               # idx 1
    win = RenderWindow(state, dataset_idx=0)
    try:
        assert len(win._channels) == 2
        for c, lut in zip(win._channels, ("Red", "Green")):
            c["lut"] = lut
            c["visible"] = True
            c["levels"] = (0.0, 10.0)
            c["transform"] = {"dx_nm": 0.0, "dy_nm": 0.0, "angle": 0.0}

        # Per-channel scalar tile: a red-only pixel and a green-only pixel.
        scalar = np.zeros((2, 4, 4), dtype=np.float32)
        scalar[0, 0, 0] = 5.0
        scalar[1, 3, 3] = 5.0
        win._last_tile_geometry = (0.0, 4.0, 0.0, 4.0)

        rgba1 = win._compose_rgba(scalar)
        red_before = rgba1[0, 0, :3].copy()
        green_before = rgba1[3, 3, :3].copy()
        assert red_before[0] > 0 and green_before[1] > 0     # sanity: colours present

        # Brighten ONLY the green channel's contrast.
        win._channels[1]["levels"] = (0.0, 2.0)
        rgba2 = win._compose_rgba(scalar)

        # The red-only pixel is unchanged; the green-only pixel changed.
        assert np.allclose(rgba2[0, 0, :3], red_before), "red channel changed when only green B/C changed"
        assert not np.allclose(rgba2[3, 3, :3], green_before), "green B/C had no effect"

        # Runtime handler: dragging the slider (with the green channel active) must
        # write ONLY the active channel's levels, not the others'.
        win._channels[0]["levels"] = (0.0, 10.0)
        win._channels[1]["levels"] = (0.0, 10.0)
        win._last_scalar_tile = scalar
        win._idx = 1                                       # green channel active
        assert win._active_channel_index() == 1
        win._on_levels_changed(0.0, 3.0)
        assert win._channels[1]["levels"] == (0.0, 3.0)    # green updated
        assert win._channels[0]["levels"] == (0.0, 10.0)   # red untouched

        # White background = subtractive composite: no-signal pixels are white,
        # signal darkens toward the hue (not additive light on black).
        win._channels[0]["levels"] = (0.0, 10.0)           # red back to a mid level
        win._white_bg = True
        rgba_w = win._compose_rgba(scalar)
        assert np.allclose(rgba_w[1, 1, :3], [1, 1, 1])    # empty region → white page
        rp = rgba_w[0, 0, :3]                              # red-only pixel
        assert rp[0] > rp[1] and rp[0] > rp[2]             # reddish on white, not white/black
    finally:
        win.close()
        _qt_app.processEvents()
