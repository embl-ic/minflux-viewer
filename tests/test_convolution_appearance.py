"""Inspection appearance is copied by channel, independent of compute pixels."""

from types import SimpleNamespace

import numpy as np
import pytest

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.ui.analysis.conv_segmentation_dialog import ConvSegmentationWindow
from minflux_viewer.ui.rendering.inspection_layer import render_appearance
from minflux_viewer.ui.rendering.precision_render_window import PrecisionRenderWindow
from minflux_viewer.ui.rendering.render_window import RenderWindow


@pytest.mark.parametrize("render_class", [RenderWindow, PrecisionRenderWindow])
@pytest.mark.parametrize("white", [False, True])
def test_nonprimary_channel_lut_gamma_inversion_and_contrast_copy(qtbot, monkeypatch, render_class, white):
    state = AppState()
    monkeypatch.setattr(state, "save_prefs", lambda: None)
    xyz = np.random.default_rng(77).normal(0., 100., (300, 3))
    for name in ("anchor", "other"):
        ds = build_localization_dataset(name=name, x_nm=xyz[:, 0], y_nm=xyz[:, 1], z_nm=xyz[:, 2])
        ds.state["overlay_id"] = "two-channel"
        state.add_dataset(ds)
    render = render_class(state, dataset_idx=0)
    qtbot.addWidget(render)
    render._white_bg = white
    render._channels[1].update(lut="Green", gamma=.7, lut_invert=True, levels=(25., 75.))
    render._last_scalar_tile = np.stack((np.ones((10, 10)), np.arange(1., 101.).reshape(10, 10)))
    owner = SimpleNamespace(_render_windows={0: render}, _advanced_render_windows={})
    copied = render_appearance(owner, state, 1)
    assert copied.contrast_pct == (24., 74.)
    assert copied.white_background == white
    ramp = np.linspace(0., 1., 256, dtype=np.float32)**.7
    expected = (render._channel_rgb_white(ramp, "Green") if white else
                render._map_norm_to_rgb(ramp, "Green", invert=True))
    np.testing.assert_array_equal(copied.lut, np.rint(expected * 255).astype(np.uint8))
    window = ConvSegmentationWindow(state, 1, owner)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitUntil(lambda: window._data_preview.pixels.size > 0 and window._data_preview._task is None
                   and not window._data_preview._pending, timeout=10000)
    np.testing.assert_array_equal(window._hist_img.lut, copied.lut)
    old_pixels = window._data_preview.pixels
    bounds = window._plot.getViewBox().viewRange()
    window._pixel_spin.setValue(.5)
    assert window._data_preview.pixels is old_pixels
    assert window._plot.getViewBox().viewRange() == bounds
    window._bc.set_target("Data")
    window._bc._on_levels(0., .01)
    assert render._channels[1]["levels"] == (25., 75.)
    render._channels[1]["levels"] = (10., 90.)
    render._last_scalar_tile = np.stack((np.ones((10, 10)), np.arange(1., 101.).reshape(10, 10)))
    window._copy_render_appearance()
    assert window._data_preview.appearance.contrast_pct == (9., 89.)
    window.close()
    render.close()
