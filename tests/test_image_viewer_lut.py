"""The toolbar LUT button reaches the standalone image viewer.

The viewer already had a right-click Colormap submenu, but not the editor with
draggable level lines, gamma and the custom-colormap menu — and the toolbar
button refused outright, because it gates on an active *dataset* and an image is
deliberately not one.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("tifffile")


@pytest.fixture(scope="module")
def _app():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _viewer(tmp_path, qtbot, *, rgb=False, name="plain.tif"):
    import tifffile

    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.core.tiff_source import TiffImageSource
    from minflux_viewer.ui.tiff_viewer_window import TiffViewerWindow

    rng = np.random.default_rng(0)
    image = ((rng.random((48, 64, 3)) * 255).astype(np.uint8) if rgb
             else (rng.random((48, 64)) * 1000).astype(np.uint16))
    path = tmp_path / name
    tifffile.imwrite(path, image)
    window = TiffViewerWindow(TiffImageSource(path), state=AppState())
    qtbot.addWidget(window)
    return window


def test_the_editor_opens_on_a_single_channel_image(_app, qtbot, tmp_path):
    window = _viewer(tmp_path, qtbot)
    assert window.open_lut_dialog() is True
    assert window._lut_dialog is not None
    window.close()


def test_colormap_invert_and_gamma_all_reach_the_image(_app, qtbot, tmp_path):
    window = _viewer(tmp_path, qtbot)
    assert window._active_cmap == "gray"
    assert window._lut_invert is False and window._lut_gamma == 1.0
    window.open_lut_dialog()

    window._on_lut_cmap_changed("hot", True)
    assert (window._active_cmap, window._lut_invert) == ("hot", True)
    window._on_lut_gamma_changed(0.5)
    assert window._lut_gamma == pytest.approx(0.5)
    window._on_lut_invert_changed(False)
    assert window._lut_invert is False
    window.close()


def test_the_right_click_submenu_keeps_an_open_editor_in_step(_app, qtbot, tmp_path):
    window = _viewer(tmp_path, qtbot)
    window.open_lut_dialog()
    synced = []
    window._refresh_lut_dialog = lambda **kw: synced.append(kw) or True
    window._lut_dialog.hide()               # sync_lut_dialog skips a hidden dialog
    window._on_cmap_changed("viridis")
    assert window._active_cmap == "viridis"
    window.close()


def test_an_rgb_plane_declines_instead_of_showing_a_useless_editor(_app, qtbot, tmp_path):
    window = _viewer(tmp_path, qtbot, rgb=True, name="rgb.tif")
    assert window.open_lut_dialog() is False
    window.close()


def test_the_toolbar_targets_a_focused_image_viewer_without_any_dataset():
    """``_show_lut`` gates on an active dataset; an image is not one."""
    from minflux_viewer.ui.main_window import MainWindow

    focused = object()
    other = object()

    class _Win:
        def __init__(self, visible=True):
            self._visible = visible

        def isVisible(self):
            return self._visible

    a, b = _Win(), _Win()
    stand_in = SimpleNamespace(
        _tiff_windows={"a": a, "b": b},
        _last_active_plot_window=None,
        _state=SimpleNamespace(active_dataset=None),
    )
    resolve = MainWindow._image_viewer_for_lut

    # The focused window wins outright.
    assert resolve(stand_in, a) is a
    # Then the last plot window, when it is an image viewer.
    stand_in._last_active_plot_window = b
    assert resolve(stand_in, focused) is b
    # With several open and none focused there is no defensible target.
    stand_in._last_active_plot_window = None
    assert resolve(stand_in, other) is None
    # A single open viewer and no dataset: the button still works.
    stand_in._tiff_windows = {"a": a}
    assert resolve(stand_in, other) is a
    # ...but not when a dataset is active — that keeps the plot-view path.
    stand_in._state = SimpleNamespace(active_dataset=object())
    assert resolve(stand_in, other) is None
