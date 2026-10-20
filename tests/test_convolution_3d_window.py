"""The separate 3D plugin owns immutable requests, not the existing 2D dialog."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import fftconvolve

from minflux_viewer.analysis import convolution_3d as conv
from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.ui.analysis.convolution_3d_window import Convolution3DWindow


@pytest.fixture
def state(qapp, monkeypatch):
    state = AppState()
    monkeypatch.setattr(state, "save_prefs", lambda: None)
    state.prefs["convolution_3d"] = {}
    rng = np.random.default_rng(38)
    xyz = np.vstack([rng.normal((30., -40., z), 2., (300, 3)) for z in (-80., 80.)])
    state.add_dataset(build_localization_dataset(name="two Z sites", x_nm=xyz[:, 0], y_nm=xyz[:, 1], z_nm=xyz[:, 2]))
    return state


def window(state, qtbot, owner=None):
    win = Convolution3DWindow(state, 0, owner)
    qtbot.addWidget(win)
    win.show()
    win._set_bounds(((-100., 100.), (-100., 100.), (-160., 160.)))
    for spin, value in zip(win._voxels, (4., 5., 2.)):
        spin.setValue(value)
    win._model.setCurrentIndex(win._model.findData("gaussian"))
    win._width.setValue(6.)
    return win


def ready(qtbot, win):
    qtbot.waitUntil(lambda: win._task is None and win._accepted_stamp is not None, timeout=15000)


def test_real_xyz_worker_publication_cache_and_display_independence(state, qtbot, monkeypatch):
    calls, published = [], []
    def recording(*a, **kw):
        calls.append((a[0].shape, threading.get_ident()))
        return fftconvolve(*a, **kw)
    monkeypatch.setattr(conv, "fftconvolve", recording)
    owner = SimpleNamespace(add_point_rois_3d=lambda *a, **kw: published.append((a, kw)))
    win = window(state, qtbot, owner)
    win._submit()
    ready(qtbot, win)
    # The 50 x 40 x 160 region plus its 4-sigma edge margin on every side.
    assert calls == [((62, 50, 184), calls[0][1])]
    assert calls[0][1] != threading.get_ident()
    assert len(win._centers) == 2
    original = win._maps["response"]
    assert original.shape == (50, 40, 160)
    for plane in ("XZ", "YZ", "XY"):
        win._plane.setCurrentText(plane)
    win._plot.getViewBox().scaleBy((.8, .8))
    win._copy_appearance()
    win._threshold.setValue(.5)
    assert not win._add.isEnabled()
    win._submit()
    ready(qtbot, win)
    assert len(calls) == 1 and win._maps["response"] is original
    win._publish()
    assert len(published) == 1 and published[0][0][1].shape == (2, 3)
    assert published[0][1]["source"] == "convolution_3d"
    assert '"backend": "cpu-scipy"' in published[0][1]["log_message"]
    assert '"voxel_xyz": [4.0, 5.0, 2.0]' in published[0][1]["log_message"]
    view_before = win._plot.getViewBox().viewRange()
    win._voxels[0].setValue(5.)
    assert win._plot.getViewBox().viewRange() == view_before
    assert win._maps is None and not win._add.isEnabled()
    win.close()
    assert state.prefs["convolution_3d"]["voxel_limit"] == 8_000_000
    assert "conv_segmentation" not in state.prefs


def test_refusal_keeps_inspection_preview_and_computes_nothing(state, qtbot, monkeypatch):
    monkeypatch.setattr(conv, "fftconvolve", lambda *a, **kw: pytest.fail("ran a refused request"))
    win = window(state, qtbot)
    win._voxel_budget.setValue(10)
    win._submit()
    assert "voxels" in win._status.text() and win._task is None
    qtbot.waitUntil(lambda: win._data_preview.pixels.size > 0)
    assert win._data_item.image is not None and win._maps is None and not win._add.isEnabled()


def test_the_default_region_runs_at_default_settings_on_thick_data(qapp, qtbot, monkeypatch):
    """Regression: the default region used to exceed the window's own 512 MiB
    budget whenever the data were thicker than about 250 nm in Z."""
    state = AppState()
    monkeypatch.setattr(state, "save_prefs", lambda: None)
    state.prefs["convolution_3d"] = {}
    rng = np.random.default_rng(5)
    xyz = rng.uniform((0., 0., 0.), (2000., 2000., 800.), size=(20000, 3))
    state.add_dataset(build_localization_dataset(name="thick", x_nm=xyz[:, 0], y_nm=xyz[:, 1], z_nm=xyz[:, 2]))
    win = Convolution3DWindow(state, 0)
    qtbot.addWidget(win)
    bounds = np.asarray(win._request().bounds_xyz)
    np.testing.assert_allclose(bounds[:, 1] - bounds[:, 0], 400., atol=.02)
    win._submit()
    ready(qtbot, win)
    # 400 nm at 4 nm voxels; the region buttons round outward, hence 100 or 101.
    assert win._maps is not None and all(n in (100, 101) for n in win._maps["shape"])


def test_filter_change_and_close_retire_inflight_xyz_results(state, qtbot, monkeypatch):
    started, release = threading.Event(), threading.Event()
    def slow(*a, **kw):
        started.set()
        assert release.wait(5)
        return fftconvolve(*a, **kw)
    monkeypatch.setattr(conv, "fftconvolve", slow)
    win = window(state, qtbot)
    win._submit()
    try:
        qtbot.waitUntil(started.is_set)
        state.datasets[0].filter_mask = np.arange(600) % 2 == 0
        state.notify_filter_changed(0)
        assert not win._add.isEnabled() and len(win._xyz) == 300
    finally:
        release.set()
    qtbot.waitUntil(lambda: win._task is None)
    assert win._maps is None and win._centers.size == 0
    started.clear()
    release.clear()
    win._submit()
    try:
        qtbot.waitUntil(started.is_set)
        start = time.perf_counter()
        win.close()
        assert time.perf_counter() - start < .5
    finally:
        release.set()


def test_late_source_replacement_never_enables_roi_publication(state, qtbot):
    published = []
    win = window(state, qtbot, SimpleNamespace(add_point_rois_3d=lambda *a, **kw: published.append(a)))
    win._submit()
    ready(qtbot, win)
    state.remove_dataset(0)
    win._publish()
    assert not published and not win._add.isEnabled() and not win._run.isEnabled()


def test_plugin_is_separate_and_rejects_2d_without_opening_window(state, monkeypatch):
    from minflux_viewer.core.plugin_registry import available
    from minflux_viewer.plugins import convolution_3d

    assert any(entry.name == "3D Convolution…" for entry in available())
    state.datasets[0].prop.num_dim = 2
    messages = []
    monkeypatch.setattr(state, "log", lambda *args: messages.append(args))
    convolution_3d._launch(state)
    assert messages and "3D" in messages[0][0]


def test_initial_region_never_clips_a_small_dataset(state, qtbot):
    win = Convolution3DWindow(state, 0)
    qtbot.addWidget(win)
    bounds = np.asarray(win._request().bounds_xyz)
    assert np.all(bounds[:, 0] <= win._xyz.min(axis=0))
    assert np.all(bounds[:, 1] >= win._xyz.max(axis=0))
    win._whole_region()
    bounds = np.asarray(win._request().bounds_xyz)
    assert np.all(bounds[:, 0] <= win._xyz.min(axis=0))
    assert np.all(bounds[:, 1] >= win._xyz.max(axis=0))
