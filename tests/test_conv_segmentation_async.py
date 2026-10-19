"""Convolution previews are bounded worker jobs, never widget-thread FFTs."""

import threading
import time

import numpy as np
import pytest
from PyQt6.QtCore import QTimer

from minflux_viewer.analysis import conv_segmentation as cs
from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.ui.analysis.conv_segmentation_dialog import ConvSegmentationWindow


@pytest.fixture
def state(qapp, monkeypatch):
    state = AppState()
    monkeypatch.setattr(state, "save_prefs", lambda: None)
    state.prefs["conv_segmentation"] = {}
    rng = np.random.default_rng(17)
    xy = rng.normal(0, 100, (100, 2))
    state.add_dataset(build_localization_dataset(name="test", x_nm=xy[:, 0], y_nm=xy[:, 1]))
    return state


def _ready(qtbot, win):
    qtbot.waitUntil(lambda: win._task is None and not win._pending
                   and win._map_key is not None, timeout=10000)
    qtbot.waitUntil(lambda: win._data_preview._task is None and not win._data_preview._pending, timeout=10000)


def _window(state, qtbot):
    win = ConvSegmentationWindow(state, 0)
    qtbot.addWidget(win)
    win._recompute_timer.setInterval(1)
    win._schedule_recompute()
    return win


def test_worker_parity_and_detection_only_cache_reuse(state, qtbot, monkeypatch):
    original, calls = cs.convolution_maps, []
    gui_thread = threading.get_ident()
    def count(*args, **kwargs):
        calls.append(threading.get_ident())
        return original(*args, **kwargs)
    monkeypatch.setattr(cs, "convolution_maps", count)
    win = _window(state, qtbot)
    _ready(qtbot, win)
    expected = original(win._xy, model_key=win._model_key(), params=win._params(),
                        pixel_size_nm=win._pixel_spin.value(),
                        zero_mean=win._zero_mean.isChecked(), sqrt_stabilize=win._sqrt.isChecked(),
                        dog=win._dog.isChecked())
    np.testing.assert_array_equal(win._H, expected["histogram"])
    np.testing.assert_array_equal(win._response, expected["response_map"])
    assert calls and gui_thread not in calls
    old_response = win._response
    win._response_spin.setValue(.7)
    _ready(qtbot, win)
    assert win._response is old_response and len(calls) == 1


def test_rapid_edits_coalesce_and_event_loop_stays_live(state, qtbot, monkeypatch):
    original, started, release, calls = cs.convolution_maps, threading.Event(), threading.Event(), []
    def slow(*args, **kwargs):
        calls.append(kwargs["params"]["diameter_nm"])
        if len(calls) == 1:
            started.set()
            assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(cs, "convolution_maps", slow)
    win = _window(state, qtbot)
    try:
        qtbot.waitUntil(started.is_set)
        heartbeat = []
        QTimer.singleShot(0, lambda: heartbeat.append(True))
        qtbot.waitUntil(lambda: bool(heartbeat))
        win._param_spins["diameter_nm"].setValue(200)
        win._param_spins["diameter_nm"].setValue(300)
        assert not win._add_btn.isEnabled()
    finally:
        release.set()
    _ready(qtbot, win)
    assert calls == [100., 300.]
    assert win._map_key[2] == tuple(win._params().items())


def test_filter_edit_retires_old_source_result(state, qtbot, monkeypatch):
    original, started, release = cs.convolution_maps, threading.Event(), threading.Event()
    def slow(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(cs, "convolution_maps", slow)
    win = _window(state, qtbot)
    try:
        qtbot.waitUntil(started.is_set)
        state.datasets[0].filter_mask = np.arange(100) < 50
        state.notify_filter_changed(0)
    finally:
        release.set()
    _ready(qtbot, win)
    assert len(win._xy) == 50
    assert win._H.sum() == 50


def test_close_does_not_wait_for_native_work_or_apply_late_result(state, qtbot, monkeypatch):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = cs.convolution_maps
    def slow(*args, **kwargs):
        started.set()
        try:
            assert release.wait(5)
            return original(*args, **kwargs)
        finally:
            finished.set()
    monkeypatch.setattr(cs, "convolution_maps", slow)
    win = _window(state, qtbot)
    try:
        qtbot.waitUntil(started.is_set)
        task, generation, stamp = win._task, win._generation, win._source_stamp()
        start = time.perf_counter()
        win.close()
        assert time.perf_counter() - start < .5
        win._computed(task, generation, stamp, (), None)  # detached queued delivery
        assert win._closed and task.is_cancelled
    finally:
        release.set()
    qtbot.waitUntil(finished.is_set)


def test_worker_failure_and_preflight_error_are_visible(state, qtbot, monkeypatch):
    original = cs.convolve_response
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic FFT failure")
    monkeypatch.setattr(cs, "convolve_response", fail)
    win = _window(state, qtbot)
    qtbot.waitUntil(lambda: "synthetic FFT" in win._info_label.text())
    assert not win._add_btn.isEnabled() and win._response.size == 0
    monkeypatch.setattr(cs, "convolve_response", original)
    win._param_spins["diameter_nm"].setValue(5000)
    win._pixel_spin.setValue(.5)
    qtbot.waitUntil(lambda: "budget" in win._info_label.text())
    assert not win._add_btn.isEnabled() and win._response.size == 0


def test_completed_jobs_release_captured_maps_and_callbacks(state, qtbot, monkeypatch):
    import gc
    import weakref

    from minflux_viewer.ui.analysis import conv_segmentation_dialog as dialog

    original, references = dialog.BackgroundTask, []
    def capture(*args, **kwargs):
        task = original(*args, **kwargs)
        references.append(weakref.ref(task))
        return task
    monkeypatch.setattr(dialog, "BackgroundTask", capture)
    win = _window(state, qtbot)
    _ready(qtbot, win)
    win._response_spin.setValue(.7)
    _ready(qtbot, win)
    qtbot.wait(10)
    gc.collect()
    assert len(references) == 2
    assert all(reference() is None for reference in references)


def test_legacy_cuda_preferences_do_not_change_cpu_tool(state, qtbot, monkeypatch):
    state.prefs["conv_segmentation"] = dict(fft_backend="cuda", cuda_python="missing")
    monkeypatch.setattr(cs, "cuda_fftconvolve", lambda *a, **kw: pytest.fail("CPU tool invoked CUDA"))
    win = _window(state, qtbot)
    _ready(qtbot, win)
    assert not hasattr(win, "_backend_combo")
    assert win._fft_provenance["backend"] == "cpu-scipy"
    win.close()
    assert "fft_backend" not in state.prefs["conv_segmentation"]


@pytest.fixture
def npc_state(state):
    from minflux_viewer.core.simulate import simulate_localizations

    xyz, tid, attrs = simulate_localizations(
        "npc", n_points=5000, locs_per_trace=4., precision_nm=1., dim=3, seed=38,
        params=dict(size_nm=5000., n_pores=100, diameter_nm=120., ring_sep_nm=55.,
                    field_curvature=.05, local_tilt_deg=1.))
    state.add_dataset(build_localization_dataset(name="3D NPC", x_nm=xyz[:, 0], y_nm=xyz[:, 1],
                                                z_nm=xyz[:, 2], tid=tid, attrs=attrs,
                                                source_version="simulation"))
    state.prefs["conv_segmentation"] = dict(
        model="disk", pixel_size=2., memory_budget_mib=512, sqrt=True, dog=True,
        min_response=.14, min_separation=80., show_detection=False, show_response=False,
        show_data=True, model_params=dict(disk=dict(diameter_nm=25., edge_nm=5.5)))
    return state


def _npc_window(npc_state, qtbot):
    win = ConvSegmentationWindow(npc_state, 1)
    qtbot.addWidget(win)
    win.show()
    return win


def _data_ready(qtbot, win):
    qtbot.waitUntil(lambda: win._task is None and not win._pending and win._data_preview.pixels.size > 0
                   and win._data_preview._task is None and not win._data_preview._pending,
                   timeout=15000)


def test_npc_budget_refusal_keeps_monitor_data_and_exact_compute_recovery(npc_state, qtbot, monkeypatch):
    calls, original = [], cs.convolution_maps
    def count(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(cs, "convolution_maps", count)
    win = _npc_window(npc_state, qtbot)
    _data_ready(qtbot, win)
    assert not calls and win._map_key is None and win._response.size == 0
    assert max(win._data_preview.pixels.shape) <= 1024 and not win._add_btn.isEnabled()
    assert win._H.size == 0  # No full scientific histogram allocated on refusal.
    assert win._hist_img.isVisible() and win._hist_img.image is not None
    assert win._preview_notice.isVisible() and "512" in win._preview_notice.text()
    assert win._budget_btn.isVisible() and win._memory_spin.value() == 512
    assert win._pixel_spin.value() == 2.
    assert win._plot.getViewBox().viewRect().intersects(win._hist_img.mapRectToParent(win._hist_img.boundingRect()))
    old = win._data_preview.pixels
    win._response_spin.setValue(.2)
    qtbot.wait(200)
    assert win._data_preview.pixels is old and not calls  # unavailable FFT is not retried for thresholds
    win._bc.set_target("Data")
    win._bc.open(win)
    assert win._hist_img.image is not None
    win._budget_btn.click()
    _ready(qtbot, win)
    assert len(calls) == 1 and win._centers.size > 0
    assert win._pixel_spin.value() == 2. and win._memory_spin.value() > 512
    assert not win._preview_notice.isVisible() and not win._budget_btn.isVisible()
    win._show_detection.setChecked(True)
    win._show_response.setChecked(True)
    assert win._resp_img.isVisible() and win._ring_scatters[0].isVisible()


def test_native_fft_failure_retains_current_input_data(state, qtbot, monkeypatch):
    win = _window(state, qtbot)
    _ready(qtbot, win)
    old = win._H
    def fail(*a, **kw):
        raise RuntimeError("backend unavailable")
    monkeypatch.setattr(cs, "convolve_response", fail)
    win._sqrt.setChecked(True)
    qtbot.waitUntil(lambda: win._task is None and "backend unavailable" in win._preview_notice.text())
    assert win._H is old and win._hist_img.image is not None
    assert win._response.size == 0 and not win._add_btn.isEnabled()


def test_initially_hidden_layers_still_fit_the_data_extent(state, qtbot):
    state.prefs["conv_segmentation"] = dict(show_detection=False, show_response=False, show_data=False)
    win = _window(state, qtbot)
    _ready(qtbot, win)
    win._show_image.setChecked(True)
    view = win._plot.getViewBox().viewRect()
    bounds = win._hist_img.mapRectToParent(win._hist_img.boundingRect())
    assert view.contains(bounds.center())
    assert view.width() >= bounds.width() and view.height() >= bounds.height()


def test_data_preview_source_edit_rejects_late_histogram(npc_state, qtbot, monkeypatch):
    from minflux_viewer.ui.rendering import inspection_layer
    original, started, release = inspection_layer.histogram_preview, threading.Event(), threading.Event()
    def slow(*a, **kw):
        started.set()
        assert release.wait(5)
        return original(*a, **kw)
    monkeypatch.setattr(inspection_layer, "histogram_preview", slow)
    win = _npc_window(npc_state, qtbot)
    try:
        qtbot.waitUntil(started.is_set)
        npc_state.datasets[1].filter_mask = np.arange(npc_state.datasets[1].prop.num_loc) % 2 == 0
        npc_state.notify_filter_changed(1)
    finally:
        release.set()
    _data_ready(qtbot, win)
    assert len(win._data_preview._xy) == len(win._xy) < npc_state.datasets[1].prop.num_loc
    np.testing.assert_array_equal(win._data_preview._xy, win._xy)
    assert win._map_key is None and not win._add_btn.isEnabled()


def test_error_banner_is_bounded_and_retains_full_diagnostics(state, qtbot):
    from types import SimpleNamespace

    win = _window(state, qtbot)
    _ready(qtbot, win)
    message = "\n".join(["external backend error"] * 500)
    win._show_map_error(message)
    assert len(win._preview_notice.text()) < 850
    assert win._preview_notice.text().count("\n") == 1
    assert win._preview_notice.toolTip() == message
    win._owner = SimpleNamespace(add_segmentation_rois=lambda *a, **kw: pytest.fail("published stale detections"))
    win._add_to_manager()
