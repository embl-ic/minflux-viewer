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


def test_cuda_requires_explicit_environment_and_cpu_recovery(state, qtbot):
    win = _window(state, qtbot)
    _ready(qtbot, win)
    win._backend_combo.setCurrentIndex(1)
    qtbot.waitUntil(lambda: "absolute path" in win._info_label.text())
    assert win._map_key is None and not win._add_btn.isEnabled()
    assert win._response.size == 0 and win._fft_provenance == {}
    win._backend_combo.setCurrentIndex(0)
    _ready(qtbot, win)
    assert win._fft_provenance["backend"] == "cpu-scipy"


def test_cuda_selection_cache_provenance_and_persistence(state, qtbot, monkeypatch):
    import sys
    from types import SimpleNamespace

    from scipy.signal import fftconvolve

    calls = []
    def accelerated(image, kernel, settings, **kwargs):
        calls.append((settings, threading.get_ident()))
        return fftconvolve(image, kernel, mode="same"), {"backend": "cuda-cupy", "dtype": "float64", "device_name": "test GPU"}
    monkeypatch.setattr(cs, "cuda_fftconvolve", accelerated)
    win = _window(state, qtbot)
    _ready(qtbot, win)
    win._cuda_python.setText(sys.executable)
    win._backend_combo.setCurrentIndex(1)
    _ready(qtbot, win)
    assert len(calls) == 1 and calls[0][1] != threading.get_ident()
    assert "cuda-cupy" in win._info_label.text()
    assert "test GPU" in win._info_label.toolTip()
    win._response_spin.setValue(.4)
    _ready(qtbot, win)
    assert len(calls) == 1 and win._fft_provenance["device_name"] == "test GPU"
    captured = []
    win._owner = SimpleNamespace(add_segmentation_rois=lambda *a, **kw: captured.append(kw) or 1)
    win._add_to_manager()
    assert captured and "test GPU" in captured[0]["log_message"]
    win._cuda_device.setValue(1)
    _ready(qtbot, win)
    assert len(calls) == 2 and calls[-1][0].device == 1
    win.close()
    assert state.prefs["conv_segmentation"]["fft_backend"] == "cuda"
    again = _window(state, qtbot)
    _ready(qtbot, again)
    assert again._backend_combo.currentData() == "cuda" and again._cuda_device.value() == 1


def test_switch_backend_discards_late_cuda_result(state, qtbot, monkeypatch):
    import sys

    from scipy.signal import fftconvolve

    started, release = threading.Event(), threading.Event()
    def accelerated(image, kernel, settings, **kwargs):
        started.set()
        assert release.wait(5)
        return fftconvolve(image, kernel, mode="same"), {"backend": "cuda-cupy"}
    monkeypatch.setattr(cs, "cuda_fftconvolve", accelerated)
    win = _window(state, qtbot)
    _ready(qtbot, win)
    win._cuda_python.setText(sys.executable)
    win._backend_combo.setCurrentIndex(1)
    try:
        qtbot.waitUntil(started.is_set)
        win._backend_combo.setCurrentIndex(0)
        assert not win._add_btn.isEnabled()
    finally:
        release.set()
    _ready(qtbot, win)
    assert win._fft_provenance["backend"] == "cpu-scipy"
