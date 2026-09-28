"""The modeless MSD workbench, including its worker and export boundaries."""

from __future__ import annotations

import json
import os
import sys
import time
import zipfile

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


def _dataset(n_tracks=8, per=30, seed=4):
    from minflux_viewer.core.dataset import build_localization_dataset

    rng = np.random.default_rng(seed)
    dt = 8.0e-4
    increments = rng.normal(0.0, 8.0, (n_tracks, per - 1, 3))
    paths = np.concatenate(
        [np.zeros((n_tracks, 1, 3)), np.cumsum(increments, axis=1)], axis=1)
    coords = paths.reshape(-1, 3)
    return build_localization_dataset(
        name="MSD test",
        x_nm=coords[:, 0],
        y_nm=coords[:, 1],
        z_nm=coords[:, 2],
        tid=np.repeat(np.arange(n_tracks), per),
        tim=np.tile(np.arange(per) * dt, n_tracks),
    )


def _state():
    from minflux_viewer.core.app_state import AppState

    state = AppState()
    state.add_dataset(_dataset())
    return state


def _wait(app, window, timeout=5.0):
    deadline = time.perf_counter() + timeout
    while window._tasks and time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.005)
    for _ in range(5):
        app.processEvents()
    assert not window._tasks, "MSD background task did not finish"


def test_msd_workbench_runs_and_presents_all_result_levels(_qt_app):
    from PyQt6.QtCore import Qt

    from minflux_viewer.ui.tracking_analysis_window import MsdAnalysisWindow

    win = MsdAnalysisWindow(_state(), 0)
    win.show()
    try:
        _wait(_qt_app, win)
        assert win.parent() is None
        assert win.windowModality() == Qt.WindowModality.NonModal
        assert win._payload is not None
        result = win._payload.result
        assert result.diagnostics.status == "ok"
        assert {
            "segment_msd",
            "pooled_time_averaged_msd",
            "ensemble_msd",
            "segment_fit",
            "pooled_fit",
        } == set(result.tables)
        assert win._fit_model.rowCount() > 0
        assert win._segment_combo.count() > 0
        assert "apparent σ" in win._summary.text()
        assert "interpolation" in win._audit.toPlainText()
        assert win._export_button.isEnabled()
    finally:
        win.close()
        _qt_app.processEvents()


def test_msd_workbench_compute_runs_off_the_gui_thread(_qt_app, monkeypatch):
    import threading

    from minflux_viewer.ui import tracking_analysis_window as module

    gui_thread = threading.get_ident()
    worker_threads = []
    original = module.run_tracking_method

    def observed(*args, **kwargs):
        worker_threads.append(threading.get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "run_tracking_method", observed)
    win = module.MsdAnalysisWindow(_state(), 0)
    win.show()
    try:
        _wait(_qt_app, win)
        assert worker_threads
        assert all(identifier != gui_thread for identifier in worker_threads)
    finally:
        win.close()
        _qt_app.processEvents()


def test_new_msd_run_cancels_and_rejects_the_old_generation(_qt_app, monkeypatch):
    from minflux_viewer.ui.tracking_analysis_window import MsdAnalysisWindow

    queued = []
    monkeypatch.setattr(
        MsdAnalysisWindow, "_start_task", lambda _self, task: queued.append(task))
    win = MsdAnalysisWindow(_state(), 0)
    win.show()
    try:
        # Run the single-shot that queues the initial analysis.
        _qt_app.processEvents()
        assert len(queued) == 1
        old_generation = win._generation
        stale = queued[0]._work(lambda _stage: None)

        win.run_analysis()
        assert len(queued) == 2
        assert queued[0].is_cancelled
        queued[1].run()
        _qt_app.processEvents()
        current = win._payload
        assert current is not None

        win._on_done(old_generation, stale)
        assert win._payload is current
    finally:
        win.close()
        _qt_app.processEvents()


def test_tracking_result_zip_contains_tables_units_and_provenance(tmp_path):
    from minflux_viewer.analysis.tracking_export import export_tracking_result_zip
    from minflux_viewer.analysis.tracking_stats import (
        mean_squared_displacement,
        trajectories_from_dataset,
    )

    result = mean_squared_displacement(
        trajectories_from_dataset(_dataset(n_tracks=2, per=12)),
        min_pairs=1,
        max_lag_points=4,
    )
    path = export_tracking_result_zip(result, tmp_path / "analysis")

    assert path.name == "analysis.zip"
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        assert "metadata.json" in names
        assert "segment_fit.csv" in names
        assert "pooled_time_averaged_msd.csv" in names
        metadata = json.loads(archive.read("metadata.json"))
        assert metadata["method_id"] == "minflux_viewer.tracking.msd"
        assert metadata["units"]["pooled_fit"]["diffusion"] == "nm²/s"
        assert metadata["provenance"]["interpolation"] == "none"


def test_dataset_change_invalidates_a_visible_result(_qt_app):
    from minflux_viewer.ui.tracking_analysis_window import MsdAnalysisWindow

    state = _state()
    win = MsdAnalysisWindow(state, 0)
    win.show()
    try:
        _wait(_qt_app, win)
        assert win._payload is not None
        state.notify_filter_changed(0)
        _qt_app.processEvents()
        assert win._payload is None
        assert not win._export_button.isEnabled()
        assert "Run again" in win._status.text()
    finally:
        win.close()
        _qt_app.processEvents()


def test_msd_workbench_can_materialize_selected_attributes(_qt_app):
    from minflux_viewer.ui.tracking_analysis_window import MsdAnalysisWindow

    state = _state()
    state.prefs["tracking"].update({
        "materialize_analysis_attributes": True,
        "analysis_attributes": ["msd_d", "step_angle"],
    })
    win = MsdAnalysisWindow(state, 0)
    win.show()
    try:
        _wait(_qt_app, win)
        dataset = state.datasets[0]
        assert "msd_d" in dataset.attr and "step_angle" in dataset.attr
        assert "msd_alpha" not in dataset.attr
        assert dataset.mfx.meta["msd_d"]["user_visible"] is True
        assert win._payload is not None, "self-notification must not invalidate result"
    finally:
        win.close()
        _qt_app.processEvents()


def test_main_menu_opens_one_modeless_msd_workbench_per_dataset(_qt_app):
    from minflux_viewer.ui.main_window import MainWindow

    state = _state()
    state.prefs.setdefault("file", {})["check_updates_on_startup"] = False
    state.save_prefs = lambda: None
    main = MainWindow(state)
    main.show()
    try:
        main._ui.actionMsdAnalysis.trigger()
        _qt_app.processEvents()
        first = main._msd_windows[0]
        assert first.isVisible()
        main._ui.actionMsdAnalysis.trigger()
        _qt_app.processEvents()
        assert main._msd_windows[0] is first
        assert "pending human approval" not in (
            main._ui.actionMsdAnalysis.statusTip().lower())
        assert not main._ui.actionParticleTracking.isVisible()
        assert main._ui.actionParticleTracking not in main._ui.menuTracking.actions()
    finally:
        main.close()
        _qt_app.processEvents()
