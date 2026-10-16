"""Regression coverage for GUI-thread-free load/save dispatch."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt6")


def _dataset(name: str, prefs: dict):
    from minflux_viewer.core.dataset import build_localization_dataset

    values = np.arange(12, dtype=float)
    return build_localization_dataset(
        name=name,
        x_nm=values,
        y_nm=values + 1,
        z_nm=np.zeros_like(values),
        prefs=prefs,
    )


@pytest.mark.parametrize("cancelled", [False, True])
def test_background_cleanup_failures_are_logged_without_losing_terminal_signals(qapp, caplog, cancelled):
    from minflux_viewer.ui.background_tasks import BackgroundTask

    seen = []

    def work(report):
        if cancelled:
            task.cancel()
        return object()

    def discard(result):
        raise RuntimeError("discard broke")

    def finish():
        raise RuntimeError("final cleanup broke")

    task = BackgroundTask(work, description="boundary test", discard_result=discard,
                          finally_callback=finish)
    task.signals.done.connect(lambda result: seen.append("done"))
    task.signals.cancelled.connect(lambda: seen.append("cancelled"))
    task.signals.finished.connect(lambda: seen.append("finished"))
    task.run()
    assert seen == ["cancelled" if cancelled else "done", "finished"]
    failures = [record for record in caplog.records if record.name.endswith("background_tasks")]
    assert len(failures) == (2 if cancelled else 1)
    assert all(record.exc_info for record in failures)
    assert "final cleanup broke" in caplog.text
    if cancelled:
        assert "discard broke" in caplog.text


def _pump_until(app, predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _window():
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.main_window import MainWindow

    state = AppState()
    state.prefs.setdefault("data", {}).update(
        {
            "show_data_info": False,
            "show_render": False,
            "compute_loc_prec": False,
            "compute_local_density": False,
            "estimate_z_scaling_factor": False,
        }
    )
    state.prefs.setdefault("plot", {})["use_fixed_z_scaling_factor"] = False
    state.save_prefs = lambda: None
    return MainWindow(state)


def test_ordinary_dataset_loader_returns_before_slow_parser_finishes(qapp) -> None:
    window = _window()
    dataset = _dataset("loaded", window._state.prefs)

    def slow_loader(_path, **_kwargs):
        time.sleep(0.3)
        return dataset

    try:
        started = time.perf_counter()
        window._start_dataset_load("slow.mat", slow_loader, kind="test data")
        elapsed = time.perf_counter() - started

        assert elapsed < 0.15
        assert len(window._state.datasets) == 0
        assert _pump_until(qapp, lambda: not window._file_io_tasks)
        assert window._state.datasets == [dataset]
    finally:
        window.close()
        qapp.processEvents()


def test_non_zarr_save_returns_before_slow_writer_finishes(
    qapp, tmp_path, monkeypatch
) -> None:
    from minflux_viewer.core import save as save_module

    window = _window()
    dataset = _dataset("saved", window._state.prefs)
    window._state.add_dataset(dataset)
    target = tmp_path / "saved.mat"

    def slow_save(_dataset, **_kwargs):
        time.sleep(0.3)
        return [Path(target)]

    monkeypatch.setattr(save_module, "save_processed", slow_save)
    try:
        started = time.perf_counter()
        window._save_as_format("mat", "MATLAB data", path=str(target))
        elapsed = time.perf_counter() - started

        assert elapsed < 0.15
        assert window._file_io_tasks
        assert _pump_until(qapp, lambda: not window._file_io_tasks)
    finally:
        window.close()
        qapp.processEvents()
