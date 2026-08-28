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
