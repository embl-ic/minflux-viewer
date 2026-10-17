"""Post-load work belongs to one dataset, identified by its key.

Stabilization Phase 3, slice 3.4. Closing any dataset used to retire every
dataset's running post-load task, so a dataset loaded just before another was
closed never received its density or precision; and the delayed post-load
start was scheduled with an index, so after an earlier close it started for
whichever dataset had moved into that position -- or for none. Both now go by
the dataset's key. The real-window versions are in
``tests/test_dataset_identity_characterization.py``; these drive the main
window's methods on a light stand-in.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.core.dataset_events import DatasetRemoved
from minflux_viewer.ui.common.background_tasks import BackgroundTask
from minflux_viewer.ui.main_window import MainWindow


def _dataset(name: str):
    x = np.linspace(0.0, 100.0, 12)
    return build_localization_dataset(name=name, x_nm=x, y_nm=x[::-1])


def _state(*names):
    st = AppState()
    for name in names:
        st.add_dataset(_dataset(name))
    return st


def _task():
    return BackgroundTask(lambda report: None, description="test")


def test_closing_a_dataset_retires_only_its_own_task():
    st = _state("A", "B")
    ka, kb = st.key_at(0), st.key_at(1)
    task_a, task_b = _task(), _task()
    stub = SimpleNamespace(_post_load_tasks={ka: task_a, kb: task_b})
    a = st.datasets[0]
    st.remove_dataset(0)

    MainWindow._retire_dataset_tasks(
        stub, DatasetRemoved(key=ka, dataset=a, old_index=0, new_active_key=kb))

    assert task_a.is_cancelled and not task_b.is_cancelled
    assert stub._post_load_tasks == {kb: task_b}


def test_a_scheduled_start_finds_its_dataset_by_key():
    st = _state("A", "B", "C")
    kb = st.key_at(1)
    started = []
    stub = SimpleNamespace(_state=st, _is_shutting_down=False,
                           _run_post_load_computations=started.append)

    st.remove_dataset(0)                      # B moves to index 0
    MainWindow._run_post_load_for_key(stub, kb)
    assert started == [0]

    st.remove_dataset(0)                      # B itself closes
    MainWindow._run_post_load_for_key(stub, kb)
    assert started == [0]


def test_a_scheduled_start_does_nothing_while_shutting_down():
    st = _state("A")
    started = []
    stub = SimpleNamespace(_state=st, _is_shutting_down=True,
                           _run_post_load_computations=started.append)
    MainWindow._run_post_load_for_key(stub, st.key_at(0))
    assert started == []


def test_a_result_is_applied_only_while_its_key_is_open():
    st = _state("A")
    a, key = st.datasets[0], st.key_at(0)
    applied, failed = [], []
    stub = SimpleNamespace(
        _state=st,
        _apply_post_load_results=lambda ds, result: applied.append((ds, result)),
        _on_post_load_failed=lambda ds, message: failed.append((ds, message)))

    MainWindow._on_post_load_done(stub, key, a, {"run": 1})
    assert applied == [(a, {"run": 1})]

    # The same object closed and opened again is a new registration: a result
    # computed for the old one is not delivered to it.
    st.remove_dataset(0)
    st.add_dataset(a)
    MainWindow._on_post_load_done(stub, key, a, {"run": 2})
    MainWindow._on_post_load_error(stub, key, a, "late failure")
    assert applied == [(a, {"run": 1})] and failed == []


def test_the_post_load_index_follows_the_dataset():
    st = _state("A", "B")
    b = st.datasets[1]
    stub = SimpleNamespace(_state=st)
    assert MainWindow._post_load_index(stub, b) == 1
    st.remove_dataset(0)
    assert MainWindow._post_load_index(stub, b) == 0
    st.remove_dataset(0)
    assert MainWindow._post_load_index(stub, b) is None
