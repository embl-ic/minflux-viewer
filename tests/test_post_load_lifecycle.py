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

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

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


def test_late_delivery_cannot_replace_newer_state_or_task():
    from minflux_viewer.core.computation_snapshot import input_signature

    for change in ("filter", "calibration", "coordinates", "attributes", "task", "shutdown"):
        st = _state("A")
        ds, key = st.datasets[0], st.key_at(0)
        task = _task()
        applied = []
        stub = SimpleNamespace(_state=st, _is_shutting_down=False,
                               _post_load_tasks={key: task},
                               _apply_post_load_results=lambda *a: applied.append(a),
                               _on_saved_filters_evaluated=lambda *a: applied.append(a),
                               _on_post_load_failed=lambda *a: applied.append(a),
                               _on_saved_filter_failed=lambda *a: applied.append(a))
        signature, revision = input_signature(ds), st.data_revision(key)
        if change == "filter":
            ds.state["filter_specs"] = [dict(attribute="tid", lo=1, hi=2)]
        elif change == "calibration":
            ds.set_z_scaling_factor(.7)
        elif change == "coordinates":
            ds.attr.replace_coordinates(loc_x=np.ones(ds.prop.num_loc))
        elif change == "attributes":
            st.notify_attributes_changed(0)  # also covers notified in-place edits
        elif change == "task":
            stub._post_load_tasks[key] = _task()
        else:
            stub._is_shutting_down = True
        MainWindow._on_post_load_done(stub, key, ds, {}, task, signature, revision)
        MainWindow._on_post_load_error(stub, key, ds, "late", task, signature, revision)
        MainWindow._on_saved_filters_done(stub, key, ds, np.ones(12), task, signature, revision)
        MainWindow._on_saved_filters_error(stub, key, ds, "late", task, signature, revision)
        assert applied == [], change


def test_current_guarded_delivery_still_applies():
    from minflux_viewer.core.computation_snapshot import input_signature

    st = _state("A")
    ds, key = st.datasets[0], st.key_at(0)
    task, applied = _task(), []
    stub = SimpleNamespace(_state=st, _is_shutting_down=False,
                           _post_load_tasks={key: task},
                           _apply_post_load_results=lambda *a: applied.append(a))
    MainWindow._on_post_load_done(stub, key, ds, {"run": 1}, task,
                                  input_signature(ds), st.data_revision(key))
    assert applied == [(ds, {"run": 1})]


@pytest.mark.parametrize("actions", list(itertools.permutations(("edit", "close-reopen", "cancel", "deliver"))))
def test_adversarial_edit_close_cancel_delivery_orders(actions):
    from minflux_viewer.core.computation_snapshot import input_signature

    state = _state("A")
    dataset, key, task = state.datasets[0], state.key_at(0), _task()
    signature, revision = input_signature(dataset), state.data_revision(key)
    applied = []
    owner = SimpleNamespace(_state=state, _is_shutting_down=False,
                            _post_load_tasks={key: task},
                            _apply_post_load_results=lambda *args: applied.append(args))
    invalidated = False
    delivered_before_change = False
    for action in actions:
        if action == "edit":
            dataset.filter_mask = np.arange(dataset.prop.num_loc) % 2 == 0
            state.notify_filter_changed(state.index_of_dataset(dataset))
            invalidated = True
        elif action == "close-reopen":
            state.remove_dataset(state.index_of_dataset(dataset))
            state.add_dataset(dataset)
            invalidated = True
        elif action == "cancel":
            task.cancel()
            invalidated = True
        else:
            delivered_before_change = not invalidated
            MainWindow._on_post_load_done(owner, key, dataset, {}, task, signature, revision)
    assert len(applied) == int(delivered_before_change)
    # Repeated queued results never revive a cancelled/retired registration.
    MainWindow._on_post_load_done(owner, key, dataset, {}, task, signature, revision)
    assert len(applied) == int(delivered_before_change)
