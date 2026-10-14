"""Typed dataset events: the one way a dataset change is announced.

Stabilization Phase 3. Slice 3.2 added events that say *which* dataset changed
-- each carries its key, and a removal carries everything a listener needs once
the dataset has left the registry -- beside the index signals, which said only
*where*, a position that shifts whenever an earlier dataset closes. Slice 3.8
retired the index signals once nothing listened to them any more.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.core.dataset_events import (
    ActiveDatasetChanged,
    ChangeAspect,
    DatasetAdded,
    DatasetChanged,
    DatasetRemoved,
    ManualAlignmentRequested,
)

RETIRED_INDEX_SIGNALS = (
    "dataset_added", "dataset_removed", "active_changed", "filter_changed",
    "attributes_changed", "calibration_changed", "overlay_transform_changed",
    "overlay_manual_alignment_requested", "roi_selection_changed",
)
TYPED_SIGNALS = (
    "added", "removed", "active_changed", "filter_changed",
    "attributes_changed", "calibration_changed", "overlay_transform_changed",
    "overlay_manual_alignment_requested", "roi_selection_changed",
)


def _dataset(name: str):
    x = np.linspace(0.0, 100.0, 12)
    return build_localization_dataset(name=name, x_nm=x, y_nm=x[::-1])


class _Recorder:
    """Every typed event, in the order it arrives."""

    def __init__(self, state: AppState):
        self.calls: list[tuple[str, object]] = []
        for name in TYPED_SIGNALS:
            getattr(state.dataset_events, name).connect(
                lambda event, name=name: self.calls.append((name, event)))

    def typed(self):
        return [event for _name, event in self.calls]

    def names(self):
        return [name for name, _ in self.calls]

    def clear(self):
        self.calls.clear()


@pytest.fixture
def state():
    st = AppState()
    for name in "ABC":
        st.add_dataset(_dataset(name))
    return st


@pytest.mark.parametrize("name", RETIRED_INDEX_SIGNALS)
def test_the_index_signals_are_gone(name):
    # A listener connecting to one now fails loudly instead of waiting for a
    # notification that never comes.
    assert not hasattr(AppState, name)
    assert not hasattr(AppState(), name)


def test_events_are_frozen_values():
    key_event = DatasetChanged(key=None, dataset=None, aspect=ChangeAspect.FILTER)
    with pytest.raises(dataclasses.FrozenInstanceError):
        key_event.aspect = ChangeAspect.ATTRIBUTES


def test_adding_announces_the_dataset_and_its_key():
    st = AppState()
    rec = _Recorder(st)
    a = _dataset("A")

    index = st.add_dataset(a)

    key = st.key_of(a)
    assert rec.names() == ["added", "active_changed"]
    added, active = rec.typed()
    assert added == DatasetAdded(key=key, dataset=a, index=index)
    assert active == ActiveDatasetChanged(key=key, dataset=a, previous_key=None)

    b = _dataset("B")
    rec.clear()
    st.add_dataset(b)
    _, active = rec.typed()
    assert active.key == st.key_of(b) and active.previous_key == key


def test_reloading_an_open_file_adds_nothing(state):
    state.set_active(2)
    previous = state.active_key
    rec = _Recorder(state)

    state.add_dataset(_dataset("A"))          # same file as the first one

    (active,) = rec.typed()                   # no DatasetAdded
    assert active == ActiveDatasetChanged(
        key=state.key_at(0), dataset=state.datasets[0], previous_key=previous)


def test_closing_an_earlier_dataset_does_not_change_the_active_one(state):
    a, b, c = state.datasets
    ka, kc = state.key_of(a), state.key_of(c)
    state.set_active(2)
    rec = _Recorder(state)

    state.remove_dataset(0)

    # C moved from 2 to 1, but the active dataset is still C: no active event.
    assert rec.names() == ["removed"]
    (removed,) = rec.typed()
    assert removed == DatasetRemoved(key=ka, dataset=a, old_index=0, new_active_key=kc)
    assert state.active_idx == 1 and state.active_dataset is c


def test_closing_the_active_dataset_names_the_one_that_takes_its_place(state):
    a, b, c = state.datasets
    kb, kc = state.key_of(b), state.key_of(c)
    state.set_active(1)
    rec = _Recorder(state)

    state.remove_dataset(1)

    assert rec.names() == ["removed", "active_changed"]
    removed, active = rec.typed()
    assert removed == DatasetRemoved(key=kb, dataset=b, old_index=1, new_active_key=kc)
    assert active == ActiveDatasetChanged(key=kc, dataset=c, previous_key=kb)


def test_closing_a_later_dataset_changes_nothing_else(state):
    a, b, c = state.datasets
    state.set_active(0)
    rec = _Recorder(state)

    state.remove_dataset(2)

    assert rec.names() == ["removed"]
    (removed,) = rec.typed()
    assert removed.dataset is c and removed.old_index == 2
    assert removed.new_active_key == state.key_of(a)


def test_closing_the_last_dataset_leaves_nothing_active():
    st = AppState()
    a = _dataset("A")
    st.add_dataset(a)
    ka = st.key_of(a)
    rec = _Recorder(st)

    st.remove_dataset(0)

    assert rec.names() == ["removed", "active_changed"]
    removed, active = rec.typed()
    assert removed == DatasetRemoved(key=ka, dataset=a, old_index=0, new_active_key=None)
    assert active == ActiveDatasetChanged(key=None, dataset=None, previous_key=ka)


def test_a_removal_listener_sees_the_settled_state(state):
    seen = {}

    def on_removed(event):
        seen["gone"] = state.index_of(event.key) is None
        seen["not_listed"] = all(ds is not event.dataset for ds in state.datasets)
        seen["active"] = state.active_key == event.new_active_key
        seen["active_dataset"] = state.active_dataset

    state.dataset_events.removed.connect(on_removed)
    state.set_active(0)
    state.remove_dataset(0)

    assert seen["gone"] and seen["not_listed"] and seen["active"]
    assert seen["active_dataset"] is state.datasets[0]


def test_switching_the_active_dataset(state):
    a, b, c = state.datasets
    state.set_active(0)
    rec = _Recorder(state)

    state.set_active(1)
    state.set_active(1)                       # no change, no event

    assert rec.names() == ["active_changed"]
    (active,) = rec.typed()
    assert active == ActiveDatasetChanged(
        key=state.key_of(b), dataset=b, previous_key=state.key_of(a))


@pytest.mark.parametrize("notify, signal, aspect", [
    ("notify_filter_changed", "filter_changed", ChangeAspect.FILTER),
    ("notify_attributes_changed", "attributes_changed", ChangeAspect.ATTRIBUTES),
    ("notify_roi_selection_changed", "roi_selection_changed", ChangeAspect.ROI_SELECTION),
    ("notify_calibration_changed", "calibration_changed", ChangeAspect.CALIBRATION),
    ("notify_overlay_transform_changed", "overlay_transform_changed",
     ChangeAspect.OVERLAY_TRANSFORM),
])
def test_a_change_carries_the_key_of_the_dataset_it_describes(state, notify, signal, aspect):
    a, b, c = state.datasets
    state.set_active(2)
    rec = _Recorder(state)

    getattr(state, notify)(1)                 # an explicit dataset
    getattr(state, notify)()                  # defaults to the active one

    assert rec.names() == [signal] * 2
    first, second = rec.typed()
    assert first == DatasetChanged(key=state.key_of(b), dataset=b, aspect=aspect)
    assert second == DatasetChanged(key=state.key_of(c), dataset=c, aspect=aspect)


def test_a_change_for_no_dataset_announces_nothing(state):
    rec = _Recorder(state)

    state.notify_filter_changed(7)            # names no open dataset
    state.request_overlay_manual_alignment(7)

    assert rec.names() == []


def test_a_manual_alignment_request_names_its_dataset(state):
    a, b, c = state.datasets
    rec = _Recorder(state)

    state.request_overlay_manual_alignment(1)

    assert rec.names() == ["overlay_manual_alignment_requested"]
    (request,) = rec.typed()
    assert request == ManualAlignmentRequested(key=state.key_of(b), dataset=b)


def test_a_dataset_is_found_by_identity_never_by_equality(state):
    # Datasets compare field by field, and two that agree on their first
    # fields reach a numpy array, whose truth value is ambiguous: a list
    # lookup such as ``ds in state.datasets`` raises for them.
    twin = _dataset("A")                      # same name and file as the first
    with pytest.raises(ValueError):
        _ = twin in state.datasets
    a, b, c = state.datasets
    assert state.index_of_dataset(c) == 2
    assert state.index_of_dataset(twin) is None
    assert state.index_of_dataset(None) is None


def test_the_events_object_belongs_to_the_state():
    st = AppState()
    assert st.dataset_events.parent() is st
