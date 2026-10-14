"""Every open dataset has a key that does not change while it is open.

Datasets were identified by their position in ``AppState.datasets``, and that
position shifts whenever an earlier dataset closes. Stabilization Phase 3, slice
3.1, gives each dataset a :class:`DatasetKey` and tracks the active dataset by
key. The index interface keeps working, derived from the key, so nothing that
reads it changes. Keys are internal: not published to scripts, never written to
a file.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.core.dataset_key import DatasetEntry, DatasetKey


def _dataset(name: str):
    x = np.linspace(0.0, 100.0, 12)
    return build_localization_dataset(name=name, x_nm=x, y_nm=x[::-1])


@pytest.fixture
def state():
    st = AppState()
    for name in "ABC":
        st.add_dataset(_dataset(name))
    return st


def _keys(state):
    return [entry.key for entry in state.entries]


def test_a_key_is_a_type_of_its_own():
    key = DatasetKey(7)
    assert str(key) == "ds-7"
    assert key == DatasetKey(7) and hash(key) == hash(DatasetKey(7))
    # Never mistaken for an index or a dataset name.
    assert key != 7 and key != "ds-7"
    with pytest.raises(TypeError):
        [0][key]
    with pytest.raises(dataclasses.FrozenInstanceError):
        key.value = 8
    for bad in (0, -1):
        with pytest.raises(ValueError):
            DatasetKey(bad)
    for bad in (True, 1.0, "1", None):
        with pytest.raises(TypeError):
            DatasetKey(bad)


def test_every_dataset_gets_its_own_key_in_list_order(state):
    keys = _keys(state)
    assert len(set(keys)) == 3
    assert all(isinstance(entry, DatasetEntry) for entry in state.entries)
    assert all(entry.dataset is ds for entry, ds in zip(state.entries, state.datasets))
    for index, ds in enumerate(state.datasets):
        key = keys[index]
        assert state.key_at(index) == key
        assert state.index_of(key) == index
        assert state.dataset_for(key) is ds
        assert state.key_of(ds) == key
    # A position past either end names no dataset; Python's negative indexing
    # must not quietly pick one from the far end.
    assert state.key_at(3) is None and state.key_at(-1) is None


def test_a_key_survives_the_close_of_an_earlier_dataset(state):
    b, c = state.datasets[1], state.datasets[2]
    kb, kc = state.key_of(b), state.key_of(c)

    state.remove_dataset(0)

    assert state.key_of(b) == kb and state.key_of(c) == kc
    assert state.index_of(kb) == 0 and state.index_of(kc) == 1


def test_a_closed_datasets_key_is_gone_and_never_reused(state):
    a = state.datasets[0]
    ka = state.key_of(a)

    state.remove_dataset(0)
    assert state.index_of(ka) is None
    assert state.dataset_for(ka) is None
    assert state.key_of(a) is None

    state.add_dataset(_dataset("D"))
    assert ka not in _keys(state)
    assert len(set(_keys(state))) == 3


def test_the_active_dataset_is_tracked_by_key(state):
    state.set_active(2)
    c, kc = state.active_dataset, state.active_key
    assert kc == state.key_of(c)

    state.remove_dataset(0)

    assert state.active_key == kc
    assert state.active_dataset is c
    assert state.active_idx == 1


def test_closing_the_active_dataset_hands_its_place_on(state):
    state.set_active(1)
    state.remove_dataset(1)           # B goes; C takes its place
    assert state.active_dataset.name == "C"
    assert state.active_key == state.key_of(state.active_dataset)

    state.remove_dataset(1)           # C was last: the one before it
    assert state.active_dataset.name == "A"

    state.remove_dataset(0)
    assert state.active_key is None
    assert state.active_idx is None and state.active_dataset is None


def test_adding_makes_the_new_dataset_active_under_its_new_key(state):
    d = _dataset("D")
    index = state.add_dataset(d)
    assert index == 3
    assert state.active_key == state.key_of(d) == state.key_at(3)


def test_reloading_the_same_file_keeps_its_key(state):
    keys = _keys(state)
    state.set_active(2)

    index = state.add_dataset(_dataset("A"))     # same file as the first one

    assert index == 0
    assert _keys(state) == keys
    assert state.active_key == keys[0]


def test_code_that_assigns_the_index_still_selects_a_key():
    # Tests set it directly; the property maps it onto the key at that slot.
    st = AppState()
    st.datasets.append(_dataset("A"))
    st._active_idx = 0
    assert st.active_dataset.name == "A"
    assert st.active_key == st.key_at(0)
    with pytest.raises(IndexError):
        st._active_idx = 1


def test_a_list_changed_from_outside_is_reconciled(state):
    # Only tests do this. Keys follow the datasets by identity, and a replaced
    # active dataset hands its slot on exactly as the index used to.
    a, b, c = state.datasets
    ka, kb, kc = _keys(state)
    state.set_active(0)

    d = _dataset("D")
    state.datasets[0] = d

    kd = state.key_of(d)
    assert kd is not None and kd not in (ka, kb, kc)
    assert state.key_of(b) == kb and state.key_of(c) == kc
    assert state.key_of(a) is None
    assert state.active_dataset is d and state.active_key == kd

    e = _dataset("E")
    state.datasets.append(e)
    assert state.index_of(state.key_of(e)) == 3


def test_the_same_dataset_listed_twice_gets_two_keys(state):
    a = state.datasets[0]
    state.datasets.append(a)
    keys = _keys(state)
    assert len(set(keys)) == len(keys) == 4
