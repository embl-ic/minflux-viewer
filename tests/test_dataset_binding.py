"""Windows and registries bound to a dataset by key (ui/dataset_binding.py).

Stabilization Phase 3, slice 3.5. A window's ``_idx`` and the main window's
registries used to hold positions, re-keyed by hand on every close. They now
hold the dataset's key and translate to a position on demand, so nothing needs
re-keying and nothing can be left pointing at the wrong dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.core.dataset_events import ActiveDatasetChanged, ChangeAspect, DatasetChanged
from minflux_viewer.ui.dataset_binding import DatasetIndex, DatasetWindowMap, event_index


def _state(*names):
    st = AppState()
    for name in names:
        x = np.linspace(0.0, 100.0, 12)
        st.add_dataset(build_localization_dataset(name=name, x_nm=x, y_nm=x[::-1]))
    return st


class _Window:
    _idx = DatasetIndex()
    _dataset_idx = DatasetIndex()

    def __init__(self, state, idx):
        self._state = state
        self._idx = idx


def test_a_window_index_follows_its_dataset():
    st = _state("A", "B", "C")
    win = _Window(st, 2)
    c = st.datasets[2]

    st.remove_dataset(0)
    assert win._idx == 1 and st.datasets[win._idx] is c
    assert win._dataset_idx == 1                     # both names, one binding

    st.remove_dataset(1)                             # its own dataset closes
    assert win._idx is None


def test_assigning_an_index_rebinds_the_window():
    st = _state("A", "B")
    win = _Window(st, 0)
    win._idx = 1
    st.remove_dataset(0)
    assert win._idx == 0 and st.datasets[0].name == "B"
    win._idx = None
    assert win._idx is None and win._dataset_key is None


def test_the_map_keeps_each_window_with_its_dataset():
    st = _state("A", "B", "C")
    windows = DatasetWindowMap(st)
    windows[0], windows[1], windows[2] = "wa", "wb", "wc"

    st.remove_dataset(0)

    assert windows[0] == "wb" and windows[1] == "wc"
    assert windows.get(0) == "wb" and 1 in windows and 2 not in windows
    assert sorted(windows.items()) == [(0, "wb"), (1, "wc")]
    assert sorted(windows.values()) == ["wb", "wc"] and len(windows) == 2
    # The closed dataset's entry is not offered at any position...
    assert "wa" not in windows.values()
    # ... and is removed by its key.
    assert windows.pop_key(st.key_at(0)) == "wb"


def test_the_map_answers_like_a_dict_for_missing_positions():
    st = _state("A")
    windows = DatasetWindowMap(st)
    assert windows.get(5) is None and windows.get(5, "x") == "x"
    assert 5 not in windows
    assert windows.pop(5, None) is None
    with pytest.raises(KeyError):
        windows[5]
    with pytest.raises(KeyError):
        windows[5] = "w"
    assert not windows


def test_the_map_has_a_slot_for_no_dataset():
    st = _state()
    windows = DatasetWindowMap(st)
    windows[None] = "filter"
    assert windows.get(None) == "filter" and list(windows) == [None]
    assert windows.pop(None) == "filter"


def test_the_map_takes_keys_too_and_discards_by_window():
    st = _state("A", "B")
    windows = DatasetWindowMap(st)
    windows[st.key_at(0)] = "wa"
    windows[1] = "wb"
    assert windows[0] == "wa" and windows[st.key_at(1)] == "wb"
    windows.discard_where(lambda w: w == "wa")
    assert windows.items() == [(1, "wb")]
    windows.clear()
    assert len(windows) == 0


def test_event_index_resolves_events_and_passes_indices_through():
    st = _state("A", "B")
    kb, b = st.key_at(1), st.datasets[1]
    event = DatasetChanged(key=kb, dataset=b, aspect=ChangeAspect.FILTER)
    assert event_index(st, event) == 1
    st.remove_dataset(0)
    assert event_index(st, event) == 0
    assert event_index(st, ActiveDatasetChanged(key=None, dataset=None, previous_key=kb)) is None
    assert event_index(st, 3) == 3 and event_index(st, None) is None
