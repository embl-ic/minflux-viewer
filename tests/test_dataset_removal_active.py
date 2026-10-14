"""Closing a dataset must keep track of which dataset is active.

Closing a dataset listed before the active one shifts it down, and closing
the active one makes another dataset active in its place. Before this, both
happened silently. The active dataset is tracked by key, so a shift keeps it
active without announcing a change; a new active dataset is announced. It must
also be settled before the removal is announced, because removal handlers
read ``active_dataset`` (``add_dataset`` follows the same order for the same
reason).
"""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset


def _dataset(name: str):
    x = np.linspace(0.0, 100.0, 12)
    return build_localization_dataset(name=name, x_nm=x, y_nm=x[::-1])


@pytest.fixture
def state(qapp):
    """A, B, C loaded, with B active; signals recorded in ``state.events``."""
    st = AppState()
    a, b, c = (_dataset(n) for n in ("A", "B", "C"))
    for ds in (a, b, c):
        st.add_dataset(ds)
    st.set_active(1)
    st.events = []
    st.dataset_events.active_changed.connect(
        lambda event: st.events.append(("active", st.index_of(event.key))))

    def on_removed(event):
        # What a removal handler sees of the active dataset.
        active = st.active_dataset
        st.events.append(("removed", event.old_index, active.name if active else None))

    st.dataset_events.removed.connect(on_removed)
    return st


def test_closing_a_dataset_before_the_active_one_follows_it(state):
    state.remove_dataset(0)  # close A

    assert state.active_dataset.name == "B"
    assert state.active_idx == 0
    # Handlers already see B active; B is still the active dataset, so no
    # change is announced -- its new position is the removal's to tell.
    assert state.events == [("removed", 0, "B")]


def test_closing_the_active_dataset_announces_its_successor(state):
    state.remove_dataset(1)  # close B

    assert state.active_dataset.name == "C"
    assert state.active_idx == 1
    assert state.events == [("removed", 1, "C"), ("active", 1)]


def test_closing_a_dataset_after_the_active_one_changes_nothing(state):
    state.remove_dataset(2)  # close C

    assert state.active_dataset.name == "B"
    assert state.active_idx == 1
    assert state.events == [("removed", 2, "B")]


def test_closing_the_active_last_dataset_falls_back_to_the_previous(state):
    state.set_active(2)
    state.events.clear()

    state.remove_dataset(2)  # close C

    assert state.active_dataset.name == "B"
    assert state.events == [("removed", 2, "B"), ("active", 1)]


def test_handlers_never_see_an_out_of_range_active_index(state):
    """C active, close A: the old order left index 2 over a 2-item list."""
    state.set_active(2)
    state.events.clear()

    state.remove_dataset(0)

    assert state.events == [("removed", 0, "C")]
    assert state.active_idx == 1 and state.active_dataset.name == "C"


def test_closing_the_only_dataset_leaves_none_active(qapp):
    st = AppState()
    st.add_dataset(_dataset("A"))
    fired = []
    st.dataset_events.active_changed.connect(fired.append)

    st.remove_dataset(0)

    assert st.active_idx is None
    assert st.active_dataset is None
    assert [event.key for event in fired] == [None]   # nothing is active now
