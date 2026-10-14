"""A Log event records which dataset it was about, as it was then.

Stabilization Phase 3, slice 3.3. A Log event used to remember its dataset by
the index it was written with, so closing an earlier dataset re-attributed it:
Generate Method Text could describe one dataset with another's metadata. An
event now records the dataset's key and a snapshot of what the method-text rules
read, taken when it is written. An event without a key (one built by hand, or by
code older than this) is matched only to the single open dataset with its exact
name, never by position.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from minflux_viewer.analysis import method_text as mt
from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.core.log_attribution import dataset_summary


def _dataset(name: str, n: int = 12, *, version: str = "m2410"):
    x = np.linspace(0.0, 100.0, n)
    ds = build_localization_dataset(name=name, x_nm=x, y_nm=x[::-1])
    ds.metadata["source_version"] = version
    ds.metadata["msr_dataset_did"] = f"did-{name}"
    return ds


def _load_event(state, name):
    marker = f"Loaded dataset '{name}':"
    (event,) = [e for e in state.log_history if e["message"].startswith(marker)]
    return event


@pytest.fixture
def state():
    st = AppState()
    for name, n in (("A", 12), ("B", 20), ("C", 30)):
        st.add_dataset(_dataset(name, n))
    return st


def test_an_event_records_its_dataset_key_and_a_snapshot(state):
    b = state.datasets[1]
    event = _load_event(state, "B")

    assert event["dataset_key"] == state.key_of(b)
    snap = event["dataset_snapshot"]
    assert snap["label"] == str(state.key_of(b))
    assert snap["name"] == "B"
    assert snap["did"] == "did-B"
    assert snap["valid_num_loc"] == 20 and snap["source_version"] == "m2410"
    assert snap["overlay_members"] is None


def test_an_event_without_a_dataset_records_none():
    st = AppState()
    st.log("nothing is open")
    (event,) = st.log_history
    assert event.get("dataset_key") is None
    assert event.get("dataset_snapshot") is None


def test_an_event_keeps_its_dataset_after_an_earlier_one_closes(state):
    b = state.datasets[1]
    event = _load_event(state, "B")

    state.remove_dataset(0)

    assert mt._ds_for(state, event) is b
    text = mt.generate_method_text(state, [event])
    assert "'B'" in text and "20 valid localizations" in text


def test_an_event_of_a_closed_dataset_names_no_open_dataset(state):
    event = _load_event(state, "A")

    state.remove_dataset(0)

    assert mt._ds_for(state, event) is None
    # ... and is still described as it was, not as whatever took its place.
    text = mt.generate_method_text(state, [event])
    assert "'A'" in text and "12 valid localizations" in text
    assert "20 valid localizations" not in text


def test_the_snapshot_is_what_was_true_when_the_event_was_written(state):
    event = _load_event(state, "B")
    state.datasets[1].metadata["source_version"] = "m2205"

    text = mt.generate_method_text(state, [event])
    assert "version m2410" in text and "m2205" not in text


def test_the_log_does_not_keep_a_closed_dataset_alive(state):
    ref = weakref.ref(state.datasets[0])
    state.log("about A", dataset_idx=0)

    state.remove_dataset(0)
    gc.collect()

    assert ref() is None
    assert any(e["message"] == "about A" for e in state.log_history)


def test_a_dataset_name_is_reported_as_it_was_logged(state):
    event = _load_event(state, "C")
    state.datasets[2].file.name = "renamed"
    assert mt._ds_name(state, event) == "C"


# ------------------------------------------------- events without a key
def _keyless(name, idx):
    return {"message": f"Loaded dataset '{name}': x", "dataset_idx": idx,
            "dataset_name": name}


def test_a_keyless_event_is_never_matched_by_its_index(state):
    # Index 0 holds A; an event about a dataset called 'Z' must not become A.
    assert mt._ds_for(state, _keyless("Z", 0)) is None


def test_a_keyless_event_matches_the_one_dataset_with_its_name(state):
    assert mt._ds_for(state, _keyless("C", 0)) is state.datasets[2]


def test_a_keyless_event_with_an_ambiguous_name_matches_nothing(state):
    state.datasets[2].file.name = "B"
    assert mt._ds_for(state, _keyless("B", 1)) is None


def test_dataset_summary_reads_the_fields_the_rules_need():
    ds = _dataset("D", 15)
    ds.metadata["overlay_reference"] = "D"
    summary = dataset_summary(ds)
    assert summary["name"] == "D" and summary["num_loc"] == 15
    assert summary["valid_num_loc"] == 15 and summary["num_dim"] in (2, 3)
    assert summary["overlay_reference"] == "D"


# ------------------------------------------------- overlays
def _overlay_state():
    st = AppState()
    for order, name in enumerate(("ch1", "ch2", "other"), start=1):
        ds = _dataset(name, 10 * order)
        if name != "other":
            ds.state["overlay_id"] = "g"
            ds.state["overlay_order"] = order
            ds.metadata["msr_dataset_name"] = name
            ds.metadata["overlay_reference"] = "ch1"
            ds.metadata["overlay_alignment_mode"] = "none"
        st.add_dataset(ds)
    return st


def test_an_overlay_event_records_every_channel():
    st = _overlay_state()
    st.log("MSR overlay loaded from 'f.msr': 2 channel(s) [ch1, ch2]; reference "
           "channel 'ch1'; channel alignment: none.", dataset_idx=0)
    event = st.log_history[-1]
    members = event["dataset_snapshot"]["overlay_members"]
    assert [m["name"] for m in members] == ["ch1", "ch2"]
    assert [m["label"] for m in members] == [str(st.key_at(0)), str(st.key_at(1))]

    st.remove_dataset(1)                      # one channel closes later
    text = mt.generate_method_text(st, [event])
    assert "comprising 2 channel(s)" in text
    assert "'ch2' (20 valid localizations" in text


# ------------------------------------------------- the event picker
def test_the_event_picker_prechecks_by_dataset_not_by_position(state):
    from minflux_viewer.plugins.generate_method_text import MethodEventSelectionDialog

    a_event, b_event = _load_event(state, "A"), _load_event(state, "B")
    state.set_active(1)                       # B is active
    state.remove_dataset(0)                   # B moves to index 0, where A was

    keys = MethodEventSelectionDialog._auto_check_keys(state)
    assert b_event["dataset_key"] in keys
    assert a_event["dataset_key"] not in keys
