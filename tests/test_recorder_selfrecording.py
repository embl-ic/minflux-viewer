"""
Macro recorder §3.3 — the published namespaces record themselves.

A script or plugin calling ``mfv`` needs no recorder-specific code. What is
asserted here is what keeps the output *usable* rather than merely complete:
reads are not recorded, arguments are round-trippable or honestly refused,
returned objects become variables, and a plugin is one step rather than its
internals.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6.QtWidgets")


@pytest.fixture
def mfv(qapp):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.core.dataset import build_localization_dataset
    from minflux_viewer.scripting import install_runtime_module

    state = AppState()
    rng = np.random.default_rng(0)
    state.add_dataset(build_localization_dataset(
        name="demo",
        x_nm=rng.normal(0, 100, 80), y_nm=rng.normal(0, 100, 80),
        z_nm=rng.normal(0, 20, 80),
        attrs={"efo": rng.uniform(1e4, 5e4, 80)},
    ))
    facade = state.mfv
    install_runtime_module(facade)
    return facade


def _codes(facade):
    return [s.code for s in facade.record.steps() if s.code]


# ---------------------------------------------------------------------------

def test_a_read_is_not_recorded(mfv):
    """A script reading forty attributes must not emit forty lines."""
    mfv.record.start()
    mfv.data.attr("efo")
    mfv.data.loc()
    mfv.data.attr_names()
    mfv.data.filter_specs()
    mfv.record.stop()

    assert mfv.record.steps() == []


def test_a_mutating_call_records_itself_as_runnable_code(mfv):
    mfv.record.start()
    mfv.data.set_filter([
        {"attribute": "efo", "mode": "per loc", "lo": 0.0, "hi": 1e9},
    ])
    mfv.record.stop()

    code = _codes(mfv)
    assert len(code) == 1
    assert code[0].startswith("mfv.data.set_filter([")
    assert "'attribute': 'efo'" in code[0]


def test_a_returned_object_becomes_a_variable(mfv):
    """
    Without binding, a later call taking that object could not be written down
    at all.
    """
    mfv.record.start()
    mfv.roi.add("rectangle", {"bounds": [0.0, 0.0, 10.0, 10.0]})
    mfv.roi.add("oval", {"bounds": [5.0, 5.0, 10.0, 10.0]})
    mfv.record.stop()

    code = _codes(mfv)
    assert code[0].startswith("roi = mfv.roi.add(")
    assert code[1].startswith("roi_2 = mfv.roi.add(")


def test_an_argument_that_cannot_be_written_records_an_honest_todo(mfv):
    """
    A line that looks runnable and is not is worse than a gap. The reason must
    reach the script, so the gap is fixable rather than mysterious.
    """
    mfv.record.start()
    mfv.data.add_attr("big", np.arange(80.0))
    mfv.record.stop()

    assert _codes(mfv) == []
    script = mfv.record.script(header=False)
    assert "# TODO" in script
    assert "80 values" in script and "too large" in script


def test_a_small_array_is_written_as_a_literal(mfv):
    from minflux_viewer.api._recording import MAX_INLINE_ARRAY

    mfv.record.start()
    mfv.plot.line(np.arange(4.0), np.arange(4.0))
    mfv.record.stop()

    code = _codes(mfv)
    assert code and "[0.0, 1.0, 2.0, 3.0]" in code[0]
    assert MAX_INLINE_ARRAY >= 4


def test_a_dataset_argument_addresses_itself_by_name(mfv):
    """A name survives into another session; a Python object does not."""
    ds = mfv.data.active()
    mfv.record.start()
    mfv.data.clear_filter(dataset=ds)
    mfv.record.stop()

    assert _codes(mfv) == ["mfv.data.clear_filter(dataset=mfv.data.get('demo'))"]


def test_a_roi_from_before_the_recording_is_refused_not_faked(mfv):
    """
    Its id is a per-session UUID, so writing it into a script would produce a
    line that runs and refers to nothing.
    """
    roi = mfv.roi.add("rectangle", {"bounds": [0.0, 0.0, 10.0, 10.0]})

    mfv.record.start()
    mfv.roi.remove(roi)
    mfv.record.stop()

    assert _codes(mfv) == []
    assert "before recording started" in mfv.record.script(header=False)


def test_view_calls_are_gui_only_and_vanish_in_silent_mode(mfv):
    mfv.record.start()
    mfv.data.clear_filter()
    mfv.record.stop()
    from minflux_viewer.api._recording import RECORDABLE
    from minflux_viewer.core.recorder import GuiClass

    assert RECORDABLE["view.render"][0] is GuiClass.GUI_ONLY
    assert RECORDABLE["data.clear_filter"][0] is GuiClass.GUI_FREE


def test_suppression_makes_a_plugin_one_step_not_its_internals(mfv):
    """
    A plugin's own ``ctx.journal.record()`` is the step worth keeping; the
    twenty API calls inside it are not.
    """
    mfv.record.start()
    with mfv.calls.suppress():
        mfv.data.clear_filter()
        mfv.roi.add("oval", {"bounds": [0.0, 0.0, 4.0, 4.0]})
    mfv.data.clear_filter()
    mfv.record.stop()

    assert len(_codes(mfv)) == 1


def test_nested_calls_record_only_the_call_the_user_made(mfv):
    """``roi.crop`` uses other namespace calls internally; one step, not three."""
    mfv.record.start()
    roi = mfv.roi.add("rectangle", {"bounds": [-200.0, -200.0, 400.0, 400.0]})
    mfv.roi.crop(roi, exact_shape=False)
    mfv.record.stop()

    code = _codes(mfv)
    assert len(code) == 2, code
    assert code[1].startswith("cropped = mfv.roi.crop(roi")


def test_the_recording_replays_and_reproduces_the_session(mfv):
    """
    The property that makes this a macro recorder rather than a log formatter.
    """
    efo = mfv.data.attr("efo", filtered=False)
    cut = float(np.median(efo))

    mfv.record.start()
    mfv.data.set_filter([
        {"attribute": "efo", "mode": "per loc", "lo": cut, "hi": float(efo.max())},
    ])
    mfv.roi.add("rectangle", {"bounds": [-500.0, -500.0, 1000.0, 1000.0]})
    mfv.record.stop()

    script = mfv.record.script(silent=True)
    expected = int(mfv.data.filter_mask().sum())

    mfv.data.clear_filter()
    for record in list(mfv.roi.list()):
        mfv.roi.remove(record)
    assert int(mfv.data.filter_mask().sum()) != expected

    with mfv.calls.suppress():           # replay must not record itself
        exec(compile(script, "recorded.py", "exec"), {})

    assert int(mfv.data.filter_mask().sum()) == expected
    assert len(mfv.roi.list()) == 1


def test_two_facades_record_independently(qapp):
    """
    The wrapper lives on the namespace *class*, so it must take its policy from
    the namespace it is called on. Two sessions must not record into each
    other -- and one recording must not be silenced by the other's state.
    """
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.core.dataset import build_localization_dataset

    def _session(name):
        state = AppState()
        state.add_dataset(build_localization_dataset(
            name=name, x_nm=np.zeros(5), y_nm=np.zeros(5)))
        return state.mfv

    first, second = _session("one"), _session("two")

    first.record.start()                    # only the first is recording
    first.data.clear_filter()
    second.data.clear_filter()
    first.record.stop()

    assert [s.code for s in first.record.steps() if s.code] == [
        "mfv.data.clear_filter()"
    ]
    assert second.record.steps() == []
