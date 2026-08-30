"""Macro-recorder contract and first core integration slice."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from minflux_viewer.core.processing_journal import ProcessingJournal
from minflux_viewer.core.recorder import GuiClass, RecordedStep, Recorder, emit_script


def test_journal_structured_fields_are_backward_compatible() -> None:
    journal = ProcessingJournal()
    old = journal.add("analysis", "Legacy entry", radius_nm=25)
    new = journal.add(
        "filter",
        "Apply filter",
        code="mfv.data.set_filter([])",
        gui_class="GUI_FREE",
        command="actionFilter",
        replace=True,
    )

    assert old.code is None and old.gui_class is None and old.command is None
    assert old.details == {"radius_nm": 25}
    assert new.gui_class is GuiClass.GUI_FREE
    assert new.details == {"replace": True}


def test_recorder_uses_the_journal_as_its_single_event_stream() -> None:
    journal = ProcessingJournal()
    recorder = Recorder(journal)
    recorder.append("ignored", code="mfv.ui.log('off')")
    assert len(journal) == 0

    recorder.start()
    recorder.append(
        "Apply a filter",
        code="mfv.data.set_filter([])",
        gui_class=GuiClass.GUI_FREE,
        command="actionFilter",
    )
    # "other" is the chatter category and stays out of a recording. A
    # macro-relevant category would surface as a TODO instead -- see
    # test_a_macro_relevant_entry_is_never_silently_dropped.
    journal.add("other", "Ordinary prose-only journal entry")
    recorder.stop()

    assert len(journal) == 2
    assert [step.command for step in recorder.steps()] == ["actionFilter"]
    assert recorder.steps()[0].code == "mfv.data.set_filter([])"


def test_emit_script_filters_only_explicit_gui_only_steps() -> None:
    steps = [
        RecordedStep(
            summary="Filter",
            code="mfv.data.clear_filter()",
            gui_class=GuiClass.GUI_FREE,
            command="actionFilter",
        ),
        RecordedStep(
            summary="Render",
            code="mfv.view.render()",
            gui_class=GuiClass.GUI_ONLY,
            command="actionRender",
        ),
        RecordedStep(summary="Undescribed", command="actionUnknown"),
    ]

    interactive = emit_script(steps, header=False)
    silent = emit_script(steps, silent=True, header=False)

    assert interactive.startswith("import mfv\n")
    assert "mfv.data.clear_filter()" in interactive
    assert "mfv.view.render()" in interactive
    assert "# TODO Undescribed (actionUnknown)" in interactive
    assert "mfv.data.clear_filter()" in silent
    assert "mfv.view.render()" not in silent
    # Unknown classification is kept: silent output must not look complete.
    assert "actionUnknown" in silent


def test_deduplication_prefers_runnable_command_event_but_preserves_data_calls() -> None:
    identity = RecordedStep(summary="Filter", command="actionFilter")
    runnable = RecordedStep(
        summary="Filter",
        command="actionFilter",
        code="mfv.data.clear_filter()",
        gui_class=GuiClass.GUI_FREE,
    )
    script = emit_script([identity, runnable, runnable], header=False)

    # The identity-only instrumentation event is replaced by its runnable form.
    assert "TODO" not in script
    # Repeated executable data operations remain semantic and are not discarded.
    assert script.count("mfv.data.clear_filter()") == 2


def test_recorder_clear_moves_only_its_cursor() -> None:
    journal = ProcessingJournal()
    recorder = Recorder(journal)
    recorder.start()
    recorder.append("first", code="mfv.ui.log('first')", gui_class="gui_free")
    recorder.clear()
    recorder.append("second", code="mfv.ui.log('second')", gui_class="gui_free")

    assert len(journal) == 2
    assert [step.summary for step in recorder.steps()] == ["second"]


def test_gui_class_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Unknown GUI class"):
        RecordedStep(gui_class="sometimes")


def test_command_hook_is_disabled_by_default_and_idempotent(qapp) -> None:
    from PyQt6.QtGui import QAction

    from minflux_viewer.ui.command_meta import CommandMeta
    from minflux_viewer.ui.main_window import MainWindow

    journal = ProcessingJournal()
    recorder = Recorder(journal)
    action = QAction("&Render…")
    meta = CommandMeta(
        summary="Open the render view.",
        category="view",
        gui_class=GuiClass.GUI_ONLY,
        record="mfv.view.render()",
    )
    # A plain class, not SimpleNamespace: the hook takes a weak reference to
    # its window, and SimpleNamespace does not support weakrefs. The real owner
    # is a QWidget, which does.
    class _Owner:
        def __init__(self):
            self._state = SimpleNamespace(recorder=recorder)

        def _record_command_action(self, act, key, command_meta):
            return MainWindow._record_command_action(self, act, key, command_meta)

    owner = _Owner()

    MainWindow._connect_command_recorder(owner, action, "actionRender", meta)
    MainWindow._connect_command_recorder(owner, action, "actionRender", meta)
    action.trigger()
    assert recorder.steps() == []

    recorder.start()
    action.trigger()
    steps = recorder.steps()
    assert len(steps) == 1
    assert steps[0].command == "actionRender"
    assert steps[0].gui_class is GuiClass.GUI_ONLY
    assert steps[0].details["command_text"] == "Render…"


def test_published_record_namespace_controls_and_extends_the_recorder(qapp) -> None:
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.scripting import ScriptError

    state = AppState()
    mfv = state.mfv

    mfv.record.start()
    mfv.record.step(
        "mfv.ui.log('plugin outcome')",
        "gui_result",
        summary="Recorded a plugin outcome",
        answer=42,
    )
    assert mfv.record.steps()[0].gui_class is GuiClass.GUI_RESULT
    assert mfv.record.steps()[0].details == {"answer": 42}
    assert "mfv.ui.log('plugin outcome')" in mfv.record.script(header=False)

    mfv.record.stop()
    mfv.record.step("mfv.ui.log('paused')")
    assert len(mfv.record.steps()) == 1

    with pytest.raises(ScriptError, match="Unknown GUI class"):
        mfv.record.step("pass", "maybe")


def test_record_namespace_rejects_empty_code_even_while_paused(qapp) -> None:
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.scripting import ScriptError

    with pytest.raises(ScriptError, match="non-empty"):
        AppState().mfv.record.step("  ")


def test_the_command_hook_does_not_keep_its_window_alive(qapp) -> None:
    """
    ⚠ The slot must hold **weak** references to the window and the action.

    Capturing them strongly builds ``window → action (its child) → connection →
    lambda → window`` on every one of the ~68 command actions. Python's GC then
    breaks those cycles at an arbitrary later moment, and collecting a QObject
    graph while Qt delivers queued events is this project's documented
    native-abort hazard -- Track C hit the same shape with its destroyed-signal
    callbacks. Measured: the strong form passed every test in isolation and
    reproducibly aborted the mixed suite inside an unrelated LUT-dialog test,
    while the suite was clean without the hook at all.
    """
    import gc
    import weakref

    from PyQt6.QtGui import QAction

    from minflux_viewer.ui.command_meta import CommandMeta
    from minflux_viewer.ui.main_window import MainWindow

    recorder = Recorder(ProcessingJournal())

    class _Owner:
        def __init__(self):
            self._state = SimpleNamespace(recorder=recorder)

        def _record_command_action(self, act, key, command_meta):
            return MainWindow._record_command_action(self, act, key, command_meta)

    owner = _Owner()
    action = QAction("&Render…")
    MainWindow._connect_command_recorder(
        owner, action, "actionRender",
        CommandMeta(summary="Open the render view.", category="view",
                    gui_class=GuiClass.GUI_ONLY, record="mfv.view.render()"),
    )

    ref = weakref.ref(owner)
    del owner
    gc.collect()
    assert ref() is None, (
        "the recorder hook is keeping its window alive; the slot must hold a "
        "weak reference to it"
    )

    # ...and the now-ownerless connection is inert rather than a crash.
    action.trigger()
    assert recorder.steps() == []


# ---------------------------------------------------------------------------
# M2 — every command is classified
# ---------------------------------------------------------------------------

def test_every_command_declares_a_gui_class() -> None:
    """
    Silent mode is only meaningful if every command says whether it is
    presentation. An unclassified command records as a TODO and is kept in the
    silent script, which is safe but useless -- so adding a command means
    adding its class.
    """
    from minflux_viewer.ui.command_meta import COMMAND_META

    missing = sorted(k for k, m in COMMAND_META.items() if m.gui_class is None)
    assert missing == [], (
        "these commands have no gui_class; see the rule above CommandMeta: "
        + ", ".join(missing)
    )


def test_the_classification_is_not_trivially_one_bucket() -> None:
    """A guard against 'classified' meaning 'everything marked the same'."""
    from collections import Counter

    from minflux_viewer.ui.command_meta import COMMAND_META

    spread = Counter(m.gui_class for m in COMMAND_META.values())
    assert len(spread) == 3, spread
    assert all(count >= 5 for count in spread.values()), spread


def test_a_declared_record_call_is_a_published_namespace_call() -> None:
    """
    A record hint is emitted verbatim into a script, so it must name something
    the API actually has.
    """
    from minflux_viewer.api import NAMESPACES
    from minflux_viewer.ui.command_meta import COMMAND_META

    for key, meta in COMMAND_META.items():
        if not meta.record:
            continue
        assert meta.record.startswith("mfv."), f"{key}: {meta.record!r}"
        namespace = meta.record.split(".")[1]
        assert namespace in NAMESPACES, f"{key} names unknown namespace {namespace!r}"


def test_silent_mode_drops_presentation_commands_and_keeps_the_rest(qapp) -> None:
    """End to end over the real registry rather than hand-built steps."""
    from minflux_viewer.ui.command_meta import COMMAND_META

    recorder = Recorder(ProcessingJournal())
    recorder.start()
    for key in ("actionRender", "actionChannelFlatten", "actionAggregate"):
        meta = COMMAND_META[key]
        recorder.record_command(
            key, summary=meta.summary, gui_class=meta.gui_class,
            code=meta.record or None,
        )
    recorder.stop()

    full = recorder.script(header=False)
    silent = recorder.script(silent=True, header=False)

    assert "mfv.view.render()" in full          # GUI_ONLY, with a runnable call
    assert "mfv.view.render()" not in silent    # ...and dropped when silent
    # the two data commands survive as TODOs until their handlers report params
    assert silent.count("# TODO") == 2


# ---------------------------------------------------------------------------
# a recording must never be mysteriously empty
# ---------------------------------------------------------------------------

def test_a_macro_relevant_entry_is_never_silently_dropped() -> None:
    """
    Reported from real use: open the recorder, drop a file, filter it — and the
    window stayed empty. Loading and filtering are exactly what somebody
    records, so showing nothing for them reads as a broken feature. An
    uninstrumented step in a macro-relevant category surfaces as a TODO naming
    what happened; only ``other`` is chatter.
    """
    journal = ProcessingJournal()
    recorder = Recorder(journal)
    recorder.start()
    journal.add("load", "Loaded dataset 'demo'")
    journal.add("other", "some incidental note")
    recorder.stop()

    script = recorder.script(header=False)
    assert "# TODO" in script and "Loaded dataset 'demo'" in script
    assert "not instrumented for recording yet" in script
    assert "incidental note" not in script


def test_a_superseding_step_collapses_to_the_state_the_user_settled_on() -> None:
    """
    Every checkbox toggle and bound edit re-applies the filter. A macro wants
    the filter that was settled on, not each intermediate drag.
    """
    recorder = Recorder(ProcessingJournal())
    recorder.start()
    for hi in (30.0, 35.0, 40.0):
        recorder.append(
            "Filtered 'demo'",
            code=f"mfv.data.set_filter([{{'hi': {hi}}}])",
            gui_class=GuiClass.GUI_RESULT, command="actionFilter",
            category="filter", supersedes_previous=True,
        )
    recorder.stop()

    codes = [s.code for s in _dedup(recorder.steps())]
    assert codes == ["mfv.data.set_filter([{'hi': 40.0}])"]


def _dedup(steps):
    from minflux_viewer.core.recorder import _deduplicate

    return _deduplicate(steps)


def test_two_different_analyses_are_both_kept() -> None:
    """Superseding is opt-in: running two analyses is not one analysis."""
    recorder = Recorder(ProcessingJournal())
    recorder.start()
    recorder.append("A", code="mfv.data.clear_filter()", command="actionA",
                    gui_class=GuiClass.GUI_FREE)
    recorder.append("B", code="mfv.data.clear_filter()", command="actionB",
                    gui_class=GuiClass.GUI_FREE)
    recorder.stop()
    assert len(_dedup(recorder.steps())) == 2
