"""A plugin run, written up and replayed from its own declaration.

These exercise the seam end to end: the runner marks the plugin as running,
what the plugin journals is attributed to it, and both consumers -- Generate
Method Text and the macro recorder -- read the same declaration.
"""

from __future__ import annotations

import pytest

from minflux_viewer.core.app_state import AppState
from minflux_viewer.plugins import PluginEntry, method as method_mod


@pytest.fixture
def spec():
    return method_mod.parse_method_spec({
        "title": "Demo analysis",
        "description": "Sites were compared against a null over {band_nm}.",
        "workflow": ["Traces below {min_locs} localization(s) were dropped."],
        "inputs": [
            {"name": "min_locs", "label": "minimum localizations"},
            {"name": "band_nm", "label": "tested band", "unit": "nm"},
        ],
        "outputs": [{"name": "ratio", "label": "observed/null ratio"}],
        "citations": [{"text": "Demo et al. 2026", "url": "https://example.org/d"}],
    })


def _registered(monkeypatch, spec, *, plugin_id="demo.analysis", launch=None):
    entry = PluginEntry(
        name="Demo analysis", tooltip="", launch=launch or (lambda *a, **k: None),
        plugin_id=plugin_id, discovered=True, method=spec,
    )
    monkeypatch.setattr("minflux_viewer.plugins.available", lambda: [entry])
    return entry


def test_a_plugin_run_is_attributed_to_it_and_carries_its_values(spec, monkeypatch):
    _registered(monkeypatch, spec)
    state = AppState()
    mfv = state.mfv

    with mfv.plugin_scope("demo.analysis", "Demo analysis", spec):
        mfv.journal.record("analysis", "Ran the demo",
                           min_locs=10, band_nm=[8.0, 25.0], ratio=2.04)

    event = state.log_history[-1]
    payload = event["method_data"]["mfv_plugin"]
    assert payload["id"] == "demo.analysis"
    assert payload["values"]["min_locs"] == 10
    # The journal entry is also a replayable macro step.
    entry = list(state.journal)[-1]
    assert entry.command == "plugin:demo.analysis"
    assert entry.code.startswith("mfv.plugins.run('demo.analysis', answers=")
    assert "'min_locs': 10" in entry.code

    # A plain script records exactly as before -- no plugin attribution.
    mfv.journal.record("analysis", "Ran a script step", value=1)
    assert state.log_history[-1].get("method_data") is None


def test_generate_method_text_renders_the_plugins_own_declaration(spec, monkeypatch):
    from minflux_viewer.analysis.method_text import generate_method_text

    _registered(monkeypatch, spec)
    state = AppState()
    with state.mfv.plugin_scope("demo.analysis", "Demo analysis", spec):
        state.mfv.journal.record("analysis", "Ran the demo",
                                 min_locs=10, band_nm=[8.0, 25.0], ratio=2.04)

    text = generate_method_text(state, state.log_history)
    assert "Demo analysis" in text
    assert "a null over 8, 25 nm" in text
    assert "below 10 localization(s)" in text
    assert "observed/null ratio: 2.04" in text
    # A declared citation travels with the method.
    assert "Demo et al. 2026" in text
    # ...and the raw journal line is not also dumped verbatim beside it.
    assert "min_locs=10, band_nm=" not in text


def test_a_value_the_run_omitted_is_called_out_in_the_generated_text(
        spec, monkeypatch):
    from minflux_viewer.analysis.method_text import generate_method_text

    _registered(monkeypatch, spec)
    state = AppState()
    with state.mfv.plugin_scope("demo.analysis", "Demo analysis", spec):
        state.mfv.journal.record("analysis", "Ran the demo", min_locs=10)

    text = generate_method_text(state, state.log_history)
    assert "<band_nm not recorded>" in text
    assert "not recorded by this run:" in text
    assert "band_nm" in text.split("not recorded by this run:")[1]


def test_a_plugin_without_a_declaration_still_reports_its_line(monkeypatch):
    """No [method] block is not an error; the run is just described plainly."""
    from minflux_viewer.analysis.method_text import generate_method_text

    _registered(monkeypatch, None)
    state = AppState()
    with state.mfv.plugin_scope("demo.analysis", "Demo analysis", None):
        state.mfv.journal.record("analysis", "Ran the demo", min_locs=10)

    text = generate_method_text(state, state.log_history)
    assert "Ran the demo" in text
    assert "min_locs=10" in text


def test_a_recorded_run_replays_by_pre_answering_the_parameter_dialog(monkeypatch):
    """This is what makes the recorded step reproducible rather than a note."""
    seen = {}

    def launch(state, parent=None):
        # Stand in for the plugin body: it asks, and must get the answers.
        seen["answers"] = state.mfv.ui.ask(
            {"min_locs": 1, "band_nm": 0.0, "untouched": "default"})

    _registered(monkeypatch, None, launch=launch)
    state = AppState()
    state.mfv.plugins.run(
        "demo.analysis", answers={"min_locs": 10, "band_nm": 25.0,
                                  "not_a_field": "ignored"})

    # Keys the plugin asked for come from the recording; one it did not ask
    # for is ignored; one the recording lacks keeps the plugin's default.
    assert seen["answers"] == {
        "min_locs": 10, "band_nm": 25.0, "untouched": "default"}
    # Staging is one-shot, so the next question is still a real question.
    assert state.mfv.take_preset_answers() is None


def test_plugins_namespace_lists_describes_and_writes_up_the_last_run(
        spec, monkeypatch):
    _registered(monkeypatch, spec)
    state = AppState()
    mfv = state.mfv

    rows = mfv.plugins.list()
    assert rows == [{"id": "demo.analysis", "name": "Demo analysis",
                     "menu_path": [], "error": "", "has_method": True}]
    assert mfv.plugins.method("demo.analysis") is spec

    with mfv.plugin_scope("demo.analysis", "Demo analysis", spec):
        mfv.journal.record("analysis", "Ran the demo",
                           min_locs=7, band_nm=[8.0, 25.0], ratio=1.5)

    assert mfv.plugins.last_run("demo.analysis")["min_locs"] == 7
    # With no values given it writes up the run that actually happened.
    assert "below 7 localization(s)" in mfv.plugins.method_text("demo.analysis")
    # ...and an explicit set overrides it, for a what-if write-up.
    assert "below 99 localization(s)" in mfv.plugins.method_text(
        "demo.analysis", {"min_locs": 99})


def test_the_namespace_refuses_what_it_cannot_do(spec, monkeypatch):
    from minflux_viewer.api._base import ApiError

    _registered(monkeypatch, None)
    state = AppState()
    with pytest.raises(ApiError, match="No plugin 'nope'"):
        state.mfv.plugins.run("nope")
    with pytest.raises(ApiError, match="declares no \\[method\\] block"):
        state.mfv.plugins.method_text("demo.analysis")


def test_a_derived_setting_is_written_up_but_never_replayed_as_an_answer():
    """``replay = false`` separates "how it was configured" from "what was asked"."""
    from minflux_viewer.api.journal import _replay_call

    spec = method_mod.parse_method_spec({
        "description": "d",
        "inputs": [
            {"name": "asked"},
            {"name": "derived", "replay": False},
        ],
        "outputs": [{"name": "result"}],
    })
    values = {"asked": 3, "derived": "from the scope", "result": 9.9}

    code = _replay_call("demo.analysis", values, spec)
    assert "'asked': 3" in code
    assert "derived" not in code            # configured, not asked
    assert "result" not in code             # an outcome, not an answer

    # It is still described, so the write-up is complete.
    text, missing = method_mod.render_method_text(spec, values)
    assert "derived: from the scope" in text
    assert missing == []

    # With no declaration at all, every dialog-shaped value is kept: a plugin
    # that ships no [method] block still records a usable replay line.
    bare = _replay_call("demo.analysis", values, None)
    assert "'derived'" in bare and "'result'" in bare


def test_the_hlyb_plugin_declares_a_method_whose_slots_all_resolve():
    """The shipped example must not reference a value nothing declares."""
    from minflux_viewer.plugins import loader

    found = next(
        plugin for plugin in loader.scan_root(loader.app_plugin_dir())
        if plugin.id == "embl.hlyb_pair_analysis")
    assert found.error == ""
    spec = found.manifest.method
    assert spec is not None
    # Every {slot} in the prose is a declared field, so its unit and
    # description are reachable and no slot can render as "not recorded"
    # merely because it was never declared.
    assert spec.slots() <= set(spec.fields)
    assert spec.workflow and spec.description and spec.limitations
    assert {item.name for item in spec.outputs} >= {
        "band_ratio", "excess_centroid_nm"}
