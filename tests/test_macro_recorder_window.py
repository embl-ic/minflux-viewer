"""Modeless Macro Recorder plugin window."""

from __future__ import annotations

from types import SimpleNamespace

from PyQt6.QtWidgets import QWidget

from minflux_viewer.core.recorder import GuiClass


def _state():
    from minflux_viewer.core.app_state import AppState

    return AppState()


def test_window_is_parentless_starts_recording_and_renders_silent_mode(qapp) -> None:
    from minflux_viewer.ui.macro_recorder_window import MacroRecorderWindow

    state = _state()
    owner = QWidget()
    window = MacroRecorderWindow(state, owner=owner)
    assert window.parent() is None
    assert state.recorder.enabled is True

    state.recorder.append(
        "Open render",
        code="mfv.view.render()",
        gui_class=GuiClass.GUI_ONLY,
        command="actionRender",
    )
    state.recorder.append(
        "Plugin result",
        code="mfv.ui.log('done')",
        gui_class=GuiClass.GUI_RESULT,
    )
    window._refresh(force=True)
    assert "mfv.view.render()" in window.code_view.toPlainText()

    window.silent_checkbox.setChecked(True)
    assert "mfv.view.render()" not in window.code_view.toPlainText()
    assert "mfv.ui.log('done')" in window.code_view.toPlainText()

    window.close()
    assert state.recorder.enabled is False
    owner.close()


def test_create_script_uses_the_public_editor_handoff(qapp) -> None:
    from minflux_viewer.ui.macro_recorder_window import MacroRecorderWindow

    received = {}

    class Editor:
        def set_script(self, text, *, source_name=None):
            received.update(text=text, source_name=source_name)

        def show(self):
            received["shown"] = True

        def raise_(self):
            pass

        def activateWindow(self):
            pass

    owner = SimpleNamespace(show_script_editor=lambda: Editor())
    state = _state()
    window = MacroRecorderWindow(state, owner=owner)
    state.recorder.append(
        "A plugin step",
        code="mfv.ui.log('recorded')",
        gui_class="gui_free",
    )

    window._create_script()

    assert received["source_name"] == "Recorded Macro"
    assert "mfv.ui.log('recorded')" in received["text"]
    assert received["shown"] is True
    window.close()


def test_save_adds_a_python_suffix_and_logs(qapp, tmp_path, monkeypatch) -> None:
    import minflux_viewer.ui.macro_recorder_window as recorder_ui

    state = _state()
    window = recorder_ui.MacroRecorderWindow(state)
    state.recorder.append("Step", code="pass", gui_class="gui_free")
    target = tmp_path / "recording"
    monkeypatch.setattr(
        recorder_ui.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(target), "Python scripts (*.py)"),
    )

    window._save()

    saved = target.with_suffix(".py")
    assert saved.is_file()
    assert "pass" in saved.read_text(encoding="utf-8")
    assert "Saved recorded macro" in state.log_history[-1]["message"]
    window.close()


def test_builtin_plugin_registry_includes_macro_recorder() -> None:
    from minflux_viewer import plugins

    plugins.ensure_loaded()
    entry = next(item for item in plugins.available() if item.name == "Macro Recorder")
    assert "record" in entry.tooltip.lower()
    assert "python" in entry.keywords
