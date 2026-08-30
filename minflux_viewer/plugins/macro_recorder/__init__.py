"""Fiji-style Python macro recorder plugin."""

from __future__ import annotations

from .. import PluginEntry, register


def _launch(state, parent=None) -> None:
    show_recorder = getattr(parent, "show_macro_recorder", None)
    if callable(show_recorder):
        show_recorder()
        return

    from ...ui.macro_recorder_window import MacroRecorderWindow
    from ...ui.modeless import show_modeless

    show_modeless(MacroRecorderWindow(state, owner=parent), parent)


register(
    PluginEntry(
        name="Macro Recorder",
        tooltip="Record viewer actions as a reusable mfv Python script.",
        launch=_launch,
        keywords=("macro", "record", "automation", "fiji", "python"),
    )
)
