"""Python Script Editor plugin."""

from __future__ import annotations

from .. import PluginEntry, register


def _launch(state, parent=None) -> None:
    if parent is not None and hasattr(parent, "_show_script_editor"):
        parent._show_script_editor()
        return
    from ...ui.script_editor_window import ScriptEditorWindow

    win = ScriptEditorWindow(state, parent=parent)
    win.show()
    win.raise_()
    win.activateWindow()


register(PluginEntry(
    name="Script Editor",
    tooltip="Open a Python script editor bound to the current MINFLUX Viewer session.",
    launch=_launch,
))
