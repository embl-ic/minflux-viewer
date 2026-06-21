"""Trace Viewer plugin — per-trace inspection (port of the MATLAB trace viewer)."""

from __future__ import annotations

from .. import PluginEntry, register


def _launch(state, parent=None) -> None:
    from .trace_viewer_window import TraceViewerWindow
    from ...ui import modeless

    active = state.active_dataset
    if active is None:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(parent, "Trace Viewer", "Load a dataset first.")
        return
    win = TraceViewerWindow(state, state.active_idx, owner=parent)
    modeless.show_modeless(win, parent)


register(PluginEntry(
    name="Trace Viewer",
    tooltip="Inspect individual molecule traces: per-trace table, time-series, "
            "animated head/tail, tile/overlay views, tail removal and export.",
    launch=_launch,
))
