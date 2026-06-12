"""
minflux_viewer.plugins.msr_reader
==================================
Qt port of the standalone ``minflux_msr_reader`` Tk application.

Provides a parse + preview + export dialog for Abberior Imspector ``.msr``
raw-data files, plus a direct "Open in MINFLUX viewer" shortcut that
hands the parsed ``mfx`` array straight to :class:`AppState` (no
intermediate files).

Registered automatically as a :class:`minflux_viewer.plugins.PluginEntry`.
"""

from __future__ import annotations

from .. import PluginEntry, register


def _launch(state, parent=None) -> None:
    # Import lazily so importing the plugin package does not pull in Qt
    # machinery until the user actually opens the dialog.
    from .msr_reader_dialog import MsrReaderDialog
    dlg = MsrReaderDialog(state, parent=parent)
    # Store on the parent so it survives the scope of this function (the dialog
    # is an unowned top-level window — see MsrReaderDialog.__init__); closing it
    # deletes it via WA_DeleteOnClose.
    if parent is not None:
        parent._plugin_msr_reader_dialog = dlg
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()


register(PluginEntry(
    name="MSR Reader",
    tooltip="Parse, preview, and export Abberior Imspector .msr files.",
    launch=_launch,
))
