"""
minflux_viewer.plugins.paraview
================================
Launch ParaView with the active dataset exported as a .vtp file.

Registered as a :class:`minflux_viewer.plugins.PluginEntry` so it appears
under **Plugins → ParaView** in the main menu (item #3 of the Phase-4b
request — moves the entry out of the View menu and lets users configure
the ParaView path from Preferences → Plugin instead of a separate menu
item).

If ParaView is missing or the configured path is wrong, a friendly
dialog explains where to download ParaView and how to point the viewer
at it via Preferences.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PyQt6.QtWidgets import QMessageBox

from .. import PluginEntry, register


def _launch(state, parent=None) -> None:
    """Plugin entry point — delegates to the per-call helper below."""
    open_in_paraview(state, parent)


def open_in_paraview(state, parent=None) -> None:
    """
    Export the active dataset as ``.vtp`` and launch ParaView.

    Tracks the spawned process on ``parent._paraview_procs`` (the main
    window) so that closing the viewer can also terminate ParaView, per
    the existing close-on-exit behaviour. Failures surface as message
    boxes that point to the Preferences page.
    """
    ds = getattr(state, "active_dataset", None)
    if ds is None:
        QMessageBox.warning(
            parent, "ParaView",
            "No dataset is loaded. Open or import a dataset first.",
        )
        return

    # Imports kept local so the plugin doesn't pull in vtk-style helpers
    # at viewer startup.
    from ...utils.paraview_export   import export_to_vtp
    from ...utils.paraview_launcher import (
        launch_paraview, ParaViewNotFoundError, not_found_instructions,
    )

    stem = Path(ds.name).stem.replace("/", "_").replace("\\", "_")
    vtp  = Path(tempfile.gettempdir()) / f"minflux_viewer_{stem}.vtp"

    state.log(f"ParaView plugin: exporting to {vtp}", "INFO")
    try:
        export_to_vtp(ds, vtp)
    except Exception as exc:
        state.log(f"ParaView export failed: {exc}", "ERROR")
        QMessageBox.critical(parent, "ParaView export error", str(exc))
        return

    state.log(
        f"ParaView plugin: launching with {ds.prop.num_loc:,} localisations…",
        "INFO",
    )
    try:
        proc = launch_paraview(vtp, prefs=state.prefs)
        # Track the subprocess on the main window so the close-on-exit
        # logic can terminate it. Falls back to no-op if `parent` doesn't
        # expose the list (e.g. when called from a unit test).
        if proc is not None and parent is not None:
            procs = getattr(parent, "_paraview_procs", None)
            if procs is not None:
                procs.append(proc)
    except ParaViewNotFoundError:
        state.log("ParaView executable not found.", "WARN")
        QMessageBox.warning(
            parent,
            "ParaView not found",
            not_found_instructions() + (
                "<br><br><b>Tip:</b> set the ParaView executable path in "
                "<i>Edit → Preferences → Plugin</i>."
            ),
        )
        return
    except Exception as exc:
        state.log(f"ParaView launch failed: {exc}", "ERROR")
        QMessageBox.critical(
            parent, "ParaView launch error",
            f"{exc}<br><br>"
            "Set the ParaView executable path in "
            "<i>Edit → Preferences → Plugin</i> if it isn't on your PATH.",
        )
        return

    # Journal entry so Generate Method Text knows about this export
    journal = getattr(state, "journal", None)
    if journal is not None:
        try:
            journal.add(
                "export",
                f"Exported '{ds.name}' to VTP and opened in ParaView.",
                vtp_path=str(vtp),
                num_loc=int(ds.prop.num_loc),
            )
        except Exception:
            pass


register(PluginEntry(
    name="ParaView",
    tooltip="Export active dataset as .vtp and open it in ParaView.",
    launch=_launch,
))
