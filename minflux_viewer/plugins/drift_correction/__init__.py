"""
minflux_viewer.plugins.drift_correction
=========================================
**Drift Correction** plugin — MINFLUX drift correction by the instrument's
fiducial (MBM) beads, as Imspector computes it, and/or fiducial-free by
time-window cross-correlation of the localizations (Ostersehlt et al. 2022).

Opens a modeless window where the user estimates the drift (2D/3D) and, if
satisfied, creates a new drift-corrected dataset (the original is kept for
comparison). See :mod:`minflux_viewer.analysis.bead_drift`,
:mod:`minflux_viewer.analysis.drift_correction` and
:mod:`.drift_correction_window`.
"""

from __future__ import annotations

from .. import PluginEntry, register


def _launch(state, parent=None) -> None:
    from ...ui.modeless import show_modeless
    from .drift_correction_window import DriftCorrectionWindow

    win = DriftCorrectionWindow(state, owner=parent)
    show_modeless(win, parent)


register(PluginEntry(
    name="Drift Correction",
    tooltip="Correct drift with the fiducial beads (as Imspector) and/or by "
            "time-window cross-correlation of the localizations (2D/3D).",
    launch=_launch,
))
