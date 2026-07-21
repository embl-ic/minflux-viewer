"""
minflux_viewer.plugins.drift_correction
=========================================
**Drift Correction** plugin — time-window auto-correlation drift correction for
MINFLUX localizations, reimplemented from pyMINFLUX (``pyminflux.correct``).

Opens a modeless window where the user estimates the drift trajectory (2D/3D)
and, if satisfied, creates a new drift-corrected dataset (the original is kept
for comparison). See :mod:`minflux_viewer.analysis.drift_correction` for the
algorithm and :mod:`.drift_correction_window` for the UI.
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
    tooltip="Estimate and correct drift by time-window auto-correlation "
            "(2D/3D, reimplemented from pyMINFLUX).",
    launch=_launch,
))
