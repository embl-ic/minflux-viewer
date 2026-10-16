'Viewer open signals'

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal

from minflux_viewer.ui.background_tasks import BackgroundTaskSignals as _BackgroundTaskSignals


class _ViewerOpenSignals(_BackgroundTaskSignals):
    """``BackgroundTask`` signals plus a ``(fraction, dataset name)`` progress."""

    progress = pyqtSignal(float, str)
