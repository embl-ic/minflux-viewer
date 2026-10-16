'Export estimate signals'

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal


class _ExportEstimateSignals(QObject):
    done = pyqtSignal(int, object)
