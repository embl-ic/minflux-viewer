'Batch export signals'

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal


class _BatchExportSignals(QObject):
    progress = pyqtSignal(float, str)          # fraction 0..1, current file name
    message = pyqtSignal(str)                  # a Log line
    finished = pyqtSignal(int, list, bool)     # exported, failures, cancelled
