'Export estimate task'

from __future__ import annotations

from PyQt6.QtCore import QRunnable

from minflux_viewer.plugins.msr_reader.export_estimate_signals import _ExportEstimateSignals


class _ExportEstimateTask(QRunnable):
    """Sample parsed arrays off-thread so the size hint never stalls the dialog."""

    def __init__(self, generation: int, components) -> None:
        super().__init__()
        self.signals = _ExportEstimateSignals()
        self._generation = int(generation)
        self._components = list(components)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:  # noqa: N802 - Qt API
        try:
            from minflux_viewer.core.export_size import estimate_export_sizes

            result = estimate_export_sizes(self._components)
        except Exception as exc:  # the estimate must never block a real export
            result = {"error": str(exc)}
        if not self._cancelled:
            self.signals.done.emit(self._generation, result)
