'Parsed export task'

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QRunnable

from minflux_viewer.msr.session_export import export_parsed_result
from minflux_viewer.plugins.msr_reader.batch_export_signals import _BatchExportSignals


class _ParsedExportTask(QRunnable):
    """Export the already-parsed single file without blocking the Qt thread."""

    def __init__(
        self, parsed, msr_state, out_dir, formats, *, field_selection,
        image_indices, on_conflict="error",
    ) -> None:
        super().__init__()
        self._on_conflict = str(on_conflict or "error")
        self.signals = _BatchExportSignals()
        self._parsed = parsed
        self._msr_state = msr_state
        self._out_dir = str(out_dir)
        self._formats = list(formats)
        self._field_selection = dict(field_selection or {})
        self._image_indices = image_indices
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:  # noqa: N802 - Qt API
        from minflux_viewer.core.task_registry import registry as task_registry

        source = Path(str((self._parsed or {}).get("msr") or "MINFLUX data"))
        failures = []
        handle = task_registry.register(
            f"MSR export — {source.name}", "msr-export", cancel=self.cancel)
        handle.start()

        def progress(done: float, total: float, label: str = "") -> None:
            fraction = done / max(total, 1e-9)
            handle.update(progress=fraction, detail=label or source.name)
            self.signals.progress.emit(fraction, label or source.name)

        try:
            self.signals.message.emit(
                f"[export] writing {source.name} as {', '.join(self._formats)} "
                f"to {self._out_dir}"
            )
            export_parsed_result(
                self._parsed,
                self._msr_state,
                self._out_dir,
                self._formats,
                image_indices=self._image_indices,
                field_selection=self._field_selection,
                log=self.signals.message.emit,
                progress=progress,
                on_conflict=self._on_conflict,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append((source, str(exc)))
            self.signals.message.emit(f"[export error] {source}: {exc}")
        handle.finish("cancelled" if self._cancelled
                      else ("failed" if failures else "done"))
        self.signals.progress.emit(1.0, "")
        self.signals.finished.emit(0 if failures else 1, failures, self._cancelled)
