'Batch export task'

from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import QRunnable

from minflux_viewer.msr.session import state_namespace_for
from minflux_viewer.msr.session_export import export_parsed_result
from minflux_viewer.plugins.msr_reader.batch_export_signals import _BatchExportSignals


class _BatchExportTask(QRunnable):
    """Parse and export a list of ``.msr`` files off the UI thread.

    Cancellation is checked between files: one file's parse+export is a single
    uninterruptible unit, so a cancel takes effect at the next file boundary
    rather than leaving a half-written export behind.
    """

    def __init__(self, files, targets, formats, tmp_dir, *, mfx_sel, mbm_sel,
                 image_names, field_selection, image_indices=frozenset(),
                 on_conflict="error"):
        super().__init__()
        self._on_conflict = str(on_conflict or "error")
        self.signals = _BatchExportSignals()
        self._files = list(files)
        self._targets = dict(targets)
        self._formats = list(formats)
        self._tmp = tmp_dir
        self._mfx_sel = mfx_sel
        self._mbm_sel = mbm_sel
        self._image_names = image_names
        # Only consulted when there is no name filter: ``None`` = every image
        # series of every file, empty = none.
        self._image_indices = image_indices
        self._field_selection = dict(field_selection or {})
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self):
        from minflux_viewer.core.task_registry import registry as task_registry
        from minflux_viewer.msr import parse_msr_general

        total = max(len(self._files), 1)
        exported = 0
        failures: list[tuple[Path, str]] = []
        log = self.signals.message.emit
        handle = task_registry.register(
            f"MSR batch export ({len(self._files)} file(s))", "msr-export",
            cancel=self.cancel)
        handle.start()
        for index, msr_path in enumerate(self._files):
            if self._cancelled:
                break
            base = index / total
            handle.update(progress=base, detail=msr_path.name)
            self.signals.progress.emit(base, msr_path.name)
            sub_out = self._targets[msr_path]
            try:
                os.makedirs(sub_out, exist_ok=True)
                log(f"[batch {index + 1}/{len(self._files)}] parse & export: "
                    f"{msr_path} -> {sub_out}")
                parsed = parse_msr_general(str(msr_path), self._tmp, log=log)
                export_parsed_result(
                    parsed, state_namespace_for(parsed), str(sub_out),
                    self._formats,
                    mfx_sel_global=self._mfx_sel, mbm_sel_global=self._mbm_sel,
                    image_names=self._image_names,
                    image_indices=self._image_indices,
                    field_selection=self._field_selection,
                    log=log,
                    progress=lambda d, t, label="", _b=base: self.signals.progress.emit(
                        _b + (d / max(t, 1e-9)) / total,
                        f"{msr_path.name} · {label}" if label else msr_path.name),
                    on_conflict=self._on_conflict,
                )
                exported += 1
            except Exception as exc:                           # noqa: BLE001
                failures.append((msr_path, str(exc)))
                log(f"[batch error] {msr_path}: {exc}")
        handle.finish("cancelled" if self._cancelled
                      else ("failed" if failures else "done"),
                      detail=f"{exported}/{len(self._files)} file(s)")
        self.signals.progress.emit(1.0, "")
        self.signals.finished.emit(exported, failures, self._cancelled)
