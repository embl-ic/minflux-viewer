'Parse worker'

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal


class _ParseWorker(QThread):
    progress = pyqtSignal(str)
    succeeded = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, msr_path: str, tmp_dir: str):
        super().__init__()
        self._msr_path = msr_path
        self._tmp_dir = tmp_dir

    def run(self):
        from minflux_viewer.core.task_registry import registry as task_registry

        # No cancel=: parsing one .msr is a single pass with no checkpoint to
        # honour a stop at, so the monitor lists it as not stoppable.
        handle = task_registry.register(
            f"Parse {Path(self._msr_path).name}", "msr-parse")
        handle.start()
        try:
            from minflux_viewer.msr import parse_msr_general
            result = parse_msr_general(
                self._msr_path,
                self._tmp_dir,
                log=lambda msg: (handle.update(detail=str(msg)[:120]),
                                 self.progress.emit(msg))[1],
            )
            handle.finish("done")
            self.succeeded.emit(result)
        except Exception as exc:
            handle.finish("failed", detail=str(exc))
            self.failed.emit(str(exc))
