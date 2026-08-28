from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QObject, pyqtSignal

from minflux_viewer.ui.background_tasks import (
    retire_background_tasks,
    shared_thread_pool,
)


class _Signals(QObject):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)


class _Task:
    def __init__(self) -> None:
        self.signals = _Signals()
        self.cancelled = False
        self.interrupted = False

    def cancel(self) -> None:
        self.cancelled = True

    def requestInterruption(self) -> None:
        self.interrupted = True


def test_retire_background_tasks_cancels_and_detaches_callbacks() -> None:
    task = _Task()
    delivered = []
    task.signals.done.connect(delivered.append)

    retire_background_tasks([task], signal_names=("done", "failed"))
    task.signals.done.emit("late result")

    assert task.cancelled is True
    assert task.interrupted is True
    assert delivered == []


def test_named_thread_pools_are_process_owned_and_reused() -> None:
    first = shared_thread_pool("test-background-pool", max_threads=1)
    second = shared_thread_pool("test-background-pool", max_threads=2)
    other = shared_thread_pool("test-background-pool-other", max_threads=1)

    assert first is second
    assert other is not first
    assert first.maxThreadCount() == 2
