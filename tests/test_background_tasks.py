from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QObject, pyqtSignal

from minflux_viewer.ui.background_tasks import (
    BackgroundTask,
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


def test_cancelled_particle_task_emits_terminal_signal_without_work() -> None:
    from minflux_viewer.ui.particle_average_dialog import _AverageTask

    called = []
    cancelled = []
    task = _AverageTask(lambda _report: called.append(True))
    task.signals.cancelled.connect(lambda: cancelled.append(True))

    task.cancel()
    task.run()

    assert called == []
    assert cancelled == [True]


def test_cancelled_zarr_task_always_emits_finished_without_work() -> None:
    from minflux_viewer.ui.main_window import _ZarrIoTask

    called = []
    finished = []
    task = _ZarrIoTask(
        lambda _report: called.append(True), description="cancelled test"
    )
    task.signals.finished.connect(lambda: finished.append(True))

    task.cancel()
    task.run()

    assert called == []
    assert finished == [True]


def test_background_task_reports_result_and_always_finishes() -> None:
    stages = []
    done = []
    finished = []
    task = BackgroundTask(
        lambda report: (report("halfway"), 42)[1],
        description="generic result",
    )
    task.signals.stage.connect(stages.append)
    task.signals.done.connect(done.append)
    task.signals.finished.connect(lambda: finished.append(True))

    task.run()

    assert stages == ["halfway"]
    assert done == [42]
    assert finished == [True]


def test_background_task_discards_a_result_cancelled_during_work() -> None:
    discarded = []
    cancelled = []
    finished = []
    task = None

    def work(_report):
        task.cancel()
        return "resource"

    task = BackgroundTask(
        work,
        description="cancel after native work",
        discard_result=discarded.append,
    )
    task.signals.cancelled.connect(lambda: cancelled.append(True))
    task.signals.finished.connect(lambda: finished.append(True))

    task.run()

    assert discarded == ["resource"]
    assert cancelled == [True]
    assert finished == [True]
