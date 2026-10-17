from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QObject, pyqtSignal

from minflux_viewer.ui.common.background_tasks import (
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
    from minflux_viewer.ui.analysis.particle_average_dialog import _AverageTask

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


@pytest.mark.parametrize("outcome", ["done", "failed", "cancelled"])
def test_public_handle_is_not_running_after_any_terminal_outcome(outcome):
    from minflux_viewer.api.run import TaskHandle

    def work(report):
        if outcome == "failed":
            raise ValueError("bad input")
        if outcome == "cancelled":
            task.cancel()
            report("checkpoint")
        return 1

    task = BackgroundTask(work, description="handle lifecycle")
    handle = TaskHandle(task)
    assert handle.is_running()
    task.run()
    assert task.status == outcome
    assert not handle.is_running()


def test_worker_failure_retains_diagnostics_but_not_frames():
    from minflux_viewer.core.task_failure import TaskExecutionError

    failures, messages = [], []
    task = BackgroundTask(lambda report: 1 / 0, description="structured failure")
    task.signals.failure.connect(failures.append)
    task.signals.failed.connect(messages.append)
    task.run()
    error = TaskExecutionError(failures[0])
    assert error.exception_type == "builtins.ZeroDivisionError"
    assert "1 / 0" in error.worker_traceback
    assert error.code == "worker_error"
    assert messages == ["division by zero"]
    assert error.__traceback__ is None
    assert task._work is None


def test_a_real_failure_does_not_become_cancellation_when_stop_was_requested():
    def work(report):
        task.cancel()
        raise OSError("disk failed")

    task = BackgroundTask(work, description="cancel plus disk failure")
    failures = []
    task.signals.failure.connect(failures.append)
    task.run()
    assert task.status == "failed"
    assert failures[0].exception_type == "builtins.OSError"


def test_committed_output_and_cancelled_delivery_are_distinct():
    from minflux_viewer.core.task_registry import registry

    def work(report):
        task.cancel()  # e.g. stop during a native writer that finishes normally
        return ["saved.npy"]

    task = BackgroundTask(work, description="commit boundary", commits_output=True)
    cancelled, discarded = [], []
    task.signals.cancelled.connect(lambda: cancelled.append(True))
    task.signals.delivery_discarded.connect(lambda: discarded.append(True))
    task.run()
    assert task.status == "done"
    assert cancelled == [] and discarded == [True]
    assert registry.get(task._handle.id).detail == "Output committed; UI delivery cancelled"


def test_pending_work_is_visible_and_pool_clear_finishes_it(qtbot):
    from threading import Event

    from minflux_viewer.core.task_registry import registry

    entered, release = Event(), Event()
    pool = shared_thread_pool("test-visible-queue", max_threads=1)
    first = BackgroundTask(lambda report: (entered.set(), release.wait(5)), description="occupy pool")
    second = BackgroundTask(lambda report: pytest.fail("cleared work ran"), description="queued work")
    finished = []
    second.signals.finished.connect(lambda: finished.append(True))
    try:
        pool.start(first)
        qtbot.waitUntil(entered.is_set, timeout=3000)
        pool.start(second)
        assert registry.get(second._handle.id).status == "queued"
        pool.clear()
        assert finished == [True]
        assert registry.get(second._handle.id).status == "cancelled"
        assert second._work is None
    finally:
        release.set()
        qtbot.waitUntil(lambda: first.status == "done", timeout=3000)
