"""Non-blocking ownership helpers for Qt background work.

Qt destroys a child ``QThreadPool`` synchronously and waits for its active
runnables.  A scheduler owned by a closing widget must therefore not own the
pool that executes its work.  Named pools in this module live for the process,
while widgets only cancel generations and detach result callbacks.
"""

from __future__ import annotations

from collections.abc import Iterable
from threading import Event
from typing import Any, Callable

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, Qt, pyqtSignal

from .qt_lifecycle import disconnect_signal, qobject_alive

_SHARED_POOLS: dict[str, QThreadPool] = {}
_LIVE_QTHREADS: set[Any] = set()
_COMMON_SIGNAL_NAMES = (
    "stage",
    "progress",
    "message",
    "result",
    "result_ready",
    "done",
    "failed",
    "cancelled",
    "finished",
)


class BackgroundTaskSignals(QObject):
    """Terminal/result signals shared by ordinary GUI-owned operations."""

    stage = pyqtSignal(str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    finished = pyqtSignal()


class BackgroundTaskCancelled(Exception):
    """Private cooperative-cancellation checkpoint raised by ``report``."""


class BackgroundTask(QRunnable):
    """Run non-Qt work on a process-owned pool and report back to the GUI.

    ``work`` receives a stage-report callback. Calling it after cancellation is
    a checkpoint that aborts the remaining Python work. Native reads/writes that
    are already inside NumPy, compression, or a file library finish normally;
    their result is then discarded instead of being delivered to a closing UI.
    """

    def __init__(
        self,
        work: Callable[[Callable[[str], None]], Any],
        *,
        description: str,
        category: str = "I/O",
        discard_result: Callable[[Any], None] | None = None,
        finally_callback: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._work = work
        self.description = str(description)
        self.category = str(category)
        self.signals = BackgroundTaskSignals()
        self._cancelled = Event()
        self._discard_result = discard_result
        self._finally_callback = finally_callback

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        self._cancelled.set()

    def run(self) -> None:  # noqa: N802 - Qt API
        from ..core.task_registry import registry as task_registry

        if self.is_cancelled:
            self._finish_cancelled(None)
            return
        handle = task_registry.register(
            self.description, self.category, cancel=self.cancel
        )
        handle.start()
        result = None

        def report(stage: str) -> None:
            if self.is_cancelled:
                raise BackgroundTaskCancelled
            text = str(stage)
            handle.update(detail=text)
            self.signals.stage.emit(text)

        try:
            result = self._work(report)
            if self.is_cancelled:
                handle.finish("cancelled")
                self._discard(result)
                self.signals.cancelled.emit()
            else:
                handle.finish("done")
                self.signals.done.emit(result)
        except BackgroundTaskCancelled:
            handle.finish("cancelled")
            self._discard(result)
            self.signals.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - exceptions cross the Qt boundary
            if self.is_cancelled:
                handle.finish("cancelled")
                self._discard(result)
                self.signals.cancelled.emit()
            else:
                handle.finish("failed", detail=str(exc))
                self.signals.failed.emit(str(exc))
        finally:
            self._run_finally()
            self.signals.finished.emit()

    def _finish_cancelled(self, result: Any) -> None:
        try:
            self._discard(result)
            self.signals.cancelled.emit()
        finally:
            self._run_finally()
            self.signals.finished.emit()

    def _discard(self, result: Any) -> None:
        if result is None or self._discard_result is None:
            return
        try:
            self._discard_result(result)
        except Exception:
            pass

    def _run_finally(self) -> None:
        if self._finally_callback is None:
            return
        try:
            self._finally_callback()
        except Exception:
            pass


def shared_thread_pool(name: str, *, max_threads: int | None = None) -> QThreadPool:
    """Return a process-owned pool whose destructor cannot block a window close."""
    key = str(name).strip()
    if not key:
        raise ValueError("A shared thread-pool name is required")
    pool = _SHARED_POOLS.get(key)
    # Test suites and embedders may destroy one QApplication and create the
    # next without unloading this module. Qt then deletes the C++ pool while
    # the Python registry still contains its wrapper.
    if pool is None or not qobject_alive(pool):
        pool = QThreadPool()
        _SHARED_POOLS[key] = pool
    if max_threads is not None:
        pool.setMaxThreadCount(max(int(max_threads), 1))
    return pool


def request_task_cancel(task: Any) -> None:
    """Cooperatively cancel a runnable/thread without waiting for it."""
    if task is None:
        return
    for method_name in ("cancel", "requestInterruption"):
        try:
            method = getattr(task, method_name, None)
            if callable(method):
                method()
        except (AttributeError, RuntimeError):
            pass


def detach_task_signals(
    task: Any, signal_names: Iterable[str] = _COMMON_SIGNAL_NAMES
) -> None:
    """Disconnect worker result signals before their receiver is destroyed."""
    try:
        signals = getattr(task, "signals", None)
    except RuntimeError:
        return
    if signals is None:
        return
    for name in signal_names:
        try:
            signal = getattr(signals, name, None)
        except RuntimeError:
            signal = None
        disconnect_signal(signal)


def retire_background_tasks(
    tasks: Iterable[Any], signal_names: Iterable[str] = _COMMON_SIGNAL_NAMES
) -> None:
    """Cancel and detach tasks; deliberately never call ``wait``/``waitForDone``."""
    for task in list(tasks):
        request_task_cancel(task)
        detach_task_signals(task, signal_names)


def clear_pool_nonblocking(pool: QThreadPool | None) -> None:
    """Drop work that has not started, leaving active work to finish safely."""
    if pool is None:
        return
    try:
        pool.clear()
    except RuntimeError:
        pass


def retain_qthread(thread: Any) -> None:
    """Keep a parentless QThread alive until its real ``finished`` signal."""
    if thread is None or thread in _LIVE_QTHREADS:
        return
    _LIVE_QTHREADS.add(thread)

    def release() -> None:
        _LIVE_QTHREADS.discard(thread)

    try:
        thread.finished.connect(release, Qt.ConnectionType.QueuedConnection)
    except (AttributeError, TypeError, RuntimeError):
        # Unknown thread-like objects used by tests remain retained for the
        # process; that is safer than destroying a possibly active worker.
        pass


def retire_qthreads(
    threads: Iterable[Any], signal_names: Iterable[str] = ()
) -> None:
    """Detach a window from QThreads and let them finish without owner blocking."""
    for thread in list(threads):
        request_task_cancel(thread)
        for name in signal_names:
            try:
                signal = getattr(thread, name, None)
            except RuntimeError:
                signal = None
            disconnect_signal(signal)
        retain_qthread(thread)
