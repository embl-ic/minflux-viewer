"""Non-blocking ownership helpers for Qt background work.

Qt destroys a child ``QThreadPool`` synchronously and waits for its active
runnables.  A scheduler owned by a closing widget must therefore not own the
pool that executes its work.  Named pools in this module live for the process,
while widgets only cancel generations and detach result callbacks.  Which
queue a named pool belongs to and how many threads it has is declared once,
in ``core/execution.py``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from threading import Event, Lock
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal

from minflux_viewer.core.execution import pool_size
from minflux_viewer.core.job_resources import JOB_RESOURCES, ResourceAdmissionError, ResourceLease
from minflux_viewer.core.task_failure import TaskFailure
from minflux_viewer.ui.common.qt_lifecycle import disconnect_signal, qobject_alive

if TYPE_CHECKING:
    from minflux_viewer.core.task_registry import TaskHandle as RegistryTaskHandle

_SHARED_POOLS: dict[str, QThreadPool] = {}
_LOG = logging.getLogger(__name__)
_LIVE_QTHREADS: set[Any] = set()
_COMMON_SIGNAL_NAMES = (
    "stage",
    "progress",
    "message",
    "result",
    "result_ready",
    "done",
    "failed",
    "failure",
    "delivery_discarded",
    "cancelled",
    "finished",
)


class BackgroundTaskSignals(QObject):
    """Terminal/result signals shared by ordinary GUI-owned operations."""

    stage = pyqtSignal(str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    failure = pyqtSignal(object)
    cancelled = pyqtSignal()
    delivery_discarded = pyqtSignal()
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
        commits_output: bool = False,
        estimated_bytes: int = 0,
        resource_lease: ResourceLease | None = None,
    ) -> None:
        super().__init__()
        self._work: Callable[[Callable[[str], None]], Any] | None = work
        self.description = str(description)
        self.category = str(category)
        self.signals = BackgroundTaskSignals()
        self._cancelled = Event()
        self._discard_result = discard_result
        self._finally_callback = finally_callback
        self._commits_output = bool(commits_output)
        self._status = "queued"
        self._state_lock = Lock()
        self._handle: RegistryTaskHandle | None = None
        self._on_start: Callable[[BackgroundTask], None] | None = None
        self._estimated_bytes = int(estimated_bytes)
        self._resource_lease = resource_lease

    @property
    def status(self) -> str:
        with self._state_lock:
            return self._status

    def prepare(self) -> RegistryTaskHandle:
        """Register before enqueueing, or on entry for a direct run()."""
        from minflux_viewer.core.task_registry import registry

        with self._state_lock:
            if self._handle is None:
                self._handle = registry.register(self.description, self.category, cancel=self.cancel)
                if self._cancelled.is_set():
                    self._handle.update(detail="Stop requested before execution")
            return self._handle

    def _finish(self, status: str, detail: str | None = None) -> None:
        with self._state_lock:
            self._status = status
        assert self._handle is not None
        self._handle.finish(status, detail=detail)

    def admit(self) -> bool:
        self.prepare()
        if self._resource_lease is not None:
            return True
        try:
            self._resource_lease = JOB_RESOURCES.acquire(self._estimated_bytes)
        except ResourceAdmissionError as exc:
            failure = TaskFailure.capture(exc, code="resource_admission")
            self._finish("failed", detail=failure.message)
            self.signals.failure.emit(failure)
            self.signals.failed.emit(failure.message)
            self._cleanup()
            self.signals.finished.emit()
            return False
        return True

    def _cleanup(self) -> None:
        self._run_finally()
        self._work = None
        self._discard_result = None
        self._finally_callback = None
        if self._resource_lease is not None:
            self._resource_lease.release()
            self._resource_lease = None

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        with self._state_lock:
            # Closing can retire a queued delivery after native work finished.
            # Record the request without relabelling an already terminal job.
            self._cancelled.set()
            if self._status in {"done", "failed", "cancelled"}:
                return
            self._status = "cancelling"

    def run(self) -> None:  # noqa: N802 - Qt API
        if self._on_start is not None:
            self._on_start(self)
            self._on_start = None
        handle = self.prepare()
        if self.is_cancelled:
            self._finish_cancelled(None)
            return
        if not self.admit():
            return
        handle.start()
        with self._state_lock:
            self._status = "cancelling" if self.is_cancelled else "running"
        result = None

        def report(stage: str) -> None:
            if self.is_cancelled:
                raise BackgroundTaskCancelled
            text = str(stage)
            handle.update(detail=text)
            self.signals.stage.emit(text)

        try:
            assert self._work is not None
            result = self._work(report)
            if self.is_cancelled:
                committed = self._commits_output
                self._finish("done" if committed else "cancelled",
                             "Output committed; UI delivery cancelled" if committed else None)
                self._discard(result)
                if committed:
                    self.signals.delivery_discarded.emit()
                else:
                    self.signals.cancelled.emit()
            else:
                self._finish("done")
                self.signals.done.emit(result)
        except BackgroundTaskCancelled:
            self._finish("cancelled")
            self._discard(result)
            self.signals.cancelled.emit()
        except (SystemExit, KeyboardInterrupt):
            self._finish("cancelled", detail="Worker requested to stop")
            self._discard(result)
            self.signals.cancelled.emit()
        except BaseException as exc:  # noqa: BLE001 - nothing escapes the Qt run boundary
            failure = TaskFailure.capture(exc)
            self._finish("failed", detail=f"{failure.exception_type}: {failure.message}")
            self.signals.failure.emit(failure)
            self.signals.failed.emit(str(exc))
        finally:
            self._cleanup()
            self.signals.finished.emit()

    def _finish_cancelled(self, result: Any) -> None:
        try:
            self._finish("cancelled")
            self._discard(result)
            self.signals.cancelled.emit()
        finally:
            self._cleanup()
            self.signals.finished.emit()


    def _discard(self, result: Any) -> None:
        if result is None or self._discard_result is None:
            return
        try:
            self._discard_result(result)
        except Exception:  # caller-supplied cleanup must not suppress terminal signals
            _LOG.exception("Could not discard result for %s", self.description)

    def _run_finally(self) -> None:
        if self._finally_callback is None:
            return
        try:
            self._finally_callback()
        except Exception:  # caller-supplied cleanup must not suppress terminal signals
            _LOG.exception("Final cleanup failed for %s", self.description)


class _TrackedThreadPool(QThreadPool):
    def __init__(self) -> None:
        super().__init__()
        self._pending: set[BackgroundTask] = set()
        self._pending_lock = Lock()

    def _started(self, task: BackgroundTask) -> None:
        with self._pending_lock:
            self._pending.discard(task)

    def start(self, runnable, priority: int = 0) -> None:
        if isinstance(runnable, BackgroundTask):
            # tryTake is unsafe for auto-deleted runnables (Qt's ABA hazard).
            # Python owners and run() retain these until actual completion.
            runnable.setAutoDelete(False)
            runnable.prepare()
            if runnable.is_cancelled:
                runnable._finish_cancelled(None)
                return
            if not runnable.admit():
                return
            runnable._on_start = self._started
            with self._pending_lock:
                self._pending.add(runnable)
        super().start(runnable, priority)

    def clear(self) -> None:
        with self._pending_lock:
            pending = list(self._pending)
        for task in pending:
            if self.tryTake(task):
                self._started(task)
                task._on_start = None
                task.cancel()
                task._finish_cancelled(None)
        super().clear()


def shared_thread_pool(name: str, *, max_threads: int | None = None) -> QThreadPool:
    """Return a process-owned pool whose destructor cannot block a window close.

    A pool the application uses is declared in ``core/execution.py``, which
    sizes it (never above the core count); *max_threads* may only repeat that
    size. A name declared nowhere needs *max_threads* (tests, ad hoc work).
    """
    key = str(name).strip()
    if not key:
        raise ValueError("A shared thread-pool name is required")
    try:
        size: int | None = pool_size(key)
    except KeyError:
        if max_threads is None:
            raise
        size = None
    if size is not None:
        if max_threads is not None and int(max_threads) != size:
            raise ValueError(
                f"thread pool {key!r} has {size} thread(s), as declared in "
                f"core/execution.py; it was asked for {max_threads}")
        max_threads = size
    pool = _SHARED_POOLS.get(key)
    # Test suites and embedders may destroy one QApplication and create the
    # next without unloading this module. Qt then deletes the C++ pool while
    # the Python registry still contains its wrapper.
    if pool is None or not qobject_alive(pool):
        pool = _TrackedThreadPool()
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
    """Keep a parentless QThread alive until its real ``finished`` signal.

    A thread that has already finished needs no protection, and retaining it
    would hold it for the life of the process because its ``finished`` signal
    can no longer fire. ``isFinished()`` is False for a thread that was never
    started, so the ordinary retain-before-``start()`` call sites are unaffected.
    """
    if thread is None or thread in _LIVE_QTHREADS:
        return
    try:
        if thread.isFinished():
            return
    except (AttributeError, RuntimeError):
        pass
    _LIVE_QTHREADS.add(thread)

    def release() -> None:
        _LIVE_QTHREADS.discard(thread)

    try:
        thread.finished.connect(release, Qt.ConnectionType.QueuedConnection)
    except (AttributeError, TypeError, RuntimeError):
        # Unknown thread-like objects used by tests remain retained for the
        # process; that is safer than destroying a possibly active worker.
        return
    try:
        # It may have finished between the check above and the connect; that
        # emission is already gone, so drop it now instead of retaining it.
        if thread.isFinished():
            release()
    except (AttributeError, RuntimeError):
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


def shared_thread_pools() -> tuple[QThreadPool, ...]:
    """Every named process-owned pool created so far, live ones only."""
    return tuple(
        pool for pool in _SHARED_POOLS.values() if qobject_alive(pool)
    )
