"""Non-blocking ownership helpers for Qt background work.

Qt destroys a child ``QThreadPool`` synchronously and waits for its active
runnables.  A scheduler owned by a closing widget must therefore not own the
pool that executes its work.  Named pools in this module live for the process,
while widgets only cancel generations and detach result callbacks.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from PyQt6.QtCore import QThreadPool, Qt

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
    "finished",
)


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
