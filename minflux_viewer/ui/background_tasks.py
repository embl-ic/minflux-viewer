"""Non-blocking ownership helpers for Qt background work.

Qt destroys a child ``QThreadPool`` synchronously and waits for its active
runnables.  A scheduler owned by a closing widget must therefore not own the
pool that executes its work.  Named pools in this module live for the process,
while widgets only cancel generations and detach result callbacks.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from PyQt6.QtCore import QThreadPool

from .qt_lifecycle import disconnect_signal

_SHARED_POOLS: dict[str, QThreadPool] = {}
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
    if pool is None:
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
