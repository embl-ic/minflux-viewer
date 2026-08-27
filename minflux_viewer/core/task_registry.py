"""One registry of the background work this application has in flight.

The app runs a lot off the UI thread -- Zarr load/save, MSR parse and export,
OME-Zarr and TIFF export, particle averaging, the update check, and the render
tile pool -- and until now each kind kept its own ad-hoc set on ``MainWindow``
purely so shutdown could drain it. Nothing could answer "what is running right
now, and can I stop it".

**A thread cannot be killed, and this module does not pretend otherwise.**
``QThread.terminate()`` is documented as unsafe (it can leave a mutex locked or
a heap half-written, so the process state afterwards is undefined), ``QRunnable``
has no terminate at all, and Python threads have no kill primitive. What every
long task here *does* have is a cooperative ``cancel()`` flag it checks at a safe
boundary. So the registry offers **request stop**, and the monitor window says
plainly that this is a request the task honours at its next checkpoint -- a task
inside one long ``np.savetxt`` will not stop until that call returns.

Qt-free and thread-safe (a task registers from its own worker thread), so the
window on top can simply poll it.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field, replace

__all__ = [
    "TASK_STATUSES",
    "ACTIVE_STATUSES",
    "TaskInfo",
    "TaskHandle",
    "TaskRegistry",
    "registry",
    "track",
]

#: Every state a tracked task can be in, in lifecycle order.
TASK_STATUSES: tuple[str, ...] = (
    "queued", "running", "done", "failed", "cancelling", "cancelled",
)
#: The states that mean the task still holds a thread.
ACTIVE_STATUSES: frozenset[str] = frozenset({"queued", "running", "cancelling"})

#: How long a finished task stays listed, so a run that has just ended is still
#: readable instead of vanishing the instant it completes.
FINISHED_LINGER_S: float = 60.0
#: Cap on retained finished tasks, so a long session cannot grow without bound.
MAX_FINISHED: int = 200


@dataclass(frozen=True)
class TaskInfo:
    """An immutable snapshot of one tracked task."""

    id: int
    name: str
    kind: str
    detail: str = ""
    status: str = "queued"
    created: float = 0.0
    started: float = 0.0
    finished: float = 0.0
    progress: float | None = None          # 0..1, or None when not measurable
    cancellable: bool = False
    thread_name: str = ""

    @property
    def active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    def elapsed(self, now: float | None = None) -> float:
        """Seconds this task has been running (or ran, once finished)."""
        start = self.started or self.created
        if not start:
            return 0.0
        end = self.finished or (now if now is not None else time.monotonic())
        return max(0.0, float(end) - float(start))


class TaskHandle:
    """The worker's side of a registration: report progress, then finish.

    Deliberately tolerant -- a worker must never fail because of its own
    bookkeeping, so every method swallows a registry that has gone away.
    """

    def __init__(self, registry: "TaskRegistry", task_id: int) -> None:
        self._registry = registry
        self.id = int(task_id)

    def start(self, detail: str | None = None) -> None:
        self._registry.update(self.id, status="running", detail=detail,
                              thread_name=threading.current_thread().name)

    def update(self, *, detail: str | None = None,
               progress: float | None = None) -> None:
        self._registry.update(self.id, detail=detail, progress=progress)

    def finish(self, status: str = "done", detail: str | None = None) -> None:
        self._registry.update(self.id, status=status, detail=detail)

    @property
    def cancelled(self) -> bool:
        """Whether a stop has been requested -- for a task polling its own flag."""
        info = self._registry.get(self.id)
        return bool(info is not None and info.status in {"cancelling", "cancelled"})


@dataclass
class _Entry:
    info: TaskInfo
    cancel: object = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class TaskRegistry:
    """Thread-safe list of the background tasks this process is running."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[int, _Entry] = {}
        self._ids = itertools.count(1)

    # -- worker side -----------------------------------------------------

    def register(self, name: str, kind: str, *, detail: str = "",
                 cancel=None) -> TaskHandle:
        """Add a task in the ``queued`` state and return its handle.

        *cancel* is the task's own cooperative stop callable; without one the
        monitor shows the task as not stoppable rather than offering a button
        that does nothing.
        """
        with self._lock:
            task_id = next(self._ids)
            self._entries[task_id] = _Entry(
                info=TaskInfo(
                    id=task_id, name=str(name), kind=str(kind), detail=str(detail),
                    status="queued", created=time.monotonic(),
                    cancellable=callable(cancel),
                ),
                cancel=cancel,
            )
            self._prune_locked()
        return TaskHandle(self, task_id)

    def update(self, task_id: int, *, status: str | None = None,
               detail: str | None = None, progress: float | None = None,
               thread_name: str | None = None) -> None:
        with self._lock:
            entry = self._entries.get(int(task_id))
            if entry is None:
                return
            info = entry.info
            changes: dict = {}
            if status is not None and status in TASK_STATUSES:
                # A stop already requested wins over a plain "running" update, so
                # a task that keeps reporting progress does not look un-cancelled.
                if not (info.status == "cancelling" and status == "running"):
                    changes["status"] = status
                if status == "running" and not info.started:
                    changes["started"] = time.monotonic()
                if status in {"done", "failed", "cancelled"} and not info.finished:
                    changes["finished"] = time.monotonic()
            if detail is not None:
                changes["detail"] = str(detail)
            if progress is not None:
                changes["progress"] = max(0.0, min(1.0, float(progress)))
            if thread_name is not None:
                changes["thread_name"] = str(thread_name)
            if changes:
                entry.info = replace(info, **changes)

    # -- monitor side ----------------------------------------------------

    def get(self, task_id: int) -> TaskInfo | None:
        with self._lock:
            entry = self._entries.get(int(task_id))
            return entry.info if entry is not None else None

    def snapshot(self) -> list[TaskInfo]:
        """Every tracked task, active ones first, newest first within a group."""
        with self._lock:
            self._prune_locked()
            infos = [entry.info for entry in self._entries.values()]
        infos.sort(key=lambda info: (not info.active, -info.id))
        return infos

    def active(self) -> list[TaskInfo]:
        return [info for info in self.snapshot() if info.active]

    def request_stop(self, task_id: int) -> bool:
        """Ask a task to stop at its next checkpoint. ``False`` when it cannot.

        This is a *request*: see the module docstring for why there is no forced
        kill. The status moves to ``cancelling`` and the task itself reports
        ``cancelled`` when it actually gets there.
        """
        with self._lock:
            entry = self._entries.get(int(task_id))
            if entry is None or not callable(entry.cancel):
                return False
            if not entry.info.active:
                return False
            cancel = entry.cancel
            entry.info = replace(entry.info, status="cancelling")
        try:
            cancel()
        except Exception:                                       # noqa: BLE001
            return False
        return True

    def clear_finished(self) -> int:
        with self._lock:
            finished = [key for key, entry in self._entries.items()
                        if not entry.info.active]
            for key in finished:
                self._entries.pop(key, None)
        return len(finished)

    def _prune_locked(self) -> None:
        now = time.monotonic()
        finished = [(key, entry) for key, entry in self._entries.items()
                    if not entry.info.active]
        stale = [key for key, entry in finished
                 if entry.info.finished and now - entry.info.finished > FINISHED_LINGER_S]
        for key in stale:
            self._entries.pop(key, None)
        finished = [key for key, entry in self._entries.items() if not entry.info.active]
        overflow = len(finished) - MAX_FINISHED
        for key in sorted(finished)[:max(0, overflow)]:
            self._entries.pop(key, None)


#: The process-wide registry. One per process by design: the monitor answers
#: "what is this application doing", which is not a per-window question.
registry = TaskRegistry()


class track:
    """Context manager registering a task for the length of a ``with`` block.

    ``with track("Save run.zarr", "zarr") as task:`` marks it running on entry
    and finishes it on exit -- ``failed`` when the block raised, ``cancelled``
    when a stop had been requested, else ``done``. The exception is never
    swallowed: reporting is bookkeeping, not error handling.
    """

    def __init__(self, name: str, kind: str, *, detail: str = "", cancel=None,
                 target: TaskRegistry | None = None) -> None:
        self._target = target if target is not None else registry
        self._name = name
        self._kind = kind
        self._detail = detail
        self._cancel = cancel
        self.handle: TaskHandle | None = None

    def __enter__(self) -> TaskHandle:
        self.handle = self._target.register(
            self._name, self._kind, detail=self._detail, cancel=self._cancel)
        self.handle.start()
        return self.handle

    def __exit__(self, exc_type, exc, tb) -> bool:
        handle = self.handle
        if handle is not None:
            if exc_type is not None:
                handle.finish("failed", detail=str(exc) if exc is not None else "")
            elif handle.cancelled:
                handle.finish("cancelled")
            else:
                handle.finish("done")
        return False
