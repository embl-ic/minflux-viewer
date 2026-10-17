"""
``mfv.run`` -- running work off the GUI thread.

Wraps ``ui/common/background_tasks.py`` and ``core/task_registry.py``, so a plugin's
computation appears in *Help > Monitor Tasks* with a working *Request stop*
button without the plugin author doing anything.

The rules a worker must obey are the project's rules, and they are not
optional:

* **A worker touches no Qt widget and no mutable viewer state.** Capture the
  inputs you need before submitting; apply results in ``on_done``, which runs
  on the GUI thread.
* **Cancellation is cooperative.** A thread cannot be killed. *Request stop*
  sets a flag; the work stops when it next calls
  :meth:`TaskContext.check_cancelled`. Work with no checkpoint runs to
  completion, and the monitor says so.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

from ._base import ApiError, Namespace


class TaskContext:
    """
    Handed to the worker function as its only argument.

    The worker runs on a pool thread, so nothing here touches Qt directly:
    progress goes through the task's own reporting channel, which marshals it
    to the GUI thread.
    """

    def __init__(self, report: Callable[[str], None], task) -> None:
        self._report = report
        self._task = task

    def progress(self, done: float, total: float | None = None, label: str = "") -> None:
        """
        Report progress. With *total*, ``done`` is a count; without it, ``done``
        is a fraction 0..1.

        This is also a **cancellation checkpoint** -- it raises out of the
        worker once a stop has been requested, which is what makes a loop that
        reports progress stoppable for free.
        """
        try:
            fraction = float(done) / float(total) if total else float(done)
        except (TypeError, ValueError, ZeroDivisionError):
            fraction = 0.0
        fraction = max(0.0, min(1.0, fraction))
        text = str(label) if label else ""
        self._report(f"{text}|{fraction:.4f}" if text else f"|{fraction:.4f}")

    def is_cancelled(self) -> bool:
        """Whether a stop has been requested. Poll this in long loops."""
        return bool(getattr(self._task, "is_cancelled", False))

    def check_cancelled(self) -> None:
        """
        Raise if a stop has been requested, otherwise return.

        The convenient form: call it at the top of each iteration and let the
        exception unwind. The task machinery catches it and reports a
        cancellation, not a failure.
        """
        self._report("")

    def log(self, message: str, level: str = "INFO") -> None:
        """
        Write to the viewer Log from inside the worker.

        Routed through the task's stage channel rather than touching
        ``AppState`` from a pool thread.
        """
        self._report(f"log:{level}:{message}")


class TaskHandle:
    """A submitted task."""

    def __init__(self, task) -> None:
        self._task = task

    def cancel(self) -> None:
        """Request a stop. Cooperative -- see the module docstring."""
        from minflux_viewer.ui.common.background_tasks import request_task_cancel

        request_task_cancel(self._task)

    def is_running(self) -> bool:
        """Whether the task is still active."""
        return getattr(self._task, "status", None) in {"queued", "running", "cancelling"}


class Run(Namespace):
    """Submit work to the shared background pool."""

    name = "run"

    #: Named process-owned pool. Widgets must never own a pool: destroying one
    #: waits for its workers in the C++ destructor, which turns a window close
    #: into a freeze.
    POOL = "mfv-script"

    def background(
        self,
        fn: Callable[[TaskContext], Any],
        *,
        on_done: Callable[[Any], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        name: str | None = None,
        kind: str = "plugin",
    ) -> TaskHandle:
        """
        Run *fn* on a pool thread and deliver its result on the GUI thread.

        *fn* receives a :class:`TaskContext` and returns any value, which is
        passed to *on_done*. A failure is passed to *on_error*, or reported to
        the Log if none is given -- it is never swallowed silently.
        """
        from minflux_viewer.core.task_failure import TaskExecutionError
        from minflux_viewer.ui.common.background_tasks import (
            BackgroundTask,
            detach_task_signals,
            shared_thread_pool,
        )

        if not callable(fn):
            raise ApiError("run.background() needs a callable.")
        label = str(name or getattr(fn, "__name__", "script task"))
        state = self._state

        # The plugin runner marks a plugin as running, and suppresses per-call
        # macro recording, only while run(ctx) executes -- and that returns as
        # soon as this task is submitted. The callbacks re-enter the context
        # they were submitted from; otherwise a result journalled from on_done
        # is not attributed to the plugin (no method text, no replay line) and
        # its API calls are recorded as macro steps beside the plugin's own.
        on_done = self._resumed(on_done)
        on_error = self._resumed(on_error)
        on_cancel = self._resumed(on_cancel)

        holder: dict[str, Any] = {}

        def work(report):
            ctx = TaskContext(report, holder.get("task"))
            return fn(ctx)

        task = BackgroundTask(work, description=label, category=str(kind))
        holder["task"] = task

        def _on_stage(text: str) -> None:
            # Runs on the GUI thread: the worker only emits strings.
            if text.startswith("log:"):
                _, _, rest = text.partition("log:")
                level, _, message = rest.partition(":")
                state.log(message, level.upper() or "INFO")
                return
            head, _, fraction = text.rpartition("|")
            if fraction:
                try:
                    state.status_progress(head or label, float(fraction))
                except ValueError:
                    pass

        def _on_failed(failure) -> None:
            if on_error is not None:
                on_error(TaskExecutionError(failure))
            else:
                state.log(f"{label} failed: {failure.traceback}", "ERROR")

        def _on_cancelled() -> None:
            state.log(f"{label}: stopped at the caller's request.", "INFO")
            if on_cancel is not None:
                on_cancel()

        task.signals.stage.connect(_on_stage)
        task.signals.failure.connect(_on_failed)
        task.signals.cancelled.connect(_on_cancelled)
        if on_done is not None:
            task.signals.done.connect(on_done)

        # Retain until Qt reports the runnable finished; a QRunnable handed to
        # a pool is otherwise collected mid-flight.
        self._facade.retain_task(task)
        def _finished() -> None:
            self._facade.release_task(task)
            detach_task_signals(task)
        task.signals.finished.connect(_finished)

        shared_thread_pool(self.POOL).start(task)
        return TaskHandle(task)

    def _resumed(self, callback):
        """*callback*, wrapped to run in the plugin scope and recording
        suppression that are active now. Unchanged when neither is."""
        facade = self._facade
        plugin = getattr(facade, "active_plugin", None)
        calls = getattr(facade, "calls", None)
        suppressed = bool(calls is not None and calls.suppressed())
        if callback is None or (plugin is None and not suppressed):
            return callback

        def resumed(*args):
            with contextlib.ExitStack() as stack:
                if suppressed:
                    stack.enter_context(calls.suppress())
                if plugin is not None:
                    stack.enter_context(facade.plugin_scope(
                        plugin["id"], plugin.get("label", ""), plugin.get("method")))
                return callback(*args)

        return resumed

    def is_cancelled(self) -> bool:
        """
        Whether the most recently submitted task has been asked to stop.

        For code deep inside a worker that does not carry its
        :class:`TaskContext`; prefer the context where you have it.
        """
        task = self._facade.last_task()
        return bool(getattr(task, "is_cancelled", False)) if task is not None else False

    def active(self) -> list:
        """Snapshot of the tasks currently registered as running."""
        from ..core.task_registry import registry

        return list(registry.active())
