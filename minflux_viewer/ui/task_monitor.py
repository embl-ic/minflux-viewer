"""Task-Manager-style view of the background work in flight — Help ▸ Monitor Tasks.

Companion to the memory monitor, and built the same way: a plain widget polling
once a second, so nothing has to push updates at it and a worker thread never
touches Qt.

Two panes, because they answer different questions:

* **Tasks** — the application's own registered work (``core/task_registry.py``):
  what it is, which subsystem started it, how long it has run, how far it has
  got, and whether it can be stopped.
* **Threads** — the raw picture underneath: Python's ``threading.enumerate()``
  and the Qt global thread pool's active/maximum count. Nothing registers these;
  they are the ground truth a tracked task list can drift from.

⚠ **Stopping is a request, not a kill.** ``QThread.terminate()`` is documented as
unsafe, ``QRunnable`` has no terminate, and Python threads cannot be killed at
all — so the button asks the task's own ``cancel()`` flag, which it honours at
its next checkpoint. A task inside one long ``savetxt`` will not stop until that
call returns. The window says so rather than offering a kill that would leave
the process in an undefined state.
"""

from __future__ import annotations

import threading

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.task_registry import registry as task_registry

_TICK_MS = 1000

#: Tasks table columns.
_TASK_COLUMNS = ("Task", "Kind", "Status", "Elapsed", "Progress", "Detail")
#: Threads table columns.
_THREAD_COLUMNS = ("Thread", "Ident", "Daemon", "Alive")

#: Status → the colour it is drawn in, so a failure is visible at a glance.
_STATUS_COLORS = {
    "running": "#2e7d32",
    "queued": "#616161",
    "cancelling": "#ef6c00",
    "cancelled": "#ef6c00",
    "failed": "#c62828",
    "done": "#616161",
}


def format_elapsed(seconds: float) -> str:
    """``12.3 s`` / ``4 min 05 s`` / ``1 h 12 min`` — compact and unambiguous."""
    value = max(0.0, float(seconds))
    if value < 60.0:
        return f"{value:.1f} s"
    if value < 3600.0:
        minutes, rest = divmod(int(value), 60)
        return f"{minutes} min {rest:02d} s"
    hours, rest = divmod(int(value), 3600)
    return f"{hours} h {rest // 60:02d} min"


def format_progress(progress: float | None) -> str:
    """A percentage, or an em dash when the task cannot measure itself."""
    return "—" if progress is None else f"{progress * 100:.0f}%"


def summary_text(tasks, pool_active: int, pool_max: int, n_threads: int) -> str:
    """The one-line header: tracked work first, then the raw thread picture."""
    active = [info for info in tasks if info.active]
    running = sum(1 for info in active if info.status == "running")
    if not tasks:
        head = "No background tasks have run yet."
    elif not active:
        head = f"No task running · {len(tasks)} recently finished"
    else:
        head = f"{len(active)} task(s) in flight · {running} running"
    return (f"{head}   |   Qt thread pool {pool_active}/{pool_max} active"
            f"   |   {n_threads} Python thread(s)")


class TaskMonitor(QWidget):
    """Floating monitor of background tasks and threads — Help ▸ Monitor Tasks."""

    TAG = "task_monitor"

    def __init__(self, state=None, parent: QWidget | None = None, *,
                 registry=None) -> None:
        super().__init__(parent)
        self._state = state
        self._registry = registry if registry is not None else task_registry

        self.setWindowTitle("Task monitor")
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.Tool)
        self.resize(760, 380)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(_TICK_MS)
        self.refresh()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        self._summary = QLabel("")
        self._summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self._summary)

        tabs = QTabWidget(self)
        self._tasks_table = self._make_table(_TASK_COLUMNS)
        self._tasks_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._tasks_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self._tasks_table.itemSelectionChanged.connect(self._sync_buttons)
        tabs.addTab(self._tasks_table, "Tasks")
        self._threads_table = self._make_table(_THREAD_COLUMNS)
        tabs.addTab(self._threads_table, "Threads")
        root.addWidget(tabs, 1)

        note = QLabel(
            "Stopping a task is a <b>request</b>: it is honoured at the task's "
            "next checkpoint. A thread cannot be force-killed — Qt's "
            "<tt>terminate()</tt> is unsafe and Python has no kill at all — so a "
            "task inside one long write finishes that write first."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(note)

        buttons = QHBoxLayout()
        self._stop_btn = QPushButton("Request stop")
        self._stop_btn.clicked.connect(self._request_stop)
        buttons.addWidget(self._stop_btn)
        self._clear_btn = QPushButton("Clear finished")
        self._clear_btn.clicked.connect(self._clear_finished)
        buttons.addWidget(self._clear_btn)
        buttons.addStretch()
        root.addLayout(buttons)
        self._sync_buttons()

    @staticmethod
    def _make_table(columns: tuple[str, ...]) -> QTableWidget:
        table = QTableWidget(0, len(columns))
        table.setHorizontalHeaderLabels(list(columns))
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        return table

    # -------------------------------------------------------------- refresh
    def refresh(self) -> None:
        tasks = self._registry.snapshot()
        pool_active, pool_max = self._pool_counts()
        threads = list(threading.enumerate())
        self._summary.setText(
            summary_text(tasks, pool_active, pool_max, len(threads)))
        self._fill_tasks(tasks)
        self._fill_threads(threads)
        self._sync_buttons()

    @staticmethod
    def _pool_counts() -> tuple[int, int]:
        try:
            from PyQt6.QtCore import QThreadPool

            pool = QThreadPool.globalInstance()
            return int(pool.activeThreadCount()), int(pool.maxThreadCount())
        except Exception:                                       # noqa: BLE001
            return 0, 0

    def _fill_tasks(self, tasks) -> None:
        table = self._tasks_table
        # Keep the selection across the once-a-second rebuild, or the Stop button
        # would disarm itself under the user's cursor.
        selected_id = self.selected_task_id()
        table.setRowCount(len(tasks))
        for row, info in enumerate(tasks):
            cells = (
                info.name, info.kind, info.status,
                format_elapsed(info.elapsed()), format_progress(info.progress),
                info.detail,
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(str(text))
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, int(info.id))
                if column == 2:
                    colour = _STATUS_COLORS.get(info.status)
                    if colour:
                        from PyQt6.QtGui import QColor

                        item.setForeground(QColor(colour))
                table.setItem(row, column, item)
            if selected_id is not None and info.id == selected_id:
                table.selectRow(row)
        table.resizeColumnsToContents()

    def _fill_threads(self, threads) -> None:
        table = self._threads_table
        table.setRowCount(len(threads))
        for row, thread in enumerate(threads):
            cells = (
                thread.name,
                str(thread.ident if thread.ident is not None else "—"),
                "yes" if thread.daemon else "no",
                "yes" if thread.is_alive() else "no",
            )
            for column, text in enumerate(cells):
                table.setItem(row, column, QTableWidgetItem(str(text)))
        table.resizeColumnsToContents()

    # -------------------------------------------------------------- actions
    def selected_task_id(self) -> int | None:
        items = self._tasks_table.selectedItems()
        for item in items:
            value = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(value, int):
                return value
        row = self._tasks_table.currentRow()
        if row < 0:
            return None
        item = self._tasks_table.item(row, 0)
        value = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        return value if isinstance(value, int) else None

    def _sync_buttons(self) -> None:
        task_id = self.selected_task_id()
        info = self._registry.get(task_id) if task_id is not None else None
        can_stop = bool(info is not None and info.active and info.cancellable)
        self._stop_btn.setEnabled(can_stop)
        if info is None:
            self._stop_btn.setToolTip("Select a task.")
        elif not info.active:
            self._stop_btn.setToolTip("This task has already finished.")
        elif not info.cancellable:
            self._stop_btn.setToolTip(
                "This task has no cancellation checkpoint, so it cannot be "
                "asked to stop. It cannot be killed either — see the note below."
            )
        else:
            self._stop_btn.setToolTip(
                "Ask this task to stop at its next checkpoint.")

    def _request_stop(self) -> None:
        task_id = self.selected_task_id()
        if task_id is None:
            return
        info = self._registry.get(task_id)
        ok = self._registry.request_stop(task_id)
        self._log(
            f"Task monitor: stop requested for '{info.name}'."
            if ok and info is not None else
            f"Task monitor: '{getattr(info, 'name', task_id)}' cannot be stopped.",
            "INFO" if ok else "WARN",
        )
        self.refresh()

    def _clear_finished(self) -> None:
        removed = self._registry.clear_finished()
        if removed:
            self._log(f"Task monitor: cleared {removed} finished task(s).")
        self.refresh()

    def _log(self, message: str, level: str = "INFO") -> None:
        log = getattr(self._state, "log", None)
        if callable(log):
            try:
                log(message, level)
            except Exception:                                   # noqa: BLE001
                pass

    # ------------------------------------------------------------- lifetime
    def closeEvent(self, event) -> None:
        try:
            self._timer.stop()
        except RuntimeError:
            pass
        super().closeEvent(event)

    def showEvent(self, event) -> None:          # noqa: N802 - Qt API
        super().showEvent(event)
        try:
            if not self._timer.isActive():
                self._timer.start(_TICK_MS)
        except RuntimeError:
            pass
        self.refresh()
