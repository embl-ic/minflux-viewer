"""Background tasks are registered, listed, and can be asked to stop.

The registry replaces "each subsystem keeps its own set on MainWindow purely so
shutdown can drain it" with one list that can also be shown. Stopping is a
cooperative request, never a kill — see ``core/task_registry.py``.
"""

from __future__ import annotations

import threading

import pytest

from minflux_viewer.core.task_registry import TaskRegistry, track


def test_a_task_moves_through_its_lifecycle():
    registry = TaskRegistry()
    handle = registry.register("Save run.zarr", "zarr")
    assert [info.status for info in registry.snapshot()] == ["queued"]

    handle.start()
    info = registry.get(handle.id)
    assert info.status == "running" and info.started > 0.0
    assert info.thread_name == threading.current_thread().name

    handle.update(progress=0.4, detail="writing mfx")
    info = registry.get(handle.id)
    assert info.progress == pytest.approx(0.4)
    assert info.detail == "writing mfx"
    assert info.active is True

    handle.finish("done")
    info = registry.get(handle.id)
    assert info.status == "done" and info.finished > 0.0 and info.active is False


def test_stopping_is_a_request_the_task_honours_itself():
    registry = TaskRegistry()
    asked = []
    handle = registry.register("MSR batch export", "msr-export",
                               cancel=lambda: asked.append(1))
    handle.start()

    assert registry.request_stop(handle.id) is True
    assert asked == [1]
    # The task is not stopped yet — it reports 'cancelling' until it reaches
    # its own checkpoint, which is the honest state and what the UI shows.
    assert registry.get(handle.id).status == "cancelling"
    assert handle.cancelled is True
    assert registry.get(handle.id).active is True

    handle.finish("cancelled")
    assert registry.get(handle.id).status == "cancelled"


def test_a_task_with_no_checkpoint_cannot_be_asked_to_stop():
    registry = TaskRegistry()
    handle = registry.register("Parse run.msr", "msr-parse")     # no cancel=
    handle.start()
    assert registry.get(handle.id).cancellable is False
    assert registry.request_stop(handle.id) is False
    assert registry.get(handle.id).status == "running"

    handle.finish("done")
    assert registry.request_stop(handle.id) is False             # already finished
    assert registry.request_stop(12345) is False                 # unknown id


def test_a_stop_request_is_not_undone_by_later_progress():
    registry = TaskRegistry()
    handle = registry.register("export", "export", cancel=lambda: None)
    handle.start()
    registry.request_stop(handle.id)
    handle.start()                                 # a task still reporting itself
    assert registry.get(handle.id).status == "cancelling"


def test_the_context_manager_reports_success_failure_and_cancellation():
    registry = TaskRegistry()
    with track("ok", "kind", target=registry):
        pass
    assert registry.snapshot()[0].status == "done"

    with pytest.raises(ValueError):
        with track("boom", "kind", target=registry):
            raise ValueError("nope")
    failed = next(i for i in registry.snapshot() if i.name == "boom")
    assert failed.status == "failed" and "nope" in failed.detail

    with track("stopped", "kind", cancel=lambda: None, target=registry) as handle:
        registry.request_stop(handle.id)
    stopped = next(i for i in registry.snapshot() if i.name == "stopped")
    assert stopped.status == "cancelled"


def test_snapshot_lists_running_work_before_finished_work():
    registry = TaskRegistry()
    old = registry.register("finished", "kind")
    old.start()
    old.finish("done")
    live = registry.register("running", "kind")
    live.start()
    assert [info.name for info in registry.snapshot()] == ["running", "finished"]
    assert [info.name for info in registry.active()] == ["running"]

    assert registry.clear_finished() == 1
    assert [info.name for info in registry.snapshot()] == ["running"]


def test_registering_from_a_worker_thread_is_safe():
    registry = TaskRegistry()
    names = []

    def work(index: int) -> None:
        with track(f"task {index}", "kind", target=registry) as handle:
            handle.update(progress=1.0)
            names.append(handle.id)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(set(names)) == 8                     # ids never collide
    assert all(info.status == "done" for info in registry.snapshot())


# ----------------------------------------------------------------- the window

@pytest.fixture(scope="module")
def _app():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_window_lists_tasks_and_arms_stop_only_where_it_can_work(_app, qtbot):
    from minflux_viewer.ui.task_monitor import TaskMonitor

    registry = TaskRegistry()
    asked = []
    stoppable = registry.register("MSR batch export", "msr-export",
                                  cancel=lambda: asked.append(1))
    stoppable.start()
    stoppable.update(progress=0.37, detail="run_03.msr")
    fixed = registry.register("Parse run.msr", "msr-parse")
    fixed.start()

    window = TaskMonitor(None, registry=registry)
    qtbot.addWidget(window)
    window.refresh()

    rows = {window._tasks_table.item(row, 0).data(256): row
            for row in range(window._tasks_table.rowCount())}
    assert len(rows) == 2
    assert "37%" in window._tasks_table.item(rows[stoppable.id], 4).text()
    assert "run_03.msr" in window._tasks_table.item(rows[stoppable.id], 5).text()

    window._tasks_table.selectRow(rows[stoppable.id])
    window._sync_buttons()
    assert window._stop_btn.isEnabled() is True
    window._request_stop()
    assert asked == [1]
    assert registry.get(stoppable.id).status == "cancelling"

    window._tasks_table.selectRow(rows[fixed.id])
    window._sync_buttons()
    assert window._stop_btn.isEnabled() is False
    assert "cannot be" in window._stop_btn.toolTip()


def test_window_reports_the_raw_thread_picture_too(_app, qtbot):
    from minflux_viewer.ui.task_monitor import TaskMonitor, summary_text

    window = TaskMonitor(None, registry=TaskRegistry())
    qtbot.addWidget(window)
    window.refresh()
    # Every live Python thread is listed, whether or not it registered a task.
    assert window._threads_table.rowCount() == len(threading.enumerate())
    assert "Python thread(s)" in window._summary.text()
    assert "No background tasks" in summary_text([], 0, 8, 3)


def test_elapsed_and_progress_are_formatted_readably():
    from minflux_viewer.ui.task_monitor import format_elapsed, format_progress

    assert format_elapsed(0.0) == "0.0 s"
    assert format_elapsed(3.25) == "3.2 s"
    assert format_elapsed(75) == "1 min 15 s"
    assert format_elapsed(4000) == "1 h 06 min"
    assert format_progress(None) == "—"            # not measurable, not "0%"
    assert format_progress(0.5) == "50%"
