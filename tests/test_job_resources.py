"""Shared admission is bounded, non-waiting and releases every terminal path."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from minflux_viewer.core.job_resources import JobResources, ResourceAdmissionError


def test_bytes_slots_nested_reservations_and_idempotent_release():
    budget = JobResources(100, 1)
    lease = budget.acquire(40)
    with pytest.raises(ResourceAdmissionError):
        budget.acquire(0)
    with budget.acquire(50, job=False):
        assert budget.reserved_bytes == 90
        with pytest.raises(ResourceAdmissionError):
            budget.acquire(11, job=False)
    lease.release()
    lease.release()
    assert budget.reserved_bytes == budget.admitted_jobs == 0


def test_reconfiguration_does_not_forget_existing_reservations():
    budget = JobResources(100, 4)
    lease = budget.acquire(80)
    budget.configure(10, 1)
    assert budget.reserved_bytes == 80
    with pytest.raises(ResourceAdmissionError):
        budget.acquire(0)
    lease.release()
    with budget.acquire(10):
        assert budget.admitted_jobs == 1


def test_a_failed_preparation_cannot_leak_an_untransferred_lease():
    import gc

    budget = JobResources(100, 1)
    lease = budget.acquire(80)
    del lease
    gc.collect()
    assert budget.reserved_bytes == budget.admitted_jobs == 0


def test_concurrent_reservation_cannot_over_admit():
    from threading import Barrier

    budget = JobResources(100, 3)
    barrier = Barrier(8)

    def acquire():
        barrier.wait(timeout=5)
        try:
            return budget.acquire(40)
        except ResourceAdmissionError:
            return None

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: acquire(), range(8)))
    leases = [item for item in results if item is not None]
    assert len(leases) == 2
    for item in leases:
        item.release()
    assert budget.reserved_bytes == budget.admitted_jobs == 0


@pytest.mark.parametrize("outcome", ["done", "failed", "cancelled"])
def test_background_terminal_paths_return_reservations(outcome, monkeypatch):
    from minflux_viewer.ui.common import background_tasks

    budget = JobResources(100, 1)
    monkeypatch.setattr(background_tasks, "JOB_RESOURCES", budget)

    def work(report):
        assert budget.reserved_bytes == 80
        if outcome == "failed":
            raise ValueError("bad input")
        if outcome == "cancelled":
            task.cancel()
            report("stop")
        return 1

    task = background_tasks.BackgroundTask(work, description="resources", estimated_bytes=80)
    task.run()
    assert task.status == outcome
    assert budget.admitted_jobs == budget.reserved_bytes == 0


def test_rejected_task_never_runs_or_retains_inputs(monkeypatch):
    from minflux_viewer.ui.common import background_tasks

    budget = JobResources(1, 1)
    monkeypatch.setattr(background_tasks, "JOB_RESOURCES", budget)
    task = background_tasks.BackgroundTask(lambda report: pytest.fail("not admitted"),
                                            description="too big", estimated_bytes=2)
    seen = []
    task.signals.failure.connect(seen.append)
    task.run()
    assert task.status == "failed" and task._work is None
    assert seen[0].code == "resource_admission"
    assert budget.reserved_bytes == budget.admitted_jobs == 0


def test_save_admission_precedes_capture_and_releases_failed_capture(monkeypatch):
    import numpy as np

    from minflux_viewer.core import save_snapshot
    from minflux_viewer.core.dataset import build_localization_dataset

    dataset = build_localization_dataset(name="save", x_nm=np.arange(5.), y_nm=np.arange(5.))
    budget = JobResources(1, 1)
    monkeypatch.setattr(save_snapshot, "JOB_RESOURCES", budget)
    called = []

    def fail(*args):
        called.append(True)
        raise ValueError("capture failed")

    monkeypatch.setattr(save_snapshot, "save_snapshot", fail)
    with pytest.raises(ResourceAdmissionError):
        save_snapshot.admitted_save_snapshot(dataset)
    assert not called
    budget.configure(10**9, 1)
    with pytest.raises(ValueError, match="capture failed"):
        save_snapshot.admitted_save_snapshot(dataset)
    assert budget.reserved_bytes == budget.admitted_jobs == 0
