"""Where background work runs, and how many threads it may use (slice 4.3).

Every named pool belongs to one of three queues, sized once in
``core/execution.py``. Analyses that start native threads of their own lease
them from one budget: each SciPy ``workers=-1`` call and each private executor
used to take every core, so two analyses at once ran twice as many threads as
the machine has and a render view settled 2.2x slower meanwhile.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

import numpy as np
import pytest

from minflux_viewer.core import execution
from minflux_viewer.core.execution import (
    POOLS,
    QUEUE_LIMITS,
    QUEUES,
    ThreadBudget,
    default_native_threads,
)

PACKAGE = Path(execution.__file__).resolve().parents[1]


def _sources():
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" not in path.parts:
            yield path, path.read_text(encoding="utf-8")


# ------------------------------------------------------------- the budget
def test_a_lease_never_takes_more_than_is_free():
    budget = ThreadBudget(6, per_lease=6)
    first = budget.acquire()
    assert first.threads == 6 and budget.free == 0
    second = budget.acquire(4)
    assert second.threads == 1, "a spent budget leaves the calling thread only"
    assert budget.free == 0
    second.release()
    first.release()
    first.release()                         # releasing twice changes nothing
    assert budget.free == 6


def test_one_lease_leaves_threads_for_the_next_analysis():
    budget = ThreadBudget(12)
    assert budget.per_lease == 8
    first, second, third = budget.acquire(), budget.acquire(), budget.acquire()
    assert (first.threads, second.threads, third.threads) == (8, 4, 1)
    for lease in (first, second, third):
        lease.release()
    assert budget.free == 12
    assert ThreadBudget(1).per_lease == 1
    assert ThreadBudget(3).per_lease == 2


def test_a_lease_asks_for_no_more_than_it_needs():
    budget = ThreadBudget(8)
    with budget.lease(3) as threads:
        assert threads == 3 and budget.free == 5
    with budget.lease(0) as threads:        # nothing to do still runs somewhere
        assert threads == 1
    assert budget.free == 8


def test_a_lease_is_returned_when_the_call_raises():
    budget = ThreadBudget(4)
    with pytest.raises(RuntimeError):
        with budget.lease():
            raise RuntimeError("analysis failed")
    assert budget.free == 4


def test_two_analyses_at_once_share_the_budget():
    budget = ThreadBudget(12)
    held = threading.Barrier(2)
    leases = []

    def analysis():
        with budget.lease() as threads:
            leases.append(threads)
            held.wait(timeout=5)            # both hold their lease at once

    workers = [threading.Thread(target=analysis) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
    assert sorted(leases) == [4, 8]
    assert budget.free == 12


def test_the_budget_leaves_the_interactive_queue_its_cores():
    assert default_native_threads(16) == 12
    assert default_native_threads(8) == 6
    assert default_native_threads(4) == 3
    assert default_native_threads(2) == 1
    assert default_native_threads(1) == 1
    assert execution.CPU_THREADS.total == default_native_threads()


def test_a_budget_needs_at_least_one_thread():
    with pytest.raises(ValueError):
        ThreadBudget(0)


# ------------------------------------------------------------- the queues
def test_every_pool_belongs_to_a_queue_within_its_limit():
    assert set(QUEUE_LIMITS) == set(QUEUES)
    for queue in QUEUES:
        members = {name: policy.threads for name, policy in POOLS.items()
                   if policy.queue == queue}
        assert members, f"the {queue} queue has no pool"
        assert sum(members.values()) <= QUEUE_LIMITS[queue], (queue, members)
    for name, policy in POOLS.items():
        assert policy.queue in QUEUES, name
        assert policy.threads >= 1, name
        assert policy.purpose, name


_POOL_NAME_PATTERNS = (
    re.compile(r'shared_thread_pool\(\s*"([^"]+)"'),
    re.compile(r'\b_?POOL(?:_NAME)?\s*=\s*"([^"]+)"'),
    re.compile(r'\bpool_name(?:\s*:\s*str)?\s*=\s*"([^"]+)"'),
)


def test_every_pool_the_application_uses_is_declared():
    used = {}
    for path, text in _sources():
        for pattern in _POOL_NAME_PATTERNS:
            for name in pattern.findall(text):
                used.setdefault(name, path.relative_to(PACKAGE).as_posix())
    assert used, "the scan found no pool at all"
    undeclared = {name: where for name, where in used.items() if name not in POOLS}
    assert not undeclared, f"pools not declared in core/execution.py: {undeclared}"
    unused = set(POOLS) - set(used)
    assert not unused, f"declared pools nothing uses: {sorted(unused)}"


def test_no_code_takes_every_core_without_a_lease():
    offenders = []
    for path, text in _sources():
        if path.name == "execution.py":
            continue
        rel = path.relative_to(PACKAGE).as_posix()
        for lineno, line in enumerate(text.splitlines(), start=1):
            code = line.split("#", 1)[0]
            if re.search(r"\bworkers\s*=\s*-1\b", code) or "cpu_count(" in code \
                    or "globalInstance().start(" in code:
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
            if "ThreadPoolExecutor(" in code and "lease.threads" not in code \
                    and "import" not in code:
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


# ------------------------------------------------- analyses use the budget
@pytest.fixture
def spent_budget(monkeypatch):
    """Every thread of the budget is leased by another analysis."""
    budget = ThreadBudget(8, per_lease=8)
    monkeypatch.setattr(execution, "CPU_THREADS", budget)
    lease = budget.acquire()
    yield budget
    lease.release()


def test_a_density_count_runs_on_one_thread_when_the_budget_is_spent(
        spent_budget, monkeypatch):
    from minflux_viewer.analysis import local_density

    seen = []
    real = local_density.cKDTree

    class Tree(real):
        def query_ball_point(self, *args, **kwargs):
            seen.append(kwargs.get("workers"))
            return super().query_ball_point(*args, **kwargs)

    pts = np.random.default_rng(0).uniform(0, 1000, (2000, 3))
    expected = local_density.local_density_kdtree(pts, 50.0)
    monkeypatch.setattr(local_density, "cKDTree", Tree)
    counts = local_density.local_density_kdtree(pts, 50.0)
    assert seen == [1]
    assert spent_budget.free == 0
    np.testing.assert_array_equal(counts, expected)


def test_a_density_count_uses_what_the_budget_has_free(monkeypatch):
    from minflux_viewer.analysis import local_density

    budget = ThreadBudget(5)
    monkeypatch.setattr(execution, "CPU_THREADS", budget)
    seen = []
    real = local_density.cKDTree

    class Tree(real):
        def query_ball_point(self, *args, **kwargs):
            seen.append(kwargs.get("workers"))
            assert budget.free == budget.total - budget.per_lease,                 "the call runs while it holds its lease"
            return super().query_ball_point(*args, **kwargs)

    monkeypatch.setattr(local_density, "cKDTree", Tree)
    local_density.local_density_kdtree(
        np.random.default_rng(0).uniform(0, 1000, (500, 3)), 50.0)
    assert seen == [budget.per_lease] == [3]
    assert budget.free == 5


def _particles(n=12, seed=3):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ang = rng.uniform(0, 2 * np.pi, 200)
        ring = np.column_stack([50 * np.cos(ang), 50 * np.sin(ang), rng.normal(0, 5, 200)])
        out.append(ring + rng.normal(0, 4, ring.shape))
    return out


def test_particle_averaging_takes_its_workers_from_the_budget(monkeypatch):
    from minflux_viewer.analysis import particle_average

    budget = ThreadBudget(3)
    monkeypatch.setattr(execution, "CPU_THREADS", budget)
    sizes = []
    real = particle_average.ThreadPoolExecutor

    def executor(max_workers):
        sizes.append(max_workers)
        assert budget.free == budget.total - max_workers
        return real(max_workers=max_workers)

    monkeypatch.setattr(particle_average, "ThreadPoolExecutor", executor)
    particles = _particles()
    result = particle_average.average_template_free(particles, n_iter=2, tol=0)
    assert sizes == [budget.per_lease] == [2]
    assert budget.free == 3
    serial = particle_average.average_template_free(particles, n_iter=2, tol=0, n_workers=1)
    np.testing.assert_allclose(result["points"], serial["points"])


def test_particle_averaging_runs_serially_when_the_budget_is_spent(
        spent_budget, monkeypatch):
    from minflux_viewer.analysis import particle_average

    def executor(max_workers):
        raise AssertionError(f"started {max_workers} threads with none free")

    monkeypatch.setattr(particle_average, "ThreadPoolExecutor", executor)
    result = particle_average.average_template_free(_particles(), n_iter=2, tol=0)
    assert result["n_particles"] == 12


# ------------------------------------------------------------ Qt side
def test_a_declared_pool_takes_its_size_from_the_policy(qapp):
    import os

    from minflux_viewer.ui.common.background_tasks import shared_thread_pool

    pool = shared_thread_pool("render-tiles")
    assert pool.maxThreadCount() == execution.pool_size("render-tiles") \
        == min(POOLS["render-tiles"].threads, os.cpu_count() or 1)
    assert shared_thread_pool("render-tiles", max_threads=pool.maxThreadCount()) is pool
    with pytest.raises(ValueError, match="core/execution.py"):
        shared_thread_pool("render-tiles", max_threads=pool.maxThreadCount() + 1)
    with pytest.raises(KeyError, match="core/execution.py"):
        shared_thread_pool("not-a-declared-pool")
    adhoc = shared_thread_pool("test-execution-adhoc", max_threads=2)
    assert adhoc.maxThreadCount() == 2
