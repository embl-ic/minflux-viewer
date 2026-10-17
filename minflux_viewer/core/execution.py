"""Where background work runs, and how many threads it may use.

Background work runs in three queues: **interactive** (what a view needs to
draw now), **analysis** (computations the user started) and **io** (reading
and writing files). Every named thread pool
(``ui/common/background_tasks.py::shared_thread_pool``) belongs to one queue and is
sized here, once; a queue's limit bounds the sum of its pools' sizes, which is
checked when this module is imported.

An analysis that starts native threads of its own -- SciPy's ``workers=``, a
private ``ThreadPoolExecutor`` -- leases them from :data:`CPU_THREADS`, one
budget for the process, sized to leave the interactive queue a quarter of the
cores. Each such call used to take every core: with two analyses running, a
render view took 2.2x as long to settle after a zoom (measured on 16 cores,
slice 4.3). One lease takes at most two thirds of the budget, so a second
analysis still gets threads of its own; a lease never waits, and when the
budget is spent the call runs on the thread it was made from, so a third
analysis is slower rather than the machine oversubscribed.

Qt-free, so analysis code can lease threads as well as the UI.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

#: The three queues, in the order the Task Monitor would list them.
QUEUES: tuple[str, ...] = ("interactive", "analysis", "io")


@dataclass(frozen=True)
class PoolPolicy:
    """One named pool: its queue, its size and what runs there."""

    queue: str
    threads: int
    purpose: str


#: Every named pool the application uses. A pool's size is its own limit; the
#: sizes of one queue's pools add up to at most that queue's limit.
POOLS: dict[str, PoolPolicy] = {
    # interactive: work a view is waiting on to draw
    "render-tiles": PoolPolicy("interactive", 4, "render view tiles"),
    "precision-render": PoolPolicy("interactive", 4, "advanced render view tiles"),
    "precision-render-ortho": PoolPolicy(
        "interactive", 2, "advanced render view, Ortho-View side panes"),
    "voronoi-field": PoolPolicy(
        "interactive", 1, "Voronoi density fields, one at a time (Qhull memory peaks)"),
    "voronoi-field-ortho": PoolPolicy(
        "interactive", 1, "Voronoi density fields of the Ortho-View side panes"),
    "tiff-plane": PoolPolicy("interactive", 1, "image viewer planes"),
    "tracking-index": PoolPolicy("interactive", 1, "Tracking View track index"),
    # analysis: computations the user started
    "post-load": PoolPolicy(
        "analysis", 2, "density, precision and Z scaling after a dataset opens"),
    "particle-average": PoolPolicy("analysis", 1, "particle averaging"),
    "drift-correction": PoolPolicy("analysis", 1, "drift correction"),
    "tracking-analysis": PoolPolicy("analysis", 1, "tracking analysis"),
    "mfv-script": PoolPolicy("analysis", 2, "mfv.run.background from scripts and plugins"),
    # io: reading and writing files
    "file-io": PoolPolicy("io", 2, "opening and saving datasets"),
    "zarr-io": PoolPolicy("io", 2, "Zarr stores"),
    "tiff-io": PoolPolicy("io", 2, "image viewer files"),
    "msr-work": PoolPolicy("io", 2, "MSR reader: parsing and exporting"),
    "msr-open": PoolPolicy("io", 1, "MSR reader: building the datasets to open"),
    "python-packages": PoolPolicy("io", 1, "installing external Python packages"),
    "update-check": PoolPolicy("io", 1, "checking for an application update"),
}

#: The most threads each queue may run at once.
QUEUE_LIMITS: dict[str, int] = {"interactive": 14, "analysis": 7, "io": 11}


def _check_policy() -> None:
    if set(QUEUE_LIMITS) != set(QUEUES):
        raise ValueError("every queue needs a limit, and only queues have one")
    for name, policy in POOLS.items():
        if policy.queue not in QUEUES:
            raise ValueError(f"pool {name!r} names no queue: {policy.queue!r}")
        if policy.threads < 1:
            raise ValueError(f"pool {name!r} needs at least one thread")
    for queue, limit in QUEUE_LIMITS.items():
        used = sum(p.threads for p in POOLS.values() if p.queue == queue)
        if used > limit:
            raise ValueError(f"the {queue} queue's pools need {used} threads, "
                             f"more than its limit of {limit}")


_check_policy()


def pool_policy(name: str) -> PoolPolicy:
    """The policy of the named pool; a name not declared here is an error."""
    try:
        return POOLS[name]
    except KeyError:
        raise KeyError(f"thread pool {name!r} is not declared in core/execution.py") from None


def pool_size(name: str) -> int:
    """How many threads the named pool runs: its declared size, never more
    than the machine has cores."""
    return max(1, min(pool_policy(name).threads, int(os.cpu_count() or 1)))


# ----------------------------------------------------------------- native threads
def default_native_threads(cores: int | None = None) -> int:
    """How many native threads analyses may run at once on *cores* cores.

    A quarter of the cores (at least one) stays with the interactive queue and
    the GUI thread; analyses share the rest.
    """
    cores = max(1, int(os.cpu_count() or 1) if cores is None else int(cores))
    return max(1, cores - max(1, cores // 4))


class Lease:
    """Threads leased from a :class:`ThreadBudget`; give them back with
    :meth:`release` (once is enough, twice changes nothing)."""

    def __init__(self, budget: ThreadBudget, taken: int) -> None:
        self._budget = budget
        self._taken = taken
        #: How many threads the leasing call may use: always at least one,
        #: because the thread making the call needs no lease.
        self.threads = max(1, taken)

    def release(self) -> None:
        taken, self._taken = self._taken, 0
        if taken:
            self._budget._give_back(taken)


class ThreadBudget:
    """A fixed number of native threads, leased by the analyses running now.

    One lease takes at most *per_lease* of them (default two thirds), so an
    analysis that starts while another holds a lease still gets several threads
    rather than one: a lease is held for a whole call and cannot be shrunk once
    it is running. Measured on 16 cores, a KD-tree range count is no faster on
    more than 8 threads (0.30 s on 8, 12 or 16) and eight times slower on one.
    """

    def __init__(self, total: int, *, per_lease: int | None = None) -> None:
        if int(total) < 1:
            raise ValueError("a thread budget needs at least one thread")
        self._total = int(total)
        cap = max(1, (2 * self._total) // 3) if per_lease is None else int(per_lease)
        self._per_lease = max(1, min(cap, self._total))
        self._free = self._total
        self._lock = threading.Lock()

    @property
    def total(self) -> int:
        return self._total

    @property
    def per_lease(self) -> int:
        return self._per_lease

    @property
    def free(self) -> int:
        with self._lock:
            return self._free

    def acquire(self, wanted: int | None = None) -> Lease:
        """Lease up to *wanted* threads (None: as many as one lease may take),
        never more than are free. Never waits."""
        with self._lock:
            limit = min(self._per_lease, self._free)
            if wanted is not None:
                limit = min(int(wanted), limit)
            taken = max(0, limit)
            self._free -= taken
        return Lease(self, taken)

    @contextmanager
    def lease(self, wanted: int | None = None) -> Iterator[int]:
        """Hold a lease for the ``with`` block; yields how many threads to use."""
        held = self.acquire(wanted)
        try:
            yield held.threads
        finally:
            held.release()

    def _give_back(self, taken: int) -> None:
        with self._lock:
            self._free = min(self._total, self._free + taken)


#: The process's budget of native threads for analyses.
CPU_THREADS = ThreadBudget(default_native_threads())


def cpu_threads(wanted: int | None = None):
    """Lease threads from :data:`CPU_THREADS` for a ``with`` block.

    ``with cpu_threads() as n: tree.query_ball_point(..., workers=n)`` -- the
    call uses what is free now, and never more than the budget.
    """
    return CPU_THREADS.lease(wanted)


def acquire_cpu_threads(wanted: int | None = None) -> Lease:
    """Lease threads from :data:`CPU_THREADS` where a ``with`` block does not
    fit (release the lease in a ``finally``)."""
    return CPU_THREADS.acquire(wanted)
