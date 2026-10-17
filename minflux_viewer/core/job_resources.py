"""Non-waiting admission for estimated transient bytes and retained job inputs.

This is not an RSS quota. Imported data, live views, cached results and native
allocations omitted by an estimate remain outside the reservation accounting.
"""

from __future__ import annotations

from threading import Lock

import psutil

DEFAULT_MEMORY_PERCENT = 25
DEFAULT_JOB_LIMIT = 32


class ResourceAdmissionError(RuntimeError):
    """Work was refused before allocation/submission, without blocking Qt."""


class ResourceLease:
    def __init__(self, owner: JobResources, size: int, job: bool = True) -> None:
        self._owner = owner
        self.size = size
        self._released = False
        self.job = bool(job)

    def release(self) -> None:
        self._owner.release(self)

    def __del__(self) -> None:
        # A preparation failure before handing the lease to a task must not
        # permanently consume admission. No Qt object or user callback here.
        self.release()

    def __enter__(self) -> ResourceLease:
        return self

    def __exit__(self, *_exc) -> None:
        self.release()


class JobResources:
    def __init__(self, budget_bytes: int, job_limit: int = DEFAULT_JOB_LIMIT) -> None:
        self._lock = Lock()
        self._bytes = 0
        self._jobs = 0
        self.configure(budget_bytes, job_limit)

    def configure(self, budget_bytes: int, job_limit: int) -> None:
        if int(budget_bytes) < 1 or int(job_limit) < 1:
            raise ValueError("Operation byte budget and job limit must be positive")
        with self._lock:
            self.budget_bytes = int(budget_bytes)
            self.job_limit = int(job_limit)

    @property
    def reserved_bytes(self) -> int:
        with self._lock:
            return self._bytes

    @property
    def admitted_jobs(self) -> int:
        with self._lock:
            return self._jobs

    def acquire(self, estimated_bytes: int = 0, *, job: bool = True) -> ResourceLease:
        size = int(estimated_bytes)
        if size < 0:
            raise ValueError("A memory estimate cannot be negative")
        with self._lock:
            if (job and self._jobs >= self.job_limit) or self._bytes + size > self.budget_bytes:
                raise ResourceAdmissionError(
                    f"Background work admission refused: {self._jobs}/{self.job_limit} jobs, "
                    f"{self._bytes / 1024**2:.1f}/{self.budget_bytes / 1024**2:.1f} MiB reserved, "
                    f"request {size / 1024**2:.1f} MiB. Wait for work to finish, reduce the "
                    "operation size, or adjust the operation budget in Preferences > Data.")
            self._jobs += int(job)
            self._bytes += size
        return ResourceLease(self, size, job)

    def release(self, lease: ResourceLease) -> None:
        with self._lock:
            if lease._owner is not self or lease._released:
                return
            lease._released = True
            self._jobs -= int(lease.job)
            self._bytes -= lease.size


JOB_RESOURCES = JobResources(int(psutil.virtual_memory().total * DEFAULT_MEMORY_PERCENT / 100))


def configure_job_resources(prefs: dict) -> None:
    data = prefs.get("data", {})
    percent = float(data.get("operation_memory_percent", DEFAULT_MEMORY_PERCENT))
    if not 0 < percent <= 100:
        raise ValueError("Operation memory percentage must be in (0, 100]")
    JOB_RESOURCES.configure(int(psutil.virtual_memory().total * percent / 100),
                            int(data.get("background_job_limit", DEFAULT_JOB_LIMIT)))
