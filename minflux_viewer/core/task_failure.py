"""Plain worker failures: diagnostics without retaining exception frames."""

from __future__ import annotations

import traceback
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskFailure:
    exception_type: str
    message: str
    traceback: str
    code: str = "worker_error"

    @classmethod
    def capture(cls, error: BaseException) -> TaskFailure:
        kind = type(error)
        return cls(f"{kind.__module__}.{kind.__qualname__}", str(error),
                   "".join(traceback.format_exception(kind, error, error.__traceback__))[-65536:])


class TaskExecutionError(RuntimeError):
    """Error delivered to script callbacks, with structured worker diagnostics."""

    def __init__(self, failure: TaskFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure
        self.exception_type = failure.exception_type
        self.worker_traceback = failure.traceback
        self.code = failure.code
