"""Same-filesystem file publication with in-process destination serialization.

All members are prepared before publication. Handled replacement failures roll
back; this is not a multi-file OS transaction or a power-loss durability promise.
If restoration fails, backups are retained and their paths appear in the error.
"""

from __future__ import annotations

import os
import tempfile
import threading
import uuid
import weakref
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from typing import Any

_LOCK_GUARD = threading.Lock()
_LOCKS: weakref.WeakValueDictionary[str, Any] = weakref.WeakValueDictionary()


@contextmanager
def destination_locks(paths: Iterable[Path]) -> Iterator[None]:
    """Serialize resolved file/directory destinations in this process only."""
    with _LOCK_GUARD:
        locks = []
        for key in sorted({os.path.normcase(str(Path(p).resolve())) for p in paths}):
            lock = _LOCKS.get(key)
            if lock is None:
                lock = threading.RLock()
                _LOCKS[key] = lock
            locks.append(lock)
    with ExitStack() as stack:
        for lock in locks:
            stack.enter_context(lock)
        yield


@contextmanager
def file_transaction(paths: Iterable[Path], *, protected: Iterable[Path] = (),
                     overwrite: bool = True) -> Iterator[dict[Path, Path]]:
    """Yield same-basename staged paths, then publish with recoverable rollback.

    Locks cover preparation through cleanup; external writers are not excluded.
    ``overwrite=False`` refuses existing destinations before preparation and
    rechecks before publication. It does not reserve paths against other processes.
    """
    originals = list(paths)
    resolved = {path: path.resolve() for path in originals}
    if len(set(resolved.values())) != len(originals):
        raise ValueError("Transaction destinations must be distinct.")
    stages: dict[Path, Path] = {}
    directories: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    committed = False
    with destination_locks([*resolved.values(), *protected]):
        if not overwrite:
            for original in originals:
                if original.is_symlink():
                    raise FileExistsError(original)
            for target in resolved.values():
                if target.exists():
                    raise FileExistsError(target)
        try:
            for original, target in resolved.items():
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and not target.is_file():
                    raise IsADirectoryError(target)
                directory = directories.get(target.parent)
                if directory is None:
                    directory = Path(tempfile.mkdtemp(prefix=".mfv-save-", dir=target.parent))
                    directories[target.parent] = directory
                stage = directory / original.name
                if stage in stages.values():
                    raise ValueError("Transaction aliases have conflicting output basenames.")
                stages[original] = stage
            yield stages
            if any(not stage.is_file() for stage in stages.values()):
                raise FileNotFoundError("A transaction member was not prepared.")
            if not overwrite:
                for target in resolved.values():
                    if target.exists():
                        raise FileExistsError(target)
            for original, target in resolved.items():
                if target.exists():
                    backup = stages[original].parent / (uuid.uuid4().hex + ".backup")
                    os.replace(target, backup)
                    backups[target] = backup
                os.replace(stages[original], target)
                published.append(target)
            committed = True
        finally:
            failures = []
            if not committed:
                for target in reversed(list(resolved.values())):
                    try:
                        if target in backups:
                            os.replace(backups[target], target)
                        elif target in published:
                            target.unlink(missing_ok=True)
                    except OSError as exc:
                        failures.append(f"{target}: {exc}; backup={backups.get(target)}")
            for stage in stages.values():
                stage.unlink(missing_ok=True)
            if committed:
                for backup in backups.values():
                    backup.unlink(missing_ok=True)
            for directory in directories.values():
                if not any(directory.iterdir()):
                    directory.rmdir()
            if failures:
                raise OSError("Export rollback failed; recovery files retained: " + "; ".join(failures))


def atomic_file_writer(writer: Callable[..., Any]) -> Callable[..., Any]:
    """Protect a single-path serializer, including direct/internal callers."""
    @wraps(writer)
    def write(path: Path, *args: Any, **kwargs: Any) -> Any:
        path = Path(path)
        with file_transaction([path]) as stages:
            return writer(stages[path], *args, **kwargs)
    return write
