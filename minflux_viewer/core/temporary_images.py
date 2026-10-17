"""Owned, locked image-cache directories and conservative abandoned cleanup."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import stat
import sys
import tempfile
import time
import uuid
from pathlib import Path

import psutil

_LOG = logging.getLogger(__name__)
_PREFIX = "minflux-viewer-images-"
_MARKER = ".owner.json"
_FORMAT = "org.minflux-viewer.temporary-images/1"
_FILE = re.compile(r"(?:[0-9a-f]{24}-.+|\.[0-9a-f]{32}\.partial)\Z")


def _plain(path: Path, *, directory: bool = False) -> bool:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        return False
    return stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)


def _lock(fd: int) -> bool:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _owner_alive(pid: int, started: float) -> bool:
    try:
        return abs(psutil.Process(pid).create_time() - started) < 0.001
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, OSError):
        return True  # unknown ownership is never permission to delete


def cleanup_abandoned_images(root: Path, *, minimum_age: float = 86400,
                             limit: int = 64) -> int:
    """Remove only old, unlocked, positively identified dead-owner flat caches.

    Legacy unmarked directories, symlinks/junctions and unknown contents are
    deliberately left alone. No recursive deletion or process termination.
    """
    root = root.resolve()
    removed = 0
    checked = 0
    for folder in root.glob(_PREFIX + "*"):
        token = folder.name.removeprefix(_PREFIX)
        if not re.fullmatch(r"[0-9a-f]{32}", token):
            continue
        checked += 1
        if checked > max(0, int(limit)):
            break
        marker = folder / _MARKER
        remove_marker = False
        fd = None
        try:
            if folder.parent != root or not _plain(folder, directory=True) or not _plain(marker):
                continue
            fd = os.open(marker, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
            if not _lock(fd):
                continue
            if os.fstat(fd).st_size > 4096:
                continue
            os.lseek(fd, 0, os.SEEK_SET)
            payload = json.loads(os.read(fd, 4097))
            if (payload.get("format") != _FORMAT or payload.get("token") != token
                    or type(payload.get("pid")) is not int or payload["pid"] <= 0):
                continue
            created, started = float(payload["created"]), float(payload["started"])
            if (not math.isfinite(created) or not math.isfinite(started)
                    or time.time() - created < max(0, minimum_age)
                    or _owner_alive(payload["pid"], started)):
                continue
            children = list(folder.iterdir())
            if any(not _plain(child) or (child.name != _MARKER and not _FILE.fullmatch(child.name))
                   for child in children):
                continue
            # Exact flat children only; never traverse another directory.
            for child in children:
                if child.name != _MARKER:
                    child.unlink()
            remove_marker = True
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            _LOG.debug("Temporary image cache left for a later cleanup: %s", folder, exc_info=True)
        finally:
            if fd is not None:
                os.close(fd)
        if remove_marker:
            try:
                marker.unlink()
                folder.rmdir()
                removed += 1
            except OSError:
                _LOG.debug("Temporary image directory removal deferred: %s", folder, exc_info=True)
    return removed


class TemporaryImages:
    def __init__(self, root: Path | None = None) -> None:
        root = Path(tempfile.gettempdir()) if root is None else root
        root = root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        cleanup_abandoned_images(root)
        token = uuid.uuid4().hex
        self.path = root / (_PREFIX + token)
        self.path.mkdir(mode=0o700)
        self.marker = self.path / _MARKER
        self._fd: int | None = os.open(self.marker, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        payload = dict(format=_FORMAT, token=token, pid=os.getpid(),
                       started=psutil.Process().create_time(), created=time.time())
        os.write(self._fd, json.dumps(payload).encode("utf-8"))
        if not _lock(self._fd):
            self.close()
            raise OSError("Could not lock the temporary image-cache owner")

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if self.path.is_dir() and list(self.path.iterdir()) == [self.marker]:
            self.marker.unlink()
            self.path.rmdir()
