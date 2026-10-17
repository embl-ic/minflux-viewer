"""Versioned, streamed extraction of embedded images with live-path leases.

The budget covers unleased cache files, not images retained by consumers. Each
returned Path keeps its exact version alive; consumers must retain that object
(not just its string) for as long as they may reopen the extracted image.
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import os
import shutil
import sys
import threading
import time
import uuid
import weakref
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .temporary_images import TemporaryImages

if sys.platform == "win32":
    from pathlib import WindowsPath as _NativePath
else:
    from pathlib import PosixPath as _NativePath

_LOG = logging.getLogger(__name__)


class _LeasedPath(_NativePath):
    __slots__ = ("_lease", "__weakref__")


@dataclass
class _Entry:
    path: Path
    size: int
    leases: int = 0
    last_used: float = 0.0


def _version(stat: os.stat_result) -> tuple:
    # On Windows 3.12 fstat/stat can expose different ctime semantics for the
    # same open file. File identity + size/mtime and the ZIP member CRC bind it.
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


class ImageMaterializationCache:
    def __init__(self, directory: Path | None = None, *, budget_bytes: int = 512 * 1024**2,
                 max_member_bytes: int = 4 * 1024**3, max_unleased_files: int = 512,
                 max_idle_seconds: float = 86400) -> None:
        if (budget_bytes < 0 or max_member_bytes <= 0 or max_unleased_files < 0
                or not 0 <= max_idle_seconds < float("inf")):
            raise ValueError("Image cache limits must be nonnegative, with a positive member limit.")
        self._directory = directory
        self.budget_bytes = int(budget_bytes)
        self.max_member_bytes = int(max_member_bytes)
        self.max_unleased_files = int(max_unleased_files)
        self.max_idle_seconds = float(max_idle_seconds)
        self._owned_directory: TemporaryImages | None = None
        self._closed = False
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.RLock()

    @property
    def retained_bytes(self) -> int:
        with self._lock:
            return sum(entry.size for entry in self._entries.values() if not entry.leases)

    def _evict(self) -> None:
        total = self.retained_bytes
        count = sum(not entry.leases for entry in self._entries.values())
        now = time.monotonic()
        for key, entry in list(self._entries.items()):
            if entry.leases:
                continue
            expired = now - entry.last_used >= self.max_idle_seconds
            if self._closed or expired or total > self.budget_bytes or count > self.max_unleased_files:
                try:
                    entry.path.unlink(missing_ok=True)
                except OSError:
                    _LOG.warning("Could not evict extracted image %s", entry.path, exc_info=True)
                    continue
                del self._entries[key]
                total -= entry.size
                count -= 1

    @property
    def retained_files(self) -> int:
        with self._lock:
            return sum(not entry.leases for entry in self._entries.values())

    def _remove_empty_directory(self) -> None:
        if self._entries:
            return
        if self._owned_directory is not None:
            self._owned_directory.close()
        elif self._directory is not None and self._directory.is_dir() and not any(self._directory.iterdir()):
            self._directory.rmdir()

    def _release(self, key: str) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.leases -= 1
                entry.last_used = time.monotonic()
                self._entries.move_to_end(key)
                try:
                    self._evict()
                    if self._closed:
                        self._remove_empty_directory()
                except OSError:
                    _LOG.exception("Could not evict an extracted image; retaining it for later cleanup")

    def _leased_path(self, key: str) -> Path:
        entry = self._entries[key]
        self._entries.move_to_end(key)
        entry.leases += 1
        path: Any = _LeasedPath(entry.path)
        path._lease = weakref.finalize(path, self._release, key)
        return path

    @staticmethod
    def _copy_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, output: Path) -> None:
        with archive.open(info) as source, output.open("xb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        if output.stat().st_size != info.file_size:
            raise OSError("Embedded image extraction size mismatch")

    def materialize(self, record: dict) -> Path:
        package = Path(record["package_path"]).resolve()
        member = str(record["path"])
        with self._lock, package.open("rb") as handle, zipfile.ZipFile(handle) as archive:
            if self._closed:
                raise ValueError("Image cache is closed")
            version = _version(os.fstat(handle.fileno()))
            members = [info for info in archive.infolist() if info.filename == member]
            if len(members) != 1:
                raise ValueError("Missing or duplicate embedded image member")
            info = members[0]
            if info.is_dir() or info.file_size > self.max_member_bytes:
                raise ValueError("Embedded image exceeds the extraction size limit")
            stamp = (str(package), version, member, info.CRC, info.file_size,
                     info.compress_size, info.header_offset)
            key = hashlib.sha256(repr(stamp).encode("utf-8")).hexdigest()
            if _version(package.stat()) != version:
                raise OSError("Image package changed while opening")
            self._evict()
            entry = self._entries.get(key)
            if entry is not None and entry.path.is_file():
                return self._leased_path(key)
            if self._directory is None:
                self._owned_directory = TemporaryImages()
                self._directory = self._owned_directory.path
            self._directory.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(self._directory).free < info.file_size:
                raise OSError("Not enough disk space to extract embedded image")
            target = self._directory / f"{key[:24]}-{Path(member).name}"
            partial = self._directory / f".{uuid.uuid4().hex}.partial"
            try:
                self._copy_member(archive, info, partial)  # ZipExtFile verifies CRC at EOF
                if (_version(os.fstat(handle.fileno())) != version
                        or _version(package.stat()) != version):
                    raise OSError("Image package changed during extraction")
                os.replace(partial, target)
            finally:
                partial.unlink(missing_ok=True)
            self._entries[key] = _Entry(target, info.file_size, last_used=time.monotonic())
            return self._leased_path(key)

    def close(self) -> None:
        """Retire the cache; defer leased files until their last consumer closes."""
        with self._lock:
            self._closed = True
            self._evict()
            try:
                self._remove_empty_directory()
            except OSError:
                _LOG.warning("Could not remove image-cache directory", exc_info=True)


IMAGE_CACHE = ImageMaterializationCache()
atexit.register(IMAGE_CACHE.close)
