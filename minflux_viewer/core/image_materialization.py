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
import tempfile
import threading
import uuid
import weakref
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _version(stat: os.stat_result) -> tuple:
    # On Windows 3.12 fstat/stat can expose different ctime semantics for the
    # same open file. File identity + size/mtime and the ZIP member CRC bind it.
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


class ImageMaterializationCache:
    def __init__(self, directory: Path | None = None, *, budget_bytes: int = 512 * 1024**2,
                 max_member_bytes: int = 4 * 1024**3) -> None:
        if budget_bytes < 0 or max_member_bytes <= 0:
            raise ValueError("Image cache limits must be nonnegative, with a positive member limit.")
        self._directory = directory
        self.budget_bytes = int(budget_bytes)
        self.max_member_bytes = int(max_member_bytes)
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.RLock()

    @property
    def retained_bytes(self) -> int:
        with self._lock:
            return sum(entry.size for entry in self._entries.values() if not entry.leases)

    def _evict(self) -> None:
        for key, entry in list(self._entries.items()):
            if self.retained_bytes <= self.budget_bytes:
                break
            if not entry.leases:
                entry.path.unlink(missing_ok=True)
                del self._entries[key]

    def _release(self, key: str) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.leases -= 1
                try:
                    self._evict()
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
            entry = self._entries.get(key)
            if entry is not None and entry.path.is_file():
                return self._leased_path(key)
            self._evict()
            if self._directory is None:
                self._directory = Path(tempfile.mkdtemp(prefix="minflux-viewer-images-"))
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
            self._entries[key] = _Entry(target, info.file_size)
            return self._leased_path(key)

    def close(self) -> None:
        """Terminal process cleanup, after consumers have closed their handles."""
        with self._lock:
            for key, entry in list(self._entries.items()):
                try:
                    entry.path.unlink(missing_ok=True)
                    del self._entries[key]
                except OSError:
                    _LOG.exception("Could not remove extracted image %s", entry.path)
            if self._directory is not None and self._directory.is_dir() and not any(self._directory.iterdir()):
                self._directory.rmdir()


IMAGE_CACHE = ImageMaterializationCache()
atexit.register(IMAGE_CACHE.close)
