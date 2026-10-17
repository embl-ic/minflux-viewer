"""Sequential CPU jobs with owned outputs and content-verified resume receipts.

No executor or UI is owned here. Exceptions propagate (fail-fast), and already
published items remain resumable. Checkpoints may raise any caller cancellation
exception; native readers/writers cannot be interrupted mid-call.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import __version__
from .dataset_io import load_dataset, save_dataset, validate_save_options
from .file_transactions import destination_locks

if TYPE_CHECKING:
    from .dataset import MinfluxDataset

Checkpoint = Callable[[], None]
Processor = Callable[["MinfluxDataset", dict[str, Any], Checkpoint], "MinfluxDataset | None"]
_RECEIPT = "receipt.json"
_MAX_RECEIPT_BYTES = 16 * 1024**2
_MAX_FILES = 100_000


def _checkpoint() -> None:
    """Default for synchronous callers that do not request cancellation."""


def _plain(path: Path) -> os.stat_result:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError(f"Batch paths must not contain links or junctions: {path}")
    return info


def _files(path: Path) -> list[Path]:
    info = _plain(path)
    if stat.S_ISREG(info.st_mode):
        return [path]
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"Batch paths must be regular files or directories: {path}")
    found = []
    for folder, directories, names in os.walk(path, followlinks=False):
        for name in directories:
            if not stat.S_ISDIR(_plain(Path(folder) / name).st_mode):
                raise ValueError("Unexpected batch directory entry")
        for name in names:
            child = Path(folder) / name
            if not stat.S_ISREG(_plain(child).st_mode):
                raise ValueError(f"Not a regular file: {child}")
            found.append(child)
            if len(found) > _MAX_FILES:
                raise ValueError("Batch file inventory exceeds 100,000 files")
    return sorted(found)


def _hash(path: Path, check: Checkpoint) -> str:
    before = _plain(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            check()
            block = stream.read(1024**2)
            if not block:
                break
            digest.update(block)
    after = _plain(path)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed while hashing: {path}")
    return digest.hexdigest()


def _inventory(path: Path, check: Checkpoint, *, exclude_receipt: bool = False) -> dict[str, str]:
    return {file.relative_to(path).as_posix() if file != path else ".": _hash(file, check)
            for file in _files(path) if not (exclude_receipt and file == path / _RECEIPT)}


def _inputs(source: Path, apply_sidecar: bool, check: Checkpoint) -> dict[str, Any]:
    from .save import metadata_sidecar_path

    data: dict[str, Any] = {"path": str(source), "sha256": _inventory(source, check)}
    sidecar = metadata_sidecar_path(source)
    if apply_sidecar and not source.name.lower().endswith((".zarr", ".zarr.zip")):
        data["sidecar_sha256"] = _hash(sidecar, check) if sidecar.exists() else None
    return data


def _json(value: Any) -> str:
    _validate_keys(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_keys(value: Any) -> None:
    if isinstance(value, dict) and any(not isinstance(key, str) for key in value):
        raise ValueError("Batch parameter keys must be strings")
    if isinstance(value, dict):
        for child in value.values():
            _validate_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_keys(child)


def run_batch(paths: Iterable[str | Path], output_dir: str | Path, process: Processor, *,
              operation: str, parameters: dict[str, Any] | None = None, fmt: str = "npy",
              content: str = "raw", filter_mode: str = "flag", apply_sidecar: bool = False,
              resume: bool = False, check_cancelled: Checkpoint | None = None,
              progress: Callable[[int, int, str], None] | None = None) -> list[dict[str, str]]:
    """Load/process/save one detached dataset at a time, never overwrite an item.

    ``operation`` identifies the callback AND its version (change it when code
    changes); JSON parameters are captured once and independently passed per item.
    A callback modifies its private dataset or returns a replacement. It must
    not touch Qt/viewer state. Resume verifies the complete request and all file
    hashes, not merely filename existence. Receipts are integrity records, not
    signatures against malicious modification. Other processes must not write
    these paths concurrently. Empty directories are not part of file inventories.
    """
    if not isinstance(operation, str) or not operation.strip() or not callable(process):
        raise ValueError("A callable processor and nonempty versioned operation ID are required")
    if parameters is not None and not isinstance(parameters, dict):
        raise ValueError("parameters must be a JSON dictionary")
    if isinstance(paths, (str, Path)):
        raise TypeError("paths must be a sequence of inputs, not one path")
    validate_save_options(fmt, content, filter_mode)
    captured = _json(parameters or {})
    check = check_cancelled or _checkpoint
    check()
    sources = [Path(path).absolute() for path in paths]
    for source in sources:
        # Validate the spelling before resolve can hide a final symlink.
        _plain(source)
    sources = [source.resolve(strict=True) for source in sources]
    if len(set(sources)) != len(sources):
        raise ValueError("Batch inputs must be distinct")
    root = Path(output_dir).absolute()
    if root.exists():
        _plain(root)
    root = root.resolve()
    if any(root == source or root.is_relative_to(source) or source.is_relative_to(root)
           for source in sources):
        raise ValueError("Batch input and output trees must not overlap")
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, str]] = []
    for index, source in enumerate(sources):
        check()
        if progress is not None:
            progress(index, len(sources), source.name)
        request = {"schema": "org.minflux-viewer.cpu-batch/1", "app_version": __version__,
                   "operation": operation, "parameters": json.loads(captured),
                   "fmt": fmt, "content": content, "filter_mode": filter_mode,
                   "apply_sidecar": bool(apply_sidecar), "input": _inputs(source, apply_sidecar, check)}
        target = root / ("item-" + hashlib.sha256(os.path.normcase(str(source)).encode("utf-8")).hexdigest()[:24])
        with destination_locks([target]):
            if target.exists() or target.is_symlink():
                if not resume:
                    raise FileExistsError(target)
                _plain(target)
                receipt = target / _RECEIPT
                if _plain(receipt).st_size > _MAX_RECEIPT_BYTES:
                    raise ValueError("Batch receipt exceeds 16 MiB")
                previous = json.loads(receipt.read_text(encoding="utf-8"))
                if (not isinstance(previous, dict) or previous.get("request") != request
                        or previous.get("outputs") != _inventory(target, check, exclude_receipt=True)):
                    raise ValueError(f"Resume refused: request or output changed: {target}")
                status = "resumed"
            else:
                with tempfile.TemporaryDirectory(prefix=".mfv-batch-", dir=root) as scratch:
                    stage = Path(scratch)
                    dataset = load_dataset(source, apply_sidecar=apply_sidecar)
                    check()
                    replacement = process(dataset, json.loads(captured), check)
                    if replacement is not None:
                        dataset = replacement
                    check()
                    save_dataset(dataset, stage / "data", fmt=fmt, content=content, filter_mode=filter_mode)
                    # Do not retain an acquisition between items or hash scans.
                    del dataset, replacement
                    check()
                    if _inputs(source, apply_sidecar, check) != request["input"]:
                        raise ValueError(f"Batch input changed during processing: {source}")
                    payload = _json({"request": request, "outputs": _inventory(stage, check)})
                    if len(payload.encode("utf-8")) > _MAX_RECEIPT_BYTES:
                        raise ValueError("Batch receipt exceeds 16 MiB")
                    (stage / _RECEIPT).write_text(payload, encoding="utf-8")
                    check()  # Last cancellation point before committing this item.
                    if target.exists():
                        raise FileExistsError(target)
                    stage.rename(target)
                status = "written"
        results.append({"source": str(source), "output_dir": str(target), "status": status})
    # No cancellation checkpoint after the last commit: committed output is real.
    return results
