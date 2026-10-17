"""Submission-time export inputs, independent of later application edits."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .computation_snapshot import readonly_array
from .dataset import AttrStore, MinfluxDataset
from .job_resources import JOB_RESOURCES


class _Capture:
    def __init__(self) -> None:
        self.memo: dict[int, Any] = {}

    def capture(self, value):
        if id(value) in self.memo:
            return self.memo[id(value)]
        if isinstance(value, Path) or callable(value):
            # A materialized Path can carry the lease keeping its file alive.
            return value
        if isinstance(value, np.ndarray):
            result = readonly_array(value)
        elif isinstance(value, AttrStore):
            result = AttrStore()
            self.memo[id(value)] = result
            for key, item in value.items():
                if key != "ftr":
                    result[key] = self.capture(item)
            object.__setattr__(result, "_sealed", True)
        elif is_dataclass(value) and not isinstance(value, type):
            result = copy.copy(value)
            self.memo[id(value)] = result
            for item in fields(value):
                object.__setattr__(result, item.name, self.capture(getattr(value, item.name)))
            if isinstance(result, MinfluxDataset):
                result.attr.bind_virtual("ftr", result, "filter_mask")
        elif isinstance(value, dict):
            result = {}
            self.memo[id(value)] = result
            result.update((key, self.capture(item)) for key, item in value.items())
        elif isinstance(value, list):
            result = []
            self.memo[id(value)] = result
            result.extend(self.capture(item) for item in value)
        elif isinstance(value, tuple):
            result = tuple(self.capture(item) for item in value)
        else:
            result = copy.deepcopy(value, self.memo)
        self.memo[id(value)] = result
        return result


def save_snapshot(ds: MinfluxDataset, context: dict | None = None) -> tuple[MinfluxDataset, dict]:
    """Freeze data and options together, deduplicating shared overlay members.

    Sealed immutable columns are shared; writable buffers and mutable metadata
    are copied. Source files remain external inputs, not copied acquisitions.
    Capture on the GUI thread after the final dialog acceptance. Published
    arrays must not concurrently be mutated from arbitrary extension threads.
    """
    capture = _Capture()
    return capture.capture(ds), capture.capture(context or {})


def estimate_save_bytes(ds: MinfluxDataset, context: dict | None = None) -> int:
    """Conservative array payload allowance for capture and writer temporaries.

    Shared arrays are counted once. Compression/library overhead is estimated,
    not an allocator guarantee; external source-image payloads are excluded.
    """
    seen: set[int] = set()

    def size(value) -> int:
        if id(value) in seen:
            return 0
        seen.add(id(value))
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        if isinstance(value, AttrStore):
            return sum(size(item) for key, item in value.items() if key != "ftr")
        if is_dataclass(value) and not isinstance(value, type):
            return sum(size(getattr(value, item.name)) for item in fields(value))
        if isinstance(value, dict):
            return sum(size(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(size(item) for item in value)
        return 0

    return 1024**2 + 3 * (size(ds) + size(context or {}))


def admitted_save_snapshot(ds: MinfluxDataset, context: dict | None = None):
    """Reserve before copying any mutable acquisition buffers."""
    lease = JOB_RESOURCES.acquire(estimate_save_bytes(ds, context))
    prepared = False
    try:
        snapshot, options = save_snapshot(ds, context)
        prepared = True
        return snapshot, options, lease
    finally:
        if not prepared:
            lease.release()
