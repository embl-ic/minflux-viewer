"""Submission-time export inputs, independent of later application edits."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import AttrStore, MinfluxDataset


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
            base = value
            immutable = not value.dtype.hasobject
            while isinstance(base, np.ndarray):
                immutable = immutable and not base.flags.writeable
                base = base.base
            immutable = immutable and (base is None or isinstance(base, bytes))
            result = value.view() if immutable else copy.deepcopy(value, self.memo)
            result.flags.writeable = False
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
