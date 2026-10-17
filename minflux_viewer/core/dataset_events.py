"""Typed events about open datasets.

``AppState`` has always announced dataset changes with an index, and an index
names a *position*, which shifts whenever an earlier dataset closes. These
events name the dataset by its :class:`~.dataset_key.DatasetKey` instead, and
carry the dataset itself. A removal is the case that needs it most: by the time
listeners run the dataset has left the registry, so :class:`DatasetRemoved`
carries everything a listener needs: the key, the dataset, where it was and
which dataset is active now.

They are the sole dataset notifications emitted by ``AppState.dataset_events``;
the positional dataset signals have been retired. Like the keys, these events
are internal: nothing here is published to scripts.

Qt-free on purpose, so the events can be built and compared in plain tests.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, fields
from typing import Any

from .dataset_key import DatasetKey


class _DatasetEvent:
    """Equality for events: the dataset by identity, everything else by value.

    A dataset is a dataclass holding numpy arrays, so its generated ``__eq__``
    compares arrays and raises for two different datasets. Which dataset an
    event is about is a question of identity anyway.
    """

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        for f in fields(self):  # type: ignore[arg-type]
            mine, theirs = getattr(self, f.name), getattr(other, f.name)
            if f.name == "dataset":
                if mine is not theirs:
                    return False
            elif mine != theirs:
                return False
        return True

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, eq=False)
class DatasetAdded(_DatasetEvent):
    """A dataset was opened; it is now at ``index``."""

    key: DatasetKey
    dataset: Any
    index: int


@dataclass(frozen=True, eq=False)
class DatasetRemoved(_DatasetEvent):
    """A dataset was closed.

    When this is emitted the registry is already settled: ``key`` no longer
    resolves, ``dataset`` is no longer listed, and ``new_active_key`` is the
    active dataset (None when nothing is left open).
    """

    key: DatasetKey
    dataset: Any
    old_index: int
    new_active_key: DatasetKey | None


@dataclass(frozen=True, eq=False)
class ActiveDatasetChanged(_DatasetEvent):
    """A different dataset became active (``key`` None: none is).

    Emitted only when the active *dataset* changes. Closing an earlier dataset
    shifts the active one's position but not the dataset, so it emits
    :class:`DatasetRemoved` and no active change.
    """

    key: DatasetKey | None
    dataset: Any
    previous_key: DatasetKey | None


class ChangeAspect(enum.Enum):
    """What changed about a dataset, one per change notification."""

    FILTER = "filter"
    ATTRIBUTES = "attributes"
    ROI_SELECTION = "roi_selection"
    CALIBRATION = "calibration"
    OVERLAY_TRANSFORM = "overlay_transform"


@dataclass(frozen=True, eq=False)
class DatasetChanged(_DatasetEvent):
    """Something about an open dataset changed; ``aspect`` says what."""

    key: DatasetKey
    dataset: Any
    aspect: ChangeAspect


@dataclass(frozen=True, eq=False)
class ManualAlignmentRequested(_DatasetEvent):
    """The UI was asked to start interactive overlay alignment for a dataset."""

    key: DatasetKey
    dataset: Any
