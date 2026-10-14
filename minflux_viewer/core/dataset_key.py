"""Stable identity of an open dataset.

A dataset's position in ``AppState.datasets`` shifts whenever an earlier one is
closed, so a position is a poor name for "this dataset": anything holding one
across a close ends up pointing at another dataset. A :class:`DatasetKey` is
assigned when a dataset is added, stays the same while it is open, and is never
given to another dataset.

Keys are internal. They are not published to scripts (``mfv`` still addresses
datasets by index, name or object) and never written to a file: a saved
dataset is matched by its acquisition identity, not by a key from one session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetKey:
    """An open dataset's identity within one session, shown as ``ds-7``.

    Deliberately its own type: it compares unequal to the integer and to the
    string it is displayed as, and cannot be used as a list index, so it can
    never be confused with a position or a dataset name.
    """

    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError(f"a dataset key is a positive int, not {self.value!r}")
        if self.value < 1:
            raise ValueError(f"a dataset key is a positive int, not {self.value}")

    def __str__(self) -> str:
        return f"ds-{self.value}"


@dataclass(frozen=True, eq=False)
class DatasetEntry:
    """One open dataset and its key, as listed by ``AppState.entries``.

    Compared by identity: two entries are the same only if they are the same
    registration, whatever the datasets' contents.
    """

    key: DatasetKey
    dataset: Any
