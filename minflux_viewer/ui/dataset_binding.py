"""Windows bound to a dataset by key, while index-based code still reads them.

A window used to remember its dataset by index, and a close re-keyed every
registry and rewrote every window's ``_idx`` by hand -- the code that dropped
the next dataset's windows from their registries and let a garbage collection
destroy them on screen. Stabilization Phase 3, slice 3.5, binds a window to its
dataset's :class:`~minflux_viewer.core.dataset_key.DatasetKey` instead:

* :class:`DatasetIndex` is the **temporary** ``_idx`` / ``_dataset_idx``
  property: it reads as the dataset's current position and an assignment
  rebinds the window to the dataset at that position. Existing index-based code
  keeps working and can no longer go stale.
* :class:`DatasetWindowMap` is a registry keyed by key that still accepts a
  position wherever a dict of positions was used, so its entries never need
  re-keying when an earlier dataset closes.
* :func:`event_index` resolves a typed dataset event (or, while the index
  signals remain, a plain index) to the current position.

All three are retired with the index signals (slice 3.8) once their callers
use keys directly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from ..core.dataset_key import DatasetKey


def event_index(state: Any, payload: Any) -> int | None:
    """The current position of the dataset *payload* names.

    *payload* is a typed dataset event (resolved through its key, so it is
    right whatever closed in between) or a plain index, passed through.
    """
    key = getattr(payload, "key", None)
    if isinstance(key, DatasetKey):
        return state.index_of(key)
    if payload is None or hasattr(payload, "key"):
        return None
    return payload


class DatasetIndex:
    """Temporary ``_idx``-style property backed by the window's dataset key.

    Reading gives the dataset's current position (None once it has closed, or
    when the window has no dataset); assigning a position binds the window to
    the dataset there. The key lives in ``_dataset_key``; the window's state is
    ``self._state``, which must be set before the first assignment.
    """

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        key = getattr(obj, "_dataset_key", None)
        if key is None:
            return None
        return obj._state.index_of(key)

    def __set__(self, obj: Any, idx: int | None) -> None:
        obj._dataset_key = None if idx is None else obj._state.key_at(idx)


class DatasetWindowMap:
    """Windows registered per dataset, keyed by dataset key.

    Indexed by position (or by key) wherever a ``dict[int, window]`` was, but
    stored by key, so an entry stays with its dataset when an earlier one
    closes. ``None`` is a key of its own (a window opened with no dataset).
    Iteration, ``items()``, ``keys()`` and ``values()`` cover the entries
    whose dataset is open, at their current positions.
    """

    def __init__(self, state: Any) -> None:
        self._state = state
        self._by_key: dict[DatasetKey | None, Any] = {}

    # -- key resolution ---------------------------------------------------
    def _key(self, idx: Any) -> DatasetKey | None:
        if idx is None or isinstance(idx, DatasetKey):
            return idx
        key = self._state.key_at(idx)
        if key is None:
            raise KeyError(idx)
        return key

    def _index(self, key: DatasetKey | None) -> int | None:
        return None if key is None else self._state.index_of(key)

    def _live(self) -> list[tuple[int | None, Any]]:
        out = []
        for key, window in list(self._by_key.items()):
            idx = self._index(key)
            if key is None or idx is not None:
                out.append((idx, window))
        return out

    # -- the dict interface the call sites use -----------------------------
    def __getitem__(self, idx: Any) -> Any:
        return self._by_key[self._key(idx)]

    def __setitem__(self, idx: Any, window: Any) -> None:
        self._by_key[self._key(idx)] = window

    def __delitem__(self, idx: Any) -> None:
        del self._by_key[self._key(idx)]

    def __contains__(self, idx: Any) -> bool:
        try:
            return self._key(idx) in self._by_key
        except KeyError:
            return False

    def __iter__(self) -> Iterator[int | None]:
        return iter([idx for idx, _window in self._live()])

    def __len__(self) -> int:
        return len(self._live())

    def __bool__(self) -> bool:
        return bool(self._live())

    def get(self, idx: Any, default: Any = None) -> Any:
        try:
            return self._by_key.get(self._key(idx), default)
        except KeyError:
            return default

    def pop(self, idx: Any, *default: Any) -> Any:
        try:
            key = self._key(idx)
        except KeyError:
            if default:
                return default[0]
            raise
        if default:
            return self._by_key.pop(key, default[0])
        return self._by_key.pop(key)

    def keys(self) -> list[int | None]:
        return [idx for idx, _window in self._live()]

    def values(self) -> list[Any]:
        return [window for _idx, window in self._live()]

    def items(self) -> list[tuple[int | None, Any]]:
        return self._live()

    def clear(self) -> None:
        self._by_key.clear()

    # -- by key ------------------------------------------------------------
    def pop_key(self, key: DatasetKey, default: Any = None) -> Any:
        """Remove and return the window of the dataset with *key*."""
        return self._by_key.pop(key, default)

    def discard_where(self, predicate: Callable[[Any], bool]) -> None:
        """Remove every entry whose window satisfies *predicate*, open or not."""
        for key, window in list(self._by_key.items()):
            if predicate(window):
                self._by_key.pop(key, None)

    def __repr__(self) -> str:
        return f"DatasetWindowMap({dict(self._live())!r})"
