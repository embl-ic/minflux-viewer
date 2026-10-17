"""Windows bound to a dataset by key, while index-based code still reads them.

A window used to remember its dataset by index, and a close re-keyed every
registry and rewrote every window's ``_idx`` by hand -- the code that dropped
the next dataset's windows from their registries and let a garbage collection
destroy them on screen. Stabilization Phase 3 binds a window to its dataset's
:class:`~minflux_viewer.core.dataset_key.DatasetKey` instead:

* :class:`DatasetIndex` is the ``_idx`` / ``_dataset_idx`` property: it reads
  as the dataset's current position (None once it has closed) and an
  assignment rebinds the window to the dataset at that position, so the
  index-based code inside a window cannot go stale. Each window declares only
  the name it uses.
* :func:`window_dataset_key` / :func:`window_dataset_index` are how code
  *outside* a window asks which dataset it shows -- through the key, whichever
  name the window uses internally.
* :class:`DatasetWindowMap` is a registry keyed by key that still accepts a
  position wherever a dict of positions was used, so its entries never need
  re-keying when an earlier dataset closes.
* :func:`event_index` / :func:`event_key` resolve a typed dataset event (or an
  index an internal caller passes) to the current position / to the key.
* :func:`window_channel_keys` / :func:`window_draws_dataset` say which datasets
  an overlay view's channels draw; a channel holds its dataset's key.

The position-backed names stay while code inside the windows still reads a
position (stabilization Phase 4).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from minflux_viewer.core.dataset_key import DatasetKey


def event_index(state: Any, payload: Any) -> int | None:
    """The current position of the dataset *payload* names.

    *payload* is a typed dataset event (resolved through its key, so it is
    right whatever closed in between) or a plain index an internal caller
    passes, returned as it is.
    """
    key = getattr(payload, "key", None)
    if isinstance(key, DatasetKey):
        return state.index_of(key)
    if payload is None or hasattr(payload, "key"):
        return None
    return payload


def event_key(state: Any, payload: Any) -> DatasetKey | None:
    """The key of the dataset *payload* names.

    *payload* is a typed dataset event, whose key is returned whether or not
    the dataset is still open, or a plain index an internal caller passes,
    resolved to the key at that position.
    """
    key = getattr(payload, "key", None)
    if isinstance(key, DatasetKey):
        return key
    if payload is None or hasattr(payload, "key"):
        return None
    return state.key_at(payload)


def window_dataset_key(window: Any) -> DatasetKey | None:
    """The key of the dataset *window* is bound to, or None.

    A closed dataset's key is still returned: it names what the window showed.
    """
    try:
        key = getattr(window, "_dataset_key", None)
    except RuntimeError:                    # a window whose C++ half is gone
        return None
    return key if isinstance(key, DatasetKey) else None


def window_channel_keys(window: Any, *, visible_only: bool = False) -> list[DatasetKey]:
    """The keys of the datasets *window*'s channel list draws, in channel order.

    An overlay view draws one channel per member. Channels that are not a
    dataset (an image channel) and windows without a channel list contribute
    nothing; with *visible_only*, neither does a hidden channel.
    """
    try:
        channels = getattr(window, "_channels", None) or []
        return [
            ch["dataset_key"] for ch in channels
            if isinstance(ch, dict) and isinstance(ch.get("dataset_key"), DatasetKey)
            and (not visible_only or ch.get("visible", True))
        ]
    except RuntimeError:                    # a window whose C++ half is gone
        return []


def window_draws_dataset(window: Any, state: Any, dataset: Any,
                         *, visible_only: bool = False) -> bool:
    """Whether a channel of *window* draws *dataset* (a position or a key)."""
    key = dataset if isinstance(dataset, DatasetKey) or dataset is None else state.key_at(dataset)
    return key is not None and key in window_channel_keys(window, visible_only=visible_only)


def window_dataset_index(window: Any) -> int | None:
    """The current position of the dataset *window* shows, or None.

    A window bound by key answers through its key, so the answer follows its
    dataset and is None once that has closed. Any other window answers with
    the plain index it keeps (``dataset_idx``, which may be a method,
    ``_dataset_idx`` or ``_idx``).
    """
    if window is None:
        return None
    key = window_dataset_key(window)
    if key is not None:
        try:
            return window._state.index_of(key)
        except (AttributeError, RuntimeError):
            return None
    for name in ("dataset_idx", "_dataset_idx", "_idx"):
        try:
            value = getattr(window, name, None)
            if callable(value):
                value = value()
        except (RuntimeError, TypeError):
            continue
        if type(value) is int:
            return value
    return None


class DatasetIndex:
    """An ``_idx``-style property backed by the window's dataset key.

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
