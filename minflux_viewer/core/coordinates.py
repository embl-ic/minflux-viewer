"""Runtime-only, byte-bounded coordinate views shared by open datasets.

The budget counts retained arrays, not process RSS: callers may still hold an
evicted view, and building a view needs temporary memory. Public APIs return
copies; internal consumers must not mutate these read-only views.
"""

from __future__ import annotations

import struct
import weakref
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future
from threading import RLock
from typing import Any

import numpy as np
import psutil

from .dataset_key import DatasetKey

DEFAULT_CACHE_PERCENT = 25
_CALIBRATED = object()
_TRANSFORM_CHUNK_ROWS = 262_144


def cache_budget_bytes(percent: float = DEFAULT_CACHE_PERCENT) -> int:
    """Convert a validated percentage of physical RAM to a retained-byte cap."""
    percent = float(percent)
    if not np.isfinite(percent) or not 0 <= percent <= 100:
        raise ValueError("Coordinate cache percentage must be between 0 and 100")
    return int(psutil.virtual_memory().total * (percent / 100))


def calibrated_coordinates(columns: tuple, z_scale: float) -> np.ndarray:
    """Build one writable array, preserving the order of Z calibration math."""
    columns = tuple(np.asarray(column, dtype=np.float64) for column in columns)
    if any(column.ndim != 1 or column.shape != columns[0].shape for column in columns):
        raise ValueError("Coordinate columns must be one-dimensional and equally sized")
    out = np.empty((len(columns[0]), 3), dtype=np.float64)
    for axis, column in enumerate(columns):
        np.multiply(column, 1e9, out=out[:, axis])
    out[:, 2] *= z_scale
    return out


def _matrix(transform: Any) -> np.ndarray | None:
    if isinstance(transform, dict):
        for name, size in (("matrix_4x4", 4), ("matrix_3x3", 3)):
            value = transform.get(name)
            if value is not None:
                matrix = np.array(value, dtype=np.float64, copy=True)
                if matrix.shape == (size, size):
                    return matrix
    return None


def apply_display_transform_nm(points_nm: np.ndarray, transform: dict | None) -> np.ndarray:
    """Apply an affine display record; never mutate the input coordinates."""
    pts = np.asarray(points_nm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return pts.copy()
    matrix = _matrix(transform)
    if matrix is None:
        return pts.copy()
    out = pts.copy()
    if matrix.shape == (4, 4):
        if out.shape[1] < 3:
            out = np.column_stack([out[:, :2], np.zeros(out.shape[0], dtype=np.float64)])
        dimensions = 3
    elif pts.shape[1] >= 2:
        dimensions = 2
    else:
        return out
    # Keep the original homogeneous matrix multiplication (including its
    # rounding), but bound its two temporary arrays independently of N.
    for start in range(0, len(out), _TRANSFORM_CHUNK_ROWS):
        part = out[start:start + _TRANSFORM_CHUNK_ROWS, :dimensions]
        homogeneous = np.column_stack([part, np.ones(len(part), dtype=np.float64)])
        part[:] = (matrix @ homogeneous.T).T[:, :dimensions]
    return out


class _Binding:
    def __init__(self, service: CoordinateService, dataset: Any, key: DatasetKey) -> None:
        self.service = weakref.ref(service)
        self.dataset = weakref.ref(dataset)
        self.key = key

    def __deepcopy__(self, memo: dict) -> None:
        # A copied dataset is not the original registration (or an open dataset).
        return None


class CoordinateService:
    """One application session's LRU; unrelated builds never hold its lock.

    Concurrent requests for the same snapshot share a Future. Removing a
    registration or clearing the cache prevents late work from repopulating it.
    No cache key retains a dataset, its source arrays, or a positional index.
    """

    def __init__(self, budget_bytes: int) -> None:
        self._lock = RLock()
        self._budget = max(0, int(budget_bytes))
        self._bytes = 0
        self._epoch = 0
        self._bindings: dict[DatasetKey, _Binding] = {}
        self._cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._pending: dict[tuple, Future] = {}

    @property
    def retained_bytes(self) -> int:
        with self._lock:
            return self._bytes

    @property
    def budget_bytes(self) -> int:
        with self._lock:
            return self._budget

    def set_budget_bytes(self, value: int) -> None:
        with self._lock:
            self._budget = max(0, int(value))
            self._trim()

    def _trim(self) -> None:
        while self._cache and self._bytes > self._budget:
            _, array = self._cache.popitem(last=False)
            self._bytes -= array.nbytes

    def attach(self, dataset: Any, key: DatasetKey) -> None:
        with self._lock:
            old = self._bindings.get(key)
            if old is not None and old.dataset() is dataset:
                return
            self.detach(key)
            if not callable(getattr(getattr(dataset, "attr", None), "coordinate_snapshot", None)):
                # Dataset-like adapters without immutable columns use the
                # uncached compatibility path, not a strong-reference binding.
                return
            binding = _Binding(self, dataset, key)
            self_ref = weakref.ref(self)

            def collected(ref: weakref.ReferenceType) -> None:
                service = self_ref()
                if service is not None:
                    with service._lock:
                        if service._bindings.get(key) is binding:
                            service.detach(key)

            binding.dataset = weakref.ref(dataset, collected)
            self._bindings[key] = binding
            dataset._coordinate_binding = binding

    def detach(self, key: DatasetKey) -> None:
        with self._lock:
            binding = self._bindings.pop(key, None)
            for cache_key in list(self._cache):
                if cache_key[0] is binding:
                    self._bytes -= self._cache.pop(cache_key).nbytes

    def clear(self) -> None:
        with self._lock:
            self._epoch += 1
            self._cache.clear()
            self._bytes = 0

    def _get(self, key: tuple, build: Callable[[], np.ndarray], *, epoch: int) -> np.ndarray:
        with self._lock:
            array = self._cache.get(key)
            if array is not None:
                self._cache.move_to_end(key)
                return array
            pending_key = (epoch, key)
            pending = self._pending.get(pending_key)
            owner = pending is None
            if pending is None:
                pending = Future()
                self._pending[pending_key] = pending
        if not owner:
            return pending.result()
        try:
            array = build()
            array.flags.writeable = False
            with self._lock:
                binding = key[0]
                if (epoch == self._epoch
                        and self._bindings.get(binding.key) is binding
                        and 0 < array.nbytes <= self._budget):
                    self._cache[key] = array
                    self._bytes += array.nbytes
                    self._trim()
            pending.set_result(array)
            return array
        except BaseException as exc:
            pending.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._pending.pop(pending_key, None)


def coordinate_view(dataset: Any, *, transform: Any = _CALIBRATED) -> np.ndarray:
    """Read-only calibrated nm, optionally transformed by an explicit record.

    Unregistered datasets work without retaining a cache. The compatibility
    branch also accepts lightweight dataset-like objects used by embedders.
    """
    store = getattr(dataset, "attr", None)
    snapshot = getattr(store, "coordinate_snapshot", None)
    if snapshot is None:
        array = np.array(dataset.loc_nm, dtype=np.float64, copy=True)
        if array.ndim == 2 and array.shape[1] == 2:
            array = np.column_stack([array, np.zeros(len(array))])
        if transform is not _CALIBRATED:
            array = apply_display_transform_nm(array, transform)
        array.flags.writeable = False
        return array
    token, columns = snapshot()
    z_scale = float(dataset.cali.z_scaling_factor)
    binding = getattr(dataset, "_coordinate_binding", None)
    service = binding.service() if binding is not None and binding.dataset() is dataset else None
    key = (binding, token, struct.pack("!d", z_scale))
    epoch = 0
    if service is not None:
        with service._lock:
            epoch = service._epoch

    def build() -> np.ndarray:
        return calibrated_coordinates(columns, z_scale)

    def calibrated() -> np.ndarray:
        return build() if service is None else service._get(key, build, epoch=epoch)

    matrix = _matrix(transform) if transform is not _CALIBRATED else None
    if matrix is None:
        array = calibrated()
    else:
        record = {f"matrix_{len(matrix)}x{len(matrix)}": matrix}

        def transformed() -> np.ndarray:
            return apply_display_transform_nm(calibrated(), record)

        array = (transformed() if service is None else service._get(
            (*key, matrix.shape, matrix.tobytes()), transformed, epoch=epoch))
    array.flags.writeable = False
    return array


def display_coordinate_view(dataset: Any) -> np.ndarray:
    """Read-only coordinates in the dataset's current display frame."""
    transform = dataset.state.get("overlay_transform") or dataset.state.get("render_transform_2d")
    return coordinate_view(dataset, transform=transform)
