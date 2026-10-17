"""Focused post-load input capture and inexpensive delivery-time validation."""

from __future__ import annotations

import copy
import weakref
from dataclasses import dataclass
from typing import Any

import numpy as np

from .dataset import AttrStore, MinfluxDataset


def readonly_array(values) -> np.ndarray:
    """Share immutable storage; isolate writable arrays and writable bases."""
    array = np.asarray(values)
    base: Any = array
    immutable = not array.dtype.hasobject
    while isinstance(base, np.ndarray):
        immutable = immutable and not base.flags.writeable
        base = base.base
    immutable = immutable and (base is None or isinstance(base, bytes))
    result = array.view() if immutable else copy.deepcopy(array)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class DatasetInputVersion:
    """Runtime-only delivery guard, without retaining the source dataset."""

    key: Any
    dataset: weakref.ReferenceType
    revision: int
    signature: tuple

    @classmethod
    def capture(cls, state, ds: MinfluxDataset) -> DatasetInputVersion:
        key = state.key_of(ds)
        if key is None:
            raise ValueError("Cannot submit work for an unregistered dataset")
        return cls(key, weakref.ref(ds), state.data_revision(key), input_signature(ds))

    def is_current(self, state) -> bool:
        ds = self.dataset()
        return (ds is not None and state.dataset_for(self.key) is ds
                and state.data_revision(self.key) == self.revision
                and input_signature(ds) == self.signature)


def _transform_signature(value) -> tuple | str:
    if isinstance(value, dict):
        return tuple((key, np.asarray(value[key], dtype=float).tobytes())
                     for key in ("matrix_4x4", "matrix_3x3") if value.get(key) is not None)
    return repr(value)


def input_signature(ds: MinfluxDataset) -> tuple:
    """Detect replacements even before the caller emits the change notification.

    In-place edits of non-coordinate arrays must still be notified through
    AppState. Do not hash entire acquisitions on Qt's thread for every job.
    The idempotent lazy raw loc_id cache is not a scientific-input revision.
    """
    stores = (ds.attr, ds.mfx_raw, ds.derived, ds.components.derived_last)
    identities = tuple(tuple((name, id(store.get(name))) for name in store.keys()
                             if name not in {"ftr", "loc_id"}) for store in stores)
    return (identities, ds.attr.coordinate_snapshot()[0],
            float(ds.cali.z_scaling_factor), repr(ds.metadata.get("z_scaling_factor_provenance")),
            ds.prop.num_loc, ds.prop.num_dim, bool(ds.metadata.get("particle_average")),
            repr(ds.state.get("filter_specs")), id(ds.state.get("filter_mask")),
            _transform_signature(ds.state.get("overlay_transform")),
            _transform_signature(ds.state.get("render_transform_2d")))


def computation_snapshot(ds: MinfluxDataset) -> MinfluxDataset:
    """Capture containers/recipes, retaining published array references.

    Column replacement and coordinate replacement cannot alter the snapshot.
    Workers see read-only views and private lazy caches. Mutable dependency
    arrays are copied only when needed by a saved filter (plus row selectors),
    not every raw measurement column or an entire acquisition deep-copy.
    """
    snapshot = copy.copy(ds)
    snapshot.components = copy.copy(ds.components)
    snapshot.prop = copy.deepcopy(ds.prop)
    snapshot.cali = copy.deepcopy(ds.cali)
    snapshot.file = copy.copy(ds.file)
    snapshot.components.metadata = copy.deepcopy(ds.metadata)
    snapshot.components.state = copy.deepcopy({
        key: value for key, value in ds.state.items() if key != "filter_mask"})
    snapshot.components.state["filter_mask"] = np.array(ds.filter_mask, copy=True)
    dependencies = {"tid", "itr", "vld"}
    for spec in snapshot.state.get("filter_specs") or []:
        dependencies.add(str(spec.get("attribute", "")))

    def capture(store: AttrStore) -> AttrStore:
        values: dict[str, Any] = {}
        for name, value in store.items():
            if name == "ftr":
                continue
            if isinstance(value, np.ndarray):
                value = np.array(value, copy=True) if value.flags.writeable and name in dependencies else value.view()
                value.flags.writeable = False
            values[name] = value
        captured = AttrStore(values)
        # Coordinates are already sealed by the owner; sharing them is safe.
        object.__setattr__(captured, "_sealed", True)
        return captured

    snapshot.attr = capture(ds.attr)
    snapshot.attr.bind_virtual("ftr", snapshot, "filter_mask")
    snapshot.components.mfx = copy.copy(ds.mfx)
    snapshot.components.mfx.attrs = snapshot.attr
    snapshot.components.mfx_raw = capture(ds.mfx_raw)
    snapshot.components.derived = capture(ds.derived)
    snapshot.components.derived_last = capture(ds.components.derived_last)
    return snapshot
