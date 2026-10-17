"""Focused post-load input capture and inexpensive delivery-time validation."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .dataset import AttrStore, MinfluxDataset


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
            repr(ds.state.get("filter_specs")), id(ds.state.get("filter_mask")))


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
