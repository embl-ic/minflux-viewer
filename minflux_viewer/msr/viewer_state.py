"""MINFLUX Viewer processing state carried inside our own ``.msr``.

Verified against Imspector 16.3 (m2410): a ``viewer`` group added to a channel's
embedded zarr store does **not** disturb it -- the file still opens, with its
images and all three MINFLUX datasets. That is the R2 experiment, and this
module is its productised form, so one ``.msr`` can be both an Abberior
measurement and a complete MINFLUX Viewer document.

**Why JSON in ``viewer/.zattrs`` and not the ``.zarr`` store's payload format.**
The self-contained ``.zarr`` writes ``viewer/{dataset,metadata,state,rois}`` as
array-encoded payload groups through the real ``zarr`` package. A ``.msr``
channel store is written by :mod:`minflux_viewer.msr.zarr2`, our own minimal v2
implementation, and mixing two zarr implementations in one store buys nothing
here: the state is a few kB of plain values. So the *content* mirrors the Zarr
viewer group while the *encoding* is one JSON blob in a group attribute.

Derived attributes are deliberately **not** carried. ``.msr`` is a raw-canonical
format -- ``den``, localization precision and the trace statistics are recomputed
on load from the data, exactly as they are for a freshly imported measurement --
and embedding them would inflate every channel for values that are reproduced
anyway. Use ``.zarr`` when frozen derived arrays matter.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

__all__ = [
    "VIEWER_GROUP",
    "STATE_ATTR",
    "STATE_FORMAT",
    "STATE_SCHEMA",
    "build_project_manifest",
    "dataset_viewer_state",
    "write_viewer_state",
    "read_viewer_state",
    "apply_viewer_state",
]

#: Group added beside ``mfx`` in a channel's embedded zarr store.
VIEWER_GROUP = "viewer"
#: Attribute on that group holding the JSON state.
STATE_ATTR = "minflux_viewer"
#: Self-identifying marker, so a reader never has to guess from shape.
STATE_FORMAT = "org.minflux-viewer.msr-state"
STATE_SCHEMA = 1

#: Metadata that is either the vendor's own (restored from the file itself) or a
#: large array (carried as a real component, never as JSON).
_SKIP_METADATA = {
    "native_zarr_root_attrs",
    "native_zarr_mfx_attrs",
    "native_zarr_mbm_attrs",
    "native_zarr_mbm_points_attrs",
    "native_zarr_search_attrs",
    "native_zarr_search_points_attrs",
    "mbm_points",
    "search_points",
    "minflux_viewer_images",
}


def _jsonable(obj: Any) -> Any:
    """Plain JSON values; numpy scalars and arrays become lists."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def build_project_manifest(datasets, *, name: str = "") -> dict:
    """The group every channel carries, so any one of them can rebuild it.

    A channel opened on its own still knows it belonged to an overlay, in which
    order, and what the other members were called -- which is what lets the
    reader regroup them however many were selected for import.
    """
    members = []
    for index, ds in enumerate(datasets):
        state = getattr(ds, "state", {}) or {}
        members.append({
            "name": str(getattr(ds, "name", "") or ""),
            "did": str((getattr(ds, "metadata", {}) or {}).get("msr_dataset_did") or ""),
            "overlay_index": state.get("overlay_index", index),
            "overlay_order": state.get("overlay_order", index),
            "overlay_lut": state.get("overlay_lut"),
        })
    anchor = datasets[0] if datasets else None
    group = ""
    if anchor is not None:
        st = getattr(anchor, "state", {}) or {}
        group = str(st.get("overlay_id") or st.get("render_group_id") or "")
    return {"name": str(name or ""), "overlay_id": group, "members": members}


def dataset_viewer_state(ds, *, manifest: dict | None = None,
                         roi_records=None, app_version: str = "") -> dict:
    """The full state payload for one channel."""
    metadata = {
        str(key): value
        for key, value in (getattr(ds, "metadata", {}) or {}).items()
        if key not in _SKIP_METADATA
    }
    cali = getattr(ds, "cali", None)
    calibration = {}
    if cali is not None:
        calibration["z_scaling_factor"] = float(getattr(cali, "z_scaling_factor", 1.0) or 1.0)
        if getattr(cali, "pixel_size", None) is not None:
            calibration["pixel_size_nm"] = float(cali.pixel_size)
        precision = getattr(cali, "loc_precision", None)
        if precision is not None:
            calibration["loc_precision_nm"] = _jsonable(precision)
    return _jsonable({
        "format": STATE_FORMAT,
        "schema": STATE_SCHEMA,
        "app_version": str(app_version or ""),
        "project": manifest or {},
        "dataset": {
            "name": str(getattr(ds, "name", "") or ""),
            "did": str(metadata.get("msr_dataset_did") or ""),
            "calibration": calibration,
        },
        "metadata": metadata,
        "state": dict(getattr(ds, "state", {}) or {}),
        "rois": list(roi_records or []),
    })


def write_viewer_state(store: dict, payload: dict) -> dict:
    """Add ``viewer/`` carrying *payload* to a channel's zarr store, in place."""
    from . import zarr2

    root = zarr2.open(store, mode="a")
    group = root.require_group(VIEWER_GROUP)
    group.attrs[STATE_ATTR] = payload
    return store


def read_viewer_state(zroot) -> dict | None:
    """The payload embedded in a channel store, or None when there is none.

    Returns None rather than raising for anything unrecognised: a ``.msr``
    without our state is the normal case, not an error.
    """
    from .io import read_zarr_attrs

    try:
        attrs = read_zarr_attrs(zroot, VIEWER_GROUP)
    except Exception:                                   # noqa: BLE001 - absent
        return None
    if not isinstance(attrs, dict):
        return None
    payload = attrs.get(STATE_ATTR)
    if isinstance(payload, str):                        # tolerated: JSON text
        try:
            payload = json.loads(payload)
        except Exception:                               # noqa: BLE001
            return None
    if not isinstance(payload, dict):
        return None
    if payload.get("format") != STATE_FORMAT:
        return None
    return payload


def apply_viewer_state(ds, payload: dict | None) -> list[str]:
    """Restore *payload* onto *ds*; returns what was applied, for the Log.

    Applied **after** the importer's own defaults, so a saved overlay id, LUT or
    transform wins over the one the import would otherwise invent. Metadata that
    describes *this* file (the source path, the DID) is left alone: it was just
    stamped from the file actually being read, which may have been moved or
    renamed since the state was written.
    """
    if not isinstance(payload, dict) or payload.get("format") != STATE_FORMAT:
        return []
    applied: list[str] = []

    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and metadata:
        keep = {"msr_source_path", "msr_dataset_key", "msr_dataset_did",
                "msr_dataset_name"}
        restored = {k: v for k, v in metadata.items() if k not in keep}
        if restored:
            ds.metadata.update(restored)
            applied.append(f"{len(restored)} metadata field(s)")

    state = payload.get("state")
    if isinstance(state, dict) and state:
        ds.state.update(state)
        applied.append(f"{len(state)} view/processing setting(s)")
        filters = state.get("filter_specs") or []
        if filters:
            applied.append(f"{len(filters)} filter row(s)")

    calibration = ((payload.get("dataset") or {}).get("calibration") or {})
    z = calibration.get("z_scaling_factor")
    if z is not None:
        try:
            ds.set_z_scaling_factor(float(z), source="restored (.msr viewer state)")
            applied.append(f"Z scaling factor {float(z):g}")
        except Exception:                               # noqa: BLE001 - defensive
            ds.cali.z_scaling_factor = float(z)

    rois = payload.get("rois")
    if isinstance(rois, list) and rois:
        ds.metadata["minflux_viewer_roi_records"] = list(rois)
        applied.append(f"{len(rois)} ROI(s)")
    return applied
