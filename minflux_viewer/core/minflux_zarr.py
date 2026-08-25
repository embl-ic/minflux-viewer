"""Self-contained MINFLUX Viewer dataset stores using physical Zarr v2.

The store is an application schema, not an unmarked collection of flat columns.
Raw acquisition facts, native MFX/MBM/search metadata, derived arrays and live
viewer state occupy separate groups so a saved dataset can be reopened without a
JSON sidecar or the original ``.msr`` file.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_ID = "org.minflux-viewer.dataset"
SCHEMA_VERSION = "1.0.0"
PROJECT_FORMAT_ID = "org.minflux-viewer.project"
PROJECT_SCHEMA_VERSION = "1.0.0"
FORMAT_ATTR = "_minflux_viewer_format"
SCHEMA_ATTR = "_minflux_viewer_schema_version"
CREATED_ATTR = "_minflux_viewer_created"
APP_VERSION_ATTR = "_minflux_viewer_app_version"
COMPONENT_ATTR = "_minflux_viewer_component"
LAYOUT_ATTR = "_minflux_viewer_layout"
ROLES_ATTR = "_minflux_viewer_roles"
FIELD_META_ATTR = "_minflux_viewer_field_metadata"
PAYLOAD_ATTR = "_minflux_viewer_payload"
RAW_FINGERPRINT_ATTR = "_minflux_viewer_raw_sha256"

_RESERVED_ATTRS = {
    FORMAT_ATTR,
    SCHEMA_ATTR,
    CREATED_ATTR,
    APP_VERSION_ATTR,
    COMPONENT_ATTR,
    LAYOUT_ATTR,
    ROLES_ATTR,
    FIELD_META_ATTR,
    PAYLOAD_ATTR,
    RAW_FINGERPRINT_ATTR,
}

_NATIVE_METADATA_KEYS = {
    "native_zarr_root_attrs",
    "native_zarr_mfx_attrs",
    "native_zarr_mbm_attrs",
    "native_zarr_mbm_points_attrs",
    "native_zarr_search_attrs",
    "native_zarr_search_points_attrs",
}
_COMPONENT_ARRAY_METADATA_KEYS = {"mbm_points", "search_points"}
_ARRAY_REF = "__minflux_viewer_array__"
_BYTES_REF = "__minflux_viewer_bytes__"
_FLOAT_REF = "__minflux_viewer_float__"
_DATETIME_REF = "__minflux_viewer_datetime__"


class MinfluxZarrError(ValueError):
    """Raised when a store is not a supported MINFLUX Viewer Zarr dataset."""


@dataclasses.dataclass
class LoadedMinfluxZarrProject:
    """One reopened Zarr acquisition bundle (one or more datasets)."""

    datasets: list
    manifest: dict
    roi_records: list[dict]
    images: list[dict]
    path: Path


def _zarr_modules():
    try:
        import zarr
        from numcodecs import Blosc
    except ImportError:
        raise ImportError(
            "MINFLUX Viewer Zarr save/load requires zarr 2.x and numcodecs."
        ) from None
    major = int(str(getattr(zarr, "__version__", "0")).split(".", 1)[0])
    if major != 2:
        raise ImportError(
            "The MINFLUX Viewer dataset writer currently requires zarr 2.x; "
            f"found zarr {getattr(zarr, '__version__', 'unknown')}."
        )
    return zarr, Blosc


def _plain_json(value: Any, *, path: str = "value") -> Any:
    """Convert a small metadata value into strict JSON-compatible objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if np.isfinite(value):
            return value
        return {_FLOAT_REF: "nan" if np.isnan(value) else ("inf" if value > 0 else "-inf")}
    if isinstance(value, np.generic):
        return _plain_json(value.item(), path=path)
    if isinstance(value, np.ndarray):
        return _plain_json(value.tolist(), path=path)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return {_DATETIME_REF: value.isoformat()}
    if isinstance(value, bytes):
        return {_BYTES_REF: base64.b64encode(value).decode("ascii")}
    if dataclasses.is_dataclass(value):
        return _plain_json(dataclasses.asdict(value), path=path)
    if isinstance(value, dict):
        return {
            str(key): _plain_json(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_plain_json(item, path=f"{path}[]") for item in value]
    raise TypeError(f"{path} contains unsupported metadata type {type(value).__name__}.")


def _restore_plain_json(value: Any) -> Any:
    if isinstance(value, list):
        return [_restore_plain_json(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {_FLOAT_REF}:
        return {"nan": np.nan, "inf": np.inf, "-inf": -np.inf}[value[_FLOAT_REF]]
    if set(value) == {_BYTES_REF}:
        return base64.b64decode(value[_BYTES_REF])
    if set(value) == {_DATETIME_REF}:
        return value[_DATETIME_REF]
    return {str(key): _restore_plain_json(item) for key, item in value.items()}


def _chunks_for(array: np.ndarray) -> tuple[int, ...]:
    if array.ndim == 0:
        return (1,)
    tail = int(np.prod(array.shape[1:], dtype=np.int64)) if array.ndim > 1 else 1
    row_bytes = max(1, int(array.dtype.itemsize) * max(1, tail))
    first = max(1, min(int(array.shape[0]) if array.shape[0] else 1, (1 << 20) // row_bytes))
    return (first, *array.shape[1:])


def _write_array(group, name: str, value: Any) -> None:
    _zarr, Blosc = _zarr_modules()
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError(f"Zarr array '{name}' has object dtype, which is not portable.")
    if array.ndim == 0:
        array = array.reshape(1)
    compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
    group.create_dataset(
        str(name), data=array, chunks=_chunks_for(array), compressor=compressor,
        overwrite=True,
    )


def _copy_native_attrs(node, values: dict | None) -> None:
    for key, value in dict(values or {}).items():
        key = str(key)
        if key in _RESERVED_ATTRS:
            continue
        node.attrs[key] = _plain_json(value, path=f"native attribute {key!r}")


def _native_attrs(node) -> dict:
    return {
        str(key): _restore_plain_json(value)
        for key, value in node.attrs.asdict().items()
        if str(key) not in _RESERVED_ATTRS
    }


def _encode_payload(value: Any, arrays_group, counter: list[int], *, path: str) -> Any:
    if isinstance(value, np.ndarray):
        key = f"a{counter[0]:06d}"
        counter[0] += 1
        _write_array(arrays_group, key, value)
        return {_ARRAY_REF: key}
    if isinstance(value, np.generic):
        return _encode_payload(value.item(), arrays_group, counter, path=path)
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {
            str(key): _encode_payload(item, arrays_group, counter, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [
            _encode_payload(item, arrays_group, counter, path=f"{path}[]")
            for item in value
        ]
    return _plain_json(value, path=path)


def _decode_payload(value: Any, arrays_group) -> Any:
    if isinstance(value, list):
        return [_decode_payload(item, arrays_group) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {_ARRAY_REF}:
        key = str(value[_ARRAY_REF])
        if key not in arrays_group:
            raise MinfluxZarrError(f"Processing payload references missing array '{key}'.")
        return np.asarray(arrays_group[key][:])
    restored = _restore_plain_json(value)
    if restored is not value and not isinstance(restored, dict):
        return restored
    return {str(key): _decode_payload(item, arrays_group) for key, item in value.items()}


def _write_payload(parent, name: str, payload: Any) -> None:
    group = parent.require_group(name)
    arrays = group.require_group("arrays")
    group.attrs[PAYLOAD_ATTR] = _encode_payload(payload, arrays, [0], path=name)


def _read_payload(parent, name: str, default: Any) -> Any:
    if name not in parent:
        return default
    group = parent[name]
    payload = group.attrs.get(PAYLOAD_ATTR, default)
    arrays = group["arrays"] if "arrays" in group else {}
    return _decode_payload(payload, arrays)


def _write_attr_store(parent, name: str, store) -> None:
    group = parent.require_group(name)
    mapping: dict[str, str] = {}
    for index, (key, value) in enumerate(store.items() if store is not None else []):
        node = f"a{index:06d}"
        _write_array(group, node, value)
        mapping[node] = str(key)
    group.attrs["names"] = mapping


def _read_attr_store(parent, name: str) -> dict[str, np.ndarray]:
    if name not in parent:
        return {}
    group = parent[name]
    mapping = dict(group.attrs.get("names", {}))
    return {
        str(original): np.asarray(group[node][:])
        for node, original in mapping.items()
        if node in group
    }


def capture_native_zarr_metadata(ds, store) -> None:
    """Attach native MFXDTA attributes/search data to a loaded dataset.

    The MSR reader calls this while its in-memory source store is available.
    The copied values are small; the multi-hundred-megabyte source store itself
    is deliberately not retained on the dataset.
    """
    if store is None:
        return
    from ..msr import zarr2
    from ..msr.io import read_zarr_attrs

    archive = zarr2.open(store, mode="r")
    ds.metadata["native_zarr_root_attrs"] = read_zarr_attrs(store, "")
    ds.metadata["native_zarr_mfx_attrs"] = read_zarr_attrs(store, "mfx")
    mbm_path = "grd/mbm" if "grd/mbm" in archive else ("mbm" if "mbm" in archive else None)
    if mbm_path:
        ds.metadata["native_zarr_mbm_attrs"] = read_zarr_attrs(store, mbm_path)
    if "grd/mbm/points" in archive:
        ds.metadata["native_zarr_mbm_points_attrs"] = read_zarr_attrs(
            store, "grd/mbm/points"
        )
    if "grd/search_0" in archive:
        ds.metadata["native_zarr_search_attrs"] = read_zarr_attrs(store, "grd/search_0")
    if "grd/search_0/points" in archive:
        ds.metadata["search_points"] = np.asarray(archive["grd/search_0/points"][:])
        ds.metadata["native_zarr_search_points_attrs"] = read_zarr_attrs(
            store, "grd/search_0/points"
        )


def _writer_info(ds) -> dict:
    cali = getattr(ds, "cali", None)
    channel = getattr(ds, "channel", None)
    file_info = getattr(ds, "file", None)
    return {
        "dataset_name": str(getattr(ds, "name", "dataset")),
        "file": {
            "name": str(getattr(file_info, "name", "") or ""),
            "folder": str(getattr(file_info, "folder", "") or ""),
            "datetime": str(getattr(file_info, "datetime", "") or ""),
            "recent_path": getattr(file_info, "recent_path", None),
        },
        "calibration": {
            "z_scaling_factor": float(getattr(cali, "z_scaling_factor", 1.0)),
            "pixel_size_nm": float(getattr(cali, "pixel_size", 4.0)),
            "loc_precision_nm": np.asarray(
                getattr(cali, "loc_precision", np.zeros(3)), dtype=float
            ),
        },
        "channel": {
            "num_channels": int(getattr(channel, "num_channels", 1)),
            "cut1": float(getattr(channel, "cut1", 0.5)),
            "cut2": float(getattr(channel, "cut2", 1.0)),
            "do_trace": bool(getattr(channel, "do_trace", False)),
            "keep_ch3": bool(getattr(channel, "keep_ch3", False)),
        },
    }


def _raw_native_attrs(values: dict | None) -> dict:
    return {
        str(key): value
        for key, value in dict(values or {}).items()
        if str(key) not in _RESERVED_ATTRS
    }


def _digest_json(hasher, label: str, value: Any) -> None:
    hasher.update(label.encode("utf-8"))
    encoded = json.dumps(
        _plain_json(value, path=label),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, "little"))
    hasher.update(encoded)


def _digest_array(hasher, label: str, value: Any) -> None:
    array = np.asarray(value)
    _digest_json(
        hasher,
        f"{label}.layout",
        {
            "shape": list(array.shape),
            "dtype": array.dtype.descr if array.dtype.fields else array.dtype.str,
        },
    )
    contiguous = np.ascontiguousarray(array)
    hasher.update(memoryview(contiguous).cast("B"))


def _mbm_points(ds):
    """The MBM points array of *ds*, or ``None``.

    ``dataset.mbm`` is normally an :class:`AttributeComponent` holding a
    ``points`` array, but a caller may reasonably have assigned the structured
    points array itself. Both carry the same fact, so both are accepted; a
    component-shaped object that is neither is an error worth naming rather than
    letting an ``AttributeError`` surface from inside a hash.
    """
    component = getattr(ds, "mbm", None)
    if component is None:
        return None
    attrs = getattr(component, "attrs", None)
    if attrs is not None:
        return attrs.get("points")
    if isinstance(component, np.ndarray):
        return component
    raise MinfluxZarrError(
        "dataset.mbm must be an AttributeComponent holding 'points', or the "
        f"points array itself; got {type(component).__name__}.")


def _raw_digest(parts) -> str:
    """Hash an ordered sequence of ``(kind, label, value)`` digest inputs.

    Both the in-memory and the on-disk fingerprint go through this one routine,
    so the two can never drift apart in ordering or encoding.
    """
    hasher = hashlib.sha256()
    hasher.update(b"minflux-viewer-canonical-raw-v1\0")
    for kind, label, value in parts:
        if kind == "json":
            _digest_json(hasher, label, value)
        else:
            _digest_array(hasher, label, value)
    return hasher.hexdigest()


def _raw_parts(
    columns,
    *,
    root_attrs,
    mfx_attrs,
    mfx_roles,
    mfx_meta,
    mbm_points=None,
    mbm_attrs=None,
    mbm_point_attrs=None,
    mbm_roles=None,
    mbm_meta=None,
    search_points=None,
    search_attrs=None,
    search_point_attrs=None,
):
    """The canonical raw digest inputs, in the one order that defines them."""
    parts = [("json", "mfx.keys", sorted(columns))]
    parts += [("array", f"mfx.{key}", columns[key]) for key in sorted(columns)]
    parts.append(("json", "root.attrs", _raw_native_attrs(root_attrs)))
    parts.append(("json", "mfx.attrs", _raw_native_attrs(mfx_attrs)))
    parts.append(("json", "mfx.roles", mfx_roles or {}))
    parts.append(("json", "mfx.field_metadata", mfx_meta or {}))

    parts.append(("json", "mbm.present", mbm_points is not None))
    if mbm_points is not None:
        parts.append(("array", "mbm.points", mbm_points))
        parts.append(("json", "mbm.attrs", mbm_attrs or {}))
        parts.append(("json", "mbm.points.attrs", mbm_point_attrs or {}))
        parts.append(("json", "mbm.roles", mbm_roles or {}))
        parts.append(("json", "mbm.field_metadata", mbm_meta or {}))

    parts.append(("json", "search.present", search_points is not None))
    if search_points is not None:
        parts.append(("array", "search.points", search_points))
        parts.append(("json", "search.attrs", _raw_native_attrs(search_attrs)))
        parts.append(
            ("json", "search.points.attrs", _raw_native_attrs(search_point_attrs))
        )
    return parts


def _dataset_raw_fingerprint(ds, *, columns=None) -> str:
    """Digest exactly the canonical/native components the writer owns as raw."""
    from .save import dataset_to_mfx_array, flatten_mfx_array

    if columns is None:
        columns = flatten_mfx_array(dataset_to_mfx_array(ds))

    points = _mbm_points(ds)
    if points is None:
        points = ds.metadata.get("mbm_points")
    mbm_attrs = mbm_point_attrs = None
    if points is not None:
        mbm_attrs = _raw_native_attrs(ds.metadata.get("native_zarr_mbm_attrs"))
        if ds.metadata.get("mbm_used") is not None:
            mbm_attrs["used"] = ds.metadata["mbm_used"]
        mbm_point_attrs = _raw_native_attrs(
            ds.metadata.get("native_zarr_mbm_points_attrs"))
        if ds.metadata.get("mbm_points_by_gri") is not None:
            mbm_point_attrs["points_by_gri"] = ds.metadata["mbm_points_by_gri"]

    return _raw_digest(_raw_parts(
        columns,
        root_attrs=ds.metadata.get("native_zarr_root_attrs"),
        mfx_attrs=ds.metadata.get("native_zarr_mfx_attrs"),
        mfx_roles=getattr(ds.mfx, "roles", {}),
        mfx_meta=getattr(ds.mfx, "meta", {}),
        mbm_points=points,
        mbm_attrs=mbm_attrs,
        mbm_point_attrs=mbm_point_attrs,
        mbm_roles=getattr(getattr(ds, "mbm", None), "roles", {}),
        mbm_meta=getattr(getattr(ds, "mbm", None), "meta", {}),
        search_points=ds.metadata.get("search_points"),
        search_attrs=ds.metadata.get("native_zarr_search_attrs"),
        search_point_attrs=ds.metadata.get("native_zarr_search_points_attrs"),
    ))


def _store_raw_fingerprint(root) -> str:
    """Digest a store's raw components by reading it, not by loading a dataset.

    Same guarantee as :func:`_dataset_raw_fingerprint` — it hashes the stored
    arrays and native attributes themselves, so an externally rewritten raw
    chunk is still detected — but it skips building a full ``MinfluxDataset``
    (property computation, derived attributes and all), which dominated the
    cost of a processing-only update on a large acquisition.
    """
    mfx_group = root["mfx"]
    columns = {str(key): np.asarray(mfx_group[key][:]) for key in mfx_group.array_keys()}

    points = mbm_attrs = mbm_point_attrs = None
    mbm_roles = mbm_meta = {}
    if "grd/mbm/points" in root:
        mbm_group = root["grd/mbm"]
        points_node = root["grd/mbm/points"]
        points = np.asarray(points_node[:])
        mbm_attrs = _native_attrs(mbm_group)
        used = mbm_group.attrs.get("used")
        if used is not None:
            mbm_attrs["used"] = _restore_plain_json(used)
        mbm_point_attrs = _native_attrs(points_node)
        by_gri = points_node.attrs.get("points_by_gri")
        if by_gri is not None:
            mbm_point_attrs["points_by_gri"] = _restore_plain_json(by_gri)
        mbm_roles = dict(mbm_group.attrs.get(ROLES_ATTR, {}))
        mbm_meta = dict(mbm_group.attrs.get(FIELD_META_ATTR, {}))

    search_points = search_attrs = search_point_attrs = None
    if "grd/search_0/points" in root:
        search_node = root["grd/search_0/points"]
        search_points = np.asarray(search_node[:])
        search_attrs = _native_attrs(root["grd/search_0"])
        search_point_attrs = _native_attrs(search_node)

    return _raw_digest(_raw_parts(
        columns,
        root_attrs=_native_attrs(root),
        mfx_attrs=_native_attrs(mfx_group),
        mfx_roles=dict(mfx_group.attrs.get(ROLES_ATTR, {})),
        mfx_meta=dict(mfx_group.attrs.get(FIELD_META_ATTR, {})),
        mbm_points=points,
        mbm_attrs=mbm_attrs,
        mbm_point_attrs=mbm_point_attrs,
        mbm_roles=mbm_roles,
        mbm_meta=mbm_meta,
        search_points=search_points,
        search_attrs=search_attrs,
        search_point_attrs=search_point_attrs,
    ))


def _write_dataset_viewer(root, ds, *, roi_records=None, image_records=None) -> None:
    """Write only the application-owned portion of one dataset store."""
    viewer = root.require_group("viewer")
    viewer.attrs[COMPONENT_ATTR] = "minflux_viewer_processing"
    _write_payload(viewer, "dataset", _writer_info(ds))
    metadata = {
        str(key): value
        for key, value in ds.metadata.items()
        if key not in _NATIVE_METADATA_KEYS | _COMPONENT_ARRAY_METADATA_KEYS
    }
    _write_payload(viewer, "metadata", metadata)
    _write_payload(viewer, "state", dict(ds.state))
    if roi_records:
        _write_payload(viewer, "rois", list(roi_records))
    if image_records:
        _write_payload(viewer, "images", list(image_records))
    _write_attr_store(viewer, "derived", ds.derived)
    _write_attr_store(viewer, "derived_last", ds.derived_last)


def _write_store(ds, target: Path, *, roi_records=None) -> None:
    zarr, _Blosc = _zarr_modules()
    from .save import dataset_to_mfx_array, flatten_mfx_array

    store = zarr.DirectoryStore(str(target))
    root = zarr.group(store=store, overwrite=True)
    _copy_native_attrs(root, ds.metadata.get("native_zarr_root_attrs"))
    root.attrs[FORMAT_ATTR] = FORMAT_ID
    root.attrs[SCHEMA_ATTR] = SCHEMA_VERSION
    root.attrs[CREATED_ATTR] = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        from .. import __version__
    except Exception:  # pragma: no cover - defensive during isolated module use
        __version__ = "unknown"
    root.attrs[APP_VERSION_ATTR] = str(__version__)

    mfx = root.require_group("mfx")
    _copy_native_attrs(mfx, ds.metadata.get("native_zarr_mfx_attrs"))
    mfx.attrs[COMPONENT_ATTR] = "mfx"
    mfx.attrs[LAYOUT_ATTR] = "columnar-m2410"
    mfx.attrs[ROLES_ATTR] = _plain_json(getattr(ds.mfx, "roles", {}), path="mfx roles")
    mfx.attrs[FIELD_META_ATTR] = _plain_json(getattr(ds.mfx, "meta", {}), path="mfx metadata")
    columns = flatten_mfx_array(dataset_to_mfx_array(ds))
    root.attrs[RAW_FINGERPRINT_ATTR] = _dataset_raw_fingerprint(ds, columns=columns)
    for key, value in columns.items():
        _write_array(mfx, str(key), value)

    points = _mbm_points(ds)
    if points is None:
        points = ds.metadata.get("mbm_points")
    search_points = ds.metadata.get("search_points")
    if points is not None or search_points is not None:
        grd = root.require_group("grd")
        if points is not None:
            mbm = grd.require_group("mbm")
            _copy_native_attrs(mbm, ds.metadata.get("native_zarr_mbm_attrs"))
            mbm.attrs[COMPONENT_ATTR] = "mbm"
            # Roles/field metadata exist only on a component; a bare points
            # array carries the same measurements without them.
            roles = getattr(getattr(ds, "mbm", None), "roles", None)
            if roles is not None:
                mbm.attrs[ROLES_ATTR] = _plain_json(roles, path="mbm roles")
            field_meta = getattr(getattr(ds, "mbm", None), "meta", None)
            if field_meta is not None:
                mbm.attrs[FIELD_META_ATTR] = _plain_json(
                    field_meta, path="mbm metadata")
            _write_array(mbm, "points", points)
            _copy_native_attrs(mbm["points"], ds.metadata.get("native_zarr_mbm_points_attrs"))
            if ds.metadata.get("mbm_points_by_gri") is not None:
                mbm["points"].attrs["points_by_gri"] = _plain_json(
                    ds.metadata["mbm_points_by_gri"], path="points_by_gri"
                )
            if ds.metadata.get("mbm_used") is not None:
                mbm.attrs["used"] = _plain_json(ds.metadata["mbm_used"], path="mbm used")
        if search_points is not None:
            search = grd.require_group("search_0")
            _copy_native_attrs(search, ds.metadata.get("native_zarr_search_attrs"))
            search.attrs[COMPONENT_ATTR] = "search_grid"
            _write_array(search, "points", search_points)
            _copy_native_attrs(
                search["points"], ds.metadata.get("native_zarr_search_points_attrs")
            )

    _write_dataset_viewer(root, ds, roi_records=roi_records)


def _validate_store(path: Path) -> None:
    zarr, _Blosc = _zarr_modules()
    root = zarr.open(str(path), mode="r")
    if root.attrs.get(FORMAT_ATTR) != FORMAT_ID:
        raise MinfluxZarrError("Written store is missing the MINFLUX Viewer format marker.")
    if root.attrs.get(SCHEMA_ATTR) != SCHEMA_VERSION:
        raise MinfluxZarrError("Written store has an unexpected schema version.")
    if "mfx" not in root:
        raise MinfluxZarrError("Written store has no mfx component.")
    required = {"loc_x", "loc_y", "itr", "vld"}
    missing = sorted(required - set(root["mfx"].array_keys()))
    if missing:
        raise MinfluxZarrError(
            "Written mfx component is incomplete; missing " + ", ".join(missing) + "."
        )


def _image_stem(name: str) -> str:
    """Filesystem-safe image stem while preserving the Imspector label."""
    forbidden = set('<>:"/\\|?*') | {chr(code) for code in range(32)}
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{index}"
        for prefix in ("COM", "LPT")
        for index in range(1, 10)
    }
    out: list[str] = []
    for char in str(name):
        if char in forbidden:
            if not out or out[-1] != "_":
                out.append("_")
        else:
            out.append(char)
    stem = "".join(out).strip().strip(".").strip() or "image"
    return f"{stem}_" if stem.upper() in reserved else stem


def _dataset_did(ds) -> str:
    native = dict(getattr(ds, "metadata", {}).get("native_zarr_mfx_attrs") or {})
    return str(
        getattr(ds, "metadata", {}).get("msr_dataset_did")
        or native.get("did")
        or ""
    )


def _export_embedded_images(
    root_path: Path,
    datasets: list,
    dataset_ids: list[str],
    dataset_paths: list[Path],
    image_specs,
) -> list[dict]:
    """Convert selected OBF images to OME-TIFF inside the Zarr directory."""
    specs = [dict(spec) for spec in (image_specs or [])]
    if not specs:
        return []
    from .obf_image_source import ObfImageSource
    from .tiff_export import export_image_series_to_tiff

    did_to_position = {
        did: position
        for position, ds in enumerate(datasets)
        if (did := _dataset_did(ds))
    }
    used: dict[Path, set[str]] = {}
    records: list[dict] = []
    for spec in specs:
        msr_path = Path(str(spec.get("msr_path") or ""))
        raw_index = int(spec.get("raw_index", -1))
        if not msr_path.is_file() or raw_index < 0:
            raise ValueError(
                f"Embedded image source is unavailable: {msr_path} (stack {raw_index})."
            )
        source_did = str(spec.get("source_did") or "")
        position = did_to_position.get(source_did)
        if position is None:
            parent = root_path / "images" / "unassigned"
            dataset_id = None
        else:
            parent = dataset_paths[position] / "images"
            dataset_id = dataset_ids[position]
        parent.mkdir(parents=True, exist_ok=True)
        stem = _image_stem(str(spec.get("name") or f"image_{raw_index}"))
        names = used.setdefault(parent, set())
        if stem in names:
            stem = f"{stem}_{raw_index}"
        names.add(stem)
        target = parent / f"{stem}.tif"
        source = ObfImageSource(msr_path, raw_stack_index=raw_index)
        try:
            export_image_series_to_tiff(source, target)
        finally:
            source.close()
        records.append({
            "name": str(spec.get("name") or stem),
            "source_did": source_did,
            "raw_stack_index": raw_index,
            "dataset_id": dataset_id,
            "path": target.relative_to(root_path).as_posix(),
            "format": "ome-tiff",
        })
    return records


def _write_image_manifest(store_path: Path, records: list[dict]) -> None:
    if not records:
        return
    zarr, _Blosc = _zarr_modules()
    root = zarr.open_group(str(store_path), mode="a")
    viewer = root.require_group("viewer")
    _write_payload(viewer, "images", records)


def _stored_image_records(path: Path) -> list[dict]:
    zarr, _Blosc = _zarr_modules()
    root = zarr.open_group(str(path), mode="r")
    if "viewer" not in root:
        return []
    records = _read_payload(root["viewer"], "images", [])
    return [dict(record) for record in records] if isinstance(records, list) else []


def _ensure_requested_images_are_present(
    existing_records: list[dict], image_specs
) -> None:
    requested = {
        (str(spec.get("source_did") or ""), int(spec.get("raw_index", -1)))
        for spec in (image_specs or [])
    }
    if not requested:
        return
    existing = {
        (str(record.get("source_did") or ""), int(record.get("raw_stack_index", -1)))
        for record in existing_records
    }
    missing = requested - existing
    if missing:
        raise MinfluxZarrError(
            "Processing-only update cannot add or replace embedded images. "
            "Choose 'Replace complete store' for this save."
        )


def _stored_raw_fingerprint(path: Path) -> str:
    zarr, _Blosc = _zarr_modules()
    root = zarr.open_group(str(path), mode="r")
    declared = str(root.attrs.get(RAW_FINGERPRINT_ATTR) or "")
    # Always hash the actual store: the recorded digest is an integrity check,
    # not an authority that could conceal an externally changed raw chunk.
    # Loading also lets pre-fingerprint stores use the same safe update path.
    actual = _store_raw_fingerprint(root)
    if declared and declared != actual:
        raise MinfluxZarrError(
            f"Stored raw data in '{path.name}' no longer matches its recorded "
            "fingerprint. Processing-only update was refused."
        )
    return actual


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _replace_viewer_directories(replacements: list[tuple[Path, Path]]) -> None:
    """Install staged viewer directories as one rollback-capable transaction."""
    journal: list[tuple[Path, Path | None]] = []
    try:
        for staged, target in replacements:
            backup = None
            if target.exists():
                backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
                os.replace(target, backup)
            journal.append((target, backup))
            os.replace(staged, target)
    except Exception:
        for target, backup in reversed(journal):
            _remove_path(target)
            if backup is not None and backup.exists():
                os.replace(backup, target)
        raise
    else:
        for _target, backup in journal:
            if backup is not None:
                _remove_path(backup)


def _preserve_unknown_viewer_content(
    existing: Path,
    staged: Path,
    *,
    known_children: set[str],
) -> None:
    """Carry forward viewer extensions this application version does not own."""
    if not existing.is_dir():
        return
    for child in existing.iterdir():
        if child.name.startswith(".") or child.name in known_children:
            continue
        destination = staged / child.name
        if child.is_dir():
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination)
    zarr, _Blosc = _zarr_modules()
    old_group = zarr.open_group(str(existing), mode="r")
    new_group = zarr.open_group(str(staged), mode="a")
    for key, value in old_group.attrs.asdict().items():
        if str(key) not in new_group.attrs:
            new_group.attrs[str(key)] = value


def update_minflux_zarr_viewer(
    ds,
    path: str | Path,
    *,
    roi_records=None,
    image_specs=None,
) -> Path:
    """Replace only ``viewer/`` after proving the canonical raw data matches."""
    target = Path(path).resolve()
    zarr, _Blosc = _zarr_modules()
    root = zarr.open_group(str(target), mode="r")
    if root.attrs.get(FORMAT_ATTR) != FORMAT_ID:
        raise MinfluxZarrError(
            "Processing-only update requires an existing single-dataset "
            "MINFLUX Viewer Zarr store."
        )
    if _stored_raw_fingerprint(target) != _dataset_raw_fingerprint(ds):
        raise MinfluxZarrError(
            "Canonical raw data differs from the existing Zarr store. "
            "Processing-only update was refused; choose 'Replace complete store' "
            "only if replacing its raw data is intentional."
        )
    image_records = _stored_image_records(target)
    _ensure_requested_images_are_present(image_records, image_specs)

    temp = Path(tempfile.mkdtemp(prefix=f".{target.name}.viewer-tmp-", dir=target.parent))
    try:
        staged_root = temp / "dataset"
        staged = zarr.group(store=zarr.DirectoryStore(str(staged_root)), overwrite=True)
        _write_dataset_viewer(
            staged,
            ds,
            roi_records=roi_records,
            image_records=image_records,
        )
        _preserve_unknown_viewer_content(
            target / "viewer",
            staged_root / "viewer",
            known_children={
                "dataset", "metadata", "state", "rois", "images",
                "derived", "derived_last",
            },
        )
        _replace_viewer_directories([(staged_root / "viewer", target / "viewer")])
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return target


ZIP_SUFFIX = ".zarr.zip"


def is_zipped_store(path) -> bool:
    """True for a sealed ``.zarr.zip`` package."""
    return str(path).lower().endswith(ZIP_SUFFIX)


def pack_minflux_zarr(store_path, zip_path=None) -> Path:
    """Seal a directory store into one ``.zarr.zip`` file.

    The zip is a **sealed archive**, deliberately not the working format. A
    processing-only update replaces members of ``viewer/``, and a zip cannot
    replace a member: writing one again appends a second copy of the same name.
    Readers then disagree — ``zipfile`` and Zarr take the last entry, while some
    archive tools take the first — and the file grows on every save. So the
    directory store stays editable and this produces a distribution copy.

    Members are stored uncompressed: the chunks are already Blosc-compressed, so
    deflating them again costs time and saves nothing. ZIP64 is enabled, which
    lifts the 4 GiB and 65,535-member ceilings.
    """
    import zipfile

    source = Path(store_path).resolve()
    if not source.is_dir():
        raise MinfluxZarrError(f"Not a Zarr directory store: '{source.name}'.")
    target = Path(zip_path) if zip_path else source.with_name(
        source.name[:-5] + ZIP_SUFFIX if source.name.lower().endswith(".zarr")
        else source.name + ZIP_SUFFIX)
    target.parent.mkdir(parents=True, exist_ok=True)

    temp = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_STORED,
                             allowZip64=True) as archive:
            for item in sorted(source.rglob("*")):
                if item.is_file():
                    archive.write(item, item.relative_to(source).as_posix())
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()
    return target


def unpack_minflux_zarr(zip_path, out_dir=None) -> Path:
    """Extract a sealed ``.zarr.zip`` back to an editable directory store."""
    import zipfile

    source = Path(zip_path).resolve()
    if not source.is_file():
        raise MinfluxZarrError(f"Not a zipped Zarr package: '{source.name}'.")
    name = source.name[:-len(ZIP_SUFFIX)] + ".zarr" if is_zipped_store(source)         else source.stem
    target = Path(out_dir) / name if out_dir else source.with_name(name)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing store: {target}")

    temp = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    try:
        with zipfile.ZipFile(source) as archive:
            seen: set[str] = set()
            for info in archive.infolist():
                if info.filename in seen:
                    raise MinfluxZarrError(
                        f"'{source.name}' contains duplicate member "
                        f"'{info.filename}'; it was written by appending rather "
                        "than repacking and its contents are ambiguous.")
                seen.add(info.filename)
            archive.extractall(temp)
        os.replace(temp, target)
    finally:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)
    return target


def write_minflux_zarr_package(
    datasets,
    path,
    *,
    overwrite: bool = True,
    roi_records=None,
    image_specs=None,
    name: str | None = None,
) -> Path:
    """Write one sealed ``.zarr.zip``: the same store, in a single file.

    The store is built in a temporary directory and packed, rather than written
    into a zip directly: the writer validates the finished tree before it is
    installed, and a zip cannot be revalidated in place. The temporary directory
    sits beside the target so the final rename stays on one filesystem.

    Accepts one dataset or a list; a list of more than one writes the project
    layout, matching :func:`write_minflux_zarr_project`.
    """
    members = list(datasets) if isinstance(datasets, (list, tuple)) else [datasets]
    if not members:
        raise MinfluxZarrError("Nothing to save: no dataset supplied.")

    target = Path(path)
    if not target.name.lower().endswith(ZIP_SUFFIX):
        target = target.with_name(target.name + ZIP_SUFFIX)
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing package: {target}")
    if target.exists() and not target.is_file():
        raise MinfluxZarrError(
            f"Package path exists and is not a file: {target}")

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-",
                                    dir=str(target.parent)))
    try:
        store = staging / (target.name[: -len(ZIP_SUFFIX)] + ".zarr")
        if len(members) > 1:
            write_minflux_zarr_project(
                members, store, overwrite=True, roi_records=roi_records,
                image_specs=image_specs, name=name,
            )
        else:
            write_minflux_zarr(
                members[0], store, overwrite=True, roi_records=roi_records,
                image_specs=image_specs,
            )
        packed = pack_minflux_zarr(store, staging / target.name)
        os.replace(packed, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def write_minflux_zarr(
    ds,
    path: str | Path,
    *,
    overwrite: bool = True,
    viewer_only: bool = False,
    roi_records=None,
    image_specs=None,
) -> Path:
    """Transactionally write one self-contained MINFLUX Viewer Zarr v2 store."""
    target = Path(path)
    if target.suffix.lower() != ".zarr":
        target = target.with_suffix(".zarr")
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing Zarr store: {target}")
    if target.exists() and not target.is_dir():
        raise ValueError(f"Zarr output path exists and is not a directory: {target}")
    if target.exists() and viewer_only:
        return update_minflux_zarr_viewer(
            ds, target, roi_records=roi_records, image_specs=image_specs,
        )

    temp = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent)))
    backup: Path | None = None
    try:
        _write_store(ds, temp, roi_records=roi_records)
        image_records = _export_embedded_images(
            temp, [ds], ["d000000"], [temp], image_specs,
        )
        _write_image_manifest(temp, image_records)
        _validate_store(temp)
        if target.exists():
            backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
            os.replace(target, backup)
        try:
            os.replace(temp, target)
        except Exception:
            if backup is not None and backup.exists() and not target.exists():
                os.replace(backup, target)
                backup = None
            raise
        if backup is not None and backup.exists():
            shutil.rmtree(backup)
            backup = None
    finally:
        if temp.exists():
            shutil.rmtree(temp)
        if backup is not None and backup.exists() and target.exists():
            shutil.rmtree(backup)
    return target


def _project_manifest(datasets: list, dataset_ids: list[str], *, name: str | None) -> dict:
    group_ids = {
        str(ds.state.get("overlay_id") or ds.state.get("render_group_id") or "")
        for ds in datasets
    }
    is_overlay = len(datasets) > 1 and len(group_ids - {""}) == 1
    return {
        "name": str(name or "MINFLUX acquisition"),
        "is_overlay": is_overlay,
        "datasets": [
            {
                "id": dataset_id,
                "name": str(getattr(ds, "name", dataset_id)),
                "order": int(ds.state.get("overlay_order", position + 1)),
                "did": _dataset_did(ds),
            }
            for position, (dataset_id, ds) in enumerate(zip(dataset_ids, datasets))
        ],
    }


def _write_project_viewer(
    root,
    datasets: list,
    dataset_ids: list[str],
    *,
    name: str | None,
    roi_records=None,
    image_records=None,
) -> None:
    """Write only the application-owned portion of a project store."""
    viewer = root.require_group("viewer")
    viewer.attrs[COMPONENT_ATTR] = "minflux_viewer_project"
    _write_payload(viewer, "project", _project_manifest(datasets, dataset_ids, name=name))
    if roi_records:
        _write_payload(viewer, "rois", list(roi_records))
    if image_records:
        _write_payload(viewer, "images", list(image_records))


def update_minflux_zarr_project_viewer(
    datasets,
    path: str | Path,
    *,
    roi_records=None,
    image_specs=None,
    name: str | None = None,
) -> Path:
    """Replace project/child viewer groups after matching every raw member."""
    members = list(datasets)
    target = Path(path).resolve()
    zarr, _Blosc = _zarr_modules()
    root = zarr.open_group(str(target), mode="r")
    if root.attrs.get(FORMAT_ATTR) != PROJECT_FORMAT_ID or "viewer" not in root:
        raise MinfluxZarrError(
            "Processing-only update requires an existing multi-dataset "
            "MINFLUX Viewer Zarr project."
        )
    manifest = _read_payload(root["viewer"], "project", {})
    specs = list(manifest.get("datasets") or []) if isinstance(manifest, dict) else []
    if len(specs) != len(members):
        raise MinfluxZarrError(
            f"Existing project has {len(specs)} dataset(s), but this save has "
            f"{len(members)}. Processing-only update was refused."
        )

    available = []
    for spec in specs:
        dataset_id = str(spec.get("id") or "")
        child = target / "datasets" / dataset_id
        available.append({
            "id": dataset_id,
            "did": str(spec.get("did") or ""),
            "fingerprint": _stored_raw_fingerprint(child),
        })

    mapped_ids: list[str] = []
    unmatched = list(available)
    for member in members:
        fingerprint = _dataset_raw_fingerprint(member)
        did = _dataset_did(member)
        candidates = [item for item in unmatched if item["fingerprint"] == fingerprint]
        if did:
            did_candidates = [item for item in candidates if item["did"] == did]
            if did_candidates:
                candidates = did_candidates
        if len(candidates) != 1:
            detail = "does not match" if not candidates else "matches ambiguously"
            raise MinfluxZarrError(
                f"Canonical raw data for '{getattr(member, 'name', 'dataset')}' "
                f"{detail} the existing project. Processing-only update was refused."
            )
        chosen = candidates[0]
        unmatched.remove(chosen)
        mapped_ids.append(chosen["id"])

    image_records = _stored_image_records(target)
    _ensure_requested_images_are_present(image_records, image_specs)
    id_remap = {
        f"d{position:06d}": dataset_id
        for position, dataset_id in enumerate(mapped_ids)
    }
    remapped_rois = []
    for record in roi_records or []:
        payload = dict(record)
        saved_id = str(payload.get("dataset_id") or "")
        if saved_id in id_remap:
            payload["dataset_id"] = id_remap[saved_id]
        remapped_rois.append(payload)

    temp = Path(tempfile.mkdtemp(prefix=f".{target.name}.viewer-tmp-", dir=target.parent))
    try:
        staged_project = temp / "project"
        staged_root = zarr.group(
            store=zarr.DirectoryStore(str(staged_project)), overwrite=True
        )
        _write_project_viewer(
            staged_root,
            members,
            mapped_ids,
            name=name,
            roi_records=remapped_rois,
            image_records=image_records,
        )
        _preserve_unknown_viewer_content(
            target / "viewer",
            staged_project / "viewer",
            known_children={"project", "rois", "images"},
        )
        replacements: list[tuple[Path, Path]] = []
        for member, dataset_id in zip(members, mapped_ids):
            staged_child = temp / "children" / dataset_id
            child_root = zarr.group(
                store=zarr.DirectoryStore(str(staged_child)), overwrite=True
            )
            _write_dataset_viewer(child_root, member)
            _preserve_unknown_viewer_content(
                target / "datasets" / dataset_id / "viewer",
                staged_child / "viewer",
                known_children={
                    "dataset", "metadata", "state", "rois", "images",
                    "derived", "derived_last",
                },
            )
            replacements.append(
                (staged_child / "viewer", target / "datasets" / dataset_id / "viewer")
            )
        replacements.append((staged_project / "viewer", target / "viewer"))
        _replace_viewer_directories(replacements)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return target


def _validate_project(path: Path) -> None:
    zarr, _Blosc = _zarr_modules()
    root = zarr.open_group(str(path), mode="r")
    if root.attrs.get(FORMAT_ATTR) != PROJECT_FORMAT_ID:
        raise MinfluxZarrError("Written store is missing the project format marker.")
    manifest = _read_payload(root["viewer"], "project", {})
    members = list(manifest.get("datasets") or []) if isinstance(manifest, dict) else []
    if not members:
        raise MinfluxZarrError("Written project contains no datasets.")
    for member in members:
        _validate_store(path / "datasets" / str(member["id"]))


def write_minflux_zarr_project(
    datasets,
    path: str | Path,
    *,
    overwrite: bool = True,
    viewer_only: bool = False,
    roi_records=None,
    image_specs=None,
    name: str | None = None,
) -> Path:
    """Transactionally write a multi-dataset acquisition/overlay bundle."""
    members = list(datasets)
    if not members:
        raise ValueError("A MINFLUX Viewer Zarr project requires at least one dataset.")
    target = Path(path)
    if target.suffix.lower() != ".zarr":
        target = target.with_suffix(".zarr")
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing Zarr project: {target}")
    if target.exists() and not target.is_dir():
        raise ValueError(f"Zarr output path exists and is not a directory: {target}")
    if target.exists() and viewer_only:
        return update_minflux_zarr_project_viewer(
            members,
            target,
            roi_records=roi_records,
            image_specs=image_specs,
            name=name,
        )

    temp = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent)))
    backup: Path | None = None
    try:
        zarr, _Blosc = _zarr_modules()
        root = zarr.group(store=zarr.DirectoryStore(str(temp)), overwrite=True)
        root.attrs[FORMAT_ATTR] = PROJECT_FORMAT_ID
        root.attrs[SCHEMA_ATTR] = PROJECT_SCHEMA_VERSION
        root.attrs[CREATED_ATTR] = datetime.now().astimezone().isoformat(timespec="seconds")
        try:
            from .. import __version__
        except Exception:  # pragma: no cover
            __version__ = "unknown"
        root.attrs[APP_VERSION_ATTR] = str(__version__)
        root.require_group("datasets")
        dataset_ids = [f"d{position:06d}" for position in range(len(members))]
        dataset_paths = [temp / "datasets" / dataset_id for dataset_id in dataset_ids]
        for ds, child in zip(members, dataset_paths):
            _write_store(ds, child)
        image_records = _export_embedded_images(
            temp, members, dataset_ids, dataset_paths, image_specs,
        )
        _write_project_viewer(
            root,
            members,
            dataset_ids,
            name=name,
            roi_records=roi_records,
            image_records=image_records,
        )
        _validate_project(temp)
        if target.exists():
            backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
            os.replace(target, backup)
        try:
            os.replace(temp, target)
        except Exception:
            if backup is not None and backup.exists() and not target.exists():
                os.replace(backup, target)
                backup = None
            raise
        if backup is not None and backup.exists():
            shutil.rmtree(backup)
            backup = None
    finally:
        if temp.exists():
            shutil.rmtree(temp)
        if backup is not None and backup.exists() and target.exists():
            shutil.rmtree(backup)
    return target


def _require_schema(root) -> str:
    if root.attrs.get(FORMAT_ATTR) != FORMAT_ID:
        raise MinfluxZarrError(
            "This is not a MINFLUX Viewer Zarr dataset (format marker missing). "
            "Legacy flat-column and raw Imspector stores are not opened by this path."
        )
    version = str(root.attrs.get(SCHEMA_ATTR, ""))
    try:
        major = int(version.split(".", 1)[0])
    except (TypeError, ValueError):
        raise MinfluxZarrError(f"Invalid MINFLUX Viewer schema version: {version!r}") from None
    if major != 1:
        raise MinfluxZarrError(
            f"Unsupported MINFLUX Viewer Zarr schema {version!r}; this build supports 1.x."
        )
    return version


def load_minflux_zarr(path: str | Path, prefs: dict | None = None):
    """Load a marked MINFLUX Viewer Zarr v2 store into the canonical model."""
    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"File not found: {source}")
    return _dataset_from_node(_open_root(source), source, prefs)


def _dataset_from_node(root, source: Path, prefs: dict | None = None):
    """Build one dataset from an already-open store node.

    Taking the **node** rather than a path is what lets a project's child
    datasets load out of a sealed ``.zarr.zip``: inside a package there is no
    ``<pkg>/datasets/d000000`` directory to open, only a group in the archive.
    """
    zarr, _Blosc = _zarr_modules()
    from .dataset import AttributeComponent
    from .loader import load_from_mfx_array
    from .save import columns_to_mfx_array

    schema = _require_schema(root)
    if "mfx" not in root:
        raise MinfluxZarrError("MINFLUX Viewer Zarr dataset has no mfx component.")
    mfx_group = root["mfx"]
    columns = {str(key): np.asarray(mfx_group[key][:]) for key in mfx_group.array_keys()}
    required = {"loc_x", "loc_y", "itr", "vld"}
    missing = sorted(required - set(columns))
    if missing:
        raise MinfluxZarrError("Incomplete mfx component; missing " + ", ".join(missing) + ".")

    ds = load_from_mfx_array(
        columns_to_mfx_array(columns),
        name=source.name,
        folder=str(source.parent),
        datetime_str=datetime.fromtimestamp(source.stat().st_mtime).strftime(
            "%Y-%b-%d, %H:%M:%S"
        ),
        recent_path=str(source),
        prefs=prefs,
    )
    ds.metadata["native_zarr_root_attrs"] = _native_attrs(root)
    ds.metadata["native_zarr_mfx_attrs"] = _native_attrs(mfx_group)
    ds.mfx.roles = dict(mfx_group.attrs.get(ROLES_ATTR, ds.mfx.roles))
    ds.mfx.meta = dict(mfx_group.attrs.get(FIELD_META_ATTR, ds.mfx.meta))

    if "grd/mbm/points" in root:
        mbm_group = root["grd/mbm"]
        points_node = root["grd/mbm/points"]
        points = np.asarray(points_node[:])
        ds.mbm = AttributeComponent(
            {"points": points},
            roles=dict(mbm_group.attrs.get(ROLES_ATTR, {})),
            meta=dict(mbm_group.attrs.get(FIELD_META_ATTR, {})),
        )
        ds.metadata["mbm_points"] = points
        ds.metadata["native_zarr_mbm_attrs"] = _native_attrs(mbm_group)
        ds.metadata["native_zarr_mbm_points_attrs"] = _native_attrs(points_node)
        points_by_gri = points_node.attrs.get("points_by_gri")
        if points_by_gri is not None:
            ds.metadata["mbm_points_by_gri"] = _restore_plain_json(points_by_gri)
        used = mbm_group.attrs.get("used")
        if used is not None:
            ds.metadata["mbm_used"] = _restore_plain_json(used)
    if "grd/search_0/points" in root:
        search_group = root["grd/search_0"]
        points_node = root["grd/search_0/points"]
        ds.metadata["search_points"] = np.asarray(points_node[:])
        ds.metadata["native_zarr_search_attrs"] = _native_attrs(search_group)
        ds.metadata["native_zarr_search_points_attrs"] = _native_attrs(points_node)

    if "viewer" in root:
        viewer = root["viewer"]
        saved_metadata = _read_payload(viewer, "metadata", {})
        if not isinstance(saved_metadata, dict):
            raise MinfluxZarrError("Viewer metadata payload must be a dictionary.")
        ds.metadata.update(saved_metadata)
        state = _read_payload(viewer, "state", {})
        if not isinstance(state, dict):
            raise MinfluxZarrError("Viewer state payload must be a dictionary.")
        ds.state.update(state)
        info = _read_payload(viewer, "dataset", {})
        if isinstance(info, dict):
            cal = dict(info.get("calibration") or {})
            if cal.get("z_scaling_factor") is not None:
                ds.cali.z_scaling_factor = float(cal["z_scaling_factor"])
            if cal.get("pixel_size_nm") is not None:
                ds.cali.pixel_size = float(cal["pixel_size_nm"])
            if cal.get("loc_precision_nm") is not None:
                ds.cali.loc_precision = np.asarray(cal["loc_precision_nm"], dtype=float)
            channel = dict(info.get("channel") or {})
            for key in ("num_channels", "cut1", "cut2", "do_trace", "keep_ch3"):
                if key in channel:
                    setattr(ds.channel, key, channel[key])
        for key, value in _read_attr_store(viewer, "derived").items():
            ds.derived[key] = value
            if np.asarray(value).ndim == 1 and np.asarray(value).size == ds.prop.num_loc:
                ds.attr[key] = value
                if key not in ds.prop.attr_names:
                    ds.prop.attr_names.append(key)
        for key, value in _read_attr_store(viewer, "derived_last").items():
            ds.derived_last[key] = value
        roi_records = _read_payload(viewer, "rois", [])
        if isinstance(roi_records, list) and roi_records:
            ds.metadata["minflux_viewer_roi_records"] = roi_records
        image_records = _read_payload(viewer, "images", [])
        if isinstance(image_records, list) and image_records:
            resolved_images = []
            for record in image_records:
                item = dict(record)
                item["absolute_path"] = str(source / str(item.get("path") or ""))
                if is_zipped_store(source):
                    item["package_path"] = str(source)
                resolved_images.append(item)
            ds.metadata["minflux_viewer_images"] = resolved_images

    mask = ds.state.get("filter_mask")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).ravel()
        if mask.size != ds.prop.num_loc:
            raise MinfluxZarrError(
                f"Saved filter mask has {mask.size} rows; expected {ds.prop.num_loc}."
            )
        ds.filter_mask = mask
    ds.metadata["source_format"] = "minflux-viewer zarr v2"
    ds.metadata["minflux_viewer_schema_version"] = schema
    ds.metadata["minflux_viewer_zarr_path"] = str(source)
    return ds


def materialize_image(record: dict) -> Path | None:
    """An on-disk path for an embedded image record, or ``None``.

    A directory store already has the file. A sealed package does not: its
    ``absolute_path`` points *inside* the zip, so the member is extracted once
    to a temp file and that path returned. Without this, images restored from a
    ``.zarr.zip`` silently fell back to reopening the original ``.msr`` -- which
    only worked while that file was still on disk at its original location.
    """
    import zipfile

    direct = Path(str(record.get("absolute_path") or ""))
    if direct.is_file():
        return direct
    package = str(record.get("package_path") or "")
    member = str(record.get("path") or "")
    if not package or not member or not Path(package).is_file():
        return None

    cache = Path(tempfile.gettempdir()) / "minflux-viewer-images"
    stamp = hashlib.sha256(f"{package}\x00{member}".encode()).hexdigest()[:16]
    target = cache / f"{stamp}-{Path(member).name}"
    if target.is_file():
        return target
    cache.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(package) as archive:
            data = archive.read(member)
    except (KeyError, OSError, zipfile.BadZipFile):
        return None
    partial = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    try:
        partial.write_bytes(data)
        os.replace(partial, target)
    finally:
        if partial.exists():
            partial.unlink()
    return target


def _open_root(source: Path):
    """Open a store root, whether it is a directory or a sealed package.

    Both loaders go through this: the GUI opens a ``.zarr`` via
    :func:`load_minflux_zarr_project` (so a multi-dataset acquisition comes back
    as an overlay) while scripts often call :func:`load_minflux_zarr`, and a
    sealed package has to work through either.
    """
    zarr, _Blosc = _zarr_modules()
    if source.is_file() and is_zipped_store(source):
        # Read-only: a sealed package cannot take a processing-only update.
        return zarr.open(store=zarr.ZipStore(str(source), mode="r"), mode="r")
    if not source.is_dir():
        raise MinfluxZarrError(
            f"Zarr dataset must be a directory or a sealed {ZIP_SUFFIX} "
            f"package: '{source.name}'.")
    return zarr.open(str(source), mode="r")


def load_minflux_zarr_project(
    path: str | Path,
    prefs: dict | None = None,
) -> LoadedMinfluxZarrProject:
    """Load either a single dataset store or a multi-dataset project bundle."""
    zarr, _Blosc = _zarr_modules()
    source = Path(path).resolve()
    root = _open_root(source)
    marker = root.attrs.get(FORMAT_ATTR)
    if marker == FORMAT_ID:
        ds = load_minflux_zarr(source, prefs=prefs)
        rois = list(ds.metadata.get("minflux_viewer_roi_records") or [])
        images = list(ds.metadata.get("minflux_viewer_images") or [])
        manifest = {
            "name": source.stem,
            "is_overlay": False,
            "datasets": [{"id": "d000000", "name": ds.name, "order": 1}],
        }
        return LoadedMinfluxZarrProject([ds], manifest, rois, images, source)
    if marker != PROJECT_FORMAT_ID:
        raise MinfluxZarrError(
            "This is not a MINFLUX Viewer Zarr dataset or project "
            "(format marker missing)."
        )
    version = str(root.attrs.get(SCHEMA_ATTR, ""))
    try:
        major = int(version.split(".", 1)[0])
    except (TypeError, ValueError):
        raise MinfluxZarrError(f"Invalid MINFLUX Viewer project schema: {version!r}") from None
    if major != 1:
        raise MinfluxZarrError(
            f"Unsupported MINFLUX Viewer project schema {version!r}; this build supports 1.x."
        )
    if "viewer" not in root:
        raise MinfluxZarrError("MINFLUX Viewer project has no viewer manifest.")
    viewer = root["viewer"]
    manifest = _read_payload(viewer, "project", {})
    if not isinstance(manifest, dict):
        raise MinfluxZarrError("MINFLUX Viewer project manifest must be a dictionary.")
    member_specs = list(manifest.get("datasets") or [])
    if not member_specs:
        raise MinfluxZarrError("MINFLUX Viewer project contains no datasets.")

    datasets = []
    by_id: dict[str, object] = {}
    for position, spec in enumerate(member_specs):
        dataset_id = str(spec.get("id") or "")
        if "datasets" not in root or dataset_id not in root["datasets"]:
            raise MinfluxZarrError(
                f"Project manifest lists dataset '{dataset_id}', but the store "
                f"has no such member.")
        child = source / "datasets" / dataset_id
        ds = _dataset_from_node(root["datasets"][dataset_id], source, prefs)
        saved_name = str(spec.get("name") or ds.name)
        ds.file.name = saved_name
        ds.file.folder = str(child)
        ds.file.recent_path = str(source)
        ds.metadata["minflux_viewer_project_path"] = str(source)
        ds.metadata["minflux_viewer_project_dataset_id"] = dataset_id
        ds.metadata["minflux_viewer_schema_version"] = version
        datasets.append(ds)
        by_id[dataset_id] = ds

    if bool(manifest.get("is_overlay")) and len(datasets) > 1:
        group_id = f"zarr:{source}"
        for position, (spec, ds) in enumerate(zip(member_specs, datasets)):
            ds.state["overlay_id"] = group_id
            ds.state["render_group_id"] = group_id
            ds.state["overlay_order"] = int(spec.get("order", position + 1))
            ds.metadata["overlay_id"] = group_id

    image_records = _read_payload(viewer, "images", [])
    images: list[dict] = []
    if isinstance(image_records, list):
        for record in image_records:
            item = dict(record)
            item["absolute_path"] = str(source / str(item.get("path") or ""))
            if is_zipped_store(source):
                item["package_path"] = str(source)
            images.append(item)
            dataset_id = item.get("dataset_id")
            if dataset_id in by_id:
                by_id[dataset_id].metadata.setdefault("minflux_viewer_images", []).append(item)

    roi_records = _read_payload(viewer, "rois", [])
    rois = list(roi_records) if isinstance(roi_records, list) else []
    for record in rois:
        dataset_id = record.get("dataset_id") if isinstance(record, dict) else None
        if dataset_id in by_id:
            by_id[dataset_id].metadata.setdefault("minflux_viewer_roi_records", []).append(record)
    return LoadedMinfluxZarrProject(datasets, manifest, rois, images, source)
