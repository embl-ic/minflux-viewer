"""OME-NGFF 0.5 / Zarr v3 export for localization datasets.

OME-NGFF currently standardizes raster images, labels, and plate layouts, but
not localization tables, processing recipes, or vector ROIs. This exporter
therefore writes a standards-compliant 2-D or 3-D density pyramid at the root
and keeps the richer MINFLUX content in a versioned extension below
``minflux/``.

The writer is intentionally independent of the application's pinned Zarr v2
dependency. It emits the small Zarr v3 core needed by this profile, avoiding a
global dependency upgrade that would affect MSR decoding.
"""

from __future__ import annotations

import json
import math
import shutil
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from gzip import compress as gzip_compress
from itertools import product
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from .. import __version__
from .loader import mfx_row_mask
from .save import build_metadata, build_snapshot_table, dataset_to_mfx_array

PROFILE_NAME = "minflux-viewer-localization"
PROFILE_VERSION = "0.1.0"
OME_NGFF_VERSION = "0.5"
ZARR_FORMAT = 3
DEFAULT_ROW_CHUNK = 65_536
DEFAULT_IMAGE_CHUNK = 256
DEFAULT_Z_CHUNK = 16
ESTIMATED_GZIP_BYTES_PER_SECOND = 120 * 1024**2
ESTIMATED_TABLE_BYTES_PER_SECOND = 180 * 1024**2
ESTIMATED_LOCALIZATIONS_PER_SECOND = 2_000_000

ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class OmeZarrExportResult:
    path: Path
    effective_pixel_size_nm: float
    levels: int
    is_3d: bool
    image_shape: tuple[int, ...]
    voxel_size_nm: tuple[float, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class OmeZarrExportEstimate:
    """Conservative preflight estimate for one OME-Zarr export."""

    is_3d: bool
    image_shape: tuple[int, ...]
    level_shapes: tuple[tuple[int, ...], ...]
    voxel_size_nm: tuple[float, ...]
    filtered_localizations: int
    estimated_output_bytes: int
    upper_output_bytes: int
    peak_working_ram_bytes: int
    dense_level_zero_bytes: int
    available_ram_bytes: int
    free_disk_bytes: int
    estimated_seconds: float
    warnings: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()


def normalize_ome_zarr_path(path: str | Path) -> Path:
    """Return *path* with the conventional ``.ome.zarr`` suffix."""
    output = Path(path)
    lower = output.name.lower()
    if lower.endswith(".ome.zarr"):
        return output
    if lower.endswith(".zarr"):
        return output.with_name(output.name[:-5] + ".ome.zarr")
    return output.with_name(output.name + ".ome.zarr")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _component(name: str) -> str:
    """Encode an arbitrary field name as one safe Zarr path component."""
    encoded = quote(str(name), safe="-_.")
    if not encoded or encoded in {".", ".."}:
        raise ValueError(f"Invalid Zarr node name: {name!r}")
    return encoded


def _group_metadata(attributes: dict | None = None) -> dict:
    return {
        "zarr_format": ZARR_FORMAT,
        "node_type": "group",
        "attributes": _jsonable(attributes or {}),
    }


def _ensure_group(root: Path, relative: str = "", attributes: dict | None = None) -> Path:
    current = root
    parts = [part for part in relative.split("/") if part]
    for part in parts:
        current = current / part
        current.mkdir(parents=True, exist_ok=True)
        metadata_path = current / "zarr.json"
        if not metadata_path.exists():
            _write_json(metadata_path, _group_metadata())
    metadata_path = current / "zarr.json"
    if not metadata_path.exists() or attributes is not None:
        existing = {}
        if metadata_path.exists():
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        merged = dict(existing.get("attributes") or {})
        if attributes:
            merged.update(_jsonable(attributes))
        _write_json(metadata_path, _group_metadata(merged))
    return current


def _zarr_dtype(dtype: np.dtype) -> tuple[str, np.dtype, bool]:
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return "bool", np.dtype(np.bool_), False
    if dtype.kind not in {"i", "u", "f"}:
        raise TypeError(f"Zarr v3 export supports real numeric arrays, not {dtype}")
    names = {
        ("i", 1): "int8",
        ("i", 2): "int16",
        ("i", 4): "int32",
        ("i", 8): "int64",
        ("u", 1): "uint8",
        ("u", 2): "uint16",
        ("u", 4): "uint32",
        ("u", 8): "uint64",
        ("f", 2): "float16",
        ("f", 4): "float32",
        ("f", 8): "float64",
    }
    name = names.get((dtype.kind, dtype.itemsize))
    if name is None:
        raise TypeError(f"Unsupported Zarr v3 dtype: {dtype}")
    return name, dtype.newbyteorder("<"), dtype.itemsize > 1


def _normal_chunk_shape(shape: tuple[int, ...], requested: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) != len(requested):
        raise ValueError("Chunk rank must match array rank.")
    return tuple(max(1, min(int(dim) if dim else 1, int(chunk))) for dim, chunk in zip(shape, requested))


def _create_array(
    root: Path,
    relative: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
    *,
    chunk_shape: tuple[int, ...],
    dimension_names: list[str | None] | None = None,
    attributes: dict | None = None,
) -> tuple[Path, tuple[int, ...], np.dtype]:
    data_type, storage_dtype, endian_applies = _zarr_dtype(dtype)
    shape = tuple(int(value) for value in shape)
    chunks = _normal_chunk_shape(shape, chunk_shape)
    node_path = root.joinpath(*relative.split("/"))
    _ensure_group(root, "/".join(relative.split("/")[:-1]))
    node_path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "zarr_format": ZARR_FORMAT,
        "node_type": "array",
        "shape": list(shape),
        "data_type": data_type,
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": list(chunks)},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": False if data_type == "bool" else 0,
        "codecs": [
            (
                {"name": "bytes", "configuration": {"endian": "little"}}
                if endian_applies
                else {"name": "bytes"}
            ),
            {"name": "gzip", "configuration": {"level": 1}},
        ],
        "storage_transformers": [],
        "dimension_names": dimension_names or [None] * len(shape),
        "attributes": _jsonable(attributes or {}),
    }
    _write_json(node_path / "zarr.json", metadata)
    return node_path, chunks, storage_dtype


def _write_chunk(
    node_path: Path,
    chunk_index: tuple[int, ...],
    block: np.ndarray,
    storage_dtype: np.dtype,
) -> None:
    chunk_path = node_path / "c"
    for index in chunk_index:
        chunk_path = chunk_path / str(index)
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = np.asarray(block, dtype=storage_dtype, order="C").tobytes(order="C")
    chunk_path.write_bytes(gzip_compress(encoded, compresslevel=1, mtime=0))


def _write_array(
    root: Path,
    relative: str,
    values: np.ndarray,
    *,
    chunk_shape: tuple[int, ...] | None = None,
    dimension_names: list[str | None] | None = None,
    attributes: dict | None = None,
    chunk_progress: Callable[[int, int], None] | None = None,
) -> None:
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    shape = tuple(int(value) for value in array.shape)
    if chunk_shape is None:
        chunk_shape = (min(max(shape[0], 1), DEFAULT_ROW_CHUNK),) + tuple(
            max(value, 1) for value in shape[1:]
        )
    node_path, chunks, storage_dtype = _create_array(
        root,
        relative,
        shape,
        array.dtype,
        chunk_shape=chunk_shape,
        dimension_names=dimension_names,
        attributes=attributes,
    )
    if any(dim == 0 for dim in shape):
        return

    chunk_counts = tuple(int(math.ceil(dim / chunk)) for dim, chunk in zip(shape, chunks))
    total_chunks = math.prod(chunk_counts)
    for done, chunk_index in enumerate(
        product(*(range(count) for count in chunk_counts)),
        start=1,
    ):
        source_slices = tuple(
            slice(index * chunk, min((index + 1) * chunk, dim))
            for index, chunk, dim in zip(chunk_index, chunks, shape)
        )
        source = np.asarray(array[source_slices], dtype=storage_dtype, order="C")
        # Zarr v3 regular chunks have their declared full shape at the edges.
        block = np.zeros(chunks, dtype=storage_dtype)
        target_slices = tuple(slice(0, size) for size in source.shape)
        block[target_slices] = source
        if np.any(block):
            _write_chunk(node_path, chunk_index, block, storage_dtype)
        if chunk_progress is not None:
            chunk_progress(done, total_chunks)


def _field_unit(name: str) -> str | None:
    if name in {"loc", "lnc", "ext"}:
        return "meter"
    if name == "tim":
        return "second"
    return None


def _write_structured_table(
    root: Path,
    relative: str,
    table: np.ndarray,
    *,
    row_name: str,
    component_role: str,
    progress: ProgressCallback | None = None,
    progress_start: float = 0.0,
    progress_span: float = 0.0,
) -> list[dict[str, Any]]:
    if table.dtype.names is None:
        raise TypeError(f"{component_role} must be a structured array.")
    _ensure_group(
        root,
        relative,
        {
            "table_role": component_role,
            "row_dimension": row_name,
            "field_order": list(table.dtype.names),
        },
    )
    schema = []
    field_count = len(table.dtype.names)
    for field_index, name in enumerate(table.dtype.names, start=1):
        values = np.asarray(table[name])
        if values.ndim == 1:
            dimensions = [row_name]
        elif values.ndim == 2:
            second = "coordinate" if values.shape[1] == 3 else f"{_component(name)}_component"
            dimensions = [row_name, second]
        else:
            raise TypeError(f"Field {name!r} has unsupported shape {values.shape}.")
        attrs = {"source_field": name}
        unit = _field_unit(name)
        if unit:
            attrs["unit"] = unit
        _write_array(
            root,
            f"{relative}/{_component(name)}",
            values,
            dimension_names=dimensions,
            attributes=attrs,
            chunk_progress=(
                (
                    lambda done, total, i=field_index, n=field_count, field=name: progress(
                        progress_start
                        + progress_span * ((i - 1) + done / max(total, 1)) / max(n, 1),
                        f"Writing raw MINFLUX field {field}",
                    )
                )
                if progress is not None
                else None
            ),
        )
        schema.append(
            {
                "name": name,
                "path": f"{relative}/{_component(name)}",
                "dtype": str(values.dtype),
                "shape": list(values.shape),
                "unit": unit,
            }
        )
    return schema


def _source_row_ids(ds, n_processed: int) -> np.ndarray:
    raw = getattr(ds, "mfx_raw", None)
    if raw is not None and len(raw):
        mask = mfx_row_mask(raw, itr="last", vld_only=True)
        if mask is None:
            raise ValueError("Raw MINFLUX store has no selectable rows.")
        ids = np.flatnonzero(mask).astype(np.uint64)
        if ids.size != n_processed:
            raise ValueError(
                "Processed rows do not align with the raw last-valid selection "
                f"({n_processed} processed, {ids.size} raw)."
            )
        return ids
    raw_table = dataset_to_mfx_array(ds)
    if len(raw_table) != n_processed:
        raise ValueError(
            "Coordinate-built dataset rows do not align with its canonical raw export."
        )
    return np.arange(n_processed, dtype=np.uint64)


@dataclass(frozen=True)
class _DensityGrid:
    is_3d: bool
    indices: np.ndarray
    origin_xyz_nm: tuple[float, float, float]
    voxel_size_nm: tuple[float, ...]
    level_shapes: tuple[tuple[int, ...], ...]


def _prepare_export_data(ds):
    raw = dataset_to_mfx_array(ds)
    processed, _dropped = build_snapshot_table(
        ds,
        include_attrs=True,
        include_derived=True,
        filter_mode="flag",
    )
    position = np.column_stack(
        [
            np.asarray(processed.pop("xnm"), dtype=np.float64),
            np.asarray(processed.pop("ynm"), dtype=np.float64),
            np.asarray(processed.pop("znm"), dtype=np.float64),
        ]
    )
    keep = np.asarray(processed.get("ftr", np.ones(position.shape[0], bool)), dtype=bool)
    return raw, processed, position, keep


def _density_grid(
    position_nm: np.ndarray,
    keep: np.ndarray,
    *,
    pixel_size_nm: float,
    z_voxel_nm: float | None,
    is_3d: bool,
    max_levels: int,
) -> _DensityGrid:
    pixel = float(pixel_size_nm)
    if not math.isfinite(pixel) or pixel <= 0.0:
        raise ValueError("XY pixel size must be a positive finite value.")
    if max_levels < 1:
        raise ValueError("At least one pyramid level is required.")
    z_depth = float(z_voxel_nm or 0.0)
    if is_3d and (not math.isfinite(z_depth) or z_depth <= 0.0):
        raise ValueError("Z voxel depth must be a positive finite value for 3-D data.")

    position = np.asarray(position_nm, dtype=np.float64)
    finite = (
        np.asarray(keep, dtype=bool)
        & np.isfinite(position[:, 0])
        & np.isfinite(position[:, 1])
    )
    if is_3d:
        finite &= np.isfinite(position[:, 2])
    if not np.any(finite):
        dimensionality = "XYZ" if is_3d else "XY"
        raise ValueError(f"No finite, unfiltered {dimensionality} localizations are available.")

    points = position[finite]

    def grid_origin(value: float, spacing: float) -> float:
        scaled = value / spacing
        nearest = round(scaled)
        if abs(scaled - nearest) < 1.0e-9:
            scaled = float(nearest)
        return math.floor(scaled) * spacing

    x0 = grid_origin(float(np.min(points[:, 0])), pixel)
    y0 = grid_origin(float(np.min(points[:, 1])), pixel)

    def voxel_indices(values: np.ndarray, origin: float, spacing: float) -> np.ndarray:
        scaled = (values - origin) / spacing
        nearest = np.rint(scaled)
        scaled = np.where(np.abs(scaled - nearest) < 1.0e-9, nearest, scaled)
        return np.floor(scaled).astype(np.int64)

    x_index = voxel_indices(points[:, 0], x0, pixel)
    y_index = voxel_indices(points[:, 1], y0, pixel)
    if is_3d:
        z0 = grid_origin(float(np.min(points[:, 2])), z_depth)
        z_index = voxel_indices(points[:, 2], z0, z_depth)
        indices = np.column_stack([z_index, y_index, x_index])
        shape = (
            int(np.max(z_index)) + 1,
            int(np.max(y_index)) + 1,
            int(np.max(x_index)) + 1,
        )
        voxel_size = (z_depth, pixel, pixel)
    else:
        z0 = 0.0
        indices = np.column_stack([y_index, x_index])
        shape = (int(np.max(y_index)) + 1, int(np.max(x_index)) + 1)
        voxel_size = (pixel, pixel)

    levels = [shape]
    while len(levels) < int(max_levels) and max(levels[-1][-2:]) > DEFAULT_IMAGE_CHUNK:
        previous = levels[-1]
        levels.append(
            (*previous[:-2], int(math.ceil(previous[-2] / 2)), int(math.ceil(previous[-1] / 2)))
        )
    return _DensityGrid(
        is_3d=is_3d,
        indices=indices,
        origin_xyz_nm=(x0, y0, z0),
        voxel_size_nm=voxel_size,
        level_shapes=tuple(levels),
    )


def _density_chunk_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) == 3:
        return (
            min(shape[0], DEFAULT_Z_CHUNK),
            min(shape[1], DEFAULT_IMAGE_CHUNK),
            min(shape[2], DEFAULT_IMAGE_CHUNK),
        )
    return (
        min(shape[0], DEFAULT_IMAGE_CHUNK),
        min(shape[1], DEFAULT_IMAGE_CHUNK),
    )


def _chunk_groups(
    indices: np.ndarray,
    chunks: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chunk_coords = indices // np.asarray(chunks, dtype=np.int64)
    order = np.lexsort(tuple(chunk_coords[:, axis] for axis in range(indices.shape[1] - 1, -1, -1)))
    sorted_coords = chunk_coords[order]
    if sorted_coords.shape[0] == 0:
        return order, np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    starts = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            np.flatnonzero(np.any(np.diff(sorted_coords, axis=0) != 0, axis=1)) + 1,
        ]
    )
    ends = np.concatenate([starts[1:], np.array([len(order)], dtype=np.int64)])
    return order, starts, ends


def _write_density_pyramid(
    root: Path,
    grid: _DensityGrid,
    *,
    progress: ProgressCallback | None,
    progress_start: float,
    progress_span: float,
) -> None:
    dimension_names = ["z", "y", "x"] if grid.is_3d else ["y", "x"]
    level_count = len(grid.level_shapes)
    for level, shape in enumerate(grid.level_shapes):
        factor = 2**level
        indices = np.array(grid.indices, copy=True)
        indices[:, -2:] //= factor
        requested_chunks = _density_chunk_shape(shape)
        node_path, chunks, storage_dtype = _create_array(
            root,
            str(level),
            shape,
            np.dtype(np.uint32),
            chunk_shape=requested_chunks,
            dimension_names=dimension_names,
            attributes={
                "image_role": (
                    "derived 3-D localization density"
                    if grid.is_3d
                    else "derived XY localization density"
                ),
                "unit": "localization count",
                "downsampling": "XY 2x2 sum; Z sampling preserved",
            },
        )
        order, starts, ends = _chunk_groups(indices, chunks)
        total_groups = max(len(starts), 1)
        level_start = progress_start + progress_span * level / level_count
        level_span = progress_span / level_count
        for group_number, (start, end) in enumerate(zip(starts, ends), start=1):
            rows = order[start:end]
            chunk_index = tuple((indices[rows[0]] // np.asarray(chunks)).tolist())
            local = indices[rows] - np.asarray(chunk_index) * np.asarray(chunks)
            block = np.zeros(chunks, dtype=np.uint32)
            np.add.at(block, tuple(local[:, axis] for axis in range(local.shape[1])), 1)
            _write_chunk(node_path, chunk_index, block, storage_dtype)
            if progress is not None:
                progress(
                    level_start + level_span * group_number / total_groups,
                    f"Writing voxel pyramid level {level + 1}/{level_count}",
                )


def _existing_disk_probe(path: Path) -> Path:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return probe


def estimate_ome_zarr_export(
    ds,
    path: str | Path,
    *,
    pixel_size_nm: float,
    z_voxel_nm: float | None = None,
    max_levels: int = 6,
) -> OmeZarrExportEstimate:
    """Estimate output size, peak RAM, and duration without allocating a volume."""
    import psutil

    raw, processed, position, keep = _prepare_export_data(ds)
    is_3d = int(getattr(ds.prop, "num_dim", 2)) >= 3
    grid = _density_grid(
        position,
        keep,
        pixel_size_nm=pixel_size_nm,
        z_voxel_nm=z_voxel_nm,
        is_3d=is_3d,
        max_levels=max_levels,
    )
    table_bytes = int(raw.nbytes + position.nbytes)
    table_bytes += sum(int(np.asarray(value).nbytes) for value in processed.values())

    density_bytes = 0
    touched_bytes = 0
    for level, shape in enumerate(grid.level_shapes):
        density_bytes += math.prod(shape) * np.dtype(np.uint32).itemsize
        chunks = _density_chunk_shape(shape)
        level_indices = np.array(grid.indices, copy=True)
        level_indices[:, -2:] //= 2**level
        _order, starts, _ends = _chunk_groups(level_indices, chunks)
        touched_bytes += len(starts) * math.prod(chunks) * np.dtype(np.uint32).itemsize

    occupied_chunk_overhead = max(1, len(grid.level_shapes) * grid.indices.shape[0]) * 16
    estimated_density_bytes = min(
        touched_bytes,
        int(touched_bytes * 0.08 + occupied_chunk_overhead),
    )
    estimated_output = int(estimated_density_bytes + table_bytes * 0.70 + 2 * 1024**2)
    upper_output = int(touched_bytes * 1.01 + table_bytes * 1.05 + 10 * 1024**2)
    largest_chunk = max(
        math.prod(_density_chunk_shape(shape)) * np.dtype(np.uint32).itemsize
        for shape in grid.level_shapes
    )
    row_count = position.shape[0]
    index_workspace = row_count * (grid.indices.shape[1] * 24 + 8)
    peak_ram = int(table_bytes + position.nbytes + index_workspace + largest_chunk * 2)
    estimated_seconds = (
        1.0
        + touched_bytes / ESTIMATED_GZIP_BYTES_PER_SECOND
        + table_bytes / ESTIMATED_TABLE_BYTES_PER_SECOND
        + grid.indices.shape[0]
        * len(grid.level_shapes)
        / ESTIMATED_LOCALIZATIONS_PER_SECOND
    )

    available_ram = int(psutil.virtual_memory().available)
    output = normalize_ome_zarr_path(path)
    free_disk = int(shutil.disk_usage(_existing_disk_probe(output.parent)).free)
    dense_level_zero = int(math.prod(grid.level_shapes[0]) * np.dtype(np.uint32).itemsize)
    warnings: list[str] = []
    blockers: list[str] = []
    if estimated_seconds >= 60:
        if estimated_seconds >= 300:
            band = "5 minutes"
        elif estimated_seconds >= 180:
            band = "3 minutes"
        else:
            band = "1 minute"
        warnings.append(f"Estimated conversion time is over {band}.")
    if dense_level_zero > available_ram:
        warnings.append(
            "The full-resolution dense stack is larger than currently available RAM; "
            "Fiji may need a virtual/lazy stack or a coarser voxel size."
        )
    if peak_ram > available_ram * 0.70:
        warnings.append("Estimated exporter working RAM exceeds 70% of available RAM.")
    if upper_output > free_disk * 0.80:
        warnings.append("Worst-case package size exceeds 80% of free disk space.")
    if peak_ram > available_ram * 0.95:
        blockers.append("Estimated exporter working RAM exceeds available system capacity.")
    if estimated_output > free_disk * 0.95:
        blockers.append("Estimated compressed package does not fit on the target disk.")
    if any(dimension > np.iinfo(np.int32).max for dimension in grid.level_shapes[0]):
        blockers.append("At least one image dimension exceeds the supported 32-bit index range.")

    return OmeZarrExportEstimate(
        is_3d=is_3d,
        image_shape=grid.level_shapes[0],
        level_shapes=grid.level_shapes,
        voxel_size_nm=grid.voxel_size_nm,
        filtered_localizations=int(grid.indices.shape[0]),
        estimated_output_bytes=estimated_output,
        upper_output_bytes=upper_output,
        peak_working_ram_bytes=peak_ram,
        dense_level_zero_bytes=dense_level_zero,
        available_ram_bytes=available_ram,
        free_disk_bytes=free_disk,
        estimated_seconds=float(estimated_seconds),
        warnings=tuple(warnings),
        blockers=tuple(blockers),
    )


def _record_dict(record: Any) -> dict[str, Any]:
    if is_dataclass(record):
        return _jsonable(asdict(record))
    if isinstance(record, dict):
        return _jsonable(record)
    raise TypeError(f"Unsupported ROI record type: {type(record).__name__}")


def _journal_dict(entry: Any) -> dict[str, Any]:
    if is_dataclass(entry):
        return _jsonable(asdict(entry))
    if isinstance(entry, dict):
        return _jsonable(entry)
    return {
        "timestamp": str(getattr(entry, "timestamp", "")),
        "category": str(getattr(entry, "category", "")),
        "summary": str(getattr(entry, "summary", "")),
        "details": _jsonable(getattr(entry, "details", {})),
    }


def _serializable_view_state(ds, warnings: list[str]) -> dict[str, Any]:
    excluded = {"filter_mask", "filter_specs", "roi_masks"}
    exported = {}
    for key, value in ds.state.items():
        if key in excluded:
            continue
        try:
            exported[str(key)] = _jsonable(value)
        except TypeError as exc:
            warnings.append(f"View-state field {key!r} was not exported: {exc}.")
    return exported


def _write_mbm(root: Path, base: str, ds) -> list[dict[str, Any]]:
    component = getattr(ds, "mbm", None)
    attrs = getattr(component, "attrs", None)
    if not attrs:
        return []
    _ensure_group(root, base, {"table_role": "MINFLUX bead/reference data"})
    schema = []
    for name, value in attrs.items():
        array = np.asarray(value)
        if array.ndim == 0:
            array = array.reshape(1)
        dimensions = ["bead_event"] + [f"{_component(name)}_component_{i}" for i in range(1, array.ndim)]
        _write_array(
            root,
            f"{base}/{_component(name)}",
            array,
            dimension_names=dimensions,
            attributes={"source_field": name},
        )
        schema.append(
            {
                "name": name,
                "path": f"{base}/{_component(name)}",
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            }
        )
    return schema


def _write_package(
    root: Path,
    ds,
    *,
    pixel_size_nm: float,
    z_voxel_nm: float | None,
    max_levels: int,
    dataset_idx: int | None,
    roi_records: Iterable[Any],
    journal_entries: Iterable[Any],
    progress: ProgressCallback | None,
) -> OmeZarrExportResult:
    warnings: list[str] = []
    if progress is not None:
        progress(0.01, "Preparing raw and processed localization tables")
    raw, processed, position, keep = _prepare_export_data(ds)
    n_processed = position.shape[0]
    source_row_id = _source_row_ids(ds, n_processed)
    is_3d = int(getattr(ds.prop, "num_dim", 2)) >= 3
    grid = _density_grid(
        position,
        keep,
        pixel_size_nm=pixel_size_nm,
        z_voxel_nm=z_voxel_nm,
        is_3d=is_3d,
        max_levels=int(max_levels),
    )
    if progress is not None:
        progress(0.06, "Building OME-NGFF image geometry")

    source_path = str(getattr(getattr(ds, "file", None), "path", "") or "")
    dataset_name = str(getattr(ds, "name", "") or "dataset")
    dataset_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"minflux-viewer:{source_path}:{dataset_name}",
        )
    )
    dataset_base = f"minflux/datasets/{dataset_id}"
    x0_nm, y0_nm, z0_nm = grid.origin_xyz_nm
    ome_datasets = []
    for level, _shape in enumerate(grid.level_shapes):
        xy_scale_um = float(pixel_size_nm) * (2 ** level) / 1000.0
        if grid.is_3d:
            scale = [float(z_voxel_nm) / 1000.0, xy_scale_um, xy_scale_um]
            translation = [z0_nm / 1000.0, y0_nm / 1000.0, x0_nm / 1000.0]
        else:
            scale = [xy_scale_um, xy_scale_um]
            translation = [y0_nm / 1000.0, x0_nm / 1000.0]
        ome_datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {"type": "scale", "scale": scale},
                    {"type": "translation", "translation": translation},
                ],
            }
        )

    axes = (
        [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]
        if grid.is_3d
        else [
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]
    )
    image_kind = "3-D localization-density volume" if grid.is_3d else "XY localization-density image"
    root_attributes = {
        "ome": {
            "version": OME_NGFF_VERSION,
            "multiscales": [
                {
                    "name": f"{dataset_name} {image_kind}",
                    "type": "localization-count-sum",
                    "metadata": {
                        "method": "XY 2x2 sum; Z sampling preserved",
                        "source": "filtered last-valid MINFLUX localizations",
                    },
                    "axes": axes,
                    "datasets": ome_datasets,
                }
            ],
        },
        "org.minflux_viewer": {
            "profile": PROFILE_NAME,
            "profile_version": PROFILE_VERSION,
            "localization_group": dataset_base,
            "density_semantics": (
                "localization count per XYZ voxel after active filters"
                if grid.is_3d
                else "localization count per XY pixel after active filters"
            ),
            "requested_pixel_size_nm": float(pixel_size_nm),
            "z_voxel_depth_nm": float(z_voxel_nm) if grid.is_3d else None,
            "image_shape": list(grid.level_shapes[0]),
        },
    }
    root.mkdir(parents=True, exist_ok=False)
    _write_json(root / "zarr.json", _group_metadata(root_attributes))
    _write_density_pyramid(
        root,
        grid,
        progress=progress,
        progress_start=0.07,
        progress_span=0.43,
    )

    _ensure_group(
        root,
        "minflux",
        {
            "profile": PROFILE_NAME,
            "profile_version": PROFILE_VERSION,
            "standard_status": (
                "MINFLUX localization tables, ROI state, and provenance are a "
                "MINFLUX Viewer extension; the root image follows OME-NGFF 0.5."
            ),
        },
    )
    _ensure_group(root, "minflux/datasets")
    _ensure_group(
        root,
        dataset_base,
        {
            "dataset_id": dataset_id,
            "name": dataset_name,
            "source_path": source_path,
        },
    )

    raw_schema = _write_structured_table(
        root,
        f"{dataset_base}/raw/mfx",
        raw,
        row_name="measurement",
        component_role="canonical all-iteration MINFLUX mfx",
        progress=progress,
        progress_start=0.50,
        progress_span=0.18,
    )
    mbm_schema = _write_mbm(root, f"{dataset_base}/raw/mbm", ds)
    if progress is not None:
        progress(0.69, "Writing processed localization table")

    processed_base = f"{dataset_base}/processed/current"
    _ensure_group(
        root,
        processed_base,
        {
            "selection": {"iteration": "last", "valid_only": True},
            "coordinates_baked": {
                "unit": "nanometer",
                "rimf": True,
                "overlay_transform": True,
            },
            "filter_representation": "ftr boolean column; rows are retained",
        },
    )
    _write_array(
        root,
        f"{processed_base}/position",
        position,
        dimension_names=["localization", "coordinate"],
        attributes={
            "unit": "nanometer",
            "coordinate_labels": ["x", "y", "z"],
        },
        chunk_progress=(
            (
                lambda done, total: progress(
                    0.69 + 0.04 * done / max(total, 1),
                    "Writing processed positions",
                )
            )
            if progress is not None
            else None
        ),
    )
    _write_array(
        root,
        f"{processed_base}/source_row_id",
        source_row_id,
        dimension_names=["localization"],
        attributes={"references": f"{dataset_base}/raw/mfx measurement rows"},
        chunk_progress=(
            (
                lambda done, total: progress(
                    0.73 + 0.02 * done / max(total, 1),
                    "Writing raw-row references",
                )
            )
            if progress is not None
            else None
        ),
    )
    processed_schema = [
        {
            "name": "position",
            "path": f"{processed_base}/position",
            "dtype": str(position.dtype),
            "shape": list(position.shape),
            "unit": "nanometer",
        },
        {
            "name": "source_row_id",
            "path": f"{processed_base}/source_row_id",
            "dtype": str(source_row_id.dtype),
            "shape": list(source_row_id.shape),
        },
    ]
    processed_count = max(len(processed), 1)
    for processed_index, (name, value) in enumerate(processed.items(), start=1):
        array = np.asarray(value)
        if array.ndim != 1 or array.size != n_processed:
            raise ValueError(
                f"Processed attribute {name!r} does not align with localization rows."
            )
        path = f"{processed_base}/{_component(name)}"
        _write_array(
            root,
            path,
            array,
            dimension_names=["localization"],
            attributes={"source_attribute": name},
            chunk_progress=(
                (
                    lambda done, total, i=processed_index, field=name: progress(
                        0.75
                        + 0.12
                        * ((i - 1) + done / max(total, 1))
                        / processed_count,
                        f"Writing processed attribute {field}",
                    )
                )
                if progress is not None
                else None
            ),
        )
        processed_schema.append(
            {
                "name": name,
                "path": path,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            }
        )

    if progress is not None:
        progress(0.88, "Writing filters, ROIs, and view state")
    roi_dicts = [_record_dict(record) for record in roi_records]
    state_base = f"{dataset_base}/state"
    _ensure_group(root, state_base)
    _write_json(
        root / state_base / "filters.json",
        {
            "version": 1,
            "specifications": ds.state.get("filter_specs") or [],
            "mask_column": f"{processed_base}/ftr" if "ftr" in processed else None,
        },
    )
    _write_json(root / state_base / "view.json", _serializable_view_state(ds, warnings))
    _write_json(root / state_base / "rois.json", {"version": 1, "rois": roi_dicts})

    roi_mask_index = []
    _ensure_group(root, f"{state_base}/roi_masks")
    roi_mask_metadata = ds.state.get("roi_masks") or {}
    for roi in roi_dicts:
        mask_key = str(roi.get("mask_key") or "")
        if not mask_key:
            continue
        mask = ds.derived.get(mask_key)
        if mask is None:
            warnings.append(
                f"ROI {roi.get('name') or roi.get('id')} references missing mask {mask_key!r}."
            )
            continue
        mask = np.asarray(mask, dtype=bool).ravel()
        if mask.size != n_processed:
            warnings.append(
                f"ROI mask {mask_key!r} has {mask.size} rows, not {n_processed}; "
                "its geometry was exported but its mask was not."
            )
            continue
        mask_path = f"{state_base}/roi_masks/{_component(str(roi['id']))}"
        _write_array(
            root,
            mask_path,
            mask,
            dimension_names=["localization"],
            attributes={"roi_id": roi["id"], "mask_key": mask_key},
        )
        roi_mask_index.append(
            {
                "roi_id": roi["id"],
                "mask_key": mask_key,
                "path": mask_path,
                "metadata": roi_mask_metadata.get(mask_key),
            }
        )
    _write_json(root / state_base / "roi_masks" / "index.json", roi_mask_index)

    if progress is not None:
        progress(0.93, "Writing provenance and source metadata")
    provenance_base = f"{dataset_base}/provenance"
    _ensure_group(root, provenance_base)
    recipe = build_metadata(ds, data_filename=None, content="raw")
    _write_json(root / provenance_base / "recipe.json", recipe)
    journal = [_journal_dict(entry) for entry in journal_entries]
    _write_json(
        root / provenance_base / "events.json",
        {"version": 1, "events": journal},
    )
    _ensure_group(root, f"{dataset_base}/metadata")
    _write_json(
        root / dataset_base / "metadata" / "source.json",
        {
            "name": dataset_name,
            "file": {
                "name": getattr(getattr(ds, "file", None), "name", ""),
                "folder": getattr(getattr(ds, "file", None), "folder", ""),
                "datetime": getattr(getattr(ds, "file", None), "datetime", ""),
            },
            "dataset_metadata": ds.metadata,
            "properties": {
                "num_localizations": int(ds.prop.num_loc),
                "num_iterations": int(ds.prop.num_itr),
                "num_dimensions": int(ds.prop.num_dim),
                "num_traces": int(ds.prop.num_traces),
            },
            "calibration": {
                "rimf": float(ds.cali.RIMF),
                "pixel_size_nm": float(ds.cali.pixel_size),
                "localization_precision_nm": ds.cali.loc_precision,
            },
            "dataset_index_at_export": dataset_idx,
        },
    )

    manifest = {
        "schema": PROFILE_NAME,
        "schema_version": PROFILE_VERSION,
        "ome_ngff_version": OME_NGFF_VERSION,
        "zarr_format": ZARR_FORMAT,
        "created": datetime.now(timezone.utc).isoformat(),
        "software": {
            "name": "MINFLUX Viewer",
            "version": __version__,
        },
        "dataset": {
            "id": dataset_id,
            "name": dataset_name,
            "path": dataset_base,
        },
        "render": {
            "kind": image_kind,
            "source": f"{processed_base}/position",
            "filter": f"{processed_base}/ftr" if "ftr" in processed else None,
            "requested_pixel_size_nm": float(pixel_size_nm),
            "z_voxel_depth_nm": float(z_voxel_nm) if grid.is_3d else None,
            "axis_order": ["z", "y", "x"] if grid.is_3d else ["y", "x"],
            "origin_nm": {"x": x0_nm, "y": y0_nm, "z": z0_nm},
            "levels": len(grid.level_shapes),
            "level_shapes": [list(shape) for shape in grid.level_shapes],
        },
        "content": {
            "raw_mfx": raw_schema,
            "raw_mbm": mbm_schema,
            "processed": processed_schema,
            "filters": f"{state_base}/filters.json",
            "view_state": f"{state_base}/view.json",
            "rois": f"{state_base}/rois.json",
            "roi_masks": roi_mask_index,
            "processing_recipe": f"{provenance_base}/recipe.json",
            "processing_events": f"{provenance_base}/events.json",
        },
        "warnings": warnings,
    }
    _write_json(root / "minflux" / "manifest.json", manifest)
    if progress is not None:
        progress(0.97, "Finalizing OME-Zarr package")

    crate_parts = [
        {"@id": f"./{level}", "@type": "File", "name": f"Density pyramid level {level}"}
        for level in range(len(grid.level_shapes))
    ]
    crate_parts.append(
        {
            "@id": f"./{dataset_base}/",
            "@type": "Dataset",
            "name": "MINFLUX localization extension",
        }
    )
    _write_json(
        root / "ro-crate-metadata.json",
        {
            "@context": "https://w3id.org/ro/crate/1.1/context",
            "@graph": [
                {
                    "@id": "ro-crate-metadata.json",
                    "@type": "CreativeWork",
                    "about": {"@id": "./"},
                    "conformsTo": {"@id": "https://w3id.org/ro/crate/1.1"},
                },
                {
                    "@id": "./",
                    "@type": "Dataset",
                    "name": f"{dataset_name} OME-NGFF export",
                    "datePublished": datetime.now(timezone.utc).isoformat(),
                    "hasPart": [{"@id": item["@id"]} for item in crate_parts],
                    "mentions": {"@id": "#minflux-viewer"},
                },
                *crate_parts,
                {
                    "@id": "#minflux-viewer",
                    "@type": "SoftwareApplication",
                    "name": "MINFLUX Viewer",
                    "softwareVersion": __version__,
                },
            ],
        },
    )
    (root / "_SUCCESS").write_text("", encoding="ascii")
    if progress is not None:
        progress(1.0, "OME-Zarr export complete")
    return OmeZarrExportResult(
        path=root,
        effective_pixel_size_nm=float(pixel_size_nm),
        levels=len(grid.level_shapes),
        is_3d=grid.is_3d,
        image_shape=grid.level_shapes[0],
        voxel_size_nm=grid.voxel_size_nm,
        warnings=tuple(warnings),
    )


def _rename_directory(source: Path, target: Path) -> None:
    """Tolerate brief Windows scanner/indexer locks on a completed directory."""
    for attempt in range(6):
        try:
            source.rename(target)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))


def write_ome_zarr(
    ds,
    path: str | Path,
    *,
    pixel_size_nm: float,
    z_voxel_nm: float | None = None,
    max_levels: int = 6,
    dataset_idx: int | None = None,
    roi_records: Iterable[Any] = (),
    journal_entries: Iterable[Any] = (),
    overwrite: bool = False,
    progress: ProgressCallback | None = None,
) -> OmeZarrExportResult:
    """Write an OME-NGFF 0.5 / Zarr v3 package transactionally."""
    output = normalize_ome_zarr_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists.")

    token = uuid.uuid4().hex
    temporary = output.parent / f".{output.name}.tmp-{token}"
    backup = output.parent / f".{output.name}.backup-{token}"
    try:
        result = _write_package(
            temporary,
            ds,
            pixel_size_nm=pixel_size_nm,
            z_voxel_nm=z_voxel_nm,
            max_levels=max_levels,
            dataset_idx=dataset_idx,
            roi_records=roi_records,
            journal_entries=journal_entries,
            progress=progress,
        )
        if output.exists():
            _rename_directory(output, backup)
        try:
            _rename_directory(temporary, output)
        except Exception:
            if backup.exists() and not output.exists():
                _rename_directory(backup, output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return OmeZarrExportResult(
            path=output,
            effective_pixel_size_nm=result.effective_pixel_size_nm,
            levels=result.levels,
            is_3d=result.is_3d,
            image_shape=result.image_shape,
            voxel_size_nm=result.voxel_size_nm,
            warnings=result.warnings,
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
