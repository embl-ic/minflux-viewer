"""Fast, value-aware size estimates for MINFLUX export formats.

The source ``.msr`` size is not a useful predictor: it is a compressed OBF
container and may also contain images.  Estimates here start from the parsed
structured arrays and sample their *actual values*.  Binary sizes are exact or
compression-sampled; text sizes are extrapolated from the same serialization
rules used by :mod:`minflux_viewer.core.save`.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExportSizeEstimate:
    """Estimated aggregate bytes and how strongly that number is supported."""

    bytes: int
    confidence: str
    note: str = ""


def format_file_size(n_bytes: int | float) -> str:
    """Format bytes with IEC units (MiB/GiB), matching filesystem-scale use."""
    value = max(0.0, float(n_bytes))
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit = units[0]
    for candidate in units:
        unit = candidate
        if value < 1024.0 or candidate == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{int(round(value))} {unit}"
    return f"{value:.2f} {unit}"


def _structured(array) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 1 or arr.dtype.names is None:
        raise ValueError("Export size estimation requires a 1-D structured array.")
    return arr


def _point_sample(array: np.ndarray, max_rows: int = 8192) -> np.ndarray:
    if array.size <= max_rows:
        return np.asarray(array)
    indices = np.linspace(0, array.size - 1, max_rows, dtype=np.int64)
    return np.asarray(array[indices])


def _block_samples(
    array: np.ndarray,
    *,
    blocks: int = 8,
    rows_per_block: int = 65_536,
) -> list[np.ndarray]:
    if array.size <= rows_per_block:
        return [np.asarray(array)]
    count = min(blocks, max(1, array.size // rows_per_block))
    starts = np.linspace(
        0, max(0, array.size - rows_per_block), count, dtype=np.int64
    )
    return [
        np.asarray(array[int(start): int(start) + rows_per_block])
        for start in starts
    ]


def _plain_record(row, names: tuple[str, ...]) -> dict:
    record = {}
    for key in names:
        value = row[key]
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, np.generic):
            value = value.item()
        record[key] = value
    return record


def _estimate_json(array: np.ndarray) -> int:
    if array.size == 0:
        return 2
    sample = _point_sample(array)
    records = [_plain_record(row, array.dtype.names) for row in sample]
    encoded = json.dumps(
        records, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    # The sample includes its brackets and inter-record commas.  Extrapolating
    # the bracket-free body gives sub-percent error on large canonical exports.
    body_per_row = max(0, len(encoded) - 2) / sample.size
    return int(round(2 + body_per_row * array.size))


def _csv_layout(array: np.ndarray):
    from .save import flatten_mfx_array

    columns = flatten_mfx_array(array)
    names = list(columns)
    formats = [
        "%d"
        if (
            np.issubdtype(np.asarray(columns[name]).dtype, np.integer)
            or np.issubdtype(np.asarray(columns[name]).dtype, np.bool_)
        )
        else "%.17g"
        for name in names
    ]
    return columns, names, formats


def _estimate_csv(array: np.ndarray) -> int:
    sample = _point_sample(array)
    columns, names, formats = _csv_layout(sample)
    header = (",".join(names) + "\n").encode("utf-8")
    if sample.size == 0:
        return len(header)
    matrix = np.column_stack(
        [np.asarray(columns[name]).ravel() for name in names]
    )
    stream = io.StringIO()
    np.savetxt(stream, matrix, delimiter=",", fmt=formats)
    body_per_row = len(stream.getvalue().encode("utf-8")) / sample.size
    return int(round(len(header) + body_per_row * array.size))


def _npy_header_bytes(array: np.ndarray) -> int:
    header = {
        "descr": np.lib.format.dtype_to_descr(array.dtype),
        "fortran_order": False,
        "shape": array.shape,
    }
    stream = io.BytesIO()
    try:
        np.lib.format.write_array_header_1_0(stream, header)
    except ValueError:
        stream = io.BytesIO()
        np.lib.format.write_array_header_2_0(stream, header)
    return len(stream.getvalue())


def _estimate_mat(array: np.ndarray, *, root: str) -> int:
    from scipy.io import savemat

    if array.size == 0:
        stream = io.BytesIO()
        savemat(
            stream,
            {root: {name: array[name] for name in array.dtype.names}},
            do_compression=True,
            long_field_names=True,
        )
        return len(stream.getvalue())

    ratios = []
    for block in _block_samples(array):
        stream = io.BytesIO()
        savemat(
            stream,
            {root: {name: block[name] for name in block.dtype.names}},
            do_compression=True,
            long_field_names=True,
        )
        if block.nbytes:
            ratios.append(len(stream.getvalue()) / block.nbytes)
    ratio = float(np.mean(ratios)) if ratios else 1.0
    return int(round(array.nbytes * ratio))


def _estimate_zarr_data(array: np.ndarray) -> int:
    """Estimate compressed array bytes, excluding images and viewer metadata."""
    try:
        from numcodecs import Blosc
    except Exception:
        # Conservative fallback when the optional runtime is not importable.
        return int(round(array.nbytes * 0.25))

    codec = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
    ratios = []
    for block in _block_samples(array):
        columns, names, _formats = _csv_layout(block)
        raw_bytes = sum(np.asarray(columns[name]).nbytes for name in names)
        if not raw_bytes:
            continue
        compressed = sum(
            len(codec.encode(np.ascontiguousarray(columns[name])))
            for name in names
        )
        ratios.append(compressed / raw_bytes)
    ratio = float(np.mean(ratios)) if ratios else 0.25
    return int(round(array.nbytes * ratio))


def estimate_export_sizes(
    components: Iterable[tuple[str, np.ndarray]],
) -> dict[str, ExportSizeEstimate]:
    """Estimate aggregate output for ``(component_name, structured_array)``.

    Component names are normally ``"mfx"`` or ``"mbm"`` and only affect the
    small MATLAB root header.  Separate MFX/MBM output files are summed.
    """
    arrays = [(str(name), _structured(value)) for name, value in components]
    totals = {name: 0 for name in ("mat", "npy", "json", "csv", "zarr")}
    for component, array in arrays:
        totals["npy"] += int(array.nbytes + _npy_header_bytes(array))
        totals["json"] += _estimate_json(array)
        totals["csv"] += _estimate_csv(array)
        totals["mat"] += _estimate_mat(array, root=component)
        totals["zarr"] += _estimate_zarr_data(array)
    return {
        "npy": ExportSizeEstimate(
            totals["npy"], "exact", "uncompressed structured payload"
        ),
        "json": ExportSizeEstimate(
            totals["json"], "high", "sampled compact record text"
        ),
        "csv": ExportSizeEstimate(
            totals["csv"], "high", "sampled decimal table text"
        ),
        "mat": ExportSizeEstimate(
            totals["mat"], "medium", "sampled MATLAB compression"
        ),
        "zarr": ExportSizeEstimate(
            totals["zarr"],
            "medium",
            "compressed numeric arrays; images/search/viewer metadata excluded",
        ),
    }


def text_export_warning(
    estimates: dict[str, ExportSizeEstimate],
    formats: Iterable[str],
    *,
    threshold_bytes: int = 1 << 30,
) -> str | None:
    """Return a concise warning when a chosen text export is exceptionally large."""
    selected = []
    for fmt in ("csv", "json"):
        estimate = estimates.get(fmt)
        if fmt in set(formats) and estimate is not None and estimate.bytes >= threshold_bytes:
            selected.append(f"{fmt.upper()} approximately {format_file_size(estimate.bytes)}")
    if not selected:
        return None
    return (
        ", ".join(selected)
        + ". Text export must format every value and text import must parse every "
        "value again; this can take minutes even though I/O is streamed. MAT, "
        "NumPy or Zarr is usually a better working format."
    )
