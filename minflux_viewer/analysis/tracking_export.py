"""Portable export of a :class:`TrackingResult` as CSV tables plus metadata."""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .tracking_stats import TrackingResult


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._")
    return name or "table"


def _table_csv(table: dict | object) -> str:
    columns = list(table)
    arrays = [np.asarray(table[column]).reshape(-1) for column in columns]
    lengths = {array.size for array in arrays}
    if len(lengths) > 1:
        raise ValueError("Tracking result table columns are not row-aligned.")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    for row in zip(*arrays):
        writer.writerow([
            item.item() if isinstance(item, np.generic) else item
            for item in row
        ])
    return stream.getvalue()


def export_tracking_result_zip(result: TrackingResult, path: str | Path) -> Path:
    """Write every result table as CSV and the audit trail as ``metadata.json``."""
    target = Path(path)
    if target.suffix.lower() != ".zip":
        target = target.with_suffix(".zip")
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "minflux-viewer-tracking-result",
        "format_version": 1,
        "method_id": result.method_id,
        "method_version": result.method_version,
        "units": result.units,
        "diagnostics": asdict(result.diagnostics),
        "provenance": result.provenance,
        "citations": list(result.citations),
        "tables": list(result.tables),
    }
    with zipfile.ZipFile(
        target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        archive.writestr(
            "metadata.json",
            json.dumps(metadata, indent=2, ensure_ascii=False, default=_json_value)
            + "\n",
        )
        for level, table in result.tables.items():
            archive.writestr(f"{_safe_name(level)}.csv", _table_csv(table))
    return target


__all__ = ["export_tracking_result_zip"]
