"""
minflux_viewer.core.filter_io
=============================
Small JSON IO layer for saved filter presets.

UI widgets can call these helpers from buttons, drag-and-drop, or main-window
file routing without duplicating schema checks or JSON read/write details.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FILTER_KEYS = {"apply", "attribute", "value_as", "min", "max"}
MINFLUX_DATA_KEYS = {"loc", "lnc", "ext", "tid", "tim", "itr", "vld", "efo", "cfr", "dcr"}


class FilterIO:
    """Namespace for filter preset read/write helpers."""

    @staticmethod
    def load_filter_json(path: str | Path) -> list[dict[str, Any]]:
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        if not is_filter_json_payload(payload):
            raise ValueError(f"'{path.name}' is not a MINFLUX Viewer filter preset.")
        return list(payload)

    @staticmethod
    def save_filter_json(path: str | Path, rows: list[dict[str, Any]]) -> None:
        path = Path(path)
        if not is_filter_json_payload(rows):
            raise ValueError("Filter rows do not match the filter preset JSON schema.")
        with path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)


def load_filter_json(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate a filter preset JSON file."""
    return FilterIO.load_filter_json(path)


def save_filter_json(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Save validated filter rows to JSON."""
    FilterIO.save_filter_json(path, rows)


def is_filter_json_file(path: str | Path) -> bool:
    """Return True only for JSON files shaped like saved filter rows."""
    try:
        load_filter_json(path)
    except Exception:
        return False
    return True


def is_filter_json_payload(payload: Any) -> bool:
    """Validate the lightweight filter-preset JSON schema."""
    if not isinstance(payload, list):
        return False
    if not payload:
        return True
    for row in payload:
        if not isinstance(row, dict):
            return False
        keys = set(row.keys())
        if keys & MINFLUX_DATA_KEYS:
            return False
        if not keys & FILTER_KEYS:
            return False
    return True
