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


# Note: the iteration selector is stored under "iteration", NOT "itr" — "itr"
# is a raw MINFLUX data-column name (see MINFLUX_DATA_KEYS below), so a filter
# row keyed "itr" would be misclassified as a data row and rejected.
FILTER_KEYS = {"apply", "attribute", "value_as", "min", "max", "iteration"}
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


# ---------------------------------------------------------------------------
# Preset rows  <->  internal filter specs
# ---------------------------------------------------------------------------
# The preset file and the dataset speak different key names for the same thing:
# a row is the saved JSON schema above, while ``ds.state["filter_specs"]`` uses
# the internal spec keys the evaluators take. The translation used to live only
# inside the Filter dialog's table, so nothing headless could read a preset; it
# belongs here, beside the schema it translates.
def filter_specs_from_rows(rows, *, only_enabled: bool = True) -> list[dict[str, Any]]:
    """Preset JSON rows -> internal filter specs.

    ``only_enabled`` keeps just the rows whose ``apply`` is set, matching what the
    Filter dialog applies (it skips unticked rows). A missing ``iteration`` stays
    missing rather than being guessed, so ``resolve_spec_iteration`` can apply its
    own backward-compatible default (effective for cfr/efc, else last).
    """
    specs: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        if only_enabled and not bool(row.get("apply", False)):
            continue
        attribute = str(row.get("attribute", "") or "")
        if not attribute:
            continue
        spec: dict[str, Any] = {
            "attribute": attribute,
            "mode": str(row.get("value_as", "per loc") or "per loc"),
            "lo": float(row.get("min", 0.0)),
            "hi": float(row.get("max", 1.0)),
            "lo_inc": bool(row.get("min_inclusive", True)),
            "hi_inc": bool(row.get("max_inclusive", True)),
        }
        if row.get("iteration", None) is not None:
            spec["itr"] = row["iteration"]
        specs.append(spec)
    return specs


def rows_from_filter_specs(specs, *, apply: bool = True) -> list[dict[str, Any]]:
    """Internal filter specs -> preset JSON rows (the inverse of the above)."""
    rows: list[dict[str, Any]] = []
    for spec in specs or []:
        if not isinstance(spec, dict):
            continue
        row: dict[str, Any] = {
            "apply": bool(apply),
            "attribute": str(spec.get("attribute", "") or ""),
            "value_as": str(spec.get("mode", "per loc") or "per loc"),
            "min": float(spec.get("lo", 0.0)),
            "max": float(spec.get("hi", 1.0)),
            "min_inclusive": bool(spec.get("lo_inc", True)),
            "max_inclusive": bool(spec.get("hi_inc", True)),
        }
        if spec.get("itr", None) is not None:
            row["iteration"] = spec["itr"]
        rows.append(row)
    return rows
