"""Helpers for user-facing dataset attribute lists."""

from __future__ import annotations

from typing import Iterable

import numpy as np


SPATIAL_VECTOR_ATTRIBUTES = {"loc", "lnc", "ext", "xyz"}
RAW_ATTRIBUTE_NAMES = (
    "vld", "itr", "tid", "loc",
    "efo", "cfr", "dcr", "tim",
    "sta", "fnl", "bot", "eot",
    "gri", "thi", "sqi", "lnc",
    "eco", "ecc", "efc", "fbg",
)
DERIVED_ATTRIBUTE_NAMES = (
    "idx", "siz", "dst", "dur", "len", "spd", "dt", "tim_trace", "den",
)
LEGACY_DERIVED_ATTRIBUTE_ALIASES = {"nLoc": "siz"}
TRACE_WISE_ATTRIBUTE_NAMES = {"siz", "dur", "len"}


def enabled_attribute_names(prefs: dict) -> set[str]:
    """Return raw attribute names enabled in Preferences > Attributes."""
    configured = set((prefs.get("attributes", {}) or {}).get("enabled", []) or [])
    return configured & set(RAW_ATTRIBUTE_NAMES)


def enabled_computed_attribute_names(prefs: dict) -> set[str]:
    """Return computed attribute names enabled in Preferences > Attributes."""
    configured = (prefs.get("attributes", {}) or {}).get("computed", None)
    if configured is None:
        return set(DERIVED_ATTRIBUTE_NAMES)
    return {LEGACY_DERIVED_ATTRIBUTE_ALIASES.get(name, name) for name in (configured or [])}


def is_trace_wise_attribute(name: str) -> bool:
    """Return True for per-track attributes expanded onto localizations."""
    return name in TRACE_WISE_ATTRIBUTE_NAMES


def plot_attribute_names(
    dataset,
    prefs: dict,
    *,
    include_derived: bool = True,
    exclude: Iterable[str] = (),
    numeric_only: bool = True,
    one_dimensional_only: bool = True,
) -> list[str]:
    """Return attributes that should appear in plot dropdowns."""
    enabled = enabled_attribute_names(prefs)
    enabled_computed = enabled_computed_attribute_names(prefs)
    exclude_set = set(exclude)
    names: list[str] = []

    for name in dataset.prop.attr_names:
        base = _base_attribute_name(name)
        if base not in enabled and not (include_derived and name in enabled_computed):
            continue
        if name in exclude_set:
            continue
        arr = np.asarray(dataset.attr[name])
        if numeric_only and arr.dtype.kind not in ("b", "i", "u", "f", "c"):
            continue
        if one_dimensional_only and arr.ndim != 1:
            continue
        names.append(name)

    return names


def _base_attribute_name(name: str) -> str:
    for base in SPATIAL_VECTOR_ATTRIBUTES:
        if name in {
            f"{base}_x",
            f"{base}_y",
            f"{base}_z",
            f"{base}_x_nm",
            f"{base}_y_nm",
            f"{base}_z_nm",
        }:
            return base
    return name
