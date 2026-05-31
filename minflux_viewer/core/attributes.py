"""Helpers for user-facing dataset attribute lists."""

from __future__ import annotations

from typing import Iterable

import numpy as np


SPATIAL_VECTOR_ATTRIBUTES = {"loc", "lnc", "ext", "xyz"}
DERIVED_ATTRIBUTE_NAMES = ("idx", "nLoc", "tim_trace", "dt", "dst", "spd", "den")


def enabled_attribute_names(prefs: dict) -> set[str]:
    """Return raw attribute names enabled in Preferences > Attributes."""
    return set((prefs.get("attributes", {}) or {}).get("enabled", []) or [])


def enabled_computed_attribute_names(prefs: dict) -> set[str]:
    """Return computed attribute names enabled in Preferences > Attributes."""
    configured = (prefs.get("attributes", {}) or {}).get("computed", None)
    if configured is None:
        return set(DERIVED_ATTRIBUTE_NAMES)
    return set(configured or [])


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
        if numeric_only and not np.issubdtype(arr.dtype, np.number):
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
