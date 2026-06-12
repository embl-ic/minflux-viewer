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

RAW_ATTRIBUTE_DESCRIPTIONS = {
    "vld": "Valid localization flag. True means a valid localization was obtained.",
    "itr": "Iteration index. This is the localization step defined in the MINFLUX sequence that iteratively improves localization precision.",
    "tid": "Trace ID. Each burst or molecule track gets a unique ID.",
    "loc": "Calculated fluorophore coordinates in metres for each iteration. These localizations may include applied corrections.",
    "efo": "Effective frequency offset: fluorophore emission frequency on the outer EOD pattern positions.",
    "cfr": "Center frequency ratio computed as efc/efo.",
    "dcr": "Detection channel ratio used for multicolor imaging, computed as detector 1 photon count divided by detector 1 plus detector 2 photon count.",
    "tim": "Time stamp in seconds.",
    "sta": "State or abort-status code.",
    "fnl": "Boolean flag indicating a final event.",
    "bot": "Boolean flag marking the beginning of a trace.",
    "eot": "Boolean flag marking the end of a trace.",
    "gri": "Grid reference.",
    "thi": "Thread value.",
    "sqi": "Sequence index.",
    "lnc": "Final localization coordinates without applied corrections. When MBM is active, this stores the raw uncorrected coordinates.",
    "eco": "Effective counts offset: background-corrected photons counted on the outer EOD pattern positions.",
    "ecc": "Effective counts center: background-corrected photons counted at the center EOD position.",
    "efc": "Effective frequency center: fluorophore emission frequency at the center EOD position.",
    "fbg": "Estimated background emission frequency, if background correction sensing was enabled.",
    "xyz": "Spatial coordinates.",
    "ext": "Physical extension of the EOD pattern diameter in the field of view.",
}

DERIVED_ATTRIBUTE_DESCRIPTIONS = {
    "idx": "Index of localization data points.",
    "siz": "The number of localizations within a track.",
    "nLoc": "The number of localizations within a track.",
    "dst": "The distance between two adjacent localisations.",
    "dur": "The time between the first and last points of a track.",
    "len": "Full sum of distance traversed, along the full length of the track.",
    "spd": "The distance between two points, in metres, divided by time between the two points, in seconds.",
    "dt": "Time interval from the previous localization in the same track.",
    "tim_trace": "Time stamp zeroed at each track start.",
    "den": "Local density at data point.",
}

AGGREGATION_DESCRIPTIONS = {
    "per loc": "Value for individual localization.",
    "trace mean": "Mean value of all localizations in a trace.",
    "trace median": "Median value of all localizations in a trace.",
    "trace min": "Minimum value of all localizations in a trace.",
    "trace max": "Maximum value of all localizations in a trace.",
    "trace stdev": "Standard deviation of all localizations in a trace.",
    "trace range": "Range of all localizations in a trace.",
}


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


def attribute_description(name: str) -> str:
    """Return the user-facing description for a raw or derived attribute."""
    if not name:
        return ""
    if name in DERIVED_ATTRIBUTE_DESCRIPTIONS:
        return DERIVED_ATTRIBUTE_DESCRIPTIONS[name]
    base = _base_attribute_name(name)
    desc = RAW_ATTRIBUTE_DESCRIPTIONS.get(base)
    if not desc:
        return "Unknown parameter, consider checking Abberior documentation."
    axis = _axis_suffix(name, base)
    if axis:
        return f"{desc} {axis.upper()} component."
    return desc


def aggregation_description(name: str) -> str:
    """Return the user-facing description for histogram aggregation modes."""
    return AGGREGATION_DESCRIPTIONS.get(name, "")


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

    for name in _dataset_attribute_keys(dataset):
        base = _base_attribute_name(name)
        if base not in enabled and not (include_derived and name in enabled_computed):
            continue
        if name in exclude_set:
            continue
        arr = np.asarray(dataset.attr[name])
        if numeric_only and arr.dtype.kind not in ("b", "i", "u", "f", "c"):
            continue
        if one_dimensional_only and arr.ndim != 1 and not _raw_store_serves_1d(dataset, name):
            continue
        names.append(name)

    return names


def _raw_store_serves_1d(dataset, name: str) -> bool:
    """True when the raw iteration store can serve *name* as a 1-D column.

    m2205-style ``iter_load="all"`` materializations keep per-iteration
    fields 2-D in ``ds.attr``; the plot windows read those through
    ``mfx_get``/``attr_values_1d``, so they stay valid dropdown entries.
    """
    raw = getattr(dataset, "mfx_raw", None)
    if raw is None or len(raw) == 0:
        return False
    val = raw.get(name)
    return val is not None and np.asarray(val).ndim == 1


def _dataset_attribute_keys(dataset) -> list[str]:
    """Return plot candidate keys, preserving prop order and appending extras."""
    keys: list[str] = []
    seen: set[str] = set()
    for name in list(getattr(getattr(dataset, "prop", None), "attr_names", []) or []):
        if name not in seen:
            keys.append(name)
            seen.add(name)
    attr = getattr(dataset, "attr", None)
    if attr is not None:
        try:
            attr_keys = attr.keys()
        except Exception:
            attr_keys = []
        for name in attr_keys:
            if name not in seen:
                keys.append(name)
                seen.add(name)
    return keys


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


def _axis_suffix(name: str, base: str) -> str:
    for suffix in ("x", "y", "z"):
        if name in {f"{base}_{suffix}", f"{base}_{suffix}_nm"}:
            return suffix
    return ""
