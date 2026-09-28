"""Materialize recurrent tracking results as auditable dataset attributes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .tracking_stats import PreparedTrajectories, TrackingResult

TRACKING_ATTRIBUTE_NAMES = (
    "msd_d",
    "msd_alpha",
    "msd_sigma_apparent",
    "step_angle",
    "track_straightness",
)

TRACKING_ATTRIBUTE_DESCRIPTIONS = {
    "msd_d": (
        "Short-lag diffusion coefficient from measured-lag MSD, expanded from "
        "each uninterrupted segment onto its localizations."
    ),
    "msd_alpha": (
        "Descriptive short-lag log-log MSD exponent, expanded from each "
        "uninterrupted segment; it is not a motion-state classification."
    ),
    "msd_sigma_apparent": (
        "Apparent localization sigma from the MSD intercept. Motion blur and "
        "MINFLUX feedback-loop dynamic error are not corrected."
    ),
    "step_angle": (
        "Turning angle in radians between consecutive displacement vectors "
        "within an uninterrupted trajectory segment."
    ),
    "track_straightness": (
        "Net displacement divided by path length for the uninterrupted segment "
        "containing this localization."
    ),
}


def _empty_rows(n_rows: int) -> np.ndarray:
    return np.full(int(n_rows), np.nan, dtype=float)


def _expanded_segment_values(
    n_rows: int,
    prepared: PreparedTrajectories,
    table: Mapping[str, np.ndarray],
    column: str,
    *,
    require_ok: bool = False,
) -> np.ndarray:
    output = _empty_rows(n_rows)
    segment_ids = np.asarray(table.get("segment_id", np.empty(0)), dtype=np.int64)
    values = np.asarray(table.get(column, np.empty(0)), dtype=float)
    reasons = np.asarray(table.get("reason", np.empty(0)), dtype=object)
    for row, segment_id in enumerate(segment_ids):
        if row >= values.size:
            break
        if require_ok and (row >= reasons.size or str(reasons[row]) != "ok"):
            continue
        source_rows = prepared.row_indices[prepared.segment_codes == int(segment_id)]
        output[source_rows] = values[row]
    return output


def tracking_attribute_arrays(
    n_rows: int,
    prepared: PreparedTrajectories,
    *,
    msd: TrackingResult | None = None,
    kinematics: TrackingResult | None = None,
    names: Iterable[str] = TRACKING_ATTRIBUTE_NAMES,
) -> dict[str, tuple[np.ndarray, str, Mapping[str, Any]]]:
    """Return row-aligned arrays, units and provenance for selected attributes."""
    requested = tuple(dict.fromkeys(str(name) for name in names))
    unknown = set(requested) - set(TRACKING_ATTRIBUTE_NAMES)
    if unknown:
        raise ValueError(f"Unknown tracking attribute(s): {', '.join(sorted(unknown))}")
    output: dict[str, tuple[np.ndarray, str, Mapping[str, Any]]] = {}

    if any(name.startswith("msd_") for name in requested):
        if msd is None:
            raise ValueError("MSD-derived attributes require an MSD result.")
        table = msd.table("segment_fit")
        unit_map = msd.units.get("segment_fit", {})
        columns = {
            "msd_d": ("diffusion", unit_map.get("diffusion", "")),
            "msd_alpha": ("alpha", "1"),
            "msd_sigma_apparent": ("apparent_sigma_nm", "nm"),
        }
        for name in requested:
            if name not in columns:
                continue
            column, unit = columns[name]
            output[name] = (
                _expanded_segment_values(
                    n_rows, prepared, table, column, require_ok=True),
                unit,
                {
                    "method_id": msd.method_id,
                    "method_version": msd.method_version,
                    "parameters": dict(msd.provenance),
                    "citations": list(msd.citations),
                },
            )

    if any(name in {"step_angle", "track_straightness"} for name in requested):
        if kinematics is None:
            raise ValueError("Kinematic attributes require a kinematics result.")
        if "step_angle" in requested:
            local = kinematics.table("localization")
            values = _empty_rows(n_rows)
            values[np.asarray(local["row_index"], dtype=np.int64)] = np.asarray(
                local["turning_angle_rad"], dtype=float)
            output["step_angle"] = (
                values,
                "rad",
                {
                    "method_id": kinematics.method_id,
                    "method_version": kinematics.method_version,
                    "parameters": dict(kinematics.provenance),
                },
            )
        if "track_straightness" in requested:
            output["track_straightness"] = (
                _expanded_segment_values(
                    n_rows,
                    prepared,
                    kinematics.table("segment"),
                    "straightness",
                ),
                "1",
                {
                    "method_id": kinematics.method_id,
                    "method_version": kinematics.method_version,
                    "parameters": dict(kinematics.provenance),
                    "scope": "uninterrupted segment",
                },
            )
    return output


def install_tracking_attributes(
    dataset,
    prepared: PreparedTrajectories,
    *,
    msd: TrackingResult | None = None,
    kinematics: TrackingResult | None = None,
    names: Iterable[str] = TRACKING_ATTRIBUTE_NAMES,
) -> tuple[str, ...]:
    """Attach selected results to a dataset on the GUI thread."""
    arrays = tracking_attribute_arrays(
        int(dataset.prop.num_loc),
        prepared,
        msd=msd,
        kinematics=kinematics,
        names=names,
    )
    installed = []
    records = dataset.metadata.setdefault("tracking_derived_attributes", {})
    for name, (values, unit, provenance) in arrays.items():
        metadata = {
            "component": "mfx",
            "source": "derived by tracking analysis",
            "description": TRACKING_ATTRIBUTE_DESCRIPTIONS[name],
            "unit": unit,
            "user_visible": True,
            "tracking_analysis": dict(provenance),
        }
        dataset.mfx.set_attr(name, values, meta=metadata)
        dataset.derived[name] = values
        if name not in dataset.prop.attr_names:
            dataset.prop.attr_names.append(name)
        records[name] = dict(provenance)
        installed.append(name)
    return tuple(installed)


__all__ = [
    "TRACKING_ATTRIBUTE_DESCRIPTIONS",
    "TRACKING_ATTRIBUTE_NAMES",
    "install_tracking_attributes",
    "tracking_attribute_arrays",
]
