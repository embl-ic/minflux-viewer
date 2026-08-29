"""Worker-safe post-load computations.

This module contains no Qt calls. The GUI submits
:func:`compute_post_load_results` to a shared pool, then applies the returned
arrays atomically on Qt's thread after confirming the dataset is still open —
so nothing here writes a *result* onto the dataset.

⚠ It is **not** free of side effects on the dataset, and claiming otherwise
would hide the one place that matters: reading through ``mfx_get`` /
``mfx_filter_mask`` populates the lazily cached ``loc_id`` column of
``ds.components.mfx_raw`` (``loader._raw_loc_id``). That write is idempotent,
is a pure function of the store, and is published under a lock, so the GUI
thread may compute it concurrently without the two disagreeing. Any *new*
worker-side cache must meet the same three conditions or be pre-computed on the
GUI thread before the task is submitted.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np


def _materialized_points_nm(ds, z_factor: float) -> np.ndarray:
    from .loader import attr_values_1d

    x = attr_values_1d(ds, "loc_x")
    y = attr_values_1d(ds, "loc_y")
    z = attr_values_1d(ds, "loc_z")
    if x is None or y is None:
        return np.empty((0, 3), dtype=float)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.zeros_like(x) if z is None else np.asarray(z, dtype=float).ravel()
    n = min(x.size, y.size, z.size)
    return np.column_stack(
        [x[:n] * 1e9, y[:n] * 1e9, z[:n] * 1e9 * float(z_factor)]
    )


def _last_valid_points_nm(ds, z_factor: float) -> np.ndarray:
    from .loader import mfx_get

    x = mfx_get(ds, "loc_x", itr="last")
    y = mfx_get(ds, "loc_y", itr="last")
    z = mfx_get(ds, "loc_z", itr="last")
    if x is None or y is None:
        return np.empty((0, 3), dtype=float)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.zeros_like(x) if z is None else np.asarray(z, dtype=float).ravel()
    n = min(x.size, y.size, z.size)
    return np.column_stack(
        [x[:n] * 1e9, y[:n] * 1e9, z[:n] * 1e9 * float(z_factor)]
    )


def compute_post_load_results(
    ds,
    prefs: dict,
    report: Callable[[str], None],
) -> dict:
    """Compute optional derived arrays for *ds* without touching Qt.

    Returns them for the GUI thread to apply; writes no result onto *ds*.
    See the module docstring for the one cache this populates as a side
    effect of reading.
    """
    data_prefs = prefs.get("data", {})
    plot_prefs = prefs.get("plot", {})
    logs: list[tuple[str, str]] = []
    result = {
        "z_update": None,
        "loc_precision": None,
        "density": None,
        "last_density": None,
        "logs": logs,
    }

    requested_z = bool(
        data_prefs.get("estimate_z_scaling_factor", False)
        or plot_prefs.get("use_fixed_z_scaling_factor", False)
    )
    z_factor = float(getattr(ds.cali, "z_scaling_factor", 1.0) or 1.0)
    if requested_z and "z_scaling_factor" not in ds.derived:
        if ds.prop.num_dim < 3:
            z_factor = 1.0
            result["z_update"] = {
                "value": z_factor,
                "source": "2D (no z correction)",
                "details": None,
            }
            logs.append(
                (f"Z scaling factor for '{ds.name}': 1.0 (2D dataset, computation skipped).", "INFO")
            )
        elif plot_prefs.get("use_fixed_z_scaling_factor", False):
            z_factor = float(plot_prefs.get("z_scaling_factor", 0.67))
            result["z_update"] = {
                "value": z_factor,
                "source": "fixed (preference)",
                "details": None,
            }
            logs.append(
                (f"Z scaling factor for '{ds.name}': {z_factor:.4g} (fixed preference value).", "INFO")
            )
        else:
            report(f"Estimating Z scaling factor for '{ds.name}'")
            value = None
            details = None
            try:
                started = time.perf_counter()
                from ..analysis.trace_analysis import estimate_anisotropy_for_dataset

                details = estimate_anisotropy_for_dataset(ds)
                elapsed = time.perf_counter() - started
                if details is not None and np.isfinite(details["z_scaling_factor"]):
                    value = float(details["z_scaling_factor"])
                if value is not None and 0.5 <= value <= 1.0:
                    z_factor = value
                    result["z_update"] = {
                        "value": value,
                        "source": "estimated (trace anisotropy)",
                        "details": details,
                    }
                    logs.append(
                        (f"Estimated Z scaling factor for '{ds.name}': {value:.4g} "
                         f"(trace anisotropy, raw last-valid z, {elapsed:.1f}s)", "INFO")
                    )
                else:
                    z_factor = 1.0
                    result["z_update"] = {
                        "value": z_factor,
                        "source": "estimate out of range (reset to 1.0)",
                        "details": None,
                    }
                    shown = f"{value:.4g}" if value is not None else "failed"
                    logs.append(
                        (f"Z scaling factor for '{ds.name}': estimate {shown} outside "
                         "[0.5, 1.0] — reset to 1.0.", "WARN")
                    )
            except Exception as exc:  # noqa: BLE001 - optional computation
                z_factor = 1.0
                result["z_update"] = {
                    "value": z_factor,
                    "source": "estimate failed (reset to 1.0)",
                    "details": None,
                }
                logs.append(
                    (f"Z scaling factor estimation failed for '{ds.name}': {exc}", "WARN")
                )

    if (
        data_prefs.get("compute_loc_prec", True)
        and "sigma_per_trace_nm" not in ds.derived
        and not ds.metadata.get("particle_average")
    ):
        report(f"Computing localization precision of '{ds.name}'")
        try:
            started = time.perf_counter()
            from ..analysis.localization_precision import stddev_per_trace
            from .loader import attr_values_1d

            loc_x = np.asarray(attr_values_1d(ds, "loc_x"))
            loc_y = np.asarray(attr_values_1d(ds, "loc_y"))
            raw_z = attr_values_1d(ds, "loc_z")
            loc_z = np.zeros_like(loc_x) if raw_z is None else np.asarray(raw_z)
            raw_tid = attr_values_1d(ds, "tid")
            tid = np.arange(loc_x.size) if raw_tid is None else np.asarray(raw_tid)
            precision = stddev_per_trace(
                np.column_stack([loc_x, loc_y, loc_z]), tid
            )
            elapsed = time.perf_counter() - started
            result["loc_precision"] = precision
            median = precision["median_sigma_xyz"]
            logs.append(
                (f"Computed localization precision for '{ds.name}' using StdDev per trace "
                 f"({elapsed:.1f}s): median sigma=({median[0]:.3g}, {median[1]:.3g}, "
                 f"{median[2]:.3g}) nm.", "INFO")
            )
        except Exception as exc:  # noqa: BLE001 - optional computation
            logs.append(
                (f"Localization precision computation skipped for '{ds.name}': {exc}", "WARN")
            )

    if (
        data_prefs.get("compute_local_density", True)
        and "den" not in ds.attr
        and not ds.metadata.get("particle_average")
    ):
        report(f"Computing local density of '{ds.name}'")
        try:
            started = time.perf_counter()
            from ..analysis.local_density import compute_local_density_for_points

            points = _materialized_points_nm(ds, z_factor)
            density, method, detail = compute_local_density_for_points(
                points, prefs, dimensions=3 if ds.prop.num_dim == 3 else 2
            )
            elapsed = time.perf_counter() - started
            result["density"] = density
            level = "WARN" if "auto fallback" in detail else "INFO"
            if level == "WARN":
                logs.append(
                    (f"Local density auto fallback for '{ds.name}': {detail}.", level)
                )
            logs.append(
                (f"Computed local density for '{ds.name}' using {method} "
                 f"({detail}, {elapsed:.1f}s).", "INFO")
            )
        except Exception as exc:  # noqa: BLE001 - optional computation
            logs.append(
                (f"Local density computation skipped for '{ds.name}': {exc}", "WARN")
            )

    if (
        data_prefs.get("compute_local_density", True)
        and len(ds.components.derived_last)
        and "den" not in ds.components.derived_last
        and not ds.metadata.get("particle_average")
    ):
        report(f"Computing last-valid density of '{ds.name}'")
        try:
            from ..analysis.local_density import compute_local_density_for_points

            points = _last_valid_points_nm(ds, z_factor)
            density, method, detail = compute_local_density_for_points(
                points, prefs, dimensions=3 if ds.prop.num_dim == 3 else 2
            )
            result["last_density"] = density
            if "auto fallback" in detail:
                logs.append(
                    (f"Last-valid local density auto fallback for '{ds.name}': {detail}.", "WARN")
                )
            logs.append(
                (f"Computed last-valid local density for '{ds.name}' using {method} ({detail}).", "INFO")
            )
        except Exception as exc:  # noqa: BLE001 - optional computation
            logs.append(
                (f"Last-valid local density computation skipped for '{ds.name}': {exc}", "WARN")
            )

    return result
