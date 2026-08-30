"""Customer HlyB/D workflow implemented against the public ``mfv`` API.

The reusable numerical engine remains in :mod:`minflux_viewer.analysis`; this
file owns the customer-specific choices and every interaction with the viewer.
It intentionally imports no Qt classes and never reaches into ``AppState``.
"""

from __future__ import annotations

import numpy as np

TITLE = "HlyB/D staged pair analysis"
ACTIVE_SCOPE = "Active dataset (infer cells)"
POOLED_SCOPE = "ROI cells from all datasets"


def _ask(ctx):
    return ctx.ui.ask(
        {
            "scope": [ACTIVE_SCOPE, POOLED_SCOPE],
            "min_loc_per_trace": 10,
            "z_scaling_factor": 0.67,
            "site_merge_nm": 4.0,
            "cell_link_nm": 180.0,
            "min_sites_per_component": 20,
            "r_max_nm": 60.0,
            "bin_nm": 0.5,
            "short_range_lo_nm": 8.0,
            "short_range_hi_nm": 25.0,
            "null_stratum_sites": 64,
            "null_replicates": 99,
            "bootstrap_replicates": 399,
            "run_sensitivity": True,
            "sensitivity_replicates": 31,
            "run_stratum_profile": True,
        },
        title=TITLE,
        descriptions={
            "scope": "Analyse the active field or pool every region ROI as one cell.",
            "z_scaling_factor": "Applied once to the raw Z coordinate in the analysis.",
            "site_merge_nm": "Hard maximum diameter for repeated-trace consolidation.",
            "cell_link_nm": "Single-link distance used to infer cells in active-dataset mode.",
            "null_replicates": "Conditional cell-surface randomizations for the main result.",
            "run_sensitivity": "Audit site, cell and null choices around the main result.",
        },
    )


def _config_values(values):
    config = {key: value for key, value in values.items() if key != "scope"}
    positive = (
        "min_loc_per_trace",
        "z_scaling_factor",
        "site_merge_nm",
        "cell_link_nm",
        "min_sites_per_component",
        "r_max_nm",
        "bin_nm",
        "short_range_hi_nm",
        "null_stratum_sites",
        "null_replicates",
        "bootstrap_replicates",
        "sensitivity_replicates",
    )
    for key in positive:
        if float(config[key]) <= 0:
            raise ValueError(f"{key} must be greater than zero.")
    if float(config["short_range_lo_nm"]) < 0:
        raise ValueError("short_range_lo_nm must not be negative.")
    if float(config["short_range_hi_nm"]) <= float(config["short_range_lo_nm"]):
        raise ValueError("short_range_hi_nm must be greater than short_range_lo_nm.")
    if float(config["r_max_nm"]) <= float(config["short_range_hi_nm"]):
        raise ValueError("r_max_nm must be greater than short_range_hi_nm.")
    return config


def _column(ctx, dataset, name):
    return np.asarray(
        ctx.data.attr(
            name,
            dataset=dataset,
            itr="last",
            vld_only=True,
            filtered=True,
        )
    ).ravel()


def _raw_columns(ctx, dataset):
    columns = [_column(ctx, dataset, name) for name in ("loc_x", "loc_y", "loc_z")]
    tid = _column(ctx, dataset, "tid")
    lengths = {values.size for values in (*columns, tid)}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 3:
        raise ValueError("The localization and trace-ID columns are not aligned.")
    try:
        tim = _column(ctx, dataset, "tim")
    except Exception:
        tim = None
    if tim is not None and tim.size != tid.size:
        tim = None
    loc_m = np.column_stack(columns).astype(float, copy=True)
    finite_z = loc_m[np.isfinite(loc_m).all(axis=1), 2]
    if finite_z.size < 3 or float(np.ptp(finite_z)) * 1e9 < 5.0:
        raise ValueError("The selected data does not contain genuine 3-D localizations.")
    return loc_m, np.array(tid, copy=True), None if tim is None else np.array(tim, copy=True)


def _capture_active(ctx):
    dataset = ctx.data.active()
    if dataset is None:
        raise ValueError("Open a dataset first, then run the analysis again.")
    properties = ctx.data.properties(dataset=dataset)
    loc_m, tid, tim = _raw_columns(ctx, dataset)
    request = {
        "scope": "active",
        "label": properties["name"],
        "loc_m": loc_m,
        "tid": tid,
        "tim": tim,
    }
    return request, dataset


def _capture_pooled(ctx):
    cells = []
    for dataset in ctx.data.datasets():
        properties = ctx.data.properties(dataset=dataset)
        rois = ctx.roi.list(dataset=dataset, region_only=True)
        if not rois:
            continue
        loc_m, tid, tim = _raw_columns(ctx, dataset)
        for position, roi in enumerate(rois, start=1):
            inside = np.asarray(
                ctx.roi.mask(roi, dataset=dataset, filtered=True), dtype=bool
            ).ravel()
            if inside.size != loc_m.shape[0]:
                raise ValueError(
                    f"ROI mask in {properties['name']!r} does not align with its data."
                )
            roi_name = str(getattr(roi, "name", "") or f"ROI {position}")
            cells.append(
                {
                    "loc_m": np.array(loc_m[inside], copy=True),
                    "tid": np.array(tid[inside], copy=True),
                    "tim": None if tim is None else np.array(tim[inside], copy=True),
                    "label": f"{properties['name']} / {roi_name}",
                    "dataset": properties["name"],
                    "roi": roi_name,
                }
            )
    if not cells:
        raise ValueError(
            "No region ROIs were found. Draw or restore a region ROI around each cell."
        )
    return {"scope": "pooled", "label": f"{len(cells)} ROI cell(s)", "cells": cells}, None


def _analyse(task, request, config):
    from minflux_viewer.analysis.hlyb_staged import (
        Staged3DConfig,
        analyze_hlyb_staged_3d,
        analyze_hlyb_staged_pooled,
    )

    task.progress(0, 1, "Preparing HlyB/D analysis")
    cfg = Staged3DConfig(**config)
    if request["scope"] == "pooled":
        result = analyze_hlyb_staged_pooled(request["cells"], cfg)
    else:
        result = analyze_hlyb_staged_3d(
            request["loc_m"], request["tid"], request["tim"], cfg
        )
    task.progress(1, 1, "HlyB/D analysis complete")
    return result


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _result_rows(label, result):
    summary = result["summary"]
    rows = [
        {
            "scope": label,
            "variant": "main",
            "n_traces": int(result["n_traces_used"]),
            "n_sites": int(result["n_sites"]),
            "n_components": int(result["n_components"]),
            "band_ratio": _number(summary.get("band_ratio")),
            "band_ratio_z": _number(summary.get("band_ratio_z")),
            "band_p": _number(summary.get("band_p")),
            "peak_nm": _number(summary.get("peak_nm")),
            "excess_centroid_nm": _number(summary.get("positive_excess_centroid_nm")),
            "robust_calibrated": str(
                result.get("robust_short_range_excess_calibrated")
            ),
        }
    ]
    for index, row in enumerate(result.get("sensitivity") or [], start=1):
        rows.append(
            {
                "scope": label,
                "variant": f"sensitivity {index}: {row.get('source', '')}",
                "n_traces": int(result["n_traces_used"]),
                "n_sites": int(row.get("n_sites", 0)),
                "n_components": int(row.get("n_components", 0)),
                "band_ratio": _number(row.get("band_ratio")),
                "band_ratio_z": _number(row.get("band_ratio_z")),
                "band_p": _number(row.get("band_p")),
                "peak_nm": _number(row.get("peak_nm")),
                "excess_centroid_nm": _number(
                    row.get("positive_excess_centroid_nm")
                ),
                "robust_calibrated": "",
            }
        )
    return rows


def _present(ctx, request, dataset, config, result):
    label = request["label"]
    ctx.results.table("HlyB/D pair analysis").add_rows(
        _result_rows(label, result)
    ).show()

    ctx.plot.line(
        result["centers_nm"],
        result["observed"],
        label="observed",
        title=f"HlyB/D pair profile — {label}",
    ).series(
        result["centers_nm"], result["null_mean"], label="conditional null"
    ).labels(
        x="pair distance (nm)", y="pair count per bin"
    ).legend().grid().show()

    sites = np.asarray(result.get("site_centers_nm", []), dtype=float)
    if sites.ndim == 2 and sites.shape[0] and sites.shape[1] >= 2:
        ctx.plot.scatter(
            sites[:, 0],
            sites[:, 1],
            color_by=np.asarray(result.get("component_labels", []), dtype=float),
            title=f"HlyB/D inferred sites — {label}",
            size=4.0,
        ).labels(x="x (nm)", y="y (nm)").grid().show()

    summary = result["summary"]
    ctx.journal.record(
        "analysis",
        "Ran the HlyB/D staged short-range pair analysis",
        dataset=dataset,
        scope=request["scope"],
        min_loc_per_trace=int(config["min_loc_per_trace"]),
        z_scaling_factor=float(config["z_scaling_factor"]),
        site_merge_nm=float(config["site_merge_nm"]),
        cell_link_nm=float(config["cell_link_nm"]),
        short_range_nm=(
            float(config["short_range_lo_nm"]),
            float(config["short_range_hi_nm"]),
        ),
        null_stratum_sites=int(config["null_stratum_sites"]),
        null_replicates=int(config["null_replicates"]),
        band_ratio=_number(summary.get("band_ratio")),
        band_ratio_z=_number(summary.get("band_ratio_z")),
        excess_centroid_nm=_number(summary.get("positive_excess_centroid_nm")),
    )
    ctx.ui.log(
        f"HlyB/D analysis on {label!r}: {result['n_sites']} inferred sites; "
        f"observed/null ratio {_number(summary.get('band_ratio')):.3f}."
    )


def run(ctx):
    """Collect parameters and data, then submit the numerical work."""
    values = _ask(ctx)
    if values is None:
        return None
    try:
        config = _config_values(values)
        if values["scope"] == POOLED_SCOPE:
            request, dataset = _capture_pooled(ctx)
        else:
            request, dataset = _capture_active(ctx)
    except Exception as exc:
        ctx.ui.error(str(exc), title=TITLE)
        return None

    def work(task):
        return _analyse(task, request, config)

    def done(result):
        _present(ctx, request, dataset, config, result)

    def failed(exc):
        ctx.ui.error(
            f"The HlyB/D analysis failed: {exc}",
            title=TITLE,
        )

    return ctx.run.background(
        work,
        on_done=done,
        on_error=failed,
        name="HlyB/D staged pair analysis",
        kind="plugin",
    )
