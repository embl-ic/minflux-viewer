"""Built-in HlyB/D workflow implemented against the public ``mfv`` API.

The reusable numerical engine remains in :mod:`minflux_viewer.analysis`; this
file owns the customer-specific choices and every interaction with the viewer.
It intentionally imports no Qt classes and never reaches into ``AppState``.
"""

from __future__ import annotations

import numpy as np

TITLE = "HlyB/D pair distance analysis"
AUTOMATIC_MODE = "Automatic E. coli detection (3D)"
ACTIVE_ROI_MODE = "Hand-draw ROI, active dataset (2D/3D)"
MULTI_ROI_MODE = "Hand-draw ROI, multiple datasets (3D)"
AUTO_BOUNDARY_EVIDENCE = "Localizations + associated acquisition image (automatic)"
LOCALIZATION_BOUNDARY_EVIDENCE = "Localizations only"

# Compatibility names used by older recordings and focused tests. The dialog
# now presents the clearer mode names above.
ACTIVE_SCOPE = AUTOMATIC_MODE
POOLED_SCOPE = MULTI_ROI_MODE
ROI_2D_SCOPE = ACTIVE_ROI_MODE


def _ask(ctx):
    return ctx.ui.ask(
        {
            "mode": [AUTOMATIC_MODE, ACTIVE_ROI_MODE, MULTI_ROI_MODE],
            "boundary_evidence": [
                AUTO_BOUNDARY_EVIDENCE,
                LOCALIZATION_BOUNDARY_EVIDENCE,
            ],
            "min_loc_per_trace": 10,
            "z_scaling_factor": 0.67,
            "site_merge_nm": 4.0,
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
            "mode": (
                "Automatic capsule-based cell detection for an active 3-D field; "
                "all drawn cell ROIs in the active dataset; or the persistent "
                "multi-dataset 3-D ROI pool."
            ),
            "boundary_evidence": (
                "For automatic 3-D detection, use the best calibrated image "
                "that contains this acquisition ROI as weak boundary support. "
                "Generated density/loc images are recognized as derived data "
                "and are not treated as independent evidence."
            ),
            "z_scaling_factor": "Applied once in 3-D; ignored for a 2-D drawn-ROI run.",
            "site_merge_nm": "Hard maximum diameter for repeated-trace consolidation.",
            "null_replicates": "3-D surface or 2-D ROI-constrained null randomizations.",
            "run_sensitivity": "Audit site, cell and null choices around the main result.",
        },
    )


def _config_values(values):
    config = {key: value for key, value in values.items()
              if key not in ("mode", "scope", "boundary_evidence")}
    # Retained in the numerical config/result schema for replay compatibility;
    # none of the three current modes uses single-link cell segmentation.
    config.setdefault("cell_link_nm", 180.0)
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


def _raw_columns(ctx, dataset, *, require_3d=True):
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
    if (require_3d
            and (finite_z.size < 3 or float(np.ptp(finite_z)) * 1e9 < 5.0)):
        raise ValueError("The selected data does not contain genuine 3-D localizations.")
    return loc_m, np.array(tid, copy=True), None if tim is None else np.array(tim, copy=True)


def _is_genuine_3d(loc_m):
    loc_m = np.asarray(loc_m, dtype=float)
    finite = loc_m[np.isfinite(loc_m).all(axis=1)]
    return bool(finite.shape[0] >= 3 and float(np.ptp(finite[:, 2])) * 1e9 >= 5.0)


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
        "source_path": properties.get("source_path", ""),
        "dataset_did": properties.get("dataset_did", ""),
    }
    return request, dataset


def _capture_active_rois(ctx):
    """Collect every stored region ROI in the active dataset as one cell."""
    dataset = ctx.data.active()
    if dataset is None:
        raise ValueError("Open a dataset and draw a region ROI around each cell first.")
    properties = ctx.data.properties(dataset=dataset)
    rois = ctx.roi.list(dataset=dataset, region_only=True)
    if not rois:
        raise ValueError(
            "No stored region ROI was found for the active dataset. Draw one "
            "around each cell and add it to ROI Manager first.")
    loc_m, tid, tim = _raw_columns(ctx, dataset, require_3d=False)
    is_2d = not _is_genuine_3d(loc_m)
    cells = []
    for position, roi in enumerate(rois, start=1):
        context = getattr(roi, "context", {}) or {}
        source_view = context.get("source_view", getattr(roi, "source_view", ""))
        if source_view and source_view not in ("render", "scatter"):
            continue
        inside = np.asarray(
            ctx.roi.mask(roi, dataset=dataset, filtered=True), dtype=bool).ravel()
        if inside.size != loc_m.shape[0]:
            raise ValueError(
                f"ROI mask in {properties['name']!r} does not align with its data.")
        if int(np.count_nonzero(inside)) < 3:
            continue
        roi_name = str(getattr(roi, "name", "") or f"ROI {position}")
        if is_2d:
            points_nm = np.asarray(
                ctx.roi.points_in(
                    roi, dataset=dataset, filtered=True, unit="nm"), dtype=float)
            if points_nm.ndim != 2 or points_nm.shape[0] != int(inside.sum()):
                raise ValueError(
                    f"ROI coordinates for {roi_name!r} do not align with its data.")
            selected_loc = np.array(points_nm[:, :2] / 1e9, copy=True)
        else:
            selected_loc = np.array(loc_m[inside], copy=True)
        cell = {
            "loc_m": selected_loc,
            "tid": np.array(tid[inside], copy=True),
            "tim": None if tim is None else np.array(tim[inside], copy=True),
            "label": f"{properties['name']} / {roi_name}",
            "dataset": properties["name"],
            "roi": roi_name,
        }
        if is_2d:
            cell["roi_geometry"] = {
                "type": roi.type, "geometry": ctx.roi.geometry(roi)}
        cells.append(cell)
    if not cells:
        raise ValueError("None of the active dataset's spatial ROIs contains enough data.")
    return {
        "scope": "active_roi",
        "label": f"{properties['name']} / {len(cells)} ROI cell(s)",
        "cells": cells,
        "is_2d": is_2d,
    }, dataset


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


def _capture_roi_2d(ctx):
    dataset = ctx.data.active()
    if dataset is None:
        raise ValueError("Open a 2-D dataset and draw a region ROI first.")
    properties = ctx.data.properties(dataset=dataset)
    rois = ctx.roi.list(dataset=dataset, region_only=True)
    active = ctx.roi.active(dataset=dataset)
    if active is not None and any(roi.id == active.id for roi in rois):
        roi = active
    else:
        raise ValueError(
            "Select one stored region ROI for the active dataset in ROI Manager. "
            "An unfinished drawing is not yet available to the plugin.")
    context = getattr(roi, "context", {}) or {}
    source_view = context.get("source_view", getattr(roi, "source_view", ""))
    if source_view and source_view not in ("render", "scatter"):
        raise ValueError("The selected ROI must be a spatial render/scatter ROI.")
    inside = np.asarray(ctx.roi.mask(roi, dataset=dataset, filtered=True), dtype=bool).ravel()
    tid = _column(ctx, dataset, "tid")
    if inside.size != tid.size:
        raise ValueError("The ROI mask does not align with the filtered trace IDs.")
    points_nm = np.asarray(
        ctx.roi.points_in(roi, dataset=dataset, filtered=True, unit="nm"), dtype=float)
    if points_nm.ndim != 2 or points_nm.shape[0] != int(inside.sum()) or points_nm.shape[1] < 2:
        raise ValueError("The ROI coordinates do not align with the selected localizations.")
    if points_nm.shape[0] < 3:
        raise ValueError("The selected ROI contains fewer than three localizations.")
    try:
        tim = _column(ctx, dataset, "tim")
    except Exception:
        tim = None
    if tim is not None and tim.size != inside.size:
        tim = None
    name = str(getattr(roi, "name", "") or "selected ROI")
    cell = {
        "loc_m": np.array(points_nm[:, :2] / 1e9, copy=True),
        "tid": np.array(tid[inside], copy=True),
        "tim": None if tim is None else np.array(tim[inside], copy=True),
        "label": f"{properties['name']} / {name}",
        "dataset": properties["name"],
        "roi": name,
        "roi_geometry": {"type": roi.type, "geometry": ctx.roi.geometry(roi)},
    }
    return {"scope": "roi2d", "label": cell["label"], "cells": [cell]}, dataset


def _analyse(task, request, config):
    from minflux_viewer.analysis.hlyb_staged import (
        Staged3DConfig,
        analyze_hlyb_staged_3d,
        analyze_hlyb_staged_pooled,
    )

    task.progress(0, 1, "Preparing HlyB/D analysis")
    cfg = Staged3DConfig(**config)
    if request["scope"] in ("pooled", "roi2d", "active_roi"):
        result = analyze_hlyb_staged_pooled(
            request["cells"], cfg,
            is_2d=bool(request.get("is_2d", request["scope"] == "roi2d")))
    else:
        result = analyze_hlyb_staged_3d(
            request["loc_m"], request["tid"], request["tim"], cfg,
            associated_image_source=(
                {
                    "source_path": request.get("source_path", ""),
                    "dataset_did": request.get("dataset_did", ""),
                }
                if request.get("use_associated_image") else None
            ),
        )
    task.progress(1, 1, "HlyB/D analysis complete")
    return result


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _result_rows(label, result, peaks):
    summary = result["summary"]
    rows = [
        {
            "scope": label,
            "dimension": "2D" if result.get("is_2d") else "3D",
            "null_model": result.get("null_model", "surface_conditioned_3d"),
            "variant": "main",
            "n_traces": int(result["n_traces_used"]),
            "n_sites": int(result["n_sites"]),
            "n_components": int(result["n_components"]),
            "band_ratio": _number(summary.get("band_ratio")),
            "band_ratio_z": _number(summary.get("band_ratio_z")),
            "band_p": _number(summary.get("band_p")),
            "binning_persistent_excess_peak_nm": (
                float(peaks[0]["distance_nm"]) if peaks else float("nan")
            ),
            "binning_persistent_excess_peaks_nm": ", ".join(
                f"{peak['distance_nm']:.2f}" for peak in sorted(
                    peaks, key=lambda item: item["distance_nm"])
            ),
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
                "dimension": "2D" if result.get("is_2d") else "3D",
                "null_model": result.get("null_model", "surface_conditioned_3d"),
                "variant": f"sensitivity {index}: {row.get('source', '')}",
                "n_traces": int(result["n_traces_used"]),
                "n_sites": int(row.get("n_sites", 0)),
                "n_components": int(row.get("n_components", 0)),
                "band_ratio": _number(row.get("band_ratio")),
                "band_ratio_z": _number(row.get("band_ratio_z")),
                "band_p": _number(row.get("band_p")),
                "binning_persistent_excess_peak_nm": float("nan"),
                "binning_persistent_excess_peaks_nm": "",
                "excess_centroid_nm": _number(
                    row.get("positive_excess_centroid_nm")
                ),
                "robust_calibrated": "",
            }
        )
    return rows


def _open_result_window(ctx, result, label):
    """Show the viewer's own staged result window.

    A generic script scatter of the site centres cannot carry the interactive
    part: the pair links are chosen by dragging a band on the pair-distance
    profile, so the histogram and the site view have to live in one window.
    That window already exists in the application, so it is reused rather than
    reimplemented here.
    """
    from minflux_viewer.ui.hlyb_staged_dialog import HlyBStagedWindow
    from minflux_viewer.ui.modeless import show_modeless

    owner = ctx.require_main_window()
    window = HlyBStagedWindow(result, title=label, owner=owner)
    show_modeless(window, owner)
    return window


def _open_collection(ctx, config):
    """Open the persistent multi-dataset ROI pool owned by the main window."""
    from minflux_viewer.analysis.hlyb_staged import Staged3DConfig

    owner = ctx.require_main_window()
    opener = getattr(owner, "show_hlyb_cell_collection", None)
    if not callable(opener):
        raise RuntimeError(
            "This viewer build does not expose the HlyB/D ROI collection window.")
    return opener(Staged3DConfig(**config))


def _present(ctx, request, dataset, config, result):
    from minflux_viewer.analysis.hlyb_staged import (
        persistent_excess_peaks,
        surface_normalized_null_envelope,
        surface_normalized_profile,
    )

    label = request["label"]
    is_2d = bool(result.get("is_2d", False))
    null_label = "ROI-conditioned 2-D null" if is_2d else "conditional surface null"
    observed = np.asarray(result["observed"], dtype=float)
    null_mean = np.asarray(result["null_mean"], dtype=float)
    edges = np.asarray(result["edges_nm"], dtype=float)
    search_hi = min(
        float(config["r_max_nm"]),
        max(40.0, float(config["short_range_hi_nm"]) + 15.0),
    )
    peaks = persistent_excess_peaks(
        observed, result["null_profiles"], edges,
        lo_nm=float(config["short_range_lo_nm"]), hi_nm=search_hi,
    )
    normalized = surface_normalized_profile(observed, null_mean, edges)
    null_lower, null_upper = surface_normalized_null_envelope(
        result["null_profiles"], edges)
    ctx.results.table("HlyB/D pair analysis").add_rows(
        _result_rows(label, result, peaks)
    ).show()

    ctx.plot.line(
        result["centers_nm"],
        result["observed"],
        label="observed",
        title=f"HlyB/D {'2-D ROI' if is_2d else '3-D'} pair profile — {label}",
    ).series(
        result["centers_nm"], result["null_mean"], label=null_label
    ).labels(
        x="pair distance (nm)", y="pair count per bin"
    ).legend().grid().show()

    ctx.plot.line(
        result["centers_nm"], normalized,
        label=f"observed / {null_label} (1.5 nm smoothing)",
        title=f"HlyB/D all-pair null-normalized — {label}",
    ).series(
        result["centers_nm"], null_lower,
        label="null 95% lower (pointwise)", color="#888888", style="--",
    ).series(
        result["centers_nm"], null_upper,
        label="null 95% upper (pointwise)", color="#888888", style="--",
    ).series(
        result["centers_nm"], np.ones_like(normalized),
        label="conditional-null baseline", color="#222222", style=":",
    ).labels(
        x=("within-ROI pair distance (nm)" if is_2d
           else "within-cell pair distance (nm)"),
        y="observed / conditional null (1 = null)",
    ).legend().grid().show()

    _open_result_window(ctx, result, label)

    summary = result["summary"]
    ctx.journal.record(
        "analysis",
        "Ran the HlyB/D staged short-range pair analysis",
        dataset=dataset,
        scope=request.get("scope_label", request["scope"]),
        mode=request.get("scope_label", request["scope"]),
        min_loc_per_trace=int(config["min_loc_per_trace"]),
        z_scaling_factor=float(config["z_scaling_factor"]),
        site_merge_nm=float(config["site_merge_nm"]),
        cell_link_nm=float(config["cell_link_nm"]),
        component_mode=result.get("component_mode", "given"),
        cell_segmentation=result.get("cell_segmentation"),
        boundary_evidence=request.get(
            "boundary_evidence", LOCALIZATION_BOUNDARY_EVIDENCE),
        associated_image_name=(
            ((result.get("cell_segmentation") or {}).get(
                "associated_image_support") or {}).get("image_name", "")
        ),
        associated_image_agreement=(
            ((result.get("cell_segmentation") or {}).get(
                "associated_image_support") or {}).get("agreement")
        ),
        short_range_nm=(
            float(config["short_range_lo_nm"]),
            float(config["short_range_hi_nm"]),
        ),
        # The band edges and the profile grid are recorded individually as
        # well, because the declared [method] block states them one at a time
        # in prose and a slot can only read a value that was recorded.
        short_range_lo_nm=float(config["short_range_lo_nm"]),
        short_range_hi_nm=float(config["short_range_hi_nm"]),
        bin_nm=float(config["bin_nm"]),
        r_max_nm=float(config["r_max_nm"]),
        null_stratum_sites=int(config["null_stratum_sites"]),
        null_replicates=int(config["null_replicates"]),
        analysis_dimension=2 if is_2d else 3,
        null_model=result.get("null_model", "surface_conditioned_3d"),
        roi_geometry=(request["cells"][0]["roi_geometry"] if is_2d else None),
        null_moved_site_fraction=result.get("null_moved_site_fraction"),
        null_swap_acceptance_fraction=result.get("null_swap_acceptance_fraction"),
        band_ratio=_number(summary.get("band_ratio")),
        band_ratio_z=_number(summary.get("band_ratio_z")),
        binning_persistent_excess_peaks_nm=[
            float(peak["distance_nm"]) for peak in peaks],
        peak_search_nm=(float(config["short_range_lo_nm"]), search_hi),
        peak_smoothing_nm=1.5,
        peak_min_binning_support_fraction=0.75,
        peak_null_max_quantile=0.95,
        normalized_curve_smoothing_nm=1.5,
        normalized_curve_min_expected_pairs=10.0,
        excess_centroid_nm=_number(summary.get("positive_excess_centroid_nm")),
    )
    peak_text = ", ".join(f"{peak['distance_nm']:.2f}" for peak in sorted(
        peaks, key=lambda item: item["distance_nm"]))
    if float(config["bin_nm"]) > 1.0:
        peak_report = "peak audit unavailable above 1 nm source bins"
    elif peaks:
        peak_report = f"binning-persistent local excess peak(s) {peak_text} nm"
    else:
        peak_report = "no binning-persistent local excess peak in the search range"
    image_support = ((result.get("cell_segmentation") or {}).get(
        "associated_image_support") or {})
    if request.get("scope") in ("active_roi", "pooled", "roi2d"):
        boundary_report = "hand-drawn ROI cell boundaries"
    elif image_support.get("used"):
        boundary_report = (
            f"cell boundaries supported by {image_support.get('image_name', 'an acquisition image')} "
            f"({float(image_support.get('agreement', float('nan'))):.3f} agreement)"
        )
    elif request.get("use_associated_image"):
        boundary_report = "no suitable independent acquisition image; localization-only fallback"
    else:
        boundary_report = "localization-only cell boundaries"
    ctx.ui.log(
        f"HlyB/D {'2-D ROI' if is_2d else '3-D'} analysis on {label!r}: "
        f"{result['n_sites']} inferred sites; "
        f"observed/null band ratio {_number(summary.get('band_ratio')):.3f}; "
        f"{boundary_report}; {peak_report} (exploratory, not assigned dimers)."
    )


def run(ctx):
    """Collect parameters and data, then submit the numerical work."""
    values = _ask(ctx)
    if values is None:
        return None
    try:
        mode = values.get("mode", values.get("scope", AUTOMATIC_MODE))
        config = _config_values(values)
        if mode == MULTI_ROI_MODE:
            return _open_collection(ctx, config)
        if mode == ACTIVE_ROI_MODE:
            request, dataset = _capture_active_rois(ctx)
            if request["is_2d"]:
                config["z_scaling_factor"] = 1.0
        else:
            request, dataset = _capture_active(ctx)
            config["component_mode"] = "shape"
            request["use_associated_image"] = (
                values.get("boundary_evidence", AUTO_BOUNDARY_EVIDENCE)
                == AUTO_BOUNDARY_EVIDENCE
            )
            request["boundary_evidence"] = values.get(
                "boundary_evidence", AUTO_BOUNDARY_EVIDENCE)
        # The dialog's own wording, not the internal token: this is what a
        # replay feeds back into the mode dropdown, and what reads best in a
        # generated method section.
        request["scope_label"] = mode
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
        name="HlyB/D pair distance analysis",
        kind="plugin",
    )
