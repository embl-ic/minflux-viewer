"""Launch path for the HlyB/D subunit pair analysis plugin.

Pulls the raw last-valid localizations of the active dataset, collects
parameters, runs :func:`minflux_viewer.analysis.hlyb_staged.analyze_hlyb_staged_3d`
and opens the modeless result window.  The Log line it emits carries a
``method_data`` payload consumed by *Plugins › Generate Method Text*.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox

TITLE = "HlyB/D Subunit Pair Analysis"

# This project's MINFLUX measurements are acquired with a fixed refractive-index
# mismatch factor, so the dialog opens on it rather than on the dataset's own
# (auto-estimated) RIMF.  The user can still override it per run.
PROJECT_RIMF = 0.67


def _localization_columns(ds):
    """Raw last-valid ``loc``/``tid``/``tim`` columns of *ds*, in metres."""
    from ...core.loader import mfx_get

    def column(attr):
        value = mfx_get(ds, attr, itr="last", vld_only=True)
        return None if value is None else np.asarray(value, dtype=float).ravel()

    lx, ly, lz, tid = (column("loc_x"), column("loc_y"),
                       column("loc_z"), column("tid"))
    finite_z = np.empty(0) if lz is None else lz[np.isfinite(lz)]
    if (lx is None or ly is None or lz is None or tid is None or lx.size < 3
            or lz.size != lx.size or finite_z.size < 3
            or np.ptp(finite_z) * 1e9 < 5.0):
        return None
    tim = column("tim")
    if tim is not None and tim.size != lx.size:
        tim = None
    return np.column_stack([lx, ly, lz]), tid, tim


def _method_payload(ds, cfg, result, *, n_localizations, has_time):
    """Serialize the run for *Generate Method Text*."""
    summary = result["summary"]
    sensitivity = [
        {key: row[key] for key in (
            "source", "site_merge_nm", "cell_link_nm", "null_stratum_sites",
            "n_sites", "n_sites_used", "n_components", "band_ratio", "band_p",
            "band_ratio_z", "peak_nm", "positive_excess_centroid_nm")}
        for row in (result.get("sensitivity") or [])
    ]
    return {
        "schema": "hlyb_staged_short_range_3d/v1",
        "input": {
            "dataset_name": ds.name,
            "source_path": str(
                ds.metadata.get("msr_source_path")
                or getattr(ds.file, "recent_path", None)
                or getattr(ds.file, "path", "") or ""),
            "n_localizations": int(n_localizations),
            "n_traces_total": int(result["n_traces_total"]),
            "n_traces_used": int(result["n_traces_used"]),
            "time_column_available": bool(has_time),
            "iteration_selector": "last",
            "valid_only": True,
        },
        "parameters": {
            "min_loc_per_trace": int(cfg.min_loc_per_trace),
            "z_scaling_factor": float(cfg.z_scaling_factor),
            "site_merge_nm": float(cfg.site_merge_nm),
            "cell_link_nm": float(cfg.cell_link_nm),
            "min_sites_per_component": int(cfg.min_sites_per_component),
            "r_max_nm": float(cfg.r_max_nm),
            "bin_nm": float(cfg.bin_nm),
            "short_range_lo_nm": float(cfg.short_range_lo_nm),
            "short_range_hi_nm": float(cfg.short_range_hi_nm),
            "null_stratum_sites": int(cfg.null_stratum_sites),
            "null_replicates": int(cfg.null_replicates),
            "bootstrap_replicates": int(cfg.bootstrap_replicates),
        },
        "site_inference": {
            "n_sites": int(result["n_sites"]),
            "n_sites_used": int(result["n_sites_used"]),
            "n_repeated_sites": int(result["n_repeated_sites"]),
            "n_traces_consolidated": int(result["n_traces_consolidated"]),
            "median_within_site_rms_nm": float(
                result["median_within_site_rms_nm"]),
        },
        "components": {
            "n_retained": int(result["n_components"]),
            "n_all": int(result["n_components_all"]),
            "n_rod_like": int(result["n_rod_like_components"]),
            "n_excluded_sites": int(result["n_excluded_sites"]),
        },
        "result": {
            key: float(summary[key]) for key in (
                "band_observed_pairs", "band_null_mean_pairs",
                "band_null_sd_pairs", "band_ratio", "band_z", "band_p",
                "band_p_resolution", "band_ratio_z",
                "null_band_ratio_mean", "null_band_ratio_sd",
                "peak_nm", "positive_excess_centroid_nm",
                "positive_excess_median_nm", "max_pointwise_z",
                "max_pointwise_p")
        },
        "robust_short_range_excess": result.get("robust_short_range_excess"),
        "robust_short_range_excess_calibrated": result.get(
            "robust_short_range_excess_calibrated"),
        "sensitivity_passes": int(result.get("sensitivity_passes", 0)),
        "sensitivity_calibrated_passes": int(
            result.get("sensitivity_calibrated_passes", 0)),
        "sensitivity_valid_variants": int(
            result.get("sensitivity_valid_variants", 0)),
        "centroid_sensitivity_range_nm": result.get(
            "centroid_sensitivity_range_nm"),
        "bootstrap": result.get("bootstrap", {}),
        "sensitivity": sensitivity,
        "stratum_profile": result.get("stratum_profile"),
        "calibrated_ratio_z": float(cfg.calibrated_ratio_z),
        "limitations": list(result.get("limitations") or []),
    }


def _log_line(ds, cfg, result) -> str:
    summary = result["summary"]
    sensitivity = result.get("sensitivity") or []
    ratios = [float(row["band_ratio"]) for row in sensitivity
              if np.isfinite(row.get("band_ratio", np.nan))]
    centroids = [float(row["positive_excess_centroid_nm"]) for row in sensitivity
                 if np.isfinite(row.get("positive_excess_centroid_nm", np.nan))]
    extra = ""
    if ratios and centroids:
        extra = (f"; sensitivity ratio {min(ratios):.2f}–{max(ratios):.2f}, "
                 f"excess centroid {min(centroids):.2f}–{max(centroids):.2f} nm")
    robust = result.get("robust_short_range_excess_calibrated")
    robustness = "PASS" if robust is True else "FAIL" if robust is False else "not run"
    return (
        f"HlyB/D subunit pair analysis on '{ds.name}': "
        f"{result['n_traces_used']:,}/{result['n_traces_total']:,} trace(s) → "
        f"{result['n_sites']:,} inferred site(s) "
        f"({result['n_sites_used']:,} in {result['n_components']} component(s)); "
        f"{cfg.short_range_lo_nm:g}–{cfg.short_range_hi_nm:g} nm observed/null "
        f"{summary['band_ratio']:.3f} ({summary['band_ratio_z']:.1f}σ vs null), "
        f"positive-excess centroid {summary['positive_excess_centroid_nm']:.2f} nm "
        f"(same-site diameter {cfg.site_merge_nm:g} nm, surface-null stratum "
        f"{cfg.null_stratum_sites} sites, {cfg.null_replicates} replicate(s), "
        f"z-scale {cfg.z_scaling_factor:g}, sensitivity {robustness}){extra}."
    )


def run_hlyb_pair_analysis(state, parent=None) -> None:
    """Plugins › HlyB/D subunit pair analysis."""
    from ...analysis.hlyb_staged import Staged3DConfig, analyze_hlyb_staged_3d
    from ...ui.hlyb_staged_dialog import HlyBStagedDialog, HlyBStagedWindow
    from ...ui.modeless import show_modeless

    idx = state.active_idx
    if idx is None or state.active_dataset is None:
        QMessageBox.information(
            parent, TITLE, "Open a dataset first — this analysis runs on the "
                           "active dataset.")
        return
    ds = state.datasets[idx]

    columns = _localization_columns(ds)
    if columns is None:
        QMessageBox.information(
            parent, TITLE,
            "The active dataset does not contain sufficient genuine 3-D "
            "localizations with trace IDs.")
        return
    loc_m, tid, tim = columns

    defaults = getattr(state, "_hlyb_staged_cfg", None)
    if defaults is None:
        defaults = Staged3DConfig(z_scaling_factor=PROJECT_RIMF)
    else:
        defaults = Staged3DConfig(
            **{**vars(defaults), "z_scaling_factor": PROJECT_RIMF})
    dlg = HlyBStagedDialog(parent, defaults=defaults)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return
    cfg = dlg.config()
    state._hlyb_staged_cfg = cfg

    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        result = analyze_hlyb_staged_3d(loc_m, tid, tim, cfg)
    except Exception as exc:                                  # noqa: BLE001
        QMessageBox.warning(parent, TITLE, f"Analysis failed: {exc}")
        return
    finally:
        if QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()

    win = HlyBStagedWindow(result, title=ds.name, owner=parent, prefs=state.prefs)
    show_modeless(win, parent)

    state.log(
        _log_line(ds, cfg, result),
        dataset_idx=idx,
        method_data=_method_payload(
            ds, cfg, result,
            n_localizations=loc_m.shape[0], has_time=tim is not None),
    )
