"""Structured CRLB provenance and its backwards-readable log line."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .precision_calculation import CRLBResult

def _bound_kind(has_background: bool, source: str, coverage: float,
                pooled: bool = False, status: str = "ideal_only") -> str:
    """Either 'background-limited; background from the …' or 'no background information'."""
    if status == "background_inconsistent":
        return "background >= orbit; no finite consistent bound"
    if status in {"unsupported_geometry", "invalid_input"}:
        return status.replace("_", " ")
    if not has_background:
        return "no background information"
    kind = f"background-limited; background from the {source}"
    if pooled:
        kind += " pooled per trace"
    if coverage < 0.995:
        kind += f" for {100.0 * coverage:.0f}% of localizations"
    return kind


def crlb_log_line(result: CRLBResult, measured: dict | None = None) -> str:
    """The Log line the method-text generator parses.

    Keep it parse-safe: no commas inside the σ_z parenthesis and no nested
    parentheses inside the σ_xy one.
    """
    z_txt = ""
    if result.has_z:
        z_kind = _bound_kind(result.has_background_z, result.background_source_z,
                             result.background_coverage_z, result.background_pooled, result.axis_status_z)
        z_txt = (f", σ_z = {result.primary_sigma_z:.2f} nm "
                 f"(L_z = {result.L_z_nm:.1f} nm; {z_kind})")
    # Legacy callers may still pass unmatched sigma_r/sigma_z; never reinterpret
    # those old medians as the new matched comparison.
    comparison = (measured if isinstance(measured, dict)
                  and any(axis in measured for axis in ("xy", "z")) else None)
    budget_txt = ""
    if comparison and "xy" in comparison:
        m = comparison["xy"]
        ratio = m.get("ratio")
        budget_txt = (f"; matched trace comparison: measured σ_r = {m['measured_nm']:.2f} nm, "
                      f"RMS bound = {m['bound_nm']:.2f} nm"
                      + (f", measured/bound = {ratio:.3f}" if ratio is not None else ""))
    n_def = "eco + ecc" if result.counts_centre else "eco"
    if result.has_z and result.per_dimension_photons and np.isfinite(result.median_n_z):
        n_txt = (f"median N = {result.median_n:.0f} photons ({n_def}; final lateral "
                 f"iteration; axial N_z = {result.median_n_z:.0f})")
    else:
        n_txt = f"median N = {result.median_n:.0f} photons ({n_def})"
    xy_kind = _bound_kind(result.has_background, result.background_source,
                          result.background_coverage, result.background_pooled, result.axis_status)
    def number(value):
        return float(value) if np.isfinite(value) else None

    def axis_info(z=False):
        return {
            "status": result.axis_status_z if z else result.axis_status,
            "primary_sigma_nm": number(result.primary_sigma_z if z else result.primary_sigma),
            "median_ideal_nm": number(result.median_sigma_z_ideal if z else result.median_sigma_ideal),
            "background_coverage": result.background_coverage_z if z else result.background_coverage,
            "invalid_background_fraction": (result.invalid_background_fraction_z if z
                                            else result.invalid_background_fraction),
            "background_source": result.background_source_z if z else result.background_source,
            "geometry_source": result.L_z_source if z else result.L_source,
            "pattern": result.pattern_z if z else result.pattern,
            "iteration": result.provenance.get("z_iteration" if z else "xy_iteration"),
            "issues": list(result.issues),
        }
    payload = dict(result.provenance, schema="crlb/v2", xy=axis_info(),
                   z=axis_info(True) if result.has_z else None,
                   count_convention=result.count_convention, issues=list(result.issues),
                   comparison=comparison, L_nm=number(result.L_nm), sigma_q_nm=number(result.sigma_q_nm),
                   median_n=number(result.median_n), input_rows=result.input_rows,
                   excluded_photon_rows=result.excluded_photon_rows)
    return (
        f"Localization precision (CRLB, Marin-Ries): median σ_xy = "
        f"{result.primary_sigma:.2f} nm ({xy_kind}), "
        f"{result.median_sigma_ideal:.2f} nm (ideal){z_txt}, "
        f"L = {result.L_nm:.1f} nm ({result.L_source}), σ_q = {result.sigma_q_nm:.0f} nm, "
        f"{n_txt}{budget_txt}; crlb_provenance="
        + json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    )
