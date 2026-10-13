"""CRLB method reporting must distinguish model bounds from measured precision."""

import json
from types import SimpleNamespace

import pytest

from minflux_viewer.analysis import method_text as mt


def _text(message):
    return mt.generate_method_text(
        SimpleNamespace(datasets=[]),
        [{"message": message, "dataset_name": "test acquisition"}],
    )


def _axis(**updates):
    axis = {
        "status": "background_estimated",
        "background_coverage": 0.9,
        "invalid_background_fraction": 0.02,
        "geometry_source": "from the acquisition sequence",
        "pattern": "hexagon",
        "iteration": 8,
        "primary_sigma_nm": 3.41,
        "median_ideal_nm": 1.21,
        "issues": [],
    }
    axis.update(updates)
    return axis


def _message(**updates):
    provenance = {
        "schema": "crlb/v2",
        "count_convention": "unknown",
        "xy": _axis(),
        "z": None,
        "L_nm": 39.6,
        "sigma_q_nm": 131.6,
        "median_n": 136,
        "comparison": None,
        "issues": [],
    }
    provenance.update(updates)
    return (
        "Localization precision (CRLB, Marin-Ries): median σ_xy = 3.41 nm "
        "(background-limited; background from the centre exposure of iteration 7 "
        "scaled x1.50 by the laser power pooled per trace for 90% of localizations), "
        "1.21 nm (ideal), L = 39.6 nm (from the acquisition sequence), "
        "σ_q = 132 nm, median N = 136 photons (eco + ecc); crlb_provenance="
        + json.dumps(provenance, allow_nan=False)
    )


def test_structured_provenance_takes_priority_over_legacy_prefix():
    text = _text(_message())
    assert "XY status: background estimated" in text
    assert "input count/rate convention is unknown" in text
    assert "Usable background coverage: 90.0%" in text
    assert "Background ≥ orbit rate affected 2.0%" in text
    assert "Photon iteration: 9 (one-based)" in text
    assert "Geometry: from the acquisition sequence" in text
    assert "centre exposure of iteration 7 scaled x1.50" in text
    assert "not achieved experimental precision or image resolution" in text
    assert "per-coordinate RMS standard deviation" in text
    assert "not lateral radial RMS error" in text
    assert "3.41 nm" in text and "1.21 nm" in text
    assert "crlb_provenance=" not in text
    assert mt.CITE_CRLB_MARIN[1] in text


@pytest.mark.parametrize("status", ["invalid_input", "unsupported_geometry", "background_inconsistent"])
def test_refused_result_is_reported_even_when_the_human_prefix_has_nan(status):
    message = _message(xy=_axis(status=status, primary_sigma_nm=None, median_ideal_nm=None))
    text = _text(message.replace("median σ_xy = 3.41", "median σ_xy = nan"))
    assert status.replace("_", " ") in text
    assert "No interpretable σ_xy headline bound" in text
    assert "nan nm" not in text
    assert "Median σ_xy =" not in text


def test_declared_detected_counts_do_not_claim_a_vendor_validation():
    text = _text(_message(count_convention="detected"))
    assert "treats the supplied counts as detected photon counts" in text
    assert "convention is unknown" not in text
    assert "not achieved experimental precision" in text


def test_per_axis_provenance_and_octahedral_model_limit_are_retained():
    text = _text(_message(
        xy=_axis(status="ideal_only", pattern="octahedron", background_coverage=0,
                 invalid_background_fraction=0, primary_sigma_nm=6.0, median_ideal_nm=6.0),
        z=_axis(status="unsupported_geometry", pattern="zline", iteration=9,
                primary_sigma_nm=None, median_ideal_nm=None,
                issues=["Multi-radius final axial pattern is unsupported."]),
    ))
    assert "XY status: ideal only" in text
    assert "zero background is not established" in text
    assert "Z status: unsupported geometry" in text
    assert "Photon iteration: 10 (one-based)" in text
    assert "isotropic three-dimensional quadratic-field approximation" in text
    assert "Multi-radius final axial pattern is unsupported" in text


def test_matched_comparison_reports_ratio_without_a_causal_noise_budget():
    text = _text(_message(comparison={
        "xy": {"measured_nm": 2.38, "bound_nm": 2.93, "ratio": 2.38 / 2.93,
               "n_traces": 103, "n_locs": 1550, "coverage": 0.8,
               "bound_kind": "background_estimated"},
        "z": {"measured_nm": 6, "bound_nm": 5, "ratio": 1.2,
              "n_traces": 100, "n_locs": 1500, "coverage": 0.78,
              "bound_kind": "ideal_only"},
    }))
    assert "RMS of its per-localization bounds use the same localizations" in text
    assert "summaries are medians over the same traces" in text
    assert "measured spread 2.38 nm versus matched bound 2.93 nm" in text
    assert "measured/bound = 0.812" in text
    assert "103 traces and 1,550 localizations" in text
    assert "Matched input-row coverage: 80.0%" in text
    assert "unexplained by the comparison" in text
    assert "No photon-independent error term is inferred" in text
    assert "biased toward" not in text
    assert "flickering" not in text
    assert "σ_fl" not in text


def test_missing_comparison_never_invents_whole_trace_precision():
    text = _text(_message(comparison=None))
    assert "Descriptive matched comparison" not in text
    assert "measured spread" not in text
    assert "σ_fl" not in text


@pytest.mark.parametrize("comparison", [None, {}])
def test_omitted_comparison_reason_is_reported(comparison):
    reason = "Tracking acquisition: whole-trace spread includes motion."
    text = _text(_message(comparison=comparison, comparison_reason=reason))
    assert f"Matched comparison omitted: {reason}" in text
    assert "Descriptive matched comparison" not in text
    assert "σ_fl" not in text


def test_duplicate_limitation_is_not_repeated():
    note = "Centre exposure convention remains unresolved."
    text = _text(_message(issues=[note], xy=_axis(issues=[note])))
    assert text.count(note) == 1


@pytest.mark.parametrize("payload", ["not json", "[]", '{"schema":"crlb/v999"}'])
def test_invalid_provenance_is_not_silently_reinterpreted_as_a_legacy_result(payload):
    message = _message().split("; crlb_provenance=", 1)[0] + "; crlb_provenance=" + payload
    text = _text(message)
    assert "provenance is invalid or uses an unsupported schema" in text
    assert "3.41 nm" not in text


@pytest.mark.parametrize("measured", [2.0, 9.5])
def test_historical_excess_is_not_promoted_to_an_identified_noise_source(measured):
    message = (
        "Localization precision (CRLB, Marin-Ries): median σ_xy = 5.20 nm "
        "(background-limited), 3.10 nm (ideal), L = 50 nm, σ_q = 250 nm, "
        f"median N = 200 photons; measured σ_r = {measured:.2f} nm (StdDev/trace) → "
        "excess σ_fl = 7.95 nm."
    )
    text = _text(message)
    assert "former quadrature subtraction is not a validated physical precision budget" in text
    assert "does not identify a noise source or estimator bias" in text
    assert "attributable to" not in text
    assert "biased toward" not in text
    assert mt.CITE_SIMUFLUX[0] not in text
    if measured < 5.2:
        assert "below-bound spread remains unexplained" in text


def test_historical_background_is_an_estimate_not_a_verified_constant():
    text = _text(_message().split("; crlb_provenance=", 1)[0])
    assert "centre exposure was used as a background estimate" in text
    assert "assuming sufficiently stable background" in text
    assert "this assumption was not established" in text
    assert "background being constant" not in text
    assert "records no fluorescence signal" not in text
