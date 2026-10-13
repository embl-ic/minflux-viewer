"""CRLB reports distinguish a conditional model from measured precision."""

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QGroupBox, QLabel, QScrollArea

from minflux_viewer.analysis import localization_precision as lp


def _case(**kwargs):
    inputs = lp.crlb_inputs_from_arrays(np.full(12, 100.0), L_nm=40.0, **kwargs)
    return lp.crlb_from_inputs(inputs), inputs


def _text(dialog):
    return "\n".join(
        [label.text() for label in dialog.findChildren(QLabel)]
        + [group.title() for group in dialog.findChildren(QGroupBox)]
    )


def _open(qtbot, result, inputs, **kwargs):
    dialog = lp.CRLBResultDialog(result, inputs, **kwargs)
    qtbot.addWidget(dialog)
    return dialog


def _matched(measured_nm=2.0, bound_nm=3.0):
    return {"xy": {
        "measured_nm": measured_nm, "bound_nm": bound_nm,
        "ratio": measured_nm / bound_nm, "n_traces": 2, "n_locs": 12,
        "coverage": 1.0, "bound_kind": "ideal_only",
    }}


def test_unknown_counts_are_conditional_not_measured_precision(qtbot):
    result, inputs = _case()
    result = replace(result, count_convention="unknown")
    dialog = _open(qtbot, result, inputs)
    text = _text(dialog)
    assert "Conditional model bound" in text
    assert "Input count convention unknown" in text
    assert "not a measured localization precision or image resolution" in text
    assert "per-coordinate RMS" in text
    assert "not a radial displacement uncertainty" in text
    assert "CRLB precision (median" not in text
    assert "250" not in dialog._sq_spin.toolTip()


def test_background_at_or_above_orbit_is_not_missing_background(qtbot):
    result, inputs = _case(efo=np.full(12, 100.0), fbg=np.full(12, 110.0))
    result = replace(result, axis_status="background_inconsistent",
                     invalid_background_fraction=1.0)
    dialog = _open(qtbot, result, inputs)
    text = _text(dialog)
    assert "background ≥ orbit rate" in text
    assert "finite bound unavailable" in text
    assert "no background information" not in text
    assert "No usable background estimate for lateral" not in text
    assert "100.0%" in text
    assert "Histogram report could not be created" not in text


def test_unsupported_and_empty_result_keeps_diagnostics_and_controls(qtbot):
    result, inputs = _case()
    result = replace(result, n_locs=0, median_sigma_ideal=float("nan"),
                     mean_sigma_ideal=float("nan"), axis_status="unsupported_geometry",
                     issues=("Final multi-radius zline is unsupported.",),
                     per_loc_sigma_ideal=np.empty(0), per_loc_n_total=np.empty(0),
                     status="No supported model for the selected pattern.")
    dialog = _open(qtbot, result, inputs)
    text = _text(dialog)
    assert "unsupported geometry" in text
    assert "Final multi-radius zline is unsupported." in text
    assert "Conditional model bound" in text
    assert "unavailable" in text
    assert "Histogram report could not be created" not in text
    assert dialog._l_spin is not None


@pytest.mark.parametrize("measured_nm", [2.0, 4.0])
def test_matched_comparison_has_no_error_attribution(qtbot, measured_nm):
    result, inputs = _case()
    dialog = _open(qtbot, result, inputs, measured=_matched(measured_nm))
    text = _text(dialog)
    assert "Matched trace comparison" in text
    assert "median trace RMS bound" in text
    assert "measured / bound" in text
    assert "2 traces; 12 locs; 100.0% coverage" in text
    assert "same coordinates and localizations" in text
    assert "photon-limited performance" in text  # explicitly *not* established
    for forbidden in ("σ<sub>fl</sub>", "Precision budget", "not reducible",
                      "biased toward", "flickering", "excess σ"):
        assert forbidden not in text
    assert ("unexplained disagreement" in text) == (measured_nm < 3.0)


def test_tracking_comparison_omission_is_visible(qtbot):
    result, inputs = _case()
    result = replace(result, provenance={"comparison_reason": "tracking experiment"})
    dialog = _open(qtbot, result, inputs)
    assert "Trace comparison omitted: tracking experiment" in _text(dialog)
    assert "Matched trace comparison" not in _text(dialog)


def test_octahedron_states_isotropic_approximation(qtbot):
    result, inputs = _case(num_dim=3)
    result = replace(result, pattern="octahedron", donut_envelope=False,
                     geometry_factor=np.sqrt(3.0), geometry_factor_z=np.sqrt(3.0))
    dialog = _open(qtbot, result, inputs)
    assert "isotropic quadratic-minimum approximation" in _text(dialog)


def test_recompute_refreshes_matched_values_instead_of_reusing_them(qtbot, monkeypatch):
    result, inputs = _case()
    dialog = _open(qtbot, result, inputs, measured=_matched())
    updated = _matched(4.0, 5.0)
    calls = []

    def compare(recomputed, source):
        calls.append((recomputed, source))
        return updated

    monkeypatch.setattr(lp, "crlb_matched_comparison", compare, raising=False)
    dialog._l_spin.setValue(50.0)
    dialog._recompute()
    assert len(calls) == 1
    assert calls[0][1] is inputs
    assert calls[0][0].L_nm == 50.0
    assert dialog._measured is updated
    assert "5.00 nm" in _text(dialog)
    assert "recomputed @ L = 50.0" in dialog._status_lbl.text()


def test_expanded_3d_report_fits_default_width_and_scrolls_vertically(qtbot):
    result, inputs = _case(num_dim=3)
    comparison = _matched()
    comparison["xy"].update(n_traces=4105, n_locs=178753, coverage=0.9779)
    comparison["z"] = dict(comparison["xy"])
    dialog = _open(qtbot, result, inputs, measured=comparison)
    dialog.show()
    dialog.resize(1120, 720)
    scroll = dialog.findChild(QScrollArea)
    qtbot.waitUntil(lambda: scroll.verticalScrollBar().maximum() > 0)
    assert scroll.horizontalScrollBar().maximum() == 0
    assert "Histogram report could not be created" not in _text(dialog)

