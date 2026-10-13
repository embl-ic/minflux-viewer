"""Corrections on top of the conditional-CRLB rework (reviewed 2026-10-13).

Each test pins one point the review found: an unsupported axis has no finite
benchmark either, a missing ratio cannot break the Log line, every caveat is
stated once, and the SimuFLUX conventions behind L and σ_q are cited where the
report relies on them.
"""

from __future__ import annotations

import json

import numpy as np

from minflux_viewer.analysis import localization_precision as lp
from minflux_viewer.analysis import method_text as mt
from minflux_viewer.core import loader


def _method_text(line: str) -> str:
    event = {"message": line, "level": "INFO", "dataset_idx": None, "dataset_name": "ds1"}
    return mt.generate_method_text(type("S", (), {"datasets": []})(), [event])


def _payload(line: str) -> dict:
    return json.loads(line.split("; crlb_provenance=", 1)[1])


def _basic(**kwargs):
    rng = np.random.default_rng(0)
    eco = rng.poisson(300, 40).astype(float)
    return lp.crlb_precision(eco, efo=np.full(40, 2.0e4), fbg=np.full(40, 500.0), **kwargs)


# ------------------------------------------------- unsupported geometry


def test_an_unsupported_axis_has_no_finite_ideal_benchmark_either():
    """The geometry factor of an unsupported pattern is unknown, so even the
    zero-background benchmark is undefined -- not the planar-orbit value."""
    r = _basic(unsupported_xy="final pattern has two radii",
               L_z_nm=39.6, unsupported_z="unrecognised axial pattern")

    assert r.axis_status == "unsupported_geometry" and np.isnan(r.primary_sigma)
    assert np.isnan(r.median_sigma_ideal)
    assert np.isnan(r.median_sigma_z_ideal)
    payload = _payload(lp.crlb_log_line(r))
    assert payload["xy"]["median_ideal_nm"] is None
    assert payload["z"]["median_ideal_nm"] is None


# ------------------------------------------------------ the Log line


def test_a_missing_ratio_does_not_break_the_log_line():
    r = _basic()
    comparison = {"xy": {"measured_nm": 1.0, "bound_nm": 0.0, "ratio": None,
                         "n_traces": 1, "n_locs": 5, "coverage": 0.1,
                         "bound_kind": "ideal only"}}
    line = lp.crlb_log_line(r, comparison)
    assert "measured σ_r = 1.00 nm" in line
    assert "measured/bound" not in line.split("; crlb_provenance")[0]


# ------------------------------------------------ one caveat, one statement


def _borrowing_dataset():
    """Two iterations: the earlier one probes the centre, the final one does not,
    so the final iteration's background is borrowed and scaled by laser power."""
    n_loc, n_itr = 60, 2
    dt = np.dtype([("vld", "?"), ("itr", "<i4"), ("tid", "<i4"), ("tim", "<f8"),
                   ("loc", "<f8", (3,)), ("eco", "<f8"), ("ecc", "<f8"),
                   ("efo", "<f8"), ("efc", "<f8"), ("fbg", "<f8")])
    mfx = np.zeros(n_loc * n_itr, dtype=dt)
    itr = np.tile(np.arange(n_itr), n_loc)
    final = itr == n_itr - 1
    mfx["itr"], mfx["vld"] = itr, True
    mfx["tid"] = np.repeat(np.arange(n_loc) // 6, n_itr)
    mfx["tim"] = np.repeat(np.linspace(0, 10, n_loc), n_itr)
    mfx["loc"][:, :2] = np.random.default_rng(1).normal(1e-6, 3e-9, (n_loc * n_itr, 2))
    mfx["eco"] = np.where(final, 300.0, 80.0)
    mfx["efo"] = np.where(final, 2.0e5, 6.0e4)
    mfx["efc"] = np.where(final, np.nan, 2.0e4)
    mfx["ecc"] = np.where(final, 0.0, 40.0)
    ds = loader.load_from_mfx_array(mfx, "synthetic")
    ds.metadata["mfx_sequence"] = {"bgcSense": False, "ctrDwellFactor": 1, "Itr": [
        {"patGeoFactor": 0.21, "Mode": {"pattern": "square", "id": "mxl3"},
         "ccrLimit": 0.8, "pwrFactor": 4},
        {"patGeoFactor": 0.11, "Mode": {"pattern": "square", "id": "mxl3"},
         "ccrLimit": -1, "pwrFactor": 6}]}
    return ds


def test_the_earlier_centre_assumption_is_stated_once():
    inputs = lp.crlb_inputs_for_dataset(_borrowing_dataset())
    r = lp.crlb_from_inputs(inputs)
    assert r.centre_reference.startswith("iteration 1 scaled x1.50")
    assert not any("laser power" in note for note in r.background_notes)

    text = _method_text(lp.crlb_log_line(r))
    assert text.count("excitation-power scaling") == 1
    assert "Earlier centre transfer assumes background scales" not in text


def test_the_count_convention_caveat_is_stated_once():
    r = _basic(count_convention="unknown")
    assert not any("convention" in issue.lower() for issue in r.issues)
    text = _method_text(lp.crlb_log_line(r))
    assert text.count("convention") == 1


def test_negative_centre_counts_still_explain_themselves():
    """The bookkeeping issue is specific information, not a restatement."""
    rng = np.random.default_rng(0)
    eco = rng.poisson(300, 10).astype(float)
    r = lp.crlb_precision(eco, ecc=np.full(10, -5.0), efo=np.full(10, 2.0e4))
    assert r.count_convention == "unknown"
    assert any("Negative ecc" in issue for issue in r.issues)


# ------------------------------------------------ SimuFLUX conventions


def test_simuflux_is_cited_for_the_pattern_scale_and_the_donut_width():
    inputs = lp.crlb_inputs_for_dataset(_borrowing_dataset())
    r = lp.crlb_from_inputs(inputs)                    # L from the sequence, default σ_q
    text = _method_text(lp.crlb_log_line(r))
    assert mt.CITE_SIMUFLUX[0] in text
    assert "patGeoFactor × 360 nm" in text


def test_simuflux_is_not_cited_when_neither_convention_is_used():
    r = _basic(L_nm=50.0, L_source="user input", sigma_q_nm=200.0)
    text = _method_text(lp.crlb_log_line(r))
    assert mt.CITE_SIMUFLUX[0] not in text


def test_the_dialog_lists_simuflux_and_each_caveat_once(qtbot):
    from PyQt6.QtWidgets import QLabel

    inputs = lp.crlb_inputs_for_dataset(_borrowing_dataset())
    r = lp.crlb_from_inputs(inputs)
    dialog = lp.CRLBResultDialog(r, inputs)
    qtbot.addWidget(dialog)
    text = "\n".join(label.text() for label in dialog.findChildren(QLabel))
    assert "Nat. Commun." in text and "17:246" in text          # SimuFLUX reference
    assert "Earlier centre transfer assumes background scales" not in text
    assert text.count("Input count convention unknown") == 1
    assert "not fully established" not in text


def test_the_stddev_docstring_cites_the_method_text_reference():
    assert "Ostersehlt et al. 2022" in lp.stddev_per_trace.__doc__
    assert "Schmidt et al. 2022" not in lp.stddev_per_trace.__doc__
