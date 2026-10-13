"""Input validity, matched statistics, and fresh-load CRLB interpretation."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from minflux_viewer.analysis import localization_precision as lp
from minflux_viewer.core.mfx_sequence import PatternGeometry


def test_contradictory_background_never_becomes_ideal_headline():
    r = lp.crlb_precision(np.array([100.]), efo=np.array([100.]), fbg=np.array([110.]))
    assert r.axis_status == "background_inconsistent"
    assert r.invalid_background_fraction == 1
    assert np.isnan(r.primary_sigma)
    assert np.isfinite(r.median_sigma_ideal)  # explicitly hypothetical benchmark
    assert "no background information" not in lp.crlb_log_line(r)


def test_contradictory_centre_is_not_silently_replaced_by_lower_fbg():
    r = lp.crlb_precision(np.array([100.]), efo=np.array([100.]),
                          efc=np.array([110.]), fbg=np.array([10.]))
    assert r.axis_status == "background_inconsistent"


def test_measured_fbg_zero_enters_pooling_but_unknown_zero_does_not():
    kwargs = dict(eco=np.full(3, 100.), efo=np.full(3, 100.), fbg=np.array([0., 0., 30.]),
                  groups=np.zeros(3))
    unknown = lp.crlb_precision(**kwargs)
    measured = lp.crlb_precision(**kwargs, fbg_zero_measured=True)
    assert unknown.background_coverage == pytest.approx(1 / 3)
    assert measured.background_coverage == 1
    assert measured.primary_sigma == pytest.approx(measured.median_sigma_ideal)
    assert measured.axis_status == "background_estimated"


@pytest.mark.parametrize("key", ["efo", "fbg", "efc", "ecc", "eco_z", "groups"])
def test_misaligned_arrays_cannot_fall_back_to_other_inputs(key):
    with pytest.raises(ValueError, match="row-aligned"):
        lp.crlb_precision(np.full(3, 100.), L_z_nm=40., **{key: np.ones(2)})


def test_negative_centre_bookkeeping_is_disclosed_without_reconstructing_counts():
    r = lp.crlb_precision(np.array([100., 200.]), ecc=np.array([-1., -2.]))
    np.testing.assert_array_equal(r.per_loc_n_total, [100., 200.])
    assert r.count_convention == "unknown"
    assert any("Negative ecc" in note for note in r.issues)


def test_xy_and_z_statuses_are_independent():
    r = lp.crlb_precision(np.full(3, 100.), efo=np.full(3, 100.), fbg=np.full(3, 110.),
                          L_z_nm=40., eco_z=np.full(3, 200.),
                          efo_z=np.full(3, 100.), fbg_z=np.full(3, 10.))
    assert np.isnan(r.primary_sigma)
    assert np.isfinite(r.primary_sigma_z)
    assert r.axis_status_z == "background_estimated"


def test_unsupported_geometry_does_not_reappear_after_user_changes_l():
    inputs = lp.crlb_inputs_from_arrays(np.full(8, 100.), num_dim=3)
    inputs["geometry_z"] = PatternGeometry(L_nm=40., pattern="zline", multi_radius=True,
                                            from_sequence=True)
    with np.errstate(invalid="ignore"):
        result = lp.crlb_from_inputs(inputs, L_z_nm=50.)
    assert result.has_z and result.axis_status_z == "unsupported_geometry"
    assert np.isnan(result.primary_sigma_z)
    assert np.isfinite(result.primary_sigma)


def test_matched_comparison_uses_identical_finite_rows_and_trace_weights():
    # Unequal trace sizes distinguish the equal-trace median from a per-row median.
    tid = np.r_[np.zeros(6), np.ones(10), np.full(4, 2)]
    x = np.r_[np.arange(6), np.arange(10), np.arange(4)].astype(float)
    inputs = lp.crlb_inputs_from_arrays(np.r_[np.full(6, 100.), np.full(14, 400.)], tid=tid)
    inputs["comparison_loc_xyz"] = np.column_stack([x, 2 * x, np.zeros(20)]) * 1e-9
    inputs["comparison_tid"] = tid
    inputs["xy"]["eco"][1] = 0  # drop from BOTH measurements and bounds
    result = lp.crlb_from_inputs(inputs)
    matched = lp.crlb_matched_comparison(result, inputs)["xy"]
    selected = result.source_indices
    trace_measured, trace_bound = [], []
    for trace in (0, 1):
        mask = tid[selected] == trace
        pts = inputs["comparison_loc_xyz"][selected[mask]] * 1e9
        trace_measured.append(np.sqrt(np.mean(np.var(pts[:, :2], axis=0, ddof=1))))
        trace_bound.append(np.sqrt(np.mean(result.per_loc_sigma_ideal[mask] ** 2)))
    assert matched["measured_nm"] == pytest.approx(np.median(trace_measured))
    assert matched["bound_nm"] == pytest.approx(np.median(trace_bound))
    assert matched["n_traces"] == 2 and matched["n_locs"] == 15
    assert matched["coverage"] == pytest.approx(.75)
    assert matched["ratio"] == pytest.approx(np.median(trace_measured) / np.median(trace_bound))


def test_background_subset_is_also_applied_to_measured_spread():
    inputs = lp.crlb_inputs_from_arrays(np.full(6, 100.), efo=np.full(6, 100.),
                                       fbg=np.array([10., 10., 10., 10., 10., np.nan]))
    inputs["comparison_loc_xyz"] = np.column_stack([np.r_[np.arange(5), 1e6], np.zeros((6, 2))]) * 1e-9
    inputs["comparison_tid"] = np.zeros(6)
    r = lp.crlb_from_inputs(inputs)
    matched = lp.crlb_matched_comparison(r, inputs)["xy"]
    assert matched["measured_nm"] == pytest.approx(np.std(np.arange(5), ddof=1) / np.sqrt(2))
    assert matched["n_locs"] == 5
    payload = json.loads(lp.crlb_log_line(r, {"xy": matched}).split("; crlb_provenance=")[1])
    assert payload["schema"] == "crlb/v2" and payload["comparison"]["xy"] == matched


_SAMPLE = Path(r"D:\Workspace\Microscopes\MINFLUX\sample data")
_TRACKING = ["1_sample_A_1-100_seq_3D_ori_exc_5.msr",
             "rotar_project_confocal_overlay/1_C5C3_PEG_surface_1-25k_overview_"
             "FLUX_seq_Tracking_2D_Rot_faster_L75nm_final_final_high_salt_plus_fuel_1.msr"]


@pytest.mark.parametrize("name", _TRACKING)
def test_fresh_tracking_load_has_no_static_comparison(name):
    path = _SAMPLE / name
    if not path.exists():
        pytest.skip("external sample absent")
    from minflux_viewer.core.app_state import DEFAULT_PREFS
    from minflux_viewer.core.tracks import dataset_tracking_role
    from minflux_viewer.msr.quick_open import load_viewer_msr
    ds = load_viewer_msr(path, prefs=copy.deepcopy(DEFAULT_PREFS))[0]
    assert dataset_tracking_role(ds)[0] is None
    inputs = lp.crlb_inputs_for_dataset(ds)
    result = lp.crlb_from_inputs(inputs)
    assert inputs["comparison_role"] == "Tracking"
    assert lp.crlb_matched_comparison(result, inputs) is None
    assert lp._measured_trace_precision(ds) is None
    assert "σ_fl" not in lp.crlb_log_line(result)
    assert dataset_tracking_role(ds)[0] is None  # read-only gate, no persistent role change
    if name.startswith("1_sample"):
        assert result.background_coverage == 1.0  # measured fbg zeros retained


def test_explicit_structure_role_overrides_conservative_tracking_hint():
    from types import SimpleNamespace
    ds = SimpleNamespace(state={"tracking_role": "Structure"},
                         metadata={"mfx_sequence": {"id": "Tracking", "Itr": [{}]}})
    assert lp._precision_role(ds)[0] == "Structure"
