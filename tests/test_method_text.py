"""Tests for the Methods-text generator (analysis/method_text.py).

Pure-Python rule registry — no Qt. A minimal fake ``state`` is enough: the
renderers only read ``state.datasets`` (empty here, so dataset names fall back to
each event's ``dataset_name``).
"""

from types import SimpleNamespace

import pytest

from minflux_viewer.analysis import method_text as mt


def _state():
    return SimpleNamespace(datasets=[])


def _ev(message, name="ds1", idx=0):
    return {"message": message, "dataset_idx": idx, "dataset_name": name}


# --- regexes fire on the real log strings -------------------------------------

STDDEV_MSG = ("Localization precision (StdDev per trace): combined (n-weighted) "
              "sigma_r = 5.30 nm, sigma_z = 12.00 nm over 1,234 of 2,000 traces "
              "(>=5 loc/trace, raw z).")
STDDEV_AUTO_MSG = ("Computed localization precision for 'ds1' using StdDev per trace: "
                   "median sigma=(5.3, 5.1, 12) nm.")
CRLB_Z_MSG = ("Localization precision (CRLB, Marin-Ries): median σ_xy = 5.20 nm "
              "(background-limited), 3.10 nm (ideal), σ_z = 12.34 nm (L_z = 100 nm), "
              "L = 50 nm, σ_q = 100 nm, median N = 200 photons.")
CRLB_2D_MSG = ("Localization precision (CRLB, Marin-Ries): median σ_xy = 5.20 nm "
               "(background-limited), 3.10 nm (ideal), L = 50 nm, σ_q = 100 nm, "
               "median N = 200 photons.")
FRC_MSG = ("Localization precision (FRC): resolution = 25.30 nm "
           "(1/7 threshold, per-localization, 50,000 points, pixel 2.00 nm).")
AGGREGATION_MSG = (
    "Aggregated 'raw' into 'raw (aggregated 500)': 1,340,965 -> 487,008 "
    "localizations; photon threshold = 500 photons per aggregated localization; "
    "photon iterations = [4] (fallback); position = photon-weighted centroid; "
    "timestamp mode = first; valid final localizations grouped per trace in time "
    "order; trailing remainder retained."
)
HLYB_TEMPLATE_MSG = (
    "HlyB subunit pair analysis (template matching 3D) on "
    "'260626-155951_minflux_mfx.mat': 6882 trace(s) → 2312 subunit(s) → "
    "196 HlyB structure(s); 858 pair(s), median distance 13.30 nm "
    "(unit Ø 14.8 nm, candidate edge 24.0 nm, min loc/trace 1, z-scale 0.67, "
    "template tol 5.0 nm, tested 3773 candidate(s), passed 873, "
    "overlap-rejected 677)."
)


def test_stddev_text_and_citation():
    txt = mt.generate_method_text(_state(), [_ev(STDDEV_MSG)])
    assert "localization precision of 'ds1'" in txt
    assert "5.30 nm" in txt and "12.00 nm" in txt
    # Ostersehlt citation + DOI inline (plain text)
    assert mt.CITE_STDDEV[0] in txt
    assert mt.CITE_STDDEV[1] in txt


def test_stddev_auto_message_matches():
    txt = mt.generate_method_text(_state(), [_ev(STDDEV_AUTO_MSG)])
    assert "median precision was (5.3, 5.1, 12) nm" in txt
    assert mt.CITE_STDDEV[1] in txt


def test_crlb_with_axial():
    txt = mt.generate_method_text(_state(), [_ev(CRLB_Z_MSG)])
    assert "Cramér-Rao" in txt
    assert "5.20 nm" in txt
    assert "axial precision (σ_z) of 12.34 nm" in txt
    assert mt.CITE_CRLB[1] in txt


def test_crlb_without_axial():
    txt = mt.generate_method_text(_state(), [_ev(CRLB_2D_MSG)])
    assert "5.20 nm" in txt
    assert "axial precision" not in txt  # no σ_z clause for 2-D
    assert mt.CITE_CRLB[0] in txt


CRLB_BUDGET_MSG = ("Localization precision (CRLB, Marin-Ries): median σ_xy = 5.20 nm "
                   "(background-limited), 3.10 nm (ideal), L = 50 nm, σ_q = 250 nm, "
                   "median N = 200 photons; measured σ_r = 9.50 nm (StdDev/trace) → "
                   "excess σ_fl = 7.95 nm.")


def test_crlb_precision_budget_excess_cites_simuflux():
    txt = mt.generate_method_text(_state(), [_ev(CRLB_BUDGET_MSG)])
    assert "excess error of σ_fl = 7.95 nm" in txt
    assert "σ_fl² + σ_CRB²" in txt
    assert mt.CITE_SIMUFLUX[0] in txt   # SimuFLUX cited only when the budget is present
    # the plain CRLB line (no budget) must NOT cite SimuFLUX
    assert mt.CITE_SIMUFLUX[0] not in mt.generate_method_text(_state(), [_ev(CRLB_2D_MSG)])


def test_frc_two_citations():
    txt = mt.generate_method_text(_state(), [_ev(FRC_MSG)])
    assert "Fourier ring correlation" in txt
    assert "25.30 nm" in txt
    assert mt.CITE_FRC_BANTERLE[0] in txt
    assert mt.CITE_FRC_NIEUWENHUIZEN[0] in txt
    assert mt.CITE_FRC_BANTERLE[1] in txt
    assert mt.CITE_FRC_NIEUWENHUIZEN[1] in txt


def test_aggregation_method_text_is_scientific_and_parameterized():
    txt = mt.generate_method_text(
        _state(),
        [_ev(AGGREGATION_MSG, name="raw (aggregated 500)")],
    )
    assert "valid final localizations" in txt
    assert "separately within each trace and in timestamp order" in txt
    assert "background-corrected effective counts (eco)" in txt
    assert "final-scale iteration(s) [4]" in txt
    assert "0-based raw iteration indices" in txt
    assert "reached or exceeded 500 photons per aggregated localization" in txt
    assert "completed groups could exceed the threshold" in txt
    assert "photon-weighted centroid Σ(P_i r_i)/ΣP_i" in txt
    assert "first contributing localization" in txt
    assert "final sub-threshold remainder of each trace was retained" in txt
    assert "1,340,965 contributing localizations to 487,008" in txt


def test_aggregation_method_text_describes_modern_weighted_time():
    msg = AGGREGATION_MSG.replace(
        "timestamp mode = first",
        "timestamp mode = photon_weighted",
    )
    txt = mt.generate_method_text(_state(), [_ev(msg)])
    assert "photon-count-weighted mean" in txt
    assert "modern flat-record convention" in txt


def test_hlyb_template_legacy_log_gets_scientific_method_text():
    txt = mt.generate_method_text(
        _state(), [_ev(HLYB_TEMPLATE_MSG, name="260626-155951_minflux_mfx.mat")])
    assert "valid final iteration" in txt
    assert "loc_x, loc_y and loc_z" in txt
    assert "Viewer filter masks and ROI selections are not applied" in txt
    assert "Laplacian-of-Gaussian" in txt
    assert "six-site, C3-symmetric HlyB model" in txt
    assert "3,773 template candidate(s) tested" in txt
    assert "873 passed" in txt
    assert "677 were subsequently rejected" in txt
    assert "858 unique within-structure pair distance(s)" in txt
    assert "median of 13.3 nm" in txt
    assert "not serialized in this legacy Log event" in txt
    assert "show all (remove template gating)" in txt


def test_hlyb_template_structured_provenance_documents_full_run():
    method_data = {
        "schema": "hlyb_template_matching_3d/v1",
        "input": {
            "dataset_name": "260626-155951_minflux_mfx.mat",
            "source_path": r"D:\\Temp\\260626-155951_minflux_mfx.mat",
            "source_format": "MATLAB",
            "source_version": "m2410",
            "n_dimensions": 3,
            "n_localizations": 50_000,
            "n_traces": 6882,
            "iteration_selector": "last",
            "valid_only": True,
            "filter_mask_applied": False,
            "coordinate_unit": "metres",
            "coordinate_fields": ["loc_x", "loc_y", "loc_z"],
            "trace_id_field": "tid",
            "z_was_synthesized": False,
        },
        "parameters": {
            "min_loc_per_trace": 1,
            "z_scaling_factor": 0.67,
            "unit_render_pixel_size_nm": 2.5,
            "basic_unit_size_nm": 0.0,
            "min_observed_subunits": 3,
            "core_a_ring_side_nm": 11.0,
            "core_b_ring_side_nm": 19.0,
            "core_twist_deg": 65.452835488,
            "core_axial_offset_nm": 0.0,
            "label_offset_nm": 2.0,
            "pair_tolerance_nm": 0.0,
            "rms_threshold_nm": 0.0,
            "max_pair_residual_nm": 0.0,
            "min_pair_match_fraction": 0.7,
        },
        "effective_parameters": {
            "basic_unit_size_nm": 14.8,
            "pair_tolerance_nm": 5.0,
            "rms_threshold_nm": 4.0,
            "max_pair_residual_nm": 8.0,
            "candidate_edge_radius_nm": 24.0,
            "max_observed_subunits": 6,
            "max_candidate_subsets_per_component": 20_000,
        },
        "template": {
            "site_labels": ["1a", "1b", "2a", "2b", "3a", "3b"],
            "class_distances_nm": {
                "neighboring domains": 8.936,
                "dimer": 10.138,
                "every second A-domain": 11.0,
                "cross-domain": 17.302,
                "every second B-domain": 19.0,
            },
        },
        "screening": {
            "n_after_trace_density": 4000,
            "n_after_log": 3000,
            "n_components": 600,
            "n_candidates_tested": 3773,
            "n_candidates_passed_thresholds": 873,
            "n_overlap_rejected": 677,
            "n_skipped_large_subsets": 12,
        },
        "result": {
            "n_subunits": 2312,
            "n_structures": 196,
            "structure_size_counts": {"3": 150, "4": 46},
            "n_pairs": 858,
            "pair_distance_median_nm": 13.3,
            "pair_distance_min_nm": 7.2,
            "pair_distance_max_nm": 22.8,
            "residual_median_abs_nm": 1.2,
            "residual_max_abs_nm": 7.8,
            "structure_rms_median_nm": 2.1,
            "match_fraction_median": 0.9,
        },
    }
    ev = _ev(HLYB_TEMPLATE_MSG, name="260626-155951_minflux_mfx.mat")
    ev["method_data"] = method_data
    txt = mt.generate_method_text(_state(), [ev])

    assert "50,000 valid localization record(s)" in txt
    assert "itr='last' (the global final iteration)" in txt
    assert "raw Z was multiplied by 0.67 (RIMF, the refractive-index mismatch factor)" in txt
    assert "basic-unit diameter was automatic (effective value 14.8 nm)" in txt
    assert "A- and B-rings with side lengths 11 and 19 nm" in txt
    assert "automatic (effective value 5 nm)" in txt
    # the structural model must state its provenance and the label-offset caveat
    assert "agree with the values annotated on the source structural diagram" in txt
    assert "2 nm per single-domain antibody at each endpoint" in txt
    assert "biased short by that amount" in txt
    assert "invariant under reflection" in txt
    # and the specificity limits must be disclosed, not implied
    assert "selection-biased downward" in txt
    assert "single contiguous range" in txt
    assert "No chance-level expectation was computed" in txt
    assert "σ = Dunit/(2√2·pixel size)" in txt
    assert "DBSCAN with ε = Dunit/2 and minPts = 1" in txt
    assert "all ordered assignments to distinct template sites" in txt
    assert "residual d_observed − d_expected" in txt
    assert "20,000" in txt
    assert "196 non-overlapping HlyB structure(s)" in txt
    assert "150 with 3 observed site(s), 46 with 4 observed site(s)" in txt
    assert "858 distances from 7.2 to 22.8 nm" in txt
    assert "show all (remove template gating)" in txt


def test_source_version_and_container_are_not_conflated():
    """source_version is the structural data version, source_format the container."""
    method_data = {
        "schema": "hlyb_template_matching_3d/v1",
        "input": {"dataset_name": "ds", "source_version": "m2410",
                  "source_format": "obf / mfxdta", "n_localizations": 10,
                  "n_traces": 5, "coordinate_fields": ["loc_x", "loc_y", "loc_z"]},
        "parameters": {}, "effective_parameters": {}, "template": {},
        "screening": {}, "result": {},
    }
    ev = _ev(HLYB_TEMPLATE_MSG, name="ds")
    ev["method_data"] = method_data
    txt = mt.generate_method_text(_state(), [ev])
    assert "source version m2410, container obf / mfxdta" in txt
    assert "source format m2410" not in txt


def test_merge_radius_conflict_with_the_shortest_model_distance_is_disclosed():
    """The detection merge radius can exceed the shortest distance being sought;
    when it does, the methods text must say the class is unrecoverable."""
    template = {"site_labels": ["1a", "1b", "2a", "2b", "3a", "3b"],
                "class_distances_nm": {"neighboring domains": 8.936, "dimer": 10.138,
                                       "every second A-domain": 11.0,
                                       "cross-domain": 17.302,
                                       "every second B-domain": 19.0}}
    base = {"schema": "hlyb_template_matching_3d/v1",
            "input": {"dataset_name": "ds", "coordinate_fields": ["loc_x", "loc_y", "loc_z"]},
            "parameters": {"min_observed_subunits": 3}, "template": template,
            "screening": {}, "result": {}}

    def render(dunit):
        data = dict(base)
        data["effective_parameters"] = {"basic_unit_size_nm": dunit,
                                        "pair_tolerance_nm": 5.0}
        ev = _ev(HLYB_TEMPLATE_MSG, name="ds")
        ev["method_data"] = data
        return mt.generate_method_text(_state(), [ev])

    conflicted = render(16.058)          # merge radius 8.03 nm >= 8.94 - 5
    assert "cannot be recovered from this run" in conflicted
    assert "must not be interpreted as a structural distance" in conflicted

    clean = render(6.0)                  # merge radius 3.0 nm, well below
    assert "all modelled classes remain resolvable in principle" in clean
    assert "property of the detection step" in clean


PAIR_FIT_MSG = (
    "HlyB pair-distance model fit on '260626-155951_minflux_mfx.mat': "
    "3,923 of 6,882 trace(s); excess above null out to 19.2 nm; best model "
    "'six_site' (next by dAIC 609.5, 10.4 with the short-range kernel released); "
    "delta 0.00 nm, sigma 3.23 nm (min loc/trace 10, z-scale 0.67, kernel "
    "empirical (consecutive traces, time-gap selected) from 224 pair(s))."
)


def _pair_fit_method_data(**overrides):
    data = {
        "schema": "hlyb_pair_distance_fit_3d/v1",
        "input": {
            "dataset_name": "ds", "source_path": "", "source_format": "",
            "source_version": "m2410", "n_localizations": 192_334,
            "n_traces_total": 6882, "n_traces_used": 3923,
            "time_column_available": True,
            "z_scaling_source": "the dataset's recorded RIMF",
        },
        "parameters": {
            "min_loc_per_trace": 10, "z_scaling_factor": 0.67, "r_max_nm": 60.0,
            "bin_nm": 0.5, "fit_r_min_nm": 1.0, "fit_r_max_nm": 45.0,
            "null_cell_nm": 50.0, "null_replicates": 8, "repeat_gap_s": 0.2,
            "repeat_max_nm": 40.0, "label_offset_bounds_nm": [0.0, 6.0],
            "fit_label_offset": True, "dimer_distance_bounds_nm": [4.0, 40.0],
            "hypotheses": ["dimer_gaussian", "dimer_uniform", "dimer_lognormal",
                           "trimer_six_site", "no_structure"],
        },
        "observable": {
            "centroid_sem_nm": [1.02, 1.05, 0.61], "sigma_floor_nm": 1.29,
            "excess_outer_nm": 19.25, "null_replicates": 8,
        },
        "repeat_kernel": {
            "source": "empirical (consecutive traces, time-gap selected)",
            "n_pairs": 224, "median_nm": 3.15, "sigma_nm": 2.05,
            "rejected_far_fraction": 0.233,
        },
        "model": {
            "class_names": ["neighboring domains", "dimer", "every second A-domain",
                            "cross-domain", "every second B-domain"],
            "class_distances_nm": [8.936, 10.138, 11.0, 17.302, 19.0],
            "class_weight": 0.2, "reference_dimer_nm": 10.138,
        },
        "distance_scan": {
            "available": True, "parameter": "distance_nm", "best_nm": 7.6,
            "ci68_nm": [7.6, 8.5], "ci95_nm": [7.6, 8.5], "step_nm": 0.9,
            "constrained": True, "ci68_below_scan_step": False,
        },
        "fits": {
            "dimer_gaussian": {
                "delta_aic": 0.0, "n_repeat_pairs": 2495.0,
                "n_structure_pairs": 6944.0, "background_scale": 0.484,
                "sigma_nm": 2.64, "parameters_at_bounds": [],
                "structure_description": "distance 8.00 nm, spread 8.77 nm",
                "distance_summary": {"median_nm": 10.0, "p16_nm": 3.75,
                                     "p84_nm": 17.75, "mode_nm": 8.0,
                                     "mean_nm": 10.7, "spread_nm": 7.0}},
            "dimer_lognormal": {"delta_aic": 28.3, "parameters_at_bounds": []},
            "dimer_uniform": {"delta_aic": 282.2, "parameters_at_bounds": []},
            "trimer_six_site": {"delta_aic": 1499.2, "parameters_at_bounds": []},
            "no_structure": {"delta_aic": 6145.4, "parameters_at_bounds": []},
        },
        "fits_relaxed_kernel": {
            "dimer_gaussian": {"delta_aic": 0.0},
            "dimer_lognormal": {"delta_aic": 22.5},
            "trimer_six_site": {"delta_aic": 78.5},
            "no_structure": {"delta_aic": 261.4},
        },
        "best_hypothesis": "dimer_gaussian",
        "best_hypothesis_relaxed": "dimer_gaussian",
    }
    data.update(overrides)
    return data


def test_pair_fit_text_documents_the_measurement_and_its_limits():
    ev = _ev(PAIR_FIT_MSG, name="ds")
    ev["method_data"] = _pair_fit_method_data()
    txt = mt.generate_method_text(_state(), [ev])

    # the observable, and why nothing is merged
    assert "3,923" in txt and "6,882" in txt
    assert "not** merged into sub-unit centres" in txt
    assert "up to 60 nm" in txt          # regression: 60 was printed as "6"
    assert "50 nm occupancy histogram" in txt
    assert "out to 19.2 nm" in txt
    # the kernel calibration must be described as time-selected, not distance-selected
    assert "selected on the acquisition time alone and never on distance" in txt
    assert "224 consecutive trace pairs" in txt
    # the published geometry must be a candidate, not an imposed constraint
    assert "was **not** imposed" in txt
    assert "fixing it would assume the result" in txt
    assert "free over 4 to 40 nm" in txt
    # the distance is reported as a distribution
    assert "median of 10 nm" in txt
    assert "central 68 % of the population between 3.75 and 17.75 nm" in txt
    # ranking and sensitivity
    assert "worse by 1499.2 AIC units" in txt
    assert "fell from 28.3 to 22.5 AIC units" in txt
    # and the claim must be bounded
    assert "does not identify individual assemblies" in txt
    assert "not a count of detected complexes" in txt


def test_pair_fit_2d_documents_the_projection_model():
    """The 2-D variant must justify itself: it is not the 3-D analysis with z
    discarded, and the foreshortening it cannot remove has to be stated."""
    data = _pair_fit_method_data()
    data["dimensions"] = 2
    data["projection"] = {
        "is_2d": True,
        "cell_mask_stats": {
            "n_cells": 4.0, "in_mask_fraction": 0.89, "median_half_width_nm": 342.0,
            "retained_fraction": 0.30, "border_mode": "relative",
            "border_fraction": 0.35, "implied_max_tilt_deg": 40.5,
        },
        "median_tilt_deg": 28.1, "median_foreshortening": 0.943,
    }
    ev = _ev(PAIR_FIT_MSG.replace("model fit on", "model fit (2D) on"), name="ds")
    ev["method_data"] = data
    txt = mt.generate_method_text(_state(), [ev])
    assert "not by discarding the axial coordinate" in txt
    assert "4 cell(s)" in txt and "shrunk inward by 0.35 of each cell's half-width" in txt
    assert "within 40.5° of face-on" in txt
    assert "median 28.1°" in txt
    assert "mean projected/true ratio of 0.943" in txt
    assert "two-dimensional (Rice) density" in txt
    # the residual limitation must be admitted
    assert "superimposes the upper and lower membrane" in txt


def test_pair_fit_3d_has_no_projection_paragraph():
    ev = _ev(PAIR_FIT_MSG, name="ds")
    ev["method_data"] = _pair_fit_method_data()
    txt = mt.generate_method_text(_state(), [ev])
    assert "Projection." not in txt


def test_pair_fit_rule_matches_both_dimension_labels():
    for msg in (PAIR_FIT_MSG,
                PAIR_FIT_MSG.replace("model fit on", "model fit (2D) on"),
                PAIR_FIT_MSG.replace("model fit on", "model fit (3D) on")):
        ev = _ev(msg, name="ds")
        ev["method_data"] = _pair_fit_method_data()
        txt = mt.generate_method_text(_state(), [ev])
        assert "ensemble pair-distance model fit was performed" in txt


def test_pair_fit_states_the_trimer_gap_and_its_biological_reading():
    """The motivating question is whether the trimer survived preparation; the
    text must answer it rather than leaving an AIC table to speak for itself."""
    ev = _ev(PAIR_FIT_MSG, name="ds")
    ev["method_data"] = _pair_fit_method_data()
    txt = mt.generate_method_text(_state(), [ev])
    assert "published six-site trimer geometry was among the candidates" in txt
    assert "not surviving sample preparation" in txt


def test_pair_fit_calls_a_wide_population_a_property_of_the_sample():
    ev = _ev(PAIR_FIT_MSG, name="ds")
    ev["method_data"] = _pair_fit_method_data()
    txt = mt.generate_method_text(_state(), [ev])
    # spread 7.0 nm against a 2.64 nm blur
    assert "exceeds the 2.64 nm positional blur" in txt
    assert "a property of the sample" in txt


def test_pair_fit_calls_a_narrow_population_unresolved():
    data = _pair_fit_method_data()
    tight = dict(data["fits"]["dimer_gaussian"]["distance_summary"], spread_nm=1.0)
    data["fits"]["dimer_gaussian"] = dict(data["fits"]["dimer_gaussian"],
                                          distance_summary=tight)
    ev = _ev(PAIR_FIT_MSG, name="ds")
    ev["method_data"] = data
    txt = mt.generate_method_text(_state(), [ev])
    assert "does not exceed the 2.64 nm positional blur" in txt
    assert "consistent with being sharp" in txt


def test_pair_fit_flags_an_unresolved_likelihood_interval():
    data = _pair_fit_method_data()
    data["distance_scan"] = dict(data["distance_scan"], ci68_below_scan_step=True,
                                 ci68_nm=[7.6, 7.6])
    ev = _ev(PAIR_FIT_MSG, name="ds")
    ev["method_data"] = data
    txt = mt.generate_method_text(_state(), [ev])
    assert "unresolved rather than tight" in txt


def test_pair_fit_flags_a_scan_that_does_not_localize_the_distance():
    data = _pair_fit_method_data()
    data["distance_scan"] = dict(data["distance_scan"], constrained=False)
    ev = _ev(PAIR_FIT_MSG, name="ds")
    ev["method_data"] = data
    txt = mt.generate_method_text(_state(), [ev])
    assert "do not localize the centre" in txt


def test_pair_fit_says_so_when_the_kernel_could_not_be_calibrated():
    data = _pair_fit_method_data()
    data["input"] = dict(data["input"], time_column_available=False)
    data["repeat_kernel"] = dict(data["repeat_kernel"], source="assumed", n_pairs=0)
    ev = _ev(PAIR_FIT_MSG, name="ds")
    ev["method_data"] = data
    txt = mt.generate_method_text(_state(), [ev])
    assert "no usable acquisition-time column was available" in txt
    assert "an assumed width" in txt


def test_pair_fit_falls_back_without_provenance():
    txt = mt.generate_method_text(_state(), [_ev(PAIR_FIT_MSG, name="ds")])
    assert "ensemble pair-distance model fit" in txt
    assert "not serialized in the Log event" in txt


def test_method_count_tolerates_grouped_numbers():
    assert mt._method_count("6,882") == "6,882"
    assert mt._method_count(6882) == "6,882"
    assert mt._method_count("nope") == "not recorded"


def test_citations_deduped():
    events = [_ev(STDDEV_MSG), _ev(STDDEV_MSG, name="ds2", idx=1)]
    txt = mt.generate_method_text(_state(), events)
    # one citation line despite two stddev events
    assert txt.count(mt.CITE_STDDEV[0]) == 1


def test_html_renders_hyperlinks():
    html = mt.generate_method_html(_state(), [_ev(STDDEV_MSG), _ev(FRC_MSG)])
    assert f'<a href="{mt.CITE_STDDEV[1]}">' in html
    assert f'<a href="{mt.CITE_FRC_BANTERLE[1]}">' in html
    assert f'<a href="{mt.CITE_FRC_NIEUWENHUIZEN[1]}">' in html
    # the visible anchor text is the URL itself (so plain copy keeps it)
    assert f'>{mt.CITE_STDDEV[1]}</a>' in html


def test_unmatched_event_kept_verbatim():
    txt = mt.generate_method_text(_state(), [_ev("Some unmapped operation happened")])
    assert "Some unmapped operation happened" in txt


def test_rimf_anisotropy_note_no_url():
    msg = "RIMF for 'ds1': 0.6375 (auto (estimate anisotropy))"
    txt = mt.generate_method_text(_state(), [_ev(msg)])
    assert "anisotropy of 'ds1' was estimated" in txt
    # custom method → inline note, no DOI
    assert "custom, MINFLUX Data Viewer" in txt


def _fake_ds(state_dict, meta, *, ndim=3, ntraces=0, num_loc=0, name="ds"):
    return SimpleNamespace(
        name=name, state=state_dict, metadata=meta,
        prop=SimpleNamespace(num_dim=ndim, num_traces=ntraces, num_loc=num_loc),
    )


def test_msr_overlay_description():
    import math
    ang = math.radians(1.4)
    c, s = math.cos(ang), math.sin(ang)
    transform = {
        "reference_channel": "ch1", "moving_channel": "ch2",
        "matrix_4x4": [[c, -s, 0, 12.3], [s, c, 0, -4.5], [0, 0, 1, 8.1], [0, 0, 0, 1]],
        "translation_nm": [12.3, -4.5], "z_translation_nm": 8.1,
        "rotation_2x2": [[c, -s], [s, c]],
        "matched_bead_count": 7, "rmse_xy_nm": 3.2, "method": "mbm xy rigid plus z translation",
    }
    ref = _fake_ds(
        {"overlay_id": "g", "overlay_order": 1},
        {"msr_dataset_name": "ch1", "valid_num_loc": 1000, "source_version": "m2410",
         "overlay_reference": "ch1", "overlay_alignment_mode": "mbm info",
         "overlay_bead_excluded": [],
         "overlay_transform": {"reference_channel": "ch1", "moving_channel": "ch1",
                               "matrix_4x4": [[1, 0, 0, 0]], "rotation_2x2": [[1, 0], [0, 1]]}},
        ndim=3, ntraces=200, num_loc=1000, name="ch1")
    mover = _fake_ds(
        {"overlay_id": "g", "overlay_order": 2},
        {"msr_dataset_name": "ch2", "valid_num_loc": 2000, "source_version": "m2410",
         "overlay_transform": transform},
        ndim=3, ntraces=300, num_loc=2000, name="ch2")
    state = SimpleNamespace(datasets=[ref, mover])
    ev = {"message": "MSR overlay loaded from 'file.msr': 2 channel(s) [ch1, ch2]; "
                     "reference channel 'ch1'; channel alignment: mbm info; "
                     "beads: all available beads used.",
          "dataset_idx": 0, "dataset_name": "ch1"}
    txt = mt.generate_method_text(state, [ev])
    assert "overlay was loaded from the .msr file 'file.msr'" in txt
    assert "'ch1' (1,000 valid localizations, 3D, 200 trace(s), m2410)" in txt
    assert "'ch2' (2,000 valid localizations, 3D, 300 trace(s), m2410)" in txt
    assert "Channel 'ch1' served as the alignment reference." in txt
    assert "MBM bead fiducials (all available beads were used)" in txt
    assert "using 7 matched bead(s) (XY RMSE 3.20 nm)" in txt
    assert "translated by +12.30 nm in X, -4.50 nm in Y, and +8.10 nm in Z" in txt
    assert "1.40° counterclockwise rotation in the XY plane" in txt
    assert "transform matrix [[" in txt


def test_empty_events():
    txt = mt.generate_method_text(_state(), [])
    assert "No log events were selected" in txt


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
