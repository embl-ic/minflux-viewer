from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.analysis.hlyb_staged import (
    Staged3DConfig,
    _band_descriptors,
    _excess_summary,
    _null_ratio_distribution,
    analyze_hlyb_staged_3d,
    infer_label_sites,
    segment_spatial_components,
    surface_conditioned_null,
)


def test_uncertainty_aware_site_inference_consolidates_repeats_not_dimers():
    sites = np.asarray([
        [0.0, 0.0, 0.0], [14.0, 0.0, 0.0],
        [100.0, 50.0, -20.0], [114.0, 50.0, -20.0],
    ])
    rng = np.random.default_rng(2)
    centroids = np.repeat(sites, 3, axis=0) + rng.normal(0, 0.35, (12, 3))
    inferred = infer_label_sites(
        centroids, np.full_like(centroids, 0.5), np.full(12, 10),
        np.arange(12.0), np.arange(12.0) + 0.1,
        merge_nm=4.0,
    )

    assert inferred["centers_nm"].shape == (4, 3)
    assert np.all(inferred["n_traces"] == 3)
    d = np.sort(inferred["centers_nm"][:, 0])
    assert d[1] - d[0] == pytest.approx(14.0, abs=1.0)
    assert d[3] - d[2] == pytest.approx(14.0, abs=1.0)


def _rod_sites(rng: np.random.Generator, n: int, *, offset_x: float = 0.0):
    u = rng.uniform(-800.0, 800.0, n)
    # Deliberately nonuniform visibility: the null must preserve it rather than
    # generate a complete cylinder.
    theta = rng.normal(0.35, 0.65, n)
    radius = rng.normal(390.0, 12.0, n)
    return np.column_stack([
        u + offset_x,
        radius * np.cos(theta),
        radius * np.sin(theta),
    ])


def test_surface_null_preserves_axial_and_radial_empirical_support():
    rng = np.random.default_rng(4)
    sites = _rod_sites(rng, 240)
    segmented = segment_spatial_components(sites, link_nm=260.0, min_sites=20)
    assert len(segmented["components"]) == 1

    null = surface_conditioned_null(
        sites, segmented["components"], r_max_nm=40.0, bin_nm=1.0,
        stratum_sites=48, replicates=9, rng_seed=3,
    )
    model = segmented["components"][0]
    original = (sites[model["indices"]] - model["center_nm"]) @ model["axes"]
    preview = (null["preview_sites_nm"] - model["center_nm"]) @ model["axes"]

    assert np.sort(preview[:, 0]) == pytest.approx(np.sort(original[:, 0]))
    assert np.sort(np.linalg.norm(preview[:, 1:], axis=1)) == pytest.approx(
        np.sort(np.linalg.norm(original[:, 1:], axis=1)))


def _localizations_from_sites(
    sites_nm: np.ndarray,
    rng: np.random.Generator,
    *,
    traces_per_site: int = 2,
    locs_per_trace: int = 10,
):
    locs = []
    tids = []
    times = []
    trace_id = 0
    for site_index, site in enumerate(sites_nm):
        for visit in range(traces_per_site):
            trace_center = site + rng.normal(0.0, 0.45, 3)
            block = trace_center + rng.normal(0.0, 0.55, (locs_per_trace, 3))
            locs.append(block * 1e-9)
            tids.extend([trace_id] * locs_per_trace)
            t0 = 10.0 * site_index + 1000.0 * visit
            times.extend(t0 + np.linspace(0.0, 0.08, locs_per_trace))
            trace_id += 1
    return np.vstack(locs), np.asarray(tids), np.asarray(times)


def _synthetic_result(*, with_dimers: bool):
    rng = np.random.default_rng(17)
    all_sites = []
    for cell in range(5):
        offset = 4000.0 * cell
        sites = _rod_sites(rng, 100, offset_x=offset)
        if with_dimers:
            anchors = _rod_sites(rng, 24, offset_x=offset)
            partners = anchors.copy()
            partners[:, 0] += 14.0
            sites = np.vstack([sites, anchors, partners])
        all_sites.append(sites)
    loc, tid, tim = _localizations_from_sites(np.vstack(all_sites), rng)
    cfg = Staged3DConfig(
        z_scaling_factor=1.0,
        cell_link_nm=300.0,
        min_sites_per_component=30,
        null_stratum_sites=48,
        null_replicates=39,
        bootstrap_replicates=79,
        run_sensitivity=False,
    )
    return analyze_hlyb_staged_3d(loc, tid, tim, cfg)


def test_staged_analysis_recovers_injected_short_range_population():
    result = _synthetic_result(with_dimers=True)
    summary = result["summary"]

    assert result["n_sites"] < result["n_traces_used"]
    assert summary["band_ratio"] > 1.10
    assert summary["band_p"] <= 0.05
    assert summary["peak_nm"] == pytest.approx(14.0, abs=1.5)
    assert result["bootstrap"]["available"]


def test_staged_analysis_does_not_create_short_range_excess_on_null_rods():
    result = _synthetic_result(with_dimers=False)
    summary = result["summary"]

    assert 0.75 < summary["band_ratio"] < 1.30
    assert summary["band_p"] > 0.025


def test_band_p_reports_the_resolution_that_censors_it():
    """The rank-based p cannot fall below 1/(replicates + 1).

    It is quoted in the report, so the floor has to travel with it -- otherwise
    a censored value reads as a measured one.
    """
    result = _synthetic_result(with_dimers=True)
    summary = result["summary"]
    replicates = result["null_profiles"].shape[0]

    assert summary["band_p_resolution"] == pytest.approx(1.0 / (replicates + 1))
    assert summary["band_p"] >= summary["band_p_resolution"] - 1e-12


def test_null_ratio_distribution_is_centred_on_one():
    """Leave-one-out ratios of exchangeable replicates must centre on unity."""
    rng = np.random.default_rng(11)
    counts = rng.normal(500.0, 25.0, 199)
    ratios = _null_ratio_distribution(counts)

    assert ratios.size == counts.size
    assert float(ratios.mean()) == pytest.approx(1.0, abs=0.01)
    # Too few replicates to estimate a spread -> refuse rather than guess.
    assert _null_ratio_distribution(counts[:2]).size == 0


def test_calibrated_ratio_z_scores_the_ratio_against_its_own_null():
    result = _synthetic_result(with_dimers=True)
    summary = result["summary"]

    assert summary["null_band_ratio_mean"] == pytest.approx(1.0, abs=0.02)
    assert summary["null_band_ratio_sd"] > 0.0
    # Unbounded, so unlike band_p it still separates strong from overwhelming.
    assert summary["band_ratio_z"] > 3.0

    null_only = _synthetic_result(with_dimers=False)["summary"]
    assert null_only["band_ratio_z"] < summary["band_ratio_z"]


def test_band_descriptors_match_the_summary_on_a_single_null_profile():
    """The bootstrap shares this helper instead of duplicating a null stack.

    It used to pass ``np.vstack([m, m])`` -- a fabricated two-replicate stack --
    whose p/z/sd were meaningless.  The descriptors must agree exactly with the
    replicate-stack summary for the keys the bootstrap actually consumes.
    """
    rng = np.random.default_rng(5)
    edges = np.arange(0.0, 60.5, 0.5)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean = rng.uniform(5.0, 40.0, centers.size)
    observed = mean + rng.uniform(0.0, 6.0, centers.size)
    band = (centers >= 8.0) & (centers < 25.0)

    direct = _band_descriptors(observed, mean, centers, band)
    viaset = _excess_summary(observed, np.vstack([mean, mean]), edges,
                             lo_nm=8.0, hi_nm=25.0)

    assert direct["band_ratio"] == pytest.approx(viaset["band_ratio"])
    assert direct["positive_excess_centroid_nm"] == pytest.approx(
        viaset["positive_excess_centroid_nm"])
    assert direct["peak_nm"] == pytest.approx(viaset["peak_nm"])


def test_stratum_profile_exposes_the_ratio_scale_dependence():
    """band_ratio is conditional on the randomization scale; the location is not.

    Reporting the ratio without its stratum would present a convention-dependent
    number as an absolute effect size.
    """
    rng = np.random.default_rng(9)
    all_sites = []
    for offset in (0.0, 2600.0, 5200.0):
        sites = _rod_sites(rng, 100, offset_x=offset)
        anchors = _rod_sites(rng, 24, offset_x=offset)
        partners = anchors.copy()
        partners[:, 0] += 14.0
        all_sites.append(np.vstack([sites, anchors, partners]))
    loc, tid, tim = _localizations_from_sites(np.vstack(all_sites), rng)
    cfg = Staged3DConfig(
        z_scaling_factor=1.0, cell_link_nm=300.0, min_sites_per_component=30,
        null_stratum_sites=48, null_replicates=39, bootstrap_replicates=79,
        run_sensitivity=False, run_stratum_profile=True,
        stratum_profile_sites=(16, 48, 128), sensitivity_replicates=19,
    )
    result = analyze_hlyb_staged_3d(loc, tid, tim, cfg)
    profile = result["stratum_profile"]

    assert [row["null_stratum_sites"] for row in profile["rows"]] == [16, 48, 128]
    # The ratio grows with the stratum: a wider window absorbs less structure.
    ratios = [row["band_ratio"] for row in profile["rows"]]
    assert ratios[0] < ratios[-1]
    # The excess location is the descriptor that survives the scan.
    centroids = [row["positive_excess_centroid_nm"] for row in profile["rows"]]
    assert max(centroids) - min(centroids) < 4.0


def test_component_bootstrap_flags_a_narrow_interval_at_few_components():
    result = _synthetic_result(with_dimers=True)
    bootstrap = result["bootstrap"]

    assert bootstrap["available"]
    assert bootstrap["narrow_ci_warning"] is (bootstrap["n_components"] < 5)


def test_sensitivity_spread_is_reported_as_the_preferred_uncertainty():
    rng = np.random.default_rng(4)
    all_sites = []
    for offset in (0.0, 2600.0, 5200.0):
        sites = _rod_sites(rng, 100, offset_x=offset)
        anchors = _rod_sites(rng, 24, offset_x=offset)
        partners = anchors.copy()
        partners[:, 0] += 14.0
        all_sites.append(np.vstack([sites, anchors, partners]))
    loc, tid, tim = _localizations_from_sites(np.vstack(all_sites), rng)
    cfg = Staged3DConfig(
        z_scaling_factor=1.0, cell_link_nm=300.0, min_sites_per_component=30,
        null_stratum_sites=48, null_replicates=39, bootstrap_replicates=79,
        run_sensitivity=True, sensitivity_replicates=19,
        run_stratum_profile=False,
    )
    result = analyze_hlyb_staged_3d(loc, tid, tim, cfg)

    lo, hi = result["centroid_sensitivity_range_nm"]
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi
    assert lo <= result["summary"]["positive_excess_centroid_nm"] <= hi
    # The calibrated flag is reported alongside the nominal-p one.
    assert result["robust_short_range_excess_calibrated"] is not None
    assert result["sensitivity_calibrated_passes"] <= result[
        "sensitivity_valid_variants"]


def test_staged_dialog_round_trips_scientific_defaults(qtbot):
    from minflux_viewer.ui.hlyb_staged_dialog import HlyBStagedDialog

    dialog = HlyBStagedDialog(defaults=Staged3DConfig())
    qtbot.addWidget(dialog)
    cfg = dialog.config()

    assert cfg.z_scaling_factor == pytest.approx(0.67)
    assert cfg.site_merge_nm == pytest.approx(4.0)
    assert cfg.short_range_lo_nm == pytest.approx(8.0)
    assert cfg.null_stratum_sites == 64
    assert cfg.run_sensitivity


def test_staged_result_window_is_modeless_and_states_non_distance_claim(qtbot):
    from minflux_viewer.ui.hlyb_staged_dialog import HlyBStagedWindow

    result = _synthetic_result(with_dimers=True)
    window = HlyBStagedWindow(result, title="synthetic")
    qtbot.addWidget(window)

    assert window.parent() is None
    report = window.layout().itemAt(1).widget().widget(2).toPlainText()
    assert "does not estimate a molecular dimer distance" in report
    assert "PRIMARY RESULT" in report


def test_batch_summary_treats_acquisitions_not_pair_counts_as_replicates():
    from scripts.analyze_hlyb_staged_batch import aggregate_acquisitions

    rows = [
        {"condition": "Bonly", "excess_centroid_nm": 12.0, "band_ratio": 1.2},
        {"condition": "Bonly", "excess_centroid_nm": 14.0, "band_ratio": 1.5},
        {"condition": "Bonly", "excess_centroid_nm": 30.0, "band_ratio": 8.0},
    ]
    summary = aggregate_acquisitions(rows)

    assert summary["all"]["n_acquisitions"] == 3
    assert summary["all"]["median_excess_centroid_nm"] == 14.0
    assert summary["Bonly"]["median_band_ratio"] == 1.5
    assert "acquisition_bootstrap_centroid_ci95_nm" in summary["Bonly"]


def test_analysis_is_a_plugin_not_an_analyze_clustering_submenu(qtbot):
    """The workflow lives in Plugins, and only in its current form.

    It is one project-specific analysis, not a family of general clustering
    tools: the retired variants (2D/3D, pair-distance fit, template matching)
    stay unexposed though their modules are kept. The two entries that do exist
    are the same staged analysis over different input scopes -- the active
    dataset, or a pool of ROI-delimited cells gathered across acquisitions --
    and both sit directly under Plugins with no submenu.
    """
    from minflux_viewer import plugins
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.command_finder import collect_commands, filter_commands
    from minflux_viewer.ui.main_window import MainWindow

    plugins.ensure_loaded()
    names = [entry.name for entry in plugins.available()]
    assert "HlyB/D subunit pair analysis" in names

    window = MainWindow(AppState())
    qtbot.addWidget(window)
    commands = collect_commands(window.menuBar())

    hlyb = [c for c in commands if "hlyb" in c.text.lower()]
    assert sorted(c.text for c in hlyb) == sorted([
        "HlyB/D subunit pair analysis",
        "HlyB/D pooled pair analysis (multi-dataset)",
    ]), [c.text for c in hlyb]
    # Directly under Plugins -- no submenu -- and each carries its implementing
    # file for the Command Finder.
    assert {c.path for c in hlyb} == {"Plugins"}
    assert all(c.source.endswith("__init__.py") for c in hlyb)

    # The retired workflows stay out of the menus.
    retired = [c.text for c in commands
               if any(word in c.text.lower()
                      for word in ("template match", "pair-distance model",
                                   "pairwise"))]
    assert retired == []

    clustering = [c.text for c in commands if c.path.endswith("Clustering")]
    assert clustering == ["DBSCAN", "K Nearest Neighbour"]

    # Findable by domain terms that are not in the menu label.
    for query in ("dimer", "surface null", "ecoli"):
        assert "HlyB/D subunit pair analysis" in [
            c.text for c in filter_commands(commands, query)]


def test_method_text_documents_parameters_and_terms():
    """Generate Method Text must be self-contained for a Methods section."""
    from types import SimpleNamespace

    from minflux_viewer.analysis.method_text import (
        RULES, _render_hlyb_staged_short_range)
    from minflux_viewer.plugins.hlyb_pair_analysis.runner import (
        _log_line, _method_payload)

    rng = np.random.default_rng(3)
    all_sites = []
    for offset in (0.0, 2600.0, 5200.0):
        sites = _rod_sites(rng, 90, offset_x=offset)
        anchors = _rod_sites(rng, 20, offset_x=offset)
        partners = anchors.copy()
        partners[:, 0] += 14.0
        all_sites.append(np.vstack([sites, anchors, partners]))
    loc, tid, tim = _localizations_from_sites(np.vstack(all_sites), rng)
    cfg = Staged3DConfig(
        z_scaling_factor=1.0, cell_link_nm=300.0, min_sites_per_component=30,
        null_stratum_sites=48, null_replicates=39, bootstrap_replicates=79,
        sensitivity_replicates=19, stratum_profile_sites=(24, 48, 96))
    result = analyze_hlyb_staged_3d(loc, tid, tim, cfg)

    ds = SimpleNamespace(name="sample", metadata={},
                         file=SimpleNamespace(path="sample.mat"))
    line = _log_line(ds, cfg, result)
    payload = _method_payload(ds, cfg, result,
                              n_localizations=loc.shape[0], has_time=True)

    match = next((pattern.match(line) for pattern, _stage, fn in RULES
                  if fn is _render_hlyb_staged_short_range
                  and pattern.match(line)), None)
    assert match is not None, "the plugin log line must match a method-text rule"
    text, _ = _render_hlyb_staged_short_range(
        match, {"method_data": payload}, None)

    for heading in ("Input data.", "Site inference.", "Parameters.",
                    "Definitions of reported terms.", "Result.",
                    "Interpretation and limitations."):
        assert heading in text, heading
    # Every operator-set parameter is stated, so the run is reproducible.
    for label in ("Same-site consolidation diameter", "Null stratum (sites)",
                  "Test band lower edge", "Null replicates",
                  "Component link distance"):
        assert label in text, label
    # Terms used in the numbers are defined.
    for term in ("inferred labelling site", "observed/null ratio",
                 "positive excess", "null stratum", "sensitivity audit"):
        assert term in text, term
    # The claim stays a distribution descriptor.
    assert "not fitted distances" in text
    assert "neither identifies pair membership" in text
