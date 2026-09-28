from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.analysis.hlyb_staged import (
    Staged3DConfig,
    _band_descriptors,
    _excess_summary,
    _inside_roi_xy,
    _null_ratio_distribution,
    _profile_components,
    analyze_hlyb_staged_3d,
    analyze_hlyb_staged_pooled,
    infer_label_sites,
    persistent_excess_peaks,
    roi_conditioned_null_2d,
    segment_spatial_components,
    surface_conditioned_null,
    surface_normalized_null_envelope,
    surface_normalized_profile,
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


def test_roi_2d_analysis_keeps_pairs_and_null_inside_drawn_shape():
    rng = np.random.default_rng(8)
    theta = rng.uniform(0.0, 2.0 * np.pi, 400)
    radius = np.sqrt(rng.uniform(0.0, 1.0, 400))
    xy = np.column_stack([800.0 * radius * np.cos(theta),
                          350.0 * radius * np.sin(theta)])
    roi = {"type": "oval", "geometry": {"bounds": [-800, -350, 1600, 700]}}
    cell = {"loc_m": xy * 1e-9, "tid": np.arange(xy.shape[0]),
            "roi_geometry": roi, "label": "drawn cell"}
    cfg = Staged3DConfig(min_loc_per_trace=1, cell_link_nm=5.0,
                         null_replicates=19, sensitivity_replicates=9,
                         stratum_profile_sites=(16, 32))
    result = analyze_hlyb_staged_pooled([cell], cfg, is_2d=True)

    assert result["schema"] == "hlyb_staged_short_range_2d_roi/v1"
    assert result["n_components"] == 1
    assert result["n_sites_used"] == result["n_sites"]
    assert result["null_moved_site_fraction"] > 0.5
    assert result["sensitivity"]
    assert len(result["stratum_profile"]["rows"]) == 2
    assert np.all(result["site_centers_nm"][:, 2] == 0.0)
    assert np.all(result["null_preview_sites_nm"][:, 2] == 0.0)
    assert _inside_roi_xy(result["null_preview_sites_nm"][:, :2], roi).all()
    from minflux_viewer.analysis.hlyb_pairwise import pair_distance_profile
    exact, _ = pair_distance_profile(result["site_centers_nm"], cfg.r_max_nm, cfg.bin_nm)
    assert np.array_equal(result["observed"], exact)


def test_roi_2d_null_respects_a_concave_polygon_and_rejects_no_mixing():
    rng = np.random.default_rng(4)
    roi = {"type": "polygon", "geometry": {"points": [
        [0, 0], [300, 0], [300, 80], [80, 80], [80, 300], [0, 300]]}}
    candidate = rng.uniform(0, 300, (400, 2))
    xy = candidate[_inside_roi_xy(candidate, roi)][:100]
    pts = np.column_stack([xy, np.zeros(xy.shape[0])])
    components = segment_spatial_components(pts, link_nm=400, min_sites=20)["components"]
    assert len(components) == 1
    components[0]["roi"] = roi
    null = roi_conditioned_null_2d(
        pts, components, r_max_nm=40, bin_nm=1,
        stratum_sites=32, replicates=9)
    assert _inside_roi_xy(null["preview_sites_nm"][:, :2], roi).all()
    assert null["moved_site_fraction"] > 0.05

    line = np.column_stack([np.linspace(10, 290, 40), np.full(40, 40.0), np.zeros(40)])
    line_component = segment_spatial_components(
        line, link_nm=20, min_sites=20)["components"][0]
    line_component["roi"] = roi
    with pytest.raises(ValueError, match="could not move"):
        roi_conditioned_null_2d(line, [line_component], replicates=9)


def test_surface_normalization_smooths_counts_and_masks_weak_denominators():
    edges = np.arange(0.0, 60.5, 0.5)
    centers = 0.5 * (edges[:-1] + edges[1:])
    null = np.full(centers.size, 20.0)
    null[centers < 8.0] = 0.0
    observed = 2.0 * null
    ratio = surface_normalized_profile(observed, null, edges)

    assert np.isnan(ratio[0])
    assert ratio[np.argmin(np.abs(centers - 14.0))] == pytest.approx(2.0)
    assert ratio[np.argmin(np.abs(centers - 30.0))] == pytest.approx(2.0)
    null_profiles = np.tile(null, (99, 1))
    lower, upper = surface_normalized_null_envelope(null_profiles, edges)
    assert np.isnan(lower[0]) and np.isnan(upper[0])
    middle = np.argmin(np.abs(centers - 30.0))
    assert lower[middle] == pytest.approx(1.0)
    assert upper[middle] == pytest.approx(1.0)


def test_persistent_peaks_keep_broad_modes_not_single_bin_spikes():
    rng = np.random.default_rng(2)
    edges = np.arange(0.0, 60.5, 0.5)
    centers = 0.5 * (edges[:-1] + edges[1:])
    background = 15.0 + 0.15 * centers
    null = rng.normal(background, np.sqrt(background), (99, centers.size))
    observed = (background
                + 30.0 * np.exp(-0.5 * ((centers - 14.0) / 2.0) ** 2)
                + 17.0 * np.exp(-0.5 * ((centers - 25.0) / 2.0) ** 2))
    observed[np.argmin(np.abs(centers - 35.0))] += 80.0

    peaks = persistent_excess_peaks(observed, null, edges)
    distances = sorted(peak["distance_nm"] for peak in peaks)
    assert distances == pytest.approx([14.0, 25.0], abs=0.5)
    assert all(peak["support_fraction"] >= 0.75 for peak in peaks)

    null_only = rng.poisson(background)
    assert persistent_excess_peaks(null_only, null, edges) == []
    assert persistent_excess_peaks(
        np.ones(30), np.ones((9, 30)), np.arange(0.0, 62.0, 2.0)) == []


def test_persistent_peak_threshold_checks_the_whole_search_range():
    rng = np.random.default_rng(7)
    edges = np.arange(0.0, 60.5, 0.5)
    centers = 0.5 * (edges[:-1] + edges[1:])
    background = 15.0 + 0.15 * centers
    false_modes = 0
    for _ in range(100):
        observed = rng.poisson(background)
        null = rng.poisson(background, (99, centers.size))
        false_modes += bool(persistent_excess_peaks(observed, null, edges))
    assert false_modes <= 10


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
    stable = persistent_excess_peaks(
        result["observed"], result["null_profiles"], result["edges_nm"])
    assert stable[0]["distance_nm"] == pytest.approx(14.0, abs=1.5)
    assert result["bootstrap"]["available"]


def test_staged_analysis_does_not_create_short_range_excess_on_null_rods():
    result = _synthetic_result(with_dimers=False)
    summary = result["summary"]

    assert 0.75 < summary["band_ratio"] < 1.30
    assert summary["band_p"] > 0.025
    assert persistent_excess_peaks(
        result["observed"], result["null_profiles"], result["edges_nm"]) == []


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
    assert window._spatial_plot.getViewBox().yInverted()
    window._view_combo.setCurrentText("XZ")
    assert not window._spatial_plot.getViewBox().yInverted()
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


def test_analysis_is_one_direct_plugins_menu_item(qtbot):
    """The workflow is discovered outside the package as one direct item.

    It is one project-specific analysis, not a family of general clustering
    tools: the retired variants stay unexposed though their numerical modules
    are kept. One Tier 2 entry chooses between active and pooled-ROI scopes.
    """
    from minflux_viewer import plugins
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.command_finder import collect_commands, filter_commands
    from minflux_viewer.ui.main_window import MainWindow

    plugins.ensure_loaded()
    assert not any(
        "hlyb" in entry.name.lower() and not entry.discovered
        for entry in plugins.available()
    )

    window = MainWindow(AppState())
    qtbot.addWidget(window)
    commands = collect_commands(window.menuBar())

    hlyb = [c for c in commands if "hlyb" in f"{c.text} {c.path}".lower()]
    assert [c.text for c in hlyb] == ["HlyB/D pair distance analysis"]
    assert hlyb[0].path == "Plugins"
    assert hlyb[0].source.replace("\\", "/").endswith(
        "plugins/hlyb_pair_analysis/main.py"
    )

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
        assert "HlyB/D pair distance analysis" in [
            c.text for c in filter_commands(commands, query)]


def test_external_plugin_exercises_the_published_extension_surfaces(monkeypatch):
    """The proving plugin asks, backgrounds, tables, plots and journals."""
    import sys
    from types import SimpleNamespace

    from minflux_viewer.plugins import loader

    loader.unload_plugin_modules()
    found = next(
        plugin for plugin in loader.scan_root(loader.app_plugin_dir())
        if plugin.id == "embl.hlyb_pair_analysis"
    )
    assert found.error == ""
    launch = loader.load_entry_callable(found)
    module = sys.modules[loader.module_name_for(found.id)]

    result = {
        "summary": {
            "band_ratio": 1.5,
            "band_ratio_z": 3.2,
            "band_p": 0.02,
            "peak_nm": 14.0,
            "positive_excess_centroid_nm": 13.8,
        },
        "n_traces_used": 3,
        "n_sites": 2,
        "n_components": 1,
        "robust_short_range_excess_calibrated": True,
        "sensitivity": [],
        "centers_nm": np.arange(0.5, 60.0, 1.0),
        "edges_nm": np.arange(0.0, 61.0, 1.0),
        "observed": np.full(60, 40.0),
        "null_mean": np.full(60, 20.0),
        "null_profiles": np.full((9, 60), 20.0),
        "site_centers_nm": np.array([[0.0, 0.0, 0.0], [14.0, 0.0, 0.0]]),
        "component_labels": np.array([0, 0]),
    }

    calls = []

    def fake_analyse(task, request, config):
        calls.append(("worker", request["scope"], config["site_merge_nm"]))
        task.progress(1, 1, "done")
        return result

    monkeypatch.setattr(module, "_analyse", fake_analyse)

    # The result window is the application's own, not a published-API
    # surface; it is verified against a real analysis result by the window
    # tests further down this file.
    def fake_window(_ctx, shown, label):
        calls.append(("result-window", label, shown is result))
        return None

    monkeypatch.setattr(module, "_open_result_window", fake_window)

    dataset = object()

    class Data:
        def active(self):
            return dataset

        def properties(self, *, dataset):
            return {"name": "proof"}

        def attr(self, name, **_kwargs):
            values = {
                "loc_x": [0.0, 1e-9, 2e-9],
                "loc_y": [0.0, 0.0, 0.0],
                "loc_z": [0.0, 6e-9, 12e-9],
                "tid": [1, 1, 1],
                "tim": [0.0, 0.1, 0.2],
            }
            return np.asarray(values[name])

    class Ui:
        def ask(self, fields, **_kwargs):
            calls.append(("ask", tuple(fields)))
            return {
                key: value[0] if isinstance(value, list) else value
                for key, value in fields.items()
            }

        def error(self, message, **_kwargs):
            raise AssertionError(message)

        def log(self, message, *_args, **_kwargs):
            calls.append(("log", message))

    class Table:
        def add_rows(self, rows):
            calls.append(("rows", list(rows)))
            return self

        def show(self):
            calls.append(("table-show",))
            return self

    class PlotHandle:
        def series(self, x, y, **kwargs):
            calls.append(("series", np.asarray(x), np.asarray(y), kwargs))
            return self

        def labels(self, **_kwargs):
            return self

        def legend(self):
            return self

        def grid(self):
            return self

        def show(self):
            calls.append(("plot-show",))
            return self

    class Plot:
        def line(self, x, y, **_kwargs):
            calls.append(("line", np.asarray(x), np.asarray(y)))
            return PlotHandle()

        def scatter(self, *_args, **_kwargs):
            calls.append(("scatter",))
            return PlotHandle()

    class Run:
        def background(self, fn, *, on_done, **kwargs):
            calls.append(("background", kwargs["name"], kwargs["kind"]))
            task = SimpleNamespace(progress=lambda *args: calls.append(("progress", args)))
            on_done(fn(task))
            return "task-handle"

    class Journal:
        def record(self, category, summary, **details):
            calls.append(("journal", category, summary, details))

    ctx = SimpleNamespace(
        data=Data(),
        roi=SimpleNamespace(),
        ui=Ui(),
        run=Run(),
        results=SimpleNamespace(table=lambda name: Table()),
        plot=Plot(),
        journal=Journal(),
    )
    assert launch(ctx) == "task-handle"
    assert {call[0] for call in calls} >= {
        "ask", "background", "worker", "rows", "table-show",
        "line", "result-window", "plot-show", "journal", "log",
    }
    asked = next(call[1] for call in calls if call[0] == "ask")
    assert asked[0] == "mode"
    assert "cell_link_nm" not in asked
    # The site view is the interactive window, not a static scatter: the band
    # selector that drives the pair links has to sit beside the histogram.
    assert ("result-window", "proof", True) in calls
    assert not any(call[0] == "scatter" for call in calls)
    plots = [call for call in calls if call[0] == "line"]
    assert len(plots) == 2
    assert plots[1][2][10:-10] == pytest.approx(np.full(40, 2.0))
    series = [call for call in calls if call[0] == "series"]
    assert len(series) == 4
    assert series[-1][2] == pytest.approx(np.ones(60))
    assert series[-1][3]["label"] == "conditional-null baseline"
    rows = next(call[1] for call in calls if call[0] == "rows")
    assert "binning_persistent_excess_peak_nm" in rows[0]


def test_plugin_2d_scope_uses_only_active_roi_display_coordinates():
    from types import SimpleNamespace

    from minflux_viewer.plugins import loader

    found = next(
        plugin for plugin in loader.scan_root(loader.app_plugin_dir())
        if plugin.id == "embl.hlyb_pair_analysis")
    loader.load_entry_callable(found)
    import sys
    module = sys.modules[loader.module_name_for(found.id)]

    dataset = object()
    roi = SimpleNamespace(id="chosen", type="rectangle", name="one cell")

    class Data:
        def active(self):
            return dataset

        def properties(self, *, dataset):
            return {"name": "2D sample", "dimensions": 2}

        def attr(self, name, **_kwargs):
            return {"tid": [10, 11, 12, 13], "tim": [0, 1, 2, 3]}[name]

    class Roi:
        def list(self, **_kwargs):
            return [roi]

        def active(self, **_kwargs):
            return roi

        def mask(self, *_args, **_kwargs):
            return [True, True, False, True]

        def points_in(self, *_args, **_kwargs):
            return [[100, 200, 900], [114, 200, 800], [150, 220, 700]]

        def geometry(self, _roi):
            return {"bounds": [0, 0, 300, 300]}

    ctx = SimpleNamespace(data=Data(), roi=Roi())
    request, selected = module._capture_roi_2d(ctx)
    cell = request["cells"][0]
    assert selected is dataset
    assert request["scope"] == "roi2d"
    assert cell["loc_m"] == pytest.approx(
        np.array([[100, 200], [114, 200], [150, 220]]) * 1e-9)
    assert np.array_equal(cell["tid"], [10, 11, 13])
    assert cell["roi_geometry"]["type"] == "rectangle"
    values = {**vars(Staged3DConfig()), "scope": module.ROI_2D_SCOPE}
    assert module._config_values(values)["z_scaling_factor"] == pytest.approx(0.67)
    roi.context = {"source_view": "histogram"}
    with pytest.raises(ValueError, match="spatial render/scatter"):
        module._capture_roi_2d(ctx)
    roi.context = {"source_view": "render"}
    ctx.roi.active = lambda **_kwargs: None
    with pytest.raises(ValueError, match="Select one stored region ROI"):
        module._capture_roi_2d(ctx)


def test_active_dataset_roi_mode_collects_every_stored_2d_cell():
    import sys
    from types import SimpleNamespace

    from minflux_viewer.plugins import loader

    found = next(
        plugin for plugin in loader.scan_root(loader.app_plugin_dir())
        if plugin.id == "embl.hlyb_pair_analysis")
    loader.load_entry_callable(found)
    module = sys.modules[loader.module_name_for(found.id)]
    dataset = object()
    rois = [SimpleNamespace(type="rectangle", name="cell 1", context={}),
            SimpleNamespace(type="rectangle", name="cell 2", context={})]
    columns = {
        "loc_x": [0, 10, 20, 100, 110, 120],
        "loc_y": [0, 0, 0, 100, 100, 100],
        "loc_z": [0, 0, 0, 0, 0, 0],
        "tid": [1, 2, 3, 4, 5, 6],
        "tim": [0, 1, 2, 3, 4, 5],
    }
    masks = {"cell 1": np.array([1, 1, 1, 0, 0, 0], dtype=bool),
             "cell 2": np.array([0, 0, 0, 1, 1, 1], dtype=bool)}

    class Data:
        def active(self):
            return dataset

        def properties(self, **_kwargs):
            return {"name": "legacy 2D"}

        def attr(self, name, **_kwargs):
            return np.asarray(columns[name], dtype=float)

    class Roi:
        def list(self, **_kwargs):
            return rois

        def mask(self, roi, **_kwargs):
            return masks[roi.name]

        def points_in(self, roi, **_kwargs):
            mask = masks[roi.name]
            return np.column_stack([
                np.asarray(columns["loc_x"])[mask] * 1e9,
                np.asarray(columns["loc_y"])[mask] * 1e9,
            ])

        def geometry(self, roi):
            return {"name": roi.name}

    request, selected = module._capture_active_rois(
        SimpleNamespace(data=Data(), roi=Roi()))
    assert selected is dataset
    assert request["scope"] == "active_roi"
    assert request["is_2d"] is True
    assert [cell["roi"] for cell in request["cells"]] == ["cell 1", "cell 2"]
    assert [cell["tid"].tolist() for cell in request["cells"]] == [
        [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_multi_dataset_mode_opens_the_persistent_pool_instead_of_a_worker(
        monkeypatch):
    import sys
    from types import SimpleNamespace

    from minflux_viewer.plugins import loader

    found = next(
        plugin for plugin in loader.scan_root(loader.app_plugin_dir())
        if plugin.id == "embl.hlyb_pair_analysis")
    loader.load_entry_callable(found)
    module = sys.modules[loader.module_name_for(found.id)]
    sentinel = object()
    seen = {}

    class Owner:
        def show_hlyb_cell_collection(self, defaults):
            seen["defaults"] = defaults
            return sentinel

    values = {**vars(Staged3DConfig()), "mode": module.MULTI_ROI_MODE}
    monkeypatch.setattr(module, "_ask", lambda _ctx: values)
    ctx = SimpleNamespace(
        require_main_window=lambda: Owner(),
        ui=SimpleNamespace(error=lambda *a, **k: pytest.fail(str(a))),
    )

    assert module.run(ctx) is sentinel
    assert isinstance(seen["defaults"], Staged3DConfig)


def test_plugin_2d_capture_matches_the_public_roi_mask(qapp):
    import sys

    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.core.dataset import build_localization_dataset
    from minflux_viewer.plugins import loader

    state = AppState()
    dataset = build_localization_dataset(
        name="drawn 2D", x_nm=[10, 20, 30, 100], y_nm=[10, 20, 30, 100],
        tid=[1, 2, 3, 4], tim=[0, 1, 2, 3])
    state.add_dataset(dataset)
    roi = state.mfv.roi.add(
        "rectangle", {"bounds": [0, 0, 50, 50]}, select=True)
    found = next(
        plugin for plugin in loader.scan_root(loader.app_plugin_dir())
        if plugin.id == "embl.hlyb_pair_analysis")
    loader.load_entry_callable(found)
    module = sys.modules[loader.module_name_for(found.id)]

    request, captured_dataset = module._capture_roi_2d(state.mfv)
    assert captured_dataset is dataset
    assert request["cells"][0]["loc_m"][:, :2] * 1e9 == pytest.approx(
        state.mfv.roi.points_in(roi)[:, :2])
    assert np.array_equal(request["cells"][0]["tid"], [1, 2, 3])


def test_method_text_documents_parameters_and_terms():
    """Generate Method Text must be self-contained for a Methods section."""
    from types import SimpleNamespace

    from minflux_viewer.analysis.hlyb_reporting import _log_line, _method_payload
    from minflux_viewer.analysis.method_text import RULES, _render_hlyb_staged_short_range

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


def test_selected_pairs_are_exactly_the_pairs_the_selected_bins_count():
    """The highlight and the curve must be one computation, not two."""
    from minflux_viewer.analysis.hlyb_staged import (
        bin_mask_for_range,
        pairs_in_bins,
        within_component_pairs,
    )

    rng = np.random.default_rng(11)
    pts = np.column_stack([
        rng.uniform(0.0, 900.0, 260),
        rng.uniform(0.0, 260.0, 260),
        rng.normal(0.0, 30.0, 260),
    ])
    components = segment_spatial_components(pts, link_nm=120.0, min_sites=20)
    labels = components["labels"]
    cfg = Staged3DConfig()
    observed, _by_component, edges = _profile_components(
        pts, labels, r_max_nm=cfg.r_max_nm, bin_nm=cfg.bin_nm)

    found = within_component_pairs(pts, labels, r_max_nm=cfg.r_max_nm)
    assert np.array_equal(
        np.histogram(found["distances_nm"], bins=edges)[0], observed.astype(int))
    assert np.all(found["pairs"][:, 0] < found["pairs"][:, 1])
    assert np.array_equal(labels[found["pairs"][:, 0]], found["component"])
    assert np.array_equal(labels[found["pairs"][:, 1]], found["component"])

    band = bin_mask_for_range(edges, cfg.short_range_lo_nm, cfg.short_range_hi_nm)
    selected = pairs_in_bins(found["distances_nm"], edges, band)
    assert selected.sum() == observed[band].sum()

    whole = pairs_in_bins(
        found["distances_nm"], edges, np.ones(edges.size - 1, dtype=bool))
    assert whole.all()
    none = pairs_in_bins(
        found["distances_nm"], edges, np.zeros(edges.size - 1, dtype=bool))
    assert not none.any()
    with pytest.raises(ValueError, match="one entry per histogram bin"):
        pairs_in_bins(found["distances_nm"], edges, band[:-1])


def test_pair_enumeration_never_crosses_a_component_or_the_radius():
    from minflux_viewer.analysis.hlyb_staged import within_component_pairs

    pts = np.array([
        [0.0, 0.0, 0.0], [12.0, 0.0, 0.0],        # component 0, 12 nm apart
        [4000.0, 0.0, 0.0], [4018.0, 0.0, 0.0],   # component 1, 18 nm apart
    ])
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    found = within_component_pairs(pts, labels, r_max_nm=60.0)
    assert found["distances_nm"] == pytest.approx([12.0, 18.0])

    excluded = within_component_pairs(pts, np.array([0, 0, -1, -1]), r_max_nm=60.0)
    assert excluded["distances_nm"] == pytest.approx([12.0])

    narrow = within_component_pairs(pts, labels, r_max_nm=15.0)
    assert narrow["distances_nm"] == pytest.approx([12.0])

    empty = within_component_pairs(pts, np.full(4, -1), r_max_nm=60.0)
    assert empty["pairs"].shape == (0, 2)
    with pytest.raises(ValueError, match="one entry per point"):
        within_component_pairs(pts, np.array([0, 0]), r_max_nm=60.0)


def test_band_selector_highlights_exactly_the_pairs_its_bins_count(qtbot):
    """Dragging the band is the indicator: links, rings and count must agree."""
    from minflux_viewer.ui.hlyb_staged_dialog import HlyBStagedWindow

    result = _synthetic_result(with_dimers=True)
    window = HlyBStagedWindow(result, title="synthetic")
    qtbot.addWidget(window)

    cfg = result["config"]
    observed = np.asarray(result["observed"], dtype=float)
    assert window._bands[0].region.movable
    # The pre-declared band is marked, not draggable: the reported result is
    # that range, whatever the selector is showing.
    assert [line.value() for line in window._declared_lines] == pytest.approx(
        [cfg.short_range_lo_nm, cfg.short_range_hi_nm])
    assert not any(line.movable for line in window._declared_lines)

    # Opens on the pre-declared band, and reproduces its reported pair count.
    assert window._band_range() == pytest.approx(
        (cfg.short_range_lo_nm, cfg.short_range_hi_nm))
    selected = window.selected_pair_mask()
    assert selected.sum() == observed[window.selected_bin_mask()].sum()
    assert selected.sum() == pytest.approx(
        result["summary"]["band_observed_pairs"])

    pairs = window._pair_table()["pairs"]
    assert np.array_equal(
        window._selected_site_indices(), np.unique(pairs[selected]))
    drawn = window._bands[0].link_item.getData()[0]
    assert drawn.size == 3 * int(selected.sum())          # two ends and a gap
    assert window._bands[0].link_item.isVisible()

    # Moving it selects a different set of bins, and the view follows.
    window._bands[0].region.setRegion((30.0, 40.0))
    moved_bins = window.selected_bin_mask()
    moved = window.selected_pair_mask()
    assert moved.sum() == observed[moved_bins].sum()
    assert not np.array_equal(moved, selected)
    window._refresh_spatial()
    assert window._bands[0].link_item.getData()[0].size == 3 * int(moved.sum())
    assert f"{int(round(float(observed[moved_bins].sum()))):,} pair(s)" \
        in window._band_label.text()

    # A range holding no bin is reported, not silently drawn as empty.
    window._bands[0].region.setRegion((cfg.r_max_nm + 10.0, cfg.r_max_nm + 20.0))
    window._refresh_spatial()
    assert "no histogram bin" in window._band_label.text()
    assert window._bands[0].link_item.getData()[0] is None or \
        window._bands[0].link_item.getData()[0].size == 0


def test_band_handles_settle_on_bin_edges_so_the_range_matches_the_pairs(qtbot):
    from minflux_viewer.ui.hlyb_staged_dialog import HlyBStagedWindow

    result = _synthetic_result(with_dimers=True)
    window = HlyBStagedWindow(result, title="synthetic")
    qtbot.addWidget(window)

    edges = np.asarray(result["edges_nm"], dtype=float)
    window._bands[0].region.setRegion((10.31, 24.87))
    before = window.selected_bin_mask().copy()
    window._snap_band_to_bins()
    lo, hi = window._band_range()

    assert lo in edges and hi in edges
    assert np.array_equal(window.selected_bin_mask(), before)


def test_toggling_a_layer_keeps_the_pan_and_zoom(qtbot):
    """The project rule for filter toggles applies to this view too."""
    from minflux_viewer.ui.hlyb_staged_dialog import HlyBStagedWindow

    result = _synthetic_result(with_dimers=True)
    window = HlyBStagedWindow(result, title="synthetic")
    qtbot.addWidget(window)

    box = window._spatial_plot.getViewBox()
    box.setRange(xRange=(100.0, 300.0), yRange=(50.0, 150.0), padding=0.0)
    chosen = np.asarray(box.viewRange(), dtype=float).ravel()

    window._raw_check.setChecked(True)
    window._merge_check.setChecked(True)
    window._pair_check.setChecked(False)
    assert np.asarray(box.viewRange()).ravel() == pytest.approx(chosen)

    # A different projection is a different picture, so that one does re-fit.
    window._view_combo.setCurrentText("XZ")
    assert not np.allclose(np.asarray(box.viewRange()).ravel(), chosen)


def test_site_and_trace_hover_report_the_consolidation(qtbot):
    from minflux_viewer.ui.hlyb_staged_dialog import HlyBStagedWindow

    result = _synthetic_result(with_dimers=True)
    window = HlyBStagedWindow(result, title="synthetic")
    qtbot.addWidget(window)

    spots = window._site_item.points()
    assert spots.size
    text = window._site_tooltip(spots[0])
    assert "Inferred label site 1" in text
    assert "Component" in text
    assert "Traces consolidated:" in text
    assert "Pairs in band A:" in text
    # pyqtgraph would otherwise add its own x/y/data tooltip over ours.
    assert window._site_item.opts["tip"] is None
    assert window._site_item.opts["hoverable"]
    assert window._trace_item.opts["tip"] is None


def test_consolidation_links_join_each_trace_to_its_site(qtbot):
    from minflux_viewer.ui.hlyb_staged_dialog import HlyBStagedWindow

    result = _synthetic_result(with_dimers=True)
    window = HlyBStagedWindow(result, title="synthetic")
    qtbot.addWidget(window)

    assert not window._merge_item.isVisible()
    window._merge_check.setChecked(True)
    xs = window._merge_item.getData()[0]
    mapping = np.asarray(result["trace_to_site"], dtype=np.int64)
    assert xs.size == 3 * int(np.sum((mapping >= 0)
                                     & (mapping < result["n_sites"])))


def test_pooled_trace_to_site_map_is_offset_per_cell_and_never_spans_two():
    """The consolidation view needs this map, and 2-D ROI mode is pooled."""
    rng = np.random.default_rng(23)
    cells = []
    for cell in range(2):
        anchor = np.array([cell * 9000.0, 0.0, 0.0])
        sites = anchor + np.column_stack([
            rng.uniform(0.0, 700.0, 60),
            rng.uniform(0.0, 240.0, 60),
            rng.normal(0.0, 25.0, 60),
        ])
        # Two traces per site, so consolidation has something to do.
        centres = np.repeat(sites, 2, axis=0)
        loc = np.repeat(centres, 12, axis=0) + rng.normal(0.0, 1.0, (centres.shape[0] * 12, 3))
        tid = np.repeat(np.arange(centres.shape[0]), 12)
        cells.append({
            "loc_m": loc * 1e-9, "tid": tid,
            "tim": np.arange(loc.shape[0], dtype=float),
            "label": f"cell {cell + 1}", "dataset": "pooled",
        })
    cfg = Staged3DConfig(min_loc_per_trace=5, null_replicates=9,
                         run_sensitivity=False, run_stratum_profile=False)
    result = analyze_hlyb_staged_pooled(cells, cfg)

    mapping = np.asarray(result["trace_to_site"], dtype=np.int64)
    traces = np.asarray(result["trace_centroids_nm"], dtype=float)
    assert mapping.size == traces.shape[0]
    assert mapping.max() < result["n_sites"]
    assert mapping.min() >= 0

    # Each site draws its traces from exactly one cell: consolidation is
    # confined per cell, so a pooled site cannot straddle two of them.
    sites = np.asarray(result["site_centers_nm"], dtype=float)
    for site in np.unique(mapping):
        members = traces[mapping == site]
        assert np.all(np.abs(members[:, 0] - sites[site, 0]) < 4500.0)


def test_roi_2d_result_opens_the_window_with_working_pair_links(qtbot):
    """The interactive view has to work in the 2-D ROI mode too."""
    from minflux_viewer.ui.hlyb_staged_dialog import HlyBStagedWindow

    rng = np.random.default_rng(31)
    centres = np.column_stack([
        rng.uniform(-700.0, 700.0, 70), rng.uniform(-260.0, 260.0, 70)])
    partners = centres + 14.0 * np.column_stack(
        [np.cos(rng.uniform(0, 2 * np.pi, 70)), np.sin(rng.uniform(0, 2 * np.pi, 70))])
    xy = np.vstack([centres, partners])
    inside = (np.abs(xy[:, 0]) < 800.0) & (np.abs(xy[:, 1]) < 350.0)
    xy = xy[inside]
    loc = np.repeat(xy, 12, axis=0) + rng.normal(0.0, 1.0, (xy.shape[0] * 12, 2))
    cell = {
        "loc_m": loc * 1e-9,
        "tid": np.repeat(np.arange(xy.shape[0]), 12),
        "roi_geometry": {"type": "rectangle",
                         "geometry": {"bounds": [-800, -350, 1600, 700]}},
        "label": "drawn cell",
    }
    cfg = Staged3DConfig(min_loc_per_trace=5, null_replicates=9,
                         run_sensitivity=False, run_stratum_profile=False)
    result = analyze_hlyb_staged_pooled([cell], cfg, is_2d=True)

    window = HlyBStagedWindow(result, title="2-D ROI")
    qtbot.addWidget(window)
    observed = np.asarray(result["observed"], dtype=float)
    assert window.selected_pair_mask().sum() == observed[
        window.selected_bin_mask()].sum()
    assert window._bands[0].link_item.getData()[0].size == 3 * int(
        window.selected_pair_mask().sum())
    # A 2-D result is flat in Z, so the XZ projection is a line, not an error.
    window._view_combo.setCurrentText("XZ")
    assert window._bands[0].link_item.isVisible()
    assert window._merge_check.isEnabled()
    # The report must name the null that was run, not the 3-D surface one.
    report = window.layout().itemAt(1).widget().widget(2).toPlainText()
    assert "ROI-conditioned 2-D null" in report
    assert "conditional surface null" not in report
    assert "XY projections" in report


def test_window_survives_a_result_with_nothing_to_pair(qtbot):
    """One site, or none, must open the window rather than raise."""
    from minflux_viewer.ui.hlyb_staged_dialog import HlyBStagedWindow

    result = _synthetic_result(with_dimers=True)
    for n in (0, 1):
        thin = dict(result)
        thin["site_centers_nm"] = np.asarray(
            result["site_centers_nm"], dtype=float)[:n]
        thin["component_labels"] = np.zeros(n, dtype=np.int64)
        thin["observed"] = np.zeros_like(
            np.asarray(result["observed"], dtype=float))
        thin["null_mean"] = np.zeros_like(thin["observed"])
        window = HlyBStagedWindow(thin, title=f"{n} site(s)")
        qtbot.addWidget(window)
        assert window.selected_pair_mask().size == 0
        assert window._selected_site_indices().size == 0
        assert "0 pair(s)" in window._band_label.text()
        assert "ratio n/a" in window._band_label.text()


def test_plugin_opens_the_window_non_owned_and_retained(qtbot):
    """The seam the published-surface test stubs out.

    A real ``MainWindow`` is deliberately not built here -- per the project's
    Qt lifecycle notes that drags in the render/post-load chain and destabilizes
    the suite -- so the owner is a plain widget, which is all ``show_modeless``
    requires. The full facade path was verified separately against a live
    window.
    """
    import sys
    from types import SimpleNamespace

    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QWidget

    from minflux_viewer.plugins import loader

    found = next(
        plugin for plugin in loader.scan_root(loader.app_plugin_dir())
        if plugin.id == "embl.hlyb_pair_analysis")
    loader.load_entry_callable(found)
    module = sys.modules[loader.module_name_for(found.id)]

    owner = QWidget()
    qtbot.addWidget(owner)
    ctx = SimpleNamespace(require_main_window=lambda: owner)

    result = _synthetic_result(with_dimers=True)
    window = module._open_result_window(ctx, result, "seam")
    # ⚠ Deliberately NOT handed to qtbot: show_modeless sets WA_DeleteOnClose
    # and the owner already retains it, so registering it here would give one
    # parentless window two cleaners racing to delete it under a queued paint.
    try:
        assert window.windowTitle().endswith("seam")
        # Non-owned and retained: the project's convention for a modeless
        # window, so the OS cannot pin it above the main window and Python
        # cannot collect it the moment the plugin returns.
        assert window.parent() is None
        assert window in owner._modeless_windows
        assert window.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        assert not window.isModal()
        # And it opened on a working selection, not an inert one.
        assert window.selected_pair_mask().sum() == np.asarray(
            result["observed"])[window.selected_bin_mask()].sum()
    finally:
        window.close()
        qtbot.wait(10)


def test_two_bands_highlight_two_populations_independently(qtbot):
    """One selection near 14 nm and one near 25 nm, each in its own colour."""
    from minflux_viewer.ui.hlyb_staged_dialog import HlyBStagedWindow

    result = _synthetic_result(with_dimers=True)
    window = HlyBStagedWindow(result, title="synthetic")
    qtbot.addWidget(window)
    observed = np.asarray(result["observed"], dtype=float)
    a, b = window._bands

    # The second band is opt-in, so the default view is the single-band one.
    assert a.enabled and not b.enabled
    assert a.region.isVisible() and not b.region.isVisible()
    assert not b.link_item.isVisible() and not b.glow_item.isVisible()
    assert window.band_readout(0) in window._band_label.text()
    assert "band B" not in window._band_label.text()

    window._band_check.setChecked(True)
    assert b.enabled and b.region.isVisible() and b.mirror.isVisible()

    a.region.setRegion((12.0, 16.0))
    b.region.setRegion((23.0, 27.0))
    window._refresh_spatial()

    # Each band counts exactly the pairs its own bins hold.
    for band in (0, 1):
        assert window.selected_pair_mask(band).sum() == observed[
            window.selected_bin_mask(band)].sum()
    assert not np.array_equal(
        window.selected_pair_mask(0), window.selected_pair_mask(1))

    # ...and draws them on its own items, in its own colour.
    for band in window._bands:
        assert band.link_item.isVisible()
        assert band.link_item.getData()[0].size == 3 * int(
            window.selected_pair_mask(band.index).sum())
        drawn = band.link_item.opts["pen"].color()
        assert (drawn.red(), drawn.green(), drawn.blue()) == band.rgb
    assert window._bands[0].rgb != window._bands[1].rgb

    # Both are described, not merely drawn.
    text = window._band_label.text()
    assert "band A 12.00–16.00 nm" in text
    assert "band B 23.00–27.00 nm" in text

    # Turning it off removes it everywhere at once.
    window._band_check.setChecked(False)
    assert not b.region.isVisible() and not b.mirror.isVisible()
    assert not b.link_item.isVisible() and not b.glow_item.isVisible()
    assert "band B" not in window._band_label.text()


def test_second_band_opens_clear_of_the_first_and_mirrors_onto_the_excess(qtbot):
    from minflux_viewer.ui.hlyb_staged_dialog import HlyBStagedWindow

    result = _synthetic_result(with_dimers=True)
    cfg = result["config"]
    window = HlyBStagedWindow(result, title="synthetic")
    qtbot.addWidget(window)

    # Two selectors stacked on the same range would look like one.
    assert window._band_range(0) == pytest.approx(
        (cfg.short_range_lo_nm, cfg.short_range_hi_nm))
    lo_b, hi_b = window._band_range(1)
    assert lo_b >= cfg.short_range_hi_nm
    assert hi_b <= cfg.r_max_nm

    window._band_check.setChecked(True)
    for band in window._bands:
        band.region.setRegion((10.0 + 10 * band.index, 20.0 + 10 * band.index))
        window._on_band_changed(band.index)
        assert band.mirror.getRegion() == pytest.approx(
            window._band_range(band.index))


def test_each_band_rings_its_own_sites_and_labels_carry_its_colour(qtbot):
    from minflux_viewer.ui.hlyb_staged_dialog import HlyBStagedWindow

    result = _synthetic_result(with_dimers=True)
    window = HlyBStagedWindow(result, title="synthetic")
    qtbot.addWidget(window)
    window._band_check.setChecked(True)
    window._bands[0].region.setRegion((12.0, 16.0))
    window._bands[1].region.setRegion((23.0, 27.0))
    window._refresh_spatial()

    for band in window._bands:
        involved = window._selected_site_indices(band.index)
        assert band.glow_item.isVisible() == (involved.size > 0)
        if involved.size:
            assert len(band.glow_item.points()) == involved.size
    # Rings are sized apart so a site in both bands shows both.
    assert window._bands[0].glow_item.opts["size"] != \
        window._bands[1].glow_item.opts["size"]

    # Labels are pooled across bands and capped once, each in its band colour.
    window._spatial_plot.getPlotItem().vb.setRange(
        xRange=(-40.0, 40.0), yRange=(-40.0, 40.0), padding=0.0)
    window._refresh_pair_labels()
    shown = {key: item for key, item in window._label_items.items()
             if item.isVisible()}
    assert len(shown) <= 100
    # A label is keyed by (band, pair) and painted in that band's colour, so a
    # number can be attributed without tracing its line back.
    assert all(isinstance(key, tuple) and key[0] in (0, 1) for key in shown)
    for (band_index, _row), item in shown.items():
        colour = item.textItem.defaultTextColor()
        assert (colour.red(), colour.green(), colour.blue()) ==             window._bands[band_index].rgb
