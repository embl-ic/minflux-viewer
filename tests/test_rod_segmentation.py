from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.analysis.rod_segmentation import (
    RodConfig,
    capsule_outline,
    detect_rods,
    seed_level_for,
)


def _capsule_points(rng, n, center, length, width, angle_deg=0.0):
    """Uniform points inside a stadium of *length* x *width* nm."""
    half_l, r = 0.5 * float(length), 0.5 * float(width)
    flat = max(half_l - r, 0.0)
    out = []
    while sum(block.shape[0] for block in out) < n:
        x = rng.uniform(-half_l, half_l, n)
        y = rng.uniform(-r, r, n)
        inside = np.abs(x) <= flat
        cap = np.hypot(np.abs(x) - flat, y) <= r
        keep = inside | cap
        out.append(np.column_stack([x[keep], y[keep]]))
    pts = np.vstack(out)[:n]
    t = np.radians(float(angle_deg))
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return pts @ rot.T + np.asarray(center, dtype=float)


def _cfg(**kwargs):
    base = dict(pixel_size_nm=20.0, min_width_nm=800.0, max_width_nm=1100.0,
                min_length_nm=1000.0, max_length_nm=8000.0)
    base.update(kwargs)
    return RodConfig(**base)


def test_detects_two_separated_rods_with_correct_size_and_orientation():
    rng = np.random.default_rng(0)
    a = _capsule_points(rng, 9000, (0.0, 0.0), 3000.0, 1000.0, 0.0)
    b = _capsule_points(rng, 9000, (0.0, 4000.0), 2400.0, 950.0, 40.0)

    result = detect_rods(np.vstack([a, b]), _cfg())

    assert len(result.accepted) == 2
    by_length = sorted(result.accepted, key=lambda r: r.length_nm)
    short, long = by_length
    assert long.length_nm == pytest.approx(3000.0, abs=250.0)
    assert long.width_nm == pytest.approx(1000.0, abs=120.0)
    assert short.length_nm == pytest.approx(2400.0, abs=250.0)
    assert abs(long.angle_deg) < 6.0
    assert short.angle_deg == pytest.approx(40.0, abs=6.0)
    # A clean rod's two independent width measures agree.
    for rod in result.accepted:
        assert rod.width_ratio == pytest.approx(1.0, abs=0.25)
        assert rod.fill_fraction > 0.8


def test_every_localization_of_a_rod_is_assigned_to_it():
    rng = np.random.default_rng(1)
    a = _capsule_points(rng, 6000, (0.0, 0.0), 3000.0, 1000.0)
    b = _capsule_points(rng, 6000, (0.0, 4000.0), 3000.0, 1000.0)
    points = np.vstack([a, b])

    result = detect_rods(points, _cfg())

    assert len(result.accepted) == 2
    assert result.component_of_point.shape == (points.shape[0],)
    assigned = result.component_of_point >= 0
    # The mask is a dilated/closed hull of the points, so every point lands in
    # a cell; the two halves must go to two different components.
    assert assigned.mean() > 0.99
    assert set(np.unique(result.component_of_point[assigned])) == {0, 1}
    first = result.component_of_point[:a.shape[0]]
    second = result.component_of_point[a.shape[0]:]
    assert len(set(np.unique(first[first >= 0]))) == 1
    assert len(set(np.unique(second[second >= 0]))) == 1
    assert set(np.unique(first[first >= 0])) != set(np.unique(second[second >= 0]))


def test_thin_bridge_between_end_to_end_cells_is_split():
    """Two cells whose caps nearly touch merge through the morphological close.

    Without splitting they pass every gate as one implausibly long rod, which
    is precisely the silent under-segmentation the split exists to prevent.
    """
    rng = np.random.default_rng(2)
    a = _capsule_points(rng, 9000, (0.0, 0.0), 3000.0, 1000.0)
    b = _capsule_points(rng, 9000, (3150.0, 0.0), 3000.0, 1000.0)
    points = np.vstack([a, b])
    # An explicit bridging length long enough to close the 150 nm gap, so the
    # merge this test is about actually happens.
    bridged = dict(smooth_nm=60.0, close_nm=120.0)

    merged = detect_rods(points, _cfg(split_touching=False, **bridged))
    assert len(merged.accepted) == 1
    assert merged.accepted[0].length_nm > 5500.0

    split = detect_rods(points, _cfg(split_touching=True, **bridged))
    assert len(split.accepted) == 2
    assert split.stats["n_split"] == 1
    for rod in split.accepted:
        assert rod.length_nm == pytest.approx(3000.0, abs=350.0)
        assert rod.width_nm == pytest.approx(1000.0, abs=120.0)


def test_side_by_side_overlap_is_rejected_rather_than_analysed():
    """Parallel cells overlapping in projection cannot be separated by any
    2-D method — the width gate must reject them instead of accepting a
    double-width object."""
    rng = np.random.default_rng(3)
    a = _capsule_points(rng, 9000, (0.0, 0.0), 3000.0, 1000.0)
    b = _capsule_points(rng, 9000, (0.0, 950.0), 3000.0, 1000.0)

    result = detect_rods(np.vstack([a, b]), _cfg())

    assert result.accepted == []
    assert len(result.rejected) == 1
    assert result.rejected[0].reject_reason == "too wide"
    assert result.rejected[0].width_nm > 1500.0


def test_round_blob_is_rejected_as_not_elongated():
    rng = np.random.default_rng(4)
    angle = rng.uniform(0, 2 * np.pi, 8000)
    radius = 480.0 * np.sqrt(rng.uniform(0, 1, 8000))
    blob = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])

    result = detect_rods(blob, _cfg(min_length_nm=1500.0))

    assert result.accepted == []
    assert result.rejected[0].reject_reason in ("too short", "not elongated")


def test_width_window_is_enforced_in_both_directions():
    rng = np.random.default_rng(5)
    thin = _capsule_points(rng, 9000, (0.0, 0.0), 3000.0, 500.0)

    result = detect_rods(thin, _cfg())

    assert result.accepted == []
    assert result.rejected[0].reject_reason == "too narrow"


def test_mask_width_overshoots_the_structure_and_the_tolerance_absorbs_it():
    """The width gate is stated in *structure* width, but measured on the
    smoothed density mask, which envelopes the structure a little wider."""
    from minflux_viewer.analysis.rod_segmentation import (
        resolve_smoothing, width_tolerance_for)

    rng = np.random.default_rng(8)
    # Sites on the membrane of a 1000 nm rod: the projected density peaks at
    # the two edges, which is the case that overshoots most.
    u = rng.uniform(-1000.0, 1000.0, 12000)
    theta = rng.uniform(0, 2 * np.pi, 12000)
    pts = np.column_stack([u, 500.0 * np.cos(theta)])

    cfg = _cfg(pixel_size_nm=25.0, smooth_nm=60.0)
    rod = detect_rods(pts, cfg).accepted[0]

    smooth, close = resolve_smoothing(pts, cfg)
    assert smooth == pytest.approx(60.0)
    assert close == pytest.approx(120.0)          # auto: twice the bridging length
    assert width_tolerance_for(cfg, smooth) == pytest.approx(120.0)
    # Measured wider than the true 1000 nm, but covered by the tolerance.
    assert 1000.0 < rod.width_nm <= 1000.0 + width_tolerance_for(cfg, smooth)

    # A window stated truthfully — these cells are 1000 nm wide, so 1050 is a
    # fair upper bound — rejects them without the tolerance and accepts them
    # with it.  That is exactly what the tolerance exists for.
    truthful = dict(pixel_size_nm=25.0, smooth_nm=60.0, max_width_nm=1050.0)
    assert detect_rods(pts, _cfg(**truthful, width_tolerance_nm=0.0)
                       ).rejected[0].reject_reason == "too wide"
    assert len(detect_rods(pts, _cfg(**truthful)).accepted) == 1


def test_bridging_length_adapts_to_labelling_sparsity():
    """Sparse labelling needs a longer bridging length than dense labelling.

    With a fixed short length, a sparsely labelled cell shatters into
    fragments — which is exactly how one real E. coli came apart.
    """
    from minflux_viewer.analysis.rod_segmentation import resolve_smoothing

    rng = np.random.default_rng(9)
    dense = _capsule_points(rng, 12000, (0.0, 0.0), 3000.0, 1000.0)
    sparse = _capsule_points(rng, 320, (0.0, 0.0), 3000.0, 1000.0)

    cfg = _cfg(pixel_size_nm=25.0)
    dense_smooth, _ = resolve_smoothing(dense, cfg)
    sparse_smooth, _ = resolve_smoothing(sparse, cfg)

    assert sparse_smooth > dense_smooth
    # Held inside the configured bounds in both cases.
    assert cfg.smooth_min_pixels * cfg.pixel_size_nm <= dense_smooth
    assert sparse_smooth <= cfg.min_width_nm / cfg.smooth_max_width_fraction

    # The sparse cell is found as one whole cell with the adaptive length.
    found = detect_rods(sparse, cfg).accepted
    assert len(found) == 1
    assert found[0].width_nm == pytest.approx(1000.0, abs=200.0)

    # With a length tuned for dense data the mask covers only the labelled
    # patches, so the cell is not recovered at all — it measures far too narrow.
    starved = detect_rods(sparse, _cfg(pixel_size_nm=25.0, smooth_nm=20.0,
                                       close_nm=40.0))
    assert starved.accepted == []
    assert max(rod.width_nm for rod in starved.rods) < 700.0


def test_sparsely_labelled_membrane_cell_is_found_whole():
    """Regression for a real single E. coli that came apart into fragments.

    The reference acquisition holds ~320 label sites on the membrane of one
    ~1000 x 3400 nm cell, each visited by a trace of ~50 localizations. The
    localization count is large but the *coverage* is sparse, and a bridging
    length tuned for dense data recovered only disconnected patches ~450 nm
    across. The bridging length has to follow the site spacing (~95 nm here).
    """
    from minflux_viewer.analysis.rod_segmentation import label_spacing_nm

    rng = np.random.default_rng(31)
    # Sites on a cylindrical membrane, projected to XY.
    n_sites = 320
    u = rng.uniform(-1700.0, 1700.0, n_sites)
    theta = rng.uniform(0, 2 * np.pi, n_sites)
    sites = np.column_stack([u, 500.0 * np.cos(theta)])
    # Each site visited by a trace: ~50 localizations piled within a few nm.
    locs = np.repeat(sites, 50, axis=0) + rng.normal(0.0, 2.0, (n_sites * 50, 2))

    assert label_spacing_nm(sites) == pytest.approx(95.0, abs=35.0)

    cfg = _cfg(pixel_size_nm=25.0)
    result = detect_rods(locs, cfg, spacing_points_nm=sites)

    assert len(result.accepted) == 1
    rod = result.accepted[0]
    assert rod.width_nm == pytest.approx(1000.0, abs=250.0)
    assert rod.length_nm == pytest.approx(3400.0, abs=400.0)

    # The pile-up must not be allowed to set the bridging length: measured on
    # the localizations it reads far too short and the cell falls apart.
    from_locs = detect_rods(locs, cfg)
    assert from_locs.stats["smooth_nm"] < result.stats["smooth_nm"]


def test_seed_level_defaults_to_a_fraction_of_the_minimum_half_width():
    cfg = _cfg(min_width_nm=800.0, seed_level_fraction=0.85)
    assert seed_level_for(cfg) == pytest.approx(340.0)
    assert seed_level_for(_cfg(seed_level_nm=200.0)) == pytest.approx(200.0)


def test_capsule_outline_traces_the_detected_cell():
    rng = np.random.default_rng(6)
    pts = _capsule_points(rng, 9000, (1000.0, -500.0), 3000.0, 1000.0, 25.0)
    rod = detect_rods(pts, _cfg()).accepted[0]

    outline = capsule_outline(rod)

    assert outline.shape[1] == 2
    assert np.allclose(outline[0], outline[-1])
    # Every outline vertex sits half a width from the centreline.
    p0, p1 = rod.endpoints_nm
    seg = p1 - p0
    t = np.clip(((outline - p0) @ seg) / max(float(seg @ seg), 1e-9), 0.0, 1.0)
    dist = np.linalg.norm(outline - (p0 + t[:, None] * seg), axis=1)
    assert np.allclose(dist, 0.5 * rod.width_nm, atol=1e-6)


def test_oversized_grid_raises_instead_of_allocating():
    pts = np.array([[0.0, 0.0], [3e6, 3e6], [0.0, 3e6]])
    with pytest.raises(ValueError, match="pixel size"):
        detect_rods(pts, _cfg(pixel_size_nm=1.0))


def test_non_finite_rows_belong_to_no_region():
    rng = np.random.default_rng(7)
    pts = _capsule_points(rng, 6000, (0.0, 0.0), 3000.0, 1000.0)
    pts = np.vstack([pts, [[np.nan, 0.0], [0.0, np.inf]]])

    result = detect_rods(pts, _cfg())

    assert len(result.accepted) == 1
    assert result.component_of_point.shape == (pts.shape[0],)
    assert np.all(result.component_of_point[-2:] == -1)


# --- integration with the staged HlyB/D workflow --------------------------

def _rod_dataset(rng, *, centers, length=3000.0, width=1000.0,
                 n_sites=340, locs_per_trace=12, pair_nm=14.0, pair_fraction=0.45):
    """Sites on the membrane of one or more rods, some as short-range pairs.

    Returns raw metre localizations, trace ids and times, as the staged entry
    point consumes them.
    """
    loc, tid, tim = [], [], []
    trace = 0
    for cx, cy in centers:
        half = 0.5 * (length - width)
        u = rng.uniform(-half, half, n_sites)
        theta = rng.uniform(0, 2 * np.pi, n_sites)
        r = 0.5 * width
        sites = np.column_stack([u + cx, r * np.cos(theta) + cy, r * np.sin(theta)])
        partner = rng.random(n_sites) < pair_fraction
        step = rng.normal(size=(int(partner.sum()), 3))
        step /= np.linalg.norm(step, axis=1, keepdims=True)
        sites = np.vstack([sites, sites[partner] + pair_nm * step])
        for site in sites:
            spread = site + rng.normal(0.0, 1.2, (locs_per_trace, 3))
            loc.append(spread)
            tid.append(np.full(locs_per_trace, trace))
            tim.append(np.full(locs_per_trace, float(trace)))
            trace += 1
    loc = np.vstack(loc)
    # The workflow expects metres and applies the z scaling itself.
    loc = loc * 1e-9
    return loc, np.concatenate(tid), np.concatenate(tim)


def _staged_cfg(**kwargs):
    from minflux_viewer.analysis.hlyb_staged import Staged3DConfig

    base = dict(component_mode="rod", z_scaling_factor=1.0, min_loc_per_trace=5,
                min_sites_per_component=20, null_replicates=9,
                run_sensitivity=False, run_stratum_profile=False,
                rod_pixel_size_nm=25.0)
    base.update(kwargs)
    return Staged3DConfig(**base)


def test_staged_analysis_runs_on_detected_rod_components():
    from minflux_viewer.analysis.hlyb_staged import analyze_hlyb_staged_3d

    rng = np.random.default_rng(11)
    loc, tid, tim = _rod_dataset(rng, centers=[(0.0, 0.0), (0.0, 4000.0)])

    result = analyze_hlyb_staged_3d(loc, tid, tim, _staged_cfg())

    assert result["component_mode"] == "rod"
    assert result["n_components"] == 2
    summary = result["rod_segmentation"]
    assert summary["n_accepted"] == 2
    assert summary["median_width_nm"] == pytest.approx(1000.0, abs=150.0)
    # The planted short-range population must survive the new component route.
    assert result["summary"]["band_ratio"] > 1.0
    assert result["summary"]["positive_excess_centroid_nm"] == pytest.approx(
        14.0, abs=3.0)


def test_rod_mode_takes_the_null_axis_from_the_measured_cell_axis():
    from minflux_viewer.analysis.hlyb_staged import (
        rod_config_for, segment_rod_components)

    rng = np.random.default_rng(12)
    # A rod along 30 degrees whose *sites* are deliberately sparse, so a
    # per-component PCA axis is a poorer estimate than the fitted cell axis.
    angle = np.radians(30.0)
    u = rng.uniform(-1200.0, 1200.0, 260)
    theta = rng.uniform(0, 2 * np.pi, 260)
    local = np.column_stack([u, 500.0 * np.cos(theta), 500.0 * np.sin(theta)])
    rot = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0]])
    sites = local @ rot.T
    dense = sites[rng.integers(0, sites.shape[0], 9000)] + rng.normal(0, 25.0, (9000, 3))

    cfg = _staged_cfg()
    segmented = segment_rod_components(
        sites, dense, rod_cfg=rod_config_for(cfg), min_sites=20, use_axis=True)

    assert len(segmented["components"]) == 1
    axis = segmented["components"][0]["axes"][:, 0]
    assert abs(float(axis[2])) < 1e-12          # the cell axis stays in-plane
    measured = np.degrees(np.arctan2(axis[1], axis[0])) % 180.0
    assert measured == pytest.approx(30.0, abs=6.0)


def test_rod_mode_sensitivity_audit_varies_the_width_window():
    from minflux_viewer.analysis.hlyb_staged import analyze_hlyb_staged_3d

    rng = np.random.default_rng(13)
    loc, tid, tim = _rod_dataset(rng, centers=[(0.0, 0.0), (0.0, 4000.0)])

    result = analyze_hlyb_staged_3d(
        loc, tid, tim,
        _staged_cfg(run_sensitivity=True, sensitivity_replicates=9,
                    sensitivity_site_merge_nm=(4.0,),
                    sensitivity_stratum_sites=(64,)))

    sources = {row["source"] for row in result["sensitivity"]}
    assert "rod width window" in sources
    assert "component link" not in sources
    scales = sorted({row["rod_width_scale"] for row in result["sensitivity"]})
    assert scales == [0.9, 1.0, 1.1]


def test_link_mode_is_unchanged_and_still_the_default():
    from minflux_viewer.analysis.hlyb_staged import Staged3DConfig

    assert Staged3DConfig().component_mode == "link"


def test_rod_mode_rejects_a_pixel_too_coarse_to_resolve_the_width():
    from minflux_viewer.analysis.hlyb_staged import (
        Staged3DConfig, analyze_hlyb_staged_3d)

    rng = np.random.default_rng(14)
    loc, tid, tim = _rod_dataset(rng, centers=[(0.0, 0.0)])
    cfg = Staged3DConfig(component_mode="rod", rod_pixel_size_nm=200.0)
    with pytest.raises(ValueError, match="eighth of the minimum"):
        analyze_hlyb_staged_3d(loc, tid, tim, cfg)


def test_no_accepted_cell_reports_why():
    from minflux_viewer.analysis.hlyb_staged import analyze_hlyb_staged_3d

    rng = np.random.default_rng(15)
    loc, tid, tim = _rod_dataset(rng, centers=[(0.0, 0.0)])
    cfg = _staged_cfg(rod_min_width_nm=2000.0, rod_max_width_nm=2600.0)
    with pytest.raises(ValueError, match="too narrow"):
        analyze_hlyb_staged_3d(loc, tid, tim, cfg)


# --- UI and method text ---------------------------------------------------

def test_dialog_round_trips_rod_settings_and_greys_out_the_unused_knob(qtbot):
    from minflux_viewer.analysis.hlyb_staged import Staged3DConfig
    from minflux_viewer.ui.hlyb_staged_dialog import HlyBStagedDialog

    dialog = HlyBStagedDialog(defaults=Staged3DConfig())
    qtbot.addWidget(dialog)

    # Link mode is the default: the rod knobs are inert.
    assert dialog.component_mode() == "link"
    assert dialog._cell_link.isEnabled()
    assert not dialog._rod_width.isEnabled()

    dialog._component_mode.setCurrentIndex(
        dialog._component_mode.findData("rod"))
    assert not dialog._cell_link.isEnabled()
    assert dialog._rod_width.isEnabled()
    assert dialog._rod_pixel.isEnabled()

    dialog._rod_min_width.setValue(850.0)
    dialog._rod_max_width.setValue(1150.0)
    cfg = dialog.config()
    assert cfg.component_mode == "rod"
    assert cfg.rod_min_width_nm == pytest.approx(850.0)
    assert cfg.rod_max_width_nm == pytest.approx(1150.0)
    assert cfg.rod_use_axis is True


def test_dialog_clamps_a_pixel_too_coarse_to_resolve_the_width(qtbot):
    from minflux_viewer.analysis.hlyb_staged import Staged3DConfig
    from minflux_viewer.ui.hlyb_staged_dialog import HlyBStagedDialog

    dialog = HlyBStagedDialog(defaults=Staged3DConfig(component_mode="rod"))
    qtbot.addWidget(dialog)
    dialog._rod_min_width.setValue(800.0)
    dialog._rod_pixel.setValue(180.0)
    dialog._accept_if_valid()

    assert dialog.config().rod_pixel_size_nm == pytest.approx(100.0)


def test_result_window_reports_the_detection_including_rejections(qtbot):
    from minflux_viewer.analysis.hlyb_staged import analyze_hlyb_staged_3d
    from minflux_viewer.ui.hlyb_staged_dialog import HlyBStagedWindow

    rng = np.random.default_rng(21)
    loc, tid, tim = _rod_dataset(rng, centers=[(0.0, 0.0), (0.0, 4000.0)])
    result = analyze_hlyb_staged_3d(loc, tid, tim, _staged_cfg())

    window = HlyBStagedWindow(result, title="synthetic")
    qtbot.addWidget(window)
    report = window.layout().itemAt(1).widget().widget(2).toPlainText()

    assert "ROD CELL DETECTION" in report
    assert "Width window:" in report
    assert "Measured widths of all regions" in report
    # The outline overlay is offered because a detection is present.
    assert window._rod_check.isEnabled()


def test_result_window_hides_the_outline_toggle_without_a_detection(qtbot):
    from minflux_viewer.analysis.hlyb_staged import analyze_hlyb_staged_3d
    from minflux_viewer.ui.hlyb_staged_dialog import HlyBStagedWindow

    rng = np.random.default_rng(22)
    loc, tid, tim = _rod_dataset(rng, centers=[(0.0, 0.0)])
    result = analyze_hlyb_staged_3d(
        loc, tid, tim, _staged_cfg(component_mode="link", cell_link_nm=180.0))

    window = HlyBStagedWindow(result, title="synthetic")
    qtbot.addWidget(window)
    report = window.layout().itemAt(1).widget().widget(2).toPlainText()

    assert "ROD CELL DETECTION" not in report
    assert not window._rod_check.isEnabled()


def test_method_text_describes_the_rod_segmentation_it_actually_used():
    from minflux_viewer.analysis.method_text import generate_method_text

    method = {
        "schema": "hlyb_staged_short_range_3d/v1",
        "input": {"dataset_name": "cells", "n_localizations": 100000,
                  "n_traces_total": 3900, "n_traces_used": 3800},
        "parameters": {
            "min_loc_per_trace": 10, "z_scaling_factor": 0.67,
            "site_merge_nm": 4.0, "component_mode": "rod",
            "cell_link_nm": 180.0, "min_sites_per_component": 20,
            "rod_min_width_nm": 800.0, "rod_max_width_nm": 1100.0,
            "rod_min_length_nm": 1000.0, "rod_max_length_nm": 8000.0,
            "rod_pixel_size_nm": 20.0, "rod_use_axis": True,
            "r_max_nm": 60.0, "bin_nm": 0.5, "short_range_lo_nm": 8.0,
            "short_range_hi_nm": 25.0, "null_stratum_sites": 64,
            "null_replicates": 99, "bootstrap_replicates": 399,
        },
        "site_inference": {"n_sites": 3300, "n_sites_used": 3100,
                           "n_repeated_sites": 400,
                           "n_traces_consolidated": 500,
                           "median_within_site_rms_nm": 1.1},
        "components": {"mode": "rod", "n_retained": 3, "n_all": 5,
                       "n_rod_like": 3, "n_excluded_sites": 200},
        "rod_segmentation": {"n_regions": 5, "n_accepted": 3, "n_split": 1},
        "result": {"band_observed_pairs": 400.0, "band_null_mean_pairs": 200.0,
                   "band_null_sd_pairs": 12.0, "band_ratio": 2.0,
                   "band_ratio_z": 20.0, "null_band_ratio_mean": 1.0,
                   "null_band_ratio_sd": 0.05, "band_p": 0.01,
                   "band_p_resolution": 0.01, "band_z": 15.0,
                   "peak_nm": 13.0, "positive_excess_centroid_nm": 13.5,
                   "positive_excess_median_nm": 13.2, "max_pointwise_z": 8.0,
                   "max_pointwise_p": 0.01},
        "sensitivity": [
            {"source": "rod width window", "rod_width_scale": 0.9,
             "band_ratio": 1.9, "band_p": 0.01,
             "positive_excess_centroid_nm": 13.2},
            {"source": "rod width window", "rod_width_scale": 1.1,
             "band_ratio": 2.1, "band_p": 0.01,
             "positive_excess_centroid_nm": 13.9},
        ],
        "limitations": [],
    }
    state = type("S", (), {"log_history": [], "datasets": []})()
    events = [{"message": "HlyB/D subunit pair analysis on 'cells': done.",
               "method_data": method, "level": "INFO"}]

    text = generate_method_text(state, events)

    assert "delineated in the XY projection" in text
    assert "800–1100 nm wide" in text
    assert "fitted long axis" in text
    assert "accepted cell-width window over 90 % and 110 %" in text
    # The link distance played no part, so it must not be quoted as a setting.
    assert "Component link distance" not in text
    assert "Cell width window (nm), upper" in text


def test_runner_log_line_and_payload_carry_the_rod_detection():
    from types import SimpleNamespace

    from minflux_viewer.analysis.hlyb_staged import analyze_hlyb_staged_3d
    from minflux_viewer.analysis.method_text import generate_method_text
    from minflux_viewer.plugins.hlyb_pair_analysis.runner import (
        _log_line, _method_payload)

    rng = np.random.default_rng(23)
    loc, tid, tim = _rod_dataset(rng, centers=[(0.0, 0.0), (0.0, 4000.0)])
    cfg = _staged_cfg()
    result = analyze_hlyb_staged_3d(loc, tid, tim, cfg)
    ds = SimpleNamespace(name="cells", metadata={}, file=None)

    line = _log_line(ds, cfg, result)
    assert "rod cell(s) detected" in line
    assert "800–1100 nm wide" in line

    payload = _method_payload(ds, cfg, result, n_localizations=loc.shape[0],
                              has_time=True)
    assert payload["parameters"]["component_mode"] == "rod"
    assert payload["parameters"]["rod_min_width_nm"] == pytest.approx(800.0)
    assert payload["components"]["mode"] == "rod"
    assert payload["rod_segmentation"]["n_accepted"] == 2

    # End to end: the real log line plus the real payload must still generate
    # method text, and it must describe the segmentation that was used.
    state = type("S", (), {"log_history": [], "datasets": []})()
    text = generate_method_text(
        state, [{"message": line, "method_data": payload, "level": "INFO"}])
    assert "delineated in the XY projection" in text
    assert "fitted long axis" in text
