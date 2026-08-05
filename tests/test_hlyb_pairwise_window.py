"""UI regressions for the HlyB pair-distance model fit, and for the retirement
of the plain 2-D / 3-D menu entries."""

from __future__ import annotations

import sys

import numpy as np
import pyqtgraph as pg
import pytest


@pytest.fixture
def _app():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(scope="module")
def result():
    """Simulated HlyB complexes, so the spatial view has real pairs to draw.

    A structureless cloud would leave the 8-14 nm band empty and the link tests
    would pass vacuously.
    """
    from minflux_viewer.analysis.hlyb_clustering import HlyBConfig, hlyb_template_model
    from minflux_viewer.analysis.hlyb_pairwise import PairFitConfig, analyze_hlyb_pairwise

    template = hlyb_template_model(HlyBConfig())["template_coords_nm"]
    rng = np.random.default_rng(3)
    pts, tids, tim, trace, clock = [], [], [], 0, 0.0
    for _ in range(150):
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        w, x, y, z = q
        rot = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        sites = (template @ rot.T) + rng.uniform(0, 2000, size=3)
        for site in sites:
            if rng.random() > 0.7:
                continue
            for _ in range(2):
                pts.append(site + rng.normal(scale=2.0, size=(20, 3)))
                tids.append(np.full(20, trace)); trace += 1
                tim.append(clock + np.arange(20) * 1e-3)
                clock += 0.05
    return analyze_hlyb_pairwise(np.concatenate(pts) * 1e-9,
                                 np.concatenate(tids), np.concatenate(tim),
                                 PairFitConfig(min_loc_per_trace=5,
                                               z_scaling_factor=1.0,
                                               null_replicates=2))


def _main_window(_app):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.main_window import MainWindow
    state = AppState()
    state.prefs.setdefault("data", {}).update({"show_data_info": False,
                                               "show_render": False})
    return MainWindow(state)


def test_dialog_builds_and_returns_a_config(_app):
    from minflux_viewer.analysis.hlyb_pairwise import PairFitConfig
    from minflux_viewer.ui.hlyb_pairwise_dialog import HlyBPairwiseDialog

    dlg = HlyBPairwiseDialog(None, defaults=PairFitConfig(z_scaling_factor=0.67))
    try:
        cfg = dlg.config()
        assert isinstance(cfg, PairFitConfig)
        assert cfg.z_scaling_factor == pytest.approx(0.67)
        # the fit window must never exceed the profile range the user asked for
        assert cfg.fit_r_max_nm <= cfg.r_max_nm
    finally:
        dlg.deleteLater()


def test_window_is_an_unparented_modeless_dialog(result, _app):
    from PyQt6.QtWidgets import QDialog
    from minflux_viewer.ui.hlyb_pairwise_dialog import HlyBPairwiseWindow

    win = HlyBPairwiseWindow(result, title="t")
    try:
        assert isinstance(win, QDialog)
        assert win.parent() is None
        assert not win.isModal()
    finally:
        win.close()
        win.deleteLater()


def test_window_view_toggles_do_not_raise(result, _app):
    from minflux_viewer.ui.hlyb_pairwise_dialog import HlyBPairwiseWindow

    win = HlyBPairwiseWindow(result, title="toggles")
    try:
        for widget in (win._show_excess, win._show_components, win._show_null):
            widget.setChecked(not widget.isChecked())
            widget.setChecked(not widget.isChecked())
        assert win._plot is not None
    finally:
        win.close()
        win.deleteLater()


def test_scatter_offers_the_projections_and_draws_the_band_links(result, _app):
    from minflux_viewer.ui.hlyb_pairwise_dialog import HlyBPairwiseWindow

    win = HlyBPairwiseWindow(result, title="scatter")
    try:
        assert [win._view_combo.itemText(i) for i in range(win._view_combo.count())] \
            == ["XY", "XZ", "YZ", "3D"]
        for view in ("XZ", "YZ", "XY"):
            win._view_combo.setCurrentText(view)
            assert win._current_view == view
            page = win._scatter_pages[view]
            xs, _ = page["links"].getData()
            assert xs is not None and len(xs) > 0, f"no links drawn in {view}"
    finally:
        win.close()
        win.deleteLater()


def test_links_are_drawn_as_disjoint_segments(result, _app):
    """Regression: the links were invisible when built on a PlotDataItem. Only
    PlotCurveItem honours a per-segment `connect` array, which is what keeps
    thousands of separate pair links from being joined into one polyline."""
    from minflux_viewer.ui.hlyb_pairwise_dialog import HlyBPairwiseWindow

    win = HlyBPairwiseWindow(result, title="segments")
    try:
        page = win._scatter_pages["XY"]
        assert isinstance(page["links"], pg.PlotCurveItem)
        xs, ys = page["links"].getData()
        # two vertices per pair, and every vertex is finite (no NaN separators)
        assert len(xs) % 2 == 0
        assert np.isfinite(xs).all() and np.isfinite(ys).all()
        connect = page["links"].opts.get("connect")
        assert isinstance(connect, np.ndarray)
        assert connect[0::2].max() == 1 and connect[1::2].max() == 0
        assert page["links"].opts.get("pen") is not None
    finally:
        win.close()
        win.deleteLater()


def test_band_region_and_spinboxes_stay_in_sync(result, _app):
    from minflux_viewer.analysis.hlyb_pairwise import pairs_in_band
    from minflux_viewer.ui.hlyb_pairwise_dialog import HlyBPairwiseWindow

    win = HlyBPairwiseWindow(result, title="band")
    try:
        win._band_region.setRegion((16.0, 20.0))
        assert win._band_lo_spin.value() == pytest.approx(16.0)
        assert win._band_hi_spin.value() == pytest.approx(20.0)
        expected = pairs_in_band(result["centroids_nm"], 16.0, 20.0).shape[0]
        assert f"{expected:,}" in win._band_count.text()

        win._band_lo_spin.setValue(5.0)
        lo, hi = win._band_region.getRegion()
        assert lo == pytest.approx(5.0)
    finally:
        win.close()
        win.deleteLater()


def test_band_region_survives_a_profile_redraw(result, _app):
    """The profile redraw calls clear(), which drops every item; the band
    selector has to be put back or it silently disappears on the first toggle."""
    from minflux_viewer.ui.hlyb_pairwise_dialog import HlyBPairwiseWindow

    win = HlyBPairwiseWindow(result, title="redraw")
    try:
        win._show_components.setChecked(False)
        win._show_excess.setChecked(True)
        assert win._band_region in win._plot.items
    finally:
        win.close()
        win.deleteLater()


def test_toggling_layers_off_clears_them(result, _app):
    from minflux_viewer.ui.hlyb_pairwise_dialog import HlyBPairwiseWindow

    win = HlyBPairwiseWindow(result, title="layers")
    try:
        win._band_check.setChecked(False)
        xs, _ = win._scatter_pages["XY"]["links"].getData()
        assert xs is None or len(xs) == 0
        win._raw_check.setChecked(False)
        assert len(win._scatter_pages["XY"]["raw"].getData()[0]) == 0
    finally:
        win.close()
        win.deleteLater()


def test_report_states_the_limits_rather_than_implying_a_resolved_distance(result):
    from minflux_viewer.ui.hlyb_pairwise_dialog import pairwise_report

    text = pairwise_report(result)
    assert "SAME-SITE SHORT-RANGE KERNEL" in text
    assert "MODEL COMPARISON" in text
    assert "SENSITIVITY" in text
    # the report must not let a reader think individual classes were resolved
    assert "NOT resolved" in text
    # and must disclose the fixed class weights that make the fit a test
    assert "fixed at 1/5" in text


def test_report_flags_a_parameter_resting_on_a_bound():
    """A fitted value at its bound is a limit, not an estimate, and saying so
    is the difference between reporting a measurement and overstating one."""
    from minflux_viewer.ui.hlyb_pairwise_dialog import pairwise_report

    fake = {
        "n_traces_total": 10, "n_traces_used": 8, "excess_outer_nm": 12.0,
        "centroid_sem_nm": np.array([1.0, 1.0, 1.0]), "sigma_floor_nm": 1.4,
        "null_replicates": 4,
        "repeat_kernel": {"source": "assumed", "n_pairs": 0, "median_nm": 3.0,
                          "rejected_far_fraction": float("nan")},
        "fits": {"six_site": {"delta_aic": 0.0, "n_complex_pairs": 100.0,
                              "label_offset_nm": 0.0, "sigma_nm": 2.0,
                              "parameters_at_bounds": ["label_offset_nm"]}},
        "best_fit": {"label_offset_nm": 0.0,
                     "parameters_at_bounds": ["label_offset_nm"]},
        "fits_relaxed_kernel": {},
        "class_distances_nm": [], "best_hypothesis": "six_site",
    }
    text = pairwise_report(fake)
    assert "sits at a bound" in text
    assert "label_offset_nm" in text


def test_hlyb_menu_offers_the_new_method_and_no_longer_the_plain_2d_3d(_app):
    """The plain 2-D / 3-D entries are retired; their analysis code stays in
    hlyb_clustering.py so a proper 2-D workflow can be rebuilt on it."""
    win = _main_window(_app)
    try:
        labels = [a.text() for a in win.menuHlyBPair.actions()]
        assert "Pair-distance model fit (3D)" in labels
        assert "Template matching (3D)" in labels
        assert "2D" not in labels
        assert "3D" not in labels
        assert not hasattr(win, "actionHlyB2D")
        assert not hasattr(win, "actionHlyB3D")
    finally:
        win.close()
        _app.processEvents()


def test_retired_analysis_functions_are_still_importable():
    """Menu removal must not delete the logic -- the 2-D cell delineation is
    explicitly kept for a future 2-D workflow."""
    from minflux_viewer.analysis.hlyb_clustering import (  # noqa: F401
        analyze_hlyb,
        analyze_hlyb_2d,
        compute_cell_mask,
    )


def test_new_command_is_traceable_to_its_source(_app):
    from minflux_viewer.ui import command_finder

    win = _main_window(_app)
    try:
        entries = command_finder.collect_commands(win.menuBar())
        match = [e for e in entries if e.text == "Pair-distance model fit (3D)"]
        assert match, "the new command must appear in the command finder"
        assert "hlyb_pairwise.py" in (match[0].source or "")
    finally:
        win.close()
        _app.processEvents()
