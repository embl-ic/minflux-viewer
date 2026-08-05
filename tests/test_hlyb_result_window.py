"""Regression: the HlyB result window must be a modeless-capable QDialog
(``show_modeless`` calls ``setModal``, which QWidget lacks)."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtWidgets import QApplication, QDialog, QMenu, QWidget

from minflux_viewer.analysis.hlyb_clustering import HlyBConfig, analyze_hlyb
from minflux_viewer.ui.hlyb_clustering_dialog import HlyBClusteringDialog, HlyBResultWindow
from minflux_viewer.ui.modeless import show_modeless


@pytest.fixture(scope="module")
def _app():
    app = QApplication.instance() or QApplication([])
    yield app


def _synthetic_result():
    tri = np.array([[0, 0, 0], [18, 0, 0], [9, 15.6, 0]], dtype=float)
    rng = np.random.default_rng(0)
    locs, tids, tid = [], [], 1
    for base in (np.array([0.0, 0, 0]), np.array([300.0, 0, 0])):
        for off in tri:
            pts = base + off + rng.normal(0, 1.0, size=(40, 3)) * np.array([1, 1, 0])
            locs.append(pts)
            tids.append(np.full(40, tid))
            tid += 1
    loc_m = np.vstack(locs) / 1e9
    tid_values = np.concatenate(tids)
    result = analyze_hlyb(
        loc_m, tid_values,
        HlyBConfig(basic_unit_size_nm=8.0, z_scaling_factor=1.0),
    )
    result["raw_color_attributes"] = {
        "tid": tid_values,
        "efo": np.linspace(10_000.0, 40_000.0, tid_values.size),
        "den": np.linspace(1.0, 25.0, tid_values.size),
    }
    return result


def test_result_window_is_modeless_capable(_app):
    win = HlyBResultWindow(_synthetic_result(), HlyBConfig(), title="t")
    assert isinstance(win, QDialog)          # QWidget has no setModal()
    owner = QWidget()
    show_modeless(win, owner)                # would raise AttributeError on a bare QWidget
    assert win.isModal() is False
    assert win in owner._modeless_windows
    win.close()


def test_scatter_object_controls_and_subunit_datatip_in_2d(_app):
    win = HlyBResultWindow(_synthetic_result(), HlyBConfig(), title="objects")
    win._view_combo.setCurrentText("XY")
    page = win._scatter_pages["XY"]

    assert win._raw_loc_checkbox.text() == "raw loc"
    assert win._subunit_detection_checkbox.text() == "sub-unit detection"
    assert win._template_match_checkbox.text() == "template match"
    assert win._pair_link_checkbox.text() == "pair link"
    assert page["raw_item"].isVisible()
    assert page["subunit_item"].isVisible()
    assert page["link_item"].isVisible()
    assert page["subunit_item"].opts["tip"] is None
    assert win._scatter_color_by == "tid"
    raw_colors = win._raw_point_colors(page["raw_indices"])
    assert np.unique(raw_colors[:, :3], axis=0).shape[0] > 1

    # pyqtgraph emits an empty NumPy array when hover state is cleared while
    # switching from the 3-D page to a newly shown 2-D projection.
    win._on_subunit_hover(None, np.empty(0, dtype=object), None)

    win._raw_loc_checkbox.setChecked(False)
    assert not page["raw_item"].isVisible()
    win._subunit_detection_checkbox.setChecked(False)
    assert not page["subunit_item"].isVisible()
    win._pair_link_checkbox.setChecked(False)
    assert not page["link_item"].isVisible()

    win._template_match_checkbox.setChecked(False)
    colors = win._subunit_colors()
    assert np.allclose(colors, colors[:1])
    tip = win._subunit_tooltip(0)
    assert "X:" in tip and "(nm)" in tip
    assert "Y:" in tip and "Z:" in tip
    assert "Template ID:" in tip
    win.close()


def test_scatter_context_menu_matches_loc_scatter_items(_app, monkeypatch):
    win = HlyBResultWindow(_synthetic_result(), HlyBConfig(), title="menu")
    win._view_combo.setCurrentText("XY")
    plot = win._scatter_pages["XY"]["plot"]
    captured = []
    monkeypatch.setattr(QMenu, "exec", lambda menu, _pos: captured.append(menu))

    win._show_scatter_context_menu(QPoint(5, 5), plot)
    menu = captured[0]
    labels = [action.text() for action in menu.actions() if not action.isSeparator()]
    assert labels == [
        "View as", "Color by", "Colormap", "Black Background", "Reset View"
    ]
    submenus = {action.text(): action.menu() for action in menu.actions() if action.menu()}
    assert [action.text() for action in submenus["View as"].actions()] == [
        "XY", "XZ", "YZ", "3D"
    ]
    assert [action.text() for action in submenus["Color by"].actions()] == [
        "tid", "efo", "den"
    ]
    assert not {"Template ID", "X", "Y", "Z"} & {
        action.text() for action in submenus["Color by"].actions()
    }
    assert plot.getPlotItem().vb.menuEnabled() is False

    subunit_colors = win._subunit_colors().copy()
    win._set_scatter_color_by("efo")
    win._set_scatter_colormap("hot")
    win._set_scatter_black_background(True)
    assert win._scatter_color_by == "efo"
    assert win._scatter_colormap == "hot"
    assert win._scatter_black_background is True
    assert np.allclose(win._subunit_colors(), subunit_colors)
    win.close()


def test_source_dataset_populates_same_raw_color_attributes_as_loc_scatter(_app):
    from minflux_viewer.core.attributes import plot_attribute_names
    from minflux_viewer.core.dataset import build_localization_dataset

    result = _synthetic_result()
    result.pop("raw_color_attributes")
    points = np.asarray(result["points_nm"], dtype=float)
    count = points.shape[0]
    prefs = {
        "attributes": {
            "enabled": ["loc", "tid", "tim", "efo"],
            "computed": ["siz", "dur", "len"],
        }
    }
    ds = build_localization_dataset(
        name="hlyb-colors",
        x_nm=points[:, 0], y_nm=points[:, 1], z_nm=points[:, 2],
        tid=np.repeat(np.arange(1, count // 40 + 1), 40)[:count],
        tim=np.arange(count, dtype=float),
        attrs={"efo": np.linspace(1_000.0, 20_000.0, count)},
        prefs=prefs,
    )
    expected = plot_attribute_names(ds, prefs, exclude=("ftr", "idx"))
    win = HlyBResultWindow(
        result, HlyBConfig(), title="dataset attributes",
        source_dataset=ds, prefs=prefs,
    )

    assert list(win._raw_color_attributes) == expected
    assert win._scatter_color_by == "tid"
    assert np.array_equal(win._raw_color_attributes["tid"], ds.attr["tid"])
    assert "efo" in win._raw_color_attributes
    assert "siz" in win._raw_color_attributes
    win.close()


def test_pair_distance_labels_appear_when_zoomed_and_follow_pair_link(_app):
    win = HlyBResultWindow(_synthetic_result(), HlyBConfig(), title="labels")
    win._view_combo.setCurrentText("XY")
    win.show()
    _app.processEvents()
    page = win._scatter_pages["XY"]
    vb = page["plot"].getPlotItem().vb

    vb.setRange(xRange=(-1000.0, 1000.0), yRange=(-1000.0, 1000.0), padding=0.0)
    win._refresh_2d_pair_labels("XY")
    assert not any(item.isVisible() for item in page["label_items"].values())

    vb.setRange(xRange=(-5.0, 25.0), yRange=(-5.0, 25.0), padding=0.0)
    win._refresh_2d_pair_labels("XY")
    visible = [item for item in page["label_items"].values() if item.isVisible()]
    assert visible
    assert all("nm" not in item.textItem.toPlainText() for item in visible)
    assert all("." in item.textItem.toPlainText() for item in visible)

    win._pair_link_checkbox.setChecked(False)
    assert not page["link_item"].isVisible()
    assert not any(item.isVisible() for item in page["label_items"].values())
    win.close()


def test_scatter_controls_apply_to_3d_when_opengl_is_available(_app):
    win = HlyBResultWindow(_synthetic_result(), HlyBConfig(), title="3d objects")
    page = win._scatter_pages.get("3D")
    if not page or page.get("kind") != "3d":
        win.close()
        pytest.skip("OpenGL result view unavailable")

    assert page["view"].contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu
    win._raw_loc_checkbox.setChecked(False)
    win._pair_link_checkbox.setChecked(False)
    win._subunit_detection_checkbox.setChecked(False)
    assert not page["raw_item"].visible()
    assert not page["link_item"].visible()
    assert not page["subunit_item"].visible()
    win.close()


def test_switch_from_zoomed_3d_preserves_center_and_scale_in_xy(_app):
    win = HlyBResultWindow(_synthetic_result(), HlyBConfig(), title="shared view")
    win.show()
    _app.processEvents()
    page_3d = win._scatter_pages.get("3D")
    if not page_3d or page_3d.get("kind") != "3d":
        win.close()
        pytest.skip("OpenGL result view unavailable")

    view = page_3d["view"]
    target_center = page_3d["anchor"] + np.array([24.0, -17.0, 9.0])
    local_center = target_center - page_3d["anchor"]
    distance = 72.0
    view.setCameraPosition(pos=pg.Vector(*local_center), distance=distance)
    source_height = max(float(view.height()), 1.0)
    fov = float(view.opts.get("fov", 60.0))
    expected_scale = 2.0 * distance * np.tan(np.deg2rad(fov) * 0.5) / source_height

    win._view_combo.setCurrentText("XY")
    _app.processEvents()
    xy_page = win._scatter_pages["XY"]
    (x0, x1), (y0, y1) = xy_page["plot"].getPlotItem().vb.viewRange()
    assert 0.5 * (x0 + x1) == pytest.approx(target_center[0], abs=1e-6)
    assert 0.5 * (y0 + y1) == pytest.approx(target_center[1], abs=1e-6)
    assert win._scatter_view_center_nm == pytest.approx(target_center)
    assert win._scatter_view_scale_nm_per_px == pytest.approx(expected_scale)

    # A second orthogonal switch keeps the common X location and retains the
    # last known Z location from the 3-D camera centre.
    win._view_combo.setCurrentText("XZ")
    _app.processEvents()
    xz_page = win._scatter_pages["XZ"]
    (x0, x1), (z0, z1) = xz_page["plot"].getPlotItem().vb.viewRange()
    assert 0.5 * (x0 + x1) == pytest.approx(target_center[0], abs=1e-6)
    assert 0.5 * (z0 + z1) == pytest.approx(target_center[2], abs=1e-6)
    win.close()


def test_result_splitter_gives_scatter_the_larger_initial_share(_app):
    win = HlyBResultWindow(_synthetic_result(), HlyBConfig(), title="splitter")
    win.show()
    _app.processEvents()
    sizes = win._result_splitter.sizes()
    assert sizes[0] >= 330
    assert sizes[0] > sizes[1]
    win.close()


def test_distance_histogram_bin_control_and_summary(_app):
    result = _synthetic_result()
    # Exercise the template-matching summary without depending on a match.
    result["template_matching"] = True
    result["model"] = {"prior_distances_nm": np.array([18.0, 30.0])}
    result["model_pair_tolerance_nm"] = 5.0
    win = HlyBResultWindow(result, HlyBConfig(), title="template")

    assert win._distance_bin_spin.minimum() == pytest.approx(0.1)
    assert win._distance_bin_spin.maximum() == pytest.approx(10.0)
    assert win._distance_bin_spin.singleStep() == pytest.approx(0.1)
    assert win._show_lognormal_fit_checkbox.text() == "show lognormal fit"
    assert not win._show_lognormal_fit_checkbox.isChecked()
    assert "Max-count bin:" in win._distance_stats_label.text()
    assert "Lognormal mean:" not in win._distance_stats_label.text()
    assert "median" not in win._summary_label().text().lower()
    assert not any(
        isinstance(item, pg.InfiniteLine)
        for item in win._distance_hist_plot.getPlotItem().items
    )

    win._distance_bin_spin.setValue(5.0)
    counts, edges = win._distance_histogram_data(win._pair_distances_nm, 5.0)
    assert counts.size > 0
    assert np.allclose(np.diff(edges), 5.0)
    assert edges[0] == pytest.approx(0.0)
    assert edges[-1] == pytest.approx(40.0)
    assert win._distance_hist_plot.viewRange()[0] == pytest.approx([0.0, 40.0])

    tip = win._distance_bar_item.tooltip_for_x(17.5)
    assert "Bin center:" in tip
    assert "Count:" in tip
    win.close()


def test_histogram_upper_bound_expands_for_returned_long_distances(_app):
    result = _synthetic_result()
    result["all_pair_distances"] = np.array([12.0, 18.0, 23.0, 690.0, 700.0])
    win = HlyBResultWindow(result, HlyBConfig(), title="long range")

    counts, edges = win._distance_histogram_data(result["all_pair_distances"], 10.0)
    assert counts.sum() == 5
    assert edges[-1] == pytest.approx(740.0)
    win._distance_bin_spin.setValue(10.0)
    assert win._distance_hist_plot.viewRange()[0] == pytest.approx([0.0, 740.0])
    win.close()


def test_show_all_overlays_exact_pre_template_pair_distribution(_app):
    result = _synthetic_result()
    win = HlyBResultWindow(result, HlyBConfig(), title="all pairs")

    assert win._show_all_pairs_checkbox.text() == "show all (remove template gating)"
    assert win._show_all_pairs_checkbox.isEnabled()
    win._distance_bin_spin.setValue(10.0)
    win._show_all_pairs_checkbox.setChecked(True)
    _app.processEvents()

    n_centers = result["subunit_centers"].shape[0]
    assert win._all_distance_bar_item is not None
    assert win._all_distance_bar_item._counts.sum() == n_centers * (n_centers - 1) // 2
    assert win._distance_bar_item._counts.sum() == result["all_pair_distances"].size
    assert "All max-count bin:" in win._distance_stats_label.text()
    assert "All detected subunits" in win._all_distance_bar_item.tooltip_for_x(15.0)
    assert win._distance_hist_plot.viewRange()[0][1] > 300.0
    win.close()


def test_show_lognormal_fit_checkbox_toggles_cached_fit_and_properties(_app):
    result = _synthetic_result()
    rng = np.random.default_rng(42)
    result["all_pair_distances"] = rng.lognormal(
        mean=np.log(18.0), sigma=0.25, size=5000)
    win = HlyBResultWindow(result, HlyBConfig(), title="lognormal")

    win._distance_bin_spin.setValue(1.0)
    win._show_lognormal_fit_checkbox.setChecked(True)
    _app.processEvents()

    fit = win._lognormal_fit_result
    assert fit is not None
    assert fit["mean_nm"] == pytest.approx(18.0 * np.exp(0.5 * 0.25**2), abs=1.0)
    assert np.isfinite(fit["mu"])
    assert fit["sigma"] > 0
    assert np.isfinite(fit["rmse_counts"])
    assert win._lognormal_curve_item is not None
    assert win._lognormal_mean_line.pen.style() == Qt.PenStyle.DashLine
    fit_text = win._lognormal_fit_text.textItem.toPlainText()
    assert "μ =" in fit_text
    assert "σ =" in fit_text
    assert "fit RMSE =" in fit_text
    assert "Lognormal mean:" in win._distance_stats_label.text()

    win._show_lognormal_fit_checkbox.setChecked(False)
    _app.processEvents()
    assert win._lognormal_fit_result is None
    assert win._lognormal_curve_item is None
    assert win._lognormal_mean_line is None

    # Re-showing the unchanged histogram reuses the fit computed on first check.
    win._show_lognormal_fit_checkbox.setChecked(True)
    _app.processEvents()
    assert win._lognormal_fit_result is fit
    win.close()


class _ViewPoint:
    def __init__(self, x, y):
        self._x = float(x)
        self._y = float(y)

    def x(self):
        return self._x

    def y(self):
        return self._y


def test_distance_histogram_replaces_default_menu_with_zoom_and_reset(_app):
    win = HlyBResultWindow(_synthetic_result(), HlyBConfig(), title="zoom menu")

    assert win.DISTANCE_ZOOM_MODES == ("horizontal", "vertical", "unconstrained")
    assert (
        win._distance_hist_plot.contextMenuPolicy()
        == Qt.ContextMenuPolicy.CustomContextMenu
    )
    assert win._distance_view_box.menuEnabled() is False

    win._distance_bin_spin.setValue(10.0)
    win._show_all_pairs_checkbox.setChecked(True)
    full_xmax = win._distance_hist_plot.viewRange()[0][1]
    assert full_xmax > 300.0

    win._toggle_distance_zoom_mode("horizontal")
    win._apply_distance_zoom_drag(_ViewPoint(0.0, 10.0), _ViewPoint(40.0, 50.0))
    (x0, x1), (y0, y1) = win._distance_view_box.viewRange()
    assert (x0, x1) == pytest.approx((0.0, 40.0))
    assert y0 == pytest.approx(0.0)
    assert y1 > y0
    assert win._distance_bin_spin.value() < 10.0

    # A manual bin-size change must re-bin in place without replacing the
    # user's inspected field of view.
    zoomed_range = win._distance_view_box.viewRange()
    win._distance_bin_spin.setValue(1.0)
    rebinned_range = win._distance_view_box.viewRange()
    assert rebinned_range[0] == pytest.approx(zoomed_range[0])
    assert rebinned_range[1] == pytest.approx(zoomed_range[1])

    win._show_lognormal_fit_checkbox.setChecked(True)
    fit_range = win._distance_view_box.viewRange()
    assert fit_range[0] == pytest.approx(rebinned_range[0])
    assert fit_range[1] == pytest.approx(rebinned_range[1])

    win._reset_distance_view()
    assert win._distance_zoom_mode is None
    assert win._distance_bin_spin.value() == pytest.approx(1.0)
    assert win._distance_hist_plot.viewRange()[0][1] > 300.0
    win.close()


def test_template_dialog_uses_consistent_core_parameters(_app):
    dlg = HlyBClusteringDialog(defaults=HlyBConfig(), mode="TEMPLATE3D")
    cfg = dlg.config()

    assert dlg._min_units.value() == 3
    assert cfg.template_core_a_ring_side_nm == pytest.approx(11.0)
    assert cfg.template_core_b_ring_side_nm == pytest.approx(19.0)
    assert cfg.template_core_twist_deg == pytest.approx(65.45, abs=0.01)
    assert cfg.template_label_offset_nm == pytest.approx(2.0)
    assert cfg.model_pair_tolerance_nm == pytest.approx(5.0)
    dlg.close()
