"""Orthogonal (XY / YZ / XZ) view mode for the coordinate views."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QApplication, QMenu

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.ui.ortho_view import (
    AXIS_COLUMNS,
    ORTHO_AXIS,
    ORTHO_AXIS_COLUMNS,
    PANE_CELLS,
    axis_columns,
    axis_labels,
    grid_stretch,
    ortho_pane_labels,
)


@pytest.fixture
def _qt_app():
    if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
        pytest.skip("No display available for Qt tests")
    return QApplication.instance() or QApplication(sys.argv)


def _dataset(*, three_d: bool = True, n: int = 600, name: str = "ortho"):
    rng = np.random.default_rng(7)
    return build_localization_dataset(
        name=name,
        x_nm=rng.normal(8000.0, 1500.0, n),
        y_nm=rng.normal(5000.0, 900.0, n),
        z_nm=rng.normal(300.0, 120.0, n) if three_d else np.zeros(n),
        tid=rng.integers(0, 60, n),
        source_version="simulation",
    )


def _state(**kwargs) -> AppState:
    state = AppState()
    state.add_dataset(_dataset(**kwargs))
    state.set_active(0)
    return state


def _window(app, state, *, size=(900, 900)):
    from minflux_viewer.ui.scatter_window import ScatterWindow

    win = ScatterWindow(state, dataset_idx=0)
    win.resize(*size)
    win.show()
    _settle(app)
    return win


def _settle(app, turns: int = 3) -> None:
    for _ in range(turns):
        app.processEvents()


def _ranges(win, plane):
    (x0, x1), (y0, y1) = win._ortho.view_box(plane).viewRange()
    return x0, x1, y0, y1


# ---------------------------------------------------------------------------
# Pure geometry
# ---------------------------------------------------------------------------

def test_pane_cells_put_xz_under_xy_and_yz_beside_it():
    """The layout is what makes the shared axes line up: XZ shares XY's
    column (so the same width), YZ shares XY's row (so the same height)."""
    assert PANE_CELLS["XY"] == (0, 0)
    assert PANE_CELLS["XZ"][1] == PANE_CELLS["XY"][1]   # same column as XY
    assert PANE_CELLS["XZ"][0] != PANE_CELLS["XY"][0]   # the row below
    assert PANE_CELLS["YZ"][0] == PANE_CELLS["XY"][0]   # same row as XY
    assert PANE_CELLS["YZ"][1] != PANE_CELLS["XY"][1]   # the column beside


def test_each_ortho_pane_shares_one_axis_with_xy():
    """Sharing an axis means sharing the *column*, in the same slot.

    The YZ pane is transposed against the standalone YZ projection for this
    reason: beside XY it must carry Y vertically. Getting it wrong draws Z
    data inside a Y range, which looks like squeezed content rather than an
    error, so it is asserted on the columns rather than on the ranges.
    """
    xy_h, xy_v = ORTHO_AXIS_COLUMNS["XY"]
    assert ORTHO_AXIS_COLUMNS["XZ"][0] == xy_h          # below: shares horizontal X
    assert ORTHO_AXIS_COLUMNS["XZ"][1] == 2             # ... its vertical is Z
    assert ORTHO_AXIS_COLUMNS["YZ"][1] == xy_v          # beside: shares vertical Y
    assert ORTHO_AXIS_COLUMNS["YZ"][0] == 2             # ... its horizontal is Z
    assert axis_labels("XZ", ortho=True) == ("X (nm)", "Z (nm)")
    assert axis_labels("YZ", ortho=True) == ("Z (nm)", "Y (nm)")

    # The standalone projections are untouched by the ortho transposition.
    assert AXIS_COLUMNS["YZ"] == (1, 2)
    assert axis_labels("YZ") == ("Y (nm)", "Z (nm)")
    assert axis_columns("YZ") == (1, 2)
    assert axis_columns("YZ", ortho=True) == (2, 1)


def test_leaving_ortho_zeroes_the_side_stretch():
    """Hiding the side panes is not enough to leave the mode.

    A hidden widget still holds its row/column open while a stretch factor is
    set, so the primary pane would keep only its share of the window.
    """
    rows, cols = grid_stretch(True, primary=3, side=1)
    assert rows == (3, 1) and cols == (3, 1)
    rows, cols = grid_stretch(False, primary=3, side=1)
    assert rows[1] == 0 and cols[1] == 0


# ---------------------------------------------------------------------------
# Scatter integration
# ---------------------------------------------------------------------------

def test_ortho_shows_three_aligned_panes(_qt_app):
    win = _window(_qt_app, _state())
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        assert win._ortho.active
        xy, yz, xz = (win._pane_plots[p] for p in ("XY", "YZ", "XZ"))
        assert xy.isVisible() and yz.isVisible() and xz.isVisible()
        # Exact alignment is the point of the grid: XZ is as wide as XY and
        # YZ is as tall, so a feature sits at the same screen position in the
        # pane that shares that axis.
        assert xz.width() == xy.width()
        assert yz.height() == xy.height()
        for plane in ("XY", "YZ", "XZ"):
            plot = win._pane_plots[plane]
            item = plot.getPlotItem()
            assert (item.getAxis("bottom").labelText,
                    item.getAxis("left").labelText) == ortho_pane_labels(plane)
    finally:
        win.close()


def test_linked_axes_agree_to_the_nanometre_after_a_zoom(_qt_app):
    """pyqtgraph maps a linked range through the ratio of the two panes'
    *screen* widths, so panes whose tick labels differ in width drift apart
    (measured at a stable 36-74 nm) unless the axis metrics are pinned."""
    win = _window(_qt_app, _state())
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        view = win._plot_2d.getPlotItem().getViewBox()
        view.setXRange(6000.0, 7000.0, padding=0)
        view.setYRange(4500.0, 5500.0, padding=0)
        _settle(_qt_app)

        xy = _ranges(win, "XY")
        yz = _ranges(win, "YZ")
        xz = _ranges(win, "XZ")
        assert xz[0] == pytest.approx(xy[0], abs=1e-6)   # shared X, low
        assert xz[1] == pytest.approx(xy[1], abs=1e-6)   # shared X, high
        assert yz[2] == pytest.approx(xy[2], abs=1e-6)   # shared Y, low
        assert yz[3] == pytest.approx(xy[3], abs=1e-6)   # shared Y, high
        # The two Z axes deliberately do NOT share a range — they share a
        # scale, and their pixel extents differ. See
        # test_both_side_panes_share_one_z_scale.
        z_centre_xz = 0.5 * (xz[2] + xz[3])
        z_centre_yz = 0.5 * (yz[0] + yz[1])
        assert z_centre_xz == pytest.approx(z_centre_yz, abs=1e-6)
    finally:
        win.close()


def test_the_side_panes_z_follows_the_xy_scale(_qt_app):
    """Zooming the primary pane carries Z with it, because the panes are
    isotropic — Z shrinks by exactly the factor the XY scale did."""
    win = _window(_qt_app, _state())
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        before = win._ortho.primary_scale_nm_per_px()
        z_before = _ranges(win, "XZ")[2:]
        win._plot_2d.getPlotItem().getViewBox().setXRange(6000.0, 6200.0, padding=0)
        _settle(_qt_app)
        after = win._ortho.primary_scale_nm_per_px()
        z_after = _ranges(win, "XZ")[2:]
        assert after < before                                  # zoomed in
        assert (z_after[1] - z_after[0]) < (z_before[1] - z_before[0])
        # Z shrank by exactly the factor the XY scale did — that is isotropy.
        assert ((z_after[1] - z_after[0]) / (z_before[1] - z_before[0])
                == pytest.approx(after / before, rel=1e-6))
    finally:
        win.close()


def test_every_pane_is_isotropic_with_xy(_qt_app):
    """Z at the XY pane's own nm/px, so a structure's proportions on screen
    are its proportions in the sample."""
    win = _window(_qt_app, _slab_state())
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        xy = win._ortho.primary_scale_nm_per_px()
        scales = _depth_scales(win)
        assert scales["XZ"] == pytest.approx(xy, rel=1e-6)
        assert scales["YZ"] == pytest.approx(xy, rel=1e-6)

        _zoom_xy(win, _qt_app, (-4000, -2000), (-4000, -2000))
        xy = win._ortho.primary_scale_nm_per_px()
        scales = _depth_scales(win)
        assert scales["XZ"] == pytest.approx(xy, rel=1e-6)
        assert scales["YZ"] == pytest.approx(xy, rel=1e-6)
    finally:
        win.close()


def test_the_z_scaling_factor_is_visible_not_merely_applied(_qt_app):
    """``loc_nm`` has always carried the calibrated Z, but a pane that
    auto-fitted Z to whatever it was handed rendered the change invisible:
    halving the factor halved the data and moved nothing on screen.

    With isotropy the pane's Z range is set by the XY zoom instead, so the
    structure really does draw half as thick.
    """
    state = _slab_state()
    ds = state.datasets[0]
    win = _window(_qt_app, state)
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        shown_before = _ranges(win, "XZ")[3] - _ranges(win, "XZ")[2]
        _xs, z_before = win._pane_scatters["XZ"].getData()
        drawn_before = float(z_before.max() - z_before.min())

        ds.set_z_scaling_factor(0.5, source="manual (test)")
        state.notify_calibration_changed(0)
        _settle(_qt_app, turns=6)

        _xs, z_after = win._pane_scatters["XZ"].getData()
        drawn_after = float(z_after.max() - z_after.min())
        shown_after = _ranges(win, "XZ")[3] - _ranges(win, "XZ")[2]

        assert drawn_after == pytest.approx(drawn_before / 2.0, rel=1e-3)
        # The window onto Z did NOT follow the data, so the halving is what the
        # eye sees. The tolerance is 2 % rather than exact because the window is
        # scale x pixels and the aspect-locked re-fit jitters the XY scale by
        # ~0.01 % — still far from the 50 % a data-following window would move.
        assert shown_after == pytest.approx(shown_before, rel=0.02)
        assert "Z scaling 0.50" in win._info_label.text()
    finally:
        win.close()


def test_a_z_range_that_does_not_fit_is_reported_as_clipped(_qt_app):
    """Isotropy's price: a pane shows ``scale x pixels`` of Z, so a tall
    structure at a deep zoom is clipped rather than squeezed. Silently showing
    part of a projection as if it were all of it would be the worse failure."""
    win = _window(_qt_app, _slab_state())
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        assert not win._ortho.depth_clipped
        assert "Z clipped" not in win._info_label.text()

        _zoom_xy(win, _qt_app, (-3100, -2900), (-3100, -2900))
        assert win._ortho.depth_clipped
        assert "Z clipped" in win._info_label.text()
    finally:
        win.close()


def test_every_pane_draws_the_same_localizations(_qt_app):
    """One thinned index set feeds all three panes, so the projections show
    the same points rather than three independent samples."""
    win = _window(_qt_app, _state())
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        counts = {}
        for plane in ("XY", "YZ", "XZ"):
            xs, ys = win._pane_scatters[plane].getData()
            counts[plane] = 0 if xs is None else len(xs)
        assert counts["XY"] > 0
        assert counts["YZ"] == counts["XY"] == counts["XZ"]

        # And each pane really plots its own two columns of the same rows.
        locs = win._current_locs(win._dataset())
        for plane in ("XY", "YZ", "XZ"):
            xs, ys = win._pane_scatters[plane].getData()
            ci, cj = ORTHO_AXIS_COLUMNS[plane]
            assert np.all(np.isin(xs, locs[:, ci]))
            assert np.all(np.isin(ys, locs[:, cj]))
    finally:
        win.close()


def test_a_shared_axis_carries_the_same_data_in_both_panes(_qt_app):
    """The range test alone cannot catch a transposed pane: the link forces
    the ranges to agree whatever is plotted inside them. So compare the
    drawn values against the linked range, which is what a reader sees.
    """
    win = _window(_qt_app, _state())
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        locs = win._current_locs(win._dataset())
        y_span = (float(locs[:, 1].min()), float(locs[:, 1].max()))
        z_span = (float(locs[:, 2].min()), float(locs[:, 2].max()))

        # YZ's vertical is Y and it is linked to XY's vertical, so the values
        # drawn there must span Y, not Z.
        _xs, ys = win._pane_scatters["YZ"].getData()
        assert float(ys.min()) == pytest.approx(y_span[0], abs=1e-6)
        assert float(ys.max()) == pytest.approx(y_span[1], abs=1e-6)
        yz_x0, yz_x1, yz_y0, yz_y1 = _ranges(win, "YZ")
        assert yz_y0 <= y_span[0] and yz_y1 >= y_span[1]
        assert yz_x0 <= z_span[0] and yz_x1 >= z_span[1]

        # XZ's horizontal is X, linked to XY's horizontal; its vertical is Z.
        xs, ys = win._pane_scatters["XZ"].getData()
        assert float(ys.min()) == pytest.approx(z_span[0], abs=1e-6)
        assert float(ys.max()) == pytest.approx(z_span[1], abs=1e-6)
        assert float(xs.min()) == pytest.approx(float(locs[:, 0].min()), abs=1e-6)
    finally:
        win.close()


def test_leaving_ortho_gives_the_primary_pane_the_whole_page(_qt_app):
    win = _window(_qt_app, _state())
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        assert win._plot_2d.width() < win._plot_page.width()

        win._axis_combo.setCurrentText("XY")
        _settle(_qt_app)
        assert not win._ortho.active
        assert not win._pane_plots["YZ"].isVisible()
        assert not win._pane_plots["XZ"].isVisible()
        assert win._plot_2d.width() == win._plot_page.width()
        assert win._plot_2d.height() == win._plot_page.height()
    finally:
        win.close()


def test_the_interactive_pane_is_xy_so_roi_and_profile_still_resolve(_qt_app):
    """Ortho is not a projection a ROI can be drawn in: the primary pane is
    XY and is the only pane that takes drawing, so everything that asks
    "which projection is this" must answer XY."""
    win = _window(_qt_app, _state())
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        assert win._active_plane() == "XY"
        assert win.roi_view_plane() == "XY"
        assert win.coordinate_view_box() is win._ortho.view_box("XY")
        assert win.roi_depth_center() is not None
        assert win.profile_localizations() is not None
        assert win.profile_localizations().shape[1] == 2
    finally:
        win.close()


def test_ortho_needs_a_3d_dataset(_qt_app, monkeypatch):
    win = _window(_qt_app, _state(three_d=False))
    try:
        assert not win._ortho_available()
        captured: list[QMenu] = []
        monkeypatch.setattr(QMenu, "exec", lambda menu, *_a: captured.append(menu))
        win._show_context_menu(QPoint(0, 0))
        view_menu = next(
            a.menu() for a in captured[0].actions() if a.text() == "View"
        )
        ortho = next(a for a in view_menu.actions() if a.text() == ORTHO_AXIS)
        assert not ortho.isEnabled()
        assert "3-D dataset" in ortho.toolTip()
    finally:
        win.close()


def test_a_saved_ortho_state_does_not_resurrect_on_2d_data(_qt_app):
    """The mode persists through the existing ``axis`` view state, so a 2-D
    dataset carrying one from elsewhere must fall back rather than open two
    panes showing a line."""
    state = _state(three_d=False)
    state.datasets[0].state["scatter_plot_state"] = {"axis": ORTHO_AXIS}
    win = _window(_qt_app, state)
    try:
        assert win._axis_combo.currentText() == "XY"
        assert not win._ortho.active
    finally:
        win.close()


def test_ortho_survives_the_save_restore_round_trip(_qt_app):
    state = _state()
    win = _window(_qt_app, state)
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        assert state.datasets[0].state["scatter_plot_state"]["axis"] == ORTHO_AXIS
    finally:
        win.close()

    reopened = _window(_qt_app, state)
    try:
        assert reopened._axis_combo.currentText() == ORTHO_AXIS
        assert reopened._ortho.active
        assert reopened._pane_plots["XZ"].isVisible()
    finally:
        reopened.close()


def test_background_and_axis_toggles_reach_every_pane(_qt_app):
    win = _window(_qt_app, _state())
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        win._set_black_background(True)
        _settle(_qt_app)
        for plane in ("XY", "YZ", "XZ"):
            brush = win._pane_plots[plane].backgroundBrush().color()
            assert (brush.red(), brush.green(), brush.blue()) == (0, 0, 0)

        win._set_current_axis_visible(False)
        _settle(_qt_app)
        for plane in ("XY", "YZ", "XZ"):
            item = win._pane_plots[plane].getPlotItem()
            assert not item.getAxis("left").isVisible()
            assert not item.getAxis("bottom").isVisible()
    finally:
        win.close()


def test_y_inversion_is_mirrored_by_the_pane_sharing_the_y_axis(_qt_app):
    """The link carries the range, not the direction: if XY inverts Y and YZ
    does not, the top of one pane is the bottom of the other."""
    state = _state()
    state.prefs.setdefault("plot", {})["scatter_xy_origin"] = "top_left"
    win = _window(_qt_app, state)
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        assert win._ortho.view_box("XY").yInverted()
        assert win._ortho.view_box("YZ").yInverted()
        assert not win._ortho.view_box("XZ").yInverted()   # its vertical is Z

        state.prefs["plot"]["scatter_xy_origin"] = "bottom_left"
        win.refresh_preferences()
        _settle(_qt_app)
        assert not win._ortho.view_box("XY").yInverted()
        assert not win._ortho.view_box("YZ").yInverted()
    finally:
        win.close()


def test_roi_highlight_is_drawn_in_every_pane(_qt_app):
    """The highlight is "these localizations", which projects truthfully into
    any plane — unlike a ROI *shape*, whose in-plane geometry is meaningless
    in the other two panes."""
    from minflux_viewer.core.roi import RoiRecord

    state = _state()
    ds = state.datasets[0]
    locs = np.asarray(ds.loc_nm, dtype=float)
    x0, x1 = float(locs[:, 0].min()), float(locs[:, 0].max())
    y0, y1 = float(locs[:, 1].min()), float(locs[:, 1].max())
    record = RoiRecord.create(
        roi_type="rectangle",
        geometry={"bounds": [x0, y0, (x1 - x0) / 2.0, (y1 - y0) / 2.0]},
        context={"source_view": "scatter", "dataset_idx": 0},
    )
    state.rois.add(record)

    win = _window(_qt_app, state)
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        from minflux_viewer.core.roi_selection import store_roi_mask

        selection = win.compute_roi_selection(record)
        assert selection is not None
        sel_ds, mask, context = selection
        assert mask.any()
        store_roi_mask(sel_ds, record, mask, context=context)
        state.rois.select([record.id])
        win._redraw_roi_highlight()
        _settle(_qt_app)

        counts = []
        for plane in ("XY", "YZ", "XZ"):
            xs, _ys = win._pane_highlights[plane].getData()
            counts.append(0 if xs is None else len(xs))
        assert counts[0] > 0
        assert counts[1] == counts[0] == counts[2]
    finally:
        win.close()


def test_overlay_channels_draw_into_every_pane(_qt_app):
    state = _state()
    state.add_dataset(_dataset(name="ortho-ch2"))
    for index, ds in enumerate(state.datasets):
        ds.state["overlay_id"] = "ortho-group"
        ds.state["overlay_index"] = index
        ds.state["overlay_order"] = index
    state.set_active(0)

    win = _window(_qt_app, state)
    try:
        assert len(win._channels) > 1
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        counts = []
        for plane in ("XY", "YZ", "XZ"):
            xs, _ys = win._pane_scatters[plane].getData()
            counts.append(0 if xs is None else len(xs))
        assert counts[0] > 0
        assert counts[1] == counts[0] == counts[2]
    finally:
        win.close()


# ---------------------------------------------------------------------------
# Shared Z scale, and the viewport-cropped projection
# ---------------------------------------------------------------------------

def _slab_state() -> AppState:
    """Two well-separated blobs at opposite Z, so a crop must change Z."""
    rng = np.random.default_rng(4)
    low = np.column_stack([rng.normal(-3000, 300, 1500),
                           rng.normal(-3000, 300, 1500),
                           rng.normal(-600, 40, 1500)])
    high = np.column_stack([rng.normal(3000, 300, 1500),
                            rng.normal(3000, 300, 1500),
                            rng.normal(600, 40, 1500)])
    xyz = np.vstack([low, high])
    state = AppState()
    state.add_dataset(build_localization_dataset(
        name="slabs", x_nm=xyz[:, 0], y_nm=xyz[:, 1], z_nm=xyz[:, 2],
        tid=rng.integers(0, 300, len(xyz)), source_version="simulation"))
    state.set_active(0)
    return state


def _depth_scales(win) -> dict[str, float]:
    """Measured nm per plot-area pixel of Z, in each side pane."""
    pixels = win._ortho.depth_pixels()
    out = {}
    for plane, vertical in (("XZ", True), ("YZ", False)):
        (x0, x1), (y0, y1) = win._ortho.view_box(plane).viewRange()
        span = (y1 - y0) if vertical else (x1 - x0)
        out[plane] = span / pixels[plane]
    return out


def _zoom_xy(win, app, x_range, y_range) -> None:
    """Stand in for a user zoom of the XY pane.

    The manual signal is part of the gesture, not decoration: the crop engages
    on ``sigRangeChangedManually`` (what a mouse drag or wheel emits), never on
    a bare ``setXRange`` -- which this window also makes programmatically, e.g.
    when a side pane pushes its own pan onto the primary.
    """
    view = win._plot_2d.getPlotItem().getViewBox()
    view.setXRange(*x_range, padding=0)
    view.setYRange(*y_range, padding=0)
    view.sigRangeChangedManually.emit(view.state["mouseEnabled"][:])
    _settle(app)
    win._ortho_timer.stop()          # the debounce, driven directly in tests
    win._redraw_ortho_projection()
    _settle(app)


def test_both_side_panes_share_one_z_scale(_qt_app):
    """Equal Z *ranges* are not equal Z *scales*: XZ shows Z down its height
    and YZ across its width, and those pixel extents differ with the window's
    proportions (measured 210 px against 162 px, so 7.36 against 9.55 nm/px —
    one feature 30% taller in one pane than it was wide in the other)."""
    win = _window(_qt_app, _slab_state())
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        scales = _depth_scales(win)
        assert scales["XZ"] == pytest.approx(scales["YZ"], rel=1e-6)

        # ... and it survives a zoom, where the Z range is re-derived.
        _zoom_xy(win, _qt_app, (-4000, -2000), (-4000, -2000))
        zoomed = _depth_scales(win)
        assert zoomed["XZ"] == pytest.approx(zoomed["YZ"], rel=1e-6)
        assert zoomed["XZ"] < scales["XZ"]        # tighter Z on a smaller region
    finally:
        win.close()


def test_the_shared_z_scale_survives_a_resize(_qt_app):
    """The scale comes from the panes' pixel extents, so the very first apply
    necessarily runs against pre-layout geometry."""
    win = _window(_qt_app, _slab_state(), size=(900, 900))
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        win.resize(1300, 700)
        _settle(_qt_app, turns=6)
        scales = _depth_scales(win)
        assert scales["XZ"] == pytest.approx(scales["YZ"], rel=1e-6)
    finally:
        win.close()


def test_side_panes_project_only_the_visible_xy_region(_qt_app):
    """An XZ pane that pools every localization at a given X, whatever its Y,
    shows the Z of everything behind and in front of the structure you zoomed
    onto — the opposite of what an orthogonal view is for."""
    state = _slab_state()
    win = _window(_qt_app, state)
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        _xs, zs = win._pane_scatters["XZ"].getData()
        assert float(zs.min()) < -400 and float(zs.max()) > 400   # both slabs

        _zoom_xy(win, _qt_app, (-4000, -2000), (-4000, -2000))
        _xs, zs = win._pane_scatters["XZ"].getData()
        assert len(zs) < 3000
        assert float(zs.max()) < 0        # only the low-Z slab survives
        assert "visible region" in win._info_label.text()
    finally:
        win.close()


def test_the_crop_lifts_while_showing_everything(_qt_app):
    """"Showing everything" is the state after a fit or a Reset View, and
    cropping to a range still being fitted would blank the side panes on the
    first draw."""
    win = _window(_qt_app, _slab_state())
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        assert win._ortho_view_rect() is None
        assert "full range" in win._info_label.text()

        _zoom_xy(win, _qt_app, (-4000, -2000), (-4000, -2000))
        assert win._ortho_view_rect() is not None

        win._reset_view()
        _settle(_qt_app)
        win._redraw_ortho_projection()
        _settle(_qt_app)
        assert win._ortho_view_rect() is None
        _xs, zs = win._pane_scatters["XZ"].getData()
        assert float(zs.max()) > 400      # the far slab is back
    finally:
        win.close()


def test_the_display_budget_is_spent_on_rows_in_view(_qt_app):
    """The crop is applied before decimation, not after: otherwise zooming
    into a sparse region would thin away the few points it contains."""
    from minflux_viewer.ui import scatter_window as sw

    state = _slab_state()
    win = _window(_qt_app, state)
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        budget = 200
        original = sw._MAX_DISPLAY_POINTS_2D
        sw._MAX_DISPLAY_POINTS_2D = budget
        try:
            _zoom_xy(win, _qt_app, (-4000, -2000), (-4000, -2000))
            _xs, zs = win._pane_scatters["XZ"].getData()
            # The whole budget lands on the zoomed slab rather than being
            # split with the 1500 points that are off screen.
            assert len(zs) == pytest.approx(budget, rel=0.2)
            assert float(zs.max()) < 0
        finally:
            sw._MAX_DISPLAY_POINTS_2D = original
    finally:
        win.close()


# ---------------------------------------------------------------------------
# Pane proportions, shared-axis labels, and the colorbar's gutter
# ---------------------------------------------------------------------------

def test_grid_gives_the_primary_pane_three_fifths_of_each_axis():
    """0.6 x 0.6 for XY, 0.4 x 0.6 for YZ, 0.6 x 0.4 for XZ.

    The side panes are isotropic, so their thickness *is* the Z range they can
    show — the 0.4 share is what buys back Z that would otherwise be clipped.
    """
    rows, cols = grid_stretch(True)
    assert rows == cols                                   # square proportions
    primary, side = rows
    assert primary / (primary + side) == pytest.approx(0.6)
    assert side / (primary + side) == pytest.approx(0.4)


def test_a_shared_axis_is_labelled_once():
    """X is carried by XY and XZ and Y by XY and YZ, with identical numbers in
    both, so naming each twice puts a redundant title mid-layout. X is named at
    the bottom of the stack, Y down its left. Z is not shared — XZ's is
    vertical, YZ's horizontal — so both Z labels stay."""
    assert ortho_pane_labels("XY") == ("", "Y (nm)")
    assert ortho_pane_labels("XZ") == ("X (nm)", "Z (nm)")
    assert ortho_pane_labels("YZ") == ("Z (nm)", "")
    # Exactly one pane names each shared axis, and two name Z.
    labels = [t for plane in ("XY", "YZ", "XZ") for t in ortho_pane_labels(plane)]
    assert labels.count("X (nm)") == 1
    assert labels.count("Y (nm)") == 1
    assert labels.count("Z (nm)") == 2
    # The standalone projections keep both of their labels.
    assert axis_labels("XY") == ("X (nm)", "Y (nm)")


def test_ortho_panes_carry_one_label_per_shared_axis(_qt_app):
    win = _window(_qt_app, _state())
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        for plane in ("XY", "YZ", "XZ"):
            item = win._pane_plots[plane].getPlotItem()
            assert (item.getAxis("bottom").labelText,
                    item.getAxis("left").labelText) == ortho_pane_labels(plane)

        # Leaving the mode gives the primary pane both labels back.
        win._axis_combo.setCurrentText("XY")
        _settle(_qt_app)
        item = win._pane_plots["XY"].getPlotItem()
        assert (item.getAxis("bottom").labelText,
                item.getAxis("left").labelText) == axis_labels("XY")
    finally:
        win.close()


def test_the_colorbar_hands_its_gutter_to_the_panes_in_ortho(_qt_app):
    """A docked bar reserves a strip flush against the YZ pane — 84 px, which
    at the default window size is most of a 92 px YZ plot area. The panes are
    isotropic, so that strip is Z range."""
    win = _window(_qt_app, _state(), size=(720, 680))
    try:
        assert win._colorbar.isVisible()
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        assert not win._colorbar.isVisible()
        assert win._plot_page.width() == win._stack.width()
        widened = win._ortho.view_box("YZ").sceneBoundingRect().width()

        win._axis_combo.setCurrentText("XY")
        _settle(_qt_app)
        assert win._colorbar.isVisible()          # given straight back

        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        assert not win._colorbar.isVisible()
        assert win._ortho.view_box("YZ").sceneBoundingRect().width() == pytest.approx(widened)
    finally:
        win.close()


@pytest.mark.parametrize("wanted", [True, False])
def test_a_colorbar_choice_made_in_ortho_survives_leaving_it(_qt_app, wanted):
    """The mode borrows the gutter; it does not get a vote over an explicit
    choice made while it holds it."""
    win = _window(_qt_app, _state(), size=(720, 680))
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        win._set_colorbar_visible(wanted)
        _settle(_qt_app)
        assert win._ortho_colorbar_restore is None    # memory dropped
        win._axis_combo.setCurrentText("XY")
        _settle(_qt_app)
        assert win._colorbar.isVisible() is wanted
    finally:
        win.close()


def test_scatter_shared_axes_match_in_direction(_qt_app):
    """The direction half of a shared axis, asserted on screen position.

    The ranges agree whatever the directions are, so a range-based check
    cannot see a flipped pane — which is how the render view shipped with its
    YZ pane upside down.
    """
    from PyQt6.QtCore import QPointF

    state = _state()
    state.prefs.setdefault("plot", {})["scatter_xy_origin"] = "top_left"
    win = _window(_qt_app, state)
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        point = (8000.0, 5000.0, 300.0)

        def screen(plane):
            view_box = win._ortho.view_box(plane)
            widget = win._pane_plots[plane]
            horizontal, vertical = ORTHO_AXIS_COLUMNS[plane]
            scene = view_box.mapViewToScene(
                QPointF(point[horizontal], point[vertical]))
            global_point = widget.viewport().mapToGlobal(scene.toPoint())
            return global_point.x(), global_point.y()

        xy = screen("XY")
        assert screen("XZ")[0] == xy[0]      # shared X
        assert screen("YZ")[1] == xy[1]      # shared Y
        assert win._ortho.view_box("YZ").yInverted() is (
            win._ortho.view_box("XY").yInverted())
        assert win._ortho.view_box("XZ").yInverted() is False
    finally:
        win.close()


# ---------------------------------------------------------------------------
# Interactive side panes (the prerequisite for drawing 3-D ROIs in them)
# ---------------------------------------------------------------------------

def test_the_side_panes_take_the_mouse(_qt_app):
    """They were read-only until the crop stopped being inferred from
    auto-range; a 3-D ROI cannot be drawn in a pane that ignores the mouse."""
    win = _window(_qt_app, _slab_state())
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        for plane in ("XZ", "YZ"):
            assert all(win._ortho.view_box(plane).state["mouseEnabled"]), plane
    finally:
        win.close()


def test_a_programmatic_range_push_does_not_engage_the_crop(_qt_app):
    """The whole reason the side panes were read-only.

    A side pane pushes its pan onto the primary with ``setXRange``, which
    *disables* pyqtgraph's auto-range -- so the old inferred rule read that as
    "the user zoomed XY" and cropped, for a gesture never made in that pane.
    """
    win = _window(_qt_app, _slab_state())
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        assert win._ortho_view_rect() is None

        view = win._plot_2d.getPlotItem().getViewBox()
        view.setXRange(-4000.0, -2000.0, padding=0)      # no manual signal
        _settle(_qt_app)

        # The old inference would now report a rect: auto-range is off.
        assert not any(view.autoRangeEnabled())
        assert win._ortho_view_rect() is None            # ...the flag does not
    finally:
        win.close()


def test_a_real_mouse_gesture_engages_the_crop(_qt_app):
    win = _window(_qt_app, _slab_state())
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        assert win._ortho_view_rect() is None

        view = win._plot_2d.getPlotItem().getViewBox()
        view.setXRange(-4000.0, -2000.0, padding=0)
        view.setYRange(-4000.0, -2000.0, padding=0)
        view.sigRangeChangedManually.emit(view.state["mouseEnabled"][:])
        _settle(_qt_app)

        assert win._ortho_view_rect() is not None
    finally:
        win.close()


def test_a_gesture_in_a_side_pane_also_engages_the_crop(_qt_app):
    """A side pane owns one of XY's axes, so zooming it genuinely narrows what
    the primary shows -- the crop must follow that too, not only an XY drag."""
    win = _window(_qt_app, _slab_state())
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        assert win._ortho_view_rect() is None

        xz = win._ortho.view_box("XZ")
        xz.setXRange(-4000.0, -2000.0, padding=0)        # XZ owns X
        xz.sigRangeChangedManually.emit(xz.state["mouseEnabled"][:])
        _settle(_qt_app)

        assert win._ortho_view_rect() is not None
    finally:
        win.close()


def test_repeated_side_pane_panning_does_not_grow_the_view(_qt_app):
    """Interaction must be range-preserving: panning moves the window onto the
    data, it never widens it.

    ⚠ This is an invariant check, not a reproduction of the reported runaway
    (ranges reaching +-100 um once ``interactive_sides`` was on in scatter).
    That needed a real drag through ``mouseDragEvent`` on a laid-out window and
    a crop tight enough to actually exclude points; offscreen the view opens far
    wider than the data, every point stays inside, and the feedback path the
    explosion travelled is never taken. The tests either side of this one are
    what carry the fix.
    """
    win = _window(_qt_app, _slab_state())
    try:
        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        _zoom_xy(win, _qt_app, (-4000, -2000), (-4000, -2000))   # crop engaged
        before = [_ranges(win, p) for p in ("XY", "XZ", "YZ")]
        spans_before = [max(r[1] - r[0], r[3] - r[2]) for r in before]

        xz = win._ortho.view_box("XZ")
        for step in range(12):
            (x0, x1), _ = xz.viewRange()
            shift = 40.0 * (1 if step % 2 == 0 else -1)
            xz.setXRange(x0 + shift, x1 + shift, padding=0)
            _settle(_qt_app)
            win._ortho_timer.stop()
            win._redraw_ortho_projection()
            _settle(_qt_app)

        after = [_ranges(win, p) for p in ("XY", "XZ", "YZ")]
        spans_after = [max(r[1] - r[0], r[3] - r[2]) for r in after]
        for plane, a, b in zip(("XY", "XZ", "YZ"), spans_before, spans_after):
            assert b == pytest.approx(a, rel=0.05), (plane, a, b)
    finally:
        win.close()


def test_entering_ortho_starts_uncropped(_qt_app):
    """Whatever rectangle the previous projection was zoomed to, the mode opens
    on the whole dataset rather than on a region set somewhere else."""
    win = _window(_qt_app, _slab_state())
    try:
        view = win._plot_2d.getPlotItem().getViewBox()
        view.setXRange(-4000.0, -2000.0, padding=0)
        view.sigRangeChangedManually.emit(view.state["mouseEnabled"][:])
        _settle(_qt_app)

        win._axis_combo.setCurrentText(ORTHO_AXIS)
        _settle(_qt_app)
        assert win._ortho_view_rect() is None
        assert "full range" in win._info_label.text()
    finally:
        win.close()
