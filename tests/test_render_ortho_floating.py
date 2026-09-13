"""Floating orthogonal windows: screen fitting, view carry-over, per-pane gates."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.ui.ortho_view import ORTHO_AXIS, fit_ortho_plot_rects


@pytest.fixture
def _qt_app():
    if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
        pytest.skip("No display available for Qt tests")
    return QApplication.instance() or QApplication(sys.argv)


def _state(n: int = 12000) -> AppState:
    rng = np.random.default_rng(5)
    state = AppState()
    state.add_dataset(build_localization_dataset(
        name="float-ortho",
        x_nm=rng.normal(0.0, 1500.0, n),
        y_nm=rng.normal(0.0, 1500.0, n),
        z_nm=rng.normal(0.0, 300.0, n),
        tid=rng.integers(0, 600, n),
        source_version="simulation",
    ))
    state.set_active(0)
    return state


def _settle(app, turns: int = 10) -> None:
    for _ in range(turns):
        app.processEvents()


def _window(app, *, size=(900, 880)):
    from minflux_viewer.ui.render_window import RenderWindow

    win = RenderWindow(_state(), dataset_idx=0)
    win.resize(*size)
    win.show()
    _settle(app)
    return win


def _float(win, app):
    """Enter the floating arrangement and drive its fit.

    The fit is deferred a turn (the grid stretch has to be applied before the
    primary pane can be measured), so it is driven explicitly here rather than
    depending on when a settle loop happens to land.
    """
    win._set_ortho_placement("floating")
    _settle(app, turns=18)
    win._ortho.fit_to_screen()
    _settle(app, turns=8)


def _frames(win):
    return [win.frameGeometry()] + [h.frameGeometry() for h in win._ortho._hosts.values()]


# ---------------------------------------------------------------------------
# 1. Screen fitting
# ---------------------------------------------------------------------------

def test_fit_plan_shrinks_until_the_whole_set_fits():
    """Floating needs more screen than embedded: the same three plot areas,
    but each now carries its own chrome and the side panes sit *outside* the
    primary rather than inside it. Left alone the arrangement runs off the
    bottom of the monitor."""
    # (left, top, right, bottom) per window. The side windows carry a title
    # bar above the plot and their out-of-plane slider below it.
    chrome = {"XY": (7, 31, 7, 52), "YZ": (2, 31, 2, 28), "XZ": (2, 31, 2, 28)}
    plan = fit_ortho_plot_rects(
        (0, 0, 1920, 1032), (600, 150, 900, 880), ratio=2 / 3, gap=8, chrome=chrome)

    right = plan["YZ"][0] + plan["YZ"][2] + chrome["YZ"][2]
    bottom = plan["XZ"][1] + plan["XZ"][3] + chrome["XZ"][3]
    assert right <= 1920 and bottom <= 1032
    assert plan["XY"][2] < 900 and plan["XY"][3] < 880          # it shrank
    # The primary keeps its shape, and the panes keep the grid's arrangement.
    assert plan["XY"][2] / plan["XY"][3] == pytest.approx(900 / 880, rel=0.02)
    assert plan["YZ"][1] == plan["XY"][1] and plan["YZ"][3] == plan["XY"][3]
    assert plan["XZ"][0] == plan["XY"][0] and plan["XZ"][2] == plan["XY"][2]


def test_fit_plan_only_ever_shrinks():
    """Enabling the mode must not grow a window the user sized."""
    small = (0, 0, 120, 120)
    plan = fit_ortho_plot_rects((0, 0, 4000, 4000), small, ratio=2 / 3)
    assert plan["XY"][2] <= max(small[2], 100)
    assert plan["XY"][3] <= max(small[3], 100)


def test_entering_floating_fits_on_the_current_screen(_qt_app):
    """⚠ Best effort, not a guarantee. Three windows have three sets of
    minimum sizes (the channel area, the depth row, each side slider), so on a
    screen too small for them the fit shrinks as far as it can and the set
    still overflows — an inherent cost of the floating arrangement that the
    embedded grid does not pay. Under pytest Qt can report an ~800x800 stub
    screen, which is exactly that case, so the containment assertion only runs
    where there is genuinely room."""
    win = _window(_qt_app, size=(980, 920))
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        _float(win, _qt_app)

        available = win.screen().availableGeometry()
        if available.width() < 1200 or available.height() < 900:
            # ⚠ On a screen smaller than the window under test, Qt's own
            # clamping is already in play and the fit cannot do its job —
            # the same best-effort case as below. Under pytest Qt can report
            # an ~800x800 stub screen, which is exactly that.
            pytest.skip(f"screen {available.width()}x{available.height()} is "
                        "too small for three windows")
        assert win.width() < 980                       # the primary gave way
        frames = _frames(win)
        assert max(g.x() + g.width() for g in frames) <= available.x() + available.width()
        assert max(g.y() + g.height() for g in frames) <= available.y() + available.height()
    finally:
        win.close()


# ---------------------------------------------------------------------------
# 2. The data view survives the resize
# ---------------------------------------------------------------------------

def test_the_displayed_region_survives_entering_floating(_qt_app):
    """A pyqtgraph view is ranged in data units, so a smaller widget at the
    same nm/px shows *less data* — the user would lose their place the moment
    the mode is enabled."""
    win = _window(_qt_app, size=(980, 920))
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        win._view_box.setXRange(-800.0, 400.0, padding=0)
        win._view_box.setYRange(-600.0, 600.0, padding=0)
        _settle(_qt_app)
        before = [tuple(r) for r in win._view_box.viewRange()]

        _float(win, _qt_app)
        after = [tuple(r) for r in win._view_box.viewRange()]

        # The centre is what must not move; an aspect-locked view may widen one
        # axis when the pane's shape changes, and cannot preserve both spans.
        for axis in (0, 1):
            centre_before = 0.5 * (before[axis][0] + before[axis][1])
            centre_after = 0.5 * (after[axis][0] + after[axis][1])
            assert centre_after == pytest.approx(centre_before, abs=2.0)
        # ... and no data that was visible may be lost.
        assert after[0][0] <= before[0][0] + 2 and after[0][1] >= before[0][1] - 2
        assert after[1][0] <= before[1][0] + 2 and after[1][1] >= before[1][1] - 2
    finally:
        win.close()


# ---------------------------------------------------------------------------
# 3. Entering from a non-XY projection
# ---------------------------------------------------------------------------

def test_entering_ortho_from_xz_carries_the_region(_qt_app):
    """The panes are always XY/YZ/XZ in the same places, so the view is carried
    across as X/Y/Z rather than as whatever the old projection happened to
    show: X stays, the XZ depth gate becomes Y, and Z keeps its centre."""
    win = _window(_qt_app, size=(900, 880))
    try:
        win._set_orientation("XZ")
        _settle(_qt_app)
        win._view_box.setXRange(-1400.0, -200.0, padding=0)
        win._all_depth_check.setChecked(False)
        _settle(_qt_app)
        win._depth_slider.set_range(-500.0, 500.0, emit=True)
        _settle(_qt_app)
        (hx0, hx1), (hz0, hz1) = win._view_box.viewRange()
        depth = tuple(win._depth_range)

        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app, turns=14)
        (ax0, ax1), (ay0, ay1) = win._view_box.viewRange()
        _, (zz0, zz1) = win._ortho.view_box("XZ").viewRange()

        assert (ax0, ax1) == pytest.approx((hx0, hx1), abs=2.0)      # X carried
        # Y came from the depth gate the XZ projection was using.
        assert 0.5 * (ay0 + ay1) == pytest.approx(0.5 * (depth[0] + depth[1]), abs=5.0)
        # Z keeps its centre; its span is set by the isotropic scale.
        assert 0.5 * (zz0 + zz1) == pytest.approx(0.5 * (hz0 + hz1), abs=5.0)
    finally:
        win.close()


def test_entering_ortho_from_yz_carries_the_region(_qt_app):
    """The transposed YZ side keeps Y authoritative and centres XY on X."""
    win = _window(_qt_app, size=(900, 880))
    try:
        win._set_orientation("YZ")
        _settle(_qt_app)
        win._view_box.setXRange(-900.0, 300.0, padding=0)  # standalone YZ: Y
        win._all_depth_check.setChecked(False)
        _settle(_qt_app)
        win._depth_slider.set_range(-400.0, 600.0, emit=True)  # off-plane X
        _settle(_qt_app)
        (hy0, hy1), (hz0, hz1) = win._view_box.viewRange()
        depth = tuple(win._depth_range)

        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app, turns=14)
        (ax0, ax1), (ay0, ay1) = win._view_box.viewRange()
        (zz0, zz1), _ = win._ortho.view_box("YZ").viewRange()

        assert (ay0, ay1) == pytest.approx((hy0, hy1), abs=2.0)  # Y carried
        assert 0.5 * (ax0 + ax1) == pytest.approx(
            0.5 * (depth[0] + depth[1]), abs=5.0
        )
        assert 0.5 * (zz0 + zz1) == pytest.approx(0.5 * (hz0 + hz1), abs=5.0)
    finally:
        win.close()


# ---------------------------------------------------------------------------
# 4. Display linking, mouse, focus, and the crosshair
# ---------------------------------------------------------------------------

def test_the_crosshair_is_on_by_default_in_ortho(_qt_app):
    """It is what makes the three panes read as one 3-D view, so it is not
    hidden behind a menu."""
    win = _window(_qt_app)
    try:
        assert not win._show_crosshair
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        assert win._show_crosshair
        assert win._ortho_crosshair.visible

        _float(win, _qt_app)
        win._ortho_crosshair.set_point((-300.0, 200.0, -250.0))
        _settle(_qt_app)
        # Still drawn in all three, now that two of them are other windows.
        assert sorted(win._ortho_crosshair._lines) == ["XY", "XZ", "YZ"]
        for plane, (h, v) in (("XY", (0, 1)), ("YZ", (2, 1)), ("XZ", (0, 2))):
            lines = win._ortho_crosshair._lines[plane]
            expected = (-300.0, 200.0, -250.0)
            assert lines[0].value() == pytest.approx(expected[h])
            assert lines[1].value() == pytest.approx(expected[v])
    finally:
        win.close()


def test_display_state_is_shared_by_all_three_panes(_qt_app):
    """Brightness/contrast, colormap, invert, per-channel LUT and the white
    background all end at ``_compose_from_cache``, so routing that one hook on
    to the side panes is what keeps the three showing the same thing."""
    win = _window(_qt_app)
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        win._ortho_timer.stop(); win._render_ortho_sides(); _settle(_qt_app)

        win._on_channel_lut(0, "Green")
        win._ortho_timer.stop(); win._render_ortho_sides(); _settle(_qt_app)
        image = win._pane_images["XZ"].image
        red = float(np.nansum(image[..., 0]))
        green = float(np.nansum(image[..., 1]))
        assert green > red, "the side pane must follow the channel colormap"

        # Levels the user sets by hand are shared exactly. Automatic exposure
        # is linked by clipping percentile (tested below), because collapsing
        # an axis makes the primary's raw count limits incomparable.
        win._on_levels_changed(0.0, 400.0)
        _settle(_qt_app)
        assert win._auto_bc is False
        assert win._channels[0]["levels"] == (0.0, 400.0)
    finally:
        win.close()


def test_no_depth_slider_in_ortho(_qt_app):
    """Each pane is a projection over the axis it does not show, and zoom now
    sets how much of that axis is in view — a slab control for one of the three
    would only disagree with the other two."""
    win = _window(_qt_app)
    try:
        win._all_depth_check.setChecked(False)
        _settle(_qt_app)
        assert win._depth_row.isVisible()

        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        assert not win._depth_row.isVisible()
        assert win._all_depth_check.isChecked(), "no gate may survive invisibly"

        win._set_orientation("XY")
        _settle(_qt_app)
        assert win._depth_row.isVisible()       # given back on the way out
    finally:
        win.close()


def test_side_panes_take_the_mouse_and_drive_the_others(_qt_app):
    """A side pane owns one shared axis and one Z axis the other side pane also
    shows, so zooming it has to reach both — the native links only run the
    other way, and are not used at all when floating."""
    win = _window(_qt_app)
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        for plane in ("XZ", "YZ"):
            assert win._ortho.view_box(plane).state["mouseEnabled"] == [True, True]

        win._view_box.setXRange(-1000.0, 1000.0, padding=0)
        win._view_box.setYRange(-1000.0, 1000.0, padding=0)
        _settle(_qt_app)

        win._ortho.view_box("XZ").setXRange(-400.0, 400.0, padding=0)
        _settle(_qt_app)
        (ax0, ax1), _ = win._view_box.viewRange()
        assert (ax0, ax1) == pytest.approx((-400.0, 400.0), abs=30.0)   # X reached XY

        win._ortho.view_box("XZ").setYRange(-150.0, 150.0, padding=0)
        _settle(_qt_app)
        (zx0, zx1), _ = win._ortho.view_box("YZ").viewRange()
        assert (zx0, zx1) == pytest.approx((-150.0, 150.0), abs=40.0)   # Z reached YZ
    finally:
        win.close()


def test_side_depth_interaction_moves_the_anchor_and_crosshair(_qt_app):
    """A side Z pan is an explicit navigation gesture, so the next refresh
    must stay there instead of snapping back to the old marker depth."""
    win = _window(_qt_app)
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app, turns=18)
        win._ortho_timer.stop(); win._render_ortho_sides(); _settle(_qt_app)
        placed = win._ortho_crosshair.point
        assert placed is not None

        target_z = placed[2] + 240.0
        win._ortho.view_box("XZ").setYRange(
            target_z - 100.0, target_z + 100.0, padding=0
        )
        _settle(_qt_app)
        assert win._ortho_crosshair.point[2] == pytest.approx(target_z, abs=1.0)
        assert win._ortho.depth_centre == pytest.approx(target_z, abs=1.0)

        # Exercise the exact refresh that used to restore the old crosshair Z.
        win._ortho_timer.stop(); win._render_ortho_sides(); _settle(_qt_app)
        _, (xz0, xz1) = win._ortho.view_box("XZ").viewRange()
        (yz0, yz1), _ = win._ortho.view_box("YZ").viewRange()
        assert 0.5 * (xz0 + xz1) == pytest.approx(target_z, abs=1.0)
        assert 0.5 * (yz0 + yz1) == pytest.approx(target_z, abs=1.0)
    finally:
        win.close()


def test_grid_lines_are_shared_by_all_ortho_panes(_qt_app):
    win = _window(_qt_app)
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        assert sorted(win._pane_grids) == ["XZ", "YZ"]

        win._set_grid_visible(True)
        assert win._grid_item.isVisible()
        assert all(grid.isVisible() for grid in win._pane_grids.values())

        win._set_white_background(True)
        expected = (35, 35, 35)
        assert win._grid_item.opts["pen"].color().getRgb()[:3] == expected
        assert all(
            grid.opts["pen"].color().getRgb()[:3] == expected
            for grid in win._pane_grids.values()
        )
    finally:
        win.close()


def test_auto_display_range_links_primary_clipping_percentiles(_qt_app):
    """Automatic linking is hidden and projection-aware: it transfers the
    clipping fractions, not XY's incomparable raw counts per pixel."""
    win = _window(_qt_app)
    try:
        primary = np.array([1.0, 2.0, 3.0, 4.0, 100.0], dtype=np.float32)
        side = np.array([10.0, 20.0, 30.0, 40.0, 1000.0], dtype=np.float32)
        win._last_scalar_tile = primary.reshape(1, 1, -1)
        win._manual_levels = (2.0, 4.0)

        levels = win._linked_ortho_auto_levels(side, win._channels[0])

        # XY clips 20% below black and 40% at/above white, so XZ/YZ use
        # their own 20th and 60th percentiles (18 and 34 here).
        assert levels == pytest.approx((18.0, 34.0))
    finally:
        win.close()


def test_the_crosshair_never_moves_by_itself(_qt_app):
    """⚠ ImageJ's Orthogonal_Views assigns ``crossLoc`` in exactly four places
    — ``run()``, ``mouseDragged``, ``mouseWheelMoved``, ``keyPressed`` — plus a
    setter, and nothing repositions it on zoom, pan, scroll, magnification
    change or resize. A marker that re-centres itself is not a marker; it is a
    readout of the view, which the view already shows.

    It does still anchor the side panes' Z: that is what makes zooming close in
    on the marked point instead of the middle of the data.
    """
    win = _window(_qt_app)
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app, turns=18)
        win._ortho_timer.stop(); win._render_ortho_sides(); _settle(_qt_app)
        placed = win._ortho_crosshair.point
        assert placed is not None                     # seeded at the centre

        win._view_box.setXRange(-300.0, 100.0, padding=0)
        win._view_box.setYRange(-200.0, 200.0, padding=0)
        _settle(_qt_app)
        win._ortho_timer.stop(); win._render_ortho_sides(); _settle(_qt_app)
        assert win._ortho_crosshair.point == placed   # a zoom does not move it

        # ... and the side panes stay centred on it.
        _, (z0, z1) = win._ortho.view_box("XZ").viewRange()
        assert 0.5 * (z0 + z1) == pytest.approx(placed[2], abs=1.0)
    finally:
        win.close()


def test_the_ortho_coordinate_is_reported_in_the_status_line(_qt_app):
    """One 3-D point, not three: the panes share their axes pairwise, so the
    crosshair (or, with it hidden, the centre of the views) is a single
    coordinate."""
    win = _window(_qt_app)
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app, turns=18)
        win._ortho_timer.stop(); win._render_ortho_sides(); _settle(_qt_app)

        text = win._info_label.text()
        assert "crosshair X=" in text and "Y=" in text and "Z=" in text

        win._set_crosshair_visible(False)
        _settle(_qt_app)
        text = win._info_label.text()
        assert "centre X=" in text
        assert "crosshair X=" not in text
    finally:
        win.close()


def test_activating_a_floating_pane_raises_the_whole_set(_qt_app):
    """Three windows that belong to one view should not be separated by another
    window landing between them."""
    win = _window(_qt_app)
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        _float(win, _qt_app)

        raised: list[str] = []
        win.raise_ = lambda: raised.append("main")
        for plane, host in win._ortho._hosts.items():
            host.raise_ = (lambda p=plane: raised.append(p))

        host = win._ortho._hosts["XZ"]
        host.isActiveWindow = lambda: True
        from PyQt6.QtCore import QEvent
        host.changeEvent(QEvent(QEvent.Type.ActivationChange))
        assert sorted(raised) == ["XZ", "YZ", "main"]
        assert raised[-1] == "main", "the primary keeps focus"
    finally:
        win.close()


def test_ortho_works_with_a_multi_channel_overlay(_qt_app):
    """The side panes compose through the same `_compose_rgba`, so an overlay
    draws all of its channels there too; the channel list stays in the primary
    window, where it belongs."""
    from minflux_viewer.core.app_state import default_prefs
    from minflux_viewer.ui.render_window import RenderWindow

    rng = np.random.default_rng(5)
    state = AppState()
    state.prefs = default_prefs()
    state.save_prefs = lambda: None
    for index in range(2):
        ds = build_localization_dataset(
            name=f"ch{index}",
            x_nm=rng.normal(200.0 * index, 1500.0, 6000),
            y_nm=rng.normal(0.0, 1500.0, 6000),
            z_nm=rng.normal(80.0 * index, 300.0, 6000),
            tid=rng.integers(0, 300, 6000),
            source_version="simulation",
        )
        ds.state["overlay_id"] = "grp"
        ds.state["overlay_index"] = index
        ds.state["overlay_order"] = index
        state.add_dataset(ds)
    state.set_active(0)

    win = RenderWindow(state, dataset_idx=0)
    win.resize(900, 880)
    win.show()
    _settle(_qt_app)
    try:
        assert len(win._channels) == 2
        assert win._ortho_available()
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        win._ortho_timer.stop(); win._render_ortho_sides(); _settle(_qt_app)

        image = win._pane_images["XZ"].image
        assert image is not None and float(np.nanmax(image)) > 0.0
        # One channel list, in the primary window.
        assert win._channel_area.isVisible()
        for plane in ("XZ", "YZ"):
            assert not hasattr(win._pane_widgets[plane], "_channel_area")
    finally:
        win.close()
