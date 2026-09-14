"""Orthogonal view mode in the render window, and its crosshair."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtWidgets import QApplication

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.ui.ortho_view import ORTHO_AXIS, ortho_pane_labels


@pytest.fixture
def _qt_app():
    if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
        pytest.skip("No display available for Qt tests")
    return QApplication.instance() or QApplication(sys.argv)


def _state(*, three_d: bool = True, n: int = 8000) -> AppState:
    rng = np.random.default_rng(5)
    state = AppState()
    state.add_dataset(build_localization_dataset(
        name="render-ortho",
        x_nm=rng.normal(0.0, 1200.0, n),
        y_nm=rng.normal(0.0, 1200.0, n),
        z_nm=rng.normal(0.0, 250.0, n) if three_d else np.zeros(n),
        tid=rng.integers(0, 400, n),
        source_version="simulation",
    ))
    state.set_active(0)
    return state


def _settle(app, turns: int = 8) -> None:
    for _ in range(turns):
        app.processEvents()


def _window(app, state, *, size=(900, 900)):
    from minflux_viewer.ui.render_window import RenderWindow

    win = RenderWindow(state, dataset_idx=0)
    win.resize(*size)
    win.show()
    _settle(app)
    return win


def _draw_sides(win, app) -> None:
    win._ortho_timer.stop()
    win._render_ortho_sides()
    _settle(app)


def _depth_scales(win) -> dict[str, float]:
    pixels = win._ortho.depth_pixels()
    out = {}
    for plane, vertical in (("XZ", True), ("YZ", False)):
        (x0, x1), (y0, y1) = win._ortho.view_box(plane).viewRange()
        out[plane] = ((y1 - y0) if vertical else (x1 - x0)) / pixels[plane]
    return out


# ---------------------------------------------------------------------------
# Layout and projection
# ---------------------------------------------------------------------------

def test_render_ortho_panes_are_aligned_and_isotropic(_qt_app):
    """Alignment needs the panes to agree on axis *visibility*, not just on
    the pinned metrics.

    The render view hides its axes by default while a plain PlotWidget shows
    them, and a hidden axis occupies no space whatever setWidth says --
    measured XZ 528 px wide against XY 590 before that was wired up.
    """
    win = _window(_qt_app, _state())
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        _draw_sides(win, _qt_app)
        assert win._ortho_active()

        xy = win._ortho.view_box("XY").sceneBoundingRect()
        yz = win._ortho.view_box("YZ").sceneBoundingRect()
        xz = win._ortho.view_box("XZ").sceneBoundingRect()
        assert xz.width() == pytest.approx(xy.width(), abs=1.0)
        assert yz.height() == pytest.approx(xy.height(), abs=1.0)

        (x0, x1), (y0, y1) = win._ortho.view_box("XY").viewRange()
        (zx0, zx1), _ = win._ortho.view_box("XZ").viewRange()
        _, (yy0, yy1) = win._ortho.view_box("YZ").viewRange()
        assert (zx0, zx1) == pytest.approx((x0, x1), abs=1e-6)
        assert (yy0, yy1) == pytest.approx((y0, y1), abs=1e-6)

        scale = win._ortho.primary_scale_nm_per_px()
        scales = _depth_scales(win)
        assert scales["XZ"] == pytest.approx(scale, rel=1e-6)
        assert scales["YZ"] == pytest.approx(scale, rel=1e-6)
    finally:
        win.close()


def test_side_panes_draw_the_projection(_qt_app):
    win = _window(_qt_app, _state())
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        _draw_sides(win, _qt_app)
        for plane in ("YZ", "XZ"):
            image = win._pane_images[plane].image
            assert image is not None
            assert image.ndim == 3 and image.shape[2] == 4      # RGBA
            assert float(np.nanmax(image)) > 0.0                # something drawn
    finally:
        win.close()


def test_the_projection_reuses_the_xy_spatial_grid(_qt_app):
    """The side panes reuse the XY (x, y) SpatialGrid: an XZ projection wants
    the locs whose X *and* Y are on screen, which is exactly the XY viewport
    query, so no per-orientation grid is built.
    """
    win = _window(_qt_app, _state())
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        xyz, indices = win._ortho_viewport_indices(0)
        assert xyz is not None and indices is not None and indices.size > 0
        (x0, x1), (y0, y1) = win._view_box.viewRange()
        xs, ys = xyz[0][indices], xyz[1][indices]
        assert xs.min() >= x0 and xs.max() <= x1
        assert ys.min() >= y0 and ys.max() <= y1

        # The same reconstruction the primary pane uses — one function, so the
        # three panes cannot take different methods.
        counts = win.render_scalar(
            xs, xyz[2][indices], (x0, x1), (-1e4, 1e4), 1.0, (0.0, 0.0), 16, 8)
        assert counts.shape == (8, 16)
    finally:
        win.close()


def test_projecting_only_the_visible_region(_qt_app):
    win = _window(_qt_app, _state())
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        _, before = win._ortho_viewport_indices(0)
        win._view_box.setXRange(-300.0, 300.0, padding=0)
        win._view_box.setYRange(-300.0, 300.0, padding=0)
        _settle(_qt_app)
        _, after = win._ortho_viewport_indices(0)
        assert after.size < before.size
    finally:
        win.close()


def test_ortho_needs_3d_localizations(_qt_app):
    win = _window(_qt_app, _state(three_d=False))
    try:
        assert not win._ortho_available()
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        assert not win._ortho_active()          # refused, not half-applied
        assert win._orientation == "XY"
    finally:
        win.close()


def test_reset_view_stays_in_ortho(_qt_app):
    """Ortho is a mode, not a projection: resetting the view inside it must
    re-fit the panes rather than drop back to a single XY view."""
    win = _window(_qt_app, _state())
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        win._view_box.setXRange(-300.0, 300.0, padding=0)
        _settle(_qt_app)
        win._reset_view()
        _settle(_qt_app)
        assert win._ortho_active()
        assert win._pane_widgets["XZ"].isVisible()
    finally:
        win.close()


def test_leaving_ortho_hides_the_side_panes_and_clears_them(_qt_app):
    win = _window(_qt_app, _state())
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        _draw_sides(win, _qt_app)
        assert win._pane_widgets["XZ"].isVisible()

        win._set_orientation("XY")
        _settle(_qt_app)
        assert not win._ortho_active()
        assert not win._pane_widgets["XZ"].isVisible()
        assert not win._pane_widgets["YZ"].isVisible()
        assert win._pane_images["XZ"].image is None
        # The primary pane reclaims the whole page (see grid_stretch).
        assert win._image_view.width() == win._plot_page.width()
    finally:
        win.close()


def test_the_interactive_pane_is_xy(_qt_app):
    """Ortho is not a projection a ROI can be drawn in, so everything that
    asks which plane this is answers XY -- the primary pane keeps the render
    window's own image view, ROI controller, B&C target and volume window."""
    win = _window(_qt_app, _state())
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        assert win._active_plane() == "XY"
        assert win.roi_view_plane() == "XY"
        assert win.coordinate_view_box() is win._ortho.view_box("XY")
        assert win._should_invert_y_axis() == win._xy_origin_top_left()
    finally:
        win.close()


def test_side_panes_carry_one_label_per_shared_axis(_qt_app):
    win = _window(_qt_app, _state())
    try:
        win._set_orientation(ORTHO_AXIS)
        win._set_axes_visible(True)
        _settle(_qt_app)
        for plane in ("XY", "YZ", "XZ"):
            item = win._ortho_plot_item(win._pane_widgets[plane])
            assert (item.getAxis("bottom").labelText,
                    item.getAxis("left").labelText) == ortho_pane_labels(plane)
    finally:
        win.close()


# ---------------------------------------------------------------------------
# Crosshair
# ---------------------------------------------------------------------------

def test_crosshair_marks_one_point_in_all_three_panes(_qt_app):
    win = _window(_qt_app, _state())
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        # On by default, and placed at the middle of the view. The placement
        # rides the debounced side render, so drive it rather than waiting —
        # and let the deferred first fit land before reading the centre, or
        # the marker is compared against a view still settling.
        assert win._ortho_crosshair.visible
        _settle(_qt_app, turns=12)
        _draw_sides(win, _qt_app)
        _draw_sides(win, _qt_app)
        placed = win._ortho_crosshair.point
        assert placed is not None
        (x0, x1), (y0, y1) = win._view_box.viewRange()
        assert placed[0] == pytest.approx(0.5 * (x0 + x1), abs=1.0)
        assert placed[1] == pytest.approx(0.5 * (y0 + y1), abs=1.0)

        win._ortho_crosshair.set_point((500.0, 700.0, -60.0))
        _draw_sides(win, _qt_app)
        # ⚠ A redraw that does not move the view must not undo the placement —
        # a click is the only way to put the marker anywhere but the middle.
        assert win._ortho_crosshair.point == (500.0, 700.0, -60.0)

        # Each pane's two lines sit at that pane's own two coordinates, so the
        # crossings are the same physical point seen three ways.
        for plane, (h, v) in (("XY", (0, 1)), ("YZ", (2, 1)), ("XZ", (0, 2))):
            lines = win._ortho_crosshair._lines[plane]
            expected = (500.0, 700.0, -60.0)
            assert lines[0].value() == pytest.approx(expected[h])
            assert lines[1].value() == pytest.approx(expected[v])
    finally:
        win.close()


def test_a_click_sets_the_two_axes_that_pane_shows(_qt_app):
    """A click says nothing about the axis the pane projects over, so that
    coordinate is carried forward rather than reset."""
    win = _window(_qt_app, _state())
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        win._set_crosshair_visible(True)
        win._ortho_crosshair.set_point((10.0, 20.0, 30.0))

        win._ortho_crosshair.update_from_pane("XZ", 500.0, -180.0)
        assert win._ortho_crosshair.point == (500.0, 20.0, -180.0)   # Y kept

        win._ortho_crosshair.update_from_pane("YZ", -60.0, 700.0)
        assert win._ortho_crosshair.point == (500.0, 700.0, -60.0)   # X kept

        win._ortho_crosshair.update_from_pane("XY", 1.0, 2.0)
        assert win._ortho_crosshair.point == (1.0, 2.0, -60.0)       # Z kept
    finally:
        win.close()


def test_a_roi_tool_wins_the_click(_qt_app):
    win = _window(_qt_app, _state())
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        win._set_crosshair_visible(True)
        win._ortho_crosshair.set_point((1.0, 2.0, 3.0))

        class _Event:
            @staticmethod
            def button():
                return Qt.MouseButton.LeftButton

            @staticmethod
            def scenePos():
                return QPointF(10.0, 10.0)

        win._state.rois.set_tool("rectangle")
        win._on_ortho_click("XY", _Event())
        assert win._ortho_crosshair.point == (1.0, 2.0, 3.0)   # untouched
        win._state.rois.set_tool(None)
    finally:
        win.close()


def test_crosshair_is_off_outside_ortho(_qt_app):
    win = _window(_qt_app, _state())
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        win._set_crosshair_visible(True)
        _settle(_qt_app)
        assert win._ortho_crosshair.visible

        win._set_orientation("XY")
        _settle(_qt_app)
        assert not win._ortho_crosshair.visible
    finally:
        win.close()


# ---------------------------------------------------------------------------
# Floating placement (the ImageJ Orthogonal_Views arrangement)
# ---------------------------------------------------------------------------

def test_floating_geometry_matches_the_embedded_arrangement():
    """YZ to the right sharing the vertical extent, XZ below sharing the
    horizontal one — the same arrangement as the grid, expressed as screen
    rectangles of the *plot area* rather than of the window frame."""
    from minflux_viewer.ui.ortho_view import floating_geometry

    plot = (100, 100, 600, 500)
    yz = floating_geometry(plot, "YZ", thickness=240, gap=8)
    xz = floating_geometry(plot, "XZ", thickness=240, gap=8)
    assert yz[1] == plot[1] and yz[3] == plot[3]      # shares Y extent
    assert yz[0] == plot[0] + plot[2] + 8             # to the right
    assert xz[0] == plot[0] and xz[2] == plot[2]      # shares X extent
    assert xz[1] == plot[1] + plot[3] + 8             # below
    with pytest.raises(ValueError):
        floating_geometry(plot, "XY", thickness=240)


def test_floating_geometry_keeps_decorated_frames_apart():
    """Plot alignment must leave room for the windows around those plots."""
    from minflux_viewer.ui.ortho_view import floating_geometry

    plot = (100, 100, 600, 500)
    owner_chrome = (7, 31, 7, 52)
    side_chrome = (2, 31, 2, 28)
    yz = floating_geometry(
        plot,
        "YZ",
        thickness=240,
        gap=8,
        owner_chrome=owner_chrome,
        side_chrome=side_chrome,
    )
    xz = floating_geometry(
        plot,
        "XZ",
        thickness=240,
        gap=8,
        owner_chrome=owner_chrome,
        side_chrome=side_chrome,
    )

    # Subtracting the side window's leading chrome gives its frame origin;
    # adding the owner's trailing chrome gives its frame edge.
    assert yz[0] - side_chrome[0] == plot[0] + plot[2] + owner_chrome[2] + 8
    assert xz[1] - side_chrome[1] == plot[1] + plot[3] + owner_chrome[3] + 8


def _float(win, app):
    win._set_ortho_placement("floating")
    _settle(app, turns=12)
    win._ortho.align_floating()
    _settle(app, turns=8)


def test_floating_panes_move_into_their_own_windows(_qt_app):
    win = _window(_qt_app, _state(), size=(700, 700))
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app, turns=18)
        # The render view has one arrangement: side panes in their own windows.
        assert win._ortho_placement() == "floating"
        assert sorted(win._ortho._hosts) == ["XZ", "YZ"]
        for plane in ("XZ", "YZ"):
            host = win._ortho._hosts[plane]
            assert host.isVisible()
            assert host.windowTitle() == f"{plane} view"
            # The pane is *moved*, not duplicated — one projection path.
            assert win._pane_widgets[plane].window() is host

        # ... and the primary pane takes the whole page back.
        assert win._image_view.width() == win._plot_page.width()
    finally:
        win.close()


def test_floating_plot_areas_align_with_the_primary_pane(_qt_app):
    """The hard part of the floating arrangement, and the reason ImageJ can
    only approximate it: matching window *frames* leaves the data misaligned
    by the axis gutters and the decorations. The target is computed for the
    plot rectangle and the frame offset measured back out."""
    win = _window(_qt_app, _state(), size=(700, 700))
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        _float(win, _qt_app)
        _draw_sides(win, _qt_app)

        xy = win._ortho._global_plot_rect("XY")
        yz = win._ortho._global_plot_rect("YZ")
        xz = win._ortho._global_plot_rect("XZ")
        assert None not in (xy, yz, xz)

        # ⚠ Screen geometry across windows is BEST EFFORT, and this tolerance
        # is the honest difference between the two placements. The embedded
        # grid is exact by construction (same row = same height, same column =
        # same width); floating has to ask the window manager, which gets a
        # vote — measured exact in isolation but a couple of px out once many
        # windows have preceded it in the same process.
        assert abs(yz[1] - xy[1]) <= 4 and abs(yz[3] - xy[3]) <= 4   # shares rows
        assert abs(xz[0] - xy[0]) <= 4 and abs(xz[2] - xy[2]) <= 4   # shares columns
    finally:
        win.close()


def test_floating_keeps_the_data_alignment_exact(_qt_app):
    """What the window manager cannot spoil.

    pyqtgraph links ranges, not pixels, and the pinned axis metrics handle the
    plot-rect ratio — so the *data* stays aligned to the nanometre even when
    the screen geometry is a few pixels out, which is the property that makes
    the floating arrangement usable at all.
    """
    win = _window(_qt_app, _state(), size=(700, 700))
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        _float(win, _qt_app)
        _draw_sides(win, _qt_app)

        (ax0, ax1), (ay0, ay1) = win._ortho.view_box("XY").viewRange()
        (zx0, zx1), _ = win._ortho.view_box("XZ").viewRange()
        _, (yy0, yy1) = win._ortho.view_box("YZ").viewRange()
        assert (zx0, zx1) == pytest.approx((ax0, ax1), abs=1e-6)
        assert (yy0, yy1) == pytest.approx((ay0, ay1), abs=1e-6)

        scale = win._ortho.primary_scale_nm_per_px()
        for plane, value in _depth_scales(win).items():
            assert value == pytest.approx(scale, rel=1e-6), plane
    finally:
        win.close()


def test_floating_windows_follow_the_owner_while_sticky(_qt_app):
    win = _window(_qt_app, _state(), size=(700, 700))
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        _float(win, _qt_app)
        before = win._ortho._global_plot_rect("YZ")

        win.move(win.x() + 140, win.y() + 30)
        _settle(_qt_app, turns=12)
        xy = win._ortho._global_plot_rect("XY")
        after = win._ortho._global_plot_rect("YZ")
        assert after != before                      # it moved
        assert after[1] == xy[1]                    # still aligned
        assert win._ortho._global_plot_rect("XZ")[0] == xy[0]
    finally:
        win.close()


def test_closing_one_side_window_leaves_the_mode(_qt_app):
    """Closing one of the extra views ends the arrangement rather than leaving
    a partial one — the same as Fiji's Orthogonal_Views. XY is the render
    view's default and what the primary pane is already showing."""
    win = _window(_qt_app, _state(), size=(700, 700))
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        _float(win, _qt_app)

        win._ortho._hosts["YZ"].close()
        _settle(_qt_app, turns=12)
        assert win._orientation == "XY"
        assert not win._ortho_active()
        assert not win._ortho._hosts
        assert not win._pane_widgets["XZ"].isVisible()
        assert win._image_view.width() == win._plot_page.width()
    finally:
        win.close()


def test_switching_back_to_embedded_reclaims_the_panes(_qt_app):
    win = _window(_qt_app, _state(), size=(700, 700))
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app)
        _float(win, _qt_app)
        assert win._ortho._hosts

        win._set_ortho_placement("embedded")
        _settle(_qt_app, turns=10)
        assert not win._ortho._hosts
        for plane in ("XZ", "YZ"):
            assert win._pane_widgets[plane].window() is win
            assert win._pane_widgets[plane].isVisible()
        assert win._image_view.width() < win._plot_page.width()
    finally:
        win.close()


def test_closing_the_render_window_takes_the_floating_windows_with_it(_qt_app):
    win = _window(_qt_app, _state(), size=(700, 700))
    win._set_orientation(ORTHO_AXIS)
    _settle(_qt_app)
    _float(win, _qt_app)
    hosts = list(win._ortho._hosts.values())
    assert hosts
    win.close()
    _settle(_qt_app, turns=8)
    assert not win._ortho._hosts
    for host in hosts:
        # Deleted outright is the good outcome; a surviving wrapper must at
        # least be hidden. A floating side window left on the desktop after
        # its render view is gone is the failure this guards.
        try:
            assert not host.isVisible()
        except RuntimeError:
            pass            # C++ object already gone, which is stronger


# ---------------------------------------------------------------------------
# Axis DIRECTION, not just range
# ---------------------------------------------------------------------------

def _screen_pos(win, plane, point):
    """Where a 3-D point lands on screen in one pane, in global pixels."""
    from PyQt6.QtCore import QPointF
    from minflux_viewer.ui.ortho_view import ORTHO_AXIS_COLUMNS

    view_box = win._ortho.view_box(plane)
    widget = win._pane_widgets[plane]
    horizontal, vertical = ORTHO_AXIS_COLUMNS[plane]
    scene = view_box.mapViewToScene(QPointF(point[horizontal], point[vertical]))
    viewport = widget.viewport() if hasattr(widget, "viewport") else widget
    global_point = viewport.mapToGlobal(scene.toPoint())
    return global_point.x(), global_point.y()


@pytest.mark.parametrize("origin", ["top_left", "bottom_left"])
def test_a_shared_axis_matches_in_direction_not_only_in_range(_qt_app, origin):
    """⚠ A shared axis has a DIRECTION as well as a range, and the link carries
    only the range.

    With XY on the default top-left origin (Y increasing downward) and the YZ
    pane left at pyqtgraph's default (Y increasing upward), the same point sat
    **51 px apart vertically** while the two Y ranges agreed to 0.000 nm the
    whole time — so every range-based assertion passed and only the picture
    showed it. This asserts screen position instead.
    """
    state = _state()
    state.prefs.setdefault("plot", {})["render_xy_origin"] = origin
    win = _window(_qt_app, state, size=(880, 860))
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app, turns=10)
        point = (600.0, 400.0, -150.0)

        xy = _screen_pos(win, "XY", point)
        yz = _screen_pos(win, "YZ", point)
        xz = _screen_pos(win, "XZ", point)
        assert xz[0] == xy[0], "X must line up between XY and XZ"
        assert yz[1] == xy[1], "Y must line up between XY and YZ"

        inverted = win._should_invert_y_axis()
        assert win._ortho.view_box("XY").yInverted() is inverted
        assert win._ortho.view_box("YZ").yInverted() is inverted   # shares Y
        # XZ's vertical is Z, which is nobody else's axis: always natural.
        assert win._ortho.view_box("XZ").yInverted() is False
    finally:
        win.close()


def test_axis_direction_survives_the_move_into_floating_windows(_qt_app):
    win = _window(_qt_app, _state(), size=(880, 860))
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app, turns=10)
        win._set_ortho_placement("floating")
        _settle(_qt_app, turns=18)

        point = (600.0, 400.0, -150.0)
        xy = _screen_pos(win, "XY", point)
        assert _screen_pos(win, "XZ", point)[0] == xy[0]
        assert _screen_pos(win, "YZ", point)[1] == xy[1]
    finally:
        win.close()


def test_the_side_panes_use_the_primary_pane_reconstruction_exactly(_qt_app):
    """One reconstruction, not two look-alikes.

    ``render_scalar`` is called by the XY pane and by both side panes, so they
    cannot take different methods — which they did: XY switches to a
    per-localization Gaussian below ``PER_LOC_SWITCH_COUNT`` while the side
    panes always histogrammed, and the side panes borrowed ``_last_px_nm``
    (whatever XY rendered at *last*, on a separate debounce) instead of sizing
    from their own view. Asserted by rendering the same plane and region
    through the primary path and through the side-pane argument rule and
    requiring the arrays to match bit for bit.
    """
    from minflux_viewer.ui.render_window import _RENDER_SIZE

    state = _state(n=30000)
    win = _window(_qt_app, state, size=(900, 880))
    try:
        win._set_orientation(ORTHO_AXIS)
        _settle(_qt_app, turns=18)

        compared = 0
        for span in ((-1500.0, 1500.0), (-400.0, 400.0), (-70.0, 70.0)):
            win._view_box.setXRange(*span, padding=0)
            win._view_box.setYRange(*span, padding=0)
            _settle(_qt_app)
            # ⚠ No settling between the render and the read: _render is
            # synchronous, but a deferred fit or a sticky re-align can move the
            # view on the next event-loop turn, and then the captured scalar
            # belongs to a range that is no longer current.
            win._render()
            primary = win._last_scalar_tile[0]
            (x0, x1), (y0, y1) = win._view_box.viewRange()

            if win._last_lod >= 0:
                # ⚠ The primary fell back to the tiled LOD pyramid, which sizes
                # its canvas from cached physical tiles rather than from the
                # viewport rule, and is XY-only (TileKey carries an
                # orientation, but no side-pane grids are built). That regime
                # is not shared and is not what this pins; the direct path,
                # which is what a zoomed-in view uses, is.
                continue

            px_nm = max(x1 - x0, y1 - y0) / float(_RENDER_SIZE)
            n_h = min(max(int(round((x1 - x0) / px_nm)), 1), win._MAX_CANVAS_PX)
            n_v = min(max(int(round((y1 - y0) / px_nm)), 1), win._MAX_CANVAS_PX)
            xyz, indices = win._ortho_viewport_indices(0)
            side_rule = win.render_scalar(
                xyz[0][indices], xyz[1][indices], (x0, x1), (y0, y1),
                px_nm, win._sigma_for_plane("XY", px_nm), n_h, n_v,
            )
            assert side_rule.shape == primary.shape, span
            assert np.abs(side_rule - primary).max() == pytest.approx(0.0, abs=1e-5), span
            compared += 1
        assert compared, "no span exercised the direct path"
    finally:
        win.close()
