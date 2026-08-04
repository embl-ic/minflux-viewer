"""
Histogram right-click Zoom (horizontal / vertical / unconstrained) + Reset View,
and the auto-range feedback loop behind the runaway zoom-out.

The three zoom modes remap the view to a dragged guide:

* ``horizontal`` — an 'H'-shaped guide across the vertical middle of the view;
  X remaps to the drawn span, Y is untouched, the bin size is refined.
* ``vertical``   — Y remaps to the drawn span; X and the bin size are untouched.
* ``unconstrained`` — a rectangle; both axes remap and the bin size is refined.

``Reset View`` is the context-menu twin of the Reset button.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from minflux_viewer.core import loader


N_LOC = 400
N_ITR = 2

_APP = None


def _qapp():
    global _APP
    try:
        from PyQt6.QtWidgets import QApplication
    except Exception:                                        # pragma: no cover
        pytest.skip("PyQt6 not available")
    if _APP is None:
        _APP = QApplication.instance() or QApplication(sys.argv)
    return _APP


def _make_dataset(n_loc=N_LOC, n_itr=N_ITR):
    """efo spread linearly over 1000..2000 so bin/zoom maths is predictable."""
    n = n_loc * n_itr
    dt = np.dtype([("vld", "?"), ("itr", "<i4"), ("tid", "<i4"),
                   ("loc", "<f8", (3,)), ("efo", "<f8"), ("eco", "<i4")])
    mfx = np.zeros(n, dtype=dt)
    itr = np.tile(np.arange(n_itr), n_loc)
    li = np.repeat(np.arange(n_loc), n_itr)
    mfx["itr"] = itr
    mfx["tid"] = li // 4
    mfx["vld"] = True
    mfx["loc"][:, 0] = np.repeat(np.linspace(0.0, 1e-6, n_loc), n_itr)
    mfx["efo"] = np.repeat(np.linspace(1000.0, 2000.0, n_loc), n_itr)
    mfx["eco"] = 50
    return loader.load_from_mfx_array(mfx, "zoom")


def _histogram(ds=None):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.histogram_window import HistogramWindow

    ds = ds if ds is not None else _make_dataset()
    state = AppState()
    state.add_dataset(ds)
    win = HistogramWindow(state, dataset_idx=0)
    win._attr_combo.setCurrentText("efo")
    return win, ds, state


class _Pt:
    """Stand-in for a pyqtgraph view-coordinate point."""

    def __init__(self, x, y):
        self._x, self._y = float(x), float(y)

    def x(self):
        return self._x

    def y(self):
        return self._y


# ---------------------------------------------------------------------------
# Menu wiring
# ---------------------------------------------------------------------------

def test_zoom_modes_are_the_three_documented_ones_in_order():
    from minflux_viewer.ui.histogram_window import HistogramWindow

    assert HistogramWindow.ZOOM_MODES == ("horizontal", "vertical", "unconstrained")


def test_arming_and_disarming_a_zoom_mode():
    _qapp()
    win, _ds, _state = _histogram()
    try:
        assert win._zoom_mode is None
        win._toggle_zoom_mode("horizontal")
        assert win._zoom_mode == "horizontal"
        # Switching to another mode replaces it...
        win._toggle_zoom_mode("vertical")
        assert win._zoom_mode == "vertical"
        # ...and re-selecting the armed mode disarms it.
        win._toggle_zoom_mode("vertical")
        assert win._zoom_mode is None
        # An unknown mode is ignored.
        win._toggle_zoom_mode("diagonal")
        assert win._zoom_mode is None
    finally:
        win.close()


class _FakeDrag:
    """Minimal stand-in for a pyqtgraph MouseDragEvent."""

    def __init__(self, start_scene, cur_scene, phase):
        self._start, self._cur, self._phase = start_scene, cur_scene, phase

    def button(self):
        from PyQt6.QtCore import Qt
        return Qt.MouseButton.LeftButton

    def buttonDownScenePos(self, *_a):
        return self._start

    def scenePos(self):
        return self._cur

    def isStart(self):
        return self._phase == "start"

    def isFinish(self):
        return self._phase == "finish"

    def accept(self):
        pass


def _drive_drag(win, view_start, view_end):
    """Run a full start/move/finish drag through the patched ViewBox handler."""
    from PyQt6.QtCore import QPointF

    vb = win._view_box
    s = vb.mapViewToScene(QPointF(*view_start))
    e = vb.mapViewToScene(QPointF(*view_end))
    for phase in ("start", "move", "finish"):
        vb.mouseDragEvent(_FakeDrag(s, e if phase != "start" else s, phase))


def test_finishing_a_drag_releases_the_zoom_tool():
    """One-shot: after the drag completes the tool disarms, so the next
    left-drag pans instead of starting another zoom."""
    _qapp()
    win, _ds, _state = _histogram()
    try:
        win.show()
        win._view_box.setRange(xRange=(1000.0, 2000.0), yRange=(0.0, 100.0), padding=0.0)
        win._toggle_zoom_mode("horizontal")
        assert win._zoom_mode == "horizontal"

        _drive_drag(win, (1400.0, 40.0), (1500.0, 40.0))

        assert win._zoom_mode is None                  # released
        assert win._zoom_preview is None               # rubber band cleaned up
        assert win._zoom_drag_start is None
        # ...and the zoom really was applied before releasing.
        (x0, x1), _y = win._view_box.viewRange()
        assert (x0, x1) == pytest.approx((1400.0, 1500.0))
    finally:
        win.close()


def test_a_second_drag_after_release_pans_instead_of_zooming():
    _qapp()
    win, _ds, _state = _histogram()
    try:
        win.show()
        win._view_box.setRange(xRange=(1000.0, 2000.0), yRange=(0.0, 100.0), padding=0.0)
        win._toggle_zoom_mode("unconstrained")
        _drive_drag(win, (1400.0, 10.0), (1500.0, 40.0))
        after_zoom = win._view_box.viewRange()
        assert win._zoom_mode is None

        # The next drag must reach pyqtgraph's own handler, not the zoom path.
        seen = []
        original = win._original_mouse_drag_event
        win._original_mouse_drag_event = lambda ev, axis=None: seen.append(ev)
        try:
            _drive_drag(win, (1420.0, 20.0), (1460.0, 30.0))
        finally:
            win._original_mouse_drag_event = original
        assert seen, "a released zoom must fall through to the pan handler"
        # The zoom path did not re-run, so the range is untouched by our drag.
        assert win._view_box.viewRange() == after_zoom
    finally:
        win.close()


def test_releasing_also_happens_for_a_degenerate_drag():
    """A stray click is still a completed gesture — it must not leave the tool
    armed (that was the reported annoyance)."""
    _qapp()
    win, _ds, _state = _histogram()
    try:
        win.show()
        win._view_box.setRange(xRange=(1000.0, 2000.0), yRange=(0.0, 100.0), padding=0.0)
        before = win._view_box.viewRange()
        win._toggle_zoom_mode("vertical")
        _drive_drag(win, (1500.0, 50.0), (1500.0, 50.0))
        assert win._zoom_mode is None
        assert win._view_box.viewRange() == before      # nothing zoomed
    finally:
        win.close()


def test_escape_disarms_the_zoom():
    _qapp()
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QKeyEvent

    win, _ds, _state = _histogram()
    try:
        win._toggle_zoom_mode("unconstrained")
        event = QKeyEvent(
            QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier
        )
        win.keyPressEvent(event)
        assert win._zoom_mode is None
    finally:
        win.close()


# ---------------------------------------------------------------------------
# Preview geometry
# ---------------------------------------------------------------------------

def _preview_xy(win):
    data = win._zoom_preview.getData()
    return np.asarray(data[0], dtype=float), np.asarray(data[1], dtype=float)


def test_horizontal_preview_is_an_H_at_the_cursor():
    """The guide is a horizontal line with end caps riding at the *mouse's* y,
    not the middle of the view, so it can be lined up against the bars."""
    _qapp()
    import pyqtgraph as pg

    win, _ds, _state = _histogram()
    try:
        win._toggle_zoom_mode("horizontal")
        win._view_box.setRange(xRange=(1000.0, 2000.0), yRange=(0.0, 100.0), padding=0.0)
        win._zoom_preview = pg.PlotDataItem()
        win._update_zoom_preview(_Pt(1200.0, 5.0), _Pt(1600.0, 90.0))
        xs, ys = _preview_xy(win)

        cursor_y = 90.0
        assert np.isclose(np.nanmin(xs), 1200.0) and np.isclose(np.nanmax(xs), 1600.0)
        # The crossbar sits at the cursor, NOT at the view middle (50).
        assert np.isclose(np.nanmean(ys), cursor_y)
        assert not np.isclose(np.nanmean(ys), 50.0)
        # ...and the caps are symmetric about it, so the shape reads as an 'H'.
        assert np.isclose(np.nanmax(ys) - cursor_y, cursor_y - np.nanmin(ys))
        assert np.nanmax(ys) > cursor_y                  # the caps have height
    finally:
        win.close()


def test_vertical_preview_rides_at_the_cursor_x():
    _qapp()
    import pyqtgraph as pg

    win, _ds, _state = _histogram()
    try:
        win._toggle_zoom_mode("vertical")
        win._view_box.setRange(xRange=(1000.0, 2000.0), yRange=(0.0, 100.0), padding=0.0)
        win._zoom_preview = pg.PlotDataItem()
        win._update_zoom_preview(_Pt(1100.0, 20.0), _Pt(1900.0, 80.0))
        xs, ys = _preview_xy(win)

        cursor_x = 1900.0
        assert np.isclose(np.nanmin(ys), 20.0) and np.isclose(np.nanmax(ys), 80.0)
        assert np.isclose(np.nanmean(xs), cursor_x)
        assert not np.isclose(np.nanmean(xs), 1500.0)     # not the view middle
        assert np.isclose(np.nanmax(xs) - cursor_x, cursor_x - np.nanmin(xs))
    finally:
        win.close()


def test_guides_follow_the_cursor_as_the_drag_moves():
    """Both guides track the mouse live, so the same drag start with a different
    cursor position puts the guide somewhere else."""
    _qapp()
    import pyqtgraph as pg

    win, _ds, _state = _histogram()
    try:
        win._view_box.setRange(xRange=(1000.0, 2000.0), yRange=(0.0, 100.0), padding=0.0)
        win._toggle_zoom_mode("horizontal")
        win._zoom_preview = pg.PlotDataItem()
        win._update_zoom_preview(_Pt(1200.0, 10.0), _Pt(1600.0, 20.0))
        low = np.nanmean(_preview_xy(win)[1])
        win._update_zoom_preview(_Pt(1200.0, 10.0), _Pt(1600.0, 70.0))
        high = np.nanmean(_preview_xy(win)[1])
        assert high > low
    finally:
        win.close()


def test_unconstrained_preview_is_a_closed_rectangle():
    _qapp()
    import pyqtgraph as pg

    win, _ds, _state = _histogram()
    try:
        win._toggle_zoom_mode("unconstrained")
        win._zoom_preview = pg.PlotDataItem()
        win._update_zoom_preview(_Pt(1200.0, 10.0), _Pt(1600.0, 40.0))
        xs, ys = _preview_xy(win)
        assert xs.size == 5 and ys.size == 5
        assert xs[0] == xs[-1] and ys[0] == ys[-1]       # closed path
        assert set(np.round(xs, 6)) == {1200.0, 1600.0}
        assert set(np.round(ys, 6)) == {10.0, 40.0}
    finally:
        win.close()


# ---------------------------------------------------------------------------
# Applying the zoom
# ---------------------------------------------------------------------------

def _visible_peak(win, x0, x1):
    """Tallest drawn bar overlapping [x0, x1]."""
    opts = win._hist_item.opts
    x = np.asarray(opts["x"], dtype=float)
    h = np.asarray(opts["height"], dtype=float)
    half = abs(float(opts.get("width", 0.0) or 0.0)) / 2.0
    inside = (x + half >= x0) & (x - half <= x1)
    return float(np.max(h[inside])) if inside.any() else 0.0


def test_horizontal_zoom_remaps_x_refines_bins_and_refits_the_height():
    """Re-binning splits each bar's counts, so the pre-zoom height would leave
    the zoomed peak squashed near the axis; the height is re-fitted instead."""
    _qapp()
    win, _ds, _state = _histogram()
    try:
        win._view_box.setRange(xRange=(1000.0, 2000.0), yRange=(0.0, 100.0), padding=0.0)
        bin_before = float(win._bin_spin.value())
        win._toggle_zoom_mode("horizontal")
        win._apply_zoom_drag(_Pt(1400.0, 12.0), _Pt(1500.0, 88.0))

        (x0, x1), (y0, y1) = win._view_box.viewRange()
        assert (x0, x1) == pytest.approx((1400.0, 1500.0))
        assert float(win._bin_spin.value()) < bin_before  # finer bins
        # Height now tracks the re-binned peak, not the old 0..100.
        peak = _visible_peak(win, 1400.0, 1500.0)
        assert y0 == pytest.approx(0.0)
        assert y1 == pytest.approx(peak * 1.05, rel=1e-6)
        assert y1 < 100.0                                 # it really did shrink
    finally:
        win.close()


def test_vertical_zoom_remaps_y_and_keeps_x_and_the_bin_size():
    _qapp()
    win, _ds, _state = _histogram()
    try:
        win._view_box.setRange(xRange=(1000.0, 2000.0), yRange=(0.0, 100.0), padding=0.0)
        bin_before = float(win._bin_spin.value())
        win._toggle_zoom_mode("vertical")
        win._apply_zoom_drag(_Pt(1100.0, 10.0), _Pt(1900.0, 40.0))

        (x0, x1), (y0, y1) = win._view_box.viewRange()
        assert (x0, x1) == pytest.approx((1000.0, 2000.0))   # X untouched
        assert (y0, y1) == pytest.approx((10.0, 40.0))
        assert float(win._bin_spin.value()) == pytest.approx(bin_before)
    finally:
        win.close()


def test_unconstrained_zoom_remaps_x_and_refits_the_height():
    _qapp()
    win, _ds, _state = _histogram()
    try:
        win._view_box.setRange(xRange=(1000.0, 2000.0), yRange=(0.0, 100.0), padding=0.0)
        bin_before = float(win._bin_spin.value())
        win._toggle_zoom_mode("unconstrained")
        win._apply_zoom_drag(_Pt(1450.0, 5.0), _Pt(1550.0, 25.0))

        (x0, x1), (y0, y1) = win._view_box.viewRange()
        assert (x0, x1) == pytest.approx((1450.0, 1550.0))
        assert float(win._bin_spin.value()) < bin_before
        # Same reasoning as the horizontal case: the drawn height is superseded
        # by a fit to the re-binned bars.
        peak = _visible_peak(win, 1450.0, 1550.0)
        assert y0 == pytest.approx(0.0)
        assert y1 == pytest.approx(peak * 1.05, rel=1e-6)
    finally:
        win.close()


# ---------------------------------------------------------------------------
# The zero floor
# ---------------------------------------------------------------------------

def test_vertical_zoom_clamps_a_below_zero_drag_to_the_axis():
    """Counts are never negative, so the band below zero is always empty."""
    _qapp()
    win, _ds, _state = _histogram()
    try:
        win._view_box.setRange(xRange=(1000.0, 2000.0), yRange=(-50.0, 100.0), padding=0.0)
        win._toggle_zoom_mode("vertical")
        win._apply_zoom_drag(_Pt(1500.0, -40.0), _Pt(1500.0, 30.0))

        _x, (y0, y1) = win._view_box.viewRange()
        assert y0 == pytest.approx(0.0)
        assert y1 == pytest.approx(30.0)
    finally:
        win.close()


def test_no_zoom_mode_puts_the_view_below_zero():
    _qapp()
    win, _ds, _state = _histogram()
    try:
        for mode, start, end in (
            ("horizontal", _Pt(1400.0, -30.0), _Pt(1500.0, -10.0)),
            ("vertical", _Pt(1500.0, -30.0), _Pt(1500.0, 20.0)),
            ("unconstrained", _Pt(1400.0, -30.0), _Pt(1500.0, 20.0)),
        ):
            win._reset_view()
            win._view_box.setRange(
                xRange=(1000.0, 2000.0), yRange=(-50.0, 100.0), padding=0.0
            )
            win._toggle_zoom_mode(mode)
            win._apply_zoom_drag(start, end)
            _x, (y0, y1) = win._view_box.viewRange()
            assert y0 >= 0.0, f"{mode} zoomed below zero"
            assert y1 > y0
    finally:
        win.close()


def test_panning_may_still_show_the_region_below_zero():
    """The floor is a zoom constraint only — a plain pan/setRange is untouched."""
    _qapp()
    win, _ds, _state = _histogram()
    try:
        win._set_zoom_mode(None)
        win._view_box.setRange(xRange=(1000.0, 2000.0), yRange=(-20.0, 60.0), padding=0.0)
        _x, (y0, _y1) = win._view_box.viewRange()
        assert y0 == pytest.approx(-20.0)
    finally:
        win.close()


def test_a_degenerate_drag_is_ignored():
    """A click without a drag must not collapse the view to zero width."""
    _qapp()
    win, _ds, _state = _histogram()
    try:
        win._view_box.setRange(xRange=(1000.0, 2000.0), yRange=(0.0, 100.0), padding=0.0)
        before = win._view_box.viewRange()
        for mode in ("horizontal", "vertical", "unconstrained"):
            win._toggle_zoom_mode(mode)
            win._apply_zoom_drag(_Pt(1500.0, 50.0), _Pt(1500.0, 50.0))
            assert win._view_box.viewRange() == before
    finally:
        win.close()


def test_zoom_actually_puts_more_bars_in_the_zoomed_window():
    """The point of re-binning: the zoomed span must show *more* detail than it
    did before. Re-running Freedman-Diaconis on the subset does not achieve this
    (it trades span for sample size), so this guards the bin-count targeting."""
    _qapp()
    from minflux_viewer.ui.histogram_window import _ZOOM_MIN_BINS

    win, _ds, _state = _histogram()
    try:
        win._view_box.setRange(xRange=(1000.0, 2000.0), yRange=(0.0, 60.0), padding=0.0)

        def bars_between(lo, hi):
            x = np.asarray(win._hist_item.opts["x"], dtype=float)
            return int(np.count_nonzero((x >= lo) & (x <= hi)))

        before = bars_between(1400.0, 1500.0)
        win._toggle_zoom_mode("horizontal")
        win._apply_zoom_drag(_Pt(1400.0, 10.0), _Pt(1500.0, 50.0))
        after = bars_between(1400.0, 1500.0)

        assert after > before
        assert after >= _ZOOM_MIN_BINS - 1        # a readable number of bars
    finally:
        win.close()


def _spiky_histogram():
    """A sharp spike on a broad tail — the shape that broke the bin choice."""
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.histogram_window import HistogramWindow

    rng = np.random.default_rng(0)
    vals = np.concatenate([rng.normal(1000.0, 1.5, 8000),
                           rng.uniform(900.0, 3000.0, 2000)])
    n_loc, n_itr = vals.size, 2
    n = n_loc * n_itr
    dt = np.dtype([("vld", "?"), ("itr", "<i4"), ("tid", "<i4"),
                   ("loc", "<f8", (3,)), ("efo", "<f8"), ("eco", "<i4")])
    mfx = np.zeros(n, dtype=dt)
    mfx["itr"] = np.tile(np.arange(n_itr), n_loc)
    li = np.repeat(np.arange(n_loc), n_itr)
    mfx["tid"] = li // 4
    mfx["vld"] = True
    mfx["loc"][:, 0] = np.repeat(np.linspace(0, 1e-6, n_loc), n_itr)
    mfx["efo"] = np.repeat(vals, n_itr)
    mfx["eco"] = 50
    ds = loader.load_from_mfx_array(mfx, "spike")
    state = AppState()
    state.add_dataset(ds)
    win = HistogramWindow(state, dataset_idx=0)
    win._attr_combo.setCurrentText("efo")
    return win, ds, state


def _bars_in(win, x0, x1):
    opts = win._hist_item.opts
    x = np.asarray(opts["x"], dtype=float)
    h = np.asarray(opts["height"], dtype=float)
    half = abs(float(opts.get("width", 0.0) or 0.0)) / 2.0
    inside = (x + half >= x0) & (x - half <= x1)
    return h[inside]


def test_zoom_over_a_sharp_peak_does_not_flood_the_window_with_bars():
    """Regression: consulting Freedman-Diaconis on the zoom window let a sharp
    concentration (tiny IQR over a wide span) collapse the bin width, producing
    ~2000 sub-pixel bars that were 91% count<=1 — a solid block of equal-height
    bars once the height auto-fitted to them."""
    _qapp()
    from minflux_viewer.ui.histogram_window import _ZOOM_MAX_BINS

    win, _ds, _state = _spiky_histogram()
    try:
        win._view_box.setRange(xRange=(900.0, 3000.0), yRange=(0.0, 100.0), padding=0.0)
        win._toggle_zoom_mode("horizontal")
        win._apply_zoom_drag(_Pt(900.0, 50.0), _Pt(1900.0, 50.0))

        bars = _bars_in(win, 900.0, 1900.0)
        # A readable number of bars, near the target — not thousands.
        assert bars.size <= _ZOOM_MAX_BINS + 2, f"{bars.size} bars in the window"
        assert bars.size >= 10
        # ...and they are not a flat comb of 0/1 counts.
        assert np.mean(bars <= 1.0) < 0.5
        assert np.unique(bars).size > 3
    finally:
        win.close()


def test_zoom_bin_width_ignores_freedman_diaconis():
    """The window bin count is set by the target rule alone, so it is
    predictable regardless of the shape inside the window."""
    _qapp()
    from minflux_viewer.ui.histogram_window import _ZOOM_MAX_BINS

    win, _ds, _state = _spiky_histogram()
    try:
        x0, x1 = 900.0, 1900.0
        width = win._zoom_bin_width(x0, x1)
        vals = np.asarray(win._vals, dtype=float)
        inside = vals[(vals >= x0) & (vals <= x1)]
        fd = win._default_bin_width(inside)
        # FD wants a far finer bin here; the target rule must win.
        assert fd < width / 10.0
        assert width == pytest.approx((x1 - x0) / _ZOOM_MAX_BINS)
    finally:
        win.close()


def test_zoom_bin_width_scales_with_the_zoom_and_respects_the_bin_cap():
    _qapp()
    win, _ds, _state = _histogram()
    try:
        wide = win._zoom_bin_width(1000.0, 2000.0)
        narrow = win._zoom_bin_width(1400.0, 1500.0)
        assert wide is not None and narrow is not None
        assert narrow < wide                              # zooming in => finer

        # Even an absurdly tiny span cannot ask for more than the 4096-bin cap
        # over the full range, or the spin box would advertise a width the
        # renderer silently clamps away.
        from minflux_viewer.ui.histogram_window import _MAX_HISTOGRAM_BINS

        tiny = win._zoom_bin_width(1499.999, 1500.001)
        full_span = 2000.0 - 1000.0
        assert tiny >= full_span / _MAX_HISTOGRAM_BINS

        # A degenerate span yields no width rather than a zero/negative one.
        assert win._zoom_bin_width(1500.0, 1500.0) is None
    finally:
        win.close()


# ---------------------------------------------------------------------------
# Reset View
# ---------------------------------------------------------------------------

def test_reset_view_matches_the_reset_button():
    """Reset View is the context-menu twin of the top-right Reset button: both
    drop the manual bin width, re-draw and re-fit."""
    _qapp()
    win, _ds, _state = _histogram()
    try:
        win._view_box.setRange(xRange=(1000.0, 2000.0), yRange=(0.0, 100.0), padding=0.0)
        win._toggle_zoom_mode("unconstrained")
        win._apply_zoom_drag(_Pt(1450.0, 5.0), _Pt(1550.0, 25.0))
        assert win._auto_bin_width is not None

        win._reset_view()
        assert win._auto_bin_width is None                # bin width back to auto
        assert win._zoom_mode is None                     # and the tool disarmed
        reset_view_range = win._view_box.viewRange()

        # The Reset button lands on exactly the same place.
        win._apply_zoom_drag(_Pt(1450.0, 5.0), _Pt(1550.0, 25.0))
        win._reset_histogram()
        assert win._view_box.viewRange() == reset_view_range
    finally:
        win.close()


# ---------------------------------------------------------------------------
# The runaway zoom-out
# ---------------------------------------------------------------------------

def test_auto_button_is_routed_to_the_deterministic_fit():
    """The floating 'A' calls PlotItem.autoBtnClicked, which pyqtgraph wires to
    the *continuous* enableAutoRange - not to vb.autoRange. It must be routed to
    the same fit the Reset button uses, or it re-enables an auto-range that keeps
    growing the view."""
    _qapp()
    win, _ds, _state = _histogram()
    try:
        plot_item = win._plot.getPlotItem()
        win._view_box.setRange(xRange=(1200.0, 1300.0), yRange=(0.0, 5.0), padding=0.0)
        plot_item.autoBtnClicked()
        x0, x1, y0, y1 = win._last_histogram_bounds
        (vx0, vx1), _vy = win._view_box.viewRange()
        assert (vx0, vx1) == pytest.approx((x0, x1))
    finally:
        win.close()


def test_repeated_auto_range_clicks_do_not_grow_the_view():
    """Regression: with a filter edit open, every 'A' click used to enlarge the
    view (the report label is anchored to the view corner and is re-positioned on
    sigRangeChanged, so including it in the auto-range bounds fed back on
    itself). The view must now be a fixed point."""
    _qapp()
    win, _ds, _state = _histogram()
    try:
        win.start_filter_edit(attr="efo", mode="per loc", lo=1200.0, hi=1600.0)
        plot_item = win._plot.getPlotItem()

        plot_item.autoBtnClicked()
        first = win._view_box.viewRange()
        for _ in range(8):
            plot_item.autoBtnClicked()
        last = win._view_box.viewRange()

        assert last[0] == pytest.approx(first[0])
        assert last[1] == pytest.approx(first[1])
    finally:
        win.close()


def test_filter_report_label_is_excluded_from_auto_range_bounds():
    """The view-anchored label must not contribute to the auto-range bounds."""
    _qapp()
    win, _ds, _state = _histogram()
    try:
        win.start_filter_edit(attr="efo", mode="per loc", lo=1200.0, hi=1600.0)
        label = win._filter_edit.get("report_label")
        region = win._filter_edit.get("region")
        assert label is not None and region is not None
        # `addItem(..., ignoreBounds=True)` works by keeping the item out of the
        # ViewBox's addedItems list, which is what childrenBounds() iterates.
        assert label not in win._view_box.addedItems
        # ...while the region, a genuine data-space item, still counts.
        assert region in win._view_box.addedItems

        # So the computed data bounds stay tied to the histogram + filter region
        # and are not stretched by the label parked at the view corner.
        x_bounds = win._view_box.childrenBounds()[0]
        hist_x0, hist_x1, _y0, _y1 = win._last_histogram_bounds
        assert x_bounds is not None
        assert x_bounds[0] >= min(hist_x0, 1200.0) - 1e-6
        assert x_bounds[1] <= max(hist_x1, 1600.0) + 1e-6
    finally:
        win.close()
