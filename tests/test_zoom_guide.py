"""The constrained-zoom rubber band rides at the cursor, in every plot.

The Attribute Plot pinned its guide to the middle of the view while both
histograms drew it at the mouse, so a vertical zoom drawn near an edge appeared
far from the cursor. The geometry now lives in one place; these tests pin both
the rule and the fact that all three plots use it.
"""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.ui.zoom_guide import (
    GUIDE_CAP_FRACTION,
    ZOOM_MODES,
    zoom_guide_points,
)

VIEW = ((0.0, 100.0), (0.0, 50.0))
#: A drag near the left edge, low down — where "pinned to the middle" was worst.
START, CURRENT = (5.0, 4.0), (9.0, 40.0)


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def test_the_vertical_bar_sits_at_the_cursors_x_not_the_view_middle():
    xs, ys = zoom_guide_points("vertical", START, CURRENT, VIEW)
    assert xs[0] == pytest.approx(CURRENT[0])          # the reported defect
    assert xs[0] != pytest.approx((VIEW[0][0] + VIEW[0][1]) / 2.0)
    # The bar spans the dragged y range, which is the range being zoomed to.
    assert _finite(ys).min() == pytest.approx(min(START[1], CURRENT[1]))
    assert _finite(ys).max() == pytest.approx(max(START[1], CURRENT[1]))
    # ...and its caps are sized from the *other* axis' visible span.
    cap = (VIEW[0][1] - VIEW[0][0]) * GUIDE_CAP_FRACTION
    assert _finite(xs).max() - _finite(xs).min() == pytest.approx(2 * cap)


def test_the_horizontal_bar_sits_at_the_cursors_y():
    xs, ys = zoom_guide_points("horizontal", START, CURRENT, VIEW)
    assert ys[0] == pytest.approx(CURRENT[1])
    assert ys[0] != pytest.approx((VIEW[1][0] + VIEW[1][1]) / 2.0)
    assert _finite(xs).min() == pytest.approx(min(START[0], CURRENT[0]))
    assert _finite(xs).max() == pytest.approx(max(START[0], CURRENT[0]))
    cap = (VIEW[1][1] - VIEW[1][0]) * GUIDE_CAP_FRACTION
    assert _finite(ys).max() - _finite(ys).min() == pytest.approx(2 * cap)


def test_unconstrained_is_a_closed_rectangle_on_the_drag():
    xs, ys = zoom_guide_points("unconstrained", START, CURRENT, VIEW)
    assert (xs[0], ys[0]) == (xs[-1], ys[-1])          # closed
    assert set(xs) == {START[0], CURRENT[0]}
    assert set(ys) == {START[1], CURRENT[1]}
    assert not np.isnan(np.asarray(xs, dtype=float)).any()   # one stroke


def test_a_zero_length_drag_is_still_drawable():
    """The first frame of a drag has start == current."""
    for mode in ZOOM_MODES:
        xs, ys = zoom_guide_points(mode, START, START, VIEW)
        assert len(xs) == len(ys)
        assert np.isfinite(_finite(xs)).all() and np.isfinite(_finite(ys)).all()


def test_an_unknown_mode_falls_back_to_the_rectangle():
    assert zoom_guide_points("nonsense", START, CURRENT, VIEW) == \
        zoom_guide_points("unconstrained", START, CURRENT, VIEW)


# ------------------------------------------------------- the live plots

@pytest.fixture(scope="module")
def _app():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _state_with_data():
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.core.dataset import build_localization_dataset

    rng = np.random.default_rng(0)
    state = AppState()
    state.add_dataset(build_localization_dataset(
        name="A", x_nm=rng.random(300) * 1000, y_nm=rng.random(300) * 1000,
        z_nm=rng.random(300) * 100))
    return state


def _guide_x(window, view_box, update, mode_attr, preview_attr, qtbot):
    """Draw a vertical guide near the left edge; return (bar x, cursor x, middle)."""
    import pyqtgraph as pg
    from PyQt6.QtCore import QPointF

    setattr(window, mode_attr, "vertical")
    item = pg.PlotDataItem()
    setattr(window, preview_attr, item)
    (vx0, vx1), (vy0, vy1) = view_box.viewRange()
    x_at = vx0 + (vx1 - vx0) * 0.15
    update(QPointF(x_at, vy0 + (vy1 - vy0) * 0.1),
           QPointF(x_at, vy0 + (vy1 - vy0) * 0.8))
    xs, _ys = item.getData()
    return float(xs[0]), float(x_at), (vx0 + vx1) / 2.0


@pytest.mark.parametrize("cpu_fix", [False, True])
def test_the_attribute_plot_draws_its_guide_at_the_cursor(_app, qtbot, cpu_fix):
    """Both Attribute Plot renderers share the code, so both are covered."""
    from minflux_viewer.ui.attribute_window import AttributeWindow

    window = AttributeWindow(_state_with_data(), dataset_idx=0, cpu_fix=cpu_fix)
    qtbot.addWidget(window)
    bar, cursor, middle = _guide_x(
        window, window._view_box, window._update_zoom_preview,
        "_zoom_mode", "_zoom_preview", qtbot)
    assert bar == pytest.approx(cursor)
    assert bar != pytest.approx(middle)


def test_the_histogram_draws_its_guide_at_the_cursor(_app, qtbot):
    from minflux_viewer.ui.histogram_window import HistogramWindow

    window = HistogramWindow(_state_with_data(), dataset_idx=0)
    qtbot.addWidget(window)
    bar, cursor, middle = _guide_x(
        window, window._view_box, window._update_zoom_preview,
        "_zoom_mode", "_zoom_preview", qtbot)
    assert bar == pytest.approx(cursor)
    assert bar != pytest.approx(middle)


def test_every_plot_with_this_tool_offers_the_same_modes():
    """One tool, one vocabulary — the drift this file exists to prevent."""
    from minflux_viewer.ui.histogram_window import HistogramWindow

    assert set(HistogramWindow.ZOOM_MODES) == set(ZOOM_MODES)

    import inspect

    from minflux_viewer.ui import attribute_window, hlyb_clustering_dialog

    for module in (attribute_window, hlyb_clustering_dialog):
        source = inspect.getsource(module)
        assert "zoom_guide_points" in source, module.__name__
        # No plot may keep a private copy of the geometry again.
        assert "* 0.08" not in source, module.__name__
