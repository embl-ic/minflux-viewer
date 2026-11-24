"""``all [stacked]`` colours a series by its iteration, never by what is drawn.

The reported bug: the Histogram passed ``len(series)`` (iterations that still
had values) as the sequence length but the *iteration index* as the position,
and the old helper clamped ``min(k, n-1)``. Whenever early iterations were
empty -- dcr is NaN in iterations 1-5 of a ten-iteration sequence, or cfr is 0
in iterations 1-2 and Hide zeros removes them -- every surviving series landed
on the ramp's last stop, so all of them were drawn (and listed) dark red. The
Attribute Separation dialog's stacked view passed no length at all and drew
every iteration dark blue.

The colour now belongs to the iteration: ``colors.iteration_color(prefs, k,
n_itr)`` with the dataset's iteration count, in every stacked view.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from minflux_viewer.colors import iteration_color
from minflux_viewer.core import loader
from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.iteration import STACKED_LABEL, ordinal


def _mfx(n_loc=120, n_itr=10, *, dcr_nan=(), cfr_zero=(), seed=0):
    """Flat m2410 array; *dcr_nan* / *cfr_zero* iterations hold no usable value."""
    rng = np.random.default_rng(seed)
    n = n_loc * n_itr
    itr = np.tile(np.arange(n_itr), n_loc)
    dt = np.dtype([
        ("vld", "?"), ("itr", "<i4"), ("tid", "<i4"), ("loc", "<f8", (3,)),
        ("dcr", "<f8"), ("cfr", "<f8"), ("efo", "<f8"), ("eco", "<f8"),
    ])
    mfx = np.zeros(n, dtype=dt)
    mfx["itr"] = itr
    mfx["tid"] = np.repeat(np.arange(n_loc) // 4, n_itr)
    mfx["vld"] = True
    mfx["loc"][:, 0] = np.repeat(rng.uniform(0.0, 1e-6, n_loc), n_itr)
    mfx["loc"][:, 1] = np.repeat(rng.uniform(0.0, 1e-6, n_loc), n_itr)
    mfx["dcr"] = rng.uniform(0.2, 0.8, n)
    mfx["cfr"] = rng.uniform(0.1, 0.6, n)
    mfx["efo"] = rng.uniform(1e4, 1e5, n)
    mfx["eco"] = rng.poisson(50, n).astype(float) + 1.0
    for k in dcr_nan:
        mfx["dcr"][itr == k] = np.nan
    for k in cfr_zero:
        mfx["cfr"][itr == k] = 0.0
    return mfx


def _state(mfx) -> AppState:
    state = AppState()
    state.add_dataset(loader.load_from_mfx_array(mfx, "stacked_colors"))
    return state


def _expected(prefs, iterations, n_itr):
    return {ordinal(k + 1): tuple(iteration_color(prefs, k, n_itr)[:3]) for k in iterations}


def _histogram(qtbot, state, attr, *, hide_zeros=False):
    from minflux_viewer.ui.plots.histogram_window import HistogramWindow

    win = HistogramWindow(state, dataset_idx=0)
    qtbot.addWidget(win)
    win._attr_combo.setCurrentText(attr)
    win._agg_combo.setCurrentText("per loc")
    win._iter_combo.setCurrentText(STACKED_LABEL)
    win._zero_chk.setChecked(hide_zeros)
    return win


def _histogram_colors(win):
    """``{label: rgb}`` as drawn (series pens) and as listed (legend)."""
    drawn = {item.name(): tuple(item.opts["pen"].color().getRgb()[:3])
             for item in win._raw_items}
    listed = {label: tuple(color[:3]) for label, color in win._legend._entries}
    return drawn, listed


def test_histogram_colours_late_iterations_by_their_own_position(qtbot):
    """dcr empty in iterations 1-5 of 10 (the reported .mat case)."""
    state = _state(_mfx(n_itr=10, dcr_nan=range(5)))
    win = _histogram(qtbot, state, "dcr")
    drawn, listed = _histogram_colors(win)

    expected = _expected(state.prefs, range(5, 10), 10)
    assert list(drawn) == ["6th", "7th", "8th", "9th", "10th"]
    assert drawn == expected
    assert len(set(drawn.values())) == 5          # were all the 10th stop
    assert listed == drawn                        # legend shows what is drawn


def test_hide_zeros_does_not_recolour_the_surviving_iterations(qtbot):
    """cfr exactly 0 in iterations 1-2 of 5 (the reported .json case)."""
    state = _state(_mfx(n_itr=5, cfr_zero=(0, 1)))
    win = _histogram(qtbot, state, "cfr")
    shown, _ = _histogram_colors(win)
    assert list(shown) == ["1st", "2nd", "3rd", "4th", "5th"]
    assert shown == _expected(state.prefs, range(5), 5)

    win._zero_chk.setChecked(True)
    hidden, listed = _histogram_colors(win)
    assert list(hidden) == ["3rd", "4th", "5th"]
    assert hidden == {label: shown[label] for label in hidden}
    assert len(set(hidden.values())) == 3          # were all the 10th stop
    assert listed == hidden


def test_histogram_and_attribute_plot_give_an_iteration_the_same_colour(qtbot):
    """The two plots sit side by side; an iteration must read alike in both."""
    from minflux_viewer.ui.plots.attribute_window import AttributeWindow

    state = _state(_mfx(n_itr=10, dcr_nan=range(5)))
    hist = _histogram(qtbot, state, "dcr")
    hist_drawn, _ = _histogram_colors(hist)

    plot = AttributeWindow(state, dataset_idx=0)
    qtbot.addWidget(plot)
    plot.set_gpu_2d(False)                        # no OpenGL view in this test
    plot._iter_combo.setCurrentText(STACKED_LABEL)
    plot_listed = {label: tuple(color[:3]) for label, color in plot._legend._entries}

    assert hist_drawn
    assert set(hist_drawn) <= set(plot_listed)
    assert {label: plot_listed[label] for label in hist_drawn} == hist_drawn


def test_separation_dialog_stacked_series_are_coloured_by_iteration(qtbot):
    """It passed no sequence length, so every iteration was the first stop."""
    from minflux_viewer.ui.analysis.attribute_separation_dialog import (
        AttributeSeparationDialog,
    )

    state = _state(_mfx(n_itr=4, dcr_nan=(0,)))
    dlg = AttributeSeparationDialog(state, 0, attribute="dcr")
    qtbot.addWidget(dlg)
    dlg._iter_combo.setCurrentText(STACKED_LABEL)

    items = [item for item in dlg._plot.getPlotItem().items
             if item.__class__.__name__ == "PlotDataItem" and item.name()]
    drawn = {item.name(): tuple(item.opts["pen"].color().getRgb()[:3]) for item in items}
    assert list(drawn) == ["2nd", "3rd", "4th"]
    assert drawn == _expected(state.prefs, range(1, 4), 4)
    assert len(set(drawn.values())) == 3
