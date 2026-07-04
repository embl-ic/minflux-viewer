"""Particle-average fit table (sortable, row → inspector) + inspector window."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest


@pytest.fixture(scope="module")
def _qt_app():
    pytest.importorskip("PyQt6")
    if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
        pytest.skip("No display available for Qt tests")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


def _image_rows():
    # three particles with distinct, deliberately out-of-order scores/angles
    return [
        {"particle": 1, "n_locs": 100, "angle": 30.0, "dx": 1.0, "dy": 2.0, "tilt": 0.0,
         "score": 0.50, "accepted": True},
        {"particle": 2, "n_locs": 250, "angle": 10.0, "dx": -3.0, "dy": 0.0, "tilt": 5.0,
         "score": 0.90, "accepted": True},
        {"particle": 3, "n_locs": 40, "angle": 300.0, "dx": 0.0, "dy": -1.0, "tilt": 2.0,
         "score": 0.10, "accepted": True},
    ]


def test_fit_table_sorts_numerically_and_maps_original_index(_qt_app):
    from PyQt6.QtCore import Qt

    from minflux_viewer.ui.particle_average_dialog import (
        IMAGE_COLS,
        ParticleFitTableWindow,
    )

    shown = []
    win = ParticleFitTableWindow(_image_rows(), columns=IMAGE_COLS,
                                 header="3 particles", on_show=lambda i: shown.append(i))
    tbl = win._tbl
    # Score is column index 6 in IMAGE_COLS.
    score_col = [c[1] for c in IMAGE_COLS].index("score")
    tbl.sortItems(score_col, Qt.SortOrder.DescendingOrder)
    # Highest score (0.90) is particle 2 → original row index 1.
    top_item = tbl.item(0, 0)
    assert top_item.text() == "2"                       # display value (particle id)
    assert top_item.data(Qt.ItemDataRole.UserRole) == 1  # original index survives sort

    tbl.selectRow(0)
    win._activate()
    assert shown == [1]                                  # on_show got the ORIGINAL index

    # Ascending → lowest score (0.10 = particle 3, orig idx 2) on top.
    tbl.sortItems(score_col, Qt.SortOrder.AscendingOrder)
    assert tbl.item(0, 0).data(Qt.ItemDataRole.UserRole) == 2
    win.close()


def test_fit_table_filter_and_rebuild(_qt_app):
    from minflux_viewer.ui.particle_average_dialog import (
        IMAGE_COLS,
        ParticleFitTableWindow,
    )

    got = []
    win = ParticleFitTableWindow(_image_rows(), columns=IMAGE_COLS, header="3 particles",
                                 on_rebuild=lambda sel: got.append(list(sel)))
    # a histogram/filter panel exists over the numeric columns
    assert hasattr(win, "_hist_col") and win._hist_col.count() > 0
    assert win._passing_mask().all()                    # no filter → all pass

    # filter Score >= 0.4 → particles 1 (0.50) and 2 (0.90) → original idx {0, 1}
    win._filters["score"] = (0.4, 1.0)
    win._refresh_filter_state()
    assert win._passing == {0, 1}
    assert "2 / 3" in win._pass_label.text()

    win._do_rebuild()
    assert got == [[0, 1]]                               # rebuild got the passing indices

    # combine a second column: n_locs >= 150 → within {0,1} only particle 2 (250)
    win._filters["n_locs"] = (150.0, 1e12)
    win._refresh_filter_state()
    assert win._passing == {1}

    win._clear_filters()
    assert win._passing_mask().all() and not win._filters
    win.close()


def test_inspector_window_builds_and_toggles(_qt_app):
    from minflux_viewer.ui.particle_inspector import ParticleInspectorWindow

    rng = np.random.default_rng(0)
    cloud = np.column_stack([rng.normal(0, 40, 200), rng.normal(0, 40, 200),
                             rng.normal(0, 3, 200)])
    phi = np.linspace(0, 2 * np.pi, 60)
    ring = np.column_stack([50 * np.cos(phi), 50 * np.sin(phi), np.zeros(60)])

    win = ParticleInspectorWindow(title="p", cloud=cloud, model_curves=[ring], is_3d=False)
    assert win._cloud_items and win._model_items
    assert all(it.isVisible() for it in win._cloud_items)
    win._model_chk.setChecked(False)
    assert not any(it.isVisible() for it in win._model_items)
    win._cloud_chk.setChecked(False)
    assert not any(it.isVisible() for it in win._cloud_items)
    win.close()


def test_inspector_disables_model_toggle_when_no_model(_qt_app):
    from minflux_viewer.ui.particle_inspector import ParticleInspectorWindow

    win = ParticleInspectorWindow(title="p", cloud=np.zeros((10, 2)), is_3d=False)
    assert not win._model_chk.isEnabled()
    win.close()


def test_inspector_axis_and_box_toggles_2d(_qt_app):
    from minflux_viewer.ui.particle_inspector import ParticleInspectorWindow

    cloud = np.column_stack([np.linspace(-50, 50, 80), np.linspace(-30, 30, 80)])
    win = ParticleInspectorWindow(title="p", cloud=cloud, is_3d=False)
    # 2-D bounding box (data-extent rectangle) exists and toggles.
    assert win._box_items and all(it.isVisible() for it in win._box_items)
    win._box_chk.setChecked(False)
    assert not any(it.isVisible() for it in win._box_items)
    # Axis toggle drives the plot's own axes.
    assert win._plot.getAxis("left").isVisible()
    win._axis_chk.setChecked(False)
    assert not win._plot.getAxis("left").isVisible()
    win._axis_chk.setChecked(True)
    assert win._plot.getAxis("left").isVisible()
    win.close()


def test_inspector_reference_helpers():
    from minflux_viewer.ui import particle_inspector as pi

    assert pi._fmt_tick(1000) == "1000"
    ticks = pi._tick_values(0.0, 100.0)
    assert ticks and all(0.0 <= t <= 100.0 for t in ticks)
    mins, maxs, spans = pi._expanded_bounds(np.array([[0, 0, 0], [10, 20, 5]], float))
    assert np.allclose(mins, [0, 0, 0]) and np.allclose(maxs, [10, 20, 5])
    _m, _M, sp = pi._expanded_bounds(np.array([[0, 0, 3], [10, 20, 3]], float))  # flat Z
    assert sp[2] > 0                                     # degenerate axis padded
    assert pi._xy_bounds(np.array([[0, 0], [10, 20]], float)) == (0.0, 0.0, 10.0, 20.0)
    assert pi._expanded_bounds(np.empty((0, 3))) is None
