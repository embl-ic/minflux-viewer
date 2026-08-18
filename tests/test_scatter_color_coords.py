"""
Regression test: coloring the scatter plot by a coordinate view (xnm/ynm/znm)
must map values to a spread of LUT bins, not a single flat color.

xnm/ynm/znm are nm *views* of the loc_x/loc_y/loc_z store and are NOT keys in
``ds.attr``; an early ``c_name in ds.attr`` guard used to reject them and color
every point with bin 0 (one color regardless of colormap).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from minflux_viewer.core.dataset import AttrStore, DataProp, FileInfo, MinfluxDataset


def _make_ds(n: int = 30, n_traces: int = 3) -> MinfluxDataset:
    per = n // n_traces
    tid = np.repeat(np.arange(n_traces), per)
    ti = np.column_stack([
        np.arange(n_traces) * per,
        np.arange(n_traces) * per + per - 1,
    ])
    prop = DataProp(
        num_loc=n, num_itr=2, num_dim=3, num_traces=n_traces,
        trace_idx=ti, num_loc_per_trace=np.full(n_traces, per),
        attr_names=["loc_x", "loc_y", "loc_z", "tid", "efo", "cfr", "ftr", "idx"],
    )
    rng = np.random.default_rng(0)
    attrs = AttrStore({
        "loc_x": np.linspace(0, 1e-6, n),          # 0 .. 1000 nm
        "loc_y": np.linspace(0, 2e-6, n),          # 0 .. 2000 nm
        "loc_z": rng.uniform(-1e-7, 1e-7, n),      # varying z
        "tid": tid.astype(float),
        "efo": np.linspace(10.0, 100.0, n),
        "cfr": rng.uniform(0.3, 0.9, n),
        "ftr": np.ones(n, dtype=bool),
        "idx": np.arange(1, n + 1, dtype=np.uint32),
    })
    return MinfluxDataset(file=FileInfo(name="synth.mat", folder="/tmp"),
                          prop=prop, attr=attrs)


@pytest.fixture
def _qt_app():
    pytest.importorskip("PyQt6")
    if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
        pytest.skip("No display available for Qt tests")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


@pytest.mark.parametrize("coord", ["xnm", "ynm", "znm"])
def test_scatter_color_by_coordinate_varies(_qt_app, coord):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.scatter_window import ScatterWindow

    state = AppState()
    state.add_dataset(_make_ds())
    ds = state.datasets[0]

    win = ScatterWindow(state, dataset_idx=0)
    try:
        win._cbar_combo.setCurrentText(coord)
        win._invalidate_color_cache()
        indices = np.arange(ds.prop.num_loc)
        values, bins, label, vmin, vmax = win._color_bins_for_points(
            None, None, None, ds, indices,
        )
        assert label == coord
        # The coordinate values must actually drive the color: more than one
        # distinct LUT bin, spanning a real data range.
        assert np.unique(bins).size > 1
        assert vmax > vmin
    finally:
        win.close()


def test_opening_lut_dialog_preserves_color_mapping(_qt_app):
    """Opening the LUT dialog must REFLECT the current color mapping, not change
    it. Bug: setBounds() on the dialog's level lines re-clamped a line sitting
    outside the new range and emitted a stale level callback, which the scatter
    recorded as manual levels and recolored with (the plot 'turned one color')."""
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.scatter_window import ScatterWindow

    state = AppState()
    state.add_dataset(_make_ds())
    ds = state.datasets[0]
    win = ScatterWindow(state, dataset_idx=0)
    try:
        win._cbar_combo.setCurrentText("efo")             # data range 10..100 (min > 0)
        win._cmap_combo.setCurrentText("glasbey")
        win._invalidate_color_cache()
        idx = np.arange(ds.prop.num_loc)
        _, bins0, _, vmin0, vmax0 = win._color_bins_for_points(None, None, None, ds, idx)
        assert win._manual_color_levels is None
        assert np.unique(bins0).size > 1                  # real spread of colors

        win.open_lut_dialog()                             # must not alter the mapping

        assert win._manual_color_levels is None           # not corrupted on open
        win._invalidate_color_cache()
        _, bins1, _, vmin1, vmax1 = win._color_bins_for_points(None, None, None, ds, idx)
        assert (vmin1, vmax1) == (vmin0, vmax0)
        assert np.array_equal(bins1, bins0)               # identical colors after open
    finally:
        win.close()


def test_color_by_change_drops_manual_levels_and_autoscales(_qt_app):
    """Changing the color-by attribute must auto-scale to the NEW attribute's
    range — a manual level range set for the old attribute must not linger."""
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.scatter_window import ScatterWindow

    state = AppState()
    state.add_dataset(_make_ds())
    ds = state.datasets[0]
    win = ScatterWindow(state, dataset_idx=0)
    try:
        idx = np.arange(ds.prop.num_loc)
        win._cbar_combo.setCurrentText("efo")             # range ~10..100
        win._on_lut_levels_changed(20.0, 30.0)            # user sets manual levels
        *_, vmin, vmax = win._color_bins_for_points(None, None, None, ds, idx)
        assert (vmin, vmax) == (20.0, 30.0)               # manual levels honoured

        win._cbar_combo.setCurrentText("cfr")             # range ~0.3..0.9
        assert win._manual_color_levels is None           # dropped on attribute change
        _, _, label, vmin2, vmax2 = win._color_bins_for_points(None, None, None, ds, idx)
        assert label == "cfr"
        assert (vmin2, vmax2) != (20.0, 30.0)
        assert vmax2 < 2.0                                # auto-scaled to cfr's own range
    finally:
        win.close()


def test_lut_dialog_follows_color_by_range(_qt_app):
    """With the LUT dialog open, switching color-by refreshes the dialog's data
    range (and UI) to the new attribute."""
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.scatter_window import ScatterWindow

    state = AppState()
    state.add_dataset(_make_ds())
    win = ScatterWindow(state, dataset_idx=0)
    try:
        win._cbar_combo.setCurrentText("efo")
        win.open_lut_dialog()
        dlg = win._lut_dialog
        assert dlg is not None
        efo_hi = dlg._data_hi
        assert efo_hi > 5.0                               # efo ~10..100

        win._cbar_combo.setCurrentText("cfr")
        win._refresh_lut_dialog(capture_baseline=False)   # (also fired via sync)
        assert dlg._data_hi < 2.0                         # now cfr ~0.3..0.9
        assert dlg._data_hi < efo_hi
    finally:
        win.close()
