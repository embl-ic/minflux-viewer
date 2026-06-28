"""
Regression test: colouring the scatter plot by a coordinate view (xnm/ynm/znm)
must map values to a spread of LUT bins, not a single flat colour.

xnm/ynm/znm are nm *views* of the loc_x/loc_y/loc_z store and are NOT keys in
``ds.attr``; an early ``c_name in ds.attr`` guard used to reject them and colour
every point with bin 0 (one colour regardless of colormap).
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
        # The coordinate values must actually drive the colour: more than one
        # distinct LUT bin, spanning a real data range.
        assert np.unique(bins).size > 1
        assert vmax > vmin
    finally:
        win.close()
