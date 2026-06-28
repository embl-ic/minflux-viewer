"""
Regression test: the Brightness/Contrast palette must close together with the
render (data viewer) window that owns it. It is a top-level always-on-top Tool
window, so it does not hide with the viewer on its own — the viewer's
``closeEvent`` has to close it explicitly, or it lingers orphaned on screen.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from minflux_viewer.core.dataset import AttrStore, DataProp, FileInfo, MinfluxDataset


def _make_ds(n: int = 12, n_traces: int = 3) -> MinfluxDataset:
    per = n // n_traces
    tid = np.repeat(np.arange(n_traces), per)
    ti = np.column_stack([
        np.arange(n_traces) * per,
        np.arange(n_traces) * per + per - 1,
    ])
    prop = DataProp(
        num_loc=n, num_itr=2, num_dim=2, num_traces=n_traces,
        trace_idx=ti, num_loc_per_trace=np.full(n_traces, per),
        attr_names=["loc_x", "loc_y", "loc_z", "tid", "efo", "cfr", "ftr", "idx"],
    )
    attrs = AttrStore({
        "loc_x": np.linspace(0, 1e-6, n),
        "loc_y": np.linspace(0, 1e-6, n),
        "loc_z": np.zeros(n),
        "tid": tid.astype(float),
        "efo": np.linspace(10.0, 100.0, n),
        "cfr": np.random.default_rng(0).uniform(0.3, 0.9, n),
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


def test_brightness_contrast_closes_with_render_window(_qt_app):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.render_window import RenderWindow

    state = AppState()
    state.add_dataset(_make_ds())

    win = RenderWindow(state, dataset_idx=0)
    win.show()
    win._show_brightness_contrast()
    bc = win._bc_dialog
    assert bc is not None
    assert bc.isVisible()

    win.close()
    assert not bc.isVisible()
