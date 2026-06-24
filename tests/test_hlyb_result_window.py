"""Regression: the HlyB result window must be a modeless-capable QDialog
(``show_modeless`` calls ``setModal``, which QWidget lacks)."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication, QDialog, QWidget

from minflux_viewer.analysis.hlyb_clustering import HlyBConfig, analyze_hlyb
from minflux_viewer.ui.hlyb_clustering_dialog import HlyBResultWindow
from minflux_viewer.ui.modeless import show_modeless


@pytest.fixture(scope="module")
def _app():
    app = QApplication.instance() or QApplication([])
    yield app


def _synthetic_result():
    tri = np.array([[0, 0, 0], [18, 0, 0], [9, 15.6, 0]], dtype=float)
    rng = np.random.default_rng(0)
    locs, tids, tid = [], [], 1
    for base in (np.array([0.0, 0, 0]), np.array([300.0, 0, 0])):
        for off in tri:
            pts = base + off + rng.normal(0, 1.0, size=(40, 3)) * np.array([1, 1, 0])
            locs.append(pts)
            tids.append(np.full(40, tid))
            tid += 1
    loc_m = np.vstack(locs) / 1e9
    return analyze_hlyb(loc_m, np.concatenate(tids),
                        HlyBConfig(basic_unit_size_nm=8.0, z_scaling_factor=1.0))


def test_result_window_is_modeless_capable(_app):
    win = HlyBResultWindow(_synthetic_result(), HlyBConfig(), title="t")
    assert isinstance(win, QDialog)          # QWidget has no setModal()
    owner = QWidget()
    show_modeless(win, owner)                # would raise AttributeError on a bare QWidget
    assert win.isModal() is False
    assert win in owner._modeless_windows
    win.close()
