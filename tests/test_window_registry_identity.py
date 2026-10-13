"""A per-dataset window leaves its registry by identity, not by its old index.

Each window removed itself from the main window's registry when destroyed,
using the dataset index it was opened at. Closing a dataset shifts every later
one down, so that index soon belongs to another dataset's window:

* closing dataset A removed B's windows, which had moved into A's slot;
* the registry held the only Python reference to those top-level windows, so
  B's windows -- still on screen -- became garbage, and the next garbage
  collection destroyed them without a close event, in the middle of the
  collection. pyqtgraph then walked its process-wide view list over
  half-destroyed views and the process aborted (reproduced with a render and
  a scatter view open for two datasets);
* B's windows were also no longer closed when B was, and closing one window
  by hand could unregister a third dataset's window.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent
from PyQt6.QtWidgets import QApplication

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.ui.main_window import MainWindow

REGISTRIES = {
    "render": "_render_windows",
    "scatter": "_scatter_windows",
    "histogram": "_histogram_windows",
    "attribute": "_attr_windows",
    "filter": "_filter_dlgs",
}


def _dataset(name: str, offset: float):
    rng = np.random.default_rng(sum(map(ord, name)))
    xyz = rng.normal(size=(300, 3)) * 80.0 + offset
    return build_localization_dataset(
        name=name, x_nm=xyz[:, 0], y_nm=xyz[:, 1], z_nm=xyz[:, 2],
        tid=np.repeat(np.arange(60), 5), source_version="simulation")


def _flush():
    for _ in range(3):
        QCoreApplication.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture
def main(qapp):
    state = AppState()
    state.prefs.setdefault("data", {}).update(
        compute_local_density=False, compute_loc_prec=False,
        estimate_z_scaling_factor=False, show_dataset_manager=False)
    win = MainWindow(state)
    try:
        yield win
    finally:
        from minflux_viewer.ui.render_window import RenderWindow
        from minflux_viewer.ui.scatter_window import ScatterWindow
        for widget in QApplication.topLevelWidgets():
            try:
                if (isinstance(widget, (RenderWindow, ScatterWindow))
                        and not sip.isdeleted(widget)):
                    widget.close()
            except RuntimeError:
                pass
        _flush()
        win.close()
        _flush()


def _open(win, names):
    datasets = [_dataset(name, 1000.0 * k) for k, name in enumerate(names)]
    for ds in datasets:
        win._state.add_dataset(ds)
    for i in range(len(datasets)):
        win._state.set_active(i)
        win._show_render(i)
        win._show_scatter(i)
        win._show_histogram(i)
        win._show_attr_plot(i)
        win._show_filter()
    _flush()
    return datasets


def test_closing_an_earlier_dataset_keeps_the_later_ones_windows(main):
    _open(main, "AB")
    # Weak references only: the registry must be what keeps B's windows alive.
    b_windows = {kind: weakref.ref(getattr(main, attr)[1])
                 for kind, attr in REGISTRIES.items()}

    main.close_datasets([0])
    _flush()

    for kind, attr in REGISTRIES.items():
        assert getattr(main, attr).get(0) is b_windows[kind](), kind
    gc.collect()
    _flush()
    for kind, ref in b_windows.items():
        assert ref() is not None and not sip.isdeleted(ref()), kind


def test_closing_a_window_after_a_reindex_unregisters_only_itself(main):
    _open(main, "ABC")
    main.close_datasets([0])
    _flush()
    b_scatter, c_scatter = main._scatter_windows[0], main._scatter_windows[1]

    b_scatter.close()
    _flush()

    assert main._scatter_windows.get(1) is c_scatter
    assert 0 not in main._scatter_windows
    fresh = main._show_scatter(0)
    assert fresh is not b_scatter and not sip.isdeleted(fresh)
