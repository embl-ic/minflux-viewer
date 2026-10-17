"""Open, close and reopen datasets and their windows, many times, in one process.

The application is one long-lived process, and its native Qt failures need the
state that earlier work leaves behind: a fresh process per test hides them. This module
keeps one main window through several cycles of opening four datasets with
their windows, closing them in a different order each time (an earlier, the
active and a later one), reopening windows for the survivors, letting the
post-load workers run, and forcing garbage collection where the
pyqtgraph/sip finalizers are most likely to meet a deleted object.

CI runs it with tools/run_qt_tests.py --strict, where a crash fails the job
even when a retry would pass. Phase 3 changes how every one of these windows
finds its dataset; this is the net under that change.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent
from PyQt6.QtWidgets import QApplication

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.ui.main_window import MainWindow

CYCLES = 4
NAMES = ("A", "B", "C", "D")

#: Close orders, by name, one per cycle: an earlier, the active, a later one.
ORDERS = (("A", "C", "D", "B"), ("D", "A", "B", "C"),
          ("B", "D", "C", "A"), ("C", "B", "A", "D"))

REGISTRIES = ("_data_windows", "_render_windows", "_scatter_windows",
              "_histogram_windows", "_attr_windows", "_filter_dlgs")


def _dataset(name: str, offset: float):
    rng = np.random.default_rng(sum(map(ord, name)))
    xyz = rng.normal(size=(400, 3)) * 80.0 + offset
    tid = np.repeat(np.arange(80), 5)
    return build_localization_dataset(
        name=name, x_nm=xyz[:, 0], y_nm=xyz[:, 1], z_nm=xyz[:, 2], tid=tid,
        source_version="simulation")


def _flush():
    for _ in range(3):
        QCoreApplication.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _index_of(win, ds) -> int:
    return next(i for i, other in enumerate(win._state.datasets) if other is ds)


def _dataset_windows():
    from minflux_viewer.ui.dialogs.filter_dialog import FilterDialog
    from minflux_viewer.ui.plots.attribute_window import AttributeWindow
    from minflux_viewer.ui.plots.histogram_window import HistogramWindow
    from minflux_viewer.ui.plots.scatter_window import ScatterWindow
    from minflux_viewer.ui.rendering.render_window import RenderWindow
    from minflux_viewer.ui.tools.data_window import DataWindow

    kinds = (AttributeWindow, DataWindow, FilterDialog, HistogramWindow,
             RenderWindow, ScatterWindow)
    found = []
    for widget in QApplication.topLevelWidgets():
        try:
            if isinstance(widget, kinds) and not sip.isdeleted(widget):
                found.append(widget)
        except RuntimeError:
            pass
    return found


def _open_windows(win, ds):
    i = _index_of(win, ds)
    win._state.set_active(i)
    win._show_render(i)
    win._show_scatter(i)
    win._show_histogram(i)
    win._show_attr_plot(i)
    win._show_filter()


def test_open_close_reopen_cycles_in_one_process(qapp, qtbot):
    state = AppState()
    data = state.prefs.setdefault("data", {})
    data.update(compute_local_density=True, compute_loc_prec=True,
                estimate_z_scaling_factor=False, show_dataset_manager=False)
    state.prefs.setdefault("plot", {})["use_fixed_z_scaling_factor"] = False
    win = MainWindow(state)
    try:
        for cycle in range(CYCLES):
            datasets = {name: _dataset(f"{name}{cycle}", 1000.0 * k)
                        for k, name in enumerate(NAMES)}
            for name in NAMES:
                state.add_dataset(datasets[name])
            for name in NAMES:
                _open_windows(win, datasets[name])
            _flush()

            for closing in ORDERS[cycle]:
                win.close_datasets([_index_of(win, datasets[closing])])
                _flush()
                survivors = [d for d in state.datasets if d in datasets.values()]
                if survivors:
                    # Reopen (or raise) every window of one survivor, as a user
                    # going back to a dataset would.
                    _open_windows(win, survivors[-1])
                    _flush()
                gc.collect()

            assert not state.datasets
            gc.collect()
            _flush()
            # Every window went with its dataset: none was dropped from its
            # registry and left open (tests/test_window_registry_identity.py).
            assert not _dataset_windows(), f"cycle {cycle}"
            for attr in REGISTRIES:
                live = [w for w in getattr(win, attr).values() if not sip.isdeleted(w)]
                assert not live, (cycle, attr)
            # Post-load work for a closed dataset must not still be pending.
            qtbot.waitUntil(lambda: not getattr(win, "_post_load_tasks", {}), timeout=5000)
    finally:
        for window in _dataset_windows():
            window.close()
        _flush()
        win.close()
        _flush()
        gc.collect()
