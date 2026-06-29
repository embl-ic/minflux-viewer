"""
Regression test: the Local Density result window (Analyze > Local Density) is a
modeless, non-owned top-level window — it must be registered on its owner so it
closes together with the main window, not left orphaned on screen.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from minflux_viewer.core.dataset import AttrStore, DataProp, FileInfo, MinfluxDataset


def _make_ds(n: int = 60, n_traces: int = 6) -> MinfluxDataset:
    per = n // n_traces
    tid = np.repeat(np.arange(n_traces), per)
    rng = np.random.default_rng(0)
    ti = np.column_stack([np.arange(n_traces) * per, np.arange(n_traces) * per + per - 1])
    prop = DataProp(
        num_loc=n, num_itr=1, num_dim=2, num_traces=n_traces,
        trace_idx=ti, num_loc_per_trace=np.full(n_traces, per),
        attr_names=["loc_x", "loc_y", "loc_z", "tid", "ftr"],
    )
    attrs = AttrStore({
        "loc_x": rng.uniform(0, 1e-6, n),
        "loc_y": rng.uniform(0, 1e-6, n),
        "loc_z": np.zeros(n),
        "tid": tid.astype(float),
        "ftr": np.ones(n, dtype=bool),
    })
    return MinfluxDataset(file=FileInfo(name="synth.mat", folder="/tmp"), prop=prop, attr=attrs)


@pytest.fixture
def _qt_app():
    pytest.importorskip("PyQt6")
    if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
        pytest.skip("No display available for Qt tests")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


def test_local_density_window_registered_and_closes_with_owner(_qt_app, monkeypatch):
    from PyQt6.QtWidgets import QDialog, QWidget

    import minflux_viewer.analysis.local_density as ld
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.modeless import close_modeless

    state = AppState()
    state.add_dataset(_make_ds())

    # Accept the options dialog without showing it (avoid a modal block).
    monkeypatch.setattr(
        ld.LocalDensityOptionsDialog, "exec",
        lambda self: QDialog.DialogCode.Accepted,
    )

    owner = QWidget()
    try:
        ld.run_local_density(owner, state)

        wins = [w for w in getattr(owner, "_modeless_windows", []) if isinstance(w, ld.LocalDensityImageWindow)]
        assert len(wins) == 1, "result window not registered as modeless on the owner"
        win = wins[0]
        assert win.isVisible()

        # Closing the owner's modeless windows (what _close_all_child_windows
        # does on main-window shutdown) must close the result window.
        close_modeless(owner)
        assert not win.isVisible()
    finally:
        close_modeless(owner)
        owner.close()
        _qt_app.processEvents()
