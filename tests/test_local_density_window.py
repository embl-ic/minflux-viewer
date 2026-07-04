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


def test_result_window_fits_image_without_view_all(_qt_app):
    """The heatmap must be visible on open — the view is fitted to the nm-placed
    image (the old bug: it stayed ranged to the pixel grid → blank until 'View All')."""
    import minflux_viewer.analysis.local_density as ld

    img = np.random.default_rng(0).random((60, 40))
    x_edges = np.linspace(60000, 66000, 61)          # image lives at nm coordinates
    y_edges = np.linspace(21000, 25000, 41)
    win = ld.LocalDensityImageWindow(img, (x_edges, y_edges), title="LD", info="i")
    try:
        win.show()
        _qt_app.processEvents()
        (xr, yr) = win._plot.getViewBox().viewRange()
        # the view range contains the image centre → visible without View All
        assert xr[0] <= 63000 <= xr[1]
        assert yr[0] <= 23000 <= yr[1]
    finally:
        win.close()
        _qt_app.processEvents()


def test_3d_view_offered_for_3d_data_and_hidden_for_2d(_qt_app):
    """A 3-D scatter is offered only when the result carries 3-D points with real Z."""
    import minflux_viewer.analysis.local_density as ld

    img = np.random.default_rng(0).random((40, 30))
    x_edges = np.linspace(0, 3000, 41)
    y_edges = np.linspace(0, 2000, 31)
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(0, 3000, 400), rng.uniform(0, 2000, 400),
                           rng.uniform(0, 800, 400)])           # real Z
    dens = rng.uniform(1, 50, 400)

    win3d = ld.LocalDensityImageWindow(img, (x_edges, y_edges), title="3D", info="i",
                                       points_nm=pts, density_values=dens, dimensions=3)
    try:
        assert win3d._pts3d is not None and win3d._view_combo.count() == 2
        win3d.show(); _qt_app.processEvents()
        win3d._view_combo.setCurrentIndex(1)                    # switch to 3-D
        _qt_app.processEvents()
        assert win3d._stack.count() == 2 and win3d._stack.currentIndex() == 1
        win3d._set_cmap("viridis")                              # recolour must not raise
    finally:
        win3d.close(); _qt_app.processEvents()

    # 2-D request (or flat Z) → no 3-D scatter offered
    win2d = ld.LocalDensityImageWindow(img, (x_edges, y_edges), title="2D", info="i",
                                       points_nm=pts, density_values=dens, dimensions=2)
    try:
        assert win2d._pts3d is None and win2d._view_combo.count() == 1
    finally:
        win2d.close(); _qt_app.processEvents()

    flat = pts.copy(); flat[:, 2] = 0.0                         # 3-D request but flat Z
    winflat = ld.LocalDensityImageWindow(img, (x_edges, y_edges), title="flat", info="i",
                                         points_nm=flat, density_values=dens, dimensions=3)
    try:
        assert winflat._pts3d is None                          # flat Z ⇒ 2-D only
    finally:
        winflat.close(); _qt_app.processEvents()


def test_display_pixel_is_finer_than_radius():
    """A large radius must not render a blocky/cubic heatmap — the display pixel is
    decoupled (~radius/5)."""
    from minflux_viewer.analysis.local_density import _image_pixel_nm

    pts = np.random.default_rng(0).uniform(0, 5000, size=(2000, 2))
    assert _image_pixel_nm(pts, 100.0) == pytest.approx(20.0)       # 5× finer than 100
    assert _image_pixel_nm(pts, 250.0) < 250.0
    # a huge field of view is capped (still coarser than radius/5 but bounded)
    big = np.random.default_rng(1).uniform(0, 500000, size=(2000, 2))
    assert _image_pixel_nm(big, 20.0) > 20.0 / 5.0


def test_run_local_density_does_not_modify_preferences(_qt_app, monkeypatch):
    """The plugin is ad-hoc exploration — running it must not overwrite or save
    the user-owned `data.local_density_*` preferences (which drive on-load den)."""
    from PyQt6.QtWidgets import QDialog, QWidget

    import minflux_viewer.analysis.local_density as ld
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.modeless import close_modeless

    state = AppState()
    state.add_dataset(_make_ds())
    state.prefs.setdefault("data", {})["local_density_radius"] = 123.0

    # Dialog accepts and reports a radius different from the preference.
    monkeypatch.setattr(
        ld.LocalDensityOptionsDialog, "exec",
        lambda self: QDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        ld.LocalDensityOptionsDialog, "options",
        lambda self: {
            "dimensions": 2, "radius_nm": 50.0, "method": "kdtree",
            "voxel_size_nm": 50.0, "smooth_sigma": 1.0,
        },
    )
    saved = {"called": False}
    monkeypatch.setattr(state, "save_prefs", lambda *a, **k: saved.__setitem__("called", True))

    owner = QWidget()
    try:
        ld.run_local_density(owner, state)
        assert state.prefs["data"]["local_density_radius"] == 123.0, "plugin overwrote the preference"
        assert saved["called"] is False, "plugin persisted preferences"
    finally:
        close_modeless(owner)
        owner.close()
        _qt_app.processEvents()
