"""UI-level integration tests for the Drift Correction plugin window."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")


def _make_drifting_dataset(seed=3):
    """A drifting 2D MINFLUX-like dataset built via the canonical builder."""
    from minflux_viewer.core.dataset import build_localization_dataset

    rng = np.random.default_rng(seed)
    n_struct = 300
    fx = rng.uniform(-800, 800, n_struct)
    fy = rng.uniform(-800, 800, n_struct)
    reps = 50
    t = np.linspace(0.0, 3000.0, n_struct * reps)
    feat = np.tile(np.arange(n_struct), reps)
    feat = feat[rng.permutation(t.size)]
    t = np.sort(t)
    dxt = 15.0 * (t / 3000.0)
    dyt = -10.0 * (t / 3000.0)
    x = fx[feat] + dxt + rng.normal(0, 1.0, t.size)
    y = fy[feat] + dyt + rng.normal(0, 1.0, t.size)
    return build_localization_dataset(
        name="drifting", x_nm=x, y_nm=y, tid=feat.astype(np.int64), tim=t)


def _bead_dataset(seed=0, n_loc=6000, n_beads=4, bad_bead=None):
    """An .msr-like dataset: ``lnc`` carries a stage drift the beads also see;
    ``loc`` holds the instrument's correction with every bead (as on disk)."""
    from minflux_viewer.core import loader

    rng = np.random.default_rng(seed)
    T = 4000.0

    def drift(t):
        t = np.asarray(t, float)
        return np.column_stack([30e-9 * t / T, -20e-9 * t / T,
                                10e-9 * np.sin(np.pi * t / T)])

    # beads sampled every 4 s, each at its own position, same drift
    bt = np.arange(0.0, T + 4.0, 4.0)
    rows = []
    for k in range(n_beads):
        pos = rng.uniform(-5e-6, 5e-6, 3) * np.array([1, 1, 0.05])
        xyz = pos + drift(bt) + rng.normal(0, 0.3e-9, (bt.size, 3))
        if k == bad_bead:                    # a bead that moves on its own
            xyz[:, 0] += 25e-9 * np.sin(2 * np.pi * bt / 900.0)
        for t_, p_ in zip(bt, xyz):
            rows.append((1000 + k, p_, t_, 1.0))
    points = np.array(rows, dtype=[("gri", "<u4"), ("xyz", "<f8", (3,)),
                                   ("tim", "<f4"), ("str", "<f4")])
    # localizations: one iteration, static structures + the drift
    t = np.sort(rng.uniform(0, T, n_loc))
    struct = rng.uniform(-1e-6, 1e-6, (300, 3)) * np.array([1, 1, 0.2])
    which = rng.integers(0, 300, n_loc)
    lnc = struct[which] + drift(t) + rng.normal(0, 1e-9, (n_loc, 3))
    online = np.mean([np.column_stack([np.interp(t, points["tim"][points["gri"] == 1000 + k],
                                                 points["xyz"][points["gri"] == 1000 + k][:, a])
                                       for a in range(3)]) for k in range(n_beads)], axis=0)
    ref = online[0]
    loc = lnc - (online - ref)
    dt = np.dtype([("vld", "?"), ("itr", "<i4"), ("tid", "<i4"), ("tim", "<f8"),
                   ("loc", "<f8", (3,)), ("lnc", "<f8", (3,))])
    mfx = np.zeros(n_loc, dtype=dt)
    mfx["vld"], mfx["tid"], mfx["tim"] = True, which, t
    mfx["loc"], mfx["lnc"] = loc, lnc
    ds = loader.load_from_mfx_array(mfx, "beads")
    ds.metadata["mbm_points"] = points
    ds.metadata["mbm_points_by_gri"] = {
        str(1000 + k): {"gri": 1000 + k, "name": f"R{k + 1}", "used": 1} for k in range(n_beads)}
    ds.metadata["mbm_used"] = [f"R{k + 1}" for k in range(n_beads)]
    ds.metadata["is_minflux"] = True
    return ds


def _multi_iteration_dataset(seed=5, n_itr=3, eff=1):
    """A drifting m2410-like dataset in which cfr/efc are measured only at
    iteration *eff* (as a real 3-D sequence measures them at 7 of 10): at the
    last iteration cfr reads 0 and efc NaN."""
    from minflux_viewer.core import loader

    rng = np.random.default_rng(seed)
    n_struct, reps = 300, 40
    n_loc = n_struct * reps
    fx = rng.uniform(-800e-9, 800e-9, n_struct)
    fy = rng.uniform(-800e-9, 800e-9, n_struct)
    t = np.sort(rng.uniform(0.0, 3000.0, n_loc))
    feat = rng.integers(0, n_struct, n_loc)
    x = fx[feat] + 15e-9 * t / 3000.0 + rng.normal(0, 1e-9, n_loc)
    y = fy[feat] - 10e-9 * t / 3000.0 + rng.normal(0, 1e-9, n_loc)
    cfr = rng.uniform(0.1, 0.9, n_loc)
    efc = rng.uniform(1e4, 1e5, n_loc)

    dt = np.dtype([("vld", "?"), ("itr", "<i4"), ("tid", "<i4"), ("tim", "<f8"),
                   ("loc", "<f8", (3,)), ("cfr", "<f8"), ("efc", "<f8"), ("efo", "<f8")])
    mfx = np.zeros(n_loc * n_itr, dtype=dt)
    mfx["vld"] = True
    mfx["itr"] = np.tile(np.arange(n_itr), n_loc)
    mfx["tid"] = np.repeat(np.arange(n_loc), n_itr)      # one localization per trace
    mfx["tim"] = np.repeat(t, n_itr)
    mfx["loc"][:, 0] = np.repeat(x, n_itr)
    mfx["loc"][:, 1] = np.repeat(y, n_itr)
    mfx["efc"] = np.nan
    at_eff = mfx["itr"] == eff
    mfx["cfr"][at_eff] = cfr
    mfx["efc"][at_eff] = efc
    mfx["efo"] = rng.uniform(5e4, 2e5, mfx.size)
    ds = loader.load_from_mfx_array(mfx, "multi")
    ds.metadata["is_minflux"] = True
    assert loader.effective_iteration_for_attr(ds, "cfr") == eff
    return ds, cfr


def _set_filter(state, ds, specs):
    ds.state["filter_specs"] = specs
    state.notify_filter_changed(state.datasets.index(ds))


_CFR_ABOVE_HALF = [{"attribute": "cfr", "mode": "per loc", "itr": "effective",
                    "lo": 0.5, "hi": 1.0, "lo_inc": True, "hi_inc": True}]


def _window(qtbot, ds):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.plugins.drift_correction.drift_correction_window import (
        DriftCorrectionWindow,
    )

    state = AppState()
    state.add_dataset(ds)
    win = DriftCorrectionWindow(state)
    qtbot.addWidget(win)
    return win, state


def _estimate(qtbot, win):
    win._on_estimate()
    qtbot.waitUntil(lambda: win._task is None, timeout=60_000)
    assert win._estimate is not None, win._stats.text()
    return win._estimate


def test_plugin_registered():
    from minflux_viewer import plugins
    plugins.ensure_loaded()
    assert "Drift Correction" in [e.name for e in plugins.available()]


def test_time_windows_estimate_and_apply(qtbot):
    from minflux_viewer.plugins.drift_correction.drift_correction_window import METHOD_WINDOWS

    win, state = _window(qtbot, _make_drifting_dataset())
    # no bead record → only the fiducial-free method is offered
    assert win._method() == METHOD_WINDOWS
    win._dims_combo.setCurrentIndex(1)          # force 2D
    win._res_spin.setValue(4.0)
    win._auto_T.setChecked(False)
    win._T_spin.setValue(400.0)
    est = _estimate(qtbot, win)
    assert est.windows.dims == 2 and est.windows.n_windows >= 2
    assert np.std(est.windows.dxt) > 1.0

    n_before = len(state.datasets)
    win._on_apply()
    assert len(state.datasets) == n_before + 1
    new = state.datasets[-1]
    assert new.metadata.get("drift_corrected") is True
    rec = new.metadata["drift_correction"]
    assert rec["method"] == METHOD_WINDOWS and rec["dims"] == 2
    assert rec["windows"]["n_windows"] == est.windows.n_windows
    assert float(getattr(new.cali, "z_scaling_factor", 1.0)) == 1.0
    assert new.prop.num_loc == state.datasets[0].prop.num_loc
    win.close()


def test_bead_method_with_the_recorded_set_reproduces_the_file(qtbot):
    from minflux_viewer.plugins.drift_correction.drift_correction_window import METHOD_BEADS

    win, _state = _window(qtbot, _bead_dataset())
    assert win._method_combo.model().item(win._method_combo.findData(METHOD_BEADS)).isEnabled()
    win._method_combo.setCurrentIndex(win._method_combo.findData(METHOD_BEADS))
    assert len(win._selected_gris()) == 4            # recorded selection by default
    est = _estimate(qtbot, win)
    assert est.recorded
    # the file's loc was made with the same beads (unsmoothed here, so within noise)
    assert max(est.beads.reproduction_nm) < 0.5
    assert "reproduces the file's recorded correction" in win._stats.text()
    win.close()


def test_suggested_selection_drops_the_deviating_bead(qtbot):
    win, _state = _window(qtbot, _bead_dataset(bad_bead=2))
    inputs, _ = win._beads_for(win._active())
    worst = max(inputs.beads, key=lambda b: b.deviation_nm)
    assert worst.gri == 1002
    win._select_suggested()
    assert 1002 not in win._selected_gris() and len(win._selected_gris()) == 3
    win.close()


def test_beads_then_residual_runs_both(qtbot):
    from minflux_viewer.plugins.drift_correction.drift_correction_window import METHOD_BOTH

    win, state = _window(qtbot, _bead_dataset())
    win._method_combo.setCurrentIndex(win._method_combo.findData(METHOD_BOTH))
    win._auto_T.setChecked(False)
    win._T_spin.setValue(800.0)
    win._res_spin.setValue(10.0)
    est = _estimate(qtbot, win)
    assert est.beads is not None and est.windows is not None
    # beads already removed the drift: the residual is small
    assert max(np.ptp(v) for v in (est.windows.dxt, est.windows.dyt)) < 5.0
    win._on_apply()
    rec = state.datasets[-1].metadata["drift_correction"]
    assert set(rec) >= {"beads", "windows"}
    win.close()


def test_non_minflux_dataset_is_rejected(qtbot):
    from minflux_viewer.core.dataset import build_localization_dataset

    rng = np.random.default_rng(0)
    ds = build_localization_dataset(
        name="plain", x_nm=rng.normal(size=500), y_nm=rng.normal(size=500))
    win, state = _window(qtbot, ds)
    assert win._validate(state.active_dataset) is not None
    win.close()


def _windows_2d(win, T=500.0):
    win._dims_combo.setCurrentIndex(1)          # force 2D
    win._res_spin.setValue(4.0)
    win._auto_T.setChecked(False)
    win._T_spin.setValue(T)


def test_filtered_estimate_uses_and_keeps_only_the_passing_rows(qtbot):
    from minflux_viewer.analysis import drift_correction as dc
    from minflux_viewer.core.loader import mfx_get

    ds, cfr = _multi_iteration_dataset()
    win, state = _window(qtbot, ds)
    _set_filter(state, ds, _CFR_ABOVE_HALF)
    _windows_2d(win)
    passing = cfr >= 0.5
    assert "pass and are used" in win._info.text()

    est = _estimate(qtbot, win)
    assert est.filtered and est.filter_attrs == ["cfr"]
    assert est.n_source == cfr.size and est.n_kept == int(passing.sum())
    # The time windows were estimated from the passing localizations only.
    x = np.asarray(mfx_get(ds, "xnm", itr="last", vld_only=True))[passing]
    y = np.asarray(mfx_get(ds, "ynm", itr="last", vld_only=True))[passing]
    t = np.asarray(mfx_get(ds, "tim", itr="last", vld_only=True))[passing]
    direct = dc.estimate_drift(x, y, None, t, resolution_nm=4.0, time_window_s=500.0, dims=2)
    np.testing.assert_allclose(est.windows.dxt, direct.dxt)
    np.testing.assert_allclose(est.windows.dyt, direct.dyt)
    assert "estimated from" in win._stats.text()

    win._on_apply()
    new = state.datasets[-1]
    assert new.prop.num_loc == est.n_kept
    # cfr/efc carried from the iteration they are measured at, not the last one's 0 / NaN
    out = np.asarray(new.attr["cfr"], dtype=float)
    np.testing.assert_allclose(np.sort(out), np.sort(cfr[passing]))
    assert np.isfinite(np.asarray(new.attr["efc"], dtype=float)).all()
    assert new.metadata["drift_correction"]["filter"] == {
        "attributes": ["cfr"], "n_kept": est.n_kept, "n_source": cfr.size,
        "not_applied": []}
    logs = [e["message"] for e in state.log_history]
    assert any("Drift correction (time windows)" in m and "passing the active filter on cfr"
               in m for m in logs)
    win.close()


def test_unfiltered_correction_still_carries_the_measured_cfr(qtbot):
    ds, cfr = _multi_iteration_dataset()
    win, state = _window(qtbot, ds)
    _windows_2d(win)
    est = _estimate(qtbot, win)
    assert not est.filtered and est.n_kept == cfr.size
    win._on_apply()
    new = state.datasets[-1]
    assert "filter" not in new.metadata["drift_correction"]
    np.testing.assert_allclose(np.sort(np.asarray(new.attr["cfr"], dtype=float)), np.sort(cfr))
    win.close()


def test_a_filter_change_makes_the_estimate_stale(qtbot, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    ds, _cfr = _multi_iteration_dataset()
    win, state = _window(qtbot, ds)
    _windows_2d(win)
    _estimate(qtbot, win)
    _set_filter(state, ds, _CFR_ABOVE_HALF)          # announced: dropped at once
    assert win._estimate is None and not win._apply_btn.isEnabled()
    assert "estimate the drift again" in win._stats.text()

    _estimate(qtbot, win)
    ds.state["filter_specs"] = []                    # not announced: caught at Apply
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    n = len(state.datasets)
    win._on_apply()
    assert len(state.datasets) == n
    assert shown and "filter changed" in shown[0]
    win.close()


def test_a_filter_that_removes_everything_is_reported(qtbot, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    ds, _cfr = _multi_iteration_dataset()
    win, state = _window(qtbot, ds)
    _set_filter(state, ds, [dict(_CFR_ABOVE_HALF[0], lo=2.0, hi=3.0)])
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    win._on_estimate()
    assert win._task is None and win._estimate is None
    assert shown and "removes every localization" in shown[0]
    win.close()


def test_bead_correction_is_applied_to_the_filtered_rows(qtbot):
    from minflux_viewer.core.loader import mfx_get
    from minflux_viewer.plugins.drift_correction.drift_correction_window import METHOD_BEADS

    ds = _bead_dataset()
    win, state = _window(qtbot, ds)
    win._method_combo.setCurrentIndex(win._method_combo.findData(METHOD_BEADS))
    _set_filter(state, ds, [{"attribute": "xnm", "mode": "per loc", "itr": "last",
                             "lo": -1e9, "hi": 0.0, "lo_inc": True, "hi_inc": True}])
    keep = np.asarray(mfx_get(ds, "xnm", itr="last", vld_only=True)) <= 0.0
    est = _estimate(qtbot, win)
    assert est.n_kept == int(keep.sum()) < keep.size
    assert est.recorded and max(est.beads.reproduction_nm) < 0.5    # beads unaffected
    disp = np.asarray(est.beads.positions_m)[keep] * 1e9
    np.testing.assert_allclose(est.coords_nm[:, :2], disp[:, :2])
    win._on_apply()
    assert state.datasets[-1].prop.num_loc == est.n_kept
    win.close()
