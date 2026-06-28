"""
Tests for post-load computed attributes (RIMF, localization precision, local
density) and the chained, event-loop-friendly scheduling that keeps the UI
responsive during dataset open.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pytest

from minflux_viewer.core.dataset import AttrStore, DataProp, FileInfo, MinfluxDataset


def _make_3d_ds(n_traces: int = 40, per: int = 12) -> MinfluxDataset:
    rng = np.random.default_rng(0)
    n = n_traces * per
    tid = np.repeat(np.arange(1, n_traces + 1), per)
    # Cluster each trace tightly around a random centre (nm → m).
    centres = rng.uniform(0, 5_000, size=(n_traces, 3))
    pts = np.repeat(centres, per, axis=0) + rng.normal(0, 8.0, size=(n, 3))
    pts_m = pts * 1e-9
    ti = np.column_stack([
        np.arange(n_traces) * per, np.arange(n_traces) * per + per - 1,
    ])
    prop = DataProp(
        num_loc=n, num_itr=1, num_dim=3, num_traces=n_traces,
        trace_idx=ti, num_loc_per_trace=np.full(n_traces, per),
        attr_names=["loc_x", "loc_y", "loc_z", "tid", "ftr"],
    )
    attrs = AttrStore({
        "loc_x": pts_m[:, 0], "loc_y": pts_m[:, 1], "loc_z": pts_m[:, 2],
        "tid": tid.astype(float), "ftr": np.ones(n, dtype=bool),
    })
    return MinfluxDataset(file=FileInfo(name="synth3d.mat", folder="/tmp"),
                          prop=prop, attr=attrs)


# ---------------------------------------------------------------------------
# Optimized grouping correctness (pure, no Qt)
# ---------------------------------------------------------------------------

def test_estimate_size_nd_grouping_matches_reference():
    from minflux_viewer.analysis.trace_analysis import estimate_size_nd_details

    rng = np.random.default_rng(3)
    ids = rng.integers(0, 200, 4000)          # many traces, id 0 present, gaps
    data = rng.normal(0, 4, (4000, 3))

    def reference_dist(col, excl):
        d = np.asarray(data[:, col], float)
        uid = np.unique(ids)
        if excl:
            uid = uid[uid != 0]
        parts = []
        for cid in uid:                       # original O(N×n_ids) approach
            pts = d[ids == cid]
            if len(pts) < 2:
                continue
            parts.append(np.abs(pts - np.median(pts)))
        out = np.concatenate(parts) if parts else np.array([])
        return np.sort(out[np.isfinite(out) & (out > 0)])

    for excl in (True, False):
        res = estimate_size_nd_details(data[:, 0], ids, exclude_zero_id=excl)
        np.testing.assert_allclose(np.sort(np.exp(res.logdist)), reference_dist(0, excl))


def test_stddev_per_trace_grouping_matches_reference():
    from minflux_viewer.analysis.localization_precision import stddev_per_trace

    rng = np.random.default_rng(4)
    tid = rng.integers(0, 100, 3000)
    loc_m = rng.normal(0, 5e-9, (3000, 3))
    res = stddev_per_trace(loc_m, tid)

    loc_nm = loc_m * 1e9
    uniq = np.unique(tid)
    ref_sig, ref_ids = [], []
    for t in uniq:
        sel = tid == t
        if int(sel.sum()) < 5:
            continue
        ref_sig.append(loc_nm[sel].std(axis=0, ddof=1))
        ref_ids.append(t)
    np.testing.assert_allclose(res["per_trace_sigma_xyz"], np.array(ref_sig))
    np.testing.assert_array_equal(res["trace_ids"], np.array(ref_ids))


# ---------------------------------------------------------------------------
# Chained post-load scheduling (Qt)
# ---------------------------------------------------------------------------

@pytest.fixture
def _qt_app():
    pytest.importorskip("PyQt6")
    if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
        pytest.skip("No display available for Qt tests")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


def _pump_until(app, predicate, timeout_s: float = 10.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_post_load_chain_and_abort(_qt_app):
    """The chained post-load fills RIMF/precision/density via the event loop,
    and a step on a removed/unknown dataset is a safe no-op. (One MainWindow:
    instantiating several pyqtgraph-heavy windows per process is teardown-fragile
    on Windows.)"""
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.main_window import MainWindow

    state = AppState()
    state.prefs.setdefault("data", {}).update({"show_data_info": False, "show_render": False})
    win = MainWindow(state)
    try:
        # Abort path: a dataset the window doesn't know is index None → no-op.
        orphan = _make_3d_ds()
        assert win._post_load_index(orphan) is None
        win._post_load_loc_prec(orphan)
        assert "sigma_per_trace_nm" not in orphan.derived

        # Happy path: chained steps complete through the event loop.
        ds = _make_3d_ds()
        state.add_dataset(ds)
        ok = _pump_until(_qt_app, lambda: "den" in ds.attr and "rimf" in ds.derived
                         and "sigma_per_trace_nm" in ds.derived)
        assert ok, "post-load chain did not finish"
        assert np.isfinite(np.asarray(ds.derived["rimf"])).all()
        assert np.asarray(ds.attr["den"]).shape[0] == ds.prop.num_loc
    finally:
        win.close()
        _qt_app.processEvents()
