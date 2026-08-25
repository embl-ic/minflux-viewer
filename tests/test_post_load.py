"""
Tests for post-load computed attributes (Z scaling factor, localization precision, local
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


def test_no_z_scaling_factor_correction_is_the_fresh_install_default():
    """A new install applies **no** Z scaling factor policy: neither *estimate Z scaling factor from
    anisotropy* nor *use fixed value* is on, so z is left as the file recorded it
    until the user opts in."""
    from minflux_viewer.core.app_state import DEFAULT_PREFS, default_prefs

    assert DEFAULT_PREFS["data"]["estimate_z_scaling_factor"] is False
    assert DEFAULT_PREFS["plot"]["use_fixed_z_scaling_factor"] is False
    assert DEFAULT_PREFS["plot"]["z_scaling_factor"] == pytest.approx(0.67)

    fresh = default_prefs()
    assert fresh["data"]["estimate_z_scaling_factor"] is False
    assert fresh["plot"]["use_fixed_z_scaling_factor"] is False
    assert fresh["plot"]["z_scaling_factor"] == pytest.approx(0.67)


def test_fresh_prefs_are_exactly_the_declared_defaults():
    """Regression: the fresh-install path is ``_migrate_prefs(deepcopy(DEFAULT_PREFS))``,
    and a fresh dict has no ``_migrations`` key — so every one-shot migration used to
    fire against the defaults and could silently overwrite them (``v036`` forced
    ``use_fixed_z_scaling_factor`` back to True, defeating any change to ``DEFAULT_PREFS``).
    Changing a default must be enough on its own."""
    from minflux_viewer.core.app_state import (
        DEFAULT_PREFS, _MIGRATION_KEYS, default_prefs,
    )

    fresh = default_prefs()
    assert set(fresh.pop("_migrations")) == set(_MIGRATION_KEYS)
    assert fresh == DEFAULT_PREFS


def test_migrations_still_rewrite_genuinely_old_saved_preferences():
    """The one-shot blocks must keep working for preferences saved before them."""
    from minflux_viewer.core.app_state import _migrate_prefs

    old = {
        "data": {"estimate_z_scaling_factor": True},
        "plot": {"use_fixed_z_scaling_factor": False, "z_scaling_factor": 1.0,
                 "render_alignment_translation_px": 4.0,
                 "render_alignment_rotation_deg": 0.5},
        "_migrations": {
            "v021_compute_show_defaults": True,
            "v035_update_check_optin": True,
        },
    }
    migrated = _migrate_prefs(old)
    # v036 still applies to this saved set (it never ran here)
    assert migrated["data"]["estimate_z_scaling_factor"] is False
    assert migrated["plot"]["use_fixed_z_scaling_factor"] is True
    assert migrated["plot"]["z_scaling_factor"] == pytest.approx(0.67)
    # …as does v037
    assert "render_alignment_translation_px" not in migrated["plot"]
    assert migrated["plot"]["render_alignment_rotation_deg"] == pytest.approx(0.1)


def test_legacy_z_scaling_preference_names_are_migrated_once():
    """Saved preferences are the only established persistence boundary for the
    terminology change; preserve their values while deleting the old keys."""
    from minflux_viewer.core.app_state import _MIGRATION_KEYS, _migrate_prefs

    migrations = {key: True for key in _MIGRATION_KEYS if "v036" not in key}
    migrations["v036_fixed_rimf_default"] = True
    old = {
        "data": {"compute_rimf": True},
        "plot": {"use_fixed_rimf": False, "rimf_value": 0.74},
        "_migrations": migrations,
    }

    migrated = _migrate_prefs(old)

    assert migrated["data"]["estimate_z_scaling_factor"] is True
    assert migrated["plot"]["use_fixed_z_scaling_factor"] is False
    assert migrated["plot"]["z_scaling_factor"] == pytest.approx(0.74)
    assert "compute_rimf" not in migrated["data"]
    assert "use_fixed_rimf" not in migrated["plot"]
    assert "rimf_value" not in migrated["plot"]
    assert migrated["_migrations"]["v036_fixed_z_scaling_factor_default"] is True
    assert "v036_fixed_rimf_default" not in migrated["_migrations"]


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
    """The chained post-load fills Z scaling factor/precision/density via the event loop,
    and a step on a removed/unknown dataset is a safe no-op. (One MainWindow:
    instantiating several pyqtgraph-heavy windows per process is teardown-fragile
    on Windows.)"""
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.main_window import MainWindow

    state = AppState()
    state.prefs.setdefault("data", {}).update({"show_data_info": False, "show_render": False})
    state.prefs["data"]["estimate_z_scaling_factor"] = False
    state.prefs.setdefault("plot", {}).update({"use_fixed_z_scaling_factor": True, "z_scaling_factor": 0.67})
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
        ok = _pump_until(_qt_app, lambda: "den" in ds.attr and "z_scaling_factor" in ds.derived
                         and "sigma_per_trace_nm" in ds.derived)
        assert ok, "post-load chain did not finish"
        assert np.isfinite(np.asarray(ds.derived["z_scaling_factor"])).all()
        assert float(ds.cali.z_scaling_factor) == pytest.approx(0.67)
        assert ds.z_scaling_factor_provenance["source"] == "fixed (preference)"
        assert np.asarray(ds.attr["den"]).shape[0] == ds.prop.num_loc

        # The Log window auto-opens on the first log message (placed beside the
        # main window, not over its menu) — so after the post-load chain's many
        # log lines it exists.
        assert win._log_win is not None

        # Averaged super-particles skip per-trace precision + local density on load
        # (their tid is a per-loc placeholder and the dense superposition makes the
        # KD-tree density heuristic mis-estimate → a long all-core freeze).
        from minflux_viewer.core.dataset import build_localization_dataset
        pa_pts = np.random.default_rng(0).normal(0, 50, (200, 3))
        ds_pa = build_localization_dataset(name="pa", x_nm=pa_pts[:, 0], y_nm=pa_pts[:, 1],
                                           z_nm=pa_pts[:, 2], prefs=state.prefs)
        ds_pa.metadata["particle_average"] = True
        ds_pa.derived["z_scaling_factor"] = 1.0
        state.add_dataset(ds_pa)                       # so _post_load_index resolves it
        win._post_load_loc_prec(ds_pa)
        assert "sigma_per_trace_nm" not in ds_pa.derived   # loc-precision skipped
        win._post_load_density(ds_pa)
        assert "den" not in ds_pa.attr                     # local density skipped
    finally:
        win.close()
        _qt_app.processEvents()


# ---------------------------------------------------------------------------
# Retired Data preferences (iter_load / load_efc_cfr / load_all_dcr)
# ---------------------------------------------------------------------------
RETIRED_DATA_PREFS = ("iter_load", "load_efc_cfr", "load_all_dcr")


def test_retired_data_preferences_are_gone():
    """These three no longer exist.

    ``iter_load`` materialized every iteration as if each were a localization
    (inflating num_loc and num_traces); browsing serves the same rows from
    ``mfx_raw`` without touching the default view. ``load_efc_cfr`` gated a
    behaviour the always-on effective-iteration machinery already guarantees,
    and ``load_all_dcr`` was read into a variable nothing ever used.
    """
    from minflux_viewer.core.app_state import DEFAULT_PREFS

    for key in RETIRED_DATA_PREFS:
        assert key not in DEFAULT_PREFS["data"], key


def test_a_stale_saved_value_for_a_retired_preference_is_inert(tmp_path):
    """An existing install carries these keys in its saved prefs; they must not
    change the load any more."""
    import numpy as np
    import scipy.io as sio

    from minflux_viewer.core.loader import load_dataset

    itr = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
    path = tmp_path / "flat.mat"
    sio.savemat(str(path), {
        "itr": itr,
        "tid": np.array([1, 1, 1, 2, 2, 2], dtype=np.int32),
        "vld": np.ones(6, dtype=bool),
        "tim": np.linspace(0.0, 1.0, 6),
        "loc": np.random.default_rng(3).normal(size=(6, 3)) * 1e-6,
        "efo": np.random.default_rng(4).random(6) * 1e5,
    })

    clean = load_dataset(path, prefs={"data": {}})
    stale = load_dataset(path, prefs={"data": {
        "iter_load": "all", "load_efc_cfr": False, "load_all_dcr": True}})

    assert clean.prop.num_loc == stale.prop.num_loc == 2      # last valid only
    assert clean.prop.num_traces == stale.prop.num_traces == 2
    assert clean.metadata["iteration_load_mode"] == "last"
    assert stale.metadata["iteration_load_mode"] == "last"


# ---------------------------------------------------------------------------
# 2D/3D dimensionality threshold (enforce_min_z_range / min_z_range_nm)
# ---------------------------------------------------------------------------
def _flat_2d_mfx(z_metres):
    """A 6-row m2410 table whose Z carries a tiny residual instead of zero."""
    import numpy as np

    dt = np.dtype([("vld", np.bool_), ("tid", np.int32), ("tim", np.float64),
                   ("itr", np.int32), ("loc", np.float64, (3,))])
    a = np.zeros(6, dt)
    a["vld"] = True
    a["tid"] = np.array([1, 1, 1, 2, 2, 2], dtype=np.int32)
    a["tim"] = np.linspace(0.0, 1.0, 6)
    a["itr"] = 2
    rng = np.random.default_rng(0)
    a["loc"][:, 0] = rng.uniform(0, 1e-6, 6)
    a["loc"][:, 1] = rng.uniform(0, 1e-6, 6)
    a["loc"][:, 2] = z_metres
    return a


def test_tiny_residual_z_is_flattened_to_2d():
    """A 2D acquisition writes a sub-picometre residual Z, not exact zero.

    Measured on a real 2D file (2_3C_measurement.msr): every localization of all
    three channels had a non-zero Z, spread over 0.0003-0.0007 nm. Judging on
    "is any Z non-zero" alone classifies that as 3D, which then runs Z scaling factor
    estimation on noise and offers the depth slider / XZ-YZ / volume views.
    """
    import numpy as np

    from minflux_viewer.core.loader import load_from_mfx_array

    # ~0.27 pm of spread around a -0.27 pm offset — the real file's scale.
    z = np.linspace(-2.68e-13, -1.3e-15, 6)
    mfx = _flat_2d_mfx(z)

    off = load_from_mfx_array(mfx, name="off", prefs={"data": {
        "enforce_min_z_range": False}})
    on = load_from_mfx_array(mfx, name="on", prefs={"data": {
        "enforce_min_z_range": True, "min_z_range_nm": 5.0}})

    assert off.prop.num_dim == 3, "without the threshold the residual reads as 3D"
    assert on.prop.num_dim == 2, "the threshold must flatten it to 2D"
    # ...and the flattened axis is genuinely zeroed, not merely relabelled.
    assert np.all(np.asarray(on.attr["loc_z"], dtype=float) == 0.0)


def test_genuine_3d_survives_the_threshold():
    """Real 3D data spans thousands of nm, far from the decision boundary."""
    import numpy as np

    from minflux_viewer.core.loader import load_from_mfx_array

    mfx = _flat_2d_mfx(np.linspace(-1.1e-6, 1.1e-6, 6))   # 2200 nm of Z
    ds = load_from_mfx_array(mfx, name="d", prefs={"data": {
        "enforce_min_z_range": True, "min_z_range_nm": 5.0}})
    assert ds.prop.num_dim == 3
