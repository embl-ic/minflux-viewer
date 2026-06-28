"""
Tests for cfr/efc effective-iteration value extraction, filtering, and the
histogram window's auto-switch to the effective iteration.

cfr/efc are only measured at one loop iteration; their materialized
last-iteration value is zero/NaN. The viewer treats the value at that
*effective* iteration as the canonical per-localization value (so histograms,
filters and scatter colour see real numbers), while still allowing per-iteration
browsing.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from minflux_viewer.core import loader


def _make_m2410_with_effective_cfr(n_loc=60, n_itr=5, eff=2, seed=0):
    """Flat m2410 mfx array where cfr/efc are measured only at iteration *eff*."""
    rng = np.random.RandomState(seed)
    n = n_loc * n_itr
    dt = np.dtype([
        ("vld", "?"), ("itr", "<i4"), ("tid", "<i4"),
        ("loc", "<f8", (3,)),
        ("cfr", "<f8"), ("efc", "<f8"), ("efo", "<f8"),
    ])
    mfx = np.zeros(n, dtype=dt)
    itr = np.tile(np.arange(n_itr), n_loc)
    mfx["itr"] = itr
    mfx["tid"] = np.repeat(np.arange(n_loc), n_itr)
    mfx["vld"] = True
    mfx["loc"][:, 0] = np.repeat(np.linspace(0.0, 1e-6, n_loc), n_itr)
    mfx["loc"][:, 1] = np.repeat(np.linspace(0.0, 1e-6, n_loc), n_itr)
    mfx["loc"][:, 2] = np.repeat(np.linspace(0.0, 5e-7, n_loc), n_itr)
    eff_rows = itr == eff
    mfx["cfr"][eff_rows] = rng.uniform(0.3, 0.8, int(eff_rows.sum()))
    mfx["efc"][:] = np.nan
    mfx["efc"][eff_rows] = rng.uniform(1e4, 1e5, int(eff_rows.sum()))
    mfx["efo"][:] = rng.uniform(1e4, 2e5, n)
    return mfx, eff


def test_effective_iteration_metadata_and_index():
    mfx, eff = _make_m2410_with_effective_cfr()
    ds = loader.load_from_mfx_array(mfx, "eff")
    assert ds.metadata.get("cfr_iteration") == eff + 1     # 1-based
    assert ds.metadata.get("efc_iteration") == eff + 1
    assert loader.effective_iteration_for_attr(ds, "cfr") == eff
    assert loader.effective_iteration_for_attr(ds, "efc") == eff
    assert loader.effective_iteration_for_attr(ds, "efo") is None
    assert loader.effective_iteration_for_attr(ds, "loc_x") is None


def test_attr_values_1d_returns_effective_not_last():
    mfx, _ = _make_m2410_with_effective_cfr()
    ds = loader.load_from_mfx_array(mfx, "eff")
    n = ds.prop.num_loc
    # Materialized last-iteration cfr is all zero (cfr only at the effective itr).
    last = np.asarray(ds.attr["cfr"]).ravel()
    assert np.count_nonzero(last) == 0
    # attr_values_1d gives the effective per-loc value: finite and in range.
    vals = np.asarray(loader.attr_values_1d(ds, "cfr")).ravel()
    assert vals.shape == (n,)
    assert np.isfinite(vals).all()
    assert vals.min() >= 0.3 - 1e-9 and vals.max() <= 0.8 + 1e-9
    efc = np.asarray(loader.attr_values_1d(ds, "efc")).ravel()
    assert np.isfinite(efc).all()


def test_effective_attr_values_none_when_no_special_iteration():
    mfx, _ = _make_m2410_with_effective_cfr()
    ds = loader.load_from_mfx_array(mfx, "eff")
    assert loader.effective_attr_values_1d(ds, "efo") is None
    assert loader.effective_attr_values_1d(ds, "loc_x") is None


def test_cfr_filter_uses_effective_values():
    mfx, _ = _make_m2410_with_effective_cfr()
    ds = loader.load_from_mfx_array(mfx, "eff")
    ds.state["filter_specs"] = [{
        "attribute": "cfr", "mode": "per loc",
        "lo": 0.4, "hi": 0.6, "lo_inc": True, "hi_inc": True,
    }]
    assert loader.apply_saved_filters(ds) is True
    expected = np.asarray(loader.attr_values_1d(ds, "cfr")).ravel()
    expected_mask = (expected >= 0.4) & (expected <= 0.6)
    assert np.array_equal(ds.filter_mask, expected_mask)
    # A genuine cfr filter keeps a non-trivial subset (not all / nothing).
    assert 0 < int(ds.filter_mask.sum()) < ds.prop.num_loc


def test_mfx_filter_mask_broadcasts_cfr_across_browse_iterations():
    """A cfr filter must not wipe out the view when browsing a non-effective
    iteration (where raw cfr is zero)."""
    mfx, _ = _make_m2410_with_effective_cfr()
    ds = loader.load_from_mfx_array(mfx, "eff")
    ds.state["filter_specs"] = [{
        "attribute": "cfr", "mode": "per loc",
        "lo": 0.4, "hi": 0.6, "lo_inc": True, "hi_inc": True,
    }]
    # Browse at the last iteration (raw cfr == 0 there).
    res = loader.mfx_filter_mask(ds, itr="last", vld_only=True)
    assert res is not None
    fmask, uneval = res
    assert uneval == []
    assert 0 < int(fmask.sum()) < fmask.size       # not zeroed out by cfr==0


def test_histogram_auto_switches_to_effective_iteration():
    try:
        from PyQt6.QtWidgets import QApplication
    except Exception:
        pytest.skip("PyQt6 not available")
    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841

    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.histogram_window import HistogramWindow

    mfx, eff = _make_m2410_with_effective_cfr()
    ds = loader.load_from_mfx_array(mfx, "eff")
    state = AppState()
    state.add_dataset(ds)

    win = HistogramWindow(state, dataset_idx=0)
    try:
        win._attr_combo.setCurrentText("efo")     # a normal attribute first
        win._attr_combo.setCurrentText("cfr")     # toggling to cfr auto-switches
        assert win._iter_combo.currentText() == "3rd"   # eff index 2 -> "3rd"
        # The effective selection routes through the materialized path so
        # filter-edit keeps working and the label matches the data shown.
        assert win._is_raw_mode() is False
    finally:
        win.close()


def test_filter_edit_on_cfr_displays_effective_iteration():
    try:
        from PyQt6.QtWidgets import QApplication
    except Exception:
        pytest.skip("PyQt6 not available")
    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841

    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.histogram_window import HistogramWindow

    mfx, eff = _make_m2410_with_effective_cfr()
    ds = loader.load_from_mfx_array(mfx, "eff")
    state = AppState()
    state.add_dataset(ds)

    win = HistogramWindow(state, dataset_idx=0)
    try:
        # Open in a non-effective state first (default last view, efo).
        win._attr_combo.setCurrentText("efo")
        win.start_filter_edit(attr="cfr", mode="per loc", lo=0.4, hi=0.6)
        # Editing a cfr filter lands on the effective iteration, materialized
        # path, with an active editable region.
        assert win._iter_combo.currentText() == "3rd"
        assert win._is_raw_mode() is False
        assert win._filter_edit is not None
        assert win._filter_edit.get("region") is not None
    finally:
        win.close()
