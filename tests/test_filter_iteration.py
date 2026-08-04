"""
Tests for iteration-aware filter specs.

A filter spec may carry an optional ``"itr"`` key recording *which* iteration
its bound was authored against, so the same logical filter is reproducible
across iteration browsing and JSON round-trips:

- ``"last"``      — the localization's final valid iteration (broadcast onto all
                    of its iteration rows; render/scatter parity).
- ``"effective"`` — cfr/efc's measured iteration.
- ``"all"``       — evaluate at each browse row's own value (per-iteration).
- ``int``         — a specific 0-based iteration (broadcast).
- absent          — legacy auto default: effective for cfr/efc, else per-browse-row.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from minflux_viewer.core import loader
from minflux_viewer.core import filter_io


N_LOC = 10
N_ITR = 4


def _make_m2410_per_iter_efo(n_loc=N_LOC, n_itr=N_ITR):
    """Flat m2410 mfx array with a deterministic per-iteration efo.

    ``efo[loc, itr] = 1000*itr + loc`` so each iteration's values live in a
    disjoint band (itr0: 0..9, itr1: 1000..1009, … itr3: 3000..3009). This makes
    an iteration-aware filter's result cleanly attributable to the iteration it
    was evaluated at. cfr is measured only at iteration ``eff=1``.
    """
    n = n_loc * n_itr
    dt = np.dtype([
        ("vld", "?"), ("itr", "<i4"), ("tid", "<i4"),
        ("loc", "<f8", (3,)),
        ("cfr", "<f8"), ("efo", "<f8"),
    ])
    mfx = np.zeros(n, dtype=dt)
    itr = np.tile(np.arange(n_itr), n_loc)
    loc_idx = np.repeat(np.arange(n_loc), n_itr)
    mfx["itr"] = itr
    mfx["tid"] = loc_idx
    mfx["vld"] = True
    mfx["loc"][:, 0] = np.repeat(np.linspace(0.0, 1e-6, n_loc), n_itr)
    mfx["loc"][:, 1] = np.repeat(np.linspace(0.0, 1e-6, n_loc), n_itr)
    mfx["efo"] = 1000.0 * itr + loc_idx
    eff = 1
    eff_rows = itr == eff
    mfx["cfr"][eff_rows] = np.linspace(0.3, 0.8, int(eff_rows.sum()))
    return mfx, eff


def _spec(attr, lo, hi, *, mode="per loc", itr=None):
    s = {"attribute": attr, "mode": mode, "lo": float(lo), "hi": float(hi),
         "lo_inc": True, "hi_inc": True}
    if itr is not None:
        s["itr"] = itr
    return s


# ---------------------------------------------------------------------------
# resolve_spec_iteration (pure logic)
# ---------------------------------------------------------------------------

def test_resolve_spec_iteration_tokens():
    mfx, eff = _make_m2410_per_iter_efo()
    ds = loader.load_from_mfx_array(mfx, "iter")

    # Absent itr (a filter file saved before the column existed): cfr/efc ->
    # "effective" (their measured iteration), every other attribute -> "last".
    assert loader.resolve_spec_iteration(ds, _spec("cfr", 0, 1)) == "effective"
    assert loader.resolve_spec_iteration(ds, _spec("efo", 0, 1)) == "last"

    # Explicit tokens pass through (case/whitespace-insensitive).
    assert loader.resolve_spec_iteration(ds, _spec("efo", 0, 1, itr="last")) == "last"
    assert loader.resolve_spec_iteration(ds, _spec("efo", 0, 1, itr="All")) == "all"
    assert loader.resolve_spec_iteration(ds, _spec("cfr", 0, 1, itr="effective")) == "effective"
    assert loader.resolve_spec_iteration(ds, _spec("efo", 0, 1, itr=2)) == 2
    assert loader.resolve_spec_iteration(ds, _spec("efo", 0, 1, itr="2")) == 2
    assert loader.resolve_spec_iteration(ds, _spec("efo", 0, 1, itr="sum")) == "sum"
    assert loader.resolve_spec_iteration(ds, _spec("efo", 0, 1, itr="Average")) == "average"

    # A stray bool is not an int iteration; unknown text falls back to "last".
    assert loader.resolve_spec_iteration(ds, _spec("efo", 0, 1, itr="weird")) == "last"


# ---------------------------------------------------------------------------
# Per-spec iteration evaluation in mfx_filter_mask
# ---------------------------------------------------------------------------

def test_itr_last_broadcasts_across_browse_iterations():
    """An itr='last' efo filter keeps/drops a localization's whole iteration
    stack by its LAST value — matching render/scatter, not a per-row test."""
    mfx, _ = _make_m2410_per_iter_efo()
    ds = loader.load_from_mfx_array(mfx, "iter")
    # Last-iteration efo values are 3000..3009; keep locs 0..4.
    ds.state["filter_specs"] = [_spec("efo", 3000, 3004, itr="last")]

    # Browse all iterations: every row of the 5 passing locs is kept -> 5*N_ITR.
    res = loader.mfx_filter_mask(ds, itr="all", vld_only=True)
    assert res is not None
    mask, uneval = res
    assert uneval == []
    assert mask.size == N_LOC * N_ITR
    assert int(mask.sum()) == 5 * N_ITR

    # And the canonical (last) selection keeps exactly those 5 localizations.
    mask_last, _ = loader.mfx_filter_mask(ds, itr="last", vld_only=True)
    assert int(mask_last.sum()) == 5


def test_absent_itr_filters_on_last():
    """A filter file with no iteration info is read as a filter on `last`, so a
    localization's whole iteration stack is kept/dropped by its last value."""
    mfx, _ = _make_m2410_per_iter_efo()
    ds = loader.load_from_mfx_array(mfx, "iter")
    ds.state["filter_specs"] = [_spec("efo", 3000, 3004)]  # no itr

    res = loader.mfx_filter_mask(ds, itr="all", vld_only=True)
    mask, uneval = res
    assert uneval == []
    # Last-iteration efo of locs 0..4 falls in [3000, 3004]; all their rows pass.
    assert int(mask.sum()) == 5 * N_ITR
    # Identical to spelling the selector out explicitly.
    ds.state["filter_specs"] = [_spec("efo", 3000, 3004, itr="last")]
    explicit, _ = loader.mfx_filter_mask(ds, itr="all", vld_only=True)
    assert np.array_equal(mask, explicit)


def test_itr_specific_iteration_broadcasts_that_iterations_value():
    """itr=<int> tests the bound against that iteration's value and broadcasts
    the pass/fail across the localization's rows."""
    mfx, _ = _make_m2410_per_iter_efo()
    ds = loader.load_from_mfx_array(mfx, "iter")

    # Iteration 1 efo values are 1000..1009; keep locs 0..4 via iteration 1.
    ds.state["filter_specs"] = [_spec("efo", 1000, 1004, itr=1)]
    mask_last, uneval = loader.mfx_filter_mask(ds, itr="last", vld_only=True)
    assert uneval == []
    assert int(mask_last.sum()) == 5           # decided by iteration 1, not last

    # The same bound at itr='last' matches nothing (last band is 3000..3009).
    ds.state["filter_specs"] = [_spec("efo", 1000, 1004, itr="last")]
    mask_last2, _ = loader.mfx_filter_mask(ds, itr="last", vld_only=True)
    assert int(mask_last2.sum()) == 0


def test_absent_cfr_itr_still_uses_effective_iteration():
    """Regression: a cfr filter with no itr must broadcast its effective-
    iteration value, not the zero/NaN last-iteration value."""
    mfx, eff = _make_m2410_per_iter_efo()
    ds = loader.load_from_mfx_array(mfx, "iter")
    ds.state["filter_specs"] = [_spec("cfr", 0.45, 0.65)]  # no itr -> effective

    res = loader.mfx_filter_mask(ds, itr="last", vld_only=True)
    mask, uneval = res
    assert uneval == []
    assert 0 < int(mask.sum()) < mask.size     # not wiped out by cfr==0 at last


# ---------------------------------------------------------------------------
# apply_saved_filters honours per-spec iteration
# ---------------------------------------------------------------------------

def test_apply_saved_filters_honours_explicit_iteration():
    mfx, _ = _make_m2410_per_iter_efo()
    ds = loader.load_from_mfx_array(mfx, "iter")

    ds.state["filter_specs"] = [_spec("efo", 1000, 1004, itr=1)]
    assert loader.apply_saved_filters(ds) is True
    # Canonical mask aligns to ds.attr (last-valid) rows = one per localization.
    assert ds.filter_mask.shape[0] == ds.prop.num_loc
    assert int(ds.filter_mask.sum()) == 5       # decided by iteration 1


def test_apply_saved_filters_backward_compatible_without_itr():
    """A spec with no itr reproduces the pre-change result exactly."""
    mfx, _ = _make_m2410_per_iter_efo()
    ds = loader.load_from_mfx_array(mfx, "iter")
    ds.state["filter_specs"] = [_spec("efo", 3000, 3004)]  # no itr
    assert loader.apply_saved_filters(ds) is True

    # Old semantics: evaluate against the materialized last-valid efo per loc.
    ref = np.asarray(loader.attr_values_1d(ds, "efo")).ravel()
    expected = (ref >= 3000) & (ref <= 3004)
    assert np.array_equal(ds.filter_mask, expected)


# ---------------------------------------------------------------------------
# JSON filter-preset round-trip of the iteration selector
# ---------------------------------------------------------------------------

def test_filter_preset_json_round_trips_iteration(tmp_path):
    rows = [
        {"apply": True, "attribute": "efo", "value_as": "per loc",
         "min": 3000.0, "max": 3004.0,
         "min_inclusive": True, "max_inclusive": True, "iteration": "last"},
        {"apply": True, "attribute": "cfr", "value_as": "per loc",
         "min": 0.45, "max": 0.65,
         "min_inclusive": True, "max_inclusive": True, "iteration": "effective"},
    ]
    # A row carrying "iteration" is still recognised as a filter preset.
    assert filter_io.is_filter_json_payload(rows) is True

    path = tmp_path / "flt.json"
    filter_io.save_filter_json(str(path), rows)
    loaded = filter_io.load_filter_json(str(path))
    assert [r.get("iteration") for r in loaded] == ["last", "effective"]


def test_filter_preset_json_rejects_itr_data_key():
    """A row keyed with the raw data-column name 'itr' must NOT validate as a
    filter preset (that is why the selector is stored under 'iteration')."""
    bad = [{"attribute": "efo", "min": 0.0, "max": 1.0, "itr": 3}]
    assert filter_io.is_filter_json_payload(bad) is False


# ---------------------------------------------------------------------------
# FilterDialog authoring persists the iteration selector (UI)
# ---------------------------------------------------------------------------

def _make_dialog_with_dataset():
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.filter_dialog import FilterDialog

    mfx, _ = _make_m2410_per_iter_efo()
    ds = loader.load_from_mfx_array(mfx, "iter")
    state = AppState()
    state.add_dataset(ds)
    dlg = FilterDialog(state, dataset_idx=0)
    return dlg, ds


def test_apply_all_persists_iteration_selector():
    try:
        from PyQt6.QtWidgets import QApplication, QCheckBox
    except Exception:
        pytest.skip("PyQt6 not available")
    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841

    from minflux_viewer.ui.filter_dialog import _COL_ENABLED, _COL_ATTR

    dlg, ds = _make_dialog_with_dataset()
    try:
        # Author an efo row and a cfr row, enable both, apply.
        dlg._add_row(attr="efo", mode="per loc", lo=3000.0, hi=3004.0,
                     enabled=True, auto_range=False)
        dlg._add_row(attr="cfr", mode="per loc", lo=0.45, hi=0.65,
                     enabled=True, auto_range=False)
        dlg._apply_all()

        specs = ds.state.get("filter_specs")
        by_attr = {s["attribute"]: s for s in specs}
        # Normal attr authored at last-valid; cfr authored at its effective iter.
        assert by_attr["efo"]["itr"] == "last"
        assert by_attr["cfr"]["itr"] == "effective"
    finally:
        dlg.close()


def test_loaded_iteration_survives_reapply(tmp_path):
    try:
        from PyQt6.QtWidgets import QApplication
    except Exception:
        pytest.skip("PyQt6 not available")
    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841

    dlg, ds = _make_dialog_with_dataset()
    try:
        # A hand-authored preset targeting iteration 1 for efo.
        rows = [{"apply": True, "attribute": "efo", "value_as": "per loc",
                 "min": 1000.0, "max": 1004.0, "min_inclusive": True,
                 "max_inclusive": True, "iteration": 1}]
        path = tmp_path / "flt.json"
        filter_io.save_filter_json(str(path), rows)

        dlg.load_filter_json(str(path))
        dlg._apply_all()

        specs = ds.state.get("filter_specs")
        assert len(specs) == 1
        # The loaded iteration (1) is preserved through _apply_all.
        assert specs[0]["itr"] == 1
        assert int(ds.filter_mask.sum()) == 5   # decided by iteration 1
    finally:
        dlg.close()
