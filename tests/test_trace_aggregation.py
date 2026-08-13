"""
Trace read-outs: the shared registry, and its use by the histogram and filter.

Three decisions this pins down:

1. A trace read-out is computed from **every** localization in the trace — the
   active filter is deliberately ignored, so a trace's mean/median/... does not
   shift as an unrelated filter is tuned. To get filtered trace values, save a
   filtered copy (``content="snapshot"``, ``filter_mode="apply"``) and reload it.
2. ``all [flatten]`` pools every raw row **equally** — no weighting by iteration.
3. The filter offers the same trace read-outs as the histogram, so any view the
   user can author is expressible as a filter.

All trace aggregation dispatches through ``utils.filters.TRACE_AGG_FUNCS``. It
used to be a table per call site, and a mode added to one but not another
silently degraded to per-localization values.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest

from minflux_viewer.core import loader
from minflux_viewer.utils.filters import (
    AGG_MODES,
    FLOAT_RESULT_MODES,
    TRACE_AGG_FUNCS,
    aggregate,
    compute_filter_mask,
    raw_spec_mask,
    raw_trace_aggregate,
    trace_agg_func,
)


N_LOC, N_ITR = 6, 3
#: value(loc, itr) = BASE[loc] + itr — every read-out is hand-checkable. The two
#: traces are deliberately spaced *differently* so that even ``stdev``/``range``
#: (which only see the spread) come out distinct; with evenly-spaced values a
#: bound on those would match both traces and prove nothing.
BASE = np.array([100.0, 200.0, 300.0, 400.0, 600.0, 900.0])
V = BASE[:, None] + np.arange(N_ITR)[None, :]
TRACES = [np.arange(0, 3), np.arange(3, 6)]

_APP = None


def _qapp():
    global _APP
    try:
        from PyQt6.QtWidgets import QApplication
    except Exception:                                        # pragma: no cover
        pytest.skip("PyQt6 not available")
    if _APP is None:
        _APP = QApplication.instance() or QApplication(sys.argv)
    return _APP


def _make_dataset():
    n = N_LOC * N_ITR
    dt = np.dtype([("vld", "?"), ("itr", "<i4"), ("tid", "<i4"),
                   ("loc", "<f8", (3,)), ("efo", "<f8"), ("eco", "<i4")])
    mfx = np.zeros(n, dtype=dt)
    itr = np.tile(np.arange(N_ITR), N_LOC)
    li = np.repeat(np.arange(N_LOC), N_ITR)
    mfx["itr"] = itr
    mfx["tid"] = li // 3
    mfx["vld"] = True
    mfx["loc"][:, 0] = np.repeat(np.linspace(0.0, 1e-6, N_LOC), N_ITR)
    mfx["efo"] = BASE[li] + itr
    mfx["eco"] = 10
    return loader.load_from_mfx_array(mfx, "agg")


def test_the_fixture_separates_every_read_out():
    """Guard the fixture itself: each trace read-out must give the two traces
    different values, or the bound tests below would pass vacuously."""
    for mode, fn in TRACE_AGG_FUNCS.items():
        a, b = (float(fn(V[t, -1])) for t in TRACES)
        assert a != pytest.approx(b), mode


def _histogram(ds=None):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.histogram_window import HistogramWindow

    ds = ds if ds is not None else _make_dataset()
    state = AppState()
    state.prefs["plot"]["histogram_values"] = list(TRACE_AGG_FUNCS)
    state.add_dataset(ds)
    win = HistogramWindow(state, dataset_idx=0)
    win._attr_combo.setCurrentText("efo")
    return win, ds, state


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

def test_registry_is_the_documented_set_in_dropdown_order():
    assert list(TRACE_AGG_FUNCS) == [
        "trace mean", "trace median", "trace min", "trace max",
        "trace 1st", "trace last", "trace stdev", "trace range",
    ]
    assert AGG_MODES == ["per loc"] + list(TRACE_AGG_FUNCS)
    # Derived statistics are float; value-returning read-outs are not.
    assert FLOAT_RESULT_MODES == {
        "trace mean", "trace median", "trace stdev", "trace range",
    }
    for mode in ("trace min", "trace max", "trace 1st", "trace last"):
        assert mode not in FLOAT_RESULT_MODES


def test_trace_agg_func_rejects_unknown_modes():
    assert trace_agg_func("per loc") is None
    for mode in TRACE_AGG_FUNCS:
        assert callable(trace_agg_func(mode))
    with pytest.raises(ValueError):
        trace_agg_func("trace nonsense")


def test_positional_read_outs_are_literal_not_order_statistics():
    a = np.array([5.0, 1.0, 9.0, 3.0])
    assert TRACE_AGG_FUNCS["trace 1st"](a) == 5.0
    assert TRACE_AGG_FUNCS["trace last"](a) == 3.0
    assert TRACE_AGG_FUNCS["trace min"](a) == 1.0
    assert TRACE_AGG_FUNCS["trace max"](a) == 9.0
    # Empty trace -> NaN rather than an IndexError.
    assert np.isnan(TRACE_AGG_FUNCS["trace 1st"](np.array([])))
    assert np.isnan(TRACE_AGG_FUNCS["trace last"](np.array([])))


def test_all_nan_trace_statistics_are_missing_without_runtime_warnings():
    """Unmapped fluorescent traces are valid missing data, not warning cases."""
    all_nan = np.array([np.nan, np.nan])
    mixed = np.array([1.0, np.nan, 3.0])
    raw = np.concatenate([all_nan, mixed])
    trace_idx = np.array([[0, 1], [2, 4]])
    counts = np.array([2, 3])
    tid = np.array([0, 0, 1, 1, 1])
    nan_skipping_modes = (
        "trace mean",
        "trace median",
        "trace min",
        "trace max",
        "trace stdev",
        "trace range",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for mode in nan_skipping_modes:
            assert np.isnan(TRACE_AGG_FUNCS[mode](all_nan)), mode
            values = aggregate(
                raw, np.ones(raw.size, dtype=bool), mode, trace_idx, 2
            )
            assert np.isnan(values[0]), mode
            assert np.isfinite(values[1]), mode
            raw_values = raw_trace_aggregate(raw, tid, mode)
            assert np.isnan(raw_values[0]), mode
            assert np.isfinite(raw_values[1]), mode

            mask = compute_filter_mask(
                raw, mode, -1.0, 10.0, trace_idx, counts, 2
            )
            assert mask.tolist() == [False, False, True, True, True], mode
            raw_mask = raw_spec_mask(raw, tid, mode, -1.0, 10.0)
            assert raw_mask.tolist() == mask.tolist(), mode


def test_histogram_trace_mean_and_median_accept_fully_unmapped_traces():
    _qapp()
    win, ds, _state = _histogram()
    try:
        raw = np.array([np.nan, np.nan, np.nan, 1.0, np.nan, 3.0])
        ftr = np.ones(raw.size, dtype=bool)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for mode in ("trace mean", "trace median"):
                values = win._aggregate(raw, ftr, mode, ds)
                assert np.isnan(values[0]), mode
                assert values[1] == pytest.approx(2.0), mode
    finally:
        win.close()


def test_every_dispatch_path_agrees_for_every_mode():
    """The four aggregation entry points must produce identical trace values —
    a mode missing from one of them used to fall through to per-row values."""
    _qapp()
    win, ds, _state = _histogram()
    try:
        raw = np.asarray(loader.attr_values_1d(ds, "efo"), dtype=float)
        ftr = np.ones(raw.size, dtype=bool)
        tid = np.asarray(loader.attr_values_1d(ds, "tid")).ravel()
        trace_idx = np.asarray(ds.prop.trace_idx, dtype=int)
        n_tr = int(ds.prop.num_traces)

        for mode in TRACE_AGG_FUNCS:
            expected = np.sort([TRACE_AGG_FUNCS[mode](V[t, -1]) for t in TRACES])
            got = {
                "histogram._aggregate": win._aggregate(raw, ftr, mode, ds),
                "filters.aggregate": aggregate(raw, ftr, mode, trace_idx, n_tr),
                "raw_trace_aggregate": raw_trace_aggregate(raw, tid, mode),
            }
            for name, vals in got.items():
                vals = np.sort(np.asarray(vals, dtype=float))
                assert vals.size == n_tr, f"{mode} via {name}: not one per trace"
                assert np.allclose(vals, expected), f"{mode} via {name}"
    finally:
        win.close()


def test_mask_builders_support_every_mode():
    """``raw_spec_mask`` / ``compute_filter_mask`` previously lacked median and
    the positional read-outs and silently tested per-row values instead."""
    _qapp()
    _win, ds, _state = _histogram()
    try:
        raw = np.asarray(loader.attr_values_1d(ds, "efo"), dtype=float)
        tid = np.asarray(loader.attr_values_1d(ds, "tid")).ravel()
        trace_idx = np.asarray(ds.prop.trace_idx, dtype=int)
        counts = np.asarray([b - a + 1 for a, b in trace_idx], dtype=int)
        n_tr = int(ds.prop.num_traces)

        for mode in TRACE_AGG_FUNCS:
            per_trace = np.asarray(
                [TRACE_AGG_FUNCS[mode](V[t, -1]) for t in TRACES], dtype=float
            )
            # A band around trace 0's value keeps exactly trace 0's rows.
            lo, hi = per_trace[0] - 0.5, per_trace[0] + 0.5
            for name, mask in (
                ("raw_spec_mask", raw_spec_mask(raw, tid, mode, lo, hi)),
                ("compute_filter_mask",
                 compute_filter_mask(raw, mode, lo, hi, trace_idx, counts, n_tr)),
            ):
                mask = np.asarray(mask, dtype=bool)
                assert mask.size == raw.size, f"{mode} via {name}"
                kept = set(np.flatnonzero(mask))
                assert kept == set(TRACES[0]), f"{mode} via {name}: kept {kept}"
    finally:
        _win.close()


def test_mask_builders_reject_unknown_modes():
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    tid = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError):
        raw_spec_mask(vals, tid, "trace nonsense", 0.0, 10.0)
    with pytest.raises(ValueError):
        compute_filter_mask(vals, "trace nonsense", 0.0, 10.0,
                            np.array([[0, 1], [2, 3]]), np.array([2, 2]), 2)


# ---------------------------------------------------------------------------
# (2) all [flatten] is an unweighted pool
# ---------------------------------------------------------------------------

def test_flatten_pools_every_raw_row_with_equal_weight():
    """`all [flatten]` is a plain pool of all rows — it does NOT weight a row by
    its iteration index."""
    _qapp()
    win, ds, _state = _histogram()
    try:
        got, _uneval = win._raw_values(ds, "efo", "all", True, "trace mean")
        got = float(np.sort(np.asarray(got, dtype=float))[0])

        rows = V[0:3].ravel()                       # trace 0's 9 raw rows
        plain = float(rows.mean())
        weighted = float((V[0:3] * (np.arange(N_ITR) + 1)).sum()
                         / (np.arange(N_ITR) + 1).sum() )
        assert got == pytest.approx(plain)
        assert got != pytest.approx(weighted)       # nothing is scaled by itr

        # per loc under flatten is simply every raw row.
        per_loc, _ = win._raw_values(ds, "efo", "all", True, "per loc")
        assert np.allclose(np.sort(per_loc), np.sort(V.ravel()))
    finally:
        win.close()


# ---------------------------------------------------------------------------
# (3) filter parity with the histogram
# ---------------------------------------------------------------------------

def test_filter_dialog_offers_the_same_trace_read_outs_as_the_histogram():
    _qapp()
    from minflux_viewer.ui import filter_dialog as fd
    from minflux_viewer.ui.histogram_window import _TRACE_AGG_MODES

    assert fd._AGG_MODES == ["per loc"] + _TRACE_AGG_MODES
    for mode in ("trace median", "trace 1st", "trace last"):
        assert mode in fd._AGG_MODES


def test_filter_can_bound_every_trace_read_out_end_to_end():
    """A spec on any trace read-out keeps exactly the traces whose value is in
    range, and keeps/drops whole traces."""
    _qapp()
    _win, ds, _state = _histogram()
    try:
        for mode in TRACE_AGG_FUNCS:
            per_trace = [TRACE_AGG_FUNCS[mode](V[t, -1]) for t in TRACES]
            lo, hi = per_trace[0] - 0.5, per_trace[0] + 0.5
            ds.state["filter_specs"] = [{
                "attribute": "efo", "mode": mode, "lo": float(lo), "hi": float(hi),
                "lo_inc": True, "hi_inc": True, "itr": "last",
            }]
            assert loader.apply_saved_filters(ds) is True, mode
            kept = set(np.flatnonzero(np.asarray(ds.filter_mask, dtype=bool)))
            assert kept == set(TRACES[0]), f"{mode}: kept {kept}"
    finally:
        _win.close()


def test_filter_spinner_values_cover_every_trace_read_out():
    _qapp()
    from minflux_viewer.ui.filter_dialog import _filter_spinner_values

    _win, ds, _state = _histogram()
    try:
        for mode in TRACE_AGG_FUNCS:
            _raw, rng = _filter_spinner_values(ds, "efo", mode, itr="last")
            expected = np.sort([TRACE_AGG_FUNCS[mode](V[t, -1]) for t in TRACES])
            assert np.allclose(np.sort(np.asarray(rng, dtype=float)), expected), mode
    finally:
        _win.close()


# ---------------------------------------------------------------------------
# (1) trace values ignore the filter; save-a-copy is the documented workaround
# ---------------------------------------------------------------------------

def test_trace_read_outs_ignore_the_active_filter():
    _qapp()
    win, ds, _state = _histogram()
    try:
        raw = np.asarray(loader.attr_values_1d(ds, "efo"), dtype=float)
        ftr = np.ones(raw.size, dtype=bool)
        ftr[0] = False                       # drop trace 0's first localization
        got = win._aggregate(raw, ftr, "trace mean", ds)
        assert got[0] == pytest.approx(V[0:3, -1].mean())      # all three locs
        assert got[0] != pytest.approx(V[1:3, -1].mean())      # not the survivors
        # ...while `per loc` does honour the mask.
        assert win._aggregate(raw, ftr, "per loc", ds).size == raw.size - 1
    finally:
        win.close()


@pytest.mark.parametrize("fmt", ["csv", "mat"])
def test_saving_a_filtered_copy_then_reloading_gives_filtered_trace_values(fmt, tmp_path):
    """The documented way to get filter-aware trace values: save the filtered
    view as a snapshot (rows dropped) and reload it as ordinary data."""
    _qapp()
    from minflux_viewer.core import save as save_mod

    win, ds, _state = _histogram()
    try:
        # Filter the way the Filter dialog does — a persisted, re-evaluable spec.
        # build_snapshot_table reads the specs, not a hand-set ds.filter_mask.
        ds.state["filter_specs"] = [{
            "attribute": "efo", "mode": "per loc", "lo": 150.0, "hi": 1e9,
            "lo_inc": True, "hi_inc": True, "itr": "last",
        }]
        assert loader.apply_saved_filters(ds) is True
        assert int(ds.filter_mask.sum()) == N_LOC - 1

        out = tmp_path / f"filtered.{fmt}"
        save_mod.save_processed(
            ds, data_path=str(out), fmt=fmt, content="snapshot", filter_mode="apply",
        )
        re_ds = loader.load_csv(str(out)) if fmt == "csv" else loader.load_dataset(str(out))
        assert re_ds.prop.num_loc == N_LOC - 1
        assert re_ds.prop.num_traces == len(TRACES)

        win2, _ds2, _s2 = _histogram(re_ds)
        try:
            rv = np.asarray(loader.attr_values_1d(re_ds, "efo"), dtype=float)
            got = win2._aggregate(rv, np.ones(rv.size, dtype=bool), "trace mean", re_ds)
            # Trace 0 now averages only its surviving localizations.
            assert np.sort(got)[0] == pytest.approx(V[1:3, -1].mean())
        finally:
            win2.close()
    finally:
        win.close()
