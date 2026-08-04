"""
Preferences > Appearance > Histogram Plot: the two option rows.

* ``show trace value(s)``     -> ``prefs["plot"]["histogram_values"]``, gating the
  histogram's **As** dropdown. Adds the positional read-outs ``trace 1st`` /
  ``trace last`` (a trace's first / last localization in time order) alongside
  the statistical ones.
* ``show all iteration value(s)`` -> ``prefs["plot"]["histogram_iterations"]``,
  gating the **pooled** entries of the histogram's **Iter** dropdown. ``last``
  and the individual iterations are never gated.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from minflux_viewer.core import loader
from minflux_viewer.core.app_state import DEFAULT_PREFS
from minflux_viewer.core.iteration import (
    AVERAGE_LABEL,
    FLATTEN_LABEL,
    POOL_KEYS,
    POOL_LABEL_BY_KEY,
    STACKED_LABEL,
    SUM_LABEL,
    iteration_labels,
)


N_LOC = 12
N_ITR = 4

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


def _make_dataset(n_loc=N_LOC, n_itr=N_ITR):
    """4 localizations per trace, efo ramping within each trace so the trace
    read-outs (1st / last / min / max) are all distinguishable."""
    n = n_loc * n_itr
    dt = np.dtype([("vld", "?"), ("itr", "<i4"), ("tid", "<i4"),
                   ("loc", "<f8", (3,)), ("efo", "<f8"), ("eco", "<i4")])
    mfx = np.zeros(n, dtype=dt)
    itr = np.tile(np.arange(n_itr), n_loc)
    li = np.repeat(np.arange(n_loc), n_itr)
    mfx["itr"] = itr
    mfx["tid"] = li // 4                       # 4 localizations per trace
    mfx["vld"] = True
    mfx["loc"][:, 0] = np.repeat(np.linspace(0.0, 1e-6, n_loc), n_itr)
    # Per-localization efo = 100, 200, 300, ... so a trace of 4 spans a decade.
    mfx["efo"] = np.repeat(100.0 * (np.arange(n_loc) + 1), n_itr)
    mfx["eco"] = 50
    return loader.load_from_mfx_array(mfx, "prefs")


def _histogram(prefs_patch=None):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.histogram_window import HistogramWindow

    ds = _make_dataset()
    state = AppState()
    if prefs_patch:
        state.prefs.setdefault("plot", {}).update(prefs_patch)
    state.add_dataset(ds)
    win = HistogramWindow(state, dataset_idx=0)
    win._attr_combo.setCurrentText("efo")
    return win, ds, state


def _combo_items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def test_default_prefs_carry_both_option_lists():
    plot = DEFAULT_PREFS["plot"]
    assert plot["histogram_values"] == ["trace mean", "trace median"]
    # All four pooled modes on by default.
    assert plot["histogram_iterations"] == list(POOL_KEYS)
    assert list(POOL_KEYS) == ["flatten", "stacked", "sum", "average"]


def test_app_state_prefs_do_not_alias_the_module_defaults():
    """Regression: ``_load_prefs`` used a shallow ``dict(DEFAULT_PREFS)``, so
    ``prefs["plot"]`` *was* ``DEFAULT_PREFS["plot"]`` and editing preferences
    corrupted the defaults for the rest of the process (it leaked between
    tests, and between a fresh-install session's Preferences edits)."""
    from minflux_viewer.core.app_state import AppState

    state = AppState()
    assert state.prefs["plot"] is not DEFAULT_PREFS["plot"]
    assert state.prefs["plot"]["histogram_iterations"] is not \
        DEFAULT_PREFS["plot"]["histogram_iterations"]

    before = list(DEFAULT_PREFS["plot"]["histogram_iterations"])
    state.prefs["plot"]["histogram_iterations"] = ["sum"]
    state.prefs["plot"]["histogram_values"].append("trace range")
    assert DEFAULT_PREFS["plot"]["histogram_iterations"] == before
    assert "trace range" not in DEFAULT_PREFS["plot"]["histogram_values"]
    # A second AppState still sees the pristine defaults.
    assert AppState().prefs["plot"]["histogram_iterations"] == before


def test_pool_label_map_matches_the_dropdown_labels():
    assert POOL_LABEL_BY_KEY == {
        "flatten": FLATTEN_LABEL,
        "stacked": STACKED_LABEL,
        "sum": SUM_LABEL,
        "average": AVERAGE_LABEL,
    }


# ---------------------------------------------------------------------------
# iteration_labels(allowed=...)
# ---------------------------------------------------------------------------

def test_iteration_labels_default_is_every_pooled_mode():
    assert iteration_labels(4) == [
        "last (4th)", FLATTEN_LABEL, STACKED_LABEL, SUM_LABEL, AVERAGE_LABEL,
        "3rd", "2nd", "1st",
    ]
    # An explicit None is the same as omitting it (a preference-less caller).
    assert iteration_labels(4, allowed=None) == iteration_labels(4)


def test_iteration_labels_gates_only_the_pooled_modes_and_keeps_the_order():
    labels = iteration_labels(4, allowed=["average", "flatten"])
    # Fixed order is preserved regardless of the order given.
    assert labels == ["last (4th)", FLATTEN_LABEL, AVERAGE_LABEL, "3rd", "2nd", "1st"]

    # last + the individual iterations are never gated.
    assert iteration_labels(4, allowed=[]) == ["last (4th)", "3rd", "2nd", "1st"]

    # Unknown keys are ignored rather than crashing.
    assert iteration_labels(4, allowed=["nonsense"]) == ["last (4th)", "3rd", "2nd", "1st"]


def test_iteration_labels_still_empty_for_single_iteration_data():
    for allowed in (None, [], list(POOL_KEYS)):
        assert iteration_labels(1, allowed=allowed) == []


# ---------------------------------------------------------------------------
# The trace read-outs
# ---------------------------------------------------------------------------

def test_trace_first_and_last_read_the_positional_values():
    """'1st'/'last' are the trace's first and last localization in stored (time)
    order - not an order statistic like min/max."""
    _qapp()
    win, ds, _state = _histogram({"histogram_values": ["trace 1st", "trace last",
                                                       "trace min", "trace max"]})
    try:
        raw = np.asarray(loader.attr_values_1d(ds, "efo"), dtype=float)
        ftr = np.ones(raw.size, dtype=bool)

        first = win._aggregate(raw, ftr, "trace 1st", ds)
        last = win._aggregate(raw, ftr, "trace last", ds)
        # Traces are 4 localizations of 100,200,300,400 / 500,600,700,800 / ...
        assert first == pytest.approx([100.0, 500.0, 900.0])
        assert last == pytest.approx([400.0, 800.0, 1200.0])

        # On this ascending data they coincide with min/max; the point is that
        # they are computed positionally, so a descending trace would flip them.
        assert first == pytest.approx(win._aggregate(raw, ftr, "trace min", ds))
        assert last == pytest.approx(win._aggregate(raw, ftr, "trace max", ds))
    finally:
        win.close()


def test_trace_first_and_last_follow_time_order_not_value_order():
    _qapp()
    win, ds, _state = _histogram()
    try:
        raw = np.asarray(loader.attr_values_1d(ds, "efo"), dtype=float)
        raw = raw[::-1].copy()                    # descending within each trace
        ftr = np.ones(raw.size, dtype=bool)
        first = win._aggregate(raw, ftr, "trace 1st", ds)
        last = win._aggregate(raw, ftr, "trace last", ds)
        assert np.all(first > last)               # reversed => 1st is the larger
        # ...and they are now the max/min respectively.
        assert first == pytest.approx(win._aggregate(raw, ftr, "trace max", ds))
        assert last == pytest.approx(win._aggregate(raw, ftr, "trace min", ds))
    finally:
        win.close()


def test_as_dropdown_lists_enabled_trace_values_in_preferences_order():
    _qapp()
    win, _ds, _state = _histogram({
        "histogram_values": ["trace range", "trace 1st", "trace mean"],
    })
    try:
        # "per loc" always first, then _TRACE_AGG_MODES order (not pref order).
        assert _combo_items(win._agg_combo) == [
            "per loc", "trace mean", "trace 1st", "trace range",
        ]
    finally:
        win.close()


def test_disabled_trace_values_are_absent_from_the_dropdown():
    _qapp()
    win, _ds, _state = _histogram({"histogram_values": ["trace mean"]})
    try:
        items = _combo_items(win._agg_combo)
        assert items == ["per loc", "trace mean"]
        for absent in ("trace 1st", "trace last", "trace median", "trace stdev"):
            assert absent not in items
    finally:
        win.close()


def test_trace_read_outs_work_on_the_raw_iteration_path_too():
    """Regression: ``trace 1st``/``trace last`` were added to the materialized
    aggregation but not to ``utils.filters.raw_trace_aggregate``, whose dispatch
    silently returned the values unchanged. Selecting them with a raw-path
    iteration (``all [flatten]``, ``all [stacked]``, an individual iteration)
    therefore plotted one value per ROW instead of one per trace."""
    _qapp()
    from minflux_viewer.utils.filters import raw_trace_aggregate

    vals = np.array([10.0, 11.0, 12.0, 20.0, 21.0, 22.0])
    tid = np.array([0, 0, 0, 1, 1, 1])
    assert raw_trace_aggregate(vals, tid, "trace 1st") == pytest.approx([10.0, 20.0])
    assert raw_trace_aggregate(vals, tid, "trace last") == pytest.approx([12.0, 22.0])
    # One value per trace, not per row.
    for mode in ("trace 1st", "trace last"):
        assert raw_trace_aggregate(vals, tid, mode).size == 2

    # Interleaved traces still group correctly (the store is not tid-contiguous).
    inter = np.array([10.0, 20.0, 11.0, 21.0])
    itid = np.array([0, 1, 0, 1])
    assert raw_trace_aggregate(inter, itid, "trace 1st") == pytest.approx([10.0, 20.0])
    assert raw_trace_aggregate(inter, itid, "trace last") == pytest.approx([11.0, 21.0])


def test_raw_trace_aggregate_rejects_an_unknown_mode():
    """It used to return the input unchanged, so a mode added to one dispatch
    table but not this one degraded to per-row values with no warning."""
    from minflux_viewer.utils.filters import raw_trace_aggregate

    vals = np.array([1.0, 2.0, 3.0, 4.0])
    tid = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError):
        raw_trace_aggregate(vals, tid, "trace nonsense")
    # "per loc" is still the documented pass-through.
    assert raw_trace_aggregate(vals, tid, "per loc") == pytest.approx(vals)


def test_every_histogram_trace_mode_is_supported_on_both_paths():
    """The materialized (_aggregate) and raw (raw_trace_aggregate) paths must
    cover the same set of read-outs, or the plot changes meaning with the
    iteration selection."""
    _qapp()
    from minflux_viewer.ui.histogram_window import _TRACE_AGG_MODES
    from minflux_viewer.utils.filters import raw_trace_aggregate

    win, ds, _state = _histogram({"histogram_values": list(_TRACE_AGG_MODES)})
    try:
        raw = np.asarray(loader.attr_values_1d(ds, "efo"), dtype=float)
        ftr = np.ones(raw.size, dtype=bool)
        tid = np.asarray(loader.attr_values_1d(ds, "tid")).ravel()
        n_traces = int(ds.prop.num_traces)
        for mode in _TRACE_AGG_MODES:
            mat = np.asarray(win._aggregate(raw, ftr, mode, ds), dtype=float)
            rawp = np.asarray(raw_trace_aggregate(raw, tid, mode), dtype=float)
            assert mat.size == n_traces, f"{mode}: materialized path"
            assert rawp.size == n_traces, f"{mode}: raw path"
            # Same rows in, same trace values out.
            assert np.allclose(np.sort(mat), np.sort(rawp)), mode
    finally:
        win.close()


def test_new_trace_modes_have_tooltips():
    from minflux_viewer.core.attributes import aggregation_description

    for mode in ("trace 1st", "trace last"):
        assert aggregation_description(mode)


# ---------------------------------------------------------------------------
# The Iter dropdown
# ---------------------------------------------------------------------------

def test_iter_dropdown_shows_all_pooled_modes_by_default():
    _qapp()
    win, _ds, _state = _histogram()
    try:
        assert _combo_items(win._iter_combo) == [
            "last (4th)", FLATTEN_LABEL, STACKED_LABEL, SUM_LABEL, AVERAGE_LABEL,
            "3rd", "2nd", "1st",
        ]
    finally:
        win.close()


def test_iter_dropdown_honours_the_preference():
    _qapp()
    win, _ds, _state = _histogram({"histogram_iterations": ["sum", "average"]})
    try:
        assert _combo_items(win._iter_combo) == [
            "last (4th)", SUM_LABEL, AVERAGE_LABEL, "3rd", "2nd", "1st",
        ]
    finally:
        win.close()


def test_turning_every_pooled_mode_off_keeps_the_plain_iterations():
    _qapp()
    win, _ds, _state = _histogram({"histogram_iterations": []})
    try:
        assert _combo_items(win._iter_combo) == ["last (4th)", "3rd", "2nd", "1st"]
        # The selector is still useful, so it stays visible.
        assert win._iter_combo.isVisible() or not win.isVisible()
    finally:
        win.close()


# ---------------------------------------------------------------------------
# Live refresh after Preferences OK
# ---------------------------------------------------------------------------

def test_refresh_preferences_repopulates_both_dropdowns():
    _qapp()
    win, _ds, state = _histogram()
    try:
        win._iter_combo.setCurrentText(STACKED_LABEL)
        state.prefs["plot"]["histogram_values"] = ["trace mean", "trace last"]
        state.prefs["plot"]["histogram_iterations"] = ["sum"]

        win.refresh_preferences()

        assert _combo_items(win._agg_combo) == ["per loc", "trace mean", "trace last"]
        assert _combo_items(win._iter_combo) == [
            "last (4th)", SUM_LABEL, "3rd", "2nd", "1st",
        ]
        # The stacked selection is gone, so it falls back to a valid label.
        assert win._iter_combo.currentText() in _combo_items(win._iter_combo)
    finally:
        win.close()


def test_refresh_preferences_keeps_a_still_valid_selection():
    _qapp()
    # Pin the enabled read-outs so the test does not ride on the defaults.
    win, _ds, state = _histogram({"histogram_values": ["trace mean", "trace median"]})
    try:
        win._iter_combo.setCurrentText(SUM_LABEL)
        win._agg_combo.setCurrentText("trace median")
        state.prefs["plot"]["histogram_iterations"] = ["sum", "average"]

        win.refresh_preferences()

        assert win._iter_combo.currentText() == SUM_LABEL
        assert win._agg_combo.currentText() == "trace median"
    finally:
        win.close()


def test_main_window_broadcasts_preferences_to_histogram_windows():
    """Regression: _refresh_plot_preferences used to cover only render/scatter,
    so the Histogram Plot options needed a window reopen to take effect."""
    import inspect

    from minflux_viewer.ui.main_window import MainWindow
    from minflux_viewer.ui.histogram_window import HistogramWindow

    src = inspect.getsource(MainWindow._refresh_plot_preferences)
    assert "_histogram_windows" in src
    assert callable(getattr(HistogramWindow, "refresh_preferences", None))


# ---------------------------------------------------------------------------
# Preferences dialog round-trip
# ---------------------------------------------------------------------------

def _prefs_dialog(state):
    from minflux_viewer.ui.preferences_dialog import PreferencesDialog

    return PreferencesDialog(state)


def test_preferences_dialog_round_trips_both_rows():
    _qapp()
    from minflux_viewer.core.app_state import AppState

    state = AppState()
    state.prefs["plot"]["histogram_values"] = ["trace mean", "trace 1st"]
    state.prefs["plot"]["histogram_iterations"] = ["flatten", "average"]
    dlg = _prefs_dialog(state)
    try:
        assert dlg._hist_trace_checks["trace mean"].isChecked()
        assert dlg._hist_trace_checks["trace 1st"].isChecked()
        assert not dlg._hist_trace_checks["trace last"].isChecked()
        assert dlg._hist_iter_checks["flatten"].isChecked()
        assert dlg._hist_iter_checks["average"].isChecked()
        assert not dlg._hist_iter_checks["stacked"].isChecked()

        # Flip a couple and write back.
        dlg._hist_trace_checks["trace last"].setChecked(True)
        dlg._hist_iter_checks["stacked"].setChecked(True)
        dlg._apply_widgets_to_draft()

        # Both lists are written back in dropdown order, not click order.
        assert dlg._draft["plot"]["histogram_values"] == [
            "trace mean", "trace 1st", "trace last",
        ]
        assert dlg._draft["plot"]["histogram_iterations"] == [
            "flatten", "stacked", "average",
        ]
    finally:
        dlg.close()


def test_preferences_dialog_offers_every_documented_option():
    _qapp()
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.histogram_window import _TRACE_AGG_MODES

    dlg = _prefs_dialog(AppState())
    try:
        assert list(dlg._hist_trace_checks) == _TRACE_AGG_MODES
        assert [cb.text() for cb in dlg._hist_trace_checks.values()] == [
            "mean", "median", "min", "max", "1st", "last", "stdev", "range",
        ]
        assert list(dlg._hist_iter_checks) == list(POOL_KEYS)
    finally:
        dlg.close()


def test_unchecking_every_trace_value_keeps_a_usable_dropdown():
    """A trace-wise attribute is forced onto a trace read-out, so the "As" list
    must never be reduced to just "per loc"."""
    _qapp()
    from minflux_viewer.core.app_state import AppState

    state = AppState()
    dlg = _prefs_dialog(state)
    try:
        for cb in dlg._hist_trace_checks.values():
            cb.setChecked(False)
        dlg._apply_widgets_to_draft()
        assert dlg._draft["plot"]["histogram_values"] == ["trace mean"]
        # The pooled row has no such floor - an empty list is a real choice.
        for cb in dlg._hist_iter_checks.values():
            cb.setChecked(False)
        dlg._apply_widgets_to_draft()
        assert dlg._draft["plot"]["histogram_iterations"] == []
    finally:
        dlg.close()
