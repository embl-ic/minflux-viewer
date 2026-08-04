"""
Iteration value-pooling (``all [sum]`` / ``all [average]``) in the filter dialog
and histogram.

Pooling collapses each localization's iterations to ONE value, aligned to the
``last`` selection rows::

    sum      ->  Σ a_i
    average  ->  mean(a_i)

so a pooled selection behaves like an ordinary per-localization attribute and
can be filtered on.

There is deliberately **no photon (``eco``) weighting** here. It was evaluated
against a real 2-colour ratiometric ``.msr`` and removed: only ``dcr`` had any
statistical justification (variance ∝ 1/N), positions did not follow the CRLB
scaling at all, and weighting a count by itself (``eco``) inflated it ~15 %.
The remaining eco-weighted DCR pooling lives in
``analysis/attribute_channels.py`` (channel separation), not here.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from minflux_viewer.core import loader
from minflux_viewer.core.iteration import (
    AVERAGE_LABEL,
    SUM_LABEL,
    filter_iteration_labels,
    iteration_selector_label,
    last_label,
    parse_iteration_label,
)


N_LOC = 8
N_ITR = 4


def _make_dataset(n_loc=N_LOC, n_itr=N_ITR, *, with_eco=True, dims=3):
    """Flat m2410 mfx array with deterministic per-iteration dcr / efo / eco.

    ``dcr[loc, itr] = 0.1*(itr+1)``, ``efo[loc, itr] = 1000*itr + loc`` — so
    every localization shares the same pooled dcr and the expected numbers can
    be written down by hand. cfr is measured only at iteration 1.
    """
    n = n_loc * n_itr
    fields = [
        ("vld", "?"), ("itr", "<i4"), ("tid", "<i4"),
        ("loc", "<f8", (3,)), ("cfr", "<f8"), ("efo", "<f8"), ("dcr", "<f8"),
    ]
    if with_eco:
        fields.append(("eco", "<i4"))
    mfx = np.zeros(n, dtype=np.dtype(fields))
    itr = np.tile(np.arange(n_itr), n_loc)
    loc_idx = np.repeat(np.arange(n_loc), n_itr)
    mfx["itr"] = itr
    mfx["tid"] = loc_idx // 2                      # 2 localizations per trace
    mfx["vld"] = True
    mfx["loc"][:, 0] = np.repeat(np.linspace(0.0, 1e-6, n_loc), n_itr)
    mfx["loc"][:, 1] = np.repeat(np.linspace(0.0, 1e-6, n_loc), n_itr)
    if dims >= 3:
        mfx["loc"][:, 2] = np.repeat(np.linspace(0.0, 1e-7, n_loc), n_itr)
    mfx["efo"] = 1000.0 * itr + loc_idx
    mfx["dcr"] = 0.1 * (itr + 1)
    if with_eco:
        mfx["eco"] = 10 * (itr + 1)
    mfx["cfr"][itr == 1] = np.linspace(0.3, 0.8, n_loc)
    return loader.load_from_mfx_array(mfx, "pool")


#: Per-localization iteration values used by the hand-checked expectations.
DCR = np.array([0.1, 0.2, 0.3, 0.4])


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def test_filter_iteration_labels_order():
    """Filter dialog order: last, individual iterations counting down, pooled."""
    assert filter_iteration_labels(4) == [
        "last (4th)", "3rd", "2nd", "1st", SUM_LABEL, AVERAGE_LABEL,
    ]
    # Single-iteration data still gets a meaningful (single) entry.
    assert filter_iteration_labels(1) == ["last (1st)"]


def test_pooled_labels_parse_to_value_pool_selectors():
    assert parse_iteration_label(SUM_LABEL) == ("sum", "single")
    assert parse_iteration_label(AVERAGE_LABEL) == ("average", "single")
    assert loader.is_value_pool_selector("sum")
    assert loader.is_value_pool_selector("average")
    assert not loader.is_value_pool_selector("all")
    assert not loader.is_value_pool_selector(2)


def test_iteration_selector_label_round_trips():
    for label in filter_iteration_labels(4):
        sel, _render = parse_iteration_label(label)
        assert iteration_selector_label(sel, 4) == label
    # The last iteration index and the semantic "last" map to the same label.
    assert iteration_selector_label(3, 4) == last_label(4) == "last (4th)"
    assert iteration_selector_label("last", 4) == "last (4th)"


# ---------------------------------------------------------------------------
# Pooling maths
# ---------------------------------------------------------------------------

def test_pooling_is_plain_sum_and_mean():
    ds = _make_dataset()
    assert np.allclose(loader.mfx_get(ds, "dcr", itr="sum"), DCR.sum())
    assert np.allclose(loader.mfx_get(ds, "dcr", itr="average"), DCR.mean())


def test_pooled_values_align_with_the_last_selection_rows():
    """Pooling yields one value per localization on the same rows as `last`,
    which is what lets pooled views reuse the per-localization machinery."""
    ds = _make_dataset()
    n = np.asarray(loader.mfx_get(ds, "dcr", itr="last")).shape[0]
    assert n == ds.prop.num_loc
    for sel in ("sum", "average"):
        assert np.asarray(loader.mfx_get(ds, "dcr", itr=sel)).shape[0] == n


def test_pooling_skips_non_finite_iterations():
    """A NaN iteration must not poison the whole localization's pooled value."""
    ds = _make_dataset()
    raw = ds.mfx_raw
    vals = np.asarray(raw["dcr"], dtype=float).copy()
    vals[np.asarray(raw["itr"]).ravel() == 0] = np.nan
    raw["dcr"] = vals
    # Iteration 0 dropped -> mean over the remaining three.
    assert np.allclose(loader.mfx_get(ds, "dcr", itr="average"), DCR[1:].mean())


# ---------------------------------------------------------------------------
# Filter specs: iteration round-trip and backward compatibility
# ---------------------------------------------------------------------------

def _spec(attr, lo, hi, **extra):
    spec = {"attribute": attr, "mode": "per loc", "lo": float(lo), "hi": float(hi),
            "lo_inc": True, "hi_inc": True}
    spec.update(extra)
    return spec


def test_legacy_spec_is_last_or_effective():
    """A filter saved before the Iter column existed: raw values at `last`, or at
    the effective iteration for cfr/efc."""
    ds = _make_dataset()
    assert loader.resolve_spec_iteration(ds, _spec("efo", 0, 1)) == "last"
    assert loader.resolve_spec_iteration(ds, _spec("cfr", 0, 1)) == "effective"


def test_legacy_spec_matches_explicit_last_spec():
    ds = _make_dataset()
    ds.state["filter_specs"] = [_spec("efo", 3000, 3003)]
    assert loader.apply_saved_filters(ds) is True
    legacy = ds.filter_mask.copy()

    ds.state["filter_specs"] = [_spec("efo", 3000, 3003, itr="last")]
    assert loader.apply_saved_filters(ds) is True
    assert np.array_equal(legacy, ds.filter_mask)
    assert int(legacy.sum()) == 4          # last-iteration efo 3000..3007


def test_a_stale_weighted_key_in_a_saved_spec_is_ignored():
    """Filters saved while the removed 'photon weighted' option existed must
    still load — the obsolete key is simply ignored, not honoured."""
    ds = _make_dataset()
    ds.state["filter_specs"] = [_spec("efo", 3000, 3003, itr="last", weighted=True)]
    assert loader.apply_saved_filters(ds) is True
    assert int(ds.filter_mask.sum()) == 4          # same as the un-weighted bound


def test_pooled_spec_filters_on_the_pooled_value():
    ds = _make_dataset()
    # Every localization's summed dcr is DCR.sum() == 1.0, so a band around it
    # keeps all of them and a band away from it keeps none.
    ds.state["filter_specs"] = [_spec("dcr", 0.9, 1.1, itr="sum")]
    assert loader.apply_saved_filters(ds) is True
    assert int(ds.filter_mask.sum()) == ds.prop.num_loc

    ds.state["filter_specs"] = [_spec("dcr", 0.0, 0.5, itr="sum")]
    loader.apply_saved_filters(ds)
    assert int(ds.filter_mask.sum()) == 0


def test_pooled_spec_broadcasts_across_browsed_iterations():
    """A pooled spec is a per-localization decision, so browsing all iterations
    keeps or drops a localization's whole stack."""
    ds = _make_dataset()
    ds.state["filter_specs"] = [_spec("efo", 0, 6003, itr="sum")]
    mask, uneval = loader.mfx_filter_mask(ds, itr="all", vld_only=True)
    assert uneval == []
    # efo sums are 6000 + 4*loc -> locs 0 (6000) and 1 (6004>6003) ... keep loc 0 only.
    assert int(mask.sum()) == N_ITR


def test_metadata_sidecar_round_trips_the_iteration(tmp_path):
    """The Save Processed Data recipe carries the spec dicts verbatim, so the
    Iter key must survive a save/reload of the processing recipe."""
    import json

    from minflux_viewer.core import save as save_mod

    ds = _make_dataset()
    ds.state["filter_specs"] = [
        _spec("dcr", 0.29, 0.31, itr="average"),
        _spec("efo", 0.0, 9e9),                       # legacy-style, no keys
    ]
    meta = save_mod.build_metadata(ds, content="raw")
    path = tmp_path / "d_metadata.json"
    path.write_text(json.dumps(meta), encoding="utf-8")

    reloaded = _make_dataset()
    assert loader.apply_metadata_sidecar(reloaded, tmp_path / "d.mat") is True
    specs = reloaded.state["filter_specs"]
    assert specs[0]["itr"] == "average"
    assert "itr" not in specs[1]
    # ...and the legacy-shaped one still resolves to `last`.
    assert loader.resolve_spec_iteration(reloaded, specs[1]) == "last"


def test_attr_values_for_selection_matches_mfx_get_on_attr_rows():
    ds = _make_dataset()
    for sel in ("last", "sum", "average", 1):
        vals = loader.attr_values_for_selection(ds, "dcr", itr=sel)
        assert vals is not None
        assert np.asarray(vals).shape[0] == ds.prop.num_loc
    # "auto" reproduces the historical materialized value exactly.
    assert np.allclose(
        np.asarray(loader.attr_values_for_selection(ds, "efo", itr="auto")),
        np.asarray(loader.attr_values_1d(ds, "efo")),
    )
    # cfr: "auto" is its effective (measured) value, "last" the raw zero.
    assert np.all(np.asarray(loader.attr_values_for_selection(ds, "cfr", itr="auto")) > 0)
    assert np.allclose(loader.attr_values_for_selection(ds, "cfr", itr="last"), 0.0)


def test_photon_weighting_api_is_gone():
    """Regression: the eco-weighting layer was removed from the value path."""
    for name in (
        "PHOTON_WEIGHT_ATTR", "has_photon_weights", "spec_is_weighted",
        "_apply_photon_weight", "_weight_attr_rows",
    ):
        assert not hasattr(loader, name), f"loader.{name} should have been removed"
    ds = _make_dataset()
    with pytest.raises(TypeError):
        loader.mfx_get(ds, "dcr", itr="average", weighted=True)


# ---------------------------------------------------------------------------
# Filter dialog UI
# ---------------------------------------------------------------------------

#: Module-level QApplication reference. PyQt6 destroys the C++ QApplication as
#: soon as the last Python reference goes, which crashes every later Qt call —
#: so the instance must be kept alive for the whole module, not per test.
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


def _dialog(ds=None):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.filter_dialog import FilterDialog

    ds = ds if ds is not None else _make_dataset()
    state = AppState()
    state.add_dataset(ds)
    return FilterDialog(state, dataset_idx=0), ds, state


def test_dialog_has_an_iter_column_and_no_weighted_column():
    _qapp()
    from minflux_viewer.ui import filter_dialog as fd

    dlg, _ds, _state = _dialog()
    try:
        headers = [dlg._table.horizontalHeaderItem(c).text() for c in range(fd._NCOLS)]
        assert headers == ["On", "Attribute", "Mode", "Iter", "Min", "Max", "", ""]
        # Iter sits between Mode and Min; there is no Weighted column at all.
        assert fd._COL_MODE < fd._COL_ITER < fd._COL_MIN
        assert not hasattr(fd, "_COL_WEIGHT")
        assert not hasattr(dlg, "_row_weighted")
    finally:
        dlg.close()


def test_dialog_row_defaults_last_and_effective_for_cfr():
    _qapp()
    from minflux_viewer.ui import filter_dialog as fd

    dlg, _ds, _state = _dialog()
    try:
        dlg._add_row(attr="efo", auto_range=False)
        dlg._add_row(attr="cfr", auto_range=False)
        assert dlg._table.cellWidget(0, fd._COL_ITER).currentText() == "last (4th)"
        assert dlg._table.cellWidget(1, fd._COL_ITER).currentText() == "2nd"  # cfr eff
        assert dlg._row_iteration(0) == "last"
        assert dlg._row_iteration(1) == "effective"
    finally:
        dlg.close()


def test_dialog_persists_the_iteration_into_specs():
    _qapp()
    from minflux_viewer.ui import filter_dialog as fd

    dlg, ds, _state = _dialog()
    try:
        dlg._add_row(attr="dcr", lo=0.0, hi=10.0, enabled=True, auto_range=False)
        dlg._table.cellWidget(0, fd._COL_ITER).setCurrentText(AVERAGE_LABEL)
        dlg._apply_all()

        spec = ds.state["filter_specs"][0]
        assert spec["itr"] == "average"
        assert "weighted" not in spec
    finally:
        dlg.close()


def test_dialog_changing_iter_reranges_the_bounds():
    """Iter changes the value space, so the Min/Max auto-range must follow —
    keeping the old numbers would silently mis-filter."""
    _qapp()
    from minflux_viewer.ui import filter_dialog as fd

    dlg, _ds, _state = _dialog()
    try:
        dlg._add_row(attr="efo", enabled=True, auto_range=True)
        # efo = 1000*itr + loc, so `last` (itr 3) spans 3000..3007.
        assert float(dlg._table.cellWidget(0, fd._COL_MAX).value()) == pytest.approx(3007.0)
        dlg._table.cellWidget(0, fd._COL_ITER).setCurrentText("2nd")
        # ...and iteration 1 spans 1000..1007.
        assert float(dlg._table.cellWidget(0, fd._COL_MAX).value()) == pytest.approx(1007.0)
    finally:
        dlg.close()


def test_dialog_json_round_trips_the_iteration(tmp_path):
    _qapp()
    from minflux_viewer.core import filter_io
    from minflux_viewer.ui import filter_dialog as fd

    dlg, _ds, _state = _dialog()
    try:
        rows = [{"apply": True, "attribute": "dcr", "value_as": "per loc",
                 "min": 0.0, "max": 1.0, "min_inclusive": True, "max_inclusive": True,
                 "iteration": "average"}]
        assert filter_io.is_filter_json_payload(rows) is True
        path = tmp_path / "flt.json"
        filter_io.save_filter_json(str(path), rows)

        dlg.load_filter_json(str(path))
        assert dlg._table.cellWidget(0, fd._COL_ITER).currentText() == AVERAGE_LABEL
        assert dlg._row_iteration(0) == "average"
    finally:
        dlg.close()


def test_dialog_loads_a_json_carrying_the_obsolete_weighted_key(tmp_path):
    """Presets written while the option existed still load; the key is ignored."""
    _qapp()
    from minflux_viewer.core import filter_io
    from minflux_viewer.ui import filter_dialog as fd

    dlg, _ds, _state = _dialog()
    try:
        rows = [{"apply": True, "attribute": "dcr", "value_as": "per loc",
                 "min": 0.0, "max": 1.0, "iteration": "average", "weighted": True}]
        assert filter_io.is_filter_json_payload(rows) is True
        path = tmp_path / "old.json"
        filter_io.save_filter_json(str(path), rows)

        dlg.load_filter_json(str(path))
        assert dlg._table.cellWidget(0, fd._COL_ITER).currentText() == AVERAGE_LABEL
    finally:
        dlg.close()


def test_dialog_loads_a_legacy_json_without_the_iteration_key(tmp_path):
    """Backward compatibility: no 'iteration' -> last (effective for cfr/efc)."""
    _qapp()
    from minflux_viewer.core import filter_io
    from minflux_viewer.ui import filter_dialog as fd

    dlg, ds, _state = _dialog()
    try:
        rows = [
            {"apply": True, "attribute": "efo", "value_as": "per loc",
             "min": 3000.0, "max": 3003.0},
            {"apply": True, "attribute": "cfr", "value_as": "per loc",
             "min": 0.3, "max": 0.9},
        ]
        path = tmp_path / "legacy.json"
        filter_io.save_filter_json(str(path), rows)

        dlg.load_filter_json(str(path))
        assert dlg._table.cellWidget(0, fd._COL_ITER).currentText() == "last (4th)"
        assert dlg._table.cellWidget(1, fd._COL_ITER).currentText() == "2nd"

        dlg._apply_all()
        specs = {s["attribute"]: s for s in ds.state["filter_specs"]}
        assert specs["efo"]["itr"] == "last"
        assert specs["cfr"]["itr"] == "effective"
        # The efo bound keeps the 4 localizations whose last efo is 3000..3003.
        assert int(ds.filter_mask.sum()) == 4
    finally:
        dlg.close()


# ---------------------------------------------------------------------------
# Histogram window
# ---------------------------------------------------------------------------

def _histogram(ds=None):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.histogram_window import HistogramWindow

    ds = ds if ds is not None else _make_dataset()
    state = AppState()
    state.add_dataset(ds)
    return HistogramWindow(state, dataset_idx=0), ds, state


def test_histogram_has_no_photon_weight_checkbox():
    """Regression: the 'photon weighted' checkbox and its helpers are gone."""
    _qapp()
    win, _ds, _state = _histogram()
    try:
        assert not hasattr(win, "_weight_chk")
        assert not hasattr(win, "_weighted")
        assert not hasattr(win, "_sync_weight_availability")
        bar = win._zero_chk.parentWidget().layout().itemAt(0).layout()
        texts = [
            bar.itemAt(i).widget().text()
            for i in range(bar.count())
            if bar.itemAt(i).widget() is not None
            and hasattr(bar.itemAt(i).widget(), "text")
        ]
        assert not any("photon" in t.lower() for t in texts)
    finally:
        win.close()


def test_histogram_iter_dropdown_offers_the_pooled_modes():
    _qapp()
    win, _ds, _state = _histogram()
    try:
        labels = [win._iter_combo.itemText(i) for i in range(win._iter_combo.count())]
        assert labels == [
            "last (4th)", "all [flatten]", "all [stacked]", SUM_LABEL, AVERAGE_LABEL,
            "3rd", "2nd", "1st",
        ]
    finally:
        win.close()


def test_histogram_pooled_view_stays_on_the_materialized_path():
    """`all [sum]`/`all [average]` yield one value per localization, so they must
    NOT fall back to the raw path — that is what keeps filter editing alive."""
    _qapp()
    win, ds, _state = _histogram()
    try:
        win._iter_combo.setCurrentText(SUM_LABEL)
        assert win._is_raw_mode() is False
        assert np.allclose(win._materialized_values(ds, "dcr"), DCR.sum())

        win._iter_combo.setCurrentText(AVERAGE_LABEL)
        assert np.allclose(win._materialized_values(ds, "dcr"), DCR.mean())
    finally:
        win.close()


def test_histogram_value_label_is_the_plain_attribute():
    _qapp()
    win, _ds, _state = _histogram()
    try:
        assert win._value_label("efo") == "efo"
        win._log_chk.setChecked(True)
        assert win._value_label("efo") == "log(efo)"
    finally:
        win.close()


def test_start_filter_edit_applies_the_iteration():
    """The filter row's Iter travels to the histogram when the eye button opens
    it, and the region stays editable (materialized path)."""
    _qapp()
    win, ds, _state = _histogram()
    try:
        win.start_filter_edit(attr="dcr", mode="per loc", lo=0.0, hi=1.0, itr="average")
        assert win._iter_combo.currentText() == AVERAGE_LABEL
        assert win._is_raw_mode() is False
        assert win._filter_edit is not None
        assert np.allclose(win._materialized_values(ds, "dcr"), DCR.mean())
    finally:
        win.close()


def test_start_filter_edit_on_a_specific_iteration_keeps_the_region():
    """A row filtering on one specific iteration is edited on that iteration's
    values gathered onto the materialized rows, so the drag region survives."""
    _qapp()
    win, ds, _state = _histogram()
    try:
        win.start_filter_edit(attr="efo", mode="per loc", lo=0.0, hi=1.0, itr=1)
        assert win._iter_combo.currentText() == "2nd"
        assert win._is_raw_mode() is False
        assert win._filter_edit is not None
        vals = np.asarray(win._materialized_values(ds, "efo"), dtype=float)
        assert np.allclose(vals, 1000.0 + np.arange(N_LOC))   # iteration 1's band
    finally:
        win.close()


def test_start_filter_edit_without_selection_keeps_previous_behaviour():
    """No itr given -> the historical default view (cfr on its effective
    iteration), unchanged."""
    _qapp()
    win, _ds, _state = _histogram()
    try:
        win.start_filter_edit(attr="cfr", mode="per loc", lo=0.3, hi=0.9)
        assert win._iter_combo.currentText() == "2nd"          # cfr effective
        assert win._is_raw_mode() is False
        assert win._edit_itr is None
    finally:
        win.close()
