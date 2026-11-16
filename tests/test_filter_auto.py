"""The Filter dialog's *Auto* button, the Process ▸ Filter preferences behind it,
and the menu places of Filter and of the precision/resolution commands."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")


@pytest.fixture(scope="module")
def _app(qapp):
    return qapp


def _dataset(seed: int = 0):
    """Traces of 1-30 localizations; efo has a one- and a two-emitter population."""
    from minflux_viewer.core.dataset import build_localization_dataset

    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, 31, 1500)
    tid = np.repeat(np.arange(lengths.size), lengths)
    n = tid.size
    double = rng.random(n) < 0.12
    efo = np.where(double, rng.normal(120_000, 10_000, n), rng.normal(60_000, 7_000, n))
    cfr = np.clip(rng.normal(0.3, 0.08, n), 0.0, None)
    outliers = rng.random(n) < 0.05
    cfr[outliers] = rng.uniform(0.8, 1.5, int(outliers.sum()))
    return build_localization_dataset(
        name="auto.mat", x_nm=rng.random(n) * 2000, y_nm=rng.random(n) * 2000,
        tid=tid, tim=np.sort(rng.random(n) * 100), attrs={"efo": efo, "cfr": cfr})


def _dialog(ds, rules=None):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.dialogs.filter_dialog import FilterDialog

    state = AppState()
    state.save_prefs = lambda: None
    if rules is not None:
        state.prefs.setdefault("process", {}).setdefault("filter", {})["auto_rules"] = rules
    state.add_dataset(ds)
    return FilterDialog(state, dataset_idx=0), state


def _rows(dialog):
    from PyQt6.QtWidgets import QCheckBox, QComboBox

    from minflux_viewer.ui.dialogs import filter_dialog as fd

    out = []
    for row in range(dialog._table.rowCount()):
        attr = dialog._table.cellWidget(row, fd._COL_ATTR)
        chk = dialog._table.cellWidget(row, fd._COL_ENABLED)
        out.append((attr.currentText() if isinstance(attr, QComboBox) else None,
                    dialog._table.cellWidget(row, fd._COL_MIN).value(),
                    dialog._table.cellWidget(row, fd._COL_MAX).value(),
                    chk.isChecked() if isinstance(chk, QCheckBox) else None))
    return out


def test_auto_adds_one_ticked_row_per_default_rule(_app):
    ds = _dataset()
    dialog, state = _dialog(ds)
    try:
        dialog._apply_auto_rules()
        rows = {attr: (lo, hi, on) for attr, lo, hi, on in _rows(dialog)}
        assert set(rows) == {"efo", "cfr", "siz"}
        assert all(on for _lo, _hi, on in rows.values())
        assert 85_000 < rows["efo"][1] < 105_000          # single/double crossing
        assert rows["siz"][0] == 10.0                             # at least 10 per trace
        assert rows["cfr"][1] < 1.0                               # robust upper bound
        mask = np.asarray(ds.filter_mask, dtype=bool)
        assert 0 < mask.sum() < ds.prop.num_loc
        assert [spec["attribute"] for spec in ds.state["filter_specs"]] == ["efo", "cfr", "siz"]
        assert rows["siz"][1] == np.max(ds.attr["siz"])
        assert dialog._info.text().startswith("Auto: efo ")
        assert any(line["message"].startswith("Auto filter on 'auto.mat': efo")
                   for line in _log_lines(state))
    finally:
        dialog.close()


def _log_lines(state):
    return list(state.log_history)


def test_the_auto_log_line_becomes_a_filtering_sentence_with_its_citation(_app):
    from minflux_viewer.analysis.method_rules import RULES

    ds = _dataset(seed=7)
    dialog, state = _dialog(ds)
    try:
        dialog._apply_auto_rules()
        line = next(entry["message"] for entry in _log_lines(state)
                    if entry["message"].startswith("Auto filter on"))
        for pattern, stage, render in RULES:
            match = pattern.search(line)
            if match:
                sentence, cites = render(match, {"message": line}, state)
                break
        else:
            pytest.fail("no method rule matches the Auto filter log line")
        assert stage == "filter"
        assert sentence.startswith("Localizations of 'auto.mat' were filtered")
        assert "efo " in sentence and "cfr " in sentence and "siz ≥ 10" in sentence
        assert [cite[0] for cite in cites] == ["Leys et al., J. Exp. Soc. Psychol. 49:764 (2013)"]
    finally:
        dialog.close()


def test_auto_updates_an_existing_row_instead_of_stacking_a_second(_app):
    ds = _dataset(seed=1)
    rules = [{"enabled": True, "attribute": "cfr", "method": "robust_spread",
              "params": {"k": 2.0, "side": "upper"}}]
    dialog, _state = _dialog(ds, rules)
    try:
        dialog._add_row(attr="cfr", mode="per loc", lo=0.0, hi=0.05, enabled=True,
                        auto_range=False)
        dialog._apply_all()
        dialog._apply_auto_rules()
        rows = _rows(dialog)
        assert [r[0] for r in rows] == ["cfr"]
        # The old, much tighter bound neither survived nor gated the new estimate.
        assert rows[0][2] > 0.3
        dialog._apply_auto_rules()                                   # idempotent
        assert [r[0] for r in _rows(dialog)] == ["cfr"]
    finally:
        dialog.close()


def test_a_rule_with_nothing_to_suggest_is_reported_and_adds_no_row(_app):
    ds = _dataset(seed=2)
    ds.attr["efo"] = np.random.default_rng(3).normal(60_000, 7_000, ds.prop.num_loc)
    rules = [{"enabled": True, "attribute": "efo", "method": "single_emitter",
              "params": {"fallback": 0.0}},
             {"enabled": True, "attribute": "nonexistent", "method": "percentile"},
             {"enabled": False, "attribute": "cfr", "method": "robust_spread"}]
    dialog, _state = _dialog(ds, rules)
    try:
        dialog._apply_auto_rules()
        assert _rows(dialog) == []
        text = dialog._info.text()
        assert "no bound was set" in text
        assert "efo: single + double-emitter model" in text and "no bound suggested" in text
        assert "nonexistent: not an attribute" in text
        assert int(np.asarray(ds.filter_mask, dtype=bool).sum()) == ds.prop.num_loc
    finally:
        dialog.close()


def test_rules_are_chained_in_order(_app):
    """The second rule is evaluated on what the first one let through."""
    ds = _dataset(seed=4)
    first = {"enabled": True, "attribute": "efo", "method": "percentile",
             "params": {"lower_pct": 0.0, "upper_pct": 50.0}}
    then = {"enabled": True, "attribute": "cfr", "method": "percentile",
            "params": {"upper_pct": 50.0}}
    dialog, _state = _dialog(ds, [first, then])
    try:
        dialog._apply_auto_rules()
        efo_hi = [r for r in _rows(dialog) if r[0] == "efo"][0][2]
        cfr_hi = [r for r in _rows(dialog) if r[0] == "cfr"][0][2]
        efo = np.asarray(ds.attr["efo"], dtype=float)
        cfr = np.asarray(ds.attr["cfr"], dtype=float)
        assert cfr_hi == pytest.approx(np.percentile(cfr[efo <= efo_hi], 50.0), rel=1e-3)
    finally:
        dialog.close()


def test_auto_cfr_clamps_both_ends_and_keeps_inclusive_endpoints(_app):
    ds = _dataset(seed=8)
    cfr = np.resize(np.array([-0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25]), ds.prop.num_loc)
    ds.attr["cfr"] = cfr
    rule = {"attribute": "cfr", "method": "robust_spread",
            "params": {"side": "both", "k": 10.0}}
    dialog, _state = _dialog(ds, [rule])
    try:
        dialog._apply_auto_rules()
        assert _rows(dialog) == [("cfr", 0.0, 1.0, True)]
        np.testing.assert_array_equal(ds.filter_mask, (cfr >= 0.0) & (cfr <= 1.0))
        assert ds.state["filter_specs"][0]["lo_inc"] is True
        assert ds.state["filter_specs"][0]["hi_inc"] is True
    finally:
        dialog.close()


def test_auto_preserves_exact_data_extrema_in_the_filter_widgets(_app):
    ds = _dataset(seed=9)
    endpoints = (0.12345678901265, 0.87654321098734)
    ds.attr["cfr"] = np.linspace(*endpoints, ds.prop.num_loc)
    rule = {"attribute": "cfr", "method": "robust_spread",
            "params": {"side": "both", "k": 10.0}}
    dialog, _state = _dialog(ds, [rule])
    try:
        dialog._apply_auto_rules()
        assert _rows(dialog) == [("cfr", *endpoints, True)]
        assert np.all(ds.filter_mask)
        dialog._table.setRowCount(0)
        dialog._restore_saved_filter_specs()
        dialog._apply_all()
        assert np.all(ds.filter_mask)
    finally:
        dialog.close()


def test_auto_bounds_use_the_whole_datasets_range_not_the_gated_one(_app):
    """A rule sees only what the rows before it let through, but its row must
    stand on its own: an open side gets the dataset's extreme and a clip uses
    the dataset's range, so the row cuts nothing once the other rows go."""
    from PyQt6.QtWidgets import QCheckBox

    from minflux_viewer.ui.dialogs import filter_dialog as fd

    ds = _dataset(seed=10)
    n = ds.prop.num_loc
    ds.attr["cfr"] = np.linspace(0.0, 0.9, n)
    ds.attr["efo"] = np.linspace(50_000.0, 90_000.0, n)       # rises with cfr
    rule = {"attribute": "cfr", "method": "robust_spread",
            "params": {"side": "upper", "k": 10.0}}
    dialog, _state = _dialog(ds, [rule])
    try:
        dialog._add_row(attr="efo", mode="per loc", lo=60_000.0, hi=80_000.0, enabled=True,
                        auto_range=False)                       # the rule sees cfr 0.225-0.675
        dialog._apply_all()
        dialog._apply_auto_rules()
        rows = {attr: (lo, hi, on) for attr, lo, hi, on in _rows(dialog)}
        assert rows["cfr"] == (0.0, 0.9, True)                  # not (0.225, 0.675)
        efo = np.asarray(ds.attr["efo"], dtype=float)
        np.testing.assert_array_equal(ds.filter_mask, (efo >= 60_000.0) & (efo <= 80_000.0))
        efo_row = next(row for row, entry in enumerate(_rows(dialog)) if entry[0] == "efo")
        chk = dialog._table.cellWidget(efo_row, fd._COL_ENABLED)
        assert isinstance(chk, QCheckBox)
        chk.setChecked(False)                                   # the cfr row on its own
        assert np.all(ds.filter_mask)
    finally:
        dialog.close()


@pytest.mark.parametrize("lengths", [np.array([9, 10, 11, 20] * 10 + [200]),
                                     np.array([1, 2] * 20)])
def test_auto_size_keeps_ten_and_the_largest_trace_with_an_old_saved_rule(_app, lengths):
    from minflux_viewer.core.dataset import build_localization_dataset

    tid = np.repeat(np.arange(lengths.size), lengths)
    ds = build_localization_dataset(name="sizes", x_nm=np.arange(tid.size),
                                    y_nm=np.zeros(tid.size), tid=tid)
    rule = {"attribute": "siz", "method": "percentile",
            "params": {"lower_pct": 0.0, "upper_pct": 99.0, "min_value": 10.0}}
    dialog, _state = _dialog(ds, [rule])
    try:
        dialog._apply_auto_rules()
        assert _rows(dialog) == [("siz", 10.0, float(lengths.max()), True)]
        np.testing.assert_array_equal(ds.filter_mask, np.repeat(lengths >= 10, lengths))
        dialog._table.setRowCount(0)
        dialog._restore_saved_filter_specs()
        dialog._apply_all()
        np.testing.assert_array_equal(ds.filter_mask, np.repeat(lengths >= 10, lengths))
    finally:
        dialog.close()


def test_deleting_a_row_does_not_misdirect_the_buttons_below_it(_app):
    """Row handlers used to capture their row index at creation."""
    from minflux_viewer.ui.dialogs import filter_dialog as fd

    ds = _dataset(seed=5)
    dialog, _state = _dialog(ds, [])
    try:
        for attr in ("efo", "cfr", "tid"):
            dialog._add_row(attr=attr, mode="per loc", enabled=False, auto_range=True)
        dialog._table.cellWidget(0, fd._COL_DELETE).click()          # delete "efo"
        assert [r[0] for r in _rows(dialog)] == ["cfr", "tid"]
        dialog._table.cellWidget(1, fd._COL_DELETE).click()          # delete "tid"
        assert [r[0] for r in _rows(dialog)] == ["cfr"]
    finally:
        dialog.close()


def test_auto_tooltip_lists_the_rules(_app):
    dialog, _state = _dialog(_dataset(seed=6))
    try:
        tip = dialog._auto_btn.toolTip()
        assert "efo: single-emitter frequency cut" in tip
        assert "cfr: robust spread" in tip
        assert "Preferences ▸ Process ▸ Filter" in tip
        assert "pyminflux" not in tip.lower()
    finally:
        dialog.close()


# ---------------------------------------------------------------------------
# Preferences ▸ Process
# ---------------------------------------------------------------------------

def test_process_page_round_trips_the_rules(_app):
    from minflux_viewer.core import auto_filter as af
    from minflux_viewer.ui.settings.prefs_process import build_process_page

    page = build_process_page()
    prefs: dict = {}
    page.load(prefs)                                   # no stored rules -> defaults
    assert [row.rule()["attribute"] for row in page.rows()] == ["efo", "cfr", "siz"]
    page.save(prefs)
    usable, skipped = af.normalize_rules(prefs["process"]["filter"]["auto_rules"])
    assert not skipped
    assert usable == af.normalize_rules(af.default_auto_rules())[0]

    page.add_button.click()
    page.rows()[-1].attribute.setCurrentText("fbg")
    page.move_row(page.rows()[-1], -1)
    page.remove_row(page.rows()[0])
    page.save(prefs)
    assert [r["attribute"] for r in prefs["process"]["filter"]["auto_rules"]] == [
        "cfr", "fbg", "siz"]
    page.restore_button.click()
    assert [row.rule()["attribute"] for row in page.rows()] == ["efo", "cfr", "siz"]
    page.deleteLater()


def test_switching_method_keeps_each_methods_values(_app):
    from minflux_viewer.ui.settings.prefs_process import AutoRuleRow, build_process_page

    page = build_process_page()
    row = AutoRuleRow(page, {"attribute": "cfr", "method": "robust_spread",
                             "params": {"k": 3.0, "side": "both"}})
    row.method.setCurrentIndex(row.method.findData("percentile"))
    row.method.setCurrentIndex(row.method.findData("robust_spread"))
    assert row.rule() == {"enabled": True, "attribute": "cfr", "method": "robust_spread",
                          "params": {"k": 3.0, "side": "both"}}
    row.deleteLater()
    page.deleteLater()


def test_size_preferences_show_the_effective_limits_for_saved_rules(_app):
    from minflux_viewer.ui.settings.prefs_process import build_process_page

    page = build_process_page()
    page.load({"process": {"filter": {"auto_rules": [
        {"attribute": "siz", "method": "percentile",
         "params": {"upper_pct": 99.0, "min_value": 10.0}}
    ]}}})
    row = page.rows()[0]
    assert row.upper_pct.value() == 100.0
    assert not row.upper_pct.isEnabled()
    row.method.setCurrentIndex(row.method.findData("robust_spread"))
    assert row.rule()["params"]["side"] == "lower"
    assert not row.side.isEnabled()
    row.attribute.setCurrentText("cfr")
    assert row.upper_pct.isEnabled() and row.side.isEnabled()
    page.deleteLater()


def test_preferences_dialog_has_a_process_page_that_saves(_app):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.settings.preferences_dialog import PreferencesDialog

    state = AppState()
    state.save_prefs = lambda: None
    dialog = PreferencesDialog(state)
    try:
        names = [dialog._page_list.item(i).text() for i in range(dialog._page_list.count())]
        assert names.index("Process") == names.index("Attributes") + 1
        page = dialog._process_page
        group_titles = [g.title() for g in page.findChildren(type(page.filter_group))]
        assert group_titles == ["Filter", "Channel", "ROI"]
        page.rows()[1].enabled.setChecked(False)
        dialog._accept()
        assert state.prefs["process"]["filter"]["auto_rules"][1]["enabled"] is False
    finally:
        dialog.close()


# ---------------------------------------------------------------------------
# Menus
# ---------------------------------------------------------------------------

def test_filter_heads_the_process_menu_and_precision_menu_is_split(_app):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.main_window import MainWindow

    state = AppState()
    state.save_prefs = lambda: None
    state.prefs.setdefault("data", {}).update({"show_data_info": False, "show_render": False})
    win = MainWindow(state)
    try:
        u = win._ui
        process = [a for a in u.menuProcess.actions()]
        assert process[0] is u.actionFilter
        assert process[1] is win.menuProcessChannel.menuAction()
        assert u.actionFilter not in u.menuEdit.actions()

        menu = u.menuLocPrecision
        assert menu.title() == "Loc Precision / Resolution"
        entries = [("—" if a.isSeparator() else a.text()) for a in menu.actions()]
        assert entries == ["CRLB (Cramer-Rao Lower Bound)", "StdDev per Trace", "—",
                           "FRC Resolution Estimation"]
    finally:
        win.close()
        _app.processEvents()
