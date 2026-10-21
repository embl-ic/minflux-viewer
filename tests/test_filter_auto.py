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
        assert dialog._info.text().startswith("Auto: efo ≤")
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
        assert "efo ≤" in sentence and "cfr ≤" in sentence and "siz 10–" in sentence
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
