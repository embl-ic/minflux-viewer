"""Inclusive / exclusive filter bounds: the table's brackets, the histogram's
solid / dashed bound lines, the preset file's flags and the Auto text."""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

#: Exactly representable, so a bound equal to a value is equal in any dtype.
_LEVELS = np.array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])


# ---------------------------------------------------------------------------
# Pure: preset flags and the range mask
# ---------------------------------------------------------------------------

def _write(path, rows):
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_a_string_flag_in_a_preset_is_refused_by_name_not_read_as_true(tmp_path):
    from minflux_viewer.core import filter_io

    path = _write(tmp_path / "hand_edited.json", [
        {"apply": True, "attribute": "cfr", "min": 0.1, "max": 0.8,
         "min_inclusive": True, "max_inclusive": "false"}])
    # Still recognized as a preset, so it reaches the loader ...
    assert filter_io.is_filter_json_file(path) is True
    # ... which names the field instead of reading "false" as True.
    with pytest.raises(ValueError, match=r"row 1: max_inclusive must be true or false, not 'false'"):
        filter_io.load_filter_json(path)
    with pytest.raises(ValueError, match="max_inclusive"):
        filter_io.filter_specs_from_rows(json.loads(path.read_text(encoding="utf-8")))
    with pytest.raises(ValueError, match="min_inclusive"):
        filter_io.save_filter_json(tmp_path / "out.json", [
            {"apply": True, "attribute": "cfr", "min": 0.1, "max": 0.8, "min_inclusive": 0}])


def test_preset_flags_absent_mean_inclusive_and_booleans_round_trip(tmp_path):
    from minflux_viewer.core import filter_io

    rows = [{"apply": True, "attribute": "cfr", "min": 0.1, "max": 0.8},
            {"apply": True, "attribute": "efo", "min": 1.0, "max": 9.0,
             "min_inclusive": False, "max_inclusive": True}]
    path = _write(tmp_path / "preset.json", rows)
    specs = filter_io.filter_specs_from_rows(filter_io.load_filter_json(path))
    assert [(s["lo_inc"], s["hi_inc"]) for s in specs] == [(True, True), (False, True)]


def test_value_range_mask_honours_each_end_and_keeps_flags_with_their_values():
    from minflux_viewer.core.roi_selection import value_range_mask

    values = _LEVELS
    inside = value_range_mask(values, 0.25, 0.75)
    assert values[inside].tolist() == [0.25, 0.375, 0.5, 0.625, 0.75]
    half_open = value_range_mask(values, 0.25, 0.75, lo_inclusive=True, hi_inclusive=False)
    assert values[half_open].tolist() == [0.25, 0.375, 0.5, 0.625]
    # Reversed ends: the exclusive flag stays with 0.75.
    reversed_ends = value_range_mask(values, 0.75, 0.25, lo_inclusive=False, hi_inclusive=True)
    assert values[reversed_ends].tolist() == [0.25, 0.375, 0.5, 0.625]


# ---------------------------------------------------------------------------
# Qt: the filter table and the histogram edit
# ---------------------------------------------------------------------------

def _dataset(n_repeat: int = 10):
    from minflux_viewer.core.dataset import build_localization_dataset

    cfr = np.tile(_LEVELS, n_repeat)
    n = cfr.size
    rng = np.random.default_rng(3)
    return build_localization_dataset(
        name="incl.mat", x_nm=rng.random(n) * 100, y_nm=rng.random(n) * 100,
        tid=np.arange(n), attrs={"cfr": cfr,
                                 "efo": rng.normal(60_000.0, 7_000.0, n)})


def _dialog(ds, *, rules=None, main=None):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.dialogs.filter_dialog import FilterDialog

    state = AppState()
    state.save_prefs = lambda: None
    if rules is not None:
        state.prefs.setdefault("process", {}).setdefault("filter", {})["auto_rules"] = rules
    state.add_dataset(ds)
    dialog = FilterDialog(state, dataset_idx=0)
    if main is not None:
        dialog._main_window = lambda: main
    return dialog, state


def _brackets(dialog, row=0):
    from minflux_viewer.ui.dialogs import filter_dialog as fd

    return (dialog._table.cellWidget(row, fd._COL_MIN_INC).text(),
            dialog._table.cellWidget(row, fd._COL_MAX_INC).text())


def _passing(ds):
    cfr = np.asarray(ds.attr["cfr"], dtype=float)
    return sorted(set(cfr[np.asarray(ds.filter_mask, dtype=bool)].tolist()))


def test_brackets_show_and_toggle_whether_a_bound_passes(qapp):
    from minflux_viewer.ui.dialogs import filter_dialog as fd

    ds = _dataset()
    dialog, _state = _dialog(ds)
    try:
        dialog._add_row(attr="cfr", lo=0.25, hi=0.75, enabled=True, auto_range=False)
        dialog._apply_all()
        assert _brackets(dialog) == ("[", "]")
        assert _passing(ds) == [0.25, 0.375, 0.5, 0.625, 0.75]

        dialog._table.cellWidget(0, fd._COL_MIN_INC).click()      # applies at once
        assert _brackets(dialog) == ("(", "]")
        assert _passing(ds) == [0.375, 0.5, 0.625, 0.75]
        dialog._table.cellWidget(0, fd._COL_MAX_INC).click()
        assert _brackets(dialog) == ("(", ")")
        assert _passing(ds) == [0.375, 0.5, 0.625]
        spec = ds.state["filter_specs"][0]
        assert (spec["lo_inc"], spec["hi_inc"]) == (False, False)
        assert "exclusive" in dialog._table.cellWidget(0, fd._COL_MIN_INC).toolTip()

        dialog._table.cellWidget(0, fd._COL_MIN_INC).click()
        assert _brackets(dialog) == ("[", ")")
        assert _passing(ds) == [0.25, 0.375, 0.5, 0.625]
    finally:
        dialog.close()


def test_brackets_follow_the_saved_state_a_preset_and_auto(qapp, tmp_path, monkeypatch):
    from minflux_viewer.ui.dialogs import filter_dialog as fd

    ds = _dataset()
    dialog, state = _dialog(ds)
    try:
        dialog._add_row(attr="cfr", lo=0.25, hi=0.75, enabled=True, auto_range=False,
                        max_inclusive=False)
        dialog._apply_all()
        assert _brackets(dialog) == ("[", ")")
        path = tmp_path / "half_open.json"
        monkeypatch.setattr(fd.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *_a, **_k: (str(path), "")))
        dialog._save_json()
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert (saved[0]["min_inclusive"], saved[0]["max_inclusive"]) == (True, False)
    finally:
        dialog.close()

    # Reopened from the dataset's own filter state.
    reopened = fd.FilterDialog(state, dataset_idx=0)
    try:
        assert _brackets(reopened) == ("[", ")")
        # A preset row, appended.
        reopened.load_filter_json(str(path))
        assert _brackets(reopened, row=1) == ("[", ")")
        # Auto rewrites a cfr row (inclusive, both ends) through blocked signals.
        reopened._table.setRowCount(0)
        reopened._add_row(attr="cfr", lo=0.25, hi=0.75, enabled=True, auto_range=False,
                          min_inclusive=False, max_inclusive=False)
        state.prefs.setdefault("process", {}).setdefault("filter", {})["auto_rules"] = [
            {"attribute": "cfr", "method": "robust_spread", "params": {"side": "both", "k": 10.0}}]
        reopened._apply_auto_rules()
        assert _brackets(reopened) == ("[", "]")
    finally:
        reopened.close()


def test_a_preset_with_a_string_flag_is_refused_with_its_reason(qapp, tmp_path, monkeypatch):
    from minflux_viewer.ui.dialogs import filter_dialog as fd

    path = _write(tmp_path / "hand_edited.json", [
        {"apply": True, "attribute": "cfr", "min": 0.1, "max": 0.8, "min_inclusive": "false"}])
    messages = []
    monkeypatch.setattr(fd.QMessageBox, "critical",
                        staticmethod(lambda _parent, _title, text: messages.append(text)))
    dialog, _state = _dialog(_dataset())
    try:
        dialog.load_filter_json(str(path))
        assert dialog._table.rowCount() == 0
        assert len(messages) == 1 and "min_inclusive must be true or false" in messages[0]
    finally:
        dialog.close()


def test_resetting_a_row_to_the_data_range_makes_both_ends_inclusive(qapp):
    from minflux_viewer.ui.dialogs import filter_dialog as fd

    ds = _dataset()
    dialog, _state = _dialog(ds)
    try:
        dialog._add_row(attr="cfr", lo=0.25, hi=0.75, enabled=True, auto_range=False,
                        min_inclusive=False, max_inclusive=False)
        # A mode change keeps the bounds, so it keeps their ends too.
        dialog._table.cellWidget(0, fd._COL_MODE).setCurrentText("trace mean")
        assert _brackets(dialog) == ("(", ")")
        # An attribute change resets the bounds to the new data's extremes.
        dialog._table.cellWidget(0, fd._COL_ATTR).setCurrentText("efo")
        assert _brackets(dialog) == ("[", "]")
    finally:
        dialog.close()


def _histogram_main(state):
    from minflux_viewer.ui.plots.histogram_window import HistogramWindow

    hist = HistogramWindow(state, dataset_idx=0)

    class _MainWindow:
        _histogram_windows: dict = {}

        def _show_histogram(self, _idx):
            return hist

    return hist, _MainWindow()


def test_histogram_edit_keeps_an_exclusive_row_exclusive(qapp):
    """Update / Finish used to hand back inclusive ends whatever the row had."""
    ds = _dataset()
    dialog, state = _dialog(ds)
    hist, main = _histogram_main(state)
    dialog._main_window = lambda: main
    try:
        dialog._add_row(attr="cfr", lo=0.25, hi=0.75, enabled=True, auto_range=False,
                        max_inclusive=False)
        dialog._apply_all()
        dialog._display_filter(0)
        edit = hist._filter_edit
        assert (edit["inclusive_min"], edit["inclusive_max"]) == (True, False)

        hist._update_filter_edit()
        assert _brackets(dialog) == ("[", ")")
        assert ds.state["filter_specs"][0]["hi_inc"] is False
        assert _passing(ds) == [0.25, 0.375, 0.5, 0.625]

        hist._finish_filter_edit()
        assert _brackets(dialog) == ("[", ")")
        assert ds.state["filter_specs"][0]["hi_inc"] is False

        # Cancel restores the row's own ends after the edit changed them.
        dialog._display_filter(0)
        hist._filter_edit["inclusive_min"] = False
        hist._update_filter_edit()
        assert _brackets(dialog) == ("(", ")")
        hist._cancel_filter_edit()
        assert _brackets(dialog) == ("[", ")")
    finally:
        dialog.close()
        hist.close()


def _pen_styles(region):
    from PyQt6.QtCore import Qt

    return ["solid" if line.pen.style() == Qt.PenStyle.SolidLine else
            "dashed" if line.pen.style() == Qt.PenStyle.DashLine else str(line.pen.style())
            for line in region.lines]


def test_histogram_draws_inclusive_bounds_solid_and_exclusive_dashed(qapp):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.plots.histogram_window import HistogramWindow

    state = AppState()
    state.add_dataset(_dataset())
    win = HistogramWindow(state, dataset_idx=0)
    try:
        win.start_filter_edit(attr="cfr", mode="per loc", lo=0.25, hi=0.75,
                              inclusive_min=True, inclusive_max=False)
        region = win._filter_edit["region"]
        assert _pen_styles(region) == ["solid", "dashed"]
        report = win._filter_edit["report_label"].textItem.toPlainText()
        assert "min: 0.25 (inclusive)" in report and "max: 0.75 (exclusive)" in report
        # The count beside it uses the same ends: 0.75 is not in the filter.
        assert "loc in filter 40 / 70" in report

        # Dragged past each other, the style follows the position.
        region.lines[0].setValue(0.9)
        assert _pen_styles(region) == ["dashed", "solid"]

        # A colour change from Preferences keeps the dashes.
        win._restyle_filter_edit()
        assert _pen_styles(region) == ["dashed", "solid"]

        # Reset to the data range: both ends inclusive again.
        win._reset_filter_region_to_current_data()
        assert _pen_styles(region) == ["solid", "solid"]
        assert win._filter_edit["inclusive_max"] is True
    finally:
        win.close()


def test_histogram_count_and_lines_follow_the_filter_bounds_popup(qapp, monkeypatch):
    from PyQt6.QtWidgets import QCheckBox, QDialog

    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.plots.histogram_window import HistogramWindow

    state = AppState()
    state.add_dataset(_dataset())
    win = HistogramWindow(state, dataset_idx=0)

    def accept_with_min_exclusive(dlg):
        boxes = dlg.findChildren(QCheckBox)
        boxes[0].setChecked(False)                    # "Min bound": Inclusive
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", accept_with_min_exclusive)
    try:
        win.start_filter_edit(attr="cfr", mode="per loc", lo=0.25, hi=0.75)
        assert _pen_styles(win._filter_edit["region"]) == ["solid", "solid"]
        assert "loc in filter 50 / 70" in win._filter_edit["report_label"].textItem.toPlainText()
        win._show_filter_bounds_dialog()
        assert _pen_styles(win._filter_edit["region"]) == ["dashed", "solid"]
        report = win._filter_edit["report_label"].textItem.toPlainText()
        assert "loc in filter 40 / 70" in report and "(exclusive)" in report
        assert win._current_filter_edit_values()[2:] == (False, True)
    finally:
        win.close()


def test_auto_text_states_two_sided_bounds_as_a_closed_interval(qapp):
    ds = _dataset(n_repeat=40)
    dialog, state = _dialog(ds, rules=[
        {"attribute": "efo", "method": "robust_spread", "params": {"side": "both", "k": 2.5}}])
    try:
        dialog._apply_auto_rules()
        text = dialog._info.text()
        # No grouping commas inside the brackets: "[40123, 79876]", not "[40,123, 79,876]".
        match = re.search(r"efo ∈ \[(\d+(?:\.\d+)?), (\d+(?:\.\d+)?)\]", text)
        assert match, text
        lo, hi = (float(v) for v in match.groups())
        assert 0.0 <= lo < hi
        assert any("efo ∈ [" in line["message"] for line in state.log_history)
    finally:
        dialog.close()
