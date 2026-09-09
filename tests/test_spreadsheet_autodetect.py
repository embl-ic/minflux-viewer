"""Value-based auto-detection for headerless / generically-named spreadsheets.

The loader guesses the MINFLUX localization roles (x/y/z → xnm/ynm/znm, id → tid,
frame → tim) and their units from each numeric column's cheap value statistics
(dtype, span, median step, unique count) when the header text doesn't name them.
"""

import numpy as np
import pytest

from minflux_viewer.core.spreadsheet_loader import (
    build_dataset_from_mapping,
    column_stats,
    guess_length_unit_from_span,
    guess_mapping,
    guess_time_unit,
    guess_units,
    read_table,
    time_unit_guess,
)


def _write_cols(path, header, cols):
    n = len(cols[0])
    lines = [",".join(header)]
    lines += [",".join(str(c[i]) for c in cols) for i in range(n)]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _write_data_only(path, cols, sep=","):
    """Write a header-less table (first row is data)."""
    n = len(cols[0])
    lines = [sep.join(str(c[i]) for c in cols) for i in range(n)]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _minflux_like(rng, n_traces=200, per=20, unit_scale=1.0, z_span_nm=200.0,
                  time_scale=1.0):
    """Generic-named MINFLUX-ish table: coords in *unit_scale* (nm=1, µm=1e-3),
    an integer trace id, and a monotone time column scaled by *time_scale*."""
    tid = np.repeat(np.arange(1, n_traces + 1), per)
    n = tid.size
    x = rng.uniform(1000.0, 3000.0, n) * unit_scale       # 2000 nm span
    y = rng.uniform(5000.0, 7000.0, n) * unit_scale
    z = rng.uniform(-z_span_nm / 2, z_span_nm / 2, n) * unit_scale
    tim = np.cumsum(rng.uniform(0.0008, 0.0012, n)) * time_scale   # ~1 ms step
    photons = rng.integers(500, 1500, n)
    return tid, x, y, z, tim, photons


# --------------------------------------------------------------- column stats

def test_column_stats_dtype_range_step_unique(tmp_path):
    p = _write_cols(tmp_path / "s.csv", ["a", "b"],
                    [[1, 1, 2, 2, 3, 3], [0.0, 10.0, 20.0, 30.0, 40.0, 50.5]])
    t = read_table(p)
    sa = column_stats(t.by_name("a"))
    assert sa.is_integer and sa.n_unique == 3 and sa.mean_count_per_unique == 2.0
    assert sa.vmin == 1 and sa.vmax == 3 and sa.ptp == 2
    sb = column_stats(t.by_name("b"))
    assert not sb.is_integer and sb.ptp == pytest.approx(50.5)   # fractional → float
    assert sb.median_abs_diff == 10.0        # 10-unit steps (last 10.5)


def test_length_unit_from_span_matches_magnitude():
    assert guess_length_unit_from_span(5000.0) == "nm"      # nm-scale numbers
    assert guess_length_unit_from_span(5.0) == "um"         # µm-scale numbers
    assert guess_length_unit_from_span(5.0e-6) == "m"       # metre-scale numbers


def _stat(median, *, is_integer=False, frac_increasing=1.0):
    from minflux_viewer.core.spreadsheet_loader import ColumnStats
    return ColumnStats(n_finite=100, is_integer=is_integer, vmin=0.0, vmax=1.0,
                       ptp=1.0, median_abs_diff=median, n_unique=100,
                       mean_count_per_unique=1.0, frac_increasing=frac_increasing)


def test_time_unit_guess_seconds_and_millis():
    assert time_unit_guess(_stat(median=1.0e-3)) == "s"     # 1 ms step in seconds
    assert time_unit_guess(_stat(median=1.0)) == "ms"       # 1 ms step in ms units
    assert time_unit_guess(_stat(median=100.0)) is None     # not time-like


# --------------------------------------------------------- headerless mapping

def test_value_based_detects_headerless_minflux_nm(tmp_path):
    rng = np.random.default_rng(0)
    tid, x, y, z, tim, ph = _minflux_like(rng)
    p = _write_cols(tmp_path / "raw.csv",
                    ["col_a", "col_b", "col_c", "group_no", "stamp", "signal"],
                    [x, y, z, tid, tim, ph])
    t = read_table(p)
    m = guess_mapping(t, use_values=True)
    assert m["x"] == "col_a" and m["y"] == "col_b" and m["z"] == "col_c"
    assert m["id"] == "group_no"
    assert m["frame"] == "stamp"
    assert guess_units(t, m) == {"x": "nm", "y": "nm", "z": "nm"}
    assert guess_time_unit(t, m) == "s"


def test_value_based_micrometre_coordinates(tmp_path):
    rng = np.random.default_rng(1)
    tid, x, y, z, tim, ph = _minflux_like(rng, unit_scale=1.0e-3)   # µm values
    p = _write_cols(tmp_path / "um.csv", ["c1", "c2"], [x, y])
    t = read_table(p)
    m = guess_mapping(t, use_values=True)
    assert m["x"] == "c1" and m["y"] == "c2"
    assert guess_units(t, m)["x"] == "um" and guess_units(t, m)["y"] == "um"


def test_value_based_flat_z_reads_as_2d(tmp_path):
    rng = np.random.default_rng(2)
    tid, x, y, z, tim, ph = _minflux_like(rng, z_span_nm=2.0)       # z < 5 nm → flat
    p = _write_cols(tmp_path / "2d.csv", ["c1", "c2", "c3"], [x, y, z])
    t = read_table(p)
    m = guess_mapping(t, use_values=True)
    assert m["x"] == "c1" and m["y"] == "c2"
    assert m["z"] is None                     # flat third axis → 2-D, z unmapped


def test_value_based_millisecond_time(tmp_path):
    rng = np.random.default_rng(3)
    tid, x, y, z, tim, ph = _minflux_like(rng, time_scale=1000.0)   # time in ms
    p = _write_cols(tmp_path / "ms.csv",
                    ["c1", "c2", "c3", "grp", "clock"], [x, y, z, tid, tim])
    t = read_table(p)
    m = guess_mapping(t, use_values=True)
    assert m["frame"] == "clock"
    assert guess_time_unit(t, m) == "ms"


def test_tid_out_of_range_is_not_detected(tmp_path):
    # Only 4 distinct ids (< 10) → not a trace id; a per-row unique id (count 1)
    # → also rejected.
    rng = np.random.default_rng(4)
    n = 400
    few = np.repeat(np.arange(4), n // 4)
    uniq = np.arange(n)
    x = rng.uniform(1000, 3000, n)
    y = rng.uniform(1000, 3000, n)
    p = _write_cols(tmp_path / "tid.csv", ["few", "uniq", "c1", "c2"],
                    [few, uniq, x, y])
    t = read_table(p)
    m = guess_mapping(t, use_values=True)
    assert m["id"] is None


def test_headerless_file_gets_positional_column_names(tmp_path):
    # First row is all-numeric → no header → positional names, first row kept.
    p = _write_data_only(tmp_path / "nohdr.csv",
                         [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    t = read_table(p)
    assert t.headers == ["1st column", "2nd column"]
    assert t.n_rows == 3                       # the numeric first row is data, not a header


def test_text_header_still_detected(tmp_path):
    p = _write_cols(tmp_path / "hdr.csv", ["xnm", "ynm"],
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    t = read_table(p)
    assert t.headers == ["xnm", "ynm"] and t.n_rows == 3


def test_metres_headerless_with_monotonic_timestamp(tmp_path):
    # Reproduces 'Nuclear Pore Model Data.txt': header-less; columns are
    # tid, time(s, monotonic), x(m), y(m), z(m) — the timestamp's span looks
    # coordinate-like, but it is monotonic so it is the frame, not x.
    rng = np.random.default_rng(7)
    n_tr, per = 300, 15
    tid = np.repeat(np.arange(1, n_tr + 1), per)
    n = tid.size
    tim = np.cumsum(rng.uniform(0.001, 0.02, n))           # monotonic seconds
    x = rng.uniform(-3.6e-6, -1.2e-6, n)                    # metres, scattered
    y = rng.uniform(-1.34e-5, -1.17e-5, n)
    z = rng.uniform(-1.7e-7, 1.5e-7, n)
    p = _write_data_only(tmp_path / "model.txt", [tid, tim, x, y, z], sep="\t")
    t = read_table(p)
    assert t.headers[0] == "1st column"
    m = guess_mapping(t, use_values=True)
    assert m["id"] == "1st column"             # tid
    assert m["frame"] == "2nd column"          # monotonic timestamp (not x!)
    assert m["x"] == "3rd column" and m["y"] == "4th column" and m["z"] == "5th column"
    assert guess_units(t, m) == {"x": "m", "y": "m", "z": "m"}
    assert guess_time_unit(t, m) == "s"
    ds = build_dataset_from_mapping(t, m, units=guess_units(t, m),
                                    time_unit=guess_time_unit(t, m))
    loc = np.asarray(ds.loc_nm, float)
    assert ds.prop.num_dim == 3
    assert -3600 < loc[:, 0].min() and loc[:, 0].max() < -1200        # metres → nm
    assert np.ptp(loc[:, 2]) < 400                                    # small real z


def test_header_match_wins_over_value_guess(tmp_path):
    # An explicit 'xnm'/'ynm' header must map even though other columns also look
    # coordinate-like by value.
    rng = np.random.default_rng(5)
    n = 300
    xnm = rng.uniform(1000, 3000, n)
    ynm = rng.uniform(1000, 3000, n)
    other = rng.uniform(1000, 3000, n)
    p = _write_cols(tmp_path / "h.csv", ["other", "xnm", "ynm"], [other, xnm, ynm])
    t = read_table(p)
    m = guess_mapping(t, use_values=True)
    assert m["x"] == "xnm" and m["y"] == "ynm"


# ---------------------------------------------------------- dataset build

def test_millisecond_time_rescaled_to_seconds(tmp_path):
    t_ms = np.arange(0.0, 100.0, 1.0)          # 1 ms steps, in milliseconds
    x = np.linspace(1000, 3000, t_ms.size)
    y = np.linspace(1000, 3000, t_ms.size)
    tid = np.repeat(np.arange(1, 11), t_ms.size // 10)
    p = _write_cols(tmp_path / "b.csv", ["xnm", "ynm", "grp", "clock"],
                    [x, y, tid, t_ms])
    t = read_table(p)
    m = guess_mapping(t, use_values=True)
    m["frame"] = "clock"
    ds = build_dataset_from_mapping(t, m, time_unit="ms")
    tim = np.asarray(ds.attr.get("tim"), float)
    assert np.isclose(tim.max(), 0.099, atol=1e-6)      # 99 ms → 0.099 s


# ------------------------------------------------------------- dialog pre-fill

def test_dialog_prefills_roles_units_and_builds(tmp_path):
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    from minflux_viewer.ui.spreadsheet_import_dialog import SpreadsheetMappingDialog

    app = QApplication.instance() or QApplication([])
    rng = np.random.default_rng(0)
    tid, x, y, z, tim, ph = _minflux_like(rng)
    p = _write_cols(tmp_path / "raw.csv",
                    ["col_a", "col_b", "col_c", "group_no", "stamp", "signal"],
                    [x, y, z, tid, tim, ph])
    dlg = SpreadsheetMappingDialog(read_table(p))
    assert dlg._role_combos["x"].currentData() == "col_a"
    assert dlg._role_combos["y"].currentData() == "col_b"
    assert dlg._role_combos["z"].currentData() == "col_c"
    assert dlg._role_combos["id"].currentData() == "group_no"
    assert dlg._role_combos["frame"].currentData() == "stamp"
    assert dlg._unit_combos["x"].currentData() == "nm"
    assert dlg._time_combo.currentData() == "s"           # time-unit control pre-filled
    ds = dlg.build_dataset()                              # OK-path build works
    assert ds.prop.num_dim == 3 and ds.prop.num_loc == tid.size
    dlg.close()
    app.processEvents()


# --------------------------------------------- the dialog's progressive depth

def _open_dialog(path):
    """``(app, dialog)`` for *path* — the dialog is never shown, so visibility is
    asserted with ``isHidden()`` (explicit hide) rather than ``isVisible()``."""
    from PyQt6.QtWidgets import QApplication

    from minflux_viewer.ui.spreadsheet_import_dialog import SpreadsheetMappingDialog

    app = QApplication.instance() or QApplication([])
    return app, SpreadsheetMappingDialog(read_table(path))


def _plain_xy(tmp_path):
    """A table whose guess fills nothing beyond the simple parameters."""
    n = 40
    return _write_cols(tmp_path / "xy.csv", ["x [nm]", "y [nm]"],
                       [np.linspace(0.0, 500.0, n), np.linspace(0.0, 900.0, n)])


def test_dialog_opens_simple_and_expands_to_the_full_parameter_set(tmp_path):
    pytest.importorskip("PyQt6")
    app, dlg = _open_dialog(_plain_xy(tmp_path))

    assert not dlg._advanced
    for role in ("x", "y", "z", "frame"):                 # the simple four
        assert not dlg._row_widgets[role][0].isHidden()
    for role in ("prec_xy", "prec_z", "id", "photons", "itr", "vld"):
        assert dlg._row_widgets[role][0].isHidden()
    assert dlg._add_btn.isHidden()                        # only offered expanded
    assert "expand" in dlg._toggle_btn.text()

    dlg._toggle_advanced()
    assert dlg._advanced
    assert all(not w.isHidden()
               for widgets in dlg._row_widgets.values() for w in widgets)
    assert not dlg._add_btn.isHidden() and "fold" in dlg._toggle_btn.text()

    dlg._toggle_advanced()                                # …and back to simple
    assert dlg._row_widgets["vld"][0].isHidden()
    dlg.close()
    app.processEvents()


def test_the_folded_view_owns_up_to_the_parameters_it_hides(tmp_path):
    """The dialog opens folded, so a guessed parameter the simple view does not
    show (here the trace id) is counted in the group title rather than lost."""
    pytest.importorskip("PyQt6")
    rng = np.random.default_rng(0)
    tid, x, y, z, tim, ph = _minflux_like(rng)
    app, dlg = _open_dialog(_write_cols(
        tmp_path / "raw.csv",
        ["col_a", "col_b", "col_c", "group_no", "stamp", "signal"],
        [x, y, z, tid, tim, ph]))

    assert not dlg._advanced
    assert dlg._current_mapping()["id"] == "group_no"     # guessed, and hidden
    assert dlg._row_widgets["id"][0].isHidden()
    assert "more mapped" in dlg._map_group.title()

    dlg._toggle_advanced()
    assert dlg._map_group.title() == "Column mapping"
    assert not dlg._row_widgets["id"][0].isHidden()
    dlg.close()
    app.processEvents()


def test_the_dialog_opens_on_the_folded_mapping_and_ten_preview_rows(tmp_path):
    pytest.importorskip("PyQt6")
    from minflux_viewer.ui.spreadsheet_import_dialog import PREVIEW_ROWS

    app, dlg = _open_dialog(_plain_xy(tmp_path))
    preview = dlg._preview
    row = preview.verticalHeader().defaultSectionSize()
    # The preview holds more rows than that; PREVIEW_ROWS is what it shows.
    assert preview.rowCount() > PREVIEW_ROWS
    assert preview.sizeHint().height() == preview._chrome() + PREVIEW_ROWS * row
    # …and the dialog opened at exactly the height those two parts want.
    assert dlg.height() == dlg.sizeHint().height()
    dlg.close()
    app.processEvents()


def test_add_parameter_offers_only_free_attributes_and_imports_the_column(tmp_path):
    pytest.importorskip("PyQt6")
    rng = np.random.default_rng(0)
    tid, x, y, z, tim, ph = _minflux_like(rng)
    app, dlg = _open_dialog(_write_cols(
        tmp_path / "raw.csv",
        ["col_a", "col_b", "col_c", "group_no", "stamp", "signal"],
        [x, y, z, tid, tim, ph]))

    dlg._toggle_advanced()
    dlg._add_extra_row()
    row = dlg._extra_rows[-1]
    choices = [row["param"].itemData(i) for i in range(row["param"].count())]
    assert "x" not in choices and "itr" not in choices    # standing parameters
    assert choices[0] == "cfr" and row["column"].currentData() is None

    row["column"].setCurrentIndex(row["column"].findData("signal"))
    assert dlg._current_mapping()["cfr"] == "signal"

    # A second row cannot claim the attribute the first one holds.
    dlg._add_extra_row()
    second = dlg._extra_rows[-1]
    assert "cfr" not in [second["param"].itemData(i)
                         for i in range(second["param"].count())]
    # …and changing the first row's parameter moves the column mapping with it.
    row["param"].setCurrentIndex(row["param"].findData("fbg"))
    mapping = dlg._current_mapping()
    assert mapping.get("fbg") == "signal" and "cfr" not in mapping

    dlg._remove_extra_row(second)
    assert len(dlg._extra_rows) == 1
    ds = dlg.build_dataset()
    assert "fbg" in ds.attr and np.allclose(np.asarray(ds.attr["fbg"]), ph)
    dlg.close()
    app.processEvents()


def test_a_removed_parameter_leaves_the_mapping(tmp_path):
    pytest.importorskip("PyQt6")
    app, dlg = _open_dialog(_plain_xy(tmp_path))
    dlg._toggle_advanced()
    dlg._add_extra_row()
    row = dlg._extra_rows[-1]
    assert row["role"] in dlg._current_mapping()
    dlg._remove_extra_row(row)
    assert row["role"] not in dlg._current_mapping()
    assert not dlg._extra_rows
    dlg.close()
    app.processEvents()
