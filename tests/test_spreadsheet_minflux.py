"""A MINFLUX table this application wrote must reload as MINFLUX data.

Two separate defects motivate this file:

* the canonical CSV the MSR reader exports still went through the generic
  column-mapping dialog, and
* a table mapped by hand to x/y/z/tid/tim pooled **every iteration of every
  localization** into one cloud and kept the invalid probes, because the generic
  mapping had no notion of ``itr`` / ``vld``.
"""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.core import loader
from minflux_viewer.core.save import save_processed
from minflux_viewer.core.spreadsheet_loader import (
    ROLES,
    build_dataset_from_mapping,
    guess_mapping,
    minflux_row_mask,
    minflux_table_kind,
    read_table,
)
from minflux_viewer.msr.export import canonical_dataset

N_LOC, N_ITR = 400, 4


def _raw_mfx() -> np.ndarray:
    """A raw canonical array: one row per (localization x iteration)."""
    rng = np.random.default_rng(3)
    n = N_LOC * N_ITR
    dtype = np.dtype([("itr", "i4"), ("tid", "u4"), ("tim", "f8"), ("vld", "?"),
                      ("loc", ("f8", 3)), ("efo", "f8"), ("cfr", "f8"),
                      ("dcr", ("f8", 2)), ("eco", "i4")])
    arr = np.zeros(n, dtype=dtype)
    arr["itr"] = np.tile(np.arange(N_ITR), N_LOC)
    arr["tid"] = np.repeat(rng.integers(1, 60, N_LOC), N_ITR)
    arr["tim"] = np.repeat(np.cumsum(rng.random(N_LOC)), N_ITR)
    arr["vld"] = np.repeat(rng.random(N_LOC) > 0.2, N_ITR)
    arr["loc"] = (np.repeat(rng.random((N_LOC, 3)) * 1e-6, N_ITR, axis=0)
                  + rng.normal(0.0, 1e-9, (n, 3)))
    arr["efo"] = rng.random(n) * 1e5
    arr["eco"] = rng.integers(100, 3000, n)
    return arr


@pytest.fixture
def canonical_csv(tmp_path):
    arr = _raw_mfx()
    ds = canonical_dataset(arr, name="run.msr", folder=str(tmp_path),
                           mbm=None, mbm_meta=None)
    paths = save_processed(
        ds, data_path=tmp_path / "run_mfx", fmt="csv", content="raw",
        include={"attrs": True, "derived": False, "recipe": False})
    expected = int((arr["vld"] & (arr["itr"] == N_ITR - 1)).sum())
    return paths[0], expected


def test_headers_identify_a_raw_or_snapshot_minflux_table(canonical_csv):
    path, _expected = canonical_csv
    assert minflux_table_kind(read_table(path).headers) == "raw"
    # Component-index spelling of the split loc vector is the same layout.
    assert minflux_table_kind(["loc_0", "loc_1", "loc_2", "itr", "vld"]) == "raw"
    # The processed table File > Save writes.
    assert minflux_table_kind(["xnm", "ynm", "znm", "tid", "tim"]) == "snapshot"
    # A generic localization table is neither, and keeps the mapping dialog.
    assert minflux_table_kind(["x [nm]", "y [nm]", "frame", "intensity"]) is None
    assert minflux_table_kind(["xnm", "ynm"]) is None


def test_canonical_csv_bypasses_the_mapping_dialog(canonical_csv, monkeypatch):
    """The dialog must not even be constructed for a table we wrote."""
    pytest.importorskip("PyQt6")
    from minflux_viewer.ui import spreadsheet_import_dialog as mod

    path, expected = canonical_csv

    def _refuse(*_args, **_kwargs):
        raise AssertionError("a canonical MINFLUX table must not need mapping")

    monkeypatch.setattr(mod, "SpreadsheetMappingDialog", _refuse)
    seen: list[str] = []
    ds = mod.import_spreadsheet(path, log=seen.append)

    assert ds.prop.num_loc == expected
    assert seen and "canonical MINFLUX raw table" in seen[0]


def test_direct_load_keeps_the_iteration_axis_and_drops_invalid_rows(canonical_csv):
    path, expected = canonical_csv
    ds = loader.load_csv(str(path))
    # One localization per valid trace event, not one per (loc x iteration) row.
    assert ds.prop.num_loc == expected
    assert int(ds.prop.num_itr) == N_ITR


def test_a_snapshot_csv_pins_the_z_factor_like_the_mat_and_npy_readers(tmp_path):
    """Baked XOR recipe: a processed snapshot must not be z-corrected twice.

    ``_flat_table_from_dict`` already pinned it for .mat/.npy; the CSV reader
    did not, so the same file behaved differently depending on its extension.
    """
    ds0 = canonical_dataset(_raw_mfx(), name="run.msr", folder=str(tmp_path),
                            mbm=None, mbm_meta=None)
    ds0.set_z_scaling_factor(0.7, source="manual (Dataset Information)")
    displayed = np.nanmax(ds0.loc_nm[:, 2])

    path = save_processed(
        ds0, data_path=tmp_path / "snap", fmt="csv", content="snapshot",
        include={"attrs": True, "derived": False, "recipe": False},
        filter_mode="flag")[0]
    assert minflux_table_kind(read_table(path).headers) == "snapshot"

    ds = loader.load_csv(str(path))
    assert ds.cali.z_scaling_factor == pytest.approx(1.0)
    assert ds.derived["z_scaling_factor"][0] == pytest.approx(1.0)
    # The baked z survives unchanged; it is not multiplied by 0.7 again.
    assert np.nanmax(ds.loc_nm[:, 2]) == pytest.approx(displayed)


def test_a_raw_csv_does_not_pin_the_z_factor(tmp_path, canonical_csv):
    """The raw export carries unscaled loc_z, so the factor stays live."""
    path, _expected = canonical_csv
    ds = loader.load_csv(str(path))
    assert "z_scaling_factor" not in ds.derived.keys()


def test_an_excel_workbook_never_takes_the_direct_route(tmp_path):
    """``load_csv`` cannot read a workbook however its columns are named."""
    pytest.importorskip("PyQt6")
    from minflux_viewer.ui.spreadsheet_import_dialog import minflux_direct_load_kind

    workbook = tmp_path / "run.xlsx"
    workbook.write_bytes(b"PK\x03\x04not-really")
    assert minflux_direct_load_kind(workbook) is None


# --- the manual mapping: itr / vld select rows -----------------------------

def test_itr_and_vld_are_offered_as_roles_and_auto_detected(canonical_csv):
    path, _expected = canonical_csv
    assert "itr" in ROLES and "vld" in ROLES
    mapping = guess_mapping(read_table(path), use_values=True)
    assert mapping["itr"] == "itr"
    assert mapping["vld"] == "vld"
    # 'eco' is the MINFLUX photon count and must be recognised as one.
    assert mapping["photons"] == "eco"


def test_mapping_itr_and_vld_reduces_rows_to_the_last_valid_iteration(canonical_csv):
    path, expected = canonical_csv
    table = read_table(path)
    units = {"x": "m", "y": "m", "z": "m"}
    base = {role: None for role in ROLES}
    base.update({"x": "loc_x", "y": "loc_y", "z": "loc_z",
                 "id": "tid", "frame": "tim", "photons": "eco"})

    without = build_dataset_from_mapping(table, dict(base), units=units)
    assert without.prop.num_loc == N_LOC * N_ITR      # the defect: every row

    mapped = dict(base, itr="itr", vld="vld")
    with_sel = build_dataset_from_mapping(table, mapped, units=units)
    assert with_sel.prop.num_loc == expected
    # The photon column keeps its canonical MINFLUX name.
    assert "eco" in with_sel.attr.keys()
    assert with_sel.attr["eco"].size == expected
    # Provenance says how much was dropped, so the mapping can be checked.
    selection = with_sel.metadata["spreadsheet_row_selection"]
    assert selection["rows_read"] == N_LOC * N_ITR
    assert selection["rows_kept"] == expected


def test_vld_alone_drops_invalid_rows_and_itr_alone_keeps_the_last(canonical_csv):
    path, _expected = canonical_csv
    table = read_table(path)
    units = {"x": "m", "y": "m", "z": "m"}
    base = {role: None for role in ROLES}
    base.update({"x": "loc_x", "y": "loc_y", "z": "loc_z"})

    itr = np.asarray(table.by_name("itr").values)
    vld = np.asarray(table.by_name("vld").values)

    vld_only = build_dataset_from_mapping(table, dict(base, vld="vld"), units=units)
    assert vld_only.prop.num_loc == int((vld != 0).sum())

    itr_only = build_dataset_from_mapping(table, dict(base, itr="itr"), units=units)
    assert itr_only.prop.num_loc == int((itr == itr.max()).sum())


def test_row_mask_reproduces_the_native_last_valid_materialization():
    itr = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=float)
    vld = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=float)

    assert minflux_row_mask(None, None) is None
    # Valid rows at the global maximum iteration -- exactly mfx_row_mask's rule.
    assert list(minflux_row_mask(itr, vld)) == [
        False, False, True, False, False, False, False, False, True]
    # The maximum is taken over the VALID rows: an invalid row at a higher
    # iteration must not move the selection off the real last iteration.
    itr_high = np.array([0, 1, 2, 9], dtype=float)
    vld_high = np.array([1, 1, 1, 0], dtype=float)
    assert list(minflux_row_mask(itr_high, vld_high)) == [False, False, True, False]
    # NaN never survives either constraint.
    assert list(minflux_row_mask(np.array([np.nan, 1.0]), None)) == [False, True]


def test_the_bypass_check_reads_only_the_header(canonical_csv, monkeypatch):
    """A table that is NOT ours must not be previewed twice."""
    pytest.importorskip("PyQt6")
    from minflux_viewer.core import spreadsheet_loader as core_mod
    from minflux_viewer.ui.spreadsheet_import_dialog import minflux_direct_load_kind

    path, _expected = canonical_csv

    def _refuse(*_args, **_kwargs):
        raise AssertionError("deciding the kind must not sample rows")

    monkeypatch.setattr(core_mod, "read_table_preview", _refuse)
    monkeypatch.setattr(core_mod, "read_table", _refuse)
    assert minflux_direct_load_kind(path) == "raw"
    assert core_mod.delimited_header_row(path)[:2] == ["loc_x", "loc_y"]
    # A file that is not delimited text is simply "not one".
    assert core_mod.delimited_header_row(path.parent / "missing.csv") == []
