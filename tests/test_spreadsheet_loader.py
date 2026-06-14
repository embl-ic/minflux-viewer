"""Tests for the generic spreadsheet localization importer (Phase 1 core)."""

import numpy as np
import pytest

from minflux_viewer.core.spreadsheet_loader import (
    DEFAULT_PIXEL_SIZE_NM,
    auto_import,
    bracket_unit,
    build_dataset_from_mapping,
    guess_mapping,
    guess_units,
    parse_unit,
    read_table,
    representative_row_indices,
)


# ---------------------------------------------------------------------------
# Synthetic exports
# ---------------------------------------------------------------------------

def _write(path, header, rows, sep=","):
    lines = [sep.join(header)] + [sep.join(str(v) for v in r) for r in rows]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _thunderstorm(tmp_path, n=12):
    hdr = ["id", "frame", "x [nm]", "y [nm]", "z [nm]",
           "uncertainty_xy [nm]", "uncertainty_z [nm]", "intensity [photon]"]
    rows = [[i + 1, i, 100.0 + i, 200.0 + 2 * i, 10.0 + i,
             5.0 + 0.1 * i, 12.0, 800 + i] for i in range(n)]
    return _write(tmp_path / "ts.csv", hdr, rows)


def _smap(tmp_path, n=12):
    hdr = ["xnm", "ynm", "znm", "locprecnm", "locprecznm", "frame", "phot", "groupindex"]
    rows = [[300.0 + i, 400.0 + i, 50.0 + i, 6.0, 14.0, i, 900 + i, i % 4]
            for i in range(n)]
    return _write(tmp_path / "smap.csv", hdr, rows)


def _picasso(tmp_path, n=12):
    # Picasso CSV export: coordinates and lp* in camera PIXELS.
    hdr = ["frame", "x", "y", "photons", "sx", "sy", "bg", "lpx", "lpy",
           "net_gradient", "group"]
    rows = [[i, 5.0 + 0.01 * i, 6.0 + 0.01 * i, 1000, 1.1, 1.1, 50,
             0.03, 0.03, 2000, i % 3] for i in range(n)]
    return _write(tmp_path / "picasso.csv", hdr, rows)


# ---------------------------------------------------------------------------
# Unit parsing
# ---------------------------------------------------------------------------

def test_parse_unit_brackets_and_suffixes():
    assert parse_unit("x [nm]") == "nm"
    assert parse_unit("x [px]") == "px"
    assert parse_unit("x [pix]") == "px"
    assert parse_unit("x [um]") == "um"
    assert parse_unit("xnm") == "nm"
    assert parse_unit("x_pix") == "px"
    assert parse_unit("x") is None          # bare → ambiguous
    assert parse_unit("locprecnm") == "nm"


def test_bracket_unit_only():
    assert bracket_unit("uncertainty_xy [nm]") == "nm"
    assert bracket_unit("lpx") is None       # no bracket → inherit coord unit
    assert bracket_unit("locprecnm") is None


# ---------------------------------------------------------------------------
# read_table + auto-mapping
# ---------------------------------------------------------------------------

def test_thunderstorm_read_and_mapping(tmp_path):
    t = read_table(_thunderstorm(tmp_path))
    assert t.detected_tool == "thunderstorm"
    assert t.n_rows == 12
    m = guess_mapping(t)
    assert m["x"] == "x [nm]" and m["y"] == "y [nm]" and m["z"] == "z [nm]"
    assert m["prec_xy"] == "uncertainty_xy [nm]"
    assert m["prec_z"] == "uncertainty_z [nm]"
    assert m["id"] == "id" and m["frame"] == "frame"
    assert m["photons"] == "intensity [photon]"
    assert guess_units(t, m) == {"x": "nm", "y": "nm", "z": "nm"}


def test_smap_read_and_mapping(tmp_path):
    t = read_table(_smap(tmp_path))
    assert t.detected_tool == "smap"
    m = guess_mapping(t)
    assert m["x"] == "xnm" and m["y"] == "ynm" and m["z"] == "znm"
    assert m["prec_xy"] == "locprecnm" and m["prec_z"] == "locprecznm"
    assert m["id"] == "groupindex"
    assert guess_units(t, m)["x"] == "nm"


def test_picasso_read_and_mapping(tmp_path):
    t = read_table(_picasso(tmp_path))
    assert t.detected_tool == "picasso"
    m = guess_mapping(t)
    assert m["x"] == "x" and m["y"] == "y"
    assert m["prec_xy"] in ("lpx", "lpy")
    assert m["id"] == "group"
    # Picasso coords are pixels → default unit px
    assert guess_units(t, m) == {"x": "px", "y": "px", "z": "px"}


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def test_build_thunderstorm_dataset(tmp_path):
    t = read_table(_thunderstorm(tmp_path))
    ds = build_dataset_from_mapping(t, guess_mapping(t))
    # canonical metres storage → loc_nm recovers the nm coordinates
    loc = np.asarray(ds.loc_nm, float)
    assert np.allclose(loc[:, 0], 100.0 + np.arange(12))
    assert np.allclose(loc[:, 1], 200.0 + 2 * np.arange(12))
    assert ds.prop.num_dim == 3
    assert np.allclose(np.asarray(ds.attr.get("loc_x")), (100.0 + np.arange(12)) * 1e-9)
    assert np.allclose(np.asarray(ds.attr.get("tid")), 1 + np.arange(12))   # from id
    assert "loc_precision_xy" in ds.attr
    assert np.allclose(np.asarray(ds.attr.get("loc_precision_xy")), 5.0 + 0.1 * np.arange(12))


def test_build_picasso_pixels_converts_with_pixel_size(tmp_path):
    t = read_table(_picasso(tmp_path))
    m = guess_mapping(t)
    ps = 100.0
    ds = build_dataset_from_mapping(t, m, pixel_size_nm=ps)
    loc = np.asarray(ds.loc_nm, float)
    assert np.allclose(loc[:, 0], (5.0 + 0.01 * np.arange(12)) * ps)
    # precision (lpx, pixels) also converted with the pixel size
    assert np.allclose(np.asarray(ds.attr.get("loc_precision_xy")), 0.03 * ps)


def test_build_picasso_pixels_without_pixel_size_raises(tmp_path):
    t = read_table(_picasso(tmp_path))
    with pytest.raises(ValueError, match="pixel size"):
        build_dataset_from_mapping(t, guess_mapping(t), pixel_size_nm=None)


def test_build_requires_x_y(tmp_path):
    t = read_table(_thunderstorm(tmp_path))
    m = guess_mapping(t)
    m["x"] = None
    with pytest.raises(ValueError, match="x and y"):
        build_dataset_from_mapping(t, m)


def test_passthrough_extra_numeric_columns(tmp_path):
    # Picasso has unmapped columns (sx, sy, bg, net_gradient) → carried through.
    t = read_table(_picasso(tmp_path))
    ds = build_dataset_from_mapping(t, guess_mapping(t), pixel_size_nm=100.0)
    assert "bg" in ds.attr and "net_gradient" in ds.attr
    assert "photons" in ds.attr          # 'photons' role mapped from 'photons'
    assert np.allclose(np.asarray(ds.attr.get("bg")), 50.0)


# ---------------------------------------------------------------------------
# Formats & helpers
# ---------------------------------------------------------------------------

def test_tsv_delimiter_autodetect(tmp_path):
    hdr = ["x [nm]", "y [nm]"]
    rows = [[1.0 * i, 2.0 * i] for i in range(30)]
    path = _write(tmp_path / "data.tsv", hdr, rows, sep="\t")
    t = read_table(path)
    assert t.n_rows == 30
    assert guess_mapping(t)["x"] == "x [nm]"


def test_xlsx_roundtrip(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["xnm", "ynm"])
    for i in range(15):
        ws.append([100.0 + i, 200.0 + i])
    p = tmp_path / "data.xlsx"
    wb.save(p)
    t = read_table(str(p))
    assert t.source_format == "xlsx" and t.n_rows == 15
    ds = build_dataset_from_mapping(t, guess_mapping(t))
    assert np.allclose(np.asarray(ds.loc_nm)[:, 0], 100.0 + np.arange(15))


def test_representative_row_indices():
    assert representative_row_indices(5) == [0, 1, 2, 3, 4]
    idx = representative_row_indices(12345)
    assert idx[:10] == list(range(10))          # dense head
    assert 100 in idx and 1000 in idx and 10000 in idx
    assert idx[-1] == 12344                      # last row always present
    assert idx == sorted(set(idx))


def test_unsupported_format_raises(tmp_path):
    p = tmp_path / "x.parquet"
    p.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        read_table(str(p))


# ---------------------------------------------------------------------------
# auto_import — the no-dialog routing entry point
# ---------------------------------------------------------------------------

def test_auto_import_paren_nm(tmp_path):
    # The exact human-friendly form that failed before: "x(nm)", "y(nm)".
    p = _write(tmp_path / "u.csv", ["x(nm)", "y(nm)"],
               [[100.0 + i, 200.0 + i] for i in range(50)])
    ds, amb = auto_import(p)
    assert ds is not None and amb is None
    assert np.allclose(ds.xnm[:3], [100.0, 101.0, 102.0], atol=1e-3)
    assert ds.attr.get("loc_x") is not None and ds.attr.get("loc_x_nm") is None


def test_auto_import_micrometre_converts(tmp_path):
    p = _write(tmp_path / "um.csv", ["x [um]", "y [um]"],
               [[0.1 * i, 0.2 * i] for i in range(50)])
    ds, _ = auto_import(p)
    assert np.ptp(ds.xnm) == pytest.approx(0.1 * 49 * 1000.0, rel=1e-3)   # µm → nm


def test_auto_import_pixels_need_size(tmp_path):
    p = _write(tmp_path / "px.csv", ["x", "y", "lpx", "lpy"],
               [[5.0, 6.0, 0.03, 0.03] for _ in range(50)])
    ds, amb = auto_import(p)
    assert ds is None and amb is not None and amb.needs_pixel_size
    ds2, amb2 = auto_import(p, pixel_size_nm=100.0)
    assert ds2 is not None and amb2 is None
    assert np.allclose(ds2.xnm, 500.0)


def test_auto_import_no_xy_is_ambiguous(tmp_path):
    p = _write(tmp_path / "w.csv", ["foo", "bar"],
               [[1.0 * i, 2.0 * i] for i in range(50)])
    ds, amb = auto_import(p)
    assert ds is None and "x and y" in amb.reason
