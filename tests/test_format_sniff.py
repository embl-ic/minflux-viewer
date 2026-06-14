"""Tests for content-based format detection (recovers mislabelled files)."""

import io

import numpy as np
import pytest

from minflux_viewer.core.format_sniff import MAGIC_FORMATS, sniff_format


def test_npy_magic(tmp_path):
    p = tmp_path / "a.npy"
    np.save(p, np.arange(10.0))
    assert sniff_format(p) == "npy"


def test_npy_mislabelled_as_json(tmp_path):
    # The motivating case: a NumPy array saved with a .json name.
    buf = io.BytesIO()
    np.save(buf, np.zeros((5, 3)))
    p = tmp_path / "data.json"
    p.write_bytes(buf.getvalue())
    assert sniff_format(p) == "npy"          # content wins over the extension
    assert "npy" in MAGIC_FORMATS


def test_mat_v73_hdf5_magic(tmp_path):
    p = tmp_path / "a.mat"
    p.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)
    assert sniff_format(p) == "mat"


def test_mat_classic_header(tmp_path):
    p = tmp_path / "a.mat"
    p.write_bytes(b"MATLAB 5.0 MAT-file, Platform: ..." + b"\x00" * 64)
    assert sniff_format(p) == "mat"


def test_xlsx_zip_magic(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    wb.active.append(["x", "y"])
    p = tmp_path / "a.xlsx"
    wb.save(p)
    assert sniff_format(p) == "xlsx"


def test_tiff_magic(tmp_path):
    p = tmp_path / "a.tif"
    p.write_bytes(b"II*\x00" + b"\x00" * 64)
    assert sniff_format(p) == "tiff"
    p2 = tmp_path / "b.tif"
    p2.write_bytes(b"MM\x00*" + b"\x00" * 64)
    assert sniff_format(p2) == "tiff"


def test_json_structure(tmp_path):
    p = tmp_path / "a.json"
    p.write_text('  \n[{"x": 1, "y": 2}]', encoding="utf-8")
    assert sniff_format(p) == "json"


def test_delimited_text(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("x [nm],y [nm],tid\n10,20,0\n11,21,0\n", encoding="utf-8")
    assert sniff_format(p) == "delimited"


def test_empty_and_unknown(tmp_path):
    empty = tmp_path / "e.dat"
    empty.write_bytes(b"")
    assert sniff_format(empty) is None
    binary = tmp_path / "b.dat"
    binary.write_bytes(bytes(range(256)) * 4)        # control-byte heavy, no magic
    assert sniff_format(binary) is None
    missing = tmp_path / "nope.dat"
    assert sniff_format(missing) is None
