"""Numeric CSV formatting must never promote integer identifiers to floats."""

import csv
import io

import numpy as np
import pytest

from minflux_viewer.core.save import _write_csv


def test_mixed_numeric_columns_preserve_integer_limits(tmp_path):
    columns = dict(
        tid=np.array([2**53 + 1, 2**64 - 1], dtype=np.uint64),
        signed=np.array([-2**63, 2**63 - 1], dtype=np.int64),
        loc_x=np.array([1.25, np.nextafter(0.1, 1)]),
        vld=np.array([True, False]))
    path = tmp_path / "exact.csv"
    _write_csv(path, columns)
    rows = list(csv.DictReader(io.StringIO(path.read_text())))
    for i, row in enumerate(rows):
        assert int(row["tid"]) == int(columns["tid"][i])
        assert int(row["signed"]) == int(columns["signed"][i])
        assert float(row["loc_x"]) == columns["loc_x"][i]
        assert int(row["vld"]) == int(columns["vld"][i])


@pytest.mark.parametrize("bad", [np.arange(3), np.zeros((2, 1))])
def test_malformed_columns_do_not_open_destination(tmp_path, bad):
    path = tmp_path / "old.csv"
    path.write_text("previous export")
    with pytest.raises(ValueError, match="align|one-dimensional"):
        _write_csv(path, {"id": np.arange(2), "bad": bad})
    assert path.read_text() == "previous export"


def test_chunk_boundary_and_ordinary_numeric_bytes(tmp_path):
    n = 100_002
    columns = {"id": np.arange(n, dtype=np.uint64) + 2**53,
               "value": np.full(n, 1.25)}
    path = tmp_path / "chunks.csv"
    _write_csv(path, columns)
    lines = path.read_text().splitlines()
    for i in (0, 99_999, 100_000, n - 1):
        assert lines[i + 1] == f"{int(columns['id'][i])},1.25"
    _write_csv(path, {"id": np.arange(2), "value": np.array([1.25, 0.1])})
    assert path.read_bytes() == b"id,value\n0,1.25\n1,0.10000000000000001\n"


def test_large_signed_trace_ids_roundtrip_canonical_reader(tmp_path):
    from minflux_viewer.core.loader_format_readers import load_csv

    path = tmp_path / "canonical.csv"
    ids = np.array([2**53 + 1, 2**53 + 3], dtype=np.int64)
    columns = dict(tid=ids, itr=np.zeros(2, dtype=np.int32), vld=np.ones(2, dtype=bool),
                   loc_x=np.array([1e-9, 2e-9]), loc_y=np.zeros(2), loc_z=np.zeros(2))
    _write_csv(path, columns)
    ds = load_csv(path)
    np.testing.assert_array_equal(ds.mfx_raw["tid"], ids)


def test_numeric_header_uses_csv_quoting(tmp_path):
    path = tmp_path / "quoted.csv"
    _write_csv(path, {"a,b": np.arange(2), "x": np.arange(2.)})
    assert next(csv.reader(io.StringIO(path.read_text()))) == ["a,b", "x"]
