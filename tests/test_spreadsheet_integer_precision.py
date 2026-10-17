"""Generic mappings retain exact IDs, not only canonical CSV imports."""

import numpy as np
import pytest

from minflux_viewer.core.spreadsheet_loader import (
    build_dataset_from_mapping,
    read_table,
    read_table_preview,
)


def test_remapped_delimited_ids_and_extra_integer_columns_stay_exact(tmp_path):
    path = tmp_path / "generic.csv"
    ids = [2**53 + 1, 2**53 + 2, 2**63 - 1]
    extras = [2**64 - 3, 2**64 - 2, 2**64 - 1]
    path.write_text("east,north,track,source_id\n" + "".join(
        f"{i}.25,{i}.5,{tid},{extra}\n" for i, (tid, extra) in enumerate(zip(ids, extras))),
        encoding="utf-8")
    table = read_table(path)
    ds = build_dataset_from_mapping(table, {"x": "east", "y": "north", "id": "track"},
                                    units={"x": "nm", "y": "nm"})
    np.testing.assert_array_equal(ds.attr["tid"], ids)
    np.testing.assert_array_equal(ds.attr["source_id"], np.array(extras, dtype=np.uint64))
    preview = read_table_preview(path, full_read_limit=1)
    assert preview.by_name("track").values.dtype.kind == "i"
    assert set(preview.by_name("track").values.tolist()) == set(ids)


@pytest.mark.parametrize("spelling", [str(2**53 + 1), f"{2**53 + 1}.0", "9.007199254740993e15"])
def test_exact_decimal_spellings_do_not_pass_through_binary_float(tmp_path, spelling):
    path = tmp_path / "generic.csv"
    path.write_text(f"x,y,track\n1,2,{spelling}\n2,3,{spelling}\n", encoding="utf-8")
    table = read_table(path)
    ds = build_dataset_from_mapping(table, {"x": "x", "y": "y", "id": "track"})
    assert int(ds.attr["tid"][0]) == 2**53 + 1


@pytest.mark.parametrize("cells", ["9007199254740993\n2,3,\n", "9007199254740993\n2,3,1.5\n",
                                  "18446744073709551615\n2,3,3\n"])
def test_invalid_or_inexact_trace_identifiers_are_refused(tmp_path, cells):
    path = tmp_path / "generic.csv"
    path.write_text("x,y,track\n1,2," + cells, encoding="utf-8")
    with pytest.raises(ValueError, match="Trace IDs"):
        build_dataset_from_mapping(read_table(path), {"x": "x", "y": "y", "id": "track"})


@pytest.mark.parametrize("as_text", [True, False])
def test_excel_large_ids_require_text_cells(tmp_path, as_text):
    from openpyxl import Workbook

    path = tmp_path / "generic.xlsx"
    book = Workbook()
    sheet = book.active
    sheet.append(["x", "y", "track"])
    sheet.append([1, 2, str(2**53 + 1) if as_text else 2**53 + 1])
    book.save(path)
    book.close()
    table = read_table(path)
    mapping = {"x": "x", "y": "y", "id": "track"}
    if as_text:
        ds = build_dataset_from_mapping(table, mapping)
        assert int(ds.attr["tid"][0]) == 2**53 + 1
    else:
        with pytest.raises(ValueError, match="precision"):
            build_dataset_from_mapping(table, mapping)
