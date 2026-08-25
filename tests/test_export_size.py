from __future__ import annotations

from pathlib import Path

import numpy as np

from minflux_viewer.core.export_size import (
    estimate_export_sizes,
    format_file_size,
    text_export_warning,
)
from minflux_viewer.core.save import (
    _write_csv,
    _write_json,
    _write_mat,
    _write_npy,
    flatten_mfx_array,
)
from minflux_viewer.core.spreadsheet_loader import (
    guess_mapping,
    guess_units,
    read_table_preview,
)


def _mfx(n: int = 4000) -> np.ndarray:
    dtype = np.dtype([
        ("loc", "<f8", (3,)),
        ("dcr", "<f2", (2,)),
        ("eco", "<u4"),
        ("efo", "<f4"),
        ("itr", "<i4"),
        ("tid", "<u4"),
        ("tim", "<f8"),
        ("vld", "?"),
    ])
    result = np.zeros(n, dtype=dtype)
    index = np.arange(n)
    result["loc"][:, 0] = np.sin(index / 20.0) * 1e-6
    result["loc"][:, 1] = np.cos(index / 23.0) * 1e-6
    result["loc"][:, 2] = (index % 17) * 1e-9
    result["dcr"][:, 0] = (index % 100) / 100.0
    result["dcr"][:, 1] = 1.0 - result["dcr"][:, 0]
    result["eco"] = index % 1000
    result["efo"] = 40_000.0 + index % 10_000
    result["itr"] = index % 10
    result["tid"] = index // 20
    result["tim"] = index * 0.00123
    result["vld"] = index % 7 != 0
    return result


def test_estimates_match_real_writers_for_complete_sample(tmp_path):
    array = _mfx()
    estimates = estimate_export_sizes([("mfx", array)])
    paths = {
        "npy": tmp_path / "data.npy",
        "mat": tmp_path / "data.mat",
        "json": tmp_path / "data.json",
        "csv": tmp_path / "data.csv",
    }
    _write_npy(paths["npy"], array)
    _write_mat(paths["mat"], array)
    _write_json(paths["json"], array)
    _write_csv(paths["csv"], flatten_mfx_array(array))

    assert estimates["npy"].bytes == paths["npy"].stat().st_size
    for fmt in ("mat", "json", "csv"):
        actual = paths[fmt].stat().st_size
        assert abs(estimates[fmt].bytes - actual) / actual < 0.01


def test_text_warning_and_human_size():
    estimates = estimate_export_sizes([("mfx", _mfx(100))])
    assert text_export_warning(estimates, ["json"], threshold_bytes=1) is not None
    assert text_export_warning(estimates, ["mat"], threshold_bytes=1) is None
    assert format_file_size(1 << 30) == "1.00 GiB"


def test_large_csv_preview_is_bounded_and_spans_file(tmp_path):
    path = tmp_path / "large.csv"
    rows = ["loc_x,loc_y,loc_z,itr,tid,tim,vld\n"]
    for index in range(5000):
        # Variable textual width exercises the length-bias correction in the
        # row-count estimate.
        rows.append(
            f"{index * 1e-9:.17g},{-index * 2e-9:.17g},0,{index % 10},"
            f"{index // 20},{index * 0.00123:.17g},{int(index % 7 != 0)}\n"
        )
    path.write_text("".join(rows), encoding="utf-8")

    table = read_table_preview(path, max_rows=100, full_read_limit=0)
    assert table.preview_only
    assert table.n_rows_is_estimate
    assert max(column.values.size for column in table.columns) <= 100
    assert abs(table.n_rows - 5000) / 5000 < 0.05
    assert table.sample_row_indices[0] == 0
    assert table.sample_row_indices[-1] >= 4900
    mapping = guess_mapping(table, use_values=True)
    assert mapping["x"] == "loc_x" and mapping["y"] == "loc_y"
    assert guess_units(table, mapping)["x"] == "m"


def test_roi_classifier_rejects_top_level_array_without_full_read(tmp_path, monkeypatch):
    from minflux_viewer.core.roi import is_roi_json_file

    path = tmp_path / "records.json"
    path.write_text('[{"loc":[0,0,0]}]', encoding="utf-8")

    def forbidden_read_text(self: Path, *args, **kwargs):
        raise AssertionError("top-level array classification must remain bounded")

    monkeypatch.setattr(Path, "read_text", forbidden_read_text)
    assert not is_roi_json_file(path)
