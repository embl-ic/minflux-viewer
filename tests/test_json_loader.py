import json

import numpy as np


def test_abberior_minflux_json_records_load(tmp_path):
    from minflux_viewer.core.loader import load_json

    path = tmp_path / "abberior_minflux.json"
    records = [
        {
            "vld": True,
            "tim": 1.0,
            "tid": 7,
            "itr": 0,
            "loc": [1e-9, 2e-9, 0.0],
            "lnc": [1e-9, 2e-9, 0.0],
            "efo": 100.0,
            "cfr": 0.25,
            "dcr": [0.4, 0.6],
        },
        {
            "vld": True,
            "tim": 1.2,
            "tid": 7,
            "itr": 1,
            "loc": [2e-9, 3e-9, 0.0],
            "lnc": [2e-9, 3e-9, 0.0],
            "efo": 120.0,
            "cfr": 0.5,
            "dcr": [0.45, 0.55],
        },
        {
            "vld": False,
            "tim": 2.0,
            "tid": 8,
            "itr": 0,
            "loc": [9e-9, 9e-9, 0.0],
            "efo": 999.0,
        },
    ]
    path.write_text(json.dumps(records), encoding="utf-8")

    ds = load_json(path)

    # The two valid records (tid=7, itr 0->1) are two iterations of ONE
    # localization; the loader materializes the last valid iteration (itr=1)
    # and drops the invalid record (tid=8). So one localization in one trace.
    assert ds.prop.num_loc == 1
    assert ds.prop.num_traces == 1
    assert ds.metadata.get("raw_num_loc") == 3      # 3 raw records read
    assert ds.metadata.get("valid_num_loc") == 1
    assert "loc_x" in ds.attr
    assert "efo" in ds.attr
    assert "dcr" in ds.attr
    assert "dcr_1" in ds.attr
    np.testing.assert_allclose(ds.loc_nm[:, :2], [[2.0, 3.0]])   # last iteration
    np.testing.assert_array_equal(ds.attr["tid"], [7])
    np.testing.assert_allclose(ds.attr["dcr"], [0.45])


def test_json_filter_discriminator_rejects_minflux_records():
    from minflux_viewer.ui.filter_dialog import is_filter_json_payload

    assert is_filter_json_payload([
        {"apply": True, "attribute": "efo", "value_as": "per loc", "min": 1, "max": 10}
    ])
    assert not is_filter_json_payload([
        {"vld": True, "tid": 1, "loc": [1e-9, 2e-9, 0.0], "efo": 10.0}
    ])
