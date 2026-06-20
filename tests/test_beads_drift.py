"""MSR bead-drift extraction (nm conversion, median re-zero, time zeroing)."""

import numpy as np

from minflux_viewer.plugins.msr_reader.beads_drift import extract_bead_drift

_DT = np.dtype([("gri", "<i4"), ("xyz", "<f8", (3,)), ("tim", "<f8"), ("str", "<f8")])


def _points():
    p = np.zeros(5, _DT)
    p["gri"] = [10, 10, 10, 20, 20]
    p["xyz"] = [[1e-9, 2e-9, 3e-9], [2e-9, 4e-9, 6e-9], [3e-9, 6e-9, 9e-9],
                [0.0, 0.0, 0.0], [1e-9, 0.0, 0.0]]
    p["tim"] = [10.0, 5.0, 0.0, 0.0, 1.0]      # bead 10 out of time order
    return p


_PBG = {"10": {"gri": 10, "name": "R1"},
        "20": {"gri": 20, "name": "R2"},
        "30": {"gri": 30, "name": "R3"}}     # R3 has no points


def test_extract_used_beads_only():
    beads = extract_bead_drift(_points(), _PBG, ["R1", "R2"])
    assert [b["gri"] for b in beads] == [10, 20]
    assert [b["rid"] for b in beads] == ["R1", "R2"]


def test_nm_median_rezero_and_time_zeroing():
    b = extract_bead_drift(_points(), _PBG, ["R1"])[0]
    assert b["n"] == 3
    # time sorted ascending and zeroed to start
    np.testing.assert_allclose(b["tim_s"], [0.0, 5.0, 10.0])
    # xyz in nm, re-zeroed to per-axis median, reordered by time
    np.testing.assert_allclose(b["xyz_nm"], [[1, 2, 3], [0, 0, 0], [-1, -2, -3]])
    np.testing.assert_allclose(np.median(b["xyz_nm"], axis=0), [0, 0, 0], atol=1e-9)


def test_unused_bead_with_no_points_skipped():
    beads = extract_bead_drift(_points(), _PBG, ["R1", "R2", "R3"])
    assert all(b["rid"] != "R3" for b in beads)   # R3 maps to gri 30, no data


def test_empty_used_falls_back_to_all_known_beads():
    beads = extract_bead_drift(_points(), _PBG, [])
    assert sorted(b["gri"] for b in beads) == [10, 20]   # R3 (gri 30) has no points


def test_missing_fields_returns_empty():
    bad = np.zeros(3, np.dtype([("foo", "<f8")]))
    assert extract_bead_drift(bad, _PBG, ["R1"]) == []
