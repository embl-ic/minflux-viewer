"""Stage B iteration browsing: selector labels, loc_id grouping, and the
derived-attribute broadcast contract (iter_load="all" must reproduce the
iter_load="last" derived values for every iteration selection)."""

import numpy as np
import pytest
import scipy.io as sio

from minflux_viewer.core.dataset import AttrStore
from minflux_viewer.core.iteration import iteration_labels, parse_iteration_label
from minflux_viewer.core.loader import (
    _build_mfx_raw_m2205_unwrapped,
    _derived_last_from_raw,
    _raw_loc_id,
    attr_matches_last_valid,
    attr_matches_selection,
    attr_values_1d,
    load_dataset,
    mfx_get,
)

DERIVED = ["dt", "dst", "spd", "siz", "dur", "len", "tim_trace"]


# ---------------------------------------------------------------------------
# Selector labels
# ---------------------------------------------------------------------------

def test_iteration_labels_fixed_order():
    # last on top, pooled modes next, then individual iterations counting down.
    assert iteration_labels(4) == [
        "last (4th)", "all [flatten]", "all [stacked]", "3rd", "2nd", "1st",
    ]


def test_iteration_labels_hidden_for_single_iteration():
    assert iteration_labels(1) == []
    assert iteration_labels(0) == []


def test_parse_iteration_label():
    assert parse_iteration_label("last (4th)") == ("last", "single")
    assert parse_iteration_label("all [flatten]") == ("all", "flatten")
    assert parse_iteration_label("all [stacked]") == ("all", "stacked")
    assert parse_iteration_label("3rd") == (2, "single")   # 1-based -> 0-based
    assert parse_iteration_label("") == ("last", "single")


# ---------------------------------------------------------------------------
# loc_id grouping
# ---------------------------------------------------------------------------

def _m2205_raw_store(n_loc=6, n_itr=3, seed=0):
    rng = np.random.default_rng(seed)
    merged = {
        "itr": np.tile(np.arange(n_itr), (n_loc, 1)),
        "loc": rng.normal(size=(n_loc, n_itr, 3)) * 1e-6,
        "tid": np.array([1, 1, 1, 2, 2, 3]),
        "tim": np.arange(n_loc, dtype=float),
        "vld": np.ones(n_loc, dtype=bool),
        "efo": rng.random((n_loc, n_itr)),
    }
    return _build_mfx_raw_m2205_unwrapped(merged, n_loc, n_itr)


def test_loc_id_m2205_grid_is_row_div_n_itr():
    raw = _m2205_raw_store()
    assert np.array_equal(_raw_loc_id(raw), np.repeat(np.arange(6), 3))


def test_loc_id_m2410_event_stream():
    # Two interleaved traces: tid 1 refines 0,1,2 then loops at itr 2 (each
    # loop row is its own localization); tid 2 rows (failed probes at itr 0)
    # interleave in time.
    raw = AttrStore({
        "itr": np.array([0, 0, 1, 2, 0, 2, 2]),
        "tid": np.array([1, 2, 1, 1, 2, 1, 1]),
        "vld": np.array([1, 0, 1, 1, 0, 1, 1], dtype=bool),
        "tim": np.arange(7, dtype=float),
    })
    loc_id = _raw_loc_id(raw)
    # tid 1 ladder rows 0,2,3 form one group; loop rows 5 and 6 are separate
    # groups; each failed probe (rows 1, 4) is its own group.
    assert np.array_equal(loc_id, np.array([0, 3, 0, 0, 4, 1, 2]))


def test_derived_last_from_raw_m2205():
    raw = _m2205_raw_store()
    out = _derived_last_from_raw(raw, 3, None)
    for key in DERIVED:
        assert key in out
        assert out.get(key).shape[0] == 6


# ---------------------------------------------------------------------------
# End-to-end broadcast contract on an m2205-unwrapped .mat file
# ---------------------------------------------------------------------------

@pytest.fixture
def m2205_mat(tmp_path):
    n_loc, n_itr = 8, 3
    rng = np.random.default_rng(42)
    path = tmp_path / "m2205_unwrapped.mat"
    sio.savemat(str(path), {
        "itr": np.tile(np.arange(n_itr, dtype=np.int32), (n_loc, 1)),
        "loc": rng.normal(size=(n_loc, n_itr, 3)) * 1e-6,
        "tid": np.array([1, 1, 1, 2, 2, 2, 3, 3], dtype=np.int32),
        "tim": np.linspace(0.0, 1.0, n_loc),
        "vld": np.ones(n_loc, dtype=bool),
        "efo": rng.random((n_loc, n_itr)) * 1e5,
    })
    return path


def test_broadcast_matches_last_load(m2205_mat):
    ds_last = load_dataset(m2205_mat, prefs={"data": {"iter_load": "last"}})
    ds_all = load_dataset(m2205_mat, prefs={"data": {"iter_load": "all"}})

    assert attr_matches_last_valid(ds_last)
    # m2205 "all" loads keep one row per localization (per-iteration fields
    # stay 2-D), so ds.attr remains last-valid-aligned here — unlike m2410.
    # The broadcast source is then ds.attr itself; derived_last is only built
    # when the materialization is NOT last-valid.
    assert attr_matches_last_valid(ds_all)
    assert len(ds_all.components.derived_last) == 0

    for attr in DERIVED:
        ref = np.asarray(ds_last.attr.get(attr), dtype=float).ravel()
        got = mfx_get(ds_all, attr, itr="last", vld_only=True)
        assert got is not None
        np.testing.assert_allclose(np.asarray(got, dtype=float), ref)


def test_broadcast_duplicates_across_iterations(m2205_mat):
    # In the fixed m2205 grid every localization has all iterations, so any
    # single-iteration view must return the identical per-loc derived values.
    ds_all = load_dataset(m2205_mat, prefs={"data": {"iter_load": "all"}})
    ref = mfx_get(ds_all, "dt", itr="last", vld_only=True)
    for k in range(3):
        got = mfx_get(ds_all, "dt", itr=k, vld_only=True)
        np.testing.assert_allclose(got, ref)
    flat = mfx_get(ds_all, "dt", itr="all", vld_only=True)
    assert flat.shape[0] == 3 * ref.shape[0]
    np.testing.assert_allclose(flat, np.repeat(ref, 3))


def test_raw_attrs_not_broadcast(m2205_mat):
    # Genuine per-iteration attributes still come from the raw store rows.
    ds_all = load_dataset(m2205_mat, prefs={"data": {"iter_load": "all"}})
    efo_0 = mfx_get(ds_all, "efo", itr=0, vld_only=True)
    efo_last = mfx_get(ds_all, "efo", itr="last", vld_only=True)
    assert efo_0.shape == efo_last.shape
    assert not np.allclose(efo_0, efo_last)


# ---------------------------------------------------------------------------
# End-to-end broadcast contract on an m2410 flat .mat file
# (variable iterations per localization: ds.attr row count depends on
# iter_load, so a "last" selection must come from derived_last)
# ---------------------------------------------------------------------------

@pytest.fixture
def m2410_mat(tmp_path):
    rng = np.random.default_rng(7)
    # Event stream: tid 1 refines 0,1,2 then loops twice at itr 2 (all valid);
    # tid 2 is a failed probe; tid 3 refines 0,1,2 (all valid).
    itr = np.array([0, 1, 2, 2, 2, 0, 0, 1, 2], dtype=np.int32)
    tid = np.array([1, 1, 1, 1, 1, 2, 3, 3, 3], dtype=np.int32)
    vld = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1], dtype=bool)
    n = itr.size
    path = tmp_path / "m2410_flat.mat"
    sio.savemat(str(path), {
        "itr": itr,
        "tid": tid,
        "vld": vld,
        "tim": np.linspace(0.0, 1.0, n),
        "loc": rng.normal(size=(n, 3)) * 1e-6,
        "efo": rng.random(n) * 1e5,
    })
    return path


def test_m2410_broadcast_matches_last_load(m2410_mat):
    ds_last = load_dataset(m2410_mat, prefs={"data": {"iter_load": "last"}})
    ds_all = load_dataset(m2410_mat, prefs={"data": {"iter_load": "all"}})

    # last-valid rows: itr==2 & vld -> 3 from tid 1, 1 from tid 3.
    assert ds_last.prop.num_loc == 4
    assert ds_all.prop.num_loc == 8          # all valid rows
    assert attr_matches_last_valid(ds_last)
    assert not attr_matches_last_valid(ds_all)

    for attr in DERIVED:
        ref = np.asarray(ds_last.attr.get(attr), dtype=float).ravel()
        got = mfx_get(ds_all, attr, itr="last", vld_only=True)
        assert got is not None, attr
        np.testing.assert_allclose(np.asarray(got, dtype=float), ref, err_msg=attr)


# ---------------------------------------------------------------------------
# Stage C: invalid-inclusive loads (only_valid_locs=False)
# ---------------------------------------------------------------------------

@pytest.fixture
def m2410_invalid_mat(tmp_path):
    rng = np.random.default_rng(11)
    # tid 1 refines 0,1,2 (valid); tid 2 reaches itr 2 but fails validation
    # there (vld False on its final row); tid 3 valid ladder 0,1,2.
    itr = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int32)
    tid = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int32)
    vld = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1], dtype=bool)
    n = itr.size
    path = tmp_path / "m2410_invalid.mat"
    sio.savemat(str(path), {
        "itr": itr,
        "tid": tid,
        "vld": vld,
        "tim": np.linspace(0.0, 1.0, n),
        "loc": rng.normal(size=(n, 3)) * 1e-6,
        "efo": rng.random(n) * 1e5,
    })
    return path


def test_invalid_inclusive_load_metadata_and_selection(m2410_invalid_mat):
    p_valid = {"data": {"iter_load": "last"}}
    p_inval = {"data": {"iter_load": "last", "only_valid_locs": False}}
    ds_v = load_dataset(m2410_invalid_mat, prefs=p_valid)
    ds_i = load_dataset(m2410_invalid_mat, prefs=p_inval)

    assert ds_v.metadata["includes_invalid"] is False
    assert ds_i.metadata["includes_invalid"] is True
    assert ds_v.prop.num_loc == 2          # valid rows at itr 2 (tid 1, 3)
    assert ds_i.prop.num_loc == 3          # plus tid 2's invalid final row

    # default-path gating: rows align with the loaded validity selection
    assert attr_matches_selection(ds_v, itr="last", vld_only=True)
    assert not attr_matches_selection(ds_v, itr="last", vld_only=False)
    assert attr_matches_selection(ds_i, itr="last", vld_only=False)
    assert not attr_matches_selection(ds_i, itr="last", vld_only=True)
    assert not attr_matches_last_valid(ds_i)


def test_invalid_inclusive_load_derived_broadcast(m2410_invalid_mat):
    p_valid = {"data": {"iter_load": "last"}}
    p_inval = {"data": {"iter_load": "last", "only_valid_locs": False}}
    ds_v = load_dataset(m2410_invalid_mat, prefs=p_valid)
    ds_i = load_dataset(m2410_invalid_mat, prefs=p_inval)

    # derived_last must be populated for the invalid-inclusive load
    for key in DERIVED:
        assert key in ds_i.components.derived_last, key

    vld_col = np.asarray(ds_i.attr.get("vld"), dtype=bool).ravel()
    for attr in DERIVED:
        ref = np.asarray(ds_v.attr.get(attr), dtype=float).ravel()
        got = mfx_get(ds_i, attr, itr="last", vld_only=False)
        assert got is not None, attr
        got = np.asarray(got, dtype=float)
        assert got.shape[0] == ds_i.prop.num_loc
        # valid rows carry the last-valid values; the invalid row is NaN
        np.testing.assert_allclose(got[vld_col], ref, err_msg=attr)
        assert np.isnan(got[~vld_col]).all(), attr
        # attr_values_1d (window default path) agrees with the broadcast
        via_attr = attr_values_1d(ds_i, attr)
        np.testing.assert_allclose(
            np.asarray(via_attr, dtype=float), got, equal_nan=True, err_msg=attr,
        )


def test_invalid_inclusive_raw_attrs_via_attr_values_1d(m2410_invalid_mat):
    p_inval = {"data": {"iter_load": "last", "only_valid_locs": False}}
    ds_i = load_dataset(m2410_invalid_mat, prefs=p_inval)
    efo = attr_values_1d(ds_i, "efo")
    np.testing.assert_allclose(
        np.asarray(efo, dtype=float),
        np.asarray(ds_i.attr.get("efo"), dtype=float).ravel(),
    )


def test_m2205_all_row_selection_still_last_valid(m2205_mat):
    # m2205 "all" loads keep one row per localization; attr rows align with
    # the last+valid selection even though iteration_load_mode == "all".
    ds_all = load_dataset(m2205_mat, prefs={"data": {"iter_load": "all"}})
    assert attr_matches_selection(ds_all, itr="last", vld_only=True)
    assert not attr_matches_selection(ds_all, itr="all", vld_only=True)


def test_m2410_flatten_inherits_group_values(m2410_mat):
    ds_all = load_dataset(m2410_mat, prefs={"data": {"iter_load": "all"}})
    dt_last = mfx_get(ds_all, "dt", itr="last", vld_only=True)
    dt_flat = mfx_get(ds_all, "dt", itr="all", vld_only=True)
    assert dt_flat.shape[0] == 8
    # tid 1 ladder rows (itr 0,1) inherit the value of their group's anchor
    # (the first itr-2 row); valid flatten rows of anchored groups are finite.
    # Flatten row order: tid1 rows 0..4, then tid3 rows 5..7 (probe is invalid).
    np.testing.assert_allclose(dt_flat[:3], dt_last[0])   # ladder + anchor
    np.testing.assert_allclose(dt_flat[3], dt_last[1])    # loop row
    np.testing.assert_allclose(dt_flat[4], dt_last[2])    # loop row
    np.testing.assert_allclose(dt_flat[5:8], dt_last[3])  # tid 3 ladder + anchor
