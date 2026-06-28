"""Save/Export redesign: processed snapshot, flat columns, CSV/npz/zarr formats.

Complements tests/test_save.py (which covers the raw-canonical + recipe path).
"""

import json

import numpy as np
import pytest

from minflux_viewer.core import loader as L
from minflux_viewer.core.save import (
    METADATA_JSON_MARKER,
    build_snapshot_table,
    columns_to_structured,
    dataset_to_mfx_array,
    flatten_mfx_array,
    save_processed,
)


def _make_mfx(n=40, seed=0):
    dt = np.dtype([
        ("vld", np.bool_), ("tid", np.int32), ("tim", np.float64),
        ("itr", np.int32), ("efo", np.float64), ("cfr", np.float64),
        ("dcr", np.float64, (2,)), ("loc", np.float64, (3,)),
    ])
    rng = np.random.default_rng(seed)
    a = np.zeros(n, dt)
    a["vld"] = True
    a["tid"] = np.repeat(np.arange(n // 4), 4)[:n]
    a["tim"] = np.linspace(0, 1, n)
    a["itr"] = 3
    a["efo"] = rng.uniform(10, 100, n)
    a["cfr"] = rng.uniform(0.3, 0.9, n)
    a["dcr"] = rng.uniform(0, 1, (n, 2))
    a["loc"] = np.column_stack([
        rng.uniform(0, 1e-6, n), rng.uniform(0, 1e-6, n), rng.uniform(0, 5e-7, n)])
    return a


def _dataset(n=40, seed=0, prefs=None):
    return L.load_from_mfx_array(_make_mfx(n, seed), name="t.mat", folder="/tmp",
                                 prefs=prefs)


# --- flatten / structured round-trip ------------------------------------------
def test_flatten_splits_vectors():
    # split logic: loc (N×3) → loc_x/y/z, dcr (N×2) → dcr_0/dcr_1, scalars kept
    dt = np.dtype([("loc", np.float64, (3,)), ("dcr", np.float64, (2,)),
                   ("efo", np.float64), ("tid", np.int32)])
    arr = np.zeros(5, dt)
    arr["loc"][:, 0] = np.arange(5)
    cols = flatten_mfx_array(arr)
    for k in ("loc_x", "loc_y", "loc_z", "dcr_0", "dcr_1", "efo", "tid"):
        assert k in cols, k
    assert all(np.asarray(v).ndim == 1 for v in cols.values())
    np.testing.assert_array_equal(cols["loc_x"], np.arange(5))


def test_flatten_real_dataset_keeps_coords():
    cols = flatten_mfx_array(dataset_to_mfx_array(_dataset()))
    for k in ("loc_x", "loc_y", "loc_z", "efo", "tid", "itr"):
        assert k in cols, k


def test_columns_to_structured_roundtrip():
    cols = {"xnm": np.arange(5.0), "tid": np.arange(5, dtype=np.int32)}
    arr = columns_to_structured(cols)
    assert arr.shape == (5,)
    assert set(arr.dtype.names) == {"xnm", "tid"}
    np.testing.assert_array_equal(arr["tid"], np.arange(5))


# --- snapshot: RIMF/transform baking + filter ---------------------------------
def test_snapshot_bakes_rimf_into_znm():
    ds = _dataset()
    ds.set_rimf(0.8, source="manual (anisotropy plugin)")
    cols, dropped = build_snapshot_table(ds)
    raw_z_nm = np.asarray(L.mfx_get(ds, "loc_z", itr="last", vld_only=True), float) * 1e9
    np.testing.assert_allclose(cols["znm"], raw_z_nm * 0.8, atol=1e-6)
    assert dropped == 0
    assert {"xnm", "ynm", "znm"} <= set(cols)


def test_snapshot_filter_flag_vs_apply():
    ds = _dataset(40)
    xn = np.asarray(L.mfx_get(ds, "xnm", itr="last", vld_only=True), float)
    lo, hi = float(np.percentile(xn, 25)), float(np.percentile(xn, 75))
    ds.state["filter_specs"] = [{"attribute": "xnm", "mode": "per loc",
                                 "lo": lo, "hi": hi, "lo_inc": True, "hi_inc": True}]
    keep = int(((xn >= lo) & (xn <= hi)).sum())

    flag, dropped_flag = build_snapshot_table(ds, filter_mode="flag")
    assert "ftr" in flag and dropped_flag == 0
    assert int(np.asarray(flag["ftr"], bool).sum()) == keep

    applied, dropped = build_snapshot_table(ds, filter_mode="apply")
    assert "ftr" not in applied
    assert applied["xnm"].size == keep
    assert dropped == xn.size - keep


def test_snapshot_include_derived_optional():
    ds = _dataset()
    # den is read by mfx_get from the last-valid materialized store (ds.attr)
    ds.attr["den"] = np.linspace(0, 1, ds.prop.num_loc)
    assert "den" not in build_snapshot_table(ds, include_derived=False)[0]
    assert "den" in build_snapshot_table(ds, include_derived=True)[0]


# --- format writers ------------------------------------------------------------
@pytest.mark.parametrize("fmt", ["csv", "npz", "zarr", "mat", "npy", "json"])
def test_snapshot_writes_each_format(tmp_path, fmt):
    ds = _dataset()
    written = save_processed(ds, data_path=tmp_path / "snap", fmt=fmt,
                             content="snapshot")
    data_path = written[0]
    assert data_path.exists()
    # sidecar written by default (include recipe)
    assert any(p.name.endswith("_metadata.json") for p in written)


def test_raw_csv_is_flattened(tmp_path):
    ds = _dataset()
    data_path, _ = save_processed(ds, data_path=tmp_path / "raw", fmt="csv",
                                  content="raw")
    header = data_path.read_text().splitlines()[0].split(",")
    assert "loc_x" in header and "loc_y" in header and "loc_z" in header
    assert "efo" in header and "tid" in header


# --- snapshot recipe + round-trip ---------------------------------------------
def test_snapshot_recipe_pins_rimf_one(tmp_path):
    ds = _dataset()
    ds.set_rimf(0.78, source="manual (anisotropy plugin)")
    _, meta_path = save_processed(ds, data_path=tmp_path / "s", fmt="csv",
                                  content="snapshot")
    meta = json.loads(meta_path.read_text())
    assert meta[METADATA_JSON_MARKER] == 1
    assert meta["content"] == "snapshot"
    assert meta["calibration"]["rimf"] == 1.0          # baked → not re-applied
    assert meta["calibration"]["baked_rimf"] == pytest.approx(0.78)
    assert meta["filters"] == []


def test_snapshot_csv_roundtrips_without_double_rimf(tmp_path):
    ds = _dataset()
    ds.set_rimf(0.8, source="manual (anisotropy plugin)")
    cols, _ = build_snapshot_table(ds)
    data_path, _ = save_processed(ds, data_path=tmp_path / "rt", fmt="csv",
                                  content="snapshot")
    re = L.load_csv(str(data_path))
    assert re.cali.RIMF == pytest.approx(1.0)          # baked snapshot not re-scaled
    assert re.prop.num_loc == cols["xnm"].size
    np.testing.assert_allclose(
        np.sort(np.asarray(re.loc_nm)[:, 0]), np.sort(cols["xnm"]), atol=1e-2)
