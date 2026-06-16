"""Saving processed data to .mat / .npy / .json / .csv (+ metadata sidecar)."""

import csv
import json

import numpy as np
import pytest

from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.core.save import (
    METADATA_JSON_MARKER,
    build_export_table,
    save_dataset,
)


def _dataset(n=20):
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1000, n)
    y = rng.uniform(0, 1000, n)
    z = rng.uniform(0, 100, n)
    tid = np.repeat(np.arange(n // 4), 4)[:n]
    attrs = {"efo": rng.uniform(50, 500, n), "cfr": rng.uniform(0, 1, n)}
    return build_localization_dataset(
        name="synthetic.mat", x_nm=x, y_nm=y, z_nm=z, tid=tid,
        tim=np.arange(n, dtype=float), attrs=attrs,
    )


def test_table_has_coords_and_ftr():
    ds = _dataset()
    table = build_export_table(ds, include_ftr=True, include_derived=True)
    assert {"xnm", "ynm", "znm"} <= set(table)
    assert "ftr" in table and table["ftr"].dtype == np.uint8
    n = ds.prop.num_loc
    assert all(len(col) == n for col in table.values())


def test_ftr_and_derived_toggles():
    ds = _dataset()
    assert "ftr" not in build_export_table(ds, include_ftr=False)
    derived_off = build_export_table(ds, include_derived=False)
    derived_on = build_export_table(ds, include_derived=True)
    # derived columns (e.g. idx) appear only when requested
    assert set(derived_on) >= set(derived_off)


@pytest.mark.parametrize("fmt", ["mat", "npy", "json", "csv"])
def test_roundtrip_each_format(tmp_path, fmt):
    ds = _dataset()
    written = save_dataset(ds, tmp_path / "out", fmt=fmt, include_metadata=True)
    assert len(written) == 2                       # data + metadata sidecar
    data_path, meta_path = written
    assert data_path.suffix == "." + fmt
    n = ds.prop.num_loc

    if fmt == "npy":
        arr = np.load(data_path)
        assert arr.shape[0] == n and "xnm" in arr.dtype.names
    elif fmt == "csv":
        rows = list(csv.reader(data_path.read_text().splitlines()))
        assert rows[0][:3] == ["xnm", "ynm", "znm"] and len(rows) == n + 1
    elif fmt == "json":
        recs = json.loads(data_path.read_text())
        assert len(recs) == n and "xnm" in recs[0]
    else:  # mat
        from scipy.io import loadmat
        m = loadmat(str(data_path))
        assert "xnm" in m and m["xnm"].size == n

    # metadata sidecar is a dict tagged with the marker (distinct from filter/
    # data JSON, which are lists)
    meta = json.loads(meta_path.read_text())
    assert isinstance(meta, dict) and meta[METADATA_JSON_MARKER] == 1
    assert meta["num_loc"] == n
    assert "rimf" in meta and "export_options" in meta


def test_metadata_can_be_skipped(tmp_path):
    ds = _dataset()
    written = save_dataset(ds, tmp_path / "out", fmt="csv", include_metadata=False)
    assert len(written) == 1 and written[0].suffix == ".csv"


def test_2d_dataset_exports_flat_z(tmp_path):
    """A 2-D (threshold-flattened) dataset must export z=0 and re-open as 2-D —
    not leak the raw mfx_raw z noise, which used to read back as 3-D."""
    from minflux_viewer.core.loader import load_from_mfx_array, load_dataset

    n = 200
    dt = np.dtype([("vld", "?"), ("tid", "<i4"), ("itr", "<i4"), ("loc", "<f8", (3,))])
    arr = np.zeros(n, dtype=dt)
    arr["vld"] = True
    arr["tid"] = np.repeat(np.arange(n // 4), 4)
    arr["loc"][:, 0] = np.linspace(0, 1e-6, n)
    arr["loc"][:, 1] = np.linspace(0, 1e-6, n)
    arr["loc"][:, 2] = np.random.default_rng(0).uniform(0, 2e-9, n)   # 2 nm noise
    prefs = {"data": {"enforce_min_z_range": True, "min_z_range_nm": 5.0}}

    ds = load_from_mfx_array(arr, name="t.mat", prefs=prefs)
    assert ds.prop.num_dim == 2
    assert np.all(build_export_table(ds)["znm"] == 0)

    written = save_dataset(ds, tmp_path / "e", fmt="mat", include_metadata=False)
    # re-opens as 2-D even without the threshold preference
    assert load_dataset(written[0], prefs=None).prop.num_dim == 2


@pytest.mark.parametrize("fmt", ["mat", "npy", "json", "csv"])
def test_export_reimport_roundtrip(tmp_path, fmt):
    """A saved dataset re-opens with correct coordinates (regression: a flat
    export must NOT be misread by the mfx parser)."""
    from minflux_viewer.core import loader as L

    ds = _dataset(40)
    written = save_dataset(ds, tmp_path / "rt", fmt=fmt, include_metadata=False)
    loadfn = {"mat": L.load_dataset, "npy": L.load_npy,
              "json": L.load_json, "csv": L.load_csv}[fmt]
    re = loadfn(written[0])

    assert re.prop.num_loc == ds.prop.num_loc
    loc_x = re.attr.get("loc_x")
    assert loc_x is not None and np.asarray(loc_x).shape == (ds.prop.num_loc,)
    # coordinates round-trip (nm), no double RIMF correction
    np.testing.assert_allclose(
        np.asarray(re.attr["loc_x"]) * 1e9, np.asarray(ds.attr["loc_x"]) * 1e9, atol=1e-3
    )
    np.testing.assert_allclose(
        np.asarray(re.attr["loc_z"]) * 1e9, np.asarray(ds.attr["loc_z"]) * 1e9, atol=1e-3
    )
    assert re.cali.RIMF == 1.0
