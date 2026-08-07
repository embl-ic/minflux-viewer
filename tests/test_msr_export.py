"""MSR export must use the canonical flat m2410 save representation."""

import numpy as np
import pytest

from minflux_viewer.core import loader as L
from minflux_viewer.msr.export import export_arrays


def _legacy_mfx(n_loc=3, n_itr=3):
    itr_dtype = np.dtype([
        ("itr", np.int32),
        ("loc", np.float64, (3,)),
        ("efo", np.float64),
        ("cfr", np.float64),
    ])
    dtype = np.dtype([
        ("itr", itr_dtype, (n_itr,)),
        ("vld", np.bool_),
        ("tid", np.int32),
        ("tim", np.float64),
    ])
    arr = np.zeros(n_loc, dtype=dtype)
    arr["vld"] = True
    arr["tid"] = np.arange(n_loc, dtype=np.int32)
    arr["tim"] = np.arange(n_loc, dtype=float)
    arr["itr"]["itr"] = np.arange(n_itr, dtype=np.int32)
    base = np.arange(n_loc * n_itr, dtype=float).reshape(n_loc, n_itr)
    arr["itr"]["loc"][:, :, 0] = base * 1e-9
    arr["itr"]["loc"][:, :, 1] = (base + 10) * 1e-9
    arr["itr"]["loc"][:, :, 2] = (base + 20) * 1e-9
    arr["itr"]["efo"] = 100.0 + base
    arr["itr"]["cfr"] = 0.5
    return arr


def _mbm():
    dtype = np.dtype([
        ("gri", np.int32),
        ("xyz", np.float64, (3,)),
        ("tim", np.float64),
        ("str", np.float64),
    ])
    arr = np.zeros(2, dtype=dtype)
    arr["gri"] = [1, 2]
    arr["xyz"] = [[1e-6, 2e-6, 3e-6], [4e-6, 5e-6, 6e-6]]
    return arr


@pytest.mark.parametrize("fmt", ["mat", "npy", "npz", "json", "csv", "zarr"])
def test_legacy_mfx_is_flat_and_all_fields_survive_each_export(tmp_path, fmt):
    written = export_arrays(
        str(tmp_path), "channel", [fmt], _legacy_mfx(), _mbm(),
        mbm_meta={"points_by_gri": {1: {"name": "R1"}}, "used": ["R1"]},
    )
    mfx_path = tmp_path / f"channel_mfx.{fmt}"
    mbm_path = tmp_path / f"channel_mbm.{fmt}"
    assert str(mfx_path) in written
    assert str(mbm_path) in written

    if fmt == "mat":
        from scipy.io import loadmat

        payload = loadmat(str(mfx_path), squeeze_me=False, struct_as_record=False)
        assert "mfx" in payload
        assert "itr_itr" not in payload
        assert "itr_loc" not in payload
        fields = payload["mfx"][0, 0]._fieldnames
        assert "itr" in fields
        assert "loc" in fields
    elif fmt == "npy":
        arr = np.load(mfx_path, allow_pickle=False)
        assert arr.dtype.names and "itr" in arr.dtype.names and "loc" in arr.dtype.names
        assert arr["itr"].ndim == 1
    elif fmt == "json":
        loaded = L.load_json(str(mfx_path))
        assert len(loaded.mfx_raw["itr"]) == 9
        assert {"efo", "cfr", "tid"} <= set(loaded.attr.keys())
    elif fmt == "csv":
        header = mfx_path.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert {"loc_x", "loc_y", "loc_z", "itr", "efo", "cfr", "tid"} <= set(header)
        loaded = L.load_csv(str(mfx_path))
        assert loaded.prop.num_loc == 9
    elif fmt == "npz":
        with np.load(mfx_path, allow_pickle=False) as payload:
            assert {"loc_x", "loc_y", "loc_z", "itr", "efo", "cfr", "tid"} <= set(payload.files)
            assert "itr_itr" not in payload.files
    elif fmt == "zarr":
        import zarr

        root = zarr.open(str(mfx_path), mode="r")
        assert {"loc_x", "loc_y", "loc_z", "itr", "efo", "cfr", "tid"} <= set(root.array_keys())
        assert "itr_itr" not in set(root.array_keys())
        loaded = L.load_zarr(str(mfx_path))
        assert len(loaded.mfx_raw["itr"]) == 9
        assert {"efo", "cfr", "tid"} <= set(loaded.attr.keys())
        with pytest.raises(ValueError, match="canonical MINFLUX Zarr export"):
            L.load_zarr(str(mbm_path))


def test_msr_export_roundtrips_canonical_mfx_and_mbm(tmp_path):
    written = export_arrays(
        str(tmp_path), "channel", ["msr"], _legacy_mfx(), _mbm(),
        mbm_meta={"points_by_gri": {1: {"name": "R1"}}, "used": ["R1"]},
    )
    path = tmp_path / "channel.msr"
    assert [str(path)] == written

    from minflux_viewer.msr.msr_parser import GeneralMSRParser

    parsed = GeneralMSRParser().parse(str(path), log=lambda *_args: None)
    recovered = parsed["mfx_map"]["channel.msr"]
    assert recovered.dtype.names and "itr" in recovered.dtype.names
    assert recovered["itr"].ndim == 1
    assert recovered.shape[0] == 9
    assert parsed["mbm_map"]["channel.msr"].shape[0] == 2


def test_export_rejects_selection_that_cannot_be_canonicalized(tmp_path):
    selected = np.zeros(3, dtype=np.dtype([("itr", np.int32), ("vld", np.bool_)]))
    with pytest.raises(ValueError, match="required 'loc'"):
        export_arrays(str(tmp_path), "channel", ["mat"], selected, None)


def test_export_refuses_to_overwrite_existing_output(tmp_path):
    export_arrays(str(tmp_path), "channel", ["npy"], _legacy_mfx(), None)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        export_arrays(str(tmp_path), "channel", ["npy"], _legacy_mfx(), None)
