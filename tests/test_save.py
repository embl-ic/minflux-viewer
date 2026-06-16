"""Save processed data: canonical-raw mfx (.mat/.npy/.json) + metadata sidecar."""

import json

import numpy as np
import pytest

from minflux_viewer.core import loader as L
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.core.save import (
    METADATA_JSON_MARKER,
    dataset_to_mfx_array,
    is_metadata_json_file,
    is_metadata_json_payload,
    metadata_sidecar_path,
    save_processed,
)

_LOAD = {"mat": L.load_dataset, "npy": L.load_npy, "json": L.load_json}


def _dataset(n=20):
    rng = np.random.default_rng(0)
    return build_localization_dataset(
        name="synthetic.mat",
        x_nm=rng.uniform(0, 1000, n), y_nm=rng.uniform(0, 1000, n),
        z_nm=rng.uniform(0, 100, n), tid=np.repeat(np.arange(n // 4), 4)[:n],
        tim=np.arange(n, dtype=float), attrs={"efo": rng.uniform(50, 500, n)},
    )


# ---------------------------------------------------------------------------
# canonical mfx reconstruction
# ---------------------------------------------------------------------------
def test_reconstruct_mfx_is_m2410_layout():
    ds = _dataset()
    mfx = dataset_to_mfx_array(ds)
    n = ds.prop.num_loc
    assert mfx.shape == (n,)
    assert mfx["loc"].shape == (n, 3)        # vectors recombined
    assert mfx["itr"].ndim == 1               # flat itr -> m2410 path
    assert "itr" in mfx.dtype.names and "vld" in mfx.dtype.names


def test_filter_and_derived_excluded_from_data():
    ds = _dataset()
    ds.filter_mask = np.ones(ds.prop.num_loc, bool)   # writes attr['ftr']
    ds.derived["den"] = np.zeros(ds.prop.num_loc)
    names = dataset_to_mfx_array(ds).dtype.names
    assert "ftr" not in names and "den" not in names


# ---------------------------------------------------------------------------
# save_processed: no-file (write data + sidecar)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fmt", ["mat", "npy", "json"])
def test_save_writes_data_and_sidecar(tmp_path, fmt):
    ds = _dataset()
    written = save_processed(ds, data_path=tmp_path / "out", fmt=fmt)
    assert len(written) == 2
    data_path, meta_path = written
    assert data_path.suffix == "." + fmt
    assert meta_path == metadata_sidecar_path(data_path)
    assert is_metadata_json_file(meta_path)


@pytest.mark.parametrize("fmt", ["mat", "npy", "json"])
def test_canonical_roundtrip(tmp_path, fmt):
    """Saved canonical raw re-opens through the normal parser with the same
    coordinates, count and dimensionality (no flat-table hack)."""
    ds = _dataset(40)
    data_path, _ = save_processed(ds, data_path=tmp_path / "rt", fmt=fmt)
    re = _LOAD[fmt](str(data_path))
    assert re.prop.num_loc == ds.prop.num_loc
    assert re.prop.num_dim == ds.prop.num_dim
    np.testing.assert_allclose(
        np.asarray(re.attr["loc_x"]) * 1e9, np.asarray(ds.attr["loc_x"]) * 1e9, atol=1e-3
    )


def test_2d_roundtrip_via_zcheck(tmp_path):
    """A 2-D dataset's raw z keeps its small noise on save; the same Z-check on
    load re-detects 2-D — no z mangling required."""
    n = 200
    dt = np.dtype([("vld", "?"), ("tid", "<i4"), ("itr", "<i4"), ("loc", "<f8", (3,))])
    arr = np.zeros(n, dtype=dt)
    arr["vld"] = True
    arr["tid"] = np.repeat(np.arange(n // 4), 4)
    arr["loc"][:, 0] = np.linspace(0, 1e-6, n)
    arr["loc"][:, 1] = np.linspace(0, 1e-6, n)
    arr["loc"][:, 2] = np.random.default_rng(0).uniform(0, 2e-9, n)   # 2 nm noise
    prefs = {"data": {"enforce_min_z_range": True, "min_z_range_nm": 5.0}}

    ds = L.load_from_mfx_array(arr, name="t.mat", prefs=prefs)
    assert ds.prop.num_dim == 2
    data_path, _ = save_processed(ds, data_path=tmp_path / "e", fmt="mat")
    assert L.load_dataset(str(data_path), prefs=prefs).prop.num_dim == 2


# ---------------------------------------------------------------------------
# save_processed: file-backed (metadata only)
# ---------------------------------------------------------------------------
def test_metadata_only_when_file_backed(tmp_path):
    ds = _dataset()
    written = save_processed(ds, data_path=None, metadata_dir=tmp_path)
    assert len(written) == 1
    assert written[0].name.endswith("_metadata.json")
    assert is_metadata_json_file(written[0])


# ---------------------------------------------------------------------------
# metadata recipe content
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fmt", ["mat", "npy", "json"])
def test_reload_restores_rimf_and_filters(tmp_path, fmt):
    """Re-opening a processed file consumes the sidecar: RIMF is restored and
    the saved filter re-applies (mask + specs visible to the filter dialog)."""
    ds = _dataset(40)
    ds.set_rimf(0.78, source="manual (anisotropy plugin)")
    xn = np.asarray(ds.attr["loc_x"]).ravel() * 1e9
    lo, hi = float(np.percentile(xn, 25)), float(np.percentile(xn, 75))
    ds.state["filter_specs"] = [
        {"attribute": "xnm", "mode": "per loc", "lo": lo, "hi": hi,
         "lo_inc": True, "hi_inc": True}
    ]
    expect = int(((xn >= lo) & (xn <= hi)).sum())

    data_path, _ = save_processed(ds, data_path=tmp_path / "p", fmt=fmt)
    re = _LOAD[fmt](str(data_path))
    assert re.cali.RIMF == pytest.approx(0.78)
    assert re.state.get("filter_specs") == ds.state["filter_specs"]
    assert int(np.asarray(re.filter_mask, bool).sum()) == expect


def test_no_sidecar_leaves_dataset_unrestored(tmp_path):
    """A data file without a metadata sidecar must not pin RIMF or filters."""
    ds = _dataset(40)
    data_path, meta_path = save_processed(ds, data_path=tmp_path / "p", fmt="npy")
    meta_path.unlink()                       # remove the sidecar
    re = _LOAD["npy"](str(data_path))
    assert not re.state.get("filter_specs")


def test_metadata_records_recipe(tmp_path):
    ds = _dataset()
    ds.set_rimf(0.85, source="manual (anisotropy plugin)")
    _, meta_path = save_processed(ds, data_path=tmp_path / "m", fmt="npy")
    meta = json.loads(meta_path.read_text())
    assert is_metadata_json_payload(meta) and meta[METADATA_JSON_MARKER] == 1
    assert meta["calibration"]["rimf"] == pytest.approx(0.85)
    assert "iteration_load_mode" in meta["gating"]
    assert "filters" in meta and isinstance(meta["filters"], list)
    # sidecar is a dict — distinct from filter/data JSON, which are lists
    assert isinstance(meta, dict)
