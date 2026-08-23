"""Regression tests for the zarr-v2 shim that keeps the ``.msr`` path working.

``zarr-python`` 3.x cannot represent the structured dtype Abberior uses for the
embedded ``mfx`` array (subarray fields ``loc``/``lnc``/``dcr``, and a nested
structured ``itr`` in m2205 files). ``minflux_viewer.msr.zarr2`` implements the
small zarr-v2 subset the ``.msr`` path needs, independently of ``zarr-python``.

These tests pin the two properties that matter: the shim reproduces exactly what
the real library reads, and what it writes stays interoperable.
"""

import json

import numpy as np
import pytest

from minflux_viewer.msr import zarr2

# --- the dtype that breaks zarr-python 3 -----------------------------------
MFX_DTYPE = np.dtype([
    ("vld", np.bool_), ("tid", np.uint32), ("tim", np.float64), ("itr", np.int32),
    ("efo", np.float32), ("cfr", np.float16), ("eco", np.uint32),
    ("dcr", np.float16, (2,)), ("loc", np.float64, (3,)), ("lnc", np.float64, (3,)),
])


def _mfx(n=500, seed=0):
    rng = np.random.default_rng(seed)
    a = np.zeros(n, MFX_DTYPE)
    a["vld"] = rng.random(n) > 0.2
    a["tid"] = np.repeat(np.arange(n // 5 + 1), 5)[:n]
    a["tim"] = np.linspace(0, 10, n)
    a["itr"] = np.tile(np.arange(5), n // 5 + 1)[:n]
    a["efo"] = rng.uniform(10, 100, n)
    a["cfr"] = rng.uniform(0, 1, n)
    a["eco"] = rng.integers(1, 500, n)
    a["dcr"] = rng.uniform(0, 1, (n, 2))
    a["loc"] = rng.uniform(0, 2e-6, (n, 3))
    a["lnc"] = rng.uniform(0, 2e-6, (n, 3))
    return a


# --- dtype JSON round trip --------------------------------------------------
def test_subarray_dtype_survives_the_json_round_trip():
    spec = zarr2.dtype_to_json(MFX_DTYPE)
    assert zarr2.dtype_from_json(spec) == MFX_DTYPE
    # the exact shape zarr v2 uses on disk, and the shape zarr-python 3 rejects
    assert ["loc", "<f8", [3]] in spec
    assert ["dcr", "<f2", [2]] in spec


def test_nested_structured_field_survives():
    """m2205 files nest a structured ``itr`` inside the record."""
    nested = np.dtype([("tid", np.uint32),
                       ("itr", np.dtype([("loc", np.float64, (3,)),
                                         ("eco", np.uint32)]), (4,))])
    assert zarr2.dtype_from_json(zarr2.dtype_to_json(nested)) == nested


def test_malformed_dtype_spec_raises():
    with pytest.raises(zarr2.ZarrV2Error):
        zarr2.dtype_from_json({"not": "a dtype"})


# --- store round trip -------------------------------------------------------
def test_write_then_read_is_byte_identical():
    src = _mfx()
    store: dict[str, bytes] = {}
    root = zarr2.open(store, mode="w")
    root["mfx"] = src
    back = np.asarray(zarr2.open(store, mode="r")["mfx"])
    assert back.dtype == src.dtype
    assert back.tobytes() == src.tobytes()


def test_chunked_array_reassembles_across_chunk_boundaries():
    src = _mfx(n=1000)
    store: dict[str, bytes] = {}
    zarr2.open(store, mode="w").create_array("mfx", src, chunk=128)
    arr = zarr2.open(store, mode="r")["mfx"]
    assert arr.nchunks == 8
    assert np.asarray(arr).tobytes() == src.tobytes()


def test_attrs_and_nested_groups():
    store: dict[str, bytes] = {}
    root = zarr2.open(store, mode="w")
    root["mfx"] = _mfx(50)
    root["mfx"].attrs["did"] = "abc-123"
    root["grd/mbm/points"] = np.zeros(4, dtype=[("gri", np.int32),
                                                ("xyz", np.float64, (3,))])
    root.require_group("mbm").attrs["used"] = [1, 2, 3]

    re_read = zarr2.open(store, mode="r")
    assert re_read["mfx"].attrs["did"] == "abc-123"
    assert re_read["grd/mbm/points"].shape == (4,)
    assert re_read["mbm"].attrs.get("used") == [1, 2, 3]
    assert "mfx" in re_read and "grd/mbm/points" in re_read
    assert "nope" not in re_read


def test_visititems_walks_every_node():
    store: dict[str, bytes] = {}
    root = zarr2.open(store, mode="w")
    root["mfx"] = _mfx(10)
    root["grd/mbm/points"] = np.zeros(2, dtype=[("gri", np.int32)])
    seen: list[str] = []
    zarr2.open(store, mode="r").visititems(lambda path, _obj: seen.append(path))
    assert sorted(seen) == ["grd", "grd/mbm", "grd/mbm/points", "mfx"]


def test_missing_chunk_reads_as_fill_value():
    src = _mfx(n=400)
    store: dict[str, bytes] = {}
    zarr2.open(store, mode="w").create_array("mfx", src, chunk=100)
    del store["mfx/2"]                       # simulate a sparse / unwritten chunk
    back = np.asarray(zarr2.open(store, mode="r")["mfx"])
    assert back.shape == src.shape
    assert np.array_equal(back[:200], src[:200])
    assert np.all(back[200:300]["tid"] == 0)


def test_directory_store_round_trip(tmp_path):
    src = _mfx(200)
    root = zarr2.open(tmp_path / "s.zarr", mode="w")
    root["mfx"] = src
    assert (tmp_path / "s.zarr" / "mfx" / ".zarray").is_file()
    back = np.asarray(zarr2.open(tmp_path / "s.zarr", mode="r")["mfx"])
    assert back.tobytes() == src.tobytes()


def test_non_zarr_store_is_rejected():
    with pytest.raises(zarr2.ZarrV2Error):
        zarr2.open({"random": b"bytes"}, mode="r")


def test_zarr_v3_store_is_rejected_not_misread():
    """No silent fallback: a v3 array must raise rather than be misparsed."""
    store = {
        ".zgroup": json.dumps({"zarr_format": 2}).encode(),
        "mfx/.zarray": json.dumps({
            "zarr_format": 3, "shape": [4], "chunks": [4],
            "dtype": "<f8", "compressor": None, "filters": None,
            "order": "C", "fill_value": 0,
        }).encode(),
    }
    with pytest.raises(zarr2.ZarrV2Error):
        zarr2.open(store, mode="r")["mfx"]


# --- parity with the real library, while it is still installed --------------
@pytest.mark.parametrize("n", [1, 333, 4096])
def test_matches_zarr_python_when_available(n):
    """The shim must reproduce zarr-python byte for byte on the same store."""
    zarr = pytest.importorskip("zarr")
    if not zarr.__version__.startswith("2."):
        pytest.skip("parity reference requires zarr-python 2.x")

    src = _mfx(n, seed=n)
    store: dict[str, bytes] = {}
    ref_root = zarr.open(zarr.storage.KVStore(store), mode="w")
    ref_root["mfx"] = src
    ref_root["mfx"].attrs["did"] = "parity"

    ours = np.asarray(zarr2.open(store, mode="r")["mfx"])
    assert ours.dtype == src.dtype
    assert ours.tobytes() == src.tobytes()
    assert zarr2.open(store, mode="r")["mfx"].attrs["did"] == "parity"


def test_our_output_is_readable_by_zarr_python():
    """What we write must stay interoperable with the wider zarr ecosystem."""
    zarr = pytest.importorskip("zarr")
    if not zarr.__version__.startswith("2."):
        pytest.skip("interop check requires zarr-python 2.x")

    src = _mfx(777)
    store: dict[str, bytes] = {}
    root = zarr2.open(store, mode="w")
    root["mfx"] = src
    root["mfx"].attrs["did"] = "interop"

    ref = zarr.open(zarr.storage.KVStore(store), mode="r")
    assert ref["mfx"][:].tobytes() == src.tobytes()
    assert dict(ref["mfx"].attrs)["did"] == "interop"


# --- the real thing ---------------------------------------------------------
def test_reads_a_real_msr_file():
    """End-to-end on real acquisition data, when the sample set is present."""
    from pathlib import Path

    sample = Path(r"D:/Workspace/Microscopes/MINFLUX/sample data/2_3C_measurement.msr")
    if not sample.is_file():
        pytest.skip("sample .msr not available on this machine")

    from minflux_viewer.msr.mfxdta import extract_zarr_store, read_obf_mfxdta_stacks

    stacks = read_obf_mfxdta_stacks(sample)
    assert stacks, "expected at least one MFXDTA stack"
    store = extract_zarr_store(stacks[0][2])
    mfx = np.asarray(zarr2.open(store, mode="r")["mfx"])

    assert mfx.dtype.names is not None
    # the fields zarr-python 3 cannot represent
    assert mfx.dtype["loc"].shape == (3,)
    assert mfx.dtype["dcr"].shape == (2,)
    assert len(mfx) > 0
    assert np.isfinite(mfx["loc"]).any()
