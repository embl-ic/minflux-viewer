"""Decoding the early 'obf / mfxdta' .msr format (embedded zarr in a uint8 stack)."""

import struct
from pathlib import Path

import numpy as np
import pytest

from minflux_viewer.msr.mfxdta import (
    MFXDTA_MAGIC,
    SOURCE_FORMAT,
    container_version,
    extract_zarr_store,
    parse_mfxdta_entries,
    read_mfxdta_mfx,
    stack_payload_is_mfxdta,
    unpack_zarr_store_to_dir,
)


def _pack_mfxdta(entries, *, version=2, ts=1654082502):
    """Serialise ``[(name, flag, data|None), …]`` into an MFXDTA blob.

    Mirrors the on-disk container layout so the parser can be tested round-trip.
    """
    head = MFXDTA_MAGIC + b"\x00" + struct.pack("<I", version) + struct.pack("<I", ts)
    head = head + b"\x00" * (147 - len(head))          # zero padding before table
    body = b""
    for name, flag, data in entries:
        nb = name.encode("latin1")
        body += struct.pack("<Q", len(nb)) + nb + bytes([flag])
        if flag != 1:                                   # leaf carries a payload
            data = data or b""
            body += struct.pack("<Q", len(data)) + data
    return head + body


def _zarr_store_for(arr):
    """Build an in-memory zarr v2 store (dict) holding ``arr`` as 'mfx'."""
    import zarr

    store: dict = {}
    root = zarr.open(zarr.storage.KVStore(store), mode="w")
    root.create_dataset("mfx", data=arr, chunks=(max(1, arr.shape[0] // 2),))
    root.create_group("grd")                            # mirrors real files
    return store


def _sample_mfx():
    dt = np.dtype([("vld", "?"), ("tid", "<i4"), ("loc", "<f8", (3,))])
    arr = np.zeros(6, dtype=dt)
    arr["vld"] = True
    arr["tid"] = [1, 1, 2, 2, 3, 3]
    arr["loc"][:, 0] = np.arange(6) * 1e-7
    arr["loc"][:, 1] = np.arange(6) * 2e-7
    return arr


def test_magic_detection():
    blob = _pack_mfxdta([("zarr", 1, None)])
    assert stack_payload_is_mfxdta(np.frombuffer(blob, dtype=np.uint8))
    assert not stack_payload_is_mfxdta(np.frombuffer(b"OMAS_BF\n....", dtype=np.uint8))
    # wrong dtype is rejected even with the right leading bytes by value
    assert not stack_payload_is_mfxdta(np.arange(20, dtype=np.int32))


def test_header_fields():
    blob = _pack_mfxdta([("zarr", 1, None)], version=2, ts=1654082502)
    assert container_version(blob) == 2


def test_entry_table_parses_groups_and_leaves():
    entries = [
        ("sync", 1, None),
        ("sync\\mfx", 1, None),
        ("sync\\mfx\\0", 0, b""),               # manifest placeholder (len 0)
        ("zarr", 1, None),
        ("zarr\\.zgroup", 0, b'{"zarr_format":2}'),
        ("zarr\\mfx\\.zarray", 0, b"ZARRAY-BYTES"),
        ("zarr\\mfx\\0", 0, b"CHUNK0"),
    ]
    store = parse_mfxdta_entries(_pack_mfxdta(entries))
    assert store["zarr\\.zgroup"] == b'{"zarr_format":2}'
    assert store["zarr\\mfx\\.zarray"] == b"ZARRAY-BYTES"
    assert store["zarr\\mfx\\0"] == b"CHUNK0"
    # group nodes carry no payload
    assert "sync" not in store


def test_extract_keeps_only_zarr_subtree_with_slashes():
    entries = [
        ("sync\\mfx\\0", 0, b""),
        ("zarr\\.zgroup", 0, b'{"zarr_format":2}'),
        ("zarr\\mfx\\.zarray", 0, b"x"),
        ("zarr\\mfx\\0", 0, b"y"),
    ]
    store = extract_zarr_store(_pack_mfxdta(entries))
    assert set(store) == {".zgroup", "mfx/.zarray", "mfx/0"}     # 'sync\' dropped, '\'→'/'


def test_extract_handles_forward_slash_keys_v3():
    # Modern (container v3) files use '/' separators instead of '\'.
    entries = [
        ("sync/mfx/0", 0, b""),
        ("zarr/.zgroup", 0, b'{"zarr_format":2}'),
        ("zarr/mfx/.zarray", 0, b"x"),
        ("zarr/mfx/0", 0, b"y"),
        ("zarr/grd/mbm/points/0", 0, b"beads"),
    ]
    store = extract_zarr_store(_pack_mfxdta(entries, version=3))
    assert set(store) == {".zgroup", "mfx/.zarray", "mfx/0", "grd/mbm/points/0"}


def test_full_roundtrip_to_mfx_array(tmp_path):
    arr = _sample_mfx()
    zstore = _zarr_store_for(arr)
    # pack the zarr store into an MFXDTA blob (backslash keys, as real files use)
    entries = [("zarr\\" + k.replace("/", "\\"), 0, v) for k, v in zstore.items()]
    blob = _pack_mfxdta(entries)

    # (a) decode straight to the structured array
    decoded = read_mfxdta_mfx(blob)
    assert decoded.dtype.names == ("vld", "tid", "loc")
    np.testing.assert_array_equal(decoded["tid"], arr["tid"])
    np.testing.assert_allclose(decoded["loc"], arr["loc"])

    # (b) unpack to disk and open as a normal zarr DirectoryStore
    import zarr

    dest = unpack_zarr_store_to_dir(extract_zarr_store(blob), tmp_path / "zarr")
    mfx = zarr.open(str(dest), mode="r")["mfx"][:]
    np.testing.assert_array_equal(mfx["tid"], arr["tid"])


def test_extract_rejects_blob_without_mfx():
    blob = _pack_mfxdta([("zarr\\.zgroup", 0, b'{"zarr_format":2}')])
    with pytest.raises(ValueError):
        extract_zarr_store(blob)


# --- Optional end-to-end smoke against a real sample file (skipped in CI) ---

_SAMPLE = Path(r"D:\Workspace\Microscopes\MINFLUX\sample data"
               r"\1_sample_A_1-100_seq_3D_ori_exc_5.msr")


@pytest.mark.skipif(not _SAMPLE.exists(), reason="local sample .msr not present")
def test_specpy_free_path_reads_early_file(tmp_path):
    """The pure-Python msr-reader path recovers the same localizations."""
    pytest.importorskip("msr_reader")
    from minflux_viewer.msr.mfxdta import read_obf_mfxdta_stacks
    from minflux_viewer.msr.msr_parser import GeneralMSRParser
    from minflux_viewer.msr import state as MFSTATE

    stacks = read_obf_mfxdta_stacks(_SAMPLE)
    assert len(stacks) == 1
    _idx, _desc, blob = stacks[0]
    assert read_mfxdta_mfx(blob).shape[0] == 152434

    # full specpy-free parse via the public parser
    res = GeneralMSRParser().parse(str(_SAMPLE), str(tmp_path), lambda *_: None)
    assert res["mode"] == "modern"
    assert res["source_format"] == "obf / mfxdta"
    assert sum(v.shape[0] for v in MFSTATE.mfx_map.values()) == 152434
