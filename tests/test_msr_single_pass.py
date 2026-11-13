"""Opening a ``.msr``: one read of the file, parallel decoding, and acquisition
ROIs found whether or not Imspector compressed its stacks.

Newer Imspector versions zlib-compress every OBF stack. The small JSON records
of the embedded zarr store -- the acquisition ROIs in its root ``.zattrs`` among
them -- are then not plain text in the file, and the whole-file byte scan that
used to read them found none: confocal matching and the acquisition-ROI overlay
silently had nothing to work with. The same scan, run three times per open,
read every byte of a multi-hundred-MB file again each time, which over a network
share dominated the open.
"""

from __future__ import annotations

import json
import struct
import zlib

import numpy as np
import pytest

from minflux_viewer.msr import writer as W

_DID_A = "00000000-0000-0000-0000-00000000000a"
_DID_B = "00000000-0000-0000-0000-00000000000b"


# --------------------------------------------------------------------- fixtures
def _mfx(n=400, seed=0):
    from minflux_viewer.core.dataset import build_localization_dataset
    from minflux_viewer.core.save import dataset_to_mfx_array
    from minflux_viewer.core.simulate import simulate_localizations

    coords, tid, attrs = simulate_localizations("npc", n_points=n, seed=seed)
    ds = build_localization_dataset(
        name="sim", x_nm=coords[:, 0], y_nm=coords[:, 1], z_nm=coords[:, 2],
        tid=tid, attrs=attrs, source_version="simulation")
    return dataset_to_mfx_array(ds)


def _roi_attrs(did, *boxes) -> bytes:
    """A root ``.zattrs`` as Imspector writes it: corners in metres."""
    rois = [{"corners": [[x0, y0], [x1, y1]], "dimensionality": 0,
             "linked_cfg": "acquiring mfx", "linked_dta": did, "name": "",
             "type": "ROI"} for x0, y0, x1, y1 in boxes]
    return json.dumps({"rois": rois, "version": "2.1"}).encode()


def _payload(did, *boxes, lead: bytes = b"", attrs_last=False, n=400, seed=0,
             root_attrs: bytes | None = None) -> bytes:
    """An MFXDTA payload whose zarr root ``.zattrs`` holds *boxes*.

    *lead* puts a record of that many bytes before the root attributes;
    *attrs_last* puts them after every array chunk instead of first (the
    order Imspector uses); *root_attrs* replaces the root record's bytes."""
    store = W.build_channel_store(W.MsrChannel(mfx=_mfx(n, seed), name=did), did)
    items = list(store.items())
    if boxes or root_attrs is not None:
        root = (".zattrs", _roi_attrs(did, *boxes) if root_attrs is None else root_attrs)
        items = items + [root] if attrs_last else [root] + items
    if lead:
        items = [("lead/blob", lead)] + items
    return W.pack_mfxdta(dict(items))


def _obf(stacks, *, description: str = "", declared=None) -> bytes:
    """OBF bytes of ``[(payload, compressed)]`` in the minimal writer's layout,
    with each stack zlib-compressed on request -- what Imspector does.
    *declared* overrides the stack sizes the headers declare."""
    head = struct.calcsize("<10sLQL")
    descr = description.encode("utf-8")

    def block(index, payload, compressed, nxt):
        data = zlib.compress(payload) if compressed else payload
        size = len(payload) if declared is None else declared[index]
        out = bytearray()
        out += struct.pack("<16sLL", W._MAGIC_STACK, 1, 1)
        out += struct.pack("<15L", size, *([0] * 14))
        out += struct.pack("<15d", *([0.0] * 15)) * 2
        out += struct.pack("<LLLLLQQQ", W._DT_UINT8, 1 if compressed else 0, 0, 0, 0,
                           0, len(data), nxt)
        out += data
        out += W._v1_footer(rank=1)
        return bytes(out)

    positions, pos = [], head + len(descr)
    for i, (payload, compressed) in enumerate(stacks):
        positions.append(pos)
        pos += len(block(i, payload, compressed, 0))
    out = bytearray(struct.pack("<10sLQL", W._MAGIC_FILE, 1,
                                positions[0] if stacks else 0, len(descr)))
    out += descr
    for i, (payload, compressed) in enumerate(stacks):
        out += block(i, payload, compressed,
                     positions[i + 1] if i + 1 < len(stacks) else 0)
    return bytes(out)


def _write(tmp_path, name, stacks, **kwargs):
    path = tmp_path / name
    path.write_bytes(_obf(stacks, **kwargs))
    return path


def _lease(threads=4):
    """A lease of exactly *threads*, independent of this machine's budget."""
    from minflux_viewer.core.execution import ThreadBudget

    return ThreadBudget(threads, per_lease=threads).acquire()


# ------------------------------------------------------------ acquisition ROIs
def test_rois_are_found_in_compressed_stacks_as_in_plain_ones(tmp_path):
    from minflux_viewer.core.acquisition_metadata import read_acquisition_rois

    a = _payload(_DID_A, (1e-6, 2e-6, 2e-6, 4e-6))
    b = _payload(_DID_B, (5e-6, 5e-6, 6e-6, 7e-6), (5.5e-6, 5e-6, 7e-6, 7e-6))
    plain = _write(tmp_path, "plain.msr", [(a, False), (b, False)])
    packed = _write(tmp_path, "packed.msr", [(a, True), (b, True)])

    # The reported failure: in the compressed file the JSON is not plain text,
    # so a byte scan of the file cannot find it.
    assert b'"rois"' in plain.read_bytes()
    assert b'"rois"' not in packed.read_bytes()

    expected = read_acquisition_rois(plain)
    assert [r.did for r in expected] == [_DID_A, _DID_B, _DID_B]
    assert expected[0].bounds == pytest.approx((1e-6, 2e-6, 1e-6, 2e-6))
    assert read_acquisition_rois(packed) == expected


def test_repeated_roi_blocks_are_not_counted_twice(tmp_path):
    """A run and its aggregated companion share a did and repeat the ROI list."""
    from minflux_viewer.core.acquisition_metadata import read_acquisition_rois

    box = (1e-6, 2e-6, 2e-6, 4e-6)
    path = _write(tmp_path, "dup.msr", [(_payload(_DID_A, box), True),
                                         (_payload(_DID_A, box, seed=1), False)])
    assert len(read_acquisition_rois(path)) == 1


@pytest.mark.parametrize("compressed", [False, True])
def test_root_attributes_far_into_the_payload_are_still_found(tmp_path, compressed):
    """Only the start of each payload is read -- but a record stored late is
    found by growing the prefix, never silently missed."""
    from minflux_viewer.core.acquisition_metadata import read_acquisition_rois

    rng = np.random.default_rng(0)
    box = (1e-6, 2e-6, 2e-6, 4e-6)
    lead = rng.integers(0, 256, 300_000, dtype=np.uint8).tobytes()   # incompressible
    late = _payload(_DID_A, box, lead=lead)
    last = _payload(_DID_B, box, attrs_last=True, n=4000)
    assert len(last) > 64 * 1024                    # beyond the first prefix
    path = _write(tmp_path, "late.msr", [(late, compressed), (last, compressed)])
    assert [r.did for r in read_acquisition_rois(path)] == [_DID_A, _DID_B]


@pytest.mark.parametrize("compressed", [False, True])
def test_a_payload_without_root_attributes_has_no_rois(tmp_path, compressed):
    """This application's own files carry no root ``.zattrs``: the walk reaches
    the end of each payload, skipping record data rather than holding it."""
    from minflux_viewer.core.acquisition_metadata import read_acquisition_rois
    from minflux_viewer.core.mfxdta_container import read_root_attrs

    payload = _payload(_DID_A, n=4000)
    path = _write(tmp_path, "none.msr", [(payload, compressed)])
    assert read_acquisition_rois(path) == []
    from msr_reader import OBFFile
    with OBFFile(str(path)) as obf:
        header = obf.stack_headers[0]
        assert read_root_attrs(obf.fd, header.data_position, header.data_length,
                               header.compressed) is None


def test_a_corrupt_stack_loses_only_its_own_rois(tmp_path):
    from minflux_viewer.core.acquisition_metadata import read_acquisition_rois

    good = _payload(_DID_B, (5e-6, 5e-6, 6e-6, 7e-6))
    data = bytearray(_obf([(_payload(_DID_A, (1e-6, 2e-6, 2e-6, 4e-6)), True),
                           (good, False)]))
    # Damage the first stack's zlib stream just after its 2-byte header.
    first_data = struct.calcsize("<10sLQL") + struct.calcsize("<16sLL") \
        + 15 * 4 + 2 * 15 * 8 + struct.calcsize("<LLLLLQQQ")
    data[first_data + 2:first_data + 12] = b"\xff" * 10
    path = tmp_path / "corrupt.msr"
    path.write_bytes(bytes(data))
    assert [r.did for r in read_acquisition_rois(path)] == [_DID_B]


# --------------------------------------------------- damaged optional metadata
_VALID_ROI = {"corners": [[1e-6, 2e-6], [2e-6, 4e-6]], "linked_dta": _DID_A, "type": "ROI"}
_DEEP_JSON = b"[" * 100_000 + b"]" * 100_000            # too deep for the decoder


@pytest.mark.parametrize("root_attrs, n_rois", [
    pytest.param(b'{"rois": 42}', 0, id="number"),
    pytest.param(b'{"rois": "corners"}', 0, id="string"),
    pytest.param(b'{"rois": {"corners": [[0, 0], [1, 1]]}}', 0, id="object"),
    pytest.param(b'{"rois": [42, null, {"corners": 5}, {"corners": [[1], "x"]}]}', 0,
                 id="bad-entries"),
    pytest.param(b"{not json", 0, id="not-json"),
    pytest.param(_DEEP_JSON, 0, id="too-deep"),
    pytest.param(json.dumps({"rois": [42, "x", _VALID_ROI]}).encode(), 1,
                 id="good-one-kept"),
])
def test_malformed_rois_never_cost_the_localizations(tmp_path, root_attrs, n_rois):
    """ROIs are optional metadata read during the parse: a malformed value
    must not reject the dataset beside it (``{"rois": 42}`` used to raise
    TypeError out of the whole parse)."""
    from minflux_viewer.core.acquisition_metadata import read_acquisition_rois
    from minflux_viewer.msr.msr_parser import parse_general

    path = _write(tmp_path, "bad.msr", [(_payload(_DID_A, root_attrs=root_attrs), True)])
    parsed = parse_general(str(path), str(tmp_path), log=lambda *_a: None)
    assert parsed["mode"] == "modern" and len(parsed["datasets"]) == 1
    assert parsed["datasets"][0]["_mfx"].tobytes() == _mfx().tobytes()
    assert len(parsed["acquisition_rois"]) == n_rois
    assert read_acquisition_rois(path) == parsed["acquisition_rois"]


def test_unreadably_nested_footer_tags_are_absent_not_fatal():
    """The stack scan and tag reading now run inside the parse, so a footer
    tag too deep for the JSON decoder must read as no tag."""
    from types import SimpleNamespace

    from minflux_viewer.core.obf_image_source import _minflux_tag
    from minflux_viewer.msr.mfxdta import minflux_stack_tags_from_obf

    footer = SimpleNamespace(tag_dictionary={"minflux": _DEEP_JSON.decode()})
    assert _minflux_tag(footer) == {}
    assert minflux_stack_tags_from_obf(SimpleNamespace(stack_footers=[footer])) == {}


def _record(name: str, data: bytes, length: int | None = None) -> bytes:
    key = name.encode()
    return (struct.pack("<Q", len(key)) + key + b"\x00"
            + struct.pack("<Q", len(data) if length is None else length) + data)


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("length", [2 ** 64 - 1, 10_000_000, 250_000])
def test_an_impossible_root_record_length_is_absent(tmp_path, monkeypatch, compressed, length):
    """A damaged length must neither size a read or inflate call (2**64 - 1
    used to raise OverflowError out of the ROI reader) nor hide the next,
    healthy stack's ROI. More payload than the first prefetch follows the
    record, so the length is really put to the stream."""
    import zlib as zlib_module

    from minflux_viewer.core import mfxdta_container
    from minflux_viewer.core.acquisition_metadata import read_acquisition_rois

    asked: list[int] = []
    real = zlib_module.decompressobj

    class _Recording:
        def __init__(self):
            self._z = real()

        def decompress(self, data, max_length=0):
            asked.append(max_length)
            return self._z.decompress(data, max_length)

        def flush(self):
            return self._z.flush()

        unconsumed_tail = property(lambda self: self._z.unconsumed_tail)
        eof = property(lambda self: self._z.eof)

    monkeypatch.setattr(mfxdta_container.zlib, "decompressobj", _Recording)
    damaged = (b"MFXDTA\x00" + struct.pack("<II", 3, 0) + bytes(17)
               + _record("zarr/.zattrs", b"{}", length) + bytes(200_000))
    healthy = _payload(_DID_B, (5e-6, 5e-6, 6e-6, 7e-6))
    path = _write(tmp_path, "damaged.msr", [(damaged, compressed), (healthy, compressed)])
    assert [r.did for r in read_acquisition_rois(path)] == [_DID_B]
    assert all(0 < size <= mfxdta_container._MAX_STEP for size in asked)
    assert bool(asked) is compressed


# ----------------------------------------------------------- stacks and labels
def test_parallel_inflate_returns_exactly_the_sequential_result(tmp_path):
    from minflux_viewer.msr.mfxdta import read_obf_mfxdta_stacks

    stacks = [(_payload(_DID_A, seed=k, n=300 + 50 * k), k % 2 == 0) for k in range(5)]
    path = _write(tmp_path, "five.msr", stacks + [(b"not an MFXDTA payload....", True)])
    sequential = read_obf_mfxdta_stacks(path)
    assert [i for i, _d, _b in sequential] == [0, 1, 2, 3, 4]
    assert [b for _i, _d, b in sequential] == [p for p, _c in stacks]
    assert read_obf_mfxdta_stacks(path, lease=_lease(4)) == sequential


def test_a_payload_unlike_its_declared_size_is_an_error(tmp_path):
    """What msr-reader's reshape to the declared shape used to reject."""
    from minflux_viewer.msr.mfxdta import read_obf_mfxdta_stacks

    payload = _payload(_DID_A)
    path = _write(tmp_path, "short.msr", [(payload, True)], declared=[len(payload) + 1])
    with pytest.raises(ValueError, match="declares"):
        read_obf_mfxdta_stacks(path)


def test_labels_come_from_the_headers(tmp_path):
    from minflux_viewer.core.acquisition_metadata import extract_did_label_map

    description = json.dumps([{"did": _DID_A, "label": "red channel"},
                              {"did": _DID_B, "label": "green"}])
    path = _write(tmp_path, "labels.msr", [(_payload(_DID_A), True)],
                  description=description)
    assert extract_did_label_map(path) == {_DID_A: "red channel", _DID_B: "green"}


# --------------------------------------------------------------- the one parse
def test_the_parse_carries_what_later_steps_would_read_again(tmp_path):
    from minflux_viewer.core.acquisition_metadata import read_acquisition_rois
    from minflux_viewer.core.execution import CPU_THREADS
    from minflux_viewer.core.obf_image_source import list_obf_image_series, scan_obf_stacks
    from minflux_viewer.msr.msr_parser import parse_general

    description = json.dumps([{"did": _DID_A, "label": "red"}])
    path = _write(tmp_path, "one.msr",
                  [(_payload(_DID_A, (1e-6, 2e-6, 2e-6, 4e-6)), True),
                   (_payload(_DID_B, (5e-6, 5e-6, 6e-6, 7e-6), seed=3), False)],
                  description=description)
    free = CPU_THREADS.free
    parsed = parse_general(str(path), str(tmp_path), log=lambda *_a: None)
    assert CPU_THREADS.free == free                  # the lease was given back
    assert parsed["mode"] == "modern"
    assert [d["display_name"] for d in parsed["datasets"]] == ["red", "minflux_1"]
    assert parsed["did_label_map"] == {_DID_A: "red"}
    assert parsed["acquisition_rois"] == read_acquisition_rois(path)
    assert len(parsed["acquisition_rois"]) == 2
    assert parsed["obf_stacks"] == scan_obf_stacks(path)
    assert parsed["image_series"] == list_obf_image_series(path) == []
    # The arrays are the ones written, decoded on several threads.
    for k, entry in enumerate(parsed["datasets"]):
        assert entry["_mfx"].tobytes() == _mfx(400, (0, 3)[k]).tobytes()


def test_an_unreadable_stack_still_ends_in_the_image_series_tree(tmp_path):
    """As before: any stack that cannot be read sends the parse to the legacy
    tree with a warning, rather than raising out of the reader."""
    from minflux_viewer.core.execution import CPU_THREADS
    from minflux_viewer.msr.msr_parser import parse_general

    payload = _payload(_DID_A)
    path = _write(tmp_path, "short.msr", [(payload, True)], declared=[len(payload) + 1])
    seen: list[str] = []
    free = CPU_THREADS.free
    parsed = parse_general(str(path), str(tmp_path), log=seen.append)
    assert CPU_THREADS.free == free
    assert parsed["mode"] == "legacy"
    assert any("could not read stacks" in line for line in seen), seen


# ------------------------------------------------------- parallel zarr decoding
def _decoded_both_ways(store, path):
    from minflux_viewer.utils import zarr2

    sequential = zarr2.open(store, mode="r")[path][:]
    with zarr2.decode_threads(_lease(4)):
        parallel = zarr2.open(store, mode="r")[path][:]
    return sequential, parallel


def test_parallel_decode_is_identical_with_missing_and_partial_chunks():
    from minflux_viewer.utils import zarr2

    rng = np.random.default_rng(1)
    rec = np.zeros(1003, dtype=[("tid", "<i4"), ("loc", "<f8", (3,)), ("vld", "?")])
    rec["tid"] = np.arange(1003)
    rec["loc"] = rng.normal(size=(1003, 3))
    rec["vld"] = rng.random(1003) > 0.5
    store: dict = {}
    zarr2.open(store, mode="w").create_array("mfx", rec, chunk=100)
    del store["mfx/3"]                                 # a missing chunk = fill value
    sequential, parallel = _decoded_both_ways(store, "mfx")
    assert parallel.tobytes() == sequential.tobytes()
    assert np.all(parallel["tid"][300:400] == 0)
    assert np.array_equal(parallel["tid"][1000:], rec["tid"][1000:])   # partial edge chunk


def test_parallel_decode_of_a_two_dimensional_chunk_grid():
    from minflux_viewer.utils import zarr2

    data = np.arange(20 * 13, dtype="<f4").reshape(20, 13)
    store: dict = {}
    zarr2.open(store, mode="w").create_array("img", data)
    meta = json.loads(store["img/.zarray"])
    # Re-chunk on both axes by hand so the 2-D grid path is exercised.
    meta["chunks"] = [7, 5]
    store = {k: v for k, v in store.items() if not k.startswith("img/")}
    store["img/.zarray"] = json.dumps(meta).encode()
    import numcodecs
    codec = numcodecs.get_codec(meta["compressor"]) if meta["compressor"] else None
    for i in range(3):
        for j in range(3):
            block = np.zeros((7, 5), "<f4")
            part = data[i * 7:(i + 1) * 7, j * 5:(j + 1) * 5]
            block[:part.shape[0], :part.shape[1]] = part
            store[f"img/{i}.{j}"] = codec.encode(block) if codec else block.tobytes()
    del store["img/1.2"]
    sequential, parallel = _decoded_both_ways(store, "img")
    assert parallel.tobytes() == sequential.tobytes()
    expected = data.copy()
    expected[7:14, 10:13] = 0
    assert np.array_equal(parallel, expected)


def test_decode_threads_belongs_to_its_block_and_thread():
    import threading

    from minflux_viewer.utils import zarr2

    lease = _lease(6)
    seen: list = []
    with zarr2.decode_threads(lease):
        seen.append(zarr2._DECODE_LEASE.get())
        worker = threading.Thread(target=lambda: seen.append(zarr2._DECODE_LEASE.get()))
        worker.start()
        worker.join()
    seen.append(zarr2._DECODE_LEASE.get())
    assert seen == [lease, None, None]


# ------------------------------------------------- choosing the route on open
def test_only_a_file_carrying_our_state_takes_the_direct_route(tmp_path):
    from minflux_viewer.core.dataset import build_localization_dataset
    from minflux_viewer.msr.msr_parser import parse_general
    from minflux_viewer.msr.quick_open import datasets_from_parsed, parsed_has_viewer_state

    plain = W.write_minflux_msr(tmp_path / "plain.msr",
                                [W.MsrChannel(mfx=_mfx(), name="ch", did=_DID_A)])
    parsed = parse_general(str(plain), str(tmp_path), log=lambda *_a: None)
    assert not parsed_has_viewer_state(parsed)

    rng = np.random.default_rng(0)
    ds = build_localization_dataset(name="mine", x_nm=rng.normal(0, 100, 60),
                                    y_nm=rng.normal(0, 100, 60))
    ds.state["overlay_lut"] = "Green"
    ours = W.write_datasets_msr(tmp_path / "ours.msr", [ds])
    parsed = parse_general(str(ours), str(tmp_path), log=lambda *_a: None)
    assert parsed_has_viewer_state(parsed)
    back = datasets_from_parsed(parsed, ours, prefs={"data": {}})
    assert len(back) == 1 and back[0].state["overlay_lut"] == "Green"
    assert back[0].metadata["msr_source_path"] == str(ours)
