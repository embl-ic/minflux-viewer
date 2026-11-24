"""What the MINFLUX JSON reader accepts, pinned by VALUE before it is rebuilt.

Each accepted coordinate form is checked for the coordinates it produces (in
nm), not merely for loading: bare ``x``/``y`` are metres in JSON while the flat
processed-table helper reads them as nanometres, so a reroute that only checked
"it loads" would have shrunk them by 1e9 unnoticed. Single-row and ``lnc``-only
files and every dict wrapper are kept. Behaviour the reader does not have yet is
marked ``xfail(strict=True)`` with the step that adds it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from minflux_viewer.core.loader_format_readers import _iter_json_records, load_json


def _write(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _two(**columns) -> list[dict]:
    """Two records from column lists."""
    return [{key: values[i] for key, values in columns.items()} for i in range(2)]


# (form, records, expected loc_nm rows)
_FORMS = [
    ("loc 3-wide (m)", _two(loc=[[1e-9, 2e-9, 3e-9], [4e-9, 5e-9, 6e-9]]),
     [[1, 2, 3], [4, 5, 6]]),
    ("loc 2-wide (m), z = 0", _two(loc=[[1e-9, 2e-9], [4e-9, 5e-9]]),
     [[1, 2, 0], [4, 5, 0]]),
    ("lnc only (m)", _two(lnc=[[1e-9, 2e-9, 3e-9], [4e-9, 5e-9, 6e-9]]),
     [[1, 2, 3], [4, 5, 6]]),
    ("loc_x/loc_y/loc_z (m)", _two(loc_x=[1e-9, 4e-9], loc_y=[2e-9, 5e-9], loc_z=[3e-9, 6e-9]),
     [[1, 2, 3], [4, 5, 6]]),
    ("bare x/y (m), z absent", _two(x=[1e-9, 4e-9], y=[2e-9, 5e-9]),
     [[1, 2, 0], [4, 5, 0]]),
    ("loc_x_nm/loc_y_nm (nm)", _two(loc_x_nm=[1.0, 4.0], loc_y_nm=[2.0, 5.0], loc_z_nm=[3.0, 6.0]),
     [[1, 2, 3], [4, 5, 6]]),
    ("xnm/ynm/znm (nm)", _two(xnm=[1.0, 4.0], ynm=[2.0, 5.0], znm=[3.0, 6.0]),
     [[1, 2, 3], [4, 5, 6]]),
    ("x_nm/y_nm (nm)", _two(x_nm=[1.0, 4.0], y_nm=[2.0, 5.0]),
     [[1, 2, 0], [4, 5, 0]]),
]


@pytest.mark.parametrize("form, records, expected", _FORMS, ids=[f[0] for f in _FORMS])
def test_coordinate_forms_keep_their_units(tmp_path, form, records, expected):
    for record, tid in zip(records, (1, 2)):
        record["tid"] = tid
    ds = load_json(_write(tmp_path / "c.json", records), apply_sidecar=False)
    np.testing.assert_allclose(np.asarray(ds.loc_nm)[:, :3], expected, rtol=1e-12, atol=1e-9,
                               err_msg=form)


def test_a_single_record_loads(tmp_path):
    ds = load_json(_write(tmp_path / "one.json",
                          [{"loc": [1e-9, 2e-9, 3e-9], "tid": 5, "itr": 0, "vld": True}]),
                   apply_sidecar=False)
    assert ds.prop.num_loc == 1
    np.testing.assert_allclose(np.asarray(ds.loc_nm)[0, :3], [1, 2, 3], atol=1e-9)


def test_a_non_minflux_json_is_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match="not recognised as MINFLUX localization JSON"):
        load_json(_write(tmp_path / "x.json", {"Itr": [{"Mode": {"id": "mflx"}}], "id": "s"}),
                  apply_sidecar=False)


def _is_pinned(ds) -> bool:
    source = (ds.metadata.get("z_scaling_factor_provenance") or {}).get("source", "")
    return "z_scaling_factor" in ds.derived and source.startswith("processed")


@pytest.mark.parametrize("family", ["xnm", "x_nm", "loc_x_nm"])
@pytest.mark.parametrize("n", [1, 2, 6])
def test_a_processed_flat_table_is_pinned_like_npy_and_mat(tmp_path, decoder, family, n):
    """Columns in nm without a ``loc`` vector are final, processed coordinates
    (what our snapshot export writes). ``.npy``/``.mat`` pin the Z scaling
    factor so the post-load estimation cannot correct Z twice -- for every nm
    spelling and every row count, one included (review 2, A2.4)."""
    x, y, z = {"xnm": ("xnm", "ynm", "znm"), "x_nm": ("x_nm", "y_nm", "z_nm"),
               "loc_x_nm": ("loc_x_nm", "loc_y_nm", "loc_z_nm")}[family]
    records = [{x: float(i), y: float(2 * i), z: float(-i), "tid": i // 2, "tim": float(i)}
               for i in range(n)]
    ds = load_json(_write(tmp_path / "flat.json", records), apply_sidecar=False)
    assert _is_pinned(ds), family
    expected = [[float(i), float(2 * i), float(-i)] for i in range(n)]
    np.testing.assert_allclose(np.asarray(ds.loc_nm)[:, :3], expected, atol=1e-9)
    assert ds.metadata["source_format"] == "json"


@pytest.mark.parametrize("records", [
    [{"x": 1e-9, "y": 2e-9, "tid": 1}],                                # metres
    [{"loc": [1e-9, 2e-9, 0.0], "xnm": 1.0, "ynm": 2.0, "tid": 1}],    # raw + derived
    [{"xnm": 1.0, "ynm": 2.0, "x_nm": 1.0, "y_nm": 2.0, "tid": 1}],    # two families
], ids=["metres", "with a loc vector", "two nm families"])
def test_only_a_single_nm_family_is_a_processed_table(tmp_path, records):
    ds = load_json(_write(tmp_path / "raw.json", records), apply_sidecar=False)
    assert not _is_pinned(ds)


def test_a_one_row_flat_mat_table_is_pinned_too(tmp_path):
    from scipy.io import savemat

    from minflux_viewer.core.loader_format_readers import load_dataset

    path = tmp_path / "one.mat"
    savemat(path, {"x_nm": np.array([5.0]), "y_nm": np.array([6.0]), "z_nm": np.array([7.0])})
    ds = load_dataset(path, apply_sidecar=False)
    assert _is_pinned(ds)
    np.testing.assert_allclose(np.asarray(ds.loc_nm)[0, :3], [5, 6, 7])


@pytest.fixture(params=["json.load", "streamed"])
def decoder(request, monkeypatch):
    """Run a test through both decoders: the size threshold picks only which."""
    import minflux_viewer.core.loader_format_readers as readers

    if request.param == "streamed":
        monkeypatch.setattr(readers, "_JSON_STREAM_BYTES", 0)
    return request.param


_WRAPPERS = ["mfx", "records", "localizations", "localisations", "data"]


@pytest.mark.parametrize("wrapper", _WRAPPERS)
@pytest.mark.parametrize("around", ["alone", "with other members"])
def test_every_wrapper_loads_in_both_decoders(tmp_path, decoder, wrapper, around):
    """Review 2, A2.1: the size of a file must not decide whether its wrapper is
    accepted -- the streamed decoder once refused every wrapped file."""
    records = _two(loc=[[1e-9, 2e-9, 0.0], [4e-9, 5e-9, 0.0]], tid=[1, 2])
    payload = {wrapper: records}
    if around != "alone":
        payload = {"version": 2, "note": {"a": [1, 2]}, wrapper: records, "after": 12345.5}
    ds = load_json(_write(tmp_path / "w.json", payload), apply_sidecar=False)
    np.testing.assert_allclose(np.asarray(ds.loc_nm)[:, :2], [[1, 2], [4, 5]], atol=1e-9)


@pytest.mark.parametrize("payload, message", [
    ({"mfx": [{"loc": [0, 0, 0]}], "data": [{"loc": [0, 0, 0]}]}, "exactly one"),
    ({"version": 2, "points": [{"loc": [0, 0, 0]}]}, None),
    ({"mfx": [1, 2, 3]}, None),
])
def test_both_decoders_refuse_the_same_objects(tmp_path, decoder, payload, message):
    with pytest.raises(ValueError, match=message):
        load_json(_write(tmp_path / "w.json", payload), apply_sidecar=False)


def test_a_value_split_across_read_chunks_is_decoded_whole(tmp_path, monkeypatch):
    """Every boundary: a 7-character read splits numbers, literals and keys."""
    import minflux_viewer.core.loader_format_readers as readers

    records = [{"loc": [i * 1.25e-9, 2e-9, 0.0], "tid": 1000 + i, "efo": 123456.789 + i,
                "vld": True} for i in range(9)]
    payload = {"version": 1234567, "mfx": records, "trailer": False}
    path = _write(tmp_path / "split.json", payload)
    monkeypatch.setattr(readers._JsonTextStream, "CHUNK", 7)
    assert list(_iter_json_records(path)) == records


def test_both_decoders_build_the_same_dataset(tmp_path, monkeypatch):
    import minflux_viewer.core.loader_format_readers as readers
    from minflux_viewer.core.save import _write_json
    from tests.test_format_equivalence import synthetic_m2410

    path = tmp_path / "acq.json"
    _write_json(path, synthetic_m2410())
    small = load_json(path, apply_sidecar=False)
    monkeypatch.setattr(readers, "_JSON_STREAM_BYTES", 0)
    streamed = load_json(path, apply_sidecar=False)
    assert list(streamed.attr.keys()) == list(small.attr.keys())
    for key in small.attr.keys():
        a, b = np.asarray(small.attr[key]), np.asarray(streamed.attr[key])
        assert a.dtype == b.dtype, key
        np.testing.assert_array_equal(a, b, err_msg=key)
    for key in ("source_version", "source_format", "cfr_iteration", "efc_iteration",
                "raw_num_itr", "raw_num_loc", "valid_num_loc"):
        assert streamed.metadata.get(key) == small.metadata.get(key), key


@pytest.mark.parametrize("key", ["vld", "itr", "tid"])
def test_a_gap_in_a_structural_field_is_an_error_naming_the_record(tmp_path, decoder, key):
    records = [{"loc": [1e-9, 2e-9, 0.0], "vld": True, "itr": 0, "tid": i} for i in range(3)]
    del records[1][key]
    with pytest.raises(ValueError, match=rf"record 1 has no '{key}'"):
        load_json(_write(tmp_path / "gap.json", records), apply_sidecar=False)


def test_a_key_first_seen_later_lines_up_with_its_rows(tmp_path, decoder):
    records = [{"loc": [i * 1e-9, 0.0, 0.0], "vld": True, "itr": 0, "tid": i} for i in range(4)]
    records[2]["efo"] = 1234.5
    ds = load_json(_write(tmp_path / "late.json", records), apply_sidecar=False)
    efo = np.asarray(ds.components.mfx_raw["efo"], dtype=float)
    np.testing.assert_array_equal(np.isnan(efo), [True, True, False, True])
    assert efo[2] == 1234.5


def test_records_without_an_iteration_are_not_called_m2410(tmp_path, decoder):
    records = [{"loc": [i * 1e-9, 0.0, 0.0], "tid": i} for i in range(3)]
    ds = load_json(_write(tmp_path / "noitr.json", records), apply_sidecar=False)
    assert ds.metadata["source_version"] == "imported"
    assert ds.metadata["source_format"] == "json"


@pytest.mark.parametrize("text, label", [
    ('[{"a": 1} {"a": 2}]', "missing comma"),
    ('[{"a": 1},]', "trailing comma"),
    ('[{"a": 1}] trailing', "content after the array"),
    ('[{"a": 1},,{"a": 2}]', "double comma"),
    ('{"mfx": [{"a": 1}] "b": 2}', "missing comma between members"),
    ('{"mfx": [{"a": 1}],}', "trailing comma in the object"),
    ('{"mfx": [{"a": 1}]', "unclosed object"),
    ('{"mfx": [{"a": 1}]} trailing', "content after the object"),
    ('{mfx: [{"a": 1}]}', "unquoted key"),
])
def test_the_streaming_decoder_rejects_what_json_rejects(tmp_path, text, label):
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)
    with pytest.raises(ValueError):
        list(_iter_json_records(path))


# ------------------------------------ the two decoders build identical columns
_BASE = {"loc": [1e-9, 2e-9, 3e-9], "tid": 1, "itr": 0, "vld": True}

#: Each case: records whose columns both decoders must build identically --
#: same keys kept, same dtypes, same values (final review, finding 2).
_EQUIVALENCE = {
    "integer beyond 2**53": [dict(_BASE, tid=9007199254740993)],
    "integer beyond int64": [dict(_BASE, big=2**63), dict(_BASE, big=1)],
    "integer beyond float64": [dict(_BASE, huge=10**400)],
    "a later float makes the column float": [dict(_BASE, a=1), dict(_BASE, a=2.5)],
    "booleans and integers": [dict(_BASE, flag=True), dict(_BASE, flag=2)],
    "booleans only": [dict(_BASE, flag=True), dict(_BASE, flag=False)],
    "a gap makes integers float": [dict(_BASE, a=7), _BASE],
    "a vector that grows": [dict(_BASE, loc=[1e-9, 2e-9]), dict(_BASE, loc=[4e-9, 5e-9, 6e-9])],
    "a vector that shrinks": [dict(_BASE, w=[1, 2, 3]), dict(_BASE, w=[4])],
    "a vector after null": [dict(_BASE, lnc=None), dict(_BASE, lnc=[4e-9, 5e-9, 6e-9])],
    "a vector after an absent key": [_BASE, dict(_BASE, lnc=[4e-9, 5e-9, 6e-9])],
    "empty vectors": [dict(_BASE, w=[]), dict(_BASE, w=[])],
    "a scalar after a vector": [dict(_BASE, w=[1, 2]), dict(_BASE, w=3)],
    "a vector after a scalar": [dict(_BASE, w=3), dict(_BASE, w=[1, 2])],
    "text in a vector": [dict(_BASE, w=[1, "2"]), dict(_BASE, w=[3, 4, 5])],
    "null in a vector": [dict(_BASE, w=[1, None]), dict(_BASE, w=[3, 4])],
    "numeric text in a column with a gap": [dict(_BASE, a="1.5"), _BASE],
    "text": [dict(_BASE, note="x"), dict(_BASE, note="y")],
    "an object": [dict(_BASE, meta={"a": 1}), dict(_BASE, meta={"a": 2})],
    "null only": [dict(_BASE, a=None), dict(_BASE, a=None)],
}


@pytest.mark.parametrize("records", list(_EQUIVALENCE.values()), ids=list(_EQUIVALENCE))
def test_both_decoders_build_identical_columns(tmp_path, records):
    from minflux_viewer.core.loader_format_readers import _json_columns, _stream_json_columns

    path = _write(tmp_path / "e.json", records)
    whole = _json_columns(json.loads(path.read_text(encoding="utf-8")))
    streamed = _stream_json_columns(path)
    assert list(streamed) == list(whole)
    for key, expected in whole.items():
        got = streamed[key]
        assert got.dtype == expected.dtype and got.shape == expected.shape, key
        np.testing.assert_array_equal(got, expected, err_msg=key)


def test_the_streamed_decoder_keeps_an_exact_integer_through_the_loader(tmp_path, decoder):
    ds = load_json(_write(tmp_path / "t.json", [dict(_BASE, tid=9007199254740993)]),
                   apply_sidecar=False)
    assert np.asarray(ds.mfx_raw["tid"]).tolist() == [9007199254740993]


def test_a_number_split_at_any_read_boundary_is_decoded_whole(tmp_path, monkeypatch):
    """``1e-09`` cut after ``1e`` used to fail: the decoder takes ``1`` and stops
    before an exponent with no digits yet (final review, finding 4)."""
    import minflux_viewer.core.loader_format_readers as readers

    payload = {"metadata": 1e-09, "scale": -2.5E+3, "n": 123456789,
               "records": [dict(_BASE, efo=6.02e23, cfr=-1e-7)]}
    path = _write(tmp_path / "x.json", payload)
    text = path.read_text(encoding="utf-8")
    for chunk in range(1, len(text) + 2):
        monkeypatch.setattr(readers._JsonTextStream, "CHUNK", chunk)
        assert list(_iter_json_records(path)) == payload["records"], chunk
