"""Classifying a ``.json`` reads a bounded prefix, once, and gets every kind right.

Several kinds share the extension -- localization data (possibly gigabytes), a
processing sidecar, a ROI set, a filter preset, an MBM bead table. Telling them
apart used to decode a data file twice before the loader decoded it a third
time. The bytes actually read are counted here (by wrapping ``open``), not the
calls to a JSON decoder.
"""

from __future__ import annotations

import builtins
import io
import json
from pathlib import Path

import pytest

from minflux_viewer.core import formats
from minflux_viewer.core.json_probe import PREFIX_BYTES, probe_json

BIG = PREFIX_BYTES * 3


def _write(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _records(n: int) -> list[dict]:
    return [{"vld": True, "itr": i % 5, "tid": i // 5, "loc": [i * 1e-9, 2e-9, 0.0],
             "efo": 1e4 + i, "cfr": 0.1, "dcr": [0.4, 0.6]} for i in range(n)]


def _big_records(min_bytes: int = BIG) -> list[dict]:
    n = 64
    while len(json.dumps(_records(n))) < min_bytes:
        n *= 2
    return _records(n)


# ------------------------------------------------------------------ the probe
def test_a_small_document_is_complete_and_decoded_once(tmp_path):
    probe = probe_json(_write(tmp_path / "s.json", {"a": 1, "b": [1, 2]}))
    assert (probe.top, probe.keys, probe.complete) == ("object", ("a", "b"), True)
    assert probe.has_key("b") is True and probe.has_key("c") is False
    assert probe.document() == {"a": 1, "b": [1, 2]}


def test_a_large_array_is_decided_by_its_first_record(tmp_path):
    probe = probe_json(_write(tmp_path / "d.json", _big_records()))
    assert probe.top == "array" and not probe.complete
    assert probe.keys == ("vld", "itr", "tid", "loc", "efo", "cfr", "dcr")


def test_a_truncated_object_is_undetermined_not_absent(tmp_path):
    payload = {"first": "x" * BIG, "later": 1}
    probe = probe_json(_write(tmp_path / "o.json", payload))
    assert probe.top == "object" and not probe.complete
    assert probe.has_key("first") is True
    assert probe.has_key("later") is None          # the prefix ended first
    assert probe.document()["later"] == 1          # one decode answers it


def test_a_brace_inside_a_string_does_not_end_the_record(tmp_path):
    probe = probe_json(_write(tmp_path / "b.json",
                              [{"note": "}", "gri": 1, "tim": 0.5, "xyz": [0, 0, 0]}] * 3))
    assert probe.keys == ("note", "gri", "tim", "xyz")


def test_a_wrapped_record_array_is_data_without_decoding(tmp_path):
    probe = probe_json(_write(tmp_path / "w.json", {"mfx": _big_records()}))
    assert probe.wraps_localizations is True and probe.conclusive
    assert probe.wrapped["mfx"] == ("vld", "itr", "tid", "loc", "efo", "cfr", "dcr")
    assert probe.has_key("rois") is None            # unread -- and it cannot matter
    assert not probe.decoded


_PAD = "x" * (PREFIX_BYTES + 6000)       # pushes what follows past the fast path


@pytest.mark.parametrize("pad", ["", _PAD], ids=["small", "large"])
@pytest.mark.parametrize("payload, expected", [
    # A localization array decides "data" wherever it is, whatever follows it.
    (lambda pad: {"data": [{"loc": [0, 0, 0], "tid": 1, "note": pad}], "rois": []}, "json"),
    (lambda pad: {"rois": [], "mfx": [{"loc": [0, 0, 0], "note": pad}]}, "json"),
    (lambda pad: {"data": [{"loc": [0, 0, 0], "note": pad}],
                  "minflux_viewer_metadata": 1}, "json"),
    # The sidecar marker first decides "sidecar" at once, whatever follows.
    (lambda pad: {"minflux_viewer_metadata": 1, "comment": pad,
                  "data": [{"loc": [0, 0, 0]}]}, "metadata_sidecar"),
    # A wrapper array of non-localizations decides nothing: the ROI set wins.
    (lambda pad: {"data": [{"label": pad}], "rois": []}, "roi_set"),
], ids=["data then rois", "rois then data", "data then marker", "marker first",
        "non-localization wrapper"])
def test_the_kind_does_not_depend_on_the_file_size(tmp_path, pad, payload, expected):
    """Final review, finding 5: ``{"data": [...], "rois": []}`` was a ROI set
    when small and data when its array outgrew the fast path."""
    path = _write(tmp_path / "p.json", payload(pad))
    fast = formats.resolve_open(path)
    deep = formats.resolve_open(path, deep=True)
    assert deep is not None and deep.key == expected
    assert fast is formats.UNDETERMINED or fast.key == expected


@pytest.mark.parametrize("pad", ["", _PAD], ids=["small", "large"])
def test_key_order_decides_only_a_sidecar_that_also_wraps_localizations(tmp_path, pad):
    """The documented exception to order-independence: rule 1 looks at the
    first key, so re-serialising an object that holds both the sidecar marker
    and a localization array (``sort_keys`` puts ``data`` first) changes its
    kind. A sidecar without such an array keeps its kind in any key order."""
    mixed = {"minflux_viewer_metadata": 1, "comment": pad, "data": [{"loc": [0, 0, 0]}]}
    plain = {"minflux_viewer_metadata": 1, "comment": pad, "data_file": "run.npy",
             "filters": []}
    cases = [(mixed, False, "metadata_sidecar"), (mixed, True, "json"),
             (plain, False, "metadata_sidecar"), (plain, True, "metadata_sidecar")]
    for payload, sort_keys, expected in cases:
        path = tmp_path / "order.json"
        path.write_text(json.dumps(payload, sort_keys=sort_keys), encoding="utf-8")
        assert formats.resolve_open(path, deep=True).key == expected, (sort_keys, payload.keys())


@pytest.mark.parametrize("records", [lambda: _records(3), _big_records], ids=["small", "large"])
def test_a_long_wrapped_array_followed_by_rois_is_data_on_the_fast_path(tmp_path, records):
    """The reviewer's case exactly: the first record closes early, the array
    outgrows the fast path, and ``rois`` follows. Decided at the first record,
    without reading on, and the same as for the small file."""
    path = _write(tmp_path / "late.json", {"data": records(), "rois": []})
    assert formats.resolve_open(path).key == "json"


def test_value_kinds_answer_list_questions_without_decoding(tmp_path):
    probe = probe_json(_write(tmp_path / "k.json", {"rois": [{"pad": "x" * BIG}], "n": 1}))
    assert not probe.complete and not probe.decoded
    assert probe.kind("rois") == "array"


@pytest.mark.parametrize("text, top", [("not json", "invalid"), ("", "empty"),
                                       ("   \n", "empty"), ("﻿[1, 2]", "array")])
def test_odd_inputs(tmp_path, text, top):
    path = tmp_path / "odd.json"
    path.write_text(text, encoding="utf-8")
    probe = probe_json(path)
    assert probe.top == top and probe.conclusive


# ------------------------------------- what the fast path cannot decide (review 2)
_FILTER_ROW = {"apply": True, "attribute": "efo", "value_as": "per loc", "min": 1, "max": 10}

_UNDECIDED = [
    ("filter preset after 70 KB of whitespace", lambda: " " * 70_000 + json.dumps([_FILTER_ROW]),
     "filter_preset"),
    ("filter preset whose first row exceeds the fast path",
     lambda: json.dumps([dict(_FILTER_ROW, note="x" * BIG)]), "filter_preset"),
    ("MBM table whose first record exceeds the fast path",
     lambda: json.dumps([{"gri": 1, "tim": 0.5, "xyz": [0, 0, 0], "note": "x" * BIG}] * 2),
     "mbm_points"),
    ("ROI set larger than the fast path", lambda: json.dumps(_roi_set(rois=2000)), "roi_set"),
    ("ROI set whose key follows a large value",
     lambda: json.dumps({"comment": "x" * BIG, "rois": []}), "roi_set"),
    ("data after whitespace", lambda: " " * 70_000 + json.dumps(_records(20)), "json"),
]


@pytest.mark.parametrize("label, text, expected", _UNDECIDED, ids=[k[0] for k in _UNDECIDED])
def test_the_fast_path_defers_instead_of_guessing_and_the_deep_path_decides(
        tmp_path, label, text, expected):
    path = tmp_path / "u.json"
    path.write_text(text(), encoding="utf-8")
    assert formats.resolve_open(path) is formats.UNDETERMINED, label
    spec = formats.resolve_open(path, deep=True)
    assert spec is not None and spec.key == expected, label


def test_a_predicate_given_a_path_never_deep_scans(tmp_path, monkeypatch):
    """Only resolve_open(deep=True) reads past the fast path; a predicate asked
    about a path it cannot decide says no instead of reading the file."""
    from minflux_viewer.core.filter_io import is_filter_json_file

    path = tmp_path / "padded.json"
    path.write_text(" " * 70_000 + json.dumps([_FILTER_ROW]), encoding="utf-8")
    counter = _counting_open(monkeypatch, path)
    assert is_filter_json_file(path) is False
    assert counter.bytes <= PREFIX_BYTES


# ------------------------------------------------------ classification table
def _sidecar(rois: int = 0) -> dict:
    return {"minflux_viewer_metadata": 1, "content": "raw", "filters": [],
            "rois": [{"id": f"r{i}", "roi_type": "rectangle", "pad": "x" * 64}
                     for i in range(rois)]}


def _roi_set(rois: int = 1) -> dict:
    return {"rois": [{"id": f"r{i}", "roi_type": "rectangle", "pad": "x" * 64}
                     for i in range(rois)]}


_KINDS = [
    ("data array, small", lambda: _records(20), "json"),
    ("data array, large", _big_records, "json"),
    ("data wrapped in 'mfx', large", lambda: {"mfx": _big_records()}, "json"),
    ("sidecar, small", _sidecar, "metadata_sidecar"),
    ("sidecar, large", lambda: _sidecar(rois=2000), "metadata_sidecar"),
    ("ROI set, small", _roi_set, "roi_set"),
    ("filter preset", lambda: [{"apply": True, "attribute": "efo", "value_as": "per loc",
                                "min": 1, "max": 10}], "filter_preset"),
    ("empty filter preset", lambda: [], "filter_preset"),
    ("MBM bead table with a brace in a string",
     lambda: [{"gri": 1, "tim": 0.5, "xyz": [0.0, 0.0, 0.0], "str": 1.0, "note": "}"}] * 4,
     "mbm_points"),
]


@pytest.mark.parametrize("label, payload, expected", _KINDS, ids=[k[0] for k in _KINDS])
def test_every_kind_resolves_to_exactly_its_format(tmp_path, label, payload, expected):
    path = _write(tmp_path / "f.json", payload())
    spec = formats.resolve_open(path)
    assert spec is not None and spec.key == expected, label
    claimed = []
    probe = probe_json(path)
    for candidate in formats.FORMATS:
        if ".json" in candidate.extensions and candidate.detect:
            if formats._load_predicate(candidate.detect)(probe):
                claimed.append(candidate.key)
    # Data is the fallback, claimed by no predicate; every other kind by its own only.
    assert claimed == ([] if expected == "json" else [expected]), f"{label}: claimed by {claimed}"


# ------------------------------------------------------------ bytes actually read
class _Counter:
    def __init__(self, target: Path) -> None:
        self.target = target.resolve()
        self.bytes = 0
        self.opens = 0


def _counting_open(monkeypatch, target: Path) -> _Counter:
    counter = _Counter(target)
    real_open = builtins.open

    class Wrapped:
        def __init__(self, handle):
            self._handle = handle

        def read(self, *args):
            data = self._handle.read(*args)
            counter.bytes += len(data)
            return data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._handle.close()

        def __getattr__(self, name):
            return getattr(self._handle, name)

    def fake_open(file, *args, **kwargs):
        handle = real_open(file, *args, **kwargs)
        try:
            same = Path(file).resolve() == counter.target
        except (TypeError, OSError):
            same = False
        if same:
            counter.opens += 1
            return Wrapped(handle)
        return handle

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(io, "open", fake_open)
    return counter


@pytest.mark.parametrize("payload", [_big_records, lambda: {"mfx": _big_records()}],
                         ids=["array", "wrapped"])
def test_resolving_a_large_data_file_reads_one_prefix(tmp_path, monkeypatch, payload):
    path = _write(tmp_path / "large.json", payload())
    assert path.stat().st_size > 2 * PREFIX_BYTES
    counter = _counting_open(monkeypatch, path)
    spec = formats.resolve_open(path)
    assert spec.key == "json"
    assert counter.bytes <= PREFIX_BYTES
    assert counter.opens == 1                        # one probe, shared by every predicate


@pytest.mark.parametrize("predicate", [
    "minflux_viewer.core.save:is_metadata_json_file",
    "minflux_viewer.core.roi:is_roi_json_file",
    "minflux_viewer.core.filter_io:is_filter_json_file",
    "minflux_viewer.core.mbm_file:is_mbm_points_file",
])
def test_each_predicate_alone_reads_at_most_a_prefix_of_data(tmp_path, monkeypatch, predicate):
    path = _write(tmp_path / "large.json", _big_records())
    counter = _counting_open(monkeypatch, path)
    assert formats._load_predicate(predicate)(path) is False
    assert counter.bytes <= PREFIX_BYTES
