"""Adversarial deterministic jobs: partial failure is never false success."""

import gc
import json
import weakref
from pathlib import Path

import numpy as np
import pytest

from minflux_viewer.api.io import batch, load, save
from minflux_viewer.core import batch_processing as jobs
from minflux_viewer.core.dataset import build_localization_dataset


class StopRequested(Exception):
    pass


def _source(tmp_path, name="source"):
    ds = build_localization_dataset(name=name, x_nm=np.arange(8.), y_nm=np.arange(8.))
    return save(ds, tmp_path / "inputs" / name)[0]


def _process(ds, params, check):
    check()
    ds.attr["score"] = np.full(ds.prop.num_loc, params.get("score", 2.))


def test_batch_publishes_owned_outputs_and_verifies_resume(tmp_path):
    source = _source(tmp_path)
    root = tmp_path / "outputs"
    first = batch([source], root, _process, operation="score@1", parameters={"score": 7})
    assert first[0]["status"] == "written"
    target = Path(first[0]["output_dir"])
    assert load(target / "data.npy").prop.num_loc == 8
    receipt = json.loads((target / "receipt.json").read_text())
    assert receipt["request"]["parameters"] == {"score": 7}
    again = batch([source], root, _process, operation="score@1", parameters={"score": 7}, resume=True)
    assert again[0]["status"] == "resumed"
    with pytest.raises(FileExistsError):
        batch([source], root, _process, operation="score@1", parameters={"score": 7})


@pytest.mark.parametrize("changed", ["operation", "parameters", "input", "output", "extra", "recipe", "receipt"])
def test_resume_refuses_stale_or_tampered_content_without_replacing_it(tmp_path, changed):
    source = _source(tmp_path)
    root = tmp_path / "outputs"
    options = dict(operation="score@1", apply_sidecar=True)
    result = batch([source], root, _process, **options)
    target = Path(result[0]["output_dir"])
    if changed == "operation":
        options["operation"] = "score@2"
    elif changed == "parameters":
        options["parameters"] = {"score": 3}
    elif changed == "input":
        source.write_bytes(b"changed input")
    elif changed == "output":
        (target / "data.npy").write_bytes(b"changed output")
    elif changed == "extra":
        (target / "unexpected.txt").write_text("extra")
    elif changed == "recipe":
        source.with_name("source_viewer_metadata.json").write_text("{}")
    else:
        (target / "receipt.json").write_text("[]")
    before = {p.name: p.read_bytes() for p in target.iterdir()}
    with pytest.raises(ValueError, match="Resume refused"):
        batch([source], root, _process, resume=True, **options)
    assert {p.name: p.read_bytes() for p in target.iterdir()} == before


@pytest.mark.parametrize("error", [ValueError, OSError, RuntimeError, MemoryError, StopRequested])
def test_failure_or_cancel_leaves_only_previously_committed_items(tmp_path, error):
    sources = [_source(tmp_path, str(i)) for i in range(3)]
    calls = []
    def process(ds, params, check):
        calls.append(ds)
        if len(calls) == 2:
            raise error("injected stop")
    with pytest.raises(error, match="injected stop"):
        batch(sources, tmp_path / "outputs", process, operation="fail@1")
    assert len(calls) == 2
    assert len(list((tmp_path / "outputs").iterdir())) == 1
    result = batch(sources, tmp_path / "outputs", _process, operation="fail@1", resume=True)
    assert [entry["status"] for entry in result] == ["resumed", "written", "written"]


def test_cancel_after_serializer_before_publication_cleans_stage(tmp_path, monkeypatch):
    source = _source(tmp_path)
    stopped = False
    real_save = jobs.save_dataset
    def save_then_stop(*args, **kwargs):
        nonlocal stopped
        real_save(*args, **kwargs)
        stopped = True
    def check():
        if stopped:
            raise StopRequested
    monkeypatch.setattr(jobs, "save_dataset", save_then_stop)
    with pytest.raises(StopRequested):
        batch([source], tmp_path / "out", _process, operation="cancel@1", check_cancelled=check)
    assert not list((tmp_path / "out").iterdir())


def test_source_edit_during_processing_is_not_published(tmp_path):
    source = _source(tmp_path)
    def process(ds, params, check):
        source.write_bytes(b"edited while processing")
    with pytest.raises(ValueError, match="input changed"):
        batch([source], tmp_path / "out", process, operation="edit@1")
    assert not list((tmp_path / "out").iterdir())


def test_parameters_are_captured_and_each_item_receives_an_independent_copy(tmp_path):
    sources = [_source(tmp_path, str(i)) for i in range(2)]
    params = {"nested": [1]}
    def process(ds, current, check):
        assert current == {"nested": [1]}
        current["nested"].append(2)
        params["nested"].append(9)
    batch(sources, tmp_path / "out", process, operation="params@1", parameters=params)


def test_many_sequential_items_do_not_retain_loaded_datasets(tmp_path):
    sources = [_source(tmp_path, str(i)) for i in range(40)]
    refs = []
    def process(ds, params, check):
        gc.collect()
        assert not any(ref() is not None for ref in refs)
        refs.append(weakref.ref(ds))
    result = batch(sources, tmp_path / "out", process, operation="release@1")
    gc.collect()
    assert len(result) == 40 and not any(ref() is not None for ref in refs)


@pytest.mark.parametrize("parameters", [{"bad": float("nan")}, {1: "non-string"}, {"bad": object()}])
def test_invalid_parameters_do_not_create_outputs(tmp_path, parameters):
    with pytest.raises((TypeError, ValueError)):
        batch([], tmp_path / "out", _process, operation="bad@1", parameters=parameters)
    assert not (tmp_path / "out").exists()


def test_duplicate_and_overlapping_paths_are_refused(tmp_path):
    source = _source(tmp_path)
    with pytest.raises(ValueError, match="distinct"):
        batch([source, source], tmp_path / "out", _process, operation="same@1")
    with pytest.raises(ValueError, match="overlap"):
        batch([source], source.parent, _process, operation="same@1")


@pytest.mark.parametrize("cancel_at", range(1, 24))
def test_cancel_at_each_checkpoint_never_leaves_partial_published_output(tmp_path, cancel_at):
    source = _source(tmp_path)
    calls = 0
    def check():
        nonlocal calls
        calls += 1
        if calls == cancel_at:
            raise StopRequested
    try:
        result = batch([source], tmp_path / "out", _process, operation="cancel-each@1",
                       check_cancelled=check)
    except StopRequested:
        assert not list((tmp_path / "out").glob("*"))
    else:
        assert len(result) == 1
        output = Path(result[0]["output_dir"])
        assert (output / "receipt.json").is_file()
        assert load(output / "data.npy").prop.num_loc == 8


@pytest.mark.parametrize("error", [OSError, ValueError, MemoryError])
def test_writer_failure_never_publishes_an_item(tmp_path, monkeypatch, error):
    source = _source(tmp_path)
    def fail(*args, **kwargs):
        raise error("injected writer failure")
    monkeypatch.setattr(jobs, "save_dataset", fail)
    with pytest.raises(error, match="writer failure"):
        batch([source], tmp_path / "out", _process, operation="write-fail@1")
    assert not list((tmp_path / "out").iterdir())
