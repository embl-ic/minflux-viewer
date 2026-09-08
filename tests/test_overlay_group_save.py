"""Saving a multi-channel overlay to the one-dataset-per-file formats.

``.mat``/``.npy``/``.json``/``.csv`` hold one dataset each, so an overlay is
several files. What makes them a set again is the folder they share plus an
``overlay`` block in each sidecar naming its siblings by filename.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from minflux_viewer.core import overlay_save as OS


def _dataset(name, did="", index=0, lut="Red", source=""):
    from minflux_viewer.core.loader import build_localization_dataset

    rng = np.random.default_rng(index)
    ds = build_localization_dataset(name=name, x_nm=rng.normal(0, 100, 30),
                                    y_nm=rng.normal(0, 100, 30))
    if did:
        ds.metadata["msr_dataset_did"] = did
    if source:
        ds.metadata["msr_source_path"] = source
    ds.state.update({"overlay_id": "grp", "overlay_index": index,
                     "overlay_lut": lut})
    return ds


def _group():
    return [_dataset("ch1", "did-1", 0, "Red", r"D:\data\run.msr"),
            _dataset("ch2", "did-2", 1, "Green", r"D:\data\run.msr"),
            _dataset("ch3", "did-3", 2, "Blue", r"D:\data\run.msr")]


# ---------------------------------------------------------------- naming
def test_names_that_sanitise_alike_do_not_overwrite_each_other():
    """``ch a`` and ``ch/a`` both become ``ch_a``; without de-duplication one
    channel would silently overwrite the other's file."""
    assert OS.unique_names(["ch a", "ch/a", "ch a"]) == ["ch a", "ch_a", "ch a_2"]
    # only genuinely illegal characters are touched, so a name stays readable
    assert OS.safe_name("1_anti_ELYS (F2)") == "1_anti_ELYS (F2)"
    assert OS.safe_name("") == "dataset"


def test_the_folder_is_named_after_what_identifies_the_set():
    """There is no overlay *group* number to borrow -- what the Dataset Manager
    shows as 'Overlay 1/2' is the channel's index within one group."""
    assert OS.default_group_name(_group()) == "run"
    # different sources: fall back to the first channel's own name
    mixed = _group()
    mixed[1].metadata["msr_source_path"] = r"D:\data\other.msr"
    assert OS.default_group_name(mixed) == "ch1"
    assert OS.default_group_name([]) == "overlay"


# ---------------------------------------------------------------- the block
def test_every_sidecar_names_every_sibling_by_filename():
    """By filename, not path: the set has to survive being moved as a whole."""
    datasets = _group()
    files = ["ch1.mat", "ch2.mat", "ch3.mat"]
    for index in range(3):
        block = OS.overlay_block(datasets, files, index=index)
        assert block["member_index"] == index and block["member_count"] == 3
        assert [m["data_file"] for m in block["members"]] == files
        assert [m["lut"] for m in block["members"]] == ["Red", "Green", "Blue"]
        assert all("/" not in m["data_file"] and "\\" not in m["data_file"]
                   for m in block["members"])


def test_a_group_of_one_is_not_a_group():
    """Nothing to regroup, and nothing to go looking for."""
    assert OS.overlay_from_metadata({"overlay": {"members": [{"data_file": "a"}]}}) is None
    assert OS.overlay_from_metadata({"overlay": {}}) is None
    assert OS.overlay_from_metadata({}) is None
    assert OS.overlay_from_metadata(None) is None


def test_the_block_is_an_addition_not_a_change(tmp_path):
    """A reader that does not know the key must be unaffected, and identity
    still comes from the same three signals."""
    from minflux_viewer.core.metadata_match import sidecar_identity
    from minflux_viewer.core.save import build_metadata, is_metadata_json_payload

    ds = _dataset("ch1", "did-1")
    block = OS.overlay_block(_group(), ["a.mat", "b.mat", "c.mat"], index=0)
    with_block = build_metadata(ds, data_filename="a.mat", overlay=block)
    without = build_metadata(ds, data_filename="a.mat")

    assert is_metadata_json_payload(with_block)
    assert sidecar_identity(with_block) == sidecar_identity(without)
    assert {k: v for k, v in with_block.items() if k != "overlay"} == without


# ---------------------------------------------------------------- the writer
def test_save_group_writes_a_file_and_a_sidecar_per_channel(tmp_path):
    datasets = _group()
    calls = []

    def fake_save(ds, **kwargs):
        calls.append(kwargs)
        return [Path(kwargs["data_path"])]

    written = OS.save_overlay_group(datasets, tmp_path / "run", "mat",
                                    names=["ch1", "ch2", "ch3"], save=fake_save)
    assert [p.name for p in written] == ["ch1.mat", "ch2.mat", "ch3.mat"]
    # every call carries the block, and each names all three siblings
    for index, call in enumerate(calls):
        assert call["overlay"]["member_index"] == index
        assert [m["data_file"] for m in call["overlay"]["members"]] == [
            "ch1.mat", "ch2.mat", "ch3.mat"]


def test_filenames_are_resolved_before_anything_is_written(tmp_path):
    """Each sidecar names its siblings, which is impossible until every name is
    known -- so a writer must never see a block with a name still to come."""
    datasets = _group()
    seen = []

    def fake_save(ds, **kwargs):
        seen.append([m["data_file"] for m in kwargs["overlay"]["members"]])
        return [Path(kwargs["data_path"])]

    OS.save_overlay_group(datasets, tmp_path / "run", "npy", save=fake_save)
    assert seen[0] == seen[1] == seen[2]
    assert seen[0] == ["ch1.npy", "ch2.npy", "ch3.npy"]


def test_saving_no_datasets_is_refused(tmp_path):
    with pytest.raises(ValueError):
        OS.save_overlay_group([], tmp_path, "mat")


# ------------------------------------------------------- recognising it again
def test_a_dataset_block_is_read_under_its_own_key():
    """⚠ Two names for one thing: the sidecar key is ``overlay``, the dataset
    key is namespaced. Reading the sidecar key off a dataset silently found
    nothing, so a saved overlay was never offered for regrouping."""
    ds = _dataset("ch1")
    ds.state.clear()
    assert OS.dataset_overlay_block(ds) is None
    ds.metadata[OS.DATASET_OVERLAY_KEY] = OS.overlay_block(
        _group(), ["a.mat", "b.mat", "c.mat"], index=0)
    assert OS.dataset_overlay_block(ds) is not None
    assert OS.OVERLAY_KEY != OS.DATASET_OVERLAY_KEY


def test_datasets_already_in_an_overlay_are_left_alone():
    """Regrouping what the user has already arranged would undo it."""
    block = OS.overlay_block(_group(), ["a.mat", "b.mat", "c.mat"], index=0)
    datasets = []
    for index in range(2):
        ds = _dataset(f"ch{index}")
        ds.state.clear()
        ds.metadata[OS.DATASET_OVERLAY_KEY] = block
        datasets.append(ds)
    assert len(OS.group_saved_datasets(datasets)[block["group_id"]]) == 2

    datasets[0].state["overlay_id"] = "already-grouped"
    assert len(OS.group_saved_datasets(datasets)[block["group_id"]]) == 1


def test_siblings_are_looked_for_only_beside_the_set(tmp_path):
    """A same-named file elsewhere could be a different acquisition."""
    block = OS.overlay_block(_group(), ["a.mat", "b.mat", "c.mat"], index=0)
    (tmp_path / "b.mat").write_bytes(b"x")
    found, absent = OS.resolve_siblings(block, tmp_path, ["a.mat"])
    assert [p.name for _m, p in found] == ["b.mat"]
    assert [m["data_file"] for m in absent] == ["c.mat"]
    # nothing missing once everything is loaded
    assert OS.missing_members(block, ["a.mat", "b.mat", "c.mat"]) == []
    # matched case-insensitively: the set is identified by recorded names
    assert OS.missing_members(block, ["A.MAT", "b.mat", "c.mat"]) == []


# ------------------------------------------------------------- the round trip
def test_group_round_trip_through_the_real_writers(tmp_path):
    from minflux_viewer.core.loader import load_npy
    from minflux_viewer.core.save import is_metadata_json_file

    datasets = _group()
    folder = tmp_path / "run"
    OS.save_overlay_group(datasets, folder, "npy",
                          names=[ds.name for ds in datasets])
    files = sorted(p.name for p in folder.iterdir())
    assert files == [
        "ch1.npy", "ch1_viewer_metadata.json",
        "ch2.npy", "ch2_viewer_metadata.json",
        "ch3.npy", "ch3_viewer_metadata.json",
    ]
    side = folder / "ch2_viewer_metadata.json"
    assert is_metadata_json_file(side)
    meta = json.loads(side.read_text(encoding="utf-8"))
    assert meta["overlay"]["member_index"] == 1
    # every sibling it names is really there
    for member in meta["overlay"]["members"]:
        assert (folder / member["data_file"]).is_file()

    # ...and loading one carries the block onto the dataset
    ds = load_npy(str(folder / "ch2.npy"), prefs={"data": {}})
    assert OS.dataset_overlay_block(ds) is not None


# -------------------------------------------------------------- the dialog
@pytest.fixture
def _app():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


def test_the_dialog_names_every_channel_and_the_folder(_app):
    from minflux_viewer.ui.save_dialog import SaveProcessedDataDialog

    names = ["1_anti_ELYS", "2_anti_GFP", "3_anti_Nup153"]
    dlg = SaveProcessedDataDialog(names[0], members=names,
                                  default_dir=r"D:\out\run",
                                  prefs={"data": {}, "file": {}})
    try:
        dlg._format.setCurrentIndex(dlg._format.findData("mat"))
        assert len(dlg._member_names) == 3
        opts = dlg.options()
        # a group is a folder plus one stem per channel, not a single path
        assert opts["data_path"] is None
        assert opts["group_names"] == names
        assert Path(opts["group_folder"]).name == "run"
        # editing a name is honoured
        dlg._member_names[1].setText("renamed")
        assert dlg.options()["group_names"][1] == "renamed"
    finally:
        dlg.close()


def test_a_single_dataset_still_uses_the_plain_path_row(_app):
    from minflux_viewer.ui.save_dialog import SaveProcessedDataDialog

    dlg = SaveProcessedDataDialog("only", members=["only"],
                                  prefs={"data": {}, "file": {}})
    try:
        assert not dlg._is_group
        dlg._format.setCurrentIndex(dlg._format.findData("mat"))
        opts = dlg.options()
        assert opts["data_path"] is not None
        assert "group_folder" not in opts
    finally:
        dlg.close()
