"""Matching a processing recipe to a dataset, and the rules around applying it.

A sidecar is a portable **recipe**, not a property of one dataset, so identity
only ever picks a default: applying a saved processing to another copy, or to
different data entirely, is deliberately allowed. That is why there is no
application-minted dataset uid -- see ``core/metadata_match``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from minflux_viewer.core.metadata_match import (
    MATCH_ACQUISITION,
    MATCH_DATA_FILE,
    MATCH_DID,
    MATCH_ORDER,
    best_match,
    dataset_identity,
    is_snapshot_recipe,
    match_kind,
    recipe_summary,
    sidecar_identity,
)


def _ds(*, did="", file_name="", acquired=""):
    return SimpleNamespace(
        name=file_name or "ds",
        file=SimpleNamespace(name=file_name, recent_path=""),
        metadata={"msr_dataset_did": did, "acquisition_date": acquired},
    )


def _meta(*, did=None, data_file=None, acquired=None, **rest):
    meta = {"minflux_viewer_metadata": 1, "content": "raw", **rest}
    if did is not None:
        meta["msr_dataset_did"] = did
    if data_file is not None:
        meta["data_file"] = data_file
    if acquired is not None:
        meta["acquisition"] = {"date": acquired}
    return meta


# ---------------------------------------------------------------- signals

def test_the_did_is_the_strongest_signal_and_wins_over_position():
    """Signal strength beats list order, or an early weak match would shadow it."""
    weak = _ds(acquired="2026-06-25T17:34:25+02:00")
    strong = _ds(did="DID-1")
    meta = _meta(did="DID-1", acquired="2026-06-25T17:34:25+02:00")
    index, kind = best_match(meta, [weak, strong])
    assert (index, kind) == (1, MATCH_DID)


def test_the_data_file_name_matches_after_the_pair_is_moved():
    """The sidecar records a bare filename, so only the name may be compared."""
    ds = _ds(file_name="D:/elsewhere/run.mat")
    assert match_kind(_meta(data_file="run.mat"), ds) == MATCH_DATA_FILE


def test_matching_is_case_insensitive_on_the_filename():
    assert match_kind(_meta(data_file="RUN.MAT"), _ds(file_name="run.mat")) \
        == MATCH_DATA_FILE


def test_acquisition_time_matches_data_re_exported_to_another_format():
    """The one signal that survives a format change: .npy has no did, and the
    filename changes, but the instrument timestamp does not."""
    ds = _ds(file_name="run.npy", acquired="2026-06-25T17:34:25+02:00")
    meta = _meta(data_file="run.mat", acquired="2026-06-25T17:34:25+02:00")
    assert match_kind(meta, ds) == MATCH_ACQUISITION


def test_an_absent_signal_never_matches():
    """Two datasets that both lack a DID are not thereby the same dataset."""
    assert match_kind(_meta(did="", data_file="", acquired=""), _ds()) is None
    assert best_match(_meta(), [_ds(), _ds()]) == (None, None)


def test_no_dataset_matches_is_reported_as_such():
    assert best_match(_meta(did="DID-1"), [_ds(did="DID-2")]) == (None, None)
    assert best_match(_meta(did="DID-1"), []) == (None, None)


def test_strength_order_is_did_then_file_then_time():
    assert MATCH_ORDER == (MATCH_DID, MATCH_DATA_FILE, MATCH_ACQUISITION)


def test_dataset_identity_reads_every_place_a_filename_hides():
    ds = SimpleNamespace(
        name="alias.mat",
        file=SimpleNamespace(name="run.mat", recent_path="C:/x/other.mat"),
        metadata={})
    assert dataset_identity(ds)[MATCH_DATA_FILE] == {
        "alias.mat", "run.mat", "other.mat"}


def test_sidecar_identity_tolerates_a_payload_missing_everything():
    assert sidecar_identity({}) == {MATCH_DID: "", MATCH_DATA_FILE: "",
                                    MATCH_ACQUISITION: ""}
    assert sidecar_identity(None)[MATCH_DID] == ""


# ------------------------------------------------------- snapshot / summary

def test_a_snapshot_recipe_is_flagged_because_it_flattens_z_elsewhere():
    """It pins z to 1.0 and carries nothing else, so applying it to other data
    only removes that dataset's Z scaling -- silently, unless we say so."""
    assert is_snapshot_recipe({"content": "snapshot"}) is True
    assert is_snapshot_recipe({"content": "raw"}) is False


def test_the_summary_names_what_would_be_applied():
    meta = _meta(calibration={"z_scaling_factor": 0.6667},
                 filters=[{}, {}], rois=[{}], acquired="2026-01-01T00:00:00+00:00")
    text = recipe_summary(meta)
    assert "0.6667" in text and "2 filter(s)" in text and "1 ROI(s)" in text
    assert recipe_summary({}) == "nothing (empty recipe)"


# ------------------------------------------------------------ the sidecar

def test_the_sidecar_now_records_the_did_so_it_can_find_its_dataset_later():
    """A .mat/.npy/.csv/.json data file cannot carry a DID; the sidecar is the
    only route by which one reaches such a dataset."""
    from minflux_viewer.core.dataset import build_localization_dataset
    from minflux_viewer.core.save import build_metadata

    rng = np.random.default_rng(0)
    ds = build_localization_dataset(name="A", x_nm=rng.random(20) * 100,
                                    y_nm=rng.random(20) * 100)
    assert "msr_dataset_did" not in build_metadata(ds)   # never invented

    ds.metadata["msr_dataset_did"] = "DID-XYZ"
    meta = build_metadata(ds)
    assert meta["msr_dataset_did"] == "DID-XYZ"
    assert json.loads(json.dumps(meta))["msr_dataset_did"] == "DID-XYZ"


def test_applying_a_recipe_carries_the_did_onto_the_dataset():
    from minflux_viewer.core.dataset import build_localization_dataset
    from minflux_viewer.core.loader import apply_metadata_recipe

    rng = np.random.default_rng(0)
    ds = build_localization_dataset(name="A", x_nm=rng.random(20) * 100,
                                    y_nm=rng.random(20) * 100)
    apply_metadata_recipe(ds, _meta(did="DID-XYZ"))
    assert ds.metadata["msr_dataset_did"] == "DID-XYZ"

    # ...but a dataset that already knows its own identity keeps it.
    apply_metadata_recipe(ds, _meta(did="DID-OTHER"))
    assert ds.metadata["msr_dataset_did"] == "DID-XYZ"


# ------------------------------------------- concatenate, never discard work

def test_filters_and_rois_concatenate_onto_what_the_dataset_already_has():
    """A recipe may be applied on top of work in progress; silently dropping
    the user's filters would be the worse failure. A repeat duplicates rows,
    which is visible in the Filter dialog and can be undone there."""
    from minflux_viewer.core.dataset import build_localization_dataset
    from minflux_viewer.core.loader import apply_metadata_recipe

    rng = np.random.default_rng(0)
    ds = build_localization_dataset(name="A", x_nm=rng.random(20) * 100,
                                    y_nm=rng.random(20) * 100)
    ds.state["filter_specs"] = [{"attribute": "idx", "mode": "per loc",
                                 "lo": 0.0, "hi": 5.0}]
    meta = _meta(filters=[{"attribute": "idx", "mode": "per loc",
                           "lo": 1.0, "hi": 9.0}],
                 rois=[{"id": "r1", "type": "rectangle"}])
    apply_metadata_recipe(ds, meta)
    assert len(ds.state["filter_specs"]) == 2
    assert len(ds.metadata["minflux_viewer_roi_records"]) == 1

    # ROIs are keyed by id, so re-applying the same recipe does not duplicate
    # them -- unlike filters, whose duplication is the visible, undoable case.
    apply_metadata_recipe(ds, meta)
    assert len(ds.metadata["minflux_viewer_roi_records"]) == 1
    assert len(ds.state["filter_specs"]) == 3


# --------------------------------------------------------- loader deferral

def test_a_loader_can_be_told_not_to_apply_the_sidecar():
    """Point 1: the UI asks first, so it needs to be able to say no. Headless
    callers keep the old behaviour, which is why True is the default."""
    import inspect

    from minflux_viewer.core import loader

    for name in ("load_dataset", "load_npy", "load_npz", "load_csv", "load_json"):
        signature = inspect.signature(getattr(loader, name))
        parameter = signature.parameters["apply_sidecar"]
        assert parameter.default is True, name
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, name


def test_read_metadata_sidecar_finds_and_parses_without_applying(tmp_path):
    from minflux_viewer.core.loader import read_metadata_sidecar

    data = tmp_path / "run.mat"
    data.write_bytes(b"")
    assert read_metadata_sidecar(data) is None       # nothing beside it

    (tmp_path / "run_metadata.json").write_text(
        json.dumps(_meta(did="DID-1", data_file="run.mat")), encoding="utf-8")
    found = read_metadata_sidecar(data)
    assert found is not None and found[0].name == "run_metadata.json"
    assert found[1]["msr_dataset_did"] == "DID-1"


def test_an_unreadable_sidecar_is_not_a_sidecar_not_an_error(tmp_path):
    """This runs speculatively for every data file opened, so it must not raise."""
    from minflux_viewer.core.loader import read_metadata_sidecar

    data = tmp_path / "run.mat"
    data.write_bytes(b"")
    (tmp_path / "run_metadata.json").write_text("{not json", encoding="utf-8")
    assert read_metadata_sidecar(data) is None


def test_reading_a_sidecar_by_its_own_path_is_not_the_sibling_lookup(tmp_path):
    """⚠ The two readers answer different questions and are easy to confuse.

    ``read_metadata_sidecar`` derives ``<stem>_metadata.json`` from a **data**
    file; handing it a path that already IS the sidecar makes it look for
    ``d_metadata_metadata.json`` and quietly find nothing — which is exactly
    how dropping a recipe on a Dataset Manager row silently did nothing.
    """
    from minflux_viewer.core.loader import read_metadata_file, read_metadata_sidecar

    side = tmp_path / "d_metadata.json"
    side.write_text(json.dumps(_meta(did="DID-1")), encoding="utf-8")

    assert read_metadata_file(side)["msr_dataset_did"] == "DID-1"
    assert read_metadata_sidecar(side) is None          # the trap

    data = tmp_path / "d.mat"
    data.write_bytes(b"")
    assert read_metadata_sidecar(data)[0] == side       # what it is for


def test_read_metadata_file_rejects_a_json_that_is_not_a_recipe(tmp_path):
    from minflux_viewer.core.loader import read_metadata_file

    other = tmp_path / "rois.json"
    other.write_text(json.dumps({"version": 1, "rois": []}), encoding="utf-8")
    assert read_metadata_file(other) is None
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert read_metadata_file(broken) is None
