"""MINFLUX acquisition sequences: recognition, validation, the display model, and
where a recorded sequence travels (sidecar, Viewer Zarr, our .msr).

All sequences here are synthetic -- the vendor's library files are third-party
and are never committed; a real-file check skips without the local folder.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from minflux_viewer.core import formats
from minflux_viewer.core.acquisition_sequence import (
    describe_key,
    is_sequence_json_file,
    load_sequence_file,
    pattern_diameters_nm,
    sequence_overview,
    sequences_for_dataset,
    sequences_from_mfx_attrs,
)

SEQUENCE_DIR = Path(r"D:\Workspace\Microscopes\MINFLUX\Sequences")


def _itr(mode_id, pattern, geo, *, dim=2, pht=100, ccr=-1.0, bgc=10000.0, pwr=1.0,
         dwell=0.001, repeat=5, wavelength=6.42e-7, **extra) -> dict:
    return {"Mode": {"id": mode_id, "dim": dim, "pattern": pattern, "phDiaAU": 1.5,
                     "strategy": "BL"},
            "patGeoFactor": geo, "phtLimit": pht, "ccrLimit": ccr, "bgcThreshold": bgc,
            "pwrFactor": pwr, "patDwellTime": dwell, "patRepeat": repeat,
            "wavelength": wavelength, "estCoeff": [[[0.0]]], "_lemd": [1e-8, 0.0],
            "_lemn": [1e-8, 0.0], **extra}


def imaging_2d() -> dict:
    return {"Itr": [_itr("prbt", "hexagon", 0.8, pht=160, repeat=1),
                    _itr("mflx", "hexagon", 0.8, pht=150),
                    _itr("mflx", "hexagon", 0.42, ccr=0.8, pwr=2.0),
                    _itr("mflx", "hexagon", 0.21, ccr=0.8, pwr=4.0),
                    _itr("mflx", "hexagon", 0.11, pht=150, ccr=2.0, pwr=6.0)],
            "bgcSense": True, "ctrDwellFactor": 1.0, "damping": 0,
            "field": {"algo": "hexgrid", "fldGeoFactor": 0.5, "stride": 1},
            "headstart": -2, "id": "Imaging_2D", "locLimit": -1, "stickiness": 2,
            "liveview": {"show": ["loc"]}}


def imaging_3d() -> dict:
    itrs = [_itr("pt3d", "hexagon", 0.8, dim=3, maxOffTime=0.003),
            _itr("mxax", "zline", [0.8, 4.0], dim=3, maxOffTime=0.003)]
    for geo in (0.8, 0.42, 0.21, 0.11):
        itrs.append(_itr("mxl3", "square", geo, dim=3, maxOffTime=0.003))
        itrs.append(_itr("mxa3", "zline2", geo, dim=3, maxOffTime=0.003))
    return {"Itr": itrs, "bgcSense": True, "headstart": -4, "id": "Imaging_3D"}


def tracking_octahedron() -> dict:
    return {"Itr": [_itr("pt3d", "hexagon", 0.8, dim=3),
                    _itr("mxax", "zline", [0.8, 4.0], dim=3),
                    _itr("mx3d", "octahedron", 0.8, dim=3),
                    _itr("mx3d", "octahedron", 0.42, dim=3, ccr=0.9),
                    _itr("mx3d", "octahedron", 0.28, dim=3)],
            "bgcSense": False, "headstart": -1, "id": "Tracking_3D", "maxOffTime": "unspecified"}


def _write(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# -------------------------------------------------------------- recognition
def test_a_sequence_file_opens_the_sequence_viewer(tmp_path):
    path = _write(tmp_path / "seq.json", imaging_2d())
    assert is_sequence_json_file(path)
    spec = formats.resolve_open(path)
    assert spec.key == "sequence" and spec.action is formats.OpenAction.SEQUENCE_VIEWER
    assert spec.folder_scan is False


def test_a_damaged_sequence_is_still_routed_to_the_sequence_loader(tmp_path):
    """Recognised by shape, so its loader can name the problem; the
    localization loader would only say what it is not."""
    damaged = imaging_2d()
    del damaged["Itr"][1]["Mode"]
    path = _write(tmp_path / "bad.json", damaged)
    assert formats.resolve_open(path).key == "sequence"
    with pytest.raises(ValueError, match="iteration 2 has no 'Mode'"):
        load_sequence_file(path)


@pytest.mark.parametrize("payload, message", [
    ([], "a sequence is a JSON object"),
    ({"Itr": []}, "non-empty list"),
    ({"Itr": [1]}, "iteration 1 is not an object"),
    ({"Itr": [{"Mode": {"dim": 2}}]}, "'Mode' without an 'id'"),
])
def test_validation_says_what_is_wrong(tmp_path, payload, message):
    with pytest.raises(ValueError, match=message):
        load_sequence_file(_write(tmp_path / "v.json", payload))


def test_stronger_markers_win_a_conflict(tmp_path):
    sidecar = {"minflux_viewer_metadata": 1, "Itr": []}
    roi_set = {"rois": [], "Itr": []}
    assert not is_sequence_json_file(_write(tmp_path / "s.json", sidecar))
    assert formats.resolve_open(tmp_path / "s.json").key == "metadata_sidecar"
    assert not is_sequence_json_file(_write(tmp_path / "r.json", roi_set))
    assert formats.resolve_open(tmp_path / "r.json").key == "roi_set"


def test_localization_data_is_never_a_sequence(tmp_path):
    records = [{"loc": [0.0, 0.0, 0.0], "itr": 0, "vld": True, "tid": 1}]
    assert not is_sequence_json_file(_write(tmp_path / "d.json", records))
    assert not is_sequence_json_file(_write(tmp_path / "w.json", {"mfx": records}))


# ------------------------------------------------------------------- values
def test_pattern_diameter_follows_the_documented_formula():
    assert pattern_diameters_nm(0.11, 6.42e-7) == pytest.approx([39.6])
    assert pattern_diameters_nm(0.8, 6.42e-7) == pytest.approx([288.0])
    assert pattern_diameters_nm([0.8, 4.0], 6.42e-7) == pytest.approx([288.0, 1440.0])
    assert pattern_diameters_nm(0.79, 6.8e-7) == pytest.approx([0.79 * 360 * 680 / 642])
    assert pattern_diameters_nm(0.11, None) == pytest.approx([39.6])     # 642 nm assumed


def test_imaging_2d_overview():
    overview = sequence_overview(imaging_2d())
    cells = [row.cells for row in overview.iterations]
    assert [c["l_nm"] for c in cells] == ["288.0", "288.0", "151.2", "75.6", "39.6"]
    assert [c["ccr"] for c in cells] == [
        "not measured", "not measured", "measured, > 0.8 rejected",
        "measured, > 0.8 rejected", "measured, not filtered (2)"]
    assert cells[0]["bgc"] == "10000 Hz above sensed background"
    assert cells[0]["max_off"] == "default 3 ms (not set)"
    assert [row.final for row in overview.iterations] == ["", "", "", "", "lateral"]
    assert overview.loop_start == 2                      # headstart -2 of 5 -> 3rd
    assert [row.in_loop for row in overview.iterations] == [False, False, True, True, True]
    headstart = dict((label, value) for label, value, _key in overview.globals)["Headstart"]
    assert headstart.startswith("-2") and "iteration 3" in headstart


def test_imaging_3d_final_lateral_and_axial():
    overview = sequence_overview(imaging_3d())
    finals = {row.index + 1: row.final for row in overview.iterations if row.final}
    assert finals == {9: "lateral", 10: "axial"}
    assert overview.iterations[1].cells["l_nm"] == "288.0 / 1440.0"
    assert overview.iterations[0].cells["max_off"] == "3 ms"


def test_octahedron_is_final_for_both_axes_and_unspecified_is_shown_as_recorded():
    overview = sequence_overview(tracking_octahedron())
    assert overview.iterations[-1].final == "lateral + axial"
    rows = {label: value for label, value, _key in overview.globals}
    assert rows["Max off-time (sequence-wide)"] == "unspecified"


def test_an_unknown_pattern_is_not_presented_as_lateral():
    sequence = imaging_2d()
    sequence["Itr"][-1]["Mode"]["pattern"] = "spiral"
    overview = sequence_overview(sequence)
    assert overview.iterations[-1].cells["role"] == "unknown"
    assert overview.iterations[-1].final == ""           # not counted as a final iteration


def test_a_positive_headstart_is_not_interpreted():
    sequence = dict(imaging_2d(), headstart=2)
    overview = sequence_overview(sequence)
    assert overview.loop_start is None
    headstart = dict((label, value) for label, value, _key in overview.globals)["Headstart"]
    assert "not documented" in headstart


def test_the_documented_keys_are_described():
    """Every key of the standard files that the wiki documents has a text."""
    sequence = imaging_2d()
    for key in ("id", "Itr", "bgcSense", "ctrDwellFactor", "damping", "headstart",
                "locLimit", "stickiness", "liveview", "field"):
        assert describe_key((key,)), key
    for key in ("Mode", "patGeoFactor", "phtLimit", "ccrLimit", "bgcThreshold",
                "patDwellTime", "patRepeat", "pwrFactor", "wavelength", "estCoeff",
                "_lemd", "_lemn"):
        assert describe_key(("root", "Itr", "[0]", key)), key
    for key in sequence["Itr"][0]["Mode"]:
        assert describe_key(("Itr", "[0]", "Mode", key)), key
    assert describe_key(("Itr", "[0]", "fldFactor")) == ""        # undocumented: no text


# ------------------------------------------------------------- per thread
def native_record() -> dict:
    """An m2410 ``mfx/.zattrs`` with an imaging and an MBM thread."""
    mbm = dict(imaging_2d(), id="seqMBM2d_v8")
    return {
        "acquisition_date": "2026-04-21T13:09:33+02:00", "did": "a-did", "version": 1,
        "scan_range": {"x": 1},
        "measurement": {"dimensionality": 2, "threads": [
            {"id": 0, "grid": "search_0", "tag": "", "sequences": [imaging_2d()],
             "detectors": [{"id": "GCM1", "expcontrol_name": "Cy5_far", "active_use": True}],
             "lasers": [{"id": "MINFLUX_642", "wavelength": 6.42e-7, "active_use": True}]},
            {"id": 1, "grid": "mbm", "tag": "mbm:0", "sequences": [mbm]},
        ]},
    }


def test_every_thread_is_listed_with_its_set_up():
    sources = sequences_from_mfx_attrs(native_record(), origin="ds")
    assert [s.label for s in sources] == ["Thread 0: Imaging_2D (grid search_0)",
                                          "Thread 1: seqMBM2d_v8 (grid mbm)"]
    details = dict(sources[0].details)
    assert details["Detectors"] == "GCM1 (Cy5_far)"
    assert details["Lasers"] == "MINFLUX_642 (642 nm)"
    assert dict(sources[1].details)["Tag"] == "mbm:0"


def test_an_m2205_record_lists_its_one_sequence():
    sources = sequences_from_mfx_attrs({"seq": imaging_2d(), "tms": 1.0})
    assert [s.label for s in sources] == ["Imaging_2D"]


class _Ds:
    def __init__(self, metadata, name="ds"):
        self.metadata = metadata
        self.name = name


def test_a_dataset_shows_its_native_record_else_its_stamped_sequence_else_nothing():
    assert len(sequences_for_dataset(_Ds({"native_zarr_mfx_attrs": native_record()}))) == 2
    assert [s.label for s in sequences_for_dataset(_Ds({"mfx_sequence": imaging_2d()}))] == [
        "Imaging_2D"]
    assert sequences_for_dataset(_Ds({})) == []


# ------------------------------------------------------- where it travels
def _m2410_dataset(native: dict | None, name: str = "acq.npy"):
    from minflux_viewer.core.loader_format_readers import load_from_mfx_array
    from tests.test_format_equivalence import synthetic_m2410

    ds = load_from_mfx_array(synthetic_m2410(), name=name)
    if native is not None:
        ds.metadata["native_zarr_mfx_attrs"] = copy.deepcopy(native)
        ds.metadata["msr_dataset_did"] = native.get("did", "")
    return ds


def test_the_sidecar_carries_the_native_record_only_when_there_is_one():
    from minflux_viewer.core.save import NATIVE_ACQUISITION_KEY, build_metadata

    with_record = build_metadata(_m2410_dataset(native_record()), data_filename="acq.npy")
    assert with_record[NATIVE_ACQUISITION_KEY]["mfx_attrs"]["measurement"]["threads"]
    without = build_metadata(_m2410_dataset(None), data_filename="acq.npy")
    assert NATIVE_ACQUISITION_KEY not in without


def test_the_record_is_restored_onto_its_own_acquisition_only():
    from minflux_viewer.core.loader_metadata_recipe import apply_metadata_recipe
    from minflux_viewer.core.save import build_metadata

    recipe = build_metadata(_m2410_dataset(native_record()), data_filename="acq.npy")

    same_file = _m2410_dataset(None, name="acq.npy")           # written beside this file
    applied = apply_metadata_recipe(same_file, recipe)
    assert len(sequences_for_dataset(same_file)) == 2
    assert same_file.metadata["mfx_sequence"]["id"] == "Imaging_2D"
    assert any(text.startswith("acquisition record") for text in applied)

    other = _m2410_dataset(None, name="other.npy")             # a different acquisition
    applied = apply_metadata_recipe(other, recipe)
    assert sequences_for_dataset(other) == []
    assert any("NOT restored" in text for text in applied)

    own = native_record()
    own["measurement"]["threads"] = own["measurement"]["threads"][:1]
    keeps = _m2410_dataset(own, name="acq.npy")                # already has its own record
    apply_metadata_recipe(keeps, recipe)
    assert len(sequences_for_dataset(keeps)) == 1


def _recipe(**changes) -> dict:
    from minflux_viewer.core.save import build_metadata

    recipe = build_metadata(_m2410_dataset(native_record()), data_filename="run.npy")
    recipe.update(changes)
    return recipe


def _restored(ds) -> bool:
    return bool(ds.metadata.get("native_zarr_mfx_attrs"))


def test_a_conflicting_acquisition_id_vetoes_a_matching_file_name():
    """Review 2, C7: a shared basename (run.npy in every folder) is not identity."""
    from minflux_viewer.core.loader_metadata_recipe import apply_metadata_recipe

    other = _m2410_dataset(None, name="run.npy")
    other.metadata["msr_dataset_did"] = "b-did"                 # its own, intrinsic
    applied = apply_metadata_recipe(other, _recipe())
    assert not _restored(other)
    assert any("NOT restored" in t and "a-did" in t and "b-did" in t for t in applied)


def test_identity_copied_from_a_recipe_is_never_evidence():
    """Review 2, C7: applying an unrelated recipe twice must not restore it the
    second time, although the first application copied the recipe's ID."""
    from minflux_viewer.core.loader_metadata_recipe import apply_metadata_recipe
    from minflux_viewer.core.metadata_match import RECIPE_IDENTITY_KEY, match_kind

    recipe = _recipe(acquisition={"date": "2026-04-21T13:09:33+02:00"})
    unrelated = _m2410_dataset(None, name="different.npy")
    apply_metadata_recipe(unrelated, copy.deepcopy(recipe))
    assert not _restored(unrelated)
    assert unrelated.metadata["msr_dataset_did"] == "a-did"     # copied, as before ...
    assert set(unrelated.metadata[RECIPE_IDENTITY_KEY]) == {"msr_dataset_did",
                                                             "acquisition_date"}
    assert match_kind(recipe, unrelated) == "did"               # ... still a dropdown default
    applied = apply_metadata_recipe(unrelated, copy.deepcopy(recipe))
    assert not _restored(unrelated)                             # ... but never evidence
    assert any("NOT restored" in t for t in applied)


def test_the_sidecar_and_its_record_must_name_the_same_acquisition():
    from minflux_viewer.core.metadata_match import acquisition_record_match
    from minflux_viewer.core.save import NATIVE_ACQUISITION_KEY

    recipe = _recipe(msr_dataset_did="z-did")                   # record says a-did
    ds = _m2410_dataset(None, name="run.npy")
    match = acquisition_record_match(recipe, ds, recipe[NATIVE_ACQUISITION_KEY]["mfx_attrs"])
    assert not match.accepted and "two different acquisition IDs" in match.reason


@pytest.mark.parametrize("date, accepted", [
    ("2026-04-21T11:09:33+00:00", True),                        # same instant, other zone
    ("2026-05-01T09:00:00+02:00", False),                       # a different acquisition
])
def test_the_acquisition_time_verifies_or_vetoes(date, accepted):
    from minflux_viewer.core.loader_metadata_recipe import apply_metadata_recipe
    from minflux_viewer.core.save import NATIVE_PROVENANCE_KEY

    ds = _m2410_dataset(None, name="run.npy")                   # same file name either way
    ds.metadata["acquisition_date"] = date                      # intrinsic
    apply_metadata_recipe(ds, _recipe())
    assert _restored(ds) is accepted
    if accepted:
        assert ds.metadata[NATIVE_PROVENANCE_KEY]["matched_by"] == "acquisition"
        assert ds.metadata[NATIVE_PROVENANCE_KEY]["verified"] is True


def test_applying_a_refused_recipe_again_is_still_refused():
    """Final review, finding 1: the first application overwrote the dataset's
    own date (and marked it recipe-supplied), so the second one fell through to
    the file name and restored the record. The replaced date is kept."""
    from minflux_viewer.core.loader_metadata_recipe import apply_metadata_recipe
    from minflux_viewer.core.metadata_match import IDENTITY_BEFORE_RECIPE_KEY
    from minflux_viewer.core.save import NATIVE_ACQUISITION_KEY

    own, theirs = "2026-01-01T12:00:00+00:00", "2026-02-01T12:00:00+00:00"
    ds = _m2410_dataset(None, name="run.npy")
    ds.metadata["acquisition_date"] = own
    recipe = _recipe(acquisition={"date": theirs})
    recipe[NATIVE_ACQUISITION_KEY]["mfx_attrs"]["acquisition_date"] = theirs
    for _ in range(3):
        applied = apply_metadata_recipe(ds, copy.deepcopy(recipe))
        assert not _restored(ds)
        assert any("NOT restored" in t and own in t for t in applied)
    assert ds.metadata[IDENTITY_BEFORE_RECIPE_KEY] == {"acquisition_date": own}
    assert ds.metadata["acquisition_date"] != own        # the recipe's date, as before


def test_an_agreeing_id_does_not_excuse_a_conflicting_time():
    """One contract: any conflict vetoes (final review, DID vs time)."""
    from minflux_viewer.core.metadata_match import acquisition_record_match
    from minflux_viewer.core.save import NATIVE_ACQUISITION_KEY

    recipe = _recipe(acquisition={"date": "2026-04-21T13:09:33+02:00"})
    ds = _m2410_dataset(None, name="run.npy")
    ds.metadata["msr_dataset_did"] = "a-did"                 # agrees
    ds.metadata["acquisition_date"] = "2026-05-01T09:00:00+02:00"   # conflicts
    match = acquisition_record_match(recipe, ds, recipe[NATIVE_ACQUISITION_KEY]["mfx_attrs"])
    assert not match.accepted and "different acquisition" in match.reason
    ds.metadata["acquisition_date"] = "2026-04-21T11:09:33Z"   # same instant
    match = acquisition_record_match(recipe, ds, recipe[NATIVE_ACQUISITION_KEY]["mfx_attrs"])
    assert match.basis == "did" and match.verified


@pytest.mark.parametrize("theirs, ours, same", [
    ("2026-04-21T13:09:33+02:00", "2026-04-21T13:09:33", True),    # zone dropped
    ("2026-04-21T13:09:33+02:00", "2026-04-21T11:09:33", True),    # naive UTC
    ("2026-04-21T13:09:33+02:00", "2026-04-21T15:00:00", False),
    ("2026-04-21T13:09:33+02:00", "2026-04-21T11:09:33+00:00", True),
])
def test_acquisition_times_compare_as_instants(theirs, ours, same):
    from minflux_viewer.core.metadata_match import _same_time

    assert _same_time(theirs, ours) is same


@pytest.mark.parametrize("sidecar, record, dataset, accepted", [
    # Re-check of stage 9: the zone-less sidecar agrees with both, the record
    # and the dataset disagree by two hours -- refused.
    ("2026-01-01T12:00:00", "2026-01-01T12:00:00+02:00", "2026-01-01T12:00:00+00:00", False),
    ("2026-01-01T12:00:00+02:00", "2026-01-01T12:00:00", "2026-01-01T12:00:00+00:00", False),
    ("2026-01-01T12:00:00+00:00", "2026-01-01T12:00:00+02:00", "2026-01-01T12:00:00", False),
    # All three agree: accepted.
    ("2026-01-01T12:00:00", "2026-01-01T12:00:00+00:00", "2026-01-01T12:00:00+00:00", True),
])
@pytest.mark.parametrize("dataset_did", ["", "a-did"])          # absent, or agreeing
def test_every_recorded_time_is_compared_with_the_dataset(sidecar, record, dataset,
                                                          accepted, dataset_did):
    """Times agree pairwise or the record is refused, even when the IDs agree:
    a time without a zone agrees with two zoned times that differ."""
    from minflux_viewer.core.loader_metadata_recipe import apply_metadata_recipe
    from minflux_viewer.core.metadata_match import acquisition_record_match
    from minflux_viewer.core.save import NATIVE_ACQUISITION_KEY

    recipe = _recipe(acquisition={"date": sidecar})
    recipe[NATIVE_ACQUISITION_KEY]["mfx_attrs"]["acquisition_date"] = record

    def target():
        ds = _m2410_dataset(None, name="target.npy")        # no file-name match
        ds.metadata["acquisition_date"] = dataset
        if dataset_did:
            ds.metadata["msr_dataset_did"] = dataset_did
        return ds

    ds = target()
    match = acquisition_record_match(recipe, ds, recipe[NATIVE_ACQUISITION_KEY]["mfx_attrs"])
    assert match.accepted is accepted
    if not accepted:
        assert "different acquisition" in match.reason
    else:
        assert match.basis == ("did" if dataset_did else "acquisition")
    ds = target()
    applied = apply_metadata_recipe(ds, copy.deepcopy(recipe))
    assert _restored(ds) is accepted
    assert any("NOT restored" in t for t in applied) is (not accepted)


def test_a_file_name_match_is_accepted_and_recorded_as_unverified(tmp_path):
    from minflux_viewer.core.loader_metadata_recipe import apply_metadata_recipe
    from minflux_viewer.core.minflux_zarr import load_minflux_zarr
    from minflux_viewer.core.save import (
        NATIVE_ACQUISITION_KEY,
        NATIVE_PROVENANCE_KEY,
        build_metadata,
        save_processed,
    )

    ds = _m2410_dataset(None, name="run.npy")
    applied = apply_metadata_recipe(ds, _recipe())
    assert _restored(ds)
    assert any("file name only" in t for t in applied)
    provenance = ds.metadata[NATIVE_PROVENANCE_KEY]
    assert (provenance["matched_by"], provenance["verified"]) == ("data_file", False)
    # It travels with the record: the next sidecar and the Viewer Zarr keep it.
    again = build_metadata(ds, data_filename="run.npy")
    assert again[NATIVE_ACQUISITION_KEY]["provenance"]["verified"] is False
    save_processed(ds, data_path=tmp_path / "run.zarr", fmt="zarr")
    assert load_minflux_zarr(tmp_path / "run.zarr").metadata[NATIVE_PROVENANCE_KEY] == provenance


def test_viewer_zarr_keeps_every_thread(tmp_path):
    from minflux_viewer.core.minflux_zarr import load_minflux_zarr
    from minflux_viewer.core.save import save_processed

    ds = _m2410_dataset(native_record())
    save_processed(ds, data_path=tmp_path / "acq.zarr", fmt="zarr")
    back = load_minflux_zarr(tmp_path / "acq.zarr")
    assert [s.label for s in sequences_for_dataset(back)] == [
        s.label for s in sequences_for_dataset(ds)]


def test_our_msr_writes_the_native_record_back(tmp_path):
    from minflux_viewer.msr.writer import MsrChannel, build_channel_store
    from minflux_viewer.utils.zarr2 import read_zarr_attrs
    from tests.test_format_equivalence import synthetic_m2410

    record = native_record()
    channel = MsrChannel(mfx=synthetic_m2410(), did="our-did", native_mfx_attrs=record,
                         acquisition_date="2026-01-01T00:00:00+00:00")
    attrs = read_zarr_attrs(build_channel_store(channel, "our-did"), "mfx")
    assert attrs["measurement"] == record["measurement"]
    assert attrs["version"] == 1 and attrs["scan_range"] == {"x": 1}
    assert attrs["did"] == "our-did"                             # our keys win
    assert attrs["acquisition_date"] == "2026-01-01T00:00:00+00:00"


def test_an_m2205_record_is_not_written_into_our_m2410_store():
    from minflux_viewer.msr.writer import MsrChannel, build_channel_store
    from minflux_viewer.utils.zarr2 import read_zarr_attrs
    from tests.test_format_equivalence import synthetic_m2410

    channel = MsrChannel(mfx=synthetic_m2410(), native_mfx_attrs={"seq": imaging_2d(), "tms": 1})
    attrs = read_zarr_attrs(build_channel_store(channel, "d"), "mfx")
    assert "seq" not in attrs and "tms" not in attrs


# ------------------------------------------------------------ folder drop
def test_a_dropped_folder_opens_the_data_and_names_the_sequences(tmp_path):
    from minflux_viewer.ui.controllers.file_open_controller import _route_path

    _write(tmp_path / "data.json", [{"loc": [0.0, 0.0, 0.0], "itr": 0, "vld": True, "tid": 1}])
    _write(tmp_path / "Imaging_2D.json", imaging_2d())

    class _Owner:
        def __init__(self):
            self.routed: list[list[str]] = []
            self.logs: list[str] = []
            self._state = self
            self._status_label = self

        def route_paths(self, paths):
            self.routed.append([Path(p).name for p in paths])

        def log(self, message, *_args, **_kwargs):
            self.logs.append(message)

        def setText(self, _text):
            pass

    owner = _Owner()
    _route_path(owner, str(tmp_path))
    assert owner.routed == [["data.json"]]
    assert any("Imaging_2D.json" in line and "not opened" in line for line in owner.logs)


# --------------------------------------------------------------- real files
def test_the_library_sequences_load_and_resolve():
    files = sorted(SEQUENCE_DIR.glob("*.json")) if SEQUENCE_DIR.is_dir() else []
    if not files:
        pytest.skip("local sequence library not available")
    for path in files:
        assert formats.resolve_open(path).key == "sequence", path.name
        overview = sequence_overview(load_sequence_file(path))
        assert overview.iterations[-1].final, path.name
        assert np.isclose(float(overview.iterations[-1].cells["l_nm"].split()[0]),
                          float(load_sequence_file(path)["Itr"][-1]["patGeoFactor"]) * 360,
                          rtol=1e-9), path.name
