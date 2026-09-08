"""Our ``.msr`` as a complete MINFLUX Viewer document.

Verified in Imspector 16.3 (m2410): a file written this way opens with its
images and all MINFLUX datasets, and the embedded ``viewer`` group does not
disturb it. What these pin down is the two halves that made that possible --
reusing the source measurement's container, and carrying the processing state
inside each channel's zarr store.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

SAMPLE = Path(r"D:\Workspace\Microscopes\MINFLUX\sample data\2_3C_measurement.msr")


# --------------------------------------------------------------- pure payload
def _dataset(name="ch", did="did-1", **state):
    from minflux_viewer.core.loader import build_localization_dataset

    rng = np.random.default_rng(0)
    ds = build_localization_dataset(name=name, x_nm=rng.normal(0, 100, 50),
                                    y_nm=rng.normal(0, 100, 50))
    ds.metadata["msr_dataset_did"] = did
    ds.state.update(state)
    return ds


def test_state_payload_is_self_identifying_and_json_clean():
    """A reader must know what it found from a marker, not from the shape."""
    from minflux_viewer.msr import viewer_state as vs

    ds = _dataset(overlay_id="g", overlay_index=0, overlay_lut="Red")
    ds.state["overlay_transform"] = {"matrix_4x4": np.eye(4).tolist()}
    payload = vs.dataset_viewer_state(ds, manifest={"name": "x", "members": []})
    assert payload["format"] == vs.STATE_FORMAT and payload["schema"] == vs.STATE_SCHEMA
    json.dumps(payload)                          # no numpy leaks


def test_bulky_and_native_metadata_stays_out_of_the_json():
    """Bead points and the vendor's own attrs are arrays / restored from the
    file itself; carrying them as JSON would inflate every channel."""
    from minflux_viewer.msr import viewer_state as vs

    ds = _dataset()
    ds.metadata["mbm_points"] = np.zeros(1000)
    ds.metadata["native_zarr_mfx_attrs"] = {"big": "x" * 10_000}
    ds.metadata["keep_me"] = 7
    payload = vs.dataset_viewer_state(ds)
    assert "mbm_points" not in payload["metadata"]
    assert "native_zarr_mfx_attrs" not in payload["metadata"]
    assert payload["metadata"]["keep_me"] == 7


def test_apply_restores_state_but_not_this_file_s_identity():
    """The source path and DID were just stamped from the file actually being
    read; a stale copy in the payload must not overwrite them."""
    from minflux_viewer.msr import viewer_state as vs

    ds = _dataset(overlay_id="g")
    ds.state["filter_specs"] = [{"attribute": "efo"}]
    payload = vs.dataset_viewer_state(ds)
    payload["metadata"]["msr_source_path"] = "/somewhere/else.msr"

    fresh = _dataset(name="fresh", did="did-2")
    fresh.metadata["msr_source_path"] = "/actual/file.msr"
    applied = vs.apply_viewer_state(fresh, payload)
    assert fresh.state["overlay_id"] == "g"
    assert len(fresh.state["filter_specs"]) == 1
    assert fresh.metadata["msr_source_path"] == "/actual/file.msr"
    assert fresh.metadata["msr_dataset_did"] == "did-2"
    assert applied


def test_unmarked_payload_is_ignored_rather_than_guessed():
    from minflux_viewer.msr import viewer_state as vs

    ds = _dataset()
    assert vs.apply_viewer_state(ds, {"state": {"overlay_id": "nope"}}) == []
    assert vs.apply_viewer_state(ds, None) == []
    assert "overlay_id" not in ds.state


def test_project_manifest_lets_one_channel_describe_the_group():
    from minflux_viewer.msr import viewer_state as vs

    members = [_dataset(name=f"c{i}", did=f"d{i}", overlay_index=i,
                        overlay_id="g", overlay_lut=c)
               for i, c in enumerate(("Red", "Green", "Blue"))]
    manifest = vs.build_project_manifest(members, name="run")
    assert manifest["overlay_id"] == "g"
    assert [m["name"] for m in manifest["members"]] == ["c0", "c1", "c2"]
    assert [m["overlay_lut"] for m in manifest["members"]] == ["Red", "Green", "Blue"]


# --------------------------------------------------------------- the DID bug
def test_channel_keeps_the_acquisition_did():
    """``resolved_did`` used to mint a fresh UUID on every save, because
    ``channel_from_dataset`` read ``metadata['did']`` while the MSR reader
    stamps ``msr_dataset_did``. That lost the acquisition identity the sidecar
    matches on, and left the container transplant with no stack to replace."""
    from minflux_viewer.msr.writer import channel_from_dataset

    ds = _dataset(name="ELYS", did="25231766-3300-413e-896e-2c2a8f35e3fb")
    channel = channel_from_dataset(ds)
    assert channel.resolved_did() == "25231766-3300-413e-896e-2c2a8f35e3fb"
    assert channel.name == "ELYS"


def test_no_source_msr_means_no_template():
    """Without a template there is no measurement object, so the caller has to
    know it is producing a viewer-only file."""
    from minflux_viewer.msr.transplant import template_for

    assert template_for([_dataset()]) is None
    a, b = _dataset(), _dataset()
    a.metadata["msr_source_path"] = "/one.msr"
    b.metadata["msr_source_path"] = "/other.msr"
    assert template_for([a, b]) is None          # two sources, no single container


def test_square_padding_matches_the_vendor_layout():
    """Imspector declares a MINFLUX stack near-square; 13,003,236 bytes is
    exactly 3606x3606 in the reference file."""
    from minflux_viewer.msr.transplant import square_pad

    padded, side = square_pad(b"x" * 13_003_236)
    assert side == 3606 and len(padded) == 3606 * 3606
    padded, side = square_pad(b"x" * 10)
    assert len(padded) == side * side >= 10


# ------------------------------------------------------- real file round trip
@pytest.mark.skipif(not SAMPLE.is_file(), reason="reference .msr not present")
def test_round_trip_keeps_the_imspector_container_and_the_state(tmp_path):
    from minflux_viewer.core.loader import load_from_mfx_array
    from minflux_viewer.core.save import save_processed
    from minflux_viewer.msr.msr_parser import parse_general
    from minflux_viewer.msr.viewer_state import apply_viewer_state, read_viewer_state
    from msr_reader.obffile import OBFFile, _read_file_header

    parsed = parse_general(str(SAMPLE), str(tmp_path), log=lambda *_a: None)
    datasets = []
    for i, entry in enumerate(parsed["datasets"]):
        ds = load_from_mfx_array(entry["_mfx"], name=entry["display_name"],
                                 prefs={"data": {}})
        ds.metadata["msr_source_path"] = str(SAMPLE)
        ds.metadata["msr_dataset_did"] = entry.get("did")
        ds.state.update({"overlay_id": "g", "overlay_index": i,
                         "overlay_lut": ["Red", "Green", "Blue"][i]})
        ds.set_z_scaling_factor(0.67, source="test")
        ds.state["filter_specs"] = [{"attribute": "efo", "mode": "per loc",
                                     "itr": "last", "lo": 1e4, "hi": 5e5,
                                     "lo_inc": True, "hi_inc": True}]
        datasets.append(ds)

    out = tmp_path / "rt.msr"
    save_processed(datasets[0], data_path=out, fmt="msr", content="raw",
                   include={"attrs": True, "derived": False, "recipe": False},
                   related_datasets=datasets,
                   roi_records=[{"id": "r1", "name": "rect", "type": "rectangle",
                                 "geometry": {"bounds": [0, 0, 10, 10], "angle": 0.0},
                                 "dataset_id": "d000000"}])

    # The container Imspector accepts: its measurement object and every stack.
    with open(out, "rb") as fd:
        head = _read_file_header(fd)
    assert head.format_version == 2
    assert head.first_stack_position > 200_000, "measurement object missing"
    assert OBFFile(str(out)).num_stacks == 36, "image stacks dropped"

    back = parse_general(str(out), str(tmp_path), log=lambda *_a: None)
    assert len(back["datasets"]) == 3
    for i, entry in enumerate(back["datasets"]):
        ds = load_from_mfx_array(entry["_mfx"], name=entry["display_name"],
                                 prefs={"data": {}})
        state = read_viewer_state(entry.get("zroot"))
        assert state, "no embedded viewer state"
        apply_viewer_state(ds, state)
        assert ds.state["overlay_id"] == "g"
        assert ds.state["overlay_index"] == i
        assert abs(ds.cali.z_scaling_factor - 0.67) < 1e-9
        assert len(ds.state["filter_specs"]) == 1
        assert len(state["project"]["members"]) == 3
    first = back["datasets"][0]
    assert read_viewer_state(first["zroot"])["rois"], "ROIs lost"


@pytest.mark.skipif(not SAMPLE.is_file(), reason="reference .msr not present")
def test_identity_transplant_reproduces_the_source_byte_for_byte(tmp_path):
    """The control that made the Imspector bisect meaningful.

    ⚠ ``stack_end_disk`` is not "where the next stack starts" -- Imspector
    leaves slack after each footer. Recomputing it corrupted 142 bytes across
    36 stacks and made every test file fail for the wrong reason.
    """
    from minflux_viewer.msr.transplant import transplant_payloads

    out = tmp_path / "identity.msr"
    transplant_payloads(SAMPLE, {}, out)
    assert out.read_bytes() == SAMPLE.read_bytes()


# ------------------------------------------------------- opening our own .msr
def test_the_viewer_group_is_described_rather_than_called_unknown():
    """It is ours, so pointing the user at Abberior documentation is wrong."""
    from minflux_viewer.msr.descriptions import VIEWER_PATH, describe_dtype, describe_path

    text = describe_path(VIEWER_PATH)
    assert "MINFLUX Viewer" in text and "Abberior documentation" not in text
    assert describe_dtype(VIEWER_PATH, None) == "MINFLUX Viewer processing state"
    # anything below it is ours too, whatever it is called
    assert describe_path(VIEWER_PATH + "/state/anything") == text
    assert "Abberior documentation" in describe_path("something_else")


def test_detection_is_a_scan_not_a_parse(tmp_path):
    """Deciding the route must not cost a full parse of a multi-GB acquisition."""
    from minflux_viewer.msr.quick_open import is_viewer_msr
    from minflux_viewer.msr.viewer_state import STATE_FORMAT

    plain = tmp_path / "plain.msr"
    plain.write_bytes(b"OMAS_BF\n\xff\xff" + b"\x00" * 4096)
    assert not is_viewer_msr(plain)

    ours = tmp_path / "ours.msr"
    ours.write_bytes(b"\x00" * 1024 + STATE_FORMAT.encode() + b"\x00" * 1024)
    assert is_viewer_msr(ours)

    assert not is_viewer_msr(tmp_path / "absent.msr")       # unreadable is not ours
    empty = tmp_path / "empty.msr"
    empty.write_bytes(b"")
    assert not is_viewer_msr(empty)                          # mmap of 0 bytes


@pytest.mark.skipif(not SAMPLE.is_file(), reason="reference .msr not present")
def test_a_vendor_msr_still_goes_to_the_reader():
    """Only a file carrying our own marker may bypass the import dialog."""
    from minflux_viewer.msr.quick_open import is_viewer_msr

    assert not is_viewer_msr(SAMPLE)


@pytest.mark.skipif(not SAMPLE.is_file(), reason="reference .msr not present")
def test_direct_open_restores_the_session(tmp_path):
    from minflux_viewer.core.loader import load_from_mfx_array
    from minflux_viewer.core.save import save_processed
    from minflux_viewer.msr.msr_parser import parse_general
    from minflux_viewer.msr.quick_open import is_viewer_msr, load_viewer_msr

    parsed = parse_general(str(SAMPLE), str(tmp_path), log=lambda *_a: None)
    datasets = []
    for i, entry in enumerate(parsed["datasets"]):
        ds = load_from_mfx_array(entry["_mfx"], name=entry["display_name"],
                                 prefs={"data": {}})
        ds.metadata["msr_source_path"] = str(SAMPLE)
        ds.metadata["msr_dataset_did"] = entry.get("did")
        ds.state.update({"overlay_id": "g", "overlay_index": i,
                         "overlay_lut": ["Red", "Green", "Blue"][i]})
        ds.set_z_scaling_factor(0.67, source="test")
        datasets.append(ds)

    out = tmp_path / "session.msr"
    save_processed(datasets[0], data_path=out, fmt="msr", content="raw",
                   include={"attrs": True, "derived": False, "recipe": False},
                   related_datasets=datasets)
    assert is_viewer_msr(out)

    back = load_viewer_msr(out, prefs={"data": {}})
    assert len(back) == 3
    for i, ds in enumerate(back):
        assert ds.state["overlay_index"] == i
        assert ds.state["overlay_lut"] == ["Red", "Green", "Blue"][i]
        assert abs(ds.cali.z_scaling_factor - 0.67) < 1e-9
        # provenance the shared stamper records
        assert ds.metadata["msr_source_path"] == str(out)
        assert ds.metadata["msr_dataset_did"] == datasets[i].metadata["msr_dataset_did"]
        assert ds.metadata.get("is_minflux") is True


# ------------------------------------------------- the MainWindow route itself
@pytest.fixture
def _app():
    pytest.importorskip("PyQt6")
    import sys
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


@pytest.mark.skipif(not SAMPLE.is_file(), reason="reference .msr not present")
def test_dropping_our_msr_adds_the_datasets_and_records_it_as_recent(_app, tmp_path):
    """Exercises MainWindow, not just the loader.

    The loader-level tests all passed while the drop handler raised
    ``AttributeError: 'MainWindow' object has no attribute '_record_recent'`` --
    that method is on AppState, and add_dataset already calls it from
    ``dataset.file.recent_path``. Nothing below MainWindow could catch it.
    """
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.core.loader import load_from_mfx_array
    from minflux_viewer.core.save import save_processed
    from minflux_viewer.msr.msr_parser import parse_general
    from minflux_viewer.ui.main_window import MainWindow

    parsed = parse_general(str(SAMPLE), str(tmp_path), log=lambda *_a: None)
    datasets = []
    for i, entry in enumerate(parsed["datasets"]):
        ds = load_from_mfx_array(entry["_mfx"], name=entry["display_name"],
                                 prefs={"data": {}})
        ds.metadata["msr_source_path"] = str(SAMPLE)
        ds.metadata["msr_dataset_did"] = entry.get("did")
        ds.state.update({"overlay_id": "g", "overlay_index": i})
        datasets.append(ds)
    out = tmp_path / "dropped.msr"
    save_processed(datasets[0], data_path=out, fmt="msr", content="raw",
                   include={"attrs": True, "derived": False, "recipe": False},
                   related_datasets=datasets)

    state = AppState()
    state.prefs.setdefault("data", {}).update({"show_data_info": False,
                                               "show_render": False})
    win = MainWindow(state)
    try:
        seen = []
        state.log_message.connect(lambda m, *_a: seen.append(m))
        win._on_viewer_msr_loaded(
            [load_from_mfx_array(e["_mfx"], name=e["display_name"],
                                 recent_path=str(out), prefs={"data": {}})
             for e in parse_general(str(out), str(tmp_path), log=lambda *_a: None)["datasets"]],
            out.name, str(out), messages=["[viewer] restored state for 'x'"])
        assert len(state.datasets) == 3
        # what the restore did is collected in the worker and logged here,
        # because a worker must not touch Qt
        assert any("restored state" in m for m in seen), seen
        recent = [str(p) for p in state.prefs.get("file", {}).get("recent_files", [])]
        assert any(out.name in r for r in recent), recent
    finally:
        win.close()
        _app.processEvents()
