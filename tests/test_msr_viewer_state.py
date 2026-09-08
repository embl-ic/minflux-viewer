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
