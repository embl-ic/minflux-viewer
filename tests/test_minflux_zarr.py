"""MINFLUX Viewer Zarr v2 schema and self-contained round trips."""

from pathlib import Path

import numpy as np
import pytest

from minflux_viewer.core import loader as L
from minflux_viewer.core.dataset import AttributeComponent
from minflux_viewer.core.minflux_zarr import (
    FORMAT_ATTR,
    FORMAT_ID,
    PROJECT_FORMAT_ID,
    RAW_FINGERPRINT_ATTR,
    SCHEMA_ATTR,
    SCHEMA_VERSION,
    MinfluxZarrError,
    capture_native_zarr_metadata,
    load_minflux_zarr_project,
    write_minflux_zarr_project,
)
from minflux_viewer.core.save import dataset_to_mfx_array, save_processed


def _raw_mfx(n_loc=12, n_itr=3):
    n = n_loc * n_itr
    dtype = np.dtype([
        ("vld", np.bool_),
        ("tid", np.int64),
        ("tim", np.float64),
        ("itr", np.int32),
        ("efo", np.float32),
        ("eco", np.int32),
        ("dcr", np.float32, (2,)),
        ("loc", np.float64, (3,)),
    ])
    raw = np.zeros(n, dtype=dtype)
    raw["vld"] = True
    raw["tid"] = np.repeat(np.arange(n_loc), n_itr)
    raw["tim"] = np.repeat(np.linspace(0.0, 11.0, n_loc), n_itr)
    raw["itr"] = np.tile(np.arange(n_itr), n_loc)
    raw["efo"] = np.arange(n, dtype=np.float32) + 10.0
    raw["eco"] = np.arange(n, dtype=np.int32) + 100
    # Deliberately not complementary: both native DCR channels are facts.
    raw["dcr"][:, 0] = np.linspace(0.1, 0.9, n)
    raw["dcr"][:, 1] = np.linspace(0.2, 0.7, n)
    raw["loc"][:, 0] = np.arange(n) * 1e-9
    raw["loc"][:, 1] = (np.arange(n) + 100) * 1e-9
    raw["loc"][:, 2] = (np.arange(n) + 200) * 1e-9
    return raw


def _points(n=4):
    dtype = np.dtype([
        ("gri", np.int32),
        ("xyz", np.float64, (3,)),
        ("tim", np.float64),
        ("str", np.float32),
    ])
    points = np.zeros(n, dtype=dtype)
    points["gri"] = np.arange(1, n + 1)
    points["xyz"] = np.arange(n * 3).reshape(n, 3) * 1e-7
    points["tim"] = np.arange(n, dtype=float)
    points["str"] = np.linspace(1.0, 2.0, n)
    return points


def _dataset(tmp_path):
    ds = L.load_from_mfx_array(
        _raw_mfx(), name="source.msr | channel", folder=str(tmp_path)
    )
    mbm = _points(4)
    search = _points(3)
    ds.mbm = AttributeComponent({"points": mbm})
    ds.metadata.update({
        "mbm_points": mbm,
        "mbm_points_by_gri": {
            "1": {"gri": 1, "name": "R1"},
            "2": {"gri": 2, "name": "R2"},
        },
        "mbm_used": ["R1", "R2"],
        "search_points": search,
        "native_zarr_root_attrs": {
            "version": "2.1",
            "rois": [{"type": "ROI", "corners": [[0.0, 0.0], [1e-6, 2e-6]]}],
        },
        "native_zarr_mfx_attrs": {
            "did": "did-123",
            "acquisition_date": "2026-06-25T12:34:56+02:00",
            "measurement": {"threads": [{"itr": 3}]},
            "scan_range": {"x": 1e-6, "y": 2e-6},
        },
        "native_zarr_mbm_attrs": {"lattice": {"spacing": 1.0}},
        "native_zarr_mbm_points_attrs": {"points_by_gri": {"1": {"name": "R1"}}},
        "native_zarr_search_attrs": {"grid": "search"},
        "native_zarr_search_points_attrs": {"description": "initial search"},
        "custom_processing_note": "round-trip me",
    })
    ds.set_z_scaling_factor(0.73, source="manual (test)")
    transform = np.eye(4)
    transform[0, 3] = 125.0
    ds.state["overlay_transform"] = {"matrix_4x4": transform.tolist()}
    ds.state["filter_specs"] = [{
        "attribute": "efo", "mode": "per loc", "itr": "last",
        "lo": 12.0, "hi": 1000.0, "lo_inc": True, "hi_inc": True,
    }]
    mask = np.ones(ds.prop.num_loc, dtype=bool)
    mask[::3] = False
    ds.filter_mask = mask
    density = np.linspace(0.0, 1.0, ds.prop.num_loc)
    ds.attr["den"] = density
    ds.derived["den"] = density
    return ds



def _await_zarr_io(window, app, *, timeout=120.0):
    """Pump Qt until a Zarr load/save worker has delivered.

    Opening and saving a store moved off the UI thread (it is CPU-bound and was
    freezing the window), so a test that calls ``_load_zarr`` and asserts
    immediately now races the worker.
    """
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if not getattr(window, "_zarr_io_tasks", []):
            app.processEvents()
            return True
        time.sleep(0.02)
    return False

def test_self_contained_schema_roundtrip(tmp_path):
    ds = _dataset(tmp_path)
    before = dataset_to_mfx_array(ds)
    written = save_processed(
        ds,
        data_path=tmp_path / "processed",
        fmt="zarr",
        content="raw",
        include={"recipe": True, "derived": True},
    )
    assert written == [tmp_path / "processed.zarr"]
    assert not (tmp_path / "processed_metadata.json").exists()

    import zarr

    root = zarr.open(str(written[0]), mode="r")
    assert root.attrs[FORMAT_ATTR] == FORMAT_ID
    assert root.attrs[SCHEMA_ATTR] == SCHEMA_VERSION
    assert {"loc_x", "loc_y", "loc_z", "dcr_0", "dcr_1"} <= set(
        root["mfx"].array_keys()
    )
    assert root.attrs["rois"][0]["type"] == "ROI"
    assert root["mfx"].attrs["did"] == "did-123"
    assert "grd/mbm/points" in root
    assert "grd/search_0/points" in root
    assert "viewer/derived" in root
    assert "viewer/state" in root

    back = L.load_zarr(written[0])
    after = dataset_to_mfx_array(back)
    assert set(after.dtype.names or ()) == set(before.dtype.names or ())
    for name in before.dtype.names or ():
        np.testing.assert_array_equal(after[name], before[name])
    np.testing.assert_array_equal(back.metadata["mbm_points"], ds.metadata["mbm_points"])
    np.testing.assert_array_equal(back.metadata["search_points"], ds.metadata["search_points"])
    assert back.metadata["mbm_used"] == ["R1", "R2"]
    assert back.metadata["custom_processing_note"] == "round-trip me"
    assert back.metadata["native_zarr_mfx_attrs"]["did"] == "did-123"
    assert back.cali.z_scaling_factor == pytest.approx(0.73)
    np.testing.assert_array_equal(back.filter_mask, ds.filter_mask)
    assert back.state["filter_specs"] == ds.state["filter_specs"]
    np.testing.assert_array_equal(back.derived["den"], ds.derived["den"])
    np.testing.assert_array_equal(back.attr["den"], ds.attr["den"])
    assert back.state["overlay_transform"]["matrix_4x4"][0][3] == 125.0
    assert back.metadata["source_format"] == "minflux-viewer zarr v2"


def test_zarr_save_is_transactional_when_replacing_store(tmp_path):
    path = tmp_path / "replace.zarr"
    first = _dataset(tmp_path)
    save_processed(first, data_path=path, fmt="zarr", content="raw")
    second = _dataset(tmp_path)
    second.metadata["custom_processing_note"] = "second save"
    save_processed(second, data_path=path, fmt="zarr", content="raw")
    assert L.load_zarr(path).metadata["custom_processing_note"] == "second save"
    assert not list(tmp_path.glob(".replace.zarr.tmp-*"))
    assert not list(tmp_path.glob(".replace.zarr.backup-*"))


def test_processing_only_update_preserves_raw_and_refuses_raw_mismatch(tmp_path):
    import os

    import zarr

    path = tmp_path / "processing-update.zarr"
    ds = _dataset(tmp_path)
    save_processed(ds, data_path=path, fmt="zarr", content="raw")
    root = zarr.open_group(str(path), mode="r")
    recorded_fingerprint = root.attrs[RAW_FINGERPRINT_ATTR]
    assert recorded_fingerprint
    # The update path also supports stores written before fingerprints existed.
    legacy_root = zarr.open_group(str(path), mode="a")
    del legacy_root.attrs[RAW_FINGERPRINT_ATTR]
    future = legacy_root["viewer"].create_group("future_extension")
    future.create_dataset("values", data=np.array([7, 8, 9]))
    legacy_root["viewer"].attrs["future_attribute"] = "keep me"

    raw_chunk = next(
        item for item in (path / "mfx" / "loc_x").iterdir()
        if item.is_file() and not item.name.startswith(".")
    )
    raw_bytes = raw_chunk.read_bytes()
    sentinel_ns = 1_650_000_000_123_456_700
    os.utime(raw_chunk, ns=(sentinel_ns, sentinel_ns))

    ds.metadata["custom_processing_note"] = "processing stage two"
    ds.state["invert"] = True
    save_processed(
        ds,
        data_path=path,
        fmt="zarr",
        content="raw",
        zarr_overwrite="viewer",
        roi_records=[{
            "id": "roi-stage-two",
            "kind": "rectangle",
            "geometry": {"bounds": [1, 2, 3, 4]},
            "dataset_id": "d000000",
        }],
    )
    assert raw_chunk.read_bytes() == raw_bytes
    assert raw_chunk.stat().st_mtime_ns == sentinel_ns
    back = L.load_zarr(path)
    assert back.metadata["custom_processing_note"] == "processing stage two"
    assert back.state["invert"] is True
    assert back.metadata["minflux_viewer_roi_records"][0]["id"] == "roi-stage-two"
    updated_root = zarr.open_group(str(path), mode="r")
    np.testing.assert_array_equal(
        updated_root["viewer/future_extension/values"][:], [7, 8, 9]
    )
    assert updated_root["viewer"].attrs["future_attribute"] == "keep me"

    mismatch = _dataset(tmp_path)
    changed = np.asarray(mismatch.mfx_raw["efo"]).copy()
    changed[0] += 1
    mismatch.mfx_raw["efo"] = changed
    mismatch.metadata["custom_processing_note"] = "must not be written"
    with pytest.raises(MinfluxZarrError, match="Canonical raw data differs"):
        save_processed(
            mismatch,
            data_path=path,
            fmt="zarr",
            content="raw",
            zarr_overwrite="viewer",
        )
    assert L.load_zarr(path).metadata["custom_processing_note"] \
        == "processing stage two"
    tampered = zarr.open_group(str(path), mode="a")
    tampered.attrs[RAW_FINGERPRINT_ATTR] = recorded_fingerprint
    tampered["mfx/efo"][0] = tampered["mfx/efo"][0] + 1
    with pytest.raises(MinfluxZarrError, match="no longer matches"):
        save_processed(
            ds,
            data_path=path,
            fmt="zarr",
            content="raw",
            zarr_overwrite="viewer",
        )
    assert not list(path.glob(".viewer.backup-*"))
    assert not list(tmp_path.glob(".processing-update.zarr.viewer-tmp-*"))


def test_unmarked_flat_zarr_is_rejected(tmp_path):
    import zarr

    path = tmp_path / "legacy.zarr"
    root = zarr.group(store=zarr.DirectoryStore(str(path)), overwrite=True)
    root.create_dataset("loc_x", data=np.arange(3.0))
    with pytest.raises(MinfluxZarrError, match="format marker"):
        L.load_zarr(path)


def test_capture_native_vendor_metadata_and_search_grid(tmp_path):
    import zarr

    path = tmp_path / "vendor.zarr"
    root = zarr.group(store=zarr.DirectoryStore(str(path)), overwrite=True)
    root.attrs.update({"version": "2.1", "rois": [{"type": "ROI"}]})
    mfx = root.create_group("mfx")
    mfx.attrs.update({"did": "native-did", "scan_range": {"x": 1.0}})
    mbm = root.create_group("grd/mbm")
    mbm.attrs["used"] = ["R1"]
    mbm_points = mbm.create_dataset("points", data=_points(2))
    mbm_points.attrs["points_by_gri"] = {"1": {"name": "R1"}}
    search = root.create_group("grd/search_0")
    search.attrs["grid"] = "search"
    search_points = search.create_dataset("points", data=_points(3))
    search_points.attrs["description"] = "native search"

    ds = L.load_from_mfx_array(_raw_mfx(), name="native", folder=str(tmp_path))
    capture_native_zarr_metadata(ds, path)
    assert ds.metadata["native_zarr_root_attrs"]["rois"][0]["type"] == "ROI"
    assert ds.metadata["native_zarr_mfx_attrs"]["did"] == "native-did"
    assert ds.metadata["native_zarr_mbm_attrs"]["used"] == ["R1"]
    assert ds.metadata["native_zarr_search_attrs"]["grid"] == "search"
    np.testing.assert_array_equal(ds.metadata["search_points"], _points(3))


def test_zarr_rejects_baked_snapshot_mode(tmp_path):
    with pytest.raises(ValueError, match="canonical raw"):
        save_processed(
            _dataset(tmp_path), data_path=tmp_path / "snapshot.zarr",
            fmt="zarr", content="snapshot",
        )


def test_file_menu_zarr_action_uses_new_save_backend():
    source = Path("minflux_viewer/ui/main_window.py").read_text(encoding="utf-8")
    assert 'QAction("Zarr (.zarr v2) format"' in source
    assert 'self._save_as_format("zarr", "Zarr v2")' in source


def test_existing_zarr_prompt_offers_processing_replace_and_cancel(tmp_path, monkeypatch):
    pytest.importorskip("PyQt6")
    from minflux_viewer.ui import main_window as main_module

    class _FakeMessageBox:
        class Icon:
            Question = object()

        class ButtonRole:
            AcceptRole = object()
            DestructiveRole = object()

        class StandardButton:
            Cancel = object()

        choice = "update"

        def __init__(self, _parent):
            self.buttons = {}
            self.clicked = None

        def setIcon(self, *_args):
            pass

        def setWindowTitle(self, *_args):
            pass

        def setText(self, *_args):
            pass

        def setInformativeText(self, *_args):
            pass

        def addButton(self, label, _role=None):
            key = "cancel" if label is self.StandardButton.Cancel else (
                "update" if str(label).startswith("Update") else "replace"
            )
            button = object()
            self.buttons[key] = button
            return button

        def setDefaultButton(self, *_args):
            pass

        def setEscapeButton(self, *_args):
            pass

        def exec(self):
            self.clicked = self.buttons[self.choice]

        def clickedButton(self):
            return self.clicked

    monkeypatch.setattr(main_module, "QMessageBox", _FakeMessageBox)
    target = tmp_path / "existing.zarr"
    target.mkdir()
    for choice, expected in (("update", "viewer"), ("replace", "replace"),
                             ("cancel", None)):
        _FakeMessageBox.choice = choice
        assert main_module.MainWindow._zarr_overwrite_mode(None, target) == expected
    assert main_module.MainWindow._zarr_overwrite_mode(
        None, tmp_path / "new-store"
    ) == "replace"


def test_multichannel_project_roundtrip_preserves_overlay_rois_and_images(
    tmp_path, monkeypatch
):
    from dataclasses import asdict
    import zarr

    from minflux_viewer.core.roi import RoiRecord
    import minflux_viewer.core.obf_image_source as obf_module
    import minflux_viewer.core.tiff_export as tiff_export

    first = _dataset(tmp_path)
    second = _dataset(tmp_path)
    first.file.name = "channel one"
    second.file.name = "channel two"
    first.metadata["native_zarr_mfx_attrs"]["did"] = "did-one"
    second.metadata["native_zarr_mfx_attrs"]["did"] = "did-two"
    for order, ds in enumerate((first, second), 1):
        ds.state.update({
            "overlay_id": "group-original",
            "render_group_id": "group-original",
            "overlay_order": order,
            "overlay_lut": ("Red", "Green")[order - 1],
        })
    second.state["overlay_transform"] = {
        "matrix_4x4": [[1, 0, 0, 42], [0, 1, 0, -7],
                       [0, 0, 1, 0], [0, 0, 0, 1]]
    }

    roi = RoiRecord.create("rectangle", {"bounds": [1.0, 2.0, 3.0, 4.0]})
    roi_payload = asdict(roi)
    roi_payload["dataset_id"] = "d000001"
    roi_payload["context"] = {"source_view": "render"}

    source_msr = tmp_path / "source.msr"
    source_msr.write_bytes(b"fake")

    class _FakeSource:
        def __init__(self, path, *, raw_stack_index):
            self.path = Path(path)
            self.raw_stack_index = raw_stack_index

        def close(self):
            pass

    monkeypatch.setattr(obf_module, "ObfImageSource", _FakeSource)
    monkeypatch.setattr(
        tiff_export, "export_image_series_to_tiff",
        lambda _source, path: Path(path).write_bytes(b"fake ome-tiff"),
    )

    path = write_minflux_zarr_project(
        [first, second], tmp_path / "acquisition.zarr",
        roi_records=[roi_payload],
        image_specs=[{
            "msr_path": str(source_msr), "raw_index": 12,
            "name": "MF(run)/density/loc", "source_did": "did-two",
        }],
        name="two channel acquisition",
    )
    root = zarr.open_group(str(path), mode="r")
    assert root.attrs[FORMAT_ATTR] == PROJECT_FORMAT_ID
    assert "datasets/d000000/mfx" in root
    assert "datasets/d000001/mfx" in root
    embedded = path / "datasets" / "d000001" / "images" / "MF(run)_density_loc.tif"
    assert embedded.is_file()

    project = load_minflux_zarr_project(path)
    assert len(project.datasets) == 2
    assert project.manifest["is_overlay"] is True
    assert project.datasets[0].state["overlay_id"] == project.datasets[1].state["overlay_id"]
    assert project.datasets[1].state["overlay_transform"]["matrix_4x4"][0][3] == 42
    assert project.roi_records[0]["geometry"]["bounds"] == [1.0, 2.0, 3.0, 4.0]
    assert project.datasets[1].metadata["minflux_viewer_images"][0]["absolute_path"] \
        == str(embedded)


def test_project_processing_update_matches_reordered_channels_and_remaps_rois(tmp_path):
    import os

    first = _dataset(tmp_path)
    second = _dataset(tmp_path)
    first.file.name = "channel one"
    second.file.name = "channel two"
    first.metadata["native_zarr_mfx_attrs"]["did"] = "did-one"
    second.metadata["native_zarr_mfx_attrs"]["did"] = "did-two"
    for order, ds in enumerate((first, second), 1):
        ds.state.update(
            overlay_id="saved-group", render_group_id="saved-group", overlay_order=order
        )
    path = write_minflux_zarr_project([first, second], tmp_path / "reorder.zarr")
    raw_chunks = [
        next(
            item for item in (path / "datasets" / dataset_id / "mfx" / "loc_x").iterdir()
            if item.is_file() and not item.name.startswith(".")
        )
        for dataset_id in ("d000000", "d000001")
    ]
    sentinels = [1_650_000_001_123_456_700, 1_650_000_002_123_456_700]
    for chunk, timestamp in zip(raw_chunks, sentinels):
        os.utime(chunk, ns=(timestamp, timestamp))

    second.state["overlay_order"] = 1
    first.state["overlay_order"] = 2
    second.state["overlay_transform"] = {"matrix_4x4": [
        [1, 0, 0, 88], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
    ]}
    write_minflux_zarr_project(
        [second, first],
        path,
        viewer_only=True,
        roi_records=[{
            "id": "roi-on-second",
            "kind": "oval",
            "geometry": {"bounds": [5, 6, 7, 8]},
            "dataset_id": "d000000",
        }],
    )
    assert [chunk.stat().st_mtime_ns for chunk in raw_chunks] == sentinels
    project = load_minflux_zarr_project(path)
    assert [ds.metadata["native_zarr_mfx_attrs"]["did"] for ds in project.datasets] \
        == ["did-two", "did-one"]
    assert project.datasets[0].state["overlay_transform"]["matrix_4x4"][0][3] == 88
    assert project.roi_records[0]["dataset_id"] == "d000001"
    assert not list(path.rglob(".viewer.backup-*"))
    assert not list(tmp_path.glob(".reorder.zarr.viewer-tmp-*"))


def test_main_window_reopens_project_as_overlay_and_restores_roi(
    tmp_path, monkeypatch
):
    pytest.importorskip("PyQt6")
    from dataclasses import asdict
    from PyQt6.QtWidgets import QApplication

    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.core.roi import RoiRecord
    from minflux_viewer.ui.main_window import MainWindow

    first = _dataset(tmp_path)
    second = _dataset(tmp_path)
    first.file.name = "channel one"
    second.file.name = "channel two"
    for order, ds in enumerate((first, second), 1):
        ds.state.update({
            "overlay_id": "saved-group", "render_group_id": "saved-group",
            "overlay_order": order,
        })
    roi = RoiRecord.create("oval", {"bounds": [1.0, 2.0, 3.0, 4.0]})
    payload = asdict(roi)
    payload["dataset_id"] = "d000001"
    payload["context"] = {"source_view": "scatter"}
    path = write_minflux_zarr_project(
        [first, second], tmp_path / "ui-project.zarr", roi_records=[payload]
    )

    _app = QApplication.instance() or QApplication([])
    state = AppState()
    state.prefs.setdefault("data", {}).update(
        show_render=False, show_data_info=False, show_dataset_manager=False,
    )
    win = MainWindow(state)
    monkeypatch.setattr(win, "_show_render", lambda *_a, **_k: None)
    try:
        win._load_zarr(str(path))
        assert _await_zarr_io(win, _app), "the load worker did not finish"
        assert len(state.datasets) == 2
        assert state.datasets[0].state["overlay_id"] == state.datasets[1].state["overlay_id"]
        restored = next(record for record in state.rois.records if record.id == roi.id)
        assert restored.context["dataset_idx"] == 1
        assert restored.geometry["bounds"] == [1.0, 2.0, 3.0, 4.0]
        win._populate_recent_menu()
        assert any(action.data() == str(path) for action in win._recent_menu.actions())
    finally:
        win.close()


def test_save_context_expands_overlay_and_captures_roi_and_linked_images(
    tmp_path, monkeypatch
):
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.core.roi import RoiRecord
    from minflux_viewer.ui.main_window import MainWindow
    import minflux_viewer.core.obf_image_source as image_module

    source = tmp_path / "three-channel.msr"
    source.write_bytes(b"fake")
    first = _dataset(tmp_path)
    second = _dataset(tmp_path)
    first.file.name = "channel-one"
    second.file.name = "channel-two"
    for order, (did, ds) in enumerate((("did-one", first), ("did-two", second)), 1):
        ds.metadata.update(msr_source_path=str(source), msr_dataset_did=did)
        ds.state.update(
            overlay_id="live-group", render_group_id="live-group", overlay_order=order
        )

    _app = QApplication.instance() or QApplication([])
    state = AppState()
    state.prefs.setdefault("data", {}).update(
        show_render=False, show_data_info=False, show_dataset_manager=False,
    )
    win = MainWindow(state)
    try:
        state.add_dataset(first)
        state.add_dataset(second)
        roi = RoiRecord.create("rectangle", {"bounds": [1, 2, 3, 4]})
        roi.context = {"dataset_idx": 1, "source_view": "render"}
        state.rois.add(roi)
        monkeypatch.setattr(image_module, "list_obf_image_series", lambda _path: [
            {"raw_index": 3, "name": "linked one", "source_did": "did-one"},
            {"raw_index": 8, "name": "unlinked", "source_did": ""},
        ])

        context = win._zarr_save_context(first)
        assert context["related_datasets"] == [first, second]
        assert context["roi_records"][0]["dataset_id"] == "d000001"
        assert "dataset_idx" not in context["roi_records"][0]["context"]
        # EVERY series of the source .msr travels with the store, not only the
        # DID-linked one. The unlinked series (confocal channels, overviews)
        # were silently dropped, which made the saved store depend on the
        # original .msr still sitting at its import path -- "View image series"
        # then fell back to reopening it and appeared to work.
        assert [(item["raw_index"], item["source_did"])
                for item in context["image_specs"]] == [(3, "did-one"), (8, "")]
    finally:
        win.close()


def test_store_fingerprint_matches_the_dataset_fingerprint(tmp_path):
    """The on-disk and in-memory digests must agree exactly.

    A processing-only update proves the raw data is unchanged by hashing the
    store itself rather than trusting its recorded digest. That hash is computed
    by reading the arrays directly instead of building a whole dataset, so the
    two routines have to stay in lockstep — if they drift, every update is
    refused with a message blaming the user's data.
    """
    import zarr

    from minflux_viewer.core.minflux_zarr import (
        _dataset_raw_fingerprint,
        _store_raw_fingerprint,
        load_minflux_zarr,
        write_minflux_zarr,
    )

    ds = _dataset(tmp_path)
    ds.mbm = AttributeComponent({"points": _points(4)}, roles={"gri": "bead id"})
    ds.metadata["mbm_points"] = _points(4)
    ds.metadata["mbm_used"] = ["R1", "R2"]
    ds.metadata["mbm_points_by_gri"] = {"1": {"name": "R1"}}
    ds.metadata["search_points"] = _points(3)
    ds.metadata["native_zarr_root_attrs"] = {"version": "2.1"}
    ds.metadata["native_zarr_mfx_attrs"] = {"did": "abc"}

    store = write_minflux_zarr(ds, tmp_path / "fp.zarr")
    root = zarr.open(str(store), mode="r")

    declared = str(root.attrs[RAW_FINGERPRINT_ATTR])
    assert _store_raw_fingerprint(root) == declared
    assert _dataset_raw_fingerprint(load_minflux_zarr(store)) == declared


def test_store_fingerprint_notices_an_externally_rewritten_chunk(tmp_path):
    """Reading arrays instead of loading a dataset must not weaken the check."""
    import zarr

    from minflux_viewer.core.minflux_zarr import (
        _store_raw_fingerprint,
        write_minflux_zarr,
    )

    store = write_minflux_zarr(_dataset(tmp_path), tmp_path / "tamper.zarr")
    before = _store_raw_fingerprint(zarr.open(str(store), mode="r"))

    handle = zarr.open(str(store), mode="a")
    handle["mfx/loc_x"][0] += 1.0
    assert _store_raw_fingerprint(zarr.open(str(store), mode="r")) != before


def test_sealed_zip_package_roundtrips_and_reads_without_unpacking(tmp_path):
    """A ``.zarr.zip`` is a distribution copy that opens directly."""
    import zipfile

    from minflux_viewer.core.minflux_zarr import (
        ZIP_SUFFIX,
        is_zipped_store,
        load_minflux_zarr,
        pack_minflux_zarr,
        unpack_minflux_zarr,
        write_minflux_zarr,
    )

    ds = _dataset(tmp_path)
    ds.state["filter_specs"] = [{"attribute": "efo", "mode": "per loc",
                                 "itr": "last", "lo": 1.0, "hi": 500.0,
                                 "lo_inc": True, "hi_inc": True}]
    store = write_minflux_zarr(ds, tmp_path / "sealed.zarr")

    package = pack_minflux_zarr(store)
    assert package.name.endswith(ZIP_SUFFIX) and is_zipped_store(package)

    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        # Chunks are already Blosc-compressed; deflating again would only cost time.
        assert {info.compress_type for info in archive.infolist()} == {zipfile.ZIP_STORED}
    assert len(names) == len(set(names)), "sealed package must not repeat members"

    opened = load_minflux_zarr(package)
    direct = load_minflux_zarr(store)
    for key in direct.components.mfx_raw.keys():
        np.testing.assert_array_equal(
            np.asarray(direct.components.mfx_raw.get(key)),
            np.asarray(opened.components.mfx_raw.get(key)),
        )
    assert opened.state["filter_specs"] == ds.state["filter_specs"]

    restored = unpack_minflux_zarr(package, out_dir=tmp_path / "out")
    assert load_minflux_zarr(restored).prop.num_loc == direct.prop.num_loc


def test_unpacking_refuses_an_appended_package(tmp_path):
    """Appending to a zip leaves two members with one name.

    ``zipfile`` and Zarr take the last, some archive tools take the first, so
    the contents are genuinely ambiguous -- name that rather than pick one.
    """
    import zipfile

    from minflux_viewer.core.minflux_zarr import (
        MinfluxZarrError,
        pack_minflux_zarr,
        unpack_minflux_zarr,
        write_minflux_zarr,
    )

    store = write_minflux_zarr(_dataset(tmp_path), tmp_path / "appended.zarr")
    package = pack_minflux_zarr(store)
    with zipfile.ZipFile(package, "a") as archive:
        archive.writestr("viewer/.zattrs", '{"tampered": true}')

    with pytest.raises(MinfluxZarrError, match="duplicate member"):
        unpack_minflux_zarr(package, out_dir=tmp_path / "out")


def test_sealed_project_loads_its_child_datasets(tmp_path):
    """A multi-dataset package must open, not just a single-dataset one.

    The project loader addressed children as filesystem paths
    (``<store>/datasets/d000000``). That exists for a directory store and does
    not exist inside a package, so every multi-channel ``.zarr.zip`` failed with
    ``FileNotFoundError`` naming ``...zarr.zip/datasets/d000000`` while
    as a directory opened fine.
    """
    from minflux_viewer.core.minflux_zarr import (
        load_minflux_zarr_project,
        pack_minflux_zarr,
        write_minflux_zarr_project,
    )

    members = []
    for index in range(3):
        ds = _dataset(tmp_path)
        ds.file.name = f"channel-{index}"
        ds.state["overlay_id"] = "grp"
        ds.state["overlay_order"] = index + 1
        members.append(ds)

    store = write_minflux_zarr_project(members, tmp_path / "acq.zarr", name="acq")
    package = pack_minflux_zarr(store)

    from_dir = load_minflux_zarr_project(store)
    from_zip = load_minflux_zarr_project(package)

    assert len(from_zip.datasets) == len(from_dir.datasets) == 3
    for expected, got in zip(from_dir.datasets, from_zip.datasets):
        assert got.prop.num_loc == expected.prop.num_loc
        np.testing.assert_array_equal(
            np.asarray(expected.components.mfx_raw.get("loc_x")),
            np.asarray(got.components.mfx_raw.get("loc_x")),
        )
    # Overlay membership survives the packing.
    assert from_zip.datasets[0].state.get("overlay_id")
    assert [d.state["overlay_order"] for d in from_zip.datasets] == [1, 2, 3]


def test_project_names_a_missing_member_instead_of_failing_obscurely(tmp_path):
    """A manifest entry with no matching group is a store problem, not a path one."""
    import shutil

    import zarr

    from minflux_viewer.core.minflux_zarr import (
        MinfluxZarrError,
        load_minflux_zarr_project,
        write_minflux_zarr_project,
    )

    store = write_minflux_zarr_project(
        [_dataset(tmp_path), _dataset(tmp_path)], tmp_path / "gap.zarr")
    shutil.rmtree(store / "datasets" / "d000001")
    zarr.open(str(store), mode="r")            # store itself still opens

    with pytest.raises(MinfluxZarrError, match="no such member"):
        load_minflux_zarr_project(store)


def test_package_images_resolve_without_the_original_msr(tmp_path):
    """A sealed package must carry its images, not borrow them.

    Image records restored from a ``.zarr.zip`` have an ``absolute_path``
    pointing inside the archive, which does not exist on disk. The viewer's
    "is this a file?" check then found nothing and fell back to reopening the
    source ``.msr`` -- so a package looked like it held every image while it was
    really reading them from a file that might move or be absent.
    """
    from minflux_viewer.core.minflux_zarr import (
        load_minflux_zarr_project,
        materialize_image,
        pack_minflux_zarr,
        write_minflux_zarr,
    )

    ds = _dataset(tmp_path)
    store = write_minflux_zarr(ds, tmp_path / "imgs.zarr")

    # Place an image the way the writer does, and record it in the manifest.
    images = store / "images"
    images.mkdir(parents=True, exist_ok=True)
    (images / "overview.tif").write_bytes(b"II*\x00fake-tiff-payload")
    import zarr
    viewer = zarr.open(str(store), mode="a")["viewer"]
    viewer.require_group("images").attrs["_minflux_viewer_payload"] = [
        {"path": "images/overview.tif", "name": "overview", "format": "ome-tiff"},
    ]

    package = pack_minflux_zarr(store)
    from_zip = load_minflux_zarr_project(package).datasets[0]
    records = from_zip.metadata.get("minflux_viewer_images") or []
    assert len(records) == 1
    assert not Path(records[0]["absolute_path"]).is_file(), "inside the archive"

    resolved = materialize_image(records[0])
    assert resolved is not None and resolved.is_file()
    assert resolved.read_bytes() == b"II*\x00fake-tiff-payload"

    # A directory store needs no extraction and must keep working.
    from_dir = load_minflux_zarr_project(store).datasets[0]
    direct = materialize_image((from_dir.metadata["minflux_viewer_images"])[0])
    assert direct is not None and direct.is_file()


def test_zarr_dataset_never_borrows_images_from_the_source_msr(tmp_path, monkeypatch):
    """A store must stand alone.

    A dataset restored from a ``.zarr`` carries its own OME-TIFFs. If it finds
    none it must say so, not reach for ``metadata["msr_source_path"]`` -- that
    file may have moved, been renamed, or never travelled with the store, and
    borrowing from it made an incomplete store look complete.
    """
    pytest.importorskip("PyQt6")
    import types

    import minflux_viewer.core.obf_image_source as image_module
    from minflux_viewer.ui.main_window import MainWindow

    ds = _dataset(tmp_path)
    msr = tmp_path / "acquisition.msr"
    msr.write_bytes(b"not really an msr")
    ds.metadata["minflux_viewer_zarr_path"] = str(tmp_path / "store.zarr")
    ds.metadata["msr_source_path"] = str(msr)

    consulted: list = []
    monkeypatch.setattr(image_module, "list_obf_image_series",
                        lambda path: consulted.append(path) or [])
    shown: list = []
    monkeypatch.setattr(
        "minflux_viewer.ui.main_window.QMessageBox.information",
        staticmethod(lambda *a, **k: shown.append(a[2] if len(a) > 2 else "")))

    # Drive the method directly: constructing a MainWindow is unnecessary for
    # the decision under test and drags in the whole render/post-load chain.
    window = types.SimpleNamespace(
        _state=types.SimpleNamespace(
            datasets=[ds],
            log=lambda *a, **k: None,
        ),
    )
    MainWindow.view_dataset_image_series(window, 0)

    assert not consulted, "the source .msr must not be opened"
    assert shown and "carries its own images" in shown[0]


def test_opening_a_store_does_not_block_the_ui(tmp_path, monkeypatch):
    """Reading a store is CPU-bound, so it must not run on the UI thread.

    On a 20.6 M-row acquisition it costs ~10 s (2.0 s decompressing, 8.2 s
    rebuilding and normalizing the structured array), which froze the whole
    window. It is not made faster by threading -- it stops blocking, which
    works because numpy and blosc release the GIL for those operations.
    """
    pytest.importorskip("PyQt6")
    import time

    from PyQt6.QtWidgets import QApplication

    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.core.minflux_zarr import write_minflux_zarr
    from minflux_viewer.ui.main_window import MainWindow

    store = write_minflux_zarr(_dataset(tmp_path), tmp_path / "async.zarr")
    app = QApplication.instance() or QApplication([])
    state = AppState()
    state.prefs.setdefault("data", {}).update(
        show_render=False, show_data_info=False, show_dataset_manager=False)
    window = MainWindow(state)
    monkeypatch.setattr(window, "_show_render", lambda *_a, **_k: None)
    try:
        started = time.perf_counter()
        window._load_zarr(str(store))
        returned = time.perf_counter() - started
        # The call hands off; it does not do the reading.
        assert returned < 1.0, f"_load_zarr blocked for {returned:.2f}s"
        assert getattr(window, "_zarr_io_tasks", None), "no worker was started"

        assert _await_zarr_io(window, app), "the load worker did not finish"
        assert len(state.datasets) == 1
        assert not window._zarr_io_tasks, "the finished task was not released"
    finally:
        window.close()


def test_a_failed_background_load_reports_instead_of_vanishing(tmp_path, monkeypatch):
    """An error on the worker must still reach the user."""
    pytest.importorskip("PyQt6")

    from PyQt6.QtWidgets import QApplication

    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow(AppState())
    shown: list = []
    monkeypatch.setattr(
        "minflux_viewer.ui.main_window.QMessageBox.critical",
        staticmethod(lambda *a, **k: shown.append(a[2] if len(a) > 2 else "")))
    try:
        window._load_zarr(str(tmp_path / "does-not-exist.zarr"))
        assert _await_zarr_io(window, app)
        assert shown, "a background failure must be reported"
        assert not window._state.datasets
    finally:
        window.close()


def test_load_zarr_accepts_a_project_store(tmp_path, recwarn):
    """The documented entry point must open both store kinds.

    A multi-dataset acquisition is marked as a *project*, and the single-dataset
    loader rejected it as "format marker missing" -- untrue, since the marker is
    present and simply says project. The GUI never hit this because it calls the
    project loader directly; scripts using the documented `load_zarr` did.
    """
    from minflux_viewer.core.loader import load_zarr
    from minflux_viewer.core.minflux_zarr import (
        write_minflux_zarr,
        write_minflux_zarr_project,
    )

    single = write_minflux_zarr(_dataset(tmp_path), tmp_path / "one.zarr")
    assert load_zarr(single).prop.num_loc > 0

    members = [_dataset(tmp_path), _dataset(tmp_path)]
    project = write_minflux_zarr_project(members, tmp_path / "many.zarr")
    opened = load_zarr(project)
    assert opened.prop.num_loc == members[0].prop.num_loc
    # Returning one of several silently would be a trap, so it says so.
    assert any("returns the first" in str(w.message) for w in recwarn)


def test_zarr2_hands_blosc_the_array_so_shuffle_works(tmp_path):
    """``.tobytes()`` gives Blosc typesize 1 and its SHUFFLE nothing to reorder.

    That made every ``.msr`` this application writes ~1.6-1.8x larger than
    needed. The values must be untouched by the change.
    """
    import numcodecs

    from minflux_viewer.msr import zarr2

    rng = np.random.default_rng(0)
    n = 300_000
    dtype = np.dtype([
        ("vld", np.bool_), ("tid", np.int64), ("tim", np.float64),
        ("itr", np.int32), ("efo", np.float32), ("loc", np.float64, (3,)),
    ])
    arr = np.zeros(n, dtype=dtype)
    arr["vld"] = True
    arr["tid"] = np.repeat(np.arange(n // 3), 3)[:n]
    arr["tim"] = np.linspace(0.0, 100.0, n)
    arr["itr"] = np.tile([0, 1, 2], n // 3 + 1)[:n]
    arr["efo"] = rng.normal(5000, 50, n).astype(np.float32)
    arr["loc"][:, 0] = np.linspace(0, 4e-6, n)

    store: dict = {}
    zarr2.open(store, mode="w")["mfx"] = arr.ravel()
    written = sum(len(v) for k, v in store.items() if k.startswith("mfx"))

    codec = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)
    as_bytes = sum(
        len(codec.encode(np.ascontiguousarray(arr[i:i + zarr2.DEFAULT_CHUNK]).tobytes()))
        for i in range(0, n, zarr2.DEFAULT_CHUNK))
    assert written < as_bytes, "shuffle must be getting the itemsize"

    back = np.asarray(zarr2.open(store, mode="r")["mfx"][:])[:n]
    assert back.dtype == arr.dtype
    for field in arr.dtype.names:
        # NB: use the ARRAY's kind — a subarray field's dtype kind is "V".
        left, right = arr[field], back[field]
        if left.dtype.kind == "f":
            np.testing.assert_allclose(left, right, rtol=0, atol=0)
        else:
            np.testing.assert_array_equal(left, right)


def _roi_payload(name, x, dataset_idx=0):
    from minflux_viewer.core.roi import RoiRecord

    record = RoiRecord.create("rectangle", {"bounds": [x, 20.0, 30.0, 40.0]},
                              name=name)
    record.context = {"dataset_idx": dataset_idx, "source_view": "render"}
    return record


def _roi_host(datasets, records):
    """A stand-in for MainWindow carrying just what the ROI helpers touch.

    Constructing a real MainWindow drags in the render/post-load chain and every
    child window; these methods only need the dataset list, the ROI store and a
    way to raise the Manager.
    """
    import types

    from minflux_viewer.core.roi import RoiStore

    store = RoiStore()
    for record in records:
        store.add(record)
    shown = []
    host = types.SimpleNamespace(
        _state=types.SimpleNamespace(
            datasets=list(datasets),
            rois=store,
            active_idx=0,
            log=lambda *a, **k: None,
            set_active=lambda idx: None,
        ),
        _roi_manager_shown=shown,
    )
    host._show_roi_manager = lambda: shown.append(True)
    host._show_filter = lambda: shown.append("filter")
    return host


@pytest.mark.parametrize("fmt", ["zarr", "mat", "csv"])
def test_every_format_saves_the_whole_roi_set(tmp_path, fmt):
    """A dataset carries all displayed ROIs, not just the active draft.

    The ROI Manager holds many at once. Only the Zarr path used to pass them,
    so a .mat/.csv sidecar stored none at all.
    """
    import json

    from minflux_viewer.core.save import save_processed
    from minflux_viewer.ui.main_window import MainWindow

    ds = _dataset(tmp_path)
    records = [_roi_payload(f"cell-{n}", 10.0 * n) for n in range(3)]
    host = _roi_host([ds], records)
    host._state.rois.active_adapter = None

    gathered = MainWindow.save_roi_records(host, ds)
    assert len(gathered) == 3

    written = save_processed(
        ds, data_path=tmp_path / f"out_{fmt}", fmt=fmt, content="raw",
        include={"attrs": True, "derived": False, "recipe": True},
        roi_records=gathered)

    if fmt == "zarr":
        from minflux_viewer.core.minflux_zarr import load_minflux_zarr_project
        stored = load_minflux_zarr_project(written[0]).roi_records
    else:
        sidecar = next(p for p in written if p.name.endswith("_metadata.json"))
        stored = json.loads(sidecar.read_text(encoding="utf-8"))["rois"]
    assert sorted(r["name"] for r in stored) == ["cell-0", "cell-1", "cell-2"]


def test_restored_rois_concatenate_and_do_not_duplicate(tmp_path):
    """Opening a second processed dataset must not discard the first one's ROIs."""
    import dataclasses

    from minflux_viewer.ui.main_window import MainWindow

    first, second = _dataset(tmp_path), _dataset(tmp_path)
    first.metadata["minflux_viewer_roi_records"] = [
        dataclasses.asdict(_roi_payload("a-1", 1.0)),
        dataclasses.asdict(_roi_payload("a-2", 2.0)),
    ]
    second.metadata["minflux_viewer_roi_records"] = [
        dataclasses.asdict(_roi_payload("b-1", 3.0)),
    ]
    host = _roi_host([first, second], [])

    MainWindow._restore_saved_rois(host, first, 0)
    assert len(host._state.rois.records) == 2
    assert host._roi_manager_shown, "the Manager must be raised"

    MainWindow._restore_saved_rois(host, second, 1)
    assert sorted(r.name for r in host._state.rois.records) == ["a-1", "a-2", "b-1"]

    # Re-opening the same dataset is idempotent: records match by id.
    MainWindow._restore_saved_rois(host, first, 0)
    assert len(host._state.rois.records) == 3


def test_saved_filters_open_the_dialog_without_changing_them(tmp_path):
    """A reopened dataset can be filtered with nothing on screen saying so.

    Each row keeps the enabled/disabled state it was saved with -- this displays
    the filter, it does not apply or alter it.
    """
    from minflux_viewer.ui.main_window import MainWindow

    ds = _dataset(tmp_path)
    specs = [
        {"attribute": "efo", "mode": "per loc", "itr": "last", "lo": 1.0,
         "hi": 900.0, "lo_inc": True, "hi_inc": True, "enabled": True},
        {"attribute": "efo", "mode": "per loc", "itr": "last", "lo": 2.0,
         "hi": 800.0, "lo_inc": True, "hi_inc": True, "enabled": False},
    ]
    ds.state["filter_specs"] = [dict(spec) for spec in specs]
    host = _roi_host([ds], [])

    MainWindow._show_saved_filters(host, ds, 0)
    assert "filter" in host._roi_manager_shown, "the Filter dialog must be shown"
    assert ds.state["filter_specs"] == specs, "showing must not rewrite them"

    # No filters means no dialog. (_dataset ships with filter_specs, so clear
    # them rather than assuming a fresh one has none.)
    plain = _dataset(tmp_path)
    plain.state.pop("filter_specs", None)
    host._roi_manager_shown.clear()
    MainWindow._show_saved_filters(host, plain, 0)
    assert not host._roi_manager_shown
