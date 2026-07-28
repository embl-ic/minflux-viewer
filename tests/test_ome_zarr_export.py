"""OME-NGFF 0.5 / Zarr v3 localization package export."""

import gzip
import json

import numpy as np
import pytest

from minflux_viewer.core import loader as L
from minflux_viewer.core import ome_zarr as OZ
from minflux_viewer.core.ome_zarr import estimate_ome_zarr_export, write_ome_zarr
from minflux_viewer.core.processing_journal import JournalEntry
from minflux_viewer.core.roi import RoiRecord


def _dataset(
    n=24,
    *,
    is_3d=True,
    x_span_nm=120.0,
    y_span_nm=180.0,
    z_span_nm=60.0,
):
    dtype = np.dtype([
        ("vld", np.bool_),
        ("tid", np.int32),
        ("tim", np.float64),
        ("itr", np.int32),
        ("efo", np.float64),
        ("loc", np.float64, (3,)),
    ])
    data = np.zeros(n, dtype=dtype)
    data["vld"] = True
    data["tid"] = np.repeat(np.arange(n // 4), 4)[:n]
    data["tim"] = np.linspace(0.0, 1.0, n)
    data["itr"] = 3
    data["efo"] = np.linspace(10.0, 100.0, n)
    z = (
        np.linspace(-z_span_nm / 2.0, z_span_nm / 2.0, n) * 1.0e-9
        if is_3d
        else np.zeros(n)
    )
    data["loc"] = np.column_stack(
        [
            np.linspace(0.0, x_span_nm * 1.0e-9, n),
            np.linspace(20e-9, (20.0 + y_span_nm) * 1.0e-9, n),
            z,
        ]
    )
    return L.load_from_mfx_array(data, name="ome-test.mat", folder="/tmp")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_ome_zarr_writes_standard_image_and_minflux_extension(tmp_path):
    ds = _dataset()
    ds.state["filter_specs"] = [{
        "attribute": "tid",
        "mode": "per loc",
        "lo": 1,
        "hi": 3,
        "lo_inc": True,
        "hi_inc": True,
    }]
    roi = RoiRecord.create(
        "rectangle",
        {"x": 0.0, "y": 0.0, "width": 50.0, "height": 60.0},
        name="cell",
        target_hint=ds.name,
    )
    roi.context["dataset_idx"] = 0
    roi.mask_key = "roi-mask"
    ds.derived[roi.mask_key] = np.arange(ds.prop.num_loc) % 2 == 0
    ds.state["roi_masks"] = {roi.mask_key: {"selection": "test"}}
    event = JournalEntry(
        timestamp="2026-07-27T12:00:00",
        category="filter",
        summary=f"Filtered {ds.name}",
        details={"attribute": "tid"},
    )

    result = write_ome_zarr(
        ds,
        tmp_path / "export",
        pixel_size_nm=5.0,
        z_voxel_nm=10.0,
        max_levels=4,
        dataset_idx=0,
        roi_records=[roi],
        journal_entries=[event],
    )
    root = result.path
    assert root.name == "export.ome.zarr"
    assert (root / "_SUCCESS").exists()

    root_meta = _read_json(root / "zarr.json")
    assert root_meta["zarr_format"] == 3
    assert root_meta["node_type"] == "group"
    assert root_meta["attributes"]["ome"]["version"] == "0.5"
    assert root_meta["attributes"]["org.minflux_viewer"]["profile_version"] == "0.1.0"

    image_meta = _read_json(root / "0" / "zarr.json")
    assert image_meta["data_type"] == "uint32"
    assert image_meta["dimension_names"] == ["z", "y", "x"]
    assert [codec["name"] for codec in image_meta["codecs"]] == ["bytes", "gzip"]
    assert (root / "0" / "c" / "0" / "0" / "0").exists()
    chunk = np.frombuffer(
        gzip.decompress((root / "0" / "c" / "0" / "0" / "0").read_bytes()),
        dtype="<u4",
    ).reshape(image_meta["chunk_grid"]["configuration"]["chunk_shape"])
    depth, height, width = image_meta["shape"]
    assert int(chunk[:depth, :height, :width].sum()) == 12
    assert image_meta["shape"] == [3, 18, 12]

    multiscale = root_meta["attributes"]["ome"]["multiscales"][0]
    assert [axis["name"] for axis in multiscale["axes"]] == ["z", "y", "x"]
    transform = multiscale["datasets"][0]["coordinateTransformations"]
    assert transform[0]["scale"] == [0.01, 0.005, 0.005]
    assert transform[1]["translation"] == [-0.02, 0.05, 0.02]
    assert result.is_3d
    assert result.image_shape == (3, 18, 12)
    assert result.voxel_size_nm == (10.0, 5.0, 5.0)

    manifest = _read_json(root / "minflux" / "manifest.json")
    base = manifest["dataset"]["path"]
    assert manifest["ome_ngff_version"] == "0.5"
    assert manifest["zarr_format"] == 3
    assert manifest["render"]["source"].endswith("/processed/current/position")
    assert manifest["render"]["filter"].endswith("/processed/current/ftr")

    raw_loc = _read_json(root / base / "raw" / "mfx" / "loc" / "zarr.json")
    assert raw_loc["shape"] == [ds.prop.num_loc, 3]
    assert raw_loc["attributes"]["unit"] == "meter"
    processed = root / base / "processed" / "current"
    assert _read_json(processed / "position" / "zarr.json")["shape"] == [
        ds.prop.num_loc,
        3,
    ]
    assert (processed / "source_row_id" / "c" / "0").exists()
    assert not (processed / "xnm").exists()
    assert not (processed / "ynm").exists()
    assert not (processed / "znm").exists()

    filters = _read_json(root / base / "state" / "filters.json")
    assert filters["specifications"][0]["attribute"] == "tid"
    rois = _read_json(root / base / "state" / "rois.json")
    assert rois["rois"][0]["name"] == "cell"
    roi_masks = _read_json(root / base / "state" / "roi_masks" / "index.json")
    assert roi_masks[0]["roi_id"] == roi.id
    assert (root / roi_masks[0]["path"] / "zarr.json").exists()

    events = _read_json(root / base / "provenance" / "events.json")
    assert events["events"][0]["summary"] == f"Filtered {ds.name}"
    assert (root / base / "provenance" / "recipe.json").exists()
    assert _read_json(root / "ro-crate-metadata.json")["@graph"][1]["@type"] == "Dataset"


def test_ome_zarr_refuses_existing_package_unless_overwrite(tmp_path):
    ds = _dataset()
    path = tmp_path / "replace.ome.zarr"
    write_ome_zarr(ds, path, pixel_size_nm=5.0, z_voxel_nm=10.0)
    with pytest.raises(FileExistsError):
        write_ome_zarr(ds, path, pixel_size_nm=5.0, z_voxel_nm=10.0)

    result = write_ome_zarr(
        ds,
        path,
        pixel_size_nm=10.0,
        z_voxel_nm=20.0,
        overwrite=True,
    )
    assert result.path == path
    assert (path / "_SUCCESS").exists()


def test_ome_zarr_keeps_2d_dataset_as_yx_image(tmp_path):
    ds = _dataset(is_3d=False)
    result = write_ome_zarr(ds, tmp_path / "two-d", pixel_size_nm=5.0)
    root_meta = _read_json(result.path / "zarr.json")
    multiscale = root_meta["attributes"]["ome"]["multiscales"][0]
    image_meta = _read_json(result.path / "0" / "zarr.json")

    assert not result.is_3d
    assert [axis["name"] for axis in multiscale["axes"]] == ["y", "x"]
    assert image_meta["dimension_names"] == ["y", "x"]
    assert len(image_meta["shape"]) == 2


def test_ome_zarr_preflight_reports_volume_resources(tmp_path):
    ds = _dataset()
    estimate = estimate_ome_zarr_export(
        ds,
        tmp_path / "estimate",
        pixel_size_nm=5.0,
        z_voxel_nm=10.0,
        max_levels=4,
    )
    assert estimate.is_3d
    assert estimate.image_shape == (7, 37, 25)
    assert estimate.voxel_size_nm == (10.0, 5.0, 5.0)
    assert estimate.filtered_localizations == ds.prop.num_loc
    assert estimate.estimated_output_bytes > 0
    assert estimate.peak_working_ram_bytes > 0
    assert estimate.estimated_seconds > 0


def test_ome_zarr_preflight_uses_five_minute_warning_band(
    tmp_path, monkeypatch
):
    ds = _dataset()
    monkeypatch.setattr(OZ, "ESTIMATED_GZIP_BYTES_PER_SECOND", 1)
    estimate = estimate_ome_zarr_export(
        ds,
        tmp_path / "slow-estimate",
        pixel_size_nm=5.0,
        z_voxel_nm=10.0,
    )
    assert estimate.estimated_seconds >= 300
    assert "Estimated conversion time is over 5 minutes." in estimate.warnings


def test_ome_zarr_progress_reaches_completion(tmp_path):
    ds = _dataset()
    updates = []
    write_ome_zarr(
        ds,
        tmp_path / "progress",
        pixel_size_nm=5.0,
        z_voxel_nm=10.0,
        progress=lambda fraction, stage: updates.append((fraction, stage)),
    )
    assert updates[0][0] > 0.0
    assert updates[-1] == (1.0, "OME-Zarr export complete")
    assert any("voxel pyramid" in stage for _fraction, stage in updates)
    assert all(a[0] <= b[0] for a, b in zip(updates, updates[1:]))


def test_ome_zarr_preserves_counts_across_3d_chunks_and_levels(tmp_path):
    ds = _dataset(
        n=32,
        x_span_nm=3_000.0,
        y_span_nm=2_200.0,
        z_span_nm=500.0,
    )
    result = write_ome_zarr(
        ds,
        tmp_path / "multi-chunk",
        pixel_size_nm=5.0,
        z_voxel_nm=10.0,
        max_levels=5,
    )
    assert result.levels == 3

    for level in range(result.levels):
        chunk_files = [
            path
            for path in (result.path / str(level) / "c").rglob("*")
            if path.is_file()
        ]
        assert len(chunk_files) > 1
        count = sum(
            int(
                np.frombuffer(
                    gzip.decompress(path.read_bytes()),
                    dtype="<u4",
                ).sum()
            )
            for path in chunk_files
        )
        assert count == ds.prop.num_loc
