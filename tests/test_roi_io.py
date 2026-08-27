"""ROI-set I/O: native JSON detection, and ImageJ export of plot-coordinate ROIs
(no longer blocked — coords written as-is, round-trips through .zip)."""

import json

import numpy as np
import pytest

from minflux_viewer.core.roi import (
    RoiRecord,
    RoiStore,
    is_roi_json_file,
    record_from_imagej,
    record_to_imagej,
)


def _freehand(points, name="freehand-1"):
    return RoiRecord.create("freehand", {"points": points, "closed": True},
                            name=name, coordinate_space="plot")


def test_is_roi_json_file(tmp_path):
    p = tmp_path / "rois.json"
    p.write_text(json.dumps({"version": 1, "rois": [
        {"id": "a", "name": "f", "type": "freehand",
         "geometry": {"points": [[0, 0], [10, 0], [5, 9]], "closed": True}}]}))
    assert is_roi_json_file(p) is True
    # a plain list (filter preset) or non-roi dict is not a ROI set
    (tmp_path / "list.json").write_text(json.dumps([{"attribute": "efo"}]))
    assert is_roi_json_file(tmp_path / "list.json") is False
    (tmp_path / "meta.json").write_text(json.dumps({"z_scaling_factor": 0.67}))
    assert is_roi_json_file(tmp_path / "meta.json") is False
    assert is_roi_json_file(tmp_path / "missing.json") is False


def test_roi_json_round_trip(tmp_path):
    store = RoiStore()
    pts = [[-21939.0, -6897.1], [-22000.0, -6900.0], [-22050.5, -6950.2]]
    store.add(_freehand(pts))
    out = tmp_path / "set.json"
    store.save(out, store.records)
    assert is_roi_json_file(out)
    loaded = RoiStore().load(out)
    assert len(loaded) == 1 and loaded[0].type == "freehand"
    np.testing.assert_allclose(loaded[0].geometry["points"], pts)


def test_plot_coordinate_freehand_exports_to_imagej_zip(tmp_path):
    pytest.importorskip("roifile")
    store = RoiStore()
    pts = [[-21939.0, -6897.1], [-22000.0, -6900.0], [-22050.5, -6950.2], [-21980.0, -7010.0]]
    store.add(_freehand(pts, name="freehand-1"))
    store.add(_freehand([[100.0, 100.0], [200.0, 100.0], [150.0, 220.0]], name="freehand-2"))
    # Previously this raised / was blocked for plot-coordinate ROIs.
    out = tmp_path / "rois.zip"
    store.save(out, store.records)
    assert out.exists() and out.stat().st_size > 0
    back = RoiStore().load(out)
    assert len(back) == 2
    # first vertex (nm) survives the ImageJ round-trip (written as pixel coords)
    assert np.allclose(back[0].geometry["points"][0], pts[0], atol=1.0)


def test_rotated_rectangle_exports_as_polygon(tmp_path):
    roifile = pytest.importorskip("roifile")
    rec = RoiRecord.create("rectangle", {"bounds": [0, 0, 100, 40], "angle": 30.0},
                           coordinate_space="plot")
    ij = record_to_imagej(rec)
    # a rotated box can't be an axis-aligned ImageJ RECT → exported as a polygon
    assert ij.roitype in (roifile.ROI_TYPE.POLYGON, roifile.ROI_TYPE.FREEHAND)
    back = record_from_imagej(ij)
    pts = np.asarray(back.geometry["points"])
    assert pts.shape[0] >= 4
    # the polygon is rotated (not axis-aligned): some vertex y differs from the corners
    assert pts[:, 1].max() > 40  # rotation pushes a corner beyond the unrotated height


def test_loading_a_roi_set_tolerates_keys_the_record_does_not_have():
    """``RoiRecord(**item)`` raised on any extra key, with a message about
    ``__init__`` that said nothing about the file.

    Saved sets legitimately carry extras — ``dataset_id`` is written per member
    of a Zarr project — and a set written by a newer version will carry more.
    """
    from minflux_viewer.core.roi import roi_record_from_dict

    record = roi_record_from_dict({
        "id": "abc", "name": "rectangle-1", "type": "rectangle",
        "geometry": {"bounds": [0.0, 0.0, 10.0, 10.0]},
        "dataset_id": "d000000", "a_field_from_the_future": 7,
    })
    assert record.name == "rectangle-1" and record.type == "rectangle"
    # Dropped, but named rather than silently discarded.
    assert record.context["unrecognised_keys"] == ["a_field_from_the_future",
                                                   "dataset_id"]


def test_a_roi_set_with_extra_keys_loads_through_the_store(tmp_path):
    import json

    from minflux_viewer.core.roi import RoiStore

    path = tmp_path / "regions.json"
    path.write_text(json.dumps({"version": 1, "rois": [
        {"id": "a", "name": "r1", "type": "rectangle",
         "geometry": {"bounds": [0, 0, 4, 4]}, "dataset_id": "d0"},
    ]}), encoding="utf-8")

    store = RoiStore()
    records = store.load(path)
    assert len(records) == 1 and records[0].name == "r1"
