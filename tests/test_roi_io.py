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
    (tmp_path / "meta.json").write_text(json.dumps({"rimf": 0.67}))
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
