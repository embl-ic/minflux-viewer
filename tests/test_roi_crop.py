"""ROI crop engine (core.roi_crop) — selection mask + subset, Qt-free."""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.core.roi import RoiRecord
from minflux_viewer.core import roi_crop as RC


def _rect(x, y, w, h, angle=0.0):
    return RoiRecord.create(
        "rectangle", {"bounds": [x, y, w, h], "angle": angle},
        coordinate_space="plot",
    )


def _dataset(x_nm, y_nm, z_nm=None, tid=None):
    n = len(x_nm)
    return build_localization_dataset(
        name="crop_src.mat",
        x_nm=np.asarray(x_nm, float), y_nm=np.asarray(y_nm, float),
        z_nm=None if z_nm is None else np.asarray(z_nm, float),
        tid=None if tid is None else np.asarray(tid),
        tim=np.arange(n, dtype=float),
    )


def test_per_loc_mask_axis_aligned():
    # 4 points; only the one at (300,300) is inside [200,600]×[200,600]
    ds = _dataset([100, 300, 700, 500], [100, 300, 700, 50])
    mask = RC.compute_crop_mask(ds, _rect(200, 200, 400, 400))
    assert mask.tolist() == [False, True, False, False]


def test_z_range_bounds_depth():
    ds = _dataset([300, 300, 300], [300, 300, 300], z_nm=[10, 100, 250])
    rect = _rect(0, 0, 1000, 1000)             # all XY inside
    mask = RC.compute_crop_mask(ds, rect, z_range=(50, 200))
    assert mask.tolist() == [False, True, False]


def test_trace_complete_by_centroid():
    # trace 0: both locs inside; trace 1: centroid (540,300) inside; trace 2:
    # one loc inside one outside but centroid (900,300) OUTSIDE → whole trace out.
    x = [300, 350,  500, 580,  300, 1500]
    y = [300, 300,  300, 300,  300, 300]
    tid = [0, 0, 1, 1, 2, 2]
    ds = _dataset(x, y, tid=tid)
    rect = _rect(200, 200, 400, 400)            # x,y in [200,600]
    per_loc = RC.compute_crop_mask(ds, rect, trace_complete=False)
    complete = RC.compute_crop_mask(ds, rect, trace_complete=True)
    # per-loc: locs with x in [200,600] inside → indices 0..4, not 5 (x=1500)
    assert per_loc.tolist() == [True, True, True, True, True, False]
    # trace-complete: trace0 centroid 325 in, trace1 centroid 540 in, trace2
    # centroid 900 out → both locs of trace2 excluded
    assert complete.tolist() == [True, True, True, True, False, False]


def test_axis_aligned_detection_and_specs():
    ds = _dataset([300], [300])
    rect = _rect(200, 200, 400, 400)
    assert RC.crop_is_axis_aligned(ds, rect) is True
    # a bounding-box crop is always axis-aligned, even of a rotated rect…
    assert RC.crop_is_axis_aligned(ds, _rect(0, 0, 10, 10, angle=30)) is True
    # …but the *exact shape* of a rotated rect is not.
    assert RC.crop_is_axis_aligned(ds, _rect(0, 0, 10, 10, angle=30), exact_shape=True) is False
    specs = RC.crop_filter_specs(rect, z_range=(50, 200))
    by_attr = {s["attribute"]: s for s in specs}
    assert by_attr["xnm"]["lo"] == 200 and by_attr["xnm"]["hi"] == 600
    assert by_attr["ynm"]["lo"] == 200 and by_attr["ynm"]["hi"] == 600
    assert by_attr["znm"]["lo"] == 50 and by_attr["znm"]["hi"] == 200
    assert all(s["mode"] == "per loc" for s in specs)
    assert RC.crop_filter_specs(rect, trace_complete=True)[0]["mode"] == "trace mean"


def test_oval_crop_bbox_vs_exact_shape():
    # 4 corner points just inside the bbox but OUTSIDE the inscribed ellipse,
    # 1 centre point inside both. bbox crop keeps all 5; exact-shape keeps centre.
    oval = RoiRecord.create("oval", {"bounds": [0, 0, 100, 100]}, coordinate_space="plot")
    x = [5, 95, 95, 5, 50]
    y = [5, 5, 95, 95, 50]
    ds = _dataset(x, y)
    bbox = RC.compute_crop_mask(ds, oval, exact_shape=False)
    shape = RC.compute_crop_mask(ds, oval, exact_shape=True)
    assert bbox.tolist() == [True, True, True, True, True]    # bounding box
    assert shape.tolist() == [False, False, False, False, True]  # exact ellipse


def test_polygon_crop_exact_shape():
    # triangle (0,0),(100,0),(0,100): bbox keeps the far corner (90,90); shape drops it
    tri = RoiRecord.create("polygon", {"points": [[0, 0], [100, 0], [0, 100]], "closed": True},
                           coordinate_space="plot")
    ds = _dataset([10, 90], [10, 90])
    assert RC.compute_crop_mask(ds, tri, exact_shape=False).tolist() == [True, True]
    assert RC.compute_crop_mask(ds, tri, exact_shape=True).tolist() == [True, False]


def test_subset_dataset_keeps_in_roi_only():
    x = [100, 300, 700, 500]
    y = [100, 300, 700, 300]
    ds = _dataset(x, y, z_nm=[0, 0, 0, 0])
    rect = _rect(200, 200, 400, 400)
    mask = RC.compute_crop_mask(ds, rect)            # keeps (300,300) and (500,300)
    sub = RC.subset_dataset(ds, mask, name="CROP_src")
    assert sub.prop.num_loc == 2
    np.testing.assert_allclose(sorted(np.asarray(sub.xnm)), [300.0, 500.0], atol=1e-6)
    assert sub.metadata["cropped_from_dataset"] == "crop_src.mat"


@pytest.mark.parametrize("text,n,expected", [
    ("1-3", 3, [1, 2, 3]),
    ("1,3", 3, [1, 3]),
    ("1,2", 3, [1, 2]),
    ("1-2", 5, [1, 2]),
    ("2-1", 3, [1, 2]),          # reversed range
    ("1, 3 ", 3, [1, 3]),        # spaces tolerated
    ("1,9", 3, [1]),             # clamp out-of-range
    ("", 3, []),
    ("x,2", 3, [2]),             # junk ignored
])
def test_parse_channel_spec(text, n, expected):
    assert RC.parse_channel_spec(text, n) == expected


def test_subset_preserves_native_z_no_rimf_bake():
    # 3-D source; subset z must equal native input nm, not RIMF-scaled.
    ds = _dataset([300, 300], [300, 300], z_nm=[40, 120])
    ds.set_rimf(0.8, source="manual (anisotropy plugin)")   # display z would scale
    mask = RC.compute_crop_mask(ds, _rect(0, 0, 1000, 1000))  # keep both
    sub = RC.subset_dataset(ds, mask, name="CROP")
    np.testing.assert_allclose(sorted(np.asarray(sub.attr["loc_z"]) * 1e9), [40.0, 120.0], atol=1e-6)
