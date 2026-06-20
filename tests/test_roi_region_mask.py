"""Enclosed-region masks for all ROI shapes (oval / polygon / freehand)."""

import numpy as np

from minflux_viewer.core.roi import RoiRecord
from minflux_viewer.core.roi_selection import (
    oval_mask,
    polygon_mask,
    roi_region_mask,
)


def test_oval_mask_inside_outside():
    # bounds (10,30,80,40) -> centre (50,50), radii (40,20)
    ov = RoiRecord.create("oval", {"bounds": [10, 30, 80, 40]}, coordinate_space="plot")
    x = np.array([50.0, 50.0, 95.0, 50.0])   # centre, near-centre, far-x, top
    y = np.array([50.0, 49.0, 50.0, 75.0])
    assert oval_mask(x, y, ov).tolist() == [True, True, False, False]


def test_polygon_figure8_both_lobes_inside():
    # self-intersecting figure-8: even-odd encloses BOTH lobes
    fig8 = [(0, 0), (20, -15), (-20, -15), (0, 0), (20, 15), (-20, 15)]
    rec = RoiRecord.create("polygon", {"points": fig8, "closed": True}, coordinate_space="plot")
    x = np.array([0.0, 0.0, 100.0])          # bottom lobe, top lobe, far outside
    y = np.array([-10.0, 10.0, 100.0])
    assert polygon_mask(x, y, rec).tolist() == [True, True, False]


def test_polygon_concave_notch_excluded():
    # a U shape: the notch is outside
    u = [(0, 0), (10, 0), (10, 10), (7, 10), (7, 3), (3, 3), (3, 10), (0, 10)]
    rec = RoiRecord.create("polygon", {"points": u, "closed": True}, coordinate_space="plot")
    x = np.array([5.0, 5.0])
    y = np.array([8.0, 1.0])                  # in the notch (out), in the base (in)
    assert polygon_mask(x, y, rec).tolist() == [False, True]


def test_roi_region_mask_dispatch_and_base():
    fh = RoiRecord.create("freehand", {"points": [[0, 0], [10, 0], [10, 10], [0, 10]],
                                        "closed": True}, coordinate_space="plot")
    x = np.array([5.0, 20.0, 5.0])
    y = np.array([5.0, 20.0, 5.0])
    base = np.array([True, True, False])
    assert roi_region_mask(x, y, fh, base_mask=base).tolist() == [True, False, False]


def test_line_and_point_have_no_region():
    line = RoiRecord.create("line", {"points": [[0, 0], [10, 10]]}, coordinate_space="plot")
    out = roi_region_mask(np.array([0.0, 5.0]), np.array([0.0, 5.0]), line)
    assert out.tolist() == [False, False]
