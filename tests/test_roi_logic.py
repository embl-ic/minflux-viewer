"""ImageJ-style shape operations (Shift adds, Alt removes) and the composite ROI."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from minflux_viewer.core import roi_logic as rl
from minflux_viewer.core.roi import RoiRecord, record_from_imagej, record_to_imagej
from minflux_viewer.core.roi_selection import REGION_ROI_TYPES, roi_region_mask


def _rect(x, y, w, h, **kw):
    return RoiRecord.create("rectangle", {"bounds": [x, y, w, h], **kw})


def _grid(lo=-10.0, hi=60.0, step=0.5):
    g = np.arange(lo, hi, step)
    xx, yy = np.meshgrid(g, g)
    return xx.ravel(), yy.ravel()


def _inside(record, x, y):
    return roi_region_mask(x, y, record)


def test_union_of_overlapping_rectangles_is_one_polygon():
    out = rl.combine(_rect(0, 0, 20, 20), _rect(10, 10, 20, 20), "or")
    assert out.type == "polygon"
    x, y = _grid()
    m = _inside(out, x, y)
    want = (((x > 0) & (x < 20) & (y > 0) & (y < 20))
            | ((x > 10) & (x < 30) & (y > 10) & (y < 30)))
    border = ((np.isin(x, [0, 10, 20, 30])) | (np.isin(y, [0, 10, 20, 30])))
    assert np.array_equal(m[~border], want[~border])


def test_alt_cuts_a_hole_and_the_result_is_a_composite():
    out = rl.combine(_rect(0, 0, 40, 40), _rect(10, 10, 10, 10), "not")
    assert out.type == "composite" and len(out.geometry["rings"]) == 2
    assert out.type in REGION_ROI_TYPES
    x = np.array([5.0, 15.0, 35.0, 45.0])
    y = np.array([5.0, 15.0, 35.0, 45.0])
    assert _inside(out, x, y).tolist() == [True, False, True, False]   # hole excluded


def test_disjoint_union_keeps_two_pieces_and_not_the_gap():
    out = rl.combine(_rect(0, 0, 10, 10), _rect(30, 0, 10, 10), "or")
    assert out.type == "composite"
    m = _inside(out, np.array([5.0, 20.0, 35.0]), np.array([5.0, 5.0, 5.0]))
    assert m.tolist() == [True, False, True]


def test_removing_everything_leaves_nothing():
    assert rl.combine(_rect(10, 10, 5, 5), _rect(0, 0, 40, 40), "not") is None


def test_a_box_that_remains_a_box_is_a_rectangle():
    out = rl.combine(_rect(0, 0, 40, 40), _rect(10, 10, 5, 5), "or")
    assert out.type == "rectangle"
    assert out.geometry["bounds"] == pytest.approx([0, 0, 40, 40])


def test_rotation_is_honoured():
    rotated = _rect(0, 0, 40, 10, angle=90.0)          # becomes 10 wide, 40 tall about (20, 5)
    out = rl.combine(rotated, _rect(100, 100, 1, 1), "or")
    m = _inside(out, np.array([20.0, 20.0, 35.0]), np.array([5.0, 20.0, 5.0]))
    assert m.tolist() == [True, True, False]


def test_a_line_takes_part_at_one_nanometre():
    line = RoiRecord.create("line", {"points": [[0.0, 0.0], [20.0, 0.0]], "closed": False})
    out = rl.combine(line, _rect(30, -5, 5, 10), "or")
    m = _inside(out, np.array([10.0, 10.0, 10.0, -0.4, 20.4, 32.0]),
                np.array([0.0, 0.45, 0.6, 0.0, 0.0, 0.0]))
    # inside the 1 nm band (±0.5 nm), its square ends extended by 0.5 nm, and the box
    assert m.tolist() == [True, True, False, True, True, True]


def test_which_shapes_can_combine():
    area, line = _rect(0, 0, 1, 1), RoiRecord.create("line", {"points": [[0, 0], [1, 1]]})
    point = RoiRecord.create("point", {"point": [0, 0]})
    assert rl.can_combine(area, "rectangle") and rl.can_combine(line, "oval")
    assert rl.can_combine(area, "magic_wand") and rl.can_combine(area, "ellipse")
    assert not rl.can_combine(area, "line")          # a line tool ignores the modifier
    assert not rl.can_combine(point, "rectangle")    # points never mix with areas
    assert not rl.can_combine(None, "rectangle")


def test_composite_keeps_its_scope_and_colour():
    prev = _rect(0, 0, 40, 40)
    prev.context = {"dataset_idx": 3, "source_view": "render"}
    prev.stroke_color = "#ff00ff"
    out = rl.combine(prev, _rect(10, 10, 5, 5), "not")
    assert out.context["dataset_idx"] == 3 and out.stroke_color == "#ff00ff"


def test_imagej_composite_round_trip():
    from roifile import ImagejRoi

    out = rl.combine(_rect(0, 0, 40, 40), _rect(10, 10, 10, 10), "not")
    roi = record_to_imagej(out)
    assert roi.composite
    back = record_from_imagej(ImagejRoi.frombytes(roi.tobytes()))
    assert back.type == "composite"
    x, y = _grid(-5, 45, 1.0)
    assert np.array_equal(_inside(back, x + 0.5, y + 0.5), _inside(out, x + 0.5, y + 0.5))


def test_xor_and_and_are_available():
    a, b = _rect(0, 0, 20, 20), _rect(10, 0, 20, 20)
    both = rl.combine(a, b, "and")
    assert both.type == "rectangle" and both.geometry["bounds"] == pytest.approx([10, 0, 10, 20])
    either = rl.combine(a, b, "xor")
    assert _inside(either, np.array([5.0, 15.0, 25.0]), np.array([5.0, 5.0, 5.0])).tolist() == \
        [True, False, True]


def _oval(x, y, w, h, **kw):
    return RoiRecord.create("oval", {"bounds": [x, y, w, h], **kw})


@pytest.mark.parametrize("oval_box", [(15.0, 5.0, 30.0, 30.0),      # overlapping: one ring
                                      (60.0, 5.0, 20.0, 20.0)])     # disjoint: two rings
def test_rectangle_and_oval_combine_to_the_same_composite_in_either_order(oval_box):
    """A flattened arc is not a set of corners: rectangle ∪ oval is a composite
    (outline, no vertex handles) whichever is drawn first, overlapping or not."""
    rect, oval = _rect(0, 0, 25, 20), _oval(*oval_box)
    a, b = rl.combine(rect, oval, "or"), rl.combine(oval, rect, "or")
    assert a.type == b.type == "composite"
    assert a.geometry["polygonal"] is False and b.geometry["polygonal"] is False
    assert len(a.geometry["rings"]) == len(b.geometry["rings"])
    x, y = _grid(-5, 90, 0.7)
    assert np.array_equal(_inside(a, x, y), _inside(b, x, y))


def test_flattened_arc_stays_within_a_fraction_of_a_nanometre():
    """Qt flattens the oval into short straight segments; how far they stray
    from the true ellipse decides whether the membership is still exact."""
    out = rl.combine(_rect(0, 0, 600, 600), _oval(400, 100, 1200, 800), "or")
    ring = np.asarray(out.geometry["rings"][0], float)
    cx, cy, ax, ay = 1000.0, 500.0, 600.0, 400.0
    on_arc = ring[(ring[:, 0] > 600.5) | (ring[:, 1] > 600.5)]   # off the rectangle's edges
    assert len(on_arc) > 50
    r = np.hypot((on_arc[:, 0] - cx) / ax, (on_arc[:, 1] - cy) / ay)
    assert np.max(np.abs(r - 1.0)) * min(ax, ay) < 0.5           # nm


def test_a_polygonal_composite_becomes_a_polygon_again():
    holed = rl.combine(_rect(0, 0, 40, 40), _rect(10, 10, 10, 10), "not")
    assert holed.type == "composite" and holed.geometry["polygonal"] is True
    filled = rl.combine(holed, _rect(5, 5, 20, 20), "or")
    assert filled.type == "rectangle"                    # the hole closed again
    notched = rl.combine(_rect(0, 0, 40, 40), _rect(30, 30, 20, 20), "not")
    assert notched.type == "polygon" and len(notched.geometry["points"]) == 6


def test_traced_and_curved_inputs_are_not_polygonal():
    free = RoiRecord.create("freehand", {"points": [[0, 0], [20, 0], [20, 20], [0, 20]],
                                         "closed": True})
    assert not rl.record_is_polygonal(free) and not rl.record_is_polygonal(_oval(0, 0, 5, 5))
    assert rl.record_is_polygonal(_rect(0, 0, 5, 5, angle=30.0))
    assert rl.combine(free, _rect(10, 10, 20, 20), "or").type == "composite"
    old = RoiRecord.create("composite", {"rings": [[[0, 0], [1, 0], [1, 1]]]})
    assert not rl.record_is_polygonal(old)              # no flag: not assumed polygonal


def test_a_composite_offers_the_enclosing_fits():
    from minflux_viewer.core.roi_convert import available_conversions, convert_roi

    comp = rl.combine(_rect(0, 0, 25, 20), _oval(15, 5, 30, 30), "or")
    targets = available_conversions(comp)
    for t in ("bounding_box", "rectangle", "oval", "ellipse", "convex_hull", "point"):
        assert t in targets
    hull = convert_roi(comp, "convex_hull")
    assert hull.type == "polygon"
    box = convert_roi(comp, "bounding_box")
    assert box.geometry["bounds"] == pytest.approx([0, 0, 45, 35], abs=0.1)
