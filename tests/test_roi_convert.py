import numpy as np
import pytest

from minflux_viewer.core.roi import RoiRecord
from minflux_viewer.core.roi_convert import (
    available_conversions,
    can_resize,
    convert_roi,
    enlarge_shrink_roi,
    skeletonize_roi,
)


def _rect(x=0, y=0, w=100, h=200):
    return RoiRecord.create("rectangle", {"bounds": [x, y, w, h]}, name="rectangle-1")


def _oval(x=0, y=0, w=100, h=200):
    return RoiRecord.create("oval", {"bounds": [x, y, w, h]}, name="oval-1")


def _polygon(points):
    return RoiRecord.create("polygon", {"points": points, "closed": True}, name="polygon-1")


def _polyline(points):
    return RoiRecord.create("polyline", {"points": points, "closed": False}, name="polyline-1")


def _point(x=10, y=20):
    return RoiRecord.create("point", {"point": [x, y, 0.0]}, name="point-1")


# -- available_conversions ------------------------------------------------

def test_available_conversions_by_type():
    assert set(available_conversions(_rect())) >= {"point", "oval", "line"}
    assert "rectangle" not in available_conversions(_rect())  # already a rectangle
    # a point can't convert to a point (same-type is meaningless)
    assert set(available_conversions(_point())) == {"rectangle", "oval"}
    # line types only grow into a region (box/oval not allowed)
    poly_targets = set(available_conversions(_polyline([[0, 0], [10, 0], [10, 10]])))
    assert poly_targets == {"point", "region", "polygon"}
    assert "rectangle" not in poly_targets and "oval" not in poly_targets
    assert available_conversions(RoiRecord.create("angle", {"points": [[0, 0], [1, 1], [2, 0]]})) == ["point"]
    assert "convex_hull" in available_conversions(_polygon([[0, 0], [10, 0], [5, 10]]))


# -- to point -------------------------------------------------------------

def test_rect_to_point_uses_centroid():
    out = convert_roi(_rect(0, 0, 100, 200), "point")
    assert out.type == "point"
    assert out.geometry["point"][:2] == [50.0, 100.0]


def test_polygon_to_point_vertex_mean():
    out = convert_roi(_polygon([[0, 0], [10, 0], [10, 10], [0, 10]]), "point")
    assert out.geometry["point"][:2] == [5.0, 5.0]


# -- bounding box / oval --------------------------------------------------

def test_oval_to_bounding_rectangle():
    out = convert_roi(_oval(0, 0, 100, 200), "rectangle")
    assert out.type == "rectangle"
    assert out.geometry["bounds"] == [0.0, 0.0, 100.0, 200.0]


def test_polygon_to_bounding_oval_oriented():
    # axis-aligned 40×60 rectangle polygon → oriented box of the same area/extent
    out = convert_roi(_polygon([[0, 0], [40, 0], [40, 60], [0, 60]]), "oval")
    assert out.type == "oval"
    x, y, w, h = out.geometry["bounds"]
    assert sorted((round(w), round(h))) == [40, 60]            # same side lengths
    assert (x + w / 2, y + h / 2) == pytest.approx((20.0, 30.0))  # same centre
    assert "angle" in out.geometry


def test_polygon_to_rectangle_oriented_for_rotated_shape():
    # a 45°-rotated thin rectangle → oriented box with ~45° angle
    c = np.array([100.0, 100.0])
    u = np.array([1.0, 1.0]) / np.sqrt(2)   # long axis (45°)
    v = np.array([-1.0, 1.0]) / np.sqrt(2)  # short axis
    corners = [c + 50 * u + 5 * v, c - 50 * u + 5 * v, c - 50 * u - 5 * v, c + 50 * u - 5 * v]
    out = convert_roi(_polygon([p.tolist() for p in corners]), "rectangle")
    assert out.type == "rectangle"
    ang = abs(float(out.geometry.get("angle", 0.0))) % 180
    assert min(abs(ang - 45), abs(ang - 135)) < 5  # oriented near 45°, not axis-aligned
    x, y, w, h = out.geometry["bounds"]
    assert max(w, h) == pytest.approx(100.0, abs=2) and min(w, h) == pytest.approx(10.0, abs=2)


# -- point sizing ---------------------------------------------------------

def test_point_to_oval_needs_size():
    with pytest.raises(ValueError):
        convert_roi(_point(), "oval")
    out = convert_roi(_point(10, 20), "oval", width=30, height=40)
    assert out.type == "oval"
    assert out.geometry["bounds"] == [10 - 15.0, 20 - 20.0, 30.0, 40.0]


# -- polyline / convex hull ----------------------------------------------

def test_polyline_to_polygon_closes():
    out = convert_roi(_polyline([[0, 0], [10, 0], [10, 10]]), "polygon")
    assert out.type == "polygon" and out.geometry["closed"] is True
    assert len(out.geometry["points"]) == 3


def test_polygon_to_convex_hull():
    # a concave 'star-ish' set; hull should drop the interior reflex vertex.
    pts = [[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]]
    out = convert_roi(_polygon(pts), "convex_hull")
    assert out.type == "polygon"
    hull = np.asarray(out.geometry["points"])
    assert hull.shape[0] == 4  # the (5,5) interior point is excluded


# -- line <-> region ------------------------------------------------------

def test_line_to_region_buffer():
    line = RoiRecord.create("line", {"points": [[0, 0], [100, 0]], "closed": False})
    out = convert_roi(line, "region", width=20)
    assert out.type == "polygon"
    pts = np.asarray(out.geometry["points"])
    assert pts.shape[0] >= 4
    # buffer extends ~10 nm on each side of the y=0 line
    assert pts[:, 1].max() > 8 and pts[:, 1].min() < -8


def test_region_to_line_is_enclosed_outline():
    from minflux_viewer.core.roi_selection import roi_region_mask

    out = convert_roi(_rect(0, 0, 100, 60), "line")
    assert out.type == "polyline" and out.geometry["closed"] is True
    pts = np.asarray(out.geometry["points"])
    assert pts.shape[0] == 4                       # rectangle outline corners
    assert pts[:, 0].min() == pytest.approx(0) and pts[:, 0].max() == pytest.approx(100)
    assert pts[:, 1].min() == pytest.approx(0) and pts[:, 1].max() == pytest.approx(60)
    # a closed line has NO enclosed-area mask (nothing is "within")
    assert roi_region_mask(np.array([50.0]), np.array([30.0]), out).tolist() == [False]


def test_oval_to_line_outline():
    out = convert_roi(_oval(0, 0, 100, 60), "line")
    assert out.type == "polyline" and out.geometry["closed"] is True
    assert np.asarray(out.geometry["points"]).shape[0] >= 16


# -- enlarge / shrink -----------------------------------------------------

def test_enlarge_rectangle_bounds():
    out = enlarge_shrink_roi(_rect(0, 0, 100, 100), 10, mode="enlarge")
    assert out.geometry["bounds"] == [-10.0, -10.0, 120.0, 120.0]


def test_shrink_rectangle_bounds():
    out = enlarge_shrink_roi(_rect(0, 0, 100, 100), 10, mode="shrink")
    assert out.geometry["bounds"] == [10.0, 10.0, 80.0, 80.0]


def test_shrink_too_much_raises():
    with pytest.raises(ValueError):
        enlarge_shrink_roi(_rect(0, 0, 10, 10), 20, mode="shrink")


def test_point_enlarge_to_oval():
    out = enlarge_shrink_roi(_point(0, 0), 50, mode="enlarge")
    assert out.type == "oval"
    assert out.geometry["bounds"] == [-25.0, -25.0, 50.0, 50.0]
    with pytest.raises(ValueError):
        enlarge_shrink_roi(_point(0, 0), 50, mode="shrink")


def test_line_enlarge_to_region():
    line = RoiRecord.create("line", {"points": [[0, 0], [100, 0]], "closed": False})
    out = enlarge_shrink_roi(line, 10, mode="enlarge")
    assert out.type == "polygon"


def test_polygon_enlarge_scales_bbox():
    out = enlarge_shrink_roi(_polygon([[0, 0], [100, 0], [100, 100], [0, 100]]), 10, mode="enlarge")
    pts = np.asarray(out.geometry["points"])
    assert pts[:, 0].min() == pytest.approx(-10) and pts[:, 0].max() == pytest.approx(110)


def test_angle_cannot_resize():
    angle = RoiRecord.create("angle", {"points": [[0, 0], [1, 1], [2, 0]]})
    assert not can_resize(angle)
    with pytest.raises(ValueError):
        enlarge_shrink_roi(angle, 10)


def test_skeletonize_rect_is_centerline():
    out = skeletonize_roi(_rect(0, 0, 20, 200))  # rect → centreline (2-pt line)
    assert out.type == "line"
    assert np.asarray(out.geometry["points"]).shape == (2, 2)


def test_skeletonize_curved_polygon_is_shape_aware():
    # an L-shaped polygon → a multi-vertex polyline that turns the corner
    poly = [[0, 0], [100, 0], [100, 20], [20, 20], [20, 100], [0, 100]]
    out = skeletonize_roi(_polygon(poly))
    assert out.type == "polyline"          # not a straight 2-point line
    pts = np.asarray(out.geometry["points"])
    assert pts.shape[0] > 2
    # the skeleton spans both arms of the L (wide x-range and y-range)
    assert np.ptp(pts[:, 0]) > 40 and np.ptp(pts[:, 1]) > 40


def test_skeletonize_square_is_horizontal_centerline():
    out = skeletonize_roi(_rect(0, 0, 100, 100))  # square → rotated-horizontal axis
    assert out.type == "line"
    pts = np.asarray(out.geometry["points"])
    assert pts[:, 1].tolist() == pytest.approx([50.0, 50.0])           # horizontal
    assert sorted(pts[:, 0].tolist()) == pytest.approx([0.0, 100.0])   # spans the width


def test_skeletonize_rotated_rect_major_axis():
    rec = RoiRecord.create("rectangle", {"bounds": [0, 0, 100, 20], "angle": 90.0})
    out = skeletonize_roi(rec)  # w>h → local-X axis, rotated 90° → vertical, length 100
    pts = np.asarray(out.geometry["points"])
    assert abs(pts[0, 0] - pts[1, 0]) < 1e-6                  # vertical
    assert abs(pts[0, 1] - pts[1, 1]) == pytest.approx(100.0, abs=1e-6)


def test_enclosed_line_enlarges_to_ring_band():
    from minflux_viewer.core.roi_selection import polygon_mask

    outline = convert_roi(_rect(-50, -50, 100, 100), "line")  # enclosed outline
    band = enlarge_shrink_roi(outline, 10, mode="enlarge")
    assert band.type == "polygon"
    # on the outline → inside the band; centre → in the hole; far → outside
    m = polygon_mask(np.array([50.0, 0.0, 500.0]), np.array([0.0, 0.0, 500.0]), band)
    assert m.tolist() == [True, False, False]


def test_convert_preserves_context_and_color():
    rec = _rect()
    rec.context = {"dataset_idx": 3, "view_plane": "XY", "depth_value": 42.0}
    rec.stroke_color = "#00ff00"
    out = convert_roi(rec, "point")
    assert out.context["dataset_idx"] == 3
    assert out.stroke_color == "#00ff00"
    assert out.geometry["point"][2] == 42.0  # depth carried onto the point
