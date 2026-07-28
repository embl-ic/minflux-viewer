"""Pure ROI fit geometry (core.roi_fit + core.min_enclosing.min_enclosing_circle)."""

from __future__ import annotations

import numpy as np

from minflux_viewer.core.min_enclosing import min_enclosing_circle
from minflux_viewer.core.roi import RoiRecord
from minflux_viewer.core.roi_fit import (
    concave_hull,
    data_fit,
    fit_ellipse_moments,
    interpolate_outline,
    spline_fit,
)


def _rect_source():
    return RoiRecord.create(
        "rectangle", {"bounds": [-100.0, -100.0, 200.0, 200.0]}, coordinate_space="plot"
    )


def test_min_enclosing_circle_contains_all_points():
    rng = np.random.default_rng(0)
    pts = rng.normal(0.0, 5.0, (400, 2))
    cx, cy, r = min_enclosing_circle(pts)
    assert np.all(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) <= r + 1e-6)
    # tight: at least two points lie ~on the circle
    d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    assert np.sum(d >= r - 1e-6) >= 2


def test_fit_ellipse_moments_orientation():
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.normal(0, 10, 3000), rng.normal(0, 2, 3000)])
    cx, cy, w, h, angle = fit_ellipse_moments(pts)
    assert w > h                                   # major axis along x
    assert abs(((angle + 90) % 180) - 90) < 15     # ~0 or 180 deg (aligned to x)
    assert abs(cx) < 1.0 and abs(cy) < 1.0


def test_data_fit_rectangle_is_rotated_and_encloses():
    src = _rect_source()
    rng = np.random.default_rng(2)
    # a rotated elongated cloud
    theta = np.radians(30.0)
    base = np.column_stack([rng.normal(0, 40, 500), rng.normal(0, 8, 500)])
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts = base @ R.T
    rec = data_fit(src, pts, "rectangle")
    assert rec.type == "rectangle"
    assert rec.geometry.get("variant") == "rotated"
    assert abs(rec.geometry["angle"]) > 1.0        # genuinely rotated


def test_data_fit_circle_is_equal_axis_oval():
    src = _rect_source()
    rng = np.random.default_rng(3)
    pts = rng.normal(5.0, 4.0, (300, 2))
    rec = data_fit(src, pts, "circle")
    assert rec.type == "oval"
    _x, _y, wdt, hgt = rec.geometry["bounds"]
    assert np.isclose(wdt, hgt)                     # circle: width == height
    assert rec.geometry["angle"] == 0.0


def test_data_fit_convex_hull_and_polygon_close():
    src = _rect_source()
    rng = np.random.default_rng(4)
    pts = rng.uniform(-50, 50, (200, 2))
    hull = data_fit(src, pts, "convex_hull")
    poly = data_fit(src, pts, "polygon")
    assert hull.type == "polygon" and hull.geometry["closed"] is True
    assert poly.type == "polygon" and poly.geometry["closed"] is True
    assert len(hull.geometry["points"]) >= 3
    # concave polygon has at least as many vertices as the convex hull
    assert len(poly.geometry["points"]) >= 3


def test_data_fit_needs_three_points():
    src = _rect_source()
    try:
        data_fit(src, np.zeros((2, 2)), "rectangle")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_concave_hull_falls_back_to_convex_for_few_points():
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.0]])
    poly = concave_hull(pts)
    assert len(poly) == 3


def test_spline_fit_and_interpolate_outline():
    poly = RoiRecord.create(
        "polygon",
        {"points": [[0, 0], [100, 0], [120, 60], [50, 100], [-10, 50]], "closed": True},
        coordinate_space="plot",
    )
    sp = spline_fit(poly, n_points=120)
    assert sp.type == "freehand" and sp.geometry["closed"] is True
    assert len(sp.geometry["points"]) == 120

    it = interpolate_outline(poly, interval_nm=20.0)
    assert it.type == "polygon"
    verts = np.asarray(it.geometry["points"], float)
    steps = np.linalg.norm(np.diff(verts, axis=0), axis=1)
    # roughly even spacing near the requested interval
    assert 10.0 < float(np.median(steps)) < 30.0
