"""Minimum-enclosing shape fits (core/min_enclosing.py)."""

import numpy as np
import pytest

from minflux_viewer.core.min_enclosing import (
    axis_aligned_enclosing_ellipse,
    ellipse_from_matrix,
    min_area_rectangle,
    min_enclosing_ellipse,
    min_enclosing_ellipsoid,
)


def _rot(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def test_min_area_rectangle_recovers_rotated_rect():
    w, h, ang = 120.0, 50.0, 30.0
    corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]]) @ _rot(ang).T + [200, 100]
    inside = (np.random.default_rng(0).uniform(-0.45, 0.45, (200, 2)) * [w, h]) @ _rot(ang).T + [200, 100]
    center, W, H, A = min_area_rectangle(np.vstack([corners, inside]))
    assert center == pytest.approx([200, 100], abs=1e-6)
    assert sorted((round(W), round(H))) == [50, 120]              # side lengths (order may swap)
    assert W * H == pytest.approx(w * h, rel=1e-6)               # exact minimum area
    a = A % 90.0
    assert min(a, 90 - a) == pytest.approx(30.0, abs=1e-3)       # orientation (mod 90)


def test_min_area_rectangle_encloses_all_points():
    pts = np.random.default_rng(1).normal(0, [40, 12], (300, 2)) @ _rot(17).T + [5, -8]
    center, W, H, A = min_area_rectangle(pts)
    t = np.radians(A); ex = np.array([np.cos(t), np.sin(t)]); ey = np.array([-ex[1], ex[0]])
    d = pts - center
    assert np.all(np.abs(d @ ex) <= W / 2 + 1e-6) and np.all(np.abs(d @ ey) <= H / 2 + 1e-6)


def test_min_area_rectangle_tighter_than_axis_bbox():
    pts = np.random.default_rng(2).normal(0, [60, 8], (300, 2)) @ _rot(40).T
    _c, W, H, _A = min_area_rectangle(pts)
    bb = float(np.ptp(pts[:, 0])) * float(np.ptp(pts[:, 1]))
    assert W * H < bb                                            # oriented beats axis-aligned


def test_min_enclosing_ellipse_recovers_ellipse():
    a, b, ang = 80.0, 40.0, -20.0
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    ell = np.column_stack([a * np.cos(t), b * np.sin(t)]) @ _rot(ang).T + [50, -30]
    cx, cy, W, H, A = min_enclosing_ellipse(ell)
    assert (cx, cy) == pytest.approx((50, -30), abs=1e-2)
    assert sorted((round(W), round(H))) == [80, 160]            # full axis lengths
    aa = A % 180.0
    assert min(aa, 180 - aa) == pytest.approx(20.0, abs=1.0)


def test_min_enclosing_ellipse_encloses_all_points():
    pts = np.random.default_rng(3).normal(0, [50, 20], (400, 2)) @ _rot(33).T + [10, 10]
    c, mat = min_enclosing_ellipsoid(pts)
    d = pts - c
    assert np.all(np.einsum("ij,jk,ik->i", d, mat, d) <= 1.0 + 1e-6)


def test_min_enclosing_ellipsoid_is_dimension_agnostic_3d():
    """The MVEE works in 3-D unchanged (room for 3-D ROIs later) and encloses all."""
    pts = np.random.default_rng(4).normal(0, [30, 20, 10], (500, 3))
    c, A = min_enclosing_ellipsoid(pts)
    assert A.shape == (3, 3) and c.shape == (3,)
    d = pts - c
    assert np.all(np.einsum("ij,jk,ik->i", d, A, d) <= 1.0 + 1e-6)


def test_ellipse_from_matrix_circle():
    cx, cy, w, h, ang = ellipse_from_matrix([1.0, 2.0], np.eye(2) / (5.0 ** 2))
    assert (cx, cy) == (1.0, 2.0)
    assert w == pytest.approx(10.0) and h == pytest.approx(10.0)   # radius 5 → diameter 10


def test_axis_aligned_enclosing_ellipse_encloses_and_axis_aligned():
    pts = np.random.default_rng(5).normal(0, [50, 15], (300, 2)) @ _rot(35).T + [7, -3]
    cx, cy, w, h = axis_aligned_enclosing_ellipse(pts)
    d = pts - [cx, cy]
    assert np.all((d[:, 0] / (w / 2)) ** 2 + (d[:, 1] / (h / 2)) ** 2 <= 1.0 + 1e-6)  # encloses


def test_axis_aligned_enclosing_ellipse_exact_for_axis_aligned_ellipse():
    a, b = 90.0, 30.0
    t = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    pts = np.column_stack([a * np.cos(t), b * np.sin(t)]) + [4, 6]
    cx, cy, w, h = axis_aligned_enclosing_ellipse(pts)
    assert (cx, cy) == pytest.approx((4, 6), abs=1e-6)
    assert (w, h) == pytest.approx((2 * a, 2 * b), rel=1e-6)   # recovers the ellipse exactly


def test_axis_aligned_min_enclosing_of_rectangle_matches_oriented():
    # a rectangle's minimum enclosing ellipse is itself axis-aligned, so the
    # axis-aligned and oriented (MVEE) fits coincide.
    rect = np.array([[-100, -40], [100, -40], [100, 40], [-100, 40]], float)
    ax = axis_aligned_enclosing_ellipse(rect)
    _cx, _cy, ow, oh, oang = min_enclosing_ellipse(rect)
    assert sorted((round(ax[2]), round(ax[3]))) == sorted((round(ow), round(oh)))
