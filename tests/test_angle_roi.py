"""Angle ROI math — angle ABC in degrees, range [0, 180]."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from minflux_viewer.ui.roi_overlay import _angle_abc


@pytest.mark.parametrize("a,b,c,expected", [
    ([0, 10], [0, 0], [10, 0], 90.0),    # right angle
    ([-5, 0], [0, 0], [10, 0], 180.0),   # straight line
    ([10, 10], [0, 0], [10, 0], 45.0),   # 45 deg
    ([10, 0], [0, 0], [5, 0], 0.0),      # BA and BC same direction
])
def test_angle_abc_known(a, b, c, expected):
    assert abs(_angle_abc(a, b, c) - expected) < 1e-6


def test_vertex_is_middle_point():
    # the angle is at B regardless of A/C ordering
    assert abs(_angle_abc([10, 0], [0, 0], [0, 10]) - 90.0) < 1e-6


def test_degenerate_returns_zero():
    assert _angle_abc([0, 0], [0, 0], [1, 0]) == 0.0   # A == B


def test_always_in_0_180():
    rng = np.random.default_rng(0)
    for _ in range(100):
        a, b, c = rng.standard_normal((3, 2)) * 100
        deg = _angle_abc(a, b, c)
        assert 0.0 <= deg <= 180.0
