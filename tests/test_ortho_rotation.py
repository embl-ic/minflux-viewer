"""The orthogonal grid's fourth cell: a rotating projection."""

import numpy as np
import pytest

from minflux_viewer.ui.ortho_rotation import ROTATION_AXES, rotated_projection


def _cube():
    g = np.linspace(-1.0, 1.0, 5)
    return np.array([[x, y, z] for x in g for y in g for z in g], dtype=float)


def test_zero_degrees_reproduces_the_neighbouring_projection():
    """It reads as a departure from a known picture, not a new one."""
    pts = _cube()
    h, v = rotated_projection(pts, 0.0, "about Y")
    assert np.allclose(h, pts[:, 0])      # X horizontal
    assert np.allclose(v, pts[:, 1])      # Y vertical -- the XY view


def test_ninety_degrees_shows_the_other_axis():
    pts = _cube()
    h, v = rotated_projection(pts, 90.0, "about Y")
    assert np.allclose(h, pts[:, 2], atol=1e-9)    # Z has swung into view
    assert np.allclose(v, pts[:, 1])               # the held axis is unmoved


def test_the_held_axis_never_moves_at_any_angle():
    """That is what keeps the pane comparable with the fixed ones beside it."""
    pts = _cube()
    for angle in (0.0, 17.0, 90.0, 180.0, 271.0, 360.0):
        _h, v = rotated_projection(pts, angle, "about Y")
        assert np.allclose(v, pts[:, 1])


def test_rotation_preserves_the_in_plane_radius():
    """A projection of a rotation, so the mixed pair keeps its length."""
    pts = _cube()
    r0 = np.hypot(pts[:, 0], pts[:, 2])
    for angle in (13.0, 45.0, 200.0):
        h, _v = rotated_projection(pts, angle, "about Y")
        assert np.all(np.abs(h) <= r0 + 1e-9)


def test_every_mode_holds_a_different_axis():
    pts = _cube()
    for mode, (_a, _b, up) in ROTATION_AXES.items():
        _h, v = rotated_projection(pts, 33.0, mode)
        assert np.allclose(v, pts[:, up]), mode


def test_a_bad_mode_is_refused_rather_than_silently_defaulted():
    with pytest.raises(ValueError, match="unknown rotation mode"):
        rotated_projection(_cube(), 10.0, "about W")


def test_two_d_input_yields_nothing():
    assert rotated_projection(np.zeros((5, 2)), 0.0) is None
