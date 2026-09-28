"""The orthogonal grid's fourth cell: a rotating projection."""

import numpy as np
import pytest

from minflux_viewer.ui.ortho_rotation import (
    ROTATION_AXES,
    rotated_projection,
    rotation_axis_labels,
)


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


@pytest.mark.parametrize(
    "mode,bottom,left",
    [
        ("about Y", "X cos θ + Z sin θ (nm)", "Y — rotation axis (nm)"),
        ("about X", "Y cos θ + Z sin θ (nm)", "X — rotation axis (nm)"),
        ("about Z", "X cos θ + Y sin θ (nm)", "Z — rotation axis (nm)"),
    ],
)
def test_rotation_axis_labels_name_the_mixed_and_held_axes(mode, bottom, left):
    assert rotation_axis_labels(mode) == (bottom, left)


def test_a_bad_mode_is_refused_rather_than_silently_defaulted():
    with pytest.raises(ValueError, match="unknown rotation mode"):
        rotated_projection(_cube(), 10.0, "about W")
    with pytest.raises(ValueError, match="unknown rotation mode"):
        rotation_axis_labels("about W")


def test_two_d_input_yields_nothing():
    assert rotated_projection(np.zeros((5, 2)), 0.0) is None


# --------------------------------------------- spinning about the crosshair
# ⚠ Without a pivot the spin is about the coordinate ORIGIN, and MINFLUX
# coordinates sit tens of microns from it: a 15 degree turn moved a cloud at
# x = 15000 nm by 485 nm, and 90 degrees put it clean off the pane.
def _far_from_origin():
    return np.array([[15000.0, 20000.0, 100.0],
                     [15050.0, 20000.0, 100.0],
                     [15000.0, 20000.0, 200.0]])


def test_a_pivot_keeps_the_zero_degree_identity():
    pts = _far_from_origin()
    pivot = pts[0]
    h, v = rotated_projection(pts, 0.0, "about Y", centre=pivot)
    assert np.allclose(h, pts[:, 0])
    assert np.allclose(v, pts[:, 1])


@pytest.mark.parametrize("angle", [0.0, 37.0, 90.0, 180.0, 271.0])
@pytest.mark.parametrize("mode", sorted(ROTATION_AXES))
def test_the_pivot_maps_to_itself_at_every_angle(angle, mode):
    """That is what keeps the marked feature in place while its surroundings
    turn -- and it is why the pane can be centred on the pivot."""
    pivot = np.array([15000.0, 20000.0, 100.0])
    axis_a, _axis_b, axis_up = ROTATION_AXES[mode]
    h, v = rotated_projection(pivot[None, :], angle, mode, centre=pivot)
    assert np.isclose(h[0], pivot[axis_a])
    assert np.isclose(v[0], pivot[axis_up])


def test_one_hundred_eighty_degrees_reflects_through_the_pivot():
    pts = _far_from_origin()
    pivot = pts[0]
    h, _v = rotated_projection(pts, 180.0, "about Y", centre=pivot)
    assert np.isclose(h[1], 14950.0)          # 15050 mirrored about 15000


def test_without_a_pivot_a_far_cloud_swings_away_from_the_pane():
    """The behaviour the pivot exists to fix, pinned so it cannot come back."""
    pts = _far_from_origin()
    h_origin, _ = rotated_projection(pts, 15.0, "about Y")
    h_pivot, _ = rotated_projection(pts, 15.0, "about Y", centre=pts[0])
    assert abs(h_origin[0] - 15000.0) > 400.0     # ~485 nm off
    assert np.isclose(h_pivot[0], 15000.0)        # fixed


def test_a_non_finite_pivot_is_ignored_rather_than_poisoning_the_projection():
    pts = _far_from_origin()
    for bad in (np.array([np.nan, 0.0, 0.0]), np.array([0.0, np.inf, 0.0])):
        h, v = rotated_projection(pts, 0.0, "about Y", centre=bad)
        assert np.all(np.isfinite(h)) and np.all(np.isfinite(v))
