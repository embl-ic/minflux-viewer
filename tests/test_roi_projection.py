"""A ROI drawn in one plane, seen in another.

Pure NumPy. The point of this layer is that a ROI is visible in all three ortho
panes — and that a *flat* one is visibly flat rather than pretending to a depth
it does not constrain.
"""

import pytest

from minflux_viewer.core.roi_projection import (
    DEGENERATE_LINE,
    FULL,
    PLANE_COLUMNS,
    flat_extent,
    project_flat_record,
)


class Rec:
    def __init__(self, type, geometry):
        self.type = type
        self.geometry = geometry


def _rect():
    return Rec("rectangle", {"bounds": [100.0, 200.0, 50.0, 80.0]})


# --------------------------------------------------------------------- extent
def test_extent_is_keyed_by_data_axis_not_by_position_in_the_geometry():
    """A rectangle drawn in XZ bounds X and Z — the second number is not 'y'."""
    e = flat_extent(_rect(), "XZ")
    assert e == {0: (100.0, 150.0), 2: (200.0, 280.0)}


def test_extent_follows_the_plane_the_roi_was_drawn_in():
    assert flat_extent(_rect(), "XY") == {0: (100.0, 150.0), 1: (200.0, 280.0)}
    assert flat_extent(_rect(), "YZ") == {1: (100.0, 150.0), 2: (200.0, 280.0)}


def test_a_rotated_rectangles_extent_covers_its_real_corners():
    rec = Rec("rectangle", {"bounds": [0.0, 0.0, 100.0, 100.0], "angle": 45.0})
    e = flat_extent(rec, "XY")
    # A square turned 45° is wider than its unrotated bounds on both axes.
    assert e[0][1] - e[0][0] == pytest.approx(100.0 * 2 ** 0.5, rel=1e-6)


def test_a_vertex_list_and_a_point_both_have_an_extent():
    poly = Rec("polygon", {"points": [[0.0, 0.0], [10.0, 4.0], [3.0, 9.0]]})
    assert flat_extent(poly, "XY") == {0: (0.0, 10.0), 1: (0.0, 9.0)}
    pt = Rec("point", {"point": [7.0, 8.0, 9.0]})
    assert flat_extent(pt, "XY") == {0: (7.0, 7.0), 1: (8.0, 8.0)}


def test_no_geometry_yields_no_extent():
    assert flat_extent(Rec("polygon", {"points": []}), "XY") is None
    assert flat_extent(_rect(), "not-a-plane") is None


# ----------------------------------------------------------------- projection
def test_its_own_plane_returns_the_full_outline():
    kind, outline = project_flat_record(_rect(), 0, 1, origin_plane="XY")
    assert kind is FULL
    assert len(outline) == 4
    assert min(p[0] for p in outline) == 100.0
    assert max(p[1] for p in outline) == 280.0


def test_the_outline_follows_the_PANE_axis_order_not_the_records():
    """An ortho YZ pane draws Z horizontally while a standalone YZ draws Y —
    asking by axis column is what makes one stored geometry right in both."""
    rec = Rec("rectangle", {"bounds": [10.0, 300.0, 5.0, 40.0]})   # Y 10..15, Z 300..340
    _k, standalone = project_flat_record(rec, 1, 2, origin_plane="YZ")   # (Y, Z)
    _k, ortho = project_flat_record(rec, 2, 1, origin_plane="YZ")        # (Z, Y)
    assert max(p[0] for p in standalone) == 15.0 and max(p[1] for p in standalone) == 340.0
    assert max(p[0] for p in ortho) == 340.0 and max(p[1] for p in ortho) == 15.0


def test_seen_edge_on_a_flat_roi_is_a_segment_at_its_depth():
    """A rectangle drawn in XY has no Z extent, so in XZ it is a line — the
    honest picture, and what distinguishes it from a cuboid."""
    kind, outline = project_flat_record(
        _rect(), 0, 2, origin_plane="XY", depth_value=42.0)
    assert kind is DEGENERATE_LINE
    assert outline == [[100.0, 42.0], [150.0, 42.0]]     # X extent, flat in Z


def test_the_other_side_view_uses_the_other_in_plane_axis():
    kind, outline = project_flat_record(
        _rect(), 1, 2, origin_plane="XY", depth_value=42.0)
    assert kind is DEGENERATE_LINE
    assert outline == [[200.0, 42.0], [280.0, 42.0]]     # Y extent, flat in Z


def test_a_roi_drawn_in_xz_is_edge_on_in_xy():
    """The rule is plane-agnostic: the degenerate axis is whichever one the
    record has no extent on."""
    kind, outline = project_flat_record(
        _rect(), 0, 1, origin_plane="XZ", depth_value=-7.0)
    assert kind is DEGENERATE_LINE
    assert outline == [[100.0, -7.0], [150.0, -7.0]]     # X extent, flat in Y


def test_without_a_recorded_depth_nothing_is_drawn_rather_than_a_guess():
    """An invented depth would put the ROI where nothing chose to put it."""
    assert project_flat_record(_rect(), 0, 2, origin_plane="XY") is None


def test_the_kind_is_the_point_of_the_return_value():
    """A caller that ignores it draws a flat ROI exactly like a volume one, and
    the user then believes a flat ROI constrains Z."""
    same, _o = project_flat_record(_rect(), 0, 1, origin_plane="XY")
    edge, _o = project_flat_record(_rect(), 0, 2, origin_plane="XY", depth_value=0.0)
    assert same is FULL and edge is DEGENERATE_LINE
    assert same != edge


def test_plane_columns_match_the_drawing_convention():
    """These are the columns a drawn record's geometry is stored in
    (roi_overlay._PLANE_PLOT_AXES); they must not drift apart."""
    from minflux_viewer.ui.roi_overlay import _PLANE_PLOT_AXES

    assert PLANE_COLUMNS == _PLANE_PLOT_AXES
