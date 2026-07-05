"""Add / delete a vertex on polyline-family ROIs (pure geometry + projection)."""

import pytest

from minflux_viewer.core.roi_vertex_edit import (
    delete_vertex,
    insert_vertex,
    min_vertices,
    nearest_edge,
    nearest_vertex,
)


def test_min_vertices():
    assert min_vertices(closed=True) == 3    # a closed region needs a triangle
    assert min_vertices(closed=False) == 2   # an open line needs a segment


def test_nearest_vertex_picks_closest():
    pts = [[0, 0], [10, 0], [10, 10], [0, 10]]
    idx, dist = nearest_vertex(pts, (9.5, 0.4))
    assert idx == 1
    assert dist == pytest.approx((0.5**2 + 0.4**2) ** 0.5)


def test_nearest_vertex_empty():
    idx, dist = nearest_vertex([], (0, 0))
    assert idx == -1 and dist == float("inf")


def test_nearest_edge_open_line_midpoint():
    pts = [[0, 0], [100, 0], [100, 100]]
    seg, t = nearest_edge(pts, (50, 3), closed=False)
    assert seg == 0 and t == pytest.approx(0.5)


def test_nearest_edge_closed_uses_closing_edge():
    # square; the point sits near the closing edge (last -> first), the left side.
    pts = [[0, 0], [10, 0], [10, 10], [0, 10]]
    seg, t = nearest_edge(pts, (-0.5, 5), closed=True)
    assert seg == 3               # closing edge index n-1 (vertex 3 -> vertex 0)
    assert t == pytest.approx(0.5)


def test_nearest_edge_open_ignores_closing_edge():
    # same square but OPEN: the closing edge is not considered, so the point near
    # where the closing edge would be falls to the nearest real edge instead.
    pts = [[0, 0], [10, 0], [10, 10], [0, 10]]
    seg_closed, _ = nearest_edge(pts, (-0.5, 5), closed=True)
    seg_open, _ = nearest_edge(pts, (-0.5, 5), closed=False)
    assert seg_closed == 3
    assert seg_open != 3          # closing edge excluded


def test_insert_vertex_2d_midpoint():
    pts = [[0, 0], [10, 0]]
    out = insert_vertex(pts, 0, 0.5)
    assert out == [[0, 0], [5, 0], [10, 0]]


def test_insert_vertex_closing_edge_appends():
    pts = [[0, 0], [10, 0], [10, 10]]   # closed triangle
    # closing edge is seg index 2 (vertex 2 -> vertex 0)
    out = insert_vertex(pts, 2, 0.5)
    assert len(out) == 4
    assert out[-1] == [5.0, 5.0]        # midpoint of (10,10)->(0,0), appended at end


def test_insert_vertex_3d_interpolates_depth():
    pts = [[0, 0, 10], [100, 0, 20]]    # 3-D polyline, z 10 -> 20
    out = insert_vertex(pts, 0, 0.5)
    assert out[1] == pytest.approx([50.0, 0.0, 15.0])


def test_delete_vertex():
    pts = [[0, 0], [5, 0], [10, 0]]
    assert delete_vertex(pts, 1) == [[0, 0], [10, 0]]
    assert delete_vertex(pts, 99) == [[0, 0], [5, 0], [10, 0]]   # out of range → no-op


# --- integration: mirror exactly what the controller composes (project → edit) ---

def test_add_vertex_on_3d_polyline_via_projection_interpolates_depth():
    """The controller projects stored (3-D) vertices to the view plane, finds the
    nearest edge there, then interpolates the STORED points at the same t — so the
    new vertex lands on the projected edge and its depth is interpolated."""
    from minflux_viewer.ui.roi_overlay import project_points

    stored = [[0, 0, 10], [100, 0, 20]]          # z 10 -> 20
    projected = project_points(stored, "XY")     # drops z -> [[0,0],[100,0]]
    seg, t = nearest_edge(projected, (50, 5), closed=False)
    new = insert_vertex(stored, seg, t)
    assert len(new) == 3
    assert new[1] == pytest.approx([50.0, 0.0, 15.0])


def test_delete_vertex_via_projection_targets_nearest_projected_vertex():
    from minflux_viewer.ui.roi_overlay import project_points

    stored = [[0, 0, 10], [50, 0, 12], [100, 0, 20]]
    projected = project_points(stored, "XY")
    idx, _ = nearest_vertex(projected, (51, 1))
    assert idx == 1
    assert delete_vertex(stored, idx) == [[0, 0, 10], [100, 0, 20]]
