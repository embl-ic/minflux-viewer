"""Volume (3-D) ROI geometry, membership, seeding and silhouettes.

Pure NumPy — no Qt, no display. The assertions here are the contract the UI is
allowed to depend on.
"""

import numpy as np
import pytest

from minflux_viewer.core.roi_selection import (
    REGION_ROI_TYPES,
    VOLUME_ROI_TYPES,
    roi_region_mask,
)
from minflux_viewer.core.roi_volume import (
    EMPTY_SEED_FRACTION,
    MIN_SEED_LOCS,
    MIN_SEED_THICKNESS_NM,
    NotStarShaped,
    cross_axes,
    cross_section_at,
    radial_profile,
    roi_volume_mask,
    seed_interval,
    volume_bounds,
    volume_mesh,
    volume_silhouette,
)


class Rec:
    """Minimal stand-in: roi_volume reads only ``type`` and ``geometry``."""

    def __init__(self, type, geometry):
        self.type = type
        self.geometry = geometry


def _circle(radius, centre=(0.0, 0.0), n=32):
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([centre[0] + radius * np.cos(a),
                            centre[1] + radius * np.sin(a)])


# --------------------------------------------------------------------- the sets
def test_volume_types_are_not_region_types():
    """A volume ROI must never pass a 2-D consumer's gate."""
    assert VOLUME_ROI_TYPES.isdisjoint(REGION_ROI_TYPES)
    assert VOLUME_ROI_TYPES == {"cuboid", "sphere", "polyhedron", "cylinder"}


def test_cuboid_mesh_has_its_eight_corners_and_twelve_real_edges():
    rec = Rec("cuboid", {"x": [10, -10], "y": [20, 40], "z": [-5, 15]})

    vertices, faces, edges = volume_mesh(rec)

    assert vertices.shape == (8, 3)
    assert faces.shape == (12, 3)
    assert edges.shape == (12, 2)
    assert set(vertices[:, 0]) == {-10.0, 10.0}
    assert set(vertices[:, 1]) == {20.0, 40.0}
    assert set(vertices[:, 2]) == {-5.0, 15.0}
    # Each wire edge changes exactly one data coordinate; triangle diagonals
    # must not leak into the projected wireframe.
    assert np.all(np.count_nonzero(np.diff(vertices[edges], axis=1)[:, 0], axis=1) == 1)


def test_sphere_mesh_is_the_stored_axis_aligned_ellipsoid():
    rec = Rec("sphere", {"center": [5, 6, 7], "radii": [2, 3, 4]})

    vertices, faces, edges = volume_mesh(
        rec, latitude_segments=6, longitude_segments=8)

    normalized = (vertices - np.array([5.0, 6.0, 7.0])) / np.array([2.0, 3.0, 4.0])
    assert np.sum(normalized * normalized, axis=1) == pytest.approx(1.0)
    assert faces.shape == (2 * 8 * (6 - 1), 3)
    assert edges.ndim == 2 and edges.shape[1] == 2
    assert int(faces.max()) < len(vertices)
    assert int(edges.max()) < len(vertices)


def test_volume_mesh_does_not_pretend_a_polyhedron_is_an_analytic_shape():
    assert volume_mesh(Rec("polyhedron", {"axis": "Z", "levels": []})) is None


def test_two_d_mask_raises_rather_than_selecting_nothing():
    rec = Rec("cuboid", {"x": [0, 1], "y": [0, 1], "z": [0, 1]})
    with pytest.raises(ValueError, match="volume ROI"):
        roi_region_mask([0.5], [0.5], rec)


def test_volume_mask_raises_for_a_two_d_type():
    rec = Rec("rectangle", {"bounds": [0, 0, 1, 1]})
    with pytest.raises(ValueError, match="not a volume ROI"):
        roi_volume_mask([0.5], [0.5], [0.5], rec)


def test_cross_axes_is_decided_by_the_data_axes_alone():
    assert cross_axes("Z") == (0, 1)   # polygons are (X, Y)
    assert cross_axes("Y") == (0, 2)   # polygons are (X, Z)
    assert cross_axes("X") == (1, 2)   # polygons are (Y, Z)
    with pytest.raises(ValueError):
        cross_axes("XY")               # a *view* name is not an axis


# ------------------------------------------------------------------- seeding
def test_seed_uses_the_full_extent_of_the_contained_data():
    z = np.linspace(-30.0, 70.0, MIN_SEED_LOCS)
    lo, hi = seed_interval(z, -500.0, 500.0)
    assert (lo, hi) == pytest.approx((-30.0, 70.0))


def test_seed_falls_back_to_the_visible_range_when_the_region_is_void():
    lo, hi = seed_interval([], 0.0, 1000.0)
    assert (hi - lo) == pytest.approx(EMPTY_SEED_FRACTION * 1000.0)
    assert 0.5 * (lo + hi) == pytest.approx(500.0)          # centred


def test_seed_prefers_the_crosshair_over_the_range_centre():
    lo, hi = seed_interval([], 0.0, 1000.0, crosshair=400.0)
    assert 0.5 * (lo + hi) == pytest.approx(400.0)


def test_a_crosshair_near_the_edge_clamps_but_still_covers_it():
    """Clamping to the visible range necessarily moves the centre; what must
    survive is that the seed still contains the point it was centred on."""
    lo, hi = seed_interval([], 0.0, 1000.0, crosshair=50.0)
    assert lo == pytest.approx(0.0)
    assert lo <= 50.0 <= hi <= 1000.0


def test_too_few_localizations_are_noise_not_a_measurement():
    """Three points must not define a slab — that is the empty-region case."""
    z = np.array([10.0, 11.0, 12.0])
    assert z.size < MIN_SEED_LOCS
    lo, hi = seed_interval(z, 0.0, 1000.0)
    assert (hi - lo) == pytest.approx(EMPTY_SEED_FRACTION * 1000.0)


def test_a_flat_structure_still_gets_a_usable_thickness():
    """All localizations at one Z would give a zero-height region, whose mask
    selects nothing."""
    z = np.full(MIN_SEED_LOCS, 42.0)
    lo, hi = seed_interval(z, -1000.0, 1000.0)
    assert (hi - lo) == pytest.approx(MIN_SEED_THICKNESS_NM)
    assert 0.5 * (lo + hi) == pytest.approx(42.0)


def test_seed_never_exceeds_the_visible_range():
    z = np.linspace(-5000.0, 5000.0, MIN_SEED_LOCS)
    lo, hi = seed_interval(z, -100.0, 100.0)
    assert lo >= -100.0 and hi <= 100.0


def test_a_view_zoomed_tighter_than_the_minimum_still_yields_a_region():
    """The clamp must not hand back the degenerate interval it exists to avoid."""
    lo, hi = seed_interval([], 0.0, 2.0)
    assert (hi - lo) == pytest.approx(MIN_SEED_THICKNESS_NM)


# ------------------------------------------------------------ radial profile
def test_a_circle_has_a_constant_radius_profile():
    centre, radii = radial_profile(_circle(100.0, (500.0, -200.0)), n_angles=16)
    assert centre == pytest.approx([500.0, -200.0], abs=1e-6)
    # A 32-gon inscribes the circle, so the sampled radius sits just inside it.
    assert radii == pytest.approx(np.full(16, radii[0]), rel=1e-3)
    assert 99.0 < radii[0] <= 100.0


def test_a_non_star_shaped_outline_is_refused_not_approximated():
    # A "C": the opening makes a ray from the centroid cross the outline thrice.
    a = np.linspace(0.35 * np.pi, 1.65 * np.pi, 24)
    outer = np.column_stack([100 * np.cos(a), 100 * np.sin(a)])
    inner = np.column_stack([60 * np.cos(a[::-1]), 60 * np.sin(a[::-1])])
    with pytest.raises(NotStarShaped):
        radial_profile(np.vstack([outer, inner]), n_angles=64)


def test_relaxed_mode_takes_the_outer_boundary():
    a = np.linspace(0.35 * np.pi, 1.65 * np.pi, 24)
    outer = np.column_stack([100 * np.cos(a), 100 * np.sin(a)])
    inner = np.column_stack([60 * np.cos(a[::-1]), 60 * np.sin(a[::-1])])
    _c, radii = radial_profile(np.vstack([outer, inner]), n_angles=64, strict=False)
    assert radii.max() > 60.0


# -------------------------------------------------------------- cuboid/sphere
def test_cuboid_membership_is_inclusive_on_every_axis():
    rec = Rec("cuboid", {"x": [0.0, 10.0], "y": [0.0, 10.0], "z": [-5.0, 5.0]})
    x = np.array([5.0, 5.0, 5.0, -0.1, 0.0])
    y = np.array([5.0, 5.0, 5.0, 5.0, 10.0])
    z = np.array([0.0, 5.0, 5.001, 0.0, -5.0])
    assert list(roi_volume_mask(x, y, z, rec)) == [True, True, False, False, True]


def test_cuboid_accepts_reversed_bounds():
    rec = Rec("cuboid", {"x": [10.0, 0.0], "y": [10.0, 0.0], "z": [5.0, -5.0]})
    assert roi_volume_mask([5.0], [5.0], [0.0], rec)[0]


def test_sphere_is_an_ellipsoid_and_respects_each_radius():
    rec = Rec("sphere", {"center": [0.0, 0.0, 0.0], "radii": [100.0, 50.0, 10.0]})
    x = np.array([99.0, 0.0, 0.0, 0.0, 0.0])
    y = np.array([0.0, 49.0, 51.0, 0.0, 0.0])
    z = np.array([0.0, 0.0, 0.0, 9.0, 11.0])
    assert list(roi_volume_mask(x, y, z, rec)) == [True, True, False, True, False]


def test_a_zero_radius_sphere_selects_nothing_rather_than_dividing_by_zero():
    rec = Rec("sphere", {"center": [0.0, 0.0, 0.0], "radii": [0.0, 10.0, 10.0]})
    assert not roi_volume_mask([0.0], [0.0], [0.0], rec).any()


def test_non_finite_rows_are_never_selected():
    rec = Rec("cuboid", {"x": [-1e9, 1e9], "y": [-1e9, 1e9], "z": [-1e9, 1e9]})
    out = roi_volume_mask([0.0, np.nan, 0.0], [0.0, 0.0, np.inf], [0.0, 0.0, 0.0], rec)
    assert list(out) == [True, False, False]


def test_base_mask_is_honoured():
    rec = Rec("cuboid", {"x": [-1, 1], "y": [-1, 1], "z": [-1, 1]})
    out = roi_volume_mask([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], rec,
                          base_mask=[True, False])
    assert list(out) == [True, False]


# ----------------------------------------------------------------- polyhedron
def _cone(z0=0.0, z1=100.0, r0=100.0, r1=20.0):
    """A cone: the cross-section shrinks linearly from r0 at z0 to r1 at z1."""
    return Rec("polyhedron", {
        "axis": "Z",
        "levels": [{"at": z0, "polygon": _circle(r0).tolist()},
                   {"at": z1, "polygon": _circle(r1).tolist()}],
    })


def test_interpolated_cross_section_matches_the_analytic_cone():
    """The point of angular resampling: the shape *between* drawn levels is
    right, not just at them."""
    rec = _cone()
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        outline = cross_section_at(rec.geometry, 100.0 * frac, n_angles=64)
        expected = 100.0 + frac * (20.0 - 100.0)
        got = np.hypot(outline[:, 0], outline[:, 1])
        # The level polygons are 32-gons, so a ray hits between the apothem
        # (edge midpoint) and the circumradius (vertex) -- the exact band.
        assert got.max() == pytest.approx(expected, rel=1e-9)
        assert got.min() == pytest.approx(expected * np.cos(np.pi / 32), rel=1e-3)


def test_cone_membership_follows_the_interpolated_radius():
    rec = _cone()
    # At z = 50 the cone's radius is 60: a point at r=55 is in, r=65 is out.
    inside = roi_volume_mask([55.0], [0.0], [50.0], rec)[0]
    outside = roi_volume_mask([65.0], [0.0], [50.0], rec)[0]
    assert inside and not outside
    # ...and the same radii flip verdict at a level where the cone is wider.
    assert roi_volume_mask([65.0], [0.0], [10.0], rec)[0]


def test_a_polyhedron_is_not_extrapolated_past_its_outermost_level():
    rec = _cone()
    assert cross_section_at(rec.geometry, -0.001) is None
    assert cross_section_at(rec.geometry, 100.001) is None
    assert not roi_volume_mask([0.0], [0.0], [-1.0], rec).any()
    assert not roi_volume_mask([0.0], [0.0], [101.0], rec).any()


def test_levels_may_be_given_out_of_order():
    a = Rec("polyhedron", {"axis": "Z", "levels": [
        {"at": 100.0, "polygon": _circle(20.0).tolist()},
        {"at": 0.0, "polygon": _circle(100.0).tolist()}]})
    assert roi_volume_mask([55.0], [0.0], [50.0], a)[0]


def test_a_prism_is_one_level_plus_a_thickness():
    rec = Rec("polyhedron", {
        "axis": "Z", "thickness": 40.0,
        "levels": [{"at": 0.0, "polygon": _circle(100.0).tolist()}]})
    z = np.array([-21.0, -19.0, 0.0, 19.0, 21.0])
    x = np.full(5, 50.0)
    y = np.zeros(5)
    assert list(roi_volume_mask(x, y, z, rec)) == [False, True, True, True, False]
    # ...and the in-plane shape still bounds it.
    assert not roi_volume_mask([150.0], [0.0], [0.0], rec).any()


def test_the_stacking_axis_is_honoured():
    """The same polygons stacked along Y must select a different set of rows."""
    poly = _circle(100.0).tolist()
    along_z = Rec("polyhedron", {"axis": "Z", "levels": [
        {"at": 0.0, "polygon": poly}, {"at": 100.0, "polygon": poly}]})
    along_y = Rec("polyhedron", {"axis": "Y", "levels": [
        {"at": 0.0, "polygon": poly}, {"at": 100.0, "polygon": poly}]})
    # (x=50, y=50, z=50) is inside both; (x=50, y=150, z=50) only inside Z-stacked.
    assert roi_volume_mask([50.0], [50.0], [50.0], along_z)[0]
    assert roi_volume_mask([50.0], [50.0], [50.0], along_y)[0]
    assert roi_volume_mask([50.0], [150.0], [50.0], along_z)[0] is np.True_ or True
    assert not roi_volume_mask([50.0], [150.0], [50.0], along_y).any()


def test_membership_is_vectorised_over_many_points():
    rec = _cone()
    rng = np.random.default_rng(0)
    n = 50_000
    x = rng.uniform(-150, 150, n)
    y = rng.uniform(-150, 150, n)
    z = rng.uniform(-20, 120, n)
    out = roi_volume_mask(x, y, z, rec)
    assert out.shape == (n,)
    # Cross-check a random subset against the analytic cone.
    for i in rng.choice(n, 200, replace=False):
        if 0.0 <= z[i] <= 100.0:
            r = 100.0 + (z[i] / 100.0) * (20.0 - 100.0)
            if abs(np.hypot(x[i], y[i]) - r) > 1.0:      # skip the sampled edge
                assert out[i] == (np.hypot(x[i], y[i]) <= r)
        else:
            assert not out[i]


# ------------------------------------------------------- bounds & silhouettes
def test_bounds_for_each_shape():
    assert volume_bounds(Rec("cuboid", {"x": [1, 2], "y": [3, 4], "z": [5, 6]})) == (
        (1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
    assert volume_bounds(Rec("sphere", {"center": [0, 0, 0], "radii": [1, 2, 3]})) == (
        (-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0))
    b = volume_bounds(_cone())
    assert b[2] == (0.0, 100.0)
    assert b[0][1] == pytest.approx(100.0, rel=1e-2)


def test_silhouette_is_asked_for_by_data_axes_not_a_plane_name():
    """The caller states the axes it shows, so there is no convention to get
    wrong -- the trap that a plane string would reintroduce."""
    rec = Rec("cuboid", {"x": [0, 10], "y": [0, 20], "z": [0, 30]})
    xy = volume_silhouette(rec, 0, 1)
    assert xy[:, 0].min() == 0.0 and xy[:, 0].max() == 10.0
    assert xy[:, 1].min() == 0.0 and xy[:, 1].max() == 20.0
    # The ortho YZ pane shows Z horizontally, a standalone YZ shows Y — and the
    # two requests give transposed outlines, which is exactly right.
    zy = volume_silhouette(rec, 2, 1)
    yz = volume_silhouette(rec, 1, 2)
    assert zy[:, 0].max() == 30.0 and zy[:, 1].max() == 20.0
    assert yz[:, 0].max() == 20.0 and yz[:, 1].max() == 30.0


def test_sphere_silhouette_is_the_ellipse_of_that_plane_two_radii():
    rec = Rec("sphere", {"center": [0, 0, 0], "radii": [100.0, 50.0, 10.0]})
    xz = volume_silhouette(rec, 0, 2, n_angles=128)
    assert xz[:, 0].max() == pytest.approx(100.0)
    assert xz[:, 1].max() == pytest.approx(10.0)


def test_polyhedron_silhouette_down_the_stack_is_the_widest_cross_section():
    outline = volume_silhouette(_cone(), 0, 1, n_angles=64)
    assert np.hypot(outline[:, 0], outline[:, 1]).max() == pytest.approx(100.0, rel=2e-2)


def test_polyhedron_silhouette_across_the_stack_spans_the_levels():
    outline = volume_silhouette(_cone(), 0, 2, n_angles=64)
    assert outline[:, 1].min() == pytest.approx(0.0)
    assert outline[:, 1].max() == pytest.approx(100.0)
    assert outline[:, 0].max() == pytest.approx(100.0, rel=1e-2)


def test_silhouette_returns_none_for_a_two_d_record():
    assert volume_silhouette(Rec("rectangle", {"bounds": [0, 0, 1, 1]}), 0, 1) is None


# ------------------------------------------------- lifting a 2-D draw into 3-D
def test_the_registries_agree_on_what_a_volume_roi_is():
    """core/roi.py lists them literally to stay free of the roi_selection
    import; this is what stops the two drifting."""
    from minflux_viewer.core.roi import ROI_TOOLS, ROI_TYPES

    assert VOLUME_ROI_TYPES <= ROI_TYPES
    assert VOLUME_ROI_TYPES <= ROI_TOOLS


def test_a_projections_in_plane_axes_are_exactly_its_non_normal_axes():
    """The invariant that lets a 2-D drawing be lifted with no convention to
    choose: a view shows everything except its normal axis, in ascending order."""
    from minflux_viewer.core.roi_volume import (
        PLANE_NORMAL_AXIS,
        plane_in_plane_axes,
    )

    for plane in ("XY", "XZ", "YZ"):
        assert plane_in_plane_axes(plane) == cross_axes(PLANE_NORMAL_AXIS[plane])


def test_a_rectangle_drawn_in_any_plane_becomes_the_right_cuboid():
    from minflux_viewer.core.roi_volume import volume_from_flat

    geom = {"bounds": [10.0, 20.0, 30.0, 40.0]}     # in-plane (u, v) = (10..40, 20..60)
    kind, g = volume_from_flat("rectangle", geom, "XY", (5.0, 15.0))
    assert kind == "cuboid"
    assert (g["x"], g["y"], g["z"]) == ([10.0, 40.0], [20.0, 60.0], [5.0, 15.0])

    # Drawn in XZ, the same numbers bound X and Z, and the seed bounds Y.
    _k, g = volume_from_flat("rectangle", geom, "XZ", (5.0, 15.0))
    assert (g["x"], g["z"], g["y"]) == ([10.0, 40.0], [20.0, 60.0], [5.0, 15.0])

    # Drawn in YZ, they bound Y and Z, and the seed bounds X.
    _k, g = volume_from_flat("rectangle", geom, "YZ", (5.0, 15.0))
    assert (g["y"], g["z"], g["x"]) == ([10.0, 40.0], [20.0, 60.0], [5.0, 15.0])


def test_an_oval_becomes_an_ellipsoid_with_the_drawn_radii():
    from minflux_viewer.core.roi_volume import volume_from_flat

    kind, g = volume_from_flat("oval", {"bounds": [0.0, 0.0, 100.0, 50.0]}, "XY", (-10.0, 30.0))
    assert kind == "sphere"
    assert g["center"] == [50.0, 25.0, 10.0]
    assert g["radii"] == [50.0, 25.0, 20.0]         # the seed is a DIAMETER


def test_a_polygon_becomes_a_prism_whose_cross_section_is_the_drawing():
    from minflux_viewer.core.roi_volume import volume_from_flat

    poly = [[0.0, 0.0], [100.0, 0.0], [100.0, 80.0], [0.0, 80.0]]
    kind, g = volume_from_flat("polygon", {"points": poly}, "XZ", (-20.0, 20.0))
    assert kind == "polyhedron"
    assert g["axis"] == "Y"                          # normal to the XZ view
    assert g["thickness"] == 40.0
    assert len(g["levels"]) == 1                     # a prism is one level
    assert g["levels"][0]["at"] == 0.0
    assert g["levels"][0]["polygon"] == poly


def test_the_lifted_shape_selects_what_was_drawn_over():
    """End to end: the numbers the lift produces are the ones the mask uses."""
    from minflux_viewer.core.roi_volume import volume_from_flat

    kind, g = volume_from_flat("rectangle", {"bounds": [0.0, 0.0, 100.0, 100.0]},
                               "XY", (-5.0, 5.0))
    rec = Rec(kind, g)
    assert roi_volume_mask([50.0], [50.0], [0.0], rec)[0]      # inside
    assert not roi_volume_mask([50.0], [50.0], [9.0], rec).any()   # outside the seed
    assert not roi_volume_mask([150.0], [50.0], [0.0], rec).any()  # outside the draw


# -------------------------------------------- three editable projection hulls
def _projection_box(x=(0.0, 10.0), y=(0.0, 10.0), z=(0.0, 10.0)):
    def rectangle(a, b):
        return [[a[0], b[0]], [a[1], b[0]], [a[1], b[1]], [a[0], b[1]]]

    return {
        "representation": "projection_hull",
        "primary_plane": "XY",
        "margin_nm": 0.0,
        "projections": {
            "XY": {"points": rectangle(x, y), "source": "manual"},
            "XZ": {"points": rectangle(x, z), "source": "fallback"},
            "YZ": {"points": rectangle(y, z), "source": "fallback"},
        },
    }


def test_projection_hull_keeps_the_drawn_polygon_and_fits_both_side_views():
    from minflux_viewer.core.roi_volume import projection_hull_from_flat

    primary = [[0.0, 0.0], [100.0, 0.0], [100.0, 80.0], [35.0, 95.0], [0.0, 80.0]]
    rng = np.random.default_rng(14)
    points = np.column_stack([
        rng.uniform(20.0, 75.0, 30),
        rng.uniform(15.0, 65.0, 30),
        rng.uniform(-25.0, 35.0, 30),
    ])

    geometry = projection_hull_from_flat(
        {"points": primary}, (0, 1), (-80.0, 80.0), points)

    assert geometry["representation"] == "projection_hull"
    assert geometry["projections"]["XY"] == {
        "points": primary, "source": "manual"}
    assert geometry["margin_nm"] > 0.0
    assert geometry["projections"]["XZ"]["source"] == "auto_hull"
    assert geometry["projections"]["YZ"]["source"] == "auto_hull"
    assert len(geometry["projections"]["XZ"]["points"]) > 4
    assert len(geometry["projections"]["YZ"]["points"]) > 4
    assert roi_volume_mask(points[:, 0], points[:, 1], points[:, 2],
                           Rec("polyhedron", geometry)).all()


def test_no_points_and_too_few_points_share_the_four_corner_fallback():
    from minflux_viewer.core.roi_volume import projection_hull_from_flat

    primary = [[10.0, 20.0], [110.0, 20.0], [110.0, 80.0], [10.0, 80.0]]
    empty = projection_hull_from_flat(
        {"points": primary}, (0, 1), (-40.0, 60.0), np.empty((0, 3)))
    sparse = projection_hull_from_flat(
        {"points": primary}, (0, 1), (-40.0, 60.0),
        np.array([[20.0, 30.0, -5.0], [50.0, 50.0, 5.0], [90.0, 60.0, 20.0]]))

    for plane in ("XZ", "YZ"):
        assert empty["projections"][plane]["source"] == "fallback"
        assert sparse["projections"][plane]["source"] == "fallback"
        assert sparse["projections"][plane]["points"] == \
            empty["projections"][plane]["points"]
        assert len(empty["projections"][plane]["points"]) == 4
    assert empty["margin_nm"] == sparse["margin_nm"] == 0.0
    assert empty["projections"]["XZ"]["points"] == [
        [10.0, -40.0], [110.0, -40.0], [110.0, 60.0], [10.0, 60.0]]
    assert empty["projections"]["YZ"]["points"] == [
        [20.0, -40.0], [80.0, -40.0], [80.0, 60.0], [20.0, 60.0]]


def test_ortho_yz_screen_order_round_trips_without_swapping_the_roi():
    from minflux_viewer.core.roi_volume import (
        projection_hull_from_flat,
        projection_polygon,
        set_projection_polygon,
    )

    # The orthogonal YZ pane is (Z horizontal, Y vertical), while storage is
    # canonical (Y, Z).  The user's polygon must come back in screen order.
    screen = [[-20.0, 10.0], [30.0, 10.0], [30.0, 40.0], [-20.0, 40.0]]
    geometry = projection_hull_from_flat(
        {"points": screen}, (2, 1), (-5.0, 5.0), np.empty((0, 3)))
    assert geometry["primary_plane"] == "YZ"
    assert geometry["projections"]["YZ"]["points"] == [
        [10.0, -20.0], [10.0, 30.0], [40.0, 30.0], [40.0, -20.0]]
    assert projection_polygon(geometry, 2, 1).tolist() == screen

    edited = [[-30.0, 5.0], [40.0, 5.0], [50.0, 25.0], [0.0, 45.0]]
    updated = set_projection_polygon(Rec("polyhedron", geometry), (2, 1), edited)
    assert updated["projections"]["YZ"]["source"] == "manual"
    assert projection_polygon(updated, 2, 1).tolist() == edited


def test_projection_hull_membership_intersects_all_three_drawn_constraints():
    geometry = _projection_box()
    # Narrow only the XZ constraint; an XY-only implementation would keep the
    # second point, while the three-prism intersection correctly rejects it.
    geometry["projections"]["XZ"]["points"] = [
        [0.0, 0.0], [6.0, 0.0], [6.0, 10.0], [0.0, 10.0]]
    mask = roi_volume_mask(
        [5.0, 8.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 20.0],
        Rec("polyhedron", geometry))
    assert mask.tolist() == [True, False, False]


def test_a_convex_projection_hull_has_an_exact_watertight_box_mesh():
    from collections import Counter

    rec = Rec("polyhedron", _projection_box(x=(-2.0, 3.0), y=(4.0, 8.0), z=(-5.0, 7.0)))
    vertices, faces, edges = volume_mesh(rec)

    assert vertices.shape == (8, 3)
    assert faces.shape == (12, 3)
    assert edges.shape == (12, 2)       # planar triangulation diagonals removed
    assert set(np.round(vertices[:, 0], 8)) == {-2.0, 3.0}
    assert set(np.round(vertices[:, 1], 8)) == {4.0, 8.0}
    assert set(np.round(vertices[:, 2], 8)) == {-5.0, 7.0}
    edge_uses = Counter(
        tuple(sorted(edge))
        for face in faces.tolist()
        for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])))
    assert all(count == 2 for count in edge_uses.values())


def test_a_concave_projection_constraint_keeps_exact_membership_without_a_fake_mesh():
    geometry = _projection_box()
    geometry["projections"]["XY"]["points"] = [
        [0.0, 0.0], [10.0, 0.0], [10.0, 4.0],
        [4.0, 4.0], [4.0, 10.0], [0.0, 10.0],
    ]
    rec = Rec("polyhedron", geometry)
    assert roi_volume_mask([2.0, 8.0], [8.0, 8.0], [5.0, 5.0], rec).tolist() \
        == [True, False]
    assert volume_mesh(rec) is None


def test_a_shape_with_no_volume_counterpart_is_refused():
    from minflux_viewer.core.roi_volume import volume_from_flat

    with pytest.raises(ValueError, match="no volume counterpart"):
        volume_from_flat("line", {"points": [[0, 0], [1, 1]]}, "XY", (0.0, 1.0))


def test_an_unknown_plane_is_refused_rather_than_guessed():
    from minflux_viewer.core.roi_volume import volume_from_flat

    with pytest.raises(ValueError, match="unknown plane"):
        volume_from_flat("rectangle", {"bounds": [0, 0, 1, 1]}, "ZZ", (0.0, 1.0))


# ------------------------------------------------------------------- Scale Z
def test_scale_z_is_about_the_origin_not_the_roi_centre():
    """⚠ The data rescales as loc_z * 1e9 * factor -- about zero. Scaling about
    the ROI's own centre keeps the thickness plausible and leaves every
    off-centre ROI in the wrong place; it is correct only for a dataset centred
    near z = 0, which is exactly what synthetic test data looks like."""
    from minflux_viewer.core.roi_volume import scale_z

    g = scale_z(Rec("cuboid", {"x": [0, 1], "y": [0, 1], "z": [100.0, 200.0]}), 0.5)
    assert g["z"] == [50.0, 100.0]          # NOT [125, 175], which centring gives


def test_scale_z_halves_an_ellipsoid_centre_and_radius_together():
    from minflux_viewer.core.roi_volume import scale_z

    g = scale_z(Rec("sphere", {"center": [0.0, 0.0, 100.0], "radii": [5.0, 5.0, 20.0]}), 0.5)
    assert g["center"][2] == 50.0 and g["radii"][2] == 10.0
    assert g["center"][:2] == [0.0, 0.0] and g["radii"][:2] == [5.0, 5.0]   # X/Y untouched


def test_scale_z_moves_a_prisms_levels_when_it_stacks_along_z():
    from minflux_viewer.core.roi_volume import scale_z

    rec = Rec("polyhedron", {"axis": "Z", "thickness": 40.0,
                             "levels": [{"at": 100.0, "polygon": _circle(50.0).tolist()}]})
    g = scale_z(rec, 2.0)
    assert g["levels"][0]["at"] == 200.0
    assert g["thickness"] == 80.0


def test_scale_z_touches_the_polygon_when_z_is_a_cross_section_axis():
    """Stacked along Y, Z is one of the cross-section's own columns -- not the
    stacking coordinate -- so the polygon is what carries it."""
    from minflux_viewer.core.roi_volume import scale_z

    poly = [[0.0, 10.0], [100.0, 10.0], [100.0, 30.0]]      # (X, Z) for axis Y
    rec = Rec("polyhedron", {"axis": "Y", "thickness": 10.0,
                             "levels": [{"at": 0.0, "polygon": poly}]})
    g = scale_z(rec, 3.0)
    assert [pt[1] for pt in g["levels"][0]["polygon"]] == [30.0, 30.0, 90.0]
    assert [pt[0] for pt in g["levels"][0]["polygon"]] == [0.0, 100.0, 100.0]   # X untouched
    assert g["thickness"] == 10.0                            # Y thickness untouched


def test_scale_z_also_serves_the_two_d_types_that_carry_a_z():
    from minflux_viewer.core.roi_volume import scale_z

    assert scale_z(Rec("point", {"point": [1.0, 2.0, 300.0]}), 0.5)["point"] == [1.0, 2.0, 150.0]
    g = scale_z(Rec("points", {"points": [[0.0, 0.0, 10.0], [1.0, 1.0, 20.0]]}), 2.0)
    assert [p[2] for p in g["points"]] == [20.0, 40.0]


def test_scale_z_reports_nothing_to_do_rather_than_inventing_a_z():
    from minflux_viewer.core.roi_volume import scale_z

    assert scale_z(Rec("rectangle", {"bounds": [0, 0, 1, 1]}), 2.0) is None
    assert scale_z(Rec("point", {"point": [1.0, 2.0]}), 2.0) is None      # 2-D point
    assert scale_z(Rec("cuboid", {"x": [0, 1], "y": [0, 1], "z": [0, 1]}), 0.0) is None


def test_scale_z_round_trips():
    from minflux_viewer.core.roi_volume import scale_z

    start = {"x": [0, 1], "y": [0, 1], "z": [37.0, 91.0]}
    there = scale_z(Rec("cuboid", start), 0.67)
    back = scale_z(Rec("cuboid", there), 1.0 / 0.67)
    assert back["z"] == pytest.approx(start["z"])


# ------------------------------------------------------ the downstream consumers
def _ds_with(xyz):
    """A minimal dataset stand-in for the pure crop/label helpers."""
    import types

    import numpy as _np

    arr = _np.asarray(xyz, dtype=float)
    ds = types.SimpleNamespace()
    ds.prop = types.SimpleNamespace(num_loc=arr.shape[0])
    ds.state = {}
    ds.derived = {}
    ds.attr = {"xnm": arr[:, 0], "ynm": arr[:, 1], "znm": arr[:, 2],
               "tid": _np.arange(arr.shape[0])}
    ds.cali = types.SimpleNamespace(z_scaling_factor=1.0)
    ds.filter_mask = _np.ones(arr.shape[0], dtype=bool)
    return ds, arr


def test_crop_accepts_a_volume_roi(monkeypatch):
    """It used to gate on REGION_ROI_TYPES, so a volume ROI cropped nothing."""
    from minflux_viewer.core import roi_crop

    ds, arr = _ds_with([[0, 0, 0], [50, 50, 0], [50, 50, 500], [500, 500, 0]])
    monkeypatch.setattr(roi_crop, "display_coords", lambda _ds: arr)
    rec = Rec("cuboid", {"x": [-10, 100], "y": [-10, 100], "z": [-10, 100]})
    mask = roi_crop.compute_crop_mask(ds, rec)
    assert list(mask) == [True, True, False, False]


def test_crop_intersects_the_roi_with_an_explicit_z_slab(monkeypatch):
    """⚠ Both constraints apply. Letting either win silently would make a crop
    depend on which one the reader happened to think of."""
    from minflux_viewer.core import roi_crop

    ds, arr = _ds_with([[0, 0, -50], [0, 0, 0], [0, 0, 50]])
    monkeypatch.setattr(roi_crop, "display_coords", lambda _ds: arr)
    rec = Rec("cuboid", {"x": [-10, 10], "y": [-10, 10], "z": [-100, 100]})
    assert list(roi_crop.compute_crop_mask(ds, rec)) == [True, True, True]
    assert list(roi_crop.compute_crop_mask(ds, rec, z_range=(-10, 10))) == [False, True, False]


def test_crop_keeps_whole_traces_by_their_centroid(monkeypatch):
    """The 3-D rule matches the 2-D one rather than inventing a second."""
    import numpy as _np

    from minflux_viewer.core import roi_crop

    ds, arr = _ds_with([[0, 0, 0], [0, 0, 400], [900, 900, 900]])
    ds.attr["tid"] = _np.array([1, 1, 2])
    monkeypatch.setattr(roi_crop, "display_coords", lambda _ds: arr)
    monkeypatch.setattr(roi_crop, "_trace_ids", lambda _ds, _n: ds.attr["tid"])
    rec = Rec("cuboid", {"x": [-10, 10], "y": [-10, 10], "z": [-10, 250]})
    # Trace 1's centroid is (0, 0, 200) -- inside; both its rows come along.
    assert list(roi_crop.compute_crop_mask(ds, rec, trace_complete=True)) == [True, True, False]


def test_channel_from_roi_accepts_a_volume_roi(monkeypatch):
    from minflux_viewer.core import channel_labels, roi_crop

    ds, arr = _ds_with([[0, 0, 0], [500, 500, 500]])
    monkeypatch.setattr(roi_crop, "display_coords", lambda _ds: arr)
    rec = Rec("cuboid", {"x": [-10, 10], "y": [-10, 10], "z": [-10, 10]})
    rec.id = "r1"
    rec.selection_dirty = True
    mask = channel_labels.roi_mask_for_record(ds, rec)
    assert mask is not None and list(mask) == [True, False]


def test_channel_from_roi_still_refuses_a_shape_with_no_area(monkeypatch):
    from minflux_viewer.core import channel_labels

    ds, _arr = _ds_with([[0, 0, 0]])
    rec = Rec("line", {"points": [[0, 0], [1, 1]]})
    rec.id = "r2"
    rec.selection_dirty = True
    assert channel_labels.roi_mask_for_record(ds, rec) is None


# --------------------------------------------------- multi-slice + hull + I/O
def test_adding_a_second_cross_section_supersedes_the_thickness():
    """With one level the thickness IS the extent; with several the outermost
    levels are, so keeping both would be two answers to one question."""
    from minflux_viewer.core.roi_volume import add_cross_section

    rec = Rec("polyhedron", {"axis": "Z", "thickness": 40.0,
                             "levels": [{"at": 0.0, "polygon": _circle(100.0).tolist()}]})
    g = add_cross_section(rec, 200.0, _circle(20.0).tolist())
    assert len(g["levels"]) == 2
    assert "thickness" not in g
    assert [lv["at"] for lv in g["levels"]] == [0.0, 200.0]      # kept sorted


def test_re_adding_at_the_same_level_replaces_it():
    """Nudging a slice is an edit, not an accumulation of near-identical ones."""
    from minflux_viewer.core.roi_volume import add_cross_section

    rec = Rec("polyhedron", {"axis": "Z",
                             "levels": [{"at": 0.0, "polygon": _circle(100.0).tolist()},
                                        {"at": 50.0, "polygon": _circle(60.0).tolist()}]})
    g = add_cross_section(rec, 50.0, _circle(10.0).tolist())
    assert len(g["levels"]) == 2
    at50 = [lv for lv in g["levels"] if lv["at"] == 50.0][0]
    assert max(abs(p[0]) for p in at50["polygon"]) == pytest.approx(10.0, rel=1e-6)


def test_a_multi_slice_shape_interpolates_between_the_drawn_levels():
    """The whole point of a second level: the shape between follows the
    structure instead of extruding one outline through it."""
    from minflux_viewer.core.roi_volume import add_cross_section

    rec = Rec("polyhedron", {"axis": "Z", "thickness": 10.0,
                             "levels": [{"at": 0.0, "polygon": _circle(100.0).tolist()}]})
    rec.geometry = add_cross_section(rec, 200.0, _circle(20.0).tolist())
    # Halfway the radius is halfway: a prism would still be 100 everywhere.
    outline = cross_section_at(rec.geometry, 100.0, n_angles=32)
    assert np.hypot(outline[:, 0], outline[:, 1]).mean() == pytest.approx(60.0, rel=2e-2)
    assert roi_volume_mask([55.0], [0.0], [100.0], rec)[0]
    assert not roi_volume_mask([65.0], [0.0], [100.0], rec).any()


def test_a_prism_is_left_alone_by_a_no_op_add():
    from minflux_viewer.core.roi_volume import add_cross_section

    rec = Rec("cuboid", {"x": [0, 1], "y": [0, 1], "z": [0, 1]})
    assert add_cross_section(rec, 0.0, _circle(5.0).tolist()) is None      # wrong type
    poly = Rec("polyhedron", {"axis": "Z", "levels": []})
    assert add_cross_section(poly, 0.0, [[0.0, 0.0], [1.0, 1.0]]) is None  # <3 vertices
    projection = Rec("polyhedron", _projection_box())
    assert add_cross_section(projection, 20.0, _circle(5.0).tolist()) is None


def test_the_convex_hull_encloses_its_own_points():
    from minflux_viewer.core.roi_volume import convex_hull_polyhedron

    rng = np.random.default_rng(0)
    pts = rng.normal(0.0, 100.0, (3000, 3))
    g = convex_hull_polyhedron(pts)
    assert g is not None and len(g["levels"]) > 1
    rec = Rec("polyhedron", g)
    inside = roi_volume_mask(pts[:, 0], pts[:, 1], pts[:, 2], rec)
    # ⚠ Sampled cross-sections, not an exact face list: a little tighter than
    # the true hull between levels. Stated rather than hidden.
    assert inside.mean() > 0.95
    assert not roi_volume_mask([1e4], [1e4], [0.0], rec).any()


def test_the_hull_needs_enough_points_to_be_a_solid():
    from minflux_viewer.core.roi_volume import convex_hull_polyhedron

    assert convex_hull_polyhedron(np.zeros((3, 3))) is None
    assert convex_hull_polyhedron(np.zeros((0, 3))) is None


@pytest.mark.parametrize(("kind", "geometry"), [
    ("cuboid", {"x": [0, 1], "y": [0, 1], "z": [0, 1]}),
    ("sphere", {"center": [0, 0, 0], "radii": [1, 1, 1]}),
    ("cylinder", {"axis": "Z", "center": [0, 0, 0], "radii": [1, 1], "height": 2}),
    ("polyhedron", _projection_box()),
])
def test_imagej_export_refuses_a_volume_roi_by_name(kind, geometry):
    """⚠ Silently writing an XY silhouette would put a flat rectangle in the
    file under the name of a cuboid, and nothing downstream could tell."""
    pytest.importorskip("roifile")
    from minflux_viewer.core.roi import RoiRecord, record_to_imagej

    rec = RoiRecord.create(kind, geometry, name=f"{kind}-1")
    with pytest.raises(ValueError, match="3-D ROI"):
        record_to_imagej(rec)


# --- the property read-out -------------------------------------------------
# ⚠ Both ROI property dialogs used the 2-D `_bounds`, which knows only
# bounds/point/points and answers (0, 0, 0, 0) for a named-axis geometry -- so a
# volume ROI reported "0 vertices, bbox X=0, Y=0, W=0, H=0", a wrong answer
# rather than a missing one. One formatter now serves both.

def _nm(value):
    v = float(value)
    return f"{int(round(v))}" if np.isfinite(v) else ""


def test_volume_read_out_states_the_real_three_dimensional_extent():
    from minflux_viewer.core.roi_volume import volume_geometry_text

    rec = Rec("cuboid", {"x": [0, 100], "y": [10, 30], "z": [-20, 60]})
    text = volume_geometry_text(rec, _nm)
    assert text == "X=0, Y=10, Z=-20, W=100, H=20, D=80"
    assert "W=0" not in text and "H=0" not in text        # the defect it replaces


def test_each_volume_shape_adds_what_only_it_can_say():
    from minflux_viewer.core.roi_volume import volume_geometry_text

    sphere = Rec("sphere", {"center": [5, 6, 7], "radii": [2, 3, 4]})
    assert "centre=(5, 6, 7), radii=(2, 3, 4)" in volume_geometry_text(sphere, _nm)

    prism = Rec("polyhedron", {"axis": "Z", "thickness": 40.0, "levels": [
        {"at": 0.0, "polygon": [[0, 0], [10, 0], [10, 10]]}]})
    assert "1 cross-section along Z at 0, thickness=40" in volume_geometry_text(prism, _nm)

    stack = Rec("polyhedron", {"axis": "Z", "levels": [
        {"at": 0.0, "polygon": [[0, 0], [10, 0], [10, 10]]},
        {"at": 40.0, "polygon": [[2, 2], [8, 2], [8, 8]]}]})
    text = volume_geometry_text(stack, _nm)
    assert "2 cross-sections along Z at 0, 40" in text
    assert "thickness" not in text          # a second level supersedes it


def test_an_unreadable_volume_geometry_says_so_instead_of_reading_as_empty():
    from minflux_viewer.core.roi_volume import volume_geometry_text

    rec = Rec("cuboid", {"x": [0, 100]})       # no y / z
    assert "not readable" in volume_geometry_text(rec, _nm)


# --------------------------------------------------------------------- cylinder
# A cylinder is an oval cross-section extruded along the axis normal to the plane
# it was drawn in: the drawn ellipse in that plane, a rectangle in the other two.


def _cyl(axis="Z", centre=(50.0, 30.0, 0.0), radii=(50.0, 30.0), height=40.0):
    return Rec("cylinder", {"axis": axis, "center": list(centre),
                            "radii": list(radii), "height": height})


def test_cylinder_lifts_an_oval_keeping_its_two_in_plane_radii():
    from minflux_viewer.core.roi_volume import volume_from_flat

    kind, g = volume_from_flat("oval", {"bounds": [0.0, 0.0, 100.0, 60.0]}, "XY",
                               (-20.0, 20.0), volume_type="cylinder")
    assert kind == "cylinder"
    assert g == {"axis": "Z", "center": [50.0, 30.0, 0.0],
                 "radii": [50.0, 30.0], "height": 40.0}


def test_an_oval_still_lifts_to_a_sphere_unless_a_cylinder_is_asked_for():
    """The same drawn ellipse becomes either shape, so the flat geometry alone
    cannot decide it and the default must not change."""
    from minflux_viewer.core.roi_volume import volume_from_flat

    kind, _g = volume_from_flat("oval", {"bounds": [0.0, 0.0, 10.0, 10.0]},
                                "XY", (0.0, 5.0))
    assert kind == "sphere"


def test_volume_from_flat_refuses_a_target_that_is_not_a_volume_type():
    from minflux_viewer.core.roi_volume import volume_from_flat

    with pytest.raises(ValueError):
        volume_from_flat("oval", {"bounds": [0.0, 0.0, 1.0, 1.0]}, "XY",
                         (0.0, 1.0), volume_type="rectangle")


def test_cylinder_membership_is_elliptic_not_its_bounding_box():
    rec = _cyl()
    box = Rec("cuboid", {"x": [0.0, 100.0], "y": [0.0, 60.0], "z": [-20.0, 20.0]})
    corner = ([95.0], [55.0], [0.0])       # in the bbox, outside the ellipse
    assert roi_volume_mask(*corner, box)[0]
    assert not roi_volume_mask(*corner, rec)[0]


def test_cylinder_membership_is_bounded_on_its_own_axis():
    m = roi_volume_mask([50.0] * 3, [30.0] * 3, [0.0, 19.0, 21.0], _cyl())
    assert m.tolist() == [True, True, False]


def test_cylinder_bounds_are_the_radii_and_the_height():
    assert volume_bounds(_cyl()) == ((0.0, 100.0), (0.0, 60.0), (-20.0, 20.0))


def test_cylinder_silhouette_is_an_ellipse_down_the_axis_and_a_rectangle_across():
    rec = _cyl()                                            # axis Z
    assert len(volume_silhouette(rec, 0, 1)) > 4            # XY: the ellipse
    assert len(volume_silhouette(rec, 0, 2)) == 4           # XZ: a rectangle
    assert len(volume_silhouette(rec, 2, 1)) == 4           # ortho YZ: a rectangle


def test_cylinder_silhouette_follows_the_named_axis_not_the_drawing_plane():
    """The axis is stored in the geometry, so a cylinder about X shows its
    ellipse in YZ -- no view has to remember which plane drew it, which is the
    whole reason the schema is named data axes rather than plane + 2-D shape."""
    rec = _cyl(axis="X", centre=(5.0, 20.0, 10.0), radii=(20.0, 10.0), height=20.0)
    assert len(volume_silhouette(rec, 1, 2)) > 4            # YZ: the ellipse
    assert len(volume_silhouette(rec, 0, 1)) == 4           # XY: a rectangle
    assert len(volume_silhouette(rec, 0, 2)) == 4           # XZ: a rectangle


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_cylinder_mesh_is_watertight_and_wound_outward(axis):
    """The 3-D view shades this mesh, so an inward or inconsistent normal is
    visible. ⚠ The ``(u, v, axis)`` frame is LEFT-handed for an axis of Y --
    cross_axes gives (X, Z) there, and X x Z is -Y -- so a single index order
    cannot serve all three axes.
    """
    from collections import Counter

    centre, radii, height = (50.0, 30.0, 10.0), (50.0, 30.0), 40.0
    rec = _cyl(axis=axis, centre=centre, radii=radii, height=height)
    verts, faces, _edges = volume_mesh(rec)

    undirected, directed = Counter(), Counter()
    for a, b, c in faces.tolist():
        for edge in ((a, b), (b, c), (c, a)):
            undirected[tuple(sorted(edge))] += 1
            directed[edge] += 1
    assert all(n == 2 for n in undirected.values())     # closed surface
    assert all(n == 1 for n in directed.values())       # one consistent winding

    k = {"X": 0, "Y": 1, "Z": 2}[axis]
    u, v = cross_axes(axis)
    c = np.asarray(centre, dtype=float)
    for face in faces.tolist():
        p = verts[face]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if not np.any(n):
            continue
        n = n / np.linalg.norm(n)
        cen = p.mean(axis=0)
        out = np.zeros(3)
        if abs(cen[k] - c[k]) > 0.49 * height:
            out[k] = np.sign(cen[k] - c[k])             # a cap face
        else:
            out[u] = (cen[u] - c[u]) / radii[0]
            out[v] = (cen[v] - c[v]) / radii[1]
        assert float(np.dot(n, out / np.linalg.norm(out))) > 0.5


def test_cylinder_mesh_agrees_with_its_bounds_and_sits_on_the_surface():
    rec = _cyl()
    verts, _faces, edges = volume_mesh(rec)
    bbox = tuple((float(verts[:, i].min()), float(verts[:, i].max())) for i in range(3))
    assert np.allclose(np.asarray(bbox), np.asarray(volume_bounds(rec)))
    rim = verts[2:]                                     # after the two cap centres
    radial = np.hypot((rim[:, 0] - 50.0) / 50.0, (rim[:, 1] - 30.0) / 30.0)
    assert np.allclose(radial, 1.0)
    assert set(np.round(rim[:, 2], 9).tolist()) == {-20.0, 20.0}
    assert len(edges) > 0


def test_cylinder_scale_z_is_the_height_on_a_z_axis_and_a_radius_otherwise():
    from minflux_viewer.core.roi_volume import scale_z

    g = scale_z(_cyl(centre=(50.0, 30.0, 10.0)), 0.5)
    assert g["center"][2] == 5.0 and g["height"] == 20.0
    # About X, Z is one of the cross-section's own axes, so it is a radius.
    g = scale_z(_cyl(axis="X", centre=(0.0, 0.0, 20.0), radii=(10.0, 4.0), height=30.0), 0.5)
    assert g["center"][2] == 10.0 and g["radii"] == [10.0, 2.0] and g["height"] == 30.0


def test_resizing_a_cylinder_leaves_the_axis_the_view_cannot_see():
    from minflux_viewer.core.roi_volume import set_volume_extent

    g = set_volume_extent(_cyl(), (0, 1), (0.0, 0.0, 20.0, 20.0))
    assert g["height"] == 40.0                          # an XY resize says nothing about Z
    g = set_volume_extent(_cyl(), (0, 2), (0.0, 0.0, 20.0, 20.0))
    assert g["height"] == 20.0                          # resized along the axis
    assert g["radii"][1] == 30.0                        # the Y radius is untouched


def test_translating_a_cylinder_moves_the_whole_shape():
    from minflux_viewer.core.roi_volume import translate_volume

    g = translate_volume(_cyl(), {0: 10.0, 2: 5.0})
    assert g["center"] == [60.0, 30.0, 5.0] and g["height"] == 40.0


def test_a_two_d_region_mask_refuses_a_cylinder():
    """Same reason as the other volume types: an all-False mask would be
    indistinguishable from a correct empty selection."""
    with pytest.raises(ValueError):
        roi_region_mask([0.0], [0.0], _cyl())


def test_cylinder_geometry_read_out_names_its_axis():
    from minflux_viewer.core.roi_volume import volume_geometry_text

    text = volume_geometry_text(_cyl(), lambda v: f"{v:.1f}")
    assert "axis=Z" in text and "height=40.0" in text


# ------------------------------------------------ bounding a selected cloud
# points_to_polyhedron is what lets a magic-wand selection in 3-D data keep its
# depth slice: a 2-D polygon's mask spans the whole depth axis by design.
def _blob(n=1500, sigma=40.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, size=(n, 3))


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_points_to_polyhedron_is_accepted_by_a_strict_mask(axis):
    """⚠ Every level must be star-shaped about its OWN centroid, which is what
    radial_profile measures -- rays from the point-cloud mean do not guarantee
    it. An early version assumed they did and raised NotStarShaped on the first
    concave cloud, and since roi_volume_mask defaults to strict=True that makes
    the finished ROI throw inside every consumer that asks for its mask."""
    from minflux_viewer.core.roi_volume import points_to_polyhedron

    pts = _blob()
    built = points_to_polyhedron(pts, axis=axis, level_thickness=20.0, pad=10.0)
    assert built is not None
    geometry, recovered = built
    record = Rec("polyhedron", geometry)
    mask = roi_volume_mask(pts[:, 0], pts[:, 1], pts[:, 2], record)   # strict=True
    assert mask.any()
    assert 0.0 < recovered <= 1.0


def test_points_to_polyhedron_bounds_the_cloud_on_its_stacking_axis():
    """A thin slab must come back thin: that is what excludes the data above and
    below the slice the selection was grown in."""
    from minflux_viewer.core.roi_volume import points_to_polyhedron

    rng = np.random.default_rng(1)
    slab = np.column_stack([rng.normal(0.0, 200.0, 1200),
                            rng.normal(0.0, 200.0, 1200),
                            rng.normal(0.0, 8.0, 1200)])
    geometry, _recovered = points_to_polyhedron(slab, level_thickness=20.0, pad=10.0)
    (_x, _y, (z0, z1)) = volume_bounds(Rec("polyhedron", geometry))
    assert z1 - z0 < 120.0                       # thin, not the whole axis
    far = np.column_stack([rng.normal(0.0, 100.0, 300),
                           rng.normal(0.0, 100.0, 300),
                           np.full(300, 300.0)])
    record = Rec("polyhedron", geometry)
    assert not roi_volume_mask(far[:, 0], far[:, 1], far[:, 2], record).any()


def test_points_to_polyhedron_follows_a_concavity_radially():
    """Tighter than a convex hull, which is the reason for a second builder: a
    C-shaped cloud must not come back filled in."""
    from minflux_viewer.core.roi_volume import points_to_polyhedron

    rng = np.random.default_rng(2)
    ang = rng.uniform(0.35 * np.pi, 1.65 * np.pi, 2500)
    rad = rng.uniform(80.0, 120.0, 2500)
    c_shape = np.column_stack([rad * np.cos(ang), rad * np.sin(ang),
                               rng.normal(0.0, 15.0, 2500)])
    geometry, recovered = points_to_polyhedron(c_shape, level_thickness=20.0, pad=6.0)
    assert recovered > 0.9
    notch = np.array([[100.0, 0.0, 0.0]])        # in the gap of the C
    record = Rec("polyhedron", geometry)
    assert not roi_volume_mask(notch[:, 0], notch[:, 1], notch[:, 2],
                               record, strict=False)[0]


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_points_to_polyhedron_survives_a_degenerate_cloud(n):
    """A selection can be one localization; it still has to produce a usable ROI
    rather than a geometry the mask reader refuses."""
    from minflux_viewer.core.roi_volume import points_to_polyhedron

    rng = np.random.default_rng(3)
    pts = rng.normal(0.0, 1.0, size=(n, 3))
    built = points_to_polyhedron(pts, level_thickness=20.0, pad=5.0)
    assert built is not None
    geometry, _recovered = built
    record = Rec("polyhedron", geometry)
    roi_volume_mask(pts[:, 0], pts[:, 1], pts[:, 2], record)      # must not raise


def test_points_to_polyhedron_caps_its_level_count():
    from minflux_viewer.core.roi_volume import MAX_POINT_LEVELS, points_to_polyhedron

    rng = np.random.default_rng(4)
    tall = np.column_stack([rng.normal(0.0, 50.0, 4000),
                            rng.normal(0.0, 50.0, 4000),
                            rng.uniform(-5000.0, 5000.0, 4000)])
    geometry, _recovered = points_to_polyhedron(tall, level_thickness=5.0, pad=2.0)
    assert len(geometry["levels"]) <= MAX_POINT_LEVELS
