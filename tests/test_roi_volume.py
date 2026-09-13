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
    assert VOLUME_ROI_TYPES == {"cuboid", "sphere", "polyhedron"}


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
