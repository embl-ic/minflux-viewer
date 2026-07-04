"""Synthetic MINFLUX sample-data generation (core/simulate.py)."""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.core.simulate import (
    STRUCTURES,
    default_params,
    simulate_attributes,
    simulate_localizations,
)


def test_all_structures_generate_and_group_into_traces():
    for key in STRUCTURES:
        coords, tid, attrs = simulate_localizations(
            key, n_points=200, locs_per_trace=3.0, precision_nm=4.0, dim=3, seed=1)
        assert coords.ndim == 2 and coords.shape[1] == 3
        assert coords.shape[0] == tid.shape[0] >= 200          # ≥1 loc per molecule
        assert np.all(np.isfinite(coords))
        # every attribute column aligns to the localizations
        for col in attrs.values():
            assert col.shape[0] == coords.shape[0]
        # ~n_points distinct traces (Poisson counts ≥ 1)
        assert np.unique(tid).size == 200
        assert tid.min() == 1 and tid.max() == 200


def test_dim_2_flattens_z():
    coords, _tid, _attrs = simulate_localizations("sphere", n_points=100, dim=2, seed=0)
    assert np.allclose(coords[:, 2], 0.0)
    coords3, *_ = simulate_localizations("sphere", n_points=100, dim=3, seed=0)
    assert np.ptp(coords3[:, 2]) > 0.0                         # 3-D keeps Z


def test_seed_is_deterministic():
    a = simulate_localizations("clusters", n_points=300, seed=42)[0]
    b = simulate_localizations("clusters", n_points=300, seed=42)[0]
    c = simulate_localizations("clusters", n_points=300, seed=43)[0]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_attribute_distributions_and_ranges():
    rng = np.random.default_rng(0)
    a = simulate_attributes(5000, rng)
    assert set(a) == {"efo", "cfr", "dcr", "eco", "ecc", "fbg"}
    assert np.all(a["efo"] > 0)                                # log-normal positive
    assert a["cfr"].min() >= 0.0 and a["cfr"].max() <= 1.0     # clipped ratios
    assert a["dcr"].min() >= 0.0 and a["dcr"].max() <= 1.0
    assert np.all(a["eco"] >= 0) and np.allclose(a["eco"], np.round(a["eco"]))  # Poisson ints
    assert np.all(a["fbg"] >= 0.0)


def test_npc_ring_geometry():
    # NPC two-ring: Z spans ~ the ring separation; XY spread ~ the field size.
    p = {**default_params("npc"), "n_pores": 1, "size_nm": 0.0,
         "diameter_nm": 100.0, "ring_sep_nm": 50.0}
    coords, _tid, _ = simulate_localizations(
        "npc", n_points=4000, locs_per_trace=1.0, precision_nm=0.0, params=p, seed=3)
    # a single pore centred at origin: two Z planes near ±25 nm (+ subunit jitter)
    assert coords[:, 2].max() > 20 and coords[:, 2].min() < -20
    radius = np.hypot(coords[:, 0], coords[:, 1])
    assert 40 < np.median(radius) < 60                         # ring radius ≈ 50 nm


def test_npc_field_curvature_and_local_tilt_add_axial_spread():
    # flat field, no local tilt → Z spread ≈ ring separation (rings + subunit jitter)
    flat = {**default_params("npc"), "n_pores": 200, "size_nm": 4000.0,
            "ring_sep_nm": 50.0, "field_curvature": 0.0, "local_tilt_deg": 0.0}
    z_flat = np.ptp(simulate_localizations(
        "npc", n_points=6000, locs_per_trace=1.0, precision_nm=0.0, params=flat, seed=1)[0][:, 2])

    # a curved field (hemisphere) tilts edge pores → much larger axial spread
    curved = {**flat, "field_curvature": 1.0}
    z_curved = np.ptp(simulate_localizations(
        "npc", n_points=6000, locs_per_trace=1.0, precision_nm=0.0, params=curved, seed=1)[0][:, 2])
    assert z_curved > 5 * z_flat

    # per-NPC local tilt (flat field) also increases axial spread
    tilted = {**flat, "local_tilt_deg": 20.0}
    z_tilt = np.ptp(simulate_localizations(
        "npc", n_points=6000, locs_per_trace=1.0, precision_nm=0.0, params=tilted, seed=1)[0][:, 2])
    assert z_tilt > z_flat


def test_npc_overlay_channels_share_scaffold():
    """3 co-registered channels: same pore scaffold, different labelling diameters."""
    from minflux_viewer.core.simulate import default_params, simulate_npc_overlay

    p = {**default_params("npc_overlay_3ch"), "n_pores": 1,
         "field_curvature": 0.0, "local_tilt_deg": 0.0}
    chans = simulate_npc_overlay(p, locs_per_trace=1.0, precision_nm=0.0, seed=1)
    assert [c["name"] for c in chans] == ["Nup-scaffold", "Nup-inner", "central-cap"]
    assert [c["lut"] for c in chans] == ["Red", "Green", "Blue"]

    # a single pore → all channels share the same XY centre (co-registered)
    cents = [c["coords"][:, :2].mean(axis=0) for c in chans]
    for c in cents[1:]:
        assert np.linalg.norm(c - cents[0]) < 5.0
    # ring radius follows each channel's own diameter (107 > 70 > 40 → radius/2)
    radii = [np.median(np.hypot(c["coords"][:, 0] - cents[0][0],
                                c["coords"][:, 1] - cents[0][1])) for c in chans]
    assert radii[0] > radii[1] > radii[2]


def test_npc_dcr_two_channel_is_bimodal():
    """One dataset, two reporters recoverable only via a bimodal ``dcr``."""
    from minflux_viewer.core.simulate import default_params, simulate_npc_dcr

    p = {**default_params("npc_dcr_2ch"), "n_pores": 20, "dcr_low": 0.25, "dcr_high": 0.75}
    coords, tid, attrs = simulate_npc_dcr(p, locs_per_trace=2.0, precision_nm=3.0, seed=1)
    dcr = attrs["dcr"]
    assert coords.shape[0] == tid.shape[0] == dcr.shape[0] > 0
    low, high = dcr[dcr < 0.5], dcr[dcr >= 0.5]
    assert low.size and high.size
    assert abs(np.mean(low) - 0.25) < 0.05 and abs(np.mean(high) - 0.75) < 0.05
    assert np.unique(tid).size > 1                       # globally-unique traces


def test_place_nonoverlapping_respects_min_distance():
    from scipy.spatial import cKDTree

    from minflux_viewer.core.simulate import place_nonoverlapping

    pts = place_nonoverlapping(300, 150.0, np.random.default_rng(0), disk_radius=2000)
    assert pts.shape[0] > 1
    d, _ = cKDTree(pts).query(pts, k=2)
    assert d[:, 1].min() >= 150.0 - 1e-6                 # no pair closer than min_dist
    # a too-dense field packs fewer (never overlapping)
    tight = place_nonoverlapping(500, 300.0, np.random.default_rng(1), disk_radius=1000)
    assert tight.shape[0] < 500
    d2, _ = cKDTree(tight).query(tight, k=2)
    assert d2[:, 1].min() >= 300.0 - 1e-6


def test_npc_scaffold_pores_do_not_overlap():
    """Distinct NPC pores are ≥ their (largest) labelling diameter apart — they
    cannot spatially overlap (unlike the co-registered channels of one pore)."""
    from scipy.spatial import cKDTree

    from minflux_viewer.core.simulate import _npc_scaffold

    centres, _u, _v, _n = _npc_scaffold(
        80, 4000.0, 0.0, 0.0, np.random.default_rng(0), min_separation=107.0)
    assert centres.shape[0] > 1
    d, _ = cKDTree(centres[:, :2]).query(centres[:, :2], k=2)
    assert d[:, 1].min() >= 107.0                         # ≥ largest diameter


def test_simulate_beads_tracks_and_drift():
    """Fiducial beads: per-gri tracks in metres with a shared common-mode drift."""
    from minflux_viewer.core.simulate import simulate_beads

    pts, pbg, used = simulate_beads(4, n_time=30, drift_nm=40.0, jitter_nm=1.0, seed=1)
    assert pts.shape[0] == 4 * 30
    assert {"gri", "xyz", "tim", "str"} <= set(pts.dtype.names)
    assert set(np.unique(pts["gri"])) == {1, 2, 3, 4}
    assert len(pbg) == 4 and used == ["R1", "R2", "R3", "R4"]
    xyz = np.asarray(pts["xyz"], float)
    assert np.abs(xyz).max() < 1e-4                      # metres (µm-scale field)
    # each bead drifts over time (common-mode stage drift is present)
    for g in (1, 2, 3, 4):
        m = pts["gri"] == g
        assert np.ptp(xyz[m], axis=0).max() > 0.0
    # 2-D beads have no Z motion
    flat, *_ = simulate_beads(3, n_time=10, dim=2, seed=2)
    assert np.allclose(np.asarray(flat["xyz"], float)[:, 2], 0.0)


def test_unknown_structure_raises():
    with pytest.raises(ValueError):
        simulate_localizations("not-a-structure", n_points=10)
