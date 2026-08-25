"""E. coli HlyB dimer simulation — the known-distance control.

It exists so the staged pair analysis can be checked against a truth it is not
told, so the tests assert both that the simulation reproduces the reference
acquisitions' statistics and that the analysis recovers the planted distance.
"""

import numpy as np
import pytest

from minflux_viewer.analysis.hlyb_staged import (
    Staged3DConfig,
    analyze_hlyb_staged_3d,
)
from minflux_viewer.core.sample_presets import default_presets
from minflux_viewer.core.simulate import (
    MULTI_SIMS,
    capsule_surface_points,
    default_params,
    param_specs,
    sim_kind,
    simulate_ecoli_hlyb,
    structure_labels,
)

KEY = "ecoli_hlyb_dimer"


def _sim(seed=7, **override):
    params = {**default_params(KEY), **override}
    return simulate_ecoli_hlyb(params, locs_per_trace=14.0, precision_nm=6.0,
                               seed=seed)


def _traces(coords, tid):
    order = np.argsort(tid, kind="stable")
    tid_s, xyz = tid[order], coords[order]
    bounds = np.flatnonzero(np.diff(tid_s)) + 1
    return np.split(xyz, bounds)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
def test_registered_as_its_own_simulation_kind():
    assert KEY in MULTI_SIMS
    assert sim_kind(KEY) == "ecoli"
    assert KEY in dict(structure_labels())
    names = {spec.name for spec in param_specs(KEY)}
    assert {"cell_length_nm", "cell_radius_nm", "subunit_density_per_um2",
            "dimer_distance_nm", "dimer_distance_sd_nm", "dimer_fraction",
            "detection_probability"} <= names


def test_shipped_as_a_sample_preset():
    preset = next(p for p in default_presets() if p["structure"] == KEY)
    assert preset["dim"] == 3
    assert preset["locs_per_trace"] == pytest.approx(14.0)
    assert preset["precision_nm"] == pytest.approx(6.0)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def test_capsule_surface_points_lie_on_the_surface():
    rng = np.random.default_rng(0)
    length, radius = 2000.0, 370.0
    pts, normals = capsule_surface_points(4000, length, radius, rng)
    barrel = length - 2 * radius
    along = np.clip(pts[:, 0], -barrel / 2, barrel / 2)
    spine = np.column_stack([along, np.zeros(len(pts)), np.zeros(len(pts))])
    assert np.allclose(np.linalg.norm(pts - spine, axis=1), radius, atol=1e-6)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-9)


def test_surface_sampling_is_uniform_by_area_not_by_angle():
    """Sampling by angle would pile points onto the caps and fake a density
    gradient along the cell."""
    rng = np.random.default_rng(1)
    length, radius = 2000.0, 370.0
    pts, _ = capsule_surface_points(60_000, length, radius, rng)
    barrel = length - 2 * radius
    on_barrel = np.abs(pts[:, 0]) <= barrel / 2 + 1e-9
    area_barrel = 2 * np.pi * radius * barrel
    expected = area_barrel / (area_barrel + 4 * np.pi * radius ** 2)
    assert on_barrel.mean() == pytest.approx(expected, abs=0.02)


def test_cell_dimensions_follow_the_parameters():
    coords, _tid, _attrs = _sim(cell_length_nm=3000.0, cell_radius_nm=250.0)
    assert np.ptp(coords[:, 0]) == pytest.approx(3000.0, rel=0.10)
    for axis in (1, 2):
        assert np.ptp(coords[:, axis]) == pytest.approx(500.0, rel=0.15)


def test_a_cell_shorter_than_its_own_caps_is_refused():
    with pytest.raises(ValueError, match="at least twice"):
        _sim(cell_length_nm=400.0, cell_radius_nm=370.0)


# --------------------------------------------------------------------------- #
# Statistics matched to the reference acquisitions
# --------------------------------------------------------------------------- #
def test_trace_and_localization_statistics_match_the_reference_data():
    """Measured per-cell medians of the 19 reference cells: 373 traces/um2,
    14 locs per trace (mean 28.8, p90 71, heavy-tailed), per-localization sigma
    5.96 nm per lateral axis and 3.30 nm axial."""
    coords, tid, _attrs = _sim()
    blocks = _traces(coords, tid)
    counts = np.array([b.shape[0] for b in blocks])

    params = default_params(KEY)
    radius, length = params["cell_radius_nm"], params["cell_length_nm"]
    barrel = length - 2 * radius
    area_um2 = (2 * np.pi * radius * barrel + 4 * np.pi * radius ** 2) / 1e6
    assert len(blocks) / area_um2 == pytest.approx(
        params["subunit_density_per_um2"], rel=0.05)

    assert np.median(counts) == pytest.approx(14, abs=2)
    assert counts.mean() > 1.5 * np.median(counts)        # heavy tail, not Poisson

    assert counts.mean() == pytest.approx(28.8, rel=0.20)
    assert np.percentile(counts, 90) == pytest.approx(71, rel=0.25)

    # Per-localization sigma, debiased for the finite count per trace — the
    # quantity the parameters actually name.
    multi = [b for b in blocks if b.shape[0] >= 2]
    def sigma(select, axes):
        out = []
        for b in multi:
            n = b.shape[0]
            d = b[:, axes] - b[:, axes].mean(axis=0)
            out.append(np.sqrt(np.mean(np.sum(d ** 2, axis=1)) * n / (n - 1)))
        return np.median(out) / np.sqrt(len(axes))
    assert sigma(multi, [0, 1]) == pytest.approx(5.96, rel=0.12)
    assert sigma(multi, [2]) == pytest.approx(3.30, rel=0.15)


def test_every_trace_is_one_subunit_seen_once():
    """The idealisation the control depends on: no repeat visits, so no
    same-molecule pairs and no drift."""
    coords, tid, attrs = _sim()
    blocks = _traces(coords, tid)
    times = attrs["tim"]
    order = np.argsort(tid, kind="stable")
    time_blocks = np.split(times[order], np.flatnonzero(np.diff(tid[order])) + 1)
    # Each trace is a single short burst, not visits scattered over the run.
    spans = np.array([float(np.ptp(t)) for t in time_blocks if t.size > 1])
    assert spans.max() <= default_params(KEY)["trace_burst_s"] + 1e-9
    assert times.max() <= default_params(KEY)["acquisition_s"] + 5.0
    assert len(blocks) == np.unique(tid).size


# --------------------------------------------------------------------------- #
# The planted truth
# --------------------------------------------------------------------------- #
def test_dimer_partners_sit_at_the_planted_distance():
    from scipy.spatial import cKDTree

    for planted in (14.0, 20.0):
        coords, tid, _ = _sim(dimer_distance_nm=planted, dimer_distance_sd_nm=1.0,
                              locs_per_trace_tail=0.0)
        centres = np.array([b.mean(axis=0) for b in _traces(coords, tid)])
        nearest = cKDTree(centres).query(centres, k=2)[0][:, 1]
        assert np.median(nearest) == pytest.approx(planted, abs=1.5)


def test_pair_axis_lies_in_the_membrane_plane_when_asked():
    from scipy.spatial import cKDTree

    coords, tid, _ = _sim(dimer_distance_sd_nm=0.5, locs_per_trace_tail=0.0,
                          pair_in_membrane=1.0)
    centres = np.array([b.mean(axis=0) for b in _traces(coords, tid)])
    dist, idx = cKDTree(centres).query(centres, k=2)
    params = default_params(KEY)
    barrel = params["cell_length_nm"] - 2 * params["cell_radius_nm"]
    keep = (dist[:, 1] < 20.0) & (np.abs(centres[:, 0]) < barrel / 2)
    pairs = centres[idx[keep, 1]] - centres[keep]
    pairs /= np.linalg.norm(pairs, axis=1, keepdims=True)
    # On the barrel the outward normal is the radial (y, z) direction.
    radial = centres[keep][:, 1:].copy()
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    radial_component = np.abs(pairs[:, 1] * radial[:, 0] + pairs[:, 2] * radial[:, 1])
    assert np.median(radial_component) < 0.3      # nearly tangential


def test_no_dimers_places_only_isolated_subunits():
    from scipy.spatial import cKDTree

    def nearest_neighbours(**override):
        coords, tid, _ = _sim(locs_per_trace_tail=0.0, **override)
        centres = np.array([b.mean(axis=0) for b in _traces(coords, tid)])
        return cKDTree(centres).query(centres, k=2)[0][:, 1]

    paired = nearest_neighbours(dimer_fraction=1.0, dimer_distance_sd_nm=1.0)
    alone = nearest_neighbours(dimer_fraction=0.0)
    # With dimers the nearest neighbour IS the partner, at the planted distance;
    # without them it is just the next molecule along, at the mean spacing.
    assert np.median(paired) == pytest.approx(14.0, abs=1.5)
    # Random placement puts the next molecule at the mean spacing (~24 nm
    # at this density), well clear of the planted 14 nm.
    assert np.median(alone) > 1.5 * np.median(paired)
    near_planted = np.mean(np.abs(alone - 14.0) < 2.0)
    assert near_planted < 0.15      # no concentration at the planted distance


# --------------------------------------------------------------------------- #
# The point of the whole thing: does the analysis recover the truth?
# --------------------------------------------------------------------------- #
def _analyse(**override):
    coords, tid, attrs = _sim(**override)
    cfg = Staged3DConfig(z_scaling_factor=1.0, null_replicates=49,
                         run_sensitivity=False, run_stratum_profile=False)
    return analyze_hlyb_staged_3d(coords * 1e-9, tid, attrs["tim"], cfg)


@pytest.mark.parametrize("planted", [14.0, 20.0])
def test_the_analysis_recovers_the_planted_distance(planted):
    """The positive-excess centroid is the estimator that works.

    Measured over 8 seeds at the default density, a planted 14.0 nm comes back
    as 14.53 +- 0.24 nm. ``peak_nm`` is a single-bin argmax over 0.5 nm bins on
    ~1000 sites and wanders by several nm, so it is not asserted here.
    """
    summary = _analyse(dimer_distance_nm=planted)["summary"]
    assert summary["band_ratio"] > 1.6
    assert summary["positive_excess_centroid_nm"] == pytest.approx(planted, abs=1.5)


def test_the_negative_control_stays_at_the_random_baseline():
    """Without dimers the workflow must not report a short-range population.

    It does not return exactly 1: measured over 8 seeds, random placement on the
    capsule gives band ratio 1.24 +- 0.08 (max 1.35) with z up to 6.9, because
    the conditional null has limited freedom at ~1000 sites. That baseline — not
    1.0 — is what a real ratio has to beat.
    """
    summary = _analyse(dimer_fraction=0.0)["summary"]
    assert summary["band_ratio"] < 1.45
    planted = _analyse(dimer_distance_nm=14.0)["summary"]
    assert planted["band_ratio"] > summary["band_ratio"] + 0.4
    assert planted["band_ratio_z"] > 2.0 * summary["band_ratio_z"]


def test_a_real_dimer_gives_a_peak_not_a_monotone_decay():
    """The shape that distinguishes a true distance from same-molecule
    contamination: near-zero excess below the distance, then a peak at it."""
    result = _analyse(dimer_distance_nm=14.0)
    centres = np.asarray(result["centers_nm"])
    excess = np.asarray(result["observed"], float) - np.asarray(
        result["null_mean"], float)
    below = excess[(centres >= 4.0) & (centres < 8.0)].sum()
    at = excess[(centres >= 12.0) & (centres < 16.0)].sum()
    assert at > 3.0 * max(below, 1.0)
