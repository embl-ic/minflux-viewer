"""Shared trace and exact pair statistics."""
import numpy as np

from minflux_viewer.analysis.pair_statistics import pair_distance_profile, trace_centroids


def test_trace_centroids_reports_standard_error_and_applies_z_scale():
    rng = np.random.default_rng(1)
    n_tr, per = 40, 25
    tid = np.repeat(np.arange(n_tr), per)
    truth = rng.uniform(0, 1000, size=(n_tr, 3))
    pts = np.repeat(truth, per, axis=0) + rng.normal(scale=5.0, size=(n_tr * per, 3))
    out = trace_centroids(pts * 1e-9, tid, z_scale=0.5, min_loc_per_trace=10)
    assert out["centroids_nm"].shape == (n_tr, 3)
    # z is scaled, x/y are not.  The centroid error is 5/sqrt(25) = 1 nm, so a
    # robust deviation is asserted rather than a per-trace bound that a single
    # 3-sigma draw would trip.
    assert np.median(np.abs(out["centroids_nm"][:, 2] - truth[:, 2] * 0.5)) < 1.5
    assert np.median(np.abs(out["centroids_nm"][:, 0] - truth[:, 0])) < 1.5
    # standard error is sd/sqrt(n), i.e. much smaller than the raw scatter
    assert np.nanmedian(out["sem_nm"]) < 2.0

def test_trace_centroids_drops_short_traces():
    tid = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    pts = np.zeros((tid.size, 3))
    out = trace_centroids(pts, tid, min_loc_per_trace=10)
    assert out["n_traces_total"] == 2
    assert out["centroids_nm"].shape[0] == 1

def test_pair_distance_profile_matches_brute_force():
    rng = np.random.default_rng(2)
    pts = rng.uniform(0, 100, size=(200, 3))
    counts, edges = pair_distance_profile(pts, 40.0, 1.0)
    from scipy.spatial.distance import pdist
    brute = np.histogram(pdist(pts), bins=edges)[0]
    assert np.array_equal(counts, brute)

def test_pair_distance_profile_has_no_exclusion_zone():
    """The whole point of this method: nothing is merged, so arbitrarily close
    pairs survive -- unlike the template pipeline, whose DBSCAN merge empties
    every bin below Dunit/2."""
    rng = np.random.default_rng(3)
    pts = np.repeat(rng.uniform(0, 500, size=(60, 3)), 2, axis=0)
    pts[1::2] += rng.normal(scale=0.4, size=(60, 3))
    counts, edges = pair_distance_profile(pts, 20.0, 0.5)
    assert counts[:4].sum() > 0
