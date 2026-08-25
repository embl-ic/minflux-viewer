"""RCC and DME drift estimators, and the data regime in which they work."""

import numpy as np
import pytest

from minflux_viewer.analysis.drift_estimators import (
    DriftTrajectory,
    _entropy_and_gradient,
    _time_bins,
    apply_drift,
    estimate_drift_dme,
    estimate_drift_rcc,
)

CELL = np.array([2000.0, 740.0, 740.0])
PRECISION = 6.0


def _scene(n_visits, *, n_molecules=800, locs_total=24_000, seed=1,
           duration=28_800.0):
    """A cell where each molecule is observed *n_visits* times over the run.

    ``n_visits=1`` is the MINFLUX regime — each molecule tracked once, so
    different time windows hold disjoint molecules. Larger values are the
    PAINT/STORM regime the cross-correlation methods were designed for.
    """
    rng = np.random.default_rng(seed)
    molecules = rng.uniform(0, 1, size=(n_molecules, 3)) * CELL
    per_visit = max(int(round(locs_total / (n_molecules * n_visits))), 1)
    xyz, t = [], []
    for molecule in molecules:
        for start in rng.uniform(0.0, 1.0, n_visits):
            xyz.append(molecule + rng.normal(0.0, PRECISION, size=(per_visit, 3)))
            t.append(np.full(per_visit, start * duration))
    return np.vstack(xyz), np.concatenate(t)


def _inject(t, total_nm, seed=5):
    rng = np.random.default_rng(seed)
    grid = np.linspace(t.min(), t.max(), 120)
    path = np.cumsum(rng.normal(size=(120, 3)), axis=0)
    path -= path.mean(axis=0)
    path *= total_nm / max(np.max(np.ptp(path, axis=0)), 1e-9)
    at = np.column_stack([np.interp(t, grid, path[:, k]) for k in range(3)])
    return at


def _residual(truth, estimate, t):
    """RMS residual after removing the free constant offset."""
    got = estimate.at(t) if estimate is not None else np.zeros_like(truth)
    r = truth - got
    r -= r.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(r ** 2, axis=1))))


# --------------------------------------------------------------------------- #
# Mechanics
# --------------------------------------------------------------------------- #
def test_trajectory_interpolates_and_is_mean_free():
    traj = DriftTrajectory(np.array([0.0, 10.0]),
                           np.array([[0.0, 0.0, 0.0], [10.0, -6.0, 2.0]]))
    mid = traj.at([5.0])
    assert mid[0] == pytest.approx([5.0, -3.0, 1.0])
    assert traj.excursion_nm() == pytest.approx([10.0, 6.0, 2.0])
    xyz = np.zeros((1, 3))
    assert apply_drift(xyz, [5.0], traj)[0] == pytest.approx([-5.0, 3.0, -1.0])


def test_time_bins_are_equal_occupancy():
    """Equal duration would leave bins nearly empty when the rate varies."""
    t = np.concatenate([np.linspace(0, 10, 900), np.linspace(10, 1000, 100)])
    labels = _time_bins(t, 10)
    counts = np.bincount(labels, minlength=10)
    assert counts.min() >= 99 and counts.max() <= 101


def test_dme_gradient_matches_finite_differences():
    """The objective's gradient, checked against the objective itself.

    Worth pinning: an inverted sign here does not crash — it silently makes DME
    return its own initialisation.
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(0)
    n, n_bins = 2000, 5
    xyz = rng.uniform(0, 400, size=(n, 3))
    labels = _time_bins(rng.uniform(0, 100, n), n_bins)
    offsets = rng.normal(0, 3, size=(n_bins, 3))
    offsets -= offsets.mean(axis=0)
    pairs = cKDTree(xyz).query_pairs(25.0, output_type="ndarray")
    left, right = pairs[:, 0], pairs[:, 1]
    cross = labels[left] != labels[right]
    left, right = left[cross], right[cross]
    inv_var = np.full(left.size, 1.0 / (2 * PRECISION ** 2))

    _energy, gradient = _entropy_and_gradient(
        xyz, labels, offsets, (left, right), inv_var, n_bins)
    numeric = np.zeros_like(gradient)
    eps = 1e-4
    for b in range(n_bins):
        for k in range(3):
            up = offsets.copy()
            up[b, k] += eps
            down = offsets.copy()
            down[b, k] -= eps
            e_up, _ = _entropy_and_gradient(xyz, labels, up, (left, right),
                                            inv_var, n_bins)
            e_dn, _ = _entropy_and_gradient(xyz, labels, down, (left, right),
                                            inv_var, n_bins)
            numeric[b, k] = (e_up - e_dn) / (2 * eps)
    assert np.allclose(gradient, numeric, atol=1e-4, rtol=1e-3)


def test_rcc_refuses_a_segment_it_cannot_correlate():
    xyz, t = _scene(4, n_molecules=60, locs_total=300)
    with pytest.raises(ValueError, match="fewer than"):
        estimate_drift_rcc(xyz, t, n_segments=40, min_localizations=50)


def test_rcc_refuses_an_unaffordable_grid():
    xyz, t = _scene(4, n_molecules=200, locs_total=4000)
    with pytest.raises(ValueError, match="too large"):
        estimate_drift_rcc(xyz, t, n_segments=4, bin_nm=0.02)


# --------------------------------------------------------------------------- #
# Recovery, in the regime the methods assume
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("estimator", ["rcc", "dme"])
def test_recovers_injected_drift_when_the_structure_is_resampled(estimator):
    """With each molecule seen many times, both methods remove most of it."""
    xyz, t = _scene(8)
    truth = _inject(t, 60.0)
    drifted = xyz + truth
    baseline = _residual(truth, None, t)
    if estimator == "rcc":
        est = estimate_drift_rcc(drifted, t, n_segments=15, bin_nm=20.0)
    else:
        est = estimate_drift_dme(drifted, t, sigma_nm=PRECISION, n_bins=15,
                                 iterations=300, step_nm=3.0)
    assert _residual(truth, est, t) < 0.45 * baseline


def test_dme_tolerates_less_resampling_than_rcc():
    """The DME paper's central claim, reproduced: at two visits per molecule
    the entropy method still works and cross-correlation does not.

    Needs a cell's worth of signal — the margin is sample-size dependent, so
    this uses the reference-scale scene (1650 molecules, 48k localizations)
    rather than the smaller one the other tests share.
    """
    xyz, t = _scene(2, n_molecules=1650, locs_total=48_000)
    truth = _inject(t, 60.0)
    drifted = xyz + truth
    baseline = _residual(truth, None, t)

    dme = estimate_drift_dme(drifted, t, sigma_nm=PRECISION, n_bins=15,
                             iterations=400, step_nm=3.0)
    assert _residual(truth, dme, t) < 0.5 * baseline

    rcc = estimate_drift_rcc(drifted, t, n_segments=15, bin_nm=20.0)
    assert _residual(truth, rcc, t) > baseline       # no better than doing nothing


def test_neither_method_works_when_every_molecule_is_seen_once():
    """The MINFLUX regime, and the reason no fiducial-free correction has
    helped on this data: disjoint molecules per time window leave nothing in
    common to align, so both estimators inject error instead of removing it."""
    xyz, t = _scene(1)
    truth = _inject(t, 60.0)
    drifted = xyz + truth
    baseline = _residual(truth, None, t)
    for est in (estimate_drift_rcc(drifted, t, n_segments=15, bin_nm=20.0),
                estimate_drift_dme(drifted, t, sigma_nm=PRECISION, n_bins=15,
                                   iterations=300, step_nm=3.0)):
        assert _residual(truth, est, t) > 2.0 * baseline


def test_dme_reports_how_it_was_initialised():
    xyz, t = _scene(8)
    est = estimate_drift_dme(xyz + _inject(t, 40.0), t, sigma_nm=PRECISION,
                             n_bins=12, iterations=100)
    assert est.method == "dme"
    assert est.info["initialised_with"] == "rcc"
    assert est.info["energy"] <= est.info["energy_start"]
