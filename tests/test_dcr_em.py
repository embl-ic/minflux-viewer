"""Two-component Gaussian-mixture EM for DCR channel separation."""

import numpy as np

from minflux_viewer.analysis.dcr_em import (
    assign,
    assign_per_trace,
    decision_boundary,
    fit_two_component_gaussian,
    goodness_of_fit,
    responsibilities,
)


def _two_peaks(seed=0, n1=4000, n2=3000, m1=0.30, s1=0.05, m2=0.70, s2=0.06):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(m1, s1, n1), rng.normal(m2, s2, n2)])


def test_recovers_known_parameters():
    g = fit_two_component_gaussian(_two_peaks())
    assert g.converged
    assert g.mu1 < g.mu2                          # component 0 is the lower peak (red)
    assert abs(g.mu1 - 0.30) < 0.01
    assert abs(g.mu2 - 0.70) < 0.01
    assert abs(g.sigma1 - 0.05) < 0.01
    assert abs(g.sigma2 - 0.06) < 0.01
    # weights ~ 4000:3000
    assert abs(g.weight1 - 4 / 7) < 0.03
    assert abs(g.weight1 + g.weight2 - 1.0) < 1e-9


def test_components_always_sorted_lower_first():
    # even if the data is generated high-first, component 0 must be the lower mean
    g = fit_two_component_gaussian(_two_peaks(m1=0.8, m2=0.2))
    assert g.mu1 < g.mu2


def test_decision_boundary_between_means():
    g = fit_two_component_gaussian(_two_peaks())
    b = decision_boundary(g)
    assert g.mu1 < b < g.mu2
    # equal-ish weights/sigmas → boundary near the midpoint
    assert abs(b - 0.5) < 0.06


def test_responsibilities_sum_to_one_and_split():
    g = fit_two_component_gaussian(_two_peaks())
    r = responsibilities(np.array([0.30, 0.70]), g)
    assert np.allclose(r.sum(axis=1), 1.0)
    assert r[0, 0] > 0.95        # 0.30 clearly red
    assert r[1, 1] > 0.95        # 0.70 clearly green


def test_assign_labels_and_confidence():
    g = fit_two_component_gaussian(_two_peaks())
    b = decision_boundary(g)
    vals = np.array([0.30, 0.70, b, np.nan])
    # pure Bayes boundary: everything finite assigned (no -1 except NaN)
    lab = assign(vals, g, min_confidence=0.5)
    assert lab[0] == 0 and lab[1] == 1 and lab[3] == -1
    # high confidence: the boundary value becomes unassigned (the overlap)
    lab_strict = assign(vals, g, min_confidence=0.9)
    assert lab_strict[2] == -1
    assert lab_strict[0] == 0 and lab_strict[1] == 1


def test_with_means_override_resorts():
    g = fit_two_component_gaussian(_two_peaks())
    g2 = g.with_means(0.65, 0.25)                 # given high-first; must re-sort
    assert g2.mu1 == 0.25 and g2.mu2 == 0.65


def test_with_params_overrides_centres_and_widths():
    g = fit_two_component_gaussian(_two_peaks())
    g2 = g.with_params(0.25, 0.70, 0.04, 0.09)    # red=comp0, green=comp1 (no re-sort)
    assert g2.mu1 == 0.25 and g2.mu2 == 0.70
    assert g2.sigma1 == 0.04 and g2.sigma2 == 0.09
    # editing the widths shifts the Bayes boundary
    b_wide = decision_boundary(g.with_params(0.25, 0.70, 0.04, 0.20))
    b_narrow = decision_boundary(g.with_params(0.25, 0.70, 0.04, 0.04))
    assert b_wide != b_narrow


def test_goodness_of_fit_small_for_good_fit():
    g = fit_two_component_gaussian(_two_peaks())
    gof = goodness_of_fit(_two_peaks(), g)
    assert gof["nrmse"] < 0.1
    assert np.isfinite(gof["bic"])


def test_assign_per_trace_keeps_whole_trace_together():
    g = fit_two_component_gaussian(_two_peaks())
    # trace 10: mostly red (mean ~0.30) but one noisy high loc; trace 20: all green
    vals = np.array([0.30, 0.32, 0.28, 0.70,   0.68, 0.72, 0.71])
    tid = np.array([10, 10, 10, 10,            20, 20, 20])
    lab = assign_per_trace(vals, tid, g, mode="trace mean", min_confidence=0.5)
    # whole trace shares one label, decided by the trace mean
    assert set(lab[:4]) == {0}      # trace 10 mean ~0.40 → still red side
    assert set(lab[4:]) == {1}      # trace 20 → green
    assert lab[3] == 0              # the noisy 0.70 loc follows its trace, not its own value


def test_assign_per_trace_transform_applied():
    # log transform must be applied to the trace value before assignment
    g = fit_two_component_gaussian(np.log(_two_peaks()))     # fit in log space
    vals = np.array([0.30, 0.30, 0.70, 0.70])               # linear dcr
    tid = np.array([1, 1, 2, 2])
    lab = assign_per_trace(vals, tid, g, mode="trace mean", min_confidence=0.5,
                           transform=lambda v: np.log(v))
    assert lab[0] == 0 and lab[2] == 1


def test_too_few_values_raises():
    import pytest
    with pytest.raises(ValueError):
        fit_two_component_gaussian([1.0])
