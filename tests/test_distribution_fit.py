"""
Generic 1-D mixture-distribution fitting (attribute-based channel separation).

Gaussian / Log-Normal go through sklearn's GaussianMixture; Gamma / Poisson
through a small scipy-backed weighted EM; Uniform is a data-adaptive quantile
partition. BIC/AIC are computed uniformly so ``auto_fit`` can compare them.
"""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.analysis.distribution_fit import (
    DISTRIBUTIONS,
    auto_fit,
    fit_mixture,
)


def _bimodal_gaussian(seed=0, n1=4000, n2=3000, m1=0.30, s1=0.05, m2=0.70, s2=0.06):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(m1, s1, n1), rng.normal(m2, s2, n2)])


# --------------------------------------------------------------------------- Gaussian
def test_gaussian_recovers_two_peaks():
    res = fit_mixture(_bimodal_gaussian(), "gaussian", 2)
    assert res.n_components == 2
    locs = res.locations                                  # sorted ascending
    assert abs(locs[0] - 0.30) < 0.02
    assert abs(locs[1] - 0.70) < 0.02
    assert abs(res.weights[0] - 4 / 7) < 0.05
    assert abs(res.weights.sum() - 1.0) < 1e-9
    # single decision boundary sits between the means
    b = res.boundaries()
    assert b.shape == (1,)
    assert locs[0] < b[0] < locs[1]


def test_gaussian_assign_and_pdf_integrates_to_weights():
    res = fit_mixture(_bimodal_gaussian(), "gaussian", 2)
    lab = res.assign(np.array([0.30, 0.70, np.nan]))
    assert lab[0] == 0 and lab[1] == 1 and lab[2] == -1
    # each weighted component density integrates (numerically) to its weight
    from scipy.integrate import trapezoid
    xs = np.linspace(-0.2, 1.2, 4000)
    comp = res.component_pdfs(xs)
    areas = trapezoid(comp, xs, axis=1)
    assert np.allclose(areas, res.weights, atol=0.02)


# --------------------------------------------------------------------------- Log-Normal
def test_lognormal_recovers_two_medians():
    rng = np.random.default_rng(1)
    x = np.concatenate([
        rng.lognormal(mean=np.log(2.0), sigma=0.3, size=4000),
        rng.lognormal(mean=np.log(10.0), sigma=0.3, size=3000),
    ])
    res = fit_mixture(x, "lognormal", 2)
    locs = res.locations                                  # component medians = exp(mu_log)
    assert abs(locs[0] - 2.0) / 2.0 < 0.2
    assert abs(locs[1] - 10.0) / 10.0 < 0.2


def test_positive_only_distributions_drop_nonpositive():
    x = np.array([-5.0, -1.0, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    res = fit_mixture(x, "lognormal", 1)                  # must not choke on <=0
    assert res.n == 5                                      # only the 5 positive values


# --------------------------------------------------------------------------- Gamma
def test_gamma_recovers_two_means():
    rng = np.random.default_rng(2)
    x = np.concatenate([
        rng.gamma(shape=9.0, scale=0.5, size=4000),       # mean 4.5
        rng.gamma(shape=9.0, scale=2.0, size=3000),       # mean 18
    ])
    res = fit_mixture(x, "gamma", 2)
    locs = res.locations                                  # component means
    assert abs(locs[0] - 4.5) / 4.5 < 0.25
    assert abs(locs[1] - 18.0) / 18.0 < 0.25


# --------------------------------------------------------------------------- Poisson
def test_poisson_recovers_two_rates():
    rng = np.random.default_rng(3)
    x = np.concatenate([
        rng.poisson(5.0, size=4000).astype(float),
        rng.poisson(50.0, size=3000).astype(float),
    ])
    res = fit_mixture(x, "poisson", 2)
    locs = res.locations                                  # component lambdas
    assert abs(locs[0] - 5.0) / 5.0 < 0.2
    assert abs(locs[1] - 50.0) / 50.0 < 0.2


# --------------------------------------------------------------------------- Uniform
def test_uniform_equal_population_partition():
    rng = np.random.default_rng(4)
    x = rng.uniform(0.0, 100.0, size=40_000)
    res = fit_mixture(x, "uniform", 4)
    assert res.n_components == 4
    assert np.allclose(res.weights, 0.25, atol=1e-9)      # equal population
    b = res.boundaries()
    assert b.shape == (3,)
    # boundaries near the 25/50/75 percentiles
    assert abs(b[0] - 25.0) < 3 and abs(b[1] - 50.0) < 3 and abs(b[2] - 75.0) < 3


# --------------------------------------------------------------------------- auto_fit
def test_auto_fit_picks_gaussian_two_components_for_bimodal_gaussian():
    res = auto_fit(_bimodal_gaussian(), max_components=3)
    assert res.distribution == "gaussian"
    assert res.n_components == 2
    assert np.isfinite(res.bic)


def test_auto_fit_single_component_for_one_peak():
    rng = np.random.default_rng(5)
    x = rng.normal(0.5, 0.05, 5000)
    res = auto_fit(x, distributions=["gaussian"], max_components=3)
    assert res.n_components == 1


# --------------------------------------------------------------------------- misc
def test_large_input_is_subsampled_and_fast():
    x = _bimodal_gaussian(n1=1_500_000, n2=1_200_000)     # ~2.7M points
    res = fit_mixture(x, "gaussian", 2, max_points=100_000)
    assert res.n == 100_000                                # fit basis was subsampled
    assert abs(res.locations[0] - 0.30) < 0.02


def test_registry_and_errors():
    assert set(DISTRIBUTIONS) == {"gaussian", "lognormal", "uniform", "gamma", "poisson"}
    with pytest.raises(ValueError):
        fit_mixture(_bimodal_gaussian(), "nope", 2)
    with pytest.raises(ValueError):
        fit_mixture([1.0], "gaussian", 2)                  # too few values
