"""Channel boundaries from the peaks of a distribution.

The point of this method over a mixture fit is that it reports what the data
supports: ask for three channels in a single hump and it says one peak, rather
than inventing two cuts.
"""

from __future__ import annotations

import numpy as np

from minflux_viewer.analysis.peak_channels import (
    peak_channel_boundaries,
    smoothed_density,
)


def _two_peaks(n: int = 4000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.r_[rng.normal(0.30, 0.03, n // 2), rng.normal(0.70, 0.03, n // 2)]


def test_two_peaks_give_one_cut_in_the_valley():
    bounds, found = peak_channel_boundaries(_two_peaks(), 2)
    assert found == 2
    assert bounds.size == 1
    assert 0.4 < float(bounds[0]) < 0.6


def test_three_peaks_give_two_cuts():
    rng = np.random.default_rng(2)
    values = np.r_[rng.normal(0.2, 0.02, 1500),
                   rng.normal(0.5, 0.02, 1500),
                   rng.normal(0.8, 0.02, 1500)]
    bounds, found = peak_channel_boundaries(values, 3)
    assert found == 3 and bounds.size == 2
    assert 0.25 < float(bounds[0]) < 0.45
    assert 0.55 < float(bounds[1]) < 0.75


def test_one_hump_reports_one_peak_however_many_are_requested():
    rng = np.random.default_rng(1)
    bounds, found = peak_channel_boundaries(rng.normal(0.5, 0.03, 3000), 3)
    assert found == 1
    assert bounds.size == 0


def test_asking_for_more_peaks_than_exist_invents_no_cuts():
    bounds, found = peak_channel_boundaries(_two_peaks(), 5)
    assert found == 2 and bounds.size == 1


def test_isolated_tail_points_are_not_peaks():
    """What the prominence floor exists for: on real data, counting every local
    maximum found 15-21 'peaks', each a lone point in a tail."""
    values = np.r_[_two_peaks(2000), np.array([2.5, 3.1, 4.2])]
    _bounds, found = peak_channel_boundaries(values, 4)
    assert found == 2


def test_density_grid_is_finite_and_non_negative():
    centres, density = smoothed_density(_two_peaks())
    assert centres.size == density.size > 0
    assert np.all(np.isfinite(density))
    assert density.min() >= 0.0 and density.max() > 0.0


def test_too_few_values_is_not_an_error():
    bounds, found = peak_channel_boundaries([1.0], 2)
    assert found == 0 and bounds.size == 0
